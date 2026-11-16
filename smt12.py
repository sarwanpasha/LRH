"""BO-SMT: Bayes-optimal layer combination (Theorem thm:bayes applied in layer-score space).
Per layer l a transfer probe (pooled LR zero-shot; prior-anchored LR few-shot) gives target scores, standardized by
the target's unlabeled statistics -> z in R^L.  Treat the target's standardized separation vector m_B as a random
effect m_B ~ N(m, P) across datasets with within-class covariance Sigma.  Then E[AUC(alpha)] =
Phi(alpha^T m / sqrt(alpha^T (2 Sigma + P) alpha)), maximized by alpha* = (2 Sigma + P)^{-1} m.
m, P, Sigma are estimated from *held-out source* score vectors: the grouped-CV fold models used to choose the pooled
LR's C score every source with a model that never saw that source (leave-sources-out).  A ridge lambda on
(2 Sigma + P) is chosen by leave-one-source-out realized AUC; lambda -> inf gives alpha propto m (matched filter / dw),
so the rule nests separation weighting.  Few-shot: Gaussian posterior for m_B given the out-of-fold labelled mean
difference d_O ~ N(m_B, 4 Sigma / n_B):  P_post = (P^-1 + n_B/4 Sigma^-1)^-1,
m_post = P_post (P^-1 m + n_B/4 Sigma^-1 d_O), alpha = (2 Sigma + P_post + lambda I)^-1 m_post.
Usage: python smt12.py --model-tag Qwen2.5-7B-Instruct --layers 8 12 ... --out results16/qw_c0.json --lofo --targets ...
"""
import argparse, glob, json, os, time
import numpy as np
import torch
from scipy.stats import norm
from smt2 import npz_memmap, auroc, md, bal_sub, F64
from smt3 import fit_lr
from smt8 import cv_oof
from smt11 import auc_to_delta, family

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))
LAMS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 1e6)


def src_stats(Z, y):
    """Z: list of (n_i x L) standardized held-out score matrices, y: labels -> per-source deltas and within covs."""
    Ds, Ss = [], []
    for z, yy in zip(Z, y):
        z = (z - z.mean(0)) / (z.std(0) + 1e-12)
        z1, z0 = z[yy == 1], z[yy == 0]
        Ds.append(z1.mean(0) - z0.mean(0))
        Ss.append(0.5 * (torch.cov(z1.T) + torch.cov(z0.T)))
    return torch.stack(Ds), torch.stack(Ss)


def moments(Ds, Ss):
    m = Ds.mean(0); Sig = Ss.mean(0)
    P = torch.cov(Ds.T) if len(Ds) > 2 else torch.zeros_like(Sig)
    return m, Sig, P


def solve(A, b, lam):
    L = A.shape[0]; t = float(torch.trace(A)) / L
    if lam >= 1e5:
        return b.clone()
    return torch.linalg.solve(A + lam * t * torch.eye(L, dtype=F64), b)


def pred_auc(a, m, A):
    return float(norm.cdf(float(a @ m) / max(float(a @ A @ a), 1e-12) ** 0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--layers", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--nbs", nargs="*", type=int, default=[16, 32, 64, 128])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--lr-per-source", type=int, default=300)
    ap.add_argument("--lofo", action="store_true")
    a = ap.parse_args()
    t0 = time.time(); LS = a.layers; nL = len(LS)
    g = torch.Generator().manual_seed(0)
    raw = {}
    for f in sorted(glob.glob(os.path.join(a.cache, f"{a.model_tag}__*.npz"))):
        nm = os.path.basename(f).split("__")[1][:-4]
        H = npz_memmap(f, "H.npy"); y = np.load(f, allow_pickle=True)["y"]
        raw[nm] = (torch.from_numpy(np.ascontiguousarray(H[:, LS, :]).astype(np.float32)), y)
    names = sorted(raw); D = {}
    for nm in names:
        X = raw[nm][0]; y = torch.from_numpy(raw[nm][1].astype(np.int64))
        sp = {0: [], 1: [], 2: []}
        for c in (0, 1):
            ic = torch.nonzero(y == c).squeeze(1); ic = ic[torch.randperm(len(ic), generator=g)]
            n = len(ic); i1, i2 = n // 2, (3 * n) // 4
            sp[0].append(ic[:i1]); sp[1].append(ic[i1:i2]); sp[2].append(ic[i2:])
        D[nm] = dict(zip(("tr", "es", "ev"), [(X[torch.cat(sp[k])], y[torch.cat(sp[k])]) for k in range(3)]))
    del raw
    MU, SD = [], []
    for li in range(nL):
        allX = torch.cat([D[k]["tr"][0][:, li, :] for k in names]).double()
        MU.append(allX.mean(0)); SD.append(allX.std(0) + 1e-6)
    del allX
    nz = lambda X, li: (X[:, li, :].double() - MU[li]) / SD[li]
    print(f"{a.model_tag} layers {LS}: loaded {time.time()-t0:.0f}s", flush=True)
    out = dict(model=a.model_tag, layers=LS, datasets=names, methods=[], info=[])
    for B in (a.targets or names):
        tB = time.time()
        srcs = [k for k in names if (family(k) != family(B) if a.lofo else k != B)]; K = len(srcs)
        XB, yB = D[B]["ev"]; Xe, ye = D[B]["es"]
        XU = torch.cat([D[B]["tr"][0], D[B]["es"][0]])
        g3 = torch.Generator().manual_seed(1000)
        picks = [torch.randperm(len(D[k]["tr"][1]), generator=g3)[: a.lr_per_source] for k in srcs]
        ys = torch.cat([D[k]["tr"][1][ii] for k, ii in zip(srcs, picks)])
        grp = torch.cat([torch.full((len(ii),), i) for i, ii in enumerate(picks)])
        W, srcauc, OOF = [], [], torch.empty(len(ys), nL, dtype=F64)
        for li in range(nL):
            Xs = torch.cat([nz(D[k]["tr"][0][ii], li) for k, ii in zip(srcs, picks)])
            best = None
            for C in (1e-3, 1e-2):
                aucs = []; sc = torch.empty(len(ys), dtype=F64)
                for f in range(3):
                    te = (grp % 3) == f
                    wv = fit_lr(Xs[~te], ys[~te], C, iters=100); wv = wv / wv.norm()
                    sc[te] = Xs[te] @ wv
                    aucs.append(np.mean([auroc(sc[te & (grp == i)], ys[te & (grp == i)]) for i in range(K) if i % 3 == f]))
                if best is None or np.mean(aucs) > best[0]:
                    best = (float(np.mean(aucs)), C, sc)
            OOF[:, li] = best[2]
            w = fit_lr(Xs, ys, best[1]); W.append(w / w.norm()); srcauc.append(best[0])
        Zs = [OOF[grp == i] for i in range(K)]; Ys = [ys[grp == i] for i in range(K)]
        Ds, Ss = src_stats(Zs, Ys)
        m, Sig, P = moments(Ds, Ss)
        # ridge by leave-one-source-out realized AUC
        lo = {}
        for lam in LAMS:
            aa = []
            for j in range(K):
                o = [i for i in range(K) if i != j]
                mj, Sj, Pj = moments(Ds[o], Ss[o])
                al = solve(2 * Sj + Pj, mj, lam)
                aa.append(auroc(Zs[j] @ al, Ys[j]))
            lo[lam] = float(np.mean(aa))
        lam = max(lo, key=lo.get)
        A = 2 * Sig + P
        alpha = solve(A, m, lam)
        dsrc = torch.tensor([auc_to_delta(s) for s in srcauc], dtype=F64).clamp(min=0)

        def scores(Ws):
            SU = torch.stack([nz(XU, li) @ Ws[li] for li in range(nL)], 1)
            m_, s_ = SU.mean(0), SU.std(0) + 1e-12
            ZB = (torch.stack([nz(XB, li) @ Ws[li] for li in range(nL)], 1) - m_) / s_
            return ZB, s_
        ZB, _ = scores(W)
        r = {f"pool_L{LS[li]}": auroc(ZB[:, li], yB) for li in range(nL)}
        sb = int(np.argmax(srcauc))
        r["pool_srcbest"] = r[f"pool_L{LS[sb]}"]
        r["pool_uni"] = auroc(ZB.sum(1), yB)
        r["pool_dw"] = auroc(ZB @ dsrc, yB)
        r["pool_mf"] = auroc(ZB @ m, yB)
        r["pool_bo"] = auroc(ZB @ alpha, yB)
        r["pool_bo0"] = auroc(ZB @ solve(A, m, 0.03), yB)
        out["methods"].append(dict(B=B, nb=0, rep=0, **r))
        g4 = torch.Generator().manual_seed(17)
        for nb in a.nbs:
            if nb > 2 * min(int((ye == 1).sum()), int((ye == 0).sum())):
                continue
            folds = 4 if nb >= 32 else 2
            for rep in range(a.reps):
                idx = bal_sub(ye, nb, g4); yb = ye[idx]
                Ws, Oo = [], []
                for li in range(nL):
                    Xb = nz(Xe[idx], li)
                    w, _, oof = cv_oof(Xb, yb, [1e-3, 1e-2, 1e-1, 1.0], lambda X, y, C, it: fit_lr(X, y, C, w0=W[li], iters=it), folds)
                    Ws.append(w); Oo.append(oof)
                ZBf, s_ = scores(Ws)
                dO = md(torch.stack(Oo, 1) / s_, yb)
                rr = {f"lrp_L{LS[li]}": auroc(ZBf[:, li], yB) for li in range(nL)}
                rr["lrp_srcbest"] = rr[f"lrp_L{LS[sb]}"]
                rr["lrp_uni"] = auroc(ZBf.sum(1), yB)
                dmix = ((nb * dO + 32 * dsrc) / (nb + 32)).clamp(min=0)
                rr["lrp_dw"] = auroc(ZBf @ dmix, yB)
                rr["lrp_bo_zs"] = auroc(ZBf @ alpha, yB)
                # Gaussian posterior for the target's separation vector
                L_ = nL; I = torch.eye(L_, dtype=F64)
                Pr = P + 1e-3 * float(torch.trace(P)) / L_ * I
                Sr = Sig + 1e-3 * float(torch.trace(Sig)) / L_ * I
                Pinv = torch.linalg.inv(Pr); Einv = (nb / 4.0) * torch.linalg.inv(Sr)
                Ppost = torch.linalg.inv(Pinv + Einv)
                mpost = Ppost @ (Pinv @ m + Einv @ dO)
                rr["lrp_bo"] = auroc(ZBf @ solve(2 * Sig + Ppost, mpost, lam), yB)
                rr["lrp_bomf"] = auroc(ZBf @ mpost, yB)
                out["methods"].append(dict(B=B, nb=nb, rep=rep, **rr))
        out["info"].append(dict(B=B, srcauc=srcauc, lam=lam, loso=lo, alpha=alpha.tolist(), m=m.tolist(),
                                pred_bo=pred_auc(alpha, m, A), pred_sb=pred_auc(torch.eye(nL, dtype=F64)[sb], m, A)))
        print(f"  {B}: srcbest={r['pool_srcbest']:.3f} uni={r['pool_uni']:.3f} dw={r['pool_dw']:.3f} mf={r['pool_mf']:.3f} "
              f"bo={r['pool_bo']:.3f} (lam {lam}) oracle={max(r[f'pool_L{l}'] for l in LS):.3f} [{time.time()-tB:.0f}s]", flush=True)
        with open(a.out + ".partial", "w") as fh:
            json.dump(out, fh)
    with open(a.out, "w") as fh:
        json.dump(out, fh)
    print("[wrote]", a.out, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
