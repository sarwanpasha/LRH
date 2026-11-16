"""ML-SMT: multi-layer second-moment transfer = per-layer EB-LR probes combined with separation-proportional
weights (no covariance inversion).  Usage:
  python smt11.py --cache cache --model-tag Qwen2.5-7B-Instruct --layers 8 12 16 20 24 28 --out results11/qw_c0.json --targets ...
For standardized layer scores with highly correlated within-class noise, the Thm-3 weights C^{-1}Delta are
ill-conditioned; under exchangeable noise (equicorrelation r -> 1 limit with equal variances) the optimal
direction within the span orthogonal to the common mode is Delta - mean(Delta), and with estimation error the
risk-minimizing shrinkage of C to the identity gives alpha propto Delta (matched filter).  Delta_l is estimated
from source held-out AUCs (zero-shot, via Delta = sqrt(2 p^2/(1 + p^2/2)), p = probit(AUC), the binormal +
Sherman-Morrison identity for unit-variance scores), shrunk toward out-of-fold labelled estimates (few-shot).
"""
import argparse, glob, json, os, time
import numpy as np
import torch
from scipy.stats import norm
from smt2 import npz_memmap, auroc, md, bal_sub, F64, novelty_rho
from smt3 import fit_lr
from smt8 import cv_oof

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))


def auc_to_delta(a):
    p = norm.ppf(min(max(a, 1e-4), 1 - 1e-4))
    return float(np.sign(p) * np.sqrt(2 * p * p / (1 + p * p / 2)))


def family(n):
    b = n.replace("neg_", "").replace("_de", "").replace("_conj", "").replace("_disj", "")
    if b in ("larger_than", "smaller_than"):
        return "numbers"
    if b.startswith("truthfulqa"):
        return "truthfulqa"
    if b.startswith("halueval"):
        return "halueval"
    return b


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
    ap.add_argument("--per-source", type=int, default=1000)
    ap.add_argument("--lofo", action="store_true", help="leave-one-family-out: drop all sources of the target's family")
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
    g2 = torch.Generator().manual_seed(1)
    Uk = {}                                                                 # per-source unit LR directions per layer
    for k in names:
        X, y = D[k]["tr"]; idx = bal_sub(y, min(a.per_source, len(y)), g2)
        Uk[k] = []
        for li in range(nL):
            w = fit_lr(nz(X[idx], li), y[idx], 0.1); Uk[k].append(w / w.norm())
    print(f"{a.model_tag} layers {LS}: loaded + per-source LR {time.time()-t0:.0f}s", flush=True)
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
        L = []                                                              # per-layer state
        for li in range(nL):
            Xs = torch.cat([nz(D[k]["tr"][0][ii], li) for k, ii in zip(srcs, picks)])
            best = None
            for C in (1e-3, 1e-2):
                aucs = []
                for f in range(3):
                    te = (grp % 3) == f
                    wv = fit_lr(Xs[~te], ys[~te], C, iters=100)
                    aucs.append(np.mean([auroc(Xs[te & (grp == i)] @ wv, ys[te & (grp == i)]) for i in range(K) if i % 3 == f]))
                if best is None or np.mean(aucs) > best[0]:
                    best = (float(np.mean(aucs)), C)
            w_pool = fit_lr(Xs, ys, best[1]); w_pool = w_pool / w_pool.norm()
            Wd = torch.stack([Uk[k][li] for k in srcs], 1); s = Wd.mean(1)
            Qf, sv, _ = torch.linalg.svd(Wd - s[:, None], full_matrices=False)
            v = sv ** 2 / (K - 1); kp = v > 1e-12 * float(v.max()); Q, v = Qf[:, kp], v[kp]
            rho = max(novelty_rho(Wd, torch.zeros(K, dtype=F64)), 1e-12); dq = v / (v + rho)
            vbar = float(v.mean()); bg = None
            for gm in (0.0, 0.1, 1.0, 10.0, 100.0):
                aucs = []
                for j in range(K):
                    o = [i for i in range(K) if i != j]; sj = Wd[:, o].mean(1)
                    Qj, svj, _ = torch.linalg.svd(Wd[:, o] - sj[:, None], full_matrices=False)
                    vj = svj ** 2 / (K - 2); gj = gm / vbar
                    bj = sj - Qj @ ((gj * vj / (1 + gj * vj)) * (Qj.T @ sj))
                    Xj, yj = D[srcs[j]]["ev"]; aucs.append(auroc(nz(Xj, li) @ bj, yj))
                if bg is None or np.mean(aucs) > bg[0]:
                    bg = (float(np.mean(aucs)), gm)
            gam = bg[1] / vbar
            w0 = w_pool - Q @ ((gam * v / (1 + gam * v)) * (Q.T @ w_pool)); w0 = w0 / w0.norm()
            L.append(dict(w_pool=w_pool, w0=w0, Q=Q, dq=dq, srcauc=best[0]))
        dsrc = torch.tensor([auc_to_delta(St["srcauc"]) for St in L], dtype=F64).clamp(min=0)

        def scores(Ws):
            SU = torch.stack([nz(XU, li) @ Ws[li] for li in range(nL)], 1)
            m_, s_ = SU.mean(0), SU.std(0) + 1e-12
            ZB = (torch.stack([nz(XB, li) @ Ws[li] for li in range(nL)], 1) - m_) / s_
            return ZB, s_
        r = {}
        for name, key in (("pool", "w_pool"), ("eb", "w0")):
            ZB, _ = scores([St[key] for St in L])
            for li in range(nL):
                r[f"{name}_L{LS[li]}"] = auroc(ZB[:, li], yB)
            r[f"{name}_srcbest"] = r[f"{name}_L{LS[int(np.argmax([St['srcauc'] for St in L]))]}"]
            r[f"{name}_uni"] = auroc(ZB.sum(1), yB)
            r[f"{name}_dw"] = auroc(ZB @ dsrc, yB)
        out["methods"].append(dict(B=B, nb=0, rep=0, **r))
        print(f"  {B}: pool srcbest={r['pool_srcbest']:.3f} uni={r['pool_uni']:.3f} dw={r['pool_dw']:.3f} | eb uni={r['eb_uni']:.3f} "
              f"dw={r['eb_dw']:.3f} | oracle={max(r[f'pool_L{l}'] for l in LS):.3f} [{time.time()-tB:.0f}s]", flush=True)
        g4 = torch.Generator().manual_seed(17)
        for nb in a.nbs:
            if nb > 2 * min(int((ye == 1).sum()), int((ye == 0).sum())):
                continue
            folds = 4 if nb >= 32 else 2
            for rep in range(a.reps):
                idx = bal_sub(ye, nb, g4); yb = ye[idx]
                rr = {}
                for name in ("lrp", "ebw"):
                    Ws, Oo = [], []
                    for li in range(nL):
                        Xb = nz(Xe[idx], li); St = L[li]
                        if name == "lrp":
                            w, _, oof = cv_oof(Xb, yb, [1e-3, 1e-2, 1e-1, 1.0], lambda X, y, C, it: fit_lr(X, y, C, w0=St["w_pool"], iters=it), folds)
                        else:
                            grid = [(C, eta) for C in (1e-3, 1e-2, 1e-1) for eta in (0.0, 0.5)]
                            w, _, oof = cv_oof(Xb, yb, grid, lambda X, y, p, it: fit_lr(X, y, p[0], w0=St["w0"], Q=St["Q"], dq=p[1] * St["dq"], iters=it), folds)
                        Ws.append(w); Oo.append(oof)
                    ZB, s_ = scores(Ws)
                    dO = md(torch.stack(Oo, 1) / s_, yb)
                    for li in range(nL):
                        rr[f"{name}_L{LS[li]}"] = auroc(ZB[:, li], yB)
                    rr[f"{name}_srcbest"] = rr[f"{name}_L{LS[int(np.argmax([St['srcauc'] for St in L]))]}"]
                    rr[f"{name}_uni"] = auroc(ZB.sum(1), yB)
                    rr[f"{name}_dws"] = auroc(ZB @ dsrc, yB)
                    dmix = ((nb * dO + 32 * dsrc) / (nb + 32)).clamp(min=0)
                    rr[f"{name}_dw"] = auroc(ZB @ dmix, yB)
                out["methods"].append(dict(B=B, nb=nb, rep=rep, **rr))
        out["info"].append(dict(B=B, srcauc=[St["srcauc"] for St in L]))
        with open(a.out + ".partial", "w") as fh:
            json.dump(out, fh)
    with open(a.out, "w") as fh:
        json.dump(out, fh)
    print("[wrote]", a.out, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
