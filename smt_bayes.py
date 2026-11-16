"""SMT-full: Bayes-optimal transfer probes in feature space AND layer space (Theorem thm:bayes applied twice).
Feature space, per layer:  w = (2 Sigma_w + gamma P + lam t I)^{-1} m,  m = pooled mean difference,
  Sigma_w = pooled within-class covariance, P = across-source covariance of per-source mean differences
  (random-effect heterogeneity), (gamma, lam) by grouped source CV.  gamma = 0 recovers covariance-corrected
  mass-mean (mm_iid); we also keep pooled LR as a second base probe.
Layer x probe space: z in R^{2L} (both probes at each layer, target-unlabeled standardized); alpha = (2S + P_z + lam I)^{-1} m_z
  estimated from held-out source score vectors (grouped CV fold models), lam by leave-one-source-out (lam -> inf: m_z).
Few-shot: per (layer, probe) prior-anchored LR (anchor = that probe), combined with the zero-shot alpha.
Usage: python smt_bayes.py --model-tag Qwen2.5-7B-Instruct --layers 8 12 ... --out results/smt_bayes_qwen7b.json --lofo --targets ...
"""
import argparse, glob, json, os, time
import numpy as np
import torch
from smt2 import npz_memmap, auroc, md, bal_sub, F64
from smt3 import fit_lr
from smt8 import cv_oof
from smt11 import family, auc_to_delta
from smt12 import src_stats, moments, solve, LAMS

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))
GL = [(gm, lm) for gm in (0.0, 1.0, 4.0) for lm in (0.1, 1.0, 10.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--layers", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--nbs", nargs="*", type=int, default=[16, 32, 64, 128])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--lr-per-source", type=int, default=2000)
    ap.add_argument("--cov-per-source", type=int, default=500)
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
        cvauc = lambda s: float(np.mean([auroc(s[grp == i], ys[grp == i]) for i in range(K)]))
        W, OOF, CV, SEL = [], [], [], []                     # W[j] for probe-layer j = 2*li + (0 lr, 1 smt)
        for li in range(nL):
            Xs = torch.cat([nz(D[k]["tr"][0][ii], li) for k, ii in zip(srcs, picks)])
            # ---- pooled LR
            best = None
            for C in (1e-3, 1e-2):
                sc = torch.empty(len(ys), dtype=F64)
                for f in range(3):
                    te = (grp % 3) == f
                    wv = fit_lr(Xs[~te], ys[~te], C, iters=100); sc[te] = Xs[te] @ (wv / wv.norm())
                au = cvauc(sc)
                if best is None or au > best[0]:
                    best = (au, C, sc)
            w = fit_lr(Xs, ys, best[1]); w_lr = w / w.norm(); oof_lr = best[2]; cv_lr = best[0]

            # ---- SMT-full feature-space Bayes direction
            def mats(tr):
                Xt, yt, gt = Xs[tr], ys[tr], grp[tr]
                m = md(Xt, yt)
                sub = torch.arange(len(yt))
                sub = sub[(sub % max(1, len(yt) // (a.cov_per_source * K))) == 0]
                Z = Xt[sub].clone(); yy = yt[sub]
                for c in (0, 1):
                    Z[yy == c] -= Z[yy == c].mean(0)
                S = Z.T @ Z / len(Z)
                dk = torch.stack([md(Xt[gt == i], yt[gt == i]) for i in torch.unique(gt).tolist()], 1)
                dc = dk - dk.mean(1, keepdim=True)
                P = dc @ dc.T / max(dk.shape[1] - 1, 1)
                return m, S, P

            def bdir(m, S, P, gm, lm):
                A = 2 * S + gm * P
                t = float(torch.trace(A)) / A.shape[0]
                return torch.linalg.solve(A + lm * t * torch.eye(A.shape[0], dtype=F64), m)
            scs = {p: torch.empty(len(ys), dtype=F64) for p in GL}
            for f in range(3):
                te = (grp % 3) == f
                m, S, P = mats(~te)
                for p in GL:
                    wv = bdir(m, S, P, *p); scs[p][te] = Xs[te] @ (wv / wv.norm())
            cva = {p: cvauc(scs[p]) for p in GL}; ps = max(cva, key=cva.get)
            m, S, P = mats(torch.ones(len(ys), dtype=torch.bool))
            w_s = bdir(m, S, P, *ps); w_s = w_s / w_s.norm()
            w_iid = bdir(m, S, P, 0.0, max((p for p in GL if p[0] == 0.0), key=cva.get)[1]); w_iid = w_iid / w_iid.norm()
            W += [w_lr, w_s]; OOF += [oof_lr, scs[ps]]; CV += [cv_lr, cva[ps]]
            SEL.append(dict(C=best[1], gl=list(ps), cv_iid=max(cva[p] for p in GL if p[0] == 0.0)))
            del Xs
            if li == 0:
                WI = []
            WI.append(w_iid)
        J = 2 * nL
        OOFm = torch.stack(OOF, 1)
        Zs = [OOFm[grp == i] for i in range(K)]; Ys = [ys[grp == i] for i in range(K)]
        Ds, Ss = src_stats(Zs, Ys)
        mz, Sz, Pz = moments(Ds, Ss)
        lo = {}
        for lam in LAMS:
            aa = []
            for j in range(K):
                o = [i for i in range(K) if i != j]
                mj, Sj, Pj = moments(Ds[o], Ss[o])
                aa.append(auroc(Zs[j] @ solve(2 * Sj + Pj, mj, lam), Ys[j]))
            lo[lam] = float(np.mean(aa))
        lam = max(lo, key=lo.get)
        alpha = solve(2 * Sz + Pz, mz, lam)
        lr_idx = list(range(0, J, 2)); sm_idx = list(range(1, J, 2))
        # layer-only BO for the LR probes (= smt12 pool_bo) for reference
        def bo_sub(idx):
            Ds_, Ss_ = Ds[:, idx], Ss[:, idx][:, :, idx]
            lo_ = {}
            for lam_ in LAMS:
                aa = []
                for j in range(K):
                    o = [i for i in range(K) if i != j]
                    mj, Sj, Pj = moments(Ds_[o], Ss_[o])
                    aa.append(auroc(Zs[j][:, idx] @ solve(2 * Sj + Pj, mj, lam_), Ys[j]))
                lo_[lam_] = float(np.mean(aa))
            l_ = max(lo_, key=lo_.get); m_, S_, P_ = moments(Ds_, Ss_)
            return solve(2 * S_ + P_, m_, l_)
        a_sm = bo_sub(sm_idx)

        def scores(Ws):
            SU = torch.stack([nz(XU, j // 2) @ Ws[j] for j in range(len(Ws))], 1)
            m_, s_ = SU.mean(0), SU.std(0) + 1e-12
            ZB = (torch.stack([nz(XB, j // 2) @ Ws[j] for j in range(len(Ws))], 1) - m_) / s_
            return ZB, s_
        ZB, _ = scores(W)
        r = {}
        for li in range(nL):
            r[f"lr_L{LS[li]}"] = auroc(ZB[:, 2 * li], yB); r[f"smt_L{LS[li]}"] = auroc(ZB[:, 2 * li + 1], yB)
            r[f"iid_L{LS[li]}"] = auroc(nz(XB, li) @ WI[li], yB)
        cvl = [CV[j] for j in lr_idx]; cvs_ = [CV[j] for j in sm_idx]; cvi = [s["cv_iid"] for s in SEL]
        r["lr_srcbest"] = r[f"lr_L{LS[int(np.argmax(cvl))]}"]
        r["smt_srcbest"] = r[f"smt_L{LS[int(np.argmax(cvs_))]}"]
        r["iid_srcbest"] = r[f"iid_L{LS[int(np.argmax(cvi))]}"]
        jb = int(np.argmax(CV)); r["both_srcbest"] = auroc(ZB[:, jb], yB)
        r["smt_bo"] = auroc(ZB[:, sm_idx] @ a_sm, yB)
        r["full_bo"] = auroc(ZB @ alpha, yB)
        r["full_mf"] = auroc(ZB @ mz, yB)
        out["methods"].append(dict(B=B, nb=0, rep=0, **r))
        g4 = torch.Generator().manual_seed(17)
        for nb in a.nbs:
            if nb > 2 * min(int((ye == 1).sum()), int((ye == 0).sum())):
                continue
            folds = 4 if nb >= 32 else 2
            for rep in range(a.reps):
                idx = bal_sub(ye, nb, g4); yb = ye[idx]
                Ws = []
                for j in range(J):
                    Xb = nz(Xe[idx], j // 2)
                    w, _, _ = cv_oof(Xb, yb, [1e-3, 1e-2, 1e-1, 1.0], lambda X, y, C, it: fit_lr(X, y, C, w0=W[j], iters=it), folds)
                    Ws.append(w)
                ZBf, _ = scores(Ws)
                rr = {}
                rr["lrp_srcbest"] = auroc(ZBf[:, 2 * int(np.argmax(cvl))], yB)
                rr["smtp_srcbest"] = auroc(ZBf[:, 2 * int(np.argmax(cvs_)) + 1], yB)
                rr["both_srcbest"] = auroc(ZBf[:, jb], yB)
                rr["smtp_bo"] = auroc(ZBf[:, sm_idx] @ a_sm, yB)
                rr["full_bo"] = auroc(ZBf @ alpha, yB)
                out["methods"].append(dict(B=B, nb=nb, rep=rep, **rr))
        out["info"].append(dict(B=B, sel=SEL, lam=lam, alpha=alpha.tolist(), cv=CV))
        print(f"  {B}: lr_sb={r['lr_srcbest']:.3f} iid_sb={r['iid_srcbest']:.3f} smt_sb={r['smt_srcbest']:.3f} smt_bo={r['smt_bo']:.3f} "
              f"full_bo={r['full_bo']:.3f} gl={[s['gl'] for s in SEL]} [{time.time()-tB:.0f}s]", flush=True)
        with open(a.out + ".partial", "w") as fh:
            json.dump(out, fh)
    with open(a.out, "w") as fh:
        json.dump(out, fh)
    print("[wrote]", a.out, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
