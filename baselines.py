"""Published-baseline suite under the exact results16 protocol (LOFO, same splits, same source subsamples).
Zero-shot (layer chosen per method by grouped source-CV AUC, as for pooled LR):
  mm      : pooled mass-mean difference (Marks & Tegmark 2023)
  mm_iid  : covariance-corrected mass-mean (Sigma_w + lam I)^{-1} mm, lam by grouped source CV
  ttpd    : TTPD general truth direction (Buerger et al. 2024): per-dataset centering, OLS on [tau, tau*polarity]
Few-shot (target labels only; layer chosen by out-of-fold AUC on the n_B labels):
  in_lr, in_mm
Usage: python baselines.py --model-tag Qwen2.5-7B-Instruct --layers 8 12 ... --out results/baselines_qwen7b.json --lofo --targets ...
"""
import argparse, glob, json, os, time
import numpy as np
import torch
from smt2 import npz_memmap, auroc, md, bal_sub, F64, polarity
from smt3 import fit_lr
from smt8 import cv_oof
from smt11 import family

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))


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
    out = dict(model=a.model_tag, layers=LS, datasets=names, methods=[])
    for B in (a.targets or names):
        tB = time.time()
        srcs = [k for k in names if (family(k) != family(B) if a.lofo else k != B)]; K = len(srcs)
        XB, yB = D[B]["ev"]; Xe, ye = D[B]["es"]
        g3 = torch.Generator().manual_seed(1000)
        picks = [torch.randperm(len(D[k]["tr"][1]), generator=g3)[: a.lr_per_source] for k in srcs]
        ys = torch.cat([D[k]["tr"][1][ii] for k, ii in zip(srcs, picks)])
        grp = torch.cat([torch.full((len(ii),), i) for i, ii in enumerate(picks)])
        pol = torch.tensor([polarity(k) for k in srcs], dtype=F64)[grp]
        tau = (2 * ys - 1).double()
        r = {}; cvs = {"mm": [], "mm_iid": [], "ttpd": []}
        for li in range(nL):
            Xs = torch.cat([nz(D[k]["tr"][0][ii], li) for k, ii in zip(srcs, picks)])
            Xc = Xs.clone()                                                    # per-dataset centered (TTPD)
            for i in range(K):
                Xc[grp == i] -= Xc[grp == i].mean(0)

            def dirs(tr):
                w_mm = md(Xs[tr], ys[tr])
                F = torch.stack([tau[tr], tau[tr] * pol[tr]], 1)
                TT = torch.linalg.solve(F.T @ F + 1e-9 * torch.eye(2, dtype=F64), F.T @ Xc[tr])
                # within-class covariance on a subsample
                sub = torch.nonzero(tr).squeeze(1)
                sub = sub[(torch.arange(len(sub)) % max(1, len(sub) // (a.cov_per_source * K))) == 0]
                Z = Xs[sub].clone(); yy = ys[sub]
                for c in (0, 1):
                    Z[yy == c] -= Z[yy == c].mean(0)
                S = Z.T @ Z / len(Z)
                return w_mm, TT[0], S
            res = {"mm": {}, "mm_iid": {}, "ttpd": {}}
            sc = {"mm": torch.empty(len(ys), dtype=F64), "ttpd": torch.empty(len(ys), dtype=F64)}
            lams = (0.1, 1.0, 10.0)
            sci = {l_: torch.empty(len(ys), dtype=F64) for l_ in lams}
            for f in range(3):
                te = (grp % 3) == f
                w_mm, w_tt, S = dirs(~te)
                sc["mm"][te] = Xs[te] @ w_mm; sc["ttpd"][te] = Xs[te] @ w_tt
                t = float(torch.trace(S)) / S.shape[0]
                for l_ in lams:
                    sci[l_][te] = Xs[te] @ torch.linalg.solve(S + l_ * t * torch.eye(S.shape[0], dtype=F64), w_mm)
            cvauc = lambda s: float(np.mean([auroc(s[grp == i], ys[grp == i]) for i in range(K)]))
            lam_auc = {l_: cvauc(sci[l_]) for l_ in lams}; lam = max(lam_auc, key=lam_auc.get)
            cvs["mm"].append(cvauc(sc["mm"])); cvs["ttpd"].append(cvauc(sc["ttpd"])); cvs["mm_iid"].append(lam_auc[lam])
            w_mm, w_tt, S = dirs(torch.ones(len(ys), dtype=torch.bool))
            t = float(torch.trace(S)) / S.shape[0]
            w_iid = torch.linalg.solve(S + lam * t * torch.eye(S.shape[0], dtype=F64), w_mm)
            ZB = nz(XB, li)
            r[f"mm_L{LS[li]}"] = auroc(ZB @ w_mm, yB)
            r[f"mm_iid_L{LS[li]}"] = auroc(ZB @ w_iid, yB)
            r[f"ttpd_L{LS[li]}"] = auroc(ZB @ w_tt, yB)
        for m_ in ("mm", "mm_iid", "ttpd"):
            r[f"{m_}_srcbest"] = r[f"{m_}_L{LS[int(np.argmax(cvs[m_]))]}"]
        out["methods"].append(dict(B=B, nb=0, rep=0, **r))
        g4 = torch.Generator().manual_seed(17)
        for nb in a.nbs:
            if nb > 2 * min(int((ye == 1).sum()), int((ye == 0).sum())):
                continue
            folds = 4 if nb >= 32 else 2
            for rep in range(a.reps):
                idx = bal_sub(ye, nb, g4); yb = ye[idx]
                rr = {}; best_lr = best_mm = None
                for li in range(nL):
                    Xb = nz(Xe[idx], li)
                    w, _, oof = cv_oof(Xb, yb, [1e-3, 1e-2, 1e-1, 1.0], lambda X, y, C, it: fit_lr(X, y, C, iters=it), folds)
                    a_lr = auroc(oof, yb); rr[f"in_lr_L{LS[li]}"] = auroc(nz(XB, li) @ w, yB)
                    if best_lr is None or a_lr > best_lr[0]:
                        best_lr = (a_lr, li)
                    # in_mm OOF
                    gcv = torch.Generator().manual_seed(rep)
                    fold = torch.empty(nb, dtype=torch.long)
                    for c in (0, 1):
                        ic = torch.nonzero(yb == c).squeeze(1); ic = ic[torch.randperm(len(ic), generator=gcv)]
                        fold[ic] = torch.arange(len(ic)) % folds
                    so = torch.empty(nb, dtype=F64)
                    for f in range(folds):
                        so[fold == f] = Xb[fold == f] @ md(Xb[fold != f], yb[fold != f])
                    a_mm = auroc(so, yb); rr[f"in_mm_L{LS[li]}"] = auroc(nz(XB, li) @ md(Xb, yb), yB)
                    if best_mm is None or a_mm > best_mm[0]:
                        best_mm = (a_mm, li)
                rr["in_lr"] = rr[f"in_lr_L{LS[best_lr[1]]}"]; rr["in_mm"] = rr[f"in_mm_L{LS[best_mm[1]]}"]
                out["methods"].append(dict(B=B, nb=nb, rep=rep, **rr))
        print(f"  {B}: mm={r['mm_srcbest']:.3f} mm_iid={r['mm_iid_srcbest']:.3f} ttpd={r['ttpd_srcbest']:.3f} [{time.time()-tB:.0f}s]", flush=True)
        with open(a.out + ".partial", "w") as fh:
            json.dump(out, fh)
    with open(a.out, "w") as fh:
        json.dump(out, fh)
    print("[wrote]", a.out, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
