"""SMT-Bayes: Bayes-optimal transfer probes in feature space and across layers.
Leave-one-family-out evaluation of SMT-Bayes against baselines on identical splits. Besides SMT-Bayes, it reports:
  zero-shot : stacked LR over the 2L held-out source scores (multi-layer baseline), uniform layer average of
              MM-iid, CORAL-style target-covariance variants (target unlabeled covariance in place of the pooled
              source covariance), an MDA-style between-domain class-mean scatter penalty, the selected (gamma,
              lambda) per layer, a binormal check of the SMT-Bayes score, and the diversity curve
              (AUROC of uniform mass-mean / MM-iid probes built from K randomly chosen source families).
  few-shot  : closed-form posterior probe (Eq. posterior + Theorem bayes) at the source-selected layer, and
              transfer-law optimal source weights (Theorem opt) with the plug-in Gram estimator.
Extra targets that must never act as sources (e.g. self-generated answers) are read from --target-cache.
"""
import argparse, glob, json, os, time, zlib
import numpy as np
import torch
from scipy.stats import norm
from smt2 import npz_memmap, auroc, md, bal_sub, F64
from smt3 import fit_lr
from smt8 import cv_oof
from smt11 import family
from smt12 import src_stats, moments, solve, LAMS

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))
GL = [(gm, lm) for gm in (0.0, 1.0, 4.0) for lm in (0.1, 1.0, 10.0)]
GB = [(gm, lm) for gm in (1.0, 4.0) for lm in (0.1, 1.0, 10.0)]
KS = (1, 2, 4, 8, 16)


def split(X, y, g):
    sp = {0: [], 1: [], 2: []}
    for c in (0, 1):
        ic = torch.nonzero(y == c).squeeze(1); ic = ic[torch.randperm(len(ic), generator=g)]
        n = len(ic); i1, i2 = n // 2, (3 * n) // 4
        sp[0].append(ic[:i1]); sp[1].append(ic[i1:i2]); sp[2].append(ic[i2:])
    return dict(zip(("tr", "es", "ev"), [(X[torch.cat(sp[k])], y[torch.cat(sp[k])]) for k in range(3)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--target-cache", default=None, help="extra target-only datasets (never used as sources)")
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--layers", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--nbs", nargs="*", type=int, default=[16, 32, 64, 128])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--lr-per-source", type=int, default=2000)
    ap.add_argument("--cov-per-source", type=int, default=500)
    ap.add_argument("--div-draws", type=int, default=10)
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
        D[nm] = split(raw[nm][0], torch.from_numpy(raw[nm][1].astype(np.int64)), g)
    del raw
    extra = []
    if a.target_cache:
        ge = torch.Generator().manual_seed(7)
        for f in sorted(glob.glob(os.path.join(a.target_cache, f"{a.model_tag}__*.npz"))):
            nm = os.path.basename(f).split("__")[1][:-4]
            H = npz_memmap(f, "H.npy"); y = np.load(f, allow_pickle=True)["y"]
            D[nm] = split(torch.from_numpy(np.ascontiguousarray(H[:, LS, :]).astype(np.float32)),
                          torch.from_numpy(y.astype(np.int64)), ge)
            extra.append(nm)
    MU, SD = [], []
    for li in range(nL):
        allX = torch.cat([D[k]["tr"][0][:, li, :] for k in names]).double()
        MU.append(allX.mean(0)); SD.append(allX.std(0) + 1e-6)
    del allX
    nz = lambda X, li: (X[:, li, :].double() - MU[li]) / SD[li]
    print(f"{a.model_tag} layers {LS}: loaded {time.time()-t0:.0f}s extra={extra}", flush=True)
    out = dict(model=a.model_tag, layers=LS, datasets=names, extra=extra, methods=[], info=[])
    for B in (a.targets or names):
        tB = time.time()
        srcs = [k for k in names if (family(k) != family(B) if a.lofo else k != B)]; K = len(srcs)
        fams = sorted(set(family(k) for k in srcs))
        XB, yB = D[B]["ev"]; Xe, ye = D[B]["es"]
        XU = torch.cat([D[B]["tr"][0], D[B]["es"][0]])
        g3 = torch.Generator().manual_seed(1000)
        picks = [torch.randperm(len(D[k]["tr"][1]), generator=g3)[: a.lr_per_source] for k in srcs]
        ys = torch.cat([D[k]["tr"][1][ii] for k, ii in zip(srcs, picks)])
        grp = torch.cat([torch.full((len(ii),), i) for i, ii in enumerate(picks)])
        cvauc = lambda s: float(np.mean([auroc(s[grp == i], ys[grp == i]) for i in range(K)]))
        W, OOF, CV, SEL, WI, OOFI, LAY = [], [], [], [], [], [], []
        for li in range(nL):
            Xs = torch.cat([nz(D[k]["tr"][0][ii], li) for k, ii in zip(srcs, picks)])
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

            def mats(tr):
                Xt, yt, gt = Xs[tr], ys[tr], grp[tr]
                m = md(Xt, yt)
                sub = torch.arange(len(yt))
                sub = sub[(sub % max(1, len(yt) // (a.cov_per_source * K))) == 0]
                Z = Xt[sub].clone(); yy = yt[sub]
                for c in (0, 1):
                    Z[yy == c] -= Z[yy == c].mean(0)
                S = Z.T @ Z / len(Z)
                gi = torch.unique(gt).tolist()
                dk = torch.stack([md(Xt[gt == i], yt[gt == i]) for i in gi], 1)
                dc = dk - dk.mean(1, keepdim=True)
                P = dc @ dc.T / max(dk.shape[1] - 1, 1)
                mu = torch.stack([Xt[(gt == i) & (yt == c)].mean(0) for i in gi for c in (0, 1)], 1)   # d x 2K class means
                cls = torch.tensor([c for i in gi for c in (0, 1)])
                Bm = torch.zeros_like(S)
                for c in (0, 1):
                    mc = mu[:, cls == c]; mc = mc - mc.mean(1, keepdim=True); Bm += mc @ mc.T / max(mc.shape[1] - 1, 1)
                return m, S, P, Bm, dk, gi

            def bdir(m, S, P, gm, lm):
                A = 2 * S + gm * P
                t = float(torch.trace(A)) / A.shape[0]
                return torch.linalg.solve(A + lm * t * torch.eye(A.shape[0], dtype=F64), m)
            scs = {p: torch.empty(len(ys), dtype=F64) for p in GL}
            scb = {p: torch.empty(len(ys), dtype=F64) for p in GB}
            for f in range(3):
                te = (grp % 3) == f
                m, S, P, Bm, _, _ = mats(~te)
                for p in GL:
                    wv = bdir(m, S, P, *p); scs[p][te] = Xs[te] @ (wv / wv.norm())
                for p in GB:
                    wv = bdir(m, S, Bm, *p); scb[p][te] = Xs[te] @ (wv / wv.norm())
            cva = {p: cvauc(scs[p]) for p in GL}; ps = max(cva, key=cva.get)
            cvb = {p: cvauc(scb[p]) for p in GB}; pb = max(cvb, key=cvb.get)
            pi_ = max((p for p in GL if p[0] == 0.0), key=cva.get)
            m, S, P, Bm, dk, gi = mats(torch.ones(len(ys), dtype=torch.bool))
            w_s = bdir(m, S, P, *ps); w_s = w_s / w_s.norm()
            w_iid = bdir(m, S, P, *pi_); w_iid = w_iid / w_iid.norm()
            w_mda = bdir(m, S, Bm, *pb); w_mda = w_mda / w_mda.norm()
            # CORAL-style: target unlabeled covariance in place of the pooled source within-class covariance
            ZU = nz(XU, li); ZU = ZU - ZU.mean(0); SU_ = ZU.T @ ZU / (len(ZU) - 1)
            w_coral = bdir(m, SU_, P, 0.0, pi_[1]); w_coral = w_coral / w_coral.norm()
            w_stgt = bdir(m, SU_, P, *ps); w_stgt = w_stgt / w_stgt.norm()
            W += [w_lr, w_s]; OOF += [oof_lr, scs[ps]]; CV += [cv_lr, cva[ps]]; WI.append(w_iid); OOFI.append(scs[pi_])
            SEL.append(dict(C=best[1], gl=list(ps), iid_lam=pi_[1], mda=list(pb), cv_iid=cva[pi_], cv_mda=cvb[pb]))
            LAY.append(dict(m=m, S=S, P=P, gm=ps[0], lm=ps[1], lmi=pi_[1], dk=dk, w_mda=w_mda, w_coral=w_coral, w_stgt=w_stgt))
            del Xs
        J = 2 * nL
        OOFm = torch.stack(OOF, 1)
        Zs = [OOFm[grp == i] for i in range(K)]; Ys = [ys[grp == i] for i in range(K)]
        Ds, Ss = src_stats(Zs, Ys)

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
            return solve(2 * S_ + P_, m_, l_), l_
        alpha, lam = bo_sub(list(range(J)))
        sm_idx = list(range(1, J, 2)); lr_idx = list(range(0, J, 2))
        a_sm, _ = bo_sub(sm_idx)
        # stacked logistic regression on per-source standardized held-out scores (multi-layer baseline)
        Zst = torch.cat([(z - z.mean(0)) / (z.std(0) + 1e-12) for z in Zs]); Yst = torch.cat(Ys)
        b_stack = fit_lr(Zst, Yst, 1.0)
        b_stack_lr = fit_lr(Zst[:, lr_idx], Yst, 1.0)

        def scores(Ws, lay):
            SU = torch.stack([nz(XU, l) @ w for w, l in zip(Ws, lay)], 1)
            m_, s_ = SU.mean(0), SU.std(0) + 1e-12
            Z = (torch.stack([nz(XB, l) @ w for w, l in zip(Ws, lay)], 1) - m_) / s_
            return Z, s_
        layJ = [j // 2 for j in range(J)]
        ZB, _ = scores(W, layJ)
        ZI, _ = scores(WI, list(range(nL)))
        r = {}
        cvl = [CV[j] for j in lr_idx]; cvs_ = [CV[j] for j in sm_idx]; cvi = [s["cv_iid"] for s in SEL]; cvm = [s["cv_mda"] for s in SEL]
        lb_lr, lb_s, lb_i, lb_m = int(np.argmax(cvl)), int(np.argmax(cvs_)), int(np.argmax(cvi)), int(np.argmax(cvm))
        r["lr_srcbest"] = auroc(ZB[:, 2 * lb_lr], yB)
        r["smt_srcbest"] = auroc(ZB[:, 2 * lb_s + 1], yB)
        r["iid_srcbest"] = auroc(ZI[:, lb_i], yB)
        r["full_bo"] = auroc(ZB @ alpha, yB)
        r["smt_bo"] = auroc(ZB[:, sm_idx] @ a_sm, yB)
        r["stack_lr"] = auroc(ZB @ b_stack, yB)
        r["stack_lronly"] = auroc(ZB[:, lr_idx] @ b_stack_lr, yB)
        r["iid_uni"] = auroc(ZI.sum(1), yB)
        r["lr_uni"] = auroc(ZB[:, lr_idx].sum(1), yB)
        r["mda_srcbest"] = auroc(nz(XB, lb_m) @ LAY[lb_m]["w_mda"], yB)
        r["coral_iidlayer"] = auroc(nz(XB, lb_i) @ LAY[lb_i]["w_coral"], yB)
        r["smt_tgtcov"] = auroc(nz(XB, lb_s) @ LAY[lb_s]["w_stgt"], yB)
        sB = ZB @ alpha; m1, m0 = sB[yB == 1], sB[yB == 0]
        r["binormal_pred"] = float(norm.cdf(float(m1.mean() - m0.mean()) / float(np.sqrt(m1.var() + m0.var()))))
        # diversity curve at the MM-iid layer (Theorem div): uniform mean-difference probes from K random families
        Ld = LAY[lb_i]; ZBd = nz(XB, lb_i)
        A_ = 2 * Ld["S"]; t_ = float(torch.trace(A_)) / A_.shape[0]
        Ainv = torch.linalg.inv(A_ + Ld["lmi"] * t_ * torch.eye(A_.shape[0], dtype=F64))
        famof = [family(srcs[i]) for i in range(K)]
        rngd = np.random.default_rng(zlib.crc32(B.encode()))
        div = {}
        for Kf in list(KS) + [len(fams)]:
            if Kf > len(fams):
                continue
            mm, ii_ = [], []
            for _ in range(a.div_draws if Kf < len(fams) else 1):
                ch = set(rngd.choice(fams, Kf, replace=False).tolist())
                cols = [i for i in range(K) if famof[i] in ch]
                u = Ld["dk"][:, cols].mean(1)
                mm.append(auroc(ZBd @ u, yB)); ii_.append(auroc(ZBd @ (Ainv @ u), yB))
            div[str(Kf)] = [float(np.mean(mm)), float(np.mean(ii_))]
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
                    Xb = nz(Xe[idx], layJ[j])
                    w, _, _ = cv_oof(Xb, yb, [1e-3, 1e-2, 1e-1, 1.0], lambda X, y, C, it: fit_lr(X, y, C, w0=W[j], iters=it), folds)
                    Ws.append(w)
                ZBf, _ = scores(Ws, layJ)
                rr = {"lrp_srcbest": auroc(ZBf[:, 2 * lb_lr], yB), "smtp_srcbest": auroc(ZBf[:, 2 * lb_s + 1], yB),
                      "full_bo": auroc(ZBf @ alpha, yB), "smtp_bo": auroc(ZBf[:, sm_idx] @ a_sm, yB)}
                # closed-form posterior probe at the SMT layer
                L_ = LAY[lb_s]; Xbn = nz(Xe[idx], lb_s); dB = md(Xbn, yb)
                S, P, m = L_["S"], L_["P"], L_["m"]
                A0 = 2 * S + L_["gm"] * P; tt = float(torch.trace(A0)) / A0.shape[0]
                Pi = L_["gm"] * P + L_["lm"] * tt * torch.eye(S.shape[0], dtype=F64)
                O = (4.0 / nb) * S
                Kg = torch.linalg.solve(Pi + O, Pi).T                     # Pi (Pi+O)^-1 (symmetric matrices)
                mpost = m + Kg @ (dB - m); Ppost = Pi - Kg @ Pi
                wpost = torch.linalg.solve(2 * S + Ppost, mpost)
                rr["post_srcbest"] = auroc(nz(XB, lb_s) @ wpost, yB)
                # transfer-law optimal source weights (Theorem opt) with plug-in Gram on target unlabeled covariance
                Li = LAY[lb_i]; Xbi = nz(Xe[idx], lb_i); dBi = md(Xbi, yb)
                Dk = Li["dk"]; ZUi = nz(XU, lb_i); ZUi = ZUi - ZUi.mean(0)
                PU = ZUi @ Dk; Agr = PU.T @ PU / (len(ZUi) - 1)
                Mh = Dk.T @ dBi
                al = torch.linalg.solve(Agr + 1e-6 * float(torch.trace(Agr)) / Agr.shape[0] * torch.eye(Agr.shape[0], dtype=F64), Mh)
                rr["thm3_opt"] = auroc(nz(XB, lb_i) @ (Dk @ al), yB)
                rr["thm3_uni"] = auroc(nz(XB, lb_i) @ Dk.mean(1), yB)
                out["methods"].append(dict(B=B, nb=nb, rep=rep, **rr))
        out["info"].append(dict(B=B, sel=SEL, lam=lam, div=div, nfam=len(fams), K=K,
                                lb=dict(lr=LS[lb_lr], smt=LS[lb_s], iid=LS[lb_i], mda=LS[lb_m])))
        print(f"  {B}: full_bo={r['full_bo']:.3f} iid={r['iid_srcbest']:.3f} stack={r['stack_lr']:.3f} mda={r['mda_srcbest']:.3f} "
              f"coral={r['coral_iidlayer']:.3f} tgtcov={r['smt_tgtcov']:.3f} gl={[s['gl'] for s in SEL]} div={div} [{time.time()-tB:.0f}s]", flush=True)
        with open(a.out + ".partial", "w") as fh:
            json.dump(out, fh)
    with open(a.out, "w") as fh:
        json.dump(out, fh)
    print("[wrote]", a.out, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
