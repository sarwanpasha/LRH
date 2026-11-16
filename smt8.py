"""EB-LR: empirical-Bayes transfer in discriminative weight space (vs LR baselines).
Usage: python smt8.py --cache cache --model-tag Qwen2.5-7B-Instruct --layer 16 --out results8/qw_L16_c0.json --targets ...

Under a shared within-class covariance, the Bayes/LR weight of dataset k is w_k = Sigma^{-1} delta_k, so the
random-effects model delta_k = s + a_k induces w_k = Sigma^{-1}s + Sigma^{-1}a_k: weight directions are
random effects around a shared direction with low-rank covariance V_w (+ rho I). We estimate them from
per-source logistic-regression directions (unit norm, standardized features) and use them as
  zero-shot: w0 = (I + gamma V_w)^{-1} b    (diversity-penalized; b = pooled-LR direction; gamma by LOSO)
  few-shot : MAP logistic regression with prior w ~ N(alpha w0, Pi), Pi = Q diag(v) Q^T + rho I, i.e. penalty
             (1/2Cn) [ ||u||^2 - sum_i eta d_i (q_i^T u)^2 ],  d_i = v_i/(v_i+rho),   w = alpha w0 + u
             (alpha unpenalized; C, eta by stratified CV on the n_B target labels, same protocol as lr_prior).
eta=0, gamma=0 recovers prior-anchored LR (lr_prior); the structured part lets target labels move the probe
freely only along directions on which source datasets disagree.
"""
import argparse, glob, json, os, math, time
import numpy as np
import torch
from smt2 import npz_memmap, auroc, md, within, bal_sub, F64, novelty_rho

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))


def fit_lr(X, y, C, w0=None, Q=None, dq=None, iters=200):
    X = X.double(); yy = y.double(); n, d = X.shape
    w = torch.zeros(d, dtype=F64, requires_grad=True)
    b = torch.zeros(1, dtype=F64, requires_grad=True)
    a = torch.zeros(1, dtype=F64, requires_grad=True)
    f0 = X @ w0 if w0 is not None else None
    params = [w, b] + ([a] if w0 is not None else [])
    opt = torch.optim.LBFGS(params, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = X @ w + b + (a * f0 if f0 is not None else 0)
        pen = w @ w
        if Q is not None:
            qw = Q.T @ w
            pen = pen - (dq * qw * qw).sum()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yy) + 0.5 / (C * n) * pen
        loss.backward()
        return loss
    opt.step(closure)
    out = w.detach()
    return out + a.detach() * w0 if w0 is not None else out


def cv_select(X, y, grid, fitter, folds, seed=0):
    g = torch.Generator().manual_seed(seed)
    fold = torch.empty(len(y), dtype=torch.long)
    for c in (0, 1):
        ic = torch.nonzero(y == c).squeeze(1); ic = ic[torch.randperm(len(ic), generator=g)]
        fold[ic] = torch.arange(len(ic)) % folds
    best = None
    for p in grid:
        sc = torch.empty(len(y), dtype=F64)
        for f in range(folds):
            tr, te = fold != f, fold == f
            sc[te] = X[te].double() @ fitter(X[tr], y[tr], p, 60)
        a_ = auroc(sc, y)
        if best is None or a_ > best[0]:
            best = (a_, p)
    return fitter(X, y, best[1], 200), best[1]


def cv_oof(X, y, grid, fitter, folds, seed=0):
    """Like cv_select but also returns out-of-fold scores (unit-norm fold models) at the selected value."""
    g = torch.Generator().manual_seed(seed)
    fold = torch.empty(len(y), dtype=torch.long)
    for c in (0, 1):
        ic = torch.nonzero(y == c).squeeze(1); ic = ic[torch.randperm(len(ic), generator=g)]
        fold[ic] = torch.arange(len(ic)) % folds
    best = None
    for p in grid:
        sc = torch.empty(len(y), dtype=F64)
        for f in range(folds):
            tr, te = fold != f, fold == f
            w_ = fitter(X[tr], y[tr], p, 60)
            sc[te] = X[te].double() @ (w_ / (w_.norm() + 1e-30))
        a_ = auroc(sc, y)
        if best is None or a_ > best[0]:
            best = (a_, p, sc.clone())
    w = fitter(X, y, best[1], 200)
    return w / (w.norm() + 1e-30), best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--nbs", nargs="*", type=int, default=[16, 32, 64, 128])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--per-source", type=int, default=1000)
    ap.add_argument("--lr-per-source", type=int, default=200)
    ap.add_argument("--src-C", type=float, default=0.1)
    a = ap.parse_args()
    t0 = time.time()
    g = torch.Generator().manual_seed(0)
    # ---- identical loading/splitting to smt2.py (same generator sequence) ----
    raw = {}
    for f in sorted(glob.glob(os.path.join(a.cache, f"{a.model_tag}__*.npz"))):
        nm = os.path.basename(f).split("__")[1][:-4]
        H = npz_memmap(f, "H.npy"); y = np.load(f, allow_pickle=True)["y"]
        raw[nm] = (np.ascontiguousarray(H[:, a.layer, :]).astype(np.float32), y)
    names = sorted(raw); D = {}
    for nm in names:
        X = torch.from_numpy(raw[nm][0]).double(); y = torch.from_numpy(raw[nm][1].astype(np.int64))
        sp = {0: [], 1: [], 2: []}
        for c in (0, 1):
            ic = torch.nonzero(y == c).squeeze(1); ic = ic[torch.randperm(len(ic), generator=g)]
            n = len(ic); i1, i2 = n // 2, (3 * n) // 4
            sp[0].append(ic[:i1]); sp[1].append(ic[i1:i2]); sp[2].append(ic[i2:])
        D[nm] = dict(zip(("tr", "es", "ev"), [(X[torch.cat(sp[k])], y[torch.cat(sp[k])]) for k in range(3)]))
    del raw
    g2 = torch.Generator().manual_seed(1)
    # global standardization (all datasets' train splits; unlabeled statistics only)
    allX = torch.cat([D[k]["tr"][0] for k in names])
    mu, sd = allX.mean(0), allX.std(0) + 1e-6
    del allX
    nz = lambda X: (X - mu) / sd
    # ---- per-dataset LR directions (unit norm), computed once per layer ----
    U = {}
    for k in names:
        X, y = D[k]["tr"]
        idx = bal_sub(y, min(a.per_source, len(y)), g2)
        w = fit_lr(nz(X[idx]), y[idx], a.src_C)
        U[k] = w / w.norm()
    print(f"{a.model_tag} L{a.layer}: per-source LR done {time.time()-t0:.0f}s", flush=True)
    out = dict(model=a.model_tag, layer=a.layer, datasets=names, methods=[], info=[])
    for B in (a.targets or names):
        tB = time.time()
        srcs = [k for k in names if k != B]; K = len(srcs)
        XB, yB = D[B]["ev"]; XBn = nz(XB); Xe, ye = D[B]["es"]
        Wd = torch.stack([U[k] for k in srcs], 1)                       # d x K unit directions
        s = Wd.mean(1)
        Qf, sv, _ = torch.linalg.svd(Wd - s[:, None], full_matrices=False)
        v = sv ** 2 / (K - 1); keep = v > 1e-12 * float(v.max()); Q, v = Qf[:, keep], v[keep]
        rho = max(novelty_rho(Wd, torch.zeros(K, dtype=F64)), 1e-12)
        dq_full = v / (v + rho)
        # pooled LR with grouped CV over C (identical protocol to smt2)
        g3 = torch.Generator().manual_seed(1000)
        sub, grp, ylist = [], [], []
        for i, k in enumerate(srcs):
            X_, y_ = D[k]["tr"]; ii = torch.randperm(len(y_), generator=g3)[: a.lr_per_source]
            sub.append(nz(X_[ii])); ylist.append(y_[ii]); grp.append(torch.full((len(ii),), i))
        Xs, ys, grp = torch.cat(sub), torch.cat(ylist), torch.cat(grp)
        bestc = None
        for C in (1e-3, 1e-2, 1e-1):
            aucs = []
            for f in range(3):
                te = (grp % 3) == f
                wv = fit_lr(Xs[~te], ys[~te], C, iters=100)
                aucs.append(np.mean([auroc(Xs[te & (grp == i)] @ wv, ys[te & (grp == i)]) for i in range(K) if i % 3 == f]))
            if bestc is None or np.mean(aucs) > bestc[0]:
                bestc = (np.mean(aucs), C)
        w_pool = fit_lr(Xs, ys, bestc[1]); w_pool = w_pool / w_pool.norm()
        # diversity penalty gamma by leave-one-source-out (using the mean-direction base; target unused)
        vbar = float(v.mean())
        best = None
        for gm in (0.0, 0.1, 1.0, 10.0, 100.0):
            aucs = []
            for j in range(K):
                o = [k for k in range(K) if k != j]
                sj = Wd[:, o].mean(1)
                Qj, svj, _ = torch.linalg.svd(Wd[:, o] - sj[:, None], full_matrices=False)
                vj = svj ** 2 / (K - 2)
                gj = gm / vbar
                bj = sj - Qj @ ((gj * vj / (1 + gj * vj)) * (Qj.T @ sj))
                Xj, yj = D[srcs[j]]["ev"]
                aucs.append(auroc(nz(Xj) @ bj, yj))
            if best is None or np.mean(aucs) > best[0]:
                best = (float(np.mean(aucs)), gm)
        gam = best[1] / vbar
        pen = lambda b: b - Q @ ((gam * v / (1 + gam * v)) * (Q.T @ b))
        w0 = pen(w_pool); w0 = w0 / w0.norm()
        w0s = pen(s)
        XUn = nz(torch.cat([D[B]["tr"][0], D[B]["es"][0]]))
        Tc = XUn - XUn.mean(0); eT, UT = torch.linalg.eigh((Tc.T @ Tc / (len(XUn) - 1)).float())
        eT, UT = eT.double().clamp(min=0), UT.double(); muT = float(eT.mean()); del Tc
        zs = dict(pooled_lr=auroc(XBn @ w_pool, yB), ebw0=auroc(XBn @ w0, yB), ebw0_mean=auroc(XBn @ w0s, yB),
                  mean_lrdir=auroc(XBn @ s, yB))
        out["methods"].append(dict(B=B, nb=0, rep=0, **zs))
        info = dict(B=B, poolC=bestc[1], gamma=best[1], rho=rho, rank=int(Q.shape[1]), sel={})
        for nb in a.nbs:
            if nb > 2 * min(int((ye == 1).sum()), int((ye == 0).sum())):
                continue
            folds = 4 if nb >= 32 else 2
            for rep in range(a.reps):
                idx = bal_sub(ye, nb, g2)
                Xb, yb = nz(Xe[idx]), ye[idx]
                Cs = [1e-3, 1e-2, 1e-1, 1.0]
                r = {}; P = {}
                P["in_lr"] = cv_oof(Xb, yb, Cs, lambda X, y, C, it: fit_lr(X, y, C, iters=it), folds)
                P["lr_prior"] = cv_oof(Xb, yb, Cs, lambda X, y, C, it: fit_lr(X, y, C, w0=w_pool, iters=it), folds)
                grid = [(C, eta) for C in Cs for eta in (0.0, 0.5, 0.9)]
                P["ebw"] = cv_oof(Xb, yb, grid, lambda X, y, p, it: fit_lr(X, y, p[0], w0=w0, Q=Q, dq=p[1] * dq_full, iters=it), folds)
                def ldaT(X, y, lm, it):                                      # target-covariance LDA (std space)
                    dd = md(X, y)
                    return UT @ ((UT.T @ dd) / (eT + lm * muT))
                P["in_ldaT"] = cv_oof(Xb, yb, [1e-1, 1.0, 10.0, 100.0], ldaT, folds)
                names_p = list(P)
                for k in names_p:
                    r[k] = auroc(XBn @ P[k][0], yB)
                Wp = torch.stack([P[k][0] for k in names_p], 1)               # d x m unit-norm probes
                O = torch.stack([P[k][2] for k in names_p], 1)                # nb x m out-of-fold scores
                SB = XBn @ Wp; SU = XUn @ Wp                                  # eval / unlabeled scores
                sdU = SU.std(0) + 1e-12
                Oz, SBz, SUz = O / sdU, SB / sdU, SU / sdU
                dO = md(Oz, yb)                                               # out-of-fold mean difference
                # SMT stacking: alpha = S^{-1} Delta, S = T_unlabeled - Delta Delta^T / 4 (Sherman-Morrison)
                Tm = torch.cov(SUz.T)
                Sw = Tm - torch.outer(dO, dO) / 4
                ev_, EV = torch.linalg.eigh(Sw); ev_ = torch.clamp(ev_, min=1e-3 * float(ev_.clamp(min=0).mean()) + 1e-12)
                al = EV @ ((EV.T @ dO) / ev_)
                r["smt_stack"] = auroc(SBz @ al, yB)
                al2 = torch.linalg.solve(Tm + 1e-6 * torch.eye(len(names_p), dtype=F64), dO)
                r["smt_stack_T"] = auroc(SBz @ al2, yB)
                # baselines: logistic stacking on out-of-fold scores; uniform average of z-scored probes
                wst = fit_lr(Oz, yb, 1.0, iters=200)
                r["stack_lr"] = auroc(SBz @ wst, yB)
                r["avg"] = auroc(SBz.sum(1), yB)
                info["sel"].setdefault(str(nb), []).append([al.tolist(), [P[k][1] if not isinstance(P[k][1], tuple) else list(P[k][1]) for k in names_p]])
                out["methods"].append(dict(B=B, nb=nb, rep=rep, **r))
        out["info"].append(info)
        print(f"  {B}: gamma={best[1]} pool={zs['pooled_lr']:.3f} ebw0={zs['ebw0']:.3f} mean={zs['ebw0_mean']:.3f} [{time.time()-tB:.0f}s]", flush=True)
        with open(a.out + ".partial", "w") as fh:
            json.dump(out, fh)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh)
    print("[wrote]", a.out, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
