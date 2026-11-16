"""SMT v2: fixed transfer-law estimators + Bayes-optimal empirical-Bayes probe (EB-SMT) vs baselines.
Usage: python smt2.py --cache cache --model-tag Qwen2.5-7B-Instruct --layer 16 --out results2/qw_L16.json

Fixes over v1
  (1) plug-in denominator: for a fixed trained direction w, w^T S_B w (S_B = within-class covariance of
      the independent target split) is unbiased for w^T Sigma_B w and is >= 0, so predictions cannot blow up.
      For the expected-over-training form, E[d_hat^T S_B d_hat] = d^T Sigma_B d + (4/n) tr(Sigma_A Sigma_B),
      so the same plug-in also estimates C_kk + Delta_kk; extrapolation to m samples adds 4 tau (1/m - 1/n) >= 0.
  (2) SMT-opt weights use the PSD Gram A = D^T S_B D with ridge.
  (3) sign/polarity handled by the random-effects model: directions on which source mean-differences
      disagree (e.g. polarity-flipped components of negated datasets) enter the posterior covariance P
      and are down-weighted by w = (2 Sigma + P)^{-1} m.

EB-SMT (ours). Whitened coordinates (Sigma_lambda = Sigma_pool + lambda mu I). Random-effects prior
  delta_B ~ N(s, V + rho I), s = mean of source mean-differences, V = their between-dataset covariance
  (noise-corrected). With n_B target labels, d_hat_B ~ N(delta_B, o I), o = (4/n_B) c_B: posterior N(m, P).
  Theorem: E_{delta_B ~ N(m,P)} AUC(w) = Phi( w^T m / sqrt(w^T (2 c Sigma + P) w) ), maximized by
  w* = (2 c Sigma + P)^{-1} m. Zero-shot: m = s, P = V + rho I. lambda is chosen by leave-one-source-out.
"""
import argparse, glob, json, os, math, time, zipfile, struct, zlib
import numpy as np
import torch
from scipy.stats import norm

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))
F64 = torch.float64


def npz_memmap(path, member):
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(member)
        assert info.compress_type == zipfile.ZIP_STORED
    with open(path, "rb") as fh:
        fh.seek(info.header_offset)
        hdr = fh.read(30)
        nlen, xlen = struct.unpack("<HH", hdr[26:30])
        fh.seek(info.header_offset + 30 + nlen + xlen)
        ver = np.lib.format.read_magic(fh)
        rd = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = rd(fh)
        off = fh.tell()
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=off, order="F" if fortran else "C")


def auroc(s, y):
    s = s.double(); r = torch.empty_like(s)
    r[torch.argsort(s)] = torch.arange(1, len(s) + 1, dtype=F64)
    n1 = y.sum().double(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def probit(a):
    return float(norm.ppf(min(max(a, 1e-6), 1 - 1e-6)))


def md(X, y):
    return X[y == 1].mean(0) - X[y == 0].mean(0)


def within(X, y):
    Z = X.clone()
    for c in (0, 1):
        Z[y == c] -= Z[y == c].mean(0)
    return Z


def bal_sub(y, n, g):
    idx = []
    for c in (0, 1):
        ic = torch.nonzero(y == c).squeeze(1)
        idx.append(ic[torch.randperm(len(ic), generator=g)[: n // 2]])
    return torch.cat(idx)


def polarity(name):
    return -1.0 if name.startswith("neg_") else 1.0


# ---------------------------------------------------------------- EB-SMT core (whitened space)
def _cg(Aop, b, iters=200, tol=1e-10):
    x = torch.zeros_like(b); r = b.clone(); p = r.clone(); rs = r @ r
    for _ in range(iters):
        Ap = Aop(p); al = rs / (p @ Ap)
        x = x + al * p; r = r - al * Ap
        rn = r @ r
        if rn < tol * (b @ b):
            break
        p = r + (rn / rs) * p; rs = rn
    return x


def eb_direction(Dt, nk_noise, rho, sdiag, dB=None, odiag=None):
    """EB-SMT in whitened coordinates with diagonal target covariance sdiag (Sigma_B ~ diag(sdiag)) and
    diagonal observation noise odiag. Prior delta_B ~ N(s, Pi), Pi = Q diag(v) Q^T + rho I.
    Posterior N(m, P); Bayes-optimal expected-AUC direction w = (2 S + P)^{-1} m.
    Returns w, m, predicted expected AUC Phi(w^T m / sqrt(w^T (2S + P) w))."""
    d, K = Dt.shape
    s = Dt.mean(1)
    Q, sv, _ = torch.linalg.svd(Dt - s[:, None], full_matrices=False)
    v = sv ** 2 / (K - 1) - nk_noise.mean()
    keep = v > 1e-12 * max(float(v.max()), 1e-30)
    Q, v = Q[:, keep], v[keep]
    Pi = lambda x: Q @ (v * (Q.T @ x)) + rho * x

    def lowrank_solve(D0, x):          # (diag(D0) + Q diag(v) Q^T)^{-1} x  (Woodbury)
        Dx = x / D0; DQ = Q / D0[:, None]
        Mi = torch.diag(1.0 / v) + Q.T @ DQ
        return Dx - DQ @ torch.linalg.solve(Mi, Q.T @ Dx)
    if dB is None:
        m = s
        Pop = Pi
        w = lowrank_solve(2 * sdiag + rho, m)
    else:
        y = lowrank_solve(rho + odiag, dB - s)
        m = s + Pi(y)
        Pop = lambda x: Pi(x) - Pi(lowrank_solve(rho + odiag, Pi(x)))
        w = _cg(lambda x: 2 * sdiag * x + Pop(x), m)
    num = float(w @ m); den = float(w @ (2 * sdiag * w + Pop(w)))
    return w, m, float(norm.cdf(num / math.sqrt(max(den, 1e-30))))


def novelty_rho(Dt, nk_noise):
    """Leave-one-source-out moment estimate of the isotropic novelty rho (whitened, per dim)."""
    d, K = Dt.shape
    vals = []
    for j in range(K):
        o = [k for k in range(K) if k != j]
        s = Dt[:, o].mean(1)
        Q, sv, _ = torch.linalg.svd(Dt[:, o] - s[:, None], full_matrices=False)
        Q = Q[:, sv > 1e-10 * max(float(sv.max()), 1e-30)]
        r = Dt[:, j] - s
        rp = r - Q @ (Q.T @ r)
        # E||rp||^2 = (d - r)(rho + noise_j) + (1/(K-1)) * avg noise of the others (mean of K-1 sources)
        vals.append(float(rp @ rp) / (d - Q.shape[1]) - float(nk_noise[j]) - float(nk_noise[o].mean()) / (K - 1))
    return max(float(np.mean(vals)), 0.0)


# ---------------------------------------------------------------- logistic helpers
def fit_lr(X, y, C, w0=None, iters=200):
    """L2 logistic regression with intercept (torch LBFGS). If w0 is given, adds an unpenalized
    scalar coefficient on the anchor score X @ w0 (prior-anchored fine-tuning)."""
    X = X.double(); yy = y.double()
    d = X.shape[1]
    w = torch.zeros(d, dtype=F64, requires_grad=True)
    b = torch.zeros(1, dtype=F64, requires_grad=True)
    a = torch.zeros(1, dtype=F64, requires_grad=True)
    f0 = (X @ w0) if w0 is not None else None
    params = [w, b] + ([a] if w0 is not None else [])
    opt = torch.optim.LBFGS(params, max_iter=iters, line_search_fn="strong_wolfe")
    n = len(yy)

    def closure():
        opt.zero_grad()
        z = X @ w + b + (a * f0 if f0 is not None else 0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yy) + 0.5 / (C * n) * (w @ w)
        loss.backward()
        return loss
    opt.step(closure)
    out = w.detach()
    if w0 is not None:
        out = out + a.detach() * w0
    return out


def cv_lr(X, y, Cs, w0=None, folds=4, seed=0):
    """Choose C by stratified k-fold CV AUC on (X, y); refit on all."""
    g = torch.Generator().manual_seed(seed)
    fold = torch.empty(len(y), dtype=torch.long)
    for c in (0, 1):
        ic = torch.nonzero(y == c).squeeze(1)
        ic = ic[torch.randperm(len(ic), generator=g)]
        fold[ic] = torch.arange(len(ic)) % folds
    best, bC = -1, Cs[0]
    for C in Cs:
        sc = torch.empty(len(y), dtype=F64)
        for f in range(folds):
            tr, te = fold != f, fold == f
            sc[te] = X[te].double() @ fit_lr(X[tr], y[tr], C, w0, iters=60)
        a = auroc(sc, y)
        if a > best:
            best, bC = a, C
    return fit_lr(X, y, bC, w0)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", nargs="*", default=None, help="subset of targets (for splitting jobs)")
    ap.add_argument("--nbs", nargs="*", type=int, default=[16, 32, 64, 128])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--lr-per-source", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-law", action="store_true")
    ap.add_argument("--synthetic", default=None, help="npz with dict of (H (n,d), y) for testing")
    a = ap.parse_args()
    t0 = time.time()
    g = torch.Generator().manual_seed(a.seed)

    # ---- load one layer of every dataset, split 50/25/25 (train / est / eval), stratified
    raw = {}
    if a.synthetic:
        z = np.load(a.synthetic, allow_pickle=True)
        for k in z.files:
            if k.endswith("__H"):
                nm = k[:-3]; raw[nm] = (z[k].astype(np.float32), z[nm + "__y"])
    else:
        for f in sorted(glob.glob(os.path.join(a.cache, f"{a.model_tag}__*.npz"))):
            nm = os.path.basename(f).split("__")[1][:-4]
            H = npz_memmap(f, "H.npy")
            y = np.load(f, allow_pickle=True)["y"]
            raw[nm] = (np.ascontiguousarray(H[:, a.layer, :]).astype(np.float32), y)
    names = sorted(raw)
    D = {}
    for nm in names:
        X = torch.from_numpy(raw[nm][0]).double(); y = torch.from_numpy(raw[nm][1].astype(np.int64))
        sp = {0: [], 1: [], 2: []}
        for c in (0, 1):
            ic = torch.nonzero(y == c).squeeze(1); ic = ic[torch.randperm(len(ic), generator=g)]
            n = len(ic); i1, i2 = n // 2, (3 * n) // 4
            sp[0].append(ic[:i1]); sp[1].append(ic[i1:i2]); sp[2].append(ic[i2:])
        tr, es, ev = [(X[torch.cat(sp[k])], y[torch.cat(sp[k])]) for k in range(3)]
        Ztr = within(*tr)
        D[nm] = dict(tr=tr, es=es, ev=ev, md=md(*tr), ntr=len(tr[1]), Ztr=Ztr, G=Ztr.T @ Ztr,
                     Zes=within(*es), mdes=md(*es))
    del raw
    d = D[names[0]]["md"].shape[0]
    Gtot = sum(D[k]["G"] for k in names); Ntot = sum(D[k]["ntr"] - 2 for k in names)
    print(f"{a.model_tag} L{a.layer}: {len(names)} datasets d={d} loaded {time.time()-t0:.0f}s", flush=True)

    out = dict(model=a.model_tag, layer=a.layer, datasets=names, law=[], law2=[], law3=[], methods=[], eb0=[])

    # ================= Part A: transfer law with plug-in (PSD) denominators =================
    for B in ([] if a.skip_law else names):
        b = D[B]; ZB = b["Zes"]; nB = ZB.shape[0]
        XB, yB = b["ev"]
        srcs = [k for k in names if k != B]
        W = torch.stack([D[k]["md"] for k in srcs])                 # K x d
        P = ZB @ W.T                                                # nB x K
        A = P.T @ P / (nB - 2)                                      # = W S_B W^T  (PSD, unbiased for C + Delta)
        M = W @ b["mdes"]
        tau = torch.stack([((D[k]["Ztr"].float() @ ZB.float().T) ** 2).sum().double() / ((D[k]["ntr"] - 2) * (nB - 2)) for k in srcs])
        for i, k in enumerate(srcs):
            obs = probit(auroc(XB @ W[i], yB))
            pr = float(M[i] / math.sqrt(2 * float(A[i, i])))
            out["law"].append(dict(A=k, B=B, obs=obs, pred=pr))
        rng = np.random.default_rng(zlib.crc32(B.encode()))
        for _ in range(20):
            r = int(rng.integers(2, 11)); S = rng.choice(len(srcs), r, replace=False)
            al = torch.zeros(len(srcs), dtype=F64); al[S] = 1.0 / r
            obs = probit(auroc(XB @ (al @ W), yB))
            out["law2"].append(dict(B=B, S=[srcs[i] for i in S], obs=obs, pred=float(al @ M / math.sqrt(2 * float(al @ A @ al)))))
        for i, k in enumerate(srcs[:: 3]):                           # sample-size extrapolation on a third of the pairs
            ii = srcs.index(k); Xa, ya = D[k]["tr"]
            for m_ in (16, 64, 256):
                if m_ >= len(ya) // 2:
                    continue
                obs = np.mean([probit(auroc(XB @ md(Xa[j], ya[j]), yB)) for j in [bal_sub(ya, m_, g) for _ in range(5)]])
                den = float(A[ii, ii]) + 4 * float(tau[ii]) * (1.0 / m_ - 1.0 / D[k]["ntr"])
                out["law3"].append(dict(A=k, B=B, m=m_, obs=float(obs), pred=float(M[ii] / math.sqrt(2 * den))))
    print(f"law done {time.time()-t0:.0f}s", flush=True)

    # ================= Part B: methods =================
    targets = a.targets or names
    lams = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    for B in targets:
        tB = time.time()
        b = D[B]; XB, yB = b["ev"]; Xe, ye = b["es"]
        srcs = [k for k in names if k != B]
        K = len(srcs)
        nk = torch.tensor([float(D[k]["ntr"]) for k in srcs], dtype=F64)
        Wraw = torch.stack([D[k]["md"] for k in srcs]).T             # d x K
        Sig = (Gtot - b["G"]) / (Ntot - (b["ntr"] - 2))              # pooled within-class covariance of sources
        mu = float(torch.trace(Sig)) / d
        ev_, U = torch.linalg.eigh(Sig)
        ev_ = torch.clamp(ev_, min=0)
        Xall = torch.cat([D[k]["tr"][0] for k in srcs]); yall = torch.cat([D[k]["tr"][1] for k in srcs])
        res = {}

        def score(w):
            return auroc(XB @ w, yB)

        def whiten_ops(lam):
            sc = 1.0 / torch.sqrt(ev_ + lam * mu)
            col = lambda v: sc[:, None] if v.dim() == 2 else sc
            fw = lambda v: col(v) * (U.T @ v)                         # original -> whitened (column/vec)
            bw = lambda v: U @ (col(v) * v)                           # whitened direction -> original score vector
            return fw, bw, sc

        Sk = {k: D[k]["G"] / (D[k]["ntr"] - 2) for k in srcs}

        def per_source_c(sc, idx):
            # tr(Sigma_lambda^{-1} Sigma_k)/d = whitened within-class scale of source k
            Sinv = (U * sc ** 2) @ U.T
            return torch.tensor([float((Sinv * Sk[k]).sum()) / d for k in idx], dtype=F64)

        # ---- choose lambda for EB-0 by leave-one-source-out AUC over sources (target never used)
        best = None
        for lam in lams:
            fw, bw, sc = whiten_ops(lam)
            Dt = fw(Wraw)
            ck = per_source_c(sc, srcs)
            noise = 4.0 / nk * ck
            rho = novelty_rho(Dt, noise)
            aucs = []
            for j in range(K):
                o = [k for k in range(K) if k != j]
                w, _, _ = eb_direction(Dt[:, o], noise[o], rho, ev_ * sc ** 2)
                Xj, yj = D[srcs[j]]["ev"]
                aucs.append(auroc(Xj @ bw(w), yj))
            sc_ = float(np.mean(aucs))
            if best is None or sc_ > best[0]:
                best = (sc_, lam, rho)
        _, lam, rho = best
        fw, bw, sc = whiten_ops(lam)
        Dt = fw(Wraw); ck = per_source_c(sc, srcs); noise = 4.0 / nk * ck
        sdiag0 = ev_ * sc ** 2                                        # whitened pooled within-class variances
        w_eb0, m0, pauc = eb_direction(Dt, noise, rho, sdiag0)

        def wtrace(Zb):                                               # tr(Sigma_lambda^{-1} S_B) from residuals
            return float(((Zb @ U) ** 2 * sc ** 2).sum()) / (Zb.shape[0] - 2)

        # ---- calibrate label trust kappa per n_B by simulated few-shot tasks on the sources (target unused)
        kappa = {}
        for nb in a.nbs:
            bestk = None
            for kp in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
                aucs = []
                for j in range(K):
                    Xej, yej = D[srcs[j]]["es"]
                    if nb > 2 * min(int((yej == 1).sum()), int((yej == 0).sum())):
                        continue
                    o = [k for k in range(K) if k != j]
                    ij = bal_sub(yej, nb, torch.Generator().manual_seed(7 + j))
                    Xb_, yb_ = Xej[ij], yej[ij]
                    sdj = sdiag0 * (wtrace(within(Xb_, yb_)) / float(sdiag0.sum()))
                    w, _, _ = eb_direction(Dt[:, o], noise[o], rho, sdj, dB=fw(md(Xb_, yb_)), odiag=4.0 / nb * kp * sdj)
                    Xj, yj = D[srcs[j]]["ev"]
                    aucs.append(auroc(Xj @ bw(w), yj))
                if aucs and (bestk is None or np.mean(aucs) > bestk[0]):
                    bestk = (float(np.mean(aucs)), kp)
            kappa[nb] = bestk[1] if bestk else 1.0
        w_eb0o = bw(w_eb0)

        # ---- zero-shot baselines
        w_uni = Wraw.mean(1)
        w_pool = md(Xall, yall)
        # covariance-corrected mass-mean (Marks & Tegmark 2023), same lambda selection rule (LOSO) for fairness
        bestl = None
        for l2 in lams:
            fw2, bw2, _ = whiten_ops(l2)
            aucs = []
            for j in range(K):
                o = [k for k in range(K) if k != j]
                Xj, yj = D[srcs[j]]["ev"]
                aucs.append(auroc(Xj @ bw2(fw2(Wraw[:, o].mean(1))), yj))
            if bestl is None or np.mean(aucs) > bestl[0]:
                bestl = (np.mean(aucs), l2)
        fw2, bw2, _ = whiten_ops(bestl[1])
        w_mmiid = bw2(fw2(w_pool))
        # TTPD general truth direction (Buerger et al. 2024): per-dataset centering, OLS on [tau, tau*p]
        Ac, Fc = [], []
        for k in srcs:
            Xk, yk = D[k]["tr"]
            tau_ = (2 * yk - 1).double(); p = polarity(k)
            Ac.append(Xk - Xk.mean(0)); Fc.append(torch.stack([tau_, tau_ * p], 1))
        Ac = torch.cat(Ac); Fc = torch.cat(Fc)
        TT = torch.linalg.solve(Fc.T @ Fc, Fc.T @ Ac)              # 2 x d
        w_ttpd = TT[0]
        # pooled logistic regression (standardized), C by grouped 3-fold over sources
        sub = torch.cat([torch.nonzero(torch.cat([torch.full((D[k]["ntr"],), i) for i, k in enumerate(srcs)]) == i).squeeze(1)[
            torch.randperm(D[k]["ntr"], generator=g)[: a.lr_per_source]] for i, k in enumerate(srcs)])
        grp = torch.cat([torch.full((D[k]["ntr"],), i) for i, k in enumerate(srcs)])[sub]
        Xs, ys = Xall[sub], yall[sub]
        mu_s, sd_s = Xs.mean(0), Xs.std(0) + 1e-6
        Xn = (Xs - mu_s) / sd_s
        bestc = None
        for C in (1e-3, 1e-2, 1e-1):
            aucs = []
            for f in range(3):
                te = (grp % 3) == f
                wv = fit_lr(Xn[~te], ys[~te], C, iters=100)
                aucs.append(np.mean([auroc(Xn[te & (grp == i)] @ wv, ys[te & (grp == i)]) for i in range(K) if i % 3 == f]))
            if bestc is None or np.mean(aucs) > bestc[0]:
                bestc = (np.mean(aucs), C)
        w_lr = fit_lr(Xn, ys, bestc[1]) / sd_s                        # back to raw scale (scores up to a constant)
        # SMT consensus (v1 E5, fixed with PSD plug-in A): weights from sources only
        zs = dict(uniform_mm=score(w_uni), pooled_mm=score(w_pool), mm_iid=score(w_mmiid), ttpd=score(w_ttpd),
                  pooled_lr=score(w_lr), eb0=score(w_eb0o))
        out["methods"].append(dict(B=B, nb=0, rep=0, **zs))
        out["eb0"].append(dict(B=B, lam=lam, rho=rho, kappa=kappa, pred_auc=pauc, obs_auc=zs["eb0"]))

        # ---- few-shot
        for nb in a.nbs:
            if nb > min(int((ye == 1).sum()), int((ye == 0).sum())) * 2:
                continue
            for rep in range(a.reps):
                idx = bal_sub(ye, nb, g)
                Xb, yb = Xe[idx], ye[idx]
                dB = md(Xb, yb); Zb = within(Xb, yb)
                r = {}
                r["in_mm"] = score(dB)
                r["in_lda"] = score(bw2(fw2(dB)))
                Xbn = (Xb - mu_s) / sd_s
                Cs = [1e-3, 1e-2, 1e-1, 1.0]
                folds = 4 if nb >= 32 else 2
                r["in_lr"] = score(cv_lr(Xbn, yb, Cs, folds=folds) / sd_s)
                r["lr_prior"] = score(cv_lr(Xbn, yb, Cs, w0=w_lr * sd_s, folds=folds) / sd_s)
                # best single + SMT-opt (fixed): plug-in PSD Gram from the n_B labels
                Pm = Zb @ Wraw
                Ag = Pm.T @ Pm / (nb - 2); Mv = Wraw.T @ dB
                pr = Mv / torch.sqrt(2 * torch.clamp(Ag.diagonal(), min=1e-30))
                r["best_single"] = score(Wraw[:, int(torch.argmax(pr))])
                al = torch.linalg.solve(Ag + 0.1 * torch.trace(Ag) / K * torch.eye(K, dtype=F64), Mv)
                r["smt_opt"] = score(Wraw @ al)
                # EB-SMT (ours): posterior with the n_B labels, target scale from its own residuals
                sdB = sdiag0 * (wtrace(Zb) / float(sdiag0.sum()))
                wq, _, _ = eb_direction(Dt, noise, rho, sdB, dB=fw(dB), odiag=4.0 / nb * kappa[nb] * sdB)
                r["eb"] = score(bw(wq))
                r.update(zs_uniform=zs["uniform_mm"])
                out["methods"].append(dict(B=B, nb=nb, rep=rep, **{k: v for k, v in r.items() if k != "zs_uniform"}))
        print(f"  {B}: lam={lam:g} rho={rho:.3g} eb0={zs['eb0']:.3f} lr={zs['pooled_lr']:.3f} ttpd={zs['ttpd']:.3f} "
              f"mmiid={zs['mm_iid']:.3f} [{time.time()-tB:.0f}s]", flush=True)
        with open(a.out + ".partial", "w") as f:
            json.dump(out, f)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f)
    print("[wrote]", a.out, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
