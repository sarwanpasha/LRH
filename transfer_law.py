"""Transfer-law validation (Theorem law, Theorem est, Proposition plugin) under leave-one-family-out pairs.
For each layer, source probes are mass-mean directions of the full training split of each source dataset.
The target moments use the labeled target estimation split (all of it, and separately n_B = 64 labels),
and the observed AUROC is measured on the target evaluation split.
Outputs single-source pairs, random uniform multi-source combinations (2 to 10 sources), and sample-size
extrapolation (source probes re-estimated from m statements).
Usage: python transfer_law.py --model-tag Qwen2.5-7B-Instruct --layers 8 12 16 20 24 28 --out results/law_qwen7b.json
"""
import argparse, glob, json, os, math, time, zlib
import numpy as np
import torch
from smt2 import npz_memmap, auroc, md, within, bal_sub, F64, probit
from smt11 import family

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--layers", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--few", type=int, default=64)
    a = ap.parse_args()
    t0 = time.time()
    raw = {}
    for f in sorted(glob.glob(os.path.join(a.cache, f"{a.model_tag}__*.npz"))):
        nm = os.path.basename(f).split("__")[1][:-4]
        H = npz_memmap(f, "H.npy"); y = np.load(f, allow_pickle=True)["y"]
        raw[nm] = (np.ascontiguousarray(H[:, a.layers, :]).astype(np.float32), y)
    names = sorted(raw)
    g = torch.Generator().manual_seed(0)
    SPL = {}
    for nm in names:
        y = torch.from_numpy(raw[nm][1].astype(np.int64)); sp = {0: [], 1: [], 2: []}
        for c in (0, 1):
            ic = torch.nonzero(y == c).squeeze(1); ic = ic[torch.randperm(len(ic), generator=g)]
            n = len(ic); i1, i2 = n // 2, (3 * n) // 4
            sp[0].append(ic[:i1]); sp[1].append(ic[i1:i2]); sp[2].append(ic[i2:])
        SPL[nm] = [torch.cat(sp[k]) for k in range(3)]
    out = dict(model=a.model_tag, layers=a.layers, datasets=names, law=[], law2=[], law3=[], few=[])
    for li, L in enumerate(a.layers):
        D = {}
        for nm in names:
            X = torch.from_numpy(raw[nm][0][:, li, :]).double(); y = torch.from_numpy(raw[nm][1].astype(np.int64))
            tr, es, ev = [(X[i], y[i]) for i in SPL[nm]]
            Ztr = within(*tr)
            D[nm] = dict(tr=tr, es=es, ev=ev, md=md(*tr), ntr=len(tr[1]), Ztr=Ztr.float(), Zes=within(*es), mdes=md(*es))
        gl = torch.Generator().manual_seed(100 + li)
        for B in names:
            b = D[B]; XB, yB = b["ev"]
            srcs = [k for k in names if family(k) != family(B)]
            W = torch.stack([D[k]["md"] for k in srcs])
            ZB = b["Zes"]; nB = ZB.shape[0]
            PZ = ZB @ W.T; A = PZ.T @ PZ / (nB - 2); M = W @ b["mdes"]
            tau = torch.stack([((D[k]["Ztr"] @ ZB.float().T) ** 2).sum().double() / ((D[k]["ntr"] - 2) * (nB - 2)) for k in srcs])
            sc = XB @ W.T
            for i, k in enumerate(srcs):
                out["law"].append(dict(L=L, A=k, B=B, obs=probit(auroc(sc[:, i], yB)), pred=float(M[i] / math.sqrt(2 * float(A[i, i])))))
            # few-label version of the target moments
            Xe, ye = b["es"]
            if a.few <= 2 * min(int((ye == 1).sum()), int((ye == 0).sum())):
                idx = bal_sub(ye, a.few, gl); Xf, yf = Xe[idx], ye[idx]
                Zf = within(Xf, yf); Pf = Zf @ W.T; Af = Pf.T @ Pf / (len(yf) - 2); Mf = W @ md(Xf, yf)
                for i, k in enumerate(srcs):
                    out["few"].append(dict(L=L, A=k, B=B, obs=probit(auroc(sc[:, i], yB)), pred=float(Mf[i] / math.sqrt(2 * float(Af[i, i]) + 1e-12))))
            rng = np.random.default_rng(zlib.crc32(f"{B}{L}".encode()))
            for _ in range(20):
                r = int(rng.integers(2, 11)); S = rng.choice(len(srcs), r, replace=False)
                al = torch.zeros(len(srcs), dtype=F64); al[S] = 1.0 / r
                out["law2"].append(dict(L=L, B=B, r=r, obs=probit(auroc(sc @ al, yB)), pred=float(al @ M / math.sqrt(2 * float(al @ A @ al)))))
            for i, k in list(enumerate(srcs))[::3]:
                Xa, ya = D[k]["tr"]
                for m_ in (16, 64, 256):
                    if m_ >= len(ya) // 2:
                        continue
                    obs = float(np.mean([probit(auroc(XB @ md(Xa[j], ya[j]), yB)) for j in [bal_sub(ya, m_, gl) for _ in range(5)]]))
                    den = float(A[i, i]) + 4 * float(tau[i]) * (1.0 / m_ - 1.0 / D[k]["ntr"])
                    out["law3"].append(dict(L=L, A=k, B=B, m=m_, obs=obs, pred=float(M[i] / math.sqrt(2 * den))))
        print(f"layer {L} done {time.time()-t0:.0f}s", flush=True)
        with open(a.out + ".partial", "w") as fh:
            json.dump(out, fh)
    with open(a.out, "w") as fh:
        json.dump(out, fh)
    print("[wrote]", a.out)


if __name__ == "__main__":
    main()
