import argparse, os, csv, json, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
QA_T = "Question: {q}\nAnswer: {a}"


def load_csv(name):
    rows = []
    with open(os.path.join(DATA, name + ".csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["statement"].strip(), int(float(r["label"]))))
    return rows


def load_truthfulqa():
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    out = []
    for ex in ds:
        out.append((QA_T.format(q=ex["question"], a=ex["best_answer"]), 1))
        out.append((QA_T.format(q=ex["question"], a=ex["incorrect_answers"][0]), 0))
    return out


def load_halueval():
    from datasets import load_dataset
    ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    out = []
    for ex in ds:
        out.append((QA_T.format(q=ex["question"], a=ex["right_answer"]), 1))
        out.append((QA_T.format(q=ex["question"], a=ex["hallucinated_answer"]), 0))
    return out


def load_fever():
    from datasets import load_dataset
    base = "hf://datasets/fever/fever@refs/convert/parquet/v1.0/"
    ds = load_dataset("parquet", data_files=base + "train/*.parquet", split="train")
    out = []
    for ex in ds:
        if ex["label"] in ("SUPPORTS", 1):
            out.append((ex["claim"].strip(), 1))
        elif ex["label"] in ("REFUTES", 0):
            out.append((ex["claim"].strip(), 0))
    return out


def _mc_pairs(q, choices, correct_idx, rng):
    wrong = [c for i, c in enumerate(choices) if i != correct_idx and str(c).strip()]
    if not wrong:
        return []
    return [(QA_T.format(q=q, a=choices[correct_idx]), 1), (QA_T.format(q=q, a=wrong[rng.integers(len(wrong))]), 0)]


def load_truthfulqa_mc():
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    out = []
    for ex in ds:
        t = ex["mc2_targets"]
        out += [(QA_T.format(q=ex["question"], a=c), int(l)) for c, l in zip(t["choices"], t["labels"])]
    return out


def load_halueval_dialogue():
    from datasets import load_dataset
    ds = load_dataset("pminervini/HaluEval", "dialogue", split="data")
    out = []
    for ex in ds:
        h = ex["dialogue_history"][-600:]
        out.append((f"Dialogue: {h}\nResponse: {ex['right_response']}", 1))
        out.append((f"Dialogue: {h}\nResponse: {ex['hallucinated_response']}", 0))
    return out


def load_halueval_summ():
    from datasets import load_dataset
    ds = load_dataset("pminervini/HaluEval", "summarization", split="data")
    out = []
    for ex in ds:
        d = ex["document"][:700]
        out.append((f"Document: {d}\nSummary: {ex['right_summary']}", 1))
        out.append((f"Document: {d}\nSummary: {ex['hallucinated_summary']}", 0))
    return out


def load_sciq():
    from datasets import load_dataset
    rng = np.random.default_rng(1); out = []
    for ex in load_dataset("allenai/sciq", split="train"):
        ch = [ex["correct_answer"], ex["distractor1"], ex["distractor2"], ex["distractor3"]]
        out += _mc_pairs(ex["question"], ch, 0, rng)
    return out


def load_medmcqa():
    from datasets import load_dataset
    rng = np.random.default_rng(2); out = []
    for ex in load_dataset("openlifescienceai/medmcqa", split="train"):
        ch = [ex["opa"], ex["opb"], ex["opc"], ex["opd"]]
        if ex["cop"] is None or not (0 <= int(ex["cop"]) < 4):
            continue
        out += _mc_pairs(ex["question"], ch, int(ex["cop"]), rng)
        if len(out) > 20000:
            break
    return out


def load_boolq():
    from datasets import load_dataset
    out = []
    for ex in load_dataset("google/boolq", split="train"):
        q = ex["question"].strip().rstrip("?") + "?"
        ans = "yes" if ex["answer"] else "no"; neg = "no" if ex["answer"] else "yes"
        out.append((QA_T.format(q=q, a=ans), 1)); out.append((QA_T.format(q=q, a=neg), 0))
    return out


def load_arc_challenge():
    from datasets import load_dataset
    rng = np.random.default_rng(3); out = []
    for split in ("train", "validation", "test"):
        for ex in load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split):
            labels = list(ex["choices"]["label"])
            if ex["answerKey"] not in labels:
                continue
            out += _mc_pairs(ex["question"], list(ex["choices"]["text"]), labels.index(ex["answerKey"]), rng)
    return out


def load_commonsense_qa():
    from datasets import load_dataset
    rng = np.random.default_rng(4); out = []
    for ex in load_dataset("tau/commonsense_qa", split="train"):
        labels = list(ex["choices"]["label"])
        if ex["answerKey"] not in labels:
            continue
        out += _mc_pairs(ex["question"], list(ex["choices"]["text"]), labels.index(ex["answerKey"]), rng)
    return out


def load_climate_fever():
    from datasets import load_dataset
    out = []
    for ex in load_dataset("tdiggelm/climate_fever", split="test"):
        if ex["claim_label"] in (0, 1):
            out.append((ex["claim"].strip(), 1 if ex["claim_label"] == 0 else 0))
    return out


def load_mmlu():
    from datasets import load_dataset
    rng = np.random.default_rng(5); out = []
    for ex in load_dataset("cais/mmlu", "all", split="test"):
        out += _mc_pairs(ex["question"], list(ex["choices"]), int(ex["answer"]), rng)
    return out


FEWSHOT = ("Question: what is the capital city of australia\nAnswer: Canberra\n\n"
           "Question: who wrote the novel pride and prejudice\nAnswer: Jane Austen\n\n"
           "Question: how many legs does a spider have\nAnswer: eight\n\n")


def _norm(s):
    import re, string
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


@torch.no_grad()
def gen_nq(model, tok, batch_size, max_q=3610):
    """Self-generated answers on NQ-Open: greedy answers of the model itself, labeled by answer match."""
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/nq_open", split="validation")
    qs = [(ex["question"], ex["answer"]) for ex in ds][:max_q]
    out = []
    for i in range(0, len(qs), batch_size):
        chunk = qs[i:i + batch_size]
        prompts = [FEWSHOT + f"Question: {q}\nAnswer:" for q, _ in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for (q, golds), t in zip(chunk, texts):
            ans = t.strip().split("\n")[0].strip()
            if not ans:
                continue
            na = _norm(ans)
            lab = int(any(_norm(gd) and (_norm(gd) == na or _norm(gd) in na) for gd in golds))
            out.append((QA_T.format(q=q, a=ans), lab))
    return out


SPECIAL = {"truthfulqa": load_truthfulqa, "halueval_qa": load_halueval, "fever": load_fever,
           "truthfulqa_mc": load_truthfulqa_mc, "halueval_dialogue": load_halueval_dialogue,
           "halueval_summ": load_halueval_summ, "sciq": load_sciq, "medmcqa": load_medmcqa, "boolq": load_boolq,
           "arc_challenge": load_arc_challenge, "commonsense_qa": load_commonsense_qa,
           "climate_fever": load_climate_fever, "mmlu": load_mmlu}


def balanced(items, max_items, seed=0):
    rng = np.random.default_rng(seed)
    pos = [it for it in items if it[1] == 1]
    neg = [it for it in items if it[1] == 0]
    k = min(len(pos), len(neg))
    if max_items:
        k = min(k, max_items // 2)
    pi = rng.permutation(len(pos))[:k]
    ni = rng.permutation(len(neg))[:k]
    out = [pos[i] for i in pi] + [neg[i] for i in ni]
    return [out[i] for i in rng.permutation(len(out))]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-items", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--layer-stride", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    tag = a.model.split("/")[-1]
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda()
    else:  # CPU: fp32 is the fast path on x86; use all allocated cores
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    model.eval()
    for name in a.datasets:
        path = os.path.join(a.outdir, f"{tag}__{name}.npz")
        if os.path.exists(path):
            print("[skip]", path, flush=True)
            continue
        t0 = time.time()
        if name == "nqgen":
            items = gen_nq(model, tok, a.batch_size)
        else:
            items = SPECIAL[name]() if name in SPECIAL else load_csv(name)
        items = balanced(items, a.max_items)
        H, ntok = [], []
        for i in range(0, len(items), a.batch_size):
            texts = [t for t, _ in items[i:i + a.batch_size]]
            enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
            hs = model(**enc, output_hidden_states=True).hidden_states
            sel = hs[::a.layer_stride] if a.layer_stride > 1 else hs
            H.append(torch.stack([h[:, -1, :] for h in sel], 1).float().cpu().numpy().astype(np.float16))
            ntok += enc["attention_mask"].sum(1).tolist()
        H = np.concatenate(H)
        y = np.array([l for _, l in items], dtype=np.int8)
        clen = np.array([len(t) for t, _ in items], dtype=np.int32)
        np.savez(path, H=H, y=y, ntok=np.array(ntok, dtype=np.int32), clen=clen,
                 layer_stride=a.layer_stride, text=np.array([t for t, _ in items], dtype=object))
        print(f"[saved] {path} H={H.shape} pos={int(y.sum())} neg={int((1-y).sum())} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
