# Truth Probes That Transfer

Code for the second-moment theory of linear probe transfer and the SMT-Bayes probe for detecting
true and false statements from the hidden states of large language models.

## Files

| File | Purpose |
|---|---|
| `extract.py` | Loads a model with Hugging Face `transformers`, builds balanced true/false statement sets, and stores the last-token hidden state of every layer (`cache/<model>__<dataset>.npz`). |
| `smt_bayes.py` | SMT-Bayes: Bayes-optimal direction `(2 Sigma + gamma V + lambda I)^{-1} s` per layer, pooled logistic regression per layer, Bayes-optimal combination across layers from held-out source scores, and the few-shot anchored refit. Also reports LR-prior and the ablation variants. |
| `baselines.py` | Mass-mean, covariance-corrected mass-mean, TTPD general truth direction, and target-only logistic regression on identical splits. |
| `smt2.py` | Transfer-law moment estimators (cross-fit and plug-in Gram), law predictions, and shared utilities. |
| `smt3.py`, `smt8.py`, `smt11.py`, `smt12.py` | Shared routines (logistic regression with anchoring, cross-validation with out-of-fold scores, dataset families, layer-space moment estimates). |
| `run_example.sh` | End-to-end example for one model. |

## Data

Datasets loaded from the Hugging Face Hub (TruthfulQA, HaluEval, FEVER, Climate-FEVER, SciQ, MedMCQA,
BoolQ, ARC-Challenge, CommonsenseQA, MMLU) are downloaded automatically by `extract.py`.
The curated true/false datasets (cities, translations, numerical comparisons, animal classes, elements,
inventors, facts, and their negated, conjunctive, disjunctive, and German variants, as well as companies,
common claims, and CounterFact statements) are taken from the public releases accompanying
Marks and Tegmark (2024) and Bürger et al. (2024). Place each as `data/<name>.csv` with columns
`statement,label`.

## Protocol

Every script uses a fixed split of each dataset into training (50%), estimation (25%), and evaluation (25%)
parts. With `--lofo`, all datasets from the target's family are removed from the sources. Few-shot budgets
are set with `--nbs` (default 16 32 64 128). Results are written as JSON with one record per target,
budget, and repetition.

## Requirements

Python 3.10 or later and the packages in `requirements.txt`. Activation extraction for 70B models needs
about 300 GB of CPU memory or a multi-GPU node; all other steps run on a CPU.
