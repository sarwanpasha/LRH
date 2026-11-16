#!/bin/bash
# Example pipeline for one model. Activations are written to ./cache, results to ./results.
# Curated true/false CSV files (statement,label) go in ./data (see README).
set -e
MODEL=Qwen/Qwen2.5-7B-Instruct
TAG=Qwen2.5-7B-Instruct
LAYERS="8 12 16 20 24 28"
mkdir -p cache results

# 1) Extract last-token hidden states (all layers) for every dataset.
python extract.py --model "$MODEL" --outdir cache --max-items 4000 --datasets \
  cities neg_cities sp_en_trans neg_sp_en_trans larger_than smaller_than \
  truthfulqa truthfulqa_mc halueval_qa halueval_dialogue halueval_summ fever climate_fever \
  sciq medmcqa boolq arc_challenge commonsense_qa mmlu

# 2) SMT-Bayes (zero-shot and few-shot), leave-one-family-out.
python smt_bayes.py --cache cache --model-tag "$TAG" --layers $LAYERS --lofo \
  --out results/smt_bayes_${TAG}.json

# 3) Published baselines (MM, MM-iid, TTPD, In-LR) on the same splits.
python baselines.py --cache cache --model-tag "$TAG" --layers $LAYERS --lofo \
  --out results/baselines_${TAG}.json

# 4) Transfer-law validation (leave-one-family-out pairs).
python transfer_law.py --cache cache --model-tag "$TAG" --layers $LAYERS --out results/law_${TAG}.json
