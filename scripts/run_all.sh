#!/usr/bin/env bash
# Full pipeline on one category: train both models, evaluate, stress-test, update README.
# Usage: bash scripts/run_all.sh [path/to/category]
set -euo pipefail
ROOT="${1:-data/VisA_pytorch/1cls/pcb1}"

python scripts/train_autoencoder.py --root "$ROOT"
python scripts/train_patchcore.py   --root "$ROOT"
python scripts/evaluate.py          --root "$ROOT"
python scripts/robustness.py        --root "$ROOT" --model patchcore
python scripts/robustness.py        --root "$ROOT" --model autoencoder
python scripts/update_readme.py
