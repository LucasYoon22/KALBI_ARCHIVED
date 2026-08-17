#!/bin/bash
set -euo pipefail

cd /users/minsuk/fm-lm-lambda
export CUDA_VISIBLE_DEVICES=2
exec /venv/main/bin/python -m \
  kalbi_v7_blocksize_microbenchmark_20260727.run_all \
  --mode full \
  --device cuda:0
