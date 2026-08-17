#!/usr/bin/env bash
set -euo pipefail

PROJECT="/users/minsuk/fm-lm-lambda"
PACKAGE="kalbi_v5_exactprox_k32s32_p100k_effbatch32"

export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

exec nice -n 10 /venv/main/bin/python \
  -m "${PACKAGE}.report_v5_eta_sweep_context_lowdim4" \
  --v3-checkpoint "${PROJECT}/runs/kalbi_v3_v19cumbeta_tabiclv11_expvarnoise_p100k_effbatch32_30k_20260723/checkpoint_step021600.pt" \
  --config "${PROJECT}/fpk_v33_kkt0.yaml" \
  --source-eval "${PROJECT}/evaluations/kalbi_v3_step21600_vs_v4_step1000_context_lowdim4_20260723" \
  --out-dir "${PROJECT}/evaluations/kalbi_v5_eta4x_8x_16x_T5_10_20_context_lowdim4_20260723" \
  --ritz-safeties 0.5 0.25 0.125 \
  --iterations 5 10 20 \
  --device cuda:0
