#!/usr/bin/env bash
set -euo pipefail

PROJECT="/users/minsuk/fm-lm-lambda"
PACKAGE="kalbi_v6_exactprox10_eta4x_v22_v20geomscaled_k32s32_p100k_effbatch32"
PREPARED="${PROJECT}/evaluations/kalbi_v21_mixed6_n500p100_n200p400_seed3107240500_20260724"
CHECKPOINT="${PROJECT}/runs/kalbi_v6_v22_v20geomscaled_tabiclv11_p100k_effbatch32_30k_20260725/checkpoint_step028600.pt"
OUTPUT="${PROJECT}/evaluations/v22_step028600_solution_selector_mixed6_raw1e4to1_20260725"

export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=7
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

cd "${PROJECT}"
exec /venv/main/bin/python \
  -m "${PACKAGE}.report_latest_v22_solution_selector_mixed6" \
  --prepared-root "${PREPARED}" \
  --checkpoint "${CHECKPOINT}" \
  --config "${PROJECT}/fpk_v33_kkt0.yaml" \
  --cd-library "${PROJECT}/canonical_cd_kernel.so" \
  --output-dir "${OUTPUT}" \
  --device cuda:0 \
  --limit 6 \
  --grid-size 41 \
  --gamma-factor 1.0 \
  --ablation-iterations 100 \
  --query-chunk 8 \
  --max-sweeps 2000 \
  --retries 2 \
  --tolerance 1e-5 \
  --zero-tolerance 1e-10
