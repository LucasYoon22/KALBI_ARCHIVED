#!/usr/bin/env bash
set -euo pipefail

PROJECT="/users/minsuk/fm-lm-lambda"
PACKAGE="kalbi_v2_v17_n200p500_joint"
CHECKPOINT="${PROJECT}/runs/kalbi_v2_v17_n200p500_quick1k_20260721/checkpoint_step010000.pt"
CONFIG="${PROJECT}/fpk_v33_kkt0.yaml"
PENALTY_CACHE="${PROJECT}/penalty_cache_300k_v12_canonical_final.pt"
OUT_ROOT="${PROJECT}/artifacts/kalbi_v2_v17_step10000_paths_selector_20260721"
LOG_DIR="${PROJECT}/logs/kalbi_v2_v17_step10000_eval_20260721"

test -s "${CHECKPOINT}"
test -s "${CONFIG}"
test -s "${PENALTY_CACHE}"
test ! -e "${OUT_ROOT}/solution_paths"
test ! -e "${OUT_ROOT}/selector_optimality"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

/venv/main/bin/python3 - "${CHECKPOINT}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if int(checkpoint.get("completed_steps", -1)) != 10_000:
    raise SystemExit("evaluation requires an exact step-10000 checkpoint")
print(f"validated_step10000_checkpoint={sys.argv[1]}", flush=True)
PY
sha256sum "${CHECKPOINT}" | tee "${LOG_DIR}/checkpoint.sha256"

export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

cd "${PROJECT}"

/venv/main/bin/python3 -m "${PACKAGE}.report_kalbi_v2_solution_paths" \
  --checkpoint "${CHECKPOINT}" \
  --config "${CONFIG}" \
  --penalty-cache "${PENALTY_CACHE}" \
  --out-dir "${OUT_ROOT}/solution_paths" \
  --device cuda:0 \
  --count 6 \
  --queries 25 \
  --reference-sweeps 16 \
  --reference-tolerance 1e-7 \
  2>&1 | tee "${LOG_DIR}/solution_paths.log"

/venv/main/bin/python3 -m "${PACKAGE}.report_kalbi_v2_selector_optimality" \
  --checkpoint "${CHECKPOINT}" \
  --config "${CONFIG}" \
  --penalty-cache "${PENALTY_CACHE}" \
  --out-dir "${OUT_ROOT}/selector_optimality" \
  --device cuda:0 \
  --count 8 \
  --lambda-grid-size 41 \
  --penalty-candidates 64 \
  --candidate-lambda-grid-size 41 \
  2>&1 | tee "${LOG_DIR}/selector_optimality.log"
