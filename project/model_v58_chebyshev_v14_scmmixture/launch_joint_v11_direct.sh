#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65536

PROJECT="${FM_PROJECT:-/users/minsuk/fm-lm-lambda}"
DATA_ROOT="${DATA_ROOT:-/users/sync_data/small_experiment_tabiclv2_graph_v2prior_N200_P400_train10gb_seed88071010_corrle095_gaussianfill_seed88071102.partial_transfer_aborted_20260711T1158Z}"
CONFIG="${CONFIG:-${PROJECT}/fpk_v33_kkt0.yaml}"
PENALTY_CACHE="${PENALTY_CACHE:-${PROJECT}/penalty_cache_100k_v9_small_experiment.pt}"
WARM_START="${WARM_START:-${PROJECT}/runs/v58chebyshev_v10_joint_N200_P400_30k_20260716_1547/checkpoint_step005400.pt}"
OUT_DIR="${OUT_DIR:?OUT_DIR must be set}"
STEPS="${STEPS:-30000}"
STOP_AFTER="${STOP_AFTER:-}"
LOG_EVERY="${LOG_EVERY:-100}"
WARM_START_SHA256="${WARM_START_SHA256:-$(sha256sum "${WARM_START}" | awk '{print $1}')}"
RESUME="${RESUME:-}"
RESUME_SHA256="${RESUME_SHA256:-}"

export FM_PROJECT="${PROJECT}"
export PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

mkdir -p "$(dirname "${OUT_DIR}")"
args=(
  --data-root "${DATA_ROOT}"
  --part part_0
  --out-dir "${OUT_DIR}"
  --config "${CONFIG}"
  --penalty-cache "${PENALTY_CACHE}"
  --warm-start "${WARM_START}"
  --warm-start-sha256 "${WARM_START_SHA256}"
  --steps "${STEPS}"
  --beta-lr 2e-4
  --selector-lr 3e-4
  --direct-head-lr 1e-5
  --warmup-steps 2000
  --batch-per-rank 1
  --grad-accum-steps 4
  --selector-grad-accum-steps 4
  --selector-every 1
  --selector-hidden 64
  --direct-positivity-beta 100
  --direct-head-init-std 1e-3
  --path-chunk 25
  --holdout-files 8
  --cache-files 2
  --checkpoint-every 200
  --log-every "${LOG_EVERY}"
  --seed 202607162306
  --max-abs-feature-corr 0.95
)
if [[ -n "${STOP_AFTER}" ]]; then
  args+=(--stop-after "${STOP_AFTER}")
fi
if [[ -n "${RESUME}" ]]; then
  if [[ -z "${RESUME_SHA256}" ]]; then
    echo "RESUME_SHA256 must be set when RESUME is set" >&2
    exit 2
  fi
  args+=(--resume "${RESUME}" --resume-sha256 "${RESUME_SHA256}")
fi

cd "${PROJECT}"
exec /venv/main/bin/torchrun \
  --standalone \
  --nproc_per_node=8 \
  -m model_v58_chebyshev_v11_direct.train_joint_v11_direct \
  "${args[@]}" \
  2>&1 | tee -a "${OUT_DIR}.launch.log"
