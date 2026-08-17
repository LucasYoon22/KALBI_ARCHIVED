#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65536

PROJECT="/users/minsuk/fm-lm-lambda"
PACKAGE="kalbi_v1_v17_n200p500_joint"
DATA_ROOT="/users/sync_data/kalbi_tabpfnv1_tabiclv2_scm_xy_50_50_groupdrop095_N200_P500_100GB_seed88073500"
CONTROL_ROOT="${DATA_ROOT}/smoke_control_128"
TRAINER_ID="kalbi_v1_v17_n200p500_quick1k_20260721"
OUT_DIR="${PROJECT}/runs/${TRAINER_ID}"
LOG_DIR="${PROJECT}/logs/${TRAINER_ID}"
LOG_PATH="${LOG_DIR}/trainer.log"
PIPELINE_CONFIG="${PROJECT}/${PACKAGE}/pipeline_config_n200p500_rolling100gb.json"
EXPECTED_PIPELINE_SHA256="$(sha256sum "${PIPELINE_CONFIG}" | awk '{print $1}')"
PENALTY_CACHE="${PROJECT}/penalty_cache_300k_v12_canonical_final.pt"
PENALTY_CACHE_SHA256="6a3fda8201de87d5e2078e6e56ad0c67e5ad2a27248865910cac8f0f36bfe1dc"
CONFIG="${PROJECT}/fpk_v33_kkt0.yaml"

test -f "${CONTROL_ROOT}/CURRENT.json"
test -f "${PENALTY_CACHE}"
test "$(sha256sum "${PENALTY_CACHE}" | awk '{print $1}')" = "${PENALTY_CACHE_SHA256}"
mkdir -p "${LOG_DIR}"

start_args=(--scratch)
if [[ -d "${OUT_DIR}" ]]; then
  latest_checkpoint="$({
    find "${OUT_DIR}" -maxdepth 1 -type f -name 'checkpoint_step*.pt' -print
  } | sort | tail -n 1)"
  if [[ -z "${latest_checkpoint}" ]]; then
    echo "Refusing to overwrite checkpoint-free run: ${OUT_DIR}" >&2
    exit 2
  fi
  resume_sha256="$(sha256sum "${latest_checkpoint}" | awk '{print $1}')"
  start_args=(--resume "${latest_checkpoint}" --resume-sha256 "${resume_sha256}")
fi

export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_P2P_DISABLE="1"
export NCCL_CUMEM_ENABLE="0"
export OMP_NUM_THREADS="4"
export MKL_NUM_THREADS="4"
export FM_INIT_TRACE="1"

set +e
/venv/main/bin/torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  -m "${PACKAGE}.train_joint_v17_stagegain_rolling" \
  --rolling-data-root "${DATA_ROOT}" \
  --rolling-control-root "${CONTROL_ROOT}" \
  --expected-pipeline-sha256 "${EXPECTED_PIPELINE_SHA256}" \
  --trainer-id "${TRAINER_ID}" \
  --out-dir "${OUT_DIR}" \
  --config "${CONFIG}" \
  --penalty-cache "${PENALTY_CACHE}" \
  --penalty-cache-sha256 "${PENALTY_CACHE_SHA256}" \
  --steps 1000 \
  --beta-lr 1e-4 \
  --feature-lr-scale 1.0 \
  --query-lr-scale 1.0 \
  --shape-lr-scale 1.0 \
  --local-lr-scale 1.0 \
  --other-lr-scale 1.0 \
  --clip-feature 20.0 \
  --clip-query 20.0 \
  --clip-shape 20.0 \
  --clip-local 20.0 \
  --clip-other 20.0 \
  --selector-lr 3e-4 \
  --direct-head-lr 1e-5 \
  --warmup-steps 100 \
  --batch-per-rank 1 \
  --grad-accum-steps 1 \
  --selector-grad-accum-steps 1 \
  --selector-update-every 4 \
  --selector-hidden 512 \
  --cache-files 2 \
  --reload-every 100 \
  --checkpoint-every 100 \
  --log-every 10 \
  --seed 202607211730 \
  --smoke-small-catalog \
  "${start_args[@]}" \
  2>&1 | tee -a "${LOG_PATH}"
trainer_rc=${PIPESTATUS[0]}
set -e
exit "${trainer_rc}"
