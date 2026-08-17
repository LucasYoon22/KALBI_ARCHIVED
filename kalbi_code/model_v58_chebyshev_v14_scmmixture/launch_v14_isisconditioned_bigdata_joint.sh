#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65536

PROJECT="/users/minsuk/fm-lm-lambda"
PACKAGE="model_v58_chebyshev_v14_isisconditioned"
DATA_ROOT="/users/sync_data/tabiclv2_graph_scm_groupdrop095_controlledSNR1to10_medianN2000_medianP10000_100GB_tailP100K_mlp512_gaussian16_seed88073000"
CONTROL_ROOT="${DATA_ROOT}/rolling_control"
TRAINER_ID="v58chebyshev_v14_isisconditioned_bigdata_30k_20260717"
OUT_DIR="${PROJECT}/runs/${TRAINER_ID}"
LOG_DIR="${PROJECT}/logs/v14_isisconditioned_rolling100gb"
LOG_PATH="${LOG_DIR}/trainer.log"

: "${WARM_START:?WARM_START must pin one immutable V58/V11 checkpoint}"
: "${WARM_START_SHA256:?WARM_START_SHA256 is required}"
: "${EXPECTED_PIPELINE_SHA256:?EXPECTED_PIPELINE_SHA256 is required}"
: "${PENALTY_CACHE_SHA256:?PENALTY_CACHE_SHA256 is required}"

PENALTY_CACHE="${PROJECT}/penalty_cache_300k_v12_canonical_final.pt"
CONFIG="${PROJECT}/fpk_v33_kkt0.yaml"

test -f "${WARM_START}"
test "$(sha256sum "${WARM_START}" | awk '{print $1}')" = "${WARM_START_SHA256}"
test -f "${PENALTY_CACHE}"
test "$(sha256sum "${PENALTY_CACHE}" | awk '{print $1}')" = "${PENALTY_CACHE_SHA256}"
test -f "${CONTROL_ROOT}/CURRENT.json"
mkdir -p "${LOG_DIR}"

resume_args=()
if [[ -d "${OUT_DIR}" ]]; then
  latest_checkpoint="$({
    find "${OUT_DIR}" -maxdepth 1 -type f -name 'checkpoint_step*.pt' -print
  } | sort | tail -n 1)"
  if [[ -n "${latest_checkpoint}" ]]; then
    resume_sha256="$(sha256sum "${latest_checkpoint}" | awk '{print $1}')"
    resume_args=(--resume "${latest_checkpoint}" --resume-sha256 "${resume_sha256}")
  else
    echo "Refusing to overwrite checkpoint-free existing V14 run: ${OUT_DIR}" >&2
    exit 2
  fi
fi

export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
set +e
/venv/main/bin/torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  -m "${PACKAGE}.train_joint_v14_isisconditioned_rolling" \
  --rolling-data-root "${DATA_ROOT}" \
  --rolling-control-root "${CONTROL_ROOT}" \
  --expected-pipeline-sha256 "${EXPECTED_PIPELINE_SHA256}" \
  --trainer-id "${TRAINER_ID}" \
  --out-dir "${OUT_DIR}" \
  --config "${CONFIG}" \
  --penalty-cache "${PENALTY_CACHE}" \
  --penalty-cache-sha256 "${PENALTY_CACHE_SHA256}" \
  --warm-start "${WARM_START}" \
  --warm-start-sha256 "${WARM_START_SHA256}" \
  --steps 30000 \
  --beta-lr 2e-4 \
  --selector-lr 3e-4 \
  --direct-head-lr 1e-5 \
  --warmup-steps 2000 \
  --batch-per-rank 1 \
  --grad-accum-steps 4 \
  --selector-grad-accum-steps 4 \
  --selector-hidden 512 \
  --cache-files 2 \
  --reload-every 500 \
  --checkpoint-every 500 \
  --log-every 10 \
  --seed 202607171410 \
  "${resume_args[@]}" \
  2>&1 | tee -a "${LOG_PATH}"
trainer_rc=${PIPESTATUS[0]}
set -e
exit "${trainer_rc}"
