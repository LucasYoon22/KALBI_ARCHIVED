#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65536

PROJECT="/users/minsuk/fm-lm-lambda"
PACKAGE="kalbi_v6_exactprox10_eta4x_v22_v20geomscaled_k32s32_p100k_effbatch32"
DATA_ROOT="/users/sync_data/kalbi_tabiclv2_scm2048_qfilter_bnn128_expvarnoise_v11_p100k_100GB_seed88076430"
CONTROL_ROOT="${DATA_ROOT}/rolling_control"
TRAINER_ID="kalbi_v6_v22_v20geomscaled_tabiclv11_p100k_effbatch32_30k_20260725"
OUT_DIR="${PROJECT}/runs/${TRAINER_ID}"
LOG_DIR="${PROJECT}/logs/${TRAINER_ID}"
LOG_PATH="${LOG_DIR}/trainer.log"
V20_OUT_DIR="${PROJECT}/runs/kalbi_v6_v20y_exactproxpg10_eta4x_tabiclv11_p100k_effbatch32_30k_20260723"
V21_OUT_DIR="${PROJECT}/runs/kalbi_v6_v21_dimraw_deriv96_tabiclv11_p100k_effbatch32_30k_20260724"
INPUT_SCALER_CACHE="${PROJECT}/${PACKAGE}/v22_input_scaler_train64.pt"
EXPECTED_PIPELINE_SHA256="74663e247f907eaf8c9905d2ef1f9683c3709f9461d4d31dc6fcc65ec2567e4b"
PENALTY_CACHE="${PROJECT}/penalty_cache_300k_v12_canonical_final.pt"
PENALTY_CACHE_SHA256="6a3fda8201de87d5e2078e6e56ad0c67e5ad2a27248865910cac8f0f36bfe1dc"
CONFIG="${PROJECT}/fpk_v33_kkt0.yaml"

test -f "${CONTROL_ROOT}/CURRENT.json"
test -f "${PENALTY_CACHE}"
test "$(sha256sum "${PENALTY_CACHE}" | awk '{print $1}')" = "${PENALTY_CACHE_SHA256}"
test -f "${INPUT_SCALER_CACHE}"
INPUT_SCALER_CACHE_SHA256="$(sha256sum "${INPUT_SCALER_CACHE}" | awk '{print $1}')"
mkdir -p "${LOG_DIR}"

latest_checkpoint=""
if [[ -d "${OUT_DIR}" ]]; then
  latest_checkpoint="$({
    find "${OUT_DIR}" -maxdepth 1 -type f -name 'checkpoint_step*.pt' -print
  } | sort -V | tail -n 1)"
fi
if [[ -n "${latest_checkpoint}" ]]; then
  resume_sha256="$(sha256sum "${latest_checkpoint}" | awk '{print $1}')"
  start_args=(--resume "${latest_checkpoint}" --resume-sha256 "${resume_sha256}")
else
  v21_checkpoint="$({
    find "${V21_OUT_DIR}" -maxdepth 1 -type f -name 'checkpoint_step*.pt' -print
  } | sort -V | tail -n 1)"
  v20_checkpoint="$({
    find "${V20_OUT_DIR}" -maxdepth 1 -type f -name 'checkpoint_step*.pt' -print
  } | sort -V | tail -n 1)"
  test -n "${v21_checkpoint}"
  test -f "${v21_checkpoint}"
  test -n "${v20_checkpoint}"
  test -f "${v20_checkpoint}"
  v21_sha256="$(sha256sum "${v21_checkpoint}" | awk '{print $1}')"
  v20_sha256="$(sha256sum "${v20_checkpoint}" | awk '{print $1}')"
  start_args=(
    --transition-v21-from "${v21_checkpoint}"
    --transition-v21-sha256 "${v21_sha256}"
    --selector-v20-from "${v20_checkpoint}"
    --selector-v20-sha256 "${v20_sha256}"
  )
fi

export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export FM_INIT_TRACE="1"

set +e
/venv/main/bin/torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  -m "${PACKAGE}.train_joint_v6_v22_k32s32_tabicl_v11_expvarnoise_rolling_8gpu" \
  --rolling-data-root "${DATA_ROOT}" \
  --rolling-control-root "${CONTROL_ROOT}" \
  --expected-pipeline-sha256 "${EXPECTED_PIPELINE_SHA256}" \
  --trainer-id "${TRAINER_ID}" \
  --out-dir "${OUT_DIR}" \
  --config "${CONFIG}" \
  --penalty-cache "${PENALTY_CACHE}" \
  --penalty-cache-sha256 "${PENALTY_CACHE_SHA256}" \
  --input-scaler-cache "${INPUT_SCALER_CACHE}" \
  --input-scaler-cache-sha256 "${INPUT_SCALER_CACHE_SHA256}" \
  --steps 30000 \
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
  --direct-head-lr 3e-4 \
  --warmup-steps 500 \
  --batch-per-rank 1 \
  --grad-accum-steps 4 \
  --selector-grad-accum-steps 4 \
  --selector-update-every 4 \
  --selector-hidden 512 \
  --cache-files 2 \
  --reload-every 100 \
  --checkpoint-every 100 \
  --log-every 10 \
  --seed 202607232355 \
  "${start_args[@]}" \
  2>&1 | tee -a "${LOG_PATH}"
trainer_rc=${PIPESTATUS[0]}
set -e
exit "${trainer_rc}"
