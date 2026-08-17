#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65536

readonly PROJECT="/users/minsuk/fm-lm-lambda"
readonly PACKAGE="kalbi_v6_exactprox10_eta4x_v24_additive512_traj3_k32s32_p100k_effbatch32"
readonly MODULE="${PACKAGE}.train_selector_only_v24_frozen_kalbi_v6_cachedtraj_v14_transition_8gpu"
readonly DATA_ROOT="/users/sync_data/kalbi_tabiclv2_v14_mixed50_linear_pn50_strictn3_snr0p2to50_p100k_100GB_target3080_seed160076430"
readonly CONTROL_ROOT="${DATA_ROOT}/rolling_control"
readonly V13_RUN="${PROJECT}/runs/kalbi_v24_selectoronly_cachedtraj3_frozenbeta_step42700_effbatch32_20260727"
readonly DEFAULT_TRAINER_ID="kalbi_v24_selectoronly_v14_snr0p2to50_frozenbeta_step42700_effbatch32_20260728"
readonly DEFAULT_OUT_DIR="${PROJECT}/runs/${DEFAULT_TRAINER_ID}"
readonly CONFIG="${PROJECT}/fpk_v33_kkt0.yaml"
readonly PENALTY_CACHE="${PROJECT}/penalty_cache_300k_v12_canonical_final.pt"
readonly PENALTY_CACHE_SHA256="6a3fda8201de87d5e2078e6e56ad0c67e5ad2a27248865910cac8f0f36bfe1dc"
readonly EXPECTED_PIPELINE_SHA256="4fd706b7b254eddb4a832b8107fa3379d128f152716c7ae9f142539064d7e707"
readonly INPUT_SCALER="${PROJECT}/${PACKAGE}/../kalbi_v6_exactprox10_eta4x_v23_independentraw_k32s32_p100k_effbatch32/v22_input_scaler_train64.pt"
readonly INPUT_SCALER_SHA256="d574f1c50ed7fa9a46b42b012524cad48961ddc9ced33b50b8d0857281202cc1"
readonly V24_SCALER="${PROJECT}/${PACKAGE}/v24_scaler_rolling256_step42700.pt"
readonly V24_SCALER_SHA256="475e34d76c0ef936d83a43dbb3a616359a4d71213f7c42ec5a5aad06b78dbab8"

trainer_id="${KALBI_TRAINER_ID:-${DEFAULT_TRAINER_ID}}"
out_dir="${KALBI_OUT_DIR:-${DEFAULT_OUT_DIR}}"

latest_checkpoint=""
if [[ -d "${out_dir}" ]]; then
  latest_checkpoint="$(
    find "${out_dir}" -maxdepth 1 -type f \
      -name 'checkpoint_selector_step*.pt' -print \
      | sort -V | tail -n 1
  )"
fi

transition_args=()
if [[ -z "${latest_checkpoint}" ]]; then
  latest_checkpoint="$(
    find "${V13_RUN}" -maxdepth 1 -type f \
      -name 'checkpoint_selector_step*.pt' -print \
      | sort -V | tail -n 1
  )"
  transition_args=(--resume-data-transition-from-v13)
fi

test -n "${latest_checkpoint}"
test -f "${latest_checkpoint}"
test -f "${PENALTY_CACHE}"
test -f "${INPUT_SCALER}"
test -f "${V24_SCALER}"
test "$(sha256sum "${PENALTY_CACHE}" | awk '{print $1}')" = "${PENALTY_CACHE_SHA256}"
test "$(sha256sum "${INPUT_SCALER}" | awk '{print $1}')" = "${INPUT_SCALER_SHA256}"
test "$(sha256sum "${V24_SCALER}" | awk '{print $1}')" = "${V24_SCALER_SHA256}"
resume_sha256="$(sha256sum "${latest_checkpoint}" | awk '{print $1}')"

stop_args=()
if [[ -n "${KALBI_STOP_AFTER:-}" ]]; then
  stop_args=(--stop-after "${KALBI_STOP_AFTER}")
fi

export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_ASYNC_ERROR_HANDLING=1
export FM_INIT_TRACE=1

cd "${PROJECT}"
exec /venv/main/bin/torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  -m "${MODULE}" \
  --rolling-data-root "${DATA_ROOT}" \
  --rolling-control-root "${CONTROL_ROOT}" \
  --expected-pipeline-sha256 "${EXPECTED_PIPELINE_SHA256}" \
  --trainer-id "${trainer_id}" \
  --out-dir "${out_dir}" \
  --config "${CONFIG}" \
  --penalty-cache "${PENALTY_CACHE}" \
  --penalty-cache-sha256 "${PENALTY_CACHE_SHA256}" \
  --input-scaler-cache "${INPUT_SCALER}" \
  --input-scaler-cache-sha256 "${INPUT_SCALER_SHA256}" \
  --v24-scaler-cache "${V24_SCALER}" \
  --v24-scaler-cache-sha256 "${V24_SCALER_SHA256}" \
  --steps 30000 \
  --selector-lr 3e-4 \
  --direct-head-lr 3e-4 \
  --warmup-steps 500 \
  --batch-per-rank 1 \
  --grad-accum-steps 4 \
  --selector-grad-accum-steps 4 \
  --selector-update-every 1 \
  --selector-hidden 512 \
  --cache-files 2 \
  --reload-every 100 \
  --checkpoint-every 100 \
  --log-every 1 \
  --seed 202607232355 \
  --resume "${latest_checkpoint}" \
  --resume-sha256 "${resume_sha256}" \
  "${transition_args[@]}" \
  "${stop_args[@]}"
