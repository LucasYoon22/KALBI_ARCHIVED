#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65536

readonly PROJECT="/users/minsuk/fm-lm-lambda"
readonly PACKAGE="kalbi_v6_exactprox10_eta4x_v23_independentraw_k32s32_p100k_effbatch32"
readonly DATA_ROOT="/users/sync_data/kalbi_tabiclv2_scm2048_qfilter_bnn128_expvarnoise_v11_p100k_100GB_target3080_seed310076430"
readonly CONTROL_ROOT="${DATA_ROOT}/rolling_control"
readonly TRAINER_ID="kalbi_v23_betaonly_target3080_effbatch32_to60k_20260726"
readonly OUT_DIR="${PROJECT}/runs/${TRAINER_ID}"
readonly CONFIG="${PROJECT}/fpk_v33_kkt0.yaml"
readonly PENALTY_CACHE="${PROJECT}/penalty_cache_300k_v12_canonical_final.pt"
readonly PENALTY_CACHE_SHA256="6a3fda8201de87d5e2078e6e56ad0c67e5ad2a27248865910cac8f0f36bfe1dc"
readonly INPUT_SCALER_CACHE="${PROJECT}/${PACKAGE}/v22_input_scaler_train64.pt"
readonly INPUT_SCALER_CACHE_SHA256="d574f1c50ed7fa9a46b42b012524cad48961ddc9ced33b50b8d0857281202cc1"
readonly EXPECTED_PIPELINE_SHA256="74663e247f907eaf8c9905d2ef1f9683c3709f9461d4d31dc6fcc65ec2567e4b"

test -f "${CONTROL_ROOT}/CURRENT.json"
test -f "${PENALTY_CACHE}"
test "$(sha256sum "${PENALTY_CACHE}" | awk '{print $1}')" = "${PENALTY_CACHE_SHA256}"
test -f "${INPUT_SCALER_CACHE}"
test "$(sha256sum "${INPUT_SCALER_CACHE}" | awk '{print $1}')" = "${INPUT_SCALER_CACHE_SHA256}"

latest_checkpoint="$(
  find "${OUT_DIR}" -maxdepth 1 -type f -name 'checkpoint_step*.pt' -print \
    | sort -V | tail -n 1
)"
test -n "${latest_checkpoint}"
resume_sha256="$(sha256sum "${latest_checkpoint}" | awk '{print $1}')"

source_mode="$(
  /venv/main/bin/python - "${latest_checkpoint}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
beta_only = checkpoint["metadata"]["exact_training_arguments"]["beta_only"]
print("beta_only" if beta_only else "joint")
PY
)"

transition_args=()
if [[ "${source_mode}" == "joint" ]]; then
  transition_args=(--resume-beta-only-from-joint)
elif [[ "${source_mode}" != "beta_only" ]]; then
  echo "Unsupported source training mode: ${source_mode}" >&2
  exit 2
fi

stop_args=()
if [[ -n "${KALBI_STOP_AFTER:-}" ]]; then
  stop_args=(--stop-after "${KALBI_STOP_AFTER}")
fi

export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export FM_INIT_TRACE=1

cd "${PROJECT}"
exec taskset -c 0-31,64-79 /venv/main/bin/torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  -m "${PACKAGE}.train_beta_only_v23_strict_resume_8gpu" \
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
  --steps 60000 \
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
  --beta-only \
  --resume "${latest_checkpoint}" \
  --resume-sha256 "${resume_sha256}" \
  "${transition_args[@]}" \
  "${stop_args[@]}"
