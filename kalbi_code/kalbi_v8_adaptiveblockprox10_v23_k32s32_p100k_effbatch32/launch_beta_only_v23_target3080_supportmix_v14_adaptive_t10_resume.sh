#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65536

readonly PROJECT="/users/minsuk/fm-lm-lambda"
readonly PACKAGE="kalbi_v8_adaptiveblockprox10_v23_k32s32_p100k_effbatch32"
readonly DATA_ROOT="/users/sync_data/kalbi_tabiclv2_v14_mixed50_linear_pn50_strictn3_snr0p2to50_p100k_100GB_target3080_seed160076430"
readonly PRODUCTION_CONTROL_ROOT="${DATA_ROOT}/rolling_control"
readonly SOURCE_CHECKPOINT="${PROJECT}/runs/kalbi_v23_betaonly_supportmix_v13_target3080_effbatch32_to60k_20260726/checkpoint_step042900.pt"
readonly SOURCE_SHA256="f6c43ffb941c508df040327fdbe47da2a5889e4b3100bdb895e3c2ca7c9d300f"
readonly DEFAULT_TRAINER_ID="kalbi_v8_adaptive_t10_betaonly_supportmix_v14_target3080_effbatch32_to60k_20260728"
readonly CONFIG="${PROJECT}/fpk_v33_kkt0.yaml"
readonly PENALTY_CACHE="${PROJECT}/penalty_cache_300k_v12_canonical_final.pt"
readonly PENALTY_CACHE_SHA256="6a3fda8201de87d5e2078e6e56ad0c67e5ad2a27248865910cac8f0f36bfe1dc"
readonly EXPECTED_PIPELINE_SHA256="4fd706b7b254eddb4a832b8107fa3379d128f152716c7ae9f142539064d7e707"

trainer_id="${KALBI_TRAINER_ID:-${DEFAULT_TRAINER_ID}}"
out_dir="${KALBI_OUT_DIR:-${PROJECT}/runs/${trainer_id}}"
control_root="${KALBI_ROLLING_CONTROL_ROOT:-${PRODUCTION_CONTROL_ROOT}}"

test -f "${SOURCE_CHECKPOINT}"
test "$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')" = "${SOURCE_SHA256}"
test -f "${PENALTY_CACHE}"
test "$(sha256sum "${PENALTY_CACHE}" | awk '{print $1}')" = "${PENALTY_CACHE_SHA256}"

latest_checkpoint=""
if [[ -d "${out_dir}" ]]; then
  latest_checkpoint="$(
    find "${out_dir}" -maxdepth 1 -type f -name 'checkpoint_step*.pt' -print \
      | sort -V | tail -n 1
  )"
fi
transition_args=()
if [[ -z "${latest_checkpoint}" ]]; then
  latest_checkpoint="${SOURCE_CHECKPOINT}"
  transition_args=(
    --resume-refinement-and-data-transition-from-v6-pg10-v13
  )
fi
resume_sha256="$(sha256sum "${latest_checkpoint}" | awk '{print $1}')"

export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd "${PROJECT}"
exec taskset -c 0-31,64-79 /venv/main/bin/torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  -m "${PACKAGE}.train_beta_only_v23_supportmix_v14_adaptive_t10_8gpu" \
  --rolling-data-root "${DATA_ROOT}" \
  --rolling-control-root "${control_root}" \
  --expected-pipeline-sha256 "${EXPECTED_PIPELINE_SHA256}" \
  --trainer-id "${trainer_id}" \
  --out-dir "${out_dir}" \
  --config "${CONFIG}" \
  --penalty-cache "${PENALTY_CACHE}" \
  --penalty-cache-sha256 "${PENALTY_CACHE_SHA256}" \
  --steps 60000 \
  --beta-lr 1e-4 \
  --feature-lr-scale 1.0 \
  --query-lr-scale 1.0 \
  --shape-lr-scale 1.0 \
  --other-lr-scale 1.0 \
  --clip-feature 20.0 \
  --clip-query 20.0 \
  --clip-shape 20.0 \
  --clip-other 20.0 \
  --warmup-steps 500 \
  --batch-per-rank 1 \
  --grad-accum-steps 4 \
  --cache-files 2 \
  --reload-every 100 \
  --checkpoint-every 100 \
  --log-every 10 \
  --seed 202607232355 \
  --resume "${latest_checkpoint}" \
  --resume-sha256 "${resume_sha256}" \
  "${transition_args[@]}"
