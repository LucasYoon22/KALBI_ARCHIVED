#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65536

readonly PROJECT="/users/minsuk/fm-lm-lambda"
readonly PACKAGE="model_v58_chebyshev_v17_scmmixture_p100k"
readonly ROOT="/users/sync_data/kalbi_tabiclv2_v12_mixed50_linear_snr_sparsity_error_p100k_100GB_target3080_seed360076430"
readonly CONFIG="${PROJECT}/${PACKAGE}/producer_config_tabicl_v12_mixedlinear_target3080_rolling100gb.json"
readonly TRAINER_ID="kalbi_v23_betaonly_target3080_effbatch32_to60k_20260726"

export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"

cd "${PROJECT}"
exec /venv/main/bin/python -u \
  -m "${PACKAGE}.rolling_curator_tabicl_v12_mixedlinear" \
  --config "${CONFIG}" \
  --root "${ROOT}" \
  --trainer-id "${TRAINER_ID}" \
  --world-size 8 \
  --bootstrap-bytes 1000000000 \
  --target-bytes 100000000000 \
  --refresh-bytes 2000000000 \
  --poll-seconds 15
