#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65536

readonly PROJECT="/users/minsuk/fm-lm-lambda"
readonly PACKAGE="model_v58_chebyshev_v17_scmmixture_p100k"
readonly ROOT="/users/sync_data/kalbi_tabiclv2_v14_mixed50_linear_pn50_strictn3_snr0p2to50_p100k_100GB_target3080_seed160076430"
readonly CONFIG="${PROJECT}/${PACKAGE}/producer_config_tabicl_v14_snr0p2_50_target3080_rolling100gb.json"
readonly TRAINER_ID="${1:-kalbi_v14_beta_trainer_pending}"

export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"

cd "${PROJECT}"
exec /venv/main/bin/python3 -u \
  -m "${PACKAGE}.rolling_curator_tabicl_v14_snr0p2_50_supportmix" \
  --config "${CONFIG}" \
  --root "${ROOT}" \
  --trainer-id "${TRAINER_ID}" \
  --world-size 8 \
  --bootstrap-bytes 1000000000 \
  --target-bytes 100000000000 \
  --refresh-bytes 2000000000 \
  --poll-seconds 15
