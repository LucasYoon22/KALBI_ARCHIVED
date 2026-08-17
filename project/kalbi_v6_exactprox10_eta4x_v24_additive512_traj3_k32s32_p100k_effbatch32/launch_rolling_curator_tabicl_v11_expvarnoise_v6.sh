#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65536

PROJECT="/users/minsuk/fm-lm-lambda"
GENERATOR_PACKAGE="model_v58_chebyshev_v17_scmmixture_p100k"
ROOT="/users/sync_data/kalbi_tabiclv2_scm2048_qfilter_bnn128_expvarnoise_v11_p100k_100GB_seed88076430"
CONFIG="${PROJECT}/${GENERATOR_PACKAGE}/producer_config_tabicl_v11_expvarnoise_rolling100gb.json"
TRAINER_ID="kalbi_v6_v20y_exactproxpg10_eta4x_tabiclv11_p100k_effbatch32_30k_20260723"

export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"

cd "${PROJECT}"
exec /venv/main/bin/python3 -u -m \
  "${GENERATOR_PACKAGE}.rolling_curator_tabicl_v11_expvarnoise" \
  --config "${CONFIG}" \
  --root "${ROOT}" \
  --trainer-id "${TRAINER_ID}" \
  --world-size 8 \
  --bootstrap-bytes 1000000000 \
  --target-bytes 100000000000 \
  --refresh-bytes 2000000000 \
  --poll-seconds 15
