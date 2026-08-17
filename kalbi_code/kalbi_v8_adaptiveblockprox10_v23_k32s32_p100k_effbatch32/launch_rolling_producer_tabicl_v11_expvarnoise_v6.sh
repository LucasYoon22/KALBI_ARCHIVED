#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65536

PROJECT="/users/minsuk/fm-lm-lambda"
V6_PACKAGE="kalbi_v6_exactprox10_eta4x_v20y_k32s32_p100k_effbatch32"
GENERATOR_PACKAGE="model_v58_chebyshev_v17_scmmixture_p100k"
ROOT="/users/sync_data/kalbi_tabiclv2_scm2048_qfilter_bnn128_expvarnoise_v11_p100k_100GB_seed88076430"
CONFIG="${PROJECT}/${GENERATOR_PACKAGE}/producer_config_tabicl_v11_expvarnoise_rolling100gb.json"
WORKER_ID="${1:?worker id is required}"

export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export ROLLING_GENERATOR_TIMEOUT_SECONDS=1800

mode_args=()
if [[ ! -f "${ROOT}/rolling_control/CURRENT.json" ]]; then
  mode_args=(--allow-before-current)
fi

cd "${PROJECT}"
exec /venv/main/bin/python3 -u -m \
  "${V6_PACKAGE}.rolling_producer_tabicl_v11_cached_history_v6" \
  --config "${CONFIG}" \
  --root "${ROOT}" \
  --worker-id "${WORKER_ID}" \
  --python /venv/main/bin/python3 \
  "${mode_args[@]}" \
  --idle-seconds 15
