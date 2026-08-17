#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65536

readonly PROJECT="/users/minsuk/fm-lm-lambda"
readonly PACKAGE="model_v58_chebyshev_v17_scmmixture_p100k"
readonly ROOT="/users/sync_data/kalbi_tabiclv2_v14_mixed50_linear_pn50_strictn3_snr0p2to50_p100k_100GB_target3080_seed160076430"
readonly CONFIG="${PROJECT}/${PACKAGE}/producer_config_tabicl_v14_snr0p2_50_target3080_rolling100gb.json"
readonly WORKER_ID="${1:?worker id is required}"

export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export ROLLING_GENERATOR_TIMEOUT_SECONDS=1800

mode_args=()
if [[ ! -f "${ROOT}/rolling_control/CURRENT.json" ]]; then
  mode_args=(--allow-before-current)
fi

cd "${PROJECT}"
exec /venv/main/bin/python3 -u \
  -m "${PACKAGE}.rolling_producer_tabicl_v14_snr0p2_50_supportmix" \
  --config "${CONFIG}" \
  --root "${ROOT}" \
  --worker-id "${WORKER_ID}" \
  --python /venv/main/bin/python3 \
  "${mode_args[@]}" \
  --idle-seconds 15
