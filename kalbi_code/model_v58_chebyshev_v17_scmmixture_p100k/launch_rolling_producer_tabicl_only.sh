#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65536

PROJECT="/users/minsuk/fm-lm-lambda"
PACKAGE="model_v58_chebyshev_v17_scmmixture_p100k"
ROOT="/users/sync_data/kalbi_tabiclv2_scm_xy_only_groupdrop095_100GB_seed1688073552"
CONFIG="${PROJECT}/${PACKAGE}/producer_config_tabicl_only_rolling100gb.json"
WORKER_ID="${1:?worker id is required}"
LOG_DIR="${PROJECT}/logs/kalbi_tabiclv2_only_p100k_rolling"

mkdir -p "${LOG_DIR}"
export FM_PROJECT="${PROJECT}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"

mode_args=()
if [[ ! -f "${ROOT}/rolling_control/CURRENT.json" ]]; then
  mode_args=(--allow-before-current)
fi

cd "${PROJECT}"
exec /venv/main/bin/python3 -u -m "${PACKAGE}.rolling_producer_tabicl_only" \
  --config "${CONFIG}" \
  --root "${ROOT}" \
  --worker-id "${WORKER_ID}" \
  --python /venv/main/bin/python3 \
  "${mode_args[@]}" \
  --idle-seconds 30 \
  2>&1 | tee -a "${LOG_DIR}/producer_${WORKER_ID}.log"
