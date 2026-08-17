#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65536

PROJECT="/users/minsuk/fm-lm-lambda"
PACKAGE="model_v58_chebyshev_v17_scmmixture_n200p500"
ROOT="/users/sync_data/kalbi_tabpfnv1_tabiclv2_scm_xy_50_50_groupdrop095_N200_P500_100GB_seed88073500"
CONFIG="${PROJECT}/${PACKAGE}/producer_config_n200p500_rolling100gb.json"
WORKER_ID="${1:?worker id is required}"
LOG_DIR="${PROJECT}/logs/v58_v17_n200p500_rolling"

mkdir -p "${LOG_DIR}"
export FM_PROJECT="${PROJECT}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"

mode_args=()
if [[ ! -f "${ROOT}/rolling_control/CURRENT.json" ]]; then
  mode_args=(--allow-before-current)
fi

cd "${PROJECT}"
exec /venv/main/bin/python3 -u -m "${PACKAGE}.rolling_producer_backpressure_fixed" \
  --config "${CONFIG}" \
  --root "${ROOT}" \
  --worker-id "${WORKER_ID}" \
  --python /venv/main/bin/python3 \
  "${mode_args[@]}" \
  --idle-seconds 30 \
  2>&1 | tee -a "${LOG_DIR}/producer_${WORKER_ID}.log"
