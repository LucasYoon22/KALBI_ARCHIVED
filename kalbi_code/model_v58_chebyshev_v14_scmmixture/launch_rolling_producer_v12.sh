#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65536

PROJECT="${FM_PROJECT:-/users/minsuk/fm-lm-lambda}"
BUNDLE="${PROJECT}/model_v58_chebyshev_v12_canonical"
ROOT="/users/sync_data/tabiclv2_graph_scm_groupdrop095_controlledSNR1to10_medianN2000_medianP10000_100GB_tailP100K_mlp512_gaussian16_seed88073000"
CONFIG="${BUNDLE}/producer_config_v12_canonical_rolling100gb.json"
WORKER_ID="${1:?worker id is required}"
LOG_DIR="${PROJECT}/logs/v12_canonical_rolling100gb"

mkdir -p "${LOG_DIR}"
export FM_PROJECT="${PROJECT}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"

# Bootstrap may fill the initial catalog before CURRENT exists.  Once CURRENT
# has been published, omit this flag so active/history shards are subtracted
# from staging capacity and the 4GB rolling high-water contract takes over.
mode_args=()
if [[ ! -f "${ROOT}/rolling_control/CURRENT.json" ]]; then
  mode_args=(--allow-before-current)
fi

cd "${PROJECT}"
/venv/main/bin/python3 -m model_v58_chebyshev_v12_canonical.rolling_producer \
  --config "${CONFIG}" \
  --root "${ROOT}" \
  --worker-id "${WORKER_ID}" \
  --python /venv/main/bin/python3 \
  "${mode_args[@]}" \
  --idle-seconds 30 \
  2>&1 | tee -a "${LOG_DIR}/producer_${WORKER_ID}.log"
