#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65536

PROJECT="${FM_PROJECT:-/users/minsuk/fm-lm-lambda}"
BUNDLE="${PROJECT}/model_v58_chebyshev_v12_canonical"
ROOT="/users/sync_data/tabiclv2_graph_scm_groupdrop095_controlledSNR1to10_medianN2000_medianP10000_100GB_tailP100K_mlp512_gaussian16_seed88073000"
CONFIG="${BUNDLE}/producer_config_v12_canonical_rolling100gb.json"
TRAINER_ID="v58chebyshev_v12_canonical_bigdata_30k_20260717"
LOG_DIR="${PROJECT}/logs/v12_canonical_rolling100gb"

mkdir -p "${LOG_DIR}"
export FM_PROJECT="${PROJECT}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"

cd "${PROJECT}"
/venv/main/bin/python3 -m model_v58_chebyshev_v12_canonical.rolling_curator \
  --config "${CONFIG}" \
  --root "${ROOT}" \
  --trainer-id "${TRAINER_ID}" \
  --world-size 8 \
  --target-bytes 100000000000 \
  --refresh-bytes 2000000000 \
  --poll-seconds 30 \
  2>&1 | tee -a "${LOG_DIR}/curator.log"
