#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65536

PROJECT="/users/minsuk/fm-lm-lambda"
PACKAGE="model_v58_chebyshev_v14_scmmixture"
ROOT="/users/sync_data/kalbi_tabpfnv1_tabiclv2_scm_xy_50_50_groupdrop095_100GB_seed88073400"
CONFIG="${PROJECT}/${PACKAGE}/producer_config_v14_scmmixture_rolling100gb.json"
TRAINER_ID="v58chebyshev_betaonly_scmmixture_30k_20260717"
LOG_DIR="${PROJECT}/logs/v14_scmmixture_rolling100gb"
LOG_PATH="${LOG_DIR}/curator_betaonly.log"

mkdir -p "${LOG_DIR}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"

/venv/main/bin/python3 -u -m "${PACKAGE}.rolling_curator" \
  --config "${CONFIG}" \
  --root "${ROOT}" \
  --trainer-id "${TRAINER_ID}" \
  --world-size 8 \
  --target-bytes 100000000000 \
  --refresh-bytes 2000000000 \
  --poll-seconds 30 \
  2>&1 | tee -a "${LOG_PATH}"
