#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65536

PROJECT="/users/minsuk/fm-lm-lambda"
SOURCE_PACKAGE="model_v58_chebyshev_v14_scmmixture"
ROOT="/users/sync_data/kalbi_tabpfnv1_tabiclv2_scm_xy_50_50_groupdrop095_100GB_seed88073400"
CONFIG="${PROJECT}/${SOURCE_PACKAGE}/producer_config_v14_scmmixture_rolling100gb.json"
TRAINER_ID="kalbi_v2_v17_p100k_effbatch32_scratch_30k_20260721"
LOG_DIR="${PROJECT}/logs/kalbi_v2_v17_p100k_rolling"

mkdir -p "${LOG_DIR}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"

exec /venv/main/bin/python3 -u -m "${SOURCE_PACKAGE}.rolling_curator" \
  --config "${CONFIG}" \
  --root "${ROOT}" \
  --trainer-id "${TRAINER_ID}" \
  --world-size 8 \
  --target-bytes 100000000000 \
  --refresh-bytes 2000000000 \
  --poll-seconds 30 \
  >> "${LOG_DIR}/curator.log" 2>&1
