#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65536

PROJECT="/users/minsuk/fm-lm-lambda"
PACKAGE="model_v58_chebyshev_v17_scmmixture_n200p500"
ROOT="/users/sync_data/kalbi_tabpfnv1_tabiclv2_scm_xy_50_50_groupdrop095_N200_P500_100GB_seed88073500"
CONFIG="${PROJECT}/${PACKAGE}/producer_config_n200p500_rolling100gb.json"
TRAINER_ID="v58chebyshev_v17_n200p500_joint_scratch_30k_20260721"
LOG_DIR="${PROJECT}/logs/v58_v17_n200p500_rolling"

mkdir -p "${LOG_DIR}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"

exec /venv/main/bin/python3 -u -m "${PACKAGE}.rolling_curator" \
  --config "${CONFIG}" \
  --root "${ROOT}" \
  --trainer-id "${TRAINER_ID}" \
  --world-size 4 \
  --target-bytes 100000000000 \
  --refresh-bytes 2000000000 \
  --poll-seconds 30 \
  2>&1 | tee -a "${LOG_DIR}/curator.log"
