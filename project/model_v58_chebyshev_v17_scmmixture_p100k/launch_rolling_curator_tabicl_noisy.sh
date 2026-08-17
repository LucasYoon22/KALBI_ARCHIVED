#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65536

PROJECT="/users/minsuk/fm-lm-lambda"
PACKAGE="model_v58_chebyshev_v17_scmmixture_p100k"
ROOT="/users/sync_data/kalbi_tabiclv2_scm_xy_noisyexp1mix_groupdrop095_100GB_seed420807355"
CONFIG="${PROJECT}/${PACKAGE}/producer_config_tabicl_noisy_rolling100gb.json"
TRAINER_ID="kalbi_v3_v17penmlp_tabicl_noisy_v6_p100k_effbatch32_30k_20260722"
LOG_DIR="${PROJECT}/logs/kalbi_tabiclv2_noisy_v6_p100k_rolling"

mkdir -p "${LOG_DIR}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"

cd "${PROJECT}"
exec /venv/main/bin/python3 -u -m "${PACKAGE}.rolling_curator_tabicl_noisy" \
  --config "${CONFIG}" \
  --root "${ROOT}" \
  --trainer-id "${TRAINER_ID}" \
  --world-size 8 \
  --target-bytes 100000000000 \
  --refresh-bytes 2000000000 \
  --poll-seconds 30 \
  2>&1 | tee -a "${LOG_DIR}/curator.log"
