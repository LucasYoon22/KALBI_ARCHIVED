#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65536

PROJECT="/users/minsuk/fm-lm-lambda"
export OUT_DIR="${PROJECT}/runs/v58chebyshev_v11_direct96_joint_N200_P400_30k_20260716_2346"
export STEPS="30000"
export LOG_EVERY="100"
export SELECTOR_GRAD_ACCUM_STEPS="4"
export WARM_START="${PROJECT}/runs/v58chebyshev_v10_joint_N200_P400_30k_20260716_1547/checkpoint_step006000.pt"
export WARM_START_SHA256="f8b63e5573f5d6134dfbee5fc9b0c97c51a4c6fec33061a307baa440e1bc74c4"
unset STOP_AFTER || true

# Resume only from a complete checkpoint belonging to this V11 run. A first
# launch migrates the compatible V10 selector state and strictly warm-starts
# the V58-Chebyshev beta weights from the immutable step-6000 checkpoint.
latest_checkpoint=""
if [[ -d "${OUT_DIR}" ]]; then
  latest_checkpoint="$(
    find "${OUT_DIR}" -maxdepth 1 -type f -name 'checkpoint_step*.pt' \
      | sort | tail -n 1
  )"
fi
if [[ -n "${latest_checkpoint}" ]]; then
  export RESUME="${latest_checkpoint}"
  export RESUME_SHA256="$(sha256sum "${RESUME}" | awk '{print $1}')"
else
  unset RESUME RESUME_SHA256 || true
fi

exec /bin/bash "${PROJECT}/model_v58_chebyshev_v11_direct/launch_joint_v11_direct.sh"
