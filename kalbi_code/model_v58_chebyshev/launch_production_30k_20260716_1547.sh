#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65536

PROJECT="/users/minsuk/fm-lm-lambda"
export OUT_DIR="${PROJECT}/runs/v58chebyshev_v10_joint_N200_P400_30k_20260716_1547"
export STEPS="30000"
export LOG_EVERY="100"
export SELECTOR_GRAD_ACCUM_STEPS="4"
unset STOP_AFTER || true

# Resume from the newest complete production checkpoint on controlled or
# unexpected restarts.  A fresh launch still falls back to the immutable
# V57-Chebyshev warm start configured by launch_joint_v10.sh.
latest_checkpoint="$(
  find "${OUT_DIR}" -maxdepth 1 -type f -name 'checkpoint_step*.pt' 2>/dev/null \
    | sort | tail -n 1
)"
if [[ -n "${latest_checkpoint}" ]]; then
  export RESUME="${latest_checkpoint}"
  export RESUME_SHA256="$(sha256sum "${RESUME}" | awk '{print $1}')"
else
  unset RESUME RESUME_SHA256 || true
fi

exec /bin/bash "${PROJECT}/model_v58_chebyshev/launch_joint_v10.sh"
