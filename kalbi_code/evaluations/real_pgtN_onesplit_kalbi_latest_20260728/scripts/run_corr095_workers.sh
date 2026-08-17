#!/usr/bin/env bash
set -euo pipefail

cd /users/minsuk/fm-lm-lambda

evaluator="evaluations/real_pgtN_onesplit_kalbi_latest_20260728/scripts/evaluate_kalbi_lasso_corr095_t100.py"
output_root="evaluations/real_pgtN_onesplit_kalbi_latest_20260728/kalbi_lasso_corr095_t100_20260728"
log_root="${output_root}/logs"
mkdir -p "${log_root}"

ranges=("1-6" "7-11" "12-16" "17-21" "22-26" "27-31" "32-36" "37-41")
pids=()

for device in "${!ranges[@]}"; do
  task_range="${ranges[$device]}"
  /venv/main/bin/python3 "${evaluator}" \
    --output-root "${output_root}" \
    --task-indices "${task_range}" \
    --device "cuda:${device}" \
    --torch-threads 2 \
    --cv-workers 2 \
    >"${log_root}/worker_${task_range}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

exit "${status}"
