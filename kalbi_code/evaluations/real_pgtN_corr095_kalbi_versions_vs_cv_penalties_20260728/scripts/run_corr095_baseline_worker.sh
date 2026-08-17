#!/bin/bash
set -euo pipefail

worker=${1:?worker index is required}
PROJECT=/users/minsuk/fm-lm-lambda
EVAL_ROOT=${PROJECT}/evaluations/real_pgtN_corr095_kalbi_versions_vs_cv_penalties_20260728
TASK_MAP=${EVAL_ROOT}/configs/baseline_worker_tasks.txt
task_indices=$(sed -n "$((worker + 1))p" "${TASK_MAP}")
if [[ -z "${task_indices}" ]]; then
    echo "missing task assignment for worker ${worker}" >&2
    exit 2
fi

export PYTHONPATH=${PROJECT}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
cd "${PROJECT}"

exec ${PROJECT}/.venv-kalbi-v24-eval/bin/python \
    ${EVAL_ROOT}/scripts/evaluate_real_pgtN_cv_baselines_gramcd.py \
    --evaluation-root "${EVAL_ROOT}" \
    --split-manifest \
    ${PROJECT}/evaluations/real_pgtN_onesplit_kalbi_latest_20260728/manifests/split_manifest.json \
    --validation-manifest \
    ${EVAL_ROOT}/manifests/gramcd_nonconvex_validation.json \
    --task-indices "${task_indices}" \
    --grid-size 100 \
    --min-ratio 1e-6 \
    --cv-workers 1 \
    --lasso-tolerance 1e-8 \
    --lasso-max-iter 30000 \
    --nonconvex-tolerance 1e-7 \
    --nonconvex-max-iter 1000
