#!/bin/bash
set -euo pipefail

PROJECT=/users/minsuk/fm-lm-lambda
EVAL_ROOT=${PROJECT}/evaluations/real_pgtN_corr095_kalbi_versions_vs_cv_penalties_20260728
SCRIPT=${EVAL_ROOT}/scripts/evaluate_corr095_kalbi_bundle.py
SPLIT=${PROJECT}/evaluations/real_pgtN_onesplit_kalbi_latest_20260728/manifests/split_manifest.json
BUNDLES=${EVAL_ROOT}/manifests/prespecified_kalbi_bundles.json
CONFIG=${PROJECT}/fpk_v33_kkt0.yaml

export PYTHONPATH=${PROJECT}
export CUDA_VISIBLE_DEVICES=7
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
cd "${PROJECT}"

applicable_tasks=1,2,5,16,17,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,40

for bundle in \
    KALBI-B-v8-v23-s20300 \
    KALBI-C-v8-v24-s03900 \
    KALBI-D-v8-v24-s13100
do
    /venv/main/bin/python3 "${SCRIPT}" \
        --evaluation-root "${EVAL_ROOT}" \
        --split-manifest "${SPLIT}" \
        --bundle-manifest "${BUNDLES}" \
        --bundle-id "${bundle}" \
        --config "${CONFIG}" \
        --device cuda:0 \
        --task-indices "${applicable_tasks}" \
        --grid-size 41 \
        --torch-threads 2
done
