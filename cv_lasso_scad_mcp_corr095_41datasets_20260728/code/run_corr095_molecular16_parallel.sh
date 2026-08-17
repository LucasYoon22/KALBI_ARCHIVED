#!/bin/bash
set -euo pipefail

project=/users/minsuk/fm-lm-lambda
evaluation_root=${project}/evaluations/real_pgtN_corr095_kalbi_versions_vs_cv_penalties_20260728

export PYTHONPATH=${project}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
cd "${project}"

exec ${project}/.venv-kalbi-v24-eval/bin/python \
    ${evaluation_root}/scripts/evaluate_real_pgtN_cv_baselines_gramcd.py \
    --evaluation-root "${evaluation_root}" \
    --split-manifest \
    ${project}/evaluations/real_pgtN_onesplit_kalbi_latest_20260728/manifests/split_manifest.json \
    --validation-manifest \
    ${evaluation_root}/manifests/gramcd_l1_scad_mcp_scalar_validation_cached.json \
    --task-indices 16 \
    --grid-size 100 \
    --min-ratio 1e-6 \
    --cv-workers 5 \
    --lasso-tolerance 1e-8 \
    --lasso-max-iter 30000 \
    --nonconvex-tolerance 1e-7 \
    --nonconvex-max-iter 1000
