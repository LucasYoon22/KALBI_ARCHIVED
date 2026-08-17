#!/usr/bin/env bash
set -euo pipefail

PROJECT="/users/minsuk/fm-lm-lambda"
PACKAGE="kalbi_v3_v17k32s32_p100k_effbatch32"
RUN_ID="kalbi_v3_v17k32s32_p100k_effbatch32_scratch_30k_20260722"
RUN_DIR="${PROJECT}/runs/${RUN_ID}"
EXPECTED_STEP="${EXPECTED_STEP:-2000}"
CHECKPOINT="${RUN_DIR}/checkpoint_step$(printf '%06d' "${EXPECTED_STEP}").pt"
CONFIG="${PROJECT}/fpk_v33_kkt0.yaml"
EVAL_ROOT="${PROJECT}/eval_data/kalbi_v3_step2000_unseen_n200p500"
MANIFEST="${EVAL_ROOT}/eval_manifest.json"
CD_LIBRARY="${PROJECT}/${PACKAGE}/canonical_cd_kernel.so"
FINAL_DIR="${PROJECT}/evaluations/kalbi_v3_step$(printf '%06d' "${EXPECTED_STEP}")_unseen_n200p500_cv"
TEMP_DIR="${FINAL_DIR}.tmp.$$.${RANDOM}"
BASELINE_PYTHON="${PROJECT}/.venv-kalbi-eval-baselines/bin/python"

test -f "${CHECKPOINT}"
test -f "${CONFIG}"
test -f "${MANIFEST}"
test -f "${CD_LIBRARY}"
test -x "${BASELINE_PYTHON}"
if [[ -e "${FINAL_DIR}" ]]; then
  printf 'Refusing existing final evaluation directory: %s\n' "${FINAL_DIR}" >&2
  exit 2
fi

mkdir -p "$(dirname "${FINAL_DIR}")"
cleanup() {
  if [[ -d "${TEMP_DIR}" ]]; then
    rm -rf -- "${TEMP_DIR}"
  fi
}
trap cleanup EXIT

export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export MPLBACKEND="Agg"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

/venv/main/bin/python -m "${PACKAGE}.report_kalbi_v3_step2000_paths" \
  --checkpoint "${CHECKPOINT}" \
  --config "${CONFIG}" \
  --eval-manifest "${MANIFEST}" \
  --cd-library "${CD_LIBRARY}" \
  --out-dir "${TEMP_DIR}/paths" \
  --device cuda:0 \
  --expected-step "${EXPECTED_STEP}" \
  --count 8 \
  --queries 41 \
  --split-seed 2026072202 \
  --max-sweeps 2000 \
  --retries 2 \
  --tolerance 1e-5 \
  --zero-tolerance 1e-10

"${BASELINE_PYTHON}" -m "${PACKAGE}.benchmark_kalbi_v3_cv_lasso_scad" \
  --paths-dir "${TEMP_DIR}/paths" \
  --paths-metrics "${TEMP_DIR}/paths/kalbi_v3_paths_metrics.json" \
  --out-dir "${TEMP_DIR}/baselines" \
  --folds 5 \
  --grid-size 25 \
  --min-ratio 1e-4 \
  --split-seed 2026072217 \
  --scad-gamma 3.7 \
  --scad-tolerance 1e-7 \
  --lasso-tolerance 1e-8

mv -- "${TEMP_DIR}" "${FINAL_DIR}"
find "${FINAL_DIR}" -type f -name '*.json' -exec \
  sed -i "s|${TEMP_DIR}|${FINAL_DIR}|g" {} +
trap - EXIT
printf 'Kalbi V3 step %s evaluation complete: %s\n' "${EXPECTED_STEP}" "${FINAL_DIR}"
