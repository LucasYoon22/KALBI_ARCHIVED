# CV LASSO, SCAD, MCP on 41 Correlation-Filtered Real Datasets

Date: 2026-07-28

## Scope

This local bundle contains the complete results and executable evaluation code
for CV LASSO, CV SCAD, and CV MCP on 41 registered real regression datasets.

The deterministic feature filter is fit using outer-training rows only:

1. Remove nonfinite or effectively constant columns.
2. Standardize each retained feature using outer-training mean and population
   standard deviation.
3. Traverse columns in stable original-column order.
4. Keep a column only when its absolute Pearson correlation with every
   previously retained column is strictly below 0.95.
5. Refit this full preprocessing pipeline independently inside every CV fold.
6. Apply the frozen outer-training transform to the outer test split.

The outer test response is first read after CV selection and outer-training
refit are frozen.

## Main Result Tables

- `results/cv_lasso_scad_mcp_41_wide.csv`
  - One row per dataset.
  - Common dataset/preprocessing fields followed by method-prefixed metrics.
- `results/cv_lasso_scad_mcp_41_long.csv`
  - 123 rows: 41 datasets times 3 methods.
  - Includes selected lambda, MSE/NMSE, R2, MAE, NNZ, KKT, and runtime.
- `results/datasets_41_after_corr095.csv`
  - One row per filtered dataset and its preprocessing fingerprints.
- `results/cv_lasso_scad_mcp_aggregate.csv`
  - Aggregate 41-task summaries.
- `results/summary.json`
  - Machine-readable cardinality and aggregate audit.

## Raw Per-Dataset Artifacts

- `results/raw_metrics_json/`
  - 41 complete per-task result JSON files.
  - Contains CV diagnostics, selected/refit statistics, test metrics, artifact
    paths, and SHA fingerprints.
- `results/coefficient_npz/`
  - 41 NPZ files.
  - Contains the 100-point lambda-ratio grid; 5-fold MSE paths; mean and
    standard-error paths; selected LASSO/SCAD/MCP coefficients; test
    predictions; retained original feature indices; and train/test row indices.
- `results/preprocessing_npz/`
  - 41 NPZ files.
  - Contains `x_mean`, `x_scale`, constant-feature mask, correlation keep
    indices, retained original indices, correlation audit, `y_mean`, `y_scale`,
    and scaler SHA.

## Solver Contract

- Lambda-ratio grid: 100 points from 1 to \(10^{-6}\).
- Selection: five-fold CV on the outer-training partition.
- LASSO:
  - sklearn cyclic coordinate descent
  - tolerance \(10^{-8}\)
  - maximum iterations 30,000
- SCAD:
  - cached-Gram coordinate descent
  - \(a=3.7\)
  - tolerance \(10^{-7}\)
  - maximum 1,000 epochs per lambda
- MCP:
  - cached-Gram coordinate descent
  - \(\gamma=3.0\)
  - tolerance \(10^{-7}\)
  - maximum 1,000 epochs per lambda

The selected ratio is multiplied by the outer-training
\(\lambda_{\max}=\max_j |X_j^\top y|/N\), and each method is refit on the
complete transformed outer-training partition before test evaluation.

## Aggregate Results

| Method | Median NMSE | Geometric mean NMSE | Median test R2 | Median NNZ |
|---|---:|---:|---:|---:|
| LASSO | 0.4270 | 0.3442 | 0.5684 | 16 |
| SCAD | 0.4213 | 0.3527 | 0.5742 | 13 |
| MCP | 0.4314 | 0.3814 | 0.5611 | 9 |

All 41 tasks passed. The largest retained absolute correlation was
0.9499982285.

## Code

- `code/evaluate_real_pgtN_cv_baselines_gramcd.py`
  - Main 41-task evaluation entrypoint.
- `code/real_pgtN_common.py`
  - Dataset loading, train-only preprocessing, correlation filtering, metrics,
    atomic output, and SHA helpers.
- `code/test_corr095_preprocessing.py`
  - Preprocessing and retained-correlation regression tests.
- `code/validate_gramcd_nonconvex_baselines.py`
  - Dense scalar-reference validation for SCAD and MCP.
- `code/validate_cached_gramcd_parity.py`
  - Cached-Gram parity validation for L1, SCAD, and MCP.
- `code/build_cv_baseline_tables.py`
  - Rebuilds the focused wide, long, dataset, and aggregate tables.
- `code/run_corr095_baseline_worker.sh`
  - Parallel worker launcher used on the source server.

## Rebuild Tables Locally

From this bundle:

```bash
python3 code/build_cv_baseline_tables.py \
  --source-root "../corr095_kalbi_versions_20260728" \
  --output-dir results
```

## Re-run Evaluation on the Source Server

The source data paths inside `manifests/split_manifest.json` refer to the
original server filesystem.

```bash
export PYTHONPATH=/users/minsuk/fm-lm-lambda

/users/minsuk/fm-lm-lambda/.venv-kalbi-v24-eval/bin/python \
  code/evaluate_real_pgtN_cv_baselines_gramcd.py \
  --evaluation-root /path/to/new_evaluation_root \
  --split-manifest manifests/split_manifest.json \
  --validation-manifest \
    manifests/gramcd_l1_scad_mcp_scalar_validation_cached.json \
  --task-indices all \
  --grid-size 100 \
  --min-ratio 1e-6 \
  --cv-workers 5 \
  --lasso-tolerance 1e-8 \
  --lasso-max-iter 30000 \
  --nonconvex-tolerance 1e-7 \
  --nonconvex-max-iter 1000
```

## Verification

- Dataset count: 41.
- Long-table rows: 123.
- Raw metric JSON files: 41.
- Coefficient/CV-path NPZ files: 41.
- Preprocessing NPZ files: 41.
- Leakage audit: PASS.
- Correlation-filter audit: PASS.
- L1/SCAD/MCP numerical validation: PASS.
- Bundle checksum manifest: `SHA256SUMS`.
