#!/usr/bin/env python3
"""Build focused 41-task CV LASSO/SCAD/MCP result tables."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METHODS = ("lasso", "scad", "mcp")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    metric_paths = sorted(
        (args.source_root / "metrics").glob("*_baselines.json")
    )
    if len(metric_paths) != 41:
        raise RuntimeError(
            f"expected 41 baseline metric files, found {len(metric_paths)}"
        )

    dataset_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    for path in metric_paths:
        metric = load_json(path)
        if metric.get("status") != "PASS":
            raise RuntimeError(f"non-PASS metric: {path}")
        if float(metric["retained_max_abs_correlation"]) >= 0.95:
            raise RuntimeError(f"correlation contract failed: {path}")
        common = {
            "registry_index": int(metric["registry_index"]),
            "task_id": metric["task_id"],
            "source": metric["source"],
            "source_family": metric["source_family"],
            "target": metric["target"],
            "N": int(metric["N"]),
            "outer_train_rows": int(metric["outer_train_rows"]),
            "outer_test_rows": int(metric["outer_test_rows"]),
            "P_raw": int(metric["P_raw"]),
            "P_after_constant_filter": int(
                metric["P_after_constant_filter"]
            ),
            "P_after_corr095": int(metric["P_after_corr095"]),
            "removed_correlated_features": int(
                metric["removed_correlated_features"]
            ),
            "removed_fraction": (
                1.0
                - int(metric["P_after_corr095"])
                / max(int(metric["P_after_constant_filter"]), 1)
            ),
            "retained_max_abs_correlation": float(
                metric["retained_max_abs_correlation"]
            ),
            "remained_high_dimensional": bool(
                metric["remained_high_dimensional"]
            ),
            "retained_original_indices_sha256": metric[
                "retained_original_indices_sha256"
            ],
            "preprocessing_sha256": metric["preprocessing_sha256"],
            "outer_split_seed": int(metric["outer_split_seed"]),
            "outer_split_sha256": metric["outer_split_sha256"],
            "zero_predictor_test_mse": float(
                metric["zero_predictor"]["mse"]
            ),
        }
        dataset_rows.append(common)
        for method in METHODS:
            result = metric["methods"][method]
            long_rows.append(
                {
                    **common,
                    "method": method.upper(),
                    "solver": result["solver"],
                    "cv_folds": int(result["cv_folds"]),
                    "cv_seed": int(result["cv_seed"]),
                    "selected_index": int(result["selected_index"]),
                    "selected_lambda_ratio": float(
                        result["selected_ratio"]
                    ),
                    "selected_raw_lambda": float(
                        result["selected_raw_lambda"]
                    ),
                    "outer_train_lambda_max": float(
                        result["outer_train_lambda_max"]
                    ),
                    "train_objective": float(
                        result["train_objective"]
                    ),
                    "projected_kkt_max": float(
                        result["train_kkt_max"]
                    ),
                    "converged_by_kkt_1e4": bool(
                        result["converged_by_kkt_1e4"]
                    ),
                    "test_mse": float(result["test_mse"]),
                    "normalized_test_mse": float(
                        result["normalized_test_mse"]
                    ),
                    "test_rmse": float(result["test_rmse"]),
                    "test_mae": float(result["test_mae"]),
                    "test_r2": float(result["test_r2"]),
                    "nnz_1e8": int(result["nnz_1e8"]),
                    "nnz_1e6": int(result["nnz_1e6"]),
                    "nnz_1e4": int(result["nnz_1e4"]),
                    "runtime_seconds": float(result["fit_seconds"]),
                }
            )

    datasets = pd.DataFrame(dataset_rows).sort_values("registry_index")
    long = pd.DataFrame(long_rows).sort_values(
        ["registry_index", "method"]
    )
    if datasets["task_id"].nunique() != 41 or long.shape[0] != 123:
        raise RuntimeError("unexpected task or method cardinality")

    value_columns = [
        "selected_lambda_ratio",
        "selected_raw_lambda",
        "outer_train_lambda_max",
        "train_objective",
        "projected_kkt_max",
        "test_mse",
        "normalized_test_mse",
        "test_rmse",
        "test_mae",
        "test_r2",
        "nnz_1e8",
        "nnz_1e6",
        "nnz_1e4",
        "runtime_seconds",
    ]
    wide_values = long.pivot(
        index="task_id", columns="method", values=value_columns
    )
    wide_values.columns = [
        f"{method.lower()}_{field}"
        for field, method in wide_values.columns
    ]
    wide = datasets.merge(
        wide_values.reset_index(), on="task_id", validate="one_to_one"
    )

    aggregate_rows = []
    for method in ("LASSO", "SCAD", "MCP"):
        values = long[long["method"] == method]
        nmse = values["normalized_test_mse"].to_numpy(dtype=np.float64)
        aggregate_rows.append(
            {
                "method": method,
                "task_count": int(values.shape[0]),
                "median_normalized_test_mse": float(np.median(nmse)),
                "geometric_mean_normalized_test_mse": float(
                    math.exp(np.mean(np.log(np.maximum(nmse, 1e-12))))
                ),
                "mean_normalized_test_mse": float(np.mean(nmse)),
                "median_test_r2": float(np.median(values["test_r2"])),
                "median_nnz_1e6": float(np.median(values["nnz_1e6"])),
                "median_projected_kkt": float(
                    np.median(values["projected_kkt_max"])
                ),
                "median_runtime_seconds": float(
                    np.median(values["runtime_seconds"])
                ),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets.to_csv(
        args.output_dir / "datasets_41_after_corr095.csv", index=False
    )
    long.to_csv(
        args.output_dir / "cv_lasso_scad_mcp_41_long.csv", index=False
    )
    wide.to_csv(
        args.output_dir / "cv_lasso_scad_mcp_41_wide.csv", index=False
    )
    aggregate.to_csv(
        args.output_dir / "cv_lasso_scad_mcp_aggregate.csv", index=False
    )
    summary = {
        "status": "PASS",
        "dataset_count": 41,
        "method_count": 3,
        "long_row_count": 123,
        "maximum_retained_abs_correlation": float(
            datasets["retained_max_abs_correlation"].max()
        ),
        "methods": aggregate.to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
