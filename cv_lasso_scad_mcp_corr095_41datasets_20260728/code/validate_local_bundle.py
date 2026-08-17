#!/usr/bin/env python3
"""Validate the focused 41-task CV baseline bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("lasso", "scad", "mcp")


def stems(paths: list[Path], suffix: str) -> set[str]:
    return {path.name.removesuffix(suffix) for path in paths}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    root = args.bundle_root
    result_root = root / "results"

    metric_paths = sorted((result_root / "raw_metrics_json").glob("*.json"))
    coefficient_paths = sorted(
        (result_root / "coefficient_npz").glob("*.npz")
    )
    preprocessing_paths = sorted(
        (result_root / "preprocessing_npz").glob("*.npz")
    )
    if not (
        len(metric_paths) == len(coefficient_paths)
        == len(preprocessing_paths)
        == 41
    ):
        raise RuntimeError("raw artifact cardinality mismatch")
    metric_ids = stems(metric_paths, "_baselines.json")
    coefficient_ids = stems(coefficient_paths, ".npz")
    preprocessing_ids = stems(preprocessing_paths, ".npz")
    if metric_ids != coefficient_ids or metric_ids != preprocessing_ids:
        raise RuntimeError("task identifiers differ across artifact types")

    for metric_path in metric_paths:
        metric = json.loads(metric_path.read_text(encoding="utf-8"))
        if metric.get("status") != "PASS":
            raise RuntimeError(f"non-PASS metric: {metric_path}")
        if set(metric["methods"]) != set(METHODS):
            raise RuntimeError(f"method mismatch: {metric_path}")
        if float(metric["retained_max_abs_correlation"]) >= 0.95:
            raise RuntimeError(f"correlation contract failed: {metric_path}")
        for method in METHODS:
            values = metric["methods"][method]
            required = (
                "test_mse",
                "normalized_test_mse",
                "test_r2",
                "selected_raw_lambda",
                "train_kkt_max",
            )
            if not np.isfinite([float(values[key]) for key in required]).all():
                raise RuntimeError(
                    f"nonfinite metric: {metric_path} {method}"
                )

    datasets = pd.read_csv(result_root / "datasets_41_after_corr095.csv")
    long = pd.read_csv(result_root / "cv_lasso_scad_mcp_41_long.csv")
    wide = pd.read_csv(result_root / "cv_lasso_scad_mcp_41_wide.csv")
    aggregate = pd.read_csv(
        result_root / "cv_lasso_scad_mcp_aggregate.csv"
    )
    if datasets.shape[0] != 41 or wide.shape[0] != 41:
        raise RuntimeError("dataset/wide table cardinality mismatch")
    if long.shape[0] != 123 or aggregate.shape[0] != 3:
        raise RuntimeError("long/aggregate table cardinality mismatch")
    if set(long["method"]) != {"LASSO", "SCAD", "MCP"}:
        raise RuntimeError("long-table method mismatch")
    if long.select_dtypes(include=[np.number]).isna().any().any():
        raise RuntimeError("numeric result table contains missing values")

    result = {
        "status": "PASS",
        "dataset_count": 41,
        "method_count": 3,
        "long_row_count": 123,
        "raw_metric_json_count": len(metric_paths),
        "coefficient_npz_count": len(coefficient_paths),
        "preprocessing_npz_count": len(preprocessing_paths),
        "maximum_retained_abs_correlation": float(
            datasets["retained_max_abs_correlation"].max()
        ),
    }
    (root / "VALIDATION.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
