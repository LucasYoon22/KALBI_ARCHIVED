#!/usr/bin/env python3
"""Create a compact, reproducible summary of the corr095 evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from real_pgtN_common import atomic_json


METHOD_ORDER = [
    "KALBI-A",
    "KALBI-B",
    "KALBI-C",
    "KALBI-D",
    "CV LASSO",
    "CV SCAD",
    "CV MCP",
]


def finite_summary(values: pd.Series) -> dict[str, float]:
    array = values.to_numpy(dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {}
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(np.max(array)),
    }


def method_summary(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in METHOD_ORDER:
        subset = frame[frame["method"] == method]
        if subset.empty:
            continue
        nmse = subset["normalized_test_mse"].to_numpy(dtype=np.float64)
        result[method] = {
            "task_count": int(subset.shape[0]),
            "median_nmse": float(np.median(nmse)),
            "geometric_mean_nmse": float(
                math.exp(np.mean(np.log(np.maximum(nmse, 1e-12))))
            ),
            "mean_nmse": float(np.mean(nmse)),
            "median_test_r2": float(np.median(subset["test_r2"])),
            "median_nnz_1e6": float(np.median(subset["nnz_1e6"])),
            "median_projected_kkt": float(
                np.median(subset["projected_kkt_max"])
            ),
            "p90_projected_kkt": float(
                np.quantile(subset["projected_kkt_max"], 0.9)
            ),
            "median_runtime_seconds": float(
                np.median(subset["runtime_seconds"])
            ),
        }
    return result


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.lower().eq("true")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    args = parser.parse_args()
    metric_root = args.evaluation_root / "metrics"
    long = pd.read_csv(metric_root / "per_task_metrics.csv")
    tasks = pd.read_csv(metric_root / "preprocessing_per_task.csv")
    pairwise = pd.read_csv(metric_root / "pairwise_statistics.csv")

    comparable = long[as_bool(long["kalbi_applicable"])].copy()
    baseline_all = long[long["method_type"] == "baseline"].copy()
    task_applicable = as_bool(tasks["kalbi_applicable"])
    task_high_dimensional = as_bool(tasks["remained_high_dimensional"])
    lasso_metrics = []
    for path in sorted(
        (args.evaluation_root / "solution_paths" / "lasso_metrics").glob(
            "*.json"
        )
    ):
        lasso_metrics.append(json.loads(path.read_text(encoding="utf-8")))

    pairwise_rows = pairwise.to_dict(orient="records")
    summary = {
        "status": "PASS",
        "task_counts": {
            "registered": int(tasks.shape[0]),
            "kalbi_applicable_after_corr095": int(
                task_applicable.sum()
            ),
            "kalbi_inapplicable_P_lt_32": int(
                (~task_applicable).sum()
            ),
            "remained_high_dimensional_after_corr095": int(
                task_high_dimensional.sum()
            ),
        },
        "feature_filter": {
            "P_after_corr095": finite_summary(tasks["P_after_corr095"]),
            "removed_fraction": finite_summary(tasks["removed_fraction"]),
            "retained_max_abs_correlation": finite_summary(
                tasks["retained_max_abs_correlation"]
            ),
        },
        "paired_25_task_method_summary": method_summary(comparable),
        "all_41_task_baseline_summary": method_summary(baseline_all),
        "pairwise_statistics": pairwise_rows,
        "lasso_solution_path": {
            "task_count": len(lasso_metrics),
            "max_projected_kkt": (
                float(max(item["max_projected_kkt"] for item in lasso_metrics))
                if lasso_metrics
                else None
            ),
            "total_convergence_warnings": int(
                sum(
                    item["convergence_warning_count"]
                    for item in lasso_metrics
                )
            ),
            "all_right_endpoints_zero": bool(
                lasso_metrics
                and all(item["endpoint_nnz_1e8"] == 0 for item in lasso_metrics)
            ),
        },
    }
    atomic_json(args.evaluation_root / "reports" / "headline_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
