#!/usr/bin/env python3
"""Validate all final corr095 evaluation artifacts and contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from real_pgtN_common import atomic_json, sha256


BUNDLES = (
    "KALBI-A-v6-v23-s20300",
    "KALBI-B-v8-v23-s20300",
    "KALBI-C-v8-v24-s03900",
    "KALBI-D-v8-v24-s13100",
)


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.evaluation_root

    baseline_paths = sorted((root / "metrics").glob("*_baselines.json"))
    if len(baseline_paths) != 41:
        raise RuntimeError(f"expected 41 baselines, found {len(baseline_paths)}")
    baseline_by_task: dict[str, dict[str, Any]] = {}
    for path in baseline_paths:
        metric = read(path)
        if metric["status"] != "PASS":
            raise RuntimeError(f"baseline not PASS: {path}")
        artifact = Path(metric["baseline_path"])
        if sha256(artifact) != metric["baseline_path_sha256"]:
            raise RuntimeError(f"baseline SHA mismatch: {artifact}")
        if not metric["test_y_accessed_after_method_freeze"]:
            raise RuntimeError(f"baseline leakage flag failed: {path}")
        if not float(metric["retained_max_abs_correlation"]) < 0.95:
            raise RuntimeError(f"correlation contract failed: {path}")
        baseline_by_task[metric["task_id"]] = metric

    applicable: set[str] | None = None
    maximum_gamma_error = 0.0
    maximum_kalbi_kkt = 0.0
    for bundle in BUNDLES:
        passed: set[str] = set()
        metric_root = root / "bundles" / bundle / "metrics"
        for path in metric_root.glob("*.json"):
            metric = read(path)
            if metric.get("status") != "PASS":
                continue
            artifact = Path(metric["path_output"])
            if sha256(artifact) != metric["path_sha256"]:
                raise RuntimeError(f"KALBI path SHA mismatch: {artifact}")
            baseline = baseline_by_task[metric["task_id"]]
            if (
                metric["retained_original_indices_sha256"]
                != baseline["retained_original_indices_sha256"]
                or metric["preprocessing_sha256"]
                != baseline["preprocessing_sha256"]
            ):
                raise RuntimeError(
                    f"KALBI/baseline preprocessing mismatch: {path}"
                )
            if not metric["test_y_accessed_after_selector_and_path_freeze"]:
                raise RuntimeError(f"KALBI leakage flag failed: {path}")
            maximum_gamma_error = max(
                maximum_gamma_error,
                float(metric["gamma_over_eta_raw_lambda_max_abs_error"]),
            )
            maximum_kalbi_kkt = max(
                maximum_kalbi_kkt,
                float(metric["selected_projected_kkt_max"]),
            )
            passed.add(metric["task_id"])
        if len(passed) != 25:
            raise RuntimeError(f"{bundle}: expected 25 PASS tasks, got {len(passed)}")
        if applicable is None:
            applicable = passed
        elif passed != applicable:
            raise RuntimeError(f"KALBI task-set mismatch: {bundle}")
    if maximum_gamma_error > 1e-6:
        raise RuntimeError(f"raw-lambda gamma contract failed: {maximum_gamma_error}")

    lasso_metrics = sorted(
        (root / "solution_paths" / "lasso_metrics").glob("*.json")
    )
    if len(lasso_metrics) != 25:
        raise RuntimeError(
            f"expected 25 LASSO solution paths, found {len(lasso_metrics)}"
        )
    maximum_lasso_kkt = 0.0
    warning_count = 0
    lasso_tasks: set[str] = set()
    for path in lasso_metrics:
        metric = read(path)
        artifact = Path(metric["path_output"])
        if sha256(artifact) != metric["path_sha256"]:
            raise RuntimeError(f"LASSO path SHA mismatch: {artifact}")
        if int(metric["endpoint_nnz_1e8"]) != 0:
            raise RuntimeError(f"LASSO endpoint is nonzero: {path}")
        maximum_lasso_kkt = max(
            maximum_lasso_kkt, float(metric["max_projected_kkt"])
        )
        warning_count += int(metric["convergence_warning_count"])
        lasso_tasks.add(metric["task_id"])
    if lasso_tasks != applicable:
        raise RuntimeError("LASSO/KALBI path task-set mismatch")

    required = (
        root / "metrics" / "per_task_metrics.csv",
        root / "metrics" / "aggregate_metrics.csv",
        root / "metrics" / "pairwise_statistics.csv",
        root / "figures" / "solution_paths_representative6.png",
        root / "figures" / "solution_paths_representative6.pdf",
        root / "figures" / "nmse_heatmap.png",
        root / "figures" / "paired_nmse_scatter.png",
        root / "figures" / "method_rank.png",
        root / "figures" / "feature_filtering.png",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing final artifacts: {missing}")

    result = {
        "status": "PASS",
        "baseline_task_count": 41,
        "kalbi_bundle_count": 4,
        "kalbi_tasks_per_bundle": 25,
        "lasso_solution_path_count": 25,
        "maximum_gamma_over_eta_raw_lambda_error": maximum_gamma_error,
        "maximum_kalbi_selected_projected_kkt": maximum_kalbi_kkt,
        "maximum_lasso_path_projected_kkt": maximum_lasso_kkt,
        "lasso_convergence_warning_count": warning_count,
        "required_artifact_sha256": {
            str(path): sha256(path) for path in required
        },
    }
    atomic_json(root / "audits" / "final_artifact_audit.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
