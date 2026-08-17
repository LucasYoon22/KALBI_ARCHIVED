#!/usr/bin/env python3
"""Validate the CD-free KALBI versus LASSO path comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from real_pgtN_common import atomic_json, load_split_manifest


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--max-lasso-kkt", type=float, default=1e-3)
    args = parser.parse_args()

    manifest = load_split_manifest(args.split_manifest)
    preprocessing = load_json(
        args.comparison_root
        / "manifests"
        / "preprocessing_reaudit.json"
    )
    failures: list[str] = []
    warnings: list[str] = []
    task_results: list[dict[str, Any]] = []
    expected_grid = np.logspace(-4.0, 0.0, 41, dtype=np.float64)

    if preprocessing.get("status") != "PASS":
        failures.append("independent preprocessing re-audit did not pass")
    if preprocessing.get("task_count") != 41:
        failures.append("preprocessing re-audit task count is not 41")

    for task in manifest["tasks"]:
        task_id = task["task_id"]
        try:
            metric = load_json(
                args.comparison_root / "metrics" / f"{task_id}.json"
            )
            if metric.get("status") != "PASS":
                raise RuntimeError("metric status is not PASS")
            if metric.get("coordinate_descent_imported_or_executed") is not False:
                raise RuntimeError("coordinate-descent execution flag is not false")
            if metric.get("same_training_only_preprocessing") is not True:
                raise RuntimeError("shared-preprocessing flag is not true")

            path_path = args.comparison_root / "paths" / f"{task_id}.npz"
            original_path = args.evaluation_root / "paths" / f"{task_id}.npz"
            with np.load(path_path, allow_pickle=False) as arrays:
                keys = set(arrays.files)
                forbidden = {
                    key
                    for key in keys
                    if "coordinate_descent" in key.lower()
                    or key.lower().startswith("cd_")
                }
                if forbidden:
                    raise RuntimeError(
                        f"coordinate-descent arrays present: {sorted(forbidden)}"
                    )
                raw = np.asarray(arrays["raw_lambdas"], dtype=np.float64)
                np.testing.assert_array_equal(raw, expected_grid)
                kalbi_beta = np.asarray(arrays["kalbi_beta"], dtype=np.float64)
                lasso_beta = np.asarray(arrays["lasso_beta"], dtype=np.float64)
                lasso_kkt = np.asarray(arrays["lasso_kkt"], dtype=np.float64)
                lasso_prediction = np.asarray(
                    arrays["lasso_test_prediction"], dtype=np.float64
                )
                lasso_test_mse = np.asarray(
                    arrays["lasso_test_mse"], dtype=np.float64
                )
                train_indices = np.asarray(
                    arrays["train_indices"], dtype=np.int64
                )
                test_indices = np.asarray(
                    arrays["test_indices"], dtype=np.int64
                )
                lasso_nnz = np.asarray(arrays["lasso_nnz"], dtype=np.int64)
                if (
                    kalbi_beta.shape != lasso_beta.shape
                    or kalbi_beta.shape[0] != raw.size
                    or kalbi_beta.shape[1] != int(metric["P_transformed"])
                ):
                    raise RuntimeError("coefficient path shape mismatch")
                if not all(
                    np.isfinite(value).all()
                    for value in (
                        kalbi_beta,
                        lasso_beta,
                        lasso_kkt,
                        lasso_prediction,
                        lasso_test_mse,
                    )
                ):
                    raise RuntimeError("non-finite output")

            with np.load(original_path, allow_pickle=False) as original:
                np.testing.assert_array_equal(
                    raw,
                    np.asarray(original["raw_lambdas"], dtype=np.float64),
                )
                np.testing.assert_array_equal(
                    train_indices,
                    np.asarray(original["train_indices"], dtype=np.int64),
                )
                np.testing.assert_array_equal(
                    test_indices,
                    np.asarray(original["test_indices"], dtype=np.int64),
                )
                np.testing.assert_allclose(
                    kalbi_beta,
                    np.asarray(original["kalbi_beta"], dtype=np.float64),
                    rtol=0.0,
                    atol=0.0,
                )

            y_store = np.load(Path(task["y_path"]), mmap_mode="r")
            y_test = np.asarray(y_store[test_indices], dtype=np.float64)
            replay_mse = np.mean(
                (lasso_prediction - y_test[None, :]) ** 2, axis=1
            )
            np.testing.assert_allclose(
                replay_mse,
                lasso_test_mse,
                rtol=5e-6,
                atol=1e-9,
            )

            max_kkt = float(np.max(lasso_kkt))
            if max_kkt > float(args.max_lasso_kkt):
                raise RuntimeError(
                    f"LASSO KKT {max_kkt:.6g} exceeds "
                    f"{args.max_lasso_kkt:.6g}"
                )
            if metric["lambda_contract"]["lasso_endpoint_zero_expected"]:
                if int(lasso_nnz[-1]) != 0:
                    raise RuntimeError("LASSO endpoint should be zero")
            if metric["lasso_convergence_warning_count"] > 0:
                warnings.append(
                    f"{task_id}: "
                    f"{metric['lasso_convergence_warning_count']} sklearn "
                    f"warnings, accepted by explicit KKT audit"
                )

            task_results.append(
                {
                    "registry_index": int(task["registry_index"]),
                    "task_id": task_id,
                    "status": "PASS",
                    "max_lasso_kkt": max_kkt,
                    "max_lasso_dual_gap": float(
                        metric["lasso_max_dual_gap"]
                    ),
                    "lasso_convergence_warning_count": int(
                        metric["lasso_convergence_warning_count"]
                    ),
                    "test_mse_replay_max_abs_error": float(
                        np.max(np.abs(replay_mse - lasso_test_mse))
                    ),
                    "same_raw_lambda_grid": True,
                    "same_training_only_preprocessing": True,
                    "coordinate_descent_imported_or_executed": False,
                }
            )
        except Exception as error:
            failures.append(
                f"{task_id}: {type(error).__name__}: {error}"
            )

    result = {
        "status": "PASS" if not failures else "FAIL",
        "task_count": len(task_results),
        "expected_task_count": 41,
        "failure_count": len(failures),
        "failures": failures,
        "warning_count": len(warnings),
        "warnings": warnings,
        "contracts": {
            "preprocessing": "independently re-audited training-only scaler",
            "raw_lambda": "same numeric 41-point grid sent to both methods",
            "raw_lambda_grid": expected_grid.tolist(),
            "lasso_features": "all transformed features",
            "coordinate_descent_imported_or_executed": False,
            "lasso_kkt_tolerance": float(args.max_lasso_kkt),
        },
        "aggregate": {
            "maximum_lasso_kkt": float(
                max(
                    (row["max_lasso_kkt"] for row in task_results),
                    default=float("nan"),
                )
            ),
            "maximum_lasso_dual_gap": float(
                max(
                    (row["max_lasso_dual_gap"] for row in task_results),
                    default=float("nan"),
                )
            ),
            "total_lasso_convergence_warnings": int(
                sum(
                    row["lasso_convergence_warning_count"]
                    for row in task_results
                )
            ),
            "maximum_test_mse_replay_abs_error": float(
                max(
                    (
                        row["test_mse_replay_max_abs_error"]
                        for row in task_results
                    ),
                    default=float("nan"),
                )
            ),
        },
        "tasks": task_results,
    }
    atomic_json(
        args.comparison_root
        / "manifests"
        / "final_validation.json",
        result,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
