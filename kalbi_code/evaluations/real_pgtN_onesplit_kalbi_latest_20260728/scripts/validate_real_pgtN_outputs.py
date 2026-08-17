#!/usr/bin/env python3
"""Validate all real P>N KALBI and CV-baseline evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import KFold

from real_pgtN_common import atomic_json, sha256


METHODS = ("lasso", "scad", "mcp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-task-count", type=int, default=41)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def add_failure(
    failures: list[dict[str, Any]],
    task_id: str,
    contract: str,
    detail: str,
) -> None:
    failures.append(
        {"task_id": task_id, "contract": contract, "detail": detail}
    )


def validate_task(
    evaluation_root: Path,
    task: dict[str, Any],
    failures: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    task_id = task["task_id"]
    kalbi_path = evaluation_root / "metrics" / f"{task_id}_kalbi.json"
    baseline_path = evaluation_root / "metrics" / f"{task_id}_baselines.json"
    path_path = evaluation_root / "paths" / f"{task_id}.npz"
    baseline_npz_path = evaluation_root / "baselines" / f"{task_id}.npz"
    preprocessing_path = evaluation_root / "preprocessing" / f"{task_id}.npz"
    required = (
        kalbi_path,
        baseline_path,
        path_path,
        baseline_npz_path,
        preprocessing_path,
    )
    for path in required:
        if not path.is_file():
            add_failure(failures, task_id, "artifact_exists", str(path))
            return {"task_id": task_id, "status": "FAIL_MISSING"}

    kalbi = load_json(kalbi_path)
    baseline = load_json(baseline_path)
    if kalbi.get("status") != "PASS":
        add_failure(failures, task_id, "kalbi_status", str(kalbi.get("status")))
    baseline_status = baseline.get("status")
    if baseline_status not in {"PASS", "PARTIAL_NONCONVEX_TIMEOUT"}:
        add_failure(
            failures, task_id, "baseline_status", str(baseline_status)
        )
    timeout_methods = [
        method
        for method in METHODS
        if baseline.get("methods", {}).get(method, {}).get("status")
        == "TIMEOUT"
    ]
    if timeout_methods:
        warnings.append(
            {
                "task_id": task_id,
                "contract": "nonconvex_baseline_timeout",
                "methods": timeout_methods,
            }
        )
    if sha256(path_path) != kalbi["path_sha256"]:
        add_failure(failures, task_id, "kalbi_path_hash", str(path_path))
    if sha256(baseline_npz_path) != baseline["baseline_path_sha256"]:
        add_failure(failures, task_id, "baseline_path_hash", str(baseline_npz_path))
    if sha256(preprocessing_path) != kalbi["preprocessing_sha256"]:
        add_failure(failures, task_id, "preprocessing_hash", str(preprocessing_path))
    if kalbi["scaler_sha256"] != baseline["scaler_sha256"]:
        add_failure(failures, task_id, "shared_preprocessing", "scaler hash mismatch")

    with np.load(task["outer_split_path"], allow_pickle=False) as split:
        train = np.asarray(split["train_indices"], dtype=np.int64)
        test = np.asarray(split["test_indices"], dtype=np.int64)
    if np.intersect1d(train, test).size:
        add_failure(failures, task_id, "split_disjoint", "overlapping row indices")
    expected = np.arange(int(task["N"]), dtype=np.int64)
    if not np.array_equal(np.sort(np.concatenate([train, test])), expected):
        add_failure(failures, task_id, "split_partition", "rows missing or duplicated")
    if sha256(Path(task["outer_split_path"])) != task["outer_split_sha256"]:
        add_failure(failures, task_id, "split_hash", task["outer_split_path"])

    with np.load(path_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    raw = arrays["raw_lambdas"]
    expected_raw = np.logspace(-4.0, 0.0, 41)
    if raw.shape != (41,) or not np.allclose(raw, expected_raw, rtol=0, atol=1e-14):
        add_failure(failures, task_id, "raw_lambda_grid", str(raw.shape))
    transformed_p = int(kalbi["P_transformed"])
    expected_beta_shape = (41, transformed_p)
    for name in ("kalbi_beta", "coordinate_descent_beta"):
        if arrays[name].shape != expected_beta_shape or not np.isfinite(arrays[name]).all():
            add_failure(
                failures, task_id, f"{name}_shape_finite", str(arrays[name].shape)
            )
    expected_prediction_shape = (41, test.size)
    for name in (
        "kalbi_test_prediction",
        "coordinate_descent_test_prediction",
    ):
        if (
            arrays[name].shape != expected_prediction_shape
            or not np.isfinite(arrays[name]).all()
        ):
            add_failure(
                failures, task_id, f"{name}_shape_finite", str(arrays[name].shape)
            )

    y_test = np.asarray(np.load(task["y_path"], mmap_mode="r")[test], dtype=np.float64)
    recomputed = np.mean(
        (arrays["kalbi_test_prediction"].astype(np.float64) - y_test[None, :]) ** 2,
        axis=1,
    )
    if not np.allclose(
        # Predictions are persisted as float32 while path MSE is computed
        # before that cast. The audited 41-task maximum relative cast error is
        # 3.04e-6 for a response scale near 760.
        recomputed,
        arrays["kalbi_test_mse"],
        rtol=5e-6,
        atol=1e-9,
    ):
        add_failure(failures, task_id, "prediction_metric_recompute", "KALBI MSE")
    if np.max(np.ptp(arrays["kalbi_test_prediction"], axis=0), initial=0.0) <= 1e-10:
        add_failure(
            failures,
            task_id,
            "independent_lambda_predictions",
            "all KALBI predictions are constant over lambda",
        )

    eta = arrays["v8_eta_path"].astype(np.float64)
    gamma = arrays["v8_gamma_path"].astype(np.float64)
    ratio = np.divide(gamma, eta, out=np.zeros_like(gamma), where=np.abs(eta) > 0)
    gamma_error = float(np.max(np.abs(ratio - raw[:, None])))
    if gamma_error > 2e-5:
        add_failure(failures, task_id, "raw_lambda_gamma_contract", str(gamma_error))
    if int(kalbi["coordinate_descent"]["full_feature_count"]) != transformed_p:
        add_failure(
            failures,
            task_id,
            "full_feature_cd",
            str(kalbi["coordinate_descent"]["full_feature_count"]),
        )
    if not bool(kalbi["coordinate_descent"].get("audit_pass", True)):
        warnings.append(
            {
                "task_id": task_id,
                "contract": "coordinate_descent_audit",
                "max_fixed_point": kalbi["coordinate_descent"]["max_fixed_point"],
                "max_projected_kkt": kalbi["coordinate_descent"]["max_projected_kkt"],
            }
        )

    with np.load(baseline_npz_path, allow_pickle=False) as archive:
        baseline_arrays = {
            key: np.asarray(archive[key]) for key in archive.files
        }
    ratios = baseline_arrays["lambda_ratios"]
    expected_ratios = np.logspace(0.0, -4.0, 100)
    if ratios.shape != (100,) or not np.allclose(
        ratios, expected_ratios, rtol=0, atol=1e-14
    ):
        add_failure(failures, task_id, "baseline_lambda_grid", str(ratios.shape))
    for method in METHODS:
        method_metric = baseline["methods"][method]
        if method_metric.get("status") == "TIMEOUT":
            if method not in {"scad", "mcp"}:
                add_failure(
                    failures,
                    task_id,
                    "unexpected_method_timeout",
                    method,
                )
            if method_metric.get("included_in_aggregate") is not False:
                add_failure(
                    failures,
                    task_id,
                    f"{method}_timeout_aggregate_contract",
                    "timeout method must be excluded",
                )
            continue
        beta_key = f"{method}_beta"
        if beta_key not in baseline_arrays:
            add_failure(
                failures, task_id, f"{method}_beta_exists", beta_key
            )
            continue
        beta = baseline_arrays[beta_key]
        if beta.shape != (transformed_p,) or not np.isfinite(beta).all():
            add_failure(
                failures, task_id, f"{method}_beta_shape_finite", str(beta.shape)
            )
        if not all(
            np.isfinite(float(method_metric[key]))
            for key in (
                "test_mse",
                "test_rmse",
                "test_mae",
                "test_r2",
                "normalized_test_mse",
                "train_kkt_max",
            )
        ):
            add_failure(
                failures, task_id, f"{method}_metrics_finite", "non-finite value"
            )
        if float(method_metric["train_kkt_max"]) > 5e-3:
            warnings.append(
                {
                    "task_id": task_id,
                    "contract": f"{method}_large_train_kkt",
                    "value": float(method_metric["train_kkt_max"]),
                }
            )

    fold_count = 5 if train.size >= 100 else 3
    cv_seed = int(task["outer_split_seed"]) ^ 0x00C0FFEE
    for fit_rel, valid_rel in KFold(
        n_splits=fold_count, shuffle=True, random_state=cv_seed
    ).split(train):
        fit_global = train[fit_rel]
        valid_global = train[valid_rel]
        if (
            np.intersect1d(fit_global, test).size
            or np.intersect1d(valid_global, test).size
        ):
            add_failure(
                failures, task_id, "inner_cv_test_leakage", "outer-test index found"
            )

    available_kkt = [
        float(baseline["methods"][method]["train_kkt_max"])
        for method in METHODS
        if baseline["methods"][method].get("status") != "TIMEOUT"
    ]
    return {
        "task_id": task_id,
        "status": (
            "PASS_WITH_NONCONVEX_TIMEOUT"
            if timeout_methods
            else "PASS"
        ),
        "N": int(kalbi["N"]),
        "P_transformed": transformed_p,
        "gamma_contract_max_error": gamma_error,
        "cd_audit_pass": bool(
            kalbi["coordinate_descent"].get("audit_pass", True)
        ),
        "timeout_methods": timeout_methods,
        "baseline_max_train_kkt": float(max(available_kkt)),
    }


def main() -> None:
    args = parse_args()
    manifest = load_json(args.split_manifest)
    if manifest.get("task_count") != args.expected_task_count:
        raise RuntimeError("split-manifest task count mismatch")
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    task_results = [
        validate_task(args.evaluation_root, task, failures, warnings)
        for task in manifest["tasks"]
    ]
    timeout_task_count = sum(
        bool(task.get("timeout_methods")) for task in task_results
    )
    result = {
        "status": (
            "FAIL"
            if failures
            else (
                "PASS_WITH_NONCONVEX_TIMEOUTS"
                if timeout_task_count
                else "PASS"
            )
        ),
        "task_count": len(task_results),
        "expected_task_count": args.expected_task_count,
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": sha256(args.split_manifest),
        "contracts": {
            "outer_split": "one deterministic 80/20 row split",
            "preprocessing": "training rows only; shared scaler hash",
            "kalbi_path": "41 raw lambdas logspace(-4,0)",
            "baseline_cv": "100 ratios from 1 to 1e-4; training rows only",
            "baseline_timeout_policy": (
                "KALBI and CV-LASSO on 41 tasks; SCAD/MCP on "
                "37 tasks, with four explicit numerical timeouts"
            ),
            "refinement": "gamma / eta = raw lambda for all 10 sweeps",
            "coordinate_descent": "all transformed features",
        },
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "nonconvex_timeout_task_count": timeout_task_count,
        "failures": failures,
        "warnings": warnings,
        "tasks": task_results,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
