#!/usr/bin/env python3
"""Fit full-feature LASSO paths on the exact corr095 KALBI raw-lambda grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

from real_pgtN_common import (
    apply_x_preprocessing,
    atomic_json,
    atomic_npz,
    fit_preprocessing,
    iter_selected_tasks,
    load_split_manifest,
    load_x,
    parse_task_indices,
    raw_prediction_metrics,
    select_rows,
    sha256,
    zero_predictor_metrics,
)


def retained_hash(indices: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(indices, dtype=np.int64).tobytes()
    ).hexdigest()


def fit_lasso_path(
    x: np.ndarray,
    y: np.ndarray,
    raw_lambdas_ascending: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    raw = np.asarray(raw_lambdas_ascending, dtype=np.float64)
    if np.any(raw <= 0) or np.any(np.diff(raw) <= 0):
        raise RuntimeError("raw lambda grid must be positive and ascending")
    model = Lasso(
        alpha=float(raw[-1]),
        fit_intercept=False,
        max_iter=int(max_iterations),
        tol=float(tolerance),
        warm_start=True,
        selection="cyclic",
    )
    coefficients: list[np.ndarray] = []
    iterations: list[int] = []
    dual_gaps: list[float] = []
    warning_counts: list[int] = []
    for raw_lambda in raw[::-1]:
        model.set_params(alpha=float(raw_lambda))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x, y)
        coefficients.append(np.asarray(model.coef_, dtype=np.float64).copy())
        iterations.append(int(model.n_iter_))
        dual_gaps.append(float(model.dual_gap_))
        warning_counts.append(
            sum(
                issubclass(item.category, ConvergenceWarning)
                for item in caught
            )
        )
    beta = np.asarray(coefficients, dtype=np.float64)[::-1]
    diagnostics = {
        "iterations": np.asarray(iterations, dtype=np.int64)[::-1],
        "dual_gaps": np.asarray(dual_gaps, dtype=np.float64)[::-1],
        "warning_counts": np.asarray(warning_counts, dtype=np.int64)[::-1],
    }
    if not np.isfinite(beta).all() or not np.isfinite(
        diagnostics["dual_gaps"]
    ).all():
        raise RuntimeError("non-finite LASSO path")
    return beta, diagnostics


def objective_and_kkt(
    x: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    raw_lambdas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    objectives: list[float] = []
    data_halves: list[float] = []
    kkt_values: list[float] = []
    for coefficient, raw_lambda in zip(
        np.asarray(beta, dtype=np.float64),
        np.asarray(raw_lambdas, dtype=np.float64),
        strict=True,
    ):
        residual = x @ coefficient - y
        gradient = x.T @ residual / x.shape[0]
        active = np.abs(coefficient) > 1e-10
        kkt = np.where(
            active,
            np.abs(gradient + raw_lambda * np.sign(coefficient)),
            np.maximum(np.abs(gradient) - raw_lambda, 0.0),
        )
        data = 0.5 * float(np.mean(residual**2))
        data_halves.append(data)
        objectives.append(data + float(raw_lambda * np.abs(coefficient).sum()))
        kkt_values.append(float(kkt.max(initial=0.0)))
    return (
        np.asarray(objectives),
        np.asarray(data_halves),
        np.asarray(kkt_values),
    )


def evaluate_task(
    task: dict[str, Any],
    evaluation_root: Path,
    reference_bundle: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    task_id = task["task_id"]
    for key, hash_key in (
        ("X_path", "X_sha256"),
        ("y_path", "y_sha256"),
        ("outer_split_path", "outer_split_sha256"),
    ):
        if sha256(Path(task[key])) != task[hash_key]:
            raise RuntimeError(f"artifact hash mismatch: {task[key]}")

    with np.load(task["outer_split_path"], allow_pickle=False) as split:
        train_indices = np.asarray(split["train_indices"], dtype=np.int64)
        test_indices = np.asarray(split["test_indices"], dtype=np.int64)
    x_store = load_x(Path(task["X_path"]))
    y_store = np.load(task["y_path"], mmap_mode="r")
    x_train_raw = select_rows(x_store, train_indices)
    y_train_raw = np.asarray(y_store[train_indices], dtype=np.float64)
    x_train, y_train, scaler = fit_preprocessing(
        x_train_raw, y_train_raw
    )
    retained = np.asarray(
        scaler["retained_original_indices"], dtype=np.int64
    )
    mask_hash = retained_hash(retained)

    bundle_root = evaluation_root / "bundles" / reference_bundle
    kalbi_metric = json.loads(
        (bundle_root / "metrics" / f"{task_id}.json").read_text(
            encoding="utf-8"
        )
    )
    baseline_metric = json.loads(
        (evaluation_root / "metrics" / f"{task_id}_baselines.json").read_text(
            encoding="utf-8"
        )
    )
    preprocessing_path = Path(kalbi_metric["preprocessing_path"])
    if (
        kalbi_metric["retained_original_indices_sha256"] != mask_hash
        or baseline_metric["retained_original_indices_sha256"] != mask_hash
        or baseline_metric["preprocessing_path"] != str(preprocessing_path)
        or kalbi_metric["preprocessing_sha256"] != sha256(preprocessing_path)
        or baseline_metric["preprocessing_sha256"]
        != kalbi_metric["preprocessing_sha256"]
    ):
        raise RuntimeError("final preprocessing mask mismatch")
    with np.load(preprocessing_path, allow_pickle=False) as preprocessing:
        for key in (
            "x_mean",
            "x_scale",
            "active",
            "correlation_keep",
            "retained_original_indices",
            "y_mean",
            "y_scale",
        ):
            np.testing.assert_array_equal(preprocessing[key], scaler[key])

    with np.load(kalbi_metric["path_output"], allow_pickle=False) as archive:
        raw_lambdas = np.asarray(archive["raw_lambdas"], dtype=np.float64)
        np.testing.assert_array_equal(
            archive["retained_original_indices"], retained
        )
        np.testing.assert_array_equal(archive["train_indices"], train_indices)
        np.testing.assert_array_equal(archive["test_indices"], test_indices)

    beta, diagnostic = fit_lasso_path(
        x_train,
        y_train,
        raw_lambdas,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
    )
    objective, data_half, kkt = objective_and_kkt(
        x_train, y_train, beta, raw_lambdas
    )
    x_test = apply_x_preprocessing(
        select_rows(x_store, test_indices), scaler
    )
    y_test_raw = np.asarray(y_store[test_indices], dtype=np.float64)
    test = raw_prediction_metrics(
        x_test,
        y_test_raw,
        beta,
        y_mean=float(scaler["y_mean"]),
        y_scale=float(scaler["y_scale"]),
    )
    zero = zero_predictor_metrics(y_test_raw, float(scaler["y_mean"]))
    lambda_max = float(
        np.max(np.abs(x_train.T @ y_train)) / x_train.shape[0]
    )

    output_path = (
        evaluation_root / "solution_paths" / "lasso" / f"{task_id}.npz"
    )
    atomic_npz(
        output_path,
        raw_lambdas=raw_lambdas,
        beta=beta.astype(np.float32),
        test_mse=np.asarray(test["mse"], dtype=np.float64),
        normalized_test_mse=(
            np.asarray(test["mse"], dtype=np.float64)
            / max(float(zero["mse"]), 1e-30)
        ),
        objective=objective,
        data_half=data_half,
        projected_kkt_max=kkt,
        nnz_1e8=np.count_nonzero(np.abs(beta) > 1e-8, axis=1),
        iterations=diagnostic["iterations"],
        dual_gaps=diagnostic["dual_gaps"],
        convergence_warning_counts=diagnostic["warning_counts"],
        retained_original_indices=retained,
        train_indices=train_indices,
        test_indices=test_indices,
    )
    metric = {
        "status": "PASS",
        "registry_index": int(task["registry_index"]),
        "task_id": task_id,
        "reference_bundle": reference_bundle,
        "P_after_corr095": int(x_train.shape[1]),
        "retained_original_indices_sha256": mask_hash,
        "scaler_sha256": scaler["scaler_sha256"],
        "lambda_max": lambda_max,
        "raw_lambda_count": int(raw_lambdas.size),
        "raw_lambda_min": float(raw_lambdas[0]),
        "raw_lambda_max": float(raw_lambdas[-1]),
        "endpoint_zero_expected": bool(raw_lambdas[-1] >= lambda_max),
        "endpoint_nnz_1e8": int(np.count_nonzero(np.abs(beta[-1]) > 1e-8)),
        "max_projected_kkt": float(kkt.max()),
        "max_dual_gap": float(diagnostic["dual_gaps"].max()),
        "convergence_warning_count": int(diagnostic["warning_counts"].sum()),
        "path_output": str(output_path),
        "path_sha256": sha256(output_path),
        "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(
        evaluation_root
        / "solution_paths"
        / "lasso_metrics"
        / f"{task_id}.json",
        metric,
    )
    return metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--reference-bundle", required=True)
    parser.add_argument("--task-indices", default="all")
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=50000)
    args = parser.parse_args()

    manifest = load_split_manifest(args.split_manifest)
    indices = parse_task_indices(
        args.task_indices, int(manifest["task_count"])
    )
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for task in iter_selected_tasks(manifest, indices):
        task_id = task["task_id"]
        metric_path = (
            args.evaluation_root
            / "solution_paths"
            / "lasso_metrics"
            / f"{task_id}.json"
        )
        if metric_path.exists():
            existing = json.loads(metric_path.read_text(encoding="utf-8"))
            path = Path(existing["path_output"])
            if (
                existing.get("status") == "PASS"
                and path.is_file()
                and sha256(path) == existing["path_sha256"]
            ):
                successes.append(existing)
                print(
                    json.dumps({"task_id": task_id, "status": "SKIP_VERIFIED"}),
                    flush=True,
                )
                continue
            raise RuntimeError(f"invalid existing LASSO path: {task_id}")
        try:
            result = evaluate_task(
                task, args.evaluation_root, args.reference_bundle, args
            )
            successes.append(result)
            print(json.dumps(result), flush=True)
        except Exception as error:
            failure = {
                "status": "FAIL",
                "registry_index": int(task["registry_index"]),
                "task_id": task_id,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            print(json.dumps(failure), flush=True)
    report = {
        "status": "PASS" if not failures else "PARTIAL",
        "success_count": len(successes),
        "failure_count": len(failures),
        "failures": failures,
    }
    atomic_json(
        args.evaluation_root / "reports" / "lasso_solution_paths.json",
        report,
    )
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
