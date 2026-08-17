#!/usr/bin/env python3
"""Compare frozen KALBI paths with full-feature LASSO on the same raw lambdas."""

from __future__ import annotations

import argparse
import json
import math
import time
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
    task_family,
    zero_predictor_metrics,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fit_lasso_raw_path(
    x: np.ndarray,
    y: np.ndarray,
    raw_lambdas_ascending: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw_grid = np.asarray(raw_lambdas_ascending, dtype=np.float64)
    if (
        raw_grid.ndim != 1
        or raw_grid.size < 2
        or not np.isfinite(raw_grid).all()
        or np.any(raw_grid <= 0)
        or np.any(np.diff(raw_grid) <= 0)
    ):
        raise RuntimeError("raw lambda grid must be finite and ascending")

    descending = raw_grid[::-1]
    model = Lasso(
        alpha=float(descending[0]),
        fit_intercept=False,
        max_iter=int(max_iterations),
        tol=float(tolerance),
        warm_start=True,
        selection="cyclic",
    )
    coefficients: list[np.ndarray] = []
    iterations: list[int] = []
    dual_gaps: list[float] = []
    convergence_warnings: list[int] = []
    for raw_lambda in descending:
        model.set_params(alpha=float(raw_lambda))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x, y)
        coefficients.append(np.asarray(model.coef_, dtype=np.float64).copy())
        iterations.append(int(model.n_iter_))
        dual_gaps.append(float(model.dual_gap_))
        convergence_warnings.append(
            sum(
                issubclass(item.category, ConvergenceWarning)
                for item in caught
            )
        )

    beta = np.asarray(coefficients, dtype=np.float64)[::-1]
    iterations_array = np.asarray(iterations, dtype=np.int64)[::-1]
    dual_gap_array = np.asarray(dual_gaps, dtype=np.float64)[::-1]
    warning_array = np.asarray(convergence_warnings, dtype=np.int64)[::-1]
    if not np.isfinite(beta).all() or not np.isfinite(dual_gap_array).all():
        raise RuntimeError("LASSO path contains non-finite values")
    return beta, {
        "iterations": iterations_array,
        "dual_gaps": dual_gap_array,
        "convergence_warnings": warning_array,
    }


def lasso_objective_and_kkt(
    x: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    raw_lambdas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    coefficient = np.asarray(beta, dtype=np.float64)
    lambdas = np.asarray(raw_lambdas, dtype=np.float64)
    objectives: list[float] = []
    data_terms: list[float] = []
    kkt_values: list[float] = []
    for row, raw_lambda in zip(coefficient, lambdas, strict=True):
        residual = x64 @ row - y64
        gradient = x64.T @ residual / x64.shape[0]
        magnitude = np.abs(row)
        active = magnitude > 1e-10
        active_residual = np.abs(
            gradient + raw_lambda * np.sign(row)
        )
        zero_residual = np.maximum(np.abs(gradient) - raw_lambda, 0.0)
        kkt = np.where(active, active_residual, zero_residual)
        data = 0.5 * float(np.mean(residual**2))
        objectives.append(data + float(raw_lambda * magnitude.sum()))
        data_terms.append(data)
        kkt_values.append(float(kkt.max(initial=0.0)))
    return (
        np.asarray(objectives),
        np.asarray(data_terms),
        np.asarray(kkt_values),
    )


def deterministic_top_features(
    kalbi_beta: np.ndarray,
    lasso_beta: np.ndarray,
    active_original_indices: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    importance = np.maximum(
        np.max(np.abs(kalbi_beta), axis=0),
        np.max(np.abs(lasso_beta), axis=0),
    )
    original = np.asarray(active_original_indices, dtype=np.int64)
    order = np.lexsort((original, -importance))
    selected = order[: min(int(count), order.size)]
    return selected.astype(np.int64), original[selected].astype(np.int64)


def evaluate_task(
    task: dict[str, Any],
    evaluation_root: Path,
    comparison_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    task_id = task["task_id"]
    split_path = Path(task["outer_split_path"])
    with np.load(split_path, allow_pickle=False) as archive:
        train_index = np.asarray(archive["train_indices"], dtype=np.int64)
        test_index = np.asarray(archive["test_indices"], dtype=np.int64)

    x_store = load_x(Path(task["X_path"]))
    y_store = np.load(Path(task["y_path"]), mmap_mode="r")
    x_train_raw = select_rows(x_store, train_index)
    y_train_raw = np.asarray(y_store[train_index], dtype=np.float64)
    x_train, y_train, scaler = fit_preprocessing(x_train_raw, y_train_raw)

    kalbi_metric = load_json(
        evaluation_root / "metrics" / f"{task_id}_kalbi.json"
    )
    baseline_metric = load_json(
        evaluation_root / "metrics" / f"{task_id}_baselines.json"
    )
    if (
        kalbi_metric["scaler_sha256"] != scaler["scaler_sha256"]
        or baseline_metric["scaler_sha256"] != scaler["scaler_sha256"]
    ):
        raise RuntimeError("KALBI/LASSO preprocessing mismatch")

    kalbi_path_path = evaluation_root / "paths" / f"{task_id}.npz"
    with np.load(kalbi_path_path, allow_pickle=False) as path:
        raw_lambdas = np.asarray(path["raw_lambdas"], dtype=np.float64)
        kalbi_beta = np.asarray(path["kalbi_beta"], dtype=np.float64)
        kalbi_test_mse = np.asarray(path["kalbi_test_mse"], dtype=np.float64)
        active_original_indices = np.asarray(
            path["active_original_indices"], dtype=np.int64
        )
        np.testing.assert_array_equal(path["train_indices"], train_index)
        np.testing.assert_array_equal(path["test_indices"], test_index)
    if kalbi_beta.shape != (raw_lambdas.size, x_train.shape[1]):
        raise RuntimeError("stored KALBI path shape mismatch")

    lasso_beta, diagnostics = fit_lasso_raw_path(
        x_train,
        y_train,
        raw_lambdas,
        tolerance=args.lasso_tolerance,
        max_iterations=args.lasso_max_iterations,
    )
    lasso_objective, lasso_data_term, lasso_kkt = (
        lasso_objective_and_kkt(
            x_train, y_train, lasso_beta, raw_lambdas
        )
    )

    x_test_raw = select_rows(x_store, test_index)
    x_test = apply_x_preprocessing(x_test_raw, scaler)
    y_test_raw = np.asarray(y_store[test_index], dtype=np.float64)
    lasso_test = raw_prediction_metrics(
        x_test,
        y_test_raw,
        lasso_beta,
        y_mean=float(scaler["y_mean"]),
        y_scale=float(scaler["y_scale"]),
    )
    zero = zero_predictor_metrics(y_test_raw, float(scaler["y_mean"]))
    normalized_kalbi = kalbi_test_mse / max(zero["mse"], 1e-30)
    normalized_lasso = lasso_test["mse"] / max(zero["mse"], 1e-30)

    selected_local, selected_original = deterministic_top_features(
        kalbi_beta,
        lasso_beta,
        active_original_indices,
        args.top_features,
    )
    lambda_max = float(
        np.max(
            np.abs(
                np.asarray(x_train, dtype=np.float64).T
                @ np.asarray(y_train, dtype=np.float64)
            )
        )
        / x_train.shape[0]
    )
    if raw_lambdas[-1] + 1e-12 < lambda_max:
        endpoint_zero_expected = False
    else:
        endpoint_zero_expected = True
        if np.count_nonzero(np.abs(lasso_beta[-1]) > 1e-10):
            raise RuntimeError("LASSO endpoint is not zero at lambda >= lambda_max")

    path_output = comparison_root / "paths" / f"{task_id}.npz"
    atomic_npz(
        path_output,
        raw_lambdas=raw_lambdas,
        active_original_indices=active_original_indices,
        kalbi_beta=kalbi_beta.astype(np.float32),
        lasso_beta=lasso_beta.astype(np.float32),
        kalbi_test_mse=kalbi_test_mse,
        lasso_test_mse=np.asarray(lasso_test["mse"], dtype=np.float64),
        kalbi_normalized_test_mse=normalized_kalbi,
        lasso_normalized_test_mse=normalized_lasso,
        lasso_test_prediction=np.asarray(
            lasso_test["prediction"], dtype=np.float32
        ),
        kalbi_nnz=np.count_nonzero(np.abs(kalbi_beta) > 1e-8, axis=1),
        lasso_nnz=np.count_nonzero(np.abs(lasso_beta) > 1e-8, axis=1),
        kalbi_l1=np.sum(np.abs(kalbi_beta), axis=1),
        lasso_l1=np.sum(np.abs(lasso_beta), axis=1),
        kalbi_l2=np.linalg.norm(kalbi_beta, axis=1),
        lasso_l2=np.linalg.norm(lasso_beta, axis=1),
        lasso_objective=lasso_objective,
        lasso_data_term=lasso_data_term,
        lasso_kkt=lasso_kkt,
        lasso_iterations=diagnostics["iterations"],
        lasso_dual_gaps=diagnostics["dual_gaps"],
        lasso_convergence_warnings=diagnostics[
            "convergence_warnings"
        ],
        top_features=selected_local,
        top_original_features=selected_original,
        train_indices=train_index,
        test_indices=test_index,
    )

    lasso_method = baseline_metric["methods"]["lasso"]
    metric = {
        "status": "PASS",
        "registry_index": int(task["registry_index"]),
        "task_id": task_id,
        "source": task["source"],
        "source_family": task_family(task["source"], task_id),
        "target": task["target"],
        "N": int(task["N"]),
        "P_raw": int(task["P_raw"]),
        "P_transformed": int(x_train.shape[1]),
        "outer_train_rows": int(train_index.size),
        "outer_test_rows": int(test_index.size),
        "scaler_sha256": scaler["scaler_sha256"],
        "same_training_only_preprocessing": True,
        "coordinate_descent_imported_or_executed": False,
        "lambda_contract": {
            "grid": "same raw lambda supplied directly to both methods",
            "minimum": float(raw_lambdas[0]),
            "maximum": float(raw_lambdas[-1]),
            "count": int(raw_lambdas.size),
            "lambda_max": lambda_max,
            "lasso_endpoint_zero_expected": endpoint_zero_expected,
        },
        "kalbi_selected_raw_lambda": float(
            kalbi_metric["selected_raw_lambda"]
        ),
        "kalbi_selected_normalized_test_mse": float(
            kalbi_metric["selected_normalized_test_mse"]
        ),
        "lasso_cv_selected_raw_lambda": float(
            lasso_method["selected_raw_lambda"]
        ),
        "lasso_cv_selected_normalized_test_mse": float(
            lasso_method["normalized_test_mse"]
        ),
        "kalbi_path_oracle_raw_lambda": float(
            raw_lambdas[int(np.argmin(normalized_kalbi))]
        ),
        "lasso_path_oracle_raw_lambda": float(
            raw_lambdas[int(np.argmin(normalized_lasso))]
        ),
        "kalbi_path_oracle_normalized_test_mse": float(
            np.min(normalized_kalbi)
        ),
        "lasso_path_oracle_normalized_test_mse": float(
            np.min(normalized_lasso)
        ),
        "lasso_max_kkt": float(np.max(lasso_kkt)),
        "lasso_max_dual_gap": float(
            np.max(diagnostics["dual_gaps"])
        ),
        "lasso_max_iterations": int(
            np.max(diagnostics["iterations"])
        ),
        "lasso_convergence_warning_count": int(
            np.sum(diagnostics["convergence_warnings"])
        ),
        "path_output": str(path_output),
        "seconds": time.perf_counter() - started,
    }
    atomic_json(
        comparison_root / "metrics" / f"{task_id}.json",
        metric,
    )
    return metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--task-indices", default="all")
    parser.add_argument("--lasso-tolerance", type=float, default=1e-7)
    parser.add_argument("--lasso-max-iterations", type=int, default=50000)
    parser.add_argument("--top-features", type=int, default=10)
    args = parser.parse_args()

    manifest = load_split_manifest(args.split_manifest)
    indices = parse_task_indices(
        args.task_indices, int(manifest["task_count"])
    )
    args.comparison_root.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    results: list[dict[str, Any]] = []
    for task in iter_selected_tasks(manifest, indices):
        task_id = task["task_id"]
        try:
            result = evaluate_task(
                task, args.evaluation_root, args.comparison_root, args
            )
            results.append(result)
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "registry_index": result["registry_index"],
                        "task_id": task_id,
                        "seconds": result["seconds"],
                        "lasso_max_kkt": result["lasso_max_kkt"],
                    }
                ),
                flush=True,
            )
        except Exception as error:
            failure = f"{task_id}: {type(error).__name__}: {error}"
            failures.append(failure)
            print(
                json.dumps({"status": "FAIL", "failure": failure}),
                flush=True,
            )

    worker_report = {
        "status": "PASS" if not failures else "FAIL",
        "task_indices": indices,
        "completed_task_count": len(results),
        "failure_count": len(failures),
        "failures": failures,
        "coordinate_descent_imported_or_executed": False,
        "results": results,
    }
    suffix = "_".join(str(index) for index in indices)
    atomic_json(
        args.comparison_root / "reports" / f"worker_{suffix}.json",
        worker_report,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
