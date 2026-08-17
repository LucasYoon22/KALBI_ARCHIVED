#!/usr/bin/env python3
"""One-split CV-LASSO/SCAD/MCP baselines for registered real P>N tasks."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import KFold
from skglm.datafits import Quadratic
from skglm.estimators import GeneralizedLinearEstimator
from skglm.penalties import MCPenalty, SCAD
from skglm.solvers import AndersonCD, GramCD

from kalbi_v8_adaptiveblockprox10_v23_k32s32_p100k_effbatch32.benchmark_kalbi_v3_cv_lasso_scad import (
    _lambda_max,
    _lasso_path,
    _scad_path,
)

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
    save_preprocessing,
    select_rows,
    sha256,
    task_family,
    zero_predictor_metrics,
)


METHODS = ("lasso", "scad", "mcp")
_CV_WORKER_CONTEXT: dict[str, Any] | None = None


def mcp_path(
    x: np.ndarray,
    y: np.ndarray,
    ratios_descending: np.ndarray,
    *,
    gamma: float,
    tolerance: float,
    max_iter: int,
    max_epochs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lambda_max = _lambda_max(x, y)
    solver = AndersonCD(
        max_iter=max_iter,
        max_epochs=max_epochs,
        p0=min(50, x.shape[1]),
        tol=tolerance,
        ws_strategy="subdiff",
        fit_intercept=False,
        warm_start=True,
        verbose=0,
    )
    estimator = GeneralizedLinearEstimator(
        datafit=Quadratic(),
        penalty=MCPenalty(lambda_max, gamma),
        solver=solver,
    )
    coefficients: list[np.ndarray] = []
    stop_criteria: list[float] = []
    iterations: list[int] = []
    for ratio in ratios_descending:
        estimator.penalty = MCPenalty(lambda_max * float(ratio), gamma)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            estimator.fit(x, y)
        beta = np.asarray(estimator.coef_, dtype=np.float64).reshape(-1)
        if not np.isfinite(beta).all():
            raise RuntimeError("MCP produced non-finite coefficients")
        coefficients.append(beta.copy())
        stop_criteria.append(float(estimator.stop_crit_))
        iterations.append(int(estimator.n_iter_))
    return (
        np.asarray(coefficients),
        np.asarray(stop_criteria, dtype=np.float64),
        np.asarray(iterations, dtype=np.int64),
    )


def gram_nonconvex_path(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    ratios_descending: np.ndarray,
    *,
    gamma: float,
    tolerance: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lambda_max = _lambda_max(x, y)
    solver = GramCD(
        max_iter=max_iter,
        use_acc=False,
        greedy_cd=True,
        tol=tolerance,
        fit_intercept=False,
        warm_start=True,
        verbose=0,
    )
    coefficient = np.zeros(x.shape[1], dtype=np.float64)
    prediction = np.zeros(x.shape[0], dtype=np.float64)
    coefficients: list[np.ndarray] = []
    stop_criteria: list[float] = []
    iteration_budgets: list[int] = []
    for ratio in ratios_descending:
        raw_lambda = lambda_max * float(ratio)
        penalty = (
            SCAD(raw_lambda, gamma)
            if method == "scad"
            else MCPenalty(raw_lambda, gamma)
        )
        coefficient, _, stop_criterion = solver.solve(
            x,
            y,
            None,
            penalty,
            coefficient,
            prediction,
        )
        coefficient = np.asarray(coefficient, dtype=np.float64).reshape(-1)
        if not np.isfinite(coefficient).all():
            raise RuntimeError(
                f"GramCD {method} produced non-finite coefficients"
            )
        prediction = np.asarray(x, dtype=np.float64) @ coefficient
        coefficients.append(coefficient.copy())
        stop_criteria.append(float(stop_criterion))
        iteration_budgets.append(int(max_iter))
    return (
        np.asarray(coefficients),
        np.asarray(stop_criteria, dtype=np.float64),
        np.asarray(iteration_budgets, dtype=np.int64),
    )


def fit_path(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    ratios: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    if method == "lasso":
        beta, iterations, dual_gaps, warning_count = _lasso_path(
            x,
            y,
            ratios,
            tolerance=args.lasso_tolerance,
            max_iter=args.lasso_max_iter,
        )
        return beta, {
            "max_iterations": int(iterations.max()),
            "max_dual_gap": float(dual_gaps.max()),
            "convergence_warning_count": int(warning_count),
        }
    if method in {"scad", "mcp"}:
        beta, stop, iterations = gram_nonconvex_path(
            method,
            x,
            y,
            ratios,
            gamma=(
                args.scad_gamma if method == "scad" else args.mcp_gamma
            ),
            tolerance=args.nonconvex_tolerance,
            max_iter=args.nonconvex_max_iter,
        )
    else:
        raise ValueError(f"unknown method: {method}")
    if not np.isfinite(stop).all():
        raise RuntimeError(f"{method} produced non-finite stop criteria")
    return beta, {
        "max_iterations": int(iterations.max()),
        "max_stop_criterion": float(stop.max()),
    }


def penalty_value(
    method: str,
    magnitude: np.ndarray,
    raw_lambda: float,
    args: argparse.Namespace,
) -> np.ndarray:
    t = np.asarray(magnitude, dtype=np.float64)
    lam = float(raw_lambda)
    if method == "lasso":
        return lam * t
    if method == "scad":
        a = float(args.scad_gamma)
        return np.where(
            t <= lam,
            lam * t,
            np.where(
                t <= a * lam,
                (-t**2 + 2.0 * a * lam * t - lam**2)
                / (2.0 * (a - 1.0)),
                0.5 * (a + 1.0) * lam**2,
            ),
        )
    if method == "mcp":
        gamma = float(args.mcp_gamma)
        return np.where(
            t <= gamma * lam,
            lam * t - t**2 / (2.0 * gamma),
            0.5 * gamma * lam**2,
        )
    raise ValueError(method)


def penalty_derivative(
    method: str,
    magnitude: np.ndarray,
    raw_lambda: float,
    args: argparse.Namespace,
) -> np.ndarray:
    t = np.asarray(magnitude, dtype=np.float64)
    lam = float(raw_lambda)
    if method == "lasso":
        return np.full_like(t, lam)
    if method == "scad":
        a = float(args.scad_gamma)
        return np.where(
            t <= lam,
            lam,
            np.where(t <= a * lam, (a * lam - t) / (a - 1.0), 0.0),
        )
    if method == "mcp":
        return np.maximum(lam - t / float(args.mcp_gamma), 0.0)
    raise ValueError(method)


def objective_and_kkt(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    raw_lambda: float,
    args: argparse.Namespace,
) -> tuple[float, float]:
    coefficient = np.asarray(beta, dtype=np.float64)
    residual = np.asarray(x, dtype=np.float64) @ coefficient - np.asarray(
        y, dtype=np.float64
    )
    gradient = np.asarray(x, dtype=np.float64).T @ residual / x.shape[0]
    magnitude = np.abs(coefficient)
    derivative = penalty_derivative(
        method, magnitude, raw_lambda, args
    )
    active = magnitude > 1e-10
    active_residual = np.abs(
        gradient + derivative * np.sign(coefficient)
    )
    zero_residual = np.maximum(np.abs(gradient) - raw_lambda, 0.0)
    kkt = np.where(active, active_residual, zero_residual)
    objective = (
        0.5 * float(np.mean(residual**2))
        + float(
            penalty_value(method, magnitude, raw_lambda, args).sum()
        )
    )
    return objective, float(kkt.max(initial=0.0))


def evaluate_cv_fold(
    method: str,
    x_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    ratios: np.ndarray,
    fold_index: int,
    fit_index: np.ndarray,
    valid_index: np.ndarray,
    args: argparse.Namespace,
) -> tuple[int, np.ndarray, dict[str, Any]]:
    x_fit, y_fit, scaler = fit_preprocessing(
        x_train_raw[fit_index], y_train_raw[fit_index]
    )
    x_valid = apply_x_preprocessing(x_train_raw[valid_index], scaler)
    beta_path, diagnostic = fit_path(method, x_fit, y_fit, ratios, args)
    metric = raw_prediction_metrics(
        x_valid,
        y_train_raw[valid_index],
        beta_path,
        y_mean=float(scaler["y_mean"]),
        y_scale=float(scaler["y_scale"]),
    )
    return (
        fold_index,
        metric["mse"],
        {
            "fold": fold_index,
            "fit_rows": int(fit_index.size),
            "valid_rows": int(valid_index.size),
            "active_features": int(np.asarray(scaler["active"]).sum()),
            **diagnostic,
        },
    )


def initialize_cv_worker(
    method: str,
    x_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    ratios: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> None:
    global _CV_WORKER_CONTEXT
    _CV_WORKER_CONTEXT = {
        "method": method,
        "x_train_raw": x_train_raw,
        "y_train_raw": y_train_raw,
        "ratios": ratios,
        "folds": folds,
        "args": args,
    }


def evaluate_cv_fold_from_context(
    fold_index: int,
) -> tuple[int, np.ndarray, dict[str, Any]]:
    if _CV_WORKER_CONTEXT is None:
        raise RuntimeError("CV worker context was not initialized")
    fit_index, valid_index = _CV_WORKER_CONTEXT["folds"][fold_index]
    return evaluate_cv_fold(
        _CV_WORKER_CONTEXT["method"],
        _CV_WORKER_CONTEXT["x_train_raw"],
        _CV_WORKER_CONTEXT["y_train_raw"],
        _CV_WORKER_CONTEXT["ratios"],
        fold_index,
        fit_index,
        valid_index,
        _CV_WORKER_CONTEXT["args"],
    )


def cross_validate(
    method: str,
    x_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    ratios: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    results: dict[int, tuple[np.ndarray, dict[str, Any]]] = {}
    workers = min(max(1, int(args.cv_workers)), len(folds))
    if workers == 1:
        for fold_index, (fit_index, valid_index) in enumerate(folds):
            index, mse, diagnostic = evaluate_cv_fold(
                method,
                x_train_raw,
                y_train_raw,
                ratios,
                fold_index,
                fit_index,
                valid_index,
                args,
            )
            results[index] = (mse, diagnostic)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("fork"),
            initializer=initialize_cv_worker,
            initargs=(
                method,
                x_train_raw,
                y_train_raw,
                ratios,
                folds,
                args,
            ),
        ) as executor:
            futures = [
                executor.submit(evaluate_cv_fold_from_context, fold_index)
                for fold_index in range(len(folds))
            ]
            for future in as_completed(futures):
                index, mse, diagnostic = future.result()
                results[index] = (mse, diagnostic)

    fold_mse = [results[index][0] for index in range(len(folds))]
    diagnostics = [results[index][1] for index in range(len(folds))]
    scores = np.asarray(fold_mse, dtype=np.float64)
    mean = scores.mean(axis=0)
    standard_error = (
        scores.std(axis=0, ddof=1) / math.sqrt(scores.shape[0])
    )
    selected_index = int(np.argmin(mean))
    return {
        "fold_mse": scores,
        "mean_mse": mean,
        "standard_error": standard_error,
        "selected_index": selected_index,
        "selected_ratio": float(ratios[selected_index]),
        "fold_diagnostics": diagnostics,
    }


def fit_outer_training(
    method: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    ratios: np.ndarray,
    selected_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    beta_path, diagnostics = fit_path(
        method,
        x_train,
        y_train,
        ratios[: selected_index + 1],
        args,
    )
    beta = beta_path[-1]
    lambda_max = _lambda_max(x_train, y_train)
    raw_lambda = float(lambda_max * ratios[selected_index])
    objective, kkt = objective_and_kkt(
        method, x_train, y_train, beta, raw_lambda, args
    )
    return {
        "beta": beta,
        "lambda_max": float(lambda_max),
        "selected_raw_lambda": raw_lambda,
        "objective": objective,
        "kkt_max": kkt,
        "diagnostics": diagnostics,
    }


def evaluate_task(
    task: dict[str, Any],
    evaluation_root: Path,
    ratios: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    task_id = task["task_id"]
    x_path = Path(task["X_path"])
    y_path = Path(task["y_path"])
    split_path = Path(task["outer_split_path"])
    for path, expected in (
        (x_path, task["X_sha256"]),
        (y_path, task["y_sha256"]),
        (split_path, task["outer_split_sha256"]),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"artifact hash mismatch: {path}")

    x_store = load_x(x_path)
    y_store = np.load(y_path, mmap_mode="r")
    with np.load(split_path, allow_pickle=False) as archive:
        train_index = np.asarray(archive["train_indices"], dtype=np.int64)
        test_index = np.asarray(archive["test_indices"], dtype=np.int64)
    x_train_raw = select_rows(x_store, train_index)
    y_train_raw = np.asarray(y_store[train_index], dtype=np.float64)
    x_train, y_train, scaler = fit_preprocessing(
        x_train_raw, y_train_raw
    )
    preprocessing_path, preprocessing_sha = save_preprocessing(
        evaluation_root, task, scaler
    )

    fold_count = 5 if train_index.size >= 100 else 3
    cv_seed = int(task["outer_split_seed"]) ^ 0x00C0FFEE
    folds = list(
        KFold(
            n_splits=fold_count,
            shuffle=True,
            random_state=cv_seed,
        ).split(x_train_raw)
    )
    method_results: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {"lambda_ratios": ratios}
    fitted: dict[str, dict[str, Any]] = {}
    method_freeze_time: float | None = None
    for method in METHODS:
        method_started = time.perf_counter()
        print(
            json.dumps(
                {
                    "task_id": task_id,
                    "status": "METHOD_START",
                    "method": method,
                    "fold_count": fold_count,
                    "cv_workers": int(args.cv_workers),
                    "grid_size": int(ratios.size),
                }
            ),
            flush=True,
        )
        cv = cross_validate(
            method,
            x_train_raw,
            y_train_raw,
            ratios,
            folds,
            args,
        )
        refit = fit_outer_training(
            method,
            x_train,
            y_train,
            ratios,
            cv["selected_index"],
            args,
        )
        fitted[method] = refit
        method_results[method] = {
            "solver": (
                "GramCD"
                if method in {"scad", "mcp"}
                else "sklearn_coordinate_descent"
            ),
            "selected_index": int(cv["selected_index"]),
            "selected_ratio": float(cv["selected_ratio"]),
            "selected_raw_lambda": float(refit["selected_raw_lambda"]),
            "outer_train_lambda_max": float(refit["lambda_max"]),
            "train_objective": float(refit["objective"]),
            "train_kkt_max": float(refit["kkt_max"]),
            "cv_folds": fold_count,
            "cv_seed": cv_seed,
            "cv_diagnostics": cv["fold_diagnostics"],
            "refit_diagnostics": refit["diagnostics"],
            "fit_seconds": time.perf_counter() - method_started,
        }
        arrays[f"{method}_fold_mse"] = cv["fold_mse"]
        arrays[f"{method}_mean_mse"] = cv["mean_mse"]
        arrays[f"{method}_standard_error"] = cv["standard_error"]
        arrays[f"{method}_beta"] = refit["beta"].astype(np.float32)
        print(
            json.dumps(
                {
                    "task_id": task_id,
                    "status": "METHOD_PASS",
                    "method": method,
                    "selected_ratio": float(cv["selected_ratio"]),
                    "seconds": time.perf_counter() - method_started,
                }
            ),
            flush=True,
        )
    method_freeze_time = time.time()

    # Test y is first materialized after all three CV choices and refits freeze.
    x_test_raw = select_rows(x_store, test_index)
    x_test = apply_x_preprocessing(x_test_raw, scaler)
    y_test_raw = np.asarray(y_store[test_index], dtype=np.float64)
    zero = zero_predictor_metrics(y_test_raw, float(scaler["y_mean"]))
    for method in METHODS:
        test = raw_prediction_metrics(
            x_test,
            y_test_raw,
            fitted[method]["beta"],
            y_mean=float(scaler["y_mean"]),
            y_scale=float(scaler["y_scale"]),
        )
        method_results[method].update(
            {
                "test_mse": float(test["mse"][0]),
                "test_rmse": float(test["rmse"][0]),
                "test_mae": float(test["mae"][0]),
                "test_r2": float(test["r2"][0]),
                "normalized_test_mse": float(
                    test["mse"][0] / max(zero["mse"], 1e-30)
                ),
                "nnz_1e8": int(
                    np.count_nonzero(
                        np.abs(fitted[method]["beta"]) > 1e-8
                    )
                ),
            }
        )
        arrays[f"{method}_test_prediction"] = test["prediction"][
            0
        ].astype(np.float32)

    path = evaluation_root / "baselines" / f"{task_id}.npz"
    atomic_npz(
        path,
        **arrays,
        train_indices=train_index,
        test_indices=test_index,
    )
    result = {
        "status": "PASS",
        "registry_index": int(task["registry_index"]),
        "task_id": task_id,
        "source": task["source"],
        "source_family": task_family(task["source"], task_id),
        "target": task["target"],
        "evidence_tier": task["evidence_tier"],
        "N": int(task["N"]),
        "P_raw": int(task["P_raw"]),
        "P_transformed": int(np.asarray(scaler["active"]).sum()),
        "outer_train_rows": int(train_index.size),
        "outer_test_rows": int(test_index.size),
        "outer_split_seed": int(task["outer_split_seed"]),
        "outer_split_sha256": task["outer_split_sha256"],
        "preprocessing_path": str(preprocessing_path),
        "preprocessing_sha256": preprocessing_sha,
        "scaler_sha256": scaler["scaler_sha256"],
        "test_y_accessed_after_method_freeze": True,
        "method_freeze_unix": method_freeze_time,
        "zero_predictor": zero,
        "methods": method_results,
        "lambda_ratio_grid": ratios.tolist(),
        "baseline_path": str(path),
        "baseline_path_sha256": sha256(path),
        "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(
        evaluation_root / "metrics" / f"{task_id}_baselines.json",
        result,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--task-indices", default="all")
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--min-ratio", type=float, default=1e-4)
    parser.add_argument("--cv-workers", type=int, default=1)
    parser.add_argument("--lasso-tolerance", type=float, default=1e-8)
    parser.add_argument("--lasso-max-iter", type=int, default=30000)
    parser.add_argument("--scad-gamma", type=float, default=3.7)
    parser.add_argument("--mcp-gamma", type=float, default=3.0)
    parser.add_argument("--nonconvex-tolerance", type=float, default=1e-7)
    parser.add_argument("--nonconvex-max-iter", type=int, default=100)
    parser.add_argument("--nonconvex-max-epochs", type=int, default=100000)
    args = parser.parse_args()

    validation = json.loads(
        args.validation_manifest.read_text(encoding="utf-8")
    )
    if validation.get("status") != "PASS":
        raise RuntimeError("nonconvex baseline validation did not pass")
    split_manifest = load_split_manifest(args.split_manifest)
    indices = parse_task_indices(
        args.task_indices, int(split_manifest["task_count"])
    )
    ratios = np.logspace(
        0.0, math.log10(args.min_ratio), args.grid_size, dtype=np.float64
    )
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for task in iter_selected_tasks(split_manifest, indices):
        task_id = task["task_id"]
        metric_path = (
            args.evaluation_root / "metrics" / f"{task_id}_baselines.json"
        )
        if metric_path.exists():
            existing = json.loads(metric_path.read_text(encoding="utf-8"))
            path = Path(existing["baseline_path"])
            if (
                existing.get("status") == "PASS"
                and path.is_file()
                and sha256(path) == existing["baseline_path_sha256"]
            ):
                print(
                    json.dumps(
                        {"task_id": task_id, "status": "SKIP_VERIFIED"}
                    ),
                    flush=True,
                )
                successes.append(existing)
                continue
            raise RuntimeError(f"invalid existing baseline output: {task_id}")
        print(
            json.dumps(
                {
                    "task_id": task_id,
                    "registry_index": task["registry_index"],
                    "status": "START",
                }
            ),
            flush=True,
        )
        try:
            result = evaluate_task(
                task, args.evaluation_root, ratios, args
            )
            successes.append(result)
            print(
                json.dumps(
                    {
                        "task_id": task_id,
                        "status": "PASS",
                        **{
                            f"{method}_r2": result["methods"][method][
                                "test_r2"
                            ]
                            for method in METHODS
                        },
                        "seconds": result["runtime_seconds"],
                    }
                ),
                flush=True,
            )
        except Exception as error:
            failure = {
                "status": "FAIL",
                "task_id": task_id,
                "registry_index": int(task["registry_index"]),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            atomic_json(
                args.evaluation_root
                / "metrics"
                / f"{task_id}_baselines_failure.json",
                failure,
            )
            print(json.dumps(failure), flush=True)

    worker = {
        "status": "PASS" if not failures else "PARTIAL",
        "task_indices": indices,
        "success_count": len(successes),
        "failure_count": len(failures),
        "grid_size": args.grid_size,
        "min_ratio": args.min_ratio,
        "scad_gamma": args.scad_gamma,
        "mcp_gamma": args.mcp_gamma,
        "failures": failures,
    }
    tag = "_".join(map(str, indices))
    atomic_json(
        args.evaluation_root / "reports" / f"baseline_worker_{tag}.json",
        worker,
    )
    print(json.dumps(worker, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
