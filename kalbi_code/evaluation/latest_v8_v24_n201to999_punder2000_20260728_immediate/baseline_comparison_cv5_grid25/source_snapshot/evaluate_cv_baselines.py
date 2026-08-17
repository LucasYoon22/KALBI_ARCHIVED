#!/usr/bin/env python3
"""Evaluate CV LASSO, SCAD, and MCP on frozen KALBI evaluation tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from skglm.datafits import Quadratic
from skglm.estimators import GeneralizedLinearEstimator
from skglm.penalties import MCPenalty, SCAD
from skglm.solvers import AndersonCD


METHODS = ("lasso", "scad", "mcp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True, choices=range(1, 5))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--grid-size", type=int, default=25)
    parser.add_argument("--min-ratio", type=float, default=1e-4)
    parser.add_argument("--cv-seed", type=int, default=2026072801)
    parser.add_argument("--lasso-tolerance", type=float, default=1e-8)
    parser.add_argument("--lasso-max-iter", type=int, default=30000)
    parser.add_argument("--scad-gamma", type=float, default=3.7)
    parser.add_argument("--mcp-gamma", type=float, default=3.0)
    parser.add_argument("--nonconvex-tolerance", type=float, default=1e-7)
    parser.add_argument("--nonconvex-max-iter", type=int, default=100)
    parser.add_argument("--nonconvex-max-epochs", type=int, default=100000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def standardize_fit(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64).reshape(-1)
    x_mean = x64.mean(axis=0)
    x_scale = x64.std(axis=0)
    x_scale[x_scale < 1e-8] = 1.0
    y_mean = float(y64.mean())
    y_scale = float(y64.std())
    if y_scale < 1e-8:
        y_scale = 1.0
    return (
        np.asfortranarray((x64 - x_mean) / x_scale),
        np.ascontiguousarray((y64 - y_mean) / y_scale),
        x_mean,
        x_scale,
        y_mean,
        y_scale,
    )


def standardize_apply(
    x: np.ndarray, x_mean: np.ndarray, x_scale: np.ndarray
) -> np.ndarray:
    return np.asfortranarray(
        (np.asarray(x, dtype=np.float64) - x_mean) / x_scale
    )


def lambda_max(x: np.ndarray, y: np.ndarray) -> float:
    value = float(np.max(np.abs(x.T @ y)) / x.shape[0])
    return max(value, 1e-12)


def prediction(
    x_standardized: np.ndarray,
    beta: np.ndarray,
    *,
    y_mean: float,
    y_scale: float,
) -> np.ndarray:
    return y_mean + y_scale * (x_standardized @ beta)


def prediction_mse(
    x_standardized: np.ndarray,
    y: np.ndarray,
    beta_path: np.ndarray,
    *,
    y_mean: float,
    y_scale: float,
) -> np.ndarray:
    predictions = y_mean + y_scale * (
        np.asarray(beta_path, dtype=np.float64)
        @ np.asarray(x_standardized, dtype=np.float64).T
    )
    return np.mean(
        (predictions - np.asarray(y, dtype=np.float64).reshape(1, -1)) ** 2,
        axis=1,
    )


def lasso_path(
    x: np.ndarray,
    y: np.ndarray,
    ratios: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    lmax = lambda_max(x, y)
    model = Lasso(
        alpha=lmax,
        fit_intercept=False,
        max_iter=args.lasso_max_iter,
        tol=args.lasso_tolerance,
        warm_start=True,
        selection="cyclic",
    )
    coefficients: list[np.ndarray] = []
    iterations: list[int] = []
    dual_gaps: list[float] = []
    warning_flags: list[int] = []
    warning_count = 0
    for ratio in ratios:
        model.set_params(alpha=lmax * float(ratio))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x, y)
        point_warning_count = sum(
            issubclass(item.category, ConvergenceWarning) for item in caught
        )
        warning_count += point_warning_count
        warning_flags.append(point_warning_count)
        beta = np.asarray(model.coef_, dtype=np.float64)
        if not np.isfinite(beta).all() or not np.isfinite(model.dual_gap_):
            raise RuntimeError("LASSO produced non-finite output")
        coefficients.append(beta.copy())
        iterations.append(int(model.n_iter_))
        dual_gaps.append(float(model.dual_gap_))
    return np.asarray(coefficients), {
        "lambda_max": lmax,
        "max_iterations": max(iterations),
        "max_dual_gap": max(dual_gaps),
        "convergence_warning_count": warning_count,
        "iterations": iterations,
        "dual_gaps": dual_gaps,
        "warning_flags": warning_flags,
    }


def nonconvex_path(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    ratios: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    lmax = lambda_max(x, y)
    gamma = args.scad_gamma if method == "scad" else args.mcp_gamma
    penalty_type = SCAD if method == "scad" else MCPenalty
    solver = AndersonCD(
        max_iter=args.nonconvex_max_iter,
        max_epochs=args.nonconvex_max_epochs,
        p0=min(50, x.shape[1]),
        tol=args.nonconvex_tolerance,
        ws_strategy="subdiff",
        fit_intercept=False,
        warm_start=True,
        verbose=0,
    )
    estimator = GeneralizedLinearEstimator(
        datafit=Quadratic(),
        penalty=penalty_type(lmax, gamma),
        solver=solver,
    )
    coefficients: list[np.ndarray] = []
    iterations: list[int] = []
    stop_criteria: list[float] = []
    for ratio in ratios:
        estimator.penalty = penalty_type(lmax * float(ratio), gamma)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            estimator.fit(x, y)
        beta = np.asarray(estimator.coef_, dtype=np.float64).reshape(-1)
        if not np.isfinite(beta).all() or not np.isfinite(estimator.stop_crit_):
            raise RuntimeError(f"{method.upper()} produced non-finite output")
        coefficients.append(beta.copy())
        iterations.append(int(estimator.n_iter_))
        stop_criteria.append(float(estimator.stop_crit_))
    return np.asarray(coefficients), {
        "lambda_max": lmax,
        "max_iterations": max(iterations),
        "max_stop_criterion": max(stop_criteria),
    }


def fit_path(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    ratios: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    if method == "lasso":
        return lasso_path(x, y, ratios, args)
    return nonconvex_path(method, x, y, ratios, args)


def cross_validate(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    ratios: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    fold_mse: list[np.ndarray] = []
    fold_diagnostics: list[dict[str, Any]] = []
    for fold_index, (fit_index, valid_index) in enumerate(folds):
        x_fit, y_fit, x_mean, x_scale, y_mean, y_scale = standardize_fit(
            x[fit_index], y[fit_index]
        )
        x_valid = standardize_apply(x[valid_index], x_mean, x_scale)
        beta_path, diagnostic = fit_path(method, x_fit, y_fit, ratios, args)
        fold_mse.append(
            prediction_mse(
                x_valid,
                y[valid_index],
                beta_path,
                y_mean=y_mean,
                y_scale=y_scale,
            )
        )
        fold_diagnostics.append({"fold": fold_index, **diagnostic})
    scores = np.asarray(fold_mse)
    mean_mse = scores.mean(axis=0)
    standard_error = scores.std(axis=0, ddof=1) / math.sqrt(scores.shape[0])
    selected_index = int(np.argmin(mean_mse))
    if method == "lasso":
        selected_warning_count = sum(
            int(diagnostic["warning_flags"][selected_index])
            for diagnostic in fold_diagnostics
        )
        selected_max_dual_gap = max(
            float(diagnostic["dual_gaps"][selected_index])
            for diagnostic in fold_diagnostics
        )
        if selected_warning_count:
            raise RuntimeError(
                "CV selected a nonconverged LASSO point: "
                f"{selected_warning_count} warnings"
            )
    else:
        selected_warning_count = 0
        selected_max_dual_gap = None
    return (
        {
            "selected_index": selected_index,
            "selected_ratio": float(ratios[selected_index]),
            "selected_convergence_warning_count": selected_warning_count,
            "selected_max_dual_gap": selected_max_dual_gap,
            "fold_diagnostics": fold_diagnostics,
        },
        {
            "fold_mse": scores,
            "mean_mse": mean_mse,
            "standard_error": standard_error,
        },
    )


def refit_and_score(
    method: str,
    x_context: np.ndarray,
    y_context: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    ratios: np.ndarray,
    selected_index: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray]:
    x_fit, y_fit, x_mean, x_scale, y_mean, y_scale = standardize_fit(
        x_context, y_context
    )
    x_eval = standardize_apply(x_test, x_mean, x_scale)
    beta_path, diagnostics = fit_path(
        method, x_fit, y_fit, ratios[: selected_index + 1], args
    )
    beta = beta_path[-1]
    test_prediction = prediction(
        x_eval, beta, y_mean=y_mean, y_scale=y_scale
    )
    test_mse = float(np.mean((y_test - test_prediction) ** 2))
    test_tss = float(np.sum((y_test - y_test.mean()) ** 2))
    test_r2 = float(1.0 - np.sum((y_test - test_prediction) ** 2) / test_tss)
    return (
        {
            "test_mse": test_mse,
            "test_r2": test_r2,
            "nnz_1e4": int(np.count_nonzero(np.abs(beta) > 1e-4)),
            "refit_diagnostics": diagnostics,
            "mse_recompute_error": float(
                abs(
                    test_mse
                    - prediction_mse(
                        x_eval,
                        y_test,
                        beta[None],
                        y_mean=y_mean,
                        y_scale=y_scale,
                    )[0]
                )
            ),
        },
        beta,
    )


def recompute_kalbi(
    entry: dict[str, Any],
    kalbi_row: dict[str, Any],
    path_file: Path,
    x: np.ndarray,
    y: np.ndarray,
    split: dict[str, np.ndarray],
) -> dict[str, Any]:
    with np.load(path_file, allow_pickle=False) as archive:
        beta = np.asarray(archive["selected_beta"], dtype=np.float64)
    test_index = split["test_indices"]
    active = split["active"].astype(bool)
    x_standardized = (
        x[test_index][:, active]
        - np.asarray(split["x_mean"], dtype=np.float64)[active]
    ) / np.asarray(split["x_scale"], dtype=np.float64)[active]
    if beta.shape != (int(active.sum()),):
        raise RuntimeError("KALBI beta/active-feature shape mismatch")
    y_mean = float(split["y_mean"])
    y_scale = float(split["y_scale"])
    y_test = y[test_index]
    test_prediction = prediction(
        x_standardized, beta, y_mean=y_mean, y_scale=y_scale
    )
    test_mse = float(np.mean((y_test - test_prediction) ** 2))
    test_tss = float(np.sum((y_test - y_test.mean()) ** 2))
    test_r2 = float(1.0 - np.sum((y_test - test_prediction) ** 2) / test_tss)
    saved_error = abs(test_mse - float(kalbi_row["selected_test_mse"]))
    if saved_error > 2e-7:
        raise RuntimeError(f"KALBI test-MSE recomputation mismatch: {saved_error}")
    return {
        "selected_raw_lambda": float(kalbi_row["selected_raw_lambda"]),
        "test_mse": test_mse,
        "test_r2": test_r2,
        "normalized_test_mse": test_mse
        / float(entry["zero_predictor_test_mse"]),
        "nnz_1e4": int(np.count_nonzero(np.abs(beta) > 1e-4)),
        "saved_test_mse_recompute_error": saved_error,
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    root = args.evaluation_root.resolve()
    task_output = args.output_root.resolve() / f"dataset_{args.task_index}"
    if task_output.exists():
        raise RuntimeError(f"refusing existing output: {task_output}")
    task_output.mkdir(parents=True)

    accepted_path = root / "data_manifest/manifest/accepted_tasks.json"
    kalbi_path = root / "metrics/full_41point/results.json"
    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
    kalbi_results = json.loads(kalbi_path.read_text(encoding="utf-8"))
    entry = accepted[args.task_index - 1]
    kalbi_row = kalbi_results[args.task_index - 1]
    if (
        entry["dataset_index"] != args.task_index
        or kalbi_row["dataset_index"] != args.task_index
        or entry["task_hash"] != kalbi_row["task_hash"]
    ):
        raise RuntimeError("task ordering/hash mismatch")

    data_path = Path(entry["accepted_data_path"])
    split_path = Path(entry["split_path"])
    if sha256(data_path) != entry["accepted_data_sha256"]:
        raise RuntimeError("data SHA256 mismatch")
    if sha256(split_path) != entry["split_sha256"]:
        raise RuntimeError("split SHA256 mismatch")
    with np.load(data_path, allow_pickle=False) as archive:
        x = np.asarray(archive["X"][0], dtype=np.float64)
        y = np.asarray(archive["y"][0], dtype=np.float64)
    with np.load(split_path, allow_pickle=False) as archive:
        split = {key: np.asarray(archive[key]) for key in archive.files}

    context_index = split["context_indices"].astype(np.int64)
    validation_index = split["validation_indices"].astype(np.int64)
    test_index = split["test_indices"].astype(np.int64)
    if (
        np.intersect1d(context_index, validation_index).size
        or np.intersect1d(context_index, test_index).size
        or np.intersect1d(validation_index, test_index).size
    ):
        raise RuntimeError("context/validation/test leakage")
    if np.unique(np.concatenate((context_index, validation_index, test_index))).size != x.shape[0]:
        raise RuntimeError("split does not cover every row exactly once")

    x_context, y_context = x[context_index], y[context_index]
    x_test, y_test = x[test_index], y[test_index]
    folds = list(
        KFold(
            n_splits=args.folds,
            shuffle=True,
            random_state=args.cv_seed + args.task_index - 1,
        ).split(x_context)
    )
    ratios = np.logspace(
        0.0, math.log10(args.min_ratio), args.grid_size, dtype=np.float64
    )
    arrays: dict[str, np.ndarray] = {"lambda_ratio_grid": ratios}
    methods: dict[str, Any] = {}
    for method in METHODS:
        method_started = time.perf_counter()
        cv, cv_arrays = cross_validate(
            method, x_context, y_context, ratios, folds, args
        )
        scored, beta = refit_and_score(
            method,
            x_context,
            y_context,
            x_test,
            y_test,
            ratios,
            cv["selected_index"],
            args,
        )
        if scored["mse_recompute_error"] > 1e-12:
            raise RuntimeError(f"{method} MSE recomputation failed")
        methods[method] = {
            **cv,
            **scored,
            "normalized_test_mse": scored["test_mse"]
            / float(entry["zero_predictor_test_mse"]),
            "runtime_seconds": time.perf_counter() - method_started,
        }
        arrays[f"{method}_fold_mse"] = cv_arrays["fold_mse"]
        arrays[f"{method}_mean_mse"] = cv_arrays["mean_mse"]
        arrays[f"{method}_standard_error"] = cv_arrays["standard_error"]
        arrays[f"{method}_beta"] = beta
        print(
            json.dumps(
                {
                    "dataset": args.task_index,
                    "method": method,
                    "selected_ratio": cv["selected_ratio"],
                    "test_mse": scored["test_mse"],
                    "test_r2": scored["test_r2"],
                    "nnz": scored["nnz_1e4"],
                }
            ),
            flush=True,
        )

    path_file = (
        root
        / f"metrics/full_41point/dataset_{args.task_index:02d}/paths.npz"
    )
    kalbi = recompute_kalbi(entry, kalbi_row, path_file, x, y, split)
    payload = {
        "status": "PASS",
        "dataset_index": args.task_index,
        "task_id": entry["task_id"],
        "task_hash": entry["task_hash"],
        "seed": int(entry["generator_seed"]),
        "outcome_branch": entry["outcome_branch"],
        "n": int(entry["n"]),
        "p": int(entry["p"]),
        "split_sizes": {
            "context": int(context_index.size),
            "validation_unused": int(validation_index.size),
            "test": int(test_index.size),
        },
        "zero_predictor_test_mse": float(entry["zero_predictor_test_mse"]),
        "kalbi": kalbi,
        **methods,
        "contract": {
            "cv_folds": args.folds,
            "cv_seed": args.cv_seed + args.task_index - 1,
            "selection_rule": "minimum mean fold-validation MSE",
            "cv_rows": "context only",
            "validation_split_use": "unused",
            "test_use": "one final score after CV selection",
            "fold_scaling": "fit rows only",
            "refit_scaling": "all context rows only",
            "lambda_ratio_grid": ratios.tolist(),
            "scad_gamma": args.scad_gamma,
            "mcp_gamma": args.mcp_gamma,
            "lasso_tolerance": args.lasso_tolerance,
            "nonconvex_tolerance": args.nonconvex_tolerance,
        },
        "provenance": {
            "accepted_tasks_sha256": sha256(accepted_path),
            "kalbi_results_sha256": sha256(kalbi_path),
            "data_sha256": sha256(data_path),
            "split_sha256": sha256(split_path),
            "kalbi_path_arrays_sha256": sha256(path_file),
            "script_sha256": sha256(Path(__file__)),
            "pid": os.getpid(),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    np.savez_compressed(task_output / "cv_paths.npz", **arrays)
    write_json(task_output / "metrics.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
