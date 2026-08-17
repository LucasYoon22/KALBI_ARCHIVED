#!/usr/bin/env python3
"""Compare Kalbi V3/V17 with cross-validated LASSO and SCAD."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from skglm import GeneralizedLinearEstimator
from skglm.datafits import Quadratic
from skglm.penalties import SCAD
from skglm.solvers import AndersonCD


INK = "#20242a"
GRID = "#d8dde3"
BLUE = "#2367a6"
ORANGE = "#cf6a32"
OLIVE = "#768b36"
GRAY = "#767d86"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _standardize_fit(
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


def _standardize_apply(
    x: np.ndarray, x_mean: np.ndarray, x_scale: np.ndarray
) -> np.ndarray:
    return np.asfortranarray(
        (np.asarray(x, dtype=np.float64) - x_mean) / x_scale
    )


def _lambda_max(x: np.ndarray, y: np.ndarray) -> float:
    value = float(np.max(np.abs(np.asarray(x).T @ np.asarray(y))) / x.shape[0])
    return max(value, 1e-12)


def _lasso_path(
    x: np.ndarray,
    y: np.ndarray,
    ratios_descending: np.ndarray,
    *,
    tolerance: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    lambda_max = _lambda_max(x, y)
    model = Lasso(
        alpha=lambda_max,
        fit_intercept=False,
        max_iter=max_iter,
        tol=tolerance,
        warm_start=True,
        selection="cyclic",
    )
    coefficients: list[np.ndarray] = []
    iterations: list[int] = []
    dual_gaps: list[float] = []
    warning_count = 0
    for ratio in ratios_descending:
        model.set_params(alpha=lambda_max * float(ratio))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x, y)
        warning_count += sum(
            issubclass(item.category, ConvergenceWarning) for item in caught
        )
        if not np.isfinite(model.coef_).all() or not np.isfinite(model.dual_gap_):
            raise RuntimeError("LASSO produced non-finite coefficients or dual gap")
        coefficients.append(np.asarray(model.coef_, dtype=np.float64).copy())
        iterations.append(int(model.n_iter_))
        dual_gaps.append(float(model.dual_gap_))
    return (
        np.asarray(coefficients),
        np.asarray(iterations, dtype=np.int64),
        np.asarray(dual_gaps, dtype=np.float64),
        int(warning_count),
    )


def _scad_path(
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
        penalty=SCAD(lambda_max, gamma),
        solver=solver,
    )
    coefficients: list[np.ndarray] = []
    stop_criteria: list[float] = []
    iterations: list[int] = []
    for ratio in ratios_descending:
        estimator.penalty = SCAD(lambda_max * float(ratio), gamma)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message="`BaseEstimator._validate_data` is deprecated.*",
            )
            estimator.fit(x, y)
        beta = np.asarray(estimator.coef_, dtype=np.float64).reshape(-1)
        if not np.isfinite(beta).all():
            raise RuntimeError("SCAD produced non-finite coefficients")
        coefficients.append(beta.copy())
        stop_criteria.append(float(estimator.stop_crit_))
        iterations.append(int(estimator.n_iter_))
    return (
        np.asarray(coefficients),
        np.asarray(stop_criteria, dtype=np.float64),
        np.asarray(iterations, dtype=np.int64),
    )


def _prediction_mse(
    x: np.ndarray,
    y: np.ndarray,
    beta_path: np.ndarray,
    *,
    y_mean: float,
    y_scale: float,
) -> np.ndarray:
    prediction = y_mean + y_scale * (
        np.asarray(beta_path, dtype=np.float64) @ np.asarray(x, dtype=np.float64).T
    )
    return np.mean(
        (prediction - np.asarray(y, dtype=np.float64).reshape(1, -1)) ** 2,
        axis=1,
    )


def _cv_method(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    ratios_descending: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    fold_mse: list[np.ndarray] = []
    fold_diagnostics: list[dict[str, Any]] = []
    for fold_index, (train_index, valid_index) in enumerate(folds):
        x_fit, y_fit, x_mean, x_scale, y_mean, y_scale = _standardize_fit(
            x[train_index], y[train_index]
        )
        x_valid = _standardize_apply(x[valid_index], x_mean, x_scale)
        if method == "lasso":
            beta_path, iterations, dual_gaps, warning_count = _lasso_path(
                x_fit,
                y_fit,
                ratios_descending,
                tolerance=float(args.lasso_tolerance),
                max_iter=int(args.lasso_max_iter),
            )
            diagnostic = {
                "fold": fold_index,
                "max_iterations": int(iterations.max()),
                "max_dual_gap": float(dual_gaps.max()),
                "convergence_warning_count": warning_count,
            }
        elif method == "scad":
            beta_path, stop_criteria, iterations = _scad_path(
                x_fit,
                y_fit,
                ratios_descending,
                gamma=float(args.scad_gamma),
                tolerance=float(args.scad_tolerance),
                max_iter=int(args.scad_max_iter),
                max_epochs=int(args.scad_max_epochs),
            )
            diagnostic = {
                "fold": fold_index,
                "max_stop_criterion": float(stop_criteria.max()),
                "max_iterations": int(iterations.max()),
            }
        else:
            raise ValueError(f"unknown method: {method}")
        fold_mse.append(
            _prediction_mse(
                x_valid,
                y[valid_index],
                beta_path,
                y_mean=y_mean,
                y_scale=y_scale,
            )
        )
        fold_diagnostics.append(diagnostic)
    scores = np.asarray(fold_mse)
    mean_mse = scores.mean(axis=0)
    standard_error = scores.std(axis=0, ddof=1) / math.sqrt(scores.shape[0])
    selected_index = int(np.argmin(mean_mse))
    return {
        "fold_mse": scores,
        "mean_mse": mean_mse,
        "standard_error": standard_error,
        "selected_index": selected_index,
        "selected_ratio": float(ratios_descending[selected_index]),
        "fold_diagnostics": fold_diagnostics,
    }


def _fit_selected(
    method: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    ratios_descending: np.ndarray,
    selected_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    x_fit, y_fit, x_mean, x_scale, y_mean, y_scale = _standardize_fit(
        x_train, y_train
    )
    x_eval = _standardize_apply(x_test, x_mean, x_scale)
    path_prefix = ratios_descending[: selected_index + 1]
    if method == "lasso":
        beta_path, iterations, dual_gaps, warning_count = _lasso_path(
            x_fit,
            y_fit,
            path_prefix,
            tolerance=float(args.lasso_tolerance),
            max_iter=int(args.lasso_max_iter),
        )
        diagnostics = {
            "max_iterations": int(iterations.max()),
            "max_dual_gap": float(dual_gaps.max()),
            "convergence_warning_count": warning_count,
        }
    elif method == "scad":
        beta_path, stop_criteria, iterations = _scad_path(
            x_fit,
            y_fit,
            path_prefix,
            gamma=float(args.scad_gamma),
            tolerance=float(args.scad_tolerance),
            max_iter=int(args.scad_max_iter),
            max_epochs=int(args.scad_max_epochs),
        )
        diagnostics = {
            "max_stop_criterion": float(stop_criteria.max()),
            "max_iterations": int(iterations.max()),
        }
    else:
        raise ValueError(f"unknown method: {method}")
    beta = beta_path[-1]
    test_mse = float(
        _prediction_mse(
            x_eval,
            y_test,
            beta[None],
            y_mean=y_mean,
            y_scale=y_scale,
        )[0]
    )
    return {
        "test_mse": test_mse,
        "nnz_1e4": int(np.count_nonzero(np.abs(beta) > 1e-4)),
        "diagnostics": diagnostics,
    }


def _geomean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64).clip(1e-12)
    return float(np.exp(np.mean(np.log(array))))


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(color=GRID, alpha=0.7, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8, colors=INK)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-dir", type=Path, required=True)
    parser.add_argument("--paths-metrics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--grid-size", type=int, default=25)
    parser.add_argument("--min-ratio", type=float, default=1e-4)
    parser.add_argument("--split-seed", type=int, default=2026072217)
    parser.add_argument("--scad-gamma", type=float, default=3.7)
    parser.add_argument("--scad-tolerance", type=float, default=1e-7)
    parser.add_argument("--scad-max-iter", type=int, default=100)
    parser.add_argument("--scad-max-epochs", type=int, default=100000)
    parser.add_argument("--lasso-tolerance", type=float, default=1e-8)
    parser.add_argument("--lasso-max-iter", type=int, default=30000)
    args = parser.parse_args()

    if args.out_dir.exists():
        raise RuntimeError(f"refusing existing output directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    path_report = json.loads(args.paths_metrics.read_text())
    rows = list(path_report["rows"])
    ratios_descending = np.logspace(
        0.0,
        math.log10(float(args.min_ratio)),
        int(args.grid_size),
        dtype=np.float64,
    )
    result_rows: list[dict[str, Any]] = []
    cv_payloads: list[dict[str, Any]] = []

    for row_index, path_row in enumerate(rows):
        task_files = sorted(args.paths_dir.glob(f"task_{row_index:02d}_*.npz"))
        if len(task_files) != 1:
            raise RuntimeError(f"expected one path payload for row {row_index}")
        with np.load(task_files[0], allow_pickle=False) as payload:
            x_train = np.asarray(payload["x_train"], dtype=np.float64)
            y_train = np.asarray(payload["y_train"], dtype=np.float64)
            x_test = np.asarray(payload["x_test"], dtype=np.float64)
            y_test = np.asarray(payload["y_test"], dtype=np.float64)
        splitter = KFold(
            n_splits=int(args.folds),
            shuffle=True,
            random_state=int(args.split_seed) + row_index,
        )
        folds = list(splitter.split(x_train))
        lasso_cv = _cv_method(
            "lasso", x_train, y_train, ratios_descending, folds, args
        )
        scad_cv = _cv_method(
            "scad", x_train, y_train, ratios_descending, folds, args
        )
        lasso_fit = _fit_selected(
            "lasso",
            x_train,
            y_train,
            x_test,
            y_test,
            ratios_descending,
            int(lasso_cv["selected_index"]),
            args,
        )
        scad_fit = _fit_selected(
            "scad",
            x_train,
            y_train,
            x_test,
            y_test,
            ratios_descending,
            int(scad_cv["selected_index"]),
            args,
        )
        zero_test_mse = float(np.mean(y_test**2))
        row = {
            "row": row_index,
            "label": path_row["label"],
            "prior": path_row["prior"],
            "kalbi_v3_v17_test_mse": float(path_row["selected_test_mse"]),
            "cv_lasso_test_mse": float(lasso_fit["test_mse"]),
            "cv_scad_test_mse": float(scad_fit["test_mse"]),
            "zero_predictor_test_mse": zero_test_mse,
            "cv_lasso_selected_ratio": float(lasso_cv["selected_ratio"]),
            "cv_scad_selected_ratio": float(scad_cv["selected_ratio"]),
            "cv_lasso_nnz_1e4": int(lasso_fit["nnz_1e4"]),
            "cv_scad_nnz_1e4": int(scad_fit["nnz_1e4"]),
            "kalbi_v3_v17_nnz_1e4": int(path_row["selected_nnz_1e4"]),
            "lasso_cv_diagnostics": lasso_cv["fold_diagnostics"],
            "scad_cv_diagnostics": scad_cv["fold_diagnostics"],
            "lasso_full_fit_diagnostics": lasso_fit["diagnostics"],
            "scad_full_fit_diagnostics": scad_fit["diagnostics"],
        }
        if not np.isfinite(
            [
                row["kalbi_v3_v17_test_mse"],
                row["cv_lasso_test_mse"],
                row["cv_scad_test_mse"],
            ]
        ).all():
            raise RuntimeError(f"non-finite held-out metric for {row['label']}")
        result_rows.append(row)
        cv_payloads.append(
            {
                "lasso_mean": lasso_cv["mean_mse"],
                "lasso_se": lasso_cv["standard_error"],
                "lasso_selected": int(lasso_cv["selected_index"]),
                "scad_mean": scad_cv["mean_mse"],
                "scad_se": scad_cv["standard_error"],
                "scad_selected": int(scad_cv["selected_index"]),
            }
        )
        print(json.dumps(row, sort_keys=True), flush=True)

    columns = 2
    panel_rows = math.ceil(len(result_rows) / columns)
    cv_figure, cv_axes = plt.subplots(
        panel_rows, columns, figsize=(13, 3.1 * panel_rows), squeeze=False
    )
    x_axis = np.log10(ratios_descending)
    for index, (row, payload) in enumerate(
        zip(result_rows, cv_payloads, strict=True)
    ):
        axis = cv_axes.flat[index]
        for mean_key, se_key, selected_key, color, label in (
            ("lasso_mean", "lasso_se", "lasso_selected", ORANGE, "LASSO"),
            ("scad_mean", "scad_se", "scad_selected", OLIVE, "SCAD"),
        ):
            mean = np.asarray(payload[mean_key])
            se = np.asarray(payload[se_key])
            selected = int(payload[selected_key])
            axis.plot(x_axis, np.log10(mean.clip(1e-12)), color=color, lw=1.4, label=label)
            lower = np.log10(np.maximum(mean - se, 1e-12))
            upper = np.log10(mean + se)
            axis.fill_between(x_axis, lower, upper, color=color, alpha=0.12)
            axis.scatter(
                [x_axis[selected]],
                [math.log10(max(float(mean[selected]), 1e-12))],
                color=color,
                edgecolor=INK,
                linewidth=0.5,
                s=35,
                zorder=4,
            )
        axis.set_title(row["label"], fontsize=9, color=INK)
        axis.set_xlabel("log10(lambda / lambda_max)", fontsize=8)
        axis.set_ylabel("log10(mean CV MSE)", fontsize=8)
        if index == 0:
            axis.legend(fontsize=8, frameon=False)
        _style_axis(axis)
    for empty_index in range(len(result_rows), panel_rows * columns):
        cv_axes.flat[empty_index].axis("off")
    cv_figure.suptitle(
        f"{args.folds}-fold CV lambda paths for LASSO and SCAD (gamma={args.scad_gamma:g})",
        fontsize=12,
        color=INK,
    )
    cv_figure.tight_layout(rect=(0, 0, 1, 0.97))
    cv_path = args.out_dir / "cv_lasso_scad_lambda_paths.png"
    cv_figure.savefig(cv_path, dpi=190, facecolor="white")
    plt.close(cv_figure)

    labels = [row["label"] for row in result_rows]
    methods = (
        ("kalbi_v3_v17_test_mse", "Kalbi V3 + V17", BLUE),
        ("cv_lasso_test_mse", "CV LASSO", ORANGE),
        ("cv_scad_test_mse", "CV SCAD", OLIVE),
    )
    comparison_figure, comparison_axes = plt.subplots(
        1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [2.25, 1.0]}
    )
    positions = np.arange(len(labels), dtype=np.float64)
    width = 0.24
    for method_index, (key, label, color) in enumerate(methods):
        values = np.asarray([row[key] for row in result_rows])
        comparison_axes[0].bar(
            positions + (method_index - 1) * width,
            np.log10(values.clip(1e-12)),
            width=width,
            color=color,
            label=label,
        )
    comparison_axes[0].set_xticks(positions, labels, rotation=32, ha="right")
    comparison_axes[0].set_ylabel("log10(held-out test MSE)", fontsize=9)
    comparison_axes[0].set_title("Outer held-out prediction performance", fontsize=11)
    comparison_axes[0].legend(fontsize=8, frameon=False, ncol=3)
    _style_axis(comparison_axes[0])

    aggregate_values = [_geomean([row[key] for row in result_rows]) for key, _, _ in methods]
    baseline_geomean = _geomean(
        [row["zero_predictor_test_mse"] for row in result_rows]
    )
    normalized = np.asarray(aggregate_values) / baseline_geomean
    comparison_axes[1].bar(
        np.arange(len(methods)),
        normalized,
        color=[color for _, _, color in methods],
        width=0.68,
    )
    comparison_axes[1].axhline(1.0, color=GRAY, ls=":", lw=1.1, label="zero predictor")
    comparison_axes[1].set_xticks(
        np.arange(len(methods)), [label for _, label, _ in methods], rotation=25, ha="right"
    )
    comparison_axes[1].set_ylabel("geometric-mean MSE / zero predictor MSE", fontsize=9)
    comparison_axes[1].set_title("Aggregate across frozen tasks", fontsize=11)
    comparison_axes[1].legend(fontsize=8, frameon=False)
    _style_axis(comparison_axes[1])
    comparison_figure.suptitle(
        f"Kalbi V3/V17 vs CV LASSO and CV SCAD | checkpoint step {path_report['checkpoint_step']}",
        fontsize=13,
        color=INK,
    )
    comparison_figure.tight_layout(rect=(0, 0, 1, 0.95))
    comparison_path = args.out_dir / "heldout_prediction_comparison.png"
    comparison_figure.savefig(comparison_path, dpi=190, facecolor="white")
    plt.close(comparison_figure)

    aggregate = {
        "geomean_kalbi_v3_v17_test_mse": aggregate_values[0],
        "geomean_cv_lasso_test_mse": aggregate_values[1],
        "geomean_cv_scad_test_mse": aggregate_values[2],
        "geomean_zero_predictor_test_mse": baseline_geomean,
        "kalbi_over_cv_lasso": aggregate_values[0] / aggregate_values[1],
        "kalbi_over_cv_scad": aggregate_values[0] / aggregate_values[2],
    }
    report = {
        "report_version": "kalbi_v3_v17_vs_cv_lasso_scad_v1",
        "source_paths_metrics": str(args.paths_metrics.resolve()),
        "source_paths_metrics_sha256": _file_sha256(args.paths_metrics),
        "checkpoint_step": int(path_report["checkpoint_step"]),
        "outer_test_labels_used_for_model_selection": False,
        "cv_contract": {
            "folds": int(args.folds),
            "shuffle": True,
            "split_seed_base": int(args.split_seed),
            "lambda_ratio_grid_descending": ratios_descending.tolist(),
            "selection_rule": "minimum mean fold validation MSE",
            "fold_preprocessing": "center and scale using fold-training rows only",
            "lasso_objective": "0.5/n * ||y-Xb||^2 + lambda*||b||_1",
            "scad_gamma": float(args.scad_gamma),
            "scad_solver": "skglm GeneralizedLinearEstimator + AndersonCD",
        },
        "rows": result_rows,
        "aggregate": aggregate,
        "figures": {
            "cv_lambda_paths": str(cv_path.resolve()),
            "heldout_prediction_comparison": str(comparison_path.resolve()),
        },
    }
    metrics_path = args.out_dir / "cv_lasso_scad_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    complete_path = args.out_dir / "EVALUATION_COMPLETE.json"
    complete_path.write_text(
        json.dumps(
            {
                "checkpoint_step": int(path_report["checkpoint_step"]),
                "paths_metrics": str(args.paths_metrics.resolve()),
                "baseline_metrics": str(metrics_path.resolve()),
                "aggregate": aggregate,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "metrics": str(metrics_path),
                "figures": report["figures"],
                "aggregate": aggregate,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
