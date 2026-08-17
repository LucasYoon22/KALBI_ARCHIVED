#!/usr/bin/env python3
"""Compare frozen KALBI results with CV LASSO, SCAD, and MCP."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import KFold
from skglm.datafits import Quadratic
from skglm.estimators import GeneralizedLinearEstimator
from skglm.penalties import MCPenalty
from skglm.solvers import AndersonCD

from kalbi_v6_exactprox10_eta4x_v24_additive512_traj3_k32s32_p100k_effbatch32.benchmark_kalbi_v3_cv_lasso_scad import (
    _lambda_max,
    _lasso_path,
    _prediction_mse,
    _scad_path,
    _standardize_apply,
    _standardize_fit,
)
from kalbi_v24_eval_n200p1000_screened4.prepare_v24_n200p1000_screened4 import (
    sha256,
    write_json,
)


METHODS = ("lasso", "scad", "mcp")


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


def fit_path(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    ratios: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    if method == "lasso":
        beta, iterations, gaps, warning_count = _lasso_path(
            x,
            y,
            ratios,
            tolerance=args.lasso_tolerance,
            max_iter=args.lasso_max_iter,
        )
        return beta, {
            "max_iterations": int(iterations.max()),
            "max_dual_gap": float(gaps.max()),
            "convergence_warning_count": int(warning_count),
        }
    if method == "scad":
        beta, stop, iterations = _scad_path(
            x,
            y,
            ratios,
            gamma=args.scad_gamma,
            tolerance=args.nonconvex_tolerance,
            max_iter=args.nonconvex_max_iter,
            max_epochs=args.nonconvex_max_epochs,
        )
    elif method == "mcp":
        beta, stop, iterations = mcp_path(
            x,
            y,
            ratios,
            gamma=args.mcp_gamma,
            tolerance=args.nonconvex_tolerance,
            max_iter=args.nonconvex_max_iter,
            max_epochs=args.nonconvex_max_epochs,
        )
    else:
        raise ValueError(f"unknown method: {method}")
    if not np.isfinite(stop).all():
        raise RuntimeError(f"{method} produced non-finite stop criteria")
    return beta, {
        "max_iterations": int(iterations.max()),
        "max_stop_criterion": float(stop.max()),
    }


def cross_validate(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    ratios: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    fold_mse: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    for fold_index, (fit_index, valid_index) in enumerate(folds):
        x_fit, y_fit, x_mean, x_scale, y_mean, y_scale = _standardize_fit(
            x[fit_index], y[fit_index]
        )
        x_valid = _standardize_apply(x[valid_index], x_mean, x_scale)
        beta, diagnostic = fit_path(method, x_fit, y_fit, ratios, args)
        fold_mse.append(
            _prediction_mse(
                x_valid,
                y[valid_index],
                beta,
                y_mean=y_mean,
                y_scale=y_scale,
            )
        )
        diagnostics.append({"fold": fold_index, **diagnostic})
    scores = np.asarray(fold_mse)
    mean = scores.mean(axis=0)
    standard_error = scores.std(axis=0, ddof=1) / math.sqrt(scores.shape[0])
    selected = int(np.argmin(mean))
    return {
        "fold_mse": scores,
        "mean_mse": mean,
        "standard_error": standard_error,
        "selected_index": selected,
        "selected_ratio": float(ratios[selected]),
        "fold_diagnostics": diagnostics,
    }


def refit_and_score(
    method: str,
    x_context: np.ndarray,
    y_context: np.ndarray,
    x_final: np.ndarray,
    y_final: np.ndarray,
    ratios: np.ndarray,
    selected_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    x_fit, y_fit, x_mean, x_scale, y_mean, y_scale = _standardize_fit(
        x_context, y_context
    )
    x_test = _standardize_apply(x_final, x_mean, x_scale)
    beta_path, diagnostics = fit_path(
        method, x_fit, y_fit, ratios[: selected_index + 1], args
    )
    beta = beta_path[-1]
    mse = float(
        _prediction_mse(
            x_test,
            y_final,
            beta[None],
            y_mean=y_mean,
            y_scale=y_scale,
        )[0]
    )
    denominator = float(np.sum((y_final - y_final.mean()) ** 2))
    prediction = y_mean + y_scale * (x_test @ beta)
    r2 = float(1.0 - np.sum((y_final - prediction) ** 2) / denominator)
    return {
        "final_mse": mse,
        "final_r2": r2,
        "nnz_1e4": int(np.count_nonzero(np.abs(beta) > 1e-4)),
        "beta": beta,
        "diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-tag", required=True)
    parser.add_argument("--task-limit", type=int, default=4)
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
    args = parser.parse_args()

    output = args.evaluation_root / args.output_tag
    if output.exists():
        raise RuntimeError(f"refusing existing output directory: {output}")
    output.mkdir(parents=True)
    preparation_path = args.evaluation_root / "metrics" / "preparation_summary.json"
    kalbi_results_path = args.evaluation_root / "full_41point" / "results.json"
    preparation = json.loads(preparation_path.read_text())
    kalbi = json.loads(kalbi_results_path.read_text())
    accepted = preparation["accepted"][: args.task_limit]
    kalbi = kalbi[: args.task_limit]
    if len(accepted) != args.task_limit or len(kalbi) != args.task_limit:
        raise RuntimeError("task count mismatch")
    ratios = np.logspace(
        0.0, math.log10(args.min_ratio), args.grid_size, dtype=np.float64
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for task_offset, (entry, kalbi_row) in enumerate(
        zip(accepted, kalbi, strict=True)
    ):
        if entry["label"] != kalbi_row["label"]:
            raise RuntimeError("preparation/KALBI task ordering mismatch")
        data_path = Path(entry["accepted_data_path"])
        split_path = Path(entry["split_path"])
        if sha256(data_path) != entry["accepted_data_sha256"]:
            raise RuntimeError("accepted data SHA256 mismatch")
        if sha256(split_path) != entry["split_sha256"]:
            raise RuntimeError("split SHA256 mismatch")
        with np.load(data_path, allow_pickle=False) as archive:
            x = np.asarray(archive["X"][0], dtype=np.float64)
            y = np.asarray(archive["y"][0], dtype=np.float64)
        with np.load(split_path, allow_pickle=False) as archive:
            context_index = np.asarray(archive["context_indices"], dtype=np.int64)
            final_index = np.asarray(archive["final_indices"], dtype=np.int64)
        if np.intersect1d(context_index, final_index).size:
            raise RuntimeError("context/final leakage")
        x_context, y_context = x[context_index], y[context_index]
        x_final, y_final = x[final_index], y[final_index]
        folds = list(
            KFold(
                n_splits=args.folds,
                shuffle=True,
                random_state=args.cv_seed + task_offset,
            ).split(x_context)
        )
        task_payload: dict[str, Any] = {}
        arrays: dict[str, np.ndarray] = {"lambda_ratios": ratios}
        for method in METHODS:
            cv = cross_validate(
                method, x_context, y_context, ratios, folds, args
            )
            fit = refit_and_score(
                method,
                x_context,
                y_context,
                x_final,
                y_final,
                ratios,
                cv["selected_index"],
                args,
            )
            task_payload[method] = {
                "selected_index": int(cv["selected_index"]),
                "selected_ratio": float(cv["selected_ratio"]),
                "final_mse": float(fit["final_mse"]),
                "final_r2": float(fit["final_r2"]),
                "normalized_final_mse": float(
                    fit["final_mse"] / entry["zero_predictor_final_mse"]
                ),
                "nnz_1e4": int(fit["nnz_1e4"]),
                "cv_diagnostics": cv["fold_diagnostics"],
                "refit_diagnostics": fit["diagnostics"],
            }
            arrays[f"{method}_fold_mse"] = cv["fold_mse"]
            arrays[f"{method}_mean_mse"] = cv["mean_mse"]
            arrays[f"{method}_standard_error"] = cv["standard_error"]
            arrays[f"{method}_beta"] = fit["beta"]
        row = {
            "dataset_index": int(entry["dataset_index"]),
            "label": entry["label"],
            "seed": int(entry["generator_seed"]),
            "n": int(entry["n"]),
            "p": int(entry["p"]),
            "kalbi_selected_raw_lambda": float(
                kalbi_row["selected_raw_lambda"]
            ),
            "kalbi_final_mse": float(kalbi_row["selected_final_mse"]),
            "kalbi_final_r2": float(kalbi_row["selected_final_r2"]),
            "kalbi_nnz_1e6": int(kalbi_row["selected_nnz_1e6"]),
            "zero_predictor_final_mse": float(
                entry["zero_predictor_final_mse"]
            ),
            **task_payload,
        }
        rows.append(row)
        np.savez_compressed(output / f"{entry['label']}_cv_paths.npz", **arrays)
        write_json(output / f"{entry['label']}_metrics.json", row)
        print(
            json.dumps(
                {
                    "label": entry["label"],
                    "kalbi_r2": row["kalbi_final_r2"],
                    **{
                        f"cv_{method}_r2": row[method]["final_r2"]
                        for method in METHODS
                    },
                }
            ),
            flush=True,
        )
    write_json(output / "results.json", rows)
    fields = [
        "dataset",
        "n",
        "p",
        "kalbi_final_r2",
        "cv_lasso_final_r2",
        "cv_scad_final_r2",
        "cv_mcp_final_r2",
        "kalbi_final_mse",
        "cv_lasso_final_mse",
        "cv_scad_final_mse",
        "cv_mcp_final_mse",
        "kalbi_nnz",
        "cv_lasso_nnz",
        "cv_scad_nnz",
        "cv_mcp_nnz",
    ]
    with (output / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row["label"],
                    "n": row["n"],
                    "p": row["p"],
                    "kalbi_final_r2": row["kalbi_final_r2"],
                    "cv_lasso_final_r2": row["lasso"]["final_r2"],
                    "cv_scad_final_r2": row["scad"]["final_r2"],
                    "cv_mcp_final_r2": row["mcp"]["final_r2"],
                    "kalbi_final_mse": row["kalbi_final_mse"],
                    "cv_lasso_final_mse": row["lasso"]["final_mse"],
                    "cv_scad_final_mse": row["scad"]["final_mse"],
                    "cv_mcp_final_mse": row["mcp"]["final_mse"],
                    "kalbi_nnz": row["kalbi_nnz_1e6"],
                    "cv_lasso_nnz": row["lasso"]["nnz_1e4"],
                    "cv_scad_nnz": row["scad"]["nnz_1e4"],
                    "cv_mcp_nnz": row["mcp"]["nnz_1e4"],
                }
            )
    manifest = {
        "status": "PASS",
        "methods": ["KALBI", "CV LASSO", "CV SCAD", "CV MCP"],
        "cv_contract": {
            "folds": args.folds,
            "seed": args.cv_seed,
            "selection": "minimum mean validation MSE",
            "lambda_ratio_grid": ratios.tolist(),
            "training_rows_only": True,
            "final_test_unused_until_refit": True,
        },
        "scad_gamma": args.scad_gamma,
        "mcp_gamma": args.mcp_gamma,
        "preparation_summary_sha256": sha256(preparation_path),
        "kalbi_results_sha256": sha256(kalbi_results_path),
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(output / "reproducibility_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
