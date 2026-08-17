#!/usr/bin/env python3
"""Compare one frozen Kalbi V6/V20 evaluation with CV LASSO and CV SCAD."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

from .benchmark_kalbi_v3_cv_lasso_scad import (
    BLUE,
    GRID,
    INK,
    OLIVE,
    ORANGE,
    _cv_method,
    _fit_selected,
    _style_axis,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-npz", type=Path, required=True)
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
    source_metrics = json.loads(args.paths_metrics.read_text(encoding="utf-8"))
    with np.load(args.paths_npz, allow_pickle=False) as payload:
        x_train = np.asarray(payload["x_train"], dtype=np.float64)
        y_train = np.asarray(payload["y_train"], dtype=np.float64)
        x_test = np.asarray(payload["x_test"], dtype=np.float64)
        y_test = np.asarray(payload["y_test"], dtype=np.float64)

    if x_train.shape[0] < int(args.folds):
        raise RuntimeError("outer training split is too small for requested CV")
    if x_train.shape[1] != int(source_metrics["p"]):
        raise RuntimeError("path payload and metrics disagree on P")

    ratios_descending = np.logspace(
        0.0,
        math.log10(float(args.min_ratio)),
        int(args.grid_size),
        dtype=np.float64,
    )
    splitter = KFold(
        n_splits=int(args.folds),
        shuffle=True,
        random_state=int(args.split_seed),
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

    kalbi_mse = float(source_metrics["selected_test_mse"])
    kalbi_nnz = int(source_metrics["selected_nnz_1e4"])
    zero_mse = float(np.mean(y_test**2))
    result = {
        "kalbi_v6_v20": {
            "test_mse": kalbi_mse,
            "nnz_1e4": kalbi_nnz,
            "selected_effective_ratio": float(
                source_metrics["selected_effective_ratio"]
            ),
        },
        "cv_lasso": {
            "test_mse": float(lasso_fit["test_mse"]),
            "nnz_1e4": int(lasso_fit["nnz_1e4"]),
            "selected_ratio": float(lasso_cv["selected_ratio"]),
            "cv_min_mean_mse": float(
                lasso_cv["mean_mse"][int(lasso_cv["selected_index"])]
            ),
            "cv_standard_error": float(
                lasso_cv["standard_error"][int(lasso_cv["selected_index"])]
            ),
            "fold_diagnostics": lasso_cv["fold_diagnostics"],
            "full_fit_diagnostics": lasso_fit["diagnostics"],
        },
        "cv_scad": {
            "test_mse": float(scad_fit["test_mse"]),
            "nnz_1e4": int(scad_fit["nnz_1e4"]),
            "selected_ratio": float(scad_cv["selected_ratio"]),
            "cv_min_mean_mse": float(
                scad_cv["mean_mse"][int(scad_cv["selected_index"])]
            ),
            "cv_standard_error": float(
                scad_cv["standard_error"][int(scad_cv["selected_index"])]
            ),
            "fold_diagnostics": scad_cv["fold_diagnostics"],
            "full_fit_diagnostics": scad_fit["diagnostics"],
        },
        "zero_predictor_test_mse": zero_mse,
    }
    result["relative"] = {
        "kalbi_over_lasso": kalbi_mse / float(lasso_fit["test_mse"]),
        "kalbi_over_scad": kalbi_mse / float(scad_fit["test_mse"]),
        "kalbi_over_zero": kalbi_mse / zero_mse,
        "lasso_over_zero": float(lasso_fit["test_mse"]) / zero_mse,
        "scad_over_zero": float(scad_fit["test_mse"]) / zero_mse,
    }

    x_axis = np.log10(ratios_descending)
    figure, axis = plt.subplots(figsize=(9.2, 5.8))
    for cv, color, label in (
        (lasso_cv, ORANGE, "LASSO"),
        (scad_cv, OLIVE, "SCAD"),
    ):
        mean = np.asarray(cv["mean_mse"])
        se = np.asarray(cv["standard_error"])
        selected = int(cv["selected_index"])
        axis.plot(x_axis, np.log10(mean.clip(1e-12)), color=color, lw=2, label=label)
        axis.fill_between(
            x_axis,
            np.log10(np.maximum(mean - se, 1e-12)),
            np.log10(mean + se),
            color=color,
            alpha=0.14,
        )
        axis.scatter(
            [x_axis[selected]],
            [math.log10(max(float(mean[selected]), 1e-12))],
            color=color,
            edgecolor=INK,
            linewidth=0.7,
            s=70,
            zorder=4,
        )
    axis.set_xlabel("log10(lambda / lambda_max)", fontsize=12)
    axis.set_ylabel("log10(mean fold-validation MSE)", fontsize=12)
    axis.set_title(
        f"{args.folds}-fold CV lambda selection | N={x_train.shape[0]}, P={x_train.shape[1]}",
        fontsize=14,
    )
    axis.legend(frameon=False, fontsize=11)
    _style_axis(axis)
    cv_path = args.out_dir / "cv_lambda_paths.png"
    figure.tight_layout()
    figure.savefig(cv_path, dpi=220, facecolor="white")
    plt.close(figure)

    labels = ["Kalbi V6 + V20", "CV LASSO", "CV SCAD"]
    mse_values = np.asarray(
        [
            result["kalbi_v6_v20"]["test_mse"],
            result["cv_lasso"]["test_mse"],
            result["cv_scad"]["test_mse"],
        ]
    )
    nnz_values = np.asarray(
        [
            result["kalbi_v6_v20"]["nnz_1e4"],
            result["cv_lasso"]["nnz_1e4"],
            result["cv_scad"]["nnz_1e4"],
        ]
    )
    colors = [BLUE, ORANGE, OLIVE]
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    bars = axes[0].bar(labels, mse_values, color=colors, width=0.68)
    axes[0].axhline(
        zero_mse, color="#767d86", ls=":", lw=1.4, label="zero predictor"
    )
    axes[0].bar_label(bars, fmt="%.4f", padding=3, fontsize=10)
    axes[0].set_ylabel("held-out test MSE", fontsize=12)
    axes[0].set_title("Outer held-out prediction error", fontsize=14)
    axes[0].legend(frameon=False, fontsize=10)
    _style_axis(axes[0])

    bars = axes[1].bar(labels, nnz_values, color=colors, width=0.68)
    axes[1].bar_label(bars, fmt="%d", padding=3, fontsize=10)
    axes[1].set_ylabel("nonzero coefficients (|beta| > 1e-4)", fontsize=12)
    axes[1].set_title("Selected model size", fontsize=14)
    _style_axis(axes[1])
    figure.suptitle(
        "Kalbi V6/V20 vs 5-fold CV LASSO and SCAD | "
        f"outer test N={x_test.shape[0]}, P={x_test.shape[1]}",
        fontsize=15,
    )
    comparison_path = args.out_dir / "heldout_mse_and_sparsity.png"
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(comparison_path, dpi=220, facecolor="white")
    plt.close(figure)

    np.savez_compressed(
        args.out_dir / "cv_paths.npz",
        lambda_ratios=ratios_descending,
        lasso_fold_mse=lasso_cv["fold_mse"],
        lasso_mean_mse=lasso_cv["mean_mse"],
        lasso_standard_error=lasso_cv["standard_error"],
        scad_fold_mse=scad_cv["fold_mse"],
        scad_mean_mse=scad_cv["mean_mse"],
        scad_standard_error=scad_cv["standard_error"],
    )
    report = {
        "report_version": "kalbi_v6_v20_vs_cv_lasso_scad_single_task_v1",
        "source_paths_npz": str(args.paths_npz.resolve()),
        "source_paths_npz_sha256": file_sha256(args.paths_npz),
        "source_paths_metrics": str(args.paths_metrics.resolve()),
        "source_paths_metrics_sha256": file_sha256(args.paths_metrics),
        "checkpoint_step": int(source_metrics["checkpoint_step"]),
        "task": {
            "shard_id": source_metrics.get("shard_id"),
            "n": int(source_metrics["n"]),
            "p": int(source_metrics["p"]),
            "outer_train_rows": int(x_train.shape[0]),
            "outer_test_rows": int(x_test.shape[0]),
            "outer_split_seed": int(source_metrics["split_seed"]),
        },
        "outer_test_labels_used_for_model_selection": False,
        "cv_contract": {
            "folds": int(args.folds),
            "shuffle": True,
            "split_seed": int(args.split_seed),
            "selection_rule": "minimum mean fold-validation MSE",
            "fold_preprocessing": "fit centering and scaling on fold-training rows only",
            "lambda_ratio_grid_descending": ratios_descending.tolist(),
            "lasso_objective": "0.5/n * ||y-Xb||^2 + lambda*||b||_1",
            "scad_gamma": float(args.scad_gamma),
            "scad_solver": "skglm GeneralizedLinearEstimator + AndersonCD",
        },
        "results": json_ready(result),
        "figures": {
            "cv_lambda_paths": str(cv_path.resolve()),
            "heldout_mse_and_sparsity": str(comparison_path.resolve()),
        },
    }
    metrics_path = args.out_dir / "comparison_metrics.json"
    metrics_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["results"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
