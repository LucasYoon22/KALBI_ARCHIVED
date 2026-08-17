#!/usr/bin/env python3
"""Compare frozen KALBI T=10/T=100 inference with CV LASSO and SCAD."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from sklearn.model_selection import KFold

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .benchmark_kalbi_v3_cv_lasso_scad import _cv_method, _fit_selected
from .eval_kalbi_v2_common import encode_dataset, file_sha256, tensors_from_arrays
from .paper_eval_unseen_dimension5 import (
    load_frozen_task,
    prediction_mse,
    split_standardize_three_way,
)
from .report_latest_v23_selectoronly_solution_selector_mixed6 import (
    SEED_STRIDE,
    load_components_v23,
    predict_ablation,
)


BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#666666"
LIGHT_GREY = "#D0D0D0"
METHODS = ("KALBI T=10", "KALBI T=100", "CV LASSO", "CV SCAD")
METHOD_COLORS = (BLUE, ORANGE, GREEN, PURPLE)


def _oracle_index(values: np.ndarray, raw_lambdas: np.ndarray) -> int:
    minimum = float(np.min(values))
    ties = np.flatnonzero(
        np.isclose(values, minimum, rtol=1e-12, atol=1e-14)
    )
    return int(ties[np.argmax(raw_lambdas[ties])])


def _plot_selector_penalty(
    tasks: list[dict[str, Any]], output: Path
) -> None:
    figure, axes = plt.subplots(
        len(tasks),
        2,
        figsize=(10.8, 2.35 * len(tasks)),
        squeeze=False,
    )
    for row, task in enumerate(tasks):
        x_axis = np.log10(task["raw_lambdas"])
        zero_mse = task["zero_predictor_test_mse"]
        selected_x = math.log10(task["selected_raw_lambda"])

        path_axis = axes[row, 0]
        path_axis.plot(
            x_axis,
            task["t10_path_mse"] / zero_mse,
            color=BLUE,
            lw=1.45,
            label="KALBI T=10",
        )
        path_axis.plot(
            x_axis,
            task["t100_path_mse"] / zero_mse,
            color=ORANGE,
            lw=1.45,
            label="KALBI T=100",
        )
        path_axis.plot(
            x_axis,
            task["cd_path_mse"] / zero_mse,
            color=GREY,
            lw=1.1,
            ls="--",
            label="Coordinate descent reference",
        )
        path_axis.axhline(1.0, color=LIGHT_GREY, lw=0.9, ls=":")
        path_axis.axvline(selected_x, color=RED, lw=1.0, ls=":")
        path_axis.scatter(
            [selected_x],
            [task["t10_selected_mse"] / zero_mse],
            color=BLUE,
            marker="o",
            s=28,
            zorder=5,
        )
        path_axis.scatter(
            [selected_x],
            [task["t100_selected_mse"] / zero_mse],
            color=ORANGE,
            marker="D",
            s=28,
            zorder=5,
        )
        path_axis.scatter(
            [x_axis[task["t100_oracle_index"]]],
            [
                task["t100_path_mse"][task["t100_oracle_index"]]
                / zero_mse
            ],
            color=ORANGE,
            marker="*",
            edgecolor="black",
            linewidth=0.35,
            s=60,
            zorder=5,
        )
        path_axis.set_ylabel(
            f"Dataset {task['dataset_index']}\nTest MSE / zero MSE"
        )
        path_axis.set_title(
            f"Held-out error (N={task['n']}, P={task['p']})",
            fontsize=10,
        )
        path_axis.grid(axis="y", color="#E7E7E7", lw=0.6)
        if row == len(tasks) - 1:
            path_axis.set_xlabel(r"$\log_{10}(\lambda_{\rm raw})$")

        penalty_axis = axes[row, 1]
        derivative_axis = penalty_axis.twinx()
        penalty_axis.plot(
            task["penalty_grid"],
            task["penalty_rho"],
            color=BLUE,
            lw=1.5,
        )
        derivative_axis.plot(
            task["penalty_grid"],
            task["penalty_drho"],
            color=ORANGE,
            lw=1.1,
            ls="--",
        )
        penalty_axis.set_ylim(bottom=0.0)
        derivative_axis.set_ylim(bottom=0.0)
        penalty_axis.set_ylabel(r"$\rho(|\beta|)$", color=BLUE)
        derivative_axis.set_ylabel(r"$\rho'(|\beta|)$", color=ORANGE)
        penalty_axis.set_title("KALBI-selected penalty", fontsize=10)
        penalty_axis.grid(axis="y", color="#E7E7E7", lw=0.6)
        if row == len(tasks) - 1:
            penalty_axis.set_xlabel(r"$|\beta|$")

    handles = [
        Line2D([0], [0], color=BLUE, lw=1.5, label="KALBI T=10"),
        Line2D([0], [0], color=ORANGE, lw=1.5, label="KALBI T=100"),
        Line2D(
            [0],
            [0],
            color=GREY,
            lw=1.1,
            ls="--",
            label="Coordinate descent reference",
        ),
        Line2D(
            [0],
            [0],
            color=RED,
            lw=1.0,
            ls=":",
            label="KALBI-selected lambda",
        ),
        Line2D(
            [0],
            [0],
            color=ORANGE,
            marker="*",
            markeredgecolor="black",
            lw=0,
            label="T=100 path oracle",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.003),
    )
    figure.subplots_adjust(
        left=0.11,
        right=0.90,
        top=0.975,
        bottom=0.075,
        hspace=0.54,
        wspace=0.38,
    )
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_baseline_comparison(
    rows: list[dict[str, Any]], output: Path
) -> None:
    normalized = np.asarray(
        [
            [
                row["kalbi_t10_normalized_mse"],
                row["kalbi_t100_normalized_mse"],
                row["cv_lasso_normalized_mse"],
                row["cv_scad_normalized_mse"],
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.0),
        gridspec_kw={"width_ratios": [3.1, 1.25]},
    )
    positions = np.arange(len(rows), dtype=np.float64)
    width = 0.19
    for index, (method, color) in enumerate(zip(METHODS, METHOD_COLORS)):
        axes[0].bar(
            positions + (index - 1.5) * width,
            normalized[:, index],
            width=width,
            color=color,
            label=method,
        )
    axes[0].axhline(
        1.0,
        color=GREY,
        lw=0.9,
        ls=":",
        label="Zero-predictor baseline",
    )
    axes[0].set_xticks(
        positions,
        [
            f"D{row['dataset_index']}\n{row['n']}x{row['p']}"
            for row in rows
        ],
    )
    axes[0].set_ylabel("Held-out MSE / zero-predictor MSE")
    axes[0].set_title("Per-dataset held-out prediction error")
    axes[0].grid(axis="y", color="#E7E7E7", lw=0.6)

    geometric_means = np.exp(np.mean(np.log(np.maximum(normalized, 1e-12)), axis=0))
    aggregate_positions = np.arange(len(METHODS))
    axes[1].bar(
        aggregate_positions,
        geometric_means,
        color=METHOD_COLORS,
        width=0.68,
    )
    axes[1].axhline(1.0, color=GREY, lw=0.9, ls=":")
    axes[1].set_xticks(
        aggregate_positions,
        ["T=10", "T=100", "LASSO", "SCAD"],
        rotation=25,
        ha="right",
    )
    axes[1].set_ylabel("Geometric mean normalized MSE")
    axes[1].set_title("Aggregate across six datasets")
    axes[1].grid(axis="y", color="#E7E7E7", lw=0.6)
    for index, value in enumerate(geometric_means):
        axes[1].text(
            index,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.985,
        top=0.90,
        bottom=0.22,
        wspace=0.30,
    )
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--paths-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--split-seed", type=int, default=2_026_072_407)
    parser.add_argument("--cv-seed", type=int, default=2_026_072_600)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-grid-size", type=int, default=25)
    parser.add_argument("--cv-min-ratio", type=float, default=1e-4)
    parser.add_argument("--lasso-tolerance", type=float, default=1e-8)
    parser.add_argument("--lasso-max-iter", type=int, default=30_000)
    parser.add_argument("--scad-gamma", type=float, default=3.7)
    parser.add_argument("--scad-tolerance", type=float, default=1e-7)
    parser.add_argument("--scad-max-iter", type=int, default=100)
    parser.add_argument("--scad-max-epochs", type=int, default=100_000)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    prepared_root = args.prepared_root.resolve()
    paths_root = args.paths_root.resolve()
    frozen = json.loads(
        (
            prepared_root
            / "reproducibility"
            / "reproducibility_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = list(frozen["accepted_tasks"])
    if len(entries) != 6:
        raise RuntimeError("expected exactly six frozen tasks")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint, model, selector = load_components_v23(
        args.checkpoint.resolve(), args.config.resolve(), device
    )
    cv_ratios = np.logspace(
        0.0,
        math.log10(float(args.cv_min_ratio)),
        int(args.cv_grid_size),
        dtype=np.float64,
    )
    rows: list[dict[str, Any]] = []
    task_plots: list[dict[str, Any]] = []
    detailed: list[dict[str, Any]] = []

    for entry in entries:
        dataset_index = int(entry["dataset_index"])
        print(
            json.dumps(
                {
                    "status": "starting",
                    "dataset": dataset_index,
                    "n": int(entry["n"]),
                    "p": int(entry["p"]),
                }
            ),
            flush=True,
        )
        x_raw, y_raw, _ = load_frozen_task(prepared_root, entry)
        split = split_standardize_three_way(
            x_raw,
            y_raw,
            int(args.split_seed) + SEED_STRIDE * (dataset_index - 1),
        )
        x_train = np.asarray(split["x_train"], dtype=np.float32)
        y_train = np.asarray(split["y_train"], dtype=np.float32)
        x_test = np.asarray(split["x_test"], dtype=np.float32)
        y_test = np.asarray(split["y_test"], dtype=np.float32)
        x_device, y_device = tensors_from_arrays(x_train, y_train, device)
        encoded = encode_dataset(model, x_device, y_device)
        lambda_max = encoded["lambda_max"].float().reshape(-1)
        with torch.inference_mode():
            selected = selector(
                x_device,
                y_device,
                lambda_max,
                return_representation=True,
            )
        selected_raw_lambda = float(selected["raw_lambda"].reshape(-1)[0])
        selected_shape = selected["penalty_shape_features"].reshape(1, -1)
        selected_d0 = selected["penalty_d0"].reshape(-1)
        _, beta_t10, beta_t100, diagnostics = predict_ablation(
            model,
            encoded,
            torch.as_tensor(
                [selected_raw_lambda],
                device=device,
                dtype=torch.float32,
            ),
            selected_shape,
            selected_d0,
            query_chunk=1,
            gamma_factor=1.0,
            ablation_iterations=100,
        )
        t10_mse = float(prediction_mse(x_test, y_test, beta_t10)[0])
        t100_mse = float(prediction_mse(x_test, y_test, beta_t100)[0])
        zero_mse = float(np.mean(np.asarray(y_test, dtype=np.float64) ** 2))

        path_file = paths_root / f"dataset_{dataset_index:02d}" / "paths.npz"
        with np.load(path_file, allow_pickle=False) as payload:
            path = {key: np.asarray(payload[key]) for key in payload.files}
        path_selected = float(path["selected_raw_lambda"])
        if not math.isclose(
            selected_raw_lambda,
            path_selected,
            rel_tol=2e-6,
            abs_tol=1e-8,
        ):
            raise RuntimeError(
                f"Dataset {dataset_index}: selector/path raw lambda mismatch"
            )
        saved_t10_mse = float(path["selected_native_test_mse"])
        if not math.isclose(
            t10_mse,
            saved_t10_mse,
            rel_tol=2e-5,
            abs_tol=2e-7,
        ):
            raise RuntimeError(
                f"Dataset {dataset_index}: native T=10 reproduction failed"
            )
        raw_lambdas = np.asarray(path["raw_lambdas"], dtype=np.float64)
        t10_path_mse = np.asarray(
            path["gamma_eta_lambda_test_mse"], dtype=np.float64
        )
        t100_path_mse = np.asarray(
            path["gamma_eta_test_mse"], dtype=np.float64
        )
        cd_path_mse = np.asarray(
            path["coordinate_descent_test_mse"], dtype=np.float64
        )
        t10_oracle_index = _oracle_index(t10_path_mse, raw_lambdas)
        t100_oracle_index = _oracle_index(t100_path_mse, raw_lambdas)

        folds = list(
            KFold(
                n_splits=int(args.cv_folds),
                shuffle=True,
                random_state=int(args.cv_seed) + dataset_index,
            ).split(x_train)
        )
        lasso_cv = _cv_method(
            "lasso", x_train, y_train, cv_ratios, folds, args
        )
        lasso_fit = _fit_selected(
            "lasso",
            x_train,
            y_train,
            x_test,
            y_test,
            cv_ratios,
            int(lasso_cv["selected_index"]),
            args,
        )
        scad_cv = _cv_method(
            "scad", x_train, y_train, cv_ratios, folds, args
        )
        scad_fit = _fit_selected(
            "scad",
            x_train,
            y_train,
            x_test,
            y_test,
            cv_ratios,
            int(scad_cv["selected_index"]),
            args,
        )

        row = {
            "dataset_index": dataset_index,
            "n": int(entry["n"]),
            "p": int(entry["p"]),
            "task_sha256": str(entry["task_sha256"]),
            "split_seed": int(args.split_seed)
            + SEED_STRIDE * (dataset_index - 1),
            "zero_predictor_test_mse": zero_mse,
            "selected_raw_lambda": selected_raw_lambda,
            "penalty_d0": float(selected_d0[0]),
            "kalbi_t10_test_mse": t10_mse,
            "kalbi_t10_normalized_mse": t10_mse / zero_mse,
            "kalbi_t10_nnz_1e4": int(
                np.count_nonzero(np.abs(beta_t10[0]) > 1e-4)
            ),
            "kalbi_t100_test_mse": t100_mse,
            "kalbi_t100_normalized_mse": t100_mse / zero_mse,
            "kalbi_t100_nnz_1e4": int(
                np.count_nonzero(np.abs(beta_t100[0]) > 1e-4)
            ),
            "kalbi_t10_path_oracle_mse": float(t10_path_mse[t10_oracle_index]),
            "kalbi_t10_path_oracle_raw_lambda": float(
                raw_lambdas[t10_oracle_index]
            ),
            "kalbi_t100_path_oracle_mse": float(
                t100_path_mse[t100_oracle_index]
            ),
            "kalbi_t100_path_oracle_raw_lambda": float(
                raw_lambdas[t100_oracle_index]
            ),
            "cv_lasso_selected_ratio": float(lasso_cv["selected_ratio"]),
            "cv_lasso_test_mse": float(lasso_fit["test_mse"]),
            "cv_lasso_normalized_mse": float(lasso_fit["test_mse"]) / zero_mse,
            "cv_lasso_nnz_1e4": int(lasso_fit["nnz_1e4"]),
            "cv_scad_selected_ratio": float(scad_cv["selected_ratio"]),
            "cv_scad_test_mse": float(scad_fit["test_mse"]),
            "cv_scad_normalized_mse": float(scad_fit["test_mse"]) / zero_mse,
            "cv_scad_nnz_1e4": int(scad_fit["nnz_1e4"]),
            "t100_over_t10_mse": t100_mse / max(t10_mse, 1e-12),
            "native_t10_reproduction_max_abs": float(
                diagnostics["native_reproduction_max_abs"]
            ),
            "t100_last_change_rms": float(
                diagnostics["gamma_eta_change_rms"][0, -1]
            ),
        }
        rows.append(row)
        task_plots.append(
            {
                "dataset_index": dataset_index,
                "n": int(entry["n"]),
                "p": int(entry["p"]),
                "raw_lambdas": raw_lambdas,
                "selected_raw_lambda": selected_raw_lambda,
                "zero_predictor_test_mse": zero_mse,
                "t10_selected_mse": t10_mse,
                "t100_selected_mse": t100_mse,
                "t10_path_mse": t10_path_mse,
                "t100_path_mse": t100_path_mse,
                "cd_path_mse": cd_path_mse,
                "t100_oracle_index": t100_oracle_index,
                "penalty_grid": np.asarray(path["penalty_grid"]),
                "penalty_rho": np.asarray(path["penalty_rho"]),
                "penalty_drho": np.asarray(path["penalty_drho"]),
            }
        )
        detailed.append(
            {
                "summary": row,
                "lasso_cv_mean_mse": lasso_cv["mean_mse"].tolist(),
                "lasso_cv_standard_error": lasso_cv[
                    "standard_error"
                ].tolist(),
                "lasso_cv_fold_diagnostics": lasso_cv["fold_diagnostics"],
                "lasso_final_diagnostics": lasso_fit["diagnostics"],
                "scad_cv_mean_mse": scad_cv["mean_mse"].tolist(),
                "scad_cv_standard_error": scad_cv[
                    "standard_error"
                ].tolist(),
                "scad_cv_fold_diagnostics": scad_cv["fold_diagnostics"],
                "scad_final_diagnostics": scad_fit["diagnostics"],
            }
        )
        print(json.dumps(row, sort_keys=True), flush=True)

    normalized = np.asarray(
        [
            [
                row["kalbi_t10_normalized_mse"],
                row["kalbi_t100_normalized_mse"],
                row["cv_lasso_normalized_mse"],
                row["cv_scad_normalized_mse"],
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    aggregate = {
        method: {
            "mean_normalized_mse": float(np.mean(normalized[:, index])),
            "geometric_mean_normalized_mse": float(
                np.exp(np.mean(np.log(np.maximum(normalized[:, index], 1e-12))))
            ),
            "median_normalized_mse": float(np.median(normalized[:, index])),
            "wins": int(
                np.count_nonzero(np.argmin(normalized, axis=1) == index)
            ),
        }
        for index, method in enumerate(METHODS)
    }
    report = {
        "evaluation": "KALBI T=10 vs T=100 post-hoc unroll and CV baselines",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint.resolve()),
        "checkpoint_step": int(checkpoint["completed_steps"]),
        "contract": {
            "beta_solver_retrained": False,
            "same_selector_raw_lambda_for_t10_t100": True,
            "same_selector_penalty_for_t10_t100": True,
            "refinement_gamma": "eta_t * raw_lambda",
            "t10_steps": 10,
            "t100_steps": 100,
            "cv_folds": int(args.cv_folds),
            "cv_ratio_grid": [
                float(cv_ratios[0]),
                float(cv_ratios[-1]),
                int(cv_ratios.size),
            ],
            "outer_test_used_for_selection": False,
        },
        "aggregate": aggregate,
        "tasks": detailed,
    }
    _write_csv(output_dir / "comparison_metrics.csv", rows)
    (output_dir / "comparison_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_selector_penalty(
        task_plots,
        output_dir / "kalbi_t10_t100_lambda_penalty_mixed6",
    )
    _plot_baseline_comparison(
        rows,
        output_dir / "kalbi_t10_t100_vs_cv_lasso_scad_mixed6",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(output_dir),
                "aggregate": aggregate,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
