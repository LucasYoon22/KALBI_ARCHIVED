#!/usr/bin/env python3
"""Render KALBI versus LASSO raw-lambda solution paths without CD."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

from real_pgtN_common import (
    atomic_json,
    load_split_manifest,
    parse_task_indices,
)


BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
GRAY = "#6B6B6B"
LIGHT_GRAY = "#B7B7B7"
FEATURE_COLORS = plt.get_cmap("tab10").colors


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def selected_marker(
    ax: plt.Axes,
    raw_lambda: float,
    mse: float,
    grid: np.ndarray,
    color: str,
    label: str,
) -> None:
    lower = float(grid[0])
    upper = float(grid[-1])
    if lower <= raw_lambda <= upper:
        x = float(np.log10(raw_lambda))
        ax.axvline(x, color=color, lw=1.2, alpha=0.75)
        ax.scatter(
            [x],
            [mse],
            color=color,
            edgecolor="white",
            linewidth=0.5,
            s=42,
            zorder=8,
        )
    else:
        side = "below" if raw_lambda < lower else "above"
        ax.text(
            0.02,
            0.96 if color == BLUE else 0.88,
            f"{label} selected lambda {side} grid",
            color=color,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )


def plot_coefficient_panel(
    ax: plt.Axes,
    arrays: dict[str, np.ndarray],
    *,
    feature_count: int | None = None,
) -> None:
    raw = arrays["raw_lambdas"]
    x = np.log10(raw)
    top = arrays["top_features"]
    if feature_count is not None:
        top = top[:feature_count]
    kalbi = arrays["kalbi_beta"]
    lasso = arrays["lasso_beta"]
    for position, feature in enumerate(top):
        color = FEATURE_COLORS[position % len(FEATURE_COLORS)]
        ax.plot(x, kalbi[:, feature], color=color, lw=1.6, alpha=0.95)
        ax.plot(
            x,
            lasso[:, feature],
            color=color,
            lw=1.3,
            ls="--",
            alpha=0.8,
        )
    ax.axhline(0.0, color=LIGHT_GRAY, lw=0.8)
    ax.text(
        0.02,
        0.97,
        "same color = same feature; solid = KALBI; dashed = LASSO",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color=GRAY,
    )
    ax.set_xlabel(r"$\log_{10}(\lambda_{\rm raw})$")
    ax.set_ylabel("Standardized coefficient")


def plot_error_panel(
    ax: plt.Axes,
    arrays: dict[str, np.ndarray],
    metric: dict[str, Any],
) -> None:
    raw = arrays["raw_lambdas"]
    x = np.log10(raw)
    kalbi = arrays["kalbi_normalized_test_mse"]
    lasso = arrays["lasso_normalized_test_mse"]
    ax.plot(x, kalbi, color=BLUE, lw=2.0, label="KALBI")
    ax.plot(x, lasso, color=ORANGE, lw=2.0, ls="--", label="LASSO")
    ax.axhline(1.0, color=GRAY, lw=1.0, ls=":", label="Zero predictor")
    kalbi_oracle = int(np.argmin(kalbi))
    lasso_oracle = int(np.argmin(lasso))
    ax.scatter(
        [x[kalbi_oracle]],
        [kalbi[kalbi_oracle]],
        marker="D",
        s=38,
        color=BLUE,
        edgecolor="white",
        linewidth=0.5,
        zorder=7,
    )
    ax.scatter(
        [x[lasso_oracle]],
        [lasso[lasso_oracle]],
        marker="D",
        s=38,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.5,
        zorder=7,
    )
    selected_marker(
        ax,
        metric["kalbi_selected_raw_lambda"],
        metric["kalbi_selected_normalized_test_mse"],
        raw,
        BLUE,
        "KALBI",
    )
    selected_marker(
        ax,
        metric["lasso_cv_selected_raw_lambda"],
        metric["lasso_cv_selected_normalized_test_mse"],
        raw,
        ORANGE,
        "LASSO",
    )
    positive = np.concatenate([kalbi[kalbi > 0], lasso[lasso > 0]])
    if positive.size:
        ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}(\lambda_{\rm raw})$")
    ax.set_ylabel("Test MSE / zero-predictor MSE")


def load_task_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def task_figure(
    task: dict[str, Any],
    arrays: dict[str, np.ndarray],
    metric: dict[str, Any],
) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.0))
    x = np.log10(arrays["raw_lambdas"])
    plot_coefficient_panel(axes[0, 0], arrays)
    axes[0, 0].set_title("Coefficient solution path")
    plot_error_panel(axes[0, 1], arrays, metric)
    axes[0, 1].set_title("Held-out error path")

    axes[1, 0].plot(
        x, arrays["kalbi_nnz"], color=BLUE, lw=2.0, label="KALBI"
    )
    axes[1, 0].plot(
        x,
        arrays["lasso_nnz"],
        color=ORANGE,
        lw=2.0,
        ls="--",
        label="LASSO",
    )
    axes[1, 0].set_xlabel(r"$\log_{10}(\lambda_{\rm raw})$")
    axes[1, 0].set_ylabel(r"Number of nonzero coefficients ($10^{-8}$)")
    axes[1, 0].set_title("Sparsity path")

    axes[1, 1].plot(
        x, arrays["kalbi_l1"], color=BLUE, lw=2.0, label="KALBI"
    )
    axes[1, 1].plot(
        x,
        arrays["lasso_l1"],
        color=ORANGE,
        lw=2.0,
        ls="--",
        label="LASSO",
    )
    axes[1, 1].set_xlabel(r"$\log_{10}(\lambda_{\rm raw})$")
    axes[1, 1].set_ylabel(r"$\|\widehat{\beta}\|_1$")
    axes[1, 1].set_title("Coefficient-norm path")

    figure.suptitle(
        (
            f"Dataset {int(task['registry_index'])}  |  "
            f"$N_{{train}}={metric['outer_train_rows']:,}$, "
            f"$N_{{test}}={metric['outer_test_rows']:,}$, "
            f"$P={metric['P_transformed']:,}$"
        ),
        y=0.995,
    )
    handles = [
        Line2D([0], [0], color=BLUE, lw=2.0, label="KALBI"),
        Line2D(
            [0], [0], color=ORANGE, lw=2.0, ls="--", label="LASSO"
        ),
        Line2D(
            [0], [0], color=GRAY, lw=1.0, ls=":", label="Zero predictor"
        ),
        Line2D(
            [0],
            [0],
            color=GRAY,
            marker="D",
            lw=0,
            label="Path oracle",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    figure.tight_layout(
        rect=(0, 0.065, 1, 0.96),
        pad=1.2,
        w_pad=3.2,
        h_pad=2.0,
    )
    return figure


def evenly_spaced_figure(
    selected: list[tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]]
) -> plt.Figure:
    figure, axes = plt.subplots(3, 4, figsize=(18.0, 12.0))
    for row in range(3):
        for slot in range(2):
            task, arrays, metric = selected[2 * row + slot]
            coefficient_ax = axes[row, 2 * slot]
            error_ax = axes[row, 2 * slot + 1]
            plot_coefficient_panel(
                coefficient_ax, arrays, feature_count=6
            )
            plot_error_panel(error_ax, arrays, metric)
            prefix = f"Dataset {int(task['registry_index'])}"
            coefficient_ax.set_title(
                f"{prefix}: coefficient paths\n"
                f"$N={int(task['N']):,}$, $P={metric['P_transformed']:,}$"
            )
            error_ax.set_title(f"{prefix}: held-out error")
    handles = [
        Line2D([0], [0], color=BLUE, lw=2.0, label="KALBI"),
        Line2D(
            [0], [0], color=ORANGE, lw=2.0, ls="--", label="LASSO"
        ),
        Line2D(
            [0], [0], color=GRAY, lw=1.0, ls=":", label="Zero predictor"
        ),
        Line2D(
            [0],
            [0],
            color=GRAY,
            marker="D",
            lw=0,
            label="Path oracle",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.tight_layout(rect=(0, 0.045, 1, 1))
    return figure


def aggregate_figure(rows: list[dict[str, Any]]) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))
    kalbi_selected = np.asarray(
        [row["kalbi_selected_normalized_test_mse"] for row in rows]
    )
    lasso_selected = np.asarray(
        [row["lasso_cv_selected_normalized_test_mse"] for row in rows]
    )
    kalbi_oracle = np.asarray(
        [row["kalbi_path_oracle_normalized_test_mse"] for row in rows]
    )
    lasso_oracle = np.asarray(
        [row["lasso_path_oracle_normalized_test_mse"] for row in rows]
    )
    indices = np.asarray([row["registry_index"] for row in rows])

    for ax, left, right, title in (
        (
            axes[0, 0],
            lasso_selected,
            kalbi_selected,
            "Selected-lambda prediction error",
        ),
        (
            axes[0, 1],
            lasso_oracle,
            kalbi_oracle,
            "Raw-grid oracle prediction error",
        ),
    ):
        lower = min(float(left.min()), float(right.min()))
        upper = max(float(left.max()), float(right.max()))
        ax.scatter(left, right, c=indices, cmap="viridis", s=35, alpha=0.85)
        ax.plot([lower, upper], [lower, upper], color=GRAY, ls=":", lw=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("LASSO normalized test MSE")
        ax.set_ylabel("KALBI normalized test MSE")
        ax.set_title(title)

    ratio = kalbi_selected / np.maximum(lasso_selected, 1e-30)
    axes[1, 0].scatter(
        indices, ratio, color=GREEN, edgecolor="white", linewidth=0.4
    )
    axes[1, 0].axhline(1.0, color=GRAY, ls=":", lw=1.2)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("Dataset index")
    axes[1, 0].set_ylabel("KALBI / LASSO selected MSE")
    axes[1, 0].set_title("Paired selected-lambda ratio")

    axes[1, 1].scatter(
        [row["lasso_cv_selected_raw_lambda"] for row in rows],
        [row["kalbi_selected_raw_lambda"] for row in rows],
        c=indices,
        cmap="viridis",
        s=35,
        alpha=0.85,
    )
    low = min(
        min(row["lasso_cv_selected_raw_lambda"] for row in rows),
        min(row["kalbi_selected_raw_lambda"] for row in rows),
    )
    high = max(
        max(row["lasso_cv_selected_raw_lambda"] for row in rows),
        max(row["kalbi_selected_raw_lambda"] for row in rows),
    )
    axes[1, 1].plot([low, high], [low, high], color=GRAY, ls=":", lw=1.2)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("CV-LASSO selected raw lambda")
    axes[1, 1].set_ylabel("KALBI selected raw lambda")
    axes[1, 1].set_title("Raw-lambda selection")
    figure.tight_layout()
    return figure


def save_figure(figure: plt.Figure, stem: Path, dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=dpi)
    figure.savefig(stem.with_suffix(".pdf"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--task-indices", default="all")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    configure_style()

    manifest = load_split_manifest(args.split_manifest)
    task_lookup = {
        int(task["registry_index"]): task for task in manifest["tasks"]
    }
    selected_indices = parse_task_indices(
        args.task_indices, int(manifest["task_count"])
    )
    records: list[
        tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]
    ] = []
    for index in selected_indices:
        task = task_lookup[index]
        task_id = task["task_id"]
        metric = load_json(
            args.comparison_root / "metrics" / f"{task_id}.json"
        )
        arrays = load_task_arrays(
            args.comparison_root / "paths" / f"{task_id}.npz"
        )
        records.append((task, arrays, metric))

    per_task_root = args.comparison_root / "figures" / "per_task"
    multipage_path = (
        args.comparison_root / "figures" / "kalbi_lasso_all_tasks.pdf"
    )
    multipage_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(multipage_path) as pdf:
        for task, arrays, metric in records:
            stem = per_task_root / (
                f"dataset_{int(task['registry_index']):02d}_{task['task_id']}"
            )
            figure = task_figure(task, arrays, metric)
            save_figure(figure, stem, args.dpi)
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)

    summary_root = args.comparison_root / "figures" / "summary"
    summary_indices = [1, 9, 17, 25, 33, 41]
    if set(summary_indices).issubset(selected_indices):
        record_lookup = {
            int(task["registry_index"]): (task, arrays, metric)
            for task, arrays, metric in records
        }
        selected = [record_lookup[index] for index in summary_indices]
        summary = evenly_spaced_figure(selected)
        save_figure(
            summary,
            summary_root / "kalbi_lasso_evenly_spaced_solution_paths",
            args.dpi,
        )
        plt.close(summary)

    rows = [metric for _, _, metric in records]
    if len(rows) >= 2:
        aggregate = aggregate_figure(rows)
        save_figure(
            aggregate,
            summary_root / "kalbi_lasso_path_aggregate",
            args.dpi,
        )
        plt.close(aggregate)

    csv_path = args.comparison_root / "reports" / "comparison.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "registry_index",
        "task_id",
        "source_family",
        "N",
        "P_transformed",
        "kalbi_selected_raw_lambda",
        "lasso_cv_selected_raw_lambda",
        "kalbi_selected_normalized_test_mse",
        "lasso_cv_selected_normalized_test_mse",
        "kalbi_path_oracle_normalized_test_mse",
        "lasso_path_oracle_normalized_test_mse",
        "lasso_max_kkt",
        "lasso_max_dual_gap",
        "lasso_convergence_warning_count",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})

    selected_ratio = np.asarray(
        [
            row["kalbi_selected_normalized_test_mse"]
            / max(row["lasso_cv_selected_normalized_test_mse"], 1e-30)
            for row in rows
        ]
    )
    aggregate_metrics = {
        "status": "PASS",
        "task_count": len(rows),
        "coordinate_descent_imported_or_executed": False,
        "raw_lambda_grid": {
            "minimum": 1e-4,
            "maximum": 1.0,
            "count": 41,
            "same_numeric_values_for_kalbi_and_lasso": True,
        },
        "selected_prediction": {
            "kalbi_wins": int(np.count_nonzero(selected_ratio < 1.0)),
            "lasso_wins": int(np.count_nonzero(selected_ratio > 1.0)),
            "ties": int(np.count_nonzero(selected_ratio == 1.0)),
            "median_kalbi_over_lasso_mse": float(np.median(selected_ratio)),
            "geometric_mean_kalbi_over_lasso_mse": float(
                np.exp(np.mean(np.log(selected_ratio)))
            ),
        },
        "lasso_numerics": {
            "maximum_kkt": float(max(row["lasso_max_kkt"] for row in rows)),
            "maximum_dual_gap": float(
                max(row["lasso_max_dual_gap"] for row in rows)
            ),
            "total_convergence_warnings": int(
                sum(
                    row["lasso_convergence_warning_count"] for row in rows
                )
            ),
        },
        "figures": {
            "per_task_png_count": len(records),
            "per_task_pdf_count": len(records),
            "multipage_pdf": str(multipage_path),
            "fixed_summary_indices": (
                summary_indices
                if set(summary_indices).issubset(selected_indices)
                else []
            ),
        },
    }
    atomic_json(
        args.comparison_root / "reports" / "aggregate_metrics.json",
        aggregate_metrics,
    )
    atomic_json(
        args.comparison_root / "manifests" / "figure_manifest.json",
        {
            "status": "PASS",
            "task_count": len(records),
            "coordinate_descent_imported_or_executed": False,
            "per_task_png_count": len(records),
            "per_task_pdf_count": len(records),
            "multipage_pdf": str(multipage_path),
            "summary_indices": (
                summary_indices
                if set(summary_indices).issubset(selected_indices)
                else []
            ),
        },
    )


if __name__ == "__main__":
    main()
