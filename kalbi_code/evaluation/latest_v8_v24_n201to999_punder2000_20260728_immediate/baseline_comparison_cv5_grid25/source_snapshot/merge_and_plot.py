#!/usr/bin/env python3
"""Validate, merge, and plot frozen KALBI versus CV baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


METHODS = ("KALBI", "CV LASSO", "CV SCAD", "CV MCP")
KEYS = ("kalbi", "lasso", "scad", "mcp")
COLORS = {
    "KALBI": "#0072B2",
    "CV LASSO": "#E69F00",
    "CV SCAD": "#009E73",
    "CV MCP": "#CC79A7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=0.8, length=3)
    ax.grid(axis="y", color="#d9dde3", linewidth=0.55, alpha=0.65)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    baseline_root = args.baseline_root.resolve()
    figure_root = baseline_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=False)
    rows = [
        load_json(baseline_root / f"dataset_{index}/metrics.json")
        for index in range(1, 5)
    ]
    for index, row in enumerate(rows, start=1):
        if row["status"] != "PASS" or row["dataset_index"] != index:
            raise RuntimeError(f"dataset {index} did not pass")
        if any(
            row[key]["mse_recompute_error"] > 1e-12
            for key in ("lasso", "scad", "mcp")
        ):
            raise RuntimeError(f"dataset {index} baseline MSE audit failed")
        if row["kalbi"]["saved_test_mse_recompute_error"] > 2e-7:
            raise RuntimeError(f"dataset {index} KALBI MSE audit failed")

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.2,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )
    dataset_labels = [f"Dataset {index}" for index in range(1, 5)]
    x = np.arange(4)
    width = 0.19

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.45), constrained_layout=True)
    for method_index, (method, key) in enumerate(zip(METHODS, KEYS, strict=True)):
        offset = (method_index - 1.5) * width
        normalized = [row[key]["normalized_test_mse"] for row in rows]
        r2 = [row[key]["test_r2"] for row in rows]
        nnz = [row[key]["nnz_1e4"] for row in rows]
        axes[0].bar(x + offset, normalized, width, color=COLORS[method], label=method)
        axes[1].bar(x + offset, r2, width, color=COLORS[method], label=method)
        axes[2].bar(x + offset, nnz, width, color=COLORS[method], label=method)
    axes[0].axhline(1.0, color="#555555", linestyle=":", linewidth=1.0)
    axes[0].set_title("Prediction error")
    axes[0].set_ylabel("Test MSE / zero-predictor MSE")
    axes[1].axhline(0.0, color="#555555", linestyle=":", linewidth=1.0)
    axes[1].set_title("Prediction performance")
    axes[1].set_ylabel(r"Test $R^2$")
    axes[2].set_title("Selected model size")
    axes[2].set_ylabel(r"Nonzero coefficients ($|\beta|>10^{-4}$)")
    for ax in axes:
        ax.set_xticks(x, dataset_labels, rotation=18, ha="right")
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    save(fig, figure_root / "kalbi_vs_cv_lasso_scad_mcp_summary")

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), constrained_layout=True)
    for index, (ax, row) in enumerate(zip(axes.flat, rows, strict=True), start=1):
        values = [row[key]["normalized_test_mse"] for key in KEYS]
        bars = ax.bar(
            np.arange(4),
            values,
            color=[COLORS[method] for method in METHODS],
            width=0.68,
        )
        ax.axhline(1.0, color="#555555", linestyle=":", linewidth=1.0)
        ax.set_xticks(np.arange(4), METHODS, rotation=18, ha="right")
        ax.set_ylabel("Test MSE / zero-predictor MSE")
        ax.set_title(
            f"Dataset {index}: {row['outcome_branch']}, "
            f"N={row['n']:,}, P={row['p']:,}"
        )
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.8,
            )
        style_axis(ax)
    save(fig, figure_root / "kalbi_vs_cv_lasso_scad_mcp_by_dataset")

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.2), constrained_layout=True)
    for index, (ax, row) in enumerate(zip(axes.flat, rows, strict=True), start=1):
        paths = np.load(
            baseline_root / f"dataset_{index}/cv_paths.npz",
            allow_pickle=False,
        )
        ratios = paths["lambda_ratio_grid"]
        x_grid = np.log10(ratios)
        for method, key in zip(METHODS[1:], KEYS[1:], strict=True):
            mean = paths[f"{key}_mean_mse"]
            standard_error = paths[f"{key}_standard_error"]
            ax.plot(x_grid, mean, color=COLORS[method], linewidth=1.6, label=method)
            ax.fill_between(
                x_grid,
                mean - standard_error,
                mean + standard_error,
                color=COLORS[method],
                alpha=0.13,
                linewidth=0,
            )
            ax.axvline(
                np.log10(row[key]["selected_ratio"]),
                color=COLORS[method],
                linewidth=0.85,
                linestyle=":",
            )
        ax.set_title(
            f"Dataset {index}: {row['outcome_branch']}, "
            f"N={row['n']:,}, P={row['p']:,}"
        )
        ax.set_xlabel(r"$\log_{10}(\lambda/\lambda_{\max})$")
        ax.set_ylabel("Fold-validation MSE")
        style_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    save(fig, figure_root / "cv_baseline_paths")

    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        for method, key in zip(METHODS, KEYS, strict=True):
            csv_rows.append(
                {
                    "dataset": row["dataset_index"],
                    "branch": row["outcome_branch"],
                    "n": row["n"],
                    "p": row["p"],
                    "method": method,
                    "test_mse": row[key]["test_mse"],
                    "normalized_test_mse": row[key]["normalized_test_mse"],
                    "test_r2": row[key]["test_r2"],
                    "nnz_1e4": row[key]["nnz_1e4"],
                    "selected_lambda_ratio": (
                        ""
                        if key == "kalbi"
                        else row[key]["selected_ratio"]
                    ),
                    "selected_raw_lambda": (
                        row[key]["selected_raw_lambda"]
                        if key == "kalbi"
                        else ""
                    ),
                }
            )
    with (baseline_root / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    write_json(baseline_root / "comparison.json", csv_rows)

    geometric_means = {}
    for method, key in zip(METHODS, KEYS, strict=True):
        normalized = np.asarray(
            [row[key]["normalized_test_mse"] for row in rows], dtype=np.float64
        )
        geometric_means[method] = float(np.exp(np.mean(np.log(normalized))))
    wins = {
        method: 0
        for method in METHODS
    }
    for row in rows:
        best = min(
            zip(METHODS, KEYS, strict=True),
            key=lambda pair: row[pair[1]]["test_mse"],
        )[0]
        wins[best] += 1
    manifest = {
        "status": "PASS",
        "task_count": 4,
        "methods": list(METHODS),
        "geometric_mean_normalized_test_mse": geometric_means,
        "test_mse_wins": wins,
        "task_metric_sha256": {
            str(index): sha256(baseline_root / f"dataset_{index}/metrics.json")
            for index in range(1, 5)
        },
        "figures": {
            path.name: sha256(path)
            for path in sorted(figure_root.iterdir())
            if path.is_file()
        },
        "source_evaluation_root": str(args.evaluation_root.resolve()),
        "source_evaluation_results_sha256": sha256(
            args.evaluation_root.resolve()
            / "metrics/full_41point/results.json"
        ),
    }
    write_json(baseline_root / "validation_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
