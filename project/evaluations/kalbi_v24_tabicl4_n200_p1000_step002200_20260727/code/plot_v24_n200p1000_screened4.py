#!/usr/bin/env python3
"""Render paper-ready KALBI V24 evaluation figures from frozen result arrays."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
    "#F0E442",
    "#6A3D9A",
    "#8C564B",
    "#17BECF",
    "#7F7F7F",
]


def display_label(value: str) -> str:
    source, index = value.split("_", maxsplit=1)
    source_label = "TabICL" if source.lower() == "tabicl" else source.title()
    return f"{source_label} {index}"


def load_results(root: Path, tag: str) -> tuple[list[dict], Path]:
    result_dir = root / tag
    metrics = json.loads((result_dir / "results.json").read_text())
    if len(metrics) != 4:
        raise RuntimeError(f"expected four datasets, found {len(metrics)}")
    return metrics, result_dir


def task_arrays(result_dir: Path, metric: dict) -> dict[str, np.ndarray]:
    task_dir = (
        result_dir
        / f"dataset_{int(metric['dataset_index']):02d}_{metric['label']}"
    )
    with np.load(task_dir / "paths.npz", allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def plot_coefficient_path(ax: plt.Axes, arrays: dict[str, np.ndarray]) -> None:
    x = np.log10(arrays["raw_lambdas"])
    features = arrays["top_features"].astype(int)
    for order, feature in enumerate(features):
        color = COLORS[order % len(COLORS)]
        ax.plot(
            x,
            arrays["kalbi_beta"][:, feature],
            color=color,
            linewidth=1.35,
        )
        ax.plot(
            x,
            arrays["coordinate_descent_beta"][:, feature],
            color=color,
            linewidth=1.0,
            linestyle="--",
            alpha=0.8,
        )
    ax.axhline(0.0, color="#808080", linewidth=0.6)
    ax.set_ylabel("Coefficient")


def plot_objective(ax: plt.Axes, arrays: dict[str, np.ndarray]) -> None:
    x = np.log10(arrays["raw_lambdas"])
    ax.plot(
        x,
        arrays["kalbi_objective"],
        color="#0072B2",
        linewidth=1.6,
        label="KALBI",
    )
    ax.plot(
        x,
        arrays["coordinate_descent_objective"],
        color="#D55E00",
        linewidth=1.4,
        linestyle="--",
        label="Coordinate descent reference",
    )
    ax.set_ylabel("Training objective")


def plot_selection(
    ax: plt.Axes,
    arrays: dict[str, np.ndarray],
    metric: dict,
) -> None:
    x = np.log10(arrays["raw_lambdas"])
    kalbi = arrays["kalbi_final_r2"]
    reference = arrays["coordinate_descent_final_r2"]
    selected_x = np.log10(float(arrays["selected_raw_lambda"]))
    oracle_index = int(arrays["oracle_index"])
    ax.plot(x, kalbi, color="#0072B2", linewidth=1.6, label="KALBI")
    ax.plot(
        x,
        reference,
        color="#D55E00",
        linewidth=1.3,
        linestyle="--",
        label="Coordinate descent reference",
    )
    if np.isfinite(metric["cv_lasso_final_r2"]):
        ax.axhline(
            metric["cv_lasso_final_r2"],
            color="#009E73",
            linewidth=1.1,
            linestyle=":",
            label="CV LASSO",
        )
    ax.axhline(0.0, color="#777777", linewidth=0.8, linestyle=":")
    ax.axvline(
        selected_x,
        color="#000000",
        linewidth=1.2,
        label="KALBI-selected lambda",
    )
    ax.scatter(
        x[oracle_index],
        kalbi[oracle_index],
        s=32,
        marker="D",
        color="#E69F00",
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
        label="Oracle lambda",
    )
    ax.set_ylabel("Final-test $R^2$")


def plot_penalty(ax: plt.Axes, arrays: dict[str, np.ndarray]) -> None:
    grid = arrays["penalty_grid"]
    keep = grid <= 1.0 + 1e-12
    rho_axis = ax
    derivative_axis = ax.twinx()
    rho_axis.plot(
        grid[keep],
        arrays["penalty_rho"][keep],
        color="#0072B2",
        linewidth=1.7,
        label=r"$\rho(t)$",
    )
    derivative_axis.plot(
        grid[keep],
        arrays["penalty_drho"][keep],
        color="#D55E00",
        linewidth=1.4,
        linestyle="--",
        label=r"$\rho'(t)$",
    )
    rho_axis.set_xlim(0.0, 1.0)
    rho_axis.set_xlabel(r"$|\beta|$")
    rho_axis.set_ylabel(r"$\rho(|\beta|)$")
    derivative_axis.set_ylabel(r"$\rho'(|\beta|)$")


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=0.8, length=3)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_composite(
    metrics: list[dict],
    arrays: list[dict[str, np.ndarray]],
    figure_dir: Path,
) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(14.2, 12.0), constrained_layout=True)
    for row, (metric, values) in enumerate(zip(metrics, arrays, strict=True)):
        plot_coefficient_path(axes[row, 0], values)
        plot_objective(axes[row, 1], values)
        plot_selection(axes[row, 2], values, metric)
        plot_penalty(axes[row, 3], values)
        axes[row, 0].text(
            0.01,
            0.97,
            (
                f"{display_label(metric['label'])}\n"
                f"$N={metric['n']:,}$, $P={metric['p']:,}$"
            ),
            transform=axes[row, 0].transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )
        for column in range(3):
            axes[row, column].set_xlabel(r"$\log_{10}(\lambda_{\rm raw})$")
        for column in range(4):
            style_axis(axes[row, column])
    axes[0, 0].set_title("Coefficient solution path")
    axes[0, 1].set_title("Penalized objective")
    axes[0, 2].set_title("Lambda selection")
    axes[0, 3].set_title("Selected penalty")
    handles, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    save_figure(fig, figure_dir / "kalbi_v24_screened4_composite")


def render_details(
    metrics: list[dict],
    arrays: list[dict[str, np.ndarray]],
    figure_dir: Path,
) -> None:
    for metric, values in zip(metrics, arrays, strict=True):
        fig, axes = plt.subplots(
            2, 2, figsize=(9.0, 6.8), constrained_layout=True
        )
        plot_coefficient_path(axes[0, 0], values)
        plot_objective(axes[0, 1], values)
        plot_selection(axes[1, 0], values, metric)
        plot_penalty(axes[1, 1], values)
        axes[0, 0].set_title(
            (
                f"{display_label(metric['label'])}  |  "
                f"N={metric['n']:,}, P={metric['p']:,}"
            )
        )
        axes[0, 1].set_title("Penalized objective")
        axes[1, 0].set_title("Lambda selection")
        axes[1, 1].set_title("Selected penalty")
        for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
            axis.set_xlabel(r"$\log_{10}(\lambda_{\rm raw})$")
        for axis in axes.reshape(-1):
            style_axis(axis)
        handles, labels = axes[1, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="outside lower center",
            ncol=3,
            frameon=False,
            fontsize=9,
        )
        name = metric["label"].lower().replace(" ", "_")
        save_figure(fig, figure_dir / f"{name}_detail")


def write_baseline_csv(metrics: list[dict], path: Path) -> None:
    fields = [
        "dataset",
        "source",
        "n",
        "p",
        "screening_cv_lasso_r2",
        "final_cv_lasso_r2",
        "kalbi_selected_r2",
        "kalbi_oracle_r2",
        "scad_status",
        "mcp_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(
                {
                    "dataset": display_label(metric["label"]),
                    "source": metric["source"],
                    "n": metric["n"],
                    "p": metric["p"],
                    "screening_cv_lasso_r2": metric[
                        "cv_lasso_screening_r2"
                    ],
                    "final_cv_lasso_r2": metric["cv_lasso_final_r2"],
                    "kalbi_selected_r2": metric["selected_final_r2"],
                    "kalbi_oracle_r2": metric["oracle_final_r2"],
                    "scad_status": "not run: no validated implementation",
                    "mcp_status": "not run: no validated implementation",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--tag", default="full_41point")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    mpl.rcParams.update(
        {
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": False,
        }
    )
    metrics, result_dir = load_results(args.evaluation_root, args.tag)
    arrays = [task_arrays(result_dir, metric) for metric in metrics]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    render_composite(metrics, arrays, args.output_dir)
    render_details(metrics, arrays, args.output_dir)
    write_baseline_csv(metrics, args.output_dir / "baseline_comparison.csv")
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": str(args.output_dir.resolve()),
                "figure_count": 10,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
