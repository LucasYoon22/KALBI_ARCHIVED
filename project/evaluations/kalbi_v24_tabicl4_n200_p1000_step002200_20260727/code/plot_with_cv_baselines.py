#!/usr/bin/env python3
"""Render KALBI paths and CV LASSO/SCAD/MCP comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


METHODS = ("KALBI", "CV LASSO", "CV SCAD", "CV MCP")
METHOD_KEYS = ("kalbi", "lasso", "scad", "mcp")
METHOD_COLORS = {
    "KALBI": "#0072B2",
    "CV LASSO": "#E69F00",
    "CV SCAD": "#009E73",
    "CV MCP": "#CC79A7",
}
COEFFICIENT_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
    "#6A3D9A",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs(
    evaluation_root: Path, baseline_root: Path
) -> tuple[list[dict], list[dict], list[dict[str, np.ndarray]]]:
    kalbi = load_json(evaluation_root / "results.json")
    baselines = load_json(baseline_root / "results.json")
    if len(kalbi) != 4 or len(baselines) != 4:
        raise RuntimeError("expected exactly four evaluation tasks")
    baseline_by_label = {row["label"]: row for row in baselines}
    arrays: list[dict[str, np.ndarray]] = []
    ordered_baselines: list[dict] = []
    for row in kalbi:
        label = row["label"]
        baseline = baseline_by_label[label]
        if (row["seed"], row["n"], row["p"]) != (
            baseline["seed"],
            baseline["n"],
            baseline["p"],
        ):
            raise RuntimeError(f"task contract mismatch for {label}")
        task_dir = (
            evaluation_root
            / f"dataset_{int(row['dataset_index']):02d}_{label}"
        )
        with np.load(task_dir / "paths.npz", allow_pickle=False) as archive:
            arrays.append({key: np.asarray(archive[key]) for key in archive.files})
        ordered_baselines.append(baseline)
    return kalbi, ordered_baselines, arrays


def task_label(index: int) -> str:
    return f"TabICL {index}"


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=0.8, length=3)


def coefficient_path(ax: plt.Axes, values: dict[str, np.ndarray]) -> None:
    x = np.log10(values["raw_lambdas"])
    for order, feature in enumerate(values["top_features"].astype(int)):
        color = COEFFICIENT_COLORS[order % len(COEFFICIENT_COLORS)]
        ax.plot(
            x,
            values["kalbi_beta"][:, feature],
            color=color,
            linewidth=1.25,
        )
        ax.plot(
            x,
            values["coordinate_descent_beta"][:, feature],
            color=color,
            linewidth=0.95,
            linestyle="--",
            alpha=0.78,
        )
    ax.axhline(0.0, color="#777777", linewidth=0.6)
    ax.set_ylabel("Coefficient")


def objective_path(ax: plt.Axes, values: dict[str, np.ndarray]) -> None:
    x = np.log10(values["raw_lambdas"])
    ax.plot(
        x,
        values["kalbi_objective"],
        color=METHOD_COLORS["KALBI"],
        linewidth=1.6,
        label="KALBI",
    )
    ax.plot(
        x,
        values["coordinate_descent_objective"],
        color="#D55E00",
        linewidth=1.3,
        linestyle="--",
        label="Coordinate descent reference",
    )
    ax.set_ylabel("Training objective")


def selection_path(
    ax: plt.Axes,
    values: dict[str, np.ndarray],
    baseline: dict,
) -> None:
    x = np.log10(values["raw_lambdas"])
    ax.plot(
        x,
        values["kalbi_final_r2"],
        color=METHOD_COLORS["KALBI"],
        linewidth=1.6,
        label="KALBI",
    )
    ax.plot(
        x,
        values["coordinate_descent_final_r2"],
        color="#D55E00",
        linewidth=1.25,
        linestyle="--",
        label="Coordinate descent reference",
    )
    line_styles = {"CV LASSO": ":", "CV SCAD": "-.", "CV MCP": (0, (3, 1, 1, 1))}
    for method, key in zip(METHODS[1:], METHOD_KEYS[1:], strict=True):
        ax.axhline(
            baseline[key]["final_r2"],
            color=METHOD_COLORS[method],
            linewidth=1.15,
            linestyle=line_styles[method],
            label=method,
        )
    selected_x = np.log10(float(values["selected_raw_lambda"]))
    oracle_index = int(values["oracle_index"])
    ax.axvline(
        selected_x,
        color="#111111",
        linewidth=1.1,
        label="KALBI-selected lambda",
    )
    ax.scatter(
        x[oracle_index],
        values["kalbi_final_r2"][oracle_index],
        color="#56B4E9",
        edgecolor="#111111",
        linewidth=0.4,
        marker="D",
        s=28,
        zorder=5,
        label="KALBI oracle",
    )
    ax.axhline(0.0, color="#888888", linewidth=0.7, linestyle=":")
    ax.set_ylabel("Final-test $R^2$")


def penalty_panel(ax: plt.Axes, values: dict[str, np.ndarray]) -> None:
    grid = values["penalty_grid"]
    keep = grid <= 1.0 + 1e-12
    derivative = ax.twinx()
    ax.plot(
        grid[keep],
        values["penalty_rho"][keep],
        color=METHOD_COLORS["KALBI"],
        linewidth=1.65,
        label=r"$\rho(t)$",
    )
    derivative.plot(
        grid[keep],
        values["penalty_drho"][keep],
        color="#D55E00",
        linewidth=1.3,
        linestyle="--",
        label=r"$\rho'(t)$",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r"$|\beta|$")
    ax.set_ylabel(r"$\rho(|\beta|)$")
    derivative.set_ylabel(r"$\rho'(|\beta|)$")
    derivative.spines["top"].set_visible(False)


def save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_composite(
    kalbi: list[dict],
    baselines: list[dict],
    arrays: list[dict[str, np.ndarray]],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(14.6, 12.2), constrained_layout=True)
    for row, (metric, baseline, values) in enumerate(
        zip(kalbi, baselines, arrays, strict=True)
    ):
        coefficient_path(axes[row, 0], values)
        objective_path(axes[row, 1], values)
        selection_path(axes[row, 2], values, baseline)
        penalty_panel(axes[row, 3], values)
        axes[row, 0].text(
            0.015,
            0.975,
            f"{task_label(row + 1)}\n$N={metric['n']:,}$, $P={metric['p']:,}$",
            transform=axes[row, 0].transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
        )
        for column in range(3):
            axes[row, column].set_xlabel(r"$\log_{10}(\lambda_{\rm raw})$")
        for column in range(4):
            style_axis(axes[row, column])
    axes[0, 0].set_title("Coefficient solution path")
    axes[0, 1].set_title("Penalized objective")
    axes[0, 2].set_title("Lambda selection and CV baselines")
    axes[0, 3].set_title("KALBI-selected penalty")
    handles, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=4,
        frameon=False,
        fontsize=8.8,
    )
    save(
        fig,
        output_dir / "kalbi_tabicl4_solution_path_lambda_selection_with_baselines",
    )


def collect_comparison_rows(
    kalbi: list[dict],
    baselines: list[dict],
    arrays: list[dict[str, np.ndarray]],
) -> list[dict]:
    rows: list[dict] = []
    for index, (metric, baseline, values) in enumerate(
        zip(kalbi, baselines, arrays, strict=True), start=1
    ):
        entries = {
            "KALBI": {
                "r2": metric["selected_final_r2"],
                "mse": metric["selected_final_mse"],
                "nnz": int(np.count_nonzero(np.abs(values["selected_beta"]) > 1e-4)),
                "selected_ratio": np.nan,
            },
            "CV LASSO": {
                "r2": baseline["lasso"]["final_r2"],
                "mse": baseline["lasso"]["final_mse"],
                "nnz": baseline["lasso"]["nnz_1e4"],
                "selected_ratio": baseline["lasso"]["selected_ratio"],
            },
            "CV SCAD": {
                "r2": baseline["scad"]["final_r2"],
                "mse": baseline["scad"]["final_mse"],
                "nnz": baseline["scad"]["nnz_1e4"],
                "selected_ratio": baseline["scad"]["selected_ratio"],
            },
            "CV MCP": {
                "r2": baseline["mcp"]["final_r2"],
                "mse": baseline["mcp"]["final_mse"],
                "nnz": baseline["mcp"]["nnz_1e4"],
                "selected_ratio": baseline["mcp"]["selected_ratio"],
            },
        }
        for method, values_for_method in entries.items():
            rows.append(
                {
                    "dataset": task_label(index),
                    "seed": metric["seed"],
                    "n": metric["n"],
                    "p": metric["p"],
                    "method": method,
                    **values_for_method,
                }
            )
    return rows


def render_summary(rows: list[dict], output_dir: Path) -> None:
    datasets = [task_label(index) for index in range(1, 5)]
    x = np.arange(len(datasets))
    width = 0.19
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    for method_index, method in enumerate(METHODS):
        method_rows = [row for row in rows if row["method"] == method]
        offset = (method_index - 1.5) * width
        axes[0].bar(
            x + offset,
            [row["r2"] for row in method_rows],
            width,
            color=METHOD_COLORS[method],
            label=method,
        )
        axes[1].bar(
            x + offset,
            [row["nnz"] for row in method_rows],
            width,
            color=METHOD_COLORS[method],
            label=method,
        )
    axes[0].axhline(0.0, color="#777777", linewidth=0.7)
    axes[0].set_ylabel("Final-test $R^2$")
    axes[0].set_title("Prediction performance")
    axes[1].set_ylabel(r"Nonzero coefficients ($|\beta|>10^{-4}$)")
    axes[1].set_title("Selected model size")
    for ax in axes:
        ax.set_xticks(x, datasets)
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    save(fig, output_dir / "kalbi_vs_cv_lasso_scad_mcp_tabicl4")


def write_comparison(rows: list[dict], output_dir: Path) -> None:
    with (output_dir / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "comparison.json").write_text(
        json.dumps(rows, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    mpl.rcParams.update(
        {
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "legend.fontsize": 9,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": False,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    kalbi, baselines, arrays = load_inputs(
        args.evaluation_root, args.baseline_root
    )
    render_composite(kalbi, baselines, arrays, args.output_dir)
    rows = collect_comparison_rows(kalbi, baselines, arrays)
    render_summary(rows, args.output_dir)
    write_comparison(rows, args.output_dir)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": str(args.output_dir.resolve()),
                "tasks": len(kalbi),
                "methods": list(METHODS),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
