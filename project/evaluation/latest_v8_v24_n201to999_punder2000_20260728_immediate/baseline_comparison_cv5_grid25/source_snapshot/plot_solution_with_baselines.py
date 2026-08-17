#!/usr/bin/env python3
"""Add CV baseline test performance to the frozen KALBI path figure."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


COLORS = {
    "CV LASSO": "#E69F00",
    "CV SCAD": "#009E73",
    "CV MCP": "#CC79A7",
}
BASELINES: dict[int, dict] = {}
BASE = None


def load_base(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_plot_latest", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load plotting module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plot_test_with_baselines(
    ax: plt.Axes,
    arrays: dict[str, np.ndarray],
    metric: dict,
    *,
    show_test_annotation: bool,
) -> None:
    baseline = BASELINES[int(metric["dataset_index"])]
    x = np.log10(arrays["raw_lambdas"])
    ax.plot(
        x,
        arrays["kalbi_test_mse_normalized"],
        color=BASE.KALBI_COLOR,
        linewidth=1.6,
    )
    ax.plot(
        x,
        arrays["coordinate_descent_test_mse_normalized"],
        color=BASE.CD_COLOR,
        linewidth=1.35,
        linestyle="--",
    )
    line_styles = {
        "CV LASSO": ":",
        "CV SCAD": "-.",
        "CV MCP": (0, (3, 1, 1, 1)),
    }
    for name, key in (
        ("CV LASSO", "lasso"),
        ("CV SCAD", "scad"),
        ("CV MCP", "mcp"),
    ):
        ax.axhline(
            baseline[key]["normalized_test_mse"],
            color=COLORS[name],
            linewidth=1.05,
            linestyle=line_styles[name],
        )
    ax.axhline(
        1.0,
        color=BASE.BASELINE_COLOR,
        linewidth=0.85,
        linestyle=":",
    )
    BASE.selected_line_or_annotation(
        ax, float(metric["selected_raw_lambda"]), float(x.min()), float(x.max())
    )
    oracle_index = int(arrays["oracle_index"])
    oracle_x = float(x[oracle_index])
    ax.axvline(
        oracle_x,
        color=BASE.ORACLE_COLOR,
        linewidth=1.1,
        linestyle=(0, (2, 1)),
    )
    ax.scatter(
        oracle_x,
        arrays["kalbi_test_mse_normalized"][oracle_index],
        color=BASE.ORACLE_COLOR,
        s=24,
        marker="D",
        edgecolor="white",
        linewidth=0.45,
        zorder=5,
    )
    if show_test_annotation:
        ax.text(
            0.99,
            0.98,
            (
                f"KALBI {baseline['kalbi']['normalized_test_mse']:.3f}\n"
                f"LASSO {baseline['lasso']['normalized_test_mse']:.3f}\n"
                f"SCAD {baseline['scad']['normalized_test_mse']:.3f}\n"
                f"MCP {baseline['mcp']['normalized_test_mse']:.3f}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.2,
            color="#333333",
        )
    ax.set_ylabel("Test MSE / zero-predictor MSE")
    ax.set_xlabel("log10(raw lambda)")


def legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=BASE.KALBI_COLOR, linewidth=1.7, label="KALBI"),
        Line2D(
            [0],
            [0],
            color=BASE.CD_COLOR,
            linewidth=1.4,
            linestyle="--",
            label="Coordinate descent reference",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["CV LASSO"],
            linewidth=1.2,
            linestyle=":",
            label="CV LASSO",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["CV SCAD"],
            linewidth=1.2,
            linestyle="-.",
            label="CV SCAD",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["CV MCP"],
            linewidth=1.2,
            linestyle=(0, (3, 1, 1, 1)),
            label="CV MCP",
        ),
        Line2D(
            [0],
            [0],
            color=BASE.SELECTED_COLOR,
            linewidth=1.2,
            linestyle="-.",
            label="KALBI-selected lambda",
        ),
        Line2D(
            [0],
            [0],
            color=BASE.ORACLE_COLOR,
            linewidth=1.2,
            linestyle=(0, (2, 1)),
            marker="D",
            markersize=4,
            label="Validation-oracle lambda",
        ),
        Line2D(
            [0],
            [0],
            color=BASE.BASELINE_COLOR,
            linewidth=1.0,
            linestyle=":",
            label="Zero-predictor baseline",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    args = parser.parse_args()
    global BASE
    BASE = load_base(args.evaluation_root / "source_snapshot/plot_latest.py")
    for index in range(1, 5):
        BASELINES[index] = json.loads(
            (args.baseline_root / f"dataset_{index}/metrics.json").read_text(
                encoding="utf-8"
            )
        )
    BASE.plot_heldout = plot_test_with_baselines
    BASE.legend_handles = legend_handles
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.6,
            "axes.labelsize": 8.6,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.0,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.75,
            "savefig.transparent": False,
        }
    )
    metrics, result_dir = BASE.load_results(args.evaluation_root)
    arrays = [
        BASE.load_arrays(result_dir, int(metric["dataset_index"]))
        for metric in metrics
    ]
    output = args.baseline_root / "figures_solution_with_baselines"
    detail = output / "detail"
    composite = output / "composite"
    detail.mkdir(parents=True)
    composite.mkdir(parents=True)
    BASE.render_details(metrics, arrays, detail)
    BASE.render_composite(metrics, arrays, composite)
    print(
        json.dumps(
            {
                "status": "PASS",
                "composite_png": str(
                    composite / "kalbi_solution_path_composite.png"
                ),
                "composite_pdf": str(
                    composite / "kalbi_solution_path_composite.pdf"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
