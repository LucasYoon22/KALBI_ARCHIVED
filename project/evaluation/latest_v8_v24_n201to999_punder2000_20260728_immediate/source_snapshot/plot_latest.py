#!/usr/bin/env python3
"""Render publication figures using only the visible name KALBI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


FEATURE_COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#332288",
    "#88CCEE",
    "#AA4499",
    "#44AA99",
    "#999933",
    "#882255",
]
KALBI_COLOR = "#0072B2"
CD_COLOR = "#D55E00"
SELECTED_COLOR = "#332288"
ORACLE_COLOR = "#E69F00"
BASELINE_COLOR = "#555555"


def load_results(root: Path) -> tuple[list[dict], Path]:
    result_dir = root / "metrics" / "full_41point"
    metrics = json.loads((result_dir / "results.json").read_text())
    if len(metrics) != 4:
        raise RuntimeError("expected four tasks")
    return metrics, result_dir


def load_arrays(result_dir: Path, index: int) -> dict[str, np.ndarray]:
    with np.load(
        result_dir / f"dataset_{index:02d}" / "paths.npz",
        allow_pickle=False,
    ) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.tick_params(direction="out", width=0.75, length=3, colors="#333333")
    ax.grid(False)


def plot_coefficient_path(ax: plt.Axes, arrays: dict[str, np.ndarray]) -> None:
    x = np.log10(arrays["raw_lambdas"])
    for order, feature in enumerate(arrays["top_features"].astype(int)):
        color = FEATURE_COLORS[order]
        ax.plot(
            x,
            arrays["kalbi_beta"][:, feature],
            color=color,
            linewidth=1.25,
            solid_capstyle="round",
        )
        ax.plot(
            x,
            arrays["coordinate_descent_beta"][:, feature],
            color=color,
            linewidth=0.95,
            linestyle="--",
            alpha=0.82,
        )
    ax.axhline(0.0, color="#999999", linewidth=0.65)
    ax.set_ylabel("Coefficient")
    ax.set_xlabel("log10(raw lambda)")


def plot_objective(ax: plt.Axes, arrays: dict[str, np.ndarray]) -> None:
    x = np.log10(arrays["raw_lambdas"])
    ax.plot(
        x,
        arrays["kalbi_objective"],
        color=KALBI_COLOR,
        linewidth=1.6,
    )
    ax.plot(
        x,
        arrays["coordinate_descent_objective"],
        color=CD_COLOR,
        linewidth=1.35,
        linestyle="--",
    )
    ax.set_ylabel("Training objective")
    ax.set_xlabel("log10(raw lambda)")


def selected_line_or_annotation(
    ax: plt.Axes, selected_raw_lambda: float, x_min: float, x_max: float
) -> None:
    selected_x = float(np.log10(selected_raw_lambda))
    if x_min <= selected_x <= x_max:
        ax.axvline(
            selected_x,
            color=SELECTED_COLOR,
            linewidth=1.15,
            linestyle="-.",
        )
        return
    side = x_min if selected_x < x_min else x_max
    direction = "<" if selected_x < x_min else ">"
    ax.axvline(
        side,
        color=SELECTED_COLOR,
        linewidth=1.15,
        linestyle="-.",
    )
    ax.annotate(
        f"selected {direction} range\n{selected_raw_lambda:.3g}",
        xy=(side, 0.96),
        xycoords=("data", "axes fraction"),
        xytext=(4 if side == x_min else -4, -2),
        textcoords="offset points",
        ha="left" if side == x_min else "right",
        va="top",
        fontsize=7.2,
        color=SELECTED_COLOR,
    )


def plot_heldout(
    ax: plt.Axes,
    arrays: dict[str, np.ndarray],
    metric: dict,
    *,
    show_test_annotation: bool,
) -> None:
    x = np.log10(arrays["raw_lambdas"])
    ax.plot(
        x,
        arrays["kalbi_validation_mse_normalized"],
        color=KALBI_COLOR,
        linewidth=1.6,
    )
    ax.plot(
        x,
        arrays["coordinate_descent_validation_mse_normalized"],
        color=CD_COLOR,
        linewidth=1.35,
        linestyle="--",
    )
    ax.axhline(
        1.0,
        color=BASELINE_COLOR,
        linewidth=0.9,
        linestyle=":",
    )
    selected_line_or_annotation(
        ax, float(metric["selected_raw_lambda"]), float(x.min()), float(x.max())
    )
    oracle_index = int(arrays["oracle_index"])
    oracle_x = float(x[oracle_index])
    ax.axvline(
        oracle_x,
        color=ORACLE_COLOR,
        linewidth=1.15,
        linestyle=(0, (2, 1)),
    )
    ax.scatter(
        oracle_x,
        arrays["kalbi_validation_mse_normalized"][oracle_index],
        color=ORACLE_COLOR,
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
                "Test MSE\n"
                f"selected {metric['selected_test_mse']:.3g}\n"
                f"oracle {metric['oracle_test_mse']:.3g}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.4,
            color="#333333",
        )
    ax.set_ylabel("Validation MSE / zero-predictor MSE")
    ax.set_xlabel("log10(raw lambda)")


def plot_penalty(ax: plt.Axes, arrays: dict[str, np.ndarray]) -> None:
    grid = arrays["penalty_grid"]
    keep = grid <= 1.0 + 1e-12
    ax.plot(
        grid[keep],
        arrays["penalty_rho"][keep],
        color=KALBI_COLOR,
        linewidth=1.7,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("|coefficient|")
    ax.set_ylabel("Selected penalty")
    ax.add_patch(
        Rectangle(
            (0.535, 0.545),
            0.465,
            0.455,
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="none",
            zorder=4,
            clip_on=False,
        )
    )
    inset = inset_axes(ax, width="43%", height="39%", loc="upper right", borderpad=0.8)
    inset.set_facecolor("white")
    inset.set_zorder(5)
    inset.add_patch(
        Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=inset.transAxes,
            facecolor="white",
            edgecolor="none",
            zorder=0,
            clip_on=False,
        )
    )
    inset.plot(
        grid[keep],
        arrays["penalty_drho"][keep],
        color=CD_COLOR,
        linewidth=1.1,
        zorder=1,
    )
    inset.set_title("Penalty derivative", fontsize=7.0, pad=2)
    inset.set_xlim(0.0, 1.0)
    inset.tick_params(labelsize=6.2, direction="out", length=2, width=0.6)
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)


def legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=KALBI_COLOR, linewidth=1.7, label="KALBI"),
        Line2D(
            [0],
            [0],
            color=CD_COLOR,
            linewidth=1.4,
            linestyle="--",
            label="Coordinate descent reference",
        ),
        Line2D(
            [0],
            [0],
            color=SELECTED_COLOR,
            linewidth=1.2,
            linestyle="-.",
            label="KALBI-selected lambda",
        ),
        Line2D(
            [0],
            [0],
            color=ORACLE_COLOR,
            linewidth=1.2,
            linestyle=(0, (2, 1)),
            marker="D",
            markersize=4,
            label="Oracle lambda",
        ),
        Line2D(
            [0],
            [0],
            color=BASELINE_COLOR,
            linewidth=1.0,
            linestyle=":",
            label="Zero-predictor baseline",
        ),
    ]


def save(fig: plt.Figure, png: Path, pdf: Path) -> None:
    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "Matplotlib"},
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Creator": "KALBI evaluation"},
    )
    plt.close(fig)


def render_composite(
    metrics: list[dict],
    arrays: list[dict[str, np.ndarray]],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(
        4,
        4,
        figsize=(14.2, 11.5),
        constrained_layout=True,
        gridspec_kw={"wspace": 0.06, "hspace": 0.07},
    )
    for row, (metric, values) in enumerate(zip(metrics, arrays, strict=True)):
        plot_coefficient_path(axes[row, 0], values)
        plot_objective(axes[row, 1], values)
        plot_heldout(
            axes[row, 2], values, metric, show_test_annotation=False
        )
        plot_penalty(axes[row, 3], values)
        axes[row, 0].text(
            0.01,
            0.97,
            f"Dataset {row + 1}\nN={metric['n']:,}, P={metric['p']:,}",
            transform=axes[row, 0].transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            color="#222222",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.86,
                "pad": 1.4,
            },
        )
        for axis in axes[row]:
            style_axis(axis)
    for axis, title in zip(
        axes[0],
        (
            "Coefficient solution path",
            "Training objective path",
            "Held-out error path",
            "Selected penalty",
        ),
        strict=True,
    ):
        axis.set_title(title, fontsize=10.8, pad=7)
    fig.legend(
        handles=legend_handles(),
        loc="outside lower center",
        ncol=5,
        frameon=False,
        fontsize=8.6,
        handlelength=2.6,
        columnspacing=1.6,
    )
    save(
        fig,
        output_dir / "kalbi_solution_path_composite.png",
        output_dir / "kalbi_solution_path_composite.pdf",
    )


def render_details(
    metrics: list[dict],
    arrays: list[dict[str, np.ndarray]],
    output_dir: Path,
) -> None:
    for row, (metric, values) in enumerate(zip(metrics, arrays, strict=True)):
        fig, axes = plt.subplots(
            2, 2, figsize=(8.1, 6.4), constrained_layout=True
        )
        plot_coefficient_path(axes[0, 0], values)
        plot_objective(axes[0, 1], values)
        plot_heldout(
            axes[1, 0], values, metric, show_test_annotation=True
        )
        plot_penalty(axes[1, 1], values)
        titles = (
            "Coefficient solution path",
            "Training objective path",
            "Held-out error path",
            "Selected penalty",
        )
        for axis, title in zip(axes.reshape(-1), titles, strict=True):
            style_axis(axis)
            axis.set_title(title, fontsize=10.2, pad=6)
        axes[0, 0].text(
            0.01,
            0.97,
            f"Dataset {row + 1}\nN={metric['n']:,}, P={metric['p']:,}",
            transform=axes[0, 0].transAxes,
            ha="left",
            va="top",
            fontsize=8.4,
            color="#222222",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.86,
                "pad": 1.4,
            },
        )
        fig.legend(
            handles=legend_handles(),
            loc="outside lower center",
            ncol=3,
            frameon=False,
            fontsize=8.0,
            handlelength=2.5,
        )
        save(
            fig,
            output_dir / f"dataset_{row + 1}_detail.png",
            output_dir / f"dataset_{row + 1}_detail.pdf",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    args = parser.parse_args()
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.6,
            "axes.labelsize": 8.6,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.2,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.75,
            "savefig.transparent": False,
        }
    )
    metrics, result_dir = load_results(args.evaluation_root)
    arrays = [
        load_arrays(result_dir, int(metric["dataset_index"]))
        for metric in metrics
    ]
    detail_dir = args.evaluation_root / "figures" / "detail"
    composite_dir = args.evaluation_root / "figures" / "composite"
    if any(detail_dir.iterdir()) or any(composite_dir.iterdir()):
        raise RuntimeError("figure output directories are not empty")
    render_details(metrics, arrays, detail_dir)
    render_composite(metrics, arrays, composite_dir)
    print(
        json.dumps(
            {
                "status": "PASS",
                "detail_png_count": 4,
                "detail_pdf_count": 4,
                "composite_png": str(
                    (composite_dir / "kalbi_solution_path_composite.png").resolve()
                ),
                "composite_pdf": str(
                    (composite_dir / "kalbi_solution_path_composite.pdf").resolve()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
