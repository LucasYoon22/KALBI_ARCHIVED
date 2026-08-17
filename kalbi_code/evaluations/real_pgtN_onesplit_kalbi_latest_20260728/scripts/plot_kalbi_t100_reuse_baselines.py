#!/usr/bin/env python3
"""Plot KALBI T=100 paths with previously computed penalized baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np


PROJECT = Path("/users/minsuk/fm-lm-lambda")
SOURCE_ROOT = PROJECT / "evaluations/real_pgtN_onesplit_kalbi_latest_20260728"
SCRIPT_ROOT = SOURCE_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from real_pgtN_common import atomic_json, load_split_manifest  # noqa: E402


BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GRAY = "#666666"
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


def finite_method(metric: dict[str, Any], name: str) -> dict[str, Any] | None:
    method = metric["baselines_reused_without_recomputation"].get(name)
    if not method:
        return None
    value = method.get("normalized_test_mse")
    raw_lambda = method.get("selected_raw_lambda")
    if value is None or raw_lambda is None:
        return None
    if not np.isfinite(value) or not np.isfinite(raw_lambda):
        return None
    return method


def coefficient_panel(
    ax: plt.Axes, arrays: dict[str, np.ndarray], feature_count: int = 6
) -> None:
    x = np.log10(arrays["raw_lambdas"])
    for position, feature in enumerate(arrays["top_features"][:feature_count]):
        color = FEATURE_COLORS[position % len(FEATURE_COLORS)]
        ax.plot(
            x,
            arrays["kalbi_beta"][:, feature],
            color=color,
            lw=1.6,
        )
        ax.plot(
            x,
            arrays["lasso_beta"][:, feature],
            color=color,
            lw=1.2,
            ls="--",
            alpha=0.82,
        )
    ax.axhline(0.0, color=LIGHT_GRAY, lw=0.8)
    ax.set_xlabel(r"$\log_{10}(\lambda_{\rm raw})$")
    ax.set_ylabel("Standardized coefficient")


def error_panel(
    ax: plt.Axes, arrays: dict[str, np.ndarray], metric: dict[str, Any]
) -> None:
    raw = arrays["raw_lambdas"]
    x = np.log10(raw)
    kalbi = arrays["kalbi_normalized_test_mse"]
    lasso = arrays["lasso_normalized_test_mse"]
    ax.plot(x, kalbi, color=BLUE, lw=2.0)
    ax.plot(x, lasso, color=ORANGE, lw=1.8, ls="--")
    ax.axhline(1.0, color=GRAY, lw=1.0, ls=":")
    ax.scatter(
        [x[int(np.argmin(kalbi))]],
        [float(np.min(kalbi))],
        marker="D",
        s=35,
        color=BLUE,
        edgecolor="white",
        linewidth=0.4,
        zorder=8,
    )
    ax.scatter(
        [x[int(np.argmin(lasso))]],
        [float(np.min(lasso))],
        marker="D",
        s=35,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.4,
        zorder=8,
    )

    marker_specs = (
        ("lasso", ORANGE, "o"),
        ("scad", GREEN, "s"),
        ("mcp", PURPLE, "^"),
    )
    for method_name, color, marker in marker_specs:
        method = finite_method(metric, method_name)
        if method is None:
            continue
        raw_lambda = float(method["selected_raw_lambda"])
        if raw[0] <= raw_lambda <= raw[-1]:
            ax.scatter(
                [np.log10(raw_lambda)],
                [float(method["normalized_test_mse"])],
                color=color,
                marker=marker,
                s=46,
                edgecolor="white",
                linewidth=0.5,
                zorder=9,
            )
    selected_raw = float(metric["selected_raw_lambda"])
    if raw[0] <= selected_raw <= raw[-1]:
        ax.axvline(np.log10(selected_raw), color=BLUE, lw=1.0, alpha=0.6)
        ax.scatter(
            [np.log10(selected_raw)],
            [float(metric["selected_normalized_test_mse"])],
            color=BLUE,
            marker="o",
            s=50,
            edgecolor="white",
            linewidth=0.5,
            zorder=9,
        )
    positive = np.concatenate([kalbi[kalbi > 0], lasso[lasso > 0]])
    if positive.size:
        ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}(\lambda_{\rm raw})$")
    ax.set_ylabel("Test MSE / zero-predictor MSE")


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def shared_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=BLUE, lw=2.0, label="KALBI, T=100"),
        Line2D(
            [0], [0], color=ORANGE, lw=1.8, ls="--", label="LASSO path"
        ),
        Line2D(
            [0],
            [0],
            color=BLUE,
            marker="o",
            lw=0,
            label="KALBI-selected",
        ),
        Line2D(
            [0],
            [0],
            color=ORANGE,
            marker="o",
            lw=0,
            label="CV LASSO",
        ),
        Line2D(
            [0],
            [0],
            color=GREEN,
            marker="s",
            lw=0,
            label="CV SCAD",
        ),
        Line2D(
            [0],
            [0],
            color=PURPLE,
            marker="^",
            lw=0,
            label="CV MCP",
        ),
        Line2D(
            [0], [0], color=GRAY, marker="D", lw=0, label="Path oracle"
        ),
    ]


def task_figure(
    task: dict[str, Any],
    arrays: dict[str, np.ndarray],
    metric: dict[str, Any],
) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    coefficient_panel(axes[0, 0], arrays, feature_count=10)
    error_panel(axes[0, 1], arrays, metric)
    x = np.log10(arrays["raw_lambdas"])
    axes[1, 0].plot(x, arrays["kalbi_nnz"], color=BLUE, lw=2.0)
    axes[1, 0].plot(
        x, arrays["lasso_nnz"], color=ORANGE, lw=1.8, ls="--"
    )
    axes[1, 0].set_xlabel(r"$\log_{10}(\lambda_{\rm raw})$")
    axes[1, 0].set_ylabel(r"Nonzero coefficients ($10^{-8}$)")
    axes[1, 1].plot(x, arrays["kalbi_l1"], color=BLUE, lw=2.0)
    axes[1, 1].plot(
        x, arrays["lasso_l1"], color=ORANGE, lw=1.8, ls="--"
    )
    axes[1, 1].set_xlabel(r"$\log_{10}(\lambda_{\rm raw})$")
    axes[1, 1].set_ylabel(r"$\|\widehat{\beta}\|_1$")
    for ax, title in zip(
        axes.flat,
        (
            "Coefficient solution path",
            "Held-out error and selected methods",
            "Sparsity path",
            "Coefficient-norm path",
        ),
        strict=True,
    ):
        ax.set_title(title)
    figure.suptitle(
        f"Dataset {task['registry_index']}  |  "
        f"$N_{{train}}={metric['outer_train_rows']:,}$, "
        f"$N_{{test}}={metric['outer_test_rows']:,}$, "
        f"$P={metric['P_transformed']:,}$",
        y=0.995,
    )
    figure.legend(
        handles=shared_handles(),
        loc="lower center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.96), w_pad=3.0, h_pad=2.0)
    return figure


def summary_figure(
    records: list[
        tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]
    ],
) -> plt.Figure:
    figure, axes = plt.subplots(3, 4, figsize=(18, 12))
    for row in range(3):
        for slot in range(2):
            task, arrays, metric = records[2 * row + slot]
            left = axes[row, 2 * slot]
            right = axes[row, 2 * slot + 1]
            coefficient_panel(left, arrays)
            error_panel(right, arrays, metric)
            prefix = f"Dataset {task['registry_index']}"
            left.set_title(
                f"{prefix}: coefficient paths\n"
                f"$N={task['N']:,}$, $P={metric['P_transformed']:,}$"
            )
            right.set_title(f"{prefix}: held-out error")
    figure.legend(
        handles=shared_handles(),
        loc="lower center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return figure


def aggregate_figure(metrics: list[dict[str, Any]]) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    index = np.asarray([row["registry_index"] for row in metrics])
    t10_selected = np.asarray(
        [row["t10_selected_normalized_test_mse"] for row in metrics]
    )
    t100_selected = np.asarray(
        [row["selected_normalized_test_mse"] for row in metrics]
    )
    t10_oracle = np.asarray(
        [row["t10_path_oracle_normalized_test_mse"] for row in metrics]
    )
    t100_oracle = np.asarray(
        [row["path_oracle_normalized_test_mse"] for row in metrics]
    )

    for ax, left, right, title in (
        (
            axes[0, 0],
            t10_selected,
            t100_selected,
            "KALBI selected-lambda error",
        ),
        (
            axes[0, 1],
            t10_oracle,
            t100_oracle,
            "KALBI raw-grid oracle error",
        ),
    ):
        finite = np.isfinite(left) & np.isfinite(right) & (left > 0) & (right > 0)
        lower = min(float(left[finite].min()), float(right[finite].min()))
        upper = max(float(left[finite].max()), float(right[finite].max()))
        ax.scatter(left, right, c=index, cmap="viridis", s=38)
        ax.plot([lower, upper], [lower, upper], color=GRAY, ls=":", lw=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("T=10 normalized test MSE")
        ax.set_ylabel("T=100 normalized test MSE")
        ax.set_title(title)

    method_names = ("KALBI T=100", "CV LASSO", "CV SCAD", "CV MCP")
    values: list[np.ndarray] = [t100_selected]
    for method_name in ("lasso", "scad", "mcp"):
        values.append(
            np.asarray(
                [
                    (
                        row["baselines_reused_without_recomputation"][
                            method_name
                        ]["normalized_test_mse"]
                        if row["baselines_reused_without_recomputation"][
                            method_name
                        ]["normalized_test_mse"]
                        is not None
                        else np.nan
                    )
                    for row in metrics
                ],
                dtype=np.float64,
            )
        )
    finite_values = [value[np.isfinite(value) & (value > 0)] for value in values]
    axes[1, 0].boxplot(
        finite_values,
        tick_labels=method_names,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#D9EAF4"},
        medianprops={"color": "#222222"},
    )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("Selected normalized test MSE")
    axes[1, 0].set_title("Previously computed baseline comparison")
    axes[1, 0].tick_params(axis="x", rotation=18)

    ratio = t100_selected / np.maximum(t10_selected, 1e-30)
    axes[1, 1].scatter(index, ratio, color=BLUE, s=34)
    axes[1, 1].axhline(1.0, color=GRAY, ls=":", lw=1.2)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Dataset index")
    axes[1, 1].set_ylabel("T=100 / T=10 selected MSE")
    axes[1, 1].set_title("Effect of 100 refinement sweeps")
    figure.tight_layout()
    return figure


def save(figure: plt.Figure, stem: Path, dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=dpi)
    figure.savefig(stem.with_suffix(".pdf"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    configure_style()
    manifest = load_split_manifest(
        SOURCE_ROOT / "manifests/split_manifest.json"
    )
    records = []
    for task in manifest["tasks"]:
        task_id = task["task_id"]
        metric_path = args.output_root / "metrics" / f"{task_id}.json"
        path_path = args.output_root / "paths" / f"{task_id}.npz"
        if not metric_path.is_file() or not path_path.is_file():
            continue
        records.append((task, load_arrays(path_path), load_json(metric_path)))
    if len(records) != manifest["task_count"]:
        raise RuntimeError(
            f"expected {manifest['task_count']} tasks, found {len(records)}"
        )

    per_task_root = args.output_root / "figures" / "per_task"
    multipage = args.output_root / "figures" / "kalbi_t100_all_tasks.pdf"
    multipage.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(multipage) as pdf:
        for task, arrays, metric in records:
            figure = task_figure(task, arrays, metric)
            stem = per_task_root / (
                f"dataset_{int(task['registry_index']):02d}_{task['task_id']}"
            )
            save(figure, stem, args.dpi)
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)

    lookup = {int(task["registry_index"]): row for task, *row in records}
    summary_indices = [1, 9, 17, 25, 33, 41]
    selected_records = [
        (
            next(task for task in manifest["tasks"] if task["registry_index"] == i),
            lookup[i][0],
            lookup[i][1],
        )
        for i in summary_indices
    ]
    summary = summary_figure(selected_records)
    summary_stem = (
        args.output_root
        / "figures/summary/kalbi_t100_lasso_paths_cv_penalties"
    )
    save(summary, summary_stem, args.dpi)
    plt.close(summary)

    metrics = [metric for _, _, metric in records]
    aggregate = aggregate_figure(metrics)
    aggregate_stem = (
        args.output_root / "figures/summary/kalbi_t100_aggregate"
    )
    save(aggregate, aggregate_stem, args.dpi)
    plt.close(aggregate)

    csv_path = args.output_root / "reports/comparison.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "registry_index",
        "task_id",
        "N",
        "P_transformed",
        "selected_raw_lambda",
        "selected_normalized_test_mse",
        "t10_selected_normalized_test_mse",
        "path_oracle_normalized_test_mse",
        "t10_path_oracle_normalized_test_mse",
        "maximum_abs_beta",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric in metrics:
            writer.writerow({key: metric[key] for key in fields})

    selected_ratio = np.asarray(
        [
            row["selected_normalized_test_mse"]
            / max(row["t10_selected_normalized_test_mse"], 1e-30)
            for row in metrics
        ]
    )
    oracle_ratio = np.asarray(
        [
            row["path_oracle_normalized_test_mse"]
            / max(row["t10_path_oracle_normalized_test_mse"], 1e-30)
            for row in metrics
        ]
    )
    report = {
        "status": "PASS",
        "task_count": len(metrics),
        "refinement_sweeps": 100,
        "baselines_recomputed": False,
        "selected_t100_over_t10": {
            "median": float(np.median(selected_ratio)),
            "geometric_mean": float(np.exp(np.mean(np.log(selected_ratio)))),
            "improved_count": int(np.count_nonzero(selected_ratio < 1)),
            "worsened_count": int(np.count_nonzero(selected_ratio > 1)),
        },
        "oracle_t100_over_t10": {
            "median": float(np.median(oracle_ratio)),
            "geometric_mean": float(np.exp(np.mean(np.log(oracle_ratio)))),
            "improved_count": int(np.count_nonzero(oracle_ratio < 1)),
            "worsened_count": int(np.count_nonzero(oracle_ratio > 1)),
        },
        "figures": {
            "summary_png": str(summary_stem.with_suffix(".png")),
            "summary_pdf": str(summary_stem.with_suffix(".pdf")),
            "aggregate_png": str(aggregate_stem.with_suffix(".png")),
            "aggregate_pdf": str(aggregate_stem.with_suffix(".pdf")),
            "multipage_pdf": str(multipage),
            "per_task_count": len(records),
        },
    }
    atomic_json(args.output_root / "reports/aggregate_metrics.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
