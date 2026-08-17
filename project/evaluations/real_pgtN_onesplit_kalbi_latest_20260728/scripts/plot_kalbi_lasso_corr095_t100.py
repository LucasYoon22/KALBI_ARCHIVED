#!/usr/bin/env python3
"""Render KALBI/LASSO results after train-only |corr| >= 0.95 filtering."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path("/users/minsuk/fm-lm-lambda")
SOURCE_ROOT = PROJECT / "evaluations/real_pgtN_onesplit_kalbi_latest_20260728"
OLD_T100_ROOT = SOURCE_ROOT / "kalbi_t100_reuse_penalized_20260728"

KALBI_COLOR = "#0072B2"
LASSO_COLOR = "#D55E00"
ORACLE_COLOR = "#009E73"
GRAY = "#555555"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_results(root: Path) -> list[dict[str, Any]]:
    metrics = [load_json(path) for path in (root / "metrics").glob("*.json")]
    metrics.sort(key=lambda item: int(item["registry_index"]))
    if len(metrics) != 41 or any(item.get("status") != "PASS" for item in metrics):
        raise RuntimeError(f"expected 41 PASS metrics, found {len(metrics)}")
    return metrics


def old_result(task_id: str) -> tuple[float, float]:
    kalbi = load_json(OLD_T100_ROOT / "metrics" / f"{task_id}.json")
    baseline = load_json(SOURCE_ROOT / "metrics" / f"{task_id}_baselines.json")
    return (
        float(kalbi["selected_normalized_test_mse"]),
        float(baseline["methods"]["lasso"]["normalized_test_mse"]),
    )


def clipped_positive(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=np.float64), 1e-6)


def draw_task_row(
    axes: np.ndarray,
    metric: dict[str, Any],
    root: Path,
    *,
    add_method_legend: bool,
) -> None:
    with np.load(root / "paths" / f"{metric['task_id']}.npz") as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    x = np.log10(data["raw_lambdas"])
    top = data["top_features"].astype(np.int64)[:8]
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, top.size)))
    for color, feature in zip(colors, top, strict=True):
        axes[0].plot(
            x,
            data["kalbi_beta"][:, feature],
            color=color,
            lw=1.5,
            alpha=0.9,
        )
        axes[0].plot(
            x,
            data["lasso_beta"][:, feature],
            color=color,
            lw=1.2,
            ls="--",
            alpha=0.75,
        )
    axes[0].axhline(0.0, color="#BBBBBB", lw=0.7)
    axes[0].set_ylabel(
        f"Dataset {metric['registry_index']}\ncoefficient",
        fontsize=10,
    )

    axes[1].plot(
        x,
        data["kalbi_normalized_test_mse"],
        color=KALBI_COLOR,
        lw=2.0,
        label="KALBI",
    )
    axes[1].plot(
        x,
        data["lasso_normalized_test_mse"],
        color=LASSO_COLOR,
        lw=2.0,
        label="LASSO",
    )
    axes[1].axhline(
        1.0, color=GRAY, lw=1.0, ls=":", label="Zero predictor"
    )
    axes[1].scatter(
        np.log10(metric["selected_raw_lambda"]),
        metric["selected_normalized_test_mse"],
        color=KALBI_COLOR,
        marker="D",
        s=35,
        zorder=5,
        label="KALBI selected",
    )
    axes[1].scatter(
        np.log10(metric["cv_lasso_selected_raw_lambda"]),
        metric["cv_lasso_normalized_test_mse"],
        color=LASSO_COLOR,
        marker="s",
        s=35,
        zorder=5,
        label="CV-LASSO selected",
    )
    axes[1].scatter(
        x[int(np.argmin(data["kalbi_normalized_test_mse"]))],
        float(np.min(data["kalbi_normalized_test_mse"])),
        color=ORACLE_COLOR,
        marker="*",
        s=60,
        zorder=5,
        label="Path oracle",
    )
    axes[1].set_yscale("log")

    axes[2].plot(
        x,
        data["kalbi_nnz"],
        color=KALBI_COLOR,
        lw=2.0,
        label="KALBI",
    )
    axes[2].plot(
        x,
        data["lasso_nnz"],
        color=LASSO_COLOR,
        lw=2.0,
        label="LASSO",
    )
    axes[2].set_yscale("symlog", linthresh=1)

    penalty_grid = data["penalty_grid"].reshape(-1)
    penalty_rho = data["penalty_rho"].reshape(-1)
    penalty_drho = data["penalty_drho"].reshape(-1)
    axes[3].plot(
        penalty_grid, penalty_rho, color=KALBI_COLOR, lw=2.0, label=r"$\rho$"
    )
    twin = axes[3].twinx()
    twin.plot(
        penalty_grid,
        penalty_drho,
        color=ORACLE_COLOR,
        lw=1.5,
        ls="--",
        label=r"$\rho'$",
    )
    twin.tick_params(axis="y", labelsize=8, colors=ORACLE_COLOR)
    twin.spines["right"].set_color(ORACLE_COLOR)

    subtitle = (
        f"N={metric['N']}, P: {metric['P_after_constant_filter']}"
        f"→{metric['P_after_correlation_filter']}"
    )
    axes[0].text(
        0.02,
        0.97,
        subtitle,
        transform=axes[0].transAxes,
        va="top",
        fontsize=8.5,
    )
    if add_method_legend:
        axes[1].legend(
            loc="best", fontsize=8, frameon=False, ncol=2
        )
        axes[3].legend(loc="upper left", fontsize=8, frameon=False)
        twin.legend(loc="upper right", fontsize=8, frameon=False)


def render_composite(
    metrics: list[dict[str, Any]], root: Path, figure_root: Path
) -> None:
    selected_indices = [1, 9, 13, 18, 31, 41]
    selected = {
        int(metric["registry_index"]): metric for metric in metrics
    }
    fig, axes = plt.subplots(
        len(selected_indices),
        4,
        figsize=(17.5, 20.5),
        constrained_layout=True,
    )
    for row, registry_index in enumerate(selected_indices):
        draw_task_row(
            axes[row],
            selected[registry_index],
            root,
            add_method_legend=row == 0,
        )
        for axis in axes[row]:
            axis.grid(False)
            axis.tick_params(labelsize=8.5)
    for axis, title in zip(
        axes[0],
        [
            "Coefficient solution path",
            "Held-out prediction error",
            "Sparsity path",
            "KALBI penalty",
        ],
        strict=True,
    ):
        axis.set_title(title, fontsize=12)
    for axis in axes[-1, :3]:
        axis.set_xlabel(r"$\log_{10}(\lambda_{\mathrm{raw}})$", fontsize=10)
    axes[-1, 3].set_xlabel(r"$|\beta|$", fontsize=10)
    fig.savefig(
        figure_root / "kalbi_lasso_corr095_t100_composite.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        figure_root / "kalbi_lasso_corr095_t100_composite.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def render_aggregate(
    metrics: list[dict[str, Any]], figure_root: Path
) -> dict[str, Any]:
    before = np.asarray(
        [item["P_after_constant_filter"] for item in metrics], dtype=float
    )
    after = np.asarray(
        [item["P_after_correlation_filter"] for item in metrics], dtype=float
    )
    kalbi = np.asarray(
        [item["selected_normalized_test_mse"] for item in metrics], dtype=float
    )
    lasso = np.asarray(
        [item["cv_lasso_normalized_test_mse"] for item in metrics], dtype=float
    )
    old = np.asarray([old_result(item["task_id"]) for item in metrics])
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), constrained_layout=True)

    axes[0].scatter(before, after, color=KALBI_COLOR, s=35)
    limit = [max(1, min(after.min(), before.min())), max(before.max(), after.max())]
    axes[0].plot(limit, limit, color=GRAY, ls=":", lw=1.2)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Features before filtering")
    axes[0].set_ylabel("Features after filtering")

    low = max(1e-3, min(kalbi.min(), lasso.min()) * 0.8)
    high = max(kalbi.max(), lasso.max()) * 1.2
    axes[1].scatter(lasso, kalbi, color=KALBI_COLOR, s=35)
    axes[1].plot([low, high], [low, high], color=GRAY, ls=":", lw=1.2)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("CV-LASSO normalized test MSE")
    axes[1].set_ylabel("KALBI normalized test MSE")

    for index in range(len(metrics)):
        axes[2].plot(
            [0, 1],
            clipped_positive([old[index, 0], kalbi[index]]),
            color=KALBI_COLOR,
            alpha=0.25,
            lw=0.8,
        )
        axes[3].plot(
            [0, 1],
            clipped_positive([old[index, 1], lasso[index]]),
            color=LASSO_COLOR,
            alpha=0.25,
            lw=0.8,
        )
    axes[2].plot(
        [0, 1],
        [np.median(old[:, 0]), np.median(kalbi)],
        color=KALBI_COLOR,
        lw=3,
        marker="o",
    )
    axes[3].plot(
        [0, 1],
        [np.median(old[:, 1]), np.median(lasso)],
        color=LASSO_COLOR,
        lw=3,
        marker="o",
    )
    for axis, label in ((axes[2], "KALBI"), (axes[3], "CV-LASSO")):
        axis.set_xticks([0, 1], ["Unfiltered", "|corr| < 0.95"])
        axis.set_yscale("log")
        axis.set_ylabel("Normalized test MSE")
        axis.set_title(label)
    for axis in axes:
        axis.grid(False)
    fig.savefig(
        figure_root / "kalbi_lasso_corr095_t100_aggregate.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        figure_root / "kalbi_lasso_corr095_t100_aggregate.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
    return {
        "task_count": len(metrics),
        "median_P_before": float(np.median(before)),
        "median_P_after": float(np.median(after)),
        "median_retained_fraction": float(np.median(after / before)),
        "tasks_with_P_below_32": int(np.count_nonzero(after < 32)),
        "maximum_post_filter_abs_corr": float(
            max(item["filter"]["post_filter_max_abs_corr"] for item in metrics)
        ),
        "kalbi_median_normalized_mse": float(np.median(kalbi)),
        "cv_lasso_median_normalized_mse": float(np.median(lasso)),
        "kalbi_wins": int(np.count_nonzero(kalbi < lasso)),
        "cv_lasso_wins": int(np.count_nonzero(lasso < kalbi)),
        "ties": int(np.count_nonzero(np.isclose(kalbi, lasso))),
        "kalbi_filter_improved_count": int(np.count_nonzero(kalbi < old[:, 0])),
        "lasso_filter_improved_count": int(np.count_nonzero(lasso < old[:, 1])),
        "kalbi_median_filtered_over_unfiltered": float(
            np.median(kalbi / old[:, 0])
        ),
        "lasso_median_filtered_over_unfiltered": float(
            np.median(lasso / old[:, 1])
        ),
    }


def write_table(metrics: list[dict[str, Any]], output: Path) -> None:
    fields = [
        "registry_index",
        "task_id",
        "N",
        "P_after_constant_filter",
        "P_after_correlation_filter",
        "selected_raw_lambda",
        "selected_normalized_test_mse",
        "cv_lasso_selected_raw_lambda",
        "cv_lasso_normalized_test_mse",
        "kalbi_path_oracle_normalized_test_mse",
        "lasso_path_oracle_normalized_test_mse",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric in metrics:
            writer.writerow({key: metric[key] for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    figure_root = args.root / "figures"
    report_root = args.root / "reports"
    figure_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    metrics = load_results(args.root)
    render_composite(metrics, args.root, figure_root)
    aggregate = render_aggregate(metrics, figure_root)
    write_table(metrics, report_root / "task_summary.csv")
    (report_root / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, sort_keys=True))


if __name__ == "__main__":
    main()
