#!/usr/bin/env python3
"""Render corr095 KALBI/LASSO solution paths without a CD reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kalbi_v6_exactprox10_eta4x_v23_independentraw_k32s32_p100k_effbatch32.utils import (
    make_penalty_grid,
)


BUNDLES = {
    "KALBI-A-v6-v23-s20300": ("KALBI-A", "#0072B2"),
    "KALBI-B-v8-v23-s20300": ("KALBI-B", "#56B4E9"),
    "KALBI-C-v8-v24-s03900": ("KALBI-C", "#009E73"),
    "KALBI-D-v8-v24-s13100": ("KALBI-D", "#D55E00"),
}
LASSO_COLOR = "#000000"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_task(evaluation_root: Path, task_id: str) -> dict[str, Any]:
    bundle_data: dict[str, dict[str, Any]] = {}
    retained: np.ndarray | None = None
    raw_lambdas: np.ndarray | None = None
    for bundle_id in BUNDLES:
        metric = load_json(
            evaluation_root
            / "bundles"
            / bundle_id
            / "metrics"
            / f"{task_id}.json"
        )
        with np.load(metric["path_output"], allow_pickle=False) as archive:
            current_retained = np.asarray(
                archive["retained_original_indices"], dtype=np.int64
            )
            current_raw = np.asarray(archive["raw_lambdas"], dtype=np.float64)
            if retained is None:
                retained = current_retained
                raw_lambdas = current_raw
            else:
                np.testing.assert_array_equal(retained, current_retained)
                np.testing.assert_allclose(raw_lambdas, current_raw)
            bundle_data[bundle_id] = {
                "metric": metric,
                "beta": np.asarray(archive["path_beta"], dtype=np.float64),
                "nmse": np.asarray(
                    archive["path_normalized_test_mse"], dtype=np.float64
                ),
                "kkt": np.asarray(
                    archive["path_projected_kkt_max"], dtype=np.float64
                ),
                "rho": np.asarray(archive["penalty_rho"], dtype=np.float64),
                "drho": np.asarray(
                    archive["penalty_drho"], dtype=np.float64
                ),
            }
    lasso_metric = load_json(
        evaluation_root
        / "solution_paths"
        / "lasso_metrics"
        / f"{task_id}.json"
    )
    with np.load(lasso_metric["path_output"], allow_pickle=False) as archive:
        np.testing.assert_array_equal(
            retained, archive["retained_original_indices"]
        )
        np.testing.assert_allclose(raw_lambdas, archive["raw_lambdas"])
        lasso = {
            "metric": lasso_metric,
            "beta": np.asarray(archive["beta"], dtype=np.float64),
            "nmse": np.asarray(
                archive["normalized_test_mse"], dtype=np.float64
            ),
            "kkt": np.asarray(
                archive["projected_kkt_max"], dtype=np.float64
            ),
        }
    baseline = load_json(
        evaluation_root / "metrics" / f"{task_id}_baselines.json"
    )
    return {
        "raw_lambdas": raw_lambdas,
        "retained": retained,
        "bundles": bundle_data,
        "lasso": lasso,
        "baseline": baseline,
    }


def top_features(task: dict[str, Any], count: int = 8) -> np.ndarray:
    paths = [task["lasso"]["beta"]]
    paths.extend(value["beta"] for value in task["bundles"].values())
    importance = np.max(
        np.stack([np.max(np.abs(path), axis=0) for path in paths]), axis=0
    )
    original = task["retained"]
    order = np.lexsort((original, -importance))
    return order[: min(int(count), order.size)]


def render_row(
    axes: np.ndarray,
    task: dict[str, Any],
    *,
    dataset_label: str,
    show_legend: bool,
) -> None:
    x = np.log10(task["raw_lambdas"])
    latest = task["bundles"]["KALBI-D-v8-v24-s13100"]
    selected = top_features(task)
    feature_colors = plt.get_cmap("tab10")(np.arange(selected.size) % 10)
    for color, local_index in zip(feature_colors, selected, strict=True):
        original = int(task["retained"][local_index])
        axes[0].plot(
            x,
            latest["beta"][:, local_index],
            color=color,
            lw=1.4,
            label=f"feature {original}",
        )
        axes[0].plot(
            x,
            task["lasso"]["beta"][:, local_index],
            color=color,
            lw=1.0,
            ls="--",
        )
    axes[0].axhline(0.0, color="0.6", lw=0.7)
    axes[0].set_title(
        f"{dataset_label}: coefficient path\n"
        "solid KALBI-D, dashed LASSO",
        fontsize=10,
    )
    axes[0].set_ylabel("Coefficient")

    for bundle_id, (label, color) in BUNDLES.items():
        value = task["bundles"][bundle_id]
        axes[1].plot(x, value["nmse"], color=color, lw=1.6, label=label)
        metric = value["metric"]
        axes[1].scatter(
            np.log10(metric["selected_raw_lambda"]),
            metric["selected_normalized_test_mse"],
            color=color,
            s=26,
            marker="o",
            zorder=4,
        )
    axes[1].plot(
        x,
        task["lasso"]["nmse"],
        color=LASSO_COLOR,
        lw=1.6,
        label="LASSO path",
    )
    lasso_cv = task["baseline"]["methods"]["lasso"]
    axes[1].scatter(
        np.log10(lasso_cv["selected_raw_lambda"]),
        lasso_cv["normalized_test_mse"],
        color=LASSO_COLOR,
        s=30,
        marker="x",
        zorder=4,
        label="CV LASSO selection",
    )
    axes[1].axhline(1.0, color="0.45", lw=0.8, ls=":")
    axes[1].set_yscale("log")
    axes[1].set_title("Held-out prediction error", fontsize=10)
    axes[1].set_ylabel("Test MSE / zero-predictor MSE")

    for bundle_id, (label, color) in BUNDLES.items():
        beta = task["bundles"][bundle_id]["beta"]
        axes[2].plot(
            x,
            np.count_nonzero(np.abs(beta) > 1e-8, axis=1),
            color=color,
            lw=1.5,
            label=label,
        )
    axes[2].plot(
        x,
        np.count_nonzero(np.abs(task["lasso"]["beta"]) > 1e-8, axis=1),
        color=LASSO_COLOR,
        lw=1.5,
        label="LASSO",
    )
    axes[2].set_yscale("symlog", linthresh=1)
    axes[2].set_title("Sparsity path", fontsize=10)
    axes[2].set_ylabel("NNZ")

    penalty_grid = make_penalty_grid().cpu().numpy()
    domain = penalty_grid <= 1.0 + 1e-12
    for bundle_id, (label, color) in BUNDLES.items():
        value = task["bundles"][bundle_id]
        axes[3].plot(
            penalty_grid[domain],
            value["rho"][domain],
            color=color,
            lw=1.7,
            label=label,
        )
    axes[3].set_title("KALBI-selected penalty", fontsize=10)
    axes[3].set_xlabel(r"$|\beta|$")
    axes[3].set_ylabel(r"$\rho(|\beta|)$")
    axes[3].set_xlim(0.0, 1.0)

    for axis in axes[:3]:
        axis.set_xlabel(r"$\log_{10}(\lambda_{\mathrm{raw}})$")
        axis.set_xlim(float(x[0]), float(x[-1]))
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(False)
    if show_legend:
        axes[0].legend(frameon=False, fontsize=6, ncol=2)
        axes[1].legend(frameon=False, fontsize=7, ncol=2)
        axes[3].legend(frameon=False, fontsize=7, ncol=2)


def save_figure(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    args = parser.parse_args()
    metrics = sorted(
        (
            args.evaluation_root
            / "bundles"
            / "KALBI-D-v8-v24-s13100"
            / "metrics"
        ).glob("*.json")
    )
    task_metrics = [
        load_json(path)
        for path in metrics
        if load_json(path).get("status") == "PASS"
    ]
    task_metrics.sort(key=lambda value: value["P_after_corr095"])
    if len(task_metrics) != 25:
        raise RuntimeError("expected 25 applicable tasks")

    positions = np.linspace(0, len(task_metrics) - 1, 6).round().astype(int)
    representative = [task_metrics[index] for index in positions]
    figure, axes = plt.subplots(
        len(representative), 4, figsize=(14.5, 3.15 * len(representative))
    )
    manifest: list[dict[str, Any]] = []
    for row_index, metric in enumerate(representative):
        task = load_task(args.evaluation_root, metric["task_id"])
        label = (
            f"Dataset {row_index + 1} "
            f"(N={metric['N']}, P={metric['P_after_corr095']})"
        )
        render_row(
            axes[row_index],
            task,
            dataset_label=label,
            show_legend=row_index == 0,
        )
        manifest.append(
            {
                "dataset_label": f"Dataset {row_index + 1}",
                "registry_index": metric["registry_index"],
                "task_id": metric["task_id"],
                "N": metric["N"],
                "P_after_corr095": metric["P_after_corr095"],
                "selection_rule": "six equally spaced filtered-P order statistics",
            }
        )
    figure.tight_layout()
    save_figure(
        figure,
        args.evaluation_root / "figures" / "solution_paths_representative6",
    )

    detail_root = args.evaluation_root / "figures" / "solution_paths_per_task"
    for metric in task_metrics:
        task = load_task(args.evaluation_root, metric["task_id"])
        figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
        render_row(
            axes.reshape(-1),
            task,
            dataset_label=(
                f"Dataset {metric['registry_index']} "
                f"(N={metric['N']}, P={metric['P_after_corr095']})"
            ),
            show_legend=True,
        )
        figure.tight_layout()
        save_figure(figure, detail_root / metric["task_id"])
    (args.evaluation_root / "manifests").mkdir(parents=True, exist_ok=True)
    (
        args.evaluation_root
        / "manifests"
        / "solution_path_representative_tasks.json"
    ).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
