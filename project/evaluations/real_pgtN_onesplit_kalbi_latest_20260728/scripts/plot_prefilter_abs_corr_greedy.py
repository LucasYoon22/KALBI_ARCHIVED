#!/usr/bin/env python3
"""Render per-task and overview pre-filter absolute-correlation heatmaps."""

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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_metrics(root: Path) -> list[dict[str, Any]]:
    values = [load_json(path) for path in (root / "metrics").glob("*.json")]
    values.sort(key=lambda item: item["registry_index"])
    if len(values) != 41 or any(item["status"] != "PASS" for item in values):
        raise RuntimeError(f"expected 41 PASS metrics, found {len(values)}")
    return values


def short_name(task_id: str, limit: int = 42) -> str:
    value = task_id.replace("hf_", "").replace("__", " / ").replace("_", " ")
    return value if len(value) <= limit else value[: limit - 1] + "…"


def render_details(
    metrics: list[dict[str, Any]], root: Path, output: Path
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        with np.load(metric["array_output"]) as archive:
            greedy = np.clip(
                np.asarray(archive["greedy_abs_corr"], dtype=np.float64),
                0.0,
                1.0,
            )
            global_view = np.clip(
                np.asarray(archive["global_abs_corr"], dtype=np.float64),
                0.0,
                1.0,
            )
        fig, axes = plt.subplots(
            1, 2, figsize=(8.6, 4.2), constrained_layout=True
        )
        images = []
        for axis, matrix, title in (
            (
                axes[0],
                greedy,
                "High-correlation greedy chain",
            ),
            (
                axes[1],
                global_view,
                "Global deterministic sample\nwith greedy ordering",
            ),
        ):
            images.append(
                axis.imshow(
                    matrix,
                    vmin=0.0,
                    vmax=1.0,
                    cmap="magma",
                    interpolation="nearest",
                    aspect="equal",
                )
            )
            axis.set_title(title, fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
        fraction = metric["threshold_pair_fractions"]["ge_0.95"]
        fig.suptitle(
            f"Dataset {metric['registry_index']}: {short_name(metric['task_id'], 58)}\n"
            f"N={metric['N']:,}, P={metric['P_before_correlation_filter']:,}"
            f"→{metric['P_after_correlation_filter']:,}; "
            f"max |corr|={min(metric['max_abs_corr'], 1.0):.6f}; "
            f"pairs ≥0.95={100*fraction:.2f}%",
            fontsize=11,
        )
        fig.colorbar(
            images[-1],
            ax=axes.tolist(),
            fraction=0.035,
            pad=0.02,
            label="Absolute Pearson correlation",
        )
        filename = (
            f"dataset_{metric['registry_index']:02d}_{metric['task_id']}.png"
        )
        fig.savefig(output / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)


def render_overview(
    metrics: list[dict[str, Any]],
    *,
    root: Path,
    output: Path,
    array_key: str,
    filename: str,
    title: str,
) -> None:
    rows, columns = 7, 6
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(15.5, 17.8),
        constrained_layout=True,
    )
    image = None
    for axis, metric in zip(axes.flat, metrics):
        with np.load(metric["array_output"]) as archive:
            matrix = np.clip(
                np.asarray(archive[array_key], dtype=np.float64), 0.0, 1.0
            )
        image = axis.imshow(
            matrix,
            vmin=0.0,
            vmax=1.0,
            cmap="magma",
            interpolation="nearest",
            aspect="equal",
        )
        fraction = metric["threshold_pair_fractions"]["ge_0.95"]
        axis.set_title(
            f"{metric['registry_index']}. "
            f"P {metric['P_before_correlation_filter']:,}"
            f"→{metric['P_after_correlation_filter']:,}\n"
            f"pairs≥.95: {100*fraction:.1f}%",
            fontsize=8,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.flat[len(metrics) :]:
        axis.axis("off")
    if image is not None:
        fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            fraction=0.018,
            pad=0.01,
            label="Absolute Pearson correlation",
        )
    fig.suptitle(title, fontsize=15)
    fig.savefig(output / f"{filename}.png", dpi=210, bbox_inches="tight")
    fig.savefig(output / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def render_summary(metrics: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    index = np.asarray([item["registry_index"] for item in metrics])
    before = np.asarray(
        [item["P_before_correlation_filter"] for item in metrics], dtype=float
    )
    after = np.asarray(
        [item["P_after_correlation_filter"] for item in metrics], dtype=float
    )
    retained = after / before
    pair_fraction = np.asarray(
        [
            item["threshold_pair_fractions"]["ge_0.95"]
            for item in metrics
        ]
    )
    nonisolated = np.asarray(
        [
            item["degree_ge_0p95"]["nonisolated_feature_count"]
            / item["P_before_correlation_filter"]
            for item in metrics
        ]
    )
    q99 = np.asarray(
        [
            item["sample_abs_corr_quantiles"]["q99"]
            for item in metrics
        ]
    )
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.2), constrained_layout=True)
    axes[0].scatter(
        np.maximum(pair_fraction, 1e-7), retained, color="#0072B2", s=38
    )
    for position, dataset_index in enumerate(index):
        axes[0].annotate(
            str(dataset_index),
            (max(pair_fraction[position], 1e-7), retained[position]),
            xytext=(3, 2),
            textcoords="offset points",
            fontsize=6.5,
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Fraction of all pairs with |corr| ≥ 0.95")
    axes[0].set_ylabel("Retained feature fraction")

    axes[1].scatter(nonisolated, retained, color="#D55E00", s=38)
    axes[1].set_xlabel("Fraction of features with ≥1 neighbor at 0.95")
    axes[1].set_ylabel("Retained feature fraction")
    axes[1].set_yscale("log")

    order = np.argsort(retained)
    positions = np.arange(len(metrics))
    axes[2].bar(
        positions,
        retained[order],
        color="#009E73",
        width=0.85,
    )
    axes[2].set_xlabel("Datasets sorted by retained fraction")
    axes[2].set_ylabel("Retained feature fraction")
    axes[2].set_yscale("log")
    axes[2].set_xticks([])

    axes[3].scatter(q99, pair_fraction, color="#CC79A7", s=38)
    axes[3].set_xlabel("Approximate q99 of |corr|")
    axes[3].set_ylabel("Fraction of pairs with |corr| ≥ 0.95")
    axes[3].set_yscale("log")
    for axis in axes:
        axis.grid(False)
    fig.savefig(
        output / "prefilter_abs_corr_summary.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        output / "prefilter_abs_corr_summary.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
    log_pair = np.log10(np.maximum(pair_fraction, 1e-12))
    log_retained = np.log10(np.maximum(retained, 1e-12))
    return {
        "task_count": len(metrics),
        "median_P_before": float(np.median(before)),
        "median_P_after": float(np.median(after)),
        "median_retained_fraction": float(np.median(retained)),
        "median_pair_fraction_ge_0p95": float(np.median(pair_fraction)),
        "pair_fraction_ge_0p95_min": float(pair_fraction.min()),
        "pair_fraction_ge_0p95_max": float(pair_fraction.max()),
        "median_nonisolated_feature_fraction": float(np.median(nonisolated)),
        "log_pair_fraction_vs_log_retained_correlation": float(
            np.corrcoef(log_pair, log_retained)[0, 1]
        ),
        "most_aggressively_filtered": [
            {
                "registry_index": int(metrics[position]["registry_index"]),
                "task_id": metrics[position]["task_id"],
                "P_before": int(before[position]),
                "P_after": int(after[position]),
                "retained_fraction": float(retained[position]),
                "pair_fraction_ge_0p95": float(pair_fraction[position]),
            }
            for position in np.argsort(retained)[:10]
        ],
    }


def write_csv(metrics: list[dict[str, Any]], output: Path) -> None:
    fields = [
        "registry_index",
        "task_id",
        "N",
        "P_before_correlation_filter",
        "P_after_correlation_filter",
        "removed_fraction",
        "max_abs_corr",
        "mean_abs_corr",
        "pair_fraction_ge_0p90",
        "pair_fraction_ge_0p95",
        "pair_fraction_ge_0p99",
        "nonisolated_feature_count_ge_0p95",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(
                {
                    "registry_index": metric["registry_index"],
                    "task_id": metric["task_id"],
                    "N": metric["N"],
                    "P_before_correlation_filter": metric[
                        "P_before_correlation_filter"
                    ],
                    "P_after_correlation_filter": metric[
                        "P_after_correlation_filter"
                    ],
                    "removed_fraction": metric["removed_fraction"],
                    "max_abs_corr": metric["max_abs_corr"],
                    "mean_abs_corr": metric["mean_abs_corr"],
                    "pair_fraction_ge_0p90": metric[
                        "threshold_pair_fractions"
                    ]["ge_0.90"],
                    "pair_fraction_ge_0p95": metric[
                        "threshold_pair_fractions"
                    ]["ge_0.95"],
                    "pair_fraction_ge_0p99": metric[
                        "threshold_pair_fractions"
                    ]["ge_0.99"],
                    "nonisolated_feature_count_ge_0p95": metric[
                        "degree_ge_0p95"
                    ]["nonisolated_feature_count"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    figure_root = args.root / "figures"
    detail_root = figure_root / "per_dataset"
    report_root = args.root / "reports"
    figure_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(args.root)
    render_details(metrics, args.root, detail_root)
    render_overview(
        metrics,
        root=args.root,
        output=figure_root,
        array_key="greedy_abs_corr",
        filename="prefilter_abs_corr_greedy_chain_overview",
        title=(
            "Pre-filter feature |correlation|: high-correlation greedy chain\n"
            "Outer-training rows only; up to 128 features per dataset"
        ),
    )
    render_overview(
        metrics,
        root=args.root,
        output=figure_root,
        array_key="global_abs_corr",
        filename="prefilter_abs_corr_global_sample_greedy_overview",
        title=(
            "Pre-filter feature |correlation|: deterministic global sample, "
            "greedy ordering\nOuter-training rows only; up to 128 features per dataset"
        ),
    )
    summary = render_summary(metrics, figure_root)
    write_csv(metrics, report_root / "prefilter_abs_corr_task_summary.csv")
    (report_root / "prefilter_abs_corr_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
