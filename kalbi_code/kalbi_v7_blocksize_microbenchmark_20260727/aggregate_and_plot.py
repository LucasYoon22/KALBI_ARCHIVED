#!/usr/bin/env python3
"""Validate, aggregate, and plot all isolated benchmark fragments."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .common import (
    METHOD_LABELS,
    METHODS,
    OUTPUT,
    PACKAGE,
    atomic_json_dump,
    sha256_file,
)
from .metrics import summarize_timing


COLORS = {
    "pg10": "#0072B2",
    "block4096": "#D55E00",
    "block256": "#009E73",
    "adaptive": "#CC79A7",
}
MARKERS = {"low": "o", "mid": "s", "high": "^"}
METHOD_ORDER = {name: index for index, name in enumerate(METHODS)}


def median(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    return statistics.median(materialized)


def percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_fragments() -> list[dict[str, Any]]:
    paths = sorted((OUTPUT / "fragments").glob("task*_rep*.json"))
    if len(paths) != 6 * 4 * 5:
        raise RuntimeError(f"expected 120 fragments, found {len(paths)}")
    fragments = [json.loads(path.read_text()) for path in paths]
    bad = [item for item in fragments if item["status"] != "OK"]
    if bad:
        raise RuntimeError(
            "full aggregation requires all cells OK: "
            + ", ".join(
                f"T{item['task_index']}:{item['method']}:"
                f"R{item['repetition']}={item['status']}"
                for item in bad
            )
        )
    keys = {
        (int(item["task_index"]), item["method"], int(item["repetition"]))
        for item in fragments
    }
    if len(keys) != len(fragments):
        raise RuntimeError("duplicate task/method/repetition fragment")
    return fragments


def validate_contracts(fragments: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for task_index in range(1, 7):
        cells = [
            item for item in fragments if int(item["task_index"]) == task_index
        ]
        hashes = {
            json.dumps(item["input_tensor_hashes"], sort_keys=True)
            for item in cells
        }
        beta_differences = {
            float(item["beta_initial_max_abs_difference"]) for item in cells
        }
        if len(hashes) != 1 or beta_differences != {0.0}:
            raise RuntimeError(f"task {task_index} common-input contract failed")
        quality_cells = [item for item in cells if item["repetition"] == 0]
        if len(quality_cells) != 4 or any(
            item["quality"] is None for item in quality_cells
        ):
            raise RuntimeError(f"task {task_index} quality fragment missing")
        report[str(task_index)] = {
            "all_method_input_hashes_identical": True,
            "beta_initial_max_abs_difference": 0.0,
            "tensor_hashes": quality_cells[0]["input_tensor_hashes"],
        }
    return report


def quality_map(
    fragments: list[dict[str, Any]],
) -> dict[tuple[int, str], dict[str, Any]]:
    return {
        (int(item["task_index"]), item["method"]): item
        for item in fragments
        if int(item["repetition"]) == 0
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_rows(
    fragments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    quality = quality_map(fragments)
    query_bank = json.loads((OUTPUT / "query_bank.json").read_text())
    raw_lambdas = [
        float(item["raw_lambda"]) for item in query_bank["queries"]
    ]
    per_query: list[dict[str, Any]] = []
    for task_index in range(1, 7):
        pg = quality[(task_index, "pg10")]["quality"]["final_objective"]
        for method in METHODS:
            cell = quality[(task_index, method)]
            metrics = cell["quality"]
            task = cell["task"]
            for query_index, final_objective in enumerate(
                metrics["final_objective"]
            ):
                pg_objective = float(pg[query_index])
                per_query.append(
                    {
                        "task_index": task_index,
                        "p_bin": task["p_bin"],
                        "n": int(task["n"]),
                        "p": int(task["p"]),
                        "p_over_n": float(task["p_over_n"]),
                        "outcome_source": task["outcome_source"],
                        "method": method,
                        "query_index": query_index,
                        "raw_lambda": raw_lambdas[query_index],
                        "f_initial": float(
                            metrics["objective_path"][query_index][0]
                        ),
                        "f_final": float(final_objective),
                        "absolute_objective_reduction": float(
                            metrics["absolute_objective_reduction"][query_index]
                        ),
                        "relative_objective_reduction": float(
                            metrics["relative_objective_reduction"][query_index]
                        ),
                        "normalized_objective_gap_vs_pg10": (
                            float(final_objective) - pg_objective
                        )
                        / max(abs(pg_objective), 1e-12),
                        "kkt_top32": float(
                            metrics["kkt"]["kkt_top32"][query_index]
                        ),
                        "kkt_full_rms": float(
                            metrics["kkt"]["kkt_full_rms"][query_index]
                        ),
                        "kkt_nonzero_rms": float(
                            metrics["kkt"]["kkt_nonzero_rms"][query_index]
                        ),
                        "kkt_zero_rms": float(
                            metrics["kkt"]["kkt_zero_rms"][query_index]
                        ),
                        "kkt_max_abs": float(
                            metrics["kkt"]["kkt_max_abs"][query_index]
                        ),
                        "nnz_1e_6": int(metrics["nnz_1e_6"][query_index]),
                        "nnz_1e_4": int(metrics["nnz_1e_4"][query_index]),
                        "coefficient_rms_change": float(
                            metrics["coefficient_rms_change"][query_index]
                        ),
                        "backtrack_count": float(
                            metrics["backtrack_count"][query_index]
                        ),
                        "rejected_block_fraction": float(
                            metrics["rejected_block_fraction"][query_index]
                        ),
                        "monotonicity_violation_count": float(
                            metrics["monotonicity_violation_count"][query_index]
                        ),
                        "block_update_count": float(
                            metrics["block_update_count"][query_index]
                        ),
                    }
                )

    per_task: list[dict[str, Any]] = []
    for task_index in range(1, 7):
        for method in METHODS:
            cells = sorted(
                (
                    item
                    for item in fragments
                    if int(item["task_index"]) == task_index
                    and item["method"] == method
                ),
                key=lambda item: int(item["repetition"]),
            )
            query_rows = [
                row
                for row in per_query
                if int(row["task_index"]) == task_index
                and row["method"] == method
            ]
            task = cells[0]["task"]
            forward = [
                item["timings"]["refinement_forward_seconds"] for item in cells
            ]
            backward = [
                item["timings"]["refinement_forward_backward_seconds"]
                for item in cells
            ]
            e2e = [
                item["timings"]["end_to_end_forward_backward_seconds"]
                for item in cells
            ]
            baseline = [
                item
                for item in fragments
                if int(item["task_index"]) == task_index
                and item["method"] == "block4096"
            ]
            baseline_by_rep = {
                int(item["repetition"]): float(
                    item["timings"]["refinement_forward_seconds"]
                )
                for item in baseline
            }
            paired_ratios = [
                float(item["timings"]["refinement_forward_seconds"])
                / baseline_by_rep[int(item["repetition"])]
                for item in cells
            ]
            per_task.append(
                {
                    "task_index": task_index,
                    "p_bin": task["p_bin"],
                    "n": int(task["n"]),
                    "p": int(task["p"]),
                    "p_over_n": float(task["p_over_n"]),
                    "outcome_source": task["outcome_source"],
                    "method": method,
                    "block_size": cells[0]["block_size"] or 0,
                    "block_count_per_sweep": cells[0][
                        "block_count_per_sweep"
                    ],
                    "median_final_objective": median(
                        row["f_final"] for row in query_rows
                    ),
                    "median_relative_objective_reduction": median(
                        row["relative_objective_reduction"]
                        for row in query_rows
                    ),
                    "median_objective_gap_vs_pg10": median(
                        row["normalized_objective_gap_vs_pg10"]
                        for row in query_rows
                    ),
                    "p90_objective_gap_vs_pg10": percentile(
                        (
                            row["normalized_objective_gap_vs_pg10"]
                            for row in query_rows
                        ),
                        0.9,
                    ),
                    "median_kkt_top32": median(
                        row["kkt_top32"] for row in query_rows
                    ),
                    "median_kkt_full_rms": median(
                        row["kkt_full_rms"] for row in query_rows
                    ),
                    "median_nnz_1e_6": median(
                        row["nnz_1e_6"] for row in query_rows
                    ),
                    "forward_median_seconds": median(forward),
                    "forward_mean_seconds": statistics.mean(forward),
                    "forward_p90_seconds": percentile(forward, 0.9),
                    "forward_std_seconds": statistics.stdev(forward),
                    "forward_cv": statistics.stdev(forward)
                    / statistics.mean(forward),
                    "forward_backward_median_seconds": median(backward),
                    "forward_backward_mean_seconds": statistics.mean(backward),
                    "forward_backward_p90_seconds": percentile(backward, 0.9),
                    "forward_backward_std_seconds": statistics.stdev(backward),
                    "forward_backward_cv": statistics.stdev(backward)
                    / statistics.mean(backward),
                    "end_to_end_median_seconds": median(e2e),
                    "end_to_end_mean_seconds": statistics.mean(e2e),
                    "end_to_end_p90_seconds": percentile(e2e, 0.9),
                    "end_to_end_std_seconds": statistics.stdev(e2e),
                    "end_to_end_cv": statistics.stdev(e2e)
                    / statistics.mean(e2e),
                    "paired_forward_ratio_vs_block4096_median": median(
                        paired_ratios
                    ),
                    "peak_allocated_gib": max(
                        max(
                            item["timings"][
                                "refinement_forward_peak_allocated_bytes"
                            ],
                            item["timings"][
                                "refinement_backward_peak_allocated_bytes"
                            ],
                            item["timings"][
                                "end_to_end_peak_allocated_bytes"
                            ],
                        )
                        / (1024.0**3)
                        for item in cells
                    ),
                    "peak_reserved_gib": max(
                        max(
                            item["timings"][
                                "refinement_forward_peak_reserved_bytes"
                            ],
                            item["timings"][
                                "refinement_backward_peak_reserved_bytes"
                            ],
                            item["timings"][
                                "end_to_end_peak_reserved_bytes"
                            ],
                        )
                        / (1024.0**3)
                        for item in cells
                    ),
                }
            )
    return per_query, per_task


def method_summary(
    rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    method: str,
    p_bin: str | None,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["method"] == method
        and (p_bin is None or row["p_bin"] == p_bin)
    ]
    selected_tasks = [
        row
        for row in task_rows
        if row["method"] == method
        and (p_bin is None or row["p_bin"] == p_bin)
    ]
    return {
        "observations": len(selected),
        "tasks": len(selected_tasks),
        "median_final_objective": median(
            row["f_final"] for row in selected
        ),
        "median_relative_objective_reduction": median(
            row["relative_objective_reduction"] for row in selected
        ),
        "median_objective_gap_vs_pg10": median(
            row["normalized_objective_gap_vs_pg10"] for row in selected
        ),
        "p90_objective_gap_vs_pg10": percentile(
            (row["normalized_objective_gap_vs_pg10"] for row in selected),
            0.9,
        ),
        "median_kkt_top32": median(row["kkt_top32"] for row in selected),
        "median_kkt_full_rms": median(
            row["kkt_full_rms"] for row in selected
        ),
        "median_forward_seconds": median(
            row["forward_median_seconds"] for row in selected_tasks
        ),
        "median_forward_ratio_vs_block4096": median(
            row["paired_forward_ratio_vs_block4096_median"]
            for row in selected_tasks
        ),
        "peak_allocated_gib": max(
            row["peak_allocated_gib"] for row in selected_tasks
        ),
    }


def aggregate_summary(
    per_query: list[dict[str, Any]],
    per_task: list[dict[str, Any]],
) -> dict[str, Any]:
    overall = {
        method: method_summary(per_query, per_task, method, None)
        for method in METHODS
    }
    by_bin = {
        p_bin: {
            method: method_summary(per_query, per_task, method, p_bin)
            for method in METHODS
        }
        for p_bin in ("low", "mid", "high")
    }
    b4096 = overall["block4096"]
    b256 = overall["block256"]
    adaptive = overall["adaptive"]
    quality_gain_256 = (
        b4096["median_final_objective"] - b256["median_final_objective"]
    )
    quality_gain_adaptive = (
        b4096["median_final_objective"]
        - adaptive["median_final_objective"]
    )
    retention = (
        quality_gain_adaptive / quality_gain_256
        if abs(quality_gain_256) > 1e-12
        else None
    )
    frontier: list[str] = []
    for method in METHODS:
        current = overall[method]
        dominated = False
        for other in METHODS:
            if other == method:
                continue
            candidate = overall[other]
            no_worse = (
                candidate["median_final_objective"]
                <= current["median_final_objective"]
                and candidate["median_kkt_top32"]
                <= current["median_kkt_top32"]
                and candidate["median_forward_seconds"]
                <= current["median_forward_seconds"]
            )
            strictly_better = (
                candidate["median_final_objective"]
                < current["median_final_objective"]
                or candidate["median_kkt_top32"]
                < current["median_kkt_top32"]
                or candidate["median_forward_seconds"]
                < current["median_forward_seconds"]
            )
            dominated = dominated or (no_worse and strictly_better)
        if not dominated:
            frontier.append(method)
    p_values = [int(row["p"]) for row in per_task]
    runtime_by_bin = {
        p_bin: by_bin[p_bin]["block256"][
            "median_forward_ratio_vs_block4096"
        ]
        for p_bin in ("low", "mid", "high")
    }
    return {
        "overall": overall,
        "by_p_bin": by_bin,
        "decision_metrics": {
            "block256_objective_change_vs_block4096_fraction": (
                b256["median_final_objective"]
                - b4096["median_final_objective"]
            )
            / max(abs(b4096["median_final_objective"]), 1e-12),
            "block256_kkt_change_vs_block4096_fraction": (
                b256["median_kkt_top32"] - b4096["median_kkt_top32"]
            )
            / max(abs(b4096["median_kkt_top32"]), 1e-12),
            "block256_runtime_multiple_vs_block4096": b256[
                "median_forward_ratio_vs_block4096"
            ],
            "block256_runtime_multiple_by_p_bin": runtime_by_bin,
            "adaptive_quality_gain_retention_vs_block256": retention,
            "block3_p90_quality_gap_within_1pct": {
                method: overall[method]["p90_objective_gap_vs_pg10"] <= 0.01
                for method in ("block4096", "block256", "adaptive")
            },
            "pareto_frontier": frontier,
            "maximum_observed_p": max(p_values),
            "near_100k_tested": max(p_values) >= 80_000,
            "near_100k_answer": (
                "not identified by the frozen six-task sample"
                if max(p_values) < 80_000
                else "measured"
            ),
        },
    }


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    figures = OUTPUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_objectives(quality: dict[tuple[int, str], dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), sharey=False)
    for task_index, axis in enumerate(axes.flat, 1):
        task = quality[(task_index, "pg10")]["task"]
        for method in METHODS:
            path = np.asarray(
                quality[(task_index, method)]["quality"]["objective_path"],
                dtype=float,
            )
            normalized = path / np.maximum(np.abs(path[:, :1]), 1e-12)
            x = np.arange(normalized.shape[1])
            axis.plot(
                x,
                np.median(normalized, axis=0),
                color=COLORS[method],
                linewidth=1.8,
                marker="o",
                markersize=3,
                label=METHOD_LABELS[method],
            )
            axis.fill_between(
                x,
                np.quantile(normalized, 0.25, axis=0),
                np.quantile(normalized, 0.75, axis=0),
                color=COLORS[method],
                alpha=0.10,
                linewidth=0,
            )
        axis.text(
            0.03,
            0.05,
            f"Task {task_index}: N={task['n']:,}, P={task['p']:,}",
            transform=axis.transAxes,
            fontsize=8,
        )
        axis.set_xlabel("Refinement update")
        axis.set_ylabel(r"$F/F_{\mathrm{initial}}$")
        axis.grid(axis="y", alpha=0.18, linewidth=0.6)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, "objective_trajectories")


def plot_pareto(per_task: list[dict[str, Any]]) -> None:
    fig, axis = plt.subplots(figsize=(6.2, 4.2))
    used_labels: set[str] = set()
    for row in per_task:
        label = METHOD_LABELS[row["method"]]
        axis.scatter(
            row["forward_median_seconds"],
            row["median_objective_gap_vs_pg10"],
            s=48,
            color=COLORS[row["method"]],
            marker=MARKERS[row["p_bin"]],
            edgecolor="white",
            linewidth=0.5,
            label=label if label not in used_labels else None,
        )
        used_labels.add(label)
    axis.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
    axis.set_xscale("log")
    axis.set_yscale("symlog", linthresh=0.01, linscale=1.0, base=10)
    axis.set_ylim(-0.2, 5.0)
    axis.set_xlabel("Refinement-only forward time (s)")
    axis.set_ylabel("Normalized objective gap vs PG10")
    axis.grid(alpha=0.18, linewidth=0.6)
    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]
    bin_handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[p_bin],
            linestyle="none",
            markerfacecolor="#777777",
            markeredgecolor="#777777",
            color="none",
            label=f"{p_bin.capitalize()} P",
        )
        for p_bin in ("low", "mid", "high")
    ]
    first_legend = axis.legend(
        handles=method_handles,
        frameon=False,
        ncol=2,
        loc="upper right",
    )
    axis.add_artist(first_legend)
    axis.legend(
        handles=bin_handles,
        frameon=False,
        ncol=3,
        loc="lower right",
    )
    save_figure(fig, "objective_runtime_pareto")


def grouped_bar(
    per_task: list[dict[str, Any]],
    field: str,
    ylabel: str,
    stem: str,
    *,
    log_y: bool,
) -> None:
    fig, axis = plt.subplots(figsize=(8.0, 4.0))
    x = np.arange(6)
    width = 0.19
    for offset, method in enumerate(METHODS):
        rows = sorted(
            (row for row in per_task if row["method"] == method),
            key=lambda row: int(row["task_index"]),
        )
        axis.bar(
            x + (offset - 1.5) * width,
            [row[field] for row in rows],
            width,
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
    axis.set_xticks(x, [f"Task {index}" for index in range(1, 7)])
    axis.set_ylabel(ylabel)
    if log_y:
        axis.set_yscale("log")
    axis.grid(axis="y", alpha=0.18, linewidth=0.6)
    axis.legend(frameon=False, ncol=2)
    save_figure(fig, stem)


def plot_runtime(per_task: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    x = np.arange(6)
    width = 0.19
    fields = (
        ("forward_median_seconds", "Forward time (s)"),
        ("forward_backward_median_seconds", "Forward + backward time (s)"),
    )
    for axis, (field, ylabel) in zip(axes, fields):
        for offset, method in enumerate(METHODS):
            rows = sorted(
                (row for row in per_task if row["method"] == method),
                key=lambda row: int(row["task_index"]),
            )
            axis.bar(
                x + (offset - 1.5) * width,
                [row[field] for row in rows],
                width,
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_xticks(x, [f"T{index}" for index in range(1, 7)])
        axis.set_ylabel(ylabel)
        axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.18, linewidth=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(fig, "forward_backward_runtime")


def plot_blocks(per_task: list[dict[str, Any]]) -> None:
    rows = sorted(
        (row for row in per_task if row["method"] == "adaptive"),
        key=lambda row: int(row["p"]),
    )
    p = np.asarray([row["p"] for row in rows])
    block_size = np.asarray([row["block_size"] for row in rows])
    block_count = np.asarray([row["block_count_per_sweep"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.5))
    axes[0].plot(p, block_size, "o-", color=COLORS["adaptive"])
    axes[0].set_xlabel("Feature count P")
    axes[0].set_ylabel("Adaptive block size")
    axes[1].plot(p, block_count, "o-", color=COLORS["block256"])
    axes[1].set_xlabel("Feature count P")
    axes[1].set_ylabel("Blocks per sweep")
    for axis in axes:
        axis.set_xscale("log")
        axis.grid(alpha=0.18, linewidth=0.6)
    fig.tight_layout()
    save_figure(fig, "block_size_count_vs_p")


def source_manifest() -> dict[str, Any]:
    paths = sorted(
        path
        for pattern in ("*.py", "*.sh", "*.conf")
        for path in PACKAGE.glob(pattern)
    )
    critical = [
        Path(
            "/users/minsuk/fm-lm-lambda/"
            "kalbi_v7_fullp_blockprox3_v23_k32s32_p100k_effbatch32/"
            "block_prox_v7.py"
        ),
        Path(
            "/users/minsuk/fm-lm-lambda/"
            "kalbi_v7_fullp_blockprox3_v23_k32s32_p100k_effbatch32/"
            "model.py"
        ),
    ]
    return {
        str(path): sha256_file(path)
        for path in paths + critical
        if path.is_file()
    }


def write_readme(summary: dict[str, Any]) -> None:
    decision = summary["decision_metrics"]
    runtime_bins = decision["block256_runtime_multiple_by_p_bin"]
    retention = decision["adaptive_quality_gain_retention_vs_block256"]
    retention_text = (
        "undefined"
        if retention is None
        else f"{100.0 * retention:.3g}%"
    )
    gap_text = ", ".join(
        f"{method}={'yes' if passed else 'no'}"
        for method, passed in decision[
            "block3_p90_quality_gap_within_1pct"
        ].items()
    )
    lines = [
        "# KALBI V7 Block-Size Microbenchmark",
        "",
        "This directory contains the immutable six-task block-size benchmark.",
        "The benchmark itself did not modify production training or rolling-data state.",
        "An unrelated process stopped the V13 trainer and started V24 GPU work during",
        "the timing window, so paired task/repetition ratios are the primary timing",
        "result; absolute seconds have limited comparability.",
        "",
        "## Frozen Design",
        "",
        "- Six tasks selected before performance inspection from catalog epoch 66.",
        "- Eight fixed raw-lambda/Direct96 queries per task.",
        "- Five isolated timing repetitions with deterministically shuffled method order.",
        "- Common standardized X/y and byte-identical frozen beta_initial per task.",
        "",
        "## Decision Snapshot",
        "",
        f"- B=256 objective change vs B=4096: "
        f"{100.0 * decision['block256_objective_change_vs_block4096_fraction']:.4g}%.",
        f"- B=256 top-32 KKT change vs B=4096: "
        f"{100.0 * decision['block256_kkt_change_vs_block4096_fraction']:.4g}%.",
        f"- B=256 forward-time multiple vs B=4096: "
        f"{decision['block256_runtime_multiple_vs_block4096']:.4g}.",
        f"- B=256 runtime multiple by P bin: low "
        f"{runtime_bins['low']:.4g}, mid {runtime_bins['mid']:.4g}, "
        f"high {runtime_bins['high']:.4g}.",
        f"- Adaptive retention of the B=256 objective gain: {retention_text}.",
        f"- Block-3 p90 objective gap within 1% of PG10: {gap_text}.",
        f"- Maximum sampled P: {decision['maximum_observed_p']:,}.",
        f"- Near-P=100K speed claim: {decision['near_100k_answer']}.",
        f"- Pareto frontier: {', '.join(decision['pareto_frontier'])}.",
        "",
        "See `aggregate_summary.json`, the two CSV files, and `figures/` for "
        "the complete results.",
        "",
    ]
    (OUTPUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_plotting()
    fragments = load_fragments()
    contract = validate_contracts(fragments)
    per_query, per_task = build_rows(fragments)
    write_csv(OUTPUT / "per_query_metrics.csv", per_query)
    write_csv(OUTPUT / "per_task_summary.csv", per_task)
    summary = aggregate_summary(per_query, per_task)
    summary["common_input_contract"] = contract
    atomic_json_dump(summary, OUTPUT / "aggregate_summary.json")
    quality = quality_map(fragments)
    trajectories: dict[str, np.ndarray] = {}
    for task_index in range(1, 7):
        for method in METHODS:
            trajectories[f"task{task_index:02d}_{method}"] = np.asarray(
                quality[(task_index, method)]["quality"]["objective_path"],
                dtype=np.float64,
            )
    np.savez_compressed(OUTPUT / "objective_trajectories.npz", **trajectories)
    timing_environment = {
        "production_gpu_sharing": "changed_during_run",
        "paired_design": True,
        "primary_timing_interpretation": (
            "Use paired task/repetition ratios. Absolute times have limited "
            "comparability because unrelated GPU workloads changed while the "
            "benchmark was running."
        ),
        "benchmark_window_utc": {
            "first_fragment": "2026-07-27T09:22:15Z",
            "run_summary": "2026-07-27T10:21:21Z",
        },
        "external_workload_events_utc": [
            {
                "time": "2026-07-27T09:32:18Z",
                "event": "kalbi_v24_scaler_builder started",
            },
            {
                "time": "2026-07-27T09:35:37Z",
                "event": "kalbi_v24_scaler_builder exited",
            },
            {
                "time": "2026-07-27T09:50:12Z",
                "event": (
                    "kalbi_v13_beta_trainer received an external SIGTERM near "
                    "step 42920; this benchmark did not stop or restart it"
                ),
            },
            {
                "time": "2026-07-27T09:50:32Z",
                "event": (
                    "kalbi_v24_selector_trainer began sharing all eight GPUs "
                    "and was externally restarted during the remaining window"
                ),
            },
        ],
        "warmups_per_isolated_cell": 2,
        "measured_repetitions": 5,
        "timing_summary": {
            method: {
                metric: summarize_timing(
                    [
                        float(item["timings"][metric])
                        for item in fragments
                        if item["method"] == method
                    ]
                )
                for metric in (
                    "refinement_forward_seconds",
                    "refinement_forward_backward_seconds",
                    "end_to_end_forward_backward_seconds",
                )
            }
            for method in METHODS
        },
        "gpu_names": sorted({item["gpu_name"] for item in fragments}),
        "torch_versions": sorted(
            {str(item["torch_version"]) for item in fragments}
        ),
    }
    atomic_json_dump(
        timing_environment, OUTPUT / "timing_environment.json"
    )
    atomic_json_dump(source_manifest(), OUTPUT / "source_sha256_manifest.json")
    plot_objectives(quality)
    plot_pareto(per_task)
    grouped_bar(
        per_task,
        "median_kkt_top32",
        "Median top-32 KKT residual",
        "task_kkt_grouped",
        log_y=True,
    )
    plot_runtime(per_task)
    plot_blocks(per_task)
    write_readme(summary)
    print(json.dumps(summary["decision_metrics"], indent=2))


if __name__ == "__main__":
    main()
