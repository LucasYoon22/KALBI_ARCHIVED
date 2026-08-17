#!/usr/bin/env python3
"""Aggregate the isolated KALBI V8 adaptive-T10 preflight benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


METHODS = ("pg10", "adaptive_t3", "adaptive_t10")
METHOD_LABELS = {
    "pg10": "Source V6 PG10",
    "adaptive_t3": "Adaptive block-prox T3",
    "adaptive_t10": "Adaptive block-prox T10",
}
OUTPUT = Path(
    os.environ.get(
        "KALBI_V8_BENCHMARK_OUTPUT",
        "/users/minsuk/fm-lm-lambda/evaluations/"
        "kalbi_v8_adaptive_t10_betaonly_preflight_20260727",
    )
)


COLORS = {
    "pg10": "#0072B2",
    "adaptive_t3": "#E69F00",
    "adaptive_t10": "#009E73",
}


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def median(values: Iterable[float]) -> float:
    return statistics.median(float(value) for value in values)


def percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot take percentile of empty values")
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_fragments() -> list[dict[str, Any]]:
    paths = sorted((OUTPUT / "fragments").glob("task*_rep*.json"))
    expected = 6 * len(METHODS) * 5
    if len(paths) != expected:
        raise RuntimeError(f"expected {expected} fragments, found {len(paths)}")
    fragments = [json.loads(path.read_text()) for path in paths]
    bad = [item for item in fragments if item.get("status") != "OK"]
    if bad:
        labels = [
            f"T{item.get('task_index')}:{item.get('method')}:"
            f"R{item.get('repetition')}={item.get('status')}"
            for item in bad
        ]
        raise RuntimeError("non-OK benchmark fragments: " + ", ".join(labels))
    keys = {
        (int(item["task_index"]), item["method"], int(item["repetition"]))
        for item in fragments
    }
    if len(keys) != expected:
        raise RuntimeError("duplicate or missing task/method/repetition cells")
    return fragments


def quality_cells(
    fragments: list[dict[str, Any]],
) -> dict[tuple[int, str], dict[str, Any]]:
    cells = {
        (int(item["task_index"]), item["method"]): item
        for item in fragments
        if int(item["repetition"]) == 0
    }
    if len(cells) != 6 * len(METHODS):
        raise RuntimeError("quality cells are incomplete")
    return cells


def validate_contracts(
    fragments: list[dict[str, Any]],
    quality: dict[tuple[int, str], dict[str, Any]],
) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for task_index in range(1, 7):
        task_cells = [
            item for item in fragments if int(item["task_index"]) == task_index
        ]
        input_hashes = {
            json.dumps(item["input_tensor_hashes"], sort_keys=True)
            for item in task_cells
        }
        initial_errors = {
            float(item["beta_initial_max_abs_difference"])
            for item in task_cells
        }
        if len(input_hashes) != 1 or initial_errors != {0.0}:
            raise RuntimeError(f"task {task_index} common-input audit failed")
        t3 = quality[(task_index, "adaptive_t3")]["quality"]
        t10 = quality[(task_index, "adaptive_t10")]["quality"]
        if t3["beta_path_sha256"] != t10["beta_path_sha256"][:4]:
            raise RuntimeError(
                f"task {task_index} T3 does not equal first three T10 sweeps"
            )
        t3_objective = np.asarray(t3["objective_path"], dtype=np.float64)
        t10_first3 = np.asarray(t10["objective_path"], dtype=np.float64)[:, :4]
        trajectory_error = float(np.max(np.abs(t3_objective - t10_first3)))
        if trajectory_error != 0.0:
            raise RuntimeError(
                f"task {task_index} T3/T10 objective prefix drift "
                f"{trajectory_error}"
            )
        for method in METHODS:
            metrics = quality[(task_index, method)]["quality"]
            trajectory = np.asarray(metrics["objective_path"], dtype=np.float64)
            if not np.isfinite(trajectory).all():
                raise RuntimeError(
                    f"task {task_index} {method} has nonfinite objective"
                )
            tolerance = 1e-6 * np.maximum(1.0, np.abs(trajectory[:, :-1]))
            if np.any(trajectory[:, 1:] > trajectory[:, :-1] + tolerance):
                raise RuntimeError(
                    f"task {task_index} {method} objective increased"
                )
        tasks[str(task_index)] = {
            "common_input_hashes": True,
            "beta_initial_exact_reproduction": True,
            "adaptive_t3_equals_t10_first_three_sweeps": True,
            "trajectory_max_abs_error": trajectory_error,
        }
    return {
        "status": "PASS",
        "tasks": tasks,
        "all_cells_ok": True,
        "all_trajectories_finite_monotone": True,
    }


def build_quality_rows(
    quality: dict[tuple[int, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    query_bank = json.loads((OUTPUT / "query_bank.json").read_text())
    raw_lambdas = [
        float(query["raw_lambda"]) for query in query_bank["queries"]
    ]
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for task_index in range(1, 7):
        pg = quality[(task_index, "pg10")]["quality"]
        pg_objective = np.asarray(pg["final_objective"], dtype=np.float64)
        for method in METHODS:
            cell = quality[(task_index, method)]
            metrics = cell["quality"]
            task = cell["task"]
            final = np.asarray(metrics["final_objective"], dtype=np.float64)
            gap = (final - pg_objective) / np.maximum(
                np.abs(pg_objective), 1e-12
            )
            for query_index, raw_lambda in enumerate(raw_lambdas):
                rows.append(
                    {
                        "task_index": task_index,
                        "p_bin": task["p_bin"],
                        "n": int(task["n"]),
                        "p": int(task["p"]),
                        "p_over_n": float(task["p_over_n"]),
                        "outcome_source": task["outcome_source"],
                        "method": method,
                        "query_index": query_index,
                        "raw_lambda": raw_lambda,
                        "initial_objective": float(
                            metrics["objective_path"][query_index][0]
                        ),
                        "final_objective": float(final[query_index]),
                        "relative_objective_reduction": float(
                            metrics["relative_objective_reduction"][query_index]
                        ),
                        "normalized_objective_gap_vs_pg10": float(
                            gap[query_index]
                        ),
                        "kkt_top32": float(
                            metrics["kkt"]["kkt_top32"][query_index]
                        ),
                        "kkt_full_rms": float(
                            metrics["kkt"]["kkt_full_rms"][query_index]
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
                    }
                )
            summaries.append(
                {
                    "task_index": task_index,
                    "p_bin": task["p_bin"],
                    "n": int(task["n"]),
                    "p": int(task["p"]),
                    "method": method,
                    "block_size": int(cell["block_size"] or 0),
                    "blocks_per_sweep": int(
                        cell["block_count_per_sweep"]
                    ),
                    "sweeps": int(cell["sweeps"]),
                    "median_final_objective": median(final),
                    "median_relative_objective_reduction": median(
                        metrics["relative_objective_reduction"]
                    ),
                    "median_objective_gap_vs_pg10": median(gap),
                    "p90_objective_gap_vs_pg10": percentile(gap, 0.9),
                    "median_kkt_top32": median(
                        metrics["kkt"]["kkt_top32"]
                    ),
                    "median_kkt_full_rms": median(
                        metrics["kkt"]["kkt_full_rms"]
                    ),
                    "median_nnz_1e_4": median(metrics["nnz_1e_4"]),
                    "median_coefficient_rms_change": median(
                        metrics["coefficient_rms_change"]
                    ),
                    "exact_prox_call_count": int(
                        metrics["exact_prox_call_count"]
                    ),
                    "backtracking_evaluation_count": int(
                        metrics["backtracking_evaluation_count"]
                    ),
                }
            )
    return rows, summaries


def summarize_timing(
    fragments: list[dict[str, Any]],
) -> dict[str, Any]:
    mode_fields = {
        "refinement_forward": "refinement_forward_seconds",
        "refinement_forward_backward":
            "refinement_forward_backward_seconds",
        "end_to_end_beta_microstep":
            "end_to_end_forward_backward_seconds",
    }
    result: dict[str, Any] = {
        "measurement_contract": {
            "independent_process_per_cell": True,
            "warmups_per_process": 2,
            "measured_repetitions": 5,
            "paired_ratio_priority_under_contention": True,
        },
        "by_task": {},
        "by_p_bin": {},
        "overall": {},
    }
    for task_index in range(1, 7):
        result["by_task"][str(task_index)] = {}
        for method in METHODS:
            cells = [
                item
                for item in fragments
                if int(item["task_index"]) == task_index
                and item["method"] == method
            ]
            summary: dict[str, Any] = {}
            for label, field in mode_fields.items():
                values = [float(cell["timings"][field]) for cell in cells]
                summary[label] = {
                    "median_seconds": median(values),
                    "mean_seconds": statistics.mean(values),
                    "p90_seconds": percentile(values, 0.9),
                    "minimum_seconds": min(values),
                    "maximum_seconds": max(values),
                }
            summary["peak_allocated_gib"] = max(
                max(
                    int(cell["timings"][key])
                    for key in cell["timings"]
                    if key.endswith("peak_allocated_bytes")
                )
                / (1024.0**3)
                for cell in cells
            )
            summary["peak_reserved_gib"] = max(
                max(
                    int(cell["timings"][key])
                    for key in cell["timings"]
                    if key.endswith("peak_reserved_bytes")
                )
                / (1024.0**3)
                for cell in cells
            )
            result["by_task"][str(task_index)][method] = summary
    for p_bin in ("low", "mid", "high", "all"):
        result["by_p_bin"][p_bin] = {}
        for method in METHODS:
            cells = [
                item
                for item in fragments
                if item["method"] == method
                and (p_bin == "all" or item["task"]["p_bin"] == p_bin)
            ]
            summary = {}
            for label, field in mode_fields.items():
                values = [float(cell["timings"][field]) for cell in cells]
                summary[label] = {
                    "median_seconds": median(values),
                    "p90_seconds": percentile(values, 0.9),
                }
            result["by_p_bin"][p_bin][method] = summary
        for method in ("adaptive_t3", "adaptive_t10"):
            for baseline in ("pg10", "adaptive_t3"):
                if method == baseline:
                    continue
                ratio_key = f"{method}_over_{baseline}"
                ratios: dict[str, float] = {}
                paired: dict[str, dict[str, float]] = {}
                for label in mode_fields:
                    numerator = result["by_p_bin"][p_bin][method][label][
                        "median_seconds"
                    ]
                    denominator = result["by_p_bin"][p_bin][baseline][label][
                        "median_seconds"
                    ]
                    ratios[label] = numerator / denominator
                    field = mode_fields[label]
                    indexed = {
                        (
                            int(cell["task_index"]),
                            int(cell["repetition"]),
                            cell["method"],
                        ): float(cell["timings"][field])
                        for cell in fragments
                        if p_bin == "all" or cell["task"]["p_bin"] == p_bin
                    }
                    paired_values = [
                        indexed[(task_index, repetition, method)]
                        / indexed[(task_index, repetition, baseline)]
                        for task_index, repetition, observed_method in indexed
                        if observed_method == method
                    ]
                    paired[label] = {
                        "median_ratio": median(paired_values),
                        "p90_ratio": percentile(paired_values, 0.9),
                    }
                result["by_p_bin"][p_bin][ratio_key] = ratios
                result["by_p_bin"][p_bin][
                    f"{ratio_key}_paired"
                ] = paired
    result["overall"] = result["by_p_bin"]["all"]
    return result


def save_figure(fig: Any, stem: str) -> None:
    plt = pyplot()
    output = OUTPUT / "figures"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_objective(
    summaries: list[dict[str, Any]],
    quality: dict[tuple[int, str], dict[str, Any]],
) -> None:
    plt = pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.2), sharex=False)
    for task_index, axis in enumerate(axes.flat, start=1):
        for method in ("pg10", "adaptive_t10", "adaptive_t3"):
            path = np.asarray(
                quality[(task_index, method)]["quality"]["objective_path"],
                dtype=np.float64,
            )
            normalized = path / np.maximum(np.abs(path[:, :1]), 1e-12)
            x = np.arange(path.shape[1])
            axis.plot(
                x,
                np.median(normalized, axis=0),
                marker="x" if method == "adaptive_t3" else "o",
                markersize=5 if method == "adaptive_t3" else 3,
                linestyle="None" if method == "adaptive_t3" else "-",
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
        task = quality[(task_index, "pg10")]["task"]
        axis.set_title(f"Task {task_index}: N={task['n']:,}, P={task['p']:,}")
        axis.set_xlabel("Refinement iteration / sweep")
        axis.set_ylabel("Median F / initial F")
        axis.grid(alpha=0.18)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, "objective_trajectories")


def plot_kkt(summaries: list[dict[str, Any]]) -> None:
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1))
    x = np.arange(1, 7)
    width = 0.24
    for offset, method in enumerate(METHODS):
        selected = [
            next(
                row
                for row in summaries
                if row["task_index"] == task and row["method"] == method
            )
            for task in range(1, 7)
        ]
        axes[0].bar(
            x + (offset - 1) * width,
            [row["median_kkt_top32"] for row in selected],
            width,
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
        axes[1].bar(
            x + (offset - 1) * width,
            [row["median_kkt_full_rms"] for row in selected],
            width,
            color=COLORS[method],
        )
    axes[0].set_title("Top-32 KKT residual")
    axes[1].set_title("Full-P KKT RMS")
    for axis in axes:
        axis.set_xlabel("Frozen task")
        axis.set_yscale("log")
        axis.set_xticks(x)
        axis.grid(axis="y", alpha=0.18)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save_figure(fig, "kkt_comparison")


def plot_runtime(timing: dict[str, Any]) -> None:
    plt = pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    bins = ("low", "mid", "high")
    x = np.arange(len(bins))
    width = 0.24
    modes = (
        ("refinement_forward", "Refinement forward"),
        ("refinement_forward_backward", "Refinement forward + backward"),
        ("end_to_end_beta_microstep", "End-to-end beta microstep"),
    )
    for axis, (field, title) in zip(axes, modes):
        for offset, method in enumerate(METHODS):
            values = [
                timing["by_p_bin"][p_bin][method][field]["median_seconds"]
                for p_bin in bins
            ]
            axis.bar(
                x + (offset - 1) * width,
                values,
                width,
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_title(title)
        axis.set_xticks(x, bins)
        axis.set_ylabel("Seconds")
        axis.grid(axis="y", alpha=0.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, "runtime_comparison")


def build_benchmark_decision(
    rows: list[dict[str, Any]],
    contracts: dict[str, Any],
) -> dict[str, Any]:
    by_method = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    summary = {}
    for method in METHODS:
        method_rows = by_method[method]
        summary[method] = {
            "median_final_objective": median(
                row["final_objective"] for row in method_rows
            ),
            "median_relative_objective_reduction": median(
                row["relative_objective_reduction"] for row in method_rows
            ),
            "median_objective_gap_vs_pg10": median(
                row["normalized_objective_gap_vs_pg10"]
                for row in method_rows
            ),
            "p90_objective_gap_vs_pg10": percentile(
                (
                    row["normalized_objective_gap_vs_pg10"]
                    for row in method_rows
                ),
                0.9,
            ),
            "median_kkt_top32": median(
                row["kkt_top32"] for row in method_rows
            ),
            "median_kkt_full_rms": median(
                row["kkt_full_rms"] for row in method_rows
            ),
        }
    return {
        "status": "PASS",
        "hard_contracts": contracts,
        "quality_is_diagnostic_not_a_hard_gate": True,
        "method_summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    fragments = load_fragments()
    quality = quality_cells(fragments)
    contracts = validate_contracts(fragments, quality)
    rows, summaries = build_quality_rows(quality)
    timing = summarize_timing(fragments)
    decision = build_benchmark_decision(rows, contracts)

    write_csv(OUTPUT / "six_task_quality.csv", rows)
    write_csv(OUTPUT / "six_task_summary.csv", summaries)
    atomic_json_dump(timing, OUTPUT / "timing_summary.json")
    atomic_json_dump(contracts, OUTPUT / "benchmark_contract_audit.json")
    atomic_json_dump(decision, OUTPUT / "benchmark_decision.json")
    if not args.skip_plots:
        plot_objective(summaries, quality)
        plot_kkt(summaries)
        plot_runtime(timing)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
