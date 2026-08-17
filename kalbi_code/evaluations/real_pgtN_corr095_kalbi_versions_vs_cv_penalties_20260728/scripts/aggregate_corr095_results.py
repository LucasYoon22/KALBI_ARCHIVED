#!/usr/bin/env python3
"""Aggregate corr095 real-data results and render publication-ready figures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

from real_pgtN_common import atomic_json, sha256


BUNDLE_LABELS = {
    "KALBI-A-v6-v23-s20300": "KALBI-A",
    "KALBI-B-v8-v23-s20300": "KALBI-B",
    "KALBI-C-v8-v24-s03900": "KALBI-C",
    "KALBI-D-v8-v24-s13100": "KALBI-D",
}
BASELINE_LABELS = {
    "lasso": "CV LASSO",
    "scad": "CV SCAD",
    "mcp": "CV MCP",
}
METHOD_ORDER = [*BUNDLE_LABELS.values(), *BASELINE_LABELS.values()]
COLORS = {
    "KALBI-A": "#0072B2",
    "KALBI-B": "#56B4E9",
    "KALBI-C": "#009E73",
    "KALBI-D": "#D55E00",
    "CV LASSO": "#000000",
    "CV SCAD": "#CC79A7",
    "CV MCP": "#E69F00",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def holm_adjust(values: list[float]) -> list[float]:
    p = np.asarray(values, dtype=np.float64)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    count = p.size
    for rank, index in enumerate(order):
        candidate = min(1.0, float((count - rank) * p[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def bootstrap_mean_log_ratio(
    left: np.ndarray,
    right: np.ndarray,
    *,
    groups: np.ndarray | None,
    seed: int,
    draws: int = 20_000,
) -> tuple[float, float]:
    difference = np.log(np.maximum(left, 1e-12)) - np.log(
        np.maximum(right, 1e-12)
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    if groups is None:
        n = difference.size
        for index in range(draws):
            estimates[index] = np.mean(
                difference[rng.integers(0, n, size=n)]
            )
    else:
        unique = np.unique(groups)
        locations = {value: np.flatnonzero(groups == value) for value in unique}
        for index in range(draws):
            sampled = rng.choice(unique, size=unique.size, replace=True)
            selected = np.concatenate([locations[value] for value in sampled])
            estimates[index] = np.mean(difference[selected])
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def classify_subgroup(row: pd.Series) -> list[str]:
    groups = ["all_applicable"]
    groups.append(
        "remained_high_dimensional"
        if bool(row["remained_high_dimensional"])
        else "became_low_dimensional"
    )
    text = f"{row['source']} {row['source_family']} {row['task_id']}".lower()
    if "moleculenet" in text or "morgan" in text:
        groups.append("molecular_fingerprint")
    elif "raman" in text or "spectra" in text:
        groups.append("spectroscopy")
    else:
        groups.append("other_tabular")
    return groups


def collect(evaluation_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    baseline_paths = sorted(
        (evaluation_root / "metrics").glob("*_baselines.json")
    )
    if len(baseline_paths) != 41:
        raise RuntimeError(
            f"expected 41 baseline metrics, found {len(baseline_paths)}"
        )
    baseline_by_task = {
        load_json(path)["task_id"]: load_json(path) for path in baseline_paths
    }
    kalbi_by_bundle: dict[str, dict[str, dict[str, Any]]] = {}
    failures: list[dict[str, Any]] = []
    for bundle_id in BUNDLE_LABELS:
        metric_root = evaluation_root / "bundles" / bundle_id / "metrics"
        passed: dict[str, dict[str, Any]] = {}
        for path in metric_root.glob("*.json"):
            value = load_json(path)
            if value.get("status") == "PASS":
                passed[value["task_id"]] = value
            elif path.name.endswith("_failure.json"):
                failures.append(value)
        kalbi_by_bundle[bundle_id] = passed

    applicable = set(kalbi_by_bundle[next(iter(BUNDLE_LABELS))])
    if len(applicable) != 25:
        raise RuntimeError(f"expected 25 KALBI-applicable tasks, got {len(applicable)}")
    for bundle_id, values in kalbi_by_bundle.items():
        if set(values) != applicable:
            raise RuntimeError(f"bundle task mismatch: {bundle_id}")

    long_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for task_id, baseline in sorted(
        baseline_by_task.items(), key=lambda item: item[1]["registry_index"]
    ):
        base_common = {
            "registry_index": int(baseline["registry_index"]),
            "task_id": task_id,
            "source": baseline["source"],
            "source_family": baseline["source_family"],
            "target": baseline["target"],
            "N": int(baseline["N"]),
            "P_raw": int(baseline["P_raw"]),
            "P_after_constant_filter": int(
                baseline["P_after_constant_filter"]
            ),
            "P_after_corr095": int(baseline["P_after_corr095"]),
            "removed_correlated_features": int(
                baseline["removed_correlated_features"]
            ),
            "removed_fraction": 1.0
            - int(baseline["P_after_corr095"])
            / max(int(baseline["P_after_constant_filter"]), 1),
            "retained_max_abs_correlation": float(
                baseline["retained_max_abs_correlation"]
            ),
            "retained_hash": baseline[
                "retained_original_indices_sha256"
            ],
            "remained_high_dimensional": bool(
                baseline["remained_high_dimensional"]
            ),
            "kalbi_applicable": task_id in applicable,
        }
        task_rows.append(base_common)
        for key, label in BASELINE_LABELS.items():
            method = baseline["methods"][key]
            long_rows.append(
                {
                    **base_common,
                    "method": label,
                    "method_type": "baseline",
                    "test_mse": float(method["test_mse"]),
                    "normalized_test_mse": float(
                        method["normalized_test_mse"]
                    ),
                    "test_r2": float(method["test_r2"]),
                    "test_mae": float(method["test_mae"]),
                    "nnz_1e8": int(method["nnz_1e8"]),
                    "nnz_1e6": int(method["nnz_1e6"]),
                    "nnz_1e4": int(method["nnz_1e4"]),
                    "selected_lambda": float(
                        method["selected_raw_lambda"]
                    ),
                    "runtime_seconds": float(method["fit_seconds"]),
                    "projected_kkt_max": float(method["train_kkt_max"]),
                }
            )
        if task_id not in applicable:
            continue
        for bundle_id, label in BUNDLE_LABELS.items():
            metric = kalbi_by_bundle[bundle_id][task_id]
            if (
                metric["retained_original_indices_sha256"]
                != base_common["retained_hash"]
                or metric["preprocessing_sha256"]
                != baseline["preprocessing_sha256"]
                or int(metric["P_after_corr095"])
                != int(baseline["P_after_corr095"])
            ):
                raise RuntimeError(
                    f"final preprocessing mismatch: {task_id} {label}"
                )
            long_rows.append(
                {
                    **base_common,
                    "method": label,
                    "method_type": "KALBI",
                    "test_mse": float(metric["selected_test_mse"]),
                    "normalized_test_mse": float(
                        metric["selected_normalized_test_mse"]
                    ),
                    "test_r2": float(metric["selected_test_r2"]),
                    "test_mae": float(metric["selected_test_mae"]),
                    "nnz_1e8": int(metric["nnz_1e8"]),
                    "nnz_1e6": int(metric["nnz_1e6"]),
                    "nnz_1e4": int(metric["nnz_1e4"]),
                    "selected_lambda": float(
                        metric["selected_raw_lambda"]
                    ),
                    "runtime_seconds": float(
                        metric["runtime"]["total_seconds"]
                    ),
                    "projected_kkt_max": float(
                        metric["selected_projected_kkt_max"]
                    ),
                }
            )
    return pd.DataFrame(long_rows), pd.DataFrame(task_rows), {
        "applicable_task_ids": sorted(applicable),
        "inapplicable_task_count": 41 - len(applicable),
        "failures": failures,
    }


def aggregate(long: pd.DataFrame, tasks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    complete = long[long["kalbi_applicable"]].copy()
    pivot = complete.pivot(
        index="task_id", columns="method", values="normalized_test_mse"
    )[METHOD_ORDER]
    ranks = pivot.rank(axis=1, method="average", ascending=True)

    aggregate_rows: list[dict[str, Any]] = []
    for subgroup in [
        "all_applicable",
        "remained_high_dimensional",
        "became_low_dimensional",
        "molecular_fingerprint",
        "spectroscopy",
        "other_tabular",
    ]:
        task_ids = []
        for _, row in tasks[tasks["kalbi_applicable"]].iterrows():
            if subgroup in classify_subgroup(row):
                task_ids.append(row["task_id"])
        if not task_ids:
            continue
        for method in METHOD_ORDER:
            values = pivot.loc[task_ids, method].to_numpy()
            r2 = complete[
                complete["task_id"].isin(task_ids)
                & (complete["method"] == method)
            ]["test_r2"].to_numpy()
            aggregate_rows.append(
                {
                    "subgroup": subgroup,
                    "method": method,
                    "task_count": len(task_ids),
                    "median_nmse": float(np.median(values)),
                    "geometric_mean_nmse": float(
                        np.exp(np.mean(np.log(np.maximum(values, 1e-12))))
                    ),
                    "median_r2": float(np.median(r2)),
                    "mean_rank": float(ranks.loc[task_ids, method].mean()),
                    "median_rank": float(
                        ranks.loc[task_ids, method].median()
                    ),
                }
            )

    pair_rows: list[dict[str, Any]] = []
    pvalues: list[float] = []
    for bundle in BUNDLE_LABELS.values():
        for baseline in BASELINE_LABELS.values():
            left = pivot[bundle].to_numpy()
            right = pivot[baseline].to_numpy()
            tolerance = 1e-9 * np.maximum(np.maximum(left, right), 1.0)
            wins = int(np.sum(left < right - tolerance))
            losses = int(np.sum(left > right + tolerance))
            ties = int(left.size - wins - losses)
            difference = np.log(np.maximum(left, 1e-12)) - np.log(
                np.maximum(right, 1e-12)
            )
            try:
                pvalue = float(
                    wilcoxon(difference, zero_method="wilcox").pvalue
                )
            except ValueError:
                pvalue = 1.0
            groups = (
                tasks.set_index("task_id")
                .loc[pivot.index, "source"]
                .to_numpy()
            )
            ci = bootstrap_mean_log_ratio(
                left, right, groups=None, seed=20260728
            )
            grouped_ci = bootstrap_mean_log_ratio(
                left, right, groups=groups, seed=20260729
            )
            pair_rows.append(
                {
                    "kalbi": bundle,
                    "baseline": baseline,
                    "task_count": int(left.size),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "mean_log_nmse_ratio": float(np.mean(difference)),
                    "median_nmse_difference": float(
                        np.median(left - right)
                    ),
                    "paired_bootstrap_log_ratio_ci_low": ci[0],
                    "paired_bootstrap_log_ratio_ci_high": ci[1],
                    "grouped_bootstrap_log_ratio_ci_low": grouped_ci[0],
                    "grouped_bootstrap_log_ratio_ci_high": grouped_ci[1],
                    "wilcoxon_p": pvalue,
                }
            )
            pvalues.append(pvalue)
    adjusted = holm_adjust(pvalues)
    for row, value in zip(pair_rows, adjusted, strict=True):
        row["wilcoxon_holm_p"] = value
    return pd.DataFrame(aggregate_rows), pd.DataFrame(pair_rows)


def render_figures(
    long: pd.DataFrame,
    tasks: pd.DataFrame,
    aggregate_frame: pd.DataFrame,
    pairwise: pd.DataFrame,
    figure_root: Path,
) -> None:
    complete = long[long["kalbi_applicable"]].copy()
    task_order = (
        tasks[tasks["kalbi_applicable"]]
        .sort_values("registry_index")["task_id"]
        .tolist()
    )
    pivot = complete.pivot(
        index="task_id", columns="method", values="normalized_test_mse"
    ).loc[task_order, METHOD_ORDER]

    figure, axis = plt.subplots(figsize=(8.4, 8.8))
    image = axis.imshow(
        np.log10(np.maximum(pivot.to_numpy(), 1e-4)),
        aspect="auto",
        cmap="RdYlBu_r",
        vmin=-1.0,
        vmax=0.5,
    )
    axis.set_xticks(np.arange(len(METHOD_ORDER)), METHOD_ORDER, rotation=35, ha="right")
    axis.set_yticks(
        np.arange(len(task_order)),
        [f"{index + 1:02d}" for index in range(len(task_order))],
    )
    axis.set_ylabel("Dataset")
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label(r"$\log_{10}$(test NMSE)")
    axis.spines[["top", "right"]].set_visible(False)
    save_figure(figure, figure_root / "nmse_heatmap")

    figure, axes = plt.subplots(4, 3, figsize=(10.5, 12.5), sharex=False, sharey=False)
    for row_index, bundle in enumerate(BUNDLE_LABELS.values()):
        for column_index, baseline in enumerate(BASELINE_LABELS.values()):
            axis = axes[row_index, column_index]
            x = pivot[baseline].to_numpy()
            y = pivot[bundle].to_numpy()
            lower = max(1e-3, min(x.min(), y.min()) * 0.8)
            upper = max(x.max(), y.max()) * 1.25
            axis.scatter(x, y, s=24, color=COLORS[bundle], alpha=0.8)
            axis.plot([lower, upper], [lower, upper], ":", color="0.35", lw=1)
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlim(lower, upper)
            axis.set_ylim(lower, upper)
            axis.set_title(f"{bundle} vs {baseline}", fontsize=10)
            if row_index == 3:
                axis.set_xlabel(f"{baseline} NMSE")
            if column_index == 0:
                axis.set_ylabel(f"{bundle} NMSE")
            axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    save_figure(figure, figure_root / "paired_nmse_scatter")

    rank_pivot = pivot.rank(axis=1, method="average", ascending=True)
    figure, axis = plt.subplots(figsize=(8.2, 4.4))
    means = rank_pivot.mean(axis=0).loc[METHOD_ORDER]
    errors = (
        rank_pivot.std(axis=0, ddof=1).loc[METHOD_ORDER]
        / math.sqrt(rank_pivot.shape[0])
    )
    x = np.arange(len(METHOD_ORDER))
    axis.errorbar(
        x,
        means,
        yerr=errors,
        fmt="o",
        capsize=4,
        color="black",
        ecolor="0.4",
    )
    for index, method in enumerate(METHOD_ORDER):
        axis.scatter(index, means[method], s=60, color=COLORS[method], zorder=3)
    axis.set_xticks(x, METHOD_ORDER, rotation=30, ha="right")
    axis.set_ylabel("Mean rank (lower is better)")
    axis.set_ylim(0.7, len(METHOD_ORDER) + 0.3)
    axis.spines[["top", "right"]].set_visible(False)
    save_figure(figure, figure_root / "method_rank")

    figure, axis = plt.subplots(figsize=(6.8, 4.6))
    matrix = np.empty((4, 3), dtype=float)
    annotations = np.empty((4, 3), dtype=object)
    for row_index, bundle in enumerate(BUNDLE_LABELS.values()):
        for column_index, baseline in enumerate(BASELINE_LABELS.values()):
            row = pairwise[
                (pairwise["kalbi"] == bundle)
                & (pairwise["baseline"] == baseline)
            ].iloc[0]
            matrix[row_index, column_index] = row["wins"] / row["task_count"]
            annotations[row_index, column_index] = (
                f"{row['wins']}/{row['ties']}/{row['losses']}"
            )
    image = axis.imshow(matrix, cmap="BrBG", vmin=0.0, vmax=1.0)
    for row_index in range(4):
        for column_index in range(3):
            axis.text(
                column_index,
                row_index,
                annotations[row_index, column_index],
                ha="center",
                va="center",
                fontsize=10,
            )
    axis.set_xticks(np.arange(3), list(BASELINE_LABELS.values()))
    axis.set_yticks(np.arange(4), list(BUNDLE_LABELS.values()))
    axis.set_xlabel("Win / tie / loss")
    figure.colorbar(image, ax=axis, label="KALBI win fraction")
    save_figure(figure, figure_root / "win_tie_loss")

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    applicable = tasks["kalbi_applicable"].to_numpy()
    axes[0].scatter(
        tasks.loc[applicable, "P_after_constant_filter"],
        tasks.loc[applicable, "P_after_corr095"],
        color="#0072B2",
        label="KALBI applicable",
        s=32,
    )
    axes[0].scatter(
        tasks.loc[~applicable, "P_after_constant_filter"],
        tasks.loc[~applicable, "P_after_corr095"],
        color="#D55E00",
        label="P after filter < 32",
        marker="x",
        s=42,
    )
    limit = float(tasks["P_after_constant_filter"].max())
    axes[0].plot([1, limit], [1, limit], ":", color="0.4")
    axes[0].axhline(32, color="0.3", lw=1, ls="--")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Features before correlation filter")
    axes[0].set_ylabel("Features after correlation filter")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].hist(
        tasks["removed_fraction"],
        bins=np.linspace(0, 1, 16),
        color="#56B4E9",
        edgecolor="white",
    )
    axes[1].set_xlabel("Fraction of features removed")
    axes[1].set_ylabel("Datasets")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    save_figure(figure, figure_root / "feature_filtering")

    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for method in METHOD_ORDER:
        subset = complete[complete["method"] == method]
        axes[0].scatter(
            subset["P_after_corr095"],
            subset["runtime_seconds"],
            s=23,
            alpha=0.75,
            label=method,
            color=COLORS[method],
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Filtered features")
    axes[0].set_ylabel("Fit or inference time (s)")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    box_values = [
        np.log10(1.0 + complete[complete["method"] == method]["nnz_1e8"])
        for method in METHOD_ORDER
    ]
    box = axes[1].boxplot(
        box_values, patch_artist=True, tick_labels=METHOD_ORDER
    )
    for patch, method in zip(box["boxes"], METHOD_ORDER, strict=True):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.7)
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_ylabel(r"$\log_{10}(1+\mathrm{NNZ})$")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    save_figure(figure, figure_root / "runtime_and_sparsity")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    args = parser.parse_args()
    metric_root = args.evaluation_root / "metrics"
    figure_root = args.evaluation_root / "figures"
    long, tasks, applicability = collect(args.evaluation_root)
    aggregate_frame, pairwise = aggregate(long, tasks)

    long.to_csv(metric_root / "per_task_metrics.csv", index=False)
    tasks.to_csv(metric_root / "preprocessing_per_task.csv", index=False)
    aggregate_frame.to_csv(metric_root / "aggregate_metrics.csv", index=False)
    pairwise.to_csv(metric_root / "pairwise_statistics.csv", index=False)
    atomic_json(
        args.evaluation_root / "manifests" / "kalbi_applicability.json",
        {
            "status": "PASS",
            "contract": "K=32 requires P_after_corr095 >= 32; no zero padding",
            **applicability,
        },
    )
    atomic_json(
        args.evaluation_root / "audits" / "correlation_filter_audit.json",
        {
            "status": "PASS",
            "threshold": 0.95,
            "task_count": int(tasks.shape[0]),
            "maximum_retained_abs_correlation": float(
                tasks["retained_max_abs_correlation"].max()
            ),
            "all_below_threshold": bool(
                (tasks["retained_max_abs_correlation"] < 0.95 + 1e-10).all()
            ),
            "minimum_P_after": int(tasks["P_after_corr095"].min()),
            "median_P_after": float(tasks["P_after_corr095"].median()),
            "maximum_P_after": int(tasks["P_after_corr095"].max()),
        },
    )
    render_figures(long, tasks, aggregate_frame, pairwise, figure_root)

    files = [
        metric_root / "per_task_metrics.csv",
        metric_root / "preprocessing_per_task.csv",
        metric_root / "aggregate_metrics.csv",
        metric_root / "pairwise_statistics.csv",
    ]
    report = {
        "status": "PASS",
        "baseline_task_count": 41,
        "kalbi_applicable_task_count": 25,
        "method_count": len(METHOD_ORDER),
        "outputs": {str(path): sha256(path) for path in files},
    }
    atomic_json(args.evaluation_root / "reports" / "aggregate_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
