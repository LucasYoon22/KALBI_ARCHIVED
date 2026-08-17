#!/usr/bin/env python3
"""Render per-task and aggregate figures for the real P>N KALBI audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from real_pgtN_common import atomic_json, sha256


METHODS = ("KALBI", "CV-LASSO", "CV-SCAD", "CV-MCP")
BASELINE_KEYS = {
    "CV-LASSO": "lasso",
    "CV-SCAD": "scad",
    "CV-MCP": "mcp",
}
COLORS = {
    "KALBI": "#0072B2",
    "Coordinate descent reference": "#D55E00",
    "CV-LASSO": "#E69F00",
    "CV-SCAD": "#009E73",
    "CV-MCP": "#CC79A7",
    "Zero predictor": "#555555",
    "Oracle": "#56B4E9",
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
    "#8C564B",
    "#17BECF",
    "#7F7F7F",
    "#BCBD22",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-task-count", type=int, default=41)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", width=0.8, length=3)
    axis.grid(False)


def save_figure(figure: plt.Figure, stem: Path) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=300, bbox_inches="tight")
    return {
        "pdf": str(pdf),
        "pdf_sha256": sha256(pdf),
        "png": str(png),
        "png_sha256": sha256(png),
    }


def finite_range(values: np.ndarray, default: tuple[float, float]) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return default
    low = float(finite.min())
    high = float(finite.max())
    if math.isclose(low, high):
        margin = max(abs(low) * 0.05, 1e-3)
        return low - margin, high + margin
    margin = 0.05 * (high - low)
    return low - margin, high + margin


def finite_method_values(
    rows: list[dict[str, Any]], method: str, suffix: str
) -> np.ndarray:
    key = method.lower().replace("cv-", "")
    values = np.asarray(
        [row[f"{key}_{suffix}"] for row in rows], dtype=np.float64
    )
    return values[np.isfinite(values)]


def coefficient_panel(
    axis: plt.Axes,
    arrays: dict[str, np.ndarray],
    *,
    cd_audit_pass: bool,
) -> None:
    x = np.log10(arrays["raw_lambdas"])
    kalbi = arrays["kalbi_beta"]
    cd = arrays["coordinate_descent_beta"]
    if cd_audit_pass:
        top = arrays["top_features"].astype(np.int64)
    else:
        score = np.max(np.abs(kalbi), axis=0)
        top = np.lexsort((np.arange(score.size), -score))[: min(12, score.size)]
    for order, feature in enumerate(top):
        color = COEFFICIENT_COLORS[order % len(COEFFICIENT_COLORS)]
        axis.plot(x, kalbi[:, feature], color=color, linewidth=1.15)
        if cd_audit_pass and np.isfinite(cd[:, feature]).any():
            axis.plot(
                x,
                cd[:, feature],
                color=color,
                linestyle="--",
                linewidth=0.9,
                alpha=0.78,
            )
    if not cd_audit_pass:
        axis.text(
            0.02,
            0.02,
            "CD reference did not meet tolerance",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color=COLORS["Coordinate descent reference"],
        )
    axis.axhline(0.0, color="#777777", linewidth=0.6)
    axis.set_xlabel(r"$\log_{10}(\lambda_{\mathrm{raw}})$")
    axis.set_ylabel("Coefficient")
    axis.set_title("Coefficient solution path")
    style_axis(axis)


def heldout_panel(
    axis: plt.Axes,
    arrays: dict[str, np.ndarray],
    kalbi_metric: dict[str, Any],
    *,
    cd_audit_pass: bool,
) -> None:
    x = np.log10(arrays["raw_lambdas"])
    zero_mse = max(float(kalbi_metric["zero_predictor"]["mse"]), 1e-30)
    kalbi_mse = arrays["kalbi_test_mse"] / zero_mse
    cd_mse = arrays["coordinate_descent_test_mse"] / zero_mse
    axis.plot(
        x,
        kalbi_mse,
        color=COLORS["KALBI"],
        linewidth=1.7,
        label="KALBI",
    )
    if cd_audit_pass and np.isfinite(cd_mse).any():
        axis.plot(
            x,
            cd_mse,
            color=COLORS["Coordinate descent reference"],
            linestyle="--",
            linewidth=1.25,
            label="Coordinate descent reference",
        )
    selected_x = math.log10(float(arrays["selected_raw_lambda"]))
    oracle_index = int(arrays["oracle_index"])
    axis.axvline(
        selected_x,
        color="#111111",
        linewidth=1.05,
        label="KALBI-selected lambda",
    )
    axis.scatter(
        x[oracle_index],
        kalbi_mse[oracle_index],
        marker="D",
        s=28,
        color=COLORS["Oracle"],
        edgecolor="#111111",
        linewidth=0.4,
        zorder=5,
        label="Oracle lambda",
    )
    axis.axhline(
        1.0,
        color=COLORS["Zero predictor"],
        linestyle=":",
        linewidth=0.9,
        label="Zero predictor",
    )
    axis.set_xlabel(r"$\log_{10}(\lambda_{\mathrm{raw}})$")
    axis.set_ylabel("Test MSE / zero-predictor MSE")
    axis.set_title("Held-out error and lambda selection")
    axis.set_yscale("log")
    style_axis(axis)


def penalty_panel(axis: plt.Axes, arrays: dict[str, np.ndarray]) -> None:
    grid = arrays["penalty_grid"]
    keep = grid <= 1.0 + 1e-12
    derivative = axis.twinx()
    axis.plot(
        grid[keep],
        arrays["penalty_rho"][keep],
        color=COLORS["KALBI"],
        linewidth=1.7,
        label=r"$\rho(t)$",
    )
    derivative.plot(
        grid[keep],
        arrays["penalty_drho"][keep],
        color=COLORS["Coordinate descent reference"],
        linestyle="--",
        linewidth=1.3,
        label=r"$\rho'(t)$",
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel(r"$|\beta|$")
    axis.set_ylabel(r"$\rho(|\beta|)$")
    derivative.set_ylabel(r"$\rho'(|\beta|)$")
    axis.set_title("KALBI-selected penalty")
    style_axis(axis)
    derivative.spines["top"].set_visible(False)
    handles_a, labels_a = axis.get_legend_handles_labels()
    handles_b, labels_b = derivative.get_legend_handles_labels()
    axis.legend(
        handles_a + handles_b,
        labels_a + labels_b,
        frameon=False,
        fontsize=8.2,
        loc="best",
    )


def performance_panel(
    axis: plt.Axes,
    row: dict[str, Any],
) -> None:
    values = [row[f"{key}_normalized_mse"] for key in ("kalbi", "lasso", "scad", "mcp")]
    positions = np.arange(len(METHODS))
    bars = axis.bar(
        positions,
        values,
        color=[COLORS[method] for method in METHODS],
        width=0.72,
    )
    for position, value, bar in zip(positions, values, bars, strict=True):
        if np.isfinite(value):
            continue
        bar.set_visible(False)
        axis.text(
            position,
            0.04,
            "timeout",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=8,
            color="#666666",
        )
    axis.axhline(
        1.0,
        color=COLORS["Zero predictor"],
        linestyle=":",
        linewidth=0.9,
    )
    axis.set_xticks(positions, METHODS, rotation=18, ha="right")
    axis.set_yscale("log")
    axis.set_ylabel("Test MSE / zero-predictor MSE")
    axis.set_title("One-split prediction comparison")
    style_axis(axis)


def render_task(
    row: dict[str, Any],
    arrays: dict[str, np.ndarray],
    output_dir: Path,
) -> tuple[plt.Figure, dict[str, str]]:
    figure, axes = plt.subplots(2, 2, figsize=(10.6, 7.7), constrained_layout=True)
    coefficient_panel(
        axes[0, 0], arrays, cd_audit_pass=row["cd_audit_pass"]
    )
    heldout_panel(
        axes[0, 1],
        arrays,
        row["kalbi_metric"],
        cd_audit_pass=row["cd_audit_pass"],
    )
    penalty_panel(axes[1, 0], arrays)
    performance_panel(axes[1, 1], row)
    figure.suptitle(
        f"Dataset {row['dataset_index']:02d}  |  "
        f"$N_{{train}}={row['kalbi_metric']['outer_train_rows']:,}$, "
        f"$N_{{test}}={row['kalbi_metric']['outer_test_rows']:,}$, "
        f"$P={row['P_transformed']:,}$",
        fontsize=12,
    )
    handles, labels = axes[0, 1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=8.4,
    )
    stem = output_dir / f"dataset_{row['dataset_index']:02d}_{row['task_id']}"
    paths = save_figure(figure, stem)
    return figure, paths


def collect_rows(
    evaluation_root: Path,
    split_manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, np.ndarray]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    arrays: list[dict[str, np.ndarray]] = []
    failures: list[dict[str, Any]] = []
    for dataset_index, task in enumerate(split_manifest["tasks"], start=1):
        task_id = task["task_id"]
        kalbi_path = evaluation_root / "metrics" / f"{task_id}_kalbi.json"
        baseline_path = evaluation_root / "metrics" / f"{task_id}_baselines.json"
        path_path = evaluation_root / "paths" / f"{task_id}.npz"
        if not (kalbi_path.is_file() and baseline_path.is_file() and path_path.is_file()):
            failures.append(
                {
                    "dataset_index": dataset_index,
                    "task_id": task_id,
                    "reason": "missing metric or path artifact",
                }
            )
            continue
        kalbi = load_json(kalbi_path)
        baseline = load_json(baseline_path)
        if kalbi.get("status") != "PASS" or baseline.get("status") not in {
            "PASS",
            "PARTIAL_NONCONVEX_TIMEOUT",
        }:
            failures.append(
                {
                    "dataset_index": dataset_index,
                    "task_id": task_id,
                    "reason": "non-PASS metric status",
                }
            )
            continue
        if kalbi["outer_split_sha256"] != baseline["outer_split_sha256"]:
            raise RuntimeError(f"outer split mismatch for {task_id}")
        if kalbi["scaler_sha256"] != baseline["scaler_sha256"]:
            raise RuntimeError(f"preprocessing mismatch for {task_id}")
        with np.load(path_path, allow_pickle=False) as archive:
            task_arrays = {
                key: np.asarray(archive[key]) for key in archive.files
            }
        zero_mse = max(float(kalbi["zero_predictor"]["mse"]), 1e-30)
        row: dict[str, Any] = {
            "dataset_index": dataset_index,
            "registry_index": int(task["registry_index"]),
            "task_id": task_id,
            "source": kalbi["source"],
            "source_family": kalbi["source_family"],
            "target": kalbi["target"],
            "evidence_tier": kalbi["evidence_tier"],
            "N": int(kalbi["N"]),
            "P_raw": int(kalbi["P_raw"]),
            "P_transformed": int(kalbi["P_transformed"]),
            "P_over_N": float(kalbi["P_transformed"] / kalbi["N"]),
            "split_sha256": kalbi["outer_split_sha256"],
            "scaler_sha256": kalbi["scaler_sha256"],
            "zero_test_mse": zero_mse,
            "kalbi_test_mse": float(kalbi["selected_test_mse"]),
            "kalbi_test_r2": float(kalbi["selected_test_r2"]),
            "kalbi_normalized_mse": float(kalbi["selected_normalized_test_mse"]),
            "kalbi_nnz": int(kalbi["selected_nnz_1e8"]),
            "kalbi_selected_raw_lambda": float(kalbi["selected_raw_lambda"]),
            "kalbi_oracle_raw_lambda": float(kalbi["oracle_raw_lambda"]),
            "kalbi_oracle_test_mse": float(kalbi["oracle_test_mse"]),
            "kalbi_oracle_normalized_mse": float(kalbi["oracle_test_mse"] / zero_mse),
            "kalbi_selected_over_oracle_mse": float(kalbi["selected_over_oracle_mse"]),
            "path_correlation_mean": float(kalbi["path_correlation_mean"]),
            "coefficient_rmse": float(kalbi["coefficient_rmse"]),
            "objective_gap_mean": float(kalbi["objective_gap_mean"]),
            "baseline_status": baseline["status"],
            "nonconvex_timeout": bool(
                baseline["status"] == "PARTIAL_NONCONVEX_TIMEOUT"
            ),
            "cd_audit_pass": bool(
                kalbi["coordinate_descent"].get("audit_pass", True)
            ),
            "penalty_d0": float(kalbi["penalty"]["d0"]),
            "penalty_max_drho": float(kalbi["penalty"]["max_drho"]),
            "penalty_derivative_tv": float(
                kalbi["penalty"]["derivative_total_variation"]
            ),
            "kalbi_metric": kalbi,
        }
        for display, key in BASELINE_KEYS.items():
            method = baseline["methods"][key]
            prefix = display.lower().replace("cv-", "")
            available = bool(
                method.get("status") != "TIMEOUT"
                and "normalized_test_mse" in method
            )
            row[f"{prefix}_available"] = available
            row[f"{prefix}_status"] = method.get("status", "PASS")
            row[f"{prefix}_test_mse"] = (
                float(method["test_mse"]) if available else float("nan")
            )
            row[f"{prefix}_test_r2"] = (
                float(method["test_r2"]) if available else float("nan")
            )
            row[f"{prefix}_normalized_mse"] = (
                float(method["normalized_test_mse"])
                if available
                else float("nan")
            )
            row[f"{prefix}_nnz"] = (
                int(method["nnz_1e8"]) if available else None
            )
            row[f"{prefix}_selected_ratio"] = (
                float(method["selected_ratio"]) if available else float("nan")
            )
        method_values = {
            "KALBI": row["kalbi_normalized_mse"],
            "CV-LASSO": row["lasso_normalized_mse"],
            "CV-SCAD": row["scad_normalized_mse"],
            "CV-MCP": row["mcp_normalized_mse"],
        }
        available_values = {
            method: value
            for method, value in method_values.items()
            if np.isfinite(value)
        }
        row["winner"] = min(available_values, key=available_values.get)
        row["winner_normalized_mse"] = float(method_values[row["winner"]])
        rows.append(row)
        arrays.append(task_arrays)
    return rows, arrays, failures


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    public: dict[str, Any] = {}
    for key, value in row.items():
        if key == "kalbi_metric":
            continue
        if isinstance(value, float) and not math.isfinite(value):
            public[key] = None
        else:
            public[key] = value
    return public


def render_method_summary(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), constrained_layout=True)
    x = np.arange(len(rows))
    for method in METHODS:
        key = method.lower().replace("cv-", "")
        axes[0, 0].plot(
            x,
            [row[f"{key}_normalized_mse"] for row in rows],
            marker="o",
            markersize=2.8,
            linewidth=0.8,
            color=COLORS[method],
            alpha=0.82,
            label=method,
        )
    axes[0, 0].axhline(1.0, color="#555555", linestyle=":", linewidth=0.8)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Dataset index")
    axes[0, 0].set_ylabel("Normalized test MSE")
    axes[0, 0].set_title("One-split performance across datasets")

    box_values = [
        finite_method_values(rows, method, "normalized_mse")
        for method in METHODS
    ]
    boxes = axes[0, 1].boxplot(
        box_values,
        tick_labels=METHODS,
        patch_artist=True,
        showfliers=True,
        widths=0.62,
    )
    for patch, method in zip(boxes["boxes"], METHODS, strict=True):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.75)
    axes[0, 1].axhline(1.0, color="#555555", linestyle=":", linewidth=0.8)
    axes[0, 1].set_yscale("log")
    axes[0, 1].tick_params(axis="x", rotation=18)
    axes[0, 1].set_ylabel("Normalized test MSE")
    axes[0, 1].set_title("Performance distribution")

    winners = Counter(row["winner"] for row in rows)
    axes[1, 0].bar(
        np.arange(len(METHODS)),
        [winners.get(method, 0) for method in METHODS],
        color=[COLORS[method] for method in METHODS],
    )
    axes[1, 0].set_xticks(np.arange(len(METHODS)), METHODS, rotation=18, ha="right")
    axes[1, 0].set_ylabel("Number of datasets")
    axes[1, 0].set_title("Lowest test MSE")

    selected = np.asarray([row["kalbi_selected_raw_lambda"] for row in rows])
    oracle = np.asarray([row["kalbi_oracle_raw_lambda"] for row in rows])
    axes[1, 1].scatter(
        oracle,
        selected,
        c=[COLORS[row["winner"]] for row in rows],
        s=28,
        edgecolor="#333333",
        linewidth=0.35,
        alpha=0.85,
    )
    limits = finite_range(np.concatenate([selected, oracle]), (1e-4, 1.0))
    positive_low = max(limits[0], 1e-6)
    positive_high = max(limits[1], positive_low * 1.01)
    axes[1, 1].plot(
        [positive_low, positive_high],
        [positive_low, positive_high],
        color="#444444",
        linestyle=":",
        linewidth=0.9,
    )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Oracle raw lambda")
    axes[1, 1].set_ylabel("KALBI-selected raw lambda")
    axes[1, 1].set_title("Lambda selection")

    for axis in axes.flat:
        style_axis(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    paths = save_figure(
        figure, output_dir / "kalbi_vs_cv_lasso_scad_mcp_all_real_pgtN"
    )
    plt.close(figure)
    return paths


def render_np_winner(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    figure, axis = plt.subplots(figsize=(7.2, 5.3), constrained_layout=True)
    for method in METHODS:
        subset = [row for row in rows if row["winner"] == method]
        if not subset:
            continue
        axis.scatter(
            [row["N"] for row in subset],
            [row["P_transformed"] for row in subset],
            s=38,
            color=COLORS[method],
            label=method,
            edgecolor="#333333",
            linewidth=0.35,
            alpha=0.82,
        )
    lower = min(min(row["N"], row["P_transformed"]) for row in rows)
    upper = max(max(row["N"], row["P_transformed"]) for row in rows)
    axis.plot([lower, upper], [lower, upper], color="#555555", linestyle=":", linewidth=0.8)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("N")
    axis.set_ylabel("P after train-only filtering")
    axis.legend(frameon=False, ncol=2)
    style_axis(axis)
    paths = save_figure(figure, output_dir / "winner_by_n_p")
    plt.close(figure)
    return paths


def render_penalty_diversity(
    rows: list[dict[str, Any]],
    arrays: list[dict[str, np.ndarray]],
    output_dir: Path,
) -> dict[str, str]:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    for row, values in zip(rows, arrays, strict=True):
        grid = values["penalty_grid"]
        keep = grid <= 1.0 + 1e-12
        color = COLORS[row["winner"]]
        axes[0].plot(grid[keep], values["penalty_rho"][keep], color=color, alpha=0.28)
        axes[1].plot(grid[keep], values["penalty_drho"][keep], color=color, alpha=0.28)
    axes[0].set_xlabel(r"$|\beta|$")
    axes[0].set_ylabel(r"$\rho(|\beta|)$")
    axes[0].set_title("Selected penalty functions")
    axes[1].set_xlabel(r"$|\beta|$")
    axes[1].set_ylabel(r"$\rho'(|\beta|)$")
    axes[1].set_title("Selected penalty derivatives")
    for axis in axes:
        axis.set_xlim(0.0, 1.0)
        style_axis(axis)
    paths = save_figure(figure, output_dir / "kalbi_selected_penalty_diversity")
    plt.close(figure)
    return paths


def render_paired_differences(
    rows: list[dict[str, Any]], output_dir: Path
) -> dict[str, str]:
    baselines = ("CV-LASSO", "CV-SCAD", "CV-MCP")
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.7), constrained_layout=True)
    for axis, method in zip(axes, baselines, strict=True):
        key = method.lower().replace("cv-", "")
        delta = np.asarray(
            [
                row["kalbi_normalized_mse"] - row[f"{key}_normalized_mse"]
                for row in rows
                if np.isfinite(row[f"{key}_normalized_mse"])
            ]
        )
        axis.axhline(0.0, color="#555555", linestyle=":", linewidth=0.9)
        axis.scatter(
            np.ones(delta.size),
            delta,
            color=COLORS[method],
            edgecolor="#333333",
            linewidth=0.3,
            s=24,
            alpha=0.75,
        )
        axis.boxplot(
            [delta],
            positions=[1],
            widths=0.35,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.2},
            boxprops={"color": "#111111"},
            whiskerprops={"color": "#111111"},
            capprops={"color": "#111111"},
        )
        axis.set_xticks([1], [method])
        axis.set_ylabel("KALBI minus baseline normalized MSE")
        axis.set_title(f"Median = {np.median(delta):.3g}")
        style_axis(axis)
    paths = save_figure(figure, output_dir / "paired_kalbi_minus_baseline_error")
    plt.close(figure)
    return paths


def render_family_macro(
    rows: list[dict[str, Any]], output_dir: Path
) -> dict[str, str]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["source_family"]].append(row)
    families = sorted(grouped, key=lambda name: (-len(grouped[name]), name))
    figure_height = max(4.2, 0.42 * len(families) + 1.6)
    figure, axis = plt.subplots(
        figsize=(10.6, figure_height), constrained_layout=True
    )
    y = np.arange(len(families))
    width = 0.18
    for method_index, method in enumerate(METHODS):
        key = method.lower().replace("cv-", "")
        values = [
            (
                float(np.exp(np.mean(np.log(finite_method_values(
                    grouped[family], method, "normalized_mse"
                )))))
                if finite_method_values(
                    grouped[family], method, "normalized_mse"
                ).size
                else float("nan")
            )
            for family in families
        ]
        axis.barh(
            y + (method_index - 1.5) * width,
            values,
            height=width,
            color=COLORS[method],
            label=method,
        )
    axis.axvline(1.0, color="#555555", linestyle=":", linewidth=0.9)
    axis.set_yticks(
        y,
        [f"{family} (n={len(grouped[family])})" for family in families],
    )
    axis.invert_yaxis()
    axis.set_xscale("log")
    axis.set_xlabel("Family geometric-mean normalized test MSE")
    axis.legend(frameon=False, ncol=2)
    style_axis(axis)
    paths = save_figure(figure, output_dir / "source_family_macro_average")
    plt.close(figure)
    return paths


def render_tier_summary(
    rows: list[dict[str, Any]], output_dir: Path
) -> dict[str, str]:
    tiers = sorted({row["evidence_tier"] for row in rows})
    figure, axes = plt.subplots(
        1, len(tiers), figsize=(5.2 * len(tiers), 4.1), constrained_layout=True
    )
    axes_array = np.atleast_1d(axes)
    for axis, tier in zip(axes_array, tiers, strict=True):
        subset = [row for row in rows if row["evidence_tier"] == tier]
        values = [
            (
                float(np.median(finite_method_values(
                    subset, method, "normalized_mse"
                )))
                if finite_method_values(
                    subset, method, "normalized_mse"
                ).size
                else float("nan")
            )
            for method in METHODS
        ]
        axis.bar(
            np.arange(len(METHODS)),
            values,
            color=[COLORS[method] for method in METHODS],
        )
        axis.axhline(1.0, color="#555555", linestyle=":", linewidth=0.9)
        axis.set_xticks(
            np.arange(len(METHODS)), METHODS, rotation=18, ha="right"
        )
        axis.set_yscale("log")
        axis.set_ylabel("Median normalized test MSE")
        axis.set_title(f"{tier.replace('_', ' ')} (n={len(subset)})")
        style_axis(axis)
    paths = save_figure(figure, output_dir / "standard_and_exploratory_summary")
    plt.close(figure)
    return paths


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    method_summary: dict[str, Any] = {}
    for method in METHODS:
        values = finite_method_values(rows, method, "normalized_mse")
        r2 = finite_method_values(rows, method, "test_r2")
        method_summary[method] = {
            "evaluated_task_count": int(values.size),
            "timeout_or_missing_task_count": int(len(rows) - values.size),
            "mean_normalized_mse": float(values.mean()),
            "median_normalized_mse": float(np.median(values)),
            "geometric_mean_normalized_mse": float(np.exp(np.mean(np.log(values)))),
            "mean_test_r2": float(r2.mean()),
            "median_test_r2": float(np.median(r2)),
            "win_count": int(sum(row["winner"] == method for row in rows)),
        }
    family: dict[str, Any] = {}
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["source_family"]].append(row)
    for name, family_rows in sorted(grouped.items()):
        family_medians: dict[str, float | None] = {}
        family_geometric_means: dict[str, float | None] = {}
        family_evaluated_counts: dict[str, int] = {}
        for method in METHODS:
            values = finite_method_values(
                family_rows, method, "normalized_mse"
            )
            family_evaluated_counts[method] = int(values.size)
            family_medians[method] = (
                float(np.median(values)) if values.size else None
            )
            family_geometric_means[method] = (
                float(np.exp(np.mean(np.log(values))))
                if values.size
                else None
            )
        family[name] = {
            "task_count": len(family_rows),
            "evaluated_task_count": family_evaluated_counts,
            "median_normalized_mse": family_medians,
            "geometric_mean_normalized_mse": family_geometric_means,
        }
    pairwise: dict[str, Any] = {}
    for method in METHODS[1:]:
        key = method.lower().replace("cv-", "")
        delta = np.asarray(
            [
                row["kalbi_normalized_mse"] - row[f"{key}_normalized_mse"]
                for row in rows
                if np.isfinite(row[f"{key}_normalized_mse"])
            ]
        )
        scale = np.asarray(
            [
                max(1.0, row["kalbi_normalized_mse"], row[f"{key}_normalized_mse"])
                for row in rows
                if np.isfinite(row[f"{key}_normalized_mse"])
            ]
        )
        tied = np.abs(delta) <= 1e-10 * scale
        pairwise[method] = {
            "paired_task_count": int(delta.size),
            "kalbi_wins": int(np.sum((delta < 0) & ~tied)),
            "ties": int(np.sum(tied)),
            "kalbi_losses": int(np.sum((delta > 0) & ~tied)),
            "mean_kalbi_minus_baseline_normalized_mse": float(delta.mean()),
            "median_kalbi_minus_baseline_normalized_mse": float(
                np.median(delta)
            ),
        }
    tiers: dict[str, Any] = {}
    tier_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tier_groups[row["evidence_tier"]].append(row)
    for name, tier_rows in sorted(tier_groups.items()):
        tier_methods: dict[str, Any] = {}
        for method in METHODS:
            values = finite_method_values(
                tier_rows, method, "normalized_mse"
            )
            tier_methods[method] = {
                "evaluated_task_count": int(values.size),
                "mean_normalized_mse": (
                    float(np.mean(values)) if values.size else None
                ),
                "median_normalized_mse": (
                    float(np.median(values)) if values.size else None
                ),
            }
        tiers[name] = {
            "task_count": len(tier_rows),
            "methods": tier_methods,
        }
    family_macro: dict[str, Any] = {}
    for method in METHODS:
        geometric = [
            value["geometric_mean_normalized_mse"][method]
            for value in family.values()
            if value["geometric_mean_normalized_mse"][method] is not None
        ]
        medians = [
            value["median_normalized_mse"][method]
            for value in family.values()
            if value["median_normalized_mse"][method] is not None
        ]
        family_macro[method] = {
            "evaluated_family_count": len(geometric),
            "mean_of_family_geometric_means": float(np.mean(geometric)),
            "median_of_family_medians": float(np.median(medians)),
        }
    return {
        "task_count": len(rows),
        "nonconvex_timeout_task_count": int(
            sum(row["nonconvex_timeout"] for row in rows)
        ),
        "methods": method_summary,
        "families": family,
        "family_macro": family_macro,
        "evidence_tiers": tiers,
        "pairwise_kalbi_vs_baselines": pairwise,
        "kalbi_selected_over_oracle_mse": {
            "mean": float(
                np.mean([row["kalbi_selected_over_oracle_mse"] for row in rows])
            ),
            "median": float(
                np.median([row["kalbi_selected_over_oracle_mse"] for row in rows])
            ),
        },
        "coordinate_descent_audit_pass_count": int(
            sum(row["cd_audit_pass"] for row in rows)
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    public = [public_row(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(public[0]))
        writer.writeheader()
        writer.writerows(public)


def main() -> None:
    args = parse_args()
    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.8,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": False,
        }
    )
    split_manifest = load_json(args.split_manifest)
    rows, arrays, failures = collect_rows(args.evaluation_root, split_manifest)
    if failures and not args.allow_partial:
        raise RuntimeError(
            f"{len(failures)} tasks lack complete KALBI/baseline artifacts"
        )
    if not rows:
        raise RuntimeError("no complete evaluation tasks")
    if (
        not args.allow_partial
        and len(rows) != args.expected_task_count
    ):
        raise RuntimeError(
            f"expected {args.expected_task_count} tasks, found {len(rows)}"
        )

    per_task = args.output_root / "per_task"
    summary = args.output_root / "summary"
    per_task.mkdir(parents=True, exist_ok=True)
    summary.mkdir(parents=True, exist_ok=True)
    multipage_path = args.output_root / "kalbi_real_pgtN_all_tasks.pdf"
    artifact_rows: list[dict[str, Any]] = []
    with PdfPages(multipage_path) as pdf:
        for row, task_arrays in zip(rows, arrays, strict=True):
            figure, paths = render_task(row, task_arrays, per_task)
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
            artifact_rows.append(
                {
                    "dataset_index": row["dataset_index"],
                    "task_id": row["task_id"],
                    **paths,
                }
            )

    summary_artifacts = {
        "method_comparison": render_method_summary(rows, summary),
        "paired_differences": render_paired_differences(rows, summary),
        "winner_by_n_p": render_np_winner(rows, summary),
        "penalty_diversity": render_penalty_diversity(rows, arrays, summary),
        "source_family_macro": render_family_macro(rows, summary),
        "evidence_tiers": render_tier_summary(rows, summary),
    }
    comparison_csv = args.output_root / "comparison.csv"
    comparison_json = args.output_root / "comparison.json"
    aggregate_json = args.output_root / "aggregate_metrics.json"
    write_csv(comparison_csv, rows)
    atomic_json(comparison_json, [public_row(row) for row in rows])
    aggregate_metrics = aggregate(rows)
    atomic_json(aggregate_json, aggregate_metrics)
    timeout_task_count = int(
        sum(row["nonconvex_timeout"] for row in rows)
    )
    manifest = {
        "status": (
            "PARTIAL"
            if failures
            else (
                "PASS_WITH_NONCONVEX_TIMEOUTS"
                if timeout_task_count
                else "PASS"
            )
        ),
        "task_count": len(rows),
        "nonconvex_timeout_task_count": timeout_task_count,
        "expected_task_count": args.expected_task_count,
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": sha256(args.split_manifest),
        "evaluation_root": str(args.evaluation_root),
        "multipage_pdf": str(multipage_path),
        "multipage_pdf_sha256": sha256(multipage_path),
        "per_task_artifacts": artifact_rows,
        "summary_artifacts": summary_artifacts,
        "comparison_csv": str(comparison_csv),
        "comparison_csv_sha256": sha256(comparison_csv),
        "comparison_json": str(comparison_json),
        "comparison_json_sha256": sha256(comparison_json),
        "aggregate_metrics": str(aggregate_json),
        "aggregate_metrics_sha256": sha256(aggregate_json),
        "failures": failures,
    }
    atomic_json(args.output_root / "figure_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "task_count": len(rows),
                "output_root": str(args.output_root),
                "multipage_pdf": str(multipage_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
