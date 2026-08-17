#!/usr/bin/env python3
"""Analyze V16 raw and 1%-stage-gained trajectory norms on 10 tasks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .selector_v16_residualtrajectory import (
    V16_SELECTOR_VERSION,
    residual_isis_trajectory_v16,
)


Tensor = torch.Tensor
BLUE = "#2F6B8A"
ORANGE = "#D97706"
INK = "#20252B"
GRID = "#D9DEE3"


def make_task(
    task_id: int,
    *,
    rows: int,
    features: int,
    block_size: int,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor, dict[str, float | int]]:
    generator = torch.Generator(device="cpu").manual_seed(seed + task_id)
    blocks = math.ceil(features / block_size)
    rho_global = 0.03 + 0.01 * (task_id % 5)
    rho_block = 0.18 + 0.02 * task_id
    independent_weight = 1.0 - rho_global - rho_block
    if independent_weight <= 0.0:
        raise ValueError("correlation weights must sum to less than one")

    global_factor = torch.randn(rows, 1, generator=generator)
    block_factors = torch.randn(rows, blocks, generator=generator)
    block_index = torch.arange(features) // block_size
    independent = torch.randn(rows, features, generator=generator)
    x = (
        math.sqrt(rho_global) * global_factor
        + math.sqrt(rho_block) * block_factors[:, block_index]
        + math.sqrt(independent_weight) * independent
    )
    x = x - x.mean(dim=0, keepdim=True)
    x = x / x.square().mean(dim=0, keepdim=True).sqrt().clamp_min(1e-6)

    support_size = 32 + 4 * (task_id % 5)
    support = torch.randperm(features, generator=generator)[:support_size]
    amplitudes = torch.exp(
        torch.linspace(math.log(1.5), math.log(0.15), support_size)
    )
    signs = torch.where(
        torch.rand(support_size, generator=generator) >= 0.5,
        torch.ones(support_size),
        -torch.ones(support_size),
    )
    beta = torch.zeros(features)
    beta[support] = amplitudes * signs
    signal = x @ beta
    snr = 2.5 + 0.5 * (task_id % 6)
    noise_scale = signal.square().mean().sqrt() / math.sqrt(snr)
    y = signal + noise_scale * torch.randn(rows, generator=generator)
    y = y - y.mean()
    y = y / y.square().mean().sqrt().clamp_min(1e-6)
    metadata: dict[str, float | int] = {
        "task_id": task_id + 1,
        "rho_global": rho_global,
        "rho_block": rho_block,
        "theoretical_within_block_corr": rho_global + rho_block,
        "theoretical_between_block_corr": rho_global,
        "support_size": support_size,
        "snr": snr,
    }
    return x, y, beta, metadata


def correlation_metrics(
    x: Tensor,
    *,
    block_size: int,
    generator: torch.Generator,
    sampled_pairs: int = 200_000,
) -> tuple[dict[str, float], Tensor]:
    rows, features = x.shape
    correlation = x.T @ x / float(rows)
    absolute = correlation.abs()
    diagonal = torch.arange(features)
    absolute[diagonal, diagonal] = 0.0
    mean_abs = float(absolute.sum() / float(features * (features - 1)))
    max_abs = float(absolute.max())

    left = torch.randint(features, (sampled_pairs,), generator=generator)
    right = torch.randint(features - 1, (sampled_pairs,), generator=generator)
    right = right + (right >= left).long()
    sampled = absolute[left, right]
    within = sampled[(left // block_size) == (right // block_size)]
    between = sampled[(left // block_size) != (right // block_size)]
    metrics = {
        "mean_abs_corr": mean_abs,
        "p95_abs_corr": float(torch.quantile(sampled, 0.95)),
        "p99_abs_corr": float(torch.quantile(sampled, 0.99)),
        "max_abs_corr": max_abs,
        "sampled_within_block_mean_abs_corr": float(within.mean()),
        "sampled_between_block_mean_abs_corr": float(between.mean()),
    }
    return metrics, correlation[:64, :64].clone()


def column_l2(values: Tensor, dimension: int) -> Tensor:
    return values.square().sum(dim=dimension).sqrt()


def add_task_lines_and_band(
    axis: plt.Axes,
    stages: np.ndarray,
    values: np.ndarray,
    *,
    color: str,
    label: str,
    linestyle: str = "-",
) -> None:
    clipped = np.clip(values, 1e-12, None)
    for task_values in clipped:
        axis.plot(
            stages,
            task_values,
            color=color,
            alpha=0.20,
            linewidth=0.9,
            linestyle=linestyle,
        )
    lower, median, upper = np.quantile(clipped, [0.1, 0.5, 0.9], axis=0)
    axis.fill_between(stages, lower, upper, color=color, alpha=0.12, linewidth=0)
    axis.plot(
        stages,
        median,
        color=color,
        linewidth=2.4,
        linestyle=linestyle,
        label=label,
    )


def style_axis(axis: plt.Axes, title: str) -> None:
    axis.set_title(title, fontsize=11, color=INK, pad=9)
    axis.set_xlabel("ISIS stage", color=INK)
    axis.set_ylabel("Column L2 norm (log scale)", color=INK)
    axis.set_yscale("log")
    axis.set_xlim(1, 256)
    axis.set_xticks([1, 64, 128, 192, 256])
    axis.grid(True, which="major", color=GRID, linewidth=0.7, alpha=0.8)
    axis.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.35)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#7A838C")
    axis.spines["bottom"].set_color("#7A838C")
    axis.tick_params(colors="#505A63", labelsize=9)


def plot_norms(
    output_path: Path,
    records: list[dict[str, np.ndarray]],
    correlation_summary: dict[str, float],
) -> None:
    stages = np.arange(1, 257)
    raw_r = np.stack([record["r_raw_l2"] for record in records])
    gained_r = np.stack([record["r_gained_l2"] for record in records])
    raw_beta = np.stack([record["beta_residual_raw_l2"] for record in records])
    gained_beta = np.stack([record["beta_residual_gained_l2"] for record in records])
    marginal_beta = np.stack([record["beta_marginal_l2"] for record in records])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor="white")
    add_task_lines_and_band(
        axes[0, 0], stages, raw_r, color=BLUE, label="Median raw R"
    )
    style_axis(axes[0, 0], "Raw residual trajectory R")
    axes[0, 0].legend(frameon=False, fontsize=9)

    add_task_lines_and_band(
        axes[0, 1], stages, gained_r, color=ORANGE, label="Median gained R"
    )
    style_axis(axes[0, 1], "R after post-hoc 1% stage gain")
    axes[0, 1].legend(frameon=False, fontsize=9)

    add_task_lines_and_band(
        axes[1, 0], stages, raw_beta, color=BLUE, label="Raw residual beta"
    )
    add_task_lines_and_band(
        axes[1, 0],
        stages,
        gained_beta,
        color=ORANGE,
        label="Residual beta with 1% gain",
        linestyle="--",
    )
    style_axis(axes[1, 0], "Residual ridge coefficient columns")
    axes[1, 0].legend(frameon=False, fontsize=9)

    add_task_lines_and_band(
        axes[1, 1],
        stages,
        marginal_beta,
        color=BLUE,
        label="Marginal beta (no gain)",
    )
    style_axis(axes[1, 1], "Marginal ridge coefficient columns")
    axes[1, 1].legend(frameon=False, fontsize=9)

    fig.suptitle(
        "V16 residual-path column norms across 10 moderately correlated tasks",
        fontsize=16,
        color=INK,
        y=0.985,
    )
    fig.text(
        0.5,
        0.955,
        (
            "N=512, P=2,048, K=4, S=256; thin lines are tasks, bold line is "
            "median, band is p10-p90. "
            f"Mean |corr|={correlation_summary['mean_abs_corr_mean']:.3f}, "
            f"mean within-block |corr|="
            f"{correlation_summary['within_block_mean_abs_corr_mean']:.3f}."
        ),
        ha="center",
        va="top",
        fontsize=10,
        color="#505A63",
    )
    fig.tight_layout(rect=(0.03, 0.03, 0.98, 0.93))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_correlation_heatmaps(
    output_path: Path,
    heatmaps: list[Tensor],
    summaries: list[dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), facecolor="white")
    image = None
    for task_index, (axis, heatmap, summary) in enumerate(
        zip(axes.flat, heatmaps, summaries, strict=True)
    ):
        image = axis.imshow(
            heatmap.numpy(),
            cmap="RdBu_r",
            vmin=-0.65,
            vmax=0.65,
            interpolation="nearest",
        )
        axis.set_title(
            f"Task {task_index + 1} | within={summary['sampled_within_block_mean_abs_corr']:.2f}",
            fontsize=9,
            color=INK,
        )
        axis.set_xticks([0, 31, 63])
        axis.set_yticks([0, 31, 63])
        axis.tick_params(labelsize=7, colors="#505A63")
        for spine in axis.spines.values():
            spine.set_visible(False)
    if image is not None:
        colorbar_axis = fig.add_axes((0.925, 0.15, 0.014, 0.68))
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Feature correlation", color=INK)
    fig.suptitle(
        "Feature-correlation heatmaps for 10 synthetic tasks",
        fontsize=16,
        color=INK,
        y=0.98,
    )
    fig.text(
        0.5,
        0.94,
        "First 64 of 2,048 features; 32-feature blocks with weak global correlation.",
        ha="center",
        fontsize=10,
        color="#505A63",
    )
    fig.subplots_adjust(
        left=0.04,
        right=0.89,
        bottom=0.06,
        top=0.88,
        wspace=0.18,
        hspace=0.25,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026071801)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--features", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.tasks != 10:
        raise ValueError("this diagnostic contract fixes tasks=10")
    if args.features < 1024:
        raise ValueError("P must be at least K*S=1024 to activate all stages")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, args.threads))
    correlation_generator = torch.Generator(device="cpu").manual_seed(args.seed + 50_000)
    records: list[dict[str, np.ndarray]] = []
    summaries: list[dict[str, Any]] = []
    heatmaps: list[Tensor] = []
    saved_tasks: list[dict[str, Any]] = []
    csv_rows: list[dict[str, float | int]] = []

    for task_index in range(args.tasks):
        x, y, beta_true, metadata = make_task(
            task_index,
            rows=args.rows,
            features=args.features,
            block_size=args.block_size,
            seed=args.seed,
        )
        corr_metrics, heatmap = correlation_metrics(
            x,
            block_size=args.block_size,
            generator=correlation_generator,
        )
        trajectory = residual_isis_trajectory_v16(x[None], y[None])
        active_stages = int(trajectory["stage_active"].sum())
        if active_stages != 256:
            raise RuntimeError(f"task {task_index + 1} has {active_stages} active stages")

        record = {
            "r_raw_l2": column_l2(trajectory["residual_trajectory_raw"][0], 0).numpy(),
            "r_gained_l2": column_l2(trajectory["residual_trajectory"][0], 0).numpy(),
            "beta_residual_raw_l2": column_l2(
                trajectory["beta_residual_raw"][0], 0
            ).numpy(),
            "beta_residual_gained_l2": column_l2(
                trajectory["beta_residual"][0], 0
            ).numpy(),
            "beta_marginal_l2": column_l2(trajectory["beta_marginal"][0], 0).numpy(),
            "stage_gain": trajectory["stage_gain"].numpy(),
            "residual_scale_before": trajectory["residual_scale_before"][0].numpy(),
            "explained_fraction": trajectory["stage_explained_fraction"][0].numpy(),
        }
        records.append(record)
        task_summary: dict[str, Any] = {
            **metadata,
            **corr_metrics,
            "active_stages": active_stages,
            "r_raw_l2_stage1": float(record["r_raw_l2"][0]),
            "r_raw_l2_stage256": float(record["r_raw_l2"][-1]),
            "r_gained_l2_stage256": float(record["r_gained_l2"][-1]),
            "beta_residual_raw_l2_stage1": float(
                record["beta_residual_raw_l2"][0]
            ),
            "beta_residual_raw_l2_stage256": float(
                record["beta_residual_raw_l2"][-1]
            ),
            "beta_residual_gained_l2_stage256": float(
                record["beta_residual_gained_l2"][-1]
            ),
        }
        summaries.append(task_summary)
        heatmaps.append(heatmap)
        saved_tasks.append(
            {
                "x": x,
                "y": y,
                "beta_true": beta_true,
                "metadata": metadata,
            }
        )
        for stage_index in range(256):
            csv_rows.append(
                {
                    "task": task_index + 1,
                    "stage": stage_index + 1,
                    "stage_gain": float(record["stage_gain"][stage_index]),
                    "r_raw_l2": float(record["r_raw_l2"][stage_index]),
                    "r_gained_l2": float(record["r_gained_l2"][stage_index]),
                    "beta_residual_raw_l2": float(
                        record["beta_residual_raw_l2"][stage_index]
                    ),
                    "beta_residual_gained_l2": float(
                        record["beta_residual_gained_l2"][stage_index]
                    ),
                    "beta_marginal_l2": float(
                        record["beta_marginal_l2"][stage_index]
                    ),
                    "residual_scale_before": float(
                        record["residual_scale_before"][stage_index]
                    ),
                    "explained_fraction": float(
                        record["explained_fraction"][stage_index]
                    ),
                }
            )

    correlation_summary = {
        "mean_abs_corr_mean": float(
            np.mean([summary["mean_abs_corr"] for summary in summaries])
        ),
        "p99_abs_corr_mean": float(
            np.mean([summary["p99_abs_corr"] for summary in summaries])
        ),
        "max_abs_corr_min": float(
            np.min([summary["max_abs_corr"] for summary in summaries])
        ),
        "max_abs_corr_max": float(
            np.max([summary["max_abs_corr"] for summary in summaries])
        ),
        "within_block_mean_abs_corr_mean": float(
            np.mean(
                [summary["sampled_within_block_mean_abs_corr"] for summary in summaries]
            )
        ),
        "between_block_mean_abs_corr_mean": float(
            np.mean(
                [summary["sampled_between_block_mean_abs_corr"] for summary in summaries]
            )
        ),
    }
    report = {
        "status": "ok",
        "selector_version": V16_SELECTOR_VERSION,
        "tasks": args.tasks,
        "N": args.rows,
        "P": args.features,
        "block_size": args.block_size,
        "stage_gain_rate": 0.01,
        "terminal_stage_gain": float(records[0]["stage_gain"][-1]),
        "correlation_summary": correlation_summary,
        "task_summaries": summaries,
    }

    with (args.output_dir / "column_norms.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    torch.save(
        {
            "selector_version": V16_SELECTOR_VERSION,
            "seed": args.seed,
            "tasks": saved_tasks,
        },
        args.output_dir / "moderate_corr_tasks_N512_P2048.pt",
    )
    plot_norms(
        args.output_dir / "column_norms.png",
        records,
        correlation_summary,
    )
    plot_correlation_heatmaps(
        args.output_dir / "correlation_heatmaps.png",
        heatmaps,
        summaries,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
