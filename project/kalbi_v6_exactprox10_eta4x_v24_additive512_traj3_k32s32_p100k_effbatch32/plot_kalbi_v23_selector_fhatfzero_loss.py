#!/usr/bin/env python3
"""Plot the normalized held-out loss for the V23 selector-only run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    for index in range(values.size):
        start = max(0, index - window + 1)
        result[index] = float(np.mean(values[start : index + 1]))
    return result


def read_series(events_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with events_path.open(encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("kind") != "selector_train_update":
                continue
            monitors = event.get("rank_monitors", [])
            losses = np.asarray(
                [monitor.get("selector_heldout_loss", math.nan) for monitor in monitors],
                dtype=float,
            )
            losses = losses[np.isfinite(losses)]
            if losses.size == 0:
                continue
            rows.append(
                {
                    "selector_step": int(event["selector_step"]),
                    "loss_mean": float(np.mean(losses)),
                    "loss_q25": float(np.quantile(losses, 0.25)),
                    "loss_q75": float(np.quantile(losses, 0.75)),
                    "loss_min": float(np.min(losses)),
                    "loss_max": float(np.max(losses)),
                    "rank_count": int(losses.size),
                    "seconds_per_step": float(
                        event.get(
                            "average_compute_seconds_per_selector_step",
                            math.nan,
                        )
                    ),
                }
            )
    if not rows:
        raise RuntimeError(f"no selector_train_update events in {events_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()

    rows = read_series(args.events)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    steps = np.asarray([row["selector_step"] for row in rows], dtype=int)
    means = np.asarray([row["loss_mean"] for row in rows], dtype=float)
    q25 = np.asarray([row["loss_q25"] for row in rows], dtype=float)
    q75 = np.asarray([row["loss_q75"] for row in rows], dtype=float)
    smooth = trailing_mean(means, args.window)

    csv_path = args.output_dir / "selector_loss_series.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    blue = "#0072B2"
    orange = "#E69F00"
    charcoal = "#2F3437"
    grid = "#D9DEE3"
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": charcoal,
            "axes.labelcolor": charcoal,
            "xtick.color": charcoal,
            "ytick.color": charcoal,
            "text.color": charcoal,
        }
    )

    fig, axis = plt.subplots(figsize=(10.8, 5.6))
    axis.fill_between(
        steps,
        q25,
        q75,
        color=blue,
        alpha=0.14,
        linewidth=0,
        label="Interquartile range across 8 ranks",
    )
    axis.plot(
        steps,
        means,
        color=blue,
        linewidth=1.15,
        marker="o",
        markersize=4.2,
        alpha=0.72,
        label="Logged rank mean",
    )
    axis.plot(
        steps,
        smooth,
        color=orange,
        linewidth=2.6,
        label=f"{args.window * 10}-step trailing mean",
    )
    checkpoint_steps = np.arange(100, int(steps[-1]) + 1, 100)
    for index, checkpoint_step in enumerate(checkpoint_steps):
        axis.axvline(
            checkpoint_step,
            color=charcoal,
            linewidth=0.8,
            linestyle=":",
            alpha=0.35,
            label="Checkpoint" if index == 0 else None,
        )

    axis.scatter(
        [steps[-1]],
        [smooth[-1]],
        color=orange,
        edgecolor="white",
        linewidth=0.9,
        s=56,
        zorder=5,
    )
    axis.annotate(
        f"{smooth[-1]:.3f}",
        xy=(steps[-1], smooth[-1]),
        xytext=(-8, 12),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=charcoal,
    )
    axis.set_title("Lambda selector training loss")
    axis.set_xlabel("Selector update")
    axis.set_ylabel(r"Mean $\log(1+\hat{F}/F_0)$")
    axis.set_xlim(max(0, int(steps[0]) - 10), int(steps[-1]) + 10)
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color=grid, linewidth=0.7, alpha=0.85)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout()

    png_path = args.output_dir / "kalbi_v23_selector_loss.png"
    pdf_path = args.output_dir / "kalbi_v23_selector_loss.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "objective": "mean(log1p(F_hat/F_zero))",
        "first_logged_step": int(steps[0]),
        "last_logged_step": int(steps[-1]),
        "logged_points": int(steps.size),
        "first_loss_mean": float(means[0]),
        "last_loss_mean": float(means[-1]),
        "first_trailing_mean": float(smooth[0]),
        "last_trailing_mean": float(smooth[-1]),
        "minimum_trailing_mean": float(np.min(smooth)),
        "relative_trailing_change": float(smooth[-1] / smooth[0] - 1.0),
        "png": str(png_path),
        "pdf": str(pdf_path),
        "csv": str(csv_path),
    }
    summary_path = args.output_dir / "selector_loss_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
