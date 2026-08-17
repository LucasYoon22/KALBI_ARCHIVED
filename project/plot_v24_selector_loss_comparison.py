#!/usr/bin/env python3
"""Plot rank-averaged V24 selector held-out loss across the V13/V14 transition."""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_updates(path: Path) -> list[dict[str, float]]:
    by_step: OrderedDict[int, dict[str, float]] = OrderedDict()
    with path.open() as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("kind") != "selector_train_update":
                continue
            step = event.get("selector_step")
            monitors = event.get("rank_monitors")
            if step is None or not isinstance(monitors, list):
                continue
            losses = [
                float(item["selector_heldout_loss"])
                for item in monitors
                if isinstance(item, dict)
                and item.get("selector_heldout_loss") is not None
                and np.isfinite(item["selector_heldout_loss"])
            ]
            if not losses:
                continue
            # A restarted run can repeat a step. The latest complete event wins.
            by_step[int(step)] = {
                "step": int(step),
                "loss": float(np.mean(losses)),
                "rank_std": float(np.std(losses)),
                "rank_count": len(losses),
                "timestamp": float(event.get("timestamp", np.nan)),
            }
    return sorted(by_step.values(), key=lambda item: item["step"])


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    if len(values) < window:
        return result
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    result[window - 1 :] = (cumulative[window:] - cumulative[:-window]) / window
    return result


def summarize(values: np.ndarray) -> dict[str, float]:
    width = min(200, len(values))
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "first_200_mean": float(np.mean(values[:width])),
        "last_200_mean": float(np.mean(values[-width:])),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-events", type=Path, required=True)
    parser.add_argument("--current-events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transition-step", type=int, default=3900)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous = [
        item
        for item in read_updates(args.previous_events)
        if item["step"] <= args.transition_step
    ]
    current = [
        item
        for item in read_updates(args.current_events)
        if item["step"] > args.transition_step
    ]
    if not previous or not current:
        raise RuntimeError("Both pre-transition and post-transition updates are required")

    previous_steps = np.array([item["step"] for item in previous])
    previous_loss = np.array([item["loss"] for item in previous])
    current_steps = np.array([item["step"] for item in current])
    current_loss = np.array([item["loss"] for item in current])
    window = 100

    comparable_count = min(len(previous), len(current))
    previous_matched = previous_loss[-comparable_count:]
    current_matched = current_loss[:comparable_count]
    relative_step = np.arange(1, comparable_count + 1)

    colors = {"previous": "#4C78A8", "current": "#E45756"}
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), constrained_layout=True)

    axes[0].scatter(
        previous_steps,
        previous_loss,
        s=5,
        alpha=0.10,
        color=colors["previous"],
        rasterized=True,
    )
    axes[0].plot(
        previous_steps,
        trailing_mean(previous_loss, window),
        color=colors["previous"],
        linewidth=2.2,
        label="Previous V24 data contract (V13), 100-step mean",
    )
    axes[0].scatter(
        current_steps,
        current_loss,
        s=7,
        alpha=0.16,
        color=colors["current"],
        rasterized=True,
    )
    axes[0].plot(
        current_steps,
        trailing_mean(current_loss, window),
        color=colors["current"],
        linewidth=2.2,
        label="Current V24 data contract (V14), 100-step mean",
    )
    axes[0].axvline(
        args.transition_step + 0.5,
        color="#333333",
        linestyle="--",
        linewidth=1.2,
        label="V13 to V14 transition",
    )
    axes[0].set_xlabel("Selector update")
    axes[0].set_ylabel("Held-out loss")
    axes[0].set_title("Full training trajectory")
    axes[0].legend(frameon=False, fontsize=8, loc="best")

    axes[1].scatter(
        relative_step,
        previous_matched,
        s=7,
        alpha=0.12,
        color=colors["previous"],
        rasterized=True,
    )
    axes[1].plot(
        relative_step,
        trailing_mean(previous_matched, window),
        color=colors["previous"],
        linewidth=2.2,
        label=f"Previous: last {comparable_count} updates",
    )
    axes[1].scatter(
        relative_step,
        current_matched,
        s=7,
        alpha=0.12,
        color=colors["current"],
        rasterized=True,
    )
    axes[1].plot(
        relative_step,
        trailing_mean(current_matched, window),
        color=colors["current"],
        linewidth=2.2,
        label=f"Current: first {comparable_count} updates",
    )
    axes[1].set_xlabel("Relative update")
    axes[1].set_ylabel("Held-out loss")
    axes[1].set_title("Matched-length comparison")
    axes[1].legend(frameon=False, fontsize=8, loc="best")

    box = axes[2].boxplot(
        [previous_matched, current_matched],
        tick_labels=["Previous V13", "Current V14"],
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.8},
    )
    box["boxes"][0].set_facecolor(colors["previous"])
    box["boxes"][1].set_facecolor(colors["current"])
    for patch in box["boxes"]:
        patch.set_alpha(0.75)
    axes[2].scatter(
        [1, 2],
        [np.mean(previous_matched), np.mean(current_matched)],
        color="#111111",
        marker="D",
        s=34,
        zorder=3,
        label="Mean",
    )
    axes[2].set_ylabel("Held-out loss")
    axes[2].set_title("Matched-window distribution")
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.65)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    png_path = args.output_dir / "v24_selector_heldout_loss_v13_vs_v14.png"
    pdf_path = args.output_dir / "v24_selector_heldout_loss_v13_vs_v14.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    csv_path = args.output_dir / "v24_selector_heldout_loss_updates.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["contract", "step", "loss", "rank_std", "rank_count", "timestamp"],
        )
        writer.writeheader()
        for contract, records in (("previous_v13", previous), ("current_v14", current)):
            for record in records:
                writer.writerow({"contract": contract, **record})

    metrics = {
        "loss_definition": "mean across 8 DDP ranks of selector_heldout_loss",
        "smoothing": f"trailing mean, window={window}",
        "transition_step": args.transition_step,
        "previous_v13": summarize(previous_loss),
        "current_v14": summarize(current_loss),
        "matched_count": comparable_count,
        "previous_matched": summarize(previous_matched),
        "current_matched": summarize(current_matched),
        "matched_mean_difference_current_minus_previous": float(
            np.mean(current_matched) - np.mean(previous_matched)
        ),
        "matched_mean_ratio_current_over_previous": float(
            np.mean(current_matched) / np.mean(previous_matched)
        ),
        "warning": (
            "The model/optimizer and loss definition continue across the boundary, "
            "but V13 and V14 use different response-generation distributions."
        ),
    }
    metrics_path = args.output_dir / "v24_selector_heldout_loss_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
