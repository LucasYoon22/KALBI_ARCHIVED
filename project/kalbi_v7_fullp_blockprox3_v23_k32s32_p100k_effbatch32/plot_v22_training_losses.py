#!/usr/bin/env python3
"""Plot KALBI V22 beta and selector training losses from JSONL events."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np


BETA_FIELDS = {
    "beta_loss": "loss",
    "beta_mse": "mse",
    "beta_objective": "objective",
    "beta_penalty": "penalty",
    "beta_kkt_top32": "kkt_top32",
}
SELECTOR_FIELDS = (
    "selector_objective",
    "selector_heldout_mse",
)


def finite_mean(values: list[object]) -> float:
    numeric = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    return mean(numeric) if numeric else math.nan


def read_events(path: Path, start_step: int) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(errors="ignore") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("kind") != "train_update":
                continue
            step = int(event.get("step", -1))
            if step < start_step:
                continue
            monitors = event.get("rank_monitors", [])
            if not monitors:
                continue

            row: dict[str, float] = {
                "step": float(step),
                "seconds_per_step": float(
                    event.get("average_compute_seconds_per_step", math.nan)
                ),
            }
            for output_name, metric_name in BETA_FIELDS.items():
                row[output_name] = finite_mean(
                    [
                        monitor.get("beta_metrics", {}).get(metric_name)
                        for monitor in monitors
                    ]
                )
            for field in SELECTOR_FIELDS:
                row[field] = finite_mean([monitor.get(field) for monitor in monitors])
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No train_update events at or after step {start_step}")
    return rows


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    finite_values = values[finite_indices]
    if finite_values.size == 0:
        return result
    for index in range(finite_values.size):
        start = max(0, index - window + 1)
        result[finite_indices[index]] = np.mean(finite_values[start : index + 1])
    return result


def endpoint_summary(values: np.ndarray, window: int = 20) -> dict[str, float]:
    values = values[np.isfinite(values)]
    width = min(window, values.size)
    if width == 0:
        return {"count": 0}
    first = float(np.mean(values[:width]))
    last = float(np.mean(values[-width:]))
    return {
        "count": int(values.size),
        "window": int(width),
        "first_window_mean": first,
        "last_window_mean": last,
        "relative_change": (last / first - 1.0) if first != 0 else math.nan,
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def write_csv(rows: list[dict[str, float]], path: Path) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, float]], output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = np.asarray([row["step"] for row in rows])
    arrays = {
        field: np.asarray([row.get(field, math.nan) for row in rows], dtype=float)
        for field in rows[0]
        if field != "step"
    }

    blue = "#0072B2"
    orange = "#E69F00"
    green = "#009E73"
    vermillion = "#D55E00"
    charcoal = "#2F3437"
    grid = "#D9DEE3"

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": charcoal,
            "axes.labelcolor": charcoal,
            "xtick.color": charcoal,
            "ytick.color": charcoal,
            "text.color": charcoal,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.2), sharex=True)
    fig.suptitle(
        f"KALBI V22 training losses | steps {int(steps[0]):,}-{int(steps[-1]):,}",
        fontsize=16,
        y=0.985,
    )

    def style_axis(axis: plt.Axes) -> None:
        axis.grid(axis="y", color=grid, linewidth=0.7, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    beta_loss = arrays["beta_loss"]
    axes[0, 0].scatter(
        steps, beta_loss, s=13, color=blue, alpha=0.23, edgecolors="none", label="Logged batch"
    )
    axes[0, 0].plot(
        steps,
        trailing_mean(beta_loss, 15),
        color=blue,
        linewidth=2.2,
        label="150-step trailing mean",
    )
    axes[0, 0].set_title("Beta training loss")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(frameon=False)
    style_axis(axes[0, 0])

    axes[0, 1].plot(
        steps,
        trailing_mean(arrays["beta_mse"], 15),
        color=orange,
        linewidth=2.1,
        label="MSE",
    )
    axes[0, 1].plot(
        steps,
        trailing_mean(arrays["beta_objective"], 15),
        color=green,
        linewidth=2.1,
        label="Penalized objective",
    )
    axes[0, 1].plot(
        steps,
        trailing_mean(arrays["beta_penalty"], 15),
        color=vermillion,
        linewidth=1.8,
        label="Penalty term",
    )
    axes[0, 1].set_title("Beta loss components")
    axes[0, 1].set_ylabel("Rank-mean value")
    axes[0, 1].legend(frameon=False)
    style_axis(axes[0, 1])

    selector_objective = arrays["selector_objective"]
    selector_mask = np.isfinite(selector_objective)
    axes[1, 0].scatter(
        steps[selector_mask],
        selector_objective[selector_mask],
        s=17,
        color=green,
        alpha=0.30,
        edgecolors="none",
        label="Selector update",
    )
    axes[1, 0].plot(
        steps[selector_mask],
        trailing_mean(selector_objective, 8)[selector_mask],
        color=green,
        linewidth=2.2,
        label="8-update trailing mean",
    )
    axes[1, 0].set_title("V22 selector objective")
    axes[1, 0].set_xlabel("Global training step")
    axes[1, 0].set_ylabel(r"mean log1p(held-out MSE / 2)")
    axes[1, 0].legend(frameon=False)
    style_axis(axes[1, 0])

    heldout_mse = arrays["selector_heldout_mse"]
    heldout_mask = np.isfinite(heldout_mse)
    axes[1, 1].scatter(
        steps[heldout_mask],
        heldout_mse[heldout_mask],
        s=17,
        color=vermillion,
        alpha=0.28,
        edgecolors="none",
        label="Selector update",
    )
    axes[1, 1].plot(
        steps[heldout_mask],
        trailing_mean(heldout_mse, 8)[heldout_mask],
        color=vermillion,
        linewidth=2.2,
        label="8-update trailing mean",
    )
    axes[1, 1].set_title("V22 selector held-out prediction error")
    axes[1, 1].set_xlabel("Global training step")
    axes[1, 1].set_ylabel("Held-out MSE")
    axes[1, 1].legend(frameon=False)
    style_axis(axes[1, 1])

    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.2, w_pad=2.0)
    png_path = output_dir / "kalbi_v22_loss_plot.png"
    pdf_path = output_dir / "kalbi_v22_loss_plot.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "step_first": int(steps[0]),
        "step_last": int(steps[-1]),
        "train_updates": int(len(rows)),
        "selector_updates": int(np.sum(selector_mask)),
        "beta_loss": endpoint_summary(beta_loss),
        "beta_mse": endpoint_summary(arrays["beta_mse"]),
        "beta_objective": endpoint_summary(arrays["beta_objective"]),
        "selector_objective": endpoint_summary(selector_objective),
        "selector_heldout_mse": endpoint_summary(heldout_mse),
        "png": str(png_path),
        "pdf": str(pdf_path),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-step", type=int, default=26000)
    args = parser.parse_args()

    rows = read_events(args.events, args.start_step)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "kalbi_v22_loss_series.csv")
    summary = plot(rows, args.output_dir)
    with (args.output_dir / "kalbi_v22_loss_summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
