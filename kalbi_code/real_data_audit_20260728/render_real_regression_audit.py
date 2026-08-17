#!/usr/bin/env python3
"""Render reviewed real-regression candidate summaries and confirmation manifests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(
    "/users/sync_data/real_regression_uci_hf_highdim_signal_screen_20260728"
)
FIGURES = ROOT / "figures"
MANIFESTS = ROOT / "manifests"
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREY = "#999999"
INK = "#252525"


def load_metrics() -> list[dict[str, Any]]:
    results = []
    for path in sorted((ROOT / "metrics").glob("*.json")):
        result = json.loads(path.read_text())
        result.setdefault("evidence_tier", "standard")
        results.append(result)
    return results


def short_label(task_id: str) -> str:
    replacements = {
        "hf_chlange_ramanspectrabioprocessanalystes": "",
        "hf_ramanbench_": "",
        "hf_greenhyperspectra__": "GreenHyper · ",
        "hf_moleculenet_": "",
        "_morgan2048": "",
        "__": " · ",
        "_": " ",
    }
    label = task_id
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label.strip().title()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "task_id",
        "source",
        "target",
        "evidence_tier",
        "N",
        "P_raw",
        "P_over_N",
        "best_model",
        "best_screening_r2",
        "best_repeated_context_median_r2",
        "test_r2",
        "test_mse",
        "zero_predictor_test_mse",
        "snr_proxy_from_test_r2",
        "sampled_max_abs_feature_corr",
        "max_abs_marginal_corr_y",
        "reality_status",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_manifests(results: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [item for item in results if item.get("status") == "accepted"]
    highdim = [item for item in accepted if item["P_raw"] > item["N"]]
    confirmed = [item for item in highdim if item.get("test_r2", -np.inf) >= 0.20]
    for item in highdim:
        item["confirmation_status"] = (
            "locked_test_confirmed"
            if item.get("test_r2", -np.inf) >= 0.20
            else "screen_pass_but_locked_test_not_confirmed"
        )
    write_csv(MANIFESTS / "screen_accepted_highdim_datasets.csv", highdim)
    write_csv(MANIFESTS / "locked_test_confirmed_highdim_datasets.csv", confirmed)
    (MANIFESTS / "screen_accepted_highdim_datasets.json").write_text(
        json.dumps(highdim, indent=2, sort_keys=True) + "\n"
    )
    (MANIFESTS / "locked_test_confirmed_highdim_datasets.json").write_text(
        json.dumps(confirmed, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "prepared_tasks": len(results),
        "screen_accepted": len(accepted),
        "screen_rejected": len(results) - len(accepted),
        "screen_accepted_P_ge_1000": sum(item["P_raw"] >= 1000 for item in accepted),
        "screen_accepted_P_gt_N": len(highdim),
        "locked_test_confirmed_P_gt_N": len(confirmed),
        "accepted_standard": sum(
            item["evidence_tier"] == "standard" for item in accepted
        ),
        "accepted_exploratory_small_n": sum(
            item["evidence_tier"] == "exploratory_small_n_highdim"
            for item in accepted
        ),
        "locked_test_confirmation_rule": "test R2 >= 0.20, diagnostic only",
        "selection_rule": (
            "screening R2 >= 0.20 and repeated context median R2 >= 0.15; "
            "locked test not used for selection"
        ),
    }
    (MANIFESTS / "real_regression_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def render_dimension_and_replication(results: list[dict[str, Any]]) -> None:
    accepted = [item for item in results if item.get("status") == "accepted"]
    highdim = [item for item in accepted if item["P_raw"] > item["N"]]
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.7))

    axis = axes[0]
    for item in results:
        accepted_item = item.get("status") == "accepted"
        exploratory = item.get("evidence_tier") == "exploratory_small_n_highdim"
        if accepted_item:
            color = ORANGE if exploratory else BLUE
            marker = "^" if exploratory else "o"
            size = 36
            alpha = 0.82
        else:
            color = GREY
            marker = "x"
            size = 24
            alpha = 0.55
        axis.scatter(
            item["P_raw"],
            item["N"],
            color=color,
            marker=marker,
            s=size,
            alpha=alpha,
            linewidths=0.8,
        )
    lower = 50
    upper = max(max(item["P_raw"], item["N"]) for item in results) * 1.15
    axis.plot([lower, upper], [lower, upper], color=INK, linestyle=":", linewidth=1)
    axis.text(1.3e4, 8.0e3, "$P>N$", color=INK, fontsize=10)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_xlabel("Feature dimension $P$")
    axis.set_ylabel("Sample size $N$")
    axis.set_title("Candidate dimensions and signal-screen status")
    axis.grid(alpha=0.15)

    axis = axes[1]
    for item in highdim:
        exploratory = item["evidence_tier"] == "exploratory_small_n_highdim"
        color = ORANGE if exploratory else BLUE
        marker = "^" if exploratory else "o"
        test_r2 = float(item["test_r2"])
        clipped_test = max(test_r2, -1.0)
        axis.scatter(
            item["best_screening_r2"],
            clipped_test,
            color=color,
            marker=marker,
            s=45,
            edgecolor=INK if test_r2 >= 0.20 else "none",
            linewidth=0.7,
            alpha=0.88,
        )
        if test_r2 < -1.0:
            axis.scatter(
                item["best_screening_r2"],
                -1.0,
                facecolors="none",
                edgecolors=color,
                marker="v",
                s=48,
                linewidth=1.0,
            )
    axis.plot([-1, 1], [-1, 1], color=INK, linestyle=":", linewidth=1)
    axis.axhline(0.20, color=GREY, linestyle="--", linewidth=1)
    axis.axvline(0.20, color=GREY, linestyle="--", linewidth=1)
    axis.set_xlim(0.15, 1.01)
    axis.set_ylim(-1.05, 1.02)
    axis.set_xlabel("Screening-holdout $R^2$")
    axis.set_ylabel("Locked-test $R^2$ (clipped below $-1$)")
    axis.set_title("Replication of accepted high-dimensional tasks")
    axis.grid(alpha=0.15)

    legend = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=BLUE,
            markeredgecolor=BLUE, label="Standard evidence", markersize=7
        ),
        Line2D(
            [0], [0], marker="^", color="none", markerfacecolor=ORANGE,
            markeredgecolor=ORANGE, label="Exploratory small-$N$", markersize=7
        ),
        Line2D(
            [0], [0], marker="x", color=GREY, label="Screen rejected",
            markersize=7, linestyle="none"
        ),
    ]
    figure.legend(handles=legend, loc="lower center", ncol=3, frameon=False)
    figure.subplots_adjust(bottom=0.18, wspace=0.28)
    figure.savefig(FIGURES / "real_regression_dimension_and_replication.png", dpi=240)
    figure.savefig(FIGURES / "real_regression_dimension_and_replication.pdf")
    plt.close(figure)


def render_highdim_ranking(results: list[dict[str, Any]]) -> None:
    confirmed = [
        item
        for item in results
        if item.get("status") == "accepted"
        and item["P_raw"] > item["N"]
        and item.get("test_r2", -np.inf) >= 0.20
    ]
    selected = sorted(confirmed, key=lambda item: item["test_r2"], reverse=True)[:25]
    selected.reverse()
    labels = [short_label(item["task_id"]) for item in selected]
    values = [item["test_r2"] for item in selected]
    colors = [
        ORANGE
        if item["evidence_tier"] == "exploratory_small_n_highdim"
        else BLUE
        for item in selected
    ]
    figure, axis = plt.subplots(figsize=(10.5, 9.0))
    positions = np.arange(len(selected))
    bars = axis.barh(positions, values, color=colors, edgecolor="white", linewidth=0.5)
    axis.set_yticks(positions, labels, fontsize=8)
    axis.set_xlim(0, 1.25)
    axis.set_xticks(np.linspace(0, 1.0, 6))
    axis.axvline(1.0, color=INK, linestyle=":", linewidth=1)
    axis.set_xlabel("Locked-test $R^2$")
    axis.set_title("High-dimensional regression tasks with replicated linear signal")
    axis.grid(axis="x", alpha=0.18)
    for bar, item in zip(bars, selected):
        axis.text(
            1.02,
            bar.get_y() + bar.get_height() / 2,
            f"{item['test_r2']:.2f}  N={item['N']}, P={item['P_raw']}",
            va="center",
            fontsize=7.5,
            color=INK,
        )
    figure.legend(
        handles=[
            Line2D([0], [0], color=BLUE, linewidth=7, label="Standard evidence"),
            Line2D([0], [0], color=ORANGE, linewidth=7, label="Exploratory small-$N$"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.62, 0.01),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(FIGURES / "confirmed_highdim_regression_ranking.png", dpi=240)
    figure.savefig(FIGURES / "confirmed_highdim_regression_ranking.pdf")
    plt.close(figure)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    MANIFESTS.mkdir(exist_ok=True)
    results = load_metrics()
    summary = write_manifests(results)
    render_dimension_and_replication(results)
    render_highdim_ranking(results)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
