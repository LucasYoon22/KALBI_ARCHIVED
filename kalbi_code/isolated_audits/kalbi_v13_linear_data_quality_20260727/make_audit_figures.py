#!/usr/bin/env python3
"""Create publication-quality diagnostic figures for the V13 data audit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/Users/minsuk.shin/.codex/tmp/kalbi_v13_data_quality_20260727/results")
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
COLORS = {
    "tabicl": "#0072B2",
    "linear_p": "#D55E00",
    "linear_n": "#009E73",
    "core": "#CC79A7",
    "bnn": "#56B4E9",
}
LABELS = {
    "tabicl": "TabICLv2 outcome",
    "linear_p": "Linear, P-referenced",
    "linear_n": "Linear, N-referenced",
}


def save(fig: plt.Figure, name: str) -> None:
    for ax in fig.axes:
        ax.title.set_fontsize(12)
    fig.savefig(OUT / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def style_axes(axes: np.ndarray) -> None:
    for ax in axes.reshape(-1):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)


def grouped_box(ax: plt.Axes, data: pd.DataFrame, column: str) -> None:
    groups = ["tabicl", "linear_p", "linear_n"]
    values = [data.loc[data.audit_stratum == group, column].dropna() for group in groups]
    bp = ax.boxplot(values, labels=[LABELS[g] for g in groups], showfliers=False, patch_artist=True)
    for patch, group in zip(bp["boxes"], groups):
        patch.set_facecolor(COLORS[group])
        patch.set_alpha(0.72)
    ax.tick_params(axis="x", rotation=18)


def main() -> None:
    metadata = pd.read_csv(ROOT / "metadata_census.csv")
    tasks = pd.read_csv(ROOT / "task_metrics.csv")
    scale_tasks = pd.read_csv(ROOT / "feature_scaling_task_metrics.csv")
    scale_summary = json.loads((ROOT / "feature_scaling_summary.json").read_text())
    unit = json.loads((ROOT / "generator_unit_test_results.json").read_text())
    replay = json.loads((ROOT / "replay/replay_verification.json").read_text())

    fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.5), constrained_layout=True)
    style_axes(axes)
    sample = metadata.sample(min(4000, len(metadata)), random_state=20260727)
    for group in ("tabicl", "linear_p", "linear_n"):
        subset = sample[sample.audit_stratum == group]
        axes[0, 0].scatter(
            subset.n, subset.p, s=8, alpha=0.38, color=COLORS[group], label=LABELS[group]
        )
    axes[0, 0].set(xscale="log", yscale="log", xlabel="N", ylabel="P", title="N-P distribution")
    axes[0, 0].legend(frameon=False, fontsize=8)

    linear = metadata[metadata.outcome_source == "linear"].copy()
    branch_order = ["p_based", "n_based"]
    branch_colors = [COLORS["linear_p"], COLORS["linear_n"]]
    for branch, color in zip(branch_order, branch_colors):
        subset = linear[linear.linear_support_reference == branch]
        axes[0, 1].hist(
            subset.linear_support_fraction_of_n,
            bins=np.linspace(0, 1 / 3, 42),
            histtype="step",
            linewidth=2,
            density=True,
            color=color,
            label=branch.replace("_", "-"),
        )
    axes[0, 1].set(xlabel="k / N", ylabel="Density", title="Support fraction of N")
    axes[0, 1].legend(frameon=False, fontsize=8)

    for branch, color in zip(branch_order, branch_colors):
        subset = linear[linear.linear_support_reference == branch]
        values = np.clip(subset.linear_support_fraction_of_p, 1e-6, 1)
        axes[0, 2].hist(
            values,
            bins=np.logspace(-6, 0, 42),
            histtype="step",
            linewidth=2,
            density=True,
            color=color,
            label=branch.replace("_", "-"),
        )
    axes[0, 2].set(xscale="log", xlabel="k / P", ylabel="Density", title="Support fraction of P")
    axes[0, 2].legend(frameon=False, fontsize=8)

    regimes = ["ultra_sparse", "sparse", "moderate", "large", "very_dense"]
    declared = np.array([0.15, 0.20, 0.25, 0.25, 0.15])
    x = np.arange(len(regimes))
    width = 0.26
    axes[1, 0].bar(x - width, declared, width, color="#777777", label="Declared")
    for offset, branch, color in zip((0, width), branch_order, branch_colors):
        fractions = (
            linear[linear.linear_support_reference == branch]
            .linear_support_regime.value_counts(normalize=True)
            .reindex(regimes, fill_value=0)
        )
        axes[1, 0].bar(x + offset, fractions, width, color=color, label=branch.replace("_", "-"))
    axes[1, 0].set_xticks(x, [r.replace("_", "\n") for r in regimes])
    axes[1, 0].set(ylabel="Task fraction", title="Support-regime frequencies")
    axes[1, 0].legend(frameon=False, fontsize=8)

    plot_linear = linear.sample(min(3500, len(linear)), random_state=17)
    for branch, color in zip(branch_order, branch_colors):
        subset = plot_linear[plot_linear.linear_support_reference == branch]
        axes[1, 1].scatter(
            subset.linear_target_snr,
            subset.linear_realized_snr,
            s=8,
            alpha=0.35,
            color=color,
            label=branch.replace("_", "-"),
        )
    limits = (1e-2, 1e3)
    axes[1, 1].plot(limits, limits, color="black", linewidth=1, linestyle="--")
    axes[1, 1].set(
        xscale="log",
        yscale="log",
        xlim=limits,
        ylim=limits,
        xlabel="Target SNR",
        ylabel="Recorded realized SNR",
        title="SNR calibration metadata",
    )
    axes[1, 1].legend(frameon=False, fontsize=8)

    for group in ("linear_p", "linear_n"):
        subset = metadata[metadata.audit_stratum == group]
        counts = subset.linear_snr_regime.value_counts(normalize=True).reindex(
            ["very_low", "low", "medium", "high", "near_noiseless"], fill_value=0
        )
        axes[1, 2].plot(
            np.arange(5),
            counts,
            marker="o",
            linewidth=2,
            color=COLORS[group],
            label=LABELS[group],
        )
    axes[1, 2].set_xticks(
        np.arange(5), ["very low", "low", "medium", "high", "near\nnoiseless"]
    )
    axes[1, 2].set(ylabel="Task fraction", title="SNR regime coverage")
    axes[1, 2].legend(frameon=False, fontsize=8)
    save(fig, "mixture_support_snr")

    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.7), constrained_layout=True)
    style_axes(axes)
    grouped_box(axes[0, 0], tasks, "y_iqr")
    axes[0, 0].axhline(0.2, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set(ylabel="Standardized IQR", title="Response IQR")

    grouped_box(axes[0, 1], tasks, "y_excess_kurtosis")
    axes[0, 1].set(yscale="symlog", ylabel="Excess kurtosis", title="Response tail shape")

    grouped_box(axes[0, 2], tasks, "top32_marginal_ridge_r2_full_rows")
    axes[0, 2].set(ylabel="R-squared", ylim=(-0.02, 1.03), title="Top-32 marginal ridge signal")

    grouped_box(axes[1, 0], tasks, "recomputed_sample_corr_max_upto2048rows")
    axes[1, 0].axhline(0.95, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set(ylabel="Sampled max |correlation|", title="Feature correlation audit")

    thresholds = ["count_std_lt_0p9", "count_std_lt_0p5", "count_std_lt_0p2"]
    threshold_labels = ["std < 0.9", "std < 0.5", "std < 0.2"]
    for pos, source in enumerate(("core", "bnn")):
        fractions = [
            scale_summary["sources"][source][threshold]["fraction"]
            for threshold in thresholds
        ]
        axes[1, 1].bar(
            np.arange(3) + (pos - 0.5) * 0.34,
            fractions,
            0.34,
            color=COLORS[source],
            label=source.upper(),
        )
    axes[1, 1].set_xticks(np.arange(3), threshold_labels)
    axes[1, 1].set(
        yscale="log",
        ylabel="Fraction of features",
        title="Post-emission feature scale drift",
    )
    axes[1, 1].legend(frameon=False, fontsize=8)

    noise = unit["noise_summary"]
    families = list(noise)
    tails = [noise[name]["tail3_mean"] for name in families]
    axes[1, 2].bar(np.arange(len(families)), tails, color="#E69F00")
    axes[1, 2].axhline(0.0027, color="black", linestyle="--", linewidth=1, label="Gaussian reference")
    axes[1, 2].set_xticks(
        np.arange(len(families)),
        [name.replace("_", "\n") for name in families],
        rotation=25,
        fontsize=8,
    )
    axes[1, 2].set(ylabel="P(|error| > 3 SD)", title="Noise-family tail rate")
    axes[1, 2].legend(frameon=False, fontsize=8)
    save(fig, "response_feature_quality")

    fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.5), constrained_layout=True)
    style_axes(axes)
    epoch_mix = (
        metadata.groupby(["catalog_epoch", "audit_stratum"]).size()
        / metadata.groupby("catalog_epoch").size()
    ).unstack(fill_value=0)
    for group in ("tabicl", "linear_p", "linear_n"):
        axes[0, 0].plot(
            epoch_mix.index,
            epoch_mix[group],
            marker="o",
            linewidth=2,
            color=COLORS[group],
            label=LABELS[group],
        )
    axes[0, 0].set(ylabel="Task fraction", xlabel="Catalog epoch", title="Mixture stability")
    axes[0, 0].legend(frameon=False, fontsize=8)

    retry = metadata.generator_rejected_attempt_count.clip(upper=10)
    axes[0, 1].hist(retry, bins=np.arange(-0.5, 11.5), color="#0072B2", alpha=0.8)
    axes[0, 1].set(
        xlabel="Rejected attempts (10 = 10+)",
        ylabel="Tasks",
        yscale="log",
        title="Generator retry burden",
    )

    for source, color in (("core", COLORS["core"]), ("bnn", COLORS["bnn"])):
        subset = scale_tasks[scale_tasks.source == source]
        axes[0, 2].hist(
            subset.std_min,
            bins=np.linspace(0.15, 1.01, 60),
            histtype="step",
            linewidth=2,
            color=color,
            label=source.upper(),
        )
    axes[0, 2].set(
        xlabel="Minimum feature std per task",
        ylabel="Tasks",
        yscale="log",
        title="Scale drift is SCM-core-specific",
    )
    axes[0, 2].legend(frameon=False, fontsize=8)

    clip_rate = (
        metadata.assign(clipped=metadata.target_pre_rescale_clip_count > 0)
        .groupby("audit_stratum")
        .clipped.mean()
        .reindex(["tabicl", "linear_p", "linear_n"])
    )
    axes[1, 0].bar(
        np.arange(3),
        clip_rate,
        color=[COLORS[group] for group in ("tabicl", "linear_p", "linear_n")],
    )
    axes[1, 0].set_xticks(np.arange(3), [LABELS[g] for g in ("tabicl", "linear_p", "linear_n")], rotation=18)
    axes[1, 0].set(ylabel="Fraction clipped", title="Response clipping incidence")

    monte = unit["support_summary"]
    p_regimes = monte["p_based"]["regime_fractions"]
    n_regimes = monte["n_based"]["regime_fractions"]
    axes[1, 1].bar(
        np.arange(5) - 0.18,
        [p_regimes[r] for r in ["ultra_sparse", "sparse", "moderate", "large", "very_dense"]],
        0.36,
        color=COLORS["linear_p"],
        label="P-referenced",
    )
    axes[1, 1].bar(
        np.arange(5) + 0.18,
        [n_regimes[r] for r in ["ultra_sparse", "sparse", "moderate", "large", "very_dense"]],
        0.36,
        color=COLORS["linear_n"],
        label="N-referenced",
    )
    axes[1, 1].set_xticks(np.arange(5), ["ultra", "sparse", "moderate", "large", "dense"])
    axes[1, 1].set(ylabel="Fraction (200k draws)", title="Conditional support sampler")
    axes[1, 1].legend(frameon=False, fontsize=8)

    checks = [
        ("Full replay\nwithout hash seed", sum(x["exact_match"] for x in replay["exact_replay_comparison"]), 6),
        ("Full replay\nPYTHONHASHSEED=0", 1, 1),
        ("Linear response\nunit replay", int(unit["same_seed_exact_replay"]), 1),
    ]
    axes[1, 2].bar(
        np.arange(3),
        [passed / total for _, passed, total in checks],
        color=["#D55E00", "#009E73", "#0072B2"],
    )
    axes[1, 2].set_xticks(np.arange(3), [name for name, _, _ in checks])
    axes[1, 2].set(ylim=(0, 1.05), ylabel="Exact-match fraction", title="Seed reproducibility")
    for index, (_, passed, total) in enumerate(checks):
        axes[1, 2].text(index, passed / total + 0.04, f"{passed}/{total}", ha="center", fontsize=9)
    save(fig, "drift_filter_reproducibility")


if __name__ == "__main__":
    main()
