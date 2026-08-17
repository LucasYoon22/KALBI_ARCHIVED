#!/usr/bin/env python3
"""Plot Top-32 correlation-selected OLS R-squared distributions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/Users/minsuk.shin/.codex/tmp/kalbi_v13_data_quality_20260727/results")
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
GROUPS = ["linear_p", "linear_n", "tabicl"]
LABELS = {
    "linear_p": "Linear, P-referenced",
    "linear_n": "Linear, N-referenced",
    "tabicl": "TabICLv2 outcome",
}
COLORS = {
    "linear_p": "#D55E00",
    "linear_n": "#009E73",
    "tabicl": "#0072B2",
}


def main() -> None:
    metrics = pd.read_csv(ROOT / "top32_corr_ols_task_metrics.csv")
    manifest = pd.read_csv(ROOT / "frozen_audit_manifest.csv")[
        ["task_hash", "linear_target_snr", "linear_snr_regime"]
    ]
    data = metrics.merge(manifest, on="task_hash", how="left", validate="one_to_one")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.7), constrained_layout=True)
    for ax in axes.reshape(-1):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)

    values = [
        data.loc[data.audit_stratum == group, "full_ols_r2"].to_numpy()
        for group in GROUPS
    ]
    violin = axes[0, 0].violinplot(
        values, positions=np.arange(3), showmeans=False, showmedians=False, showextrema=False
    )
    for body, group in zip(violin["bodies"], GROUPS):
        body.set_facecolor(COLORS[group])
        body.set_edgecolor("black")
        body.set_alpha(0.68)
    box = axes[0, 0].boxplot(
        values,
        positions=np.arange(3),
        widths=0.18,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(0.85)
    axes[0, 0].set_xticks(
        np.arange(3), [LABELS[group] for group in GROUPS], rotation=12
    )
    axes[0, 0].set(
        ylabel="In-sample R-squared",
        ylim=(-0.02, 1.02),
        title="Top-32 correlation-selected OLS",
    )

    for group in GROUPS:
        ordered = np.sort(
            data.loc[data.audit_stratum == group, "full_ols_r2"].to_numpy()
        )
        probability = np.arange(1, ordered.size + 1) / ordered.size
        axes[0, 1].step(
            ordered,
            probability,
            where="post",
            linewidth=2,
            color=COLORS[group],
            label=LABELS[group],
        )
    axes[0, 1].set(
        xlabel="In-sample R-squared",
        ylabel="Empirical CDF",
        xlim=(0, 1),
        ylim=(0, 1.01),
        title="Full distribution",
    )
    axes[0, 1].legend(frameon=False, fontsize=8)

    for group in ("linear_p", "linear_n"):
        subset = data[data.audit_stratum == group]
        axes[1, 0].scatter(
            subset.linear_target_snr,
            subset.full_ols_r2,
            s=18,
            alpha=0.48,
            color=COLORS[group],
            label=LABELS[group],
        )
    axes[1, 0].set(
        xscale="log",
        xlabel="Target SNR",
        ylabel="In-sample R-squared",
        ylim=(-0.02, 1.02),
        title="Linear tasks by target SNR",
    )
    axes[1, 0].legend(frameon=False, fontsize=8)

    heldout = [
        data.loc[data.audit_stratum == group, "split_test_ols_r2"].to_numpy()
        for group in GROUPS
    ]
    clipped = [np.clip(value, -1.0, 1.0) for value in heldout]
    bp = axes[1, 1].boxplot(
        clipped,
        positions=np.arange(3),
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, group in zip(bp["boxes"], GROUPS):
        patch.set_facecolor(COLORS[group])
        patch.set_alpha(0.68)
    axes[1, 1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_xticks(
        np.arange(3), [LABELS[group] for group in GROUPS], rotation=12
    )
    axes[1, 1].set(
        ylabel="Held-out R-squared (display clipped at -1)",
        ylim=(-1.05, 1.05),
        title="Train-half selection and OLS diagnostic",
    )
    for position, raw in enumerate(heldout):
        below = 100.0 * np.mean(raw < -1.0)
        axes[1, 1].text(
            position,
            -0.94,
            f"{below:.1f}% < -1",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )

    for ax in axes.reshape(-1):
        ax.title.set_fontsize(12)
    fig.savefig(OUT / "top32_corr_ols_r2_distribution.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "top32_corr_ols_r2_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
