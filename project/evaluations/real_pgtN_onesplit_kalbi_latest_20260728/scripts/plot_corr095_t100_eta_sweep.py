#!/usr/bin/env python3
"""Plot and summarize the correlation-filtered T=100 eta sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FACTORS = (1, 2, 5, 10)
COLORS = {
    1: "#0072B2",
    2: "#009E73",
    5: "#E69F00",
    10: "#D55E00",
}
LASSO_COLOR = "#6A3D9A"
GRAY = "#555555"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_metrics(root: Path) -> list[dict[str, Any]]:
    values = [load_json(path) for path in (root / "metrics").glob("*.json")]
    values.sort(key=lambda item: item["registry_index"])
    if len(values) != 41 or any(item["status"] != "PASS" for item in values):
        raise RuntimeError(f"expected 41 PASS metrics, got {len(values)}")
    return values


def factor_key(factor: int) -> str:
    return f"eta{factor}x"


def aggregate(metrics: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        row: dict[str, Any] = {
            "registry_index": metric["registry_index"],
            "task_id": metric["task_id"],
            "N": metric["N"],
            "P": metric["P_after_correlation_filter"],
        }
        for factor in FACTORS:
            value = metric["factors"][factor_key(factor)]
            row[f"selected_eta{factor}x"] = value[
                "selected_normalized_test_mse"
            ]
            row[f"oracle_eta{factor}x"] = value[
                "path_oracle_normalized_test_mse"
            ]
            row[f"accepted_first_eta_ratio_eta{factor}x"] = value[
                "accepted_first_sweep_eta_ratio_median"
            ]
            row[f"backtracks_eta{factor}x"] = value[
                "mean_backtracks_per_query_sweep"
            ]
        rows.append(row)
    selected = np.asarray(
        [[row[f"selected_eta{factor}x"] for factor in FACTORS] for row in rows]
    )
    oracle = np.asarray(
        [[row[f"oracle_eta{factor}x"] for factor in FACTORS] for row in rows]
    )
    accepted = np.asarray(
        [
            [
                row[f"accepted_first_eta_ratio_eta{factor}x"]
                for factor in FACTORS
            ]
            for row in rows
        ]
    )
    backtracks = np.asarray(
        [[row[f"backtracks_eta{factor}x"] for factor in FACTORS] for row in rows]
    )
    selected_winner = np.argmin(selected, axis=1)
    oracle_winner = np.argmin(oracle, axis=1)
    summary = {
        "task_count": len(rows),
        "factors": FACTORS,
        "selected_median": {
            f"{factor}x": float(np.median(selected[:, index]))
            for index, factor in enumerate(FACTORS)
        },
        "selected_geometric_mean": {
            f"{factor}x": float(
                np.exp(np.mean(np.log(np.maximum(selected[:, index], 1e-30))))
            )
            for index, factor in enumerate(FACTORS)
        },
        "selected_improved_vs_1x_count": {
            f"{factor}x": int(np.count_nonzero(selected[:, index] < selected[:, 0]))
            for index, factor in enumerate(FACTORS[1:], start=1)
        },
        "selected_median_ratio_vs_1x": {
            f"{factor}x": float(np.median(selected[:, index] / selected[:, 0]))
            for index, factor in enumerate(FACTORS[1:], start=1)
        },
        "selected_winner_count": {
            f"{factor}x": int(np.count_nonzero(selected_winner == index))
            for index, factor in enumerate(FACTORS)
        },
        "oracle_median": {
            f"{factor}x": float(np.median(oracle[:, index]))
            for index, factor in enumerate(FACTORS)
        },
        "oracle_winner_count": {
            f"{factor}x": int(np.count_nonzero(oracle_winner == index))
            for index, factor in enumerate(FACTORS)
        },
        "accepted_first_eta_ratio_median": {
            f"{factor}x": float(np.median(accepted[:, index]))
            for index, factor in enumerate(FACTORS)
        },
        "mean_backtracks_median": {
            f"{factor}x": float(np.median(backtracks[:, index]))
            for index, factor in enumerate(FACTORS)
        },
        "maximum_majorization_ratio": float(
            max(
                value["maximum_majorization_ratio"]
                for metric in metrics
                for value in metric["factors"].values()
            )
        ),
        "maximum_objective_increase": float(
            max(
                value["maximum_objective_increase"]
                for metric in metrics
                for value in metric["factors"].values()
            )
        ),
        "monotonicity_violation_count": int(
            sum(
                value["monotonicity_violation_count"]
                for metric in metrics
                for value in metric["factors"].values()
            )
        ),
        "eta_1x_reproduction_max_abs_error": float(
            max(item["eta_1x_reproduction_max_abs_error"] for item in metrics)
        ),
    }
    return summary, rows


def render_aggregate(
    metrics: list[dict[str, Any]], root: Path, figure_root: Path
) -> None:
    selected = np.asarray(
        [
            [
                metric["factors"][factor_key(factor)][
                    "selected_normalized_test_mse"
                ]
                for factor in FACTORS
            ]
            for metric in metrics
        ]
    )
    oracle = np.asarray(
        [
            [
                metric["factors"][factor_key(factor)][
                    "path_oracle_normalized_test_mse"
                ]
                for factor in FACTORS
            ]
            for metric in metrics
        ]
    )
    accepted = np.asarray(
        [
            [
                metric["factors"][factor_key(factor)][
                    "accepted_first_sweep_eta_ratio_median"
                ]
                for factor in FACTORS
            ]
            for metric in metrics
        ]
    )
    backtracks = np.asarray(
        [
            [
                metric["factors"][factor_key(factor)][
                    "mean_backtracks_per_query_sweep"
                ]
                for factor in FACTORS
            ]
            for metric in metrics
        ]
    )
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.2), constrained_layout=True)
    for task in range(selected.shape[0]):
        axes[0].plot(
            FACTORS, selected[task], color="#999999", alpha=0.25, lw=0.8
        )
        axes[1].plot(
            FACTORS, oracle[task], color="#999999", alpha=0.25, lw=0.8
        )
    for axis, data, label in (
        (axes[0], selected, "KALBI-selected normalized MSE"),
        (axes[1], oracle, "Path-oracle normalized MSE"),
    ):
        axis.plot(
            FACTORS,
            np.median(data, axis=0),
            color="#111111",
            marker="o",
            lw=2.5,
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks(FACTORS, [f"{factor}x" for factor in FACTORS])
        axis.set_xlabel("Requested initial eta multiplier")
        axis.set_ylabel(label)

    for index, factor in enumerate(FACTORS):
        axes[2].scatter(
            np.full(accepted.shape[0], factor),
            accepted[:, index],
            color=COLORS[factor],
            s=18,
            alpha=0.65,
        )
    axes[2].plot(FACTORS, FACTORS, color=GRAY, ls=":", lw=1.2)
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xticks(FACTORS, [f"{factor}x" for factor in FACTORS])
    axes[2].set_xlabel("Requested initial eta multiplier")
    axes[2].set_ylabel("Accepted first-sweep eta ratio")

    axes[3].boxplot(
        [backtracks[:, index] for index in range(len(FACTORS))],
        tick_labels=[f"{factor}x" for factor in FACTORS],
        showfliers=False,
    )
    axes[3].set_xlabel("Requested initial eta multiplier")
    axes[3].set_ylabel("Mean backtracks per query/sweep")
    for axis in axes:
        axis.grid(False)
    fig.savefig(
        figure_root / "corr095_t100_eta_sweep_aggregate.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        figure_root / "corr095_t100_eta_sweep_aggregate.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def render_paths(
    metrics: list[dict[str, Any]], root: Path, figure_root: Path
) -> None:
    selected_indices = [1, 9, 13, 18, 31, 41]
    by_index = {item["registry_index"]: item for item in metrics}
    fig, axes = plt.subplots(
        len(selected_indices),
        3,
        figsize=(13.5, 18.5),
        constrained_layout=True,
    )
    for row_index, registry_index in enumerate(selected_indices):
        metric = by_index[registry_index]
        with np.load(root / "paths" / f"{metric['task_id']}.npz") as archive:
            data = {key: np.asarray(archive[key]) for key in archive.files}
        x = np.log10(data["raw_lambdas"])
        top = data["top_features"].astype(np.int64)[:6]
        colors = plt.cm.tab10(np.linspace(0, 1, max(1, top.size)))
        for color, feature in zip(colors, top, strict=True):
            axes[row_index, 0].plot(
                x,
                data["eta1x_beta"][:-1, feature],
                color=color,
                lw=1.6,
            )
            axes[row_index, 0].plot(
                x,
                data["eta10x_beta"][:-1, feature],
                color=color,
                lw=1.2,
                ls="--",
            )
        axes[row_index, 0].axhline(0, color="#BBBBBB", lw=0.7)
        axes[row_index, 0].text(
            0.02,
            0.97,
            f"N={metric['N']}, P={metric['P_after_correlation_filter']}",
            transform=axes[row_index, 0].transAxes,
            va="top",
            fontsize=8.5,
        )
        axes[row_index, 0].set_ylabel(
            f"Dataset {registry_index}\ncoefficient"
        )

        for factor in FACTORS:
            axes[row_index, 1].plot(
                x,
                data[f"eta{factor}x_normalized_test_mse"][:-1],
                color=COLORS[factor],
                lw=1.8,
                label=f"eta {factor}x",
            )
        axes[row_index, 1].plot(
            x,
            data["lasso_normalized_test_mse"],
            color=LASSO_COLOR,
            lw=1.3,
            ls=":",
            label="LASSO path",
        )
        axes[row_index, 1].axhline(1.0, color=GRAY, lw=0.8, ls=":")
        axes[row_index, 1].scatter(
            np.log10(float(data["cv_lasso_raw_lambda"])),
            float(data["cv_lasso_normalized_test_mse"]),
            color=LASSO_COLOR,
            marker="s",
            s=30,
            zorder=5,
            label="CV-LASSO",
        )
        axes[row_index, 1].set_yscale("log")

        for factor in FACTORS:
            nnz = np.count_nonzero(
                np.abs(data[f"eta{factor}x_beta"][:-1]) > 1e-8, axis=1
            )
            axes[row_index, 2].plot(
                x,
                nnz,
                color=COLORS[factor],
                lw=1.8,
                label=f"eta {factor}x",
            )
        axes[row_index, 2].set_yscale("symlog", linthresh=1)
        if row_index == 0:
            axes[row_index, 1].legend(
                fontsize=7.5, frameon=False, ncol=2
            )
    for axis, title in zip(
        axes[0],
        [
            "Coefficient path: eta 1x (solid), eta 10x (dashed)",
            "Held-out prediction error",
            "Sparsity path",
        ],
        strict=True,
    ):
        axis.set_title(title, fontsize=11)
    for axis in axes[-1]:
        axis.set_xlabel(r"$\log_{10}(\lambda_{\mathrm{raw}})$")
    for axis in axes.flat:
        axis.grid(False)
        axis.tick_params(labelsize=8)
    fig.savefig(
        figure_root / "corr095_t100_eta_sweep_paths.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        figure_root / "corr095_t100_eta_sweep_paths.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    figure_root = args.root / "figures"
    report_root = args.root / "reports"
    figure_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(args.root)
    summary, rows = aggregate(metrics)
    render_aggregate(metrics, args.root, figure_root)
    render_paths(metrics, args.root, figure_root)
    write_csv(rows, report_root / "eta_sweep_task_summary.csv")
    (report_root / "eta_sweep_aggregate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
