#!/usr/bin/env python3
"""Compare V17 selected lambda/penalty with fixed holdout query oracles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .canonical_penalty_cache import CanonicalPenaltyCache
from .eval_kalbi_v2_common import (
    encode_dataset,
    file_sha256,
    load_components,
    load_task,
    predict_from_encoded,
    select_balanced_entries,
    select_smallest_p_regime_entries,
    split_standardize,
    tensors_from_arrays,
    validation_mse,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--penalty-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--lambda-grid-size", type=int, default=41)
    parser.add_argument("--penalty-candidates", type=int, default=32)
    parser.add_argument("--candidate-lambda-grid-size", type=int, default=25)
    parser.add_argument("--split-seed", type=int, default=2026072117)
    parser.add_argument(
        "--selection",
        choices=("balanced-shard", "smallest-p-by-prior-regime"),
        default="balanced-shard",
    )
    args = parser.parse_args()

    if args.out_dir.exists():
        raise RuntimeError(f"refusing existing output directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint, model, selector = load_components(
        args.checkpoint, args.config, device
    )
    step = int(checkpoint["completed_steps"])
    catalog = checkpoint["catalog_payload"]
    data_root = Path(str(catalog["data_root"])).resolve()
    entries = (
        select_smallest_p_regime_entries(catalog, int(args.count))
        if args.selection == "smallest-p-by-prior-regime"
        else select_balanced_entries(catalog, int(args.count))
    )
    cache = CanonicalPenaltyCache(args.penalty_cache, seed=2026072117)
    cache.load(expected_count=300_000)
    candidate_indices = np.linspace(
        0, cache.pool_size - 1, int(args.penalty_candidates)
    ).round().astype(np.int64)
    candidate_encoded = cache.encoded_indices(candidate_indices, device)
    candidate_ratios = torch.logspace(
        -6.0,
        0.0,
        int(args.candidate_lambda_grid_size),
        device=device,
        dtype=torch.float32,
    )
    selected_ratio_base = torch.logspace(
        -6.0,
        0.0,
        int(args.lambda_grid_size),
        device=device,
        dtype=torch.float32,
    )

    rows: list[dict[str, object]] = []
    penalty_curves: list[dict[str, np.ndarray | float | str]] = []
    for row, entry in enumerate(entries):
        X_raw, y_raw, max_corr = load_task(data_root, entry)
        X_train, y_train, X_test, y_test = split_standardize(
            X_raw, y_raw, int(args.split_seed) + row
        )
        X_device, y_device = tensors_from_arrays(X_train, y_train, device)
        encoded_dataset = encode_dataset(model, X_device, y_device)
        lambda_max = encoded_dataset["lambda_max"].float().reshape(-1)
        with torch.inference_mode():
            selected = selector(
                X_device,
                y_device,
                lambda_max,
                return_representation=True,
            )
        selected_lambda = selected["lambdas"].reshape(-1)
        selected_shape = selected["penalty_shape_features"].reshape(1, -1)
        selected_d0 = selected["penalty_d0"].reshape(-1)
        selected_effective_ratio = float(
            selected["effective_lambda_ratio"].reshape(-1)[0]
        )
        selected_beta = predict_from_encoded(
            model,
            encoded_dataset,
            selected_lambda,
            selected_shape,
            selected_d0,
        )
        selected_mse = float(validation_mse(X_test, y_test, selected_beta)[0])

        ratio_grid = torch.unique(torch.cat([
            selected_ratio_base,
            torch.as_tensor(
                [selected_effective_ratio], device=device, dtype=torch.float32
            ),
        ])).sort().values
        selected_penalty_lambdas = (
            ratio_grid * lambda_max[0] / selected_d0[0].clamp_min(1e-12)
        )
        selected_penalty_betas = predict_from_encoded(
            model,
            encoded_dataset,
            selected_penalty_lambdas,
            selected_shape.expand(ratio_grid.numel(), -1),
            selected_d0.expand(ratio_grid.numel()),
        )
        selected_penalty_mses = validation_mse(
            X_test, y_test, selected_penalty_betas
        )
        lambda_oracle_index = int(np.argmin(selected_penalty_mses))
        lambda_oracle_mse = float(selected_penalty_mses[lambda_oracle_index])
        lambda_oracle_ratio = float(ratio_grid[lambda_oracle_index])

        penalty_count = int(candidate_indices.size)
        ratio_count = int(candidate_ratios.numel())
        shape_grid = candidate_encoded["shape_features"].repeat_interleave(
            ratio_count, dim=0
        )
        d0_grid = candidate_encoded["d0"].repeat_interleave(ratio_count)
        ratio_cross = candidate_ratios.repeat(penalty_count)
        lambda_grid = ratio_cross * lambda_max[0] / d0_grid.clamp_min(1e-12)
        candidate_betas = predict_from_encoded(
            model,
            encoded_dataset,
            lambda_grid,
            shape_grid,
            d0_grid,
            chunk_size=64,
        )
        candidate_mses = validation_mse(X_test, y_test, candidate_betas)
        cache_oracle_flat = int(np.argmin(candidate_mses))
        cache_oracle_penalty_position = cache_oracle_flat // ratio_count
        cache_oracle_ratio_position = cache_oracle_flat % ratio_count
        cache_oracle_mse = float(candidate_mses[cache_oracle_flat])
        cache_oracle_ratio = float(candidate_ratios[cache_oracle_ratio_position])
        cache_oracle_penalty_index = int(
            candidate_indices[cache_oracle_penalty_position]
        )

        joint_oracle_mse = min(lambda_oracle_mse, cache_oracle_mse)
        joint_source = (
            "selected_penalty_lambda_oracle"
            if lambda_oracle_mse <= cache_oracle_mse
            else "cache_penalty_lambda_grid"
        )
        baseline_zero_mse = float(np.mean(np.asarray(y_test, dtype=np.float64) ** 2))
        result = {
            "row": row,
            "prior": entry["prior"],
            "shard_id": int(entry["shard_id"]),
            "n": int(entry["n"]),
            "p": int(entry["p"]),
            "regime": (
                "n_gt_p" if int(entry["n"]) > int(entry["p"]) else "p_gt_n"
            ),
            "max_abs_feature_corr": max_corr,
            "lambda_max": float(lambda_max[0]),
            "selected_effective_ratio": selected_effective_ratio,
            "selected_raw_lambda": float(selected_lambda[0]),
            "selected_penalty_d0": float(selected_d0[0]),
            "selected_validation_mse": selected_mse,
            "selected_penalty_lambda_oracle_ratio": lambda_oracle_ratio,
            "selected_penalty_lambda_oracle_mse": lambda_oracle_mse,
            "cache_grid_oracle_penalty_index": cache_oracle_penalty_index,
            "cache_grid_oracle_ratio": cache_oracle_ratio,
            "cache_grid_oracle_mse": cache_oracle_mse,
            "joint_candidate_oracle_mse": joint_oracle_mse,
            "joint_candidate_oracle_source": joint_source,
            "selected_over_lambda_oracle": selected_mse / max(lambda_oracle_mse, 1e-12),
            "selected_over_joint_oracle": selected_mse / max(joint_oracle_mse, 1e-12),
            "zero_prediction_mse": baseline_zero_mse,
        }
        rows.append(result)

        selected_rho = selected["penalty_rho"].reshape(-1).float().cpu().numpy()
        selected_drho = selected["penalty_drho"].reshape(-1).float().cpu().numpy()
        if joint_source == "selected_penalty_lambda_oracle":
            oracle_rho = selected_rho
            oracle_drho = selected_drho
            oracle_label = "same penalty, lambda oracle"
            oracle_ratio = lambda_oracle_ratio
        else:
            oracle_rho = candidate_encoded["rho"][
                cache_oracle_penalty_position
            ].float().cpu().numpy()
            oracle_drho = candidate_encoded["drho"][
                cache_oracle_penalty_position
            ].float().cpu().numpy()
            oracle_label = f"cache penalty {cache_oracle_penalty_index}"
            oracle_ratio = cache_oracle_ratio
        penalty_curves.append({
            "selected_rho": selected_rho,
            "selected_drho": selected_drho,
            "oracle_rho": oracle_rho,
            "oracle_drho": oracle_drho,
            "oracle_label": oracle_label,
            "oracle_ratio": oracle_ratio,
        })
        np.savez_compressed(
            args.out_dir / f"row_{row:02d}_selector_grid.npz",
            selected_ratio_grid=ratio_grid.cpu().numpy(),
            selected_penalty_validation_mse=selected_penalty_mses,
            cache_candidate_indices=candidate_indices,
            cache_effective_ratio_grid=candidate_ratios.cpu().numpy(),
            cache_grid_validation_mse=candidate_mses.reshape(
                penalty_count, ratio_count
            ),
        )
        print(json.dumps(result, sort_keys=True), flush=True)

    labels = [
        f"{row['prior'].replace('_v1', '').replace('_v2', '')}-"
        f"{'N>P' if row['regime'] == 'n_gt_p' else 'P>N'}-{row['row']}"
        for row in rows
    ]
    x_axis = np.arange(len(rows))
    selected_mses = np.asarray([row["selected_validation_mse"] for row in rows])
    lambda_oracle_mses = np.asarray([
        row["selected_penalty_lambda_oracle_mse"] for row in rows
    ])
    joint_mses = np.asarray([row["joint_candidate_oracle_mse"] for row in rows])
    selected_ratios = np.asarray([row["selected_effective_ratio"] for row in rows])
    lambda_oracle_ratios = np.asarray([
        row["selected_penalty_lambda_oracle_ratio"] for row in rows
    ])

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    width = 0.26
    axes[0, 0].bar(x_axis - width, np.log10(selected_mses.clip(1e-12)), width, label="V17 selected")
    axes[0, 0].bar(x_axis, np.log10(lambda_oracle_mses.clip(1e-12)), width, label="same penalty, lambda oracle")
    axes[0, 0].bar(x_axis + width, np.log10(joint_mses.clip(1e-12)), width, label="joint candidate oracle")
    axes[0, 0].set_ylabel("log10(validation MSE)")
    axes[0, 0].set_title("Holdout prediction error")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].scatter(
        np.log10(lambda_oracle_ratios.clip(1e-12)),
        np.log10(selected_ratios.clip(1e-12)),
        c=np.arange(len(rows)),
        cmap="viridis",
        s=55,
    )
    bounds = [-6.0, 0.0]
    axes[0, 1].plot(bounds, bounds, "k:", lw=1)
    axes[0, 1].set_xlim(bounds)
    axes[0, 1].set_ylim(bounds)
    axes[0, 1].set_xlabel("log10(lambda oracle ratio)")
    axes[0, 1].set_ylabel("log10(V17 selected ratio)")
    axes[0, 1].set_title("Effective lambda selection")

    axes[1, 0].bar(
        x_axis - width / 2,
        selected_mses / np.maximum(lambda_oracle_mses, 1e-12),
        width,
        label="selected / lambda oracle",
    )
    axes[1, 0].bar(
        x_axis + width / 2,
        selected_mses / np.maximum(joint_mses, 1e-12),
        width,
        label="selected / joint oracle",
    )
    axes[1, 0].axhline(1.0, color="black", ls=":", lw=1)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("validation MSE ratio")
    axes[1, 0].set_title("Selector regret")
    axes[1, 0].legend(fontsize=8)

    cache_mses = np.asarray([row["cache_grid_oracle_mse"] for row in rows])
    axes[1, 1].bar(x_axis, cache_mses / np.maximum(lambda_oracle_mses, 1e-12))
    axes[1, 1].axhline(1.0, color="black", ls=":", lw=1)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("cache-grid best / selected-penalty lambda best")
    axes[1, 1].set_title("Penalty selection opportunity")

    for axis in (axes[0, 0], axes[1, 0], axes[1, 1]):
        axis.set_xticks(x_axis, labels, rotation=35, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 1].grid(alpha=0.25)
    figure.suptitle(
        f"Kalbi V2 + V17 lambda/penalty selection | checkpoint step {step}",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    summary_figure = args.out_dir / f"kalbi_v2_v17_step{step:06d}_selector_optimality.png"
    figure.savefig(summary_figure, dpi=180)
    plt.close(figure)

    knots = candidate_encoded["knots"].float().cpu().numpy()
    curve_figure, curve_axes = plt.subplots(
        len(rows), 1, figsize=(11, 2.6 * len(rows)), squeeze=False
    )
    for index, (row, curves) in enumerate(zip(rows, penalty_curves, strict=True)):
        axis = curve_axes[index, 0]
        axis.plot(knots, curves["selected_rho"], color="#1261a0", label="V17 selected rho")
        axis.plot(knots, curves["oracle_rho"], color="#d34e24", ls="--", label=str(curves["oracle_label"]))
        twin = axis.twinx()
        twin.plot(knots, curves["selected_drho"], color="#1261a0", alpha=0.35)
        twin.plot(knots, curves["oracle_drho"], color="#d34e24", ls="--", alpha=0.35)
        axis.set_title(
            f"row {index} {row['prior']} | selected r={row['selected_effective_ratio']:.3g}, "
            f"oracle r={float(curves['oracle_ratio']):.3g}",
            fontsize=9,
        )
        axis.set_ylabel("rho")
        twin.set_ylabel("rho'", fontsize=8)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, loc="best")
    curve_axes[-1, 0].set_xlabel("|beta|")
    curve_figure.tight_layout()
    curve_path = args.out_dir / f"kalbi_v2_v17_step{step:06d}_selected_vs_oracle_penalties.png"
    curve_figure.savefig(curve_path, dpi=170)
    plt.close(curve_figure)

    aggregate = {
        "geomean_selected_over_lambda_oracle": float(np.exp(np.mean(np.log(selected_mses / np.maximum(lambda_oracle_mses, 1e-12))))),
        "geomean_selected_over_joint_oracle": float(np.exp(np.mean(np.log(selected_mses / np.maximum(joint_mses, 1e-12))))),
        "median_selected_validation_mse": float(np.median(selected_mses)),
        "median_lambda_oracle_validation_mse": float(np.median(lambda_oracle_mses)),
        "median_joint_oracle_validation_mse": float(np.median(joint_mses)),
    }
    report = {
        "report_version": "kalbi_v2_v17_fixed_holdout_selector_grid_oracle_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "step": step,
        "same_training_catalog_tasks": True,
        "task_selection": args.selection,
        "split_seed": int(args.split_seed),
        "candidate_penalties": int(args.penalty_candidates),
        "candidate_lambda_grid_size": int(args.candidate_lambda_grid_size),
        "oracle_definition": "best holdout MSE under current Kalbi decoder over selected-penalty lambda grid plus fixed cache penalty/lambda grid",
        "aggregate": aggregate,
        "rows": rows,
        "summary_figure": str(summary_figure),
        "penalty_figure": str(curve_path),
    }
    (args.out_dir / "selector_optimality_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "summary_figure": str(summary_figure),
        "penalty_figure": str(curve_path),
        **aggregate,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
