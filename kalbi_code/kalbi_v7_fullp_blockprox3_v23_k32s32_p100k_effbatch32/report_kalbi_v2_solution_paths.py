#!/usr/bin/env python3
"""Render Kalbi V2 solution paths against a coordinate-descent reference."""

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
    coordinate_descent_path,
    encode_dataset,
    file_sha256,
    load_components,
    load_task,
    objective_path,
    predict_from_encoded,
    rho_abs,
    segments_from_knots,
    select_balanced_entries,
    select_smallest_p_regime_entries,
    standardize_full,
    tensors_from_arrays,
    vector_correlation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--penalty-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--queries", type=int, default=25)
    parser.add_argument("--reference-sweeps", type=int, default=16)
    parser.add_argument("--reference-tolerance", type=float, default=1e-7)
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
    checkpoint, model, _selector = load_components(
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

    cache = CanonicalPenaltyCache(args.penalty_cache, seed=20260721)
    cache.load(expected_count=300_000)
    penalty_indices = np.linspace(
        0, cache.pool_size - 1, int(args.count)
    ).round().astype(np.int64)
    effective_ratios = np.logspace(
        -6.0, 0.0, int(args.queries), dtype=np.float64
    )

    figure, axes = plt.subplots(
        int(args.count), 3, figsize=(18, 3.35 * int(args.count)), squeeze=False
    )
    rows: list[dict[str, object]] = []
    for row, (entry, penalty_index) in enumerate(
        zip(entries, penalty_indices, strict=True)
    ):
        X_raw, y_raw, max_corr = load_task(data_root, entry)
        X, y = standardize_full(X_raw, y_raw)
        X_device, y_device = tensors_from_arrays(X, y, device)
        encoded_dataset = encode_dataset(model, X_device, y_device)
        lambda_max = float(encoded_dataset["lambda_max"].reshape(-1)[0])
        encoded_penalty = cache.encoded_indices(
            np.asarray([penalty_index], dtype=np.int64), device
        )
        shape = encoded_penalty["shape_features"].expand(
            len(effective_ratios), -1
        )
        d0_scalar = float(encoded_penalty["d0"].reshape(-1)[0])
        d0 = encoded_penalty["d0"].expand(len(effective_ratios))
        lambdas = effective_ratios * lambda_max / max(d0_scalar, 1e-12)
        model_beta = predict_from_encoded(
            model,
            encoded_dataset,
            torch.as_tensor(lambdas, device=device, dtype=torch.float32),
            shape,
            d0,
        ).numpy()
        segments = segments_from_knots(
            encoded_penalty["knots"].detach().cpu().numpy(),
            encoded_penalty["rho"][0].detach().cpu().numpy(),
            encoded_penalty["drho"][0].detach().cpu().numpy(),
            tail_hi=max(20.0, 2.0 + 2.0 * float(np.abs(model_beta).max())),
        )
        gram = np.asarray(X, dtype=np.float64).T @ np.asarray(X, dtype=np.float64) / X.shape[0]
        linear = np.asarray(X, dtype=np.float64).T @ np.asarray(y, dtype=np.float64) / X.shape[0]
        reference_from_model = coordinate_descent_path(
            gram,
            linear,
            lambdas,
            segments,
            model_beta,
            sweeps=int(args.reference_sweeps),
            tolerance=float(args.reference_tolerance),
        )
        reference_from_zero = coordinate_descent_path(
            gram,
            linear,
            lambdas,
            segments,
            np.zeros_like(model_beta),
            sweeps=int(args.reference_sweeps),
            tolerance=float(args.reference_tolerance),
        )
        model_objective = objective_path(X, y, model_beta, lambdas, segments)
        objective_model_init = objective_path(
            X, y, reference_from_model, lambdas, segments
        )
        objective_zero_init = objective_path(
            X, y, reference_from_zero, lambdas, segments
        )
        candidate_objectives = np.stack(
            [model_objective, objective_model_init, objective_zero_init], axis=1
        )
        best_candidate = np.argmin(candidate_objectives, axis=1)
        reference_beta = np.empty_like(model_beta)
        for query_index, candidate_index in enumerate(best_candidate):
            reference_beta[query_index] = (
                model_beta[query_index]
                if candidate_index == 0
                else reference_from_model[query_index]
                if candidate_index == 1
                else reference_from_zero[query_index]
            )
        reference_objective = candidate_objectives[
            np.arange(len(effective_ratios)), best_candidate
        ]
        if bool((reference_objective > model_objective + 1e-8).any()):
            raise RuntimeError("CD reference objective exceeded Kalbi initialization")
        f_zero = 0.5 * float(np.mean(np.asarray(y, dtype=np.float64) ** 2))
        model_curve = np.log1p(model_objective / max(f_zero, 1e-12))
        reference_curve = np.log1p(reference_objective / max(f_zero, 1e-12))
        loss_gap = model_curve - reference_curve
        point_correlations = np.asarray([
            vector_correlation(model_beta[index], reference_beta[index])
            for index in range(len(effective_ratios))
        ])
        metrics = {
            "row": row,
            "prior": entry["prior"],
            "shard_id": int(entry["shard_id"]),
            "n": int(entry["n"]),
            "p": int(entry["p"]),
            "regime": (
                "n_gt_p" if int(entry["n"]) > int(entry["p"]) else "p_gt_n"
            ),
            "penalty_index": int(penalty_index),
            "lambda_max": lambda_max,
            "penalty_d0": d0_scalar,
            "max_abs_feature_corr": max_corr,
            "mean_loss_gap": float(loss_gap.mean()),
            "loss_curve_rmse": float(np.sqrt(np.mean(loss_gap * loss_gap))),
            "max_abs_loss_gap": float(np.max(np.abs(loss_gap))),
            "mean_path_correlation": float(point_correlations.mean()),
            "model_mean_nnz_1e4": float((np.abs(model_beta) > 1e-4).sum(1).mean()),
            "reference_mean_nnz_1e4": float((np.abs(reference_beta) > 1e-4).sum(1).mean()),
        }
        rows.append(metrics)
        np.savez_compressed(
            args.out_dir / f"row_{row:02d}_paths.npz",
            X=X,
            y=y,
            effective_lambda_ratios=effective_ratios,
            raw_lambdas=lambdas,
            model_beta=model_beta,
            reference_beta=reference_beta,
            model_objective=model_objective,
            reference_objective=reference_objective,
            model_loss_curve=model_curve,
            reference_loss_curve=reference_curve,
            point_correlations=point_correlations,
        )

        grid = np.linspace(0.0, 6.0, 240)
        axes[row, 0].plot(
            grid, [rho_abs(value, segments) for value in grid], color="#1261a0"
        )
        axes[row, 0].set_title(
            f"row {row} canonical penalty {penalty_index}", fontsize=8
        )
        axes[row, 0].set_xlabel("|beta|", fontsize=7)
        axes[row, 0].set_ylabel("rho", fontsize=7)
        axes[row, 0].grid(alpha=0.25)

        score = np.maximum(
            np.max(np.abs(model_beta), axis=0),
            np.max(np.abs(reference_beta), axis=0),
        )
        top = np.argsort(-score)[:12]
        x_axis = np.log10(effective_ratios)
        for feature in top:
            line = axes[row, 1].plot(
                x_axis, model_beta[:, feature], lw=1.0, alpha=0.85
            )[0]
            axes[row, 1].plot(
                x_axis,
                reference_beta[:, feature],
                lw=0.9,
                ls=":",
                color=line.get_color(),
                alpha=0.65,
            )
        axes[row, 1].axhline(0.0, color="black", lw=0.4)
        axes[row, 1].set_title(
            f"{entry['prior']} N={entry['n']}, P={entry['p']} "
            f"corr={point_correlations.mean():.3f}",
            fontsize=8,
        )
        axes[row, 1].set_xlabel("log10(effective lambda ratio)", fontsize=7)
        axes[row, 1].set_ylabel("beta_j", fontsize=7)
        axes[row, 1].grid(alpha=0.25)

        axes[row, 2].plot(
            x_axis, reference_curve, "o:", ms=3, color="black", label="CD reference"
        )
        axes[row, 2].plot(
            x_axis, model_curve, "^-", ms=3, color="#d34e24", label="Kalbi V2"
        )
        axes[row, 2].set_title(
            f"loss RMSE={metrics['loss_curve_rmse']:.4g}", fontsize=8
        )
        axes[row, 2].set_xlabel("log10(effective lambda ratio)", fontsize=7)
        axes[row, 2].set_ylabel("log1p(F/F0)", fontsize=7)
        axes[row, 2].grid(alpha=0.25)
        axes[row, 2].legend(fontsize=7)
        for axis in axes[row]:
            axis.tick_params(labelsize=7)
        print(json.dumps(metrics, sort_keys=True), flush=True)

    figure.suptitle(
        f"Kalbi V2 K32/S32 solution paths vs CD | checkpoint step {step}",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.993))
    figure_path = args.out_dir / f"kalbi_v2_step{step:06d}_solution_paths_vs_cd.png"
    figure.savefig(figure_path, dpi=170)
    plt.close(figure)
    aggregate = {
        "mean_loss_gap": float(np.mean([row["mean_loss_gap"] for row in rows])),
        "mean_loss_curve_rmse": float(np.mean([row["loss_curve_rmse"] for row in rows])),
        "max_abs_loss_gap": float(np.max([row["max_abs_loss_gap"] for row in rows])),
        "mean_path_correlation": float(np.mean([row["mean_path_correlation"] for row in rows])),
    }
    report = {
        "report_version": "kalbi_v2_k32s32_canonical_cd_path_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "step": step,
        "same_training_catalog_tasks": True,
        "task_selection": args.selection,
        "effective_lambda_ratios": effective_ratios.tolist(),
        "reference": "lower objective of coordinate descent initialized from Kalbi and zero",
        "reference_sweeps": int(args.reference_sweeps),
        "aggregate": aggregate,
        "rows": rows,
        "figure": str(figure_path),
    }
    (args.out_dir / "solution_path_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"figure": str(figure_path), **aggregate}, sort_keys=True))


if __name__ == "__main__":
    main()
