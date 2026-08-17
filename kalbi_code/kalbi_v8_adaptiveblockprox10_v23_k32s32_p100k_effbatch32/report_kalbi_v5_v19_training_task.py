#!/usr/bin/env python3
"""Evaluate Kalbi V5 exact-prox and Selector V19 on one training task."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .eval_kalbi_v2_common import (
    encode_dataset,
    file_sha256,
    load_task,
    predict_from_encoded,
    split_standardize,
    tensors_from_arrays,
    vector_correlation,
)
from .exact_canonical_cd import (
    CanonicalCDKernel,
    canonical_penalty_values,
    objective_components,
    solve_multistart_path,
)
from .model import (
    KALBI_RIDGE,
    KALBI_STAGES,
    KALBI_TOPK,
    MODEL_VERSION,
    V3_SOURCE_MODEL_VERSION,
    build_model,
)
from .selector_v19_k32s32 import DirectPenaltyMLPQuerySelectorV19StageGain
from train_fpk import _load_yaml


INK = "#20242A"
MID = "#687078"
GRID = "#D8DDE3"
BLUE = "#2367A6"
GOLD = "#D39B22"
OLIVE = "#768B36"
FEATURE_COLORS = (
    "#2367A6",
    "#CF6A32",
    "#768B36",
    "#B64F7C",
    "#D39B22",
    "#535B65",
    "#6F8FB3",
    "#9D704B",
    "#8B769D",
    "#4F8A83",
    "#A5657A",
    "#6C7F47",
)


def load_components_v19(
    checkpoint_path: Path, config_path: Path, device: torch.device
) -> tuple[dict[str, Any], torch.nn.Module, torch.nn.Module]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    exact = checkpoint.get("metadata", {}).get("exact_training_arguments", {})
    expected = {
        "kalbi_isis_topk": KALBI_TOPK,
        "kalbi_isis_stages": KALBI_STAGES,
        "kalbi_isis_ridge": KALBI_RIDGE,
        "selector_stages": 32,
        "selector_topk": 32,
        "selector_cumulative_joint_ridge_path": True,
    }
    for key, value in expected.items():
        if exact.get(key) != value:
            raise RuntimeError(
                f"checkpoint architecture mismatch for {key}: "
                f"{exact.get(key)!r} != {value!r}"
            )

    model = build_model(_load_yaml(str(config_path)), device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    selector = DirectPenaltyMLPQuerySelectorV19StageGain(
        hidden=int(exact.get("selector_hidden", 512)),
        lambda_min_ratio=1e-6,
        lambda_max_ratio=1.0,
        initial_lambda_ratio=0.1,
        derivative_floor=1e-6,
        squareplus_scale=0.02,
        direct_head_init_std=1e-3,
    )
    selector.load_state_dict(checkpoint["selector_state_dict"], strict=True)
    selector.to(device).eval()
    return checkpoint, model, selector


def select_entry(
    catalog: Mapping[str, Any], shard_id: int
) -> dict[str, Any]:
    matches = [
        dict(entry)
        for entry in catalog["files"]
        if int(entry["shard_id"]) == int(shard_id)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one shard {shard_id}, found {len(matches)}")
    return matches[0]


def objective_paths(
    x: np.ndarray,
    y: np.ndarray,
    betas: np.ndarray,
    lambdas: np.ndarray,
    knots: np.ndarray,
    relu_b: float,
    relu_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(
        [
            objective_components(
                x, y, beta, raw_lambda, knots, relu_b, relu_weights
            )
            for beta, raw_lambda in zip(betas, lambdas, strict=True)
        ],
        dtype=np.float64,
    )
    return values[:, 0], values[:, 1], values[:, 2]


def safe_loss_curve(objective: np.ndarray, f_zero: float) -> np.ndarray:
    ratio = np.asarray(objective, dtype=np.float64) / max(float(f_zero), 1e-12)
    result = np.full(ratio.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(ratio) & (ratio > -1.0)
    result[valid] = np.log1p(ratio[valid])
    return result


def test_mse(x: np.ndarray, y: np.ndarray, betas: np.ndarray) -> np.ndarray:
    predictions = np.asarray(betas, dtype=np.float64) @ np.asarray(
        x, dtype=np.float64
    ).T
    return np.mean(
        (predictions - np.asarray(y, dtype=np.float64)[None, :]) ** 2,
        axis=1,
    )


def style_axis(axis: plt.Axes) -> None:
    axis.grid(color=GRID, alpha=0.72, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=10, colors=INK)


def load_generated_task(
    npz_path: Path,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    with np.load(npz_path, allow_pickle=False) as data:
        x_batch = np.asarray(data["X"])
        y_batch = np.asarray(data["y"])
        if x_batch.ndim != 3 or x_batch.shape[0] != 1:
            raise RuntimeError(f"expected one generated task in X, got {x_batch.shape}")
        if y_batch.ndim != 2 or y_batch.shape[0] != 1:
            raise RuntimeError(f"expected one generated task in y, got {y_batch.shape}")
        x = x_batch[0].astype(np.float32, copy=False)
        y = y_batch[0].astype(np.float32, copy=False)
        if y.shape != (x.shape[0],):
            raise RuntimeError(f"generated X/y shape mismatch: {x.shape}, {y.shape}")
        max_corr = float(np.asarray(data["core_exact_max_abs_corr"]).reshape(-1)[0])
        batch_seed = int(np.asarray(data["batch_seed"]).reshape(-1)[0])
    manifest_path = npz_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = [entry for entry in manifest["files"] if entry["path"] == npz_path.name]
    if len(files) != 1:
        raise RuntimeError(f"manifest entry mismatch for {npz_path}")
    entry = dict(files[0])
    if int(entry["n"]) != x.shape[0] or int(entry["p"]) != x.shape[1]:
        raise RuntimeError("manifest N/P does not match generated arrays")
    if not bool(entry.get("full_pair_exact_certificate")):
        raise RuntimeError("small generated task lacks a full-pair correlation certificate")
    if not max_corr < 0.95:
        raise RuntimeError(f"generated task violates max-correlation contract: {max_corr}")
    provenance = {
        "task_sha256": file_sha256(npz_path),
        "relative_path": str(npz_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "generator": manifest["generator"],
        "dataset_contract": manifest["dataset_contract"],
        "batch_seed": batch_seed,
        "n": int(x.shape[0]),
        "p": int(x.shape[1]),
    }
    return x, y, max_corr, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cd-library", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--shard-id", type=int)
    source.add_argument("--data-npz", type=Path)
    parser.add_argument("--task-label", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--queries", type=int, default=41)
    parser.add_argument("--split-seed", type=int, default=2026072319)
    parser.add_argument("--max-sweeps", type=int, default=2000)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--zero-tolerance", type=float, default=1e-10)
    args = parser.parse_args()

    if args.out_dir.exists():
        raise RuntimeError(f"refusing existing output directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint, model, selector = load_components_v19(
        args.checkpoint, args.config, device
    )
    step = int(checkpoint["completed_steps"])
    if (
        checkpoint.get("metadata", {}).get("model_version")
        != V3_SOURCE_MODEL_VERSION
    ):
        raise RuntimeError("checkpoint is not the strict Kalbi V3 source contract")

    catalog = checkpoint["catalog_payload"]
    if args.data_npz is None:
        entry = select_entry(catalog, int(args.shard_id))
        data_root = Path(str(catalog["data_root"])).resolve()
        x_raw, y_raw, max_corr = load_task(data_root, entry)
        provenance = {
            "task_sha256": entry["sha256"],
            "relative_path": entry["relative_path"],
            "manifest": None,
            "manifest_sha256": None,
            "generator": entry.get("prior"),
            "dataset_contract": entry.get("dataset_contract"),
            "batch_seed": entry.get("batch_seed"),
            "n": int(entry["n"]),
            "p": int(entry["p"]),
        }
        source_description = f"active training shard {entry['shard_id']}"
        source_kind = "active_training_catalog"
        shard_id: int | None = int(entry["shard_id"])
        catalog_epoch: int | None = int(catalog["epoch"])
    else:
        x_raw, y_raw, max_corr, provenance = load_generated_task(args.data_npz)
        source_description = args.task_label or args.data_npz.parent.name
        source_kind = "independently_generated_v11"
        shard_id = None
        catalog_epoch = None
    x_train, y_train, x_test, y_test = split_standardize(
        x_raw, y_raw, args.split_seed
    )
    x_device, y_device = tensors_from_arrays(x_train, y_train, device)
    encoded = encode_dataset(model, x_device, y_device)
    lambda_max_tensor = encoded["lambda_max"].float().reshape(-1)
    lambda_max = float(lambda_max_tensor[0])
    with torch.inference_mode():
        selected = selector(
            x_device,
            y_device,
            lambda_max_tensor,
            return_representation=True,
        )

    selected_ratio = float(selected["effective_lambda_ratio"].reshape(-1)[0])
    selected_raw_lambda = float(selected["lambdas"].reshape(-1)[0])
    selected_d0 = float(selected["penalty_d0"].reshape(-1)[0])
    selected_shape = selected["penalty_shape_features"].reshape(1, -1)
    base_ratios = np.logspace(-6.0, 0.0, int(args.queries), dtype=np.float64)
    ratios = np.unique(np.concatenate((base_ratios, np.asarray([selected_ratio]))))
    ratios.sort()
    raw_lambdas = ratios * lambda_max / max(selected_d0, 1e-12)
    shape_queries = selected_shape.expand(ratios.size, -1)
    d0_queries = selected["penalty_d0"].reshape(-1).expand(ratios.size)
    model_beta_raw = predict_from_encoded(
        model,
        encoded,
        torch.as_tensor(raw_lambdas, device=device, dtype=torch.float32),
        shape_queries,
        d0_queries,
        chunk_size=8,
    ).numpy().astype(np.float64)

    knots = (
        selector.penalty_parameter_head.penalty_grid.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    relu_b = float(selected["penalty_relu_b"].reshape(-1)[0])
    relu_weights = (
        selected["penalty_relu_weights"]
        .reshape(-1, selected["penalty_relu_weights"].shape[-1])[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    kernel = CanonicalCDKernel(args.cd_library)
    solved = solve_multistart_path(
        kernel=kernel,
        x=x_train,
        y=y_train,
        raw_lambdas=raw_lambdas,
        knots=knots,
        relu_b=relu_b,
        relu_weights=relu_weights,
        model_beta=model_beta_raw,
        max_sweeps=int(args.max_sweeps),
        retries=int(args.retries),
        tolerance=float(args.tolerance),
        zero_tolerance=float(args.zero_tolerance),
    )
    model_beta = solved["clipped_model_beta"]
    reference_beta = solved["reference_beta"]
    fixed_point_ceiling = max(2e-3, 5.0 * float(args.tolerance))
    if float(np.max(solved["reference_fixed_point_max"])) > fixed_point_ceiling:
        raise RuntimeError("exact-CD fixed-point residual exceeded audit ceiling")

    model_data, model_penalty, model_objective = objective_paths(
        x_train,
        y_train,
        model_beta,
        raw_lambdas,
        knots,
        relu_b,
        relu_weights,
    )
    cd_data, cd_penalty, cd_objective = objective_paths(
        x_train,
        y_train,
        reference_beta,
        raw_lambdas,
        knots,
        relu_b,
        relu_weights,
    )
    if bool((cd_objective > model_objective + 2e-8).any()):
        raise RuntimeError("exact CD reference exceeded its Kalbi start")
    f_zero = 0.5 * float(np.mean(np.asarray(y_train, dtype=np.float64) ** 2))
    model_loss = safe_loss_curve(model_objective, f_zero)
    cd_loss = safe_loss_curve(cd_objective, f_zero)
    model_test_mse = test_mse(x_test, y_test, model_beta)
    cd_test_mse = test_mse(x_test, y_test, reference_beta)
    selected_index = int(np.argmin(np.abs(ratios - selected_ratio)))
    model_oracle_index = int(np.argmin(model_test_mse))
    cd_oracle_index = int(np.argmin(cd_test_mse))
    correlations = np.asarray(
        [
            vector_correlation(model_beta[index], reference_beta[index])
            for index in range(ratios.size)
        ]
    )

    metrics = {
        "report_version": "kalbi_v5_exactprox_v19_training_task_exactcd_v1",
        "evaluated_model_version": MODEL_VERSION,
        "source_model_version": V3_SOURCE_MODEL_VERSION,
        "source_kind": source_kind,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_step": step,
        "catalog_epoch": catalog_epoch,
        "shard_id": shard_id,
        **provenance,
        "train_rows": int(x_train.shape[0]),
        "test_rows": int(x_test.shape[0]),
        "max_abs_feature_corr": float(max_corr),
        "split_seed": int(args.split_seed),
        "lambda_max": lambda_max,
        "selected_penalty_d0": selected_d0,
        "selected_effective_ratio": selected_ratio,
        "selected_raw_lambda": selected_raw_lambda,
        "selected_test_mse": float(model_test_mse[selected_index]),
        "model_path_oracle_ratio": float(ratios[model_oracle_index]),
        "model_path_oracle_test_mse": float(model_test_mse[model_oracle_index]),
        "exact_cd_path_oracle_ratio": float(ratios[cd_oracle_index]),
        "exact_cd_path_oracle_test_mse": float(cd_test_mse[cd_oracle_index]),
        "selected_over_model_path_oracle": float(
            model_test_mse[selected_index]
            / max(float(model_test_mse[model_oracle_index]), 1e-12)
        ),
        "selected_over_exact_cd_oracle": float(
            model_test_mse[selected_index]
            / max(float(cd_test_mse[cd_oracle_index]), 1e-12)
        ),
        "selected_nnz_1e4": int(
            np.count_nonzero(np.abs(model_beta[selected_index]) > 1e-4)
        ),
        "mean_solution_path_correlation_with_cd": float(np.mean(correlations)),
        "loss_curve_rmse_vs_cd": float(np.sqrt(np.mean((model_loss - cd_loss) ** 2))),
        "reference_fixed_point_max": float(
            np.max(solved["reference_fixed_point_max"])
        ),
        "reference_projected_kkt_max": float(
            np.max(solved["reference_projected_kkt_max"])
        ),
    }

    np.savez_compressed(
        args.out_dir / "small_training_task_paths.npz",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        effective_lambda_ratios=ratios,
        raw_lambdas=raw_lambdas,
        kalbi_v5_beta=model_beta.astype(np.float32),
        exact_cd_beta=reference_beta.astype(np.float32),
        kalbi_v5_data_half=model_data,
        kalbi_v5_raw_penalty=model_penalty,
        kalbi_v5_objective=model_objective,
        exact_cd_data_half=cd_data,
        exact_cd_raw_penalty=cd_penalty,
        exact_cd_objective=cd_objective,
        kalbi_v5_train_loss_curve=model_loss,
        exact_cd_train_loss_curve=cd_loss,
        kalbi_v5_test_mse=model_test_mse,
        exact_cd_test_mse=cd_test_mse,
        selected_index=np.asarray(selected_index, dtype=np.int64),
        model_oracle_index=np.asarray(model_oracle_index, dtype=np.int64),
        cd_oracle_index=np.asarray(cd_oracle_index, dtype=np.int64),
        selected_penalty_rho=selected["penalty_rho"].reshape(-1).float().cpu().numpy(),
        selected_penalty_drho=selected["penalty_drho"].reshape(-1).float().cpu().numpy(),
    )

    x_axis = np.log10(ratios)
    score = np.maximum(
        np.max(np.abs(model_beta), axis=0),
        np.max(np.abs(reference_beta), axis=0),
    )
    top = np.argsort(-score)[:12]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.1))
    penalty_grid = np.linspace(0.0, float(knots[-1]), 320)
    axes[0].plot(
        penalty_grid,
        canonical_penalty_values(
            penalty_grid, knots, relu_b, relu_weights
        ),
        color=OLIVE,
        lw=2.0,
    )
    axes[0].set_xlabel("|coefficient|", fontsize=12)
    axes[0].set_ylabel("selected penalty rho", fontsize=12)
    axes[0].set_title(f"V19 selected penalty | d0={selected_d0:.3g}", fontsize=13)
    style_axis(axes[0])

    for color_index, feature in enumerate(top):
        color = FEATURE_COLORS[color_index % len(FEATURE_COLORS)]
        axes[1].plot(x_axis, model_beta[:, feature], color=color, lw=1.3)
        axes[1].plot(
            x_axis,
            reference_beta[:, feature],
            color=color,
            lw=1.0,
            ls=":",
            alpha=0.8,
        )
    axes[1].axvline(math.log10(selected_ratio), color=GOLD, ls="--", lw=1.5)
    axes[1].axhline(0.0, color=INK, lw=0.6)
    axes[1].set_yscale("symlog", linthresh=0.02)
    axes[1].set_xlabel("log10(effective lambda ratio)", fontsize=12)
    axes[1].set_ylabel("coefficient", fontsize=12)
    axes[1].set_title("Coefficient paths | solid Kalbi, dotted exact CD", fontsize=13)
    style_axis(axes[1])

    axes[2].plot(
        x_axis, model_loss, color=BLUE, marker="o", ms=3.2, lw=1.7, label="Kalbi V5"
    )
    axes[2].plot(x_axis, cd_loss, color=INK, ls=":", lw=1.7, label="bounded exact CD")
    axes[2].axvline(
        math.log10(selected_ratio),
        color=GOLD,
        ls="--",
        lw=1.5,
        label="V19 selected lambda",
    )
    axes[2].set_xlabel("log10(effective lambda ratio)", fontsize=12)
    axes[2].set_ylabel("log1p(F/F0)", fontsize=12)
    axes[2].set_title(
        f"Training objective | path RMSE={metrics['loss_curve_rmse_vs_cd']:.3g}",
        fontsize=13,
    )
    axes[2].legend(frameon=False, fontsize=10)
    style_axis(axes[2])
    figure.suptitle(
        f"Kalbi V5 exact-prox + Selector V19 | {source_description} | "
        f"N={provenance['n']}, P={provenance['p']} | V3 source step {step}",
        fontsize=15,
        color=INK,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95), w_pad=2.0)
    solution_path = args.out_dir / "solution_path_vs_exact_cd.png"
    figure.savefig(solution_path, dpi=220, facecolor="white")
    plt.close(figure)

    model_log_mse = np.log10(np.asarray(model_test_mse).clip(1e-12))
    cd_log_mse = np.log10(np.asarray(cd_test_mse).clip(1e-12))
    figure, axes = plt.subplots(1, 2, figsize=(15.2, 5.8))
    axis = axes[0]
    axis.plot(x_axis, model_log_mse, color=BLUE, lw=2.0, label="Kalbi V5 path")
    axis.plot(x_axis, cd_log_mse, color=INK, ls=":", lw=1.8, label="exact CD path")
    axis.scatter(
        [x_axis[selected_index]],
        [model_log_mse[selected_index]],
        marker="D",
        s=80,
        color=GOLD,
        edgecolor=INK,
        linewidth=0.7,
        zorder=5,
        label="V19 selected",
    )
    axis.scatter(
        [x_axis[model_oracle_index]],
        [model_log_mse[model_oracle_index]],
        marker="*",
        s=150,
        facecolor="white",
        edgecolor=BLUE,
        linewidth=1.5,
        zorder=5,
        label="Kalbi path oracle",
    )
    axis.scatter(
        [x_axis[cd_oracle_index]],
        [cd_log_mse[cd_oracle_index]],
        marker="x",
        s=90,
        color=INK,
        linewidth=1.6,
        zorder=5,
        label="exact CD oracle",
    )
    axis.set_xlabel("log10(effective lambda ratio)", fontsize=13)
    axis.set_ylabel("log10(held-out MSE)", fontsize=13)
    axis.set_title(
        f"V19 lambda selection | selected/oracle={metrics['selected_over_model_path_oracle']:.3f}",
        fontsize=14,
    )
    axis.legend(frameon=False, fontsize=11, ncol=2)
    style_axis(axis)

    penalty_axis = axes[1]
    selected_rho = (
        selected["penalty_rho"].reshape(-1).detach().float().cpu().numpy()
    )
    selected_drho = (
        selected["penalty_drho"].reshape(-1).detach().float().cpu().numpy()
    )
    penalty_axis.plot(knots, selected_rho, color=OLIVE, lw=2.1, label="rho(t)")
    penalty_axis.set_xlabel("|coefficient|", fontsize=13)
    penalty_axis.set_ylabel("selected penalty rho(t)", fontsize=13, color=OLIVE)
    penalty_axis.tick_params(axis="y", labelcolor=OLIVE)
    derivative_axis = penalty_axis.twinx()
    derivative_axis.plot(
        knots,
        selected_drho,
        color=GOLD,
        lw=1.7,
        ls="--",
        label="rho'(t)",
    )
    derivative_axis.set_ylabel("penalty derivative", fontsize=13, color=GOLD)
    derivative_axis.tick_params(axis="y", labelcolor=GOLD, labelsize=10)
    derivative_axis.spines["top"].set_visible(False)
    penalty_axis.set_title(f"V19 selected penalty | d0={selected_d0:.3g}", fontsize=14)
    penalty_axis.legend(loc="upper left", frameon=False, fontsize=11)
    derivative_axis.legend(loc="upper right", frameon=False, fontsize=11)
    style_axis(penalty_axis)
    figure.tight_layout()
    selector_path = args.out_dir / "lambda_penalty_selector.png"
    figure.savefig(selector_path, dpi=220, facecolor="white")
    plt.close(figure)

    metrics["figures"] = {
        "solution_path": str(solution_path.resolve()),
        "lambda_penalty_selector": str(selector_path.resolve()),
    }
    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
