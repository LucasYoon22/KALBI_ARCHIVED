#!/usr/bin/env python3
"""Plot one generated TabICLv2 task under one random canonical penalty."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

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
    predict_from_encoded,
    standardize_full,
    tensors_from_arrays,
    vector_correlation,
)
from .exact_canonical_cd import (
    CanonicalCDKernel,
    canonical_penalty_values,
    objective_components,
    solve_multistart_path,
)
from .model import MODEL_VERSION


INK = "#20242a"
GRID = "#d8dde3"
BLUE = "#2367a6"
ORANGE = "#cf6a32"
FEATURE_COLORS = (
    "#2367a6",
    "#cf6a32",
    "#768b36",
    "#b64f7c",
    "#d39b22",
    "#535b65",
    "#6f8fb3",
    "#9d704b",
    "#8b769d",
    "#4f8a83",
    "#a45c68",
    "#47788f",
)


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_generated_task(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) != 1:
        raise RuntimeError("evaluation manifest must contain exactly one task")
    entry = dict(entries[0])
    task_metadata = entry.get("task_metadata")
    if not isinstance(task_metadata, list) or len(task_metadata) != 1:
        raise RuntimeError("evaluation task metadata is missing")
    task = dict(task_metadata[0])
    path = (manifest_path.parent / str(entry["path"])).resolve()
    if manifest_path.parent.resolve() not in path.parents or not path.is_file():
        raise RuntimeError(f"evaluation task is missing or outside root: {path}")
    if int(entry.get("bytes", -1)) != path.stat().st_size:
        raise RuntimeError("evaluation task byte certificate is inconsistent")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "X",
            "y",
            "latent_scm_signal",
            "response_noise_unit",
            "response_noise_sigma",
            "response_noise_family_id",
            "exact_max_abs_corr",
            "corr_threshold",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise RuntimeError(f"evaluation task lacks arrays: {missing}")
        x = np.asarray(archive["X"])
        y = np.asarray(archive["y"])
        max_corr = float(np.asarray(archive["exact_max_abs_corr"]).reshape(-1)[0])
        threshold = float(np.asarray(archive["corr_threshold"]).reshape(-1)[0])
        sigma = float(np.asarray(archive["response_noise_sigma"]).reshape(-1)[0])
        family_id = int(
            np.asarray(archive["response_noise_family_id"]).reshape(-1)[0]
        )
    if x.shape != (1, 200, 600) or y.shape != (1, 200):
        raise RuntimeError(f"expected N=200 P=600, got X={x.shape}, y={y.shape}")
    if x.dtype != np.float16 or y.dtype != np.float32:
        raise RuntimeError(f"unexpected task dtypes: X={x.dtype}, y={y.dtype}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError("evaluation task contains nonfinite values")
    if not 0.0 <= max_corr < threshold or abs(threshold - 0.95) > 1e-6:
        raise RuntimeError("evaluation feature-correlation certificate failed")
    if abs(float(y.mean())) > 2e-6 or abs(float(y.std()) - 1.0) > 2e-6:
        raise RuntimeError("evaluation response is not standardized")
    if not math.isclose(
        sigma,
        float(task["response_noise_sigma"]),
        rel_tol=0.0,
        abs_tol=2e-6,
    ):
        raise RuntimeError("evaluation response-noise sigma is inconsistent")
    if family_id != int(task["response_noise_family_id"]):
        raise RuntimeError("evaluation response-noise family is inconsistent")
    entry["task_sha256"] = file_sha256(path)
    entry["max_abs_feature_corr"] = max_corr
    return manifest, entry, x[0], y[0]


def _objective_paths(
    x: np.ndarray,
    y: np.ndarray,
    betas: np.ndarray,
    lambdas: np.ndarray,
    knots: np.ndarray,
    relu_b: float,
    relu_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    components = np.asarray(
        [
            objective_components(
                x, y, beta, raw_lambda, knots, relu_b, relu_weights
            )
            for beta, raw_lambda in zip(betas, lambdas, strict=True)
        ],
        dtype=np.float64,
    )
    return components[:, 0], components[:, 1], components[:, 2]


def _loss_curve(objective: np.ndarray, f_zero: float) -> np.ndarray:
    ratio = np.asarray(objective, dtype=np.float64) / max(f_zero, 1e-12)
    if not np.all(np.isfinite(ratio) & (ratio > -1.0)):
        raise RuntimeError("objective path cannot be transformed with log1p")
    return np.log1p(ratio)


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(color=GRID, alpha=0.7, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8, colors=INK)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--penalty-cache", type=Path, required=True)
    parser.add_argument("--cd-library", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--penalty-seed", type=int, default=2607221601)
    parser.add_argument("--queries", type=int, default=41)
    parser.add_argument("--max-sweeps", type=int, default=2000)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--zero-tolerance", type=float, default=1e-10)
    args = parser.parse_args()

    if args.out_dir.exists():
        raise RuntimeError(f"refusing existing output directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint, model, _selector = load_components(
        args.checkpoint, args.config, device
    )
    step = int(checkpoint["completed_steps"])
    metadata = checkpoint.get("metadata", {})
    if metadata.get("model_version") != MODEL_VERSION:
        raise RuntimeError("checkpoint is not the strict Kalbi V3 model contract")

    manifest, entry, x_raw, y_raw = _load_generated_task(args.task_manifest)
    x, y = standardize_full(x_raw, y_raw)
    x_device, y_device = tensors_from_arrays(x, y, device)
    encoded_dataset = encode_dataset(model, x_device, y_device)
    lambda_max = float(encoded_dataset["lambda_max"].reshape(-1)[0])

    cache = CanonicalPenaltyCache(args.penalty_cache, seed=args.penalty_seed)
    cache_metadata = cache.load(expected_count=300_000)
    rng = np.random.default_rng(int(args.penalty_seed))
    penalty_index = int(rng.integers(0, cache.pool_size))
    encoded_penalty = cache.encoded_indices(
        np.asarray([penalty_index], dtype=np.int64), device
    )
    effective_ratios = np.logspace(
        -6.0, 0.0, int(args.queries), dtype=np.float64
    )
    shape = encoded_penalty["shape_features"].expand(
        effective_ratios.size, -1
    )
    d0_scalar = float(encoded_penalty["d0"].reshape(-1)[0])
    d0 = encoded_penalty["d0"].expand(effective_ratios.size)
    raw_lambdas = effective_ratios * lambda_max / max(d0_scalar, 1e-12)
    model_beta = predict_from_encoded(
        model,
        encoded_dataset,
        torch.as_tensor(raw_lambdas, device=device, dtype=torch.float32),
        shape,
        d0,
        chunk_size=8,
    ).numpy().astype(np.float64)

    knots = (
        encoded_penalty["knots"].detach().cpu().numpy().astype(np.float64)
    )
    rho_knots = (
        encoded_penalty["rho"][0].detach().cpu().numpy().astype(np.float64)
    )
    drho_knots = (
        encoded_penalty["drho"][0].detach().cpu().numpy().astype(np.float64)
    )
    relu_b = float(encoded_penalty["relu_b"].reshape(-1)[0])
    relu_weights = (
        encoded_penalty["relu_weights"][0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    bound = float(knots[-1])
    domain_valid = np.max(np.abs(model_beta), axis=1) <= bound + 2e-5

    kernel = CanonicalCDKernel(args.cd_library)
    solved = solve_multistart_path(
        kernel=kernel,
        x=x,
        y=y,
        raw_lambdas=raw_lambdas,
        knots=knots,
        relu_b=relu_b,
        relu_weights=relu_weights,
        model_beta=model_beta,
        max_sweeps=int(args.max_sweeps),
        retries=int(args.retries),
        tolerance=float(args.tolerance),
        zero_tolerance=float(args.zero_tolerance),
    )
    fixed_point_ceiling = max(2e-3, 5.0 * float(args.tolerance))
    if float(np.max(solved["reference_fixed_point_max"])) > fixed_point_ceiling:
        raise RuntimeError("exact-CD fixed-point residual exceeded audit ceiling")
    clipped_beta = solved["clipped_model_beta"]
    reference_beta = solved["reference_beta"]
    actual_data, actual_penalty, actual_objective = _objective_paths(
        x, y, model_beta, raw_lambdas, knots, relu_b, relu_weights
    )
    clipped_data, clipped_penalty, clipped_objective = _objective_paths(
        x, y, clipped_beta, raw_lambdas, knots, relu_b, relu_weights
    )
    cd_data, cd_penalty, cd_objective = _objective_paths(
        x, y, reference_beta, raw_lambdas, knots, relu_b, relu_weights
    )
    if bool((cd_objective > clipped_objective + 2e-8).any()):
        raise RuntimeError("exact CD reference exceeded its Kalbi start")
    f_zero = 0.5 * float(np.mean(np.asarray(y, dtype=np.float64) ** 2))
    actual_curve = _loss_curve(actual_objective, f_zero)
    actual_curve[~domain_valid] = np.nan
    model_curve = _loss_curve(clipped_objective, f_zero)
    cd_curve = _loss_curve(cd_objective, f_zero)
    loss_gap = model_curve - cd_curve
    correlations = np.asarray(
        [
            vector_correlation(clipped_beta[index], reference_beta[index])
            for index in range(effective_ratios.size)
        ],
        dtype=np.float64,
    )

    task_metadata = manifest["files"][0]["task_metadata"][0]
    metrics = {
        "checkpoint_step": step,
        "n": 200,
        "p": 600,
        "task_seed": int(manifest["files"][0]["batch_seed"]),
        "task_sha256": entry["task_sha256"],
        "max_abs_feature_corr": float(entry["max_abs_feature_corr"]),
        "response_noise_family": task_metadata["response_noise_family"],
        "response_noise_sigma": float(task_metadata["response_noise_sigma"]),
        "latent_signal_noisy_r2_ceiling": float(
            task_metadata["latent_signal_noisy_r2_ceiling"]
        ),
        "penalty_seed": int(args.penalty_seed),
        "penalty_index": penalty_index,
        "penalty_d0": d0_scalar,
        "penalty_rho_at_zero": float(
            canonical_penalty_values(
                np.asarray([0.0]), knots, relu_b, relu_weights
            )[0]
        ),
        "penalty_min_knot_derivative": float(np.min(drho_knots)),
        "lambda_max": lambda_max,
        "canonical_beta_bound": bound,
        "actual_max_abs_beta": float(np.max(np.abs(model_beta))),
        "domain_violation_query_fraction": float(np.mean(~domain_valid)),
        "mean_loss_gap": float(np.mean(loss_gap)),
        "loss_curve_rmse": float(np.sqrt(np.mean(loss_gap * loss_gap))),
        "max_abs_loss_gap": float(np.max(np.abs(loss_gap))),
        "mean_path_correlation": float(np.mean(correlations)),
        "minimum_path_correlation": float(np.min(correlations)),
        "model_mean_nnz_1e4": float(
            np.mean(np.sum(np.abs(clipped_beta) > 1e-4, axis=1))
        ),
        "exact_cd_mean_nnz_1e4": float(
            np.mean(np.sum(np.abs(reference_beta) > 1e-4, axis=1))
        ),
        "reference_projected_kkt_max": float(
            np.max(solved["reference_projected_kkt_max"])
        ),
        "reference_fixed_point_max": float(
            np.max(solved["reference_fixed_point_max"])
        ),
        "reference_sweeps_median": float(np.median(solved["reference_sweeps"])),
        "reference_sweeps_max": int(np.max(solved["reference_sweeps"])),
    }

    np.savez_compressed(
        args.out_dir / "random_penalty_solution_paths.npz",
        x=x,
        y=y,
        effective_lambda_ratios=effective_ratios,
        raw_lambdas=raw_lambdas,
        penalty_index=np.asarray([penalty_index], dtype=np.int64),
        penalty_knots=knots,
        penalty_rho_knots=rho_knots,
        penalty_drho_knots=drho_knots,
        penalty_relu_b=np.asarray([relu_b]),
        penalty_relu_weights=relu_weights,
        model_beta=model_beta.astype(np.float32),
        clipped_model_beta=clipped_beta.astype(np.float32),
        exact_cd_beta=reference_beta.astype(np.float32),
        model_domain_valid=domain_valid,
        actual_data_half=actual_data,
        actual_raw_penalty=actual_penalty,
        actual_objective=actual_objective,
        clipped_data_half=clipped_data,
        clipped_raw_penalty=clipped_penalty,
        clipped_objective=clipped_objective,
        exact_cd_data_half=cd_data,
        exact_cd_raw_penalty=cd_penalty,
        exact_cd_objective=cd_objective,
        actual_valid_loss_curve=actual_curve,
        model_loss_curve=model_curve,
        exact_cd_loss_curve=cd_curve,
        point_correlations=correlations,
        reference_projected_kkt_max=solved["reference_projected_kkt_max"],
        reference_fixed_point_max=solved["reference_fixed_point_max"],
        reference_sweeps=solved["reference_sweeps"],
        reference_selected_start=solved["reference_selected_start"],
    )

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    grid = np.linspace(0.0, bound, 320)
    rho_grid = canonical_penalty_values(grid, knots, relu_b, relu_weights)
    axes[0].plot(grid, rho_grid, color=BLUE, linewidth=2.0)
    axes[0].scatter([0.0], [rho_grid[0]], color=ORANGE, s=28, zorder=3)
    axes[0].set_title(
        f"Random canonical penalty #{penalty_index}\n"
        f"rho(0)={rho_grid[0]:.2e}, d0={d0_scalar:.3g}",
        fontsize=10,
    )
    axes[0].set_xlabel("|beta|", fontsize=9)
    axes[0].set_ylabel("rho(|beta|)", fontsize=9)

    score = np.maximum(
        np.max(np.abs(clipped_beta), axis=0),
        np.max(np.abs(reference_beta), axis=0),
    )
    top = np.argsort(-score)[:12]
    x_axis = np.log10(effective_ratios)
    for line_index, feature in enumerate(top):
        color = FEATURE_COLORS[line_index % len(FEATURE_COLORS)]
        axes[1].plot(
            x_axis,
            clipped_beta[:, feature],
            color=color,
            linewidth=1.35,
            alpha=0.9,
        )
        axes[1].plot(
            x_axis,
            reference_beta[:, feature],
            color=color,
            linewidth=1.0,
            linestyle=":",
            alpha=0.8,
        )
    axes[1].axhline(0.0, color=INK, linewidth=0.6)
    axes[1].set_title(
        "Top-12 coefficient paths\nsolid: Kalbi V3, dotted: exact CD",
        fontsize=10,
    )
    axes[1].set_xlabel("log10(effective lambda ratio)", fontsize=9)
    axes[1].set_ylabel("coefficient", fontsize=9)

    axes[2].plot(
        x_axis,
        cd_curve,
        "o:",
        markersize=3.5,
        color=INK,
        label="bounded exact CD",
    )
    axes[2].plot(
        x_axis,
        model_curve,
        "^-",
        markersize=3.5,
        color=ORANGE,
        label="Kalbi V3",
    )
    if bool(np.isfinite(actual_curve).any()) and bool((~domain_valid).any()):
        axes[2].plot(
            x_axis,
            actual_curve,
            "x",
            markersize=4,
            color=BLUE,
            label="actual Kalbi, domain-valid",
        )
    axes[2].set_title(
        f"Objective path | RMSE={metrics['loss_curve_rmse']:.4f}\n"
        f"mean coefficient correlation={metrics['mean_path_correlation']:.3f}",
        fontsize=10,
    )
    axes[2].set_xlabel("log10(effective lambda ratio)", fontsize=9)
    axes[2].set_ylabel("log1p(F / F0)", fontsize=9)
    axes[2].legend(fontsize=8)
    for axis in axes:
        _style_axis(axis)

    figure.suptitle(
        f"Kalbi V3 random-penalty solution path | checkpoint {step} | "
        f"noisy TabICLv2 N=200, P=600\n"
        f"noise={task_metadata['response_noise_family']}, "
        f"sigma={float(task_metadata['response_noise_sigma']):.3f}, "
        f"max|feature corr|={float(entry['max_abs_feature_corr']):.3f}",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure_path = args.out_dir / (
        f"kalbi_v3_step{step:06d}_noisy_tabicl_n200p600_"
        f"random_penalty_{penalty_index}_solution_path.png"
    )
    figure.savefig(figure_path, dpi=190, bbox_inches="tight")
    plt.close(figure)

    report = {
        "report_version": "kalbi_v3_generated_noisy_tabicl_random_penalty_path_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config": str(args.config.resolve()),
        "task_manifest": str(args.task_manifest.resolve()),
        "task_manifest_sha256": _json_sha256(manifest),
        "penalty_cache": str(args.penalty_cache.resolve()),
        "penalty_cache_metadata": cache_metadata,
        "cd_library": str(args.cd_library.resolve()),
        "cd_library_sha256": file_sha256(args.cd_library),
        "cd_settings": {
            "max_sweeps_per_attempt": int(args.max_sweeps),
            "retries": int(args.retries),
            "coordinate_tolerance": float(args.tolerance),
            "zero_tolerance": float(args.zero_tolerance),
            "independent_fixed_point_ceiling": fixed_point_ceiling,
            "starts": ["high-to-low warm", "Kalbi clipped to domain", "zero"],
        },
        "effective_lambda_ratios": effective_ratios.tolist(),
        "metrics": metrics,
        "figure": str(figure_path.resolve()),
        "arrays": str(
            (args.out_dir / "random_penalty_solution_paths.npz").resolve()
        ),
    }
    report_path = args.out_dir / "random_penalty_solution_path_metrics.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "figure": str(figure_path),
                "metrics": metrics,
                "report": str(report_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
