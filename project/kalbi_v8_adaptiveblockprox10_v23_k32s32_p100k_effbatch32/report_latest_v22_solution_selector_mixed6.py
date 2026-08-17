#!/usr/bin/env python3
"""Evaluate KALBI on a zero-KKT-normalized grid with a raw-lambda x-axis.

This is an evaluation-only iteration-count comparison. Both refinement paths
start from the same learned KALBI warm start and use the same V22 penalty,
raw-lambda grid, and exact gamma=eta*raw_lambda proximal-gradient contract.
The additional branch unrolls that update for a configurable number of steps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
    }
)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from . import model as model_module
from .eval_kalbi_v2_common import (
    encode_dataset,
    file_sha256,
    tensors_from_arrays,
    vector_correlation,
)
from .exact_canonical_cd import CanonicalCDKernel, solve_multistart_path
from .exact_direct96_prox import exact_direct96_prox
from .model import (
    KALBI_BETA_BOUND,
    KALBI_EXACT_PG_FEATURE_CHUNK,
    KALBI_EXACT_PG_STEPS,
    KALBI_RIDGE,
    KALBI_STAGES,
    KALBI_TOPK,
    MODEL_VERSION,
    build_model,
)
from .paper_eval_unseen_dimension5 import (
    load_frozen_task,
    objective_paths,
    prediction_mse,
    split_standardize_three_way,
)
from .selector_v22_v20geometry_scaled import (
    DirectPenaltyMLPQuerySelectorV22V20GeometryScaled,
    V22_SELECTOR_VERSION,
)
from .utils import strip_convexity_flag
from train_fpk import _load_yaml


EVALUATION_VERSION = (
    "kalbi_v22_latest_checkpoint_solution_selector_mixed6_"
    "raw_lambda_long_unroll_v1"
)
BETA_RAW_LAMBDA_MIN = 1e-6
BETA_RAW_LAMBDA_MAX = 10.0
INK = "#202124"
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#666666"
LIGHT_GREY = "#D0D0D0"
FEATURE_COLORS = (
    BLUE,
    ORANGE,
    GREEN,
    RED,
    PURPLE,
    "#56B4E9",
    "#8A6D3B",
    "#4C566A",
)
SEED_STRIDE = 1_000_003


def zero_kkt_raw_lambda_grid(
    zero_kkt_ratios: np.ndarray,
    lambda_max: float,
    d0: float,
) -> tuple[np.ndarray, np.ndarray]:
    ratios = np.asarray(zero_kkt_ratios, dtype=np.float64)
    if (
        ratios.ndim != 1
        or ratios.size == 0
        or not np.isfinite(ratios).all()
        or np.any(ratios <= 0.0)
    ):
        raise ValueError("zero-KKT ratios must be a finite positive vector")
    if not math.isfinite(lambda_max) or lambda_max <= 0.0:
        raise ValueError("lambda_max must be finite and positive")
    if not math.isfinite(d0) or d0 <= 1e-8:
        raise ValueError("penalty d0 must be finite and greater than 1e-8")

    raw_lambdas = ratios * float(lambda_max) / float(d0)
    recovered = raw_lambdas * float(d0) / float(lambda_max)
    relative_error = np.abs(recovered - ratios) / np.maximum(ratios, 1e-15)
    if float(np.max(relative_error)) > 1e-6:
        raise RuntimeError("zero-KKT/raw-lambda round-trip audit failed")
    if (
        float(raw_lambdas.min()) < BETA_RAW_LAMBDA_MIN
        or float(raw_lambdas.max()) > BETA_RAW_LAMBDA_MAX
    ):
        raise RuntimeError(
            "zero-KKT grid leaves beta-solver raw-lambda support: "
            f"[{raw_lambdas.min():.8g}, {raw_lambdas.max():.8g}] "
            f"not within [{BETA_RAW_LAMBDA_MIN}, {BETA_RAW_LAMBDA_MAX}]"
        )
    return raw_lambdas, recovered


def audit_direct_raw_lambda_grid(
    raw_lambdas: np.ndarray,
    lambda_max: float,
    d0: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(raw_lambdas, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.isfinite(values).all()
        or np.any(values <= 0.0)
    ):
        raise ValueError("raw-lambda grid must be a finite positive vector")
    if float(values.min()) < BETA_RAW_LAMBDA_MIN:
        raise RuntimeError("raw-lambda grid falls below beta-solver support")
    if float(values.max()) > BETA_RAW_LAMBDA_MAX:
        raise RuntimeError("raw-lambda grid exceeds beta-solver support")
    if not math.isfinite(lambda_max) or lambda_max <= 0.0:
        raise ValueError("lambda_max must be finite and positive")
    if not math.isfinite(d0) or d0 <= 1e-8:
        raise ValueError("penalty d0 must be finite and greater than 1e-8")
    zero_kkt_ratios = values * float(d0) / float(lambda_max)
    recovered = (
        zero_kkt_ratios * float(lambda_max) / float(d0)
    )
    np.testing.assert_allclose(recovered, values, rtol=1e-12, atol=1e-14)
    return values.copy(), zero_kkt_ratios


def verify_prediction_mse_rows(
    x: np.ndarray,
    y: np.ndarray,
    betas: np.ndarray,
    vectorized: np.ndarray,
) -> None:
    expected = np.asarray(
        [
            np.mean(
                (
                    np.asarray(x, dtype=np.float64)
                    @ np.asarray(beta, dtype=np.float64)
                    - np.asarray(y, dtype=np.float64)
                )
                ** 2
            )
            for beta in betas
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        np.asarray(vectorized, dtype=np.float64),
        expected,
        rtol=1e-10,
        atol=1e-12,
    )


def load_components_v22(
    checkpoint_path: Path,
    config_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], torch.nn.Module, torch.nn.Module]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    metadata = checkpoint.get("metadata", {})
    exact = metadata.get("exact_training_arguments", {})
    expected = {
        "kalbi_isis_topk": KALBI_TOPK,
        "kalbi_isis_stages": KALBI_STAGES,
        "kalbi_isis_ridge": KALBI_RIDGE,
        "selector_topk": 32,
        "selector_stages": 32,
        "selector_public_lambda": (
            "raw_lambda_after_single_v20_effective_ratio_conversion"
        ),
    }
    for key, value in expected.items():
        if exact.get(key) != value:
            raise RuntimeError(
                f"checkpoint contract mismatch for {key}: "
                f"{exact.get(key)!r} != {value!r}"
            )
    if metadata.get("model_version") != MODEL_VERSION:
        raise RuntimeError("checkpoint model version mismatch")
    if metadata.get("selector_version") != V22_SELECTOR_VERSION:
        raise RuntimeError("checkpoint selector version mismatch")

    model = build_model(_load_yaml(str(config_path)), device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    selector = DirectPenaltyMLPQuerySelectorV22V20GeometryScaled(
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


def refine_gamma_eta_once(
    model: torch.nn.Module,
    encoded: dict[str, Any],
    beta: torch.Tensor,
    raw_lambda: torch.Tensor,
    d0: torch.Tensor,
    penalty_shape: torch.Tensor,
    gamma_factor: float,
) -> dict[str, torch.Tensor]:
    """Apply one stable gamma=gamma_factor*eta*raw_lambda step.

    Passing the effective raw-lambda value to the step selector is deliberate:
    makes both its curvature cap and its candidate-direction majorization test
    correspond to the prox used below. The reported objective
    remains the original raw-lambda objective and is computed outside.
    """

    if not math.isfinite(float(gamma_factor)) or float(gamma_factor) <= 0.0:
        raise ValueError("gamma_factor must be finite and positive")
    with torch.autocast(
        device_type=beta.device.type,
        enabled=False,
    ):
        beta0 = beta.float()
        x_centered = encoded["Xc"].float()
        y_centered = encoded["yc"].float()
        sample_count = max(1, int(x_centered.shape[1]))
        prediction0 = torch.matmul(beta0, x_centered.transpose(1, 2))
        residual0 = prediction0 - y_centered[:, None, :]
        data_gradient = (
            torch.matmul(residual0, x_centered) / float(sample_count)
        )
        raw_lambda32 = raw_lambda.float()
        if raw_lambda32.shape != beta0.shape[:2]:
            raise RuntimeError(
                "raw_lambda shape must match beta batch/query dimensions"
            )
        if not bool(
            (
                torch.isfinite(raw_lambda32)
                & (raw_lambda32 > 0.0)
            ).all().item()
        ):
            raise RuntimeError("raw_lambda must be finite and positive")
        effective_lambda = raw_lambda32 * float(gamma_factor)
        with torch.no_grad():
            step = model_module._select_exact_pg_step(
                model,
                encoded,
                beta0,
                residual0,
                data_gradient,
                effective_lambda,
                penalty_shape,
            )
        step_size = step["step_size"]
        gamma = step_size * effective_lambda
        proposal = beta0 - step_size[..., None] * data_gradient
        refined, prox_diagnostics = exact_direct96_prox(
            proposal,
            gamma,
            strip_convexity_flag(penalty_shape).float(),
            model._kalbi_v6_penalty_grid.float(),
            feature_chunk=KALBI_EXACT_PG_FEATURE_CHUNK,
            curvature_margin=model_module.KALBI_EXACT_PG_CURVATURE_MARGIN,
        )
        feature_mask = encoded.get("feature_mask")
        if feature_mask is not None:
            refined = refined * feature_mask[:, None, :].float()
        refined = refined.clamp(-KALBI_BETA_BOUND, KALBI_BETA_BOUND)
        if not bool(torch.isfinite(refined).all().item()):
            raise RuntimeError(
                "gamma=eta*raw_lambda refinement produced nonfinite beta"
            )
        native_d0 = strip_convexity_flag(penalty_shape).float()[..., 1]
        return {
            "beta": refined,
            "step_size": step_size,
            "gamma": gamma,
            "backtracks": step["backtracks"],
            "majorization_ratio": step["majorization_ratio"],
            "minimum_curvature_denominator": prox_diagnostics[
                "minimum_denominator"
            ],
            "zero_threshold": prox_diagnostics["zero_threshold"],
            "change_rms": (refined - beta0).square().mean(-1).sqrt(),
            "active_fraction": (refined != 0.0).float().mean(-1),
            "d0_contract_error": (native_d0 - d0.float()).abs(),
        }


def predict_ablation(
    model: torch.nn.Module,
    encoded: dict[str, Any],
    raw_lambdas: torch.Tensor,
    penalty_shape: torch.Tensor,
    d0: torch.Tensor,
    *,
    query_chunk: int,
    gamma_factor: float,
    ablation_iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    raw_lambdas = raw_lambdas.reshape(-1).float()
    penalty_shape = penalty_shape.reshape(raw_lambdas.numel(), -1).float()
    d0 = d0.reshape(-1).float()
    if int(ablation_iterations) <= 0:
        raise ValueError("ablation_iterations must be positive")
    if int(ablation_iterations) < int(KALBI_EXACT_PG_STEPS):
        raise ValueError(
            "ablation_iterations must cover the native refinement length"
        )
    lambda_max = encoded["lambda_max"].float().reshape(-1)[0]
    beta0_parts: list[torch.Tensor] = []
    normal_parts: list[torch.Tensor] = []
    eta_parts: list[torch.Tensor] = []
    normal_gamma_parts: list[torch.Tensor] = []
    normal_eta_parts: list[torch.Tensor] = []
    eta_gamma_parts: list[torch.Tensor] = []
    eta_step_parts: list[torch.Tensor] = []
    eta_backtrack_parts: list[torch.Tensor] = []
    eta_change_parts: list[torch.Tensor] = []
    manual_native_parts: list[torch.Tensor] = []

    with torch.inference_mode():
        for start in range(0, raw_lambdas.numel(), int(query_chunk)):
            stop = min(start + int(query_chunk), raw_lambdas.numel())
            chunk_lambda = raw_lambdas[start:stop][None]
            chunk_shape = penalty_shape[start:stop][None]
            chunk_d0 = d0[start:stop][None]
            output = model.decode_queries(
                encoded,
                lambdas=chunk_lambda,
                lambda_max=lambda_max.expand(1, stop - start),
                penalty_shape_features=chunk_shape,
                penalty_d0=chunk_d0,
                return_aux=True,
            )
            beta0 = output["kalbi_v6_beta0_q"].float()
            beta_eta = beta0
            beta_at_native_steps: torch.Tensor | None = None
            eta_history: list[dict[str, torch.Tensor]] = []
            for iteration in range(int(ablation_iterations)):
                refinement = refine_gamma_eta_once(
                    model,
                    encoded,
                    beta_eta,
                    chunk_lambda,
                    chunk_d0,
                    chunk_shape,
                    gamma_factor,
                )
                beta_eta = refinement["beta"]
                eta_history.append(refinement)
                if iteration + 1 == int(KALBI_EXACT_PG_STEPS):
                    beta_at_native_steps = beta_eta.clone()
            if beta_at_native_steps is None:
                raise RuntimeError("native-step beta was not captured")

            beta0_parts.append(beta0[0].cpu())
            normal_parts.append(output["beta_q"][0].float().cpu())
            manual_native_parts.append(beta_at_native_steps[0].float().cpu())
            eta_parts.append(beta_eta[0].float().cpu())
            normal_gamma_parts.append(
                output["kalbi_v6_refinement_gamma_path_q"][0].float().cpu()
            )
            normal_eta_parts.append(
                output["kalbi_v6_refinement_step_size_path_q"][0]
                .float()
                .cpu()
            )
            eta_gamma_parts.append(
                torch.stack(
                    [item["gamma"][0] for item in eta_history], dim=-1
                ).float().cpu()
            )
            eta_step_parts.append(
                torch.stack(
                    [item["step_size"][0] for item in eta_history], dim=-1
                ).float().cpu()
            )
            eta_backtrack_parts.append(
                torch.stack(
                    [item["backtracks"][0] for item in eta_history], dim=-1
                ).float().cpu()
            )
            eta_change_parts.append(
                torch.stack(
                    [item["change_rms"][0] for item in eta_history], dim=-1
                ).float().cpu()
            )

    beta0 = torch.cat(beta0_parts).numpy().astype(np.float64)
    normal = torch.cat(normal_parts).numpy().astype(np.float64)
    manual_native = (
        torch.cat(manual_native_parts).numpy().astype(np.float64)
    )
    gamma_eta = torch.cat(eta_parts).numpy().astype(np.float64)
    native_reproduction_max_abs = float(
        np.max(np.abs(manual_native - normal), initial=0.0)
    )
    if math.isclose(
        float(gamma_factor), 1.0, rel_tol=0.0, abs_tol=1e-12
    ) and native_reproduction_max_abs > 2e-6:
        raise RuntimeError(
            "manual raw-lambda PG does not reproduce native T="
            f"{KALBI_EXACT_PG_STEPS}: max_abs="
            f"{native_reproduction_max_abs:.6g}"
        )
    diagnostics = {
        "normal_gamma": torch.cat(normal_gamma_parts).numpy(),
        "normal_step_size": torch.cat(normal_eta_parts).numpy(),
        "gamma_eta_gamma": torch.cat(eta_gamma_parts).numpy(),
        "gamma_eta_step_size": torch.cat(eta_step_parts).numpy(),
        "gamma_eta_backtracks": torch.cat(eta_backtrack_parts).numpy(),
        "gamma_eta_change_rms": torch.cat(eta_change_parts).numpy(),
        "native_reproduction_max_abs": native_reproduction_max_abs,
    }
    return beta0, normal, gamma_eta, diagnostics


def lower_objective_cd_reference(
    *,
    kernel: CanonicalCDKernel,
    x: np.ndarray,
    y: np.ndarray,
    raw_lambdas: np.ndarray,
    knots: np.ndarray,
    relu_b: float,
    relu_weights: np.ndarray,
    normal_beta: np.ndarray,
    gamma_eta_beta: np.ndarray,
    max_sweeps: int,
    retries: int,
    tolerance: float,
    zero_tolerance: float,
) -> dict[str, Any]:
    common = {
        "kernel": kernel,
        "x": x,
        "y": y,
        "raw_lambdas": raw_lambdas,
        "knots": knots,
        "relu_b": relu_b,
        "relu_weights": relu_weights,
        "max_sweeps": max_sweeps,
        "retries": retries,
        "tolerance": tolerance,
        "zero_tolerance": zero_tolerance,
    }
    from_normal = solve_multistart_path(
        **common, model_beta=normal_beta
    )
    from_gamma_eta = solve_multistart_path(
        **common, model_beta=gamma_eta_beta
    )
    choose_eta = (
        from_gamma_eta["reference_objective"]
        < from_normal["reference_objective"]
    )
    reference_beta = np.where(
        choose_eta[:, None],
        from_gamma_eta["reference_beta"],
        from_normal["reference_beta"],
    )
    fixed_point = np.where(
        choose_eta,
        from_gamma_eta["reference_fixed_point_max"],
        from_normal["reference_fixed_point_max"],
    )
    projected_kkt = np.where(
        choose_eta,
        from_gamma_eta["reference_projected_kkt_max"],
        from_normal["reference_projected_kkt_max"],
    )
    reference_objective = np.where(
        choose_eta,
        from_gamma_eta["reference_objective"],
        from_normal["reference_objective"],
    )
    sweeps = np.where(
        choose_eta,
        from_gamma_eta["reference_sweeps"],
        from_normal["reference_sweeps"],
    )
    return {
        "reference_beta": reference_beta,
        "reference_objective": reference_objective,
        "reference_projected_kkt_max": projected_kkt,
        "reference_fixed_point_max": fixed_point,
        "reference_sweeps": sweeps,
        "gamma_eta_start_selected_fraction": float(np.mean(choose_eta)),
    }


def safe_log_objective(
    objective: np.ndarray, zero_objective: float
) -> np.ndarray:
    ratio = np.asarray(objective, dtype=np.float64) / max(
        float(zero_objective), 1e-12
    )
    result = np.full_like(ratio, np.nan)
    valid = np.isfinite(ratio) & (ratio > -1.0)
    result[valid] = np.log1p(ratio[valid])
    return result


def path_metrics(
    beta: np.ndarray,
    reference: np.ndarray,
    objective: np.ndarray,
    reference_objective: np.ndarray,
    zero_kkt_ratios: np.ndarray,
) -> dict[str, float]:
    low = zero_kkt_ratios <= 1e-2
    correlations = np.asarray(
        [
            vector_correlation(candidate, target)
            for candidate, target in zip(beta, reference, strict=True)
        ],
        dtype=np.float64,
    )
    squared_error = (beta - reference) ** 2
    objective_gap = objective - reference_objective
    return {
        "path_correlation": float(np.mean(correlations)),
        "coefficient_rmse": float(np.sqrt(np.mean(squared_error))),
        "objective_gap_mean": float(np.mean(objective_gap)),
        "low_ratio_path_correlation": float(np.mean(correlations[low])),
        "low_ratio_coefficient_rmse": float(
            np.sqrt(np.mean(squared_error[low]))
        ),
        "low_ratio_objective_gap_mean": float(
            np.mean(objective_gap[low])
        ),
        "nnz_at_ratio_1e-4": int(np.sum(np.abs(beta[0]) > 1e-6)),
    }


def heldout_path_diagnostics(
    mse: np.ndarray,
    beta: np.ndarray,
    raw_lambdas: np.ndarray,
    zero_kkt_ratios: np.ndarray,
    zero_predictor_mse: float,
) -> dict[str, float]:
    values = np.asarray(mse, dtype=np.float64)
    oracle_index = int(np.argmin(values))
    minimum = float(values[oracle_index])
    maximum = float(np.max(values))
    return {
        "test_mse_min": minimum,
        "test_mse_max": maximum,
        "relative_test_mse_span": float(
            (maximum - minimum) / max(minimum, 1e-12)
        ),
        "oracle_raw_lambda": float(raw_lambdas[oracle_index]),
        "oracle_zero_kkt_ratio": float(zero_kkt_ratios[oracle_index]),
        "oracle_mse_over_zero": float(
            minimum / max(float(zero_predictor_mse), 1e-12)
        ),
        "oracle_improvement_over_zero": float(
            1.0 - minimum / max(float(zero_predictor_mse), 1e-12)
        ),
        "first_to_last_coefficient_rms": float(
            np.sqrt(np.mean((beta[-1] - beta[0]) ** 2))
        ),
    }


def endpoint_diagnostics(
    beta: np.ndarray,
    objective: np.ndarray,
    zero_objective: float,
) -> dict[str, float | int]:
    endpoint = np.asarray(beta[-1], dtype=np.float64)
    return {
        "nnz": int(np.count_nonzero(np.abs(endpoint) > 1e-6)),
        "max_abs_beta": float(np.max(np.abs(endpoint), initial=0.0)),
        "objective_minus_zero": float(
            np.asarray(objective, dtype=np.float64)[-1]
            - float(zero_objective)
        ),
    }


def style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8, colors=INK, length=3)
    axis.grid(False)


def top_features(reference_beta: np.ndarray, count: int = 6) -> np.ndarray:
    score = np.max(np.abs(reference_beta), axis=0)
    return np.lexsort((np.arange(score.size), -score))[:count]


def draw_task_row(
    axes: np.ndarray,
    task: dict[str, Any],
    *,
    show_titles: bool,
    show_xlabels: bool,
) -> None:
    raw_lambdas = task["raw_lambdas"]
    x_axis = np.log10(raw_lambdas)
    normal = task["normal_beta"]
    gamma_eta = task["gamma_eta_beta"]
    reference = task["reference_beta"]
    for color_index, feature in enumerate(task["top_features"]):
        color = FEATURE_COLORS[color_index % len(FEATURE_COLORS)]
        axes[0].plot(x_axis, normal[:, feature], color=color, lw=1.25)
        axes[0].plot(
            x_axis, gamma_eta[:, feature], color=color, lw=1.05, ls="--"
        )
        axes[0].plot(
            x_axis, reference[:, feature], color=color, lw=0.9, ls=":"
        )
    axes[0].axhline(0.0, color=LIGHT_GREY, lw=0.6)
    axes[0].set_yscale("symlog", linthresh=0.02)
    axes[0].set_ylabel(
        (
            f"Dataset {task['dataset_index']}\n"
            f"N={task['n']:,}, P={task['p']:,}\ncoefficient"
        ),
        fontsize=8.5,
    )

    axes[1].plot(
        x_axis, task["normal_loss"], color=BLUE, lw=1.6
    )
    axes[1].plot(
        x_axis, task["gamma_eta_loss"], color=RED, lw=1.5, ls="--"
    )
    axes[1].plot(
        x_axis, task["reference_loss"], color=INK, lw=1.2, ls=":"
    )
    axes[1].set_ylabel("log1p(F/F0)", fontsize=8.5)

    axes[2].plot(
        x_axis,
        task["normal_mse"] / task["zero_predictor_mse"],
        color=BLUE,
        lw=1.6,
    )
    axes[2].plot(
        x_axis,
        task["gamma_eta_mse"] / task["zero_predictor_mse"],
        color=RED,
        lw=1.5,
        ls="--",
    )
    axes[2].plot(
        x_axis,
        task["reference_mse"] / task["zero_predictor_mse"],
        color=INK,
        lw=1.2,
        ls=":",
    )
    axes[2].axhline(
        1.0, color=GREY, lw=0.8, ls=(0, (4, 3)), alpha=0.8
    )
    axes[2].set_ylabel("test MSE / zero MSE", fontsize=8.5)

    axes[3].plot(
        x_axis,
        np.sum(np.abs(normal) > 1e-6, axis=1),
        color=BLUE,
        lw=1.6,
    )
    axes[3].plot(
        x_axis,
        np.sum(np.abs(gamma_eta) > 1e-6, axis=1),
        color=RED,
        lw=1.5,
        ls="--",
    )
    axes[3].plot(
        x_axis,
        np.sum(np.abs(reference) > 1e-6, axis=1),
        color=INK,
        lw=1.2,
        ls=":",
    )
    axes[3].set_ylabel("nonzero coefficients", fontsize=8.5)

    selected_raw_lambda = task["selected_raw_lambda"]
    if raw_lambdas[0] <= selected_raw_lambda <= raw_lambdas[-1]:
        selected_x = math.log10(selected_raw_lambda)
        for axis in axes:
            axis.axvline(
                selected_x, color=ORANGE, lw=0.9, ls="-.", alpha=0.9
            )
    for axis in axes:
        axis.set_xlim(float(x_axis[0]), float(x_axis[-1]))
        style_axis(axis)
        if show_xlabels:
            axis.set_xlabel(
                r"$\log_{10}(\mathrm{raw}\ \lambda)$", fontsize=8.5
            )
    if show_titles:
        axes[0].set_title(
            f"Coefficient paths\nN={task['n']:,}, P={task['p']:,}",
            fontsize=9.5,
        )
        axes[1].set_title("Training objective", fontsize=9.5)
        axes[2].set_title("Held-out prediction", fontsize=9.5)
        axes[3].set_title("Sparsity path", fontsize=9.5)


def save_detail(output: Path, task: dict[str, Any]) -> None:
    figure, row = plt.subplots(1, 4, figsize=(14.5, 3.4))
    draw_task_row(row, task, show_titles=True, show_xlabels=True)
    handles = [
        Line2D(
            [0], [0], color=BLUE, lw=1.8,
            label=rf"KALBI, $T={KALBI_EXACT_PG_STEPS}$",
        ),
        Line2D(
            [0],
            [0],
            color=RED,
            lw=1.7,
            ls="--",
            label=task["gamma_label"],
        ),
        Line2D(
            [0], [0], color=INK, lw=1.4, ls=":",
            label="Coordinate descent reference",
        ),
        Line2D(
            [0], [0], color=ORANGE, lw=1.1, ls="-.",
            label="KALBI-selected lambda",
        ),
        Line2D(
            [0], [0], color=GREY, lw=0.9, ls=(0, (4, 3)),
            label="Zero-predictor baseline",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.02),
    )
    figure.subplots_adjust(
        left=0.07, right=0.99, top=0.86, bottom=0.24, wspace=0.4
    )
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_composite(output: Path, tasks: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(
        len(tasks), 4, figsize=(14.5, 3.0 * len(tasks)), squeeze=False
    )
    for row_index, task in enumerate(tasks):
        draw_task_row(
            axes[row_index],
            task,
            show_titles=row_index == 0,
            show_xlabels=row_index == len(tasks) - 1,
        )
    handles = [
        Line2D(
            [0], [0], color=BLUE, lw=1.8,
            label=rf"KALBI, $T={KALBI_EXACT_PG_STEPS}$",
        ),
        Line2D(
            [0],
            [0],
            color=RED,
            lw=1.7,
            ls="--",
            label=tasks[0]["gamma_label"],
        ),
        Line2D(
            [0], [0], color=INK, lw=1.4, ls=":",
            label="Coordinate descent reference",
        ),
        Line2D(
            [0], [0], color=ORANGE, lw=1.1, ls="-.",
            label="KALBI-selected lambda",
        ),
        Line2D(
            [0], [0], color=GREY, lw=0.9, ls=(0, (4, 3)),
            label="Zero-predictor baseline",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.99,
        top=0.965,
        bottom=0.075,
        wspace=0.38,
        hspace=0.36,
    )
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_selector_composite(
    output: Path, tasks: list[dict[str, Any]]
) -> None:
    figure, axes = plt.subplots(
        len(tasks),
        2,
        figsize=(11.5, 3.0 * len(tasks) + 1.0),
        squeeze=False,
    )
    for row_index, task in enumerate(tasks):
        x_axis = np.log10(task["raw_lambdas"])
        zero_mse = float(task["zero_predictor_mse"])
        heldout = axes[row_index, 0]
        heldout.plot(
            x_axis,
            task["normal_mse"] / zero_mse,
            color=BLUE,
            lw=1.7,
        )
        heldout.plot(
            x_axis,
            task["gamma_eta_mse"] / zero_mse,
            color=RED,
            lw=1.4,
            ls="--",
        )
        heldout.plot(
            x_axis,
            task["reference_mse"] / zero_mse,
            color=INK,
            lw=1.1,
            ls=":",
        )
        heldout.axhline(
            1.0, color=GREY, lw=0.8, ls=(0, (4, 3)), alpha=0.8
        )
        selected_x = math.log10(task["selected_raw_lambda"])
        heldout.axvline(
            selected_x, color=ORANGE, lw=1.0, ls="-.", alpha=0.9
        )
        heldout.scatter(
            [selected_x],
            [task["selected_native_mse"] / zero_mse],
            marker="D",
            s=28,
            color=ORANGE,
            edgecolor=INK,
            linewidth=0.4,
            zorder=5,
        )
        oracle_index = int(task["native_oracle_index"])
        heldout.scatter(
            [x_axis[oracle_index]],
            [task["normal_mse"][oracle_index] / zero_mse],
            marker="*",
            s=70,
            color="white",
            edgecolor=BLUE,
            linewidth=1.0,
            zorder=5,
        )
        heldout.set_xlim(float(x_axis[0]), float(x_axis[-1]))
        heldout.set_ylabel(
            f"Dataset {task['dataset_index']}\nMSE / zero MSE",
            fontsize=8.5,
        )
        style_axis(heldout)

        penalty = axes[row_index, 1]
        penalty.plot(
            task["penalty_grid"],
            task["penalty_rho"],
            color=GREEN,
            lw=1.7,
        )
        derivative = penalty.twinx()
        derivative.plot(
            task["penalty_grid"],
            task["penalty_drho"],
            color=ORANGE,
            lw=1.3,
            ls="--",
        )
        penalty.set_xlim(
            float(task["penalty_grid"][0]),
            float(task["penalty_grid"][-1]),
        )
        penalty.set_ylabel(r"$\rho(|\beta|)$", fontsize=8.5, color=GREEN)
        derivative.set_ylabel(
            r"$\rho'(|\beta|)$", fontsize=8.5, color=ORANGE
        )
        style_axis(penalty)
        derivative.spines["top"].set_visible(False)
        derivative.tick_params(
            labelsize=8, colors=ORANGE, length=3
        )
        if row_index == 0:
            heldout.set_title("Held-out lambda selection", fontsize=9.5)
            penalty.set_title("KALBI penalty", fontsize=9.5)
        if row_index == len(tasks) - 1:
            heldout.set_xlabel(
                r"$\log_{10}(\mathrm{raw}\ \lambda)$", fontsize=8.5
            )
            penalty.set_xlabel(r"$|\beta|$", fontsize=8.5)

    handles = [
        Line2D(
            [0], [0], color=BLUE, lw=1.8,
            label=rf"KALBI, $T={KALBI_EXACT_PG_STEPS}$",
        ),
        Line2D(
            [0], [0], color=RED, lw=1.6, ls="--",
            label=f"KALBI, T={tasks[0]['ablation_iterations']}",
        ),
        Line2D(
            [0], [0], color=INK, lw=1.3, ls=":",
            label="Coordinate descent reference",
        ),
        Line2D(
            [0], [0], color=ORANGE, lw=1.1, ls="-.",
            marker="D", markersize=4,
            label="KALBI-selected lambda",
        ),
        Line2D(
            [0], [0], color=BLUE, lw=0, marker="*",
            markerfacecolor="white", markersize=8,
            label="KALBI path oracle",
        ),
        Line2D(
            [0], [0], color=GREY, lw=0.9, ls=(0, (4, 3)),
            label="Zero-predictor baseline",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.subplots_adjust(
        left=0.10,
        right=0.91,
        top=0.965,
        bottom=0.22 if len(tasks) == 1 else 0.10,
        wspace=0.42,
        hspace=0.34,
    )
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def evaluate_task(
    *,
    prepared_root: Path,
    entry: dict[str, Any],
    split_seed: int,
    model: torch.nn.Module,
    selector: torch.nn.Module,
    device: torch.device,
    kernel: CanonicalCDKernel,
    raw_lambda_grid: np.ndarray,
    query_chunk: int,
    max_sweeps: int,
    retries: int,
    tolerance: float,
    zero_tolerance: float,
    gamma_factor: float,
    ablation_iterations: int,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    x_raw, y_raw, provenance = load_frozen_task(prepared_root, entry)
    split = split_standardize_three_way(x_raw, y_raw, split_seed)
    x_train = np.asarray(split["x_train"], dtype=np.float32)
    y_train = np.asarray(split["y_train"], dtype=np.float32)
    x_test = np.asarray(split["x_test"], dtype=np.float32)
    y_test = np.asarray(split["y_test"], dtype=np.float32)
    x_device, y_device = tensors_from_arrays(x_train, y_train, device)
    encoded = encode_dataset(model, x_device, y_device)
    lambda_max_tensor = encoded["lambda_max"].float().reshape(-1)
    lambda_max_encoded = float(lambda_max_tensor[0])
    lambda_max = float(
        np.max(
            np.abs(
                np.asarray(x_train, dtype=np.float64).T
                @ np.asarray(y_train, dtype=np.float64)
                / float(x_train.shape[0])
            )
        )
    )
    lambda_max_relative_error = abs(lambda_max_encoded - lambda_max) / max(
        lambda_max, 1e-12
    )
    if lambda_max_relative_error > 1e-5:
        raise RuntimeError(
            "encoded/direct lambda_max mismatch: "
            f"encoded={lambda_max_encoded:.9g}, direct={lambda_max:.9g}"
        )
    with torch.inference_mode():
        selected = selector(
            x_device,
            y_device,
            lambda_max_tensor,
            return_representation=True,
        )
    max_drho = float(selected["penalty_max_drho"].reshape(-1)[0])
    d0_scalar = float(selected["penalty_d0"].reshape(-1)[0])
    raw_lambdas, zero_kkt_ratios = audit_direct_raw_lambda_grid(
        raw_lambda_grid,
        lambda_max,
        d0_scalar,
    )
    recovered_zero_kkt_ratios = (
        raw_lambdas * d0_scalar / lambda_max
    )
    force_ratios = raw_lambdas * max_drho / lambda_max
    selected_raw_lambda = float(selected["raw_lambda"].reshape(-1)[0])
    selected_force_ratio = selected_raw_lambda * max_drho / lambda_max
    selected_zero_kkt_ratio = selected_raw_lambda * d0_scalar / lambda_max
    shape = selected["penalty_shape_features"].reshape(1, -1).expand(
        zero_kkt_ratios.size, -1
    )
    d0 = selected["penalty_d0"].reshape(-1).expand(zero_kkt_ratios.size)
    beta0, normal_beta, gamma_eta_beta, diagnostics = predict_ablation(
        model,
        encoded,
        torch.as_tensor(raw_lambdas, device=device, dtype=torch.float32),
        shape,
        d0,
        query_chunk=query_chunk,
        gamma_factor=gamma_factor,
        ablation_iterations=ablation_iterations,
    )
    knots = (
        selector.penalty_parameter_head.penalty_grid.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    relu_b = float(selected["penalty_relu_b"].reshape(-1)[0])
    relu_weights = (
        selected["penalty_relu_weights"].reshape(-1, knots.size)[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    penalty_rho = (
        selected["penalty_rho"].reshape(-1).detach().cpu().numpy()
        .astype(np.float64)
    )
    penalty_drho = (
        selected["penalty_drho"].reshape(-1).detach().cpu().numpy()
        .astype(np.float64)
    )
    if penalty_rho.shape != knots.shape or penalty_drho.shape != knots.shape:
        raise RuntimeError("selector penalty grid output shape mismatch")
    if not (
        np.isfinite(penalty_rho).all()
        and np.isfinite(penalty_drho).all()
    ):
        raise RuntimeError("selector penalty contains nonfinite values")
    if abs(float(penalty_rho[0])) > 1e-8:
        raise RuntimeError("selector penalty violates rho(0)=0")
    if abs(float(penalty_rho[-1]) - 1.0) > 2e-6:
        raise RuntimeError("selector penalty violates rho(6)=1")
    if float(np.min(penalty_drho)) < -1e-8:
        raise RuntimeError("selector penalty derivative is negative")
    solved = lower_objective_cd_reference(
        kernel=kernel,
        x=x_train,
        y=y_train,
        raw_lambdas=raw_lambdas,
        knots=knots,
        relu_b=relu_b,
        relu_weights=relu_weights,
        normal_beta=normal_beta,
        gamma_eta_beta=gamma_eta_beta,
        max_sweeps=max_sweeps,
        retries=retries,
        tolerance=tolerance,
        zero_tolerance=zero_tolerance,
    )
    reference_beta = solved["reference_beta"]
    if float(np.max(solved["reference_fixed_point_max"])) > max(
        2e-3, 5.0 * tolerance
    ):
        raise RuntimeError("coordinate-descent fixed-point audit failed")
    if float(np.max(solved["reference_projected_kkt_max"])) > max(
        2e-3, 5.0 * tolerance
    ):
        raise RuntimeError("coordinate-descent projected-KKT audit failed")

    objective = {}
    loss = {}
    mse = {}
    zero_objective = 0.5 * float(np.mean(y_train.astype(np.float64) ** 2))
    for name, beta in (
        ("beta0", beta0),
        ("normal", normal_beta),
        ("gamma_eta", gamma_eta_beta),
        ("reference", reference_beta),
    ):
        _, _, objective[name] = objective_paths(
            x_train,
            y_train,
            beta,
            raw_lambdas,
            knots,
            relu_b,
            relu_weights,
        )
        loss[name] = safe_log_objective(objective[name], zero_objective)
        mse[name] = prediction_mse(x_test, y_test, beta)
        verify_prediction_mse_rows(x_test, y_test, beta, mse[name])

    with torch.inference_mode():
        selected_native_beta = model.decode_queries(
            encoded,
            lambdas=torch.as_tensor(
                [[selected_raw_lambda]],
                device=device,
                dtype=torch.float32,
            ),
            lambda_max=lambda_max_tensor[0].expand(1, 1),
            penalty_shape_features=selected[
                "penalty_shape_features"
            ].reshape(1, 1, -1),
            penalty_d0=selected["penalty_d0"].reshape(1, 1),
        )[0, 0].float().cpu().numpy().astype(np.float64)
    selected_native_mse = float(
        prediction_mse(
            x_test,
            y_test,
            selected_native_beta.reshape(1, -1),
        )[0]
    )
    native_minimum = float(np.min(mse["normal"]))
    native_ties = np.flatnonzero(
        np.isclose(
            mse["normal"],
            native_minimum,
            rtol=1e-12,
            atol=1e-14,
        )
    )
    native_oracle_index = int(
        native_ties[np.argmax(raw_lambdas[native_ties])]
    )
    native_oracle_mse = float(mse["normal"][native_oracle_index])
    selector_mse_ratio = selected_native_mse / max(
        native_oracle_mse, 1e-12
    )

    zero_predictor_mse = float(
        np.mean(np.asarray(y_test, dtype=np.float64) ** 2)
    )

    normal_metrics = path_metrics(
        normal_beta,
        reference_beta,
        objective["normal"],
        objective["reference"],
        zero_kkt_ratios,
    )
    gamma_eta_metrics = path_metrics(
        gamma_eta_beta,
        reference_beta,
        objective["gamma_eta"],
        objective["reference"],
        zero_kkt_ratios,
    )
    normal_metrics["objective_improvement_over_beta0_fraction"] = float(
        np.mean(objective["normal"] <= objective["beta0"] + 1e-10)
    )
    gamma_eta_metrics["objective_improvement_over_beta0_fraction"] = float(
        np.mean(objective["gamma_eta"] <= objective["beta0"] + 1e-10)
    )
    normal_metrics["heldout"] = heldout_path_diagnostics(
        mse["normal"],
        normal_beta,
        raw_lambdas,
        zero_kkt_ratios,
        zero_predictor_mse,
    )
    gamma_eta_metrics["heldout"] = heldout_path_diagnostics(
        mse["gamma_eta"],
        gamma_eta_beta,
        raw_lambdas,
        zero_kkt_ratios,
        zero_predictor_mse,
    )
    reference_metrics = {
        "heldout": heldout_path_diagnostics(
            mse["reference"],
            reference_beta,
            raw_lambdas,
            zero_kkt_ratios,
            zero_predictor_mse,
        ),
        "endpoint": endpoint_diagnostics(
            reference_beta,
            objective["reference"],
            zero_objective,
        ),
    }
    normal_metrics["endpoint"] = endpoint_diagnostics(
        normal_beta,
        objective["normal"],
        zero_objective,
    )
    gamma_eta_metrics["endpoint"] = endpoint_diagnostics(
        gamma_eta_beta,
        objective["gamma_eta"],
        zero_objective,
    )
    endpoint_zero_kkt_residual = max(
        lambda_max - float(raw_lambdas[-1]) * d0_scalar,
        0.0,
    )
    metric = {
        "dataset_index": int(entry["dataset_index"]),
        "dataset_id": str(entry["dataset_id"]),
        "n": int(entry["n"]),
        "p": int(entry["p"]),
        "task_sha256": str(entry["task_sha256"]),
        "split_seed": int(split_seed),
        "lambda_max": lambda_max,
        "lambda_max_encoded": lambda_max_encoded,
        "lambda_max_relative_error": lambda_max_relative_error,
        "penalty_max_drho": max_drho,
        "penalty_d0": d0_scalar,
        "selected_raw_lambda": selected_raw_lambda,
        "selected_force_ratio": selected_force_ratio,
        "selected_zero_kkt_ratio": selected_zero_kkt_ratio,
        "raw_lambda_gamma_factor": float(gamma_factor),
        "ablation_iterations": int(ablation_iterations),
        "raw_lambda_min": float(raw_lambdas.min()),
        "raw_lambda_max": float(raw_lambdas.max()),
        "zero_kkt_ratio_min": float(zero_kkt_ratios.min()),
        "zero_kkt_ratio_max": float(zero_kkt_ratios.max()),
        "force_ratio_min": float(force_ratios.min()),
        "force_ratio_max": float(force_ratios.max()),
        "zero_predictor_test_mse": zero_predictor_mse,
        "endpoint_zero_kkt_residual": endpoint_zero_kkt_residual,
        "endpoint_zero_kkt_feasible": bool(
            endpoint_zero_kkt_residual <= max(1e-10, 1e-8 * lambda_max)
        ),
        "normal": normal_metrics,
        "gamma_eta": gamma_eta_metrics,
        "reference": reference_metrics,
        "selector": {
            "selected_raw_lambda": selected_raw_lambda,
            "selected_zero_kkt_ratio": selected_zero_kkt_ratio,
            "selected_test_mse": selected_native_mse,
            "oracle_raw_lambda": float(
                raw_lambdas[native_oracle_index]
            ),
            "oracle_zero_kkt_ratio": float(
                zero_kkt_ratios[native_oracle_index]
            ),
            "oracle_test_mse": native_oracle_mse,
            "selected_over_oracle_mse": selector_mse_ratio,
            "rho_at_zero": float(penalty_rho[0]),
            "rho_at_six": float(penalty_rho[-1]),
            "d0": float(penalty_drho[0]),
            "max_drho": float(np.max(penalty_drho)),
            "derivative_total_variation": float(
                np.sum(np.abs(np.diff(penalty_drho)))
            ),
            "concavity_violation": bool(
                np.any(np.diff(penalty_drho) > 1e-8)
            ),
        },
        "diagnostics": {
            "normal_eta_median": float(
                np.median(diagnostics["normal_step_size"])
            ),
            "normal_gamma_median": float(
                np.median(diagnostics["normal_gamma"])
            ),
            "gamma_eta_eta_median": float(
                np.median(diagnostics["gamma_eta_step_size"])
            ),
            "gamma_eta_gamma_median": float(
                np.median(diagnostics["gamma_eta_gamma"])
            ),
            "gamma_eta_backtrack_fraction": float(
                np.mean(diagnostics["gamma_eta_backtracks"] > 0)
            ),
            "gamma_eta_last_change_rms_median": float(
                np.median(diagnostics["gamma_eta_change_rms"][:, -1])
            ),
            "native_t10_reproduction_max_abs": float(
                diagnostics["native_reproduction_max_abs"]
            ),
            "cd_max_fixed_point": float(
                np.max(solved["reference_fixed_point_max"])
            ),
            "cd_max_projected_kkt": float(
                np.max(solved["reference_projected_kkt_max"])
            ),
            "cd_median_sweeps": float(
                np.median(solved["reference_sweeps"])
            ),
            "cd_gamma_eta_start_selected_fraction": float(
                solved["gamma_eta_start_selected_fraction"]
            ),
        },
        "provenance": {
            "data_path": provenance["data_path"],
            "manifest_path": provenance["manifest_path"],
        },
    }

    task_data = {
        "dataset_index": int(entry["dataset_index"]),
        "n": int(entry["n"]),
        "p": int(entry["p"]),
        "raw_lambdas": raw_lambdas,
        "zero_kkt_ratios": zero_kkt_ratios,
        "selected_raw_lambda": selected_raw_lambda,
        "selected_native_mse": selected_native_mse,
        "native_oracle_index": native_oracle_index,
        "zero_predictor_mse": zero_predictor_mse,
        "ablation_iterations": int(ablation_iterations),
        "penalty_grid": knots,
        "penalty_rho": penalty_rho,
        "penalty_drho": penalty_drho,
        "gamma_label": (
            rf"KALBI, $T={int(ablation_iterations)}$"
        ),
        "normal_beta": normal_beta,
        "gamma_eta_beta": gamma_eta_beta,
        "reference_beta": reference_beta,
        "normal_loss": loss["normal"],
        "gamma_eta_loss": loss["gamma_eta"],
        "reference_loss": loss["reference"],
        "normal_mse": mse["normal"],
        "gamma_eta_mse": mse["gamma_eta"],
        "reference_mse": mse["reference"],
        "top_features": top_features(reference_beta),
    }
    task_dir = output_dir / f"dataset_{int(entry['dataset_index']):02d}"
    task_dir.mkdir(parents=True)
    np.savez_compressed(
        task_dir / "paths.npz",
        zero_kkt_ratios=zero_kkt_ratios,
        recovered_zero_kkt_ratios=recovered_zero_kkt_ratios,
        force_ratios=force_ratios,
        raw_lambdas=raw_lambdas,
        selected_raw_lambda=np.asarray(selected_raw_lambda),
        selected_zero_kkt_ratio=np.asarray(selected_zero_kkt_ratio),
        selected_force_ratio=np.asarray(selected_force_ratio),
        selected_native_beta=selected_native_beta.astype(np.float32),
        selected_native_test_mse=np.asarray(selected_native_mse),
        native_oracle_index=np.asarray(native_oracle_index),
        native_oracle_test_mse=np.asarray(native_oracle_mse),
        penalty_grid=knots,
        penalty_rho=penalty_rho,
        penalty_drho=penalty_drho,
        zero_predictor_test_mse=np.asarray(zero_predictor_mse),
        beta0=beta0.astype(np.float32),
        gamma_eta_lambda_beta=normal_beta.astype(np.float32),
        gamma_eta_beta=gamma_eta_beta.astype(np.float32),
        coordinate_descent_beta=reference_beta.astype(np.float32),
        gamma_eta_lambda_objective=objective["normal"],
        gamma_eta_objective=objective["gamma_eta"],
        coordinate_descent_objective=objective["reference"],
        gamma_eta_lambda_test_mse=mse["normal"],
        gamma_eta_test_mse=mse["gamma_eta"],
        coordinate_descent_test_mse=mse["reference"],
        coordinate_descent_projected_kkt_max=solved[
            "reference_projected_kkt_max"
        ],
        coordinate_descent_fixed_point_max=solved[
            "reference_fixed_point_max"
        ],
    )
    (task_dir / "metrics.json").write_text(
        json.dumps(metric, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_detail(task_dir / "detail", task_data)
    return metric, task_data


def write_metrics_csv(path: Path, metrics: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for item in metrics:
        row: dict[str, Any] = {
            "dataset_id": item["dataset_id"],
            "n": item["n"],
            "p": item["p"],
            "lambda_max": item["lambda_max"],
            "d0": item["penalty_d0"],
            "max_drho": item["penalty_max_drho"],
            "raw_lambda_min": item["raw_lambda_min"],
            "raw_lambda_max": item["raw_lambda_max"],
            "selected_raw_lambda": item["selected_raw_lambda"],
            "selected_zero_kkt_ratio": item["selected_zero_kkt_ratio"],
            "selected_force_ratio": item["selected_force_ratio"],
            "selector_selected_test_mse": item["selector"][
                "selected_test_mse"
            ],
            "selector_oracle_raw_lambda": item["selector"][
                "oracle_raw_lambda"
            ],
            "selector_oracle_test_mse": item["selector"][
                "oracle_test_mse"
            ],
            "selector_selected_over_oracle_mse": item["selector"][
                "selected_over_oracle_mse"
            ],
            "zero_predictor_test_mse": item["zero_predictor_test_mse"],
            "endpoint_zero_kkt_feasible": item[
                "endpoint_zero_kkt_feasible"
            ],
            "cd_max_projected_kkt": item["diagnostics"][
                "cd_max_projected_kkt"
            ],
            "cd_max_fixed_point": item["diagnostics"][
                "cd_max_fixed_point"
            ],
        }
        for mode in ("normal", "gamma_eta", "reference"):
            heldout = item[mode]["heldout"]
            endpoint = item[mode]["endpoint"]
            for key, value in heldout.items():
                row[f"{mode}_{key}"] = value
            for key, value in endpoint.items():
                row[f"{mode}_endpoint_{key}"] = value
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cd-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--gamma-factor", type=float, default=1.0)
    parser.add_argument("--ablation-iterations", type=int, default=100)
    parser.add_argument("--query-chunk", type=int, default=8)
    parser.add_argument("--split-seed", type=int, default=2_026_072_407)
    parser.add_argument("--max-sweeps", type=int, default=2000)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--zero-tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    if not math.isfinite(args.gamma_factor) or args.gamma_factor <= 0.0:
        parser.error("--gamma-factor must be finite and positive")
    if int(args.ablation_iterations) <= 0:
        parser.error("--ablation-iterations must be positive")

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    prepared_root = args.prepared_root.resolve()
    frozen = json.loads(
        (
            prepared_root
            / "reproducibility"
            / "reproducibility_manifest.json"
        ).read_text(encoding="utf-8")
    )
    if (
        frozen.get("dimension_regime")
        != "mixed6_fixed_n500p100_n200p400"
    ):
        raise RuntimeError("evaluation requires the frozen mixed-six task set")
    entries = list(frozen["accepted_tasks"])[: int(args.limit)]
    if len(entries) != int(args.limit):
        raise RuntimeError("not enough frozen mixed-six tasks")
    for entry in entries:
        dataset_index = int(entry["dataset_index"])
        expected_shape = (
            (500, 100) if dataset_index <= 3 else (200, 400)
        )
        if (int(entry["n"]), int(entry["p"])) != expected_shape:
            raise RuntimeError(f"mixed-six task contract failed: {entry}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint, model, selector = load_components_v22(
        args.checkpoint.resolve(), args.config.resolve(), device
    )
    kernel = CanonicalCDKernel(args.cd_library.resolve())
    raw_lambda_grid = np.logspace(
        -4.0, 0.0, int(args.grid_size), dtype=np.float64
    )
    metrics: list[dict[str, Any]] = []
    plot_data: list[dict[str, Any]] = []
    for entry in entries:
        dataset_index = int(entry["dataset_index"])
        print(
            json.dumps(
                {
                    "status": "starting",
                    "dataset": dataset_index,
                    "n": int(entry["n"]),
                    "p": int(entry["p"]),
                }
            ),
            flush=True,
        )
        metric, task_data = evaluate_task(
            prepared_root=prepared_root,
            entry=entry,
            split_seed=int(args.split_seed)
            + SEED_STRIDE * (dataset_index - 1),
            model=model,
            selector=selector,
            device=device,
            kernel=kernel,
            raw_lambda_grid=raw_lambda_grid,
            query_chunk=int(args.query_chunk),
            max_sweeps=int(args.max_sweeps),
            retries=int(args.retries),
            tolerance=float(args.tolerance),
            zero_tolerance=float(args.zero_tolerance),
            gamma_factor=float(args.gamma_factor),
            ablation_iterations=int(args.ablation_iterations),
            output_dir=output_dir,
        )
        metrics.append(metric)
        plot_data.append(task_data)
        print(json.dumps(metric, sort_keys=True), flush=True)

    factor_tag = str(args.gamma_factor).replace(".", "p")
    iteration_tag = int(args.ablation_iterations)
    composite = (
        output_dir
        / (
            f"zero_kkt_raw_lambda_factor_{factor_tag}_"
            f"iter{iteration_tag}_mixed6_composite"
        )
    )
    save_composite(composite, plot_data)
    selector_composite = (
        output_dir
        / (
            f"v22_lambda_penalty_selector_step"
            f"{int(checkpoint['completed_steps']):06d}_mixed6"
        )
    )
    save_selector_composite(selector_composite, plot_data)
    write_metrics_csv(output_dir / "metrics.csv", metrics)
    aggregate = {
        mode: {
            key: float(np.mean([item[mode][key] for item in metrics]))
            for key in (
                "path_correlation",
                "coefficient_rmse",
                "objective_gap_mean",
                "low_ratio_path_correlation",
                "low_ratio_coefficient_rmse",
                "low_ratio_objective_gap_mean",
                "nnz_at_ratio_1e-4",
                "objective_improvement_over_beta0_fraction",
            )
        }
        for mode in ("normal", "gamma_eta")
    }
    penalty_stack = np.stack(
        [np.asarray(item["penalty_rho"], dtype=np.float64) for item in plot_data]
    )
    pairwise_penalty_rms = [
        float(np.sqrt(np.mean((penalty_stack[left] - penalty_stack[right]) ** 2)))
        for left in range(penalty_stack.shape[0])
        for right in range(left + 1, penalty_stack.shape[0])
    ]
    selector_summary = {
        "mean_selected_over_oracle_mse": float(
            np.mean(
                [
                    item["selector"]["selected_over_oracle_mse"]
                    for item in metrics
                ]
            )
        ),
        "max_selected_over_oracle_mse": float(
            np.max(
                [
                    item["selector"]["selected_over_oracle_mse"]
                    for item in metrics
                ]
            )
        ),
        "selected_raw_lambda_min": float(
            np.min([item["selected_raw_lambda"] for item in metrics])
        ),
        "selected_raw_lambda_max": float(
            np.max([item["selected_raw_lambda"] for item in metrics])
        ),
        "penalty_pairwise_rho_rms_mean": float(
            np.mean(pairwise_penalty_rms)
            if pairwise_penalty_rms
            else 0.0
        ),
        "penalty_pairwise_rho_rms_min": float(
            np.min(pairwise_penalty_rms)
            if pairwise_penalty_rms
            else 0.0
        ),
        "penalty_pair_count": int(len(pairwise_penalty_rms)),
    }
    report = {
        "evaluation_version": EVALUATION_VERSION,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint.resolve()),
        "checkpoint_step": int(checkpoint["completed_steps"]),
        "model_version": MODEL_VERSION,
        "selector_version": V22_SELECTOR_VERSION,
        "comparison": {
            "normal": "gamma=eta*raw_lambda",
            "ablation": (
                f"gamma={float(args.gamma_factor):g}*eta*raw_lambda "
                "with matched raw-lambda curvature safety and "
                f"{int(args.ablation_iterations)} iterations"
            ),
            "raw_lambda_gamma_factor": float(args.gamma_factor),
            "ablation_iterations": int(args.ablation_iterations),
            "normal_iterations": int(KALBI_EXACT_PG_STEPS),
            "shared": (
                "same frozen tasks, train-only preprocessing, V22 penalty, "
                "common raw-lambda grid [1e-4, 1], learned beta warm "
                "start, and exact gamma=eta*raw_lambda mechanics; normal "
                "KALBI uses 10 iterations and the long-unroll branch uses "
                f"{int(args.ablation_iterations)} iterations"
            ),
            "x_axis": "log10(raw lambda)",
            "raw_lambda_grid": [
                float(raw_lambda_grid[0]),
                float(raw_lambda_grid[-1]),
            ],
            "grid_size": int(raw_lambda_grid.size),
        },
        "aggregate": aggregate,
        "selector_summary": selector_summary,
        "tasks": metrics,
        "metrics_csv": str(output_dir / "metrics.csv"),
        "composite_png": str(composite.with_suffix(".png")),
        "composite_pdf": str(composite.with_suffix(".pdf")),
        "selector_composite_png": str(
            selector_composite.with_suffix(".png")
        ),
        "selector_composite_pdf": str(
            selector_composite.with_suffix(".pdf")
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
