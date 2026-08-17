#!/usr/bin/env python3
"""CPU-safe architecture and invariance preflight for the V17 selector."""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any

import torch

from .selector_v17_stagegain import (
    DirectPenaltyMLPQuerySelectorV17StageGain,
    V17_SELECTOR_VERSION,
)


Tensor = torch.Tensor


class PreflightError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def standardize_context(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    x_centered = x - x.mean(dim=1, keepdim=True)
    x_scale = x_centered.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
    x_standard = x_centered / x_scale
    y_centered = y - y.mean(dim=1, keepdim=True)
    y_scale = y_centered.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
    return x_standard, y_centered / y_scale


def synthetic_task(
    batch: int,
    rows: int,
    features: int,
    seed: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    x = torch.randn(batch, rows, features, generator=generator)
    latent = torch.randn(batch, rows, 6, generator=generator)
    mixed = min(features, 48)
    if mixed:
        weights = torch.randn(batch, 6, mixed, generator=generator) * 0.35
        x[:, :, :mixed] += torch.einsum("bnl,blp->bnp", latent, weights)
    beta = torch.zeros(batch, features)
    support = min(features, 12)
    beta[:, :support] = torch.linspace(1.4, 0.2, support)[None, :]
    signs = torch.where(
        torch.arange(support) % 2 == 0,
        torch.ones(support),
        -torch.ones(support),
    )
    beta[:, :support] *= signs[None, :]
    noise = 0.35 * torch.randn(batch, rows, generator=generator)
    y = torch.einsum("bnp,bp->bn", x, beta) + noise
    x, y = standardize_context(x, y)
    return x.to(device), y.to(device)


def lambda_max(x: Tensor, y: Tensor) -> Tensor:
    return torch.einsum("bnp,bn->bp", x.float(), y.float()).abs().amax(-1) / float(
        x.shape[1]
    )


def direct_cache(seed: int, rows: int = 2048) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    cache = 0.08 * torch.randn(rows, 96, generator=generator)
    cache[:, 0] += 1.0 / 6.0
    return cache


def maximum_difference(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    keys = (
        "lambda_ratio",
        "penalty_rho",
        "penalty_drho",
        "v17_stage_interaction",
        "v17_task_code",
    )
    return {
        key: float((left[key].detach().float() - right[key].detach().float()).abs().max())
        for key in keys
    }


def require_invariance(name: str, differences: dict[str, float], tolerance: float) -> None:
    worst_name = max(differences, key=differences.get)
    worst = differences[worst_name]
    require(
        math.isfinite(worst) and worst <= float(tolerance),
        f"{name} invariance failed: {worst_name} difference={worst}",
    )


def module_gradient_norm(module: torch.nn.Module) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += parameter.grad.detach().double().square().sum().cpu()
    return float(total.sqrt())


def run_preflight(device: torch.device, seed: int) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    if device.type == "cpu":
        torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    selector = DirectPenaltyMLPQuerySelectorV17StageGain().to(device)
    calibration = selector.fit_penalty_parameter_cache(direct_cache(seed + 1))
    selector.eval()

    x, y = synthetic_task(2, 96, 160, seed + 2, device)
    lmax = lambda_max(x, y)
    started = time.perf_counter()
    base = selector(
        x,
        y,
        lmax,
        return_representation=True,
    )
    base_seconds = time.perf_counter() - started

    require(base["lambdas"].shape == (2, 1), "V17 lambda output shape drift")
    require(
        base["penalty_shape_features"].shape == (2, 1, 196),
        "V17 native196 shape drift",
    )
    require(base["penalty_rho"].shape == (2, 96), "V17 rho shape drift")
    require(
        base["penalty_relu_weights"].shape == (2, 96),
        "V17 ReLU2 weight shape drift",
    )
    base_trajectory = base["v17_trajectory"]
    residual_trajectory = base_trajectory["residual_trajectory"]
    residual_trajectory_raw = base_trajectory["residual_trajectory_raw"]
    beta_residual = base_trajectory["beta_residual"]
    beta_residual_raw = base_trajectory["beta_residual_raw"]
    beta_marginal = base_trajectory["beta_marginal"]
    beta_marginal_raw = base_trajectory["beta_marginal_raw"]
    stage_gain = base_trajectory["stage_gain"]
    stage_gain_rate = float(base_trajectory["stage_gain_rate"])
    residual_scale_before = base_trajectory["residual_scale_before"]
    residual_energy_after = base_trajectory["residual_energy_ratio"]
    stage_explained_fraction = base_trajectory["stage_explained_fraction"]
    stage_active = base_trajectory["stage_active"]
    scale_floor = float(base_trajectory["residual_scale_floor"])
    expected_stage_gain = torch.pow(
        torch.tensor(
            1.0 + stage_gain_rate,
            device=stage_gain.device,
            dtype=stage_gain.dtype,
        ),
        torch.arange(stage_gain.numel(), device=stage_gain.device),
    )
    stage_gain_schedule_error = float(
        (stage_gain - expected_stage_gain).detach().abs().max()
    )
    require(
        stage_gain_schedule_error <= 1e-6,
        "V17 0.5% stage gain schedule failed",
    )
    residual_beta_gain_reconstruction_error = float(
        (beta_residual - beta_residual_raw * stage_gain[None, None, :])
        .detach()
        .abs()
        .max()
    )
    require(
        residual_beta_gain_reconstruction_error <= 2e-6,
        "V17 residual coefficient stage-gain reconstruction failed",
    )
    marginal_beta_gain_reconstruction_error = float(
        (beta_marginal - beta_marginal_raw * stage_gain[None, None, :])
        .detach()
        .abs()
        .max()
    )
    require(
        marginal_beta_gain_reconstruction_error <= 2e-6,
        "V17 marginal coefficient stage-gain reconstruction failed",
    )
    residual_gain_reconstruction_error = float(
        (
            residual_trajectory
            - residual_trajectory_raw * stage_gain[None, None, :]
        )
        .detach()
        .abs()
        .max()
    )
    require(
        residual_gain_reconstruction_error <= 2e-6,
        "V17 residual trajectory stage-gain reconstruction failed",
    )
    residual_energy_before = torch.cat(
        [
            torch.ones_like(residual_energy_after[:, :1]),
            residual_energy_after[:, :-1],
        ],
        dim=-1,
    )
    expected_scale_before = residual_energy_before.sqrt().clamp_min(scale_floor)
    scale_sequence_error = float(
        (residual_scale_before - expected_scale_before)
        .detach()
        .abs()
        .max()
    )
    require(
        scale_sequence_error <= 2e-6,
        "V17 residual scale trajectory is not stage-continuous",
    )
    expected_explained_fraction = (
        (residual_energy_before - residual_energy_after)
        / residual_energy_before.clamp_min(scale_floor**2)
    ).clamp(0.0, 1.0) * stage_active.float()
    explained_fraction_error = float(
        (stage_explained_fraction - expected_explained_fraction)
        .detach()
        .abs()
        .max()
    )
    require(
        explained_fraction_error <= 2e-6,
        "V17 stage explained-fraction reconstruction failed",
    )
    require(
        bool((base["penalty_relu_weights"][:, -1] == 0.0).all()),
        "V17 dead final penalty weight is not exact zero",
    )
    require(
        float(base["penalty_rho6_error"].detach().abs().max()) <= 1e-5,
        "V17 rho(6)=1 contract failed",
    )
    require(
        float(base["penalty_min_drho"].detach().min()) > 0.0,
        "V17 penalty derivative is not positive",
    )
    reconstructed_ratio = (
        base["lambdas"][:, 0]
        * base["penalty_d0"][:, 0]
        / lmax.clamp_min(1e-12)
    )
    require(
        float(
            (reconstructed_ratio - base["lambda_ratio"])
            .detach()
            .abs()
            .max()
        )
        <= 1e-6,
        "V17 effective lambda-ratio contract failed",
    )

    row_generator = torch.Generator(device="cpu").manual_seed(seed + 3)
    row_permutation = torch.randperm(x.shape[1], generator=row_generator).to(device)
    row_output = selector(
        x.index_select(1, row_permutation),
        y.index_select(1, row_permutation),
        lmax,
        return_representation=True,
    )
    row_differences = maximum_difference(base, row_output)
    require_invariance("row permutation", row_differences, 2e-4)

    feature_generator = torch.Generator(device="cpu").manual_seed(seed + 4)
    feature_permutation = torch.randperm(
        x.shape[2], generator=feature_generator
    ).to(device)
    feature_sign = torch.where(
        torch.rand(x.shape[2], generator=feature_generator) >= 0.5,
        torch.ones(x.shape[2]),
        -torch.ones(x.shape[2]),
    ).to(device)
    transformed_x = x.index_select(2, feature_permutation) * feature_sign[None, None, :]
    transformed_lmax = lambda_max(transformed_x, y)
    feature_output = selector(
        transformed_x,
        y,
        transformed_lmax,
        return_representation=True,
    )
    feature_differences = maximum_difference(base, feature_output)
    require_invariance("feature permutation/sign", feature_differences, 3e-4)

    response_sign_output = selector(
        x,
        -y,
        lmax,
        return_representation=True,
    )
    response_sign_differences = maximum_difference(base, response_sign_output)
    require_invariance("global response sign", response_sign_differences, 3e-4)

    response_scale = 3.25
    scaled_lmax = lmax * response_scale
    response_scale_output = selector(
        x,
        y * response_scale,
        scaled_lmax,
        return_representation=True,
    )
    response_scale_differences = maximum_difference(base, response_scale_output)
    require_invariance("response scale representation", response_scale_differences, 3e-4)
    require(
        float(
            (
                response_scale_output["lambdas"]
                / base["lambdas"].clamp_min(1e-12)
                - response_scale
            ).detach().abs().max()
        )
        <= 3e-4,
        "V17 raw lambda did not scale with response",
    )

    small_x, small_y = synthetic_task(1, 37, 23, seed + 5, device)
    small = selector(
        small_x,
        small_y,
        lambda_max(small_x, small_y),
        return_representation=True,
    )
    small_trajectory = small["v17_trajectory"]
    active_stages = int(small_trajectory["stage_active"].sum())
    require(active_stages == math.ceil(23 / 4), "V17 partial-stage padding failed")
    selected = small_trajectory["selected_indices"]
    selected_nonnegative = selected[selected >= 0]
    require(
        selected_nonnegative.numel() == 23,
        "V17 small-P path did not consume each available feature once",
    )
    require(
        int(selected_nonnegative.unique().numel()) == 23,
        "V17 small-P path selected a feature twice",
    )

    masked_x, masked_y = synthetic_task(1, 64, 80, seed + 6, device)
    feature_mask = torch.ones(1, 80, dtype=torch.bool, device=device)
    feature_mask[:, -17:] = False
    masked = selector(
        masked_x,
        masked_y,
        lambda_max(masked_x, masked_y),
        feature_mask=feature_mask,
        return_representation=True,
    )
    masked_indices = masked["v17_trajectory"]["selected_indices"]
    require(
        not bool((masked_indices >= 63).any()),
        "V17 selected a masked feature",
    )

    selector.train()
    selector.zero_grad(set_to_none=True)
    gradient_output = selector(
        x,
        y,
        lmax,
        return_representation=True,
    )
    loss = (
        gradient_output["lambdas"].mean()
        + gradient_output["penalty_rho"].square().mean()
        + 0.1 * gradient_output["penalty_drho"].square().mean()
    )
    loss.backward()
    gradient_modules = {
        "row_query": selector.row_query,
        "row_key": selector.row_key,
        "interaction_proj": selector.interaction_proj,
        "residual_coefficient_proj": selector.residual_coefficient_proj,
        "marginal_coefficient_proj": selector.marginal_coefficient_proj,
        "stage_metadata_proj": selector.stage_metadata_proj,
        "fusion_mlp": selector.fusion_mlp,
        "lambda_head": selector.lambda_head,
        "penalty_head": selector.penalty_parameter_head,
    }
    gradient_norms = {
        name: module_gradient_norm(module)
        for name, module in gradient_modules.items()
    }
    require(
        all(math.isfinite(value) and value > 0.0 for value in gradient_norms.values()),
        f"V17 gradient coverage failed: {gradient_norms}",
    )
    proposal = selector.penalty_parameter_head._last_parameter_proposal
    require(proposal is not None and proposal.grad is not None, "V17 Direct96 gradient missing")
    direct_gradient = proposal.grad.detach().float().reshape(-1, 96)
    active_direct_coordinates = int((direct_gradient.abs().sum(0) > 0.0).sum())
    require(
        active_direct_coordinates == 96,
        f"V17 Direct96 gradient coverage={active_direct_coordinates}/96",
    )

    report = selector.architecture_report()
    require(report["selector_version"] == V17_SELECTOR_VERSION, "V17 version drift")
    return {
        "status": "ok",
        "device": str(device),
        "selector_version": V17_SELECTOR_VERSION,
        "architecture": report,
        "calibration": calibration,
        "base_forward_seconds": base_seconds,
        "base_shapes": {
            "residual_trajectory": list(
                base["v17_trajectory"]["residual_trajectory"].shape
            ),
            "residual_trajectory_raw": list(
                base["v17_trajectory"]["residual_trajectory_raw"].shape
            ),
            "beta_residual": list(base["v17_trajectory"]["beta_residual"].shape),
            "beta_residual_raw": list(
                base["v17_trajectory"]["beta_residual_raw"].shape
            ),
            "beta_marginal": list(base["v17_trajectory"]["beta_marginal"].shape),
            "beta_marginal_raw": list(
                base["v17_trajectory"]["beta_marginal_raw"].shape
            ),
            "residual_scale_before": list(
                base["v17_trajectory"]["residual_scale_before"].shape
            ),
            "stage_interaction": list(base["v17_stage_interaction"].shape),
            "task_code": list(base["v17_task_code"].shape),
            "native_penalty": list(base["penalty_shape_features"].shape),
        },
        "invariance_max_abs": {
            "row_permutation": row_differences,
            "feature_permutation_sign": feature_differences,
            "response_sign": response_sign_differences,
            "response_scale": response_scale_differences,
        },
        "small_p_active_stages": active_stages,
        "all_column_stage_gain": {
            "rate": stage_gain_rate,
            "first_gain": float(stage_gain[0]),
            "terminal_gain": float(stage_gain[-1]),
            "schedule_error": stage_gain_schedule_error,
            "residual_beta_reconstruction_error": (
                residual_beta_gain_reconstruction_error
            ),
            "marginal_beta_reconstruction_error": (
                marginal_beta_gain_reconstruction_error
            ),
            "residual_reconstruction_error": residual_gain_reconstruction_error,
            "stage_sequence_error": scale_sequence_error,
            "explained_fraction_error": explained_fraction_error,
            "active_scale_min": float(
                residual_scale_before[stage_active].detach().min()
            ),
            "active_scale_max": float(
                residual_scale_before[stage_active].detach().max()
            ),
        },
        "gradient_norms": gradient_norms,
        "active_direct96_gradient_coordinates": active_direct_coordinates,
        "rho6_error_max": float(
            base["penalty_rho6_error"].detach().abs().max()
        ),
        "min_drho": float(base["penalty_min_drho"].detach().min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    device = torch.device(args.device)
    result = run_preflight(device, args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
