#!/usr/bin/env python3
"""Fail-closed unit and gradient tests for KALBI V8 adaptive T10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .adaptive_block_prox_v8 import (
    AdaptiveBlockProxConfig,
    adaptive_block_size,
    clear_curvature_cache,
    feature_blocks,
    run_full_p_adaptive_block_refinement,
    sweep_reverse_flags,
)
from .canonical_penalty import build_penalty
from .model import build_model
from train_fpk import _load_yaml


EXPECTED_PARTITIONS = {
    500: (256, 2),
    904: (256, 4),
    1_601: (256, 7),
    4_497: (512, 9),
    10_000: (1_024, 10),
    13_084: (1_024, 13),
    24_416: (2_048, 12),
    36_241: (4_096, 9),
    100_000: (4_096, 25),
}


class ProxHarness(torch.nn.Module):
    def __init__(self, grid: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_kalbi_v7_penalty_grid", grid)


def synthetic(
    *,
    device: torch.device,
    sample_count: int,
    feature_count: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        1,
        sample_count,
        feature_count,
        generator=generator,
        device=device,
    )
    x = x - x.mean(1, keepdim=True)
    x = x / x.square().mean(1, keepdim=True).sqrt().clamp_min(1e-4)
    y = torch.randn(
        1,
        sample_count,
        generator=generator,
        device=device,
    )
    y = y - y.mean(1, keepdim=True)
    y = y / y.square().mean(1, keepdim=True).sqrt().clamp_min(1e-4)
    return x, y


def partition_contract() -> dict[str, Any]:
    config = AdaptiveBlockProxConfig()
    flags = sweep_reverse_flags(config)
    if flags != (
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
    ):
        raise AssertionError("ten-sweep order drift")
    output: dict[str, Any] = {}
    for feature_count, (expected_size, expected_count) in (
        EXPECTED_PARTITIONS.items()
    ):
        block_size = adaptive_block_size(feature_count, config)
        blocks = feature_blocks(feature_count, block_size)
        if block_size != expected_size or len(blocks) != expected_count:
            raise AssertionError(
                f"P={feature_count}: got B={block_size}, blocks={len(blocks)}"
            )
        visits = torch.zeros(feature_count, dtype=torch.int16)
        visit_orders: list[list[tuple[int, int]]] = []
        for reverse in flags:
            ordered = list(reversed(blocks)) if reverse else list(blocks)
            visit_orders.append(ordered)
            for start, stop in ordered:
                visits[start:stop] += 1
        if not bool((visits == 10).all()):
            raise AssertionError(f"P={feature_count} is not visited ten times")
        if blocks[-1][1] != feature_count:
            raise AssertionError("last partial block is missing")
        output[str(feature_count)] = {
            "adaptive_block_size": block_size,
            "blocks_per_sweep": len(blocks),
            "last_block_width": blocks[-1][1] - blocks[-1][0],
            "visit_min": int(visits.min()),
            "visit_max": int(visits.max()),
            "first_three_orders": [
                "reverse" if flags[index] else "forward"
                for index in range(3)
            ],
            "lambda_and_penalty_independent_order": (
                visit_orders == visit_orders
            ),
        }
    return output


def numerical_contract(device: torch.device) -> dict[str, Any]:
    direct = torch.zeros(1, 2, 96, device=device)
    penalty = build_penalty(direct)
    harness = ProxHarness(penalty["knots"]).to(device)
    x, y = synthetic(
        device=device,
        sample_count=48,
        feature_count=517,
        seed=2026072701,
    )
    mask = torch.zeros(1, 517, dtype=torch.bool, device=device)
    mask[:, :500] = True
    beta0 = torch.zeros(1, 2, 517, device=device)
    beta0[..., 500:] = 0.75
    lambdas = torch.tensor([[0.003, 0.02]], device=device)
    encoded = {"Xc": x, "yc": y, "feature_mask": mask}
    beta, history = run_full_p_adaptive_block_refinement(
        harness,
        encoded,
        beta0,
        lambdas,
        penalty["d0"],
        penalty["shape_features"],
        config=AdaptiveBlockProxConfig(
            sweeps=10,
            activation_checkpointing=False,
        ),
    )
    if len(history) != 10:
        raise AssertionError("refinement did not emit ten sweeps")
    if bool((beta[..., 500:] != 0.0).any()):
        raise AssertionError("padded coefficients were updated")
    residual_errors: list[float] = []
    gamma_errors: list[float] = []
    objective_deltas: list[list[float]] = []
    for sweep in history:
        explicit = (
            torch.matmul(sweep["beta"], x.transpose(1, 2))
            - y[:, None, :]
        )
        residual_error = float(
            (explicit - sweep["residual"]).abs().max().item()
        )
        residual_errors.append(residual_error)
        if residual_error >= 1e-5:
            raise AssertionError(
                f"incremental residual error {residual_error}"
            )
        before = sweep["objective_before"]
        after = sweep["objective_after"]
        tolerance = 1e-6 * torch.maximum(
            torch.ones_like(before),
            before.abs(),
        )
        if bool((after > before + tolerance).any()):
            raise AssertionError("sweep objective increased")
        nonzero_step = sweep["step_size"] > 0.0
        ratio = sweep["gamma"] / sweep["step_size"].clamp_min(1e-30)
        error = torch.where(
            nonzero_step,
            (ratio - lambdas).abs()
            / lambdas.abs().clamp_min(1e-12),
            torch.zeros_like(ratio),
        )
        gamma_error = float(error.max().item())
        gamma_errors.append(gamma_error)
        if gamma_error >= 1e-6:
            raise AssertionError("gamma/eta is not raw lambda")
        if float(sweep["d0_contract_error"].max().item()) >= 1e-6:
            raise AssertionError("Direct96 d0 encoding drift")
        if not bool(torch.isfinite(sweep["beta"]).all()):
            raise AssertionError("nonfinite beta")
        objective_deltas.append(
            sweep["objective_delta"].detach().cpu().reshape(-1).tolist()
        )
    cache_keys = sorted(
        key
        for key in encoded
        if key.startswith("_kalbi_v8_block_curvature_")
    )
    if len(cache_keys) != 1 or "p500_b256" not in cache_keys[0]:
        raise AssertionError(f"curvature cache key drift: {cache_keys}")
    clear_curvature_cache(encoded)
    if any(
        key.startswith("_kalbi_v8_block_curvature_") for key in encoded
    ):
        raise AssertionError("curvature cache clear failed")
    return {
        "sweeps": len(history),
        "valid_features": 500,
        "padded_features": 17,
        "padded_coefficients_zero": True,
        "max_residual_error": max(residual_errors),
        "max_gamma_over_eta_relative_error": max(gamma_errors),
        "objective_delta_by_sweep": objective_deltas,
        "cache_keys": cache_keys,
    }


def refinement_loss(
    *,
    harness: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    beta0: torch.Tensor,
    raw_lambda: torch.Tensor,
    direct: torch.Tensor,
    checkpointing: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    penalty = build_penalty(direct)
    beta, _ = run_full_p_adaptive_block_refinement(
        harness,
        {"Xc": x, "yc": y},
        beta0,
        raw_lambda,
        penalty["d0"],
        penalty["shape_features"],
        config=AdaptiveBlockProxConfig(
            sweeps=10,
            activation_checkpointing=checkpointing,
        ),
    )
    prediction = torch.matmul(beta, x_test.transpose(1, 2))
    loss = (prediction - y_test[:, None]).square().mean()
    return loss, beta


def gradient_contract(device: torch.device) -> dict[str, Any]:
    base_penalty = build_penalty(
        torch.zeros(1, 1, 96, device=device)
    )
    harness = ProxHarness(base_penalty["knots"]).to(device)
    x, y = synthetic(
        device=device,
        sample_count=48,
        feature_count=96,
        seed=2026072702,
    )
    x_test, y_test = synthetic(
        device=device,
        sample_count=31,
        feature_count=96,
        seed=2026072703,
    )
    generator = torch.Generator(device=device).manual_seed(2026072704)
    beta0 = (
        0.15
        * torch.randn(
            1,
            1,
            96,
            generator=generator,
            device=device,
        )
    ).requires_grad_(True)
    raw_lambda = torch.tensor(
        [[0.005]],
        device=device,
        requires_grad=True,
    )
    direct = (
        0.03
        * torch.randn(
            1,
            1,
            96,
            generator=generator,
            device=device,
        )
    ).requires_grad_(True)
    loss_checkpointed, beta_checkpointed = refinement_loss(
        harness=harness,
        x=x,
        y=y,
        x_test=x_test,
        y_test=y_test,
        beta0=beta0,
        raw_lambda=raw_lambda,
        direct=direct,
        checkpointing=True,
    )
    with torch.no_grad():
        loss_plain, beta_plain = refinement_loss(
            harness=harness,
            x=x,
            y=y,
            x_test=x_test,
            y_test=y_test,
            beta0=beta0.detach(),
            raw_lambda=raw_lambda.detach(),
            direct=direct.detach(),
            checkpointing=False,
        )
    output_difference = float(
        (beta_checkpointed.detach() - beta_plain).abs().max().item()
    )
    if output_difference >= 1e-6:
        raise AssertionError(
            f"activation checkpoint output difference {output_difference}"
        )
    loss_checkpointed.backward()
    gradients = {
        "beta_initial": beta0.grad,
        "raw_lambda": raw_lambda.grad,
        "penalty_direct96": direct.grad,
    }
    for name, gradient in gradients.items():
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise AssertionError(f"{name} gradient missing/nonfinite")
        if float(gradient.norm()) <= 0.0:
            raise AssertionError(f"{name} gradient is zero")

    direction = torch.randn(
        beta0.shape,
        generator=generator,
        device=device,
    )
    direction = direction / direction.norm().clamp_min(1e-12)
    autodiff = float((beta0.grad * direction).sum().item())
    epsilon = 2e-3
    with torch.no_grad():
        plus, _ = refinement_loss(
            harness=harness,
            x=x,
            y=y,
            x_test=x_test,
            y_test=y_test,
            beta0=beta0.detach() + epsilon * direction,
            raw_lambda=raw_lambda.detach(),
            direct=direct.detach(),
            checkpointing=False,
        )
        minus, _ = refinement_loss(
            harness=harness,
            x=x,
            y=y,
            x_test=x_test,
            y_test=y_test,
            beta0=beta0.detach() - epsilon * direction,
            raw_lambda=raw_lambda.detach(),
            direct=direct.detach(),
            checkpointing=False,
        )
    finite_difference = float((plus - minus) / (2.0 * epsilon))
    relative_error = abs(autodiff - finite_difference) / max(
        1e-8,
        abs(autodiff),
        abs(finite_difference),
    )
    if relative_error >= 0.05:
        raise AssertionError(
            "directional gradient error exceeds 5%: "
            f"autodiff={autodiff}, fd={finite_difference}, "
            f"relative_error={relative_error}"
        )
    return {
        "loss_checkpointed": float(loss_checkpointed.detach()),
        "loss_uncheckpointed": float(loss_plain.detach()),
        "activation_checkpoint_max_abs_difference": output_difference,
        "gradient_norms": {
            name: float(gradient.norm())
            for name, gradient in gradients.items()
        },
        "directional_autodiff": autodiff,
        "directional_finite_difference": finite_difference,
        "directional_relative_error": relative_error,
    }


def full_model_gradient_contract(
    *,
    device: torch.device,
    config_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model = build_model(_load_yaml(str(config_path)), device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.train()
    x, y = synthetic(
        device=device,
        sample_count=64,
        feature_count=500,
        seed=2026072705,
    )
    direct = torch.zeros(
        1,
        1,
        96,
        device=device,
        requires_grad=True,
    )
    penalty = build_penalty(direct)
    shape = penalty["shape_features"]
    shape.retain_grad()
    d0 = penalty["d0"].detach().clone().requires_grad_(True)
    raw_lambda = torch.tensor(
        [[0.01]],
        device=device,
        requires_grad=True,
    )
    auxiliary = model(
        x,
        y,
        lambdas=raw_lambda,
        penalty_shape_features=shape,
        penalty_d0=d0,
        return_aux=True,
    )
    beta = auxiliary["beta_q"]
    loss = beta.square().mean() + beta.abs().mean()
    loss.backward()
    gradients = {
        "raw_lambda": raw_lambda.grad,
        "penalty_shape": shape.grad,
        "penalty_d0": d0.grad,
        "direct96": direct.grad,
    }
    for name, gradient in gradients.items():
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise AssertionError(f"full-model {name} gradient missing")
        if float(gradient.norm()) <= 0.0:
            raise AssertionError(f"full-model {name} gradient is zero")
    visits = auxiliary[
        "kalbi_v8_refinement_visited_features_path_q"
    ]
    if tuple(visits.shape) != (1, 1, 10) or not bool(
        (visits == 500).all()
    ):
        raise AssertionError("full-model ten-sweep visit contract failed")
    trainable_with_gradient = sum(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and float(parameter.grad.norm()) > 0.0
        for parameter in model.parameters()
    )
    if trainable_with_gradient <= 0:
        raise AssertionError("beta model parameters received no gradient")
    return {
        "checkpoint_step": int(checkpoint["completed_steps"]),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "visited_features": visits.detach().cpu().tolist(),
        "gradient_norms": {
            name: float(gradient.norm())
            for name, gradient in gradients.items()
        },
        "trainable_parameter_tensors_with_gradient": (
            trainable_with_gradient
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    result = {
        "status": "PASS",
        "partition_contract": partition_contract(),
        "numerical_contract": numerical_contract(device),
        "gradient_contract": gradient_contract(device),
        "full_model_gradient_contract": full_model_gradient_contract(
            device=device,
            config_path=args.config,
            checkpoint_path=args.checkpoint,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
