#!/usr/bin/env python3
"""Fail-closed contract tests for KALBI V7 full-P block-prox refinement."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from .block_prox_v7 import (
    BLOCK_SIZE,
    SWEEP_REVERSE,
    feature_blocks,
    run_full_p_block_refinement,
)
from .canonical_penalty import build_penalty
from .model import build_model
from .selector_v23_independent_rawlambda import (
    DirectPenaltyMLPQuerySelectorV23IndependentRawLambda,
    V17_HIDDEN_DIM,
    V23_INITIAL_RAW_LAMBDA,
    V23_RAW_LAMBDA_MAX,
    V23_RAW_LAMBDA_MIN,
)
from train_fpk import _load_yaml


class _ProxHarness(torch.nn.Module):
    def __init__(self, grid: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_kalbi_v7_penalty_grid", grid)


def _penalty(
    *,
    device: torch.device,
    query_count: int = 1,
    direct: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if direct is None:
        direct = torch.zeros(
            1, query_count, 96, device=device, dtype=torch.float32
        )
    payload = build_penalty(direct)
    return (
        payload["shape_features"],
        payload["d0"],
        payload["knots"],
    )


def _synthetic(
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
        1, sample_count, generator=generator, device=device
    )
    y = y - y.mean(1, keepdim=True)
    y = y / y.square().mean(1, keepdim=True).sqrt().clamp_min(1e-4)
    return x, y


def test_visit_contract(device: torch.device) -> dict[str, Any]:
    shape, d0, grid = _penalty(device=device)
    harness = _ProxHarness(grid).to(device)
    results: dict[str, Any] = {}
    for feature_count in (500, 10_000, 100_000):
        blocks = feature_blocks(feature_count)
        visits = torch.zeros(feature_count, dtype=torch.int16)
        for reverse in SWEEP_REVERSE:
            order = (
                range(len(blocks) - 1, -1, -1)
                if reverse
                else range(len(blocks))
            )
            for block_index in order:
                start, stop = blocks[block_index]
                visits[start:stop] += 1
        if not bool((visits == 3).all()):
            raise AssertionError(
                f"P={feature_count} does not receive exactly three visits"
            )

        x, y = _synthetic(
            device=device,
            sample_count=8 if feature_count >= 10_000 else 24,
            feature_count=feature_count,
            seed=1700 + feature_count,
        )
        beta0 = torch.zeros(
            1, 1, feature_count, device=device, dtype=torch.float32
        )
        beta, history = run_full_p_block_refinement(
            harness,
            {"Xc": x, "yc": y},
            beta0,
            torch.tensor([[0.02]], device=device),
            d0,
            shape,
        )
        if len(history) != 3:
            raise AssertionError("refinement did not emit exactly three sweeps")
        for sweep_index, sweep in enumerate(history):
            observed = float(sweep["visited_features"].item())
            if observed != float(feature_count):
                raise AssertionError(
                    f"P={feature_count}, sweep={sweep_index}: "
                    f"visited={observed}"
                )
            before = sweep["objective_before"]
            after = sweep["objective_after"]
            tolerance = 1e-6 * torch.maximum(
                torch.ones_like(before), before.abs()
            )
            if bool((after > before + tolerance).any()):
                raise AssertionError("a full sweep increased the objective")
        if not bool(torch.isfinite(beta).all()):
            raise AssertionError("refined beta is nonfinite")
        results[str(feature_count)] = {
            "blocks": len(blocks),
            "block_size": BLOCK_SIZE,
            "visit_min": int(visits.min()),
            "visit_max": int(visits.max()),
            "objective_delta_by_sweep": [
                float(item["objective_delta"].item()) for item in history
            ],
        }

    x, y = _synthetic(
        device=device, sample_count=24, feature_count=500, seed=1901
    )
    mask = torch.ones(1, 500, device=device, dtype=torch.bool)
    mask[:, -13:] = False
    beta0 = torch.zeros(1, 1, 500, device=device)
    beta0[..., -13:] = 0.5
    beta, _ = run_full_p_block_refinement(
        harness,
        {"Xc": x, "yc": y, "feature_mask": mask},
        beta0,
        torch.tensor([[0.02]], device=device),
        d0,
        shape,
    )
    if bool((beta[..., -13:] != 0.0).any()):
        raise AssertionError("feature mask did not zero invalid coefficients")
    results["feature_mask"] = {"invalid_features_zero": True}
    return results


def _heldout_loss(
    *,
    harness: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    beta0: torch.Tensor,
    raw_lambda: torch.Tensor,
    direct: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    payload = build_penalty(direct)
    beta, _ = run_full_p_block_refinement(
        harness,
        {"Xc": x, "yc": y},
        beta0,
        raw_lambda,
        payload["d0"],
        payload["shape_features"],
    )
    prediction = torch.matmul(beta, x_test.transpose(1, 2))
    return (prediction - y_test[:, None]).square().mean(), beta


def test_gradient_contract(device: torch.device) -> dict[str, Any]:
    _, _, grid = _penalty(device=device)
    harness = _ProxHarness(grid).to(device)
    x, y = _synthetic(
        device=device, sample_count=48, feature_count=96, seed=20260727
    )
    x_test, y_test = _synthetic(
        device=device, sample_count=31, feature_count=96, seed=20260728
    )
    generator = torch.Generator(device=device).manual_seed(20260729)
    beta0 = (
        0.15
        * torch.randn(
            1, 1, 96, generator=generator, device=device
        )
    ).requires_grad_(True)
    raw_lambda = torch.tensor(
        [[0.005]], device=device, requires_grad=True
    )
    direct = (
        0.03
        * torch.randn(
            1, 1, 96, generator=generator, device=device
        )
    ).requires_grad_(True)
    loss, beta = _heldout_loss(
        harness=harness,
        x=x,
        y=y,
        x_test=x_test,
        y_test=y_test,
        beta0=beta0,
        raw_lambda=raw_lambda,
        direct=direct,
    )
    loss.backward()
    gradients = {
        "beta0": beta0.grad,
        "raw_lambda": raw_lambda.grad,
        "direct96": direct.grad,
    }
    for name, gradient in gradients.items():
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise AssertionError(f"{name} gradient is missing/nonfinite")
        if float(gradient.norm()) <= 0.0:
            raise AssertionError(f"{name} gradient is exactly zero")

    direction = torch.randn(
        direct.shape, generator=generator, device=device
    )
    direction = direction / direction.norm().clamp_min(1e-12)
    autodiff = float((direct.grad * direction).sum())
    epsilon = 2e-3
    with torch.no_grad():
        plus, _ = _heldout_loss(
            harness=harness,
            x=x,
            y=y,
            x_test=x_test,
            y_test=y_test,
            beta0=beta0.detach(),
            raw_lambda=raw_lambda.detach(),
            direct=direct.detach() + epsilon * direction,
        )
        minus, _ = _heldout_loss(
            harness=harness,
            x=x,
            y=y,
            x_test=x_test,
            y_test=y_test,
            beta0=beta0.detach(),
            raw_lambda=raw_lambda.detach(),
            direct=direct.detach() - epsilon * direction,
        )
    finite_difference = float((plus - minus) / (2.0 * epsilon))
    relative_error = abs(autodiff - finite_difference) / max(
        1e-8, abs(autodiff), abs(finite_difference)
    )
    if relative_error > 0.05:
        raise AssertionError(
            "Direct96 directional-gradient relative error exceeds 5%: "
            f"autodiff={autodiff}, finite_difference={finite_difference}, "
            f"relative_error={relative_error}"
        )
    return {
        "loss": float(loss.detach()),
        "beta_finite": bool(torch.isfinite(beta).all()),
        "gradient_norms": {
            name: float(gradient.norm())
            for name, gradient in gradients.items()
        },
        "directional_autodiff": autodiff,
        "directional_finite_difference": finite_difference,
        "directional_relative_error": relative_error,
    }


def test_full_model_gradient(
    *,
    device: torch.device,
    config_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model = build_model(_load_yaml(str(config_path)), device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.train()
    x, y = _synthetic(
        device=device, sample_count=64, feature_count=500, seed=20260730
    )
    penalty = build_penalty(
        torch.zeros(1, 1, 96, device=device)
    )
    raw_lambda = torch.tensor(
        [[0.01]], device=device, requires_grad=True
    )
    shape = penalty["shape_features"].detach().requires_grad_(True)
    d0 = penalty["d0"].detach().requires_grad_(True)
    auxiliary = model(
        x,
        y,
        lambdas=raw_lambda,
        penalty_shape_features=shape,
        penalty_d0=d0,
        return_aux=True,
    )
    beta = auxiliary["beta_q"]
    loss = (beta.square().mean() + beta.abs().mean())
    loss.backward()
    values = {
        "raw_lambda": raw_lambda.grad,
        "penalty_shape_features": shape.grad,
        "penalty_d0": d0.grad,
    }
    for name, gradient in values.items():
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise AssertionError(f"full-model {name} gradient is missing/nonfinite")
        if float(gradient.norm()) <= 0.0:
            raise AssertionError(f"full-model {name} gradient is zero")
    visited = auxiliary["kalbi_v7_refinement_visited_features_path_q"]
    if tuple(visited.shape) != (1, 1, 3) or not bool(
        (visited == 500).all()
    ):
        raise AssertionError("full model violated the three-sweep visit contract")
    trainable_gradient_count = sum(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and float(parameter.grad.norm()) > 0.0
        for parameter in model.parameters()
    )
    if trainable_gradient_count <= 0:
        raise AssertionError("full-model beta parameters received no gradient")
    return {
        "checkpoint_step": int(checkpoint["completed_steps"]),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "visited_features": visited.detach().cpu().tolist(),
        "input_gradient_norms": {
            name: float(gradient.norm())
            for name, gradient in values.items()
        },
        "trainable_parameter_tensors_with_gradient": trainable_gradient_count,
    }


def test_selector_to_refinement_gradient(
    *,
    device: torch.device,
    config_path: Path,
    beta_checkpoint_path: Path,
    selector_checkpoint_path: Path,
    input_scaler_path: Path,
) -> dict[str, Any]:
    beta_checkpoint = torch.load(
        beta_checkpoint_path, map_location="cpu", weights_only=False
    )
    selector_checkpoint = torch.load(
        selector_checkpoint_path, map_location="cpu", weights_only=False
    )
    model = build_model(_load_yaml(str(config_path)), device)
    model.load_state_dict(
        beta_checkpoint["model_state_dict"], strict=True
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    selector = DirectPenaltyMLPQuerySelectorV23IndependentRawLambda(
        hidden=V17_HIDDEN_DIM,
        raw_lambda_min=V23_RAW_LAMBDA_MIN,
        raw_lambda_max=V23_RAW_LAMBDA_MAX,
        initial_raw_lambda=V23_INITIAL_RAW_LAMBDA,
        derivative_floor=1e-6,
        squareplus_scale=0.02,
        direct_head_init_std=1e-3,
    ).to(device)
    selector.load_state_dict(
        selector_checkpoint["selector_state_dict"], strict=True
    )
    scaler = torch.load(
        input_scaler_path, map_location="cpu", weights_only=False
    )
    selector.load_input_scaler_cache(scaler)
    selector.train()

    x, y = _synthetic(
        device=device, sample_count=80, feature_count=500, seed=20260731
    )
    x_train, x_test = x[:, :64], x[:, 64:]
    y_train, y_test = y[:, :64], y[:, 64:]
    with torch.no_grad():
        encoded = model.encode_dataset(model.precompute(x_train, y_train))
    query = selector(
        x_train,
        y_train,
        encoded["lambda_max"].float(),
        return_representation=True,
    )
    for name in ("lambdas", "penalty_shape_features", "penalty_d0"):
        query[name].retain_grad()
    auxiliary = model.decode_queries(
        encoded,
        lambdas=query["lambdas"],
        lambda_max=encoded["lambda_max"].float()[:, None],
        penalty_shape_features=query["penalty_shape_features"],
        penalty_d0=query["penalty_d0"],
        return_aux=True,
    )
    beta = auxiliary["beta_q"][:, 0].float()
    prediction = torch.matmul(x_test.float(), beta.unsqueeze(-1)).squeeze(-1)
    f_hat = 0.5 * (prediction - y_test.float()).square().mean(-1)
    f_zero = 0.5 * y_test.float().square().mean(-1).clamp_min(1e-12)
    loss = torch.log1p(f_hat / f_zero).mean()
    loss.backward()

    output_gradient_norms: dict[str, float] = {}
    for name in ("lambdas", "penalty_shape_features", "penalty_d0"):
        gradient = query[name].grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise AssertionError(
                f"selector output {name} gradient is missing/nonfinite"
            )
        norm = float(gradient.norm())
        if norm <= 0.0:
            raise AssertionError(f"selector output {name} gradient is zero")
        output_gradient_norms[name] = norm

    parameter_groups = {
        "lambda_head": tuple(selector.lambda_head.parameters()),
        "penalty_mlp": tuple(selector.penalty_mlp.parameters()),
        "penalty_parameter_head": tuple(
            selector.penalty_parameter_head.parameters()
        ),
    }
    parameter_gradient_norms: dict[str, float] = {}
    for name, parameters in parameter_groups.items():
        gradients = [
            parameter.grad.detach().float().reshape(-1)
            for parameter in parameters
            if parameter.grad is not None
        ]
        if not gradients:
            raise AssertionError(f"selector {name} received no gradient")
        norm = float(torch.cat(gradients).norm())
        if not math.isfinite(norm) or norm <= 0.0:
            raise AssertionError(
                f"selector {name} gradient is zero/nonfinite"
            )
        parameter_gradient_norms[name] = norm
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("frozen beta parameters received selector gradients")
    return {
        "beta_checkpoint_step": int(beta_checkpoint["completed_steps"]),
        "selector_checkpoint_step": int(
            selector_checkpoint["completed_steps"]
        ),
        "selector_objective": float(loss.detach()),
        "raw_lambda": float(query["raw_lambda"].detach().reshape(-1)[0]),
        "output_gradient_norms": output_gradient_norms,
        "parameter_gradient_norms": parameter_gradient_norms,
        "visited_features": auxiliary[
            "kalbi_v7_refinement_visited_features_path_q"
        ].detach().cpu().tolist(),
        "beta_parameters_frozen": True,
        "heldout_y_used_only_in_loss": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "visit",
            "gradient",
            "full-model",
            "selector-gradient",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--selector-checkpoint", type=Path)
    parser.add_argument("--input-scaler", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("KALBI V7 contract tests require CUDA")
    device = torch.device("cuda:0")
    result: dict[str, Any] = {}
    if args.mode in ("visit", "all"):
        result["visit"] = test_visit_contract(device)
    if args.mode in ("gradient", "all"):
        result["gradient"] = test_gradient_contract(device)
    if args.mode in ("full-model", "all"):
        if args.config is None or args.checkpoint is None:
            parser.error("--config and --checkpoint are required for full-model")
        result["full_model"] = test_full_model_gradient(
            device=device,
            config_path=args.config,
            checkpoint_path=args.checkpoint,
        )
    if args.mode in ("selector-gradient", "all"):
        if (
            args.config is None
            or args.checkpoint is None
            or args.selector_checkpoint is None
            or args.input_scaler is None
        ):
            parser.error(
                "--config, --checkpoint, --selector-checkpoint, and "
                "--input-scaler are required for selector-gradient"
            )
        result["selector_gradient"] = (
            test_selector_to_refinement_gradient(
                device=device,
                config_path=args.config,
                beta_checkpoint_path=args.checkpoint,
                selector_checkpoint_path=args.selector_checkpoint,
                input_scaler_path=args.input_scaler,
            )
        )
    serialized = json.dumps(result, indent=2, sort_keys=True)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
