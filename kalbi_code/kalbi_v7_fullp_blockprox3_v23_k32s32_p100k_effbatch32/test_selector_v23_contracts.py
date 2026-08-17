#!/usr/bin/env python3
"""Contract tests for V23 independent raw-lambda output and V22 migration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .model import KALBI_EXACT_PG_STEPS, build_model
from .selector_v23_independent_rawlambda import (
    DirectPenaltyMLPQuerySelectorV23IndependentRawLambda,
    V23_INITIAL_RAW_LAMBDA,
    V23_LAMBDA_HEAD_STATE_KEYS,
    V23_LAMBDA_OUTPUT_CONTRACT,
    V23_RAW_LAMBDA_MAX,
    V23_RAW_LAMBDA_MIN,
    load_v22_state_without_lambda_head,
)
from train_fpk import _load_yaml


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def standardized_task(
    rows: int, features: int, *, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, rows, features, generator=generator)
    latent = torch.zeros(features)
    latent[: min(12, features)] = torch.randn(
        min(12, features), generator=generator
    )
    y = torch.einsum("bnp,p->bn", x, latent)
    y += 0.4 * torch.randn(1, rows, generator=generator)
    x = (x - x.mean(1, keepdim=True)) / x.std(
        1, keepdim=True, unbiased=False
    ).clamp_min(1e-6)
    y = (y - y.mean(1, keepdim=True)) / y.std(
        1, keepdim=True, unbiased=False
    ).clamp_min(1e-6)
    return x.to(device), y.to(device)


def lambda_maximum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bnp,bn->bp", x.float(), y.float()).abs().amax(-1) / x.shape[1]


def build_selector() -> DirectPenaltyMLPQuerySelectorV23IndependentRawLambda:
    return DirectPenaltyMLPQuerySelectorV23IndependentRawLambda(
        hidden=512,
        raw_lambda_min=V23_RAW_LAMBDA_MIN,
        raw_lambda_max=V23_RAW_LAMBDA_MAX,
        initial_raw_lambda=V23_INITIAL_RAW_LAMBDA,
        derivative_floor=1e-6,
        squareplus_scale=0.02,
        direct_head_init_std=1e-3,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v22-checkpoint", type=Path, required=True)
    parser.add_argument("--scaler-cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(
        args.v22_checkpoint, map_location="cpu", weights_only=False
    )
    scaler = torch.load(
        args.scaler_cache, map_location="cpu", weights_only=False
    )
    selector = build_selector()
    import_report = load_v22_state_without_lambda_head(
        selector,
        checkpoint["selector_state_dict"],
    )
    selector.load_input_scaler_cache(scaler)
    selector.to(device).eval()

    source_state = checkpoint["selector_state_dict"]
    migrated_state = selector.state_dict()
    for key in source_state:
        if key not in V23_LAMBDA_HEAD_STATE_KEYS:
            require(
                torch.equal(source_state[key].cpu(), migrated_state[key].cpu()),
                f"shared V22 tensor changed: {key}",
            )
    require(
        float(selector.lambda_head.weight.detach().abs().max()) == 0.0,
        "V23 lambda weight was not reset",
    )

    x, y = standardized_task(72, 112, seed=2026072601, device=device)
    lambda_max = lambda_maximum(x, y)
    with torch.no_grad():
        first = selector(
            x, y, lambda_max, return_representation=True
        )
    require(
        first["lambda_output_contract"] == V23_LAMBDA_OUTPUT_CONTRACT,
        "V23 output contract drift",
    )
    require(
        V23_RAW_LAMBDA_MIN
        <= float(first["raw_lambda"])
        <= V23_RAW_LAMBDA_MAX,
        "raw lambda left its public support",
    )
    require(
        torch.allclose(
            first["zero_kkt_ratio"],
            first["raw_lambda"] * first["penalty_d0"].reshape(-1) / lambda_max,
            atol=1e-7,
            rtol=1e-6,
        ),
        "zero-KKT telemetry formula drift",
    )
    require(
        torch.allclose(
            first["force_ratio"],
            first["raw_lambda"]
            * first["penalty_drho"].amax(-1)
            / lambda_max,
            atol=1e-7,
            rtol=1e-6,
        ),
        "force-ratio telemetry formula drift",
    )

    original_penalty_state = {
        key: value.detach().clone()
        for key, value in selector.penalty_parameter_head.state_dict().items()
    }
    with torch.no_grad():
        selector.penalty_parameter_head.projection.bias.add_(
            torch.linspace(
                -1.5,
                1.5,
                selector.penalty_parameter_head.projection.bias.numel(),
                device=device,
            )
        )
        changed_penalty = selector(
            x, y, lambda_max, return_representation=True
        )
    require(
        torch.equal(first["raw_lambda"], changed_penalty["raw_lambda"]),
        "raw lambda changed when only the penalty head changed",
    )
    require(
        not torch.equal(first["penalty_drho"], changed_penalty["penalty_drho"]),
        "penalty mutation failed to change rho prime",
    )
    selector.penalty_parameter_head.load_state_dict(
        original_penalty_state, strict=True
    )

    selector.train()
    selector.zero_grad(set_to_none=True)
    lambda_only = selector(
        x, y, lambda_max, return_representation=True
    )["raw_lambda"].sum()
    lambda_only.backward()
    lambda_grad = selector.lambda_head.weight.grad
    penalty_grad = selector.penalty_parameter_head.projection.weight.grad
    require(
        lambda_grad is not None
        and bool(torch.isfinite(lambda_grad).all())
        and float(lambda_grad.norm()) > 0.0,
        "independent raw-lambda head has no gradient",
    )
    require(
        penalty_grad is None or float(penalty_grad.norm()) == 0.0,
        "raw-lambda-only loss leaked through penalty normalization",
    )

    model = build_model(_load_yaml(str(args.config)), device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selector.zero_grad(set_to_none=True)
    with torch.no_grad():
        encoded = model.encode_dataset(model.precompute(x, y))
    query = selector(
        x,
        y,
        encoded["lambda_max"].float(),
        return_representation=True,
    )
    aux = model.decode_queries(
        encoded,
        lambdas=query["lambdas"],
        lambda_max=encoded["lambda_max"].float()[:, None],
        penalty_shape_features=query["penalty_shape_features"],
        penalty_d0=query["penalty_d0"],
        return_aux=True,
    )
    require(
        int(aux["kalbi_v6_refinement_steps"]) == KALBI_EXACT_PG_STEPS,
        "refinement iteration count drift",
    )
    eta = aux["kalbi_v6_refinement_step_size_path_q"].float()
    gamma = aux["kalbi_v6_refinement_gamma_path_q"].float()
    raw = query["raw_lambda"].float()[:, None, None]
    require(
        torch.allclose(gamma, eta * raw, atol=2e-7, rtol=2e-6),
        "beta solver did not receive V23 raw lambda",
    )
    loss = (
        aux["beta_q"].float().square().mean()
        + 1e-4 * query["penalty_rho"].float().square().mean()
    )
    loss.backward()
    require(
        selector.lambda_head.weight.grad is not None
        and float(selector.lambda_head.weight.grad.norm()) > 0.0,
        "beta path has no lambda-head gradient",
    )
    require(
        selector.penalty_parameter_head.projection.weight.grad is not None
        and float(
            selector.penalty_parameter_head.projection.weight.grad.norm()
        )
        > 0.0,
        "beta path has no penalty-head gradient",
    )

    reloaded = build_selector()
    reloaded.load_state_dict(selector.state_dict(), strict=True)
    reloaded.to(device).eval()
    selector.eval()
    with torch.no_grad():
        repeated = reloaded(
            x, y, lambda_max, return_representation=True
        )
        original = selector(
            x, y, lambda_max, return_representation=True
        )
    for key in ("raw_lambda", "penalty_rho", "penalty_drho"):
        require(
            torch.equal(original[key], repeated[key]),
            f"strict reload changed {key}",
        )

    print(
        json.dumps(
            {
                "status": "PASS",
                "parameter_count": sum(
                    parameter.numel() for parameter in selector.parameters()
                ),
                "import_report": import_report,
                "raw_lambda": float(original["raw_lambda"]),
                "lambda_max": float(lambda_max),
                "d0": float(original["penalty_d0"]),
                "max_drho": float(original["penalty_drho"].amax()),
                "zero_kkt_ratio": float(original["zero_kkt_ratio"]),
                "force_ratio": float(original["force_ratio"]),
                "lambda_only_lambda_head_grad_norm": float(
                    lambda_grad.norm()
                ),
                "lambda_only_penalty_head_grad_norm": (
                    0.0 if penalty_grad is None else float(penalty_grad.norm())
                ),
                "beta_path_lambda_head_grad_norm": float(
                    selector.lambda_head.weight.grad.norm()
                ),
                "beta_path_penalty_head_grad_norm": float(
                    selector.penalty_parameter_head.projection.weight.grad.norm()
                ),
                "gamma_over_eta_error": float(
                    (gamma - eta * raw).abs().max()
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
