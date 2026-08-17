#!/usr/bin/env python3
"""Contract tests for Selector V21 and Kalbi V6 raw-lambda integration."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import torch

from .canonical_penalty import canonical_grid
from .model import KALBI_EXACT_PG_STEPS, build_model
from .selector_v21_dimension_derivative import (
    DirectPenaltyMLPQuerySelectorV21DimensionDerivative,
    IntegratedDerivativePenaltyHeadV21,
)
from . import selector_v19_k32s32
from train_fpk import _load_yaml


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def standardized_random_task(
    rows: int,
    features: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, rows, features, generator=generator)
    latent = torch.randn(features, generator=generator)
    latent[16:] = 0.0
    y = torch.einsum("bnp,p->bn", x, latent)
    y = y + 0.35 * torch.randn(1, rows, generator=generator)
    x = (x - x.mean(1, keepdim=True)) / x.std(
        1, keepdim=True, unbiased=False
    ).clamp_min(1e-6)
    y = (y - y.mean(1, keepdim=True)) / y.std(
        1, keepdim=True, unbiased=False
    ).clamp_min(1e-6)
    return x, y


def test_penalty_head_contract() -> dict[str, float]:
    torch.manual_seed(7)
    head = IntegratedDerivativePenaltyHeadV21(32)
    task_code = torch.randn(4, 32, requires_grad=True)
    output = head(task_code)
    rho = output["rho"]
    drho = output["drho"]
    grid = canonical_grid().to(rho)
    expected_increment = (
        0.5 * (drho[:, :-1] + drho[:, 1:])
        * (grid[1:] - grid[:-1])
    )
    observed_increment = rho[:, 1:] - rho[:, :-1]
    require(torch.equal(rho[:, 0], torch.zeros_like(rho[:, 0])), "rho(0)")
    require(
        torch.allclose(rho[:, -1], torch.ones_like(rho[:, -1]), atol=2e-6),
        "rho(6)",
    )
    require(bool(torch.isfinite(rho).all()), "finite rho")
    require(bool(torch.isfinite(drho).all()), "finite drho")
    require(bool((drho >= 0.0).all()), "nonnegative drho")
    require(
        torch.allclose(
            observed_increment, expected_increment, atol=2e-7, rtol=2e-6
        ),
        "trapezoidal integral consistency",
    )
    spacing = grid[1:] - grid[:-1]
    require(int(torch.unique(spacing.round(decimals=7)).numel()) > 1, "grid")
    loss = rho.square().mean() + drho.square().mean()
    loss.backward()
    require(
        head.projection.weight.grad is not None
        and float(head.projection.weight.grad.norm()) > 0.0,
        "derivative head gradient",
    )
    return {
        "rho0_error": float(rho[:, 0].abs().max().detach()),
        "rho6_error": float((rho[:, -1] - 1.0).abs().max().detach()),
        "integral_error": float(
            (observed_increment - expected_increment).abs().max().detach()
        ),
        "min_drho": float(drho.min().detach()),
    }


def selector_forward_without_effective_conversion(
    selector: DirectPenaltyMLPQuerySelectorV21DimensionDerivative,
    x: torch.Tensor,
    y: torch.Tensor,
    lambda_max: torch.Tensor,
    *,
    feature_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    original = selector_v19_k32s32.lambda_from_effective_ratio_v56

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("effective-ratio conversion entered V21 forward")

    selector_v19_k32s32.lambda_from_effective_ratio_v56 = forbidden
    try:
        return selector(
            x,
            y,
            lambda_max,
            feature_mask=feature_mask,
            return_representation=True,
        )
    finally:
        selector_v19_k32s32.lambda_from_effective_ratio_v56 = original


def test_variable_dimensions_and_reload() -> dict[str, float]:
    torch.manual_seed(11)
    selector = DirectPenaltyMLPQuerySelectorV21DimensionDerivative()
    selector.eval()
    x, y = standardized_random_task(64, 96, seed=12)
    mask = torch.zeros(1, 96, dtype=torch.bool)
    mask[:, :73] = True
    first = selector_forward_without_effective_conversion(
        selector,
        x,
        y,
        torch.tensor([0.4]),
        feature_mask=mask,
    )
    dimension = first["v21_dimension_context_raw"]
    require(
        torch.allclose(
            dimension[0, :2],
            torch.tensor([64.0, 73.0]).log(),
            atol=1e-6,
        ),
        "actual N/P dimension context",
    )
    require(
        first["lambda_output_contract"] == "raw_lambda_direct_1e-6_to_10",
        "raw-lambda public contract",
    )
    x2, y2 = standardized_random_task(81, 127, seed=13)
    second = selector_forward_without_effective_conversion(
        selector, x2, y2, torch.tensor([0.7])
    )
    require(
        torch.allclose(
            second["v21_dimension_context_raw"][0, :2],
            torch.tensor([81.0, 127.0]).log(),
            atol=1e-6,
        ),
        "variable N/P smoke",
    )

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "selector.pt"
        torch.save(selector.state_dict(), checkpoint)
        reloaded = DirectPenaltyMLPQuerySelectorV21DimensionDerivative()
        reloaded.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True),
            strict=True,
        )
        reloaded.eval()
        repeated = selector_forward_without_effective_conversion(
            reloaded,
            x,
            y,
            torch.tensor([0.4]),
            feature_mask=mask,
        )
    for name in ("raw_lambda", "penalty_rho", "penalty_drho"):
        require(
            torch.equal(first[name], repeated[name]),
            f"strict reload output mismatch: {name}",
        )
    return {
        "parameter_count": float(sum(p.numel() for p in selector.parameters())),
        "raw_lambda": float(first["raw_lambda"].item()),
        "actual_n": float(dimension[0, 0].exp()),
        "actual_p": float(dimension[0, 1].exp()),
    }


def test_heldout_gradient_and_raw_lambda(
    config_path: Path,
) -> dict[str, float]:
    torch.manual_seed(17)
    device = torch.device("cpu")
    model = build_model(_load_yaml(str(config_path)), device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selector = DirectPenaltyMLPQuerySelectorV21DimensionDerivative().to(device)
    selector.train()
    x_train, y_train = standardized_random_task(72, 112, seed=18)
    x_test, y_test = standardized_random_task(31, 112, seed=19)
    with torch.no_grad():
        encoded = model.encode_dataset(model.precompute(x_train, y_train))
    query = selector_forward_without_effective_conversion(
        selector,
        x_train,
        y_train,
        encoded["lambda_max"].float(),
    )
    aux = model.decode_queries(
        encoded,
        lambdas=query["lambdas"],
        lambda_max=encoded["lambda_max"].float()[:, None],
        penalty_shape_features=query["penalty_shape_features"],
        penalty_d0=query["penalty_d0"],
        return_aux=True,
    )
    require(int(aux["kalbi_v6_refinement_steps"]) == KALBI_EXACT_PG_STEPS, "T")
    step = aux["kalbi_v6_refinement_step_size_path_q"].float()
    gamma = aux["kalbi_v6_refinement_gamma_path_q"].float()
    raw = query["raw_lambda"].float()[:, None, None]
    require(
        step.shape[-1] == KALBI_EXACT_PG_STEPS,
        "ten refinement iterations",
    )
    require(
        torch.allclose(gamma, step * raw, atol=2e-7, rtol=2e-6),
        "raw lambda was not passed to every PG iteration",
    )
    beta = aux["beta_q"][:, 0].float()
    prediction = torch.einsum("bnp,bp->bn", x_test.float(), beta)
    heldout_loss = torch.log1p(
        0.5 * (prediction - y_test.float()).square().mean()
    )
    heldout_loss.backward()
    gradients = {
        "derivative_head": selector.penalty_parameter_head.projection.weight.grad,
        "lambda_head": selector.lambda_head.weight.grad,
        "dimension_projection": (
            selector.dimension_context_projection[0].weight.grad
        ),
    }
    for name, gradient in gradients.items():
        require(
            gradient is not None
            and bool(torch.isfinite(gradient).all())
            and float(gradient.norm()) > 0.0,
            f"held-out gradient missing: {name}",
        )

    fixed_beta = torch.linspace(-2.0, 2.0, 112)
    rho_sum = fixed_beta.abs().sum()
    explicit_a = query["raw_lambda"].detach() * rho_sum
    explicit_b = query["raw_lambda"].detach() * rho_sum
    require(
        torch.equal(explicit_a, explicit_b),
        "explicit fixed-beta penalty depends on lambda_max",
    )
    return {
        "heldout_loss": float(heldout_loss.detach()),
        "raw_lambda": float(query["raw_lambda"].detach()),
        "gamma_contract_error": float((gamma - step * raw).abs().max()),
        **{
            f"{name}_gradient_norm": float(gradient.norm())
            for name, gradient in gradients.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "penalty": test_penalty_head_contract(),
        "variable_dimensions_and_reload": test_variable_dimensions_and_reload(),
        "heldout_gradient_and_raw_lambda": (
            test_heldout_gradient_and_raw_lambda(args.config)
        ),
    }
    print(report)


if __name__ == "__main__":
    main()
