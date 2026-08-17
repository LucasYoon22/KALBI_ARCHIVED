#!/usr/bin/env python3
"""One authoritative Direct96 -> native196/ReLU2 penalty contract.

Both random beta queries and the learned V11 selector call ``build_penalty``.
The cache therefore stores only the 96 pre-feasibility MLP parameters; it does
not serialize an independently evaluated rho/drho tensor.
"""

from __future__ import annotations

from typing import Any

import torch

from penalty_relu_mlp_pwl_v33 import _v33_grid_t


Tensor = torch.Tensor
GRID_SIZE = 96
EFFECTIVE_WEIGHT_DIM = GRID_SIZE - 1
DIRECT_PARAMETER_DIM = 1 + EFFECTIVE_WEIGHT_DIM
NATIVE_DIM = 2 * GRID_SIZE + 4
CONTRACT_VERSION = (
    "v12_canonical_direct96_squareplus_positive_"
    "adjacent_integral_native196_meta4_relu2_exact_shared_v1"
)
DEFAULT_DERIVATIVE_FLOOR = 1e-6
DEFAULT_SQUAREPLUS_SCALE = 2e-2


def canonical_grid(
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    return _v33_grid_t(
        GRID_SIZE,
        6.0,
        device=torch.device("cpu") if device is None else torch.device(device),
        dtype=dtype,
    )


def _basis(knots: Tensor) -> Tensor:
    return (
        knots[:, None] - knots[:EFFECTIVE_WEIGHT_DIM][None, :]
    ).clamp_min(0.0)


def stable_squareplus(value: Tensor, scale: float) -> Tensor:
    """Strictly positive squareplus with a nonzero finite-float tail.

    The negative branch avoids the cancellation in ``(x + sqrt(...))/2``.
    Unlike the previous beta=100 softplus, finite inputs do not acquire an
    exactly zero Jacobian merely because a proposal is far below zero.
    """
    width = torch.as_tensor(float(scale), device=value.device, dtype=value.dtype)
    if float(scale) <= 0.0:
        raise ValueError("squareplus scale must be positive")
    root = torch.sqrt(value.square() + 4.0 * width.square())
    positive = 0.5 * (value + root)
    negative = 2.0 * width.square() / (root - value).clamp_min(
        torch.finfo(value.dtype).tiny
    )
    return torch.where(value >= 0.0, positive, negative)


def stable_squareplus_jacobian(value: Tensor, scale: float) -> Tensor:
    """Analytic non-cancelling derivative of :func:`stable_squareplus`."""
    width = torch.as_tensor(float(scale), device=value.device, dtype=value.dtype)
    if float(scale) <= 0.0:
        raise ValueError("squareplus scale must be positive")
    root = torch.sqrt(value.square() + 4.0 * width.square())
    positive_value = 0.5 * (value + root)
    negative_value = 2.0 * width.square() / (root - value).clamp_min(
        torch.finfo(value.dtype).tiny
    )
    positive_jacobian = 0.5 * (1.0 + value / root)
    negative_jacobian = negative_value / root
    return torch.where(value >= 0.0, positive_jacobian, negative_jacobian)


def derivative_knots_to_relu2(
    derivative: Tensor, knots: Tensor
) -> tuple[Tensor, Tensor]:
    intervals = knots[1:] - knots[:-1]
    if bool((intervals <= 0.0).any().item()):
        raise ValueError("canonical knots must be strictly increasing")
    slopes = (derivative[..., 1:] - derivative[..., :-1]) / intervals
    weights = torch.cat(
        [slopes[..., :1], slopes[..., 1:] - slopes[..., :-1]], dim=-1
    )
    if weights.shape[-1] != EFFECTIVE_WEIGHT_DIM:
        raise RuntimeError("canonical effective-weight width drift")
    full_weights = torch.cat(
        [weights, torch.zeros_like(weights[..., :1])], dim=-1
    )
    return derivative[..., 0], full_weights


def native196_from_derivative_knots(
    derivative: Tensor, knots: Tensor
) -> dict[str, Tensor]:
    intervals = knots[1:] - knots[:-1]
    increments = 0.5 * (
        derivative[..., 1:] + derivative[..., :-1]
    ) * intervals
    rho = torch.cat(
        [torch.zeros_like(derivative[..., :1]), increments.cumsum(dim=-1)],
        dim=-1,
    )

    one = torch.ones((), device=knots.device, dtype=knots.dtype)
    index = int(torch.searchsorted(knots, one, right=True).item()) - 1
    index = max(0, min(index, GRID_SIZE - 2))
    offset = one - knots[index]
    interval = knots[index + 1] - knots[index]
    local_curvature = (
        derivative[..., index + 1] - derivative[..., index]
    ) / interval
    rho_one = (
        rho[..., index]
        + derivative[..., index] * offset
        + 0.5 * local_curvature * offset.square()
    )
    tv = (
        derivative[..., 1:] - derivative[..., :-1]
    ).abs().sum(dim=-1)
    linear = torch.stack((rho, derivative), dim=-1).reshape(
        *rho.shape[:-1], 2 * GRID_SIZE
    )
    metadata4 = torch.stack(
        [derivative[..., 0], rho_one, derivative[..., -1], tv], dim=-1
    )
    native = torch.cat([linear, metadata4], dim=-1)
    if native.shape[-1] != NATIVE_DIM:
        raise RuntimeError("canonical native196 width drift")
    return {
        "shape_features": native,
        "linear_features": linear,
        "rho": rho,
        "drho": derivative,
        "metadata4": metadata4,
        "d0": derivative[..., 0],
    }


def build_penalty(
    direct_parameters: Tensor,
    *,
    derivative_floor: float = DEFAULT_DERIVATIVE_FLOOR,
    squareplus_scale: float = DEFAULT_SQUAREPLUS_SCALE,
    knots: Tensor | None = None,
) -> dict[str, Tensor]:
    """Decode Direct96 proposals into the complete public penalty payload."""
    proposal = direct_parameters.float()
    if proposal.shape[-1] != DIRECT_PARAMETER_DIM:
        raise ValueError(
            f"Direct96 width {proposal.shape[-1]} != {DIRECT_PARAMETER_DIM}"
        )
    if not bool(torch.isfinite(proposal).all().item()):
        raise ValueError("Direct96 proposal contains nonfinite values")
    if float(derivative_floor) <= 0.0:
        raise ValueError("derivative floor must be positive")
    grid = (
        canonical_grid(device=proposal.device, dtype=proposal.dtype)
        if knots is None
        else knots.to(proposal)
    )
    basis = _basis(grid)
    proposal_drho = proposal[..., :1] + torch.einsum(
        "...k,tk->...t", proposal[..., 1:], basis
    )
    shifted = proposal_drho - float(derivative_floor)
    positive_drho = float(derivative_floor) + stable_squareplus(
        shifted, float(squareplus_scale)
    )
    positivity_jacobian = stable_squareplus_jacobian(
        shifted, float(squareplus_scale)
    )
    if not bool(torch.isfinite(positivity_jacobian).all().item()) or bool(
        (positivity_jacobian <= 0.0).any().item()
    ):
        raise RuntimeError("canonical positivity map acquired a zero/nonfinite Jacobian")
    mass = torch.trapezoid(positive_drho, grid, dim=-1)
    if not bool(torch.isfinite(mass).all().item()) or bool((mass <= 0.0).any().item()):
        raise RuntimeError("canonical penalty has invalid positive mass")
    derivative = positive_drho / mass[..., None]
    encoded = native196_from_derivative_knots(derivative, grid)
    relu_b, relu_weights = derivative_knots_to_relu2(derivative, grid)
    encoded.update(
        {
            "direct_parameters": proposal,
            "proposal_drho": proposal_drho,
            "positive_drho": positive_drho,
            "positivity_jacobian": positivity_jacobian,
            "mass": mass,
            "relu_b": relu_b,
            "relu_weights": relu_weights,
            "knots": grid,
            "min_drho": derivative.amin(dim=-1),
            "rho6_error": (encoded["rho"][..., -1] - 1.0).abs(),
            "dead_last_weight_max_abs": relu_weights[..., -1].abs(),
        }
    )
    return encoded


def summarize_contract(encoded: dict[str, Tensor]) -> dict[str, Any]:
    drho = encoded["drho"].detach().float()
    rho = encoded["rho"].detach().float()
    return {
        "contract_version": CONTRACT_VERSION,
        "min_drho": float(drho.amin()),
        "rho6_error_max": float((rho[..., -1] - 1.0).abs().amax()),
        "dead_last_weight_max_abs": float(
            encoded["relu_weights"].detach().float()[..., -1].abs().amax()
        ),
        "finite": bool(
            torch.isfinite(encoded["shape_features"]).all().item()
        ),
    }
