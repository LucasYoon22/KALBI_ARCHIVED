#!/usr/bin/env python3
"""Numerical and autograd checks for the Kalbi V5 scalar prox."""

from __future__ import annotations

import json

import torch

from .canonical_penalty import build_penalty, canonical_grid
from .exact_direct96_prox import (
    curvature_safe_step_cap,
    exact_direct96_prox,
    interpolate_direct96_penalty,
    unpack_direct96_penalty,
)


def native_from_rho_drho(rho: torch.Tensor, drho: torch.Tensor) -> torch.Tensor:
    linear = torch.stack((rho, drho), dim=-1).reshape(*rho.shape[:-1], 192)
    metadata = torch.stack(
        (
            drho[..., 0],
            rho[..., -1] / 6.0,
            drho[..., -1],
            (drho[..., 1:] - drho[..., :-1]).abs().sum(-1),
        ),
        dim=-1,
    )
    return torch.cat((linear, metadata), dim=-1)


def check_known_penalties(grid: torch.Tensor) -> dict[str, float]:
    values = torch.linspace(-5.0, 5.0, 1001).reshape(1, 1, -1)

    ridge_rho = (grid.square() / 36.0).reshape(1, 1, -1)
    ridge_drho = (grid / 18.0).reshape(1, 1, -1)
    ridge_native = native_from_rho_drho(ridge_rho, ridge_drho)
    ridge_gamma = torch.tensor([[1.7]])
    ridge, _ = exact_direct96_prox(
        values, ridge_gamma, ridge_native, grid, feature_chunk=127
    )
    ridge_expected = values / (1.0 + 2.0 * ridge_gamma[..., None] / 36.0)
    ridge_error = float((ridge - ridge_expected).abs().max())

    lasso_rho = (grid / 6.0).reshape(1, 1, -1)
    lasso_drho = torch.full_like(lasso_rho, 1.0 / 6.0)
    lasso_native = native_from_rho_drho(lasso_rho, lasso_drho)
    lasso_gamma = torch.tensor([[0.8]])
    lasso, _ = exact_direct96_prox(
        values, lasso_gamma, lasso_native, grid, feature_chunk=131
    )
    threshold = lasso_gamma[..., None] / 6.0
    lasso_expected = torch.sign(values) * torch.relu(values.abs() - threshold)
    lasso_error = float((lasso - lasso_expected).abs().max())
    return {
        "ridge_max_abs_error": ridge_error,
        "lasso_max_abs_error": lasso_error,
    }


def check_random_penalties(grid: torch.Tensor) -> dict[str, float]:
    generator = torch.Generator().manual_seed(20260723)
    direct = torch.randn(6, 96, generator=generator) * 0.35
    direct.requires_grad_(True)
    encoded = build_penalty(direct)
    native = encoded["shape_features"].reshape(2, 3, 196)
    pieces = unpack_direct96_penalty(native, grid)
    lambdas = torch.tensor([[0.2, 0.6, 1.0], [0.15, 0.45, 0.9]])
    cap = curvature_safe_step_cap(lambdas, pieces.kappa)
    step = torch.minimum(torch.full_like(cap, 0.4), 0.5 * cap)
    gamma = step * lambdas
    values = (torch.randn(2, 3, 257, generator=generator) * 1.5).clamp(-5.5, 5.5)
    values.requires_grad_(True)
    prox, diagnostics = exact_direct96_prox(
        values,
        gamma,
        native,
        grid,
        feature_chunk=73,
    )
    magnitude = prox.abs()
    _, drho, _ = interpolate_direct96_penalty(magnitude, pieces, grid)
    stationarity = magnitude - values.abs() + gamma[..., None] * drho
    interior = (magnitude > 1e-6) & (magnitude < float(grid[-1]) - 1e-6)
    stationarity_error = float(stationarity[interior].abs().max().detach())
    zero_valid = bool(
        (
            values.abs()[magnitude <= 1e-8]
            <= diagnostics["zero_threshold"][..., None].expand_as(values)[
                magnitude <= 1e-8
            ]
            + 2e-5
        ).all()
    )
    loss = prox.square().mean() + 0.03 * prox.mean()
    loss.backward()
    if values.grad is None or direct.grad is None:
        raise RuntimeError("exact prox autograd path is missing")
    return {
        "stationarity_max_abs_error": stationarity_error,
        "minimum_curvature_denominator": float(
            diagnostics["minimum_denominator"].min().detach()
        ),
        "zero_region_valid": float(zero_valid),
        "value_gradient_norm": float(values.grad.norm()),
        "direct96_gradient_norm": float(direct.grad.norm()),
        "direct96_gradient_nonzero_fraction": float(
            (direct.grad != 0.0).float().mean()
        ),
    }


def check_finite_difference_gradients(grid: torch.Tensor) -> dict[str, float]:
    def make_native(theta: torch.Tensor) -> torch.Tensor:
        drho = 0.12 + theta * (grid / float(grid[-1]))
        increments = (
            0.5
            * (drho[1:] + drho[:-1])
            * (grid[1:] - grid[:-1])
        )
        rho = torch.cat(
            (torch.zeros(1, dtype=grid.dtype), increments.cumsum(0))
        )
        return native_from_rho_drho(
            rho.reshape(1, 1, -1), drho.reshape(1, 1, -1)
        )

    theta = torch.tensor(0.08, requires_grad=True)
    value = torch.tensor([[[2.37]]], requires_grad=True)
    gamma = torch.tensor([[0.41]])
    prox, _ = exact_direct96_prox(
        value, gamma, make_native(theta), grid
    )
    grad_value, grad_theta = torch.autograd.grad(
        prox.sum(), (value, theta)
    )

    epsilon = 5e-4

    def evaluate(value_scalar: float, theta_scalar: float) -> float:
        candidate, _ = exact_direct96_prox(
            torch.tensor([[[value_scalar]]]),
            gamma,
            make_native(torch.tensor(theta_scalar)),
            grid,
        )
        return float(candidate)

    finite_value = (
        evaluate(float(value.detach()) + epsilon, float(theta.detach()))
        - evaluate(float(value.detach()) - epsilon, float(theta.detach()))
    ) / (2.0 * epsilon)
    finite_theta = (
        evaluate(float(value.detach()), float(theta.detach()) + epsilon)
        - evaluate(float(value.detach()), float(theta.detach()) - epsilon)
    ) / (2.0 * epsilon)
    return {
        "value_gradient_finite_difference_error": abs(
            float(grad_value) - finite_value
        ),
        "penalty_gradient_finite_difference_error": abs(
            float(grad_theta) - finite_theta
        ),
    }


def check_dense_reference(grid: torch.Tensor) -> dict[str, float]:
    generator = torch.Generator().manual_seed(8807)
    direct = torch.randn(3, 96, generator=generator) * 0.25
    encoded = build_penalty(direct)
    native = encoded["shape_features"][:, None, :]
    pieces = unpack_direct96_penalty(native, grid)
    lambdas = torch.tensor([[0.3], [0.7], [1.1]])
    cap = curvature_safe_step_cap(lambdas, pieces.kappa)
    step = torch.minimum(torch.full_like(cap, 0.25), 0.5 * cap)
    gamma = step * lambdas
    values = torch.linspace(-4.8, 4.8, 97).reshape(1, 1, -1).expand(3, 1, -1)
    prox, _ = exact_direct96_prox(values, gamma, native, grid, feature_chunk=41)

    dense = torch.linspace(0.0, 6.0, 30001)
    max_location_error = 0.0
    max_objective_excess = 0.0
    for penalty_index in range(3):
        dense_values = dense.reshape(1, 1, -1)
        rho_dense, _, _ = interpolate_direct96_penalty(
            dense_values,
            DirectPenaltySlice(pieces, penalty_index),
            grid,
        )
        observed = values[penalty_index, 0].abs()
        objectives = (
            0.5 * (dense[None, :] - observed[:, None]).square()
            + gamma[penalty_index, 0] * rho_dense[0, 0][None, :]
        )
        reference_index = objectives.argmin(-1)
        reference = dense[reference_index]
        predicted = prox[penalty_index, 0].abs()
        max_location_error = max(
            max_location_error, float((predicted - reference).abs().max())
        )
        predicted_rho, _, _ = interpolate_direct96_penalty(
            predicted.reshape(1, 1, -1),
            DirectPenaltySlice(pieces, penalty_index),
            grid,
        )
        predicted_objective = (
            0.5 * (predicted - observed).square()
            + gamma[penalty_index, 0] * predicted_rho[0, 0]
        )
        reference_objective = objectives.gather(
            1, reference_index[:, None]
        ).squeeze(1)
        max_objective_excess = max(
            max_objective_excess,
            float((predicted_objective - reference_objective).max()),
        )
    return {
        "dense_reference_location_max_abs_error": max_location_error,
        "dense_reference_objective_max_excess": max_objective_excess,
    }


def DirectPenaltySlice(pieces, index: int):
    return type(pieces)(
        rho=pieces.rho[index : index + 1],
        drho=pieces.drho[index : index + 1],
        slopes=pieces.slopes[index : index + 1],
        kappa=pieces.kappa[index : index + 1],
    )


def main() -> None:
    grid = canonical_grid(dtype=torch.float32)
    result = {
        **check_known_penalties(grid),
        **check_random_penalties(grid),
        **check_finite_difference_gradients(grid),
        **check_dense_reference(grid),
    }
    limits = {
        "ridge_max_abs_error": 2e-5,
        "lasso_max_abs_error": 2e-5,
        "stationarity_max_abs_error": 3e-5,
        "dense_reference_location_max_abs_error": 3e-4,
        "dense_reference_objective_max_excess": 2e-6,
        "value_gradient_finite_difference_error": 5e-4,
        "penalty_gradient_finite_difference_error": 5e-4,
    }
    for name, limit in limits.items():
        if result[name] > limit:
            raise RuntimeError(f"{name}={result[name]} exceeds {limit}")
    if result["minimum_curvature_denominator"] < 0.5 - 2e-5:
        raise RuntimeError("curvature safety margin failed")
    if result["zero_region_valid"] != 1.0:
        raise RuntimeError("zero-threshold condition failed")
    if result["value_gradient_norm"] <= 0.0:
        raise RuntimeError("value gradient is zero")
    if result["direct96_gradient_norm"] <= 0.0:
        raise RuntimeError("Direct96 gradient is zero")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
