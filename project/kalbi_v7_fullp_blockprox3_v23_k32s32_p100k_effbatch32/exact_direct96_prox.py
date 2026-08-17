"""Analytic proximal operator for the canonical Direct96 penalty.

Direct96 represents ``rho'`` by values on a fixed 96-point grid.  Linear
interpolation of those derivative knots makes ``rho`` piecewise quadratic.
For a curvature-safe proximal scale, ``t + gamma * rho'(t)`` is strictly
increasing, so every scalar proximal problem has one global minimizer.  The
minimizer is found by a batched interval search and one closed-form solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


Tensor = torch.Tensor

GRID_SIZE = 96
NATIVE_DIM = 2 * GRID_SIZE + 4
DEFAULT_CURVATURE_MARGIN = 0.9
DEFAULT_FEATURE_CHUNK = 4096


@dataclass(frozen=True)
class Direct96PenaltyPieces:
    rho: Tensor
    drho: Tensor
    slopes: Tensor
    kappa: Tensor


def unpack_direct96_penalty(native196: Tensor, grid: Tensor) -> Direct96PenaltyPieces:
    """Return the coherent piecewise-quadratic penalty representation."""

    native = native196.float()
    if native.shape[-1] != NATIVE_DIM:
        raise ValueError(
            f"Direct96 native width must be {NATIVE_DIM}, got {native.shape[-1]}"
        )
    grid = grid.to(device=native.device, dtype=native.dtype)
    if grid.ndim != 1 or grid.shape[0] != GRID_SIZE:
        raise ValueError(f"penalty grid must have shape ({GRID_SIZE},)")
    intervals = grid[1:] - grid[:-1]
    if bool((intervals <= 0.0).any().item()):
        raise ValueError("penalty grid must be strictly increasing")
    pairs = native[..., : 2 * GRID_SIZE].reshape(
        *native.shape[:-1], GRID_SIZE, 2
    )
    rho = pairs[..., 0]
    drho = pairs[..., 1]
    slopes = (drho[..., 1:] - drho[..., :-1]) / intervals
    kappa = (-slopes.amin(dim=-1)).clamp_min(0.0)
    return Direct96PenaltyPieces(
        rho=rho,
        drho=drho,
        slopes=slopes,
        kappa=kappa,
    )


def curvature_safe_step_cap(
    lambdas: Tensor,
    kappa: Tensor,
    *,
    margin: float = DEFAULT_CURVATURE_MARGIN,
) -> Tensor:
    """Largest step ensuring ``1 + step*lambda*rho'' >= 1-margin``."""

    if not 0.0 < float(margin) < 1.0:
        raise ValueError("curvature margin must lie strictly between zero and one")
    lam = lambdas.float().clamp_min(0.0)
    curvature = kappa.float().clamp_min(0.0)
    if lam.shape != curvature.shape:
        raise ValueError("lambda and kappa shapes must match")
    product = lam * curvature
    infinity = torch.full_like(product, torch.inf)
    return torch.where(product > 0.0, float(margin) / product, infinity)


def interpolate_direct96_penalty(
    magnitude: Tensor,
    pieces: Direct96PenaltyPieces,
    grid: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Evaluate rho, rho-prime, and rho-double-prime at nonnegative inputs."""

    values = magnitude.float().clamp(0.0, float(grid[-1]))
    prefix = pieces.rho.shape[:-1]
    if values.shape[: len(prefix)] != prefix:
        raise ValueError("magnitude and penalty prefix shapes differ")
    grid = grid.to(device=values.device, dtype=values.dtype)
    upper = torch.searchsorted(grid, values.contiguous(), right=True).clamp(
        1, GRID_SIZE - 1
    )
    lower = upper - 1
    lower_grid = grid[lower]
    offset = values - lower_grid
    lower_rho = torch.gather(pieces.rho, -1, lower)
    lower_drho = torch.gather(pieces.drho, -1, lower)
    slope = torch.gather(pieces.slopes, -1, lower)
    rho = lower_rho + lower_drho * offset + 0.5 * slope * offset.square()
    drho = lower_drho + slope * offset
    return rho, drho, slope


def exact_direct96_prox(
    values: Tensor,
    gamma: Tensor,
    native196: Tensor,
    grid: Tensor,
    *,
    feature_chunk: int = DEFAULT_FEATURE_CHUNK,
    curvature_margin: float = DEFAULT_CURVATURE_MARGIN,
    check_contract: bool = True,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Apply the exact constrained scalar prox to every feature.

    The leading shape of ``values`` is ``(..., P)``.  ``gamma`` has the same
    leading shape without ``P`` and equals ``step_size * lambda``.  The prox is
    constrained to the Direct96 domain ``[0, grid[-1]]``.
    """

    if int(feature_chunk) <= 0:
        raise ValueError("feature_chunk must be positive")
    value = values.float()
    gamma = gamma.float().clamp_min(0.0)
    if value.shape[:-1] != gamma.shape:
        raise ValueError("prox value and gamma prefix shapes differ")
    if native196.shape[:-1] != gamma.shape:
        raise ValueError("prox penalty and gamma prefix shapes differ")
    pieces = unpack_direct96_penalty(native196, grid)
    denominator_floor = 1.0 - float(curvature_margin)
    minimum_denominator = 1.0 - gamma * pieces.kappa
    if check_contract and bool(
        (minimum_denominator < denominator_floor - 5e-6).any().item()
    ):
        raise RuntimeError("Direct96 prox received a curvature-unsafe gamma")

    grid = grid.to(device=value.device, dtype=value.dtype)
    gamma_column = gamma[..., None]
    h_knots = grid + gamma_column * pieces.drho
    flat_count = int(gamma.numel())
    feature_count = int(value.shape[-1])
    flat_values = value.reshape(flat_count, feature_count)
    flat_gamma = gamma.reshape(flat_count, 1)
    flat_h = h_knots.reshape(flat_count, GRID_SIZE).contiguous()
    flat_drho = pieces.drho.reshape(flat_count, GRID_SIZE)
    flat_slopes = pieces.slopes.reshape(flat_count, GRID_SIZE - 1)
    outputs: list[Tensor] = []

    for start in range(0, feature_count, int(feature_chunk)):
        stop = min(feature_count, start + int(feature_chunk))
        current = flat_values[:, start:stop]
        magnitude = current.abs()
        upper = torch.searchsorted(
            flat_h,
            magnitude.contiguous(),
            right=True,
        )
        zero = upper == 0
        saturated = upper >= GRID_SIZE
        lower = (upper - 1).clamp(0, GRID_SIZE - 2)
        upper_interval = lower + 1
        lower_grid = grid[lower]
        upper_grid = grid[upper_interval]
        lower_drho = torch.gather(flat_drho, 1, lower)
        slope = torch.gather(flat_slopes, 1, lower)
        denominator = (1.0 + flat_gamma * slope).clamp_min(
            denominator_floor
        )
        stationary = (
            magnitude
            - flat_gamma * (lower_drho - slope * lower_grid)
        ) / denominator
        stationary = torch.maximum(
            lower_grid, torch.minimum(stationary, upper_grid)
        )
        positive = torch.where(zero, torch.zeros_like(stationary), stationary)
        positive = torch.where(
            saturated,
            torch.full_like(positive, float(grid[-1])),
            positive,
        )
        outputs.append(torch.sign(current) * positive)

    prox = torch.cat(outputs, dim=-1).reshape_as(value)
    diagnostics = {
        "kappa": pieces.kappa,
        "minimum_denominator": minimum_denominator,
        "zero_threshold": gamma * pieces.drho[..., 0],
        "saturated_fraction": (
            prox.abs() >= float(grid[-1]) - 1e-7
        ).float().mean(dim=-1),
    }
    return prox.to(values.dtype), diagnostics


__all__ = [
    "DEFAULT_CURVATURE_MARGIN",
    "DEFAULT_FEATURE_CHUNK",
    "Direct96PenaltyPieces",
    "curvature_safe_step_cap",
    "exact_direct96_prox",
    "interpolate_direct96_penalty",
    "unpack_direct96_penalty",
]
