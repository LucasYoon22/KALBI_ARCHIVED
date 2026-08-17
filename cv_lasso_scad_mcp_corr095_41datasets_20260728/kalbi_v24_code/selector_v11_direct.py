#!/usr/bin/env python3
"""V11 direct-parameter lambda/penalty selector for V58-Chebyshev.

Unlike V10, the learned penalty head has 96 outputs and uses no PCA basis.  The
96 unmixed coordinates propose the identifiable fixed-knot ReLU-squared parameters

    [base slope b, 95 effective ReLU-squared weights].

Each coordinate is de-standardized independently with the 100k random-penalty
cache mean and scale; there is no rotation or dimension reduction.  A
differentiable feasibility map reconstructs the derivative at all 96
knots, makes it strictly positive, normalizes its integral so rho(6)=1, and
maps the valid derivative back to the exact fixed-knot MLP parameters.  The
native196 tensor, d0, and lambda are then rebuilt through the same differentiable
public contract used by V58.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from penalty_relu_mlp_pwl_v33 import batch_segments_to_relu_params

from .selector_v10 import (
    DirectPenaltyMLPQuerySelectorV10,
    EFFECTIVE_WEIGHT_DIM,
    PENALTY_GRID_SIZE,
    Tensor,
    _basis_matrices,
    native196_from_relu2_parameters,
    normalize_relu2_parameters,
)


DIRECT_PARAMETER_DIM = 1 + EFFECTIVE_WEIGHT_DIM
V11_DIRECT_PENALTY_ENCODING_VERSION = (
    "v11_v58cheb_native196_fixedknot_relu2_direct96_meta4_"
    "positiveknotderivative_splittrunk_v1"
)


def native196_from_derivative_knots(
    derivative: Tensor, knots: Tensor
) -> dict[str, Tensor]:
    """Build native196 stably from the exact piecewise-linear derivative.

    The equivalent global hinge-basis weights can be large and cancellation
    prone in float32.  Integrating adjacent derivative knots is algebraically
    identical but numerically stable and keeps every native coordinate on the
    direct head's computation graph.
    """
    drho = derivative.float()
    grid = knots.to(drho)
    if drho.shape[-1] != PENALTY_GRID_SIZE:
        raise ValueError(f"direct derivative shape drift: {tuple(drho.shape)}")
    intervals = grid[1:] - grid[:-1]
    increments = 0.5 * (drho[..., 1:] + drho[..., :-1]) * intervals
    rho = torch.cat(
        [torch.zeros_like(drho[..., :1]), increments.cumsum(dim=-1)], dim=-1
    )

    one = torch.ones((), device=grid.device, dtype=grid.dtype)
    index = int(torch.searchsorted(grid, one, right=True).item()) - 1
    index = max(0, min(index, PENALTY_GRID_SIZE - 2))
    offset = one - grid[index]
    interval = grid[index + 1] - grid[index]
    local_slope = (drho[..., index + 1] - drho[..., index]) / interval
    rho_at_one = (
        rho[..., index]
        + drho[..., index] * offset
        + 0.5 * local_slope * offset.square()
    )
    derivative_tv = (drho[..., 1:] - drho[..., :-1]).abs().sum(dim=-1)
    linear = torch.stack((rho, drho), dim=-1).reshape(
        *rho.shape[:-1], 2 * PENALTY_GRID_SIZE
    )
    metadata = torch.stack(
        [drho[..., 0], rho_at_one, drho[..., -1], derivative_tv], dim=-1
    )
    native = torch.cat([linear, metadata], dim=-1)
    return {
        "shape_features": native,
        "linear_features": linear,
        "rho": rho,
        "drho": drho,
        "metadata4": metadata,
    }


def derivative_knots_to_relu2_parameters(
    derivative: Tensor, knots: Tensor
) -> tuple[Tensor, Tensor]:
    """Convert positive knot derivatives to the exact ReLU-squared parameters.

    The derivative of a fixed-knot ReLU-squared network is piecewise linear.
    Its first interval slope is the knot-zero weight, and every later weight is
    the change in adjacent interval slope.
    """
    values = derivative.float()
    grid = knots.to(values)
    if values.shape[-1] != PENALTY_GRID_SIZE:
        raise ValueError(
            f"derivative width {values.shape[-1]} != {PENALTY_GRID_SIZE}"
        )
    intervals = grid[1:] - grid[:-1]
    if bool((intervals <= 0.0).any().item()):
        raise ValueError("penalty knots must be strictly increasing")
    slopes = (values[..., 1:] - values[..., :-1]) / intervals
    weights = torch.cat(
        [slopes[..., :1], slopes[..., 1:] - slopes[..., :-1]], dim=-1
    )
    if weights.shape[-1] != EFFECTIVE_WEIGHT_DIM:
        raise RuntimeError(f"direct weight shape drift: {tuple(weights.shape)}")
    return values[..., 0], weights


class DirectReLU2PenaltyParameterHeadV11(nn.Module):
    """Emit 96 direct parameter proposals and map them to a valid penalty."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        derivative_floor: float = 1e-4,
        positivity_beta: float = 100.0,
        head_init_std: float = 1e-3,
        std_floor_fraction: float = 1e-3,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.latent_rank = DIRECT_PARAMETER_DIM
        self.derivative_floor = float(derivative_floor)
        self.positivity_beta = float(positivity_beta)
        self.std_floor_fraction = float(std_floor_fraction)
        if self.derivative_floor <= 0.0:
            raise ValueError("derivative_floor must be positive")
        if self.positivity_beta <= 0.0:
            raise ValueError("positivity_beta must be positive")
        if self.std_floor_fraction <= 0.0:
            raise ValueError("std_floor_fraction must be positive")

        self.projection = nn.Linear(self.hidden_dim, DIRECT_PARAMETER_DIM)
        nn.init.normal_(self.projection.weight, mean=0.0, std=float(head_init_std))
        nn.init.zeros_(self.projection.bias)

        # The exact grid is installed by the V10 module and shared with V58.
        from .selector_v10 import _v33_grid_t

        knots = _v33_grid_t(
            PENALTY_GRID_SIZE,
            6.0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.register_buffer("penalty_grid", knots)
        self.register_buffer("parameter_mean", torch.zeros(DIRECT_PARAMETER_DIM))
        self.register_buffer("parameter_std", torch.ones(DIRECT_PARAMETER_DIM))
        self.register_buffer("reference_b", torch.tensor(1.0 / 6.0))
        self.register_buffer(
            "reference_weights", torch.zeros(EFFECTIVE_WEIGHT_DIM)
        )
        self.register_buffer("explained_variance", torch.tensor(1.0))
        self.register_buffer("cache_size", torch.tensor(0, dtype=torch.long))
        self.register_buffer("calibrated", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("cache_parameter_std_min", torch.tensor(float("nan")))
        self.register_buffer("cache_parameter_std_max", torch.tensor(float("nan")))
        self._last_latent_raw: Tensor | None = None
        self._last_parameter_proposal: Tensor | None = None

    @torch.no_grad()
    def fit_parameter_cache(
        self, relu_b: Tensor, relu_weights: Tensor
    ) -> dict[str, Any]:
        normalized = normalize_relu2_parameters(
            relu_b.detach().cpu(),
            relu_weights.detach().cpu(),
            self.penalty_grid.cpu(),
        )
        parameters = torch.cat(
            [normalized["b"][:, None], normalized["weights"]], dim=-1
        ).float()
        if parameters.shape[0] <= 1 or parameters.shape[1] != DIRECT_PARAMETER_DIM:
            raise ValueError(f"invalid direct parameter cache: {parameters.shape}")
        mean = parameters.mean(dim=0)
        std = parameters.std(dim=0, unbiased=True)
        positive_std = std[std > 0.0]
        if positive_std.numel() == 0:
            raise ValueError("direct parameter cache has zero variance")
        std_floor = positive_std.median() * self.std_floor_fraction
        std = std.clamp_min(std_floor)

        self.parameter_mean.copy_(mean)
        self.parameter_std.copy_(std)
        # The 96 learned coordinates stay axis-aligned with the 96 MLP
        # parameters.  Zero is the cache mean and one unit is one coordinate's
        # cache standard deviation.  This is diagonal preconditioning only:
        # no PCA rotation, mixing, tanh, or dimension reduction is used.
        self.projection.bias.zero_()
        self.reference_b.copy_(mean[0])
        self.reference_weights.copy_(mean[1:])
        self.cache_size.copy_(torch.tensor(parameters.shape[0], dtype=torch.long))
        self.cache_parameter_std_min.copy_(std.amin())
        self.cache_parameter_std_max.copy_(std.amax())
        self.calibrated.fill_(True)

        reference = native196_from_relu2_parameters(
            self.reference_b[None],
            self.reference_weights[None],
            self.penalty_grid,
        )
        return {
            "cache_size": int(parameters.shape[0]),
            "direct_parameter_dim": DIRECT_PARAMETER_DIM,
            "pca_used": False,
            "head_outputs_actual_mlp_parameter_proposal": True,
            "internal_coordinatewise_standardization": True,
            "parameter_std_min": float(std.amin()),
            "parameter_std_max": float(std.amax()),
            "reference_rho6_error": float(
                (reference["rho"][..., -1] - 1.0).abs().amax()
            ),
            "reference_min_drho": float(reference["drho"].amin()),
            "dead_last_weight_max_abs": float(
                normalized["dead_last_weight_max_abs"]
            ),
        }

    @torch.no_grad()
    def fit_segment_cache(self, segments: list[Any]) -> dict[str, Any]:
        _, b, weights = batch_segments_to_relu_params(
            segments,
            K=PENALTY_GRID_SIZE,
            t_max=6.0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        return self.fit_parameter_cache(b, weights)

    def decode_direct_parameters(self, raw: Tensor) -> dict[str, Tensor]:
        if not bool(self.calibrated.item()):
            raise RuntimeError("V11 direct penalty head must be cache-calibrated")
        raw32 = raw.float()
        if raw32.shape[-1] != DIRECT_PARAMETER_DIM:
            raise ValueError(
                f"direct parameter width {raw32.shape[-1]} != {DIRECT_PARAMETER_DIM}"
            )
        standardized = raw32
        proposal = self.parameter_mean.to(raw32) + self.parameter_std.to(
            raw32
        ) * standardized
        self._last_parameter_proposal = proposal
        if proposal.requires_grad:
            proposal.retain_grad()
        proposal_b = proposal[..., 0]
        proposal_weights = proposal[..., 1:]
        grid_relu, _ = _basis_matrices(self.penalty_grid.to(raw32))
        proposal_drho = proposal_b[..., None] + torch.einsum(
            "...k,tk->...t", proposal_weights, grid_relu
        )

        # A smooth pointwise map guarantees positive derivative at every knot.
        # Piecewise linear interpolation then guarantees positivity everywhere
        # on [0, 6].  Integral normalization removes the one redundant scale
        # degree of freedom while preserving all shape gradients.
        positive_drho = self.derivative_floor + F.softplus(
            proposal_drho - self.derivative_floor,
            beta=self.positivity_beta,
        )
        grid = self.penalty_grid.to(raw32)
        mass = torch.trapezoid(positive_drho, grid, dim=-1).clamp_min(1e-12)
        normalized_drho = positive_drho / mass[..., None]
        b, weights = derivative_knots_to_relu2_parameters(
            normalized_drho, grid
        )
        encoded = native196_from_derivative_knots(normalized_drho, grid)
        full_weights = torch.cat(
            [weights, torch.zeros_like(weights[..., :1])], dim=-1
        )
        encoded.update({"relu_b": b, "relu_weights": full_weights})
        reconstruction_error = (encoded["drho"] - normalized_drho).abs().amax(
            dim=-1
        )
        encoded.update(
            {
                "latent_raw": raw32,
                "latent_unit": standardized,
                "direct_parameter_proposal": proposal,
                "proposal_drho": proposal_drho,
                "validity_alpha": torch.ones_like(mass),
                "validity_danger": (
                    self.derivative_floor - proposal_drho.amin(dim=-1)
                ).clamp_min(0.0),
                "validity_projection_active": (
                    proposal_drho.amin(dim=-1) <= self.derivative_floor
                ),
                "rho6_error": (encoded["rho"][..., -1] - 1.0).abs(),
                "direct_reconstruction_error": reconstruction_error,
            }
        )
        return encoded

    def forward(self, hidden: Tensor) -> dict[str, Tensor]:
        raw = self.projection(hidden.float()).float()
        self._last_latent_raw = raw
        if raw.requires_grad:
            raw.retain_grad()
        return self.decode_direct_parameters(raw)


class DirectPenaltyMLPQuerySelectorV11(DirectPenaltyMLPQuerySelectorV10):
    """V10 context/lambda trunks with a no-PCA direct96 penalty head."""

    def __init__(
        self,
        *,
        positivity_beta: float = 100.0,
        direct_head_init_std: float = 1e-3,
        **kwargs: Any,
    ):
        # V10 initializes all shared context and split-trunk components.  Its
        # PCA head is immediately replaced and is absent from the final schema.
        hidden = int(kwargs.get("hidden", 256))
        kwargs.pop("parameter_rank", None)
        super().__init__(parameter_rank=48, **kwargs)
        self.penalty_parameter_head = DirectReLU2PenaltyParameterHeadV11(
            hidden,
            derivative_floor=float(kwargs.get("derivative_floor", 1e-4)),
            positivity_beta=float(positivity_beta),
            head_init_std=float(direct_head_init_std),
        )
