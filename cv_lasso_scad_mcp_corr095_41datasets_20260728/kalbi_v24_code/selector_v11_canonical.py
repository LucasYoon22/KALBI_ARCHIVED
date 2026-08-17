#!/usr/bin/env python3
"""V11 Canonical-v2 selector using the shared Direct96 builder."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from .canonical_penalty import (
    CONTRACT_VERSION,
    DEFAULT_DERIVATIVE_FLOOR,
    DEFAULT_SQUAREPLUS_SCALE,
    DIRECT_PARAMETER_DIM,
    build_penalty,
    canonical_grid,
)
from .selector_v10 import (
    DirectPenaltyMLPQuerySelectorV10,
    lambda_from_effective_ratio_v56,
)


Tensor = torch.Tensor


class CanonicalDirect96Head(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        derivative_floor: float = DEFAULT_DERIVATIVE_FLOOR,
        squareplus_scale: float = DEFAULT_SQUAREPLUS_SCALE,
        head_init_std: float = 1e-3,
        std_floor_fraction: float = 1e-3,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.latent_rank = DIRECT_PARAMETER_DIM
        self.derivative_floor = float(derivative_floor)
        self.squareplus_scale = float(squareplus_scale)
        self.std_floor_fraction = float(std_floor_fraction)
        self.projection = nn.Linear(self.hidden_dim, DIRECT_PARAMETER_DIM)
        nn.init.normal_(self.projection.weight, 0.0, float(head_init_std))
        nn.init.zeros_(self.projection.bias)
        self.register_buffer("penalty_grid", canonical_grid())
        self.register_buffer("parameter_mean", torch.zeros(DIRECT_PARAMETER_DIM))
        self.register_buffer("parameter_std", torch.ones(DIRECT_PARAMETER_DIM))
        self.register_buffer("reference_b", torch.tensor(1.0 / 6.0))
        self.register_buffer("reference_weights", torch.zeros(95))
        self.register_buffer("explained_variance", torch.tensor(1.0))
        self.register_buffer("cache_size", torch.tensor(0, dtype=torch.long))
        self.register_buffer("calibrated", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("cache_parameter_std_min", torch.tensor(float("nan")))
        self.register_buffer("cache_parameter_std_max", torch.tensor(float("nan")))
        self._last_latent_raw: Tensor | None = None
        self._last_parameter_proposal: Tensor | None = None

    @torch.no_grad()
    def fit_parameter_cache(self, direct_parameters: Tensor) -> dict[str, Any]:
        parameters = direct_parameters.detach().cpu().float()
        if parameters.ndim != 2 or parameters.shape[1] != DIRECT_PARAMETER_DIM:
            raise ValueError(f"invalid Direct96 cache shape {tuple(parameters.shape)}")
        if parameters.shape[0] <= 1 or not bool(torch.isfinite(parameters).all()):
            raise ValueError("Direct96 cache must be finite and nontrivial")
        mean = parameters.mean(0)
        std = parameters.std(0, unbiased=True)
        positive = std[std > 0.0]
        if positive.numel() != DIRECT_PARAMETER_DIM:
            raise ValueError("Direct96 cache has a zero-variance coordinate")
        floor = positive.median() * self.std_floor_fraction
        std = std.clamp_min(floor)
        self.parameter_mean.copy_(mean)
        self.parameter_std.copy_(std)
        self.cache_size.fill_(parameters.shape[0])
        self.cache_parameter_std_min.copy_(std.amin())
        self.cache_parameter_std_max.copy_(std.amax())
        self.calibrated.fill_(True)
        reference = build_penalty(
            mean[None],
            derivative_floor=self.derivative_floor,
            squareplus_scale=self.squareplus_scale,
            knots=self.penalty_grid,
        )
        self.reference_b.copy_(reference["relu_b"][0])
        self.reference_weights.copy_(reference["relu_weights"][0, :-1])
        return {
            "cache_size": int(parameters.shape[0]),
            "direct_parameter_dim": DIRECT_PARAMETER_DIM,
            "pca_used": False,
            "canonical_contract_version": CONTRACT_VERSION,
            "parameter_std_min": float(std.amin()),
            "parameter_std_max": float(std.amax()),
            "reference_min_drho": float(reference["drho"].amin()),
            "reference_rho6_error": float(reference["rho6_error"].amax()),
        }

    def decode_direct_parameters(self, raw: Tensor) -> dict[str, Tensor]:
        if not bool(self.calibrated.item()):
            raise RuntimeError("canonical V11 head is not cache calibrated")
        raw32 = raw.float()
        proposal = self.parameter_mean.to(raw32) + self.parameter_std.to(raw32) * raw32
        self._last_parameter_proposal = proposal
        if proposal.requires_grad:
            proposal.retain_grad()
        encoded = build_penalty(
            proposal,
            derivative_floor=self.derivative_floor,
            squareplus_scale=self.squareplus_scale,
            knots=self.penalty_grid,
        )
        encoded.update(
            {
                "latent_raw": raw32,
                "latent_unit": raw32,
                "direct_parameter_proposal": proposal,
                "validity_alpha": torch.ones_like(encoded["mass"]),
                "validity_danger": (
                    self.derivative_floor - encoded["proposal_drho"].amin(-1)
                ).clamp_min(0.0),
                "validity_projection_active": (
                    encoded["proposal_drho"].amin(-1) <= self.derivative_floor
                ),
                "direct_reconstruction_error": torch.zeros_like(encoded["mass"]),
            }
        )
        return encoded

    def forward(self, hidden: Tensor) -> dict[str, Tensor]:
        raw = self.projection(hidden.float()).float()
        self._last_latent_raw = raw
        if raw.requires_grad:
            raw.retain_grad()
        return self.decode_direct_parameters(raw)


class DirectPenaltyMLPQuerySelectorV11Canonical(DirectPenaltyMLPQuerySelectorV10):
    def __init__(
        self,
        *,
        squareplus_scale: float = DEFAULT_SQUAREPLUS_SCALE,
        direct_head_init_std: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        hidden = int(kwargs.get("hidden", 256))
        kwargs.pop("parameter_rank", None)
        super().__init__(parameter_rank=48, **kwargs)
        self.penalty_parameter_head = CanonicalDirect96Head(
            hidden,
            derivative_floor=float(kwargs.get("derivative_floor", DEFAULT_DERIVATIVE_FLOOR)),
            squareplus_scale=float(squareplus_scale),
            head_init_std=float(direct_head_init_std),
        )

    def fit_penalty_parameter_cache(
        self, direct_parameters: Tensor, relu_weights: Tensor | None = None
    ) -> dict[str, Any]:
        if relu_weights is not None:
            raise ValueError(
                "canonical V11 calibration accepts one Direct96 tensor, not b/weights"
            )
        return self.penalty_parameter_head.fit_parameter_cache(direct_parameters)

    def forward(
        self,
        ds_feat: Tensor,
        t_feat: Tensor,
        lambda_max: Tensor,
        path_feat: Tensor,
        *,
        U: Tensor | None = None,
        y_context: Tensor | None = None,
    ) -> dict[str, Tensor | str]:
        hidden = (
            self.ds_proj(ds_feat.float())
            + self.t_proj(t_feat.float())
            + self.path_proj(path_feat.float())
        )
        if U is not None and y_context is not None:
            if U.dim() == 4:
                U = U[:, 0]
            if U.shape[-1] != self.krylov_dim and U.shape[1] == self.krylov_dim:
                U = U.transpose(1, 2)
            token = self.y_token_mlp(
                self.u_proj(U.float())
                + self.y_proj(y_context.float().unsqueeze(-1))
            )
            pooled = torch.cat(
                [
                    token.mean(dim=1),
                    token.std(dim=1, unbiased=False),
                    token.amax(dim=1),
                ],
                dim=-1,
            )
            hidden = hidden + self.y_pool_proj(pooled)
        lambda_hidden = self.lambda_net(hidden.float())
        penalty_hidden = self.penalty_net(hidden.float())
        lambda_unit = torch.sigmoid(self.lambda_head(lambda_hidden).squeeze(-1))
        lambda_ratio = torch.exp(
            math.log(self.lambda_min_ratio)
            + (math.log(self.lambda_max_ratio) - math.log(self.lambda_min_ratio))
            * lambda_unit
        ).clamp(self.lambda_min_ratio, self.lambda_max_ratio)
        encoded = self.penalty_parameter_head(penalty_hidden)
        shape = encoded["shape_features"]
        d0 = encoded["d0"]
        lambda_scale = lambda_from_effective_ratio_v56(
            lambda_ratio, lambda_max.float(), d0
        )
        reference_direct = self.penalty_parameter_head.parameter_mean.to(shape)[None]
        reference_native = build_penalty(
            reference_direct,
            derivative_floor=self.penalty_parameter_head.derivative_floor,
            squareplus_scale=self.penalty_parameter_head.squareplus_scale,
            knots=self.penalty_parameter_head.penalty_grid,
        )["shape_features"][0]
        reference_distance = (
            (shape - reference_native).square().mean(-1).sqrt()
            / reference_native.square().mean().sqrt().clamp_min(1e-8)
        )
        return {
            "lambdas": lambda_scale["lambdas"][:, None],
            "lambda_ratio": lambda_ratio,
            "effective_lambda_ratio": lambda_scale["effective_ratio"],
            "penalty_shape_features": shape[:, None, :],
            "penalty_d0": d0[:, None],
            "penalty_relu_b": encoded["relu_b"],
            "penalty_relu_weights": encoded["relu_weights"],
            "penalty_rho": encoded["rho"],
            "penalty_drho": encoded["drho"],
            "penalty_metadata4": encoded["metadata4"],
            "penalty_latent_raw": encoded["latent_raw"],
            "penalty_latent_unit": encoded["latent_unit"],
            "penalty_validity_alpha": encoded["validity_alpha"],
            "penalty_validity_danger": encoded["validity_danger"],
            "penalty_validity_projection_active": encoded[
                "validity_projection_active"
            ],
            "penalty_rho6_error": encoded["rho6_error"],
            "penalty_min_drho": encoded["drho"].amin(-1),
            "penalty_max_drho": encoded["drho"].amax(-1),
            "penalty_derivative_tv": encoded["metadata4"][..., 3],
            "penalty_positivity_jacobian_min": encoded[
                "positivity_jacobian"
            ].amin(-1),
            "penalty_reference_distance": reference_distance,
            "penalty_reference_slope": lambda_scale["reference_slope"],
            "penalty_ridge_branch": lambda_scale["ridge_branch"],
            "penalty_parameter_explained_variance": (
                self.penalty_parameter_head.explained_variance.to(shape)
            ),
            "penalty_encoding_version": CONTRACT_VERSION,
        }
