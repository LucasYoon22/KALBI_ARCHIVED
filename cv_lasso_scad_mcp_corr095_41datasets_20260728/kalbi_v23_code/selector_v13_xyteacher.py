#!/usr/bin/env python3
"""V13 response-conditioned selector for the V58-Chebyshev solver.

V12 injected raw ``y`` additively into response-seeded Krylov row tokens and
also spent 150 hard decoder queries constructing reference solution-path
features.  V13 replaces both with an explicit, cheap, symmetry-preserving
summary of the linear X-y relationship:

* normalized marginal correlations for the top 32 response-aligned features;
* the upper triangle of their response-aligned local Gram matrix; and
* ten global marginal-signal statistics.

The response alignment makes the summary invariant to row permutation,
feature permutation/sign flips, and a global response sign flip (up to top-k
tie ordering).  A four-depth fusion MLP receives geometry, response, and their
elementwise product before the separate lambda and penalty trunks.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from .canonical_penalty import build_penalty
from .selector_v10 import lambda_from_effective_ratio_v56
from .selector_v12_canonical import (
    DirectPenaltyMLPQuerySelectorV12Canonical,
    V12_HIDDEN_DIM,
    V12_MLP_DEPTH,
)
from .utils import ResidualMLP


Tensor = torch.Tensor

V13_SELECTOR_VERSION = (
    "v13_v58cheb_canonical_direct96_xy_top32_alignedgram_"
    "hardgridteacher_hidden512_depth4_v1"
)
V13_HIDDEN_DIM = V12_HIDDEN_DIM
V13_MLP_DEPTH = V12_MLP_DEPTH
V13_RESPONSE_TOPK = 32
V13_RESPONSE_GLOBAL_DIM = 10
V13_RESPONSE_TRIU_DIM = V13_RESPONSE_TOPK * (V13_RESPONSE_TOPK + 1) // 2
V13_RESPONSE_FEATURE_DIM = (
    V13_RESPONSE_TOPK + V13_RESPONSE_TRIU_DIM + V13_RESPONSE_GLOBAL_DIM
)


@torch.no_grad()
def response_aligned_xy_features(
    encoded: dict[str, Any],
    x_train: Tensor,
    y_train: Tensor,
    *,
    topk: int = V13_RESPONSE_TOPK,
    eps: float = 1e-8,
) -> Tensor:
    """Return a fixed-width, response-aligned sufficient-statistic sketch.

    ``encoded['Xty']`` already equals X^T y / N.  The training tensors have
    train-only standardization, but the explicit normalization below keeps the
    contract correct when a nearly constant column was zeroed.
    """
    if x_train.ndim != 3 or y_train.ndim != 2:
        raise ValueError(
            f"expected x=(B,N,P), y=(B,N), got {x_train.shape}, {y_train.shape}"
        )
    batch, rows, features = x_train.shape
    if y_train.shape != (batch, rows):
        raise ValueError("X/y row alignment drift")
    if int(topk) != V13_RESPONSE_TOPK:
        raise ValueError(f"V13 top-k is fixed at {V13_RESPONSE_TOPK}")
    if features < int(topk):
        raise ValueError(f"V13 requires P >= {topk}, got P={features}")

    xty = encoded["Xty"].float()
    gram_diag = encoded["gram_diag"].float().clamp_min(0.0)
    if xty.shape != (batch, features) or gram_diag.shape != (batch, features):
        raise ValueError("encoded marginal-statistic shape drift")
    y_energy = y_train.float().square().mean(dim=1, keepdim=True).clamp_min(eps)
    denominator = (gram_diag * y_energy).clamp_min(eps).sqrt()
    correlation = torch.nan_to_num(
        xty / denominator, nan=0.0, posinf=0.0, neginf=0.0
    ).clamp(-1.0, 1.0)
    absolute = correlation.abs()
    top_absolute, indices = torch.topk(
        absolute, k=int(topk), dim=-1, largest=True, sorted=True
    )
    selected_correlation = torch.gather(correlation, 1, indices)
    selected_x = torch.gather(
        x_train.float(),
        2,
        indices[:, None, :].expand(batch, rows, int(topk)),
    )
    local_gram = torch.einsum("bnk,bnl->bkl", selected_x, selected_x) / float(rows)
    response_sign = torch.where(
        selected_correlation >= 0.0,
        torch.ones_like(selected_correlation),
        -torch.ones_like(selected_correlation),
    )
    aligned_gram = (
        response_sign[:, :, None]
        * local_gram
        * response_sign[:, None, :]
    ).clamp(-1.0, 1.0)
    triu = torch.triu_indices(int(topk), int(topk), device=x_train.device)
    aligned_upper = aligned_gram[:, triu[0], triu[1]]

    root_n = math.sqrt(float(rows))
    z_absolute = (absolute * root_n).clamp_max(1e6)
    global_stats = torch.cat(
        [
            absolute.mean(-1, keepdim=True),
            absolute.square().mean(-1, keepdim=True).sqrt(),
            absolute.amax(-1, keepdim=True),
            absolute.std(-1, keepdim=True, unbiased=False),
            top_absolute.mean(-1, keepdim=True),
            top_absolute.amin(-1, keepdim=True),
            torch.log1p(z_absolute.mean(-1, keepdim=True)),
            torch.log1p(z_absolute.amax(-1, keepdim=True)),
            (z_absolute > 2.0).float().mean(-1, keepdim=True),
            (z_absolute > 3.0).float().mean(-1, keepdim=True),
        ],
        dim=-1,
    )
    result = torch.cat([top_absolute, aligned_upper, global_stats], dim=-1)
    if result.shape != (batch, V13_RESPONSE_FEATURE_DIM):
        raise RuntimeError(f"V13 response feature shape drift: {result.shape}")
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("V13 response features are nonfinite")
    return result


class DirectPenaltyMLPQuerySelectorV13XYTeacher(
    DirectPenaltyMLPQuerySelectorV12Canonical
):
    """V12 canonical penalty head with explicit response-conditioned fusion."""

    def __init__(
        self,
        *,
        hidden: int = V13_HIDDEN_DIM,
        depth: int = V13_MLP_DEPTH,
        response_feat_dim: int = V13_RESPONSE_FEATURE_DIM,
        **kwargs: Any,
    ) -> None:
        if int(hidden) != V13_HIDDEN_DIM or int(depth) != V13_MLP_DEPTH:
            raise ValueError("V13 fixes hidden=512 and MLP depth=4")
        if int(response_feat_dim) != V13_RESPONSE_FEATURE_DIM:
            raise ValueError(
                f"V13 fixes response feature width={V13_RESPONSE_FEATURE_DIM}"
            )
        # The inherited path adapter is constructed only to keep the canonical
        # V12 initialization path simple, then removed from the final schema.
        super().__init__(
            hidden=int(hidden), depth=int(depth), path_feat_dim=1, **kwargs
        )
        for name in (
            "path_proj",
            "u_proj",
            "y_proj",
            "y_token_mlp",
            "y_pool_proj",
        ):
            delattr(self, name)
        self.response_proj = nn.Sequential(
            nn.LayerNorm(V13_RESPONSE_FEATURE_DIM),
            nn.Linear(V13_RESPONSE_FEATURE_DIM, int(hidden)),
        )
        self.response_net = ResidualMLP(
            int(hidden), int(hidden), int(hidden), depth=int(depth),
            zero_init_last=False, act=nn.GELU,
        )
        self.fusion_net = ResidualMLP(
            3 * int(hidden), int(hidden), int(hidden), depth=int(depth),
            zero_init_last=False, act=nn.GELU,
        )
        self.selector_version = V13_SELECTOR_VERSION
        self.response_feature_dim = V13_RESPONSE_FEATURE_DIM
        self.response_topk = V13_RESPONSE_TOPK

    def architecture_report(self) -> dict[str, Any]:
        def report(module: nn.Module) -> dict[str, int]:
            return {
                "residual_mlp_depth_contract": V13_MLP_DEPTH,
                "linear_modules": sum(
                    isinstance(item, nn.Linear) for item in module.modules()
                ),
                "parameters": sum(p.numel() for p in module.parameters()),
            }

        return {
            "selector_version": V13_SELECTOR_VERSION,
            "hidden_dim": V13_HIDDEN_DIM,
            "all_mlp_depth": V13_MLP_DEPTH,
            "response_topk": V13_RESPONSE_TOPK,
            "response_feature_dim": V13_RESPONSE_FEATURE_DIM,
            "raw_y_token_branch_present": False,
            "reference_path_decoder_features_present": False,
            "response_mlp": report(self.response_net),
            "fusion_mlp": report(self.fusion_net),
            "lambda_mlp": report(self.lambda_net),
            "penalty_mlp": report(self.penalty_net),
            "total_parameters": sum(p.numel() for p in self.parameters()),
        }

    def forward(
        self,
        ds_feat: Tensor,
        t_feat: Tensor,
        lambda_max: Tensor,
        response_feat: Tensor,
    ) -> dict[str, Tensor | str]:
        if response_feat.shape[-1] != V13_RESPONSE_FEATURE_DIM:
            raise ValueError("V13 response feature width drift")
        geometry = self.ds_proj(ds_feat.float()) + self.t_proj(t_feat.float())
        response = self.response_net(self.response_proj(response_feat.float()))
        geometry_unit = torch.nn.functional.layer_norm(
            geometry.float(), (geometry.shape[-1],)
        )
        response_unit = torch.nn.functional.layer_norm(
            response.float(), (response.shape[-1],)
        )
        fused = self.fusion_net(
            torch.cat(
                [geometry_unit, response_unit, geometry_unit * response_unit],
                dim=-1,
            )
        )
        lambda_hidden = self.lambda_net(fused.float())
        penalty_hidden = self.penalty_net(fused.float())
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
            "penalty_encoding_version": self.selector_version,
        }

