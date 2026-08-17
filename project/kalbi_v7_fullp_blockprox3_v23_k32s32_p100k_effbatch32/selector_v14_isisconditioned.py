#!/usr/bin/env python3
"""V14 selector with an actual train-only hard-ISIS28 pilot path.

The Krylov/Gram representation is confined to ``range(X^T)``.  For ``P>N``
that is sufficient for prediction but it does not expose the original-coordinate
support competition needed by a sparse penalty selector.  V14 therefore uses a
two-pass inference contract:

1. evaluate one fixed canonical penalty/effective-ratio query with the exact
   production V58 hard ISIS28 solver on training rows only;
2. summarize the ordered 28-stage original-coordinate support path, selected
   X-y/Gram statistics, and the final pilot residual/KKT state; and
3. inject that fixed-width ISIS representation into the final lambda and
   Direct96 penalty selector.

Held-out rows are used only by the training loss/teacher after the final selector
query has been emitted.  The pilot is fixed rather than learned, so the discrete
support path cannot create a dead first-stage selector branch.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from .canonical_penalty import build_penalty
from .selector_v10 import lambda_from_effective_ratio_v56
from .selector_v13_xyteacher import (
    DirectPenaltyMLPQuerySelectorV13XYTeacher,
    V13_HIDDEN_DIM,
    V13_MLP_DEPTH,
    V13_RESPONSE_FEATURE_DIM,
)
from .utils import ResidualMLP


Tensor = torch.Tensor

V14_SELECTOR_VERSION = (
    "v14_v58cheb_canonical_direct96_xy_hardisis28pilot_"
    "support_residual_injected_hidden512_depth4_v1"
)
V14_HIDDEN_DIM = V13_HIDDEN_DIM
V14_MLP_DEPTH = V13_MLP_DEPTH
V14_ISIS_STAGES = 28
V14_ISIS_TOPK = 32
V14_ISIS_STAGE_DIM = 16
V14_ISIS_GLOBAL_DIM = 12
V14_ISIS_FEATURE_DIM = (
    V14_ISIS_STAGES * V14_ISIS_STAGE_DIM + V14_ISIS_GLOBAL_DIM
)
V14_PILOT_EFFECTIVE_RATIO = 0.1


def _gather_last(values: Tensor, indices: Tensor) -> Tensor:
    return torch.gather(values, -1, indices)


@torch.no_grad()
def hard_isis28_path_features(
    encoded: dict[str, Any],
    x_train: Tensor,
    y_train: Tensor,
    pilot_aux: dict[str, Any],
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Build the invariant fixed-width representation of one hard ISIS28 run."""
    if x_train.ndim != 3 or y_train.ndim != 2:
        raise ValueError("V14 ISIS features require X=(B,N,P), y=(B,N)")
    batch, rows, features = x_train.shape
    if y_train.shape != (batch, rows):
        raise ValueError("V14 ISIS X/y row alignment drift")
    if int(pilot_aux.get("v56fast_isis_requested_stages", -1)) != V14_ISIS_STAGES:
        raise ValueError("V14 requires the production 28-stage ISIS path")
    if int(pilot_aux.get("v56fast_isis_topk", -1)) != V14_ISIS_TOPK:
        raise ValueError("V14 requires the production ISIS TopK=32 path")

    selected = pilot_aux.get("v56fast_isis_selected_indices")
    active = pilot_aux.get("v56fast_isis_stage_active")
    beta_q = pilot_aux.get("beta_q")
    if not all(torch.is_tensor(item) for item in (selected, active, beta_q)):
        raise ValueError("V14 pilot aux lacks hard ISIS support/beta tensors")
    expected_selected = (batch, 1, V14_ISIS_STAGES, V14_ISIS_TOPK)
    expected_active = (batch, 1, V14_ISIS_STAGES)
    if tuple(selected.shape) != expected_selected:
        raise ValueError(
            f"V14 ISIS selected-index shape drift: {tuple(selected.shape)}"
        )
    if tuple(active.shape) != expected_active:
        raise ValueError(f"V14 ISIS active-stage shape drift: {tuple(active.shape)}")
    if tuple(beta_q.shape) != (batch, 1, features):
        raise ValueError(f"V14 pilot beta shape drift: {tuple(beta_q.shape)}")

    indices = selected[:, 0].long()
    stage_active = active[:, 0].bool()
    safe_indices = indices.clamp(0, max(0, features - 1))
    chosen_valid = (indices >= 0) & (indices < features)
    if bool((chosen_valid != stage_active.unsqueeze(-1)).any()):
        raise ValueError("V14 inactive/active ISIS index sentinel contract drift")

    xty = encoded["Xty"].float()
    gram_diag = encoded["gram_diag"].float().clamp_min(0.0)
    if xty.shape != (batch, features) or gram_diag.shape != (batch, features):
        raise ValueError("V14 encoded marginal-statistic shape drift")
    y_energy = y_train.float().square().mean(-1, keepdim=True).clamp_min(eps)
    marginal = torch.nan_to_num(
        xty / (gram_diag * y_energy).clamp_min(eps).sqrt(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp(-1.0, 1.0)
    selected_marginal = _gather_last(
        marginal[:, None, :].expand(batch, V14_ISIS_STAGES, features),
        safe_indices,
    )
    selected_abs = selected_marginal.abs()

    pilot_beta = beta_q[:, 0].float()
    selected_beta_abs = _gather_last(
        pilot_beta.abs()[:, None, :].expand(batch, V14_ISIS_STAGES, features),
        safe_indices,
    )

    flat_indices = safe_indices.reshape(batch, -1)
    selected_x = torch.gather(
        x_train.float(),
        2,
        flat_indices[:, None, :].expand(batch, rows, flat_indices.shape[-1]),
    ).reshape(batch, rows, V14_ISIS_STAGES, V14_ISIS_TOPK)
    local_gram = torch.einsum("bnsk,bnsl->bskl", selected_x, selected_x) / float(
        max(1, rows)
    )
    response_sign = torch.where(
        selected_marginal >= 0.0,
        torch.ones_like(selected_marginal),
        -torch.ones_like(selected_marginal),
    )
    aligned_gram = (
        response_sign[..., :, None]
        * local_gram
        * response_sign[..., None, :]
    )
    diagonal = local_gram.diagonal(dim1=-2, dim2=-1).clamp_min(0.0)
    eye = torch.eye(
        V14_ISIS_TOPK, dtype=torch.bool, device=x_train.device
    ).view(1, 1, V14_ISIS_TOPK, V14_ISIS_TOPK)
    offdiag_abs = local_gram.abs().masked_fill(eye, 0.0)
    aligned_offdiag = aligned_gram.masked_fill(eye, 0.0)
    offdiag_denominator = float(V14_ISIS_TOPK * (V14_ISIS_TOPK - 1))

    stage_position = torch.linspace(
        1.0 / V14_ISIS_STAGES,
        1.0,
        V14_ISIS_STAGES,
        device=x_train.device,
        dtype=torch.float32,
    )[None, :].expand(batch, -1)
    active_float = stage_active.float()
    cumulative_support_fraction = (
        stage_active.float().cumsum(-1) * float(V14_ISIS_TOPK) / float(features)
    ).clamp(0.0, 1.0)
    stage_stats = torch.stack(
        [
            active_float,
            stage_position,
            cumulative_support_fraction,
            selected_abs.mean(-1),
            selected_abs.amax(-1),
            selected_abs.amin(-1),
            selected_abs.std(-1, unbiased=False),
            torch.log1p(selected_beta_abs.mean(-1)),
            torch.log1p(selected_beta_abs.amax(-1)),
            (selected_beta_abs > 1e-3).float().mean(-1),
            diagonal.mean(-1),
            diagonal.std(-1, unbiased=False),
            offdiag_abs.sum((-1, -2)) / offdiag_denominator,
            offdiag_abs.amax((-1, -2)),
            aligned_offdiag.sum((-1, -2)) / offdiag_denominator,
            torch.log1p(
                diagonal.amax(-1)
                / diagonal.amin(-1).clamp_min(max(eps, 1e-6))
            ),
        ],
        dim=-1,
    )
    value_mask = active_float.unsqueeze(-1)
    stage_stats = torch.cat(
        [stage_stats[..., :2], stage_stats[..., 2:] * value_mask], dim=-1
    )

    residual = y_train.float() - torch.einsum(
        "bnp,bp->bn", x_train.float(), pilot_beta
    )
    residual_energy = residual.square().mean(-1, keepdim=True).clamp_min(eps)
    residual_rhs = torch.einsum("bnp,bn->bp", x_train.float(), residual) / float(
        max(1, rows)
    )
    residual_correlation = torch.nan_to_num(
        residual_rhs / (gram_diag * residual_energy).clamp_min(eps).sqrt(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).abs().clamp(0.0, 1e6)
    residual_top = torch.topk(
        residual_correlation,
        k=min(V14_ISIS_TOPK, features),
        dim=-1,
        largest=True,
        sorted=True,
    ).values
    z_residual = residual_correlation * math.sqrt(float(rows))
    global_stats = torch.cat(
        [
            (residual_energy / y_energy).sqrt(),
            residual_correlation.mean(-1, keepdim=True),
            residual_correlation.square().mean(-1, keepdim=True).sqrt(),
            residual_correlation.amax(-1, keepdim=True),
            residual_correlation.std(-1, keepdim=True, unbiased=False),
            residual_top.mean(-1, keepdim=True),
            residual_top.amin(-1, keepdim=True),
            (z_residual > 2.0).float().mean(-1, keepdim=True),
            (z_residual > 3.0).float().mean(-1, keepdim=True),
            (pilot_beta.abs() > 1e-3).float().mean(-1, keepdim=True),
            torch.log1p(pilot_beta.norm(dim=-1, keepdim=True)),
            stage_active.float().mean(-1, keepdim=True),
        ],
        dim=-1,
    )
    result = torch.cat([stage_stats.reshape(batch, -1), global_stats], dim=-1)
    result = torch.nan_to_num(
        result, nan=0.0, posinf=20.0, neginf=-20.0
    ).clamp(-20.0, 20.0)
    if result.shape != (batch, V14_ISIS_FEATURE_DIM):
        raise RuntimeError(f"V14 ISIS feature shape drift: {tuple(result.shape)}")
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("V14 ISIS features are nonfinite")
    return result


class DirectPenaltyMLPQuerySelectorV14ISISConditioned(
    DirectPenaltyMLPQuerySelectorV13XYTeacher
):
    """Final selector conditioned on a fixed actual hard-ISIS28 pilot path."""

    def __init__(
        self,
        *,
        hidden: int = V14_HIDDEN_DIM,
        depth: int = V14_MLP_DEPTH,
        isis_feat_dim: int = V14_ISIS_FEATURE_DIM,
        **kwargs: Any,
    ) -> None:
        if int(hidden) != V14_HIDDEN_DIM or int(depth) != V14_MLP_DEPTH:
            raise ValueError("V14 fixes hidden=512 and MLP depth=4")
        if int(isis_feat_dim) != V14_ISIS_FEATURE_DIM:
            raise ValueError(
                f"V14 fixes ISIS feature width={V14_ISIS_FEATURE_DIM}"
            )
        super().__init__(hidden=int(hidden), depth=int(depth), **kwargs)
        self.isis_proj = nn.Sequential(
            nn.LayerNorm(V14_ISIS_FEATURE_DIM),
            nn.Linear(V14_ISIS_FEATURE_DIM, int(hidden)),
        )
        self.isis_net = ResidualMLP(
            int(hidden),
            int(hidden),
            int(hidden),
            depth=int(depth),
            zero_init_last=False,
            act=nn.GELU,
        )
        self.fusion_net = ResidualMLP(
            6 * int(hidden),
            int(hidden),
            int(hidden),
            depth=int(depth),
            zero_init_last=False,
            act=nn.GELU,
        )
        self.selector_version = V14_SELECTOR_VERSION
        self.isis_feature_dim = V14_ISIS_FEATURE_DIM
        self.isis_stages = V14_ISIS_STAGES
        self.isis_topk = V14_ISIS_TOPK
        self.pilot_effective_ratio = V14_PILOT_EFFECTIVE_RATIO

    @torch.no_grad()
    def fixed_pilot_query(self, lambda_max: Tensor) -> dict[str, Tensor | str]:
        if lambda_max.ndim != 1:
            raise ValueError("V14 pilot lambda_max must be a batch vector")
        batch = int(lambda_max.shape[0])
        direct = self.penalty_parameter_head.parameter_mean.to(lambda_max)[None]
        direct = direct.expand(batch, -1)
        encoded = build_penalty(
            direct,
            derivative_floor=self.penalty_parameter_head.derivative_floor,
            squareplus_scale=self.penalty_parameter_head.squareplus_scale,
            knots=self.penalty_parameter_head.penalty_grid,
        )
        ratio = lambda_max.new_full((batch,), V14_PILOT_EFFECTIVE_RATIO)
        scaled = lambda_from_effective_ratio_v56(
            ratio, lambda_max.float(), encoded["d0"]
        )
        return {
            "lambdas": scaled["lambdas"][:, None],
            "lambda_ratio": ratio,
            "effective_lambda_ratio": scaled["effective_ratio"],
            "penalty_shape_features": encoded["shape_features"][:, None, :],
            "penalty_d0": encoded["d0"][:, None],
            "pilot_contract": (
                "fixed_cache_mean_direct96_effective_ratio_0p1_"
                "trainonly_actual_hard_isis28_v1"
            ),
        }

    def architecture_report(self) -> dict[str, Any]:
        def report(module: nn.Module) -> dict[str, int]:
            return {
                "residual_mlp_depth_contract": V14_MLP_DEPTH,
                "linear_modules": sum(
                    isinstance(item, nn.Linear) for item in module.modules()
                ),
                "parameters": sum(p.numel() for p in module.parameters()),
            }

        return {
            "selector_version": V14_SELECTOR_VERSION,
            "hidden_dim": V14_HIDDEN_DIM,
            "all_mlp_depth": V14_MLP_DEPTH,
            "response_feature_dim": V13_RESPONSE_FEATURE_DIM,
            "isis_feature_dim": V14_ISIS_FEATURE_DIM,
            "isis_stages": V14_ISIS_STAGES,
            "isis_topk": V14_ISIS_TOPK,
            "pilot_effective_ratio": V14_PILOT_EFFECTIVE_RATIO,
            "pilot_is_train_only_actual_hard_isis28": True,
            "raw_y_token_branch_present": False,
            "reference_path_decoder_features_present": False,
            "response_mlp": report(self.response_net),
            "isis_mlp": report(self.isis_net),
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
        isis_feat: Tensor,
    ) -> dict[str, Tensor | str]:
        if response_feat.shape[-1] != V13_RESPONSE_FEATURE_DIM:
            raise ValueError("V14 response feature width drift")
        if isis_feat.shape[-1] != V14_ISIS_FEATURE_DIM:
            raise ValueError("V14 ISIS feature width drift")
        geometry = self.ds_proj(ds_feat.float()) + self.t_proj(t_feat.float())
        response = self.response_net(self.response_proj(response_feat.float()))
        isis = self.isis_net(self.isis_proj(isis_feat.float()))
        geometry_unit = torch.nn.functional.layer_norm(
            geometry.float(), (geometry.shape[-1],)
        )
        response_unit = torch.nn.functional.layer_norm(
            response.float(), (response.shape[-1],)
        )
        isis_unit = torch.nn.functional.layer_norm(
            isis.float(), (isis.shape[-1],)
        )
        fused = self.fusion_net(
            torch.cat(
                [
                    geometry_unit,
                    response_unit,
                    isis_unit,
                    geometry_unit * response_unit,
                    geometry_unit * isis_unit,
                    response_unit * isis_unit,
                ],
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


__all__ = [
    "DirectPenaltyMLPQuerySelectorV14ISISConditioned",
    "V14_HIDDEN_DIM",
    "V14_ISIS_FEATURE_DIM",
    "V14_ISIS_GLOBAL_DIM",
    "V14_ISIS_STAGE_DIM",
    "V14_ISIS_STAGES",
    "V14_ISIS_TOPK",
    "V14_MLP_DEPTH",
    "V14_PILOT_EFFECTIVE_RATIO",
    "V14_SELECTOR_VERSION",
    "hard_isis28_path_features",
]
