#!/usr/bin/env python3
"""Selector V21 with concatenative dimension context and integrated rho'.

V21 retains the V20 full-P K32/S32 residual-ISIS trajectory and symmetric
context-y branch. It changes the query decoder in three ways:

* projected interaction, coefficient, stage, response, and dimension codes are
  concatenated before the fusion network;
* the public lambda output is a raw lambda in the beta solver's training
  support [1e-6, 10]; and
* the penalty head emits derivative-knot logits only. Positive derivatives are
  integrated on the canonical nonuniform grid and jointly normalized so that
  rho(0)=0 and rho(6)=1.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .canonical_penalty import (
    DIRECT_PARAMETER_DIM,
    canonical_grid,
    derivative_knots_to_relu2,
    native196_from_derivative_knots,
)
from .selector_v17_k32s32 import (
    V17_HIDDEN_DIM,
    V17_MAX_PROBE_ROWS,
    V17_STAGES,
    V17_STAGE_GAIN_RATE,
    V17_TOPK,
)
from .selector_v19_k32s32 import V19_FEATURE_CHUNK_SIZE
from .selector_v20_yconditioned import (
    DirectPenaltyMLPQuerySelectorV20YConditioned,
)


Tensor = torch.Tensor

V21_SELECTOR_VERSION = (
    "v21_k32s32_fullp_isis_dimensionconcat_yconditioned_"
    "rawlambda1e-6to10_derivative96_softplus_trapezoid_rho6_v1"
)
V21_RAW_LAMBDA_MIN = 1e-6
V21_RAW_LAMBDA_MAX = 10.0
V21_INITIAL_RAW_LAMBDA = 0.1
V21_DERIVATIVE_EPS = 1e-8
V21_DIMENSION_MEAN = (
    math.log(1_000.0),
    math.log(5_000.0),
    math.log(0.1),
)
V21_DIMENSION_SCALE = (2.0, 2.5, 2.0)
V21_FUSION_BRANCHES = (
    "residual_gram",
    "beta_trajectory",
    "stage_metadata",
    "context_y",
    "dimension_scale",
)
V21_V20_SHARED_STATE_PREFIXES = (
    "row_query.",
    "row_key.",
    "interaction_proj.",
    "residual_coefficient_proj.",
    "marginal_coefficient_proj.",
    "cumulative_coefficient_proj.",
    "coefficient_concat_proj.",
    "stage_metadata_proj.",
    "response_token_projection.",
    "response_pool_projection.",
)


def _softplus_inverse(value: float) -> float:
    if value <= 0.0:
        raise ValueError("softplus inverse requires a positive value")
    return math.log(math.expm1(float(value)))


class IntegratedDerivativePenaltyHeadV21(nn.Module):
    """Map a task code to one internally consistent Direct96 penalty."""

    def __init__(
        self,
        hidden: int,
        *,
        derivative_eps: float = V21_DERIVATIVE_EPS,
        head_init_std: float = 1e-3,
    ) -> None:
        super().__init__()
        if int(hidden) <= 0:
            raise ValueError("V21 penalty hidden width must be positive")
        if float(derivative_eps) <= 0.0:
            raise ValueError("V21 derivative epsilon must be positive")
        self.hidden = int(hidden)
        self.derivative_eps = float(derivative_eps)
        self.projection = nn.Linear(self.hidden, DIRECT_PARAMETER_DIM)
        nn.init.normal_(
            self.projection.weight, mean=0.0, std=float(head_init_std)
        )
        nn.init.constant_(
            self.projection.bias, _softplus_inverse(1.0 / 6.0)
        )
        grid = canonical_grid()
        self.register_buffer("penalty_grid", grid)
        reference = torch.zeros(DIRECT_PARAMETER_DIM, dtype=torch.float32)
        reference[0] = 1.0 / 6.0
        self.register_buffer("parameter_mean", reference)
        self.register_buffer("explained_variance", torch.tensor(0.0))

        # Compatibility attributes used only by inherited diagnostics.
        self.derivative_floor = 1e-6
        self.squareplus_scale = 2e-2
        self.latent_rank = DIRECT_PARAMETER_DIM
        self._last_parameter_proposal: Tensor | None = None

    def fit_parameter_cache(
        self,
        direct_parameters: Tensor,
    ) -> dict[str, Any]:
        if direct_parameters.ndim != 2:
            raise ValueError("V21 calibration cache must be a matrix")
        if direct_parameters.shape[-1] != DIRECT_PARAMETER_DIM:
            raise ValueError("V21 calibration cache width drift")
        return {
            "mode": "derivative_only_no_cache_fit",
            "cache_rows_audited": int(direct_parameters.shape[0]),
            "initial_penalty": "rho(t)=t/6",
            "rho0": 0.0,
            "rho6": 1.0,
        }

    def forward(self, task_code: Tensor) -> dict[str, Tensor]:
        logits = self.projection(task_code).float()
        if logits.shape[-1] != DIRECT_PARAMETER_DIM:
            raise RuntimeError("V21 derivative-logit width drift")
        if not bool(torch.isfinite(logits).all().item()):
            raise RuntimeError("V21 derivative logits are nonfinite")
        if logits.requires_grad:
            logits.retain_grad()
        self._last_parameter_proposal = logits

        positive = F.softplus(logits) + self.derivative_eps
        grid = self.penalty_grid.to(positive)
        mass = torch.trapezoid(positive, grid, dim=-1)
        if not bool(torch.isfinite(mass).all().item()) or bool(
            (mass <= self.derivative_eps).any().item()
        ):
            raise RuntimeError("V21 derivative integral is invalid")
        derivative = positive / mass[..., None]
        encoded = native196_from_derivative_knots(derivative, grid)
        relu_b, relu_weights = derivative_knots_to_relu2(derivative, grid)
        rho = encoded["rho"]
        rho6_error = (rho[..., -1] - 1.0).abs()
        if not bool(torch.isfinite(rho).all().item()) or bool(
            (rho6_error > 2e-5).any().item()
        ):
            raise RuntimeError("V21 penalty normalization contract failed")

        zeros = torch.zeros_like(mass)
        encoded.update(
            {
                "direct_parameters": logits,
                "proposal_drho": logits,
                "positive_drho": positive,
                "positivity_jacobian": torch.sigmoid(logits),
                "mass": mass,
                "relu_b": relu_b,
                "relu_weights": relu_weights,
                "knots": grid,
                "min_drho": derivative.amin(dim=-1),
                "rho6_error": rho6_error,
                "dead_last_weight_max_abs": relu_weights[..., -1].abs(),
                "latent_raw": logits,
                "latent_unit": torch.sigmoid(logits),
                "validity_alpha": torch.ones_like(mass),
                "validity_danger": zeros,
                "validity_projection_active": torch.zeros_like(
                    mass, dtype=torch.bool
                ),
            }
        )
        return encoded


class _ConcatenativeFusionV21(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        hidden = int(hidden)
        input_width = len(V21_FUSION_BRANCHES) * hidden
        self.input_width = input_width
        self.hidden = hidden
        self.input_norm = nn.LayerNorm(input_width)
        self.input_projection = nn.Linear(input_width, hidden)
        self.input_activation = nn.GELU()
        self.residual_norms = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(3)]
        )
        self.residual_linears = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(3)]
        )
        self.residual_activation = nn.GELU()
        self.output_norm = nn.LayerNorm(hidden)

    def forward(self, values: Tensor) -> Tensor:
        hidden = self.input_activation(
            self.input_projection(self.input_norm(values))
        )
        for norm, linear in zip(
            self.residual_norms, self.residual_linears, strict=True
        ):
            hidden = hidden + self.residual_activation(linear(norm(hidden)))
        return self.output_norm(hidden)


class DirectPenaltyMLPQuerySelectorV21DimensionDerivative(
    DirectPenaltyMLPQuerySelectorV20YConditioned
):
    """V20 trajectory with raw lambda and derivative-integrated penalty."""

    def __init__(
        self,
        *,
        hidden: int = V17_HIDDEN_DIM,
        raw_lambda_min: float = V21_RAW_LAMBDA_MIN,
        raw_lambda_max: float = V21_RAW_LAMBDA_MAX,
        initial_raw_lambda: float = V21_INITIAL_RAW_LAMBDA,
        derivative_eps: float = V21_DERIVATIVE_EPS,
        derivative_head_init_std: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("lambda_min_ratio", None)
        kwargs.pop("lambda_max_ratio", None)
        kwargs.pop("initial_lambda_ratio", None)
        kwargs.pop("derivative_floor", None)
        kwargs.pop("squareplus_scale", None)
        kwargs.pop("direct_head_init_std", None)
        super().__init__(
            hidden=hidden,
            lambda_min_ratio=float(raw_lambda_min),
            lambda_max_ratio=float(raw_lambda_max),
            initial_lambda_ratio=float(initial_raw_lambda),
            derivative_floor=1e-6,
            squareplus_scale=2e-2,
            direct_head_init_std=1e-3,
            **kwargs,
        )
        self.raw_lambda_min = float(raw_lambda_min)
        self.raw_lambda_max = float(raw_lambda_max)
        self.initial_raw_lambda = float(initial_raw_lambda)
        self.fusion_mlp = _ConcatenativeFusionV21(self.hidden)
        self.dimension_context_projection = nn.Sequential(
            nn.Linear(3, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
        )
        self.register_buffer(
            "dimension_context_mean",
            torch.tensor(V21_DIMENSION_MEAN, dtype=torch.float32),
        )
        self.register_buffer(
            "dimension_context_scale",
            torch.tensor(V21_DIMENSION_SCALE, dtype=torch.float32),
        )
        self.penalty_parameter_head = IntegratedDerivativePenaltyHeadV21(
            self.hidden,
            derivative_eps=float(derivative_eps),
            head_init_std=float(derivative_head_init_std),
        )
        self.selector_version = V21_SELECTOR_VERSION

    def encode_context(
        self,
        x_context: Tensor,
        y_context: Tensor,
        *,
        feature_mask: Tensor | None = None,
    ) -> dict[str, Tensor | int | float | str | bool]:
        trajectory = dict(
            super().encode_context(
                x_context,
                y_context,
                feature_mask=feature_mask,
            )
        )
        batch, rows, features = x_context.shape
        context_n = torch.full(
            (batch,),
            float(rows),
            dtype=torch.float32,
            device=x_context.device,
        )
        if feature_mask is None:
            context_p = torch.full(
                (batch,),
                float(features),
                dtype=torch.float32,
                device=x_context.device,
            )
        else:
            if feature_mask.shape != (batch, features):
                raise ValueError("V21 feature mask shape drift")
            context_p = feature_mask.float().sum(dim=-1)
        if bool((context_n <= 0.0).any().item()) or bool(
            (context_p <= 0.0).any().item()
        ):
            raise ValueError("V21 actual N/P must be positive")
        trajectory.update(
            {
                "v21_context_n": context_n,
                "v21_context_p": context_p,
                "v21_actual_dimension_source": (
                    "tensor_rows_and_valid_feature_mask"
                ),
            }
        )
        return trajectory

    def _build_fusion_input(
        self,
        *,
        interaction_code: Tensor,
        coefficient_code: Tensor,
        metadata_code: Tensor,
        trajectory: dict[str, Tensor | int | float | str | bool],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        response_probe = trajectory.get("response_probe")
        context_n = trajectory.get("v21_context_n")
        context_p = trajectory.get("v21_context_p")
        lambda_max = trajectory.get("v21_lambda_max")
        if not all(
            torch.is_tensor(value)
            for value in (response_probe, context_n, context_p, lambda_max)
        ):
            raise TypeError("V21 fusion context is incomplete")
        response_probe = response_probe.float()
        context_n = context_n.float()
        context_p = context_p.float()
        lambda_max = lambda_max.float()
        batch = interaction_code.shape[0]
        if response_probe.ndim != 2 or response_probe.shape[0] != batch:
            raise ValueError("V21 response probe shape drift")
        if any(value.shape != (batch,) for value in (context_n, context_p, lambda_max)):
            raise ValueError("V21 dimension context shape drift")

        response_tokens = self.response_token_projection(
            response_probe.unsqueeze(-1)
        )
        response_pooled = torch.cat(
            [
                response_tokens.mean(dim=1),
                response_tokens.std(dim=1, unbiased=False),
                response_tokens.amax(dim=1),
            ],
            dim=-1,
        )
        response_code = self.response_pool_projection(response_pooled)
        dimension_raw = torch.stack(
            [
                context_n.clamp_min(1.0).log(),
                context_p.clamp_min(1.0).log(),
                lambda_max.clamp_min(1e-12).log(),
            ],
            dim=-1,
        )
        dimension_normalized = (
            dimension_raw - self.dimension_context_mean.to(dimension_raw)
        ) / self.dimension_context_scale.to(dimension_raw)
        dimension_code = self.dimension_context_projection(
            dimension_normalized
        )
        fusion_input = torch.cat(
            [
                interaction_code,
                coefficient_code,
                metadata_code,
                response_code,
                dimension_code,
            ],
            dim=-1,
        )
        return fusion_input, {
            "v20_response_tokens": response_tokens,
            "v20_response_pooled": response_pooled,
            "v20_response_code": response_code,
            "v21_dimension_context_raw": dimension_raw,
            "v21_dimension_context_normalized": dimension_normalized,
            "v21_dimension_context_code": dimension_code,
            "v21_fusion_input": fusion_input,
            "v21_interaction_code": interaction_code,
            "v21_coefficient_code": coefficient_code,
            "v21_metadata_code": metadata_code,
            "v21_response_code": response_code,
        }

    def forward_from_trajectory(
        self,
        trajectory: dict[str, Tensor | int | float | str | bool],
        lambda_max: Tensor,
        *,
        return_representation: bool = False,
    ) -> dict[str, Any]:
        extended = dict(trajectory)
        extended["v21_lambda_max"] = lambda_max
        residual = extended["residual_trajectory"]
        beta_residual = extended["beta_residual"]
        beta_marginal = extended["beta_marginal"]
        beta_cumulative = extended["beta_cumulative"]
        stage_active = extended["stage_active"]
        residual_scale_before = extended["residual_scale_before"]
        stage_explained_fraction = extended["stage_explained_fraction"]
        if not all(
            torch.is_tensor(value)
            for value in (
                residual,
                beta_residual,
                beta_marginal,
                beta_cumulative,
                stage_active,
                residual_scale_before,
                stage_explained_fraction,
            )
        ):
            raise TypeError("V21 trajectory tensor contract is incomplete")

        residual = residual.float()
        beta_residual = beta_residual.float()
        beta_marginal = beta_marginal.float()
        beta_cumulative = beta_cumulative.float()
        stage_active = stage_active.bool()
        residual_scale_before = residual_scale_before.float()
        stage_explained_fraction = stage_explained_fraction.float()
        batch, probe_rows, stages = residual.shape
        if stages != self.stages:
            raise ValueError("V21 residual stage width drift")
        if beta_residual.shape != (batch, self.topk, self.stages):
            raise ValueError("V21 residual coefficient width drift")
        if beta_marginal.shape != beta_residual.shape:
            raise ValueError("V21 marginal coefficient width drift")
        if beta_cumulative.shape != (
            batch,
            self.cumulative_slot_width,
            self.stages,
        ):
            raise ValueError("V21 cumulative coefficient width drift")
        if stage_active.shape != (batch, self.stages):
            raise ValueError("V21 stage mask width drift")
        if residual_scale_before.shape != (batch, self.stages):
            raise ValueError("V21 residual scale width drift")
        if stage_explained_fraction.shape != (batch, self.stages):
            raise ValueError("V21 stage explained-fraction width drift")
        if lambda_max.shape != (batch,):
            raise ValueError(
                f"V21 lambda_max must be {(batch,)}, got {tuple(lambda_max.shape)}"
            )

        stage_mask = stage_active[:, None, :].float()
        residual_masked = residual * stage_mask
        flat_rows = residual_masked.reshape(batch * probe_rows, self.stages)
        row_query = self.row_query(flat_rows).reshape(
            batch, probe_rows, self.stages
        ) * stage_mask
        row_key = self.row_key(flat_rows).reshape(
            batch, probe_rows, self.stages
        ) * stage_mask
        interaction = torch.einsum(
            "bns,bnt->bst", row_query, row_key
        ) / float(max(1, probe_rows))
        interaction_code = self.interaction_proj(
            interaction.reshape(batch, -1)
        )

        coefficient_mask = stage_active[:, None, :].float()
        beta_residual = beta_residual * coefficient_mask
        beta_marginal = beta_marginal * coefficient_mask
        beta_cumulative = beta_cumulative * coefficient_mask
        residual_code = self.residual_coefficient_proj(
            beta_residual.reshape(batch, -1)
        )
        marginal_code = self.marginal_coefficient_proj(
            beta_marginal.reshape(batch, -1)
        )
        cumulative_code = self.cumulative_coefficient_proj(
            beta_cumulative.reshape(batch, -1)
        )
        coefficient_concat = torch.cat(
            [residual_code, marginal_code, cumulative_code], dim=-1
        )
        coefficient_code = self.coefficient_concat_proj(coefficient_concat)

        metadata_mask = stage_active.float()
        stage_metadata = torch.cat(
            [
                metadata_mask,
                torch.log(
                    residual_scale_before.clamp_min(self.residual_scale_floor)
                )
                * metadata_mask,
                stage_explained_fraction.clamp(0.0, 1.0) * metadata_mask,
            ],
            dim=-1,
        )
        metadata_code = self.stage_metadata_proj(stage_metadata)
        fusion_input, fusion_representation = self._build_fusion_input(
            interaction_code=interaction_code,
            coefficient_code=coefficient_code,
            metadata_code=metadata_code,
            trajectory=extended,
        )
        task_code = self.fusion_mlp(fusion_input)

        lambda_unit = torch.sigmoid(self.lambda_head(task_code).squeeze(-1))
        raw_lambda = torch.exp(
            math.log(self.raw_lambda_min)
            + (
                math.log(self.raw_lambda_max)
                - math.log(self.raw_lambda_min)
            )
            * lambda_unit
        ).clamp(self.raw_lambda_min, self.raw_lambda_max)
        penalty_code = self.penalty_mlp(task_code)
        encoded = self.penalty_parameter_head(penalty_code)
        shape = encoded["shape_features"]
        derivative = encoded["drho"].float()
        d0 = encoded["d0"].float()
        lmax = lambda_max.float().clamp_min(1e-12)
        max_drho = derivative.amax(dim=-1)
        raw_over_lmax = raw_lambda / lmax
        force_ratio = raw_lambda * max_drho / lmax
        zero_kkt_ratio = raw_lambda * d0 / lmax
        concavity_violations = (
            derivative[..., 1:] - derivative[..., :-1] > 1e-7
        ).float()
        reference_derivative = torch.full_like(derivative, 1.0 / 6.0)
        reference_shape = native196_from_derivative_knots(
            reference_derivative,
            self.penalty_parameter_head.penalty_grid.to(derivative),
        )["shape_features"]
        reference_distance = (
            (shape - reference_shape).square().mean(-1).sqrt()
            / reference_shape.square().mean(-1).sqrt().clamp_min(1e-8)
        )

        result: dict[str, Any] = {
            "lambdas": raw_lambda[:, None],
            "raw_lambda": raw_lambda,
            "lambda_ratio": raw_over_lmax,
            "raw_lambda_over_lambda_max": raw_over_lmax,
            "effective_lambda_ratio": force_ratio,
            "force_ratio": force_ratio,
            "zero_kkt_ratio": zero_kkt_ratio,
            "penalty_shape_features": shape[:, None, :],
            "penalty_d0": d0[:, None],
            "penalty_relu_b": encoded["relu_b"],
            "penalty_relu_weights": encoded["relu_weights"],
            "penalty_rho": encoded["rho"],
            "penalty_drho": derivative,
            "penalty_metadata4": encoded["metadata4"],
            "penalty_latent_raw": encoded["latent_raw"],
            "penalty_latent_unit": encoded["latent_unit"],
            "penalty_validity_alpha": encoded["validity_alpha"],
            "penalty_validity_danger": encoded["validity_danger"],
            "penalty_validity_projection_active": encoded[
                "validity_projection_active"
            ],
            "penalty_rho6_error": encoded["rho6_error"],
            "penalty_min_drho": encoded["min_drho"],
            "penalty_max_drho": max_drho,
            "penalty_derivative_tv": encoded["metadata4"][..., 3],
            "penalty_positivity_jacobian_min": encoded[
                "positivity_jacobian"
            ].amin(-1),
            "penalty_reference_distance": reference_distance,
            "penalty_reference_slope": max_drho,
            "penalty_ridge_branch": d0 <= 1e-7,
            "penalty_parameter_explained_variance": (
                self.penalty_parameter_head.explained_variance.to(shape)
            ),
            "penalty_concavity_violation_fraction": (
                concavity_violations.mean(dim=-1)
            ),
            "penalty_encoding_version": self.selector_version,
            "lambda_output_contract": "raw_lambda_direct_1e-6_to_10",
        }
        if return_representation:
            result.update(
                {
                    "v17_stage_interaction": interaction,
                    "v17_task_code": task_code,
                    "v17_penalty_code": penalty_code,
                    "v17_trajectory": extended,
                    "v19_stage_interaction": interaction,
                    "v19_residual_coefficient_code": residual_code,
                    "v19_marginal_coefficient_code": marginal_code,
                    "v19_cumulative_coefficient_code": cumulative_code,
                    "v19_coefficient_concat": coefficient_concat,
                    "v19_coefficient_code": coefficient_code,
                    "v19_task_code": task_code,
                    "v19_penalty_code": penalty_code,
                    "v19_trajectory": extended,
                    "v20_task_code": task_code,
                    "v20_penalty_code": penalty_code,
                    "v20_trajectory": extended,
                    "v21_task_code": task_code,
                    "v21_penalty_code": penalty_code,
                    "v21_trajectory": extended,
                    **fusion_representation,
                }
            )
        return result

    def architecture_report(self) -> dict[str, Any]:
        report = super().architecture_report()
        report.update(
            {
                "selector_version": self.selector_version,
                "architecture_generation": 21,
                "v21_lineage": (
                    "v20_fullp_yconditioned_plus_dimension_concat_"
                    "rawlambda_plus_integrated_derivative"
                ),
                "fusion_branches": list(V21_FUSION_BRANCHES),
                "fusion_mode": "project_each_then_concat_5H_layernorm_residual512",
                "fusion_input_width": len(V21_FUSION_BRANCHES) * self.hidden,
                "dimension_context": [
                    "log_actual_N",
                    "log_actual_P",
                    "log_context_lambda_max",
                ],
                "dimension_context_fixed_mean": list(V21_DIMENSION_MEAN),
                "dimension_context_fixed_scale": list(V21_DIMENSION_SCALE),
                "dimension_projection": [3, self.hidden, self.hidden],
                "lambda_public_output": "raw_lambda",
                "raw_lambda_support": [
                    self.raw_lambda_min,
                    self.raw_lambda_max,
                ],
                "effective_ratios_are_telemetry_only": True,
                "penalty_head_output": "96_positive_derivative_knots_only",
                "penalty_positive_map": "softplus_plus_1e-8",
                "penalty_integration": "nonuniform_grid_trapezoid",
                "penalty_scale_constraint": "rho(0)=0_rho(6)=1",
                "rho_and_drho_independent_heads": False,
                "concavity_constrained": False,
                "total_parameters": sum(p.numel() for p in self.parameters()),
            }
        )
        return report


def load_v20_representation_state_dict(
    selector: DirectPenaltyMLPQuerySelectorV21DimensionDerivative,
    source_state_dict: Mapping[str, Tensor],
) -> dict[str, Any]:
    """Import only the V20 trajectory and response representation weights."""

    target = selector.state_dict()
    source_keys = {
        key
        for key in source_state_dict
        if key.startswith(V21_V20_SHARED_STATE_PREFIXES)
    }
    target_keys = {
        key
        for key in target
        if key.startswith(V21_V20_SHARED_STATE_PREFIXES)
    }
    if source_keys != target_keys:
        raise RuntimeError(
            "V20-to-V21 shared representation key mismatch: "
            f"missing={sorted(target_keys - source_keys)}, "
            f"unexpected={sorted(source_keys - target_keys)}"
        )
    imported_parameters = 0
    for key in sorted(target_keys):
        source = source_state_dict[key]
        destination = target[key]
        if source.shape != destination.shape or source.dtype != destination.dtype:
            raise RuntimeError(
                f"V20-to-V21 tensor contract mismatch for {key}: "
                f"source={tuple(source.shape)}/{source.dtype}, "
                f"target={tuple(destination.shape)}/{destination.dtype}"
            )
        target[key] = source.detach().clone()
        imported_parameters += int(source.numel())
    incompatible = selector.load_state_dict(target, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("V20-to-V21 strict merged-state load failed")
    forbidden_prefixes = (
        "fusion_mlp.",
        "dimension_context_projection.",
        "lambda_head.",
        "penalty_mlp.",
        "penalty_parameter_head.",
    )
    forbidden = sorted(
        key
        for key in source_keys
        if key.startswith(forbidden_prefixes)
    )
    if forbidden:
        raise RuntimeError(
            f"V20 output/new branch state was imported: {forbidden}"
        )
    return {
        "source": "V20 selector representation",
        "imported_keys": sorted(source_keys),
        "imported_tensor_count": len(source_keys),
        "imported_parameter_count": imported_parameters,
        "shared_prefixes": list(V21_V20_SHARED_STATE_PREFIXES),
        "fresh_prefixes": list(forbidden_prefixes),
        "strict_key_audit": True,
    }


__all__ = [
    "DirectPenaltyMLPQuerySelectorV21DimensionDerivative",
    "IntegratedDerivativePenaltyHeadV21",
    "load_v20_representation_state_dict",
    "V19_FEATURE_CHUNK_SIZE",
    "V21_DERIVATIVE_EPS",
    "V21_FUSION_BRANCHES",
    "V21_INITIAL_RAW_LAMBDA",
    "V21_RAW_LAMBDA_MAX",
    "V21_RAW_LAMBDA_MIN",
    "V21_SELECTOR_VERSION",
    "V21_V20_SHARED_STATE_PREFIXES",
    "V17_HIDDEN_DIM",
    "V17_MAX_PROBE_ROWS",
    "V17_STAGES",
    "V17_STAGE_GAIN_RATE",
    "V17_TOPK",
]
