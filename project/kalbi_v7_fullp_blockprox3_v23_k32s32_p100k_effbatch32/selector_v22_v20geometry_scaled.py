#!/usr/bin/env python3
"""V22: V20 selector with calibrated support-geometry conditioning."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn

from .canonical_penalty import build_penalty
from .selector_v10 import lambda_from_effective_ratio_v56
from .selector_v17_k32s32 import _ThreeLinearResidualMap
from .selector_v19_k32s32 import (
    V19_FEATURE_CHUNK_SIZE,
    V19_PENALTY_RESIDUAL_LAYERS,
    V22_CORRELATION_QUANTILES,
    V22_GEOMETRY_STAT_NAMES,
)
from .selector_v20_yconditioned import (
    DirectPenaltyMLPQuerySelectorV20YConditioned,
    V17_HIDDEN_DIM,
    V17_MAX_PROBE_ROWS,
    V17_STAGES,
    V17_STAGE_GAIN_RATE,
    V17_TOPK,
)


Tensor = torch.Tensor

V22_SELECTOR_VERSION = (
    "v22_v20canonical_k32s32_fullpisis_supportgeometry13x32_"
    "finalcorrquantile10_logNlogPlambdaMax_fixedcache_v1"
)
V22_SCALER_VERSION = "v22_fixed_source_scaler_welford_v1"
V22_SCALER_CLIP = 8.0
V22_SCALER_STD_FLOOR = 1e-6
V22_AUXILIARY_GATE_INIT = 0.05


def _scaler_shapes(stages: int, topk: int) -> dict[str, tuple[int, ...]]:
    slots = int(stages) * int(topk)
    return {
        "residual": (int(stages),),
        "interaction": (int(stages), int(stages)),
        "beta_residual": (int(topk), int(stages)),
        "beta_marginal": (int(topk), int(stages)),
        "beta_cumulative": (slots, int(stages)),
        "stage_metadata": (3 * int(stages),),
        "response": (1,),
        "geometry": (int(stages), len(V22_GEOMETRY_STAT_NAMES)),
        "correlation_quantiles": (len(V22_CORRELATION_QUANTILES),),
        "dimension": (3,),
    }


class FixedInputScalerV22(nn.Module):
    """Checkpointed fixed training-distribution source normalizers."""

    def __init__(
        self,
        *,
        stages: int,
        topk: int,
        clip: float = V22_SCALER_CLIP,
        std_floor: float = V22_SCALER_STD_FLOOR,
    ) -> None:
        super().__init__()
        self.shapes = _scaler_shapes(stages, topk)
        self.clip = float(clip)
        self.std_floor = float(std_floor)
        if self.clip <= 0.0 or self.std_floor <= 0.0:
            raise ValueError("V22 scaler clip and std floor must be positive")
        for name, shape in self.shapes.items():
            self.register_buffer(f"{name}_mean", torch.zeros(shape))
            self.register_buffer(f"{name}_std", torch.ones(shape))
            self.register_buffer(f"{name}_count", torch.zeros(shape, dtype=torch.long))
        self.register_buffer("cache_loaded", torch.tensor(False, dtype=torch.bool))
        self.cache_metadata: dict[str, Any] = {}

    def source_ready(self, name: str) -> bool:
        count = getattr(self, f"{name}_count")
        return bool((count > 0).any().item())

    @torch.no_grad()
    def load_cache_payload(self, payload: Mapping[str, Any]) -> None:
        if payload.get("version") != V22_SCALER_VERSION:
            raise RuntimeError("V22 input-scaler cache version mismatch")
        sources = payload.get("sources")
        if not isinstance(sources, Mapping):
            raise TypeError("V22 input-scaler cache has no sources")
        for name, shape in self.shapes.items():
            source = sources.get(name)
            if not isinstance(source, Mapping):
                raise RuntimeError(f"V22 scaler cache is missing {name}")
            mean = torch.as_tensor(source["mean"], dtype=torch.float32)
            std = torch.as_tensor(source["std"], dtype=torch.float32)
            count = torch.as_tensor(source["count"], dtype=torch.long)
            if (
                tuple(mean.shape) != shape
                or tuple(std.shape) != shape
                or tuple(count.shape) != shape
            ):
                raise RuntimeError(
                    f"V22 scaler shape drift for {name}: "
                    f"{tuple(mean.shape)}/{tuple(std.shape)}/{tuple(count.shape)} "
                    f"!= {shape}"
                )
            if not bool(torch.isfinite(mean).all()) or not bool(
                torch.isfinite(std).all()
            ):
                raise RuntimeError(f"V22 scaler {name} is nonfinite")
            structural_zero = count <= 0
            std = torch.where(
                structural_zero,
                torch.ones_like(std),
                std.clamp_min(self.std_floor),
            )
            mean = torch.where(structural_zero, torch.zeros_like(mean), mean)
            getattr(self, f"{name}_mean").copy_(mean)
            getattr(self, f"{name}_std").copy_(std)
            getattr(self, f"{name}_count").copy_(count)
        self.cache_metadata = dict(payload.get("metadata", {}))
        self.cache_loaded.fill_(True)

    def scale(
        self,
        name: str,
        values: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> Tensor:
        if name not in self.shapes:
            raise KeyError(f"unknown V22 scaler source {name}")
        if not self.source_ready(name):
            raise RuntimeError(f"V22 scaler source {name} is not calibrated")
        expected = self.shapes[name]
        if tuple(values.shape[-len(expected):]) != expected:
            raise ValueError(
                f"V22 scaler {name} trailing shape "
                f"{tuple(values.shape[-len(expected):])} != {expected}"
            )
        mean = getattr(self, f"{name}_mean").to(values)
        std = getattr(self, f"{name}_std").to(values).clamp_min(self.std_floor)
        scaled = ((values.float() - mean) / std).clamp(
            -self.clip, self.clip
        )
        if mask is not None:
            scaled = scaled * mask.to(scaled).float()
        return scaled

    def architecture_report(self) -> dict[str, Any]:
        return {
            "version": V22_SCALER_VERSION,
            "clip": self.clip,
            "std_floor": self.std_floor,
            "source_shapes": {
                name: list(shape) for name, shape in self.shapes.items()
            },
            "cache_loaded": bool(self.cache_loaded.item()),
            "cache_metadata": dict(self.cache_metadata),
        }


def transform_geometry_for_scaling(geometry: Tensor, eps: float = 1e-8) -> Tensor:
    if geometry.shape[-1] != len(V22_GEOMETRY_STAT_NAMES):
        raise ValueError("V22 geometry width drift")
    transformed = geometry.float().clone()
    for index in (1, 2, 4, 5, 6):
        transformed[..., index] = transformed[..., index].clamp_min(eps).log()
    transformed[..., 9] = torch.log1p(transformed[..., 9].clamp_min(0.0))
    transformed[..., 11] = transformed[..., 11].clamp_min(eps).log()
    transformed[..., 12] = torch.log1p(transformed[..., 12].clamp_min(0.0))
    return transformed


class DirectPenaltyMLPQuerySelectorV22V20GeometryScaled(
    DirectPenaltyMLPQuerySelectorV20YConditioned
):
    """V20 output contract with calibrated cumulative-support geometry."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.input_scaler = FixedInputScalerV22(
            stages=self.stages, topk=self.topk
        )
        geometry_width = self.stages * len(V22_GEOMETRY_STAT_NAMES)
        self.geometry_projection = nn.Sequential(
            nn.LayerNorm(geometry_width),
            nn.Linear(geometry_width, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
        )
        self.correlation_quantile_projection = nn.Sequential(
            nn.LayerNorm(len(V22_CORRELATION_QUANTILES)),
            nn.Linear(len(V22_CORRELATION_QUANTILES), self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
        )
        self.dimension_context_projection = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
        )
        self.auxiliary_fusion = nn.Sequential(
            nn.LayerNorm(3 * self.hidden),
            nn.Linear(3 * self.hidden, self.hidden),
            nn.GELU(),
            _ThreeLinearResidualMap(self.hidden),
        )
        self.auxiliary_gate = nn.Parameter(
            torch.full((self.hidden,), V22_AUXILIARY_GATE_INIT)
        )
        self.selector_version = V22_SELECTOR_VERSION

    @torch.no_grad()
    def load_input_scaler_cache(self, payload: Mapping[str, Any]) -> None:
        self.input_scaler.load_cache_payload(payload)

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
        trajectory["v22_context_n"] = torch.full(
            (batch,),
            float(rows),
            device=x_context.device,
            dtype=torch.float32,
        )
        trajectory["v22_context_p"] = (
            torch.full(
                (batch,),
                float(features),
                device=x_context.device,
                dtype=torch.float32,
            )
            if feature_mask is None
            else feature_mask.float().sum(dim=-1)
        )
        return trajectory

    def _source_tensors(
        self,
        trajectory: Mapping[str, Any],
        lambda_max: Tensor,
        *,
        scale_residual_for_interaction: bool,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        residual = trajectory["residual_trajectory"].float()
        beta_residual = trajectory["beta_residual"].float()
        beta_marginal = trajectory["beta_marginal"].float()
        beta_cumulative = trajectory["beta_cumulative"].float()
        stage_active = trajectory["stage_active"].bool()
        residual_scale_before = trajectory["residual_scale_before"].float()
        stage_explained_fraction = trajectory[
            "stage_explained_fraction"
        ].float()
        response_probe = trajectory["response_probe"].float()
        geometry = transform_geometry_for_scaling(
            trajectory["cumulative_geometry_raw"].float()
        )
        correlation_quantiles = trajectory[
            "support_correlation_quantiles"
        ].float()
        context_n = trajectory["v22_context_n"].float()
        context_p = trajectory["v22_context_p"].float()

        batch, probe_rows, stages = residual.shape
        stage_mask = stage_active[:, None, :]
        residual_masked = residual * stage_mask.float()
        residual_for_interaction = residual_masked
        if scale_residual_for_interaction:
            residual_for_interaction = self.input_scaler.scale(
                "residual", residual_masked, mask=stage_mask
            )
        flat_rows = residual_for_interaction.reshape(
            batch * probe_rows, stages
        )
        query = self.row_query(flat_rows).reshape(
            batch, probe_rows, stages
        ) * stage_mask.float()
        key = self.row_key(flat_rows).reshape(
            batch, probe_rows, stages
        ) * stage_mask.float()
        interaction = torch.einsum(
            "bns,bnt->bst", query, key
        ) / float(max(1, probe_rows))

        coefficient_mask = stage_active[:, None, :]
        selected_valid = trajectory["cumulative_selected_valid"].bool()
        slot_indices = torch.arange(
            self.cumulative_slot_width,
            device=residual.device,
        )[:, None]
        stage_limits = (
            torch.arange(self.stages, device=residual.device)[None, :] + 1
        ) * self.topk
        triangular = slot_indices < stage_limits
        cumulative_mask = (
            selected_valid[:, :, None]
            & stage_active[:, None, :]
            & triangular[None, :, :]
        )
        metadata_mask = torch.cat(
            [stage_active, stage_active, stage_active], dim=-1
        )
        stage_metadata = torch.cat(
            [
                stage_active.float(),
                torch.log(
                    residual_scale_before.clamp_min(
                        self.residual_scale_floor
                    )
                )
                * stage_active.float(),
                stage_explained_fraction.clamp(0.0, 1.0)
                * stage_active.float(),
            ],
            dim=-1,
        )
        response = response_probe.unsqueeze(-1)
        geometry_mask = stage_active[:, :, None]
        quantile_valid = (
            trajectory["support_correlation_valid_pair_count"].reshape(-1)
            > 0
        )[:, None]
        dimension = torch.stack(
            [
                context_n.clamp_min(1.0).log(),
                context_p.clamp_min(1.0).log(),
                lambda_max.float().clamp_min(1e-12).log(),
            ],
            dim=-1,
        )
        sources = {
            "residual": residual_masked,
            "interaction": interaction,
            "beta_residual": beta_residual * coefficient_mask.float(),
            "beta_marginal": beta_marginal * coefficient_mask.float(),
            "beta_cumulative": beta_cumulative * cumulative_mask.float(),
            "stage_metadata": stage_metadata,
            "response": response,
            "geometry": geometry * geometry_mask.float(),
            "correlation_quantiles": correlation_quantiles,
            "dimension": dimension,
        }
        masks = {
            "residual": stage_mask.expand_as(residual_masked),
            "interaction": (
                stage_active[:, :, None] & stage_active[:, None, :]
            ),
            "beta_residual": coefficient_mask.expand_as(beta_residual),
            "beta_marginal": coefficient_mask.expand_as(beta_marginal),
            "beta_cumulative": cumulative_mask,
            "stage_metadata": metadata_mask,
            "response": torch.ones_like(response, dtype=torch.bool),
            "geometry": geometry_mask.expand_as(geometry),
            "correlation_quantiles": quantile_valid.expand_as(
                correlation_quantiles
            ),
            "dimension": torch.ones_like(dimension, dtype=torch.bool),
        }
        return sources, masks

    @torch.no_grad()
    def collect_scaler_sources(
        self,
        trajectory: Mapping[str, Any],
        lambda_max: Tensor,
        *,
        scale_residual_for_interaction: bool,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        return self._source_tensors(
            trajectory,
            lambda_max,
            scale_residual_for_interaction=scale_residual_for_interaction,
        )

    def forward_from_trajectory(
        self,
        trajectory: dict[str, Tensor | int | float | str | bool],
        lambda_max: Tensor,
        *,
        return_representation: bool = False,
    ) -> dict[str, Any]:
        if not bool(self.input_scaler.cache_loaded.item()):
            raise RuntimeError("V22 input-scaling cache is not loaded")
        sources, masks = self._source_tensors(
            trajectory,
            lambda_max,
            scale_residual_for_interaction=True,
        )
        scaled = {
            name: self.input_scaler.scale(
                name, value, mask=masks[name]
            )
            for name, value in sources.items()
        }
        batch = scaled["interaction"].shape[0]
        interaction_code = self.interaction_proj(
            scaled["interaction"].reshape(batch, -1)
        )
        residual_code = self.residual_coefficient_proj(
            scaled["beta_residual"].reshape(batch, -1)
        )
        marginal_code = self.marginal_coefficient_proj(
            scaled["beta_marginal"].reshape(batch, -1)
        )
        cumulative_code = self.cumulative_coefficient_proj(
            scaled["beta_cumulative"].reshape(batch, -1)
        )
        coefficient_concat = torch.cat(
            [residual_code, marginal_code, cumulative_code], dim=-1
        )
        coefficient_code = self.coefficient_concat_proj(coefficient_concat)
        metadata_code = self.stage_metadata_proj(
            scaled["stage_metadata"]
        )
        response_tokens = self.response_token_projection(
            scaled["response"]
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
        v20_fusion_input = (
            interaction_code + coefficient_code + metadata_code + response_code
        )
        v20_task_code = self.fusion_mlp(v20_fusion_input)

        geometry_code = self.geometry_projection(
            scaled["geometry"].reshape(batch, -1)
        )
        correlation_quantile_code = self.correlation_quantile_projection(
            scaled["correlation_quantiles"]
        )
        dimension_code = self.dimension_context_projection(
            scaled["dimension"]
        )
        auxiliary_concat = torch.cat(
            [geometry_code, correlation_quantile_code, dimension_code],
            dim=-1,
        )
        auxiliary_code = self.auxiliary_fusion(auxiliary_concat)
        task_code = (
            v20_task_code
            + self.auxiliary_gate.to(v20_task_code) * auxiliary_code
        )

        lambda_unit = torch.sigmoid(
            self.lambda_head(task_code).squeeze(-1)
        )
        lambda_ratio = torch.exp(
            math.log(self.lambda_min_ratio)
            + (
                math.log(self.lambda_max_ratio)
                - math.log(self.lambda_min_ratio)
            )
            * lambda_unit
        ).clamp(self.lambda_min_ratio, self.lambda_max_ratio)
        penalty_code = self.penalty_mlp(task_code)
        encoded = self.penalty_parameter_head(penalty_code)
        shape = encoded["shape_features"]
        d0 = encoded["d0"]
        reference_slope = self._lambda_reference_slope(encoded)
        lambda_scale = lambda_from_effective_ratio_v56(
            lambda_ratio,
            lambda_max.float(),
            d0,
            reference_slope=reference_slope,
        )
        reference_direct = self.penalty_parameter_head.parameter_mean.to(
            shape
        )[None]
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
        raw_lambda = lambda_scale["lambdas"]
        lambda_max_safe = lambda_max.float().clamp_min(1e-12)
        raw_lambda_over_lambda_max = raw_lambda / lambda_max_safe
        force_ratio = (
            raw_lambda * encoded["drho"].float().amax(dim=-1)
            / lambda_max_safe
        )
        zero_kkt_ratio = raw_lambda * d0.float() / lambda_max_safe
        concavity_violations = (
            encoded["drho"][..., 1:] - encoded["drho"][..., :-1] > 1e-7
        ).float()
        result: dict[str, Any] = {
            "lambdas": raw_lambda[:, None],
            "raw_lambda": raw_lambda,
            "lambda_ratio": lambda_ratio,
            "raw_lambda_over_lambda_max": raw_lambda_over_lambda_max,
            "effective_lambda_ratio": lambda_scale["effective_ratio"],
            "force_ratio": force_ratio,
            "zero_kkt_ratio": zero_kkt_ratio,
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
            "penalty_concavity_violation_fraction": (
                concavity_violations.mean(dim=-1)
            ),
            "penalty_parameter_explained_variance": (
                self.penalty_parameter_head.explained_variance.to(shape)
            ),
            "penalty_encoding_version": self.selector_version,
            "lambda_output_contract": (
                "v20_effective_ratio_head_converted_once_to_raw_lambda"
            ),
        }
        if return_representation:
            result.update(
                {
                    "v22_task_code": task_code,
                    "v22_v20_task_code": v20_task_code,
                    "v22_penalty_code": penalty_code,
                    "v22_geometry_code": geometry_code,
                    "v22_correlation_quantile_code": correlation_quantile_code,
                    "v22_dimension_code": dimension_code,
                    "v22_auxiliary_code": auxiliary_code,
                    "v22_auxiliary_gate": self.auxiliary_gate,
                    "v22_scaled_sources": scaled,
                    "v22_trajectory": trajectory,
                    "v17_stage_interaction": scaled["interaction"],
                    "v17_task_code": task_code,
                    "v17_penalty_code": penalty_code,
                    "v17_trajectory": trajectory,
                    "v19_stage_interaction": scaled["interaction"],
                    "v19_residual_coefficient_code": residual_code,
                    "v19_marginal_coefficient_code": marginal_code,
                    "v19_cumulative_coefficient_code": cumulative_code,
                    "v19_coefficient_concat": coefficient_concat,
                    "v19_coefficient_code": coefficient_code,
                    "v19_task_code": task_code,
                    "v19_penalty_code": penalty_code,
                    "v19_trajectory": trajectory,
                    "v20_response_tokens": response_tokens,
                    "v20_response_pooled": response_pooled,
                    "v20_response_code": response_code,
                    "v20_fusion_input": v20_fusion_input,
                    "v20_task_code": task_code,
                    "v20_penalty_code": penalty_code,
                    "v20_trajectory": trajectory,
                    "v21_interaction_code": interaction_code,
                    "v21_coefficient_code": coefficient_code,
                    "v21_metadata_code": metadata_code,
                    "v21_response_code": response_code,
                    "v21_dimension_context_code": dimension_code,
                    "v21_task_code": task_code,
                    "v21_penalty_code": penalty_code,
                    "v21_trajectory": trajectory,
                }
            )
        return result

    def architecture_report(self) -> dict[str, Any]:
        report = super().architecture_report()
        report.update(
            {
                "selector_version": self.selector_version,
                "architecture_generation": 22,
                "lineage": "v20_plus_fixedscaled_support_geometry",
                "v20_lambda_contract_preserved": True,
                "v20_canonical_direct96_preserved": True,
                "geometry_stat_names": list(V22_GEOMETRY_STAT_NAMES),
                "geometry_shape": [
                    self.stages,
                    len(V22_GEOMETRY_STAT_NAMES),
                ],
                "correlation_quantiles": list(V22_CORRELATION_QUANTILES),
                "dimension_context": ["log_N", "log_P", "log_lambda_max"],
                "input_scaler": self.input_scaler.architecture_report(),
                "auxiliary_gate_initialization": V22_AUXILIARY_GATE_INIT,
                "full_gram_projection_used": False,
                "eigendecomposition_used": False,
                "total_parameters": sum(p.numel() for p in self.parameters()),
            }
        )
        return report


def load_v20_state_dict(
    selector: DirectPenaltyMLPQuerySelectorV22V20GeometryScaled,
    state_dict: Mapping[str, Tensor],
) -> dict[str, Any]:
    """Import every V20 tensor and require all mismatches to be V22 additions."""

    incompatible = selector.load_state_dict(state_dict, strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    allowed_prefixes = (
        "input_scaler.",
        "geometry_projection.",
        "correlation_quantile_projection.",
        "dimension_context_projection.",
        "auxiliary_fusion.",
        "auxiliary_gate",
    )
    forbidden_missing = [
        key for key in missing
        if not key.startswith(allowed_prefixes)
    ]
    if unexpected or forbidden_missing:
        raise RuntimeError(
            "V20-to-V22 state mismatch: "
            f"forbidden_missing={forbidden_missing}, "
            f"unexpected={unexpected}"
        )
    imported_parameters = sum(
        int(value.numel())
        for key, value in state_dict.items()
        if key in selector.state_dict()
    )
    return {
        "missing_v22_keys": missing,
        "unexpected_keys": unexpected,
        "imported_tensor_count": len(state_dict),
        "imported_parameter_and_buffer_elements": imported_parameters,
        "strict_v20_key_audit": True,
    }


__all__ = [
    "DirectPenaltyMLPQuerySelectorV22V20GeometryScaled",
    "FixedInputScalerV22",
    "V22_AUXILIARY_GATE_INIT",
    "V22_SCALER_CLIP",
    "V22_SCALER_STD_FLOOR",
    "V22_SCALER_VERSION",
    "V22_SELECTOR_VERSION",
    "V17_HIDDEN_DIM",
    "V17_MAX_PROBE_ROWS",
    "V17_STAGES",
    "V17_STAGE_GAIN_RATE",
    "V17_TOPK",
    "load_v20_state_dict",
    "transform_geometry_for_scaling",
    "V19_FEATURE_CHUNK_SIZE",
    "V19_PENALTY_RESIDUAL_LAYERS",
]
