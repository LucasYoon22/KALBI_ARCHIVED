#!/usr/bin/env python3
"""V18 lambda selector with full-P residual ISIS and pre-norm penalty blocks.

V18 changes two parts of V17:

* Every ISIS stage scores every valid feature against the current residual.
  There is no fixed marginal-correlation candidate pool. Features selected by
  an earlier stage are masked permanently, so selection is without replacement.
* V18 makes each of the three width-preserving linear layers its own pre-norm
  residual block and removes the post-residual LayerNorm.
* The V17 fusion trunk, lambda head, and canonical Direct96
  projection/decoder remain unchanged.

Full-P stage scoring is exact but feature-chunked for bounded temporary memory.
Chunking changes only execution order, not the global top-k result.

The penalty path is therefore:

    task_code[512]
      -> 3 x (LayerNorm[512] -> Linear[512,512] -> GELU -> residual add)
      -> Linear[512,96] in CanonicalDirect96Head

There is no LayerNorm after the final residual add and no 96-dimensional
LayerNorm in this selector.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from .selector_v17_k32s32 import (
    DirectPenaltyMLPQuerySelectorV17StageGain,
    V17_HIDDEN_DIM,
    V17_INITIAL_LAMBDA_RATIO,
    V17_MAX_PROBE_ROWS,
    V17_RIDGE_FLOOR,
    V17_RIDGE_RATIO,
    V17_RESIDUAL_SCALE_FLOOR,
    V17_STAGES,
    V17_STAGE_GAIN_RATE,
    V17_TOPK,
    _batch_gather_rows,
    _content_stratified_rows,
)


Tensor = torch.Tensor

V18_SELECTOR_VERSION = (
    "v18_k32s32_direct96_residualtrajectory_fullp_isis32x32_"
    "stagegain0p5pct_hidden512_probe2048_chunk4096_noreplacement_"
    "penalty_preln_reslinear3_nopostnorm_v1"
)
V18_PENALTY_RESIDUAL_LAYERS = 3
V18_FEATURE_CHUNK_SIZE = 4096


def residual_isis_trajectory_v18_all_features(
    x_context: Tensor,
    y_context: Tensor,
    *,
    feature_mask: Tensor | None = None,
    stages: int = V17_STAGES,
    topk: int = V17_TOPK,
    max_probe_rows: int = V17_MAX_PROBE_ROWS,
    feature_chunk_size: int = V18_FEATURE_CHUNK_SIZE,
    ridge_ratio: float = V17_RIDGE_RATIO,
    ridge_floor: float = V17_RIDGE_FLOOR,
    residual_scale_floor: float = V17_RESIDUAL_SCALE_FLOOR,
    stage_gain_rate: float = V17_STAGE_GAIN_RATE,
    eps: float = 1e-8,
) -> dict[str, Tensor | int | float | str | bool]:
    """Build a full-P, without-replacement residual-ISIS trajectory.

    At every stage, all valid and not-yet-selected features are standardized
    on the probe rows and scored against the current residual. The feature
    scan is chunked only to bound temporary memory; a running global top-k
    makes the result identical to scoring a materialized B x P matrix.
    """

    if x_context.ndim != 3 or y_context.ndim != 2:
        raise ValueError(
            f"V18 expects X=(B,N,P), y=(B,N), got "
            f"{tuple(x_context.shape)} and {tuple(y_context.shape)}"
        )
    batch, rows, features = x_context.shape
    if y_context.shape != (batch, rows):
        raise ValueError("V18 context X/y row alignment drift")
    if rows < 2 or features < 1:
        raise ValueError(f"V18 requires N>=2 and P>=1, got N={rows}, P={features}")
    if int(stages) < 1 or int(topk) < 1:
        raise ValueError("V18 stages and topk must be positive")
    if int(max_probe_rows) < 2 or int(feature_chunk_size) < 1:
        raise ValueError("V18 probe rows and feature chunk size must be positive")
    if float(ridge_ratio) < 0.0 or float(ridge_floor) <= 0.0:
        raise ValueError("V18 ridge ratio/floor contract is invalid")
    if float(residual_scale_floor) <= 0.0:
        raise ValueError("V18 residual scale floor must be positive")
    if not math.isfinite(float(stage_gain_rate)) or float(stage_gain_rate) < 0.0:
        raise ValueError("V18 stage gain rate must be finite and nonnegative")

    if feature_mask is None:
        valid_features = torch.ones(
            batch, features, device=x_context.device, dtype=torch.bool
        )
    else:
        if feature_mask.shape != (batch, features):
            raise ValueError(
                f"feature_mask must be {(batch, features)}, got "
                f"{tuple(feature_mask.shape)}"
            )
        valid_features = feature_mask.to(device=x_context.device, dtype=torch.bool)

    row_indices = _content_stratified_rows(y_context, int(max_probe_rows))
    y_probe_raw = _batch_gather_rows(y_context, row_indices).float()
    probe_rows = int(y_probe_raw.shape[1])
    y_centered = y_probe_raw - y_probe_raw.mean(dim=1, keepdim=True)
    y_scale = y_centered.square().mean(dim=1, keepdim=True).sqrt().clamp_min(eps)
    y_unit = y_centered / y_scale

    # Feature statistics are cached as O(BP) tensors. No B x Nprobe x P
    # standardized matrix or fixed candidate pool is materialized.
    feature_mean = torch.empty(
        batch, features, device=x_context.device, dtype=torch.float32
    )
    feature_scale = torch.empty_like(feature_mean)
    feature_diag = torch.empty_like(feature_mean)
    effective_valid = valid_features.clone()
    chunk_size = int(feature_chunk_size)
    for start in range(0, features, chunk_size):
        end = min(features, start + chunk_size)
        x_chunk_raw = _batch_gather_rows(
            x_context[:, :, start:end], row_indices
        ).float()
        mean = x_chunk_raw.mean(dim=1)
        second = x_chunk_raw.square().mean(dim=1)
        variance = (second - mean.square()).clamp_min(0.0)
        scale_raw = variance.sqrt()
        scale = scale_raw.clamp_min(eps)
        x_chunk = (x_chunk_raw - mean[:, None, :]) / scale[:, None, :]
        diag = x_chunk.square().mean(dim=1).clamp_min(eps)
        feature_mean[:, start:end] = mean
        feature_scale[:, start:end] = scale
        feature_diag[:, start:end] = diag
        effective_valid[:, start:end] &= scale_raw > eps

    residual = y_unit.clone()
    selected_features = torch.zeros_like(effective_valid)
    residual_columns: list[Tensor] = []
    residual_betas_raw: list[Tensor] = []
    marginal_betas: list[Tensor] = []
    selected_global_indices: list[Tensor] = []
    stage_active_columns: list[Tensor] = []
    residual_energy_columns: list[Tensor] = []
    residual_scale_before_columns: list[Tensor] = []
    explained_fraction_columns: list[Tensor] = []

    select_width = int(topk)
    for _stage in range(int(stages)):
        residual_energy_before = residual.square().mean(dim=1, keepdim=True)
        residual_scale_before = residual_energy_before.sqrt().clamp_min(
            float(residual_scale_floor)
        )

        # Exact global top-k over all P, assembled from exact per-chunk top-k.
        best_scores = torch.full(
            (batch, select_width),
            -torch.inf,
            device=x_context.device,
            dtype=torch.float32,
        )
        best_indices = torch.full(
            (batch, select_width),
            -1,
            device=x_context.device,
            dtype=torch.long,
        )
        for start in range(0, features, chunk_size):
            end = min(features, start + chunk_size)
            x_chunk_raw = _batch_gather_rows(
                x_context[:, :, start:end], row_indices
            ).float()
            x_chunk = (
                x_chunk_raw - feature_mean[:, None, start:end]
            ) / feature_scale[:, None, start:end]
            rhs = torch.einsum("bnm,bn->bm", x_chunk, residual) / float(
                probe_rows
            )
            score = rhs.abs() / (
                feature_diag[:, start:end].sqrt() * residual_scale_before
            ).clamp_min(eps)
            available = (
                effective_valid[:, start:end]
                & ~selected_features[:, start:end]
            )
            score = torch.nan_to_num(
                score, nan=0.0, posinf=0.0, neginf=0.0
            ).masked_fill(~available, -torch.inf)
            local_width = min(select_width, end - start)
            chunk_scores, chunk_positions = torch.topk(
                score,
                k=local_width,
                dim=-1,
                largest=True,
                sorted=True,
            )
            chunk_indices = chunk_positions + int(start)
            combined_scores = torch.cat([best_scores, chunk_scores], dim=-1)
            combined_indices = torch.cat([best_indices, chunk_indices], dim=-1)
            best_scores, keep = torch.topk(
                combined_scores,
                k=select_width,
                dim=-1,
                largest=True,
                sorted=True,
            )
            best_indices = torch.gather(combined_indices, 1, keep)

        selected_valid = torch.isfinite(best_scores) & (best_indices >= 0)
        global_indices = best_indices.masked_fill(~selected_valid, -1)
        safe_indices = global_indices.clamp_min(0)

        x_selected_all_rows = torch.gather(
            x_context,
            2,
            safe_indices[:, None, :].expand(batch, rows, select_width),
        ).float()
        x_stage_raw = _batch_gather_rows(x_selected_all_rows, row_indices)
        selected_mean = torch.gather(feature_mean, 1, safe_indices)
        selected_scale = torch.gather(feature_scale, 1, safe_indices)
        x_stage = (
            (x_stage_raw - selected_mean[:, None, :])
            / selected_scale[:, None, :]
            * selected_valid[:, None, :].float()
        )

        gram = torch.einsum("bnk,bnl->bkl", x_stage, x_stage) / float(probe_rows)
        rhs_res_stage = torch.einsum(
            "bnk,bn->bk", x_stage, residual
        ) / float(probe_rows)
        rhs_marg_stage = torch.einsum(
            "bnk,bn->bk", x_stage, y_unit
        ) / float(probe_rows)
        diagonal = gram.diagonal(dim1=-2, dim2=-1)
        active_count = selected_valid.sum(dim=-1).clamp_min(1).float()
        mean_diagonal = (
            diagonal * selected_valid.float()
        ).sum(dim=-1) / active_count
        alpha = (float(ridge_ratio) * mean_diagonal).clamp_min(float(ridge_floor))
        eye = torch.eye(
            select_width, device=x_context.device, dtype=torch.float32
        )[None, :, :]
        system = gram + (alpha[:, None, None] + float(eps)) * eye
        beta_residual_raw = torch.linalg.solve(
            system, rhs_res_stage.unsqueeze(-1)
        ).squeeze(-1)
        beta_marginal = torch.linalg.solve(
            system, rhs_marg_stage.unsqueeze(-1)
        ).squeeze(-1)
        beta_residual_raw = beta_residual_raw * selected_valid.float()
        beta_marginal = beta_marginal * selected_valid.float()

        residual = residual - torch.einsum(
            "bnk,bk->bn", x_stage, beta_residual_raw
        )
        update_counts = torch.zeros(
            batch,
            features,
            device=x_context.device,
            dtype=torch.int32,
        )
        update_counts.scatter_add_(
            1, safe_indices, selected_valid.to(dtype=torch.int32)
        )
        selected_features |= update_counts > 0

        stage_active = selected_valid.any(dim=-1)
        residual_energy_after = residual.square().mean(dim=1)
        explained_fraction = (
            (residual_energy_before.squeeze(-1) - residual_energy_after)
            / residual_energy_before.squeeze(-1).clamp_min(
                float(residual_scale_floor) ** 2
            )
        ).clamp(0.0, 1.0) * stage_active.float()
        residual_columns.append(residual.clone())
        residual_betas_raw.append(beta_residual_raw)
        marginal_betas.append(beta_marginal)
        selected_global_indices.append(global_indices)
        stage_active_columns.append(stage_active)
        residual_energy_columns.append(residual_energy_after)
        residual_scale_before_columns.append(residual_scale_before.squeeze(-1))
        explained_fraction_columns.append(explained_fraction)

    residual_trajectory_raw = torch.stack(residual_columns, dim=-1)
    response_sign = torch.where(
        y_unit >= 0.0, torch.ones_like(y_unit), -torch.ones_like(y_unit)
    )
    residual_trajectory_raw = (
        residual_trajectory_raw * response_sign.unsqueeze(-1)
    )
    beta_residual_raw = torch.stack(residual_betas_raw, dim=-1)
    stage_gain = torch.pow(
        torch.tensor(
            1.0 + float(stage_gain_rate),
            device=x_context.device,
            dtype=torch.float32,
        ),
        torch.arange(int(stages), device=x_context.device, dtype=torch.float32),
    )
    residual_trajectory = residual_trajectory_raw * stage_gain[None, None, :]
    beta_residual = beta_residual_raw * stage_gain[None, None, :]
    beta_marginal_raw = torch.stack(marginal_betas, dim=-1)
    beta_marginal = beta_marginal_raw * stage_gain[None, None, :]
    selected_indices = torch.stack(selected_global_indices, dim=1)
    stage_active = torch.stack(stage_active_columns, dim=-1)
    residual_energy_ratio = torch.stack(residual_energy_columns, dim=-1)
    residual_scale_before = torch.stack(residual_scale_before_columns, dim=-1)
    stage_explained_fraction = torch.stack(explained_fraction_columns, dim=-1)

    expected_r = (batch, probe_rows, int(stages))
    expected_b = (batch, int(topk), int(stages))
    if residual_trajectory.shape != expected_r:
        raise RuntimeError(
            f"V18 residual trajectory shape {tuple(residual_trajectory.shape)} "
            f"!= {expected_r}"
        )
    if (
        beta_residual.shape != expected_b
        or beta_residual_raw.shape != expected_b
        or beta_marginal.shape != expected_b
        or beta_marginal_raw.shape != expected_b
    ):
        raise RuntimeError("V18 coefficient trajectory shape drift")
    if selected_indices.shape != (batch, int(stages), int(topk)):
        raise RuntimeError("V18 selected-index shape drift")
    tensors = (
        residual_trajectory,
        residual_trajectory_raw,
        beta_residual,
        beta_residual_raw,
        beta_marginal,
        beta_marginal_raw,
        residual_energy_ratio,
        residual_scale_before,
        stage_explained_fraction,
        stage_gain,
    )
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise RuntimeError("V18 trajectory contains nonfinite values")

    all_feature_indices = torch.arange(
        features, device=x_context.device, dtype=torch.long
    )[None, :].expand(batch, -1)
    return {
        "residual_trajectory": residual_trajectory,
        "residual_trajectory_raw": residual_trajectory_raw,
        "beta_residual": beta_residual,
        "beta_residual_raw": beta_residual_raw,
        "beta_marginal": beta_marginal,
        "beta_marginal_raw": beta_marginal_raw,
        "selected_indices": selected_indices,
        "stage_active": stage_active,
        "residual_energy_ratio": residual_energy_ratio,
        "residual_scale_before": residual_scale_before,
        "stage_explained_fraction": stage_explained_fraction,
        "stage_gain": stage_gain,
        "probe_row_indices": row_indices,
        "candidate_indices": all_feature_indices,
        "candidate_valid": effective_valid,
        "probe_rows": probe_rows,
        "candidate_pool_width": features,
        "feature_chunk_size": chunk_size,
        "ridge_ratio": float(ridge_ratio),
        "residual_scale_floor": float(residual_scale_floor),
        "stage_gain_rate": float(stage_gain_rate),
        "all_features_rescored_each_stage": True,
        "selection_without_replacement": True,
        "trajectory_contract": (
            "response_quantile_rows_fullP_residual_rescore_each_stage_"
            "top32_without_replacement_localridge_stagegain0p5pct_v1"
        ),
    }


class _PreNormLinearResidualLayer(nn.Module):
    """One width-preserving pre-norm linear residual layer."""

    def __init__(self, width: int) -> None:
        super().__init__()
        width = int(width)
        self.norm = nn.LayerNorm(width)
        self.linear = nn.Linear(width, width)
        self.act = nn.GELU()

    def forward(self, values: Tensor) -> Tensor:
        return values + self.act(self.linear(self.norm(values)))


class _ThreePreNormLinearResidualLayers(nn.Module):
    """Three independent pre-norm residual layers with no output norm."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = int(width)
        self.layers = nn.ModuleList(
            [
                _PreNormLinearResidualLayer(self.width)
                for _ in range(V18_PENALTY_RESIDUAL_LAYERS)
            ]
        )

    def forward(self, values: Tensor) -> Tensor:
        for layer in self.layers:
            values = layer(values)
        return values


class DirectPenaltyMLPQuerySelectorV18StageGain(
    DirectPenaltyMLPQuerySelectorV17StageGain
):
    """K32/S32 selector with full-P ISIS and the V18 penalty MLP."""

    def __init__(
        self,
        *,
        feature_chunk_size: int = V18_FEATURE_CHUNK_SIZE,
        **kwargs: Any,
    ) -> None:
        # Retain the parent constructor's legacy candidate-pool argument only
        # for state/constructor compatibility. V18 encode_context never uses it.
        kwargs.pop("candidate_pool", None)
        super().__init__(candidate_pool=V17_STAGES * V17_TOPK, **kwargs)
        if int(feature_chunk_size) < 1:
            raise ValueError("V18 feature chunk size must be positive")
        self.feature_chunk_size = int(feature_chunk_size)
        self.penalty_mlp = _ThreePreNormLinearResidualLayers(self.hidden)
        self.selector_version = V18_SELECTOR_VERSION

    def encode_context(
        self,
        x_context: Tensor,
        y_context: Tensor,
        *,
        feature_mask: Tensor | None = None,
    ) -> dict[str, Tensor | int | float | str | bool]:
        return residual_isis_trajectory_v18_all_features(
            x_context,
            y_context,
            feature_mask=feature_mask,
            stages=self.stages,
            topk=self.topk,
            max_probe_rows=self.max_probe_rows,
            feature_chunk_size=self.feature_chunk_size,
            ridge_ratio=self.ridge_ratio,
            ridge_floor=self.ridge_floor,
            residual_scale_floor=self.residual_scale_floor,
            stage_gain_rate=self.stage_gain_rate,
        )

    def architecture_report(self) -> dict[str, Any]:
        report = super().architecture_report()
        report.update(
            {
                "selector_version": self.selector_version,
                "candidate_pool": None,
                "candidate_pool_used": False,
                "legacy_constructor_candidate_pool": self.candidate_pool,
                "all_features_rescored_each_stage": True,
                "selection_without_replacement": True,
                "feature_chunk_size": self.feature_chunk_size,
                "trajectory_contract": (
                    "fullP_residual_rescore_each_stage_top32_"
                    "without_replacement"
                ),
                "penalty_mlp_linear_layers": V18_PENALTY_RESIDUAL_LAYERS,
                "penalty_mlp_residual_blocks": V18_PENALTY_RESIDUAL_LAYERS,
                "penalty_mlp_layer_norms": V18_PENALTY_RESIDUAL_LAYERS,
                "penalty_mlp_residual_granularity": (
                    "one_pre_norm_linear_gelu_per_residual"
                ),
                "penalty_mlp_post_norm": False,
                "penalty_mlp_output_dim": self.hidden,
                "direct96_head_input_dim": self.hidden,
                "direct96_head_output_dim": int(
                    self.penalty_parameter_head.latent_rank
                ),
                "v17_trajectory_encoder_unchanged": False,
                "v18_fullp_trajectory_encoder": True,
                "v17_fusion_mlp_unchanged": True,
                "v17_lambda_head_unchanged": True,
                "v17_direct96_head_unchanged": True,
                "total_parameters": sum(p.numel() for p in self.parameters()),
            }
        )
        return report


__all__ = [
    "DirectPenaltyMLPQuerySelectorV18StageGain",
    "V18_FEATURE_CHUNK_SIZE",
    "V18_PENALTY_RESIDUAL_LAYERS",
    "V18_SELECTOR_VERSION",
    "V17_HIDDEN_DIM",
    "V17_INITIAL_LAMBDA_RATIO",
    "V17_MAX_PROBE_ROWS",
    "V17_RIDGE_FLOOR",
    "V17_RIDGE_RATIO",
    "V17_RESIDUAL_SCALE_FLOOR",
    "V17_STAGES",
    "V17_STAGE_GAIN_RATE",
    "V17_TOPK",
    "residual_isis_trajectory_v18_all_features",
]
