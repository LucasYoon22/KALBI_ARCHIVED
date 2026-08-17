#!/usr/bin/env python3
"""V19 lambda selector with full-P ISIS and a cumulative-ridge beta branch.

V19 keeps the validated V18 selection path:

* every residual-ISIS stage scores every valid feature in P;
* features selected by earlier stages are excluded permanently;
* exact global top-32 selection is assembled with memory-bounded chunks; and
* the existing stage-local residual and marginal ridge betas are retained.

In addition, V19 jointly refits y on every feature selected up through each
stage.  This cumulative ridge fit is separate from the stage-local residual
update, so it does not change the residual used for later full-P screening.
The complete zero-padded triangular cumulative-beta path is encoded and its
code is concatenated with the two existing local-beta codes.

The penalty path remains three independent pre-norm Linear-GELU residual
blocks followed directly by the canonical Direct96 linear head.  There is no
post-residual LayerNorm and no 96-dimensional LayerNorm.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from .canonical_penalty import build_penalty
from .selector_v10 import lambda_from_effective_ratio_v56
from .selector_v17_k32s32 import (
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
)
from .selector_v18_k32s32 import (
    DirectPenaltyMLPQuerySelectorV18StageGain,
    V18_FEATURE_CHUNK_SIZE,
    V18_PENALTY_RESIDUAL_LAYERS,
    residual_isis_trajectory_v18_all_features,
)


Tensor = torch.Tensor

V19_SELECTOR_VERSION = (
    "v19_k32s32_direct96_residualtrajectory_fullp_isis32x32_"
    "stagegain0p5pct_hidden512_probe2048_chunk4096_noreplacement_"
    "localres_localmarg_cumulativejointbeta_concat_"
    "penalty_preln_reslinear3_nopostnorm_v2"
)
V19_FEATURE_CHUNK_SIZE = V18_FEATURE_CHUNK_SIZE
V19_PENALTY_RESIDUAL_LAYERS = V18_PENALTY_RESIDUAL_LAYERS
V19_CUMULATIVE_SLOT_WIDTH = V17_STAGES * V17_TOPK
V19_CUMULATIVE_PATH_WIDTH = V19_CUMULATIVE_SLOT_WIDTH * V17_STAGES
V22_GEOMETRY_STAT_NAMES = (
    "regularized_logdet",
    "cholesky_pivot_min",
    "cholesky_pivot_max",
    "log_pivot_range",
    "schur_diag_min",
    "schur_diag_mean",
    "schur_diag_max",
    "support_abs_corr_max",
    "support_abs_corr_mean",
    "gram_frobenius",
    "participation_ratio_fraction",
    "cumulative_residual_energy",
    "cumulative_coefficient_norm",
)
V22_CORRELATION_QUANTILES = (
    0.99,
    0.97,
    0.95,
    0.90,
    0.80,
    0.70,
    0.60,
    0.50,
    0.40,
    0.30,
)


@torch.no_grad()
def _cumulative_ridge_beta_path_v19(
    x_context: Tensor,
    y_context: Tensor,
    trajectory: dict[str, Tensor | int | float | str | bool],
    *,
    stages: int,
    topk: int,
    ridge_ratio: float,
    ridge_floor: float,
    eps: float,
) -> dict[str, Tensor | int | float | str | bool]:
    """Jointly refit all selected features at every stage.

    The selected columns are ordered by (stage, within-stage rank).  At stage
    s, a ridge fit uses the first ``(s + 1) * topk`` slots.  Coefficients are
    stored in a ``(stages * topk) x stages`` zero-padded triangular path.

    A fixed per-task ridge alpha is computed from the standardized diagonal of
    all selected columns.  The same alpha at every prefix permits an exact
    block-Cholesky expansion rather than recomputing a growing factorization
    from scratch at all 32 stages.
    """

    selected_indices = trajectory["selected_indices"]
    probe_row_indices = trajectory["probe_row_indices"]
    stage_active = trajectory["stage_active"]
    stage_gain = trajectory["stage_gain"]
    if not all(
        torch.is_tensor(value)
        for value in (
            selected_indices,
            probe_row_indices,
            stage_active,
            stage_gain,
        )
    ):
        raise TypeError("V19 cumulative fit requires tensor trajectory fields")

    selected_indices = selected_indices.long()
    probe_row_indices = probe_row_indices.long()
    stage_active = stage_active.bool()
    stage_gain = stage_gain.float()
    batch, rows, features = x_context.shape
    expected_selected = (batch, int(stages), int(topk))
    if selected_indices.shape != expected_selected:
        raise ValueError(
            f"V19 selected-index shape {tuple(selected_indices.shape)} "
            f"!= {expected_selected}"
        )
    if stage_active.shape != (batch, int(stages)):
        raise ValueError("V19 cumulative stage mask width drift")
    if stage_gain.shape != (int(stages),):
        raise ValueError("V19 cumulative stage-gain width drift")

    slot_width = int(stages) * int(topk)
    flat_indices = selected_indices.reshape(batch, slot_width)
    selected_valid = flat_indices >= 0
    safe_indices = flat_indices.clamp(0, max(0, features - 1))
    x_selected_all_rows = torch.gather(
        x_context,
        2,
        safe_indices[:, None, :].expand(batch, rows, slot_width),
    ).float()
    x_probe_raw = _batch_gather_rows(
        x_selected_all_rows, probe_row_indices
    ).float()
    probe_rows = int(x_probe_raw.shape[1])
    selected_mean = x_probe_raw.mean(dim=1)
    selected_variance = (
        x_probe_raw.square().mean(dim=1) - selected_mean.square()
    ).clamp_min(0.0)
    selected_scale_raw = selected_variance.sqrt()
    selected_valid = selected_valid & (selected_scale_raw > float(eps))
    selected_scale = selected_scale_raw.clamp_min(float(eps))
    x_selected = (
        (x_probe_raw - selected_mean[:, None, :])
        / selected_scale[:, None, :]
        * selected_valid[:, None, :].float()
    )

    y_probe_raw = _batch_gather_rows(y_context, probe_row_indices).float()
    y_centered = y_probe_raw - y_probe_raw.mean(dim=1, keepdim=True)
    y_scale = y_centered.square().mean(dim=1, keepdim=True).sqrt().clamp_min(
        float(eps)
    )
    y_unit = y_centered / y_scale

    diagonal = x_selected.square().mean(dim=1)
    selected_count = selected_valid.sum(dim=-1)
    active_count = selected_count.clamp_min(1).float()
    mean_diagonal = (
        diagonal * selected_valid.float()
    ).sum(dim=-1) / active_count
    cumulative_alpha = (
        float(ridge_ratio) * mean_diagonal
    ).clamp_min(float(ridge_floor))
    rhs_all = torch.einsum("bnm,bn->bm", x_selected, y_unit) / float(
        probe_rows
    )

    beta_columns: list[Tensor] = []
    residual_energy_columns: list[Tensor] = []
    geometry_columns: list[Tensor] = []
    pair_value_chunks: list[Tensor] = []
    pair_mask_chunks: list[Tensor] = []
    cholesky: Tensor | None = None
    running_pair_abs_sum = torch.zeros(
        batch, device=x_context.device, dtype=torch.float32
    )
    running_pair_count = torch.zeros_like(running_pair_abs_sum)
    running_pair_abs_max = torch.zeros_like(running_pair_abs_sum)
    running_gram_frobenius_sq = torch.zeros_like(running_pair_abs_sum)
    running_trace = torch.zeros_like(running_pair_abs_sum)
    zero_column = torch.zeros(
        batch,
        slot_width,
        device=x_context.device,
        dtype=torch.float32,
    )
    zero_energy = torch.zeros(
        batch, device=x_context.device, dtype=torch.float32
    )

    for stage in range(int(stages)):
        start = stage * int(topk)
        end = start + int(topk)
        active_now = stage_active[:, stage]
        if not bool(active_now.any().item()):
            beta_columns.extend(
                zero_column.clone() for _ in range(stage, int(stages))
            )
            residual_energy_columns.extend(
                zero_energy.clone() for _ in range(stage, int(stages))
            )
            geometry_columns.extend(
                torch.zeros(
                    batch,
                    len(V22_GEOMETRY_STAT_NAMES),
                    device=x_context.device,
                    dtype=torch.float32,
                )
                for _ in range(stage, int(stages))
            )
            break

        x_new = x_selected[:, :, start:end]
        valid_new = selected_valid[:, start:end]
        eye_new = torch.eye(
            int(topk), device=x_context.device, dtype=torch.float32
        )[None, :, :]
        correlation_new = (
            torch.einsum("bnk,bnl->bkl", x_new, x_new)
            / float(probe_rows)
        )
        diagonal_block = (
            correlation_new + cumulative_alpha[:, None, None] * eye_new
        )
        valid_new_matrix = valid_new[:, :, None] & valid_new[:, None, :]
        running_gram_frobenius_sq = (
            running_gram_frobenius_sq
            + (
                correlation_new.square()
                * valid_new_matrix.float()
            ).sum(dim=(-2, -1))
        )
        running_trace = (
            running_trace
            + (
                torch.diagonal(correlation_new, dim1=-2, dim2=-1)
                * valid_new.float()
            ).sum(dim=-1)
        )

        upper = torch.triu_indices(
            int(topk),
            int(topk),
            offset=1,
            device=x_context.device,
        )
        within_values = correlation_new[:, upper[0], upper[1]].abs()
        within_mask = valid_new[:, upper[0]] & valid_new[:, upper[1]]
        pair_value_chunks.append(within_values)
        pair_mask_chunks.append(within_mask)
        running_pair_abs_sum = (
            running_pair_abs_sum
            + (within_values * within_mask.float()).sum(dim=-1)
        )
        running_pair_count = (
            running_pair_count + within_mask.float().sum(dim=-1)
        )
        running_pair_abs_max = torch.maximum(
            running_pair_abs_max,
            within_values.masked_fill(~within_mask, 0.0).amax(dim=-1),
        )

        if cholesky is None:
            schur = diagonal_block
            lower_new = torch.linalg.cholesky(
                0.5 * (schur + schur.transpose(-1, -2))
                + float(eps) * eye_new
            )
        else:
            x_previous = x_selected[:, :, :start]
            valid_previous = selected_valid[:, :start]
            cross = torch.einsum(
                "bni,bnj->bij", x_previous, x_new
            ) / float(probe_rows)
            cross_mask = valid_previous[:, :, None] & valid_new[:, None, :]
            cross_values = cross.abs().reshape(batch, -1)
            cross_mask_flat = cross_mask.reshape(batch, -1)
            pair_value_chunks.append(cross_values)
            pair_mask_chunks.append(cross_mask_flat)
            running_pair_abs_sum = (
                running_pair_abs_sum
                + (cross_values * cross_mask_flat.float()).sum(dim=-1)
            )
            running_pair_count = (
                running_pair_count + cross_mask_flat.float().sum(dim=-1)
            )
            running_pair_abs_max = torch.maximum(
                running_pair_abs_max,
                cross_values.masked_fill(~cross_mask_flat, 0.0).amax(dim=-1),
            )
            running_gram_frobenius_sq = (
                running_gram_frobenius_sq
                + 2.0
                * (
                    cross.square() * cross_mask.float()
                ).sum(dim=(-2, -1))
            )
            solved_cross = torch.linalg.solve_triangular(
                cholesky, cross, upper=False
            )
            schur = diagonal_block - torch.einsum(
                "bik,bil->bkl", solved_cross, solved_cross
            )
            schur = 0.5 * (schur + schur.transpose(-1, -2))
            lower_new = torch.linalg.cholesky(
                schur + float(eps) * eye_new
            )

        expanded = torch.zeros(
            batch,
            end,
            end,
            device=x_context.device,
            dtype=torch.float32,
        )
        if cholesky is not None:
            expanded[:, :start, :start] = cholesky
            expanded[:, start:end, :start] = solved_cross.transpose(-1, -2)
        expanded[:, start:end, start:end] = lower_new
        cholesky = expanded

        beta_prefix = torch.cholesky_solve(
            rhs_all[:, :end].unsqueeze(-1), cholesky
        ).squeeze(-1)
        beta_prefix = beta_prefix * selected_valid[:, :end].float()
        beta_padded = zero_column.clone()
        beta_padded[:, :end] = beta_prefix
        beta_padded = beta_padded * active_now[:, None].float()
        beta_columns.append(beta_padded)

        prediction = torch.einsum(
            "bnm,bm->bn", x_selected[:, :, :end], beta_prefix
        )
        residual_energy = (
            y_unit - prediction
        ).square().mean(dim=1) * active_now.float()
        residual_energy_columns.append(residual_energy)

        valid_prefix = selected_valid[:, :end]
        valid_count = valid_prefix.float().sum(dim=-1).clamp_min(1.0)
        cholesky_diagonal = torch.diagonal(
            cholesky, dim1=-2, dim2=-1
        )
        pivot_min = cholesky_diagonal.masked_fill(
            ~valid_prefix, torch.inf
        ).amin(dim=-1)
        pivot_max = cholesky_diagonal.masked_fill(
            ~valid_prefix, 0.0
        ).amax(dim=-1)
        pivot_min = torch.where(
            torch.isfinite(pivot_min), pivot_min, torch.ones_like(pivot_min)
        ).clamp_min(float(eps))
        pivot_max = pivot_max.clamp_min(float(eps))
        regularized_logdet = (
            2.0
            * (
                cholesky_diagonal.clamp_min(float(eps)).log()
                * valid_prefix.float()
            ).sum(dim=-1)
        )
        log_pivot_range = (
            pivot_max.log() - pivot_min.log()
        ).clamp_min(0.0)

        schur_diagonal = torch.diagonal(
            schur + float(eps) * eye_new, dim1=-2, dim2=-1
        )
        schur_min = schur_diagonal.masked_fill(
            ~valid_new, torch.inf
        ).amin(dim=-1)
        schur_max = schur_diagonal.masked_fill(
            ~valid_new, 0.0
        ).amax(dim=-1)
        schur_count = valid_new.float().sum(dim=-1).clamp_min(1.0)
        schur_mean = (
            schur_diagonal * valid_new.float()
        ).sum(dim=-1) / schur_count
        schur_min = torch.where(
            torch.isfinite(schur_min), schur_min, torch.ones_like(schur_min)
        ).clamp_min(float(eps))
        schur_mean = schur_mean.clamp_min(float(eps))
        schur_max = schur_max.clamp_min(float(eps))

        pair_mean = running_pair_abs_sum / running_pair_count.clamp_min(1.0)
        gram_frobenius = running_gram_frobenius_sq.clamp_min(0.0).sqrt()
        participation_ratio = (
            running_trace.square()
            / running_gram_frobenius_sq.clamp_min(float(eps))
        )
        participation_fraction = (
            participation_ratio / valid_count
        ).clamp(0.0, 1.0)
        coefficient_norm = (
            beta_prefix.square() * valid_prefix.float()
        ).sum(dim=-1).sqrt()
        geometry = torch.stack(
            [
                regularized_logdet,
                pivot_min,
                pivot_max,
                log_pivot_range,
                schur_min,
                schur_mean,
                schur_max,
                running_pair_abs_max,
                pair_mean,
                gram_frobenius,
                participation_fraction,
                residual_energy,
                coefficient_norm,
            ],
            dim=-1,
        )
        geometry_columns.append(geometry * active_now[:, None].float())

    if len(beta_columns) != int(stages):
        raise RuntimeError("V19 cumulative beta stage count drift")
    beta_cumulative_raw = torch.stack(beta_columns, dim=-1)
    beta_cumulative = beta_cumulative_raw * stage_gain[None, None, :]
    cumulative_residual_energy = torch.stack(residual_energy_columns, dim=-1)
    cumulative_geometry_raw = torch.stack(geometry_columns, dim=1)
    if cumulative_geometry_raw.shape != (
        batch,
        int(stages),
        len(V22_GEOMETRY_STAT_NAMES),
    ):
        raise RuntimeError("V22 cumulative geometry shape drift")

    all_pair_values = torch.cat(pair_value_chunks, dim=-1)
    all_pair_masks = torch.cat(pair_mask_chunks, dim=-1)
    quantile_levels = torch.tensor(
        V22_CORRELATION_QUANTILES,
        device=x_context.device,
        dtype=torch.float32,
    )
    support_correlation_quantiles = []
    valid_pair_counts = all_pair_masks.sum(dim=-1)
    for batch_index in range(batch):
        valid_values = all_pair_values[batch_index][all_pair_masks[batch_index]]
        if valid_values.numel() == 0:
            quantiles = torch.zeros_like(quantile_levels)
        else:
            quantiles = torch.quantile(
                valid_values.float(), quantile_levels
            )
        support_correlation_quantiles.append(quantiles)
    support_correlation_quantiles = torch.stack(
        support_correlation_quantiles, dim=0
    )
    expected_beta = (batch, slot_width, int(stages))
    if beta_cumulative.shape != expected_beta:
        raise RuntimeError(
            f"V19 cumulative beta shape {tuple(beta_cumulative.shape)} "
            f"!= {expected_beta}"
        )
    tensors = (
        beta_cumulative,
        beta_cumulative_raw,
        cumulative_residual_energy,
        cumulative_alpha,
        cumulative_geometry_raw,
        support_correlation_quantiles,
    )
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise RuntimeError("V19 cumulative trajectory contains nonfinite values")

    return {
        "beta_cumulative": beta_cumulative,
        "beta_cumulative_raw": beta_cumulative_raw,
        "cumulative_selected_valid": selected_valid,
        "cumulative_selected_count": selected_count,
        "cumulative_residual_energy": cumulative_residual_energy,
        "cumulative_ridge_alpha": cumulative_alpha,
        "cumulative_geometry_raw": cumulative_geometry_raw,
        "cumulative_geometry_stat_names": V22_GEOMETRY_STAT_NAMES,
        "support_correlation_quantiles": support_correlation_quantiles,
        "support_correlation_quantile_levels": V22_CORRELATION_QUANTILES,
        "support_correlation_valid_pair_count": valid_pair_counts,
        "cumulative_slot_width": slot_width,
        "cumulative_path_width": slot_width * int(stages),
        "cumulative_refit_each_stage": True,
        "cumulative_refit_changes_screening_residual": False,
        "cumulative_beta_layout": (
            "selection_order_slots_by_stage_zero_padded_triangular"
        ),
        "cumulative_solver": "exact_block_cholesky_fixed_task_ridge_alpha",
    }


@torch.no_grad()
def residual_isis_trajectory_v19_all_features(
    x_context: Tensor,
    y_context: Tensor,
    *,
    feature_mask: Tensor | None = None,
    stages: int = V17_STAGES,
    topk: int = V17_TOPK,
    max_probe_rows: int = V17_MAX_PROBE_ROWS,
    feature_chunk_size: int = V19_FEATURE_CHUNK_SIZE,
    ridge_ratio: float = V17_RIDGE_RATIO,
    ridge_floor: float = V17_RIDGE_FLOOR,
    residual_scale_floor: float = V17_RESIDUAL_SCALE_FLOOR,
    stage_gain_rate: float = V17_STAGE_GAIN_RATE,
    eps: float = 1e-8,
) -> dict[str, Tensor | int | float | str | bool]:
    """Run full-P ISIS plus the separate cumulative joint-ridge path."""

    trajectory = dict(
        residual_isis_trajectory_v18_all_features(
            x_context,
            y_context,
            feature_mask=feature_mask,
            stages=stages,
            topk=topk,
            max_probe_rows=max_probe_rows,
            feature_chunk_size=feature_chunk_size,
            ridge_ratio=ridge_ratio,
            ridge_floor=ridge_floor,
            residual_scale_floor=residual_scale_floor,
            stage_gain_rate=stage_gain_rate,
            eps=eps,
        )
    )
    trajectory.update(
        _cumulative_ridge_beta_path_v19(
            x_context,
            y_context,
            trajectory,
            stages=int(stages),
            topk=int(topk),
            ridge_ratio=float(ridge_ratio),
            ridge_floor=float(ridge_floor),
            eps=float(eps),
        )
    )
    trajectory["trajectory_contract"] = (
        "response_quantile_rows_fullP_residual_rescore_each_stage_"
        "top32_without_replacement_localridge_plus_"
        "cumulative_joint_ridge_beta_stagegain0p5pct_v2"
    )
    return trajectory


class DirectPenaltyMLPQuerySelectorV19StageGain(
    DirectPenaltyMLPQuerySelectorV18StageGain
):
    """V19 selector with local and cumulative beta representations."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cumulative_slot_width = self.stages * self.topk
        self.cumulative_path_width = self.cumulative_slot_width * self.stages
        self.cumulative_coefficient_proj = nn.Sequential(
            nn.LayerNorm(self.cumulative_path_width),
            nn.Linear(self.cumulative_path_width, self.hidden),
        )
        self.coefficient_concat_proj = nn.Sequential(
            nn.LayerNorm(3 * self.hidden),
            nn.Linear(3 * self.hidden, self.hidden),
        )
        self.selector_version = V19_SELECTOR_VERSION

    def encode_context(
        self,
        x_context: Tensor,
        y_context: Tensor,
        *,
        feature_mask: Tensor | None = None,
    ) -> dict[str, Tensor | int | float | str | bool]:
        return residual_isis_trajectory_v19_all_features(
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
                "architecture_generation": 19,
                "v19_contract_frozen": True,
                "v19_lineage": "v18_fullp_plus_cumulative_joint_beta",
                "parameterization_identical_to_v18": False,
                "v18_shared_modules_retained": True,
                "all_features_rescored_each_stage": True,
                "selection_without_replacement": True,
                "candidate_pool": None,
                "candidate_pool_used": False,
                "feature_chunk_size": self.feature_chunk_size,
                "local_beta_branches": ["residual", "marginal"],
                "cumulative_refit_each_stage": True,
                "cumulative_refit_target": "standardized_y",
                "cumulative_refit_changes_screening_residual": False,
                "cumulative_slot_width": self.cumulative_slot_width,
                "cumulative_path_shape": [
                    self.cumulative_slot_width,
                    self.stages,
                ],
                "cumulative_solver": (
                    "exact_block_cholesky_fixed_task_ridge_alpha"
                ),
                "coefficient_fusion": (
                    "concat(local_residual_code,local_marginal_code,"
                    "cumulative_joint_code)->linear512"
                ),
                "coefficient_concat_input_dim": 3 * self.hidden,
                "coefficient_concat_output_dim": self.hidden,
                "stage_gain_targets": [
                    "R",
                    "B_res",
                    "B_marg",
                    "B_cumulative",
                ],
                "penalty_mlp_post_norm": False,
                "total_parameters": sum(p.numel() for p in self.parameters()),
            }
        )
        return report

    def _build_fusion_input(
        self,
        *,
        interaction_code: Tensor,
        coefficient_code: Tensor,
        metadata_code: Tensor,
        trajectory: dict[str, Tensor | int | float | str | bool],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        del trajectory
        return interaction_code + coefficient_code + metadata_code, {}

    def _lambda_reference_slope(
        self, encoded: dict[str, Tensor]
    ) -> Tensor:
        """Retain V19's nonsmooth effective-lambda definition."""
        return encoded["d0"]

    def forward_from_trajectory(
        self,
        trajectory: dict[str, Tensor | int | float | str | bool],
        lambda_max: Tensor,
        *,
        return_representation: bool = False,
    ) -> dict[str, Any]:
        residual = trajectory["residual_trajectory"]
        beta_residual = trajectory["beta_residual"]
        beta_marginal = trajectory["beta_marginal"]
        beta_cumulative = trajectory["beta_cumulative"]
        stage_active = trajectory["stage_active"]
        residual_energy = trajectory["residual_energy_ratio"]
        residual_scale_before = trajectory["residual_scale_before"]
        stage_explained_fraction = trajectory["stage_explained_fraction"]
        if not all(
            torch.is_tensor(value)
            for value in (
                residual,
                beta_residual,
                beta_marginal,
                beta_cumulative,
                stage_active,
                residual_energy,
                residual_scale_before,
                stage_explained_fraction,
            )
        ):
            raise TypeError("V19 trajectory tensor contract is incomplete")

        residual = residual.float()
        beta_residual = beta_residual.float()
        beta_marginal = beta_marginal.float()
        beta_cumulative = beta_cumulative.float()
        stage_active = stage_active.bool()
        residual_energy = residual_energy.float()
        residual_scale_before = residual_scale_before.float()
        stage_explained_fraction = stage_explained_fraction.float()
        batch, probe_rows, stages = residual.shape
        if stages != self.stages:
            raise ValueError("V19 residual stage width drift")
        if beta_residual.shape != (batch, self.topk, self.stages):
            raise ValueError("V19 residual coefficient width drift")
        if beta_marginal.shape != beta_residual.shape:
            raise ValueError("V19 marginal coefficient width drift")
        if beta_cumulative.shape != (
            batch,
            self.cumulative_slot_width,
            self.stages,
        ):
            raise ValueError("V19 cumulative coefficient width drift")
        if stage_active.shape != (batch, self.stages):
            raise ValueError("V19 stage mask width drift")
        if residual_energy.shape != (batch, self.stages):
            raise ValueError("V19 residual energy width drift")
        if residual_scale_before.shape != (batch, self.stages):
            raise ValueError("V19 residual scale width drift")
        if stage_explained_fraction.shape != (batch, self.stages):
            raise ValueError("V19 stage explained-fraction width drift")
        if lambda_max.shape != (batch,):
            raise ValueError(
                f"V19 lambda_max must be {(batch,)}, got {tuple(lambda_max.shape)}"
            )

        stage_mask = stage_active[:, None, :].float()
        residual_masked = residual * stage_mask
        flat_rows = residual_masked.reshape(batch * probe_rows, self.stages)
        query = self.row_query(flat_rows).reshape(
            batch, probe_rows, self.stages
        ) * stage_mask
        key = self.row_key(flat_rows).reshape(
            batch, probe_rows, self.stages
        ) * stage_mask
        interaction = torch.einsum("bns,bnt->bst", query, key) / float(
            max(1, probe_rows)
        )
        interaction_code = self.interaction_proj(interaction.reshape(batch, -1))

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
            trajectory=trajectory,
        )
        task_code = self.fusion_mlp(fusion_input)

        lambda_unit = torch.sigmoid(self.lambda_head(task_code).squeeze(-1))
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

        result: dict[str, Any] = {
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
        if return_representation:
            result.update(
                {
                    "v17_stage_interaction": interaction,
                    "v17_task_code": task_code,
                    "v17_penalty_code": penalty_code,
                    "v17_trajectory": trajectory,
                    "v19_stage_interaction": interaction,
                    "v19_residual_coefficient_code": residual_code,
                    "v19_marginal_coefficient_code": marginal_code,
                    "v19_cumulative_coefficient_code": cumulative_code,
                    "v19_coefficient_concat": coefficient_concat,
                    "v19_coefficient_code": coefficient_code,
                    "v19_task_code": task_code,
                    "v19_penalty_code": penalty_code,
                    "v19_trajectory": trajectory,
                    **fusion_representation,
                }
            )
        return result


__all__ = [
    "DirectPenaltyMLPQuerySelectorV19StageGain",
    "V19_CUMULATIVE_PATH_WIDTH",
    "V19_CUMULATIVE_SLOT_WIDTH",
    "V19_FEATURE_CHUNK_SIZE",
    "V19_PENALTY_RESIDUAL_LAYERS",
    "V19_SELECTOR_VERSION",
    "residual_isis_trajectory_v19_all_features",
]
