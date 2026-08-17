#!/usr/bin/env python3
"""V16 direct lambda/penalty selector from an ISIS residual trajectory.

The selector consumes only standardized context ``X`` and ``y``. A fixed,
penalty-independent ridge trajectory is used as a geometry encoder; it is not
the final beta estimate. The trajectory follows the handwritten V16 design:

* select K features per residual-ISIS stage;
* retain raw residual and marginal local ridge coefficients;
* apply a deterministic 1% stage gain to residual coefficients and residuals;
* form a row-permutation-invariant stage interaction matrix from residuals;
* fuse the interaction and coefficient codes at width 1024; and
* emit one effective lambda ratio and one canonical Direct96 penalty.

The expensive iterative path is restricted to a one-pass marginal candidate
pool and at most ``max_probe_rows`` content-stratified rows. This changes the
large-task cost from O(stages*N*P) to O(n_probe*P + stages*n_probe*M).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from .canonical_penalty import (
    DEFAULT_DERIVATIVE_FLOOR,
    DEFAULT_SQUAREPLUS_SCALE,
    build_penalty,
)
from .selector_v10 import lambda_from_effective_ratio_v56
from .selector_v11_canonical import CanonicalDirect96Head


Tensor = torch.Tensor

V16_SELECTOR_VERSION = (
    "v16_direct96_residualtrajectory_isis4x256_"
    "stagegain1pct_hidden512_candidate1024_probe2048_rawresidual_v4"
)
V16_STAGES = 256
V16_TOPK = 4
V16_CANDIDATE_POOL = 1024
V16_MAX_PROBE_ROWS = 2048
V16_HIDDEN_DIM = 512
V16_RIDGE_RATIO = 1e-3
V16_RIDGE_FLOOR = 1e-5
V16_RESIDUAL_SCALE_FLOOR = 1e-4
V16_STAGE_GAIN_RATE = 0.01
V16_LAMBDA_MIN_RATIO = 1e-6
V16_LAMBDA_MAX_RATIO = 1.0
V16_INITIAL_LAMBDA_RATIO = 0.1


def _batch_gather_rows(values: Tensor, indices: Tensor) -> Tensor:
    if values.ndim == 2:
        return torch.gather(values, 1, indices)
    if values.ndim == 3:
        return torch.gather(
            values,
            1,
            indices[:, :, None].expand(-1, -1, values.shape[-1]),
        )
    raise ValueError(f"row gather requires rank 2 or 3, got {values.ndim}")


def _content_stratified_rows(y: Tensor, max_rows: int) -> Tensor:
    """Select approximately uniform response quantiles without using row order."""
    batch, rows = y.shape
    if rows <= int(max_rows):
        return torch.arange(rows, device=y.device)[None, :].expand(batch, -1)
    order = torch.argsort(y.float(), dim=1, stable=True)
    positions = torch.linspace(
        0,
        rows - 1,
        int(max_rows),
        device=y.device,
        dtype=torch.float64,
    ).round().long()
    return torch.gather(order, 1, positions[None, :].expand(batch, -1))


def _pad_last(values: Tensor, width: int, value: float | int = 0) -> Tensor:
    current = int(values.shape[-1])
    if current > int(width):
        raise ValueError(f"cannot pad width {current} down to {width}")
    if current == int(width):
        return values
    shape = list(values.shape)
    shape[-1] = int(width) - current
    padding = torch.full(
        shape,
        value,
        device=values.device,
        dtype=values.dtype,
    )
    return torch.cat([values, padding], dim=-1)


@torch.no_grad()
def residual_isis_trajectory_v16(
    x_context: Tensor,
    y_context: Tensor,
    *,
    feature_mask: Tensor | None = None,
    stages: int = V16_STAGES,
    topk: int = V16_TOPK,
    candidate_pool: int = V16_CANDIDATE_POOL,
    max_probe_rows: int = V16_MAX_PROBE_ROWS,
    ridge_ratio: float = V16_RIDGE_RATIO,
    ridge_floor: float = V16_RIDGE_FLOOR,
    residual_scale_floor: float = V16_RESIDUAL_SCALE_FLOOR,
    stage_gain_rate: float = V16_STAGE_GAIN_RATE,
    eps: float = 1e-8,
) -> dict[str, Tensor | int | float | str]:
    """Build the fixed-width V16 path from variable-size context tensors.

    Inputs are expected to use the same context-only standardization as V58.
    Candidate columns are re-centered and re-scaled on probe rows, making the
    path robust to small preprocessing drift and feature rescaling.
    """
    if x_context.ndim != 3 or y_context.ndim != 2:
        raise ValueError(
            f"V16 expects X=(B,N,P), y=(B,N), got "
            f"{tuple(x_context.shape)} and {tuple(y_context.shape)}"
        )
    batch, rows, features = x_context.shape
    if y_context.shape != (batch, rows):
        raise ValueError("V16 context X/y row alignment drift")
    if rows < 2 or features < 1:
        raise ValueError(f"V16 requires N>=2 and P>=1, got N={rows}, P={features}")
    if int(stages) < 1 or int(topk) < 1 or int(candidate_pool) < 1:
        raise ValueError("V16 stages, topk, and candidate_pool must be positive")
    if int(max_probe_rows) < 2:
        raise ValueError("V16 max_probe_rows must be at least two")
    if float(ridge_ratio) < 0.0 or float(ridge_floor) <= 0.0:
        raise ValueError("V16 ridge ratio/floor contract is invalid")
    if float(residual_scale_floor) <= 0.0:
        raise ValueError("V16 residual scale floor must be positive")
    if not math.isfinite(float(stage_gain_rate)) or float(stage_gain_rate) < 0.0:
        raise ValueError("V16 stage gain rate must be finite and nonnegative")

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
    x_probe = _batch_gather_rows(x_context, row_indices).float()
    y_probe_raw = _batch_gather_rows(y_context, row_indices).float()
    probe_rows = int(x_probe.shape[1])

    y_centered = y_probe_raw - y_probe_raw.mean(dim=1, keepdim=True)
    y_scale = y_centered.square().mean(dim=1, keepdim=True).sqrt().clamp_min(eps)
    y_unit = y_centered / y_scale

    x_mean = x_probe.mean(dim=1)
    x_second = x_probe.square().mean(dim=1)
    x_variance = (x_second - x_mean.square()).clamp_min(0.0)
    x_scale = x_variance.sqrt().clamp_min(eps)
    initial_rhs = torch.einsum("bnp,bn->bp", x_probe, y_unit) / float(probe_rows)
    initial_score = torch.nan_to_num(
        initial_rhs.abs() / x_scale,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    negative_large = torch.finfo(torch.float32).min
    initial_score = initial_score.masked_fill(~valid_features, negative_large)

    pool_width = min(features, max(int(candidate_pool), int(stages) * int(topk)))
    _, candidate_indices = torch.topk(
        initial_score,
        k=pool_width,
        dim=-1,
        largest=True,
        sorted=True,
    )
    candidate_valid = torch.gather(valid_features, 1, candidate_indices)
    candidate_rhs = torch.gather(initial_rhs, 1, candidate_indices)
    candidate_mean = torch.gather(x_mean, 1, candidate_indices)
    candidate_scale = torch.gather(x_scale, 1, candidate_indices)
    candidate_sign = torch.where(
        candidate_rhs >= 0.0,
        torch.ones_like(candidate_rhs),
        -torch.ones_like(candidate_rhs),
    )
    x_candidates = torch.gather(
        x_probe,
        2,
        candidate_indices[:, None, :].expand(batch, probe_rows, pool_width),
    )
    x_candidates = (
        (x_candidates - candidate_mean[:, None, :])
        / candidate_scale[:, None, :]
        * candidate_sign[:, None, :]
    )
    x_candidates = x_candidates * candidate_valid[:, None, :].float()
    candidate_diag = x_candidates.square().mean(dim=1).clamp_min(eps)
    residual = y_unit.clone()
    selected_candidates = torch.zeros_like(candidate_valid)
    residual_columns: list[Tensor] = []
    residual_betas_raw: list[Tensor] = []
    marginal_betas: list[Tensor] = []
    selected_global_indices: list[Tensor] = []
    stage_active_columns: list[Tensor] = []
    residual_energy_columns: list[Tensor] = []
    residual_scale_before_columns: list[Tensor] = []
    explained_fraction_columns: list[Tensor] = []

    select_width = min(int(topk), pool_width)
    for _stage in range(int(stages)):
        rhs_residual = torch.einsum(
            "bnm,bn->bm", x_candidates, residual
        ) / float(probe_rows)
        residual_energy_before = residual.square().mean(dim=1, keepdim=True)
        residual_scale_before = residual_energy_before.sqrt().clamp_min(
            float(residual_scale_floor)
        )
        score = rhs_residual.abs() / (
            candidate_diag.sqrt() * residual_scale_before
        ).clamp_min(eps)
        available = candidate_valid & ~selected_candidates
        score = torch.nan_to_num(
            score, nan=0.0, posinf=0.0, neginf=0.0
        ).masked_fill(~available, negative_large)
        _, positions_raw = torch.topk(
            score,
            k=select_width,
            dim=-1,
            largest=True,
            sorted=True,
        )
        selected_valid_raw = torch.gather(available, 1, positions_raw)

        x_stage_raw = torch.gather(
            x_candidates,
            2,
            positions_raw[:, None, :].expand(batch, probe_rows, select_width),
        )
        x_stage_raw = x_stage_raw * selected_valid_raw[:, None, :].float()
        global_indices_raw = torch.gather(candidate_indices, 1, positions_raw)

        x_stage = _pad_last(x_stage_raw, int(topk), value=0.0)
        selected_valid = _pad_last(selected_valid_raw, int(topk), value=0)
        global_indices = _pad_last(
            global_indices_raw,
            int(topk),
            value=-1,
        ).masked_fill(~selected_valid, -1)

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
        alpha = (
            float(ridge_ratio) * mean_diagonal
        ).clamp_min(float(ridge_floor))
        eye = torch.eye(
            int(topk), device=x_context.device, dtype=torch.float32
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
        update = torch.zeros_like(selected_candidates)
        update.scatter_(1, positions_raw, selected_valid_raw)
        selected_candidates |= update

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
        y_unit >= 0.0,
        torch.ones_like(y_unit),
        -torch.ones_like(y_unit),
    )
    residual_trajectory_raw = residual_trajectory_raw * response_sign.unsqueeze(-1)
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
    beta_marginal = torch.stack(marginal_betas, dim=-1)
    selected_indices = torch.stack(selected_global_indices, dim=1)
    stage_active = torch.stack(stage_active_columns, dim=-1)
    residual_energy_ratio = torch.stack(residual_energy_columns, dim=-1)
    residual_scale_before = torch.stack(residual_scale_before_columns, dim=-1)
    stage_explained_fraction = torch.stack(explained_fraction_columns, dim=-1)

    expected_r = (batch, probe_rows, int(stages))
    expected_b = (batch, int(topk), int(stages))
    if residual_trajectory.shape != expected_r:
        raise RuntimeError(
            f"V16 residual trajectory shape {tuple(residual_trajectory.shape)} "
            f"!= {expected_r}"
        )
    if (
        beta_residual.shape != expected_b
        or beta_residual_raw.shape != expected_b
        or beta_marginal.shape != expected_b
    ):
        raise RuntimeError("V16 coefficient trajectory shape drift")
    if selected_indices.shape != (batch, int(stages), int(topk)):
        raise RuntimeError("V16 selected-index shape drift")
    tensors = (
        residual_trajectory,
        residual_trajectory_raw,
        beta_residual,
        beta_residual_raw,
        beta_marginal,
        residual_energy_ratio,
        residual_scale_before,
        stage_explained_fraction,
        stage_gain,
    )
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise RuntimeError("V16 trajectory contains nonfinite values")

    return {
        "residual_trajectory": residual_trajectory,
        "residual_trajectory_raw": residual_trajectory_raw,
        "beta_residual": beta_residual,
        "beta_residual_raw": beta_residual_raw,
        "beta_marginal": beta_marginal,
        "selected_indices": selected_indices,
        "stage_active": stage_active,
        "residual_energy_ratio": residual_energy_ratio,
        "residual_scale_before": residual_scale_before,
        "stage_explained_fraction": stage_explained_fraction,
        "stage_gain": stage_gain,
        "probe_row_indices": row_indices,
        "candidate_indices": candidate_indices,
        "candidate_valid": candidate_valid,
        "probe_rows": probe_rows,
        "candidate_pool_width": pool_width,
        "ridge_ratio": float(ridge_ratio),
        "residual_scale_floor": float(residual_scale_floor),
        "stage_gain_rate": float(stage_gain_rate),
        "trajectory_contract": (
            "response_quantile_rows_allP_marginalscreen_"
            "aligned_candidate_rawresidual_stagegain1pct_v4"
        ),
    }


class _TwoLinearResidualMap(nn.Module):
    """Two-linear MLP with the handwritten outer residual and LayerNorm."""

    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.pre_norm = nn.LayerNorm(int(width))
        self.fc1 = nn.Linear(int(width), int(hidden))
        self.fc2 = nn.Linear(int(hidden), int(width))
        self.post_norm = nn.LayerNorm(int(width))
        self.act = nn.GELU()

    def forward(self, values: Tensor) -> Tensor:
        delta = self.fc2(self.act(self.fc1(self.pre_norm(values))))
        return self.post_norm(values + delta)


class _ThreeLinearResidualMap(nn.Module):
    """Three-linear width-preserving MLP with an outer residual and LayerNorm."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.pre_norm = nn.LayerNorm(int(width))
        self.fc1 = nn.Linear(int(width), int(width))
        self.fc2 = nn.Linear(int(width), int(width))
        self.fc3 = nn.Linear(int(width), int(width))
        self.post_norm = nn.LayerNorm(int(width))
        self.act = nn.GELU()

    def forward(self, values: Tensor) -> Tensor:
        hidden = self.act(self.fc1(self.pre_norm(values)))
        hidden = self.act(self.fc2(hidden))
        return self.post_norm(values + self.fc3(hidden))


class DirectPenaltyMLPQuerySelectorV16ResidualTrajectory(nn.Module):
    """Direct V16 selector implementing the two-page residual-path design."""

    def __init__(
        self,
        *,
        stages: int = V16_STAGES,
        topk: int = V16_TOPK,
        candidate_pool: int = V16_CANDIDATE_POOL,
        max_probe_rows: int = V16_MAX_PROBE_ROWS,
        hidden: int = V16_HIDDEN_DIM,
        ridge_ratio: float = V16_RIDGE_RATIO,
        ridge_floor: float = V16_RIDGE_FLOOR,
        residual_scale_floor: float = V16_RESIDUAL_SCALE_FLOOR,
        stage_gain_rate: float = V16_STAGE_GAIN_RATE,
        lambda_min_ratio: float = V16_LAMBDA_MIN_RATIO,
        lambda_max_ratio: float = V16_LAMBDA_MAX_RATIO,
        initial_lambda_ratio: float = V16_INITIAL_LAMBDA_RATIO,
        derivative_floor: float = DEFAULT_DERIVATIVE_FLOOR,
        squareplus_scale: float = DEFAULT_SQUAREPLUS_SCALE,
        direct_head_init_std: float = 1e-3,
    ) -> None:
        super().__init__()
        if int(stages) != V16_STAGES or int(topk) != V16_TOPK:
            raise ValueError(
                f"V16 fixes stages={V16_STAGES} and topk={V16_TOPK}"
            )
        if int(hidden) != V16_HIDDEN_DIM:
            raise ValueError(f"V16 fixes hidden={V16_HIDDEN_DIM}")
        if int(candidate_pool) < int(stages) * int(topk):
            raise ValueError("V16 candidate pool cannot be smaller than K*stages")
        if float(residual_scale_floor) <= 0.0:
            raise ValueError("V16 residual scale floor must be positive")
        if not math.isclose(
            float(stage_gain_rate), V16_STAGE_GAIN_RATE, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"V16 fixes stage_gain_rate={V16_STAGE_GAIN_RATE}")
        if not (
            0.0 < float(lambda_min_ratio)
            <= float(initial_lambda_ratio)
            <= float(lambda_max_ratio)
        ):
            raise ValueError("V16 lambda-ratio bounds are invalid")

        self.stages = int(stages)
        self.topk = int(topk)
        self.candidate_pool = int(candidate_pool)
        self.max_probe_rows = int(max_probe_rows)
        self.hidden = int(hidden)
        self.ridge_ratio = float(ridge_ratio)
        self.ridge_floor = float(ridge_floor)
        self.residual_scale_floor = float(residual_scale_floor)
        self.stage_gain_rate = float(stage_gain_rate)
        self.lambda_min_ratio = float(lambda_min_ratio)
        self.lambda_max_ratio = float(lambda_max_ratio)
        self.selector_version = V16_SELECTOR_VERSION

        self.row_query = _TwoLinearResidualMap(self.stages, self.hidden)
        self.row_key = _TwoLinearResidualMap(self.stages, self.hidden)
        interaction_width = self.stages * self.stages
        coefficient_width = self.topk * self.stages
        metadata_width = 3 * self.stages
        self.interaction_proj = nn.Sequential(
            nn.LayerNorm(interaction_width),
            nn.Linear(interaction_width, self.hidden),
        )
        self.residual_coefficient_proj = nn.Sequential(
            nn.LayerNorm(coefficient_width),
            nn.Linear(coefficient_width, self.hidden),
        )
        self.marginal_coefficient_proj = nn.Sequential(
            nn.LayerNorm(coefficient_width),
            nn.Linear(coefficient_width, self.hidden),
        )
        self.stage_metadata_proj = nn.Sequential(
            nn.LayerNorm(metadata_width),
            nn.Linear(metadata_width, self.hidden),
        )
        self.fusion_mlp = _ThreeLinearResidualMap(self.hidden)
        self.lambda_head = nn.Linear(self.hidden, 1)
        nn.init.zeros_(self.lambda_head.weight)
        log_lo = math.log(self.lambda_min_ratio)
        log_hi = math.log(self.lambda_max_ratio)
        initial_unit = (
            math.log(float(initial_lambda_ratio)) - log_lo
        ) / (log_hi - log_lo)
        initial_unit = min(max(initial_unit, 1e-6), 1.0 - 1e-6)
        nn.init.constant_(
            self.lambda_head.bias,
            math.log(initial_unit / (1.0 - initial_unit)),
        )
        self.penalty_parameter_head = CanonicalDirect96Head(
            self.hidden,
            derivative_floor=float(derivative_floor),
            squareplus_scale=float(squareplus_scale),
            head_init_std=float(direct_head_init_std),
        )

    def fit_penalty_parameter_cache(
        self,
        direct_parameters: Tensor,
        relu_weights: Tensor | None = None,
    ) -> dict[str, Any]:
        if relu_weights is not None:
            raise ValueError("V16 calibration accepts one Direct96 tensor")
        return self.penalty_parameter_head.fit_parameter_cache(direct_parameters)

    def encode_context(
        self,
        x_context: Tensor,
        y_context: Tensor,
        *,
        feature_mask: Tensor | None = None,
    ) -> dict[str, Tensor | int | float | str]:
        return residual_isis_trajectory_v16(
            x_context,
            y_context,
            feature_mask=feature_mask,
            stages=self.stages,
            topk=self.topk,
            candidate_pool=self.candidate_pool,
            max_probe_rows=self.max_probe_rows,
            ridge_ratio=self.ridge_ratio,
            ridge_floor=self.ridge_floor,
            residual_scale_floor=self.residual_scale_floor,
            stage_gain_rate=self.stage_gain_rate,
        )

    def architecture_report(self) -> dict[str, Any]:
        return {
            "selector_version": self.selector_version,
            "stages": self.stages,
            "topk_per_stage": self.topk,
            "candidate_pool": self.candidate_pool,
            "max_probe_rows": self.max_probe_rows,
            "hidden_dim": self.hidden,
            "row_mlp_linear_layers": 2,
            "fusion_mlp_linear_layers": 3,
            "interaction_shape": [self.stages, self.stages],
            "coefficient_shape": [self.topk, self.stages],
            "residual_coefficient_normalization": "none",
            "residual_stage_gain": "1.01^(stage_index_zero_based)",
            "residual_stage_gain_rate": self.stage_gain_rate,
            "terminal_residual_stage_gain": (1.0 + self.stage_gain_rate)
            ** (self.stages - 1),
            "residual_trajectory_gain_applied_posthoc": True,
            "marginal_coefficient_stage_gain": False,
            "residual_scale_floor": self.residual_scale_floor,
            "stage_metadata_coordinates": [
                "active",
                "log_residual_scale_before",
                "explained_fraction",
            ],
            "uses_beta_solver_input": False,
            "uses_penalty_candidates": False,
            "outputs_direct96": True,
            "total_parameters": sum(p.numel() for p in self.parameters()),
        }

    def forward_from_trajectory(
        self,
        trajectory: dict[str, Tensor | int | float | str],
        lambda_max: Tensor,
        *,
        return_representation: bool = False,
    ) -> dict[str, Any]:
        residual = trajectory["residual_trajectory"]
        beta_residual = trajectory["beta_residual"]
        beta_marginal = trajectory["beta_marginal"]
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
                stage_active,
                residual_energy,
                residual_scale_before,
                stage_explained_fraction,
            )
        ):
            raise TypeError("V16 trajectory tensor contract is incomplete")

        residual = residual.float()
        beta_residual = beta_residual.float()
        beta_marginal = beta_marginal.float()
        stage_active = stage_active.bool()
        residual_energy = residual_energy.float()
        residual_scale_before = residual_scale_before.float()
        stage_explained_fraction = stage_explained_fraction.float()
        batch, probe_rows, stages = residual.shape
        if stages != self.stages:
            raise ValueError("V16 residual stage width drift")
        if beta_residual.shape != (batch, self.topk, self.stages):
            raise ValueError("V16 residual coefficient width drift")
        if beta_marginal.shape != beta_residual.shape:
            raise ValueError("V16 marginal coefficient width drift")
        if stage_active.shape != (batch, self.stages):
            raise ValueError("V16 stage mask width drift")
        if residual_energy.shape != (batch, self.stages):
            raise ValueError("V16 residual energy width drift")
        if residual_scale_before.shape != (batch, self.stages):
            raise ValueError("V16 residual scale width drift")
        if stage_explained_fraction.shape != (batch, self.stages):
            raise ValueError("V16 stage explained-fraction width drift")
        if lambda_max.shape != (batch,):
            raise ValueError(
                f"V16 lambda_max must be {(batch,)}, got {tuple(lambda_max.shape)}"
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
        coefficient_code = self.residual_coefficient_proj(
            beta_residual.reshape(batch, -1)
        ) + self.marginal_coefficient_proj(beta_marginal.reshape(batch, -1))
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
        task_code = self.fusion_mlp(
            interaction_code + coefficient_code + metadata_code
        )

        lambda_unit = torch.sigmoid(self.lambda_head(task_code).squeeze(-1))
        lambda_ratio = torch.exp(
            math.log(self.lambda_min_ratio)
            + (
                math.log(self.lambda_max_ratio)
                - math.log(self.lambda_min_ratio)
            )
            * lambda_unit
        ).clamp(self.lambda_min_ratio, self.lambda_max_ratio)
        encoded = self.penalty_parameter_head(task_code)
        shape = encoded["shape_features"]
        d0 = encoded["d0"]
        lambda_scale = lambda_from_effective_ratio_v56(
            lambda_ratio,
            lambda_max.float(),
            d0,
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
                    "v16_stage_interaction": interaction,
                    "v16_task_code": task_code,
                    "v16_trajectory": trajectory,
                }
            )
        return result

    def forward(
        self,
        x_context: Tensor,
        y_context: Tensor,
        lambda_max: Tensor,
        *,
        feature_mask: Tensor | None = None,
        return_representation: bool = False,
    ) -> dict[str, Any]:
        trajectory = self.encode_context(
            x_context,
            y_context,
            feature_mask=feature_mask,
        )
        return self.forward_from_trajectory(
            trajectory,
            lambda_max,
            return_representation=return_representation,
        )


__all__ = [
    "DirectPenaltyMLPQuerySelectorV16ResidualTrajectory",
    "V16_CANDIDATE_POOL",
    "V16_HIDDEN_DIM",
    "V16_INITIAL_LAMBDA_RATIO",
    "V16_MAX_PROBE_ROWS",
    "V16_RIDGE_FLOOR",
    "V16_RIDGE_RATIO",
    "V16_RESIDUAL_SCALE_FLOOR",
    "V16_SELECTOR_VERSION",
    "V16_STAGES",
    "V16_STAGE_GAIN_RATE",
    "V16_TOPK",
    "residual_isis_trajectory_v16",
]
