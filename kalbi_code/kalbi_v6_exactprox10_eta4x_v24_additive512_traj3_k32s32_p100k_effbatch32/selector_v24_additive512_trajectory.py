#!/usr/bin/env python3
"""V24 selector: V23 proposal plus additive global/solver/proposal corrections."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn

from .canonical_penalty import build_penalty, canonical_grid
from .exact_direct96_prox import (
    exact_direct96_prox,
    interpolate_direct96_penalty,
    unpack_direct96_penalty,
)
from .model import (
    KALBI_EXACT_PG_CURVATURE_MARGIN,
    KALBI_EXACT_PG_FEATURE_CHUNK,
    _select_exact_pg_step,
)
from .selector_v23_independent_rawlambda import (
    DirectPenaltyMLPQuerySelectorV23IndependentRawLambda,
    V23_RAW_LAMBDA_MAX,
    V23_RAW_LAMBDA_MIN,
)


Tensor = torch.Tensor

V24_SELECTOR_VERSION = (
    "v24_v23_additive512_global_solvertrajectory3_proposal_"
    "residual4_independent_rawlambda_v1"
)
V24_PROBE_STEPS = 3
V24_TRAJECTORY_STATS = 14
V24_GLOBAL_DIM = 33
V24_PROPOSAL_DIM = 21
V24_SCALER_VERSION = "v24_fixed_coordinate_scaler_global33_traj3x14_prop21_v1"
V24_SCALER_TRANSFORM_CONTRACT = (
    "v24_global33_traj3x14_prop21_coordinate_transforms_"
    "median_mad_clip8_v1"
)
V24_LAMBDA_OUTPUT_CONTRACT = (
    "v23_independent_raw_lambda_plus_residual_loglambda_correction_v1"
)


class FixedCoordinateScalerV24(nn.Module):
    """Checkpointed fixed scaler for the three new V24 source branches."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("global_center", torch.zeros(V24_GLOBAL_DIM))
        self.register_buffer("global_scale", torch.ones(V24_GLOBAL_DIM))
        self.register_buffer(
            "trajectory_center",
            torch.zeros(V24_PROBE_STEPS, V24_TRAJECTORY_STATS),
        )
        self.register_buffer(
            "trajectory_scale",
            torch.ones(V24_PROBE_STEPS, V24_TRAJECTORY_STATS),
        )
        self.register_buffer("proposal_center", torch.zeros(V24_PROPOSAL_DIM))
        self.register_buffer("proposal_scale", torch.ones(V24_PROPOSAL_DIM))
        self.register_buffer("cache_loaded", torch.tensor(False))
        self.register_buffer("cache_size", torch.tensor(0, dtype=torch.long))
        self.register_buffer("cache_sha256_bytes", torch.zeros(32, dtype=torch.uint8))

    @staticmethod
    def transform(name: str, value: Tensor) -> Tensor:
        source = value.float()
        if name == "global":
            return torch.cat(
                [
                    source[..., :9],
                    source[..., 9:11].sign()
                    * torch.log1p(source[..., 9:11].abs()),
                    source[..., 11:13].clamp(0.0, 1.0),
                    torch.log1p(source[..., 13:29].clamp_min(0.0)),
                    torch.log(source[..., 29:33].clamp_min(1e-12)),
                ],
                dim=-1,
            )
        if name == "trajectory":
            return torch.cat(
                [
                    torch.log1p(source[..., 0:2].clamp_min(0.0)),
                    torch.log(source[..., 2:4].clamp_min(1e-12)),
                    torch.log1p(source[..., 4:5].clamp_min(0.0)),
                    source[..., 5:8].clamp(0.0, 1.0),
                    torch.log(source[..., 8:14].clamp_min(1e-12)),
                ],
                dim=-1,
            )
        if name == "proposal":
            return torch.cat(
                [
                    source[..., :1],
                    torch.log(source[..., 1:14].clamp_min(1e-12)),
                    torch.log1p(source[..., 14:15].clamp_min(0.0)),
                    source[..., 15:17].clamp(0.0, 1.0),
                    torch.log(source[..., 17:18].clamp_min(1e-12)),
                    torch.log1p(source[..., 18:19].clamp_min(0.0)),
                    torch.log(source[..., 19:20].clamp_min(1e-12)),
                    torch.log1p(source[..., 20:21].clamp_min(0.0)),
                ],
                dim=-1,
            )
        raise KeyError(f"unknown V24 scaler source {name!r}")

    @torch.no_grad()
    def load_cache_payload(self, payload: Mapping[str, Any]) -> None:
        if payload.get("version") != V24_SCALER_VERSION:
            raise ValueError("V24 scaler-cache version mismatch")
        if payload.get("transform_contract") != V24_SCALER_TRANSFORM_CONTRACT:
            raise ValueError("V24 scaler transform-contract mismatch")
        for name in ("global", "trajectory", "proposal"):
            center = torch.as_tensor(payload[f"{name}_center"]).float()
            scale = torch.as_tensor(payload[f"{name}_scale"]).float()
            target_center = getattr(self, f"{name}_center")
            target_scale = getattr(self, f"{name}_scale")
            if (
                center.shape != target_center.shape
                or scale.shape != target_scale.shape
            ):
                raise ValueError(f"V24 {name} scaler shape mismatch")
            if not bool(
                torch.isfinite(center).all() and torch.isfinite(scale).all()
            ):
                raise ValueError(f"V24 {name} scaler is nonfinite")
            target_center.copy_(center)
            target_scale.copy_(scale.clamp_min(1e-4))
        self.cache_size.fill_(int(payload["cache_size"]))
        cache_sha = str(payload["cache_sha256"])
        if len(cache_sha) != 64:
            raise ValueError("V24 scaler SHA256 must have 64 hex characters")
        self.cache_sha256_bytes.copy_(
            torch.tensor(list(bytes.fromhex(cache_sha)), dtype=torch.uint8)
        )
        self.cache_loaded.fill_(True)

    def scale(self, name: str, value: Tensor) -> Tensor:
        if not bool(self.cache_loaded.item()):
            raise RuntimeError("V24 fixed input scaler is not loaded")
        transformed = self.transform(name, value)
        center = getattr(self, f"{name}_center").to(transformed)
        scale = getattr(self, f"{name}_scale").to(transformed)
        return ((transformed - center) / scale).clamp(-8.0, 8.0)


class _ResidualLinear512(nn.Module):
    def __init__(self, width: int = 512) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear = nn.Linear(width, width)
        self.gate = nn.Parameter(torch.tensor(0.01))

    def forward(self, value: Tensor) -> tuple[Tensor, Tensor]:
        update = torch.nn.functional.gelu(self.linear(self.norm(value)))
        return value + self.gate * update, update


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    weight = mask.float()
    return (value * weight).sum(-1) / weight.sum(-1).clamp_min(1.0)


def _masked_rms(value: Tensor, mask: Tensor) -> Tensor:
    return _masked_mean(value.square(), mask).clamp_min(0.0).sqrt()


def _masked_fraction(value: Tensor, mask: Tensor) -> Tensor:
    return _masked_mean(value.float(), mask)


def _quantiles(value: Tensor, levels: tuple[float, ...]) -> Tensor:
    return torch.quantile(
        value.float(),
        torch.tensor(levels, device=value.device, dtype=torch.float32),
        dim=-1,
    ).transpose(0, 1)


class DirectPenaltyMLPQuerySelectorV24Additive512(
    DirectPenaltyMLPQuerySelectorV23IndependentRawLambda
):
    """Two-pass V24 selector with a differentiable three-step solver probe."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.hidden != 512:
            raise ValueError("V24 fixes hidden_dim=512")
        self.v24_scaler = FixedCoordinateScalerV24()
        self.global_projection = nn.Sequential(
            nn.Linear(V24_GLOBAL_DIM, self.hidden),
            nn.GELU(),
            nn.LayerNorm(self.hidden),
        )
        self.trajectory_projection = nn.Sequential(
            nn.Linear(V24_PROBE_STEPS * V24_TRAJECTORY_STATS, self.hidden),
            nn.GELU(),
            nn.LayerNorm(self.hidden),
        )
        self.proposal_projection = nn.Sequential(
            nn.Linear(V24_PROPOSAL_DIM, self.hidden),
            nn.GELU(),
            nn.LayerNorm(self.hidden),
        )
        self.initial_fusion_norm = nn.LayerNorm(self.hidden)
        self.initial_norm_gate = nn.Parameter(torch.tensor(0.0))
        self.corrected_fusion_norm = nn.LayerNorm(self.hidden)
        self.global_gate = nn.Parameter(torch.tensor(1e-3))
        self.trajectory_gate = nn.Parameter(torch.tensor(1e-3))
        self.proposal_gate = nn.Parameter(torch.tensor(1e-3))
        self.residual_layers = nn.ModuleList(
            [_ResidualLinear512(self.hidden) for _ in range(4)]
        )
        self.final_norm = nn.LayerNorm(self.hidden)
        self.delta_loglambda_head = nn.Linear(self.hidden, 1)
        self.delta_penalty_latent_head = nn.Linear(self.hidden, 96)
        nn.init.zeros_(self.delta_loglambda_head.weight)
        nn.init.zeros_(self.delta_loglambda_head.bias)
        nn.init.zeros_(self.delta_penalty_latent_head.weight)
        nn.init.zeros_(self.delta_penalty_latent_head.bias)
        self.register_buffer("_kalbi_v6_penalty_grid", canonical_grid())
        self.selector_version = V24_SELECTOR_VERSION

    @torch.no_grad()
    def load_v24_scaler_cache(self, payload: Mapping[str, Any]) -> None:
        self.v24_scaler.load_cache_payload(payload)

    def _raw_lambda(self, task_code: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        logit = self.lambda_head(task_code).squeeze(-1)
        unit = torch.sigmoid(logit)
        log_lambda = (
            math.log(self.raw_lambda_min)
            + (math.log(self.raw_lambda_max) - math.log(self.raw_lambda_min))
            * unit
        )
        return log_lambda.exp(), logit, unit

    def _global_summary(
        self,
        x: Tensor,
        y: Tensor,
        feature_mask: Tensor | None,
    ) -> Tensor:
        batch, rows, features = x.shape
        mask = (
            torch.ones(batch, features, device=x.device, dtype=torch.bool)
            if feature_mask is None
            else feature_mask.bool()
        )
        y32 = y.float()
        x32 = x.float()
        yq = _quantiles(
            y32, (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
        )
        ymean = y32.mean(-1, keepdim=True)
        ystd = y32.std(-1, unbiased=False, keepdim=True).clamp_min(1e-6)
        yz = (y32 - ymean) / ystd
        ymoments = torch.stack(
            [
                yz.pow(3).mean(-1),
                yz.pow(4).mean(-1) - 3.0,
                (yz.abs() > 2.0).float().mean(-1),
                (yz.abs() > 3.0).float().mean(-1),
            ],
            dim=-1,
        )
        marginal = (
            torch.einsum("bnp,bn->bp", x32, y32) / float(max(1, rows))
        ).abs()
        marginal = marginal * mask.float()
        valid_marginal = torch.where(
            mask, marginal, torch.full_like(marginal, float("nan"))
        )
        # Masked quantiles are evaluated per task to avoid padding pollution.
        mq = torch.stack(
            [
                torch.quantile(
                    valid_marginal[index][mask[index]],
                    torch.tensor(
                        (0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999, 1.0),
                        device=x.device,
                    ),
                )
                for index in range(batch)
            ],
            dim=0,
        )
        total = marginal.sum(-1).clamp_min(1e-12)
        sorted_marginal = marginal.sort(dim=-1, descending=True).values
        concentration = torch.stack(
            [
                _masked_mean(marginal, mask),
                _masked_rms(marginal, mask),
                marginal.amax(-1),
                sorted_marginal[:, :1].sum(-1) / total,
                sorted_marginal[:, :8].sum(-1) / total,
                sorted_marginal[:, :32].sum(-1) / total,
            ],
            dim=-1,
        )
        gram_diag = x32.square().mean(1) * mask.float()
        trace = gram_diag.sum(-1).clamp_min(1e-8)
        # Four deterministic matrix-free probes estimate ||X'X/N||_F^2.
        # Data-derived deterministic probes are feature-permutation equivariant.
        # Coordinate-index random vectors would break the selector contract.
        feature_signature = (
            marginal
            + 0.1 * x32.pow(3).mean(1).abs()
            + 0.01 * x32.pow(4).mean(1)
        )
        probes = torch.stack(
            [
                torch.sign(
                    torch.sin(
                        (feature_signature + 1e-4)
                        * float(k + 1)
                        * 12.9898
                    )
                )
                for k in range(4)
            ],
            dim=1,
        )
        probes = probes * mask[:, None, :].float()
        xv = torch.einsum("bnp,bkp->bkn", x32, probes)
        gv = torch.einsum("bnp,bkn->bkp", x32, xv) / float(max(1, rows))
        frob_sq = gv.square().sum(-1).mean(-1).clamp_min(1e-8)
        effective_rank = trace.square() / frob_sq
        power = probes[:, 0]
        for _ in range(3):
            xp = torch.einsum("bnp,bp->bn", x32, power)
            power = (
                torch.einsum("bnp,bn->bp", x32, xp) / float(max(1, rows))
            ) * mask.float()
            power = power / power.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-8)
        xp = torch.einsum("bnp,bp->bn", x32, power)
        leading = xp.square().sum(-1) / float(max(1, rows))
        spectral = torch.stack(
            [
                trace / mask.sum(-1).clamp_min(1),
                frob_sq.sqrt() / trace,
                effective_rank / mask.sum(-1).clamp_min(1),
                leading / trace,
            ],
            dim=-1,
        )
        result = torch.cat([yq, ymoments, mq, concentration, spectral], dim=-1)
        if result.shape[-1] != V24_GLOBAL_DIM:
            raise RuntimeError("V24 global-summary width drift")
        return result

    def _proposal_summary(
        self,
        raw_lambda: Tensor,
        encoded: Mapping[str, Tensor],
        lambda_max: Tensor,
    ) -> Tensor:
        drho = encoded["drho"].float()
        d0 = encoded["d0"].float()
        lambda_max_safe = lambda_max.float().clamp_min(1e-12)
        max_drho = drho.amax(-1)
        ratios = torch.stack(
            [
                raw_lambda / lambda_max_safe,
                raw_lambda * d0 / lambda_max_safe,
                raw_lambda * max_drho / lambda_max_safe,
            ],
            dim=-1,
        )
        drho_q = _quantiles(
            drho, (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)
        )
        differences = drho[..., 1:] - drho[..., :-1]
        extra = torch.stack(
            [
                differences.abs().sum(-1),
                (differences > 1e-7).float().mean(-1),
                (drho[..., -1] <= 0.02 * max_drho).float(),
            ],
            dim=-1,
        )
        result = torch.cat(
            [
                raw_lambda.clamp_min(1e-12).log()[:, None],
                ratios,
                torch.stack([d0, drho.amin(-1), max_drho], dim=-1),
                drho_q,
                extra,
                encoded["metadata4"].float(),
            ],
            dim=-1,
        )
        if result.shape[-1] != V24_PROPOSAL_DIM:
            raise RuntimeError("V24 proposal-summary width drift")
        return result

    def _probe(
        self,
        solver_context: Mapping[str, Tensor],
        raw_lambda: Tensor,
        encoded: Mapping[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        x = solver_context["Xc"].float()
        y = solver_context["yc"].float()
        feature_mask = solver_context.get("feature_mask")
        batch, rows, features = x.shape
        valid = (
            torch.ones(batch, features, device=x.device, dtype=torch.bool)
            if feature_mask is None
            else feature_mask.bool()
        )
        beta = torch.zeros(batch, 1, features, device=x.device)
        lam = raw_lambda[:, None]
        native = encoded["shape_features"][:, None, :]
        d0 = encoded["d0"][:, None]
        pieces = unpack_direct96_penalty(native, self._kalbi_v6_penalty_grid)
        statistics: list[Tensor] = []
        eta_values: list[Tensor] = []
        gamma_values: list[Tensor] = []
        for _ in range(V24_PROBE_STEPS):
            prediction = torch.matmul(beta, x.transpose(1, 2))
            residual = prediction - y[:, None, :]
            gradient = torch.matmul(residual, x) / float(max(1, rows))
            with torch.no_grad():
                step = _select_exact_pg_step(
                    self, dict(solver_context), beta, residual, gradient,
                    lam, native,
                )
            eta = step["step_size"]
            gamma = eta * lam
            proposal = beta - eta[..., None] * gradient
            beta_next, prox_diag = exact_direct96_prox(
                proposal,
                gamma,
                native,
                self._kalbi_v6_penalty_grid,
                feature_chunk=KALBI_EXACT_PG_FEATURE_CHUNK,
                curvature_margin=KALBI_EXACT_PG_CURVATURE_MARGIN,
            )
            if feature_mask is not None:
                beta_next = beta_next * valid[:, None, :].float()
            magnitude = proposal.abs()
            h_knots = self._kalbi_v6_penalty_grid.to(native) + (
                gamma[..., None] * pieces.drho
            )
            upper = torch.searchsorted(
                h_knots.reshape(batch, 96),
                magnitude.reshape(batch, features).contiguous(),
                right=True,
            )
            zero_branch = upper == 0
            saturated = upper >= 96
            lower = (upper - 1).clamp(0, 94)
            slope = torch.gather(
                pieces.slopes.reshape(batch, 95), 1, lower
            ).reshape(batch, 1, features)
            denominator = 1.0 + gamma[..., None] * slope
            prox_derivative = torch.where(
                zero_branch[:, None] | saturated[:, None],
                torch.zeros_like(denominator),
                denominator.reciprocal(),
            )
            z_rms = _masked_rms(proposal[:, 0], valid).clamp_min(1e-8)
            boundary = (
                (proposal[:, 0].abs() - gamma[:, 0, None] * d0[:, 0, None])
                .abs() / z_rms[:, None]
            )
            boundary_q05 = torch.stack(
                [
                    torch.quantile(boundary[i][valid[i]], 0.05)
                    for i in range(batch)
                ]
            )
            prediction_next = torch.matmul(beta_next, x.transpose(1, 2))
            residual_next = prediction_next - y[:, None, :]
            gradient_next = torch.matmul(residual_next, x) / float(max(1, rows))
            _, drho_beta, _ = interpolate_direct96_penalty(
                beta_next.abs(), pieces, self._kalbi_v6_penalty_grid
            )
            active = beta_next != 0.0
            active_kkt = gradient_next + (
                lam[..., None] * drho_beta * beta_next.sign()
            )
            zero_kkt = (
                gradient_next.abs() - lam[..., None] * d0[..., None]
            ).clamp_min(0.0)
            kkt = torch.where(active, active_kkt.abs(), zero_kkt)
            stat = torch.stack(
                [
                    _masked_mean(prox_derivative[:, 0], valid),
                    torch.where(
                        valid, prox_derivative[:, 0],
                        torch.full_like(prox_derivative[:, 0], -torch.inf),
                    ).amax(-1).clamp_min(0.0),
                    torch.where(
                        valid, denominator[:, 0],
                        torch.full_like(denominator[:, 0], torch.inf),
                    ).amin(-1),
                    _masked_mean(denominator[:, 0], valid),
                    boundary_q05,
                    _masked_fraction(active[:, 0], valid),
                    _masked_fraction(zero_branch, valid),
                    _masked_fraction(saturated, valid),
                    _masked_rms(beta_next[:, 0], valid),
                    _masked_rms((beta_next - beta)[:, 0], valid),
                    _masked_rms(gradient[:, 0], valid),
                    _masked_rms(kkt[:, 0], valid),
                    eta[:, 0],
                    gamma[:, 0],
                ],
                dim=-1,
            )
            statistics.append(stat)
            eta_values.append(eta[:, 0])
            gamma_values.append(gamma[:, 0])
            beta = beta_next
        return torch.stack(statistics, dim=1), {
            "beta": beta,
            "eta": torch.stack(eta_values, dim=1),
            "gamma": torch.stack(gamma_values, dim=1),
        }

    def forward(
        self,
        x_context: Tensor,
        y_context: Tensor,
        lambda_max: Tensor,
        *,
        feature_mask: Tensor | None = None,
        solver_context: Mapping[str, Tensor] | None = None,
        return_representation: bool = False,
    ) -> dict[str, Any]:
        if solver_context is None:
            raise ValueError("V24 requires the beta solver's encoded context")
        if not bool(self.v24_scaler.cache_loaded.item()):
            raise RuntimeError("V24 source scaler is not loaded")
        trajectory = self.encode_context(
            x_context, y_context, feature_mask=feature_mask
        )
        base = super().forward_from_trajectory(
            trajectory, lambda_max, return_representation=True
        )
        v23_task = base["v22_task_code"]
        global_summary = self._global_summary(
            x_context, y_context, feature_mask
        )
        global_scaled = self.v24_scaler.scale("global", global_summary)
        global_code = self.global_projection(global_scaled)
        initial_fused = v23_task + self.global_gate * global_code
        u0 = initial_fused + self.initial_norm_gate * (
            self.initial_fusion_norm(initial_fused) - initial_fused
        )
        initial_lambda, initial_logit, initial_unit = self._raw_lambda(u0)
        initial_penalty_code = self.penalty_mlp(u0)
        initial_encoded = self.penalty_parameter_head(initial_penalty_code)
        solver_trajectory, probe = self._probe(
            solver_context, initial_lambda, initial_encoded
        )
        trajectory_scaled = self.v24_scaler.scale(
            "trajectory", solver_trajectory
        )
        trajectory_code = self.trajectory_projection(
            trajectory_scaled.reshape(x_context.shape[0], -1)
        )
        proposal_summary = self._proposal_summary(
            initial_lambda, initial_encoded, lambda_max
        )
        proposal_scaled = self.v24_scaler.scale("proposal", proposal_summary)
        proposal_code = self.proposal_projection(proposal_scaled)
        hidden = self.corrected_fusion_norm(
            u0
            + self.trajectory_gate * trajectory_code
            + self.proposal_gate * proposal_code
        )
        residual_updates: list[Tensor] = []
        for layer in self.residual_layers:
            hidden, update = layer(hidden)
            residual_updates.append(update)
        corrected_code = self.final_norm(hidden)
        delta_loglambda = self.delta_loglambda_head(corrected_code).squeeze(-1)
        corrected_loglambda = (
            initial_lambda.clamp_min(1e-12).log() + delta_loglambda
        ).clamp(math.log(V23_RAW_LAMBDA_MIN), math.log(V23_RAW_LAMBDA_MAX))
        raw_lambda = corrected_loglambda.exp()
        delta_latent = self.delta_penalty_latent_head(corrected_code)
        corrected_latent = initial_encoded["latent_raw"] + delta_latent
        encoded = self.penalty_parameter_head.decode_direct_parameters(
            corrected_latent
        )
        lambda_max_safe = lambda_max.float().clamp_min(1e-12)
        d0 = encoded["d0"]
        max_drho = encoded["drho"].amax(-1)
        concavity = (
            encoded["drho"][..., 1:] - encoded["drho"][..., :-1] > 1e-7
        ).float().mean(-1)
        base.update(
            {
                "lambdas": raw_lambda[:, None],
                "raw_lambda": raw_lambda,
                "lambda_ratio": raw_lambda / lambda_max_safe,
                "raw_lambda_over_lambda_max": raw_lambda / lambda_max_safe,
                "effective_lambda_ratio": raw_lambda * d0 / lambda_max_safe,
                "zero_kkt_ratio": raw_lambda * d0 / lambda_max_safe,
                "force_ratio": raw_lambda * max_drho / lambda_max_safe,
                "raw_lambda_logit": initial_logit,
                "raw_lambda_unit": initial_unit,
                "penalty_shape_features": encoded["shape_features"][:, None],
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
                "penalty_max_drho": max_drho,
                "penalty_derivative_tv": encoded["metadata4"][..., 3],
                "penalty_positivity_jacobian_min": encoded[
                    "positivity_jacobian"
                ].amin(-1),
                "penalty_reference_slope": max_drho,
                "penalty_ridge_branch": d0 <= 1e-7,
                "penalty_concavity_violation_fraction": concavity,
                "penalty_encoding_version": self.selector_version,
                "lambda_output_contract": V24_LAMBDA_OUTPUT_CONTRACT,
                "v24_initial_raw_lambda": initial_lambda,
                "v24_delta_loglambda": delta_loglambda,
                "v24_initial_d0": initial_encoded["d0"],
                "v24_initial_max_drho": initial_encoded["drho"].amax(-1),
                "v24_global_code_norm": global_code.norm(dim=-1),
                "v24_trajectory_code_norm": trajectory_code.norm(dim=-1),
                "v24_proposal_code_norm": proposal_code.norm(dim=-1),
                "v24_residual_update_rms": torch.stack(
                    [value.square().mean(-1).sqrt()
                     for value in residual_updates],
                    dim=-1,
                ),
                "v24_probe_eta": probe["eta"],
                "v24_probe_gamma": probe["gamma"],
                "v24_probe_gamma_contract_error": (
                    probe["gamma"]
                    / probe["eta"].clamp_min(1e-30)
                    - initial_lambda[:, None]
                ).abs(),
            }
        )
        if return_representation:
            base.update(
                {
                    "v24_task_code": v23_task,
                    "v24_global_summary": global_summary,
                    "v24_global_scaled": global_scaled,
                    "v24_global_code": global_code,
                    "v24_initial_task_code": u0,
                    "v24_solver_trajectory": solver_trajectory,
                    "v24_trajectory_scaled": trajectory_scaled,
                    "v24_trajectory_code": trajectory_code,
                    "v24_proposal_summary": proposal_summary,
                    "v24_proposal_scaled": proposal_scaled,
                    "v24_proposal_code": proposal_code,
                    "v24_corrected_task_code": corrected_code,
                    "v24_initial_penalty_latent": initial_encoded["latent_raw"],
                    "v24_corrected_penalty_latent": corrected_latent,
                }
            )
        else:
            for key in list(base):
                if key.startswith(
                    ("v17_", "v19_", "v20_", "v21_", "v22_")
                ):
                    base.pop(key)
        return base

    def architecture_report(self) -> dict[str, Any]:
        report = super().architecture_report()
        report.update(
            {
                "selector_version": self.selector_version,
                "architecture_generation": 24,
                "lineage": "v23_plus_additive512_solver_trajectory_correction",
                "probe_steps": V24_PROBE_STEPS,
                "trajectory_shape": [V24_PROBE_STEPS, V24_TRAJECTORY_STATS],
                "global_summary_dim": V24_GLOBAL_DIM,
                "proposal_summary_dim": V24_PROPOSAL_DIM,
                "residual_512_linear_layers": len(self.residual_layers),
                "fusion": "additive_512d",
                "raw_lambda_contract": V24_LAMBDA_OUTPUT_CONTRACT,
                "probe_gamma_contract": "eta_times_initial_raw_lambda",
                "v24_scaler_version": V24_SCALER_VERSION,
                "v24_scaler_transform_contract": (
                    V24_SCALER_TRANSFORM_CONTRACT
                ),
                "v24_scaler_cache_size": int(self.v24_scaler.cache_size.item()),
                "total_parameters": sum(p.numel() for p in self.parameters()),
            }
        )
        return report


@torch.no_grad()
def load_v23_state_into_v24(
    selector: DirectPenaltyMLPQuerySelectorV24Additive512,
    state_dict: Mapping[str, Tensor],
) -> dict[str, Any]:
    """Strictly import every V23 tensor and retain only explicit V24 additions."""
    current = selector.state_dict()
    imported: list[str] = []
    unexpected: list[str] = []
    mismatched: list[str] = []
    for key, value in state_dict.items():
        if key not in current:
            unexpected.append(key)
        elif current[key].shape != value.shape:
            mismatched.append(key)
        else:
            current[key] = value.detach().clone()
            imported.append(key)
    if unexpected or mismatched:
        raise RuntimeError(
            f"V23->V24 import failed: unexpected={unexpected}, "
            f"shape_mismatched={mismatched}"
        )
    selector.load_state_dict(current, strict=True)
    new_keys = sorted(set(current) - set(imported))
    allowed_prefixes = (
        "v24_scaler.",
        "global_projection.",
        "trajectory_projection.",
        "proposal_projection.",
        "initial_fusion_norm.",
        "initial_norm_gate",
        "corrected_fusion_norm.",
        "global_gate",
        "trajectory_gate",
        "proposal_gate",
        "residual_layers.",
        "final_norm.",
        "delta_loglambda_head.",
        "delta_penalty_latent_head.",
        "_kalbi_v6_penalty_grid",
    )
    forbidden = [
        key for key in new_keys if not key.startswith(allowed_prefixes)
    ]
    if forbidden:
        raise RuntimeError(f"unclassified V24 state keys: {forbidden}")
    return {
        "imported_count": len(imported),
        "new_count": len(new_keys),
        "imported_keys": sorted(imported),
        "new_keys": new_keys,
        "unexpected": unexpected,
        "shape_mismatched": mismatched,
    }


def make_identity_v24_scaler_payload(cache_size: int = 1) -> dict[str, Any]:
    """Bootstrap payload; production should replace it with collected moments."""
    return {
        "version": V24_SCALER_VERSION,
        "transform_contract": V24_SCALER_TRANSFORM_CONTRACT,
        "cache_size": int(cache_size),
        "cache_sha256": "0" * 64,
        "global_center": torch.zeros(V24_GLOBAL_DIM),
        "global_scale": torch.ones(V24_GLOBAL_DIM),
        "trajectory_center": torch.zeros(
            V24_PROBE_STEPS, V24_TRAJECTORY_STATS
        ),
        "trajectory_scale": torch.ones(
            V24_PROBE_STEPS, V24_TRAJECTORY_STATS
        ),
        "proposal_center": torch.zeros(V24_PROPOSAL_DIM),
        "proposal_scale": torch.ones(V24_PROPOSAL_DIM),
    }
