#!/usr/bin/env python3
"""V10 lambda/penalty selector for V58-Chebyshev native196.

The penalty head predicts whitened coordinates of the *actual* fixed-knot
one-hidden-layer ReLU-squared penalty network used by the random-penalty
generator.  Those coordinates are decoded to its identifiable parameters::

    rho(t)  = b |t| + 0.5 sum_k w_k ReLU(|t| - knot_k)^2
    drho(t) = b     +     sum_k w_k ReLU( t  - knot_k)

The knot at t=6 has no effect on [0, 6], so the head predicts the 95 effective
weights and appends the exact dead value w_95=0 only for interoperability.
The shared penalty scale is fixed by rho(6)=1.  All 196 V56 coordinates are
then rebuilt differentiably as 192 interleaved rho/drho values and four
continuous metadata values [d0, rho(1), drho(6), TV(drho)].

Independent coordinate scaling of the 95 MLP weights is ill conditioned
because valid random penalties rely on strong cancellations.  V10 therefore
fits a PCA-whitened parameter basis from the random-penalty cache and predicts
its leading coordinates.  A smooth always-feasible radial map keeps drho above
a small floor without the exact zero radial gradient of a hard boundary clamp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import ResidualMLP
from penalty_relu_mlp_pwl_v33 import (
    _v33_grid_t,
    batch_segments_to_relu_params,
)

Tensor = torch.Tensor

PENALTY_GRID_SIZE = 96
EFFECTIVE_WEIGHT_DIM = PENALTY_GRID_SIZE - 1
V56_META_DIM = 4
V56_NATIVE_DIM = 2 * PENALTY_GRID_SIZE + V56_META_DIM
DEFAULT_PARAMETER_RANK = 48
V10_PENALTY_ENCODING_VERSION = (
    "v10_v58cheb_native196_fixedknot_relu2_mlp_pca48_meta4_"
    "smoothvalid_splittrunk_v1"
)


def _upper_tri_flat(value: Tensor, max_dim: int | None = None) -> Tensor:
    if value.dim() == 4:
        value = value[:, 0]
    width = value.shape[-1]
    indices = torch.triu_indices(width, width, device=value.device)
    result = value[..., indices[0], indices[1]]
    if max_dim is None:
        return result
    if result.shape[-1] >= max_dim:
        return result[..., :max_dim]
    return F.pad(result, (0, max_dim - result.shape[-1]))


def summarize_encoded(
    encoded: dict[str, Any], t_feat_dim: int
) -> tuple[Tensor, Tensor]:
    """Local copy of the V10 dataset summary; no versioned model dependency."""
    Xty = encoded["Xty"].float()
    gram_diag = encoded["gram_diag"].float().clamp_min(1e-12)
    lambda_max = encoded["lambda_max"].float().clamp_min(1e-12)
    batch = Xty.shape[0]
    sample_count = torch.full(
        (batch, 1), float(encoded["N"]), device=Xty.device
    )
    feature_count = torch.full(
        (batch, 1), float(encoded["P"]), device=Xty.device
    )
    absolute = Xty.abs()
    dataset_features = torch.cat(
        [
            torch.log(sample_count),
            torch.log(feature_count),
            torch.log(lambda_max).unsqueeze(-1),
            torch.log1p(absolute.mean(dim=-1, keepdim=True)),
            torch.log1p(absolute.amax(dim=-1, keepdim=True)),
            torch.log1p(
                absolute.square().mean(dim=-1, keepdim=True).sqrt()
            ),
            torch.log1p(gram_diag.mean(dim=-1, keepdim=True)),
            torch.log1p(gram_diag.amax(dim=-1, keepdim=True)),
            torch.log1p(
                gram_diag.square().mean(dim=-1, keepdim=True).sqrt()
            ),
        ],
        dim=-1,
    )
    dataset_features = torch.nan_to_num(
        dataset_features, nan=0.0, posinf=20.0, neginf=-20.0
    ).clamp(-20.0, 20.0)
    krylov_features = _upper_tri_flat(
        encoded["T_krylov"].float(), t_feat_dim
    )
    krylov_features = torch.nan_to_num(
        krylov_features, nan=0.0, posinf=20.0, neginf=-20.0
    ).clamp(-20.0, 20.0)
    return dataset_features, krylov_features


def selector_beta_with_inactive_ste(
    decode_aux: dict[str, Tensor],
    *,
    leak: float = 0.05,
    temperature_fraction: float = 0.10,
) -> Tensor:
    """Keep the V56 hard beta forward value while repairing inactive gradients."""
    leak = float(leak)
    temperature_fraction = float(temperature_fraction)
    if not (0.0 < leak <= 1.0):
        raise ValueError(f"selector STE leak must be in (0, 1], got {leak}")
    if temperature_fraction <= 0.0:
        raise ValueError("selector STE temperature_fraction must be positive")
    required = ("beta_q", "u_hat_q", "tau_hat_q")
    missing = [key for key in required if key not in decode_aux]
    if missing:
        raise KeyError(f"selector STE requires decoder aux fields {missing}")

    hard_beta = decode_aux["beta_q"].float()
    u_hat = decode_aux["u_hat_q"].float()
    tau_hat = decode_aux["tau_hat_q"].float().clamp_min(0.0)
    if hard_beta.shape != u_hat.shape or hard_beta.shape != tau_hat.shape:
        raise ValueError(
            f"selector STE shape mismatch: beta={tuple(hard_beta.shape)} "
            f"u={tuple(u_hat.shape)} tau={tuple(tau_hat.shape)}"
        )
    margin = u_hat.abs() - tau_hat
    inactive = margin.le(0.0).to(u_hat.dtype).detach()
    scale = (u_hat.detach().abs() + tau_hat.detach()).clamp_min(1e-3)
    temperature = (temperature_fraction * scale).clamp_min(1e-4)
    smooth_magnitude = (
        leak * margin
        + (1.0 - leak) * temperature * F.softplus(margin / temperature)
    )
    surrogate = u_hat.sign().detach() * smooth_magnitude
    return hard_beta + inactive * (surrogate - surrogate.detach())


def _basis_matrices(knots: Tensor, *, t_max: float = 6.0) -> tuple[Tensor, Tensor]:
    effective_knots = knots[:EFFECTIVE_WEIGHT_DIM]
    grid_relu = (knots[:, None] - effective_knots[None, :]).clamp_min(0.0)
    mass_coeff = 0.5 * (float(t_max) - effective_knots).clamp_min(0.0).square()
    return grid_relu, mass_coeff


def normalize_relu2_parameters(
    relu_b: Tensor,
    relu_weights: Tensor,
    knots: Tensor,
    *,
    minimum_mass: float = 1e-8,
) -> dict[str, Tensor]:
    """Remove the lambda-confounded global scale by enforcing rho(6)=1."""
    b = relu_b.float()
    weights = relu_weights.float()
    if b.ndim != 1:
        raise ValueError(f"relu_b must be a vector, got {tuple(b.shape)}")
    if weights.shape != (b.numel(), PENALTY_GRID_SIZE):
        raise ValueError(
            f"relu_weights must be ({b.numel()},{PENALTY_GRID_SIZE}), "
            f"got {tuple(weights.shape)}"
        )
    if not bool(torch.isfinite(b).all().item()) or not bool(
        torch.isfinite(weights).all().item()
    ):
        raise ValueError("nonfinite ReLU-squared penalty parameters")
    _, mass_coeff = _basis_matrices(knots.to(weights), t_max=6.0)
    mass = 6.0 * b + (weights[:, :EFFECTIVE_WEIGHT_DIM] * mass_coeff).sum(-1)
    if bool((mass <= float(minimum_mass)).any().item()):
        raise ValueError("penalty cache contains nonpositive rho(6) mass")
    return {
        "b": b / mass,
        "weights": weights[:, :EFFECTIVE_WEIGHT_DIM] / mass[:, None],
        "mass": mass,
        "dead_last_weight_max_abs": weights[:, -1].abs().amax(),
    }


def native196_from_relu2_parameters(
    relu_b: Tensor,
    effective_weights: Tensor,
    knots: Tensor,
) -> dict[str, Tensor]:
    """Build the V56 native196 tensor with no discrete metadata coordinate."""
    b = relu_b.float()
    weights = effective_weights.float()
    if b.shape != weights.shape[:-1] or weights.shape[-1] != EFFECTIVE_WEIGHT_DIM:
        raise ValueError(
            f"parameter mismatch b={tuple(b.shape)} weights={tuple(weights.shape)}"
        )
    grid = knots.to(weights)
    grid_relu, _ = _basis_matrices(grid)
    drho = b[..., None] + torch.einsum("...k,tk->...t", weights, grid_relu)
    rho = b[..., None] * grid + 0.5 * torch.einsum(
        "...k,tk->...t", weights, grid_relu.square()
    )
    one_relu = (torch.ones((), device=grid.device, dtype=grid.dtype) - grid[
        :EFFECTIVE_WEIGHT_DIM
    ]).clamp_min(0.0)
    rho_at_one = b + 0.5 * (weights * one_relu.square()).sum(-1)
    derivative_tv = (drho[..., 1:] - drho[..., :-1]).abs().sum(-1)
    linear = torch.stack((rho, drho), dim=-1).reshape(
        *rho.shape[:-1], 2 * PENALTY_GRID_SIZE
    )
    metadata = torch.stack(
        [drho[..., 0], rho_at_one, drho[..., -1], derivative_tv], dim=-1
    )
    native = torch.cat([linear, metadata], dim=-1)
    if native.shape[-1] != V56_NATIVE_DIM:
        raise RuntimeError(f"V10 native contract failed: {tuple(native.shape)}")
    full_weights = torch.cat([weights, torch.zeros_like(weights[..., :1])], dim=-1)
    return {
        "shape_features": native,
        "linear_features": linear,
        "rho": rho,
        "drho": drho,
        "metadata4": metadata,
        "relu_b": b,
        "relu_weights": full_weights,
    }


def lambda_from_effective_ratio_v56(
    lambda_ratio: Tensor,
    lambda_max: Tensor,
    d0: Tensor,
    *,
    reference_slope: Tensor | None = None,
    minimum_d0: float = 1e-7,
) -> dict[str, Tensor]:
    """Keep lambda*reference_slope/lambda_max at the requested ratio.

    Older selectors omit ``reference_slope`` and retain the nonsmooth
    ``rho'(0)`` contract. General-penalty selectors may provide another
    positive, scale-equivariant derivative summary while preserving ``d0`` as
    the actual origin derivative consumed by the beta solver.
    """
    origin_slope = d0.float()
    slope = (
        origin_slope
        if reference_slope is None
        else reference_slope.float()
    )
    if slope.shape != origin_slope.shape:
        raise ValueError(
            "lambda reference slope shape must match d0: "
            f"{tuple(slope.shape)} != {tuple(origin_slope.shape)}"
        )
    if not bool(torch.isfinite(slope).all().item()) or bool(
        (slope <= float(minimum_d0)).any().item()
    ):
        if reference_slope is None:
            raise ValueError(
                "V10 supports only nonsmooth cache penalties with positive d0"
            )
        raise ValueError("generalized lambda reference slope must be finite and positive")
    lmax = lambda_max.float().clamp_min(1e-12)
    lambdas = lambda_ratio.float() * lmax / slope
    return {
        "lambdas": lambdas,
        "d0": origin_slope,
        "reference_slope": slope,
        "ridge_branch": origin_slope <= float(minimum_d0),
        "effective_ratio": lambdas * slope / lmax,
    }


class ReLU2PenaltyParameterHeadV10(nn.Module):
    """Predict PCA-whitened coordinates and decode actual penalty MLP parameters."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        latent_rank: int = DEFAULT_PARAMETER_RANK,
        derivative_floor: float = 1e-4,
        boundary_safety: float = 0.99,
        projection_power: float = 8.0,
        latent_span: float = 1.0,
        head_init_std: float = 1e-3,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.latent_rank = int(latent_rank)
        self.derivative_floor = float(derivative_floor)
        self.boundary_safety = float(boundary_safety)
        self.projection_power = float(projection_power)
        self.latent_span = float(latent_span)
        if not (1 <= self.latent_rank <= EFFECTIVE_WEIGHT_DIM):
            raise ValueError("latent_rank must be between 1 and 95")
        if self.derivative_floor <= 0.0:
            raise ValueError("derivative_floor must be positive")
        if not (0.0 < self.boundary_safety < 1.0):
            raise ValueError("boundary_safety must be in (0,1)")
        if self.projection_power < 2.0:
            raise ValueError("projection_power must be at least 2")
        if self.latent_span <= 0.0:
            raise ValueError("latent_span must be positive")

        self.projection = nn.Linear(self.hidden_dim, self.latent_rank)
        nn.init.normal_(self.projection.weight, mean=0.0, std=float(head_init_std))
        nn.init.zeros_(self.projection.bias)
        knots = _v33_grid_t(
            PENALTY_GRID_SIZE,
            6.0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.register_buffer("penalty_grid", knots)
        self.register_buffer(
            "reference_weights", torch.zeros(EFFECTIVE_WEIGHT_DIM)
        )
        self.register_buffer("reference_b", torch.tensor(1.0 / 6.0))
        self.register_buffer(
            "parameter_basis", torch.zeros(EFFECTIVE_WEIGHT_DIM, self.latent_rank)
        )
        self.register_buffer(
            "parameter_eigenvalues", torch.zeros(self.latent_rank)
        )
        self.register_buffer("parameter_total_variance", torch.tensor(0.0))
        self.register_buffer("explained_variance", torch.tensor(0.0))
        self.register_buffer("cache_mass_min", torch.tensor(float("nan")))
        self.register_buffer("cache_mass_max", torch.tensor(float("nan")))
        self.register_buffer("cache_d0_min", torch.tensor(float("nan")))
        self.register_buffer("cache_d0_max", torch.tensor(float("nan")))
        self.register_buffer("cache_size", torch.tensor(0, dtype=torch.long))
        self.register_buffer("calibrated", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def fit_parameter_cache(
        self,
        relu_b: Tensor,
        relu_weights: Tensor,
    ) -> dict[str, Any]:
        """Fit the identifiable scale-normalized MLP-parameter manifold."""
        normalized = normalize_relu2_parameters(
            relu_b.detach().cpu(),
            relu_weights.detach().cpu(),
            self.penalty_grid.cpu(),
        )
        b = normalized["b"].double()
        weights = normalized["weights"].double()
        if weights.shape[0] <= self.latent_rank:
            raise ValueError(
                f"calibration needs more than {self.latent_rank} penalties, "
                f"got {weights.shape[0]}"
            )
        reference_weights = weights.mean(dim=0)
        _, mass_coeff = _basis_matrices(self.penalty_grid.double())
        reference_b = (1.0 - (reference_weights * mass_coeff).sum()) / 6.0
        reference = native196_from_relu2_parameters(
            reference_b[None].float(),
            reference_weights[None].float(),
            self.penalty_grid,
        )
        reference_drho = reference["drho"][0]
        if float(reference_drho.amin()) <= self.derivative_floor:
            raise ValueError(
                "cache mean is too close to the derivative boundary: "
                f"min={float(reference_drho.amin()):.6g}"
            )

        centered = weights - reference_weights
        covariance = centered.T @ centered / float(weights.shape[0] - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order].clamp_min(0.0)
        eigenvectors = eigenvectors[:, order]
        chosen_values = eigenvalues[: self.latent_rank]
        chosen_vectors = eigenvectors[:, : self.latent_rank]
        # Resolve the arbitrary eigenvector signs so calibration is reproducible.
        pivots = chosen_vectors.abs().argmax(dim=0)
        signs = chosen_vectors[pivots, torch.arange(self.latent_rank)].sign()
        signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)
        chosen_vectors = chosen_vectors * signs
        basis = chosen_vectors * chosen_values.sqrt()[None, :]
        explained = chosen_values.sum() / eigenvalues.sum().clamp_min(1e-30)

        self.reference_weights.copy_(reference_weights.float())
        self.reference_b.copy_(reference_b.float())
        self.parameter_basis.copy_(basis.float())
        self.parameter_eigenvalues.copy_(chosen_values.float())
        self.parameter_total_variance.copy_(eigenvalues.sum().float())
        self.explained_variance.copy_(explained.float())
        self.cache_mass_min.copy_(normalized["mass"].amin().float())
        self.cache_mass_max.copy_(normalized["mass"].amax().float())
        self.cache_d0_min.copy_(b.amin().float())
        self.cache_d0_max.copy_(b.amax().float())
        self.cache_size.copy_(torch.tensor(weights.shape[0], dtype=torch.long))
        self.calibrated.fill_(True)
        rank_checkpoints = sorted(
            {rank for rank in (1, 2, 3, 4, 8, 16, 32, self.latent_rank)
             if rank <= self.latent_rank}
        )
        return {
            "cache_size": int(weights.shape[0]),
            "latent_rank": self.latent_rank,
            "explained_variance": float(explained),
            "mass_min": float(normalized["mass"].amin()),
            "mass_max": float(normalized["mass"].amax()),
            "normalized_d0_min": float(b.amin()),
            "normalized_d0_max": float(b.amax()),
            "dead_last_weight_max_abs": float(
                normalized["dead_last_weight_max_abs"]
            ),
            "explained_variance_by_rank": {
                str(rank): float(
                    eigenvalues[:rank].sum()
                    / eigenvalues.sum().clamp_min(1e-30)
                )
                for rank in rank_checkpoints
            },
            "selected_eigenvalue_condition": float(
                chosen_values[0]
                / chosen_values[-1].clamp_min(1e-30)
            ),
        }

    @torch.no_grad()
    def fit_segment_cache(self, segments: list[Any]) -> dict[str, Any]:
        """Convenience calibration from the generator's segment objects."""
        _, b, weights = batch_segments_to_relu_params(
            segments,
            K=PENALTY_GRID_SIZE,
            t_max=6.0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        return self.fit_parameter_cache(b, weights)

    def decode_latent(self, latent_unit: Tensor) -> dict[str, Tensor]:
        if not bool(self.calibrated.item()):
            raise RuntimeError("V10 penalty head must be calibrated from the cache")
        z = latent_unit.float()
        if z.shape[-1] != self.latent_rank:
            raise ValueError(
                f"latent dimension {z.shape[-1]} != {self.latent_rank}"
            )
        delta_weights = self.latent_span * torch.einsum(
            "...r,kr->...k", z, self.parameter_basis.to(z)
        )
        _, mass_coeff = _basis_matrices(self.penalty_grid.to(z))
        delta_b = -(delta_weights * mass_coeff).sum(-1) / 6.0
        grid_relu, _ = _basis_matrices(self.penalty_grid.to(z))
        delta_drho = delta_b[..., None] + torch.einsum(
            "...k,tk->...t", delta_weights, grid_relu
        )
        reference = native196_from_relu2_parameters(
            self.reference_b.to(z).expand(z.shape[:-1]),
            self.reference_weights.to(z).expand(
                *z.shape[:-1], EFFECTIVE_WEIGHT_DIM
            ),
            self.penalty_grid.to(z),
        )
        reference_drho = reference["drho"]
        available = (reference_drho - self.derivative_floor).clamp_min(0.0)
        danger_by_grid = torch.where(
            delta_drho < 0.0,
            (-delta_drho) / available.clamp_min(1e-20),
            torch.zeros_like(delta_drho),
        )
        danger = danger_by_grid.amax(dim=-1)
        # Smooth, always-feasible radial projection.  If r is the worst
        # fraction of available derivative margin consumed by the proposed
        # delta, alpha*r is strictly below boundary_safety.  Unlike a hard
        # min-clamp this retains a radial gradient near the boundary.
        scaled_danger = danger / self.boundary_safety
        alpha = (1.0 + scaled_danger.pow(self.projection_power)).pow(
            -1.0 / self.projection_power
        )
        effective_delta = alpha[..., None] * delta_weights
        weights = self.reference_weights.to(z) + effective_delta
        b = (1.0 - (weights * mass_coeff).sum(-1)) / 6.0
        encoded = native196_from_relu2_parameters(
            b, weights, self.penalty_grid.to(z)
        )
        encoded.update(
            {
                "latent_unit": z,
                "delta_weights_unscaled": delta_weights,
                "validity_alpha": alpha,
                "validity_danger": danger,
                "validity_projection_active": danger > self.boundary_safety,
                "rho6_error": (encoded["rho"][..., -1] - 1.0).abs(),
            }
        )
        return encoded

    def forward(self, hidden: Tensor) -> dict[str, Tensor]:
        raw = self.projection(hidden.float()).float()
        # DDP wraps each returned tensor in a distinct sink.  Retaining the
        # public latent_raw sibling after DDP forward therefore does not
        # observe the gradient carried by the returned shape tensor.  Keep
        # the actual internal ancestor for exact gradient telemetry.
        self._last_latent_raw = raw
        if raw.requires_grad:
            raw.retain_grad()
        latent = torch.tanh(raw)
        encoded = self.decode_latent(latent)
        encoded["latent_raw"] = raw
        return encoded


class DirectPenaltyMLPQuerySelectorV10(nn.Module):
    """Dataset-conditioned V10 selector that emits lambda and penalty MLP params."""

    def __init__(
        self,
        *,
        ds_feat_dim: int,
        t_feat_dim: int,
        krylov_dim: int,
        shape_dim: int = V56_NATIVE_DIM,
        path_feat_dim: int = 150,
        hidden: int = 256,
        depth: int = 3,
        lambda_min_ratio: float = 1e-6,
        lambda_max_ratio: float = 1.0,
        initial_lambda_ratio: float = 0.1,
        parameter_rank: int = DEFAULT_PARAMETER_RANK,
        derivative_floor: float = 1e-4,
        boundary_safety: float = 0.99,
        projection_power: float = 8.0,
        latent_span: float = 1.0,
        penalty_head_init_std: float = 1e-3,
    ):
        super().__init__()
        self.shape_dim = int(shape_dim)
        self.krylov_dim = int(krylov_dim)
        self.lambda_min_ratio = float(lambda_min_ratio)
        self.lambda_max_ratio = float(lambda_max_ratio)
        if self.shape_dim != V56_NATIVE_DIM:
            raise ValueError(f"V10/V56 requires shape_dim={V56_NATIVE_DIM}")
        if not (
            self.lambda_min_ratio
            <= float(initial_lambda_ratio)
            <= self.lambda_max_ratio
        ):
            raise ValueError("initial lambda ratio is outside selector bounds")

        self.ds_proj = nn.Linear(int(ds_feat_dim), int(hidden))
        self.t_proj = nn.Sequential(
            nn.LayerNorm(int(t_feat_dim)), nn.Linear(int(t_feat_dim), int(hidden))
        )
        self.path_proj = nn.Sequential(
            nn.LayerNorm(int(path_feat_dim)), nn.Linear(int(path_feat_dim), int(hidden))
        )
        self.u_proj = nn.Linear(int(krylov_dim), int(hidden))
        self.y_proj = nn.Linear(1, int(hidden))
        self.y_token_mlp = ResidualMLP(
            int(hidden), int(hidden), int(hidden), depth=2,
            zero_init_last=False, act=nn.GELU,
        )
        self.y_pool_proj = nn.Linear(int(hidden) * 3, int(hidden))
        # V9's zero-initialized final trunk layer made every task identical at
        # initialization and gave the penalty-head weights zero first-step grad.
        self.lambda_net = ResidualMLP(
            int(hidden), int(hidden), int(hidden), depth=int(depth),
            zero_init_last=False, act=nn.GELU,
        )
        self.penalty_net = ResidualMLP(
            int(hidden), int(hidden), int(hidden), depth=int(depth),
            zero_init_last=False, act=nn.GELU,
        )
        self.lambda_head = nn.Linear(int(hidden), 1)
        nn.init.zeros_(self.lambda_head.weight)
        log_lo = math.log(self.lambda_min_ratio)
        log_hi = math.log(self.lambda_max_ratio)
        initial_unit = (
            math.log(float(initial_lambda_ratio)) - log_lo
        ) / (log_hi - log_lo)
        initial_unit = min(max(initial_unit, 1e-6), 1.0 - 1e-6)
        nn.init.constant_(
            self.lambda_head.bias, math.log(initial_unit / (1.0 - initial_unit))
        )
        self.penalty_parameter_head = ReLU2PenaltyParameterHeadV10(
            int(hidden),
            latent_rank=int(parameter_rank),
            derivative_floor=float(derivative_floor),
            boundary_safety=float(boundary_safety),
            projection_power=float(projection_power),
            latent_span=float(latent_span),
            head_init_std=float(penalty_head_init_std),
        )

    def fit_penalty_parameter_cache(
        self, relu_b: Tensor, relu_weights: Tensor
    ) -> dict[str, Any]:
        return self.penalty_parameter_head.fit_parameter_cache(relu_b, relu_weights)

    def fit_penalty_segment_cache(
        self, segments: list[Any]
    ) -> dict[str, Any]:
        return self.penalty_parameter_head.fit_segment_cache(segments)

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
        h = (
            self.ds_proj(ds_feat.float())
            + self.t_proj(t_feat.float())
            + self.path_proj(path_feat.float())
        )
        if U is not None and y_context is not None:
            if U.dim() == 4:
                U = U[:, 0]
            if U.shape[-1] != self.krylov_dim and U.shape[1] == self.krylov_dim:
                U = U.transpose(1, 2)
            tok = self.y_token_mlp(
                self.u_proj(U.float()) + self.y_proj(y_context.float().unsqueeze(-1))
            )
            pooled = torch.cat(
                [
                    tok.mean(dim=1),
                    tok.std(dim=1, unbiased=False),
                    tok.amax(dim=1),
                ],
                dim=-1,
            )
            h = h + self.y_pool_proj(pooled)
        fused = h.float()
        lambda_hidden = self.lambda_net(fused)
        penalty_hidden = self.penalty_net(fused)

        lam_unit = torch.sigmoid(self.lambda_head(lambda_hidden).squeeze(-1))
        log_lo = math.log(self.lambda_min_ratio)
        log_hi = math.log(self.lambda_max_ratio)
        lambda_ratio = torch.exp(log_lo + (log_hi - log_lo) * lam_unit).clamp(
            self.lambda_min_ratio, self.lambda_max_ratio
        )
        encoded = self.penalty_parameter_head(penalty_hidden)
        shape = encoded["shape_features"]
        d0 = encoded["drho"][..., 0]
        lambda_scale = lambda_from_effective_ratio_v56(
            lambda_ratio, lambda_max.float(), d0
        )
        reference_native = native196_from_relu2_parameters(
            self.penalty_parameter_head.reference_b.to(shape)[None],
            self.penalty_parameter_head.reference_weights.to(shape)[None],
            self.penalty_parameter_head.penalty_grid.to(shape),
        )["shape_features"][0]
        reference_distance = (
            (shape - reference_native).square().mean(-1).sqrt()
            / reference_native.square().mean().sqrt().clamp_min(1e-8)
        )
        return {
            "lambdas": lambda_scale["lambdas"][:, None],
            "lambda_ratio": lambda_ratio,
            "effective_lambda_ratio": lambda_scale["effective_ratio"],
            "penalty_shape_features": shape[:, None, :].float(),
            "penalty_d0": d0[:, None].float(),
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
            "penalty_reference_distance": reference_distance,
            "penalty_reference_slope": lambda_scale["reference_slope"],
            "penalty_ridge_branch": lambda_scale["ridge_branch"],
            "penalty_parameter_explained_variance": (
                self.penalty_parameter_head.explained_variance.to(shape)
            ),
            "penalty_encoding_version": V10_PENALTY_ENCODING_VERSION,
        }


@dataclass
class SelectorOutputV10:
    beta: Tensor
    lambdas: Tensor
    lambda_ratio: Tensor
    penalty_shape_features: Tensor
    penalty_d0: Tensor
    penalty_relu_b: Tensor
    penalty_relu_weights: Tensor
    penalty_rho: Tensor
    penalty_drho: Tensor
    penalty_metadata4: Tensor
    penalty_latent_raw: Tensor
    penalty_validity_alpha: Tensor
    penalty_validity_danger: Tensor


class FrozenV56PathFeatureSelectorV10(nn.Module):
    """Freeze V56 parameters but retain the selector-to-beta computation graph."""

    def __init__(
        self,
        frozen_fm: nn.Module,
        *,
        t_feat_dim: int,
        krylov_dim: int,
        path_feat_dim: int = 150,
        selector_hidden: int = 256,
        lambda_min_ratio: float = 1e-6,
        lambda_max_ratio: float = 1.0,
        parameter_rank: int = DEFAULT_PARAMETER_RANK,
        selector_ste_leak: float = 0.05,
        selector_ste_temperature_fraction: float = 0.10,
    ):
        super().__init__()
        self.frozen_fm = frozen_fm.eval()
        for parameter in self.frozen_fm.parameters():
            parameter.requires_grad_(False)
        self.selector_ste_leak = float(selector_ste_leak)
        self.selector_ste_temperature_fraction = float(
            selector_ste_temperature_fraction
        )
        self.selector = DirectPenaltyMLPQuerySelectorV10(
            ds_feat_dim=9,
            t_feat_dim=int(t_feat_dim),
            krylov_dim=int(krylov_dim),
            shape_dim=V56_NATIVE_DIM,
            path_feat_dim=int(path_feat_dim),
            hidden=int(selector_hidden),
            lambda_min_ratio=float(lambda_min_ratio),
            lambda_max_ratio=float(lambda_max_ratio),
            parameter_rank=int(parameter_rank),
        )

    def fit_penalty_parameter_cache(
        self, relu_b: Tensor, relu_weights: Tensor
    ) -> dict[str, Any]:
        return self.selector.fit_penalty_parameter_cache(relu_b, relu_weights)

    def forward(
        self, X_context: Tensor, y_context: Tensor, path_feat: Tensor
    ) -> SelectorOutputV10:
        with torch.no_grad():
            pre = self.frozen_fm.precompute(X_context, y_context)
            encoded_ds = self.frozen_fm.encode_dataset(pre)
            ds_feat, t_feat = summarize_encoded(
                encoded_ds, self.selector.t_proj[0].normalized_shape[0]
            )
        query = self.selector(
            ds_feat,
            t_feat,
            encoded_ds["lambda_max"].float(),
            path_feat,
            U=encoded_ds.get("U"),
            y_context=y_context,
        )
        decode_aux = self.frozen_fm.decode_queries(
            encoded_ds,
            lambdas=query["lambdas"],
            lambda_max=encoded_ds["lambda_max"].float()[:, None],
            penalty_shape_features=query["penalty_shape_features"],
            penalty_d0=query["penalty_d0"],
            return_aux=True,
        )
        beta_grid = selector_beta_with_inactive_ste(
            decode_aux,
            leak=self.selector_ste_leak,
            temperature_fraction=self.selector_ste_temperature_fraction,
        )
        return SelectorOutputV10(
            beta=beta_grid[:, 0],
            lambdas=query["lambdas"][:, 0],
            lambda_ratio=query["lambda_ratio"],
            penalty_shape_features=query["penalty_shape_features"][:, 0],
            penalty_d0=query["penalty_d0"][:, 0],
            penalty_relu_b=query["penalty_relu_b"],
            penalty_relu_weights=query["penalty_relu_weights"],
            penalty_rho=query["penalty_rho"],
            penalty_drho=query["penalty_drho"],
            penalty_metadata4=query["penalty_metadata4"],
            penalty_latent_raw=query["penalty_latent_raw"],
            penalty_validity_alpha=query["penalty_validity_alpha"],
            penalty_validity_danger=query["penalty_validity_danger"],
        )


DirectPenaltyQuerySelector = DirectPenaltyMLPQuerySelectorV10
