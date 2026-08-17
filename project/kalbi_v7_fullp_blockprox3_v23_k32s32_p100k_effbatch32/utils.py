"""Kalbi V2 neural modules and numerical trajectory builders.

Kalbi V2 has no local Krylov or Chebyshev approximation.  It constructs two
lambda-independent coefficient trajectories from centered ``X`` and ``y``:

* ``A[:, :, s]`` is the K-feature ridge increment fitted to the current
  residual at stage ``s``.
* ``B[:, :, s]`` is a full ridge refit of the original centered target on all
  features selected through stage ``s``.

Both ridge paths use the same fixed regularization parameter.  The discrete
support is selected by absolute centered Pearson correlation and is shared by
all lambda/penalty queries.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


Tensor = torch.Tensor

GRID_SIZE = 96
META_DIM = 4
NATIVE_DIM = 2 * GRID_SIZE + META_DIM
EXPANDED_DIM = 4 * GRID_SIZE + META_DIM
PROX_COND_DIM = 6 + EXPANDED_DIM + GRID_SIZE
FEATURE_FILM_DIM = 6 + EXPANDED_DIM
LEGACY_NATIVE_DIM = NATIVE_DIM + 1
LEGACY_CONVEXITY_FLAG_INDEX = LEGACY_NATIVE_DIM - 1
LOCAL_ISIS_GRADIENT_CONTRACT = (
    "lambda_independent_centered_correlation_ab_trajectory_no_local_basis_v2"
)

_FIXED_RIDGE_NUMERICS = {
    "solve_calls": 0,
    "systems": 0,
    "float32_failed_systems": 0,
    "float64_fallback_calls": 0,
    "float64_fallback_systems": 0,
    "float64_jitter_calls": 0,
    "max_float64_relative_jitter": 0.0,
}


def fixed_ridge_numerical_diagnostics() -> dict[str, int | float]:
    """Return process-local cumulative diagnostics for fixed-ridge solves."""
    return dict(_FIXED_RIDGE_NUMERICS)


def strip_convexity_flag(shape: Tensor) -> Tensor:
    """Convert legacy native197/meta5 to native196/meta4 without detaching."""
    if shape.shape[-1] == NATIVE_DIM:
        return shape
    if shape.shape[-1] != LEGACY_NATIVE_DIM:
        raise ValueError(
            f"expected native {LEGACY_NATIVE_DIM} or {NATIVE_DIM}, "
            f"got {shape.shape[-1]}"
        )
    return shape[..., :LEGACY_CONVEXITY_FLAG_INDEX]


def native196_to_expanded388(shape: Tensor, lambdas: Tensor) -> Tensor:
    """Differentiably deinterleave and lambda-expand the penalty input."""
    shape = strip_convexity_flag(shape)
    if lambdas.shape != shape.shape[:-1]:
        raise ValueError(
            f"lambda/shape prefix mismatch: {tuple(lambdas.shape)} vs "
            f"{tuple(shape.shape[:-1])}"
        )
    pairs = shape[..., : 2 * GRID_SIZE].reshape(
        *shape.shape[:-1], GRID_SIZE, 2
    )
    rho = pairs[..., 0]
    drho = pairs[..., 1]
    meta4 = shape[..., 2 * GRID_SIZE :]
    lam = lambdas.unsqueeze(-1)
    expanded = torch.cat([rho, drho, lam * rho, lam * drho, meta4], dim=-1)
    if expanded.shape[-1] != EXPANDED_DIM:
        raise RuntimeError(f"expanded-shape contract failed: {expanded.shape}")
    return expanded


def make_penalty_grid(size: int = GRID_SIZE, t_max: float = 6.0) -> Tensor:
    """Reproduce the fixed 96-point penalty grid."""
    eps = 0.05
    n_lo = size // 2
    n_hi = size - n_lo
    lo = [
        math.exp(
            math.log(eps)
            + (math.log(0.3 + eps) - math.log(eps)) * i / (n_lo - 1)
        )
        - eps
        for i in range(n_lo)
    ]
    hi_full = [
        math.exp(
            math.log(0.3 + eps)
            + (math.log(t_max + eps) - math.log(0.3 + eps)) * i / n_hi
        )
        - eps
        for i in range(n_hi + 1)
    ]
    return torch.tensor(lo + hi_full[1:], dtype=torch.float32)


def make_prox_film_cond(
    model: nn.Module,
    lam: Tensor,
    lmx: Tensor,
    d0: Tensor,
    penalty_shape_features: Tensor,
) -> Tensor:
    """Build the 490-D lambda/shape condition used by Kalbi V2."""
    ce = model.cond_embed
    eps = ce.eps
    device = lam.device
    batch_size, query_count = lam.shape
    ratio = (lam / lmx.clamp_min(eps)).clamp_min(eps)
    threshold_ratio = (
        lam * d0.clamp_min(eps) / lmx.clamp_min(eps)
    ).clamp_min(eps)
    lambda_features = torch.stack(
        [
            ratio.log().clamp(-8.0, 1.0) / 4.0,
            ratio.clamp(0.0, 10.0).sqrt() / math.sqrt(10.0),
            torch.tanh(lmx.clamp_min(eps).log() / 5.0),
            threshold_ratio.log().clamp(-8.0, 1.0) / 4.0,
            torch.tanh(d0 / 2.0),
            (lam + 1e-7).log(),
        ],
        dim=-1,
    )
    lambda_normalized = (
        lambda_features - ce._lambda_mean.to(device)
    ) / (ce._lambda_scale.to(device) + 0.1)
    native = strip_convexity_flag(
        penalty_shape_features.to(device=device, dtype=torch.float32)
    )
    expanded = native196_to_expanded388(native, lam.to(torch.float32))
    shape_normalized = (
        expanded - model._shape_mean_388.to(device)
    ) / (model._shape_scale_388.to(device) + 0.1)
    logt = model._logt_grid.to(device).view(1, 1, -1).expand(
        batch_size, query_count, -1
    )
    result = torch.cat([lambda_normalized, shape_normalized, logt], dim=-1)
    if result.shape[-1] != PROX_COND_DIM:
        raise RuntimeError(f"prox condition must be 490-D, got {result.shape}")
    return result.reshape(batch_size * query_count, -1)


class _ResidualBlock(nn.Module):
    def __init__(self, hidden: int, act: type[nn.Module] = nn.SiLU):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.lin = nn.Linear(hidden, hidden)
        self.act = act()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.act(self.lin(self.norm(x)))


class ResidualMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        depth: int = 2,
        zero_init_last: bool = False,
        act: type[nn.Module] = nn.SiLU,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("ResidualMLP depth must be >= 1")
        self.norm = nn.LayerNorm(in_dim)
        self.in_proj = nn.Linear(in_dim, hidden)
        self.act = act()
        self.blocks = nn.ModuleList(
            _ResidualBlock(hidden, act=act) for _ in range(depth - 1)
        )
        self.out = nn.Linear(hidden, out_dim)
        if zero_init_last:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def __getitem__(self, idx: int) -> nn.Module:
        if idx == -1:
            return self.out
        raise IndexError("ResidualMLP only exposes [-1]")

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.in_proj(self.norm(x)))
        for block in self.blocks:
            h = block(h)
        return self.out(h)


class ResidualPenaltyShapeEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 256,
        hidden: int = 256,
        depth: int = 3,
        zero_init_last: bool = True,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.net = ResidualMLP(
            in_dim=self.in_dim,
            hidden=int(hidden),
            out_dim=self.out_dim,
            depth=int(depth),
            zero_init_last=zero_init_last,
            act=nn.GELU,
        )

    def forward(self, shape_features: Tensor) -> Tensor:
        if shape_features.ndim != 3 or shape_features.shape[-1] != self.in_dim:
            raise ValueError(
                f"shape_features must be (B,Q,{self.in_dim}), "
                f"got {tuple(shape_features.shape)}"
            )
        batch_size, query_count, width = shape_features.shape
        return self.net(
            shape_features.reshape(batch_size * query_count, width)
        ).view(batch_size, query_count, -1)


def _apply_film(hidden: Tensor, gamma: Tensor, bias: Tensor) -> Tensor:
    return hidden * (1.0 + torch.tanh(gamma)) + bias


class ABTrajectoryThresholdFiLMHead(nn.Module):
    """Estimate correction and terminal threshold from A/B trajectory rows.

    The input feature width is fixed at ``2 * stages + 5`` and therefore does
    not depend on N or P.  Lambda and penalty-shape features enter only through
    FiLM, while both outputs remain permutation equivariant over P.  The
    correction is support-masked; the threshold multiplier is emitted for all
    features so the dense global Krylov tail can be set exactly to zero.
    """

    def __init__(
        self,
        stages: int,
        film_dim: int = FEATURE_FILM_DIM,
        hidden: int = 512,
        film_hidden: int = 512,
        feature_chunk: int = 2048,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.stages = int(stages)
        self.hidden = int(hidden)
        self.feature_chunk = int(feature_chunk)
        self.use_checkpoint = bool(use_checkpoint)
        # A, B, Xty, gram diagonal, log|Xty|, base coefficient, support flag.
        token_dim = 2 * self.stages + 5
        self.input_proj = nn.Linear(token_dim, self.hidden)
        self.input_norm = nn.LayerNorm(self.hidden)
        self.layer1 = nn.Linear(self.hidden, self.hidden)
        self.norm1 = nn.LayerNorm(self.hidden)
        self.layer2 = nn.Linear(self.hidden, self.hidden)
        self.norm2 = nn.LayerNorm(self.hidden)
        self.layer3 = nn.Linear(self.hidden, self.hidden)
        self.norm3 = nn.LayerNorm(self.hidden)
        self.act = nn.GELU()
        # Channel 0 is the support-local correction. Channel 1 is the learned
        # log multiplier for the terminal, penalty-aware threshold.
        self.out = nn.Linear(self.hidden, 2)
        self.film_gen = nn.Sequential(
            nn.LayerNorm(int(film_dim)),
            nn.Linear(int(film_dim), int(film_hidden)),
            nn.GELU(),
            nn.Linear(int(film_hidden), 6 * self.hidden),
        )

        # Small nonzero initialization is required so selector shape gradients
        # exist from the first joint-training update.
        nn.init.normal_(self.out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.out.bias)
        nn.init.normal_(self.film_gen[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.film_gen[-1].bias)

    def _forward_chunk(
        self,
        trajectory: Tensor,
        xty: Tensor,
        gram_diag: Tensor,
        log_abs_xty: Tensor,
        base: Tensor,
        support: Tensor,
        film: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size, feature_count, trajectory_dim = trajectory.shape
        query_count = base.shape[1]
        if trajectory_dim != 2 * self.stages:
            raise ValueError(
                f"trajectory width must be {2 * self.stages}, "
                f"got {trajectory_dim}"
            )
        shared = torch.cat(
            [
                trajectory,
                xty.unsqueeze(-1),
                gram_diag.unsqueeze(-1),
                log_abs_xty.unsqueeze(-1),
                support.to(trajectory.dtype).unsqueeze(-1),
            ],
            dim=-1,
        )
        shared = shared[:, None].expand(
            batch_size, query_count, feature_count, shared.shape[-1]
        )
        token = torch.cat([shared, base.unsqueeze(-1)], dim=-1)
        token = torch.nan_to_num(
            token.float(), nan=0.0, posinf=20.0, neginf=-20.0
        ).clamp(-20.0, 20.0)
        hidden = self.act(self.input_norm(self.input_proj(token)))
        film = film.view(batch_size, query_count, 1, 6 * self.hidden)
        gamma1, bias1, gamma2, bias2, gamma3, bias3 = film.chunk(6, dim=-1)
        residual = hidden
        hidden = _apply_film(
            self.act(self.norm1(self.layer1(hidden))), gamma1, bias1
        ) + residual
        hidden = _apply_film(
            self.act(self.norm2(self.layer2(hidden))), gamma2, bias2
        ) + hidden
        hidden = _apply_film(
            self.act(self.norm3(self.layer3(hidden))), gamma3, bias3
        ) + hidden
        output = self.out(hidden)
        correction = output[..., 0]
        log_threshold_multiplier = output[..., 1]
        return (
            correction * support[:, None, :].to(correction.dtype),
            log_threshold_multiplier,
        )

    def forward(
        self,
        A: Tensor,
        B: Tensor,
        Xty: Tensor,
        gram_diag: Tensor,
        log_abs_Xty: Tensor,
        base: Tensor,
        support: Tensor,
        film_cond: Tensor,
    ) -> dict[str, Tensor]:
        if A.shape != B.shape or A.ndim != 3:
            raise ValueError(
                f"A and B must match (batch,P,stages), got {A.shape}, {B.shape}"
            )
        batch_size, feature_count, stages = A.shape
        if stages != self.stages:
            raise ValueError(f"expected {self.stages} stages, got {stages}")
        if base.ndim != 3 or base.shape[0] != batch_size or base.shape[2] != feature_count:
            raise ValueError(f"base must be (batch,query,P), got {base.shape}")
        query_count = base.shape[1]
        if film_cond.shape[:2] != (batch_size, query_count):
            raise ValueError(
                f"film condition prefix mismatch: {film_cond.shape} vs "
                f"{(batch_size, query_count)}"
            )
        trajectory = torch.cat([A, B], dim=-1)
        film = self.film_gen(film_cond)
        correction_outputs: list[Tensor] = []
        threshold_outputs: list[Tensor] = []
        chunk = max(1, self.feature_chunk)
        for start in range(0, feature_count, chunk):
            end = min(feature_count, start + chunk)
            args = (
                trajectory[:, start:end],
                Xty[:, start:end],
                gram_diag[:, start:end],
                log_abs_Xty[:, start:end],
                base[:, :, start:end],
                support[:, start:end],
                film,
            )
            if self.training and self.use_checkpoint:
                correction, log_threshold_multiplier = checkpoint(
                    self._forward_chunk, *args, use_reentrant=False
                )
            else:
                correction, log_threshold_multiplier = self._forward_chunk(*args)
            correction_outputs.append(correction)
            threshold_outputs.append(log_threshold_multiplier)
        return {
            "correction": torch.cat(correction_outputs, dim=-1),
            "log_threshold_multiplier": torch.cat(
                threshold_outputs, dim=-1
            ),
        }


def _deterministic_topk_disjoint(
    score: Tensor,
    valid_mask: Tensor,
    selected_mask: Tensor,
    topk: int,
) -> tuple[Tensor, Tensor]:
    if score.ndim != 2:
        raise ValueError(f"score must be (B,P), got {tuple(score.shape)}")
    if valid_mask.shape != score.shape or selected_mask.shape != score.shape:
        raise ValueError("valid and selected masks must match score")
    feature_count = score.shape[-1]
    k = int(topk)
    if not 1 <= k <= feature_count:
        raise ValueError(f"topk must be in [1,P], got {k} for P={feature_count}")
    available = valid_mask.to(torch.bool) & ~selected_mask.to(torch.bool)
    active = available.sum(dim=-1) >= k
    finite = torch.nan_to_num(
        score.float(),
        nan=0.0,
        posinf=torch.finfo(torch.float32).max,
        neginf=0.0,
    ).clamp_min(0.0)
    score_bits = finite.contiguous().view(torch.int32).to(torch.int64)
    index = torch.arange(feature_count, device=score.device, dtype=torch.int64)
    tie_priority = (feature_count - 1 - index).view(1, feature_count)
    key = score_bits * (feature_count + 1) + tie_priority
    key = key.masked_fill(~available, -1)
    indices = torch.topk(key, k=k, dim=-1).indices.sort(dim=-1).values
    return indices, active


def _ridge_system(
    X: Tensor,
    target: Tensor,
    ridge: float,
    *,
    dual: bool,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Construct the normalized primal or dual ridge system in ``dtype``."""
    sample_count = int(X.shape[1])
    Xd = X.to(dtype=dtype)
    yd = target.to(dtype=dtype)
    if dual:
        system = torch.bmm(Xd, Xd.transpose(1, 2)) / float(max(1, sample_count))
        rhs = yd / float(max(1, sample_count))
        width = sample_count
    else:
        system = torch.bmm(Xd.transpose(1, 2), Xd) / float(max(1, sample_count))
        rhs = torch.bmm(Xd.transpose(1, 2), yd.unsqueeze(-1)).squeeze(-1)
        rhs = rhs / float(max(1, sample_count))
        width = int(X.shape[2])
    system = 0.5 * (system + system.transpose(1, 2))
    eye = torch.eye(width, device=X.device, dtype=dtype).expand(
        int(X.shape[0]), width, width
    )
    return system + float(ridge) * eye, rhs


def _solve_fixed_ridge_system(
    X: Tensor,
    target: Tensor,
    ridge: float,
    *,
    dual: bool,
) -> Tensor:
    """Solve in FP32, rebuilding the exact same ridge system in FP64 on failure."""
    system, rhs = _ridge_system(
        X, target, ridge, dual=dual, dtype=torch.float32
    )
    if not bool(torch.isfinite(system).all() and torch.isfinite(rhs).all()):
        raise RuntimeError("fixed-ridge system contains nonfinite values")
    solution, info = torch.linalg.solve_ex(
        system, rhs.unsqueeze(-1), check_errors=False
    )
    finite = torch.isfinite(solution).all(dim=(-2, -1))
    solved = info.eq(0) & finite
    _FIXED_RIDGE_NUMERICS["solve_calls"] += 1
    _FIXED_RIDGE_NUMERICS["systems"] += int(X.shape[0])
    if bool(solved.all()):
        return solution.squeeze(-1)

    failed = int((~solved).sum().item())
    _FIXED_RIDGE_NUMERICS["float32_failed_systems"] += failed
    _FIXED_RIDGE_NUMERICS["float64_fallback_calls"] += 1
    _FIXED_RIDGE_NUMERICS["float64_fallback_systems"] += failed

    # Recompute X'X or XX' in float64. Casting the already rounded FP32 Gram
    # matrix would not restore a 1e-6 diagonal shift lost to FP32 resolution.
    system64, rhs64 = _ridge_system(
        X, target, ridge, dual=dual, dtype=torch.float64
    )
    solution64, info64 = torch.linalg.solve_ex(
        system64, rhs64.unsqueeze(-1), check_errors=False
    )
    finite64 = torch.isfinite(solution64).all(dim=(-2, -1))
    solved64 = info64.eq(0) & finite64
    if not bool(solved64.all()):
        eye64 = torch.eye(
            int(system64.shape[-1]),
            device=system64.device,
            dtype=torch.float64,
        ).expand_as(system64)
        diagonal_scale = system64.diagonal(dim1=-2, dim2=-1).abs().amax(-1)
        diagonal_scale = diagonal_scale.clamp_min(1.0)
        unresolved = ~solved64
        output64 = torch.empty_like(solution64)
        output64[solved64] = solution64[solved64]
        for multiplier in (64.0, 4096.0, 262144.0):
            relative = torch.finfo(torch.float64).eps * multiplier
            jitter = diagonal_scale * relative
            candidate, candidate_info = torch.linalg.solve_ex(
                system64 + jitter[:, None, None] * eye64,
                rhs64.unsqueeze(-1),
                check_errors=False,
            )
            candidate_ok = (
                candidate_info.eq(0)
                & torch.isfinite(candidate).all(dim=(-2, -1))
                & unresolved
            )
            output64[candidate_ok] = candidate[candidate_ok]
            unresolved = unresolved & ~candidate_ok
            if bool(candidate_ok.any()):
                _FIXED_RIDGE_NUMERICS["float64_jitter_calls"] += 1
                _FIXED_RIDGE_NUMERICS["max_float64_relative_jitter"] = max(
                    float(_FIXED_RIDGE_NUMERICS["max_float64_relative_jitter"]),
                    float(relative),
                )
            if not bool(unresolved.any()):
                break
        if bool(unresolved.any()):
            raise RuntimeError(
                "fixed-ridge solve failed after exact FP64 rebuild and jitter"
            )
        solution64 = output64
    return solution64.squeeze(-1).to(dtype=torch.float32)


def _fixed_ridge_fit(X: Tensor, target: Tensor, ridge: float) -> Tensor:
    """Solve normalized ridge using the smaller primal/dual system.

    The statistical ridge remains exactly ``ridge``. FP64 is used only when an
    FP32 solver reports a singular/nonfinite system; numerical jitter is a last
    resort after rebuilding the Gram/kernel matrix in FP64.
    """
    if X.ndim != 3 or target.ndim != 2 or X.shape[:2] != target.shape:
        raise ValueError(f"ridge shape mismatch: X={X.shape}, target={target.shape}")
    batch_size, sample_count, width = X.shape
    alpha = float(ridge)
    if alpha <= 0.0:
        raise ValueError(f"ridge must be positive, got {alpha}")
    Xf = X.float()
    yf = target.float()
    if width <= sample_count:
        return _solve_fixed_ridge_system(
            Xf, yf, alpha, dual=False
        )
    dual = _solve_fixed_ridge_system(Xf, yf, alpha, dual=True)
    return torch.bmm(Xf.transpose(1, 2), dual.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def build_kalbi_ab_trajectory(
    X: Tensor,
    y: Tensor,
    feature_mask: Tensor | None,
    topk: int = 32,
    stages: int = 32,
    ridge: float = 1e-6,
    eps: float = 1e-8,
) -> dict[str, Tensor]:
    """Build disjoint centered-correlation ISIS A/B coefficient paths."""
    if X.ndim != 3:
        raise ValueError(f"X must be (B,N,P), got {tuple(X.shape)}")
    if y.ndim == 3 and y.shape[-1] == 1:
        y = y.squeeze(-1)
    if y.ndim != 2 or y.shape != X.shape[:2]:
        raise ValueError(f"y must be (B,N), got {tuple(y.shape)}")
    batch_size, sample_count, feature_count = X.shape
    selected_width = int(topk)
    requested_stages = int(stages)
    if requested_stages <= 0:
        raise ValueError(f"stages must be positive, got {requested_stages}")
    if selected_width > feature_count:
        raise ValueError(
            f"topk={selected_width} exceeds feature count P={feature_count}"
        )

    Xc = X.float() - X.float().mean(dim=1, keepdim=True)
    yc = y.float() - y.float().mean(dim=1, keepdim=True)
    x_rms = Xc.square().mean(dim=1).sqrt()
    if feature_mask is None:
        valid = torch.ones(
            batch_size, feature_count, dtype=torch.bool, device=X.device
        )
    else:
        if feature_mask.shape != (batch_size, feature_count):
            raise ValueError(
                f"feature_mask must be (B,P), got {tuple(feature_mask.shape)}"
            )
        valid = feature_mask.to(device=X.device, dtype=torch.bool)
    valid &= x_rms > float(eps)

    selected = torch.zeros_like(valid)
    residual = yc.clone()
    A_columns: list[Tensor] = []
    B_columns: list[Tensor] = []
    index_columns: list[Tensor] = []
    active_columns: list[Tensor] = []
    residual_rms: list[Tensor] = [
        residual.square().mean(dim=-1).sqrt()
    ]

    for _stage in range(requested_stages):
        residual_centered = residual - residual.mean(dim=-1, keepdim=True)
        residual_scale = residual_centered.square().mean(dim=-1).sqrt()
        covariance = torch.einsum("bnp,bn->bp", Xc, residual_centered)
        covariance = covariance / float(max(1, sample_count))
        correlation = covariance.abs() / (
            x_rms * residual_scale.unsqueeze(-1)
        ).clamp_min(float(eps))
        indices, active = _deterministic_topk_disjoint(
            correlation, valid, selected, selected_width
        )
        safe_indices = indices.masked_fill(~active.unsqueeze(-1), 0)

        new_mask = torch.zeros_like(selected)
        new_mask.scatter_(-1, safe_indices, True)
        selected |= new_mask & active.unsqueeze(-1)

        X_new_all = torch.gather(
            Xc,
            2,
            safe_indices[:, None, :].expand(
                batch_size, sample_count, selected_width
            ),
        )
        a_selected = Xc.new_zeros(batch_size, selected_width)
        if bool(active.any()):
            a_selected[active] = _fixed_ridge_fit(
                X_new_all[active], residual_centered[active], ridge
            )
        a_column = Xc.new_zeros(batch_size, feature_count)
        a_column.scatter_add_(-1, safe_indices, a_selected)

        fitted_increment = torch.einsum("bnk,bk->bn", X_new_all, a_selected)
        residual = residual - fitted_increment
        residual = residual - residual.mean(dim=-1, keepdim=True)

        index_columns.append(
            safe_indices.masked_fill(~active.unsqueeze(-1), -1)
        )
        active_columns.append(active)
        joint_indices = torch.stack(index_columns, dim=1).reshape(
            batch_size, -1
        )
        joint_width = joint_indices.shape[-1]
        safe_joint_indices = joint_indices.masked_fill(
            ~active.unsqueeze(-1), 0
        )
        X_joint_all = torch.gather(
            Xc,
            2,
            safe_joint_indices[:, None, :].expand(
                batch_size, sample_count, joint_width
            ),
        )
        b_selected = Xc.new_zeros(batch_size, joint_width)
        if bool(active.any()):
            b_selected[active] = _fixed_ridge_fit(
                X_joint_all[active], yc[active], ridge
            )
        b_column = Xc.new_zeros(batch_size, feature_count)
        b_column.scatter_add_(-1, safe_joint_indices, b_selected)

        A_columns.append(a_column)
        B_columns.append(b_column)
        residual_rms.append(residual.square().mean(dim=-1).sqrt())

    A = torch.stack(A_columns, dim=-1)
    B = torch.stack(B_columns, dim=-1)
    indices_out = torch.stack(index_columns, dim=1)
    active_out = torch.stack(active_columns, dim=1)
    residual_rms_out = torch.stack(residual_rms, dim=1)
    support = selected & valid
    if A.shape != (batch_size, feature_count, requested_stages):
        raise RuntimeError(f"A trajectory shape drifted: {A.shape}")
    if B.shape != A.shape:
        raise RuntimeError(f"B trajectory shape drifted: {B.shape}")
    return {
        "A": A,
        "B": B,
        "selected_indices": indices_out,
        "stage_active": active_out,
        "support_mask": support,
        "residual_rms": residual_rms_out,
        "ridge": Xc.new_full((batch_size,), float(ridge)),
    }


__all__ = [
    "ABTrajectoryThresholdFiLMHead",
    "EXPANDED_DIM",
    "FEATURE_FILM_DIM",
    "GRID_SIZE",
    "LOCAL_ISIS_GRADIENT_CONTRACT",
    "META_DIM",
    "NATIVE_DIM",
    "PROX_COND_DIM",
    "ResidualMLP",
    "ResidualPenaltyShapeEncoder",
    "build_kalbi_ab_trajectory",
    "make_penalty_grid",
    "make_prox_film_cond",
    "native196_to_expanded388",
    "strip_convexity_flag",
]
