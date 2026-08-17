"""Self-contained V58 neural modules and numerical helpers.

This module intentionally contains no imports from versioned model modules.
The only model-level dependency of the refactored package is the stable
``model_fpk`` bootstrap used by :mod:`model_v58_chebyshev.model`.

V58 freezes only the discrete ISIS support decision.  Once an index set is
chosen, residual/RHS recomputation, the local Krylov/ridge solve, mixer
features, lambda-penalty threshold, soft threshold, scatter, and stage update
remain in the autograd graph.  This preserves V57's hard forward while making
its continuous fixed-support map differentiable.
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
    "detach_discrete_support_recompute_continuous_local_path_v1"
)


def strip_convexity_flag(shape: Tensor) -> Tensor:
    """Convert legacy native197/meta5 to native196/meta4 without detaching."""
    if shape.shape[-1] == NATIVE_DIM:
        return shape
    if shape.shape[-1] != LEGACY_NATIVE_DIM:
        raise ValueError(
            f"expected native {LEGACY_NATIVE_DIM} or {NATIVE_DIM}, got {shape.shape[-1]}"
        )
    return shape[..., :LEGACY_CONVEXITY_FLAG_INDEX]


def native196_to_expanded388(shape: Tensor, lambdas: Tensor) -> Tensor:
    """Differentiably deinterleave and lambda-expand the public penalty input."""
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
    """Reproduce the fixed V33/V56 96-point penalty grid exactly."""
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
    """Build the 490-D V57 query condition."""
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
        raise IndexError("ResidualMLP only exposes [-1] for Sequential compatibility")

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
        if shape_features.dim() != 3:
            raise ValueError(
                f"shape_features must be (B,Q,D), got {tuple(shape_features.shape)}"
            )
        if shape_features.shape[-1] != self.in_dim:
            raise ValueError(
                f"shape_features last dim {shape_features.shape[-1]} "
                f"!= in_dim {self.in_dim}"
            )
        batch_size, query_count, width = shape_features.shape
        return self.net(
            shape_features.reshape(batch_size * query_count, width)
        ).view(batch_size, query_count, -1)


class ResidualQueryMixerGramAware(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        num_channels: int,
        krylov_dim: int,
        t_feat_dim: int,
        hidden: int = 256,
        proj_dim: int = 256,
        mix_scale: float = 2.0,
        zero_init: bool = True,
    ):
        super().__init__()
        self.num_channels = int(num_channels)
        self.krylov_dim = int(krylov_dim)
        self.mix_scale = float(mix_scale)
        self.cond_proj = nn.Linear(int(cond_dim), int(proj_dim))
        self.t_proj = nn.Sequential(
            nn.LayerNorm(int(t_feat_dim)), nn.Linear(int(t_feat_dim), int(proj_dim))
        )
        self.net = ResidualMLP(
            in_dim=int(proj_dim),
            hidden=int(hidden),
            out_dim=self.num_channels * self.krylov_dim,
            depth=3,
            zero_init_last=zero_init,
            act=nn.GELU,
        )

    def forward(self, cond: Tensor, t_feat: Tensor) -> Tensor:
        batch_size, query_count, _ = cond.shape
        hidden = self.cond_proj(cond) + self.t_proj(t_feat).unsqueeze(1)
        raw = self.net(hidden).view(
            batch_size, query_count, self.num_channels, self.krylov_dim
        )
        return self.mix_scale * torch.tanh(raw)


class ChunkedSharedFeatureEmissionHead(nn.Module):
    """Consolidated final V57 per-feature head with legacy state-dict names."""

    def __init__(
        self,
        num_channels: int,
        cond_dim: int,
        hidden: int = 512,
        tau_scale_bound: float = 2.0,
        zero_init: bool = True,
        use_bilinear: bool = True,
        dropout: float = 0.0,
        film_dim: int = FEATURE_FILM_DIM,
        film_hidden: int = 512,
        feature_chunk: int = 4096,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.tau_scale_bound = tau_scale_bound
        self.use_bilinear = use_bilinear
        self.num_channels = num_channels
        if use_bilinear:
            triu_row, triu_col = torch.triu_indices(num_channels, num_channels)
            self.register_buffer("_triu_row", triu_row)
            self.register_buffer("_triu_col", triu_col)
            channel_feature_dim = num_channels + triu_row.shape[0]
        else:
            channel_feature_dim = num_channels
        in_dim = channel_feature_dim + 3 + cond_dim
        self.input_proj = nn.Linear(in_dim, hidden)
        self.input_norm = nn.LayerNorm(in_dim)
        self.layer1 = nn.Linear(in_dim, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.layer3 = nn.Linear(hidden, hidden)
        self.norm3 = nn.LayerNorm(hidden)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out = nn.Linear(hidden, 2)
        if zero_init:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        self.ubase_proj = nn.Linear(1, hidden)
        nn.init.zeros_(self.ubase_proj.bias)
        self.film_gen = nn.Sequential(
            nn.LayerNorm(film_dim),
            nn.Linear(film_dim, film_hidden),
            nn.ReLU(),
            nn.Linear(film_hidden, 6 * hidden),
        )
        nn.init.zeros_(self.film_gen[-1].weight)
        nn.init.zeros_(self.film_gen[-1].bias)
        self.feature_chunk = int(feature_chunk)
        self.use_checkpoint = bool(use_checkpoint)

    def _forward_chunk(
        self,
        S: Tensor,
        Xty: Tensor,
        gram_diag: Tensor,
        log_abs_Xty: Tensor,
        cond: Tensor,
        film_cond: Tensor | None,
        u_base: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        batch_size, query_count, _, feature_count = S.shape
        s_flat = S.permute(0, 1, 3, 2)
        if self.use_bilinear:
            s_i = s_flat[..., self._triu_row]
            s_j = s_flat[..., self._triu_col]
            s_features = torch.cat([s_flat, s_i * s_j], dim=-1)
        else:
            s_features = s_flat
        x_features = Xty[:, None, :, None].expand(
            batch_size, query_count, feature_count, 1
        )
        g_features = gram_diag[:, None, :, None].expand(
            batch_size, query_count, feature_count, 1
        )
        l_features = log_abs_Xty[:, None, :, None].expand(
            batch_size, query_count, feature_count, 1
        )
        c_features = cond[:, :, None, :].expand(
            batch_size, query_count, feature_count, -1
        )
        token = torch.cat(
            [s_features, x_features, g_features, l_features, c_features], dim=-1
        )
        token_flat = token.reshape(batch_size * query_count, feature_count, -1)
        if film_cond is not None:
            film = self.film_gen(film_cond).reshape(
                batch_size * query_count, 1, -1
            )
            g1, b1, g2, b2, g3, b3 = film.chunk(6, dim=-1)
        else:
            g1 = b1 = g2 = b2 = g3 = b3 = None

        def apply_film(h: Tensor, gamma: Tensor | None, bias: Tensor | None) -> Tensor:
            return h if gamma is None else h * (1.0 + torch.tanh(gamma)) + bias

        x0 = self.input_norm(token_flat)
        residual = self.input_proj(x0)
        ubase = (
            self.ubase_proj(
                u_base.reshape(batch_size * query_count, feature_count, 1)
            )
            if u_base is not None
            else 0.0
        )
        hidden = apply_film(self.act(self.norm1(self.layer1(x0) + ubase)), g1, b1)
        hidden = self.dropout(hidden)
        hidden = hidden + residual
        hidden = apply_film(self.act(self.norm2(self.layer2(hidden))), g2, b2) + hidden
        hidden = self.dropout(hidden)
        hidden = apply_film(self.act(self.norm3(self.layer3(hidden))), g3, b3) + hidden
        hidden = self.dropout(hidden)
        raw = self.out(hidden).view(batch_size, query_count, feature_count, 2)
        return raw[..., 0], raw[..., 1].clamp(
            -self.tau_scale_bound, self.tau_scale_bound
        )

    def forward(
        self,
        S: Tensor,
        Xty: Tensor,
        gram_diag: Tensor,
        log_abs_Xty: Tensor,
        cond: Tensor,
        film_cond: Tensor | None = None,
        u_base: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        feature_count = S.shape[-1]
        chunk = max(1, int(self.feature_chunk))
        if feature_count <= chunk:
            return self._forward_chunk(
                S, Xty, gram_diag, log_abs_Xty, cond, film_cond, u_base
            )
        outputs_u: list[Tensor] = []
        outputs_log: list[Tensor] = []
        for start in range(0, feature_count, chunk):
            end = min(feature_count, start + chunk)
            args = (
                S[..., start:end],
                Xty[..., start:end],
                gram_diag[..., start:end],
                log_abs_Xty[..., start:end],
                cond,
                film_cond,
                u_base[..., start:end] if u_base is not None else None,
            )
            if self.training and self.use_checkpoint:
                u, log_tau = checkpoint(
                    self._forward_chunk, *args, use_reentrant=False
                )
            else:
                u, log_tau = self._forward_chunk(*args)
            outputs_u.append(u)
            outputs_log.append(log_tau)
        return torch.cat(outputs_u, dim=-1), torch.cat(outputs_log, dim=-1)


def soft_threshold(z: Tensor, tau: Tensor) -> Tensor:
    return torch.sign(z) * torch.relu(z.abs() - tau.clamp_min(0.0))


def pad_last(x: Tensor, width: int) -> Tensor:
    if x.shape[-1] >= width:
        return x[..., :width]
    return torch.nn.functional.pad(x, (0, width - x.shape[-1]))


def solve_from_phi(gram: Tensor, rhs: Tensor, alpha: Tensor, phi: Tensor) -> Tensor:
    if rhs.dim() == 3 and rhs.shape[1] == 1:
        rhs_vector = rhs[:, 0, :]
    elif rhs.dim() == 2:
        rhs_vector = rhs
    else:
        raise ValueError(
            f"rhs must be (BQ,1,K) or (BQ,K), got {tuple(rhs.shape)}"
        )
    gram_phi = torch.bmm(gram, phi.transpose(1, 2)).transpose(1, 2)
    projected = torch.bmm(phi, gram_phi.transpose(1, 2))
    projected_rhs = torch.einsum("bmk,bk->bm", phi, rhs_vector)
    width = projected.shape[-1]
    eye = torch.eye(
        width, device=projected.device, dtype=projected.dtype
    ).expand(projected.shape[0], width, width)
    ridge = alpha.reshape(-1, 1, 1).to(projected.dtype)
    coefficients = torch.linalg.solve(
        projected + ridge * eye + 1e-6 * eye,
        projected_rhs.unsqueeze(-1),
    ).squeeze(-1)
    return torch.einsum("bm,bmk->bk", coefficients, phi)


def local_orthonormal_krylov_basis(
    gram: Tensor,
    rhs: Tensor,
    steps: int,
    max_rank: int | None = None,
    atol: float = 1e-12,
    breakdown_eps_multiplier: float = 64.0,
) -> Tensor:
    if rhs.ndim != 3:
        raise ValueError(f"rhs must be (B,Q,K), got {tuple(rhs.shape)}")
    batch_size, _, width = rhs.shape
    if gram.shape != (batch_size, width, width):
        raise ValueError(
            f"gram must be (B,K,K), got gram={tuple(gram.shape)} "
            f"rhs={tuple(rhs.shape)}"
        )
    rank_cap = width if max_rank is None else max(0, min(int(max_rank), width))
    basis_size = min(int(steps), width, rank_cap)
    if basis_size < 1:
        raise ValueError(
            "Krylov basis requires at least one permitted row, got "
            f"steps={steps}, K={width}, max_rank={max_rank}"
        )
    q = rhs
    dtype_eps = torch.finfo(q.dtype).eps
    if q.dtype in (torch.float16, torch.bfloat16):
        numerical_rank_rtol = 1e-2
    elif q.dtype == torch.float32:
        numerical_rank_rtol = 2e-3
    else:
        numerical_rank_rtol = 1e-10
    numerical_rank_rtol = max(
        numerical_rank_rtol,
        float(breakdown_eps_multiplier) * float(dtype_eps),
    )
    operator_scale = gram.abs().sum(dim=-1).amax(dim=-1).view(batch_size, 1, 1)
    rhs_scale = rhs.norm(dim=-1, keepdim=True)
    absolute_tol = torch.full_like(rhs_scale, float(atol))
    basis: list[Tensor] = []
    for step in range(basis_size):
        for _ in range(2):
            for previous in basis:
                projection = (q * previous).sum(dim=-1, keepdim=True)
                q = q - projection * previous
        q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
        norm = q.norm(dim=-1, keepdim=True)
        reference_scale = rhs_scale if step == 0 else operator_scale
        tolerance = torch.maximum(
            absolute_tol, numerical_rank_rtol * reference_scale
        )
        alive = norm > tolerance
        q = torch.where(alive, q / norm.clamp_min(float(atol)), torch.zeros_like(q))
        basis.append(q)
        q = torch.einsum("bij,bqj->bqi", gram, q)
    return torch.stack(basis, dim=2)


def local_chebyshev_basis_rowbound_recurrence(
    gram: Tensor,
    rhs: Tensor,
    steps: int,
    *,
    max_rank: int | None = None,
    atol: float = 1e-12,
) -> Tensor:
    """Build the exact V57-Chebyshev row-bound three-term recurrence.

    The local Gram operator is mapped approximately to ``[-1, 1]`` by

    ``Ahat = 2 * gram / max_i(sum_j(abs(gram_ij))) - I``.

    Starting from the L2-normalized RHS, apply
    ``t[j+1] = 2 * Ahat @ t[j] - t[j-1]`` and L2-normalize every emitted row.
    There is no eigendecomposition and no projection against earlier rows.
    """
    if rhs.ndim != 3:
        raise ValueError(f"rhs must be (B,Q,K), got {tuple(rhs.shape)}")
    batch_size, _, width = rhs.shape
    if gram.shape != (batch_size, width, width):
        raise ValueError(
            f"Chebyshev shape mismatch gram={tuple(gram.shape)} "
            f"rhs={tuple(rhs.shape)}"
        )
    rank_cap = width if max_rank is None else max(0, min(int(max_rank), width))
    basis_size = min(int(steps), width, rank_cap)
    if basis_size < 1:
        raise ValueError(
            "Chebyshev recurrence requires at least one permitted row, got "
            f"steps={steps}, width={width}, max_rank={max_rank}"
        )
    row_bound = gram.abs().sum(dim=-1).amax(dim=-1).view(
        batch_size, 1, 1
    )
    row_bound = row_bound.clamp_min(max(float(atol), 1e-3))

    def apply_scaled_operator(vector: Tensor) -> Tensor:
        gram_vector = torch.einsum("bij,bqj->bqi", gram, vector)
        return 2.0 * gram_vector / row_bound - vector

    def normalize(vector: Tensor) -> Tensor:
        vector = torch.nan_to_num(
            vector, nan=0.0, posinf=0.0, neginf=0.0
        )
        norm = vector.norm(dim=-1, keepdim=True)
        alive = norm > float(atol)
        return torch.where(
            alive,
            vector / norm.clamp_min(float(atol)),
            torch.zeros_like(vector),
        )

    previous = normalize(rhs)
    basis: list[Tensor] = [previous]
    if basis_size == 1:
        return torch.stack(basis, dim=2)
    current = normalize(apply_scaled_operator(previous))
    basis.append(current)
    for _step in range(2, basis_size):
        following = normalize(
            2.0 * apply_scaled_operator(current) - previous
        )
        basis.append(following)
        previous, current = current, following
    return torch.stack(basis, dim=2)


def solve_from_general_phi(
    gram: Tensor, rhs: Tensor, alpha: Tensor, phi: Tensor
) -> Tensor:
    """Ridge solve in a non-orthonormal row basis.

    Chebyshev rows are L2-normalized but not orthogonal.  Therefore the
    projected ridge term is ``alpha * phi @ phi.T`` rather than ``alpha * I``.
    The production N=200/P=400 contract uses 32 steps on a width-32 support
    and takes the exact full local solve, but this keeps lower-step ablations
    geometrically correct.
    """
    rhs_vector = rhs[:, 0, :] if rhs.ndim == 3 else rhs
    gram_phi = torch.bmm(gram, phi.transpose(1, 2)).transpose(1, 2)
    projected = torch.bmm(phi, gram_phi.transpose(1, 2))
    projected_rhs = torch.einsum("bmk,bk->bm", phi, rhs_vector)
    basis_gram = torch.bmm(phi, phi.transpose(1, 2))
    width = projected.shape[-1]
    eye = torch.eye(
        width, device=projected.device, dtype=projected.dtype
    ).expand(projected.shape[0], width, width)
    ridge = alpha.reshape(-1, 1, 1).to(projected.dtype)
    coefficients = torch.linalg.solve(
        projected + ridge * basis_gram + 1e-6 * eye,
        projected_rhs.unsqueeze(-1),
    ).squeeze(-1)
    return torch.einsum("bm,bmk->bk", coefficients, phi)


def select_new_isis_indices(
    score: Tensor,
    valid_mask: Tensor,
    selected_mask: Tensor,
    topk: int,
) -> tuple[Tensor, Tensor]:
    if score.ndim != 3:
        raise ValueError(f"score must be (B,Q,P), got {tuple(score.shape)}")
    if valid_mask.shape != score.shape or selected_mask.shape != score.shape:
        raise ValueError(
            "valid_mask and selected_mask must match score shape, got "
            f"score={tuple(score.shape)} valid={tuple(valid_mask.shape)} "
            f"selected={tuple(selected_mask.shape)}"
        )
    k = int(topk)
    feature_count = int(score.shape[-1])
    if k <= 0:
        raise ValueError(f"topk must be positive, got {k}")
    if k > feature_count:
        raise ValueError(
            f"exact topk={k} selection is impossible with P={feature_count}"
        )
    available = valid_mask.to(torch.bool) & ~selected_mask.to(torch.bool)
    active = available.sum(dim=-1) >= k
    finite_score = torch.nan_to_num(
        score.float(),
        nan=0.0,
        posinf=torch.finfo(torch.float32).max,
        neginf=0.0,
    ).clamp_min(0.0)
    score_bits = finite_score.contiguous().view(torch.int32).to(torch.int64)
    global_index = torch.arange(
        feature_count, device=score.device, dtype=torch.int64
    )
    tie_priority = (feature_count - 1 - global_index).view(
        *([1] * (score.ndim - 1)), feature_count
    )
    key = score_bits * (feature_count + 1) + tie_priority
    key = key.masked_fill(~available, -1)
    indices = torch.topk(key, k=k, dim=-1).indices
    return indices.sort(dim=-1).values, active


def valid_support_fraction(
    beta_abs: Tensor, valid_mask: Tensor, threshold: float = 1e-3
) -> Tensor:
    if beta_abs.shape != valid_mask.shape:
        raise ValueError(
            f"beta_abs and valid_mask must match, got "
            f"{tuple(beta_abs.shape)} and {tuple(valid_mask.shape)}"
        )
    valid = valid_mask.to(torch.bool)
    return (
        ((beta_abs > float(threshold)) & valid).sum(dim=-1).float()
        / valid.sum(dim=-1).clamp_min(1).float()
    )


def solve_local_ridge(
    gram: Tensor,
    rhs: Tensor,
    alpha: Tensor,
    phi: Tensor,
    requested_steps: int,
) -> Tensor:
    width = int(gram.shape[-1])
    if int(requested_steps) >= width:
        rhs_vector = rhs[:, 0, :] if rhs.ndim == 3 else rhs
        eye = torch.eye(width, device=gram.device, dtype=gram.dtype).expand(
            gram.shape[0], width, width
        )
        system = gram + (
            alpha.reshape(-1, 1, 1).to(gram.dtype) + 1e-6
        ) * eye
        return torch.linalg.solve(system, rhs_vector.unsqueeze(-1)).squeeze(-1)
    return solve_from_general_phi(gram, rhs, alpha, phi)


def support_memory_local_mixer_correction(
    model: nn.Module,
    encoded: dict,
    beta: Tensor,
    lam: Tensor,
    d0: Tensor,
    alpha: Tensor,
    topk: int,
    steps: int,
    beta_bonus: float,
    stage_frac: float,
    valid_mask: Tensor,
    selected_mask: Tensor,
    fixed_indices: Tensor | None = None,
    fixed_active: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """V58 correction with detached support and differentiable local math.

    ``fixed_indices``/``fixed_active`` are an audit-only override.  They freeze
    the support plan across perturbed forwards while all continuous quantities
    are recomputed from the current beta/lambda/penalty inputs.
    """
    X = encoded["Xc"].float()
    y = encoded["yc"].float()
    batch_size, sample_count, feature_count = X.shape
    _, query_count = lam.shape
    selected_width = int(topk)
    expected_indices_shape = (batch_size, query_count, selected_width)
    expected_active_shape = (batch_size, query_count)

    # The discrete decision is the only detached part of V58 local ISIS.
    with torch.no_grad():
        if (fixed_indices is None) != (fixed_active is None):
            raise ValueError(
                "fixed_indices and fixed_active must be provided together"
            )
        if fixed_indices is None:
            selection_residual = y[:, None, :] - torch.einsum(
                "bnp,bqp->bqn", X, beta.float()
            )
            selection_rhs = torch.einsum(
                "bnp,bqn->bqp", X, selection_residual
            ) / float(max(1, sample_count))
            selection_diag = encoded["gram_diag"].float().clamp_min(
                max(float(model.cfg.diag_eps), 1e-3)
            )
            selection_residual_score = (
                selection_rhs.abs() / selection_diag[:, None, :].sqrt()
            )
            selection_beta_abs = beta.float().abs()
            selection_beta_scale = selection_beta_abs.amax(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            selection_residual_scale = selection_residual_score.amax(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            selection_memory_score = selection_residual_scale * (
                selection_beta_abs / selection_beta_scale
            )
            selection_score = selection_residual_score + float(beta_bonus) * (
                selection_memory_score
            )
            indices, active = select_new_isis_indices(
                selection_score,
                valid_mask=valid_mask,
                selected_mask=selected_mask,
                topk=selected_width,
            )
        else:
            if tuple(fixed_indices.shape) != expected_indices_shape:
                raise ValueError(
                    "fixed_indices shape mismatch: expected "
                    f"{expected_indices_shape}, got {tuple(fixed_indices.shape)}"
                )
            if tuple(fixed_active.shape) != expected_active_shape:
                raise ValueError(
                    "fixed_active shape mismatch: expected "
                    f"{expected_active_shape}, got {tuple(fixed_active.shape)}"
                )
            raw_indices = fixed_indices.to(device=beta.device, dtype=torch.long)
            active = fixed_active.to(device=beta.device, dtype=torch.bool)
            active_indices = raw_indices[active]
            if active_indices.numel():
                if bool((active_indices < 0).any()) or bool(
                    (active_indices >= feature_count).any()
                ):
                    raise ValueError("fixed active ISIS index is out of range")
                sorted_indices = active_indices.sort(dim=-1).values
                if bool((sorted_indices[:, 1:] == sorted_indices[:, :-1]).any()):
                    raise ValueError("fixed active ISIS support contains duplicates")
            indices = raw_indices.masked_fill(~active.unsqueeze(-1), 0)
            available = valid_mask.to(torch.bool) & ~selected_mask.to(torch.bool)
            chosen_available = torch.gather(available, -1, indices)
            if bool((~chosen_available & active.unsqueeze(-1)).any()):
                raise ValueError(
                    "fixed active ISIS support selects an invalid or reused feature"
                )

    # Recompute every continuous quantity outside no_grad using detached
    # support indices.  The operation order matches V57's hard forward.
    residual = y[:, None, :] - torch.einsum(
        "bnp,bqp->bqn", X, beta.float()
    )
    rhs_full = torch.einsum("bnp,bqn->bqp", X, residual) / float(
        max(1, sample_count)
    )
    diag_full = encoded["gram_diag"].float().clamp_min(
        max(float(model.cfg.diag_eps), 1e-3)
    )
    residual_score = rhs_full.abs() / diag_full[:, None, :].sqrt()
    beta_abs = beta.float().abs()
    beta_scale = beta_abs.amax(dim=-1, keepdim=True).clamp_min(1e-6)
    residual_scale = residual_score.amax(dim=-1, keepdim=True).clamp_min(1e-6)
    memory_score = residual_scale * (beta_abs / beta_scale)
    score = residual_score + float(beta_bonus) * memory_score
    rhs_all = torch.gather(rhs_full, -1, indices)
    score_selected_all = torch.gather(score, -1, indices)
    active_flat = active.reshape(-1)
    if not bool(active_flat.any()):
        return (
            X.new_zeros(batch_size, query_count, feature_count),
            indices,
            active,
        )
    X_selected_all = torch.gather(
        X[:, None, :, :].expand(
            batch_size, query_count, sample_count, feature_count
        ),
        3,
        indices[:, :, None, :].expand(
            batch_size, query_count, sample_count, selected_width
        ),
    )
    X_selected = X_selected_all.reshape(
        batch_size * query_count, sample_count, selected_width
    )[active_flat]
    rhs = rhs_all.reshape(batch_size * query_count, selected_width)[active_flat]
    score_selected = score_selected_all.reshape(
        batch_size * query_count, selected_width
    )[active_flat]
    gram = torch.einsum("ank,anl->akl", X_selected, X_selected) / float(
        max(1, sample_count)
    )
    diag = gram.diagonal(dim1=-2, dim2=-1).clamp_min(
        max(float(model.cfg.diag_eps), 1e-3)
    )
    batch_queries = batch_size * query_count
    lam_active = lam.float().reshape(batch_queries)[active_flat]
    d0_active = d0.float().reshape(batch_queries)[active_flat]
    alpha_active = alpha.float().reshape(batch_queries, 1)[active_flat]
    tau = lam_active.unsqueeze(-1) * d0_active.unsqueeze(-1) / diag
    rhs_for_basis = rhs.unsqueeze(1)
    centered_rank_cap = min(max(sample_count - 1, 0), selected_width)
    if centered_rank_cap < 1:
        return (
            X.new_zeros(batch_size, query_count, feature_count),
            indices,
            torch.zeros_like(active),
        )
    phi = local_chebyshev_basis_rowbound_recurrence(
        gram,
        rhs_for_basis,
        steps=steps,
        max_rank=centered_rank_cap,
    ).squeeze(1)
    raw_deterministic = solve_local_ridge(
        gram,
        rhs_for_basis,
        alpha_active,
        phi,
        requested_steps=steps,
    )
    uy = torch.einsum("amk,ak->am", phi, rhs)
    uy = uy / uy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    beta_active_rows = beta.float().reshape(
        batch_queries, feature_count
    )[active_flat]
    beta_abs_active = beta_abs.reshape(
        batch_queries, feature_count
    )[active_flat]
    beta_norm = beta_active_rows.norm(dim=-1)
    deterministic_norm = raw_deterministic.float().norm(dim=-1)
    valid_active = valid_mask.reshape(
        batch_queries, feature_count
    )[active_flat]
    beta_support_fraction = valid_support_fraction(
        beta_abs_active, valid_active, threshold=1e-3
    )
    stats = torch.stack(
        [
            torch.log1p(beta_norm),
            torch.log1p(deterministic_norm),
            torch.log1p(score_selected.mean(dim=-1).clamp_min(0.0)),
            torch.log1p(score_selected.amax(dim=-1).clamp_min(0.0)),
            torch.log1p(lam_active.clamp_min(1e-8)),
            torch.log1p(d0_active.clamp_min(0.0)),
            torch.log1p(alpha_active[:, 0].clamp_min(1e-8)),
            beta_support_fraction,
            torch.full_like(beta_norm, float(stage_frac)),
        ],
        dim=-1,
    )
    stats = torch.nan_to_num(
        stats, nan=0.0, posinf=20.0, neginf=-20.0
    ).clamp(-20.0, 20.0)
    uy_padded = pad_last(uy, int(model.v46_max_local_steps))
    mixer_input = torch.cat([stats, uy_padded], dim=-1)

    coefficients = model.v46_local_mixer(
        mixer_input.to(next(model.v46_local_mixer.parameters()).dtype)
    ).float()
    basis_width = phi.shape[-2]
    coefficients = coefficients[..., :basis_width].clamp(-2.0, 2.0)
    learned_raw = torch.einsum("am,amk->ak", coefficients, phi)
    raw = raw_deterministic + float(model.v46_learned_scale) * learned_raw
    delta_active = soft_threshold(raw, tau)
    delta_selected = X.new_zeros(batch_queries, selected_width)
    delta_selected[active_flat] = delta_active
    correction = X.new_zeros(batch_queries, feature_count)
    correction.scatter_add_(
        -1, indices.reshape(batch_queries, selected_width), delta_selected
    )
    correction = correction.reshape(batch_size, query_count, feature_count)
    correction = torch.nan_to_num(
        correction, nan=0.0, posinf=20.0, neginf=-20.0
    ).clamp(-20.0, 20.0)
    return correction, indices, active
