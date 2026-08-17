from __future__ import annotations

import ast
import inspect
import math
import textwrap
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


def _sep(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ##############################################################################
# SECTION A -- Krylov backbone (grid-evaluated penalty conditioning)
# ##############################################################################


# ═══════════════════════════════════════════════════════════════════════════════
# Penalty registry
# ═══════════════════════════════════════════════════════════════════════════════

PENALTY_REGISTRY = {
    "l1": 0,
    "lasso": 0,
    "l2": 1,
    "ridge": 1,
    "elastic_net": 2,
    "enet": 2,
    "scad": 3,
    "mcp": 4,
}
NUM_PENALTY_TYPES = 5


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _maybe_disable_autocast(device_type: str):
    enabled = False
    if device_type == "cuda":
        enabled = torch.is_autocast_enabled()
    elif device_type == "cpu":
        try:
            enabled = torch.is_autocast_enabled("cpu")
        except TypeError:
            try:
                enabled = torch.is_autocast_cpu_enabled()
            except AttributeError:
                enabled = False
    if enabled and device_type in ("cpu", "cuda"):
        return torch.autocast(device_type=device_type, enabled=False)
    return nullcontext()


def _normalize_feature_mask(
    feature_mask: Optional[Tensor],
    B: int,
    P: int,
    device: torch.device,
) -> Optional[Tensor]:
    if feature_mask is None:
        return None
    if feature_mask.dim() == 1:
        feature_mask = feature_mask.unsqueeze(0).expand(B, P)
    if feature_mask.shape != (B, P):
        raise ValueError(
            f"feature_mask: expected ({B}, {P}), got {tuple(feature_mask.shape)}"
        )
    return feature_mask.to(device=device, dtype=torch.bool)


def _safe_logabs(x: Tensor, eps: float = 1e-12) -> Tensor:
    return torch.log(x.abs().clamp_min(eps))


def _as_2d_queries(
    x: Union[Tensor, float, int],
    B: int,
    name: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    t = torch.as_tensor(x, device=device, dtype=dtype)
    if t.dim() == 0:
        return t.view(1, 1).expand(B, 1)
    if t.dim() == 1:
        if t.shape[0] == B:
            return t.view(B, 1)
        return t.view(1, -1).expand(B, -1)
    if t.dim() == 2:
        if t.shape[0] == 1 and B > 1:
            return t.expand(B, -1)
        if t.shape[0] != B:
            raise ValueError(f"{name}: expected first dim {B}, got {tuple(t.shape)}")
        return t
    raise ValueError(f"{name}: expected scalar, (B,), (Q,), or (B,Q); got {tuple(t.shape)}")


def _broadcast_queries_like(
    x: Union[Tensor, float, int],
    B: int,
    Q: int,
    name: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    t = torch.as_tensor(x, device=device, dtype=dtype)
    if t.dim() == 0:
        return t.view(1, 1).expand(B, Q)
    if t.dim() == 1:
        if t.shape[0] == B:
            return t.view(B, 1).expand(B, Q)
        if t.shape[0] == Q:
            return t.view(1, Q).expand(B, Q)
        if t.shape[0] == 1:
            return t.view(1, 1).expand(B, Q)
    if t.dim() == 2:
        if t.shape == (B, Q):
            return t
        if t.shape == (B, 1):
            return t.expand(B, Q)
        if t.shape == (1, Q):
            return t.expand(B, Q)
        if t.shape == (1, 1):
            return t.expand(B, Q)
    raise ValueError(
        f"{name}: cannot broadcast shape {tuple(t.shape)} to ({B}, {Q})"
    )


def _resolve_penalty_type_grid(
    penalty_type_id: Optional[Union[str, int, Tensor]],
    B: int,
    Q: int,
    device: torch.device,
) -> Tensor:
    if penalty_type_id is None:
        return torch.zeros(B, Q, device=device, dtype=torch.long)

    if isinstance(penalty_type_id, str):
        pid = PENALTY_REGISTRY[penalty_type_id.lower()]
        return torch.full((B, Q), pid, device=device, dtype=torch.long)

    if isinstance(penalty_type_id, int):
        return torch.full((B, Q), penalty_type_id, device=device, dtype=torch.long)

    t = torch.as_tensor(penalty_type_id, device=device)
    if t.dim() == 0:
        return t.to(torch.long).view(1, 1).expand(B, Q)
    if t.dim() == 1:
        if t.shape[0] == B:
            return t.to(torch.long).view(B, 1).expand(B, Q)
        if t.shape[0] == Q:
            return t.to(torch.long).view(1, Q).expand(B, Q)
        if t.shape[0] == 1:
            return t.to(torch.long).view(1, 1).expand(B, Q)
    if t.dim() == 2:
        if t.shape == (B, Q):
            return t.to(torch.long)
        if t.shape == (B, 1):
            return t.to(torch.long).expand(B, Q)
        if t.shape == (1, Q):
            return t.to(torch.long).expand(B, Q)
        if t.shape == (1, 1):
            return t.to(torch.long).expand(B, Q)
    raise ValueError(
        f"penalty_type_id: cannot broadcast shape {tuple(t.shape)} to ({B}, {Q})"
    )


def _resolve_penalty_param_grid(
    penalty_params: Optional[Tensor],
    B: int,
    Q: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    out_dim: int = 2,
) -> Tensor:
    if penalty_params is None:
        return torch.zeros(B, Q, out_dim, device=device, dtype=dtype)

    t = torch.as_tensor(penalty_params, device=device, dtype=dtype)
    if t.dim() == 1:
        if t.shape[0] > out_dim:
            t = t[:out_dim]
        if t.shape[0] < out_dim:
            t = torch.cat([t, t.new_zeros(out_dim - t.shape[0])], dim=0)
        return t.view(1, 1, out_dim).expand(B, Q, out_dim)

    if t.dim() == 2:
        if t.shape[1] > out_dim:
            t = t[:, :out_dim]
        if t.shape[1] < out_dim:
            t = torch.cat([t, t.new_zeros(t.shape[0], out_dim - t.shape[1])], dim=1)
        if t.shape[0] == B:
            return t.view(B, 1, out_dim).expand(B, Q, out_dim)
        if t.shape[0] == Q:
            return t.view(1, Q, out_dim).expand(B, Q, out_dim)
        if t.shape[0] == 1:
            return t.view(1, 1, out_dim).expand(B, Q, out_dim)

    if t.dim() == 3:
        if t.shape[-1] > out_dim:
            t = t[..., :out_dim]
        if t.shape[-1] < out_dim:
            pad = t.new_zeros(*t.shape[:-1], out_dim - t.shape[-1])
            t = torch.cat([t, pad], dim=-1)
        if t.shape == (B, Q, out_dim):
            return t
        if t.shape == (B, 1, out_dim):
            return t.expand(B, Q, out_dim)
        if t.shape == (1, Q, out_dim):
            return t.expand(B, Q, out_dim)
        if t.shape == (1, 1, out_dim):
            return t.expand(B, Q, out_dim)

    raise ValueError(
        f"penalty_params: cannot broadcast shape {tuple(t.shape)} to ({B}, {Q}, {out_dim})"
    )


def center_xy(
    X: Tensor,
    y: Tensor,
    feature_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Center X and y for profiled-intercept sparse regression."""
    if y.dim() == 3 and y.size(-1) == 1:
        y = y.squeeze(-1)

    B, N, P = X.shape
    fm = _normalize_feature_mask(feature_mask, B, P, X.device)

    if fm is not None:
        X = X * fm.unsqueeze(1).to(X.dtype)

    x_mean = X.mean(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)
    Xc = X - x_mean
    yc = y - y_mean

    if fm is not None:
        Xc = Xc * fm.unsqueeze(1).to(Xc.dtype)

    return Xc, yc, x_mean, y_mean


def compute_lambda_pair(
    X: Tensor,
    y: Tensor,
    ratio_range: Tuple[float, float] = (1e-3, 1.0),
    feature_mask: Optional[Tensor] = None,
    eps: float = 1e-12,
) -> Tuple[Tensor, Tensor]:
    """Compute one sampled lambda and lambda_max per task on the centered problem."""
    if y.dim() == 3 and y.size(-1) == 1:
        y = y.squeeze(-1)

    Xc, yc, _, _ = center_xy(X, y, feature_mask=feature_mask)
    B, N, _ = Xc.shape

    with _maybe_disable_autocast(X.device.type):
        Xcf = Xc.float()
        ycf = yc.float()
        Xt_y = torch.bmm(Xcf.transpose(1, 2), ycf.unsqueeze(-1)).squeeze(-1)
        lam_max = (Xt_y.abs().amax(dim=1) / float(max(1, N))).clamp_min(eps)

        lo, hi = float(ratio_range[0]), float(ratio_range[1])
        u = torch.rand(B, device=X.device, dtype=torch.float32)
        lam = torch.exp(
            torch.log(lam_max * lo)
            + u * (torch.log(lam_max * hi) - torch.log(lam_max * lo))
        )
    return lam, lam_max


def recover_intercept(x_mean: Tensor, y_mean: Tensor, beta: Tensor) -> Tensor:
    """Recover the profiled intercept from centered-solve coefficients."""
    xm = x_mean.squeeze(1)
    ym = y_mean.squeeze(1)
    if beta.dim() == 2:
        return ym - (xm * beta).sum(dim=-1)
    if beta.dim() == 3:
        return ym.unsqueeze(1) - (xm.unsqueeze(1) * beta).sum(dim=-1)
    raise ValueError(f"beta must have dim 2 or 3, got {beta.dim()}")


def _sample_to_feature_projection(Xc: Tensor, v: Tensor) -> Tensor:
    """X^T v / N in fp32."""
    B, N, _ = Xc.shape
    return torch.einsum("bnp,bn->bp", Xc, v) / float(max(1, N))


def _row_gram_apply(Xc: Tensor, v: Tensor) -> Tensor:
    """G(v) = X (X^T v / N) in fp32 without forming X^T X."""
    proj = _sample_to_feature_projection(Xc, v)
    return torch.einsum("bnp,bp->bn", Xc, proj)


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KrylovQueryDecoderConfig:
    # Exact preprocessing
    cond_eps: float = 1e-12
    diag_eps: float = 1e-8

    # Query conditioning
    condition_np: bool = True
    cond_dim: int = 24
    cond_hidden: int = 64
    penalty_embed_dim: int = 8

    # Krylov encoder
    krylov_dim: int = 32                # FPK v2: was 16, raised per 이현지 sweep
    krylov_reorth_passes: int = 2
    krylov_eps: float = 1e-7

    # Decoder
    num_decoder_channels: int = 12
    use_bilinear_channels: bool = True
    query_mixer_hidden: int = 64
    query_mix_scale: float = 2.0
    feature_head_hidden: int = 512      # FPK v2: was 128, raised per 교수님 명시
    tau_scale_bound: float = 2.0

    # Sparse head
    prox_hidden: int = 64
    prox_mode: str = "learned"  # learned | soft_threshold
    zero_init_query_mixer: bool = True
    zero_init_feature_head: bool = True
    zero_init_learned_prox: bool = True

    # Optional rank-aware SLOPE-style prox. Disabled by default.
    use_slope_prox: bool = False
    slope_calibrated: bool = True
    slope_hidden: int = 64
    slope_iso_mode: str = "cummin"  # "cummin", "pava", or "none"
    slope_eps: float = 1e-8
    slope_return_order: bool = False
    slope_rho_mode: str = "learned"  # "learned" or "fixed"
    slope_fixed_rho: float = 0.0
    slope_rho_init_logit: float = -5.0
    slope_log_scale_mode: str = "learned"  # "learned" or "fixed_zero"
    slope_log_scale_min: float = -3.0
    slope_log_scale_max: float = 3.0
    slope_curve_weight: float = 0.0
    slope_mean_log_scale_weight: float = 0.0
    slope_entropy_weight: float = 0.0

    # Correlation-aware one-shot base
    use_krylov_ridge_base: bool = True
    base_mode: str = "blend"  # blend | krylov_only | diag_only
    base_head_hidden: int = 64
    base_alpha_log_min: float = -6.0
    base_alpha_log_max: float = 3.0

    # Encoding cache
    retain_U: bool = True
    retain_centered_data_in_encoding: bool = False

    # Optional progressive mode
    prefix_delta: int = 50


# ═══════════════════════════════════════════════════════════════════════════════
# Conditioning
# ═══════════════════════════════════════════════════════════════════════════════

class PenaltyAwareCondEmbed(nn.Module):
    """(lambda, lambda_max, N, P, penalty_type, penalty_params) -> (B, Q, d)."""

    def __init__(
        self,
        out_dim: int,
        hidden: int = 64,
        eps: float = 1e-12,
        use_np: bool = True,
        penalty_embed_dim: int = 8,
    ):
        super().__init__()
        self.eps = eps
        self.use_np = use_np
        self.pen_embed = nn.Embedding(NUM_PENALTY_TYPES, penalty_embed_dim)
        numeric_dim = 4 + (4 if use_np else 0) + 2
        self.mlp = nn.Sequential(
            nn.Linear(numeric_dim + penalty_embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(
        self,
        lambdas: Tensor,
        lambda_max: Tensor,
        N: Optional[Union[int, float, Tensor]] = None,
        P: Optional[Union[int, float, Tensor]] = None,
        penalty_type_id: Optional[Union[str, int, Tensor]] = None,
        penalty_params: Optional[Tensor] = None,
    ) -> Tensor:
        if lambdas.dim() != 2:
            raise ValueError(f"lambdas must have shape (B,Q), got {tuple(lambdas.shape)}")
        if lambda_max.shape != lambdas.shape:
            raise ValueError(
                f"lambda_max must match lambdas shape, got {tuple(lambda_max.shape)} vs {tuple(lambdas.shape)}"
            )

        device = self.mlp[0].weight.device
        dtype = self.mlp[0].weight.dtype
        eps = self.eps
        B, Q = lambdas.shape

        lam = lambdas.to(device=device, dtype=torch.float32).clamp_min(eps)
        lmx = lambda_max.to(device=device, dtype=torch.float32).clamp_min(eps)
        ratio = (lam / lmx).clamp_min(eps)

        parts = [
            torch.stack([lam.log(), lmx.log(), ratio.log(), ratio], dim=-1)
        ]

        if self.use_np:
            if N is None:
                Nt = torch.full((B, Q), 1.0, device=device, dtype=torch.float32)
            else:
                Nt = _broadcast_queries_like(N, B, Q, "N", device=device, dtype=torch.float32).clamp_min(1.0)
            if P is None:
                Pt = torch.full((B, Q), 1.0, device=device, dtype=torch.float32)
            else:
                Pt = _broadcast_queries_like(P, B, Q, "P", device=device, dtype=torch.float32).clamp_min(1.0)
            npr = (Nt / Pt).clamp_min(eps)
            parts.append(torch.stack([Nt.log(), Pt.log(), npr.log(), npr], dim=-1))

        pp = _resolve_penalty_param_grid(
            penalty_params=penalty_params,
            B=B,
            Q=Q,
            device=device,
            dtype=torch.float32,
            out_dim=2,
        )
        parts.append(pp)

        pid = _resolve_penalty_type_grid(
            penalty_type_id=penalty_type_id,
            B=B,
            Q=Q,
            device=device,
        )

        numeric = torch.cat(parts, dim=-1)
        pen = self.pen_embed(pid)
        feat = torch.cat([numeric, pen], dim=-1).to(dtype)
        out = self.mlp(feat.view(B * Q, -1)).view(B, Q, -1)
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# Sparse output head
# ═══════════════════════════════════════════════════════════════════════════════

class LearnedProxOp(nn.Module):
    """Penalty-conditioned shrinkage.

    With zero init, this starts from standard soft-thresholding.
    """

    def __init__(self, cond_dim: int, hidden: int = 64, zero_init: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 + cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, u_hat: Tensor, tau_hat: Tensor, cond: Tensor) -> Tensor:
        if u_hat.dim() != 2 or tau_hat.dim() != 2 or cond.dim() != 2:
            raise ValueError("LearnedProxOp expects (BQ,P), (BQ,P), (BQ,D)")
        BQ, P = u_hat.shape
        tau = tau_hat.expand(BQ, P).clamp_min(1e-12)
        au = u_hat.abs()
        cond_feat = cond.unsqueeze(1).expand(BQ, P, -1)
        tok = torch.cat(
            [au.unsqueeze(-1), tau.unsqueeze(-1), torch.zeros_like(au).unsqueeze(-1), cond_feat],
            dim=-1,
        )
        raw = self.net(tok)
        scale = raw[..., 0].clamp(-2.0, 2.0)
        shift = raw[..., 1].clamp(-2.0, 2.0)
        return torch.sign(u_hat) * torch.relu(au * (1.0 + scale) - tau * (1.0 + shift))


def isotonic_nonnegative_decreasing(x: Tensor, mode: str) -> Tensor:
    """Project/approximate x onto nonnegative decreasing sequences along last dim."""
    mode_l = str(mode).lower()
    if mode_l == "none":
        return torch.relu(x)
    if mode_l == "cummin":
        y = torch.relu(x)
        return torch.cummin(y, dim=-1).values
    if mode_l != "pava":
        raise ValueError(f"slope_iso_mode must be 'cummin', 'pava', or 'none'; got {mode!r}")

    P = x.shape[-1]
    if x.is_cuda and P > 512:
        raise RuntimeError("Use slope_iso_mode='cummin' for CUDA training with large P.")

    y_flat = torch.relu(x).reshape(-1, P)
    rows: List[Tensor] = []
    for row in y_flat:
        levels: List[Tensor] = []
        counts: List[int] = []
        for j in range(P):
            levels.append(row[j])
            counts.append(1)
            while len(levels) >= 2 and bool((levels[-2] < levels[-1]).detach()):
                c1 = counts[-2]
                c2 = counts[-1]
                merged = (levels[-2] * c1 + levels[-1] * c2) / float(c1 + c2)
                levels[-2] = merged
                counts[-2] = c1 + c2
                levels.pop()
                counts.pop()
        rows.append(torch.cat([level.expand(count) for level, count in zip(levels, counts)], dim=0))
    return torch.stack(rows, dim=0).reshape_as(x)


class RankScheduleNet(nn.Module):
    """Learn a positive rank-dependent shrinkage schedule for sorted coefficients."""

    def __init__(
        self,
        cond_dim: int,
        hidden: int = 64,
        rho_init_logit: float = -5.0,
        rho_mode: str = "learned",
        fixed_rho: float = 0.0,
        log_scale_mode: str = "learned",
        log_scale_min: float = -3.0,
        log_scale_max: float = 3.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.eps = float(eps)
        self.rho_mode = str(rho_mode)
        self.fixed_rho = float(fixed_rho)
        self.log_scale_mode = str(log_scale_mode)
        self.log_scale_min = float(log_scale_min)
        self.log_scale_max = float(log_scale_max)
        if self.rho_mode not in {"learned", "fixed"}:
            raise ValueError(f"slope_rho_mode must be 'learned' or 'fixed'; got {rho_mode!r}")
        if self.log_scale_mode not in {"learned", "fixed_zero"}:
            raise ValueError(
                f"slope_log_scale_mode must be 'learned' or 'fixed_zero'; got {log_scale_mode!r}"
            )
        self.rank_net = nn.Sequential(
            nn.LayerNorm(cond_dim + 3),
            nn.Linear(cond_dim + 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.query_head = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.rank_net[-1].weight)
        nn.init.zeros_(self.rank_net[-1].bias)
        nn.init.zeros_(self.query_head[-1].weight)
        nn.init.zeros_(self.query_head[-1].bias)
        with torch.no_grad():
            self.query_head[-1].bias[1] = float(rho_init_logit)

    def forward(self, cond: Tensor, P: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        if cond.dim() != 3:
            raise ValueError(f"cond must have shape (B,Q,D), got {tuple(cond.shape)}")
        B, Q, D = cond.shape
        device = cond.device
        dtype = cond.dtype
        eps = self.eps

        r = torch.linspace(1.0 / float(P), 1.0, int(P), device=device, dtype=torch.float32)
        r_safe = r.clamp_min(eps)
        one_minus_r = (1.0 - r).clamp_min(eps)
        rank_feat = torch.stack([r, r_safe.log(), one_minus_r.log()], dim=-1).to(dtype)
        rank_feat = rank_feat.view(1, 1, P, 3).expand(B, Q, P, 3)
        cond_feat = cond[:, :, None, :].expand(B, Q, P, D)
        token = torch.cat([rank_feat, cond_feat], dim=-1)

        raw_inc = self.rank_net(token).squeeze(-1)
        inc = torch.nn.functional.softplus(raw_inc) + eps
        mono = torch.flip(torch.cumsum(torch.flip(inc, dims=(-1,)), dim=-1), dims=(-1,))
        mono = mono / mono.mean(dim=-1, keepdim=True).clamp_min(eps)

        raw_query = self.query_head(cond)
        if self.log_scale_mode == "fixed_zero":
            log_scale = raw_query[..., 0].detach().new_zeros(raw_query[..., 0].shape)
        else:
            log_scale = raw_query[..., 0].clamp(self.log_scale_min, self.log_scale_max)
        if self.rho_mode == "fixed":
            rho = raw_query[..., 1].detach().new_full(raw_query[..., 1].shape, self.fixed_rho)
        else:
            rho = torch.sigmoid(raw_query[..., 1])
        rank_scale = log_scale.exp().unsqueeze(-1) * ((1.0 - rho).unsqueeze(-1) + rho.unsqueeze(-1) * mono)
        aux = {
            "slope_rho_q": rho,
            "slope_log_scale_q": log_scale,
            "slope_mono_shape_q": mono,
            "slope_rank_scale_q": rank_scale,
            "slope_mean_log_rank_scale_q": torch.log(rank_scale.clamp_min(eps)).mean(dim=-1),
            "slope_rank_scale_first_q": rank_scale[..., 0],
            "slope_rank_scale_last_q": rank_scale[..., -1],
        }
        return rank_scale, aux


class LearnedSLOPEProx(nn.Module):
    """Rank-aware sorted shrinkage layer with an independent-prox initialization."""

    def __init__(
        self,
        cond_dim: int,
        hidden: int = 64,
        slope_calibrated: bool = True,
        slope_iso_mode: str = "cummin",
        slope_rho_mode: str = "learned",
        slope_fixed_rho: float = 0.0,
        slope_rho_init_logit: float = -5.0,
        slope_log_scale_mode: str = "learned",
        slope_log_scale_min: float = -3.0,
        slope_log_scale_max: float = 3.0,
        slope_eps: float = 1e-8,
        slope_return_order: bool = False,
    ):
        super().__init__()
        self.slope_calibrated = bool(slope_calibrated)
        self.slope_iso_mode = str(slope_iso_mode)
        self.slope_eps = float(slope_eps)
        self.slope_return_order = bool(slope_return_order)
        self.rank_schedule = RankScheduleNet(
            cond_dim=cond_dim,
            hidden=hidden,
            rho_init_logit=slope_rho_init_logit,
            rho_mode=slope_rho_mode,
            fixed_rho=slope_fixed_rho,
            log_scale_mode=slope_log_scale_mode,
            log_scale_min=slope_log_scale_min,
            log_scale_max=slope_log_scale_max,
            eps=slope_eps,
        )

    def forward(
        self,
        u_hat: Tensor,
        tau_local: Tensor,
        gram_diag: Tensor,
        cond: Tensor,
        lambdas: Tensor,
        return_aux: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        if u_hat.dim() != 3:
            raise ValueError(f"u_hat must have shape (B,Q,P), got {tuple(u_hat.shape)}")
        if tau_local.shape != u_hat.shape:
            raise ValueError(f"tau_local must match u_hat shape, got {tuple(tau_local.shape)} vs {tuple(u_hat.shape)}")
        B, Q, P = u_hat.shape
        if gram_diag.shape != (B, P):
            raise ValueError(f"gram_diag must have shape ({B},{P}), got {tuple(gram_diag.shape)}")
        if cond.shape[:2] != (B, Q):
            raise ValueError(f"cond first dims must be ({B},{Q}), got {tuple(cond.shape)}")

        work_dtype = u_hat.dtype
        sqrt_d = gram_diag.to(device=u_hat.device, dtype=work_dtype).clamp_min(self.slope_eps).sqrt()

        if self.slope_calibrated:
            u_std = u_hat * sqrt_d[:, None, :]
            tau_std = tau_local * sqrt_d[:, None, :]
            a, order = torch.sort(u_std.abs(), dim=-1, descending=True)
            sign_sorted = torch.gather(torch.sign(u_std), dim=-1, index=order)
            base_tau_sorted = torch.gather(tau_std, dim=-1, index=order)
        else:
            u_std = u_hat
            a, order = torch.sort(u_std.abs(), dim=-1, descending=True)
            sign_sorted = torch.gather(torch.sign(u_std), dim=-1, index=order)
            base_tau_sorted = lambdas.to(device=u_hat.device, dtype=work_dtype).unsqueeze(-1).expand_as(a)

        rank_scale, sched_aux = self.rank_schedule(cond, P)
        rank_scale = rank_scale.to(device=u_hat.device, dtype=work_dtype)
        tau_sorted = base_tau_sorted * rank_scale
        raw_mag = a - tau_sorted

        mag_ind = torch.relu(raw_mag)
        mag_iso = isotonic_nonnegative_decreasing(raw_mag, mode=self.slope_iso_mode).to(dtype=work_dtype)
        rho = sched_aux["slope_rho_q"].to(device=u_hat.device, dtype=work_dtype)
        mag_sorted = (
            (1.0 - rho).unsqueeze(-1) * mag_ind.to(dtype=work_dtype)
            + rho.unsqueeze(-1) * mag_iso
        ).to(dtype=work_dtype)

        theta_sorted = (sign_sorted * mag_sorted).to(dtype=u_std.dtype)
        theta = torch.zeros_like(u_std).scatter(dim=-1, index=order, src=theta_sorted)
        beta = theta / sqrt_d[:, None, :] if self.slope_calibrated else theta

        if not return_aux:
            return beta

        aux: Dict[str, Tensor] = {
            "slope_tau_sorted_q": tau_sorted,
            "slope_rank_scale_q": rank_scale,
            "slope_rho_q": sched_aux["slope_rho_q"],
            "slope_log_scale_q": sched_aux["slope_log_scale_q"],
            "slope_mono_shape_q": sched_aux["slope_mono_shape_q"],
            "slope_mean_log_rank_scale_q": sched_aux["slope_mean_log_rank_scale_q"],
            "slope_rank_scale_first_q": sched_aux["slope_rank_scale_first_q"],
            "slope_rank_scale_last_q": sched_aux["slope_rank_scale_last_q"],
            "slope_mag_sorted_q": mag_sorted,
            "slope_effective_nnz_q": (beta.abs() > 1e-4).to(beta.dtype).sum(dim=-1),
        }
        if self.slope_return_order:
            aux["slope_order_q"] = order
        return beta, aux


def _aux_zero(aux: Mapping[str, Tensor]) -> Tensor:
    for value in aux.values():
        if torch.is_tensor(value):
            return value.new_zeros(())
    return torch.zeros(())


def slope_curve_regularizer(aux: Mapping[str, Tensor], weight: float = 1.0) -> Tensor:
    """Penalize curvature of the learned log rank schedule when SLOPE aux exists."""
    if weight <= 0.0 or "slope_rank_scale_q" not in aux:
        return _aux_zero(aux)
    rank_scale = aux["slope_rank_scale_q"]
    if rank_scale.shape[-1] < 3:
        return rank_scale.new_zeros(())
    log_s = torch.log(rank_scale.clamp_min(1e-12))
    d2 = log_s[..., 2:] - 2.0 * log_s[..., 1:-1] + log_s[..., :-2]
    return float(weight) * d2.square().mean()


def slope_mean_log_scale_regularizer(aux: Mapping[str, Tensor], weight: float = 1.0) -> Tensor:
    """Penalize global rank-scale shifts so shape cannot hide as threshold rescaling."""
    if weight <= 0.0 or "slope_rank_scale_q" not in aux:
        return _aux_zero(aux)
    rank_scale = aux["slope_rank_scale_q"]
    log_s = torch.log(rank_scale.clamp_min(1e-12))
    mean_log = log_s.mean(dim=-1)
    return float(weight) * mean_log.square().mean()


def slope_entropy_regularizer(aux: Mapping[str, Tensor], weight: float = 1.0) -> Tensor:
    """Optional concentration penalty on normalized rank scales; disabled by default."""
    if weight <= 0.0 or "slope_rank_scale_q" not in aux:
        return _aux_zero(aux)
    rank_scale = aux["slope_rank_scale_q"].clamp_min(1e-12)
    prob = rank_scale / rank_scale.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    entropy = -(prob * prob.log()).sum(dim=-1)
    max_entropy = math.log(float(rank_scale.shape[-1]))
    flatness = entropy / max(max_entropy, 1e-12)
    return float(weight) * flatness.mean()


def slope_rank_curve_regularizer(aux: Mapping[str, Tensor], weight: float = 1.0) -> Tensor:
    """Backward-compatible alias for the first SLOPE experiment."""
    return slope_curve_regularizer(aux, weight=weight)


# ═══════════════════════════════════════════════════════════════════════════════
# Krylov encoder
# ═══════════════════════════════════════════════════════════════════════════════

class KrylovBasisBuilder(nn.Module):
    """Parameter-free Krylov/Lanczos-style basis builder in sample space."""

    def __init__(
        self,
        krylov_dim: int = 8,
        reorth_passes: int = 2,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.krylov_dim = krylov_dim
        self.reorth_passes = reorth_passes
        self.eps = eps

    @torch.no_grad()
    def forward(self, Xc: Tensor, yc: Tensor) -> Dict[str, Tensor]:
        with _maybe_disable_autocast(Xc.device.type):
            Xf = Xc.detach().float()
            yf = yc.detach().float()

            B, N, P = Xf.shape
            M = self.krylov_dim

            U = Xf.new_zeros(B, M, N)
            basis_proj = Xf.new_zeros(B, M, P)
            valid = torch.zeros(B, M, device=Xf.device, dtype=torch.bool)
            basis_norms = Xf.new_zeros(B, M)
            candidate_norms = Xf.new_zeros(B, M)

            seed = yf
            seed_norm = seed.norm(dim=-1, keepdim=True)
            fallback = Xf.square().sum(dim=-1).sqrt()
            fallback = fallback - fallback.mean(dim=-1, keepdim=True)
            fallback_norm = fallback.norm(dim=-1, keepdim=True)

            use_fallback = (seed_norm <= self.eps) & (fallback_norm > self.eps)
            seed = torch.where(use_fallback, fallback, seed)
            seed_norm = seed.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            q = seed / seed_norm

            for m in range(M):
                q_norm = q.norm(dim=-1)
                is_valid = q_norm > self.eps
                valid[:, m] = is_valid
                q = torch.where(
                    is_valid.unsqueeze(-1),
                    q / q_norm.clamp_min(self.eps).unsqueeze(-1),
                    torch.zeros_like(q),
                )

                U[:, m, :] = q
                basis_norms[:, m] = q.norm(dim=-1)

                proj = _sample_to_feature_projection(Xf, q)
                basis_proj[:, m, :] = proj

                if m + 1 < M:
                    w = torch.einsum("bnp,bp->bn", Xf, proj)
                    used = U[:, : m + 1, :]
                    for _ in range(max(1, self.reorth_passes)):
                        coeff = torch.einsum("bkn,bn->bk", used, w)
                        w = w - torch.einsum("bkn,bk->bn", used, coeff)
                    cand_norm = w.norm(dim=-1)
                    candidate_norms[:, m] = cand_norm
                    q = torch.where(
                        (cand_norm > self.eps).unsqueeze(-1),
                        w / cand_norm.clamp_min(self.eps).unsqueeze(-1),
                        torch.zeros_like(w),
                    )

            gram = torch.einsum("bmn,bkn->bmk", U, U)
            eye = torch.eye(M, device=Xf.device, dtype=gram.dtype).unsqueeze(0).expand(B, -1, -1)
            valid_pair = (valid.unsqueeze(1) & valid.unsqueeze(2)).to(gram.dtype)
            diff = (gram - eye) * valid_pair
            offdiag = diff - torch.diag_embed(torch.diagonal(diff, dim1=1, dim2=2))

            orth_error = diff.abs().amax(dim=(1, 2))
            offdiag_error = offdiag.abs().amax(dim=(1, 2))
            diag_error = (
                (torch.diagonal(gram, dim1=1, dim2=2) - 1.0) * valid.to(gram.dtype)
            ).abs().amax(dim=1)

            Uy = torch.einsum("bmn,bn->bm", U, yf)  # (B, M)

            # T = U^T (X X^T / N) U  =  (X^T U)^T (X^T U) / N
            T_krylov = float(max(1, N)) * torch.einsum("bmp,bkp->bmk", basis_proj, basis_proj)
            T_krylov = 0.5 * (T_krylov + T_krylov.transpose(1, 2))

            return {
                "U": U,
                "basis_proj": basis_proj,
                "Uy": Uy,
                "T_krylov": T_krylov,
                "diagnostics": {
                    "basis_norms": basis_norms,
                    "candidate_norms": candidate_norms,
                    "valid_mask": valid,
                    "orthogonality_error": orth_error,
                    "offdiag_error": offdiag_error,
                    "diag_error": diag_error,
                },
            }


# ═══════════════════════════════════════════════════════════════════════════════
# Decoder
# ═══════════════════════════════════════════════════════════════════════════════

class QueryMixer(nn.Module):
    """Maps each query embedding to C signed mixtures over M basis vectors."""

    def __init__(
        self,
        cond_dim: int,
        num_channels: int,
        krylov_dim: int,
        hidden: int = 64,
        mix_scale: float = 2.0,
        zero_init: bool = True,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.krylov_dim = krylov_dim
        self.mix_scale = mix_scale
        self.net = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_channels * krylov_dim),
        )
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, cond: Tensor) -> Tensor:
        if cond.dim() != 3:
            raise ValueError(f"cond must have shape (B,Q,D), got {tuple(cond.shape)}")
        B, Q, _ = cond.shape
        raw = self.net(cond).view(B, Q, self.num_channels, self.krylov_dim)
        return self.mix_scale * torch.tanh(raw)


class SharedFeatureEmissionHead(nn.Module):
    """Shared per-feature head.

    Optionally augments the channel features with bilinear cross terms.
    """

    def __init__(
        self,
        num_channels: int,
        cond_dim: int,
        hidden: int = 128,
        tau_scale_bound: float = 2.0,
        zero_init: bool = True,
        use_bilinear: bool = True,
    ):
        super().__init__()
        self.tau_scale_bound = tau_scale_bound
        self.use_bilinear = use_bilinear
        self.num_channels = num_channels

        # Upper-triangle indices for bilinear features
        if use_bilinear:
            triu_row, triu_col = torch.triu_indices(num_channels, num_channels)
            self.register_buffer("_triu_row", triu_row)
            self.register_buffer("_triu_col", triu_col)
            bilinear_dim = triu_row.shape[0]   # C*(C+1)/2
            channel_feat_dim = num_channels + bilinear_dim
        else:
            bilinear_dim = 0
            channel_feat_dim = num_channels

        in_dim = channel_feat_dim + 3 + cond_dim
        self.backbone = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.out = nn.Linear(hidden, 2)
        if zero_init:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(
        self,
        S: Tensor,
        Xty: Tensor,
        gram_diag: Tensor,
        log_abs_Xty: Tensor,
        cond: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if S.dim() != 4:
            raise ValueError(f"S must have shape (B,Q,C,P), got {tuple(S.shape)}")
        B, Q, C, P = S.shape

        s_flat = S.permute(0, 1, 3, 2)  # (B, Q, P, C)

        if self.use_bilinear:
            # Upper-triangle pairwise products: C*(C+1)/2 features
            s_i = s_flat[..., self._triu_row]   # (B, Q, P, C*(C+1)/2)
            s_j = s_flat[..., self._triu_col]
            s_bilinear = s_i * s_j
            s_feat = torch.cat([s_flat, s_bilinear], dim=-1)
        else:
            s_feat = s_flat

        x_feat = Xty[:, None, :, None].expand(B, Q, P, 1)
        g_feat = gram_diag[:, None, :, None].expand(B, Q, P, 1)
        l_feat = log_abs_Xty[:, None, :, None].expand(B, Q, P, 1)
        c_feat = cond[:, :, None, :].expand(B, Q, P, -1)

        token = torch.cat([s_feat, x_feat, g_feat, l_feat, c_feat], dim=-1)
        raw = self.out(self.backbone(token.reshape(B * Q, P, -1))).view(B, Q, P, 2)

        u_resid = raw[..., 0]
        log_tau_scale = raw[..., 1].clamp(-self.tau_scale_bound, self.tau_scale_bound)
        return u_resid, log_tau_scale




class CorrelationAwareBaseHead(nn.Module):
    """Query-conditioned mixer for the diagonal base and Krylov-ridge base."""

    def __init__(
        self,
        cond_dim: int,
        hidden: int = 64,
        zero_init: bool = True,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        # bias[1] controls the initial ridge/diag mixture
        with torch.no_grad():
            self.net[-1].bias[1] = 0.5

    def forward(self, cond: Tensor) -> Tensor:
        if cond.dim() != 3:
            raise ValueError(f"cond must have shape (B,Q,D), got {tuple(cond.shape)}")
        return self.net(cond)

# ═══════════════════════════════════════════════════════════════════════════════
# Main model
# ═══════════════════════════════════════════════════════════════════════════════

class KrylovRowSpaceQueryDecoder(nn.Module):
    """One-shot sparse regression model with a Krylov row-space query decoder."""

    def __init__(self, cfg: KrylovQueryDecoderConfig):
        super().__init__()
        self.cfg = cfg

        self.cond_embed = PenaltyAwareCondEmbed(
            out_dim=cfg.cond_dim,
            hidden=cfg.cond_hidden,
            eps=cfg.cond_eps,
            use_np=cfg.condition_np,
            penalty_embed_dim=cfg.penalty_embed_dim,
        )

        self.basis_builder = KrylovBasisBuilder(
            krylov_dim=cfg.krylov_dim,
            reorth_passes=cfg.krylov_reorth_passes,
            eps=cfg.krylov_eps,
        )

        self.query_mixer = QueryMixer(
            cond_dim=cfg.cond_dim,
            num_channels=cfg.num_decoder_channels,
            krylov_dim=cfg.krylov_dim,
            hidden=cfg.query_mixer_hidden,
            mix_scale=cfg.query_mix_scale,
            zero_init=cfg.zero_init_query_mixer,
        )

        self.feature_head = SharedFeatureEmissionHead(
            num_channels=cfg.num_decoder_channels,
            cond_dim=cfg.cond_dim,
            hidden=cfg.feature_head_hidden,
            tau_scale_bound=cfg.tau_scale_bound,
            zero_init=cfg.zero_init_feature_head,
            use_bilinear=cfg.use_bilinear_channels,
        )

        self.base_head = CorrelationAwareBaseHead(
            cond_dim=cfg.cond_dim,
            hidden=cfg.base_head_hidden,
            zero_init=True,
        )

        self.learned_prox = LearnedProxOp(
            cond_dim=cfg.cond_dim,
            hidden=cfg.prox_hidden,
            zero_init=cfg.zero_init_learned_prox,
        )
        if getattr(cfg, "use_slope_prox", False):
            self.learned_slope_prox = LearnedSLOPEProx(
                cond_dim=cfg.cond_dim,
                hidden=cfg.slope_hidden,
                slope_calibrated=cfg.slope_calibrated,
                slope_iso_mode=cfg.slope_iso_mode,
                slope_rho_mode=cfg.slope_rho_mode,
                slope_fixed_rho=cfg.slope_fixed_rho,
                slope_rho_init_logit=cfg.slope_rho_init_logit,
                slope_log_scale_mode=cfg.slope_log_scale_mode,
                slope_log_scale_min=cfg.slope_log_scale_min,
                slope_log_scale_max=cfg.slope_log_scale_max,
                slope_eps=cfg.slope_eps,
                slope_return_order=cfg.slope_return_order,
            )

    def precompute(
        self,
        X: Tensor,
        y: Tensor,
        feature_mask: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Exact preprocessing. No X^T X is formed."""
        if y.dim() == 3 and y.size(-1) == 1:
            y = y.squeeze(-1)

        B, N, P = X.shape
        fm = _normalize_feature_mask(feature_mask, B, P, X.device)

        Xc, yc, x_mean, y_mean = center_xy(X, y, feature_mask=fm)

        with _maybe_disable_autocast(X.device.type):
            Xcf = Xc.float()
            ycf = yc.float()
            Xty = torch.bmm(Xcf.transpose(1, 2), ycf.unsqueeze(-1)).squeeze(-1) / float(max(1, N))
            gram_diag = Xcf.square().mean(dim=1)
            lambda_max = Xty.abs().amax(dim=1).clamp_min(self.cfg.cond_eps)
            log_abs_Xty = _safe_logabs(Xty, self.cfg.cond_eps)

        return {
            "Xc": Xc,
            "yc": yc,
            "x_mean": x_mean.float(),
            "y_mean": y_mean.float(),
            "Xty": Xty,
            "gram_diag": gram_diag,
            "log_abs_Xty": log_abs_Xty,
            "lambda_max": lambda_max,
            "feature_mask": fm,
            "B": B,
            "N": N,
            "P": P,
        }

    def encode_dataset(
        self,
        precomputed: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Encode one dataset once into a compact row-space representation."""
        basis = self.basis_builder(precomputed["Xc"], precomputed["yc"])

        encoded = {
            "Xty": precomputed["Xty"],
            "gram_diag": precomputed["gram_diag"],
            "log_abs_Xty": precomputed["log_abs_Xty"],
            "lambda_max": precomputed["lambda_max"],
            "x_mean": precomputed["x_mean"],
            "y_mean": precomputed["y_mean"],
            "feature_mask": precomputed["feature_mask"],
            "basis_proj": basis["basis_proj"],
            "Uy_krylov": basis["Uy"],
            "T_krylov": basis["T_krylov"],
            "basis_diagnostics": basis["diagnostics"],
            "B": precomputed["B"],
            "N": precomputed["N"],
            "P": precomputed["P"],
        }
        if self.cfg.retain_U:
            encoded["U"] = basis["U"]
        if self.cfg.retain_centered_data_in_encoding:
            encoded["Xc"] = precomputed["Xc"]
            encoded["yc"] = precomputed["yc"]
        return encoded

    def decode_queries(
        self,
        encoded: Dict[str, Any],
        lambdas: Tensor,
        lambda_max: Optional[Tensor] = None,
        penalty_type_id: Optional[Union[str, int, Tensor]] = None,
        penalty_params: Optional[Tensor] = None,
        return_aux: bool = False,
        return_row_state: bool = False,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """Decode one or many queries from an encoded dataset."""
        B = encoded["B"]
        P = encoded["P"]
        N = encoded["N"]
        device = encoded["Xty"].device

        lam = _as_2d_queries(lambdas, B, "lambdas", device=device, dtype=torch.float32)
        B_check, Q = lam.shape
        if B_check != B:
            raise ValueError(f"lambdas batch mismatch: expected {B}, got {B_check}")

        lmx_source = encoded["lambda_max"] if lambda_max is None else lambda_max
        lmx = _broadcast_queries_like(
            lmx_source, B, Q, "lambda_max", device=device, dtype=torch.float32
        )

        cond = self.cond_embed(
            lambdas=lam,
            lambda_max=lmx,
            N=float(N),
            P=float(P),
            penalty_type_id=penalty_type_id,
            penalty_params=penalty_params,
        )

        A = self.query_mixer(cond)
        with _maybe_disable_autocast(device.type):
            basis_proj_f = encoded["basis_proj"].float()
            S = torch.einsum(
                "bqcm,bmp->bqcp",
                A.float(),
                basis_proj_f,
            )

        model_dtype = cond.dtype
        S = S.to(model_dtype)
        Xty = encoded["Xty"].to(model_dtype)
        gram_diag = encoded["gram_diag"].to(model_dtype)
        log_abs_Xty = encoded["log_abs_Xty"].to(model_dtype)

        u_resid, log_tau_scale = self.feature_head(
            S=S,
            Xty=Xty,
            gram_diag=gram_diag,
            log_abs_Xty=log_abs_Xty,
            cond=cond,
        )

        denom = gram_diag.clamp_min(self.cfg.diag_eps)
        u_diag = Xty[:, None, :] / denom[:, None, :]
        tau_base = lam.to(model_dtype).unsqueeze(-1) / denom[:, None, :]

        base_mode = getattr(self.cfg, "base_mode", "blend")
        if not self.cfg.use_krylov_ridge_base:
            base_mode = "diag_only"
        if base_mode not in ("blend", "krylov_only", "diag_only"):
            raise ValueError(
                f"base_mode must be 'blend', 'krylov_only', or 'diag_only'; got {base_mode!r}"
            )

        if base_mode in ("blend", "krylov_only"):
            base_raw = self.base_head(cond)
            log_alpha_scale = base_raw[..., 0].clamp(
                self.cfg.base_alpha_log_min,
                self.cfg.base_alpha_log_max,
            )

            alpha = lam * torch.exp(log_alpha_scale) + self.cfg.diag_eps

            with _maybe_disable_autocast(device.type):
                T = encoded["T_krylov"].float()
                Uy = encoded["Uy_krylov"].float()
                Phi = encoded["basis_proj"].float()
                M = T.shape[-1]

                eye = torch.eye(M, device=device, dtype=torch.float32).view(1, 1, M, M)
                rhs = Uy[:, None, :, None].expand(B, Q, M, 1)
                sys = T[:, None, :, :] + alpha[..., None, None].float() * eye + self.cfg.diag_eps * eye

                coeff = torch.linalg.solve(sys, rhs).squeeze(-1)
                u_krylov = torch.einsum("bqm,bmp->bqp", coeff, Phi)

            if base_mode == "blend":
                mix = torch.sigmoid(base_raw[..., 1])
                u_base = mix.unsqueeze(-1) * u_krylov.to(model_dtype) + (1.0 - mix.unsqueeze(-1)) * u_diag
            else:
                mix = torch.ones(B, Q, device=device, dtype=model_dtype)
                u_base = u_krylov.to(model_dtype)
        else:
            mix = torch.zeros(B, Q, device=device, dtype=model_dtype)
            alpha = lam.new_full((B, Q), float("nan"))
            u_base = u_diag

        u_hat = u_base + u_resid
        tau_hat = tau_base * torch.exp(log_tau_scale)

        slope_aux: Dict[str, Tensor] = {}
        if getattr(self.cfg, "use_slope_prox", False):
            beta, slope_aux = self.learned_slope_prox(
                u_hat=u_hat,
                tau_local=tau_hat,
                gram_diag=gram_diag,
                cond=cond,
                lambdas=lam.to(model_dtype),
                return_aux=True,
            )
        else:
            prox_mode = getattr(self.cfg, "prox_mode", "learned")
            if prox_mode == "learned":
                beta = self.learned_prox(
                    u_hat.reshape(B * Q, P),
                    tau_hat.reshape(B * Q, P),
                    cond.reshape(B * Q, -1),
                ).view(B, Q, P)
            elif prox_mode == "soft_threshold":
                beta = torch.sign(u_hat) * torch.relu(u_hat.abs() - tau_hat)
            else:
                raise ValueError(
                    f"prox_mode must be 'learned' or 'soft_threshold'; got {prox_mode!r}"
                )

        fm = encoded["feature_mask"]
        if fm is not None:
            beta = beta * fm[:, None, :].to(beta.dtype)
            u_hat = u_hat * fm[:, None, :].to(u_hat.dtype)
            if slope_aux:
                slope_aux["slope_effective_nnz_q"] = (beta.abs() > 1e-4).to(beta.dtype).sum(dim=-1)

        if not return_aux:
            return beta

        out: Dict[str, Tensor] = {
            "beta_q": beta,
            "u_base_q": u_base,
            "u_hat_q": u_hat,
            "tau_hat_q": tau_hat,
            "mixing_weights": A,
            "query_embedding": cond,
            "channel_projection_q": S,
            "base_mix_q": mix,
            "base_alpha_q": alpha.to(model_dtype),
        }
        out.update(slope_aux)
        if return_row_state:
            if "U" not in encoded:
                raise ValueError("return_row_state=True requires cfg.retain_U=True")
            with _maybe_disable_autocast(device.type):
                Z = torch.einsum(
                    "bqcm,bmn->bqcn",
                    A.float(),
                    encoded["U"].float(),
                )
            out["row_state_q"] = Z.to(model_dtype)
        return out

    def solve_queries(
        self,
        encoded: Dict[str, Any],
        lambdas: Tensor,
        lambda_max: Optional[Tensor] = None,
        penalty_type_id: Optional[Union[str, int, Tensor]] = None,
        penalty_params: Optional[Tensor] = None,
        return_aux: bool = False,
        return_row_state: bool = False,
    ) -> Union[Tensor, Dict[str, Any]]:
        dec = self.decode_queries(
            encoded=encoded,
            lambdas=lambdas,
            lambda_max=lambda_max,
            penalty_type_id=penalty_type_id,
            penalty_params=penalty_params,
            return_aux=True,
            return_row_state=return_row_state,
        )

        beta_q = dec["beta_q"]
        intercept_q = recover_intercept(
            x_mean=encoded["x_mean"],
            y_mean=encoded["y_mean"],
            beta=beta_q,
        )

        beta = beta_q[:, 0, :] if beta_q.shape[1] == 1 else beta_q
        intercept = intercept_q[:, 0] if intercept_q.shape[1] == 1 else intercept_q

        if not return_aux:
            return beta

        out: Dict[str, Any] = dict(dec)
        out["beta"] = beta
        out["intercept"] = intercept
        out["intercept_q"] = intercept_q
        out["basis_diagnostics"] = encoded["basis_diagnostics"]
        out["lambda_max_dataset"] = encoded["lambda_max"]
        return out

    def forward(
        self,
        X: Tensor,
        y: Tensor,
        lambdas: Tensor,
        lambda_max: Optional[Tensor] = None,
        feature_mask: Optional[Tensor] = None,
        penalty_type_id: Optional[Union[str, int, Tensor]] = None,
        penalty_params: Optional[Tensor] = None,
        return_aux: bool = False,
        return_row_state: bool = False,
    ) -> Union[Tensor, Dict[str, Any]]:
        precomputed = self.precompute(X, y, feature_mask=feature_mask)
        encoded = self.encode_dataset(precomputed)
        return self.solve_queries(
            encoded=encoded,
            lambdas=lambdas,
            lambda_max=lambda_max,
            penalty_type_id=penalty_type_id,
            penalty_params=penalty_params,
            return_aux=return_aux,
            return_row_state=return_row_state,
        )

    def forward_progressive(
        self,
        X: Tensor,
        y: Tensor,
        lambdas: Tensor,
        checkpoints: Optional[Sequence[int]] = None,
        lambda_max: Optional[Tensor] = None,
        feature_mask: Optional[Tensor] = None,
        penalty_type_id: Optional[Union[str, int, Tensor]] = None,
        penalty_params: Optional[Tensor] = None,
        return_row_state: bool = False,
    ) -> Dict[str, Any]:
        """Decode a sequence of prefixes independently."""
        if y.dim() == 3 and y.size(-1) == 1:
            y = y.squeeze(-1)

        B, N, P = X.shape
        lam = _as_2d_queries(lambdas, B, "lambdas", device=X.device, dtype=torch.float32)
        _, Q = lam.shape

        if checkpoints is None:
            delta = max(1, int(self.cfg.prefix_delta))
            checkpoints = list(range(delta, N + 1, delta))
            if checkpoints[-1] != N:
                checkpoints.append(N)

        beta_path: List[Tensor] = []
        intercept_path: List[Tensor] = []
        lambda_max_path: List[Tensor] = []
        row_state_path: List[Tensor] = []

        for nk in checkpoints:
            pre = self.precompute(X[:, :nk, :], y[:, :nk], feature_mask=feature_mask)
            enc = self.encode_dataset(pre)

            if lambda_max is None:
                lmx_k = enc["lambda_max"]
            else:
                lmx_k = lambda_max

            out = self.solve_queries(
                encoded=enc,
                lambdas=lam,
                lambda_max=lmx_k,
                penalty_type_id=penalty_type_id,
                penalty_params=penalty_params,
                return_aux=True,
                return_row_state=return_row_state,
            )

            beta_path.append(out["beta_q"])
            intercept_path.append(out["intercept_q"])
            lambda_max_path.append(
                _broadcast_queries_like(
                    lmx_k,
                    B=B,
                    Q=Q,
                    name="lambda_max",
                    device=X.device,
                    dtype=torch.float32,
                )
            )
            if return_row_state:
                row_state_path.append(out["row_state_q"])

        beta_grid = torch.stack(beta_path, dim=1)
        intercept_grid = torch.stack(intercept_path, dim=1)
        lambda_max_grid = torch.stack(lambda_max_path, dim=1)

        beta_final_q = beta_grid[:, -1]
        intercept_final_q = intercept_grid[:, -1]
        beta_final = beta_final_q[:, 0, :] if Q == 1 else beta_final_q
        intercept_final = intercept_final_q[:, 0] if Q == 1 else intercept_final_q

        out: Dict[str, Any] = {
            "beta": beta_final,
            "intercept": intercept_final,
            "beta_path": beta_grid,
            "intercept_path": intercept_grid,
            "checkpoints": list(checkpoints),
            "lambda_max_path": lambda_max_grid,
        }
        if return_row_state:
            out["row_state_path"] = torch.stack(row_state_path, dim=1)
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# Data-only lasso objectives
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_beta_q(beta_hat: Tensor) -> Tensor:
    if beta_hat.dim() == 2:
        return beta_hat.unsqueeze(1)
    if beta_hat.dim() == 3:
        return beta_hat
    raise ValueError(f"beta_hat must have shape (B,P) or (B,Q,P), got {tuple(beta_hat.shape)}")


def lambda_query_weights(
    lambdas: Tensor,
    lambda_max: Tensor,
    gamma: float = 0.75,
) -> Tensor:
    """Low-lambda upweighting; weights are normalized within each task."""
    if lambdas.dim() == 1:
        lambdas = lambdas.unsqueeze(1)
    B, Q = lambdas.shape
    lmx = _broadcast_queries_like(
        lambda_max,
        B=B,
        Q=Q,
        name="lambda_max",
        device=lambdas.device,
        dtype=torch.float32,
    ).clamp_min(5e-3)
    ratio = (lambdas.float() / lmx).clamp_min(5e-3)
    w = ratio.pow(-gamma)
    return w / w.mean(dim=1, keepdim=True)


def smooth_nnz_penalty(
    beta_hat: Tensor,
    lambdas: Tensor,
    lambda_max: Tensor,
    target_scale: float = 0.15,
    threshold: float = 5e-4,
    temperature: float = 1e-4,
    query_weight_gamma: float = 0.75,
) -> Tensor:
    """Differentiable excess-NNZ penalty weighted toward low-lambda queries."""
    beta_q = _ensure_beta_q(beta_hat)
    B, Q, P = beta_q.shape
    lam = _broadcast_queries_like(
        lambdas,
        B=B,
        Q=Q,
        name="lambdas",
        device=beta_q.device,
        dtype=torch.float32,
    )
    lmx = _broadcast_queries_like(
        lambda_max,
        B=B,
        Q=Q,
        name="lambda_max",
        device=beta_q.device,
        dtype=torch.float32,
    ).clamp_min(1e-12)
    ratio = (lam / lmx).clamp(1e-3, 1.0)
    temp = max(float(temperature), 1e-12)
    p_active = torch.sigmoid((beta_q.float().abs() - float(threshold)) / temp)
    smooth_nnz = p_active.sum(dim=-1)
    target_nnz = float(P) * float(target_scale) * (1.0 - torch.sqrt(ratio))
    excess = torch.relu(smooth_nnz - target_nnz)
    qw = lambda_query_weights(lam, lmx, gamma=query_weight_gamma)
    return ((excess.square() / float(max(1, P * P))) * qw).sum(dim=1).div(qw.sum(dim=1)).mean()


def lasso_objective_per_query(
    X: Tensor,
    y: Tensor,
    beta_hat: Tensor,
    lambdas: Tensor,
    feature_mask: Optional[Tensor] = None,
    precomputed: Optional[Dict[str, Any]] = None,
) -> Tensor:
    """Return the lasso objective for each task/query pair, shape (B, Q)."""
    beta_q = _ensure_beta_q(beta_hat)
    if precomputed is None or ("Xc" not in precomputed) or ("yc" not in precomputed):
        Xc_tmp, yc_tmp, _, _ = center_xy(X, y, feature_mask=feature_mask)
        precomputed = {"Xc": Xc_tmp, "yc": yc_tmp}

    Xc = precomputed["Xc"]
    yc = precomputed["yc"]
    B, N, P = Xc.shape
    _, Q, P_beta = beta_q.shape
    if P_beta != P:
        raise ValueError(f"beta_hat last dim mismatch: expected {P}, got {P_beta}")

    lam = _broadcast_queries_like(
        lambdas,
        B=B,
        Q=Q,
        name="lambdas",
        device=Xc.device,
        dtype=torch.float32,
    )

    with _maybe_disable_autocast(Xc.device.type):
        Xcf = Xc.float()
        ycf = yc.float()
        beta_f = beta_q.float()
        pred = torch.einsum("bnp,bqp->bqn", Xcf, beta_f)
        resid = ycf.unsqueeze(1) - pred
        data = 0.5 * resid.square().mean(dim=-1)
        reg = lam * beta_f.abs().sum(dim=-1)
        return data + reg


def _squared_error_data_term_per_query(
    X: Tensor,
    y: Tensor,
    beta_hat: Tensor,
    feature_mask: Optional[Tensor] = None,
    precomputed: Optional[Dict[str, Any]] = None,
) -> Tuple[Tensor, Tensor]:
    """Return ((1 / 2N) ||yc - Xc beta||^2, beta_q) with shape (B, Q)."""
    beta_q = _ensure_beta_q(beta_hat)
    if precomputed is None or ("Xc" not in precomputed) or ("yc" not in precomputed):
        Xc_tmp, yc_tmp, _, _ = center_xy(X, y, feature_mask=feature_mask)
        precomputed = {"Xc": Xc_tmp, "yc": yc_tmp}

    Xc = precomputed["Xc"]
    yc = precomputed["yc"]
    B, N, P = Xc.shape
    _, _, P_beta = beta_q.shape
    if P_beta != P:
        raise ValueError(f"beta_hat last dim mismatch: expected {P}, got {P_beta}")

    with _maybe_disable_autocast(Xc.device.type):
        Xcf = Xc.float()
        ycf = yc.float()
        beta_f = beta_q.float()
        pred = torch.einsum("bnp,bqp->bqn", Xcf, beta_f)
        resid = ycf.unsqueeze(1) - pred
        data = 0.5 * resid.square().mean(dim=-1)
        return data, beta_f


def ridge_objective_per_query(
    X: Tensor,
    y: Tensor,
    beta_hat: Tensor,
    lambdas: Tensor,
    feature_mask: Optional[Tensor] = None,
    precomputed: Optional[Dict[str, Any]] = None,
) -> Tensor:
    """Return (1 / 2N) ||yc - Xc beta||^2 + lambda ||beta||_2^2."""
    data, beta_f = _squared_error_data_term_per_query(
        X=X,
        y=y,
        beta_hat=beta_hat,
        feature_mask=feature_mask,
        precomputed=precomputed,
    )
    B, Q, _ = beta_f.shape
    lam = _broadcast_queries_like(
        lambdas,
        B=B,
        Q=Q,
        name="lambdas",
        device=beta_f.device,
        dtype=torch.float32,
    )
    return data + lam * beta_f.square().sum(dim=-1)


def mcp_objective_per_query(
    X: Tensor,
    y: Tensor,
    beta_hat: Tensor,
    lambdas: Tensor,
    gamma: float = 3.0,
    feature_mask: Optional[Tensor] = None,
    precomputed: Optional[Dict[str, Any]] = None,
) -> Tensor:
    """Return squared-error data term plus separable MCP penalty."""
    if gamma <= 1.0:
        raise ValueError(f"MCP gamma should be > 1.0, got {gamma}")
    data, beta_f = _squared_error_data_term_per_query(
        X=X,
        y=y,
        beta_hat=beta_hat,
        feature_mask=feature_mask,
        precomputed=precomputed,
    )
    B, Q, _ = beta_f.shape
    lam = _broadcast_queries_like(
        lambdas,
        B=B,
        Q=Q,
        name="lambdas",
        device=beta_f.device,
        dtype=torch.float32,
    ).unsqueeze(-1)
    abs_beta = beta_f.abs()
    threshold = gamma * lam
    penalty = torch.where(
        abs_beta <= threshold,
        lam * abs_beta - beta_f.square() / (2.0 * gamma),
        0.5 * gamma * lam.square(),
    )
    return data + penalty.sum(dim=-1)


def penalty_objective_per_query(
    X: Tensor,
    y: Tensor,
    beta_hat: Tensor,
    lambdas: Tensor,
    penalty_loss_type: str = "lasso",
    mcp_gamma: float = 3.0,
    feature_mask: Optional[Tensor] = None,
    precomputed: Optional[Dict[str, Any]] = None,
) -> Tensor:
    """Penalty-aware objective dispatcher for lasso, ridge, and MCP."""
    penalty = penalty_loss_type.lower()
    if penalty == "lasso":
        return lasso_objective_per_query(
            X=X,
            y=y,
            beta_hat=beta_hat,
            lambdas=lambdas,
            feature_mask=feature_mask,
            precomputed=precomputed,
        )
    if penalty == "ridge":
        return ridge_objective_per_query(
            X=X,
            y=y,
            beta_hat=beta_hat,
            lambdas=lambdas,
            feature_mask=feature_mask,
            precomputed=precomputed,
        )
    if penalty == "mcp":
        return mcp_objective_per_query(
            X=X,
            y=y,
            beta_hat=beta_hat,
            lambdas=lambdas,
            gamma=mcp_gamma,
            feature_mask=feature_mask,
            precomputed=precomputed,
        )
    raise ValueError(f"Unsupported penalty_loss_type {penalty_loss_type!r}")


def penalty_objective_training_loss(
    X: Tensor,
    y: Tensor,
    beta_hat: Tensor,
    lambdas: Tensor,
    lambda_max: Tensor,
    penalty_loss_type: str = "lasso",
    mcp_gamma: float = 3.0,
    feature_mask: Optional[Tensor] = None,
    precomputed: Optional[Dict[str, Any]] = None,
    objective_weight: float = 1.0,
    query_weight_gamma: float = 0.75,
) -> Tensor:
    """Low-lambda-weighted objective-only training loss."""
    obj_q = penalty_objective_per_query(
        X=X,
        y=y,
        beta_hat=beta_hat,
        lambdas=lambdas,
        penalty_loss_type=penalty_loss_type,
        mcp_gamma=mcp_gamma,
        feature_mask=feature_mask,
        precomputed=precomputed,
    )
    qw = lambda_query_weights(lambdas, lambda_max, gamma=query_weight_gamma)
    obj = ((obj_q * qw).sum(dim=1) / qw.sum(dim=1)).mean()
    return objective_weight * obj


def lasso_kkt_topk_per_query(
    X: Tensor,
    y: Tensor,
    beta_hat: Tensor,
    lambdas: Tensor,
    feature_mask: Optional[Tensor] = None,
    precomputed: Optional[Dict[str, Any]] = None,
    beta_zero_tol: float = 1e-8,
    inactive_topk: int = 256,
    active_weight: float = 4.0,
    inactive_weight: float = 1.0,
) -> Tensor:
    """Active-set stationarity + top-k inactive lasso KKT violations, shape (B, Q)."""
    beta_q = _ensure_beta_q(beta_hat)
    if precomputed is None or ("Xc" not in precomputed) or ("yc" not in precomputed):
        Xc_tmp, yc_tmp, _, _ = center_xy(X, y, feature_mask=feature_mask)
        precomputed = {"Xc": Xc_tmp, "yc": yc_tmp}

    Xc = precomputed["Xc"]
    yc = precomputed["yc"]

    B, N, P = Xc.shape
    _, Q, P_beta = beta_q.shape
    if P_beta != P:
        raise ValueError(f"beta_hat last dim mismatch: expected {P}, got {P_beta}")

    lam = _broadcast_queries_like(
        lambdas,
        B=B,
        Q=Q,
        name="lambdas",
        device=Xc.device,
        dtype=torch.float32,
    )

    fm = _normalize_feature_mask(feature_mask, B, P, Xc.device)

    with _maybe_disable_autocast(Xc.device.type):
        Xcf = Xc.float()
        ycf = yc.float()
        beta_f = beta_q.float()

        pred = torch.einsum("bnp,bqp->bqn", Xcf, beta_f)
        grad = torch.einsum("bnp,bqn->bqp", Xcf, pred - ycf.unsqueeze(1)) / float(max(1, N))

        active = beta_f.abs() > beta_zero_tol
        if fm is not None:
            active = active & fm[:, None, :]

        stationarity = grad + lam.unsqueeze(-1) * beta_f.sign()
        active_denom = active.float().sum(dim=-1).clamp_min(1.0)
        active_loss_q = (stationarity.square() * active.float()).sum(dim=-1) / active_denom

        inactive_violation = torch.relu(grad.abs() - lam.unsqueeze(-1))
        inactive_violation = inactive_violation.masked_fill(active, 0.0)
        if fm is not None:
            inactive_violation = inactive_violation.masked_fill(~fm[:, None, :], 0.0)

        k = min(int(inactive_topk), P)
        topv = inactive_violation.topk(k, dim=-1).values
        inactive_loss_q = topv.square().mean(dim=-1)

        return active_weight * active_loss_q + inactive_weight * inactive_loss_q


def lasso_monotone_l1_only(
    beta_hat: Tensor,
    lambdas: Tensor,
) -> Tensor:
    """Only enforce monotone total L1 mass as lambda decreases."""
    beta_q = _ensure_beta_q(beta_hat)
    B, Q, P = beta_q.shape
    if Q < 2:
        return beta_q.new_zeros(())

    lam = _broadcast_queries_like(
        lambdas,
        B=B,
        Q=Q,
        name="lambdas",
        device=beta_q.device,
        dtype=torch.float32,
    )

    order = lam.argsort(dim=1, descending=True)
    beta_sorted = torch.gather(beta_q, 1, order.unsqueeze(-1).expand(B, Q, P))
    l1 = beta_sorted.abs().sum(dim=-1)
    return torch.relu(l1[:, :-1] - l1[:, 1:]).square().mean()


def lasso_training_loss_v2(
    X: Tensor,
    y: Tensor,
    beta_hat: Tensor,
    lambdas: Tensor,
    lambda_max: Tensor,
    feature_mask: Optional[Tensor] = None,
    precomputed: Optional[Dict[str, Any]] = None,
    objective_weight: float = 1.0,
    kkt_weight: float = 0.5,
    path_weight: float = 0.0,
    query_weight_gamma: float = 0.75,
    kkt_inactive_topk: int = 256,
) -> Tensor:
    """Low-lambda-weighted training loss for one-shot Krylov decoding."""
    obj_q = lasso_objective_per_query(
        X=X,
        y=y,
        beta_hat=beta_hat,
        lambdas=lambdas,
        feature_mask=feature_mask,
        precomputed=precomputed,
    )
    kkt_q = lasso_kkt_topk_per_query(
        X=X,
        y=y,
        beta_hat=beta_hat,
        lambdas=lambdas,
        feature_mask=feature_mask,
        precomputed=precomputed,
        inactive_topk=kkt_inactive_topk,
    )

    qw = lambda_query_weights(lambdas, lambda_max, gamma=query_weight_gamma)
    obj = ((obj_q * qw).sum(dim=1) / qw.sum(dim=1)).mean()
    kkt = ((kkt_q * qw).sum(dim=1) / qw.sum(dim=1)).mean()

    loss = objective_weight * obj + kkt_weight * kkt
    if path_weight > 0.0:
        loss = loss + path_weight * lasso_monotone_l1_only(beta_hat, lambdas)
    return loss

def lasso_objective_loss(
    X: Tensor,
    y: Tensor,
    beta_hat: Tensor,
    lambdas: Tensor,
    feature_mask: Optional[Tensor] = None,
    precomputed: Optional[Dict[str, Any]] = None,
) -> Tensor:
    """(1 / 2N) ||yc - Xc beta||^2 + lambda ||beta||_1."""
    beta_q = _ensure_beta_q(beta_hat)
    if precomputed is None or ("Xc" not in precomputed) or ("yc" not in precomputed):
        Xc_tmp, yc_tmp, _, _ = center_xy(X, y, feature_mask=feature_mask)
        precomputed = {"Xc": Xc_tmp, "yc": yc_tmp}

    Xc = precomputed["Xc"]
    yc = precomputed["yc"]

    B, N, P = Xc.shape
    _, Q, P_beta = beta_q.shape
    if P_beta != P:
        raise ValueError(f"beta_hat last dim mismatch: expected {P}, got {P_beta}")

    lam = _broadcast_queries_like(
        lambdas,
        B=B,
        Q=Q,
        name="lambdas",
        device=Xc.device,
        dtype=torch.float32,
    )

    with _maybe_disable_autocast(Xc.device.type):
        Xcf = Xc.float()
        ycf = yc.float()
        beta_f = beta_q.float()
        pred = torch.einsum("bnp,bqp->bqn", Xcf, beta_f)
        resid = ycf.unsqueeze(1) - pred
        data = 0.5 * resid.square().mean(dim=-1)
        reg = lam * beta_f.abs().sum(dim=-1)
        return (data + reg).mean()


def lasso_kkt_loss(
    X: Tensor,
    y: Tensor,
    beta_hat: Tensor,
    lambdas: Tensor,
    feature_mask: Optional[Tensor] = None,
    precomputed: Optional[Dict[str, Any]] = None,
    beta_zero_tol: float = 1e-8,
) -> Tensor:
    """Stationarity + subgradient feasibility residual for lasso."""
    beta_q = _ensure_beta_q(beta_hat)
    if precomputed is None or ("Xc" not in precomputed) or ("yc" not in precomputed):
        Xc_tmp, yc_tmp, _, _ = center_xy(X, y, feature_mask=feature_mask)
        precomputed = {"Xc": Xc_tmp, "yc": yc_tmp}

    Xc = precomputed["Xc"]
    yc = precomputed["yc"]

    B, N, P = Xc.shape
    _, Q, P_beta = beta_q.shape
    if P_beta != P:
        raise ValueError(f"beta_hat last dim mismatch: expected {P}, got {P_beta}")

    lam = _broadcast_queries_like(
        lambdas,
        B=B,
        Q=Q,
        name="lambdas",
        device=Xc.device,
        dtype=torch.float32,
    )

    with _maybe_disable_autocast(Xc.device.type):
        Xcf = Xc.float()
        ycf = yc.float()
        beta_f = beta_q.float()

        pred = torch.einsum("bnp,bqp->bqn", Xcf, beta_f)
        grad = torch.einsum("bnp,bqn->bqp", Xcf, pred - ycf.unsqueeze(1)) / float(max(1, N))

        active = beta_f.abs() > beta_zero_tol
        stationarity = grad + lam.unsqueeze(-1) * beta_f.sign()
        active_loss = (stationarity.square() * active.float()).mean()

        inactive_violation = torch.relu(grad.abs() - lam.unsqueeze(-1))
        inactive_loss = (inactive_violation.square() * (~active).float()).mean()

        return active_loss + inactive_loss


def lasso_weak_path_regularization(
    beta_hat: Tensor,
    lambdas: Tensor,
    smooth_weight: float = 0.05,
) -> Tensor:
    """Weak path regularization across lambda queries."""
    beta_q = _ensure_beta_q(beta_hat)
    B, Q, P = beta_q.shape
    if Q < 2:
        return beta_q.new_zeros(())

    lam = _broadcast_queries_like(
        lambdas,
        B=B,
        Q=Q,
        name="lambdas",
        device=beta_q.device,
        dtype=torch.float32,
    )

    order = lam.argsort(dim=1, descending=True)
    beta_sorted = torch.gather(beta_q, 1, order.unsqueeze(-1).expand(B, Q, P))
    lam_sorted = torch.gather(lam, 1, order)

    l1 = beta_sorted.abs().sum(dim=-1)
    monotone_violation = torch.relu(l1[:, :-1] - l1[:, 1:])

    dlog = (lam_sorted[:, :-1].log() - lam_sorted[:, 1:].log()).abs().clamp_min(1e-6)
    smooth = (beta_sorted[:, 1:, :] - beta_sorted[:, :-1, :]).abs().mean(dim=-1) / dlog

    return monotone_violation.square().mean() + smooth_weight * smooth.mean()


def lasso_training_loss(
    X: Tensor,
    y: Tensor,
    beta_hat: Tensor,
    lambdas: Tensor,
    feature_mask: Optional[Tensor] = None,
    precomputed: Optional[Dict[str, Any]] = None,
    objective_weight: float = 1.0,
    kkt_weight: float = 0.25,
    path_weight: float = 0.05,
) -> Tensor:
    obj = lasso_objective_loss(
        X=X,
        y=y,
        beta_hat=beta_hat,
        lambdas=lambdas,
        feature_mask=feature_mask,
        precomputed=precomputed,
    )
    kkt = lasso_kkt_loss(
        X=X,
        y=y,
        beta_hat=beta_hat,
        lambdas=lambdas,
        feature_mask=feature_mask,
        precomputed=precomputed,
    )
    path = lasso_weak_path_regularization(
        beta_hat=beta_hat,
        lambdas=lambdas,
    )
    return objective_weight * obj + kkt_weight * kkt + path_weight * path


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    torch.manual_seed(0)
    try:
        torch.set_num_threads(min(4, torch.get_num_threads()))
    except Exception:
        pass

    def _assert_no_loops(fn) -> None:
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
        loops = [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While, ast.AsyncFor))]
        assert len(loops) == 0, f"{fn.__name__} contains explicit Python loops"

    device = torch.device("cpu")
    cfg = KrylovQueryDecoderConfig(
        krylov_dim=8,
        num_decoder_channels=6,
        use_bilinear_channels=True,
        cond_dim=24,
        cond_hidden=64,
        query_mixer_hidden=64,
        feature_head_hidden=96,
        prox_hidden=48,
        retain_U=True,
        retain_centered_data_in_encoding=False,
    )
    model = KrylovRowSpaceQueryDecoder(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    B, N, P, Q = 3, 64, 128, 5
    X = torch.randn(B, N, P, device=device)
    beta_true = torch.zeros(B, P, device=device)
    beta_true[:, :7] = torch.randn(B, 7, device=device)
    y = torch.einsum("bnp,bp->bn", X, beta_true) + 0.05 * torch.randn(B, N, device=device)

    _, lambda_max = compute_lambda_pair(X, y, ratio_range=(1e-2, 0.5))
    ratios = torch.tensor([1.0, 0.5, 0.2, 0.05, 0.01], device=device, dtype=torch.float32)
    lambdas = lambda_max[:, None] * ratios[None, :]

    # 1. Shape test
    _sep("Test 1: shape test for B, N, P, Q")
    pre = model.precompute(X, y)
    enc = model.encode_dataset(pre)
    out = model.solve_queries(enc, lambdas=lambdas, return_aux=True)
    print(f"params:                  {n_params:,}")
    print(f"beta_q shape:            {tuple(out['beta_q'].shape)}")
    print(f"intercept_q shape:       {tuple(out['intercept_q'].shape)}")
    assert out["beta_q"].shape == (B, Q, P)
    assert out["intercept_q"].shape == (B, Q)

    # 2. Finite forward
    _sep("Test 2: finite forward")
    out_fwd = model(X, y, lambdas=lambdas, return_aux=True)
    finite_ok = (
        torch.isfinite(out_fwd["beta_q"]).all()
        and torch.isfinite(out_fwd["intercept_q"]).all()
        and torch.isfinite(out_fwd["u_hat_q"]).all()
        and torch.isfinite(out_fwd["tau_hat_q"]).all()
    )
    print(f"finite forward:          {finite_ok}")
    assert finite_ok

    # 3. Backward pass finite
    _sep("Test 3: backward pass finite")
    model.train()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    beta_pred = model(X, y, lambdas=lambdas)
    loss = lasso_training_loss(X, y, beta_pred, lambdas=lambdas)
    print(f"training loss:           {loss.item():.6f}")
    loss.backward()
    grads_ok = all(
        (p.grad is None) or torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )
    print(f"finite gradients:        {grads_ok}")
    assert grads_ok
    model.eval()

    # 4. Feature permutation
    _sep("Test 4: permutation equivariance over features")
    perm = torch.randperm(P, device=device)
    beta_a = model(X, y, lambdas=lambdas)
    beta_b = model(X[:, :, perm], y, lambdas=lambdas)
    perm_err = (beta_a[:, :, perm] - beta_b).abs().max().item()
    print(f"max permutation error:   {perm_err:.6e}")
    assert perm_err < 1e-5

    # 5. Query-order invariance
    _sep("Test 5: query-order invariance")
    order = torch.tensor([4, 2, 0, 1, 3], device=device)
    beta_ord_a = model(X, y, lambdas=lambdas)
    beta_ord_b = model(X, y, lambdas=lambdas[:, order])
    order_err = (beta_ord_a[:, order, :] - beta_ord_b).abs().max().item()
    print(f"max query reorder error: {order_err:.6e}")
    assert order_err < 1e-6

    # 6. Single-query vs batched-query consistency
    _sep("Test 6: one-query vs batched-query consistency")
    q_idx = 2
    beta_single = model(X, y, lambdas=lambdas[:, q_idx])
    beta_batch = model(X, y, lambdas=lambdas)
    one_vs_batch_err = (beta_single - beta_batch[:, q_idx, :]).abs().max().item()
    print(f"max single/batch error:  {one_vs_batch_err:.6e}")
    assert one_vs_batch_err < 1e-6

    # 7. Different P sizes
    _sep("Test 7: p-agnostic behavior across different P")
    for P_t in [63, 257, 1024]:
        X_t = torch.randn(2, 48, P_t, device=device)
        y_t = torch.randn(2, 48, device=device)
        _, lmx_t = compute_lambda_pair(X_t, y_t)
        lam_t = lmx_t[:, None] * torch.tensor([1.0, 0.2, 0.05], device=device)
        beta_t = model(X_t, y_t, lambdas=lam_t)
        ok = torch.isfinite(beta_t).all().item()
        print(f"P={P_t:>4d} -> shape {tuple(beta_t.shape)}, finite={ok}")
        assert beta_t.shape == (2, 3, P_t)
        assert ok

    # 8. Scaling sanity
    _sep("Test 8: scaling sanity at N=500, P=5000")
    X_big = torch.randn(1, 500, 5000, device=device)
    y_big = torch.randn(1, 500, device=device)
    _, lmx_big = compute_lambda_pair(X_big, y_big)
    lam_big = lmx_big[:, None] * torch.tensor([1.0, 0.3, 0.1, 0.03], device=device)

    t0 = time.perf_counter()
    with torch.no_grad():
        big_out = model(X_big, y_big, lambdas=lam_big, return_aux=True)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    print(f"beta_q shape:            {tuple(big_out['beta_q'].shape)}")
    print(f"forward time (cpu):      {dt_ms:.1f} ms")
    assert big_out["beta_q"].shape == (1, 4, 5000)
    assert torch.isfinite(big_out["beta_q"]).all()

    # 9. Orthogonality diagnostics
    _sep("Test 9: Krylov basis orthogonality diagnostics")
    diag = out["basis_diagnostics"]
    orth_err = diag["orthogonality_error"].max().item()
    offdiag_err = diag["offdiag_error"].max().item()
    valid_norms = diag["basis_norms"][diag["valid_mask"]]
    norm_err = (valid_norms - 1.0).abs().max().item() if valid_norms.numel() > 0 else 0.0
    print(f"orthogonality error:     {orth_err:.6e}")
    print(f"offdiag error:           {offdiag_err:.6e}")
    print(f"basis norm error:        {norm_err:.6e}")
    assert orth_err < 5e-4
    assert offdiag_err < 5e-4
    assert norm_err < 5e-6

    # 10. AST check on default forward
    _sep("Test 10: AST check on default forward")
    _assert_no_loops(KrylovRowSpaceQueryDecoder.decode_queries)
    _assert_no_loops(KrylovRowSpaceQueryDecoder.solve_queries)
    _assert_no_loops(KrylovRowSpaceQueryDecoder.forward)
    print("decode_queries / solve_queries / forward contain no explicit Python loops.")
    print("Basis construction uses explicit loops; the default query path does not.")

    # 11. Prefix mode
    _sep("Test 11: prefix mode")
    prog = model.forward_progressive(
        X=X,
        y=y,
        lambdas=lambdas,
        checkpoints=[16, 32, 48, 64],
    )
    print(f"beta_path shape:         {tuple(prog['beta_path'].shape)}")
    print(f"intercept_path shape:    {tuple(prog['intercept_path'].shape)}")
    assert prog["beta_path"].shape == (B, 4, Q, P)
    assert prog["intercept_path"].shape == (B, 4, Q)
    assert torch.isfinite(prog["beta_path"]).all()

    print(f"\n{'=' * 72}")
    print("All tests passed.")


# ##############################################################################
# SECTION B -- Penalty-shape encoding
# (merged from krylov/penalty/evaluators.py, features.py, encoder.py)
#
#   * evaluators.py : PID_* family ids, standardized rho/drho/d0
#   * features.py   : bounded 101-dim shape-feature builder
#   * encoder.py    : PenaltyShapeEncoder + the patched conditioning and
#                     prox heads (renamed: see module docstring)
#
# Self-contained: depends only on torch/nn and reuses `_broadcast_queries_like`
# from Section A (the penalty package shipped its own identical copy).
# ##############################################################################


# Keep these in sync with PENALTY_REGISTRY above.
PID_LASSO = 0
PID_RIDGE = 1
PID_ENET = 2
PID_SCAD = 3
PID_MCP = 4


def default_penalty_grid(
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Near-zero-heavy positive grid used by the shape-feature builder.

    Most points lie below 1.0 because that is where sparse penalties
    actually differ from each other; the points above 1.0 only need to
    capture saturation behavior (MCP/SCAD flattening, Ridge growth).
    """
    return torch.tensor(
        [
            0.0,
            1e-4, 3e-4, 1e-3, 3e-3,
            1e-2, 2e-2, 5e-2, 7.5e-2,
            1e-1, 1.5e-1, 2e-1, 3e-1,
            5e-1, 7.5e-1,
            1.0, 1.25, 1.5, 2.0, 2.5,
            3.0, 4.0, 6.0, 8.0,
        ],
        device=device,
        dtype=dtype,
    )


def _raw_penalty_rho_drho(
    t: Tensor,
    penalty_type_id: Tensor,
    penalty_params: Tensor,
    eps: float = 1e-12,
) -> Tuple[Tensor, Tensor]:
    """Raw (unstandardized) rho and rho' for the supported families.

    Parameters
    ----------
    t:
        Nonnegative tensor of shape (B, Q, G) or (B, Q, P).
    penalty_type_id:
        (B, Q) long tensor of family IDs.
    penalty_params:
        (B, Q, K) float tensor. The first slot is interpreted family-wise:
        - Elastic Net:  L1 fraction alpha in [0, 1]
        - MCP:          gamma >= 1.01
        - SCAD:         a    >= 2.01
        For Lasso/Ridge the slot is unused.

    Returns
    -------
    rho, drho:
        Tensors with the same shape as `t`.

    Notes
    -----
    `torch.where` is used rather than masking-and-summing because every
    branch is cheap and we want a clean autograd path. Parameter clamps
    are applied before the closed-form branches so that gradients into
    `t` are well-defined everywhere; sampled parameters should be
    detached upstream.
    """
    pid = penalty_type_id.to(torch.long).unsqueeze(-1)

    # Robust extraction of the first param slot. Shape: (B, Q, 1).
    if penalty_params.numel() > 0:
        p0 = penalty_params[..., 0:1]
    else:
        p0 = torch.zeros_like(t[..., :1])

    alpha = p0.clamp(0.0, 1.0)        # Elastic Net L1 fraction
    gamma = p0.clamp(1.01, 50.0)      # MCP concavity radius
    a_scad = p0.clamp(2.01, 50.0)     # SCAD shape

    # --- Lasso: rho(t) = t -------------------------------------------------
    rho_lasso = t
    drho_lasso = torch.ones_like(t)

    # --- Ridge: rho(t) = t^2 / 2 ------------------------------------------
    rho_ridge = 0.5 * t.square()
    drho_ridge = t

    # --- Elastic Net: alpha * L1 + (1 - alpha) * Ridge --------------------
    rho_enet = alpha * t + 0.5 * (1.0 - alpha) * t.square()
    drho_enet = alpha + (1.0 - alpha) * t

    # --- MCP --------------------------------------------------------------
    # rho(t) = t - t^2/(2 gamma) for t <= gamma, else gamma/2.
    inside = t <= gamma
    rho_mcp = torch.where(inside, t - t.square() / (2.0 * gamma), 0.5 * gamma)
    drho_mcp = torch.relu(1.0 - t / gamma)

    # --- SCAD -------------------------------------------------------------
    a_safe = (a_scad - 1.0).clamp_min(eps)
    rho_scad_quad = (-t.square() + 2.0 * a_scad * t - 1.0) / (2.0 * a_safe)
    rho_scad = torch.where(
        t <= 1.0,
        t,
        torch.where(t <= a_scad, rho_scad_quad, 0.5 * (a_scad + 1.0)),
    )
    drho_scad_quad = (a_scad - t) / a_safe
    drho_scad = torch.where(
        t <= 1.0,
        torch.ones_like(t),
        torch.where(t <= a_scad, drho_scad_quad, torch.zeros_like(t)),
    ).clamp_min(0.0)

    # --- Dispatch ---------------------------------------------------------
    rho = torch.where(
        pid == PID_LASSO, rho_lasso,
        torch.where(
            pid == PID_RIDGE, rho_ridge,
            torch.where(
                pid == PID_ENET, rho_enet,
                torch.where(pid == PID_SCAD, rho_scad, rho_mcp),
            ),
        ),
    )

    drho = torch.where(
        pid == PID_LASSO, drho_lasso,
        torch.where(
            pid == PID_RIDGE, drho_ridge,
            torch.where(
                pid == PID_ENET, drho_enet,
                torch.where(pid == PID_SCAD, drho_scad, drho_mcp),
            ),
        ),
    )

    return rho, drho


def standardized_penalty_rho_drho(
    t: Tensor,
    penalty_type_id: Tensor,
    penalty_params: Tensor,
    t_ref: float = 1.0,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute standardized rho, drho, and the standardized d0 = rho'(0+).

    Standardization enforces rho(0) = 0 and rho(t_ref) = 1, so that
    `p` and `c*p` map to the same standardized shape; the overall
    multiplier `c` is absorbed into lambda upstream.

    Returns
    -------
    rho: tensor with the same shape as `t`
    drho: tensor with the same shape as `t`
    d0: tensor with shape `t.shape[:-1]` (one scalar per query)
    """
    rho_raw, drho_raw = _raw_penalty_rho_drho(
        t, penalty_type_id, penalty_params, eps=eps
    )

    # Standardization constants: evaluate at t=0 and t=t_ref.
    # Shapes: (..., 1).
    t0 = torch.zeros_like(t[..., :1])
    rho0, drho0 = _raw_penalty_rho_drho(
        t0, penalty_type_id, penalty_params, eps=eps
    )
    tref = torch.full_like(t[..., :1], float(t_ref))
    rho_ref, _ = _raw_penalty_rho_drho(
        tref, penalty_type_id, penalty_params, eps=eps
    )

    scale = (rho_ref - rho0).clamp_min(eps)

    rho = (rho_raw - rho0) / scale
    drho = drho_raw / scale
    d0 = (drho0 / scale).squeeze(-1)

    return rho, drho, d0






# Feature dimensionality at the default grid (G = 24, channels = 4, meta = 5).
DEFAULT_GRID_SIZE = 24
DEFAULT_GRID_CHANNELS = 4
DEFAULT_META_DIM = 5
DEFAULT_SHAPE_FEATURE_DIM = DEFAULT_GRID_SIZE * DEFAULT_GRID_CHANNELS + DEFAULT_META_DIM


def make_penalty_features(
    penalty_type_id: Tensor,
    penalty_params: Tensor,
    grid_t: Optional[Tensor] = None,
    eps: float = 1e-8,
    detach_inputs: bool = True,
) -> Dict[str, Tensor]:
    """Build conditioning features and standardized d0 for a batch of queries.

    Parameters
    ----------
    penalty_type_id:
        (B, Q) long tensor.
    penalty_params:
        (B, Q, K) float tensor. K>=1; only the first slot is read here.
    grid_t:
        Optional (G,) positive grid; falls back to `default_penalty_grid`.
    eps:
        Numerical floor for the standardization denominator.
    detach_inputs:
        If True (default), inputs are detached before feature construction.
        We never want gradients flowing into sampled penalty parameters
        through the conditioning path; we want them only through the
        downstream `t = |beta|/s` evaluation in the loss.

    Returns
    -------
    dict with keys:
        shape_features: (B, Q, D_raw) bounded conditioning vector
        d0:             (B, Q) standardized rho'(0+); used by tau_base
        grid_t:         (G,) the grid used (for caching / diagnostics)
        grid_rho:       (B, Q, G) standardized rho(t)  -- diagnostic
        grid_drho:      (B, Q, G) standardized rho'(t) -- diagnostic
        grid_curv:      (B, Q, G) finite-difference curvature -- diagnostic
        meta:           (B, Q, 5)  the bounded metadata block
    """
    if detach_inputs:
        penalty_type_id = penalty_type_id.detach()
        penalty_params = penalty_params.detach()

    # Force fp32 internally. The finite-difference curvature channel lives
    # at scales as small as O(1e-4) in the near-zero region; bf16/fp16
    # inputs would degrade it precisely where the redesign needs it sharp.
    # The conditioning encoder recasts to model dtype afterward.
    device = (
        penalty_params.device if penalty_params.numel() > 0
        else penalty_type_id.device
    )
    dtype = torch.float32
    penalty_params = penalty_params.to(device=device, dtype=dtype)

    if penalty_type_id.dim() != 2:
        raise ValueError(
            f"penalty_type_id must have shape (B, Q), got {tuple(penalty_type_id.shape)}"
        )
    B, Q = penalty_type_id.shape

    if grid_t is None:
        grid_t = default_penalty_grid(device=device, dtype=dtype)
    G = grid_t.numel()

    t = grid_t.view(1, 1, G).expand(B, Q, G)

    rho, drho, d0 = standardized_penalty_rho_drho(
        t=t,
        penalty_type_id=penalty_type_id,
        penalty_params=penalty_params,
        eps=eps,
    )

    # Finite-difference curvature. We pad the leading slot with the first
    # finite-difference so the curv channel has length G.
    dt = (grid_t[1:] - grid_t[:-1]).clamp_min(eps)
    curv_mid = (drho[..., 1:] - drho[..., :-1]) / dt.view(1, 1, -1)
    curv = torch.cat([curv_mid[..., :1], curv_mid], dim=-1)

    # Channel 0: positional encoding of t. FPK v2: changed from
    # `log1p(grid_t) / log1p(grid_t[-1])` to `log(t+1e-3) / log(2+1e-3)`.
    # The legacy formula saturated near t ~ 1 (because log1p makes the
    # near-zero differences extremely compressed); the new formula gives
    # a finer-grained encoding in the active region t in [1e-4, 2] while
    # still bounding values in roughly [-1, 1] across the grid.
    LOG_T_EPS = 1e-3
    log_denom = math.log(2.0 + LOG_T_EPS)
    t_norm = (torch.log(grid_t.clamp_min(0.0) + LOG_T_EPS) / log_denom)
    t_feat = t_norm.view(1, 1, G).expand(B, Q, G)

    # Bounded value/derivative/curvature channels.
    rho_feat = torch.tanh(rho / 4.0)
    drho_feat = torch.tanh(drho / 2.0)
    curv_feat = torch.tanh(curv / 2.0)

    grid_feat = torch.stack(
        [rho_feat, drho_feat],
        dim=-1,
    )  # (B, Q, G, 2)

    # Metadata: a handful of single-scalar features that summarize the
    # penalty shape. Kept bounded with tanh / explicit binary sign.
    area = torch.trapezoid(rho, grid_t, dim=-1) / grid_t[-1].clamp_min(eps)
    tail_slope = drho[..., -1]
    near_slope = drho[..., 1]  # at the first nonzero grid point
    has_kink = (d0 > 1e-6).to(dtype)

    meta = torch.stack(
        [
            torch.tanh(d0 / 2.0),
            torch.tanh(near_slope / 2.0),
            torch.tanh(tail_slope / 2.0),
            torch.tanh(area / 4.0),
            2.0 * has_kink - 1.0,
        ],
        dim=-1,
    )  # (B, Q, 5)

    shape_features = torch.cat(
        [grid_feat.reshape(B, Q, G * 2), meta],
        dim=-1,
    )

    return {
        "shape_features": shape_features,
        "d0": d0,
        "grid_t": grid_t,
        "grid_rho": rho,
        "grid_drho": drho,
        "grid_curv": curv,
        "meta": meta,
    }


# ============================================================================
# make_penalty_features_from_segments — random-θ companion to
# make_penalty_features. Verified against the 5 known families to ≤5e-4 in
# shape_features and ≤1e-6 in d0. See dg_fpk v5.1 priority notebook.
# ============================================================================

def make_penalty_features_from_segments(
    segments_grid,
    grid_t=None,
    eps=1e-8,
    device=None,
):
    """Random-penalty companion to ``make_penalty_features``.

    Accepts a (B, Q) nested list of ``SegmentList`` objects (the segment
    representation, already standardized to rho(0)=0, rho(1)=1) and
    returns the SAME dict that ``make_penalty_features`` returns for
    known families: ``shape_features``, ``d0``, ``grid_t``, ``grid_rho``,
    ``grid_drho``, ``grid_curv``, ``meta``.

    FPK v2 notes:
      - ``device=None`` defaults to CPU (FPK v2: was auto-CUDA, which
        caused unexpected GPU allocation when the caller wanted CPU
        sample-time computation. Caller should pass device explicitly.)
      - Same updated ``t_norm`` formula as ``make_penalty_features``.
      - Round-trip checked against the 5 known families: max diff
        <= 5e-4 in ``shape_features`` and <= 1e-6 in ``d0``.
    """
    if device is None:
        device = torch.device("cpu")
    dtype = torch.float32

    if grid_t is None:
        grid_t = default_penalty_grid(device=device, dtype=dtype)
    G = grid_t.numel()

    B = len(segments_grid)
    Q = len(segments_grid[0]) if B > 0 else 0
    rho = torch.zeros(B, Q, G, device=device, dtype=dtype)
    drho = torch.zeros(B, Q, G, device=device, dtype=dtype)
    d0 = torch.zeros(B, Q, device=device, dtype=dtype)

    grid_np = grid_t.detach().cpu().numpy().astype("float64")
    for b in range(B):
        if len(segments_grid[b]) != Q:
            raise ValueError(
                f"segments_grid[{b}] length {len(segments_grid[b])} != Q={Q}"
            )
        for q in range(Q):
            segs = segments_grid[b][q]
            for g_idx, tg in enumerate(grid_np):
                seg_match = None
                for seg in segs:
                    if seg.lo <= tg <= seg.hi:
                        seg_match = seg
                        break
                if seg_match is None:
                    seg_match = segs[-1]
                rho[b, q, g_idx] = float(seg_match.eval_rho(float(tg)))
                drho[b, q, g_idx] = float(seg_match.eval_drho(float(tg)))
            d0[b, q] = float(segs[0].eval_drho(segs[0].lo))

    dt = (grid_t[1:] - grid_t[:-1]).clamp_min(eps)
    curv_mid = (drho[..., 1:] - drho[..., :-1]) / dt.view(1, 1, -1)
    curv = torch.cat([curv_mid[..., :1], curv_mid], dim=-1)

    # Same t_norm formula as make_penalty_features (FPK v2 patch).
    LOG_T_EPS = 1e-3
    log_denom = math.log(2.0 + LOG_T_EPS)
    t_norm = torch.log(grid_t.clamp_min(0.0) + LOG_T_EPS) / log_denom
    t_feat = t_norm.view(1, 1, G).expand(B, Q, G)
    rho_feat = torch.tanh(rho / 4.0)
    drho_feat = torch.tanh(drho / 2.0)
    curv_feat = torch.tanh(curv / 2.0)
    grid_feat = torch.stack([rho_feat, drho_feat], dim=-1)

    area = torch.trapezoid(rho, grid_t, dim=-1) / grid_t[-1].clamp_min(eps)
    tail_slope = drho[..., -1]
    near_slope = drho[..., 1]
    has_kink = (d0 > 1e-6).to(dtype)
    meta = torch.stack(
        [
            torch.tanh(d0 / 2.0),
            torch.tanh(near_slope / 2.0),
            torch.tanh(tail_slope / 2.0),
            torch.tanh(area / 4.0),
            2.0 * has_kink - 1.0,
        ],
        dim=-1,
    )

    shape_features = torch.cat(
        [grid_feat.reshape(B, Q, G * 2), meta],
        dim=-1,
    )
    return {
        "shape_features": shape_features,
        "d0": d0,
        "grid_t": grid_t,
        "grid_rho": rho,
        "grid_drho": drho,
        "grid_curv": curv,
        "meta": meta,
    }


class PenaltyShapeEncoder(nn.Module):
    """Bounded deterministic penalty-shape features -> learned penalty code.

    A small LayerNorm-MLP that maps the (B, Q, D_raw) shape-feature
    tensor produced by ``make_penalty_features`` into a (B, Q, out_dim)
    code. Zero-initializing the last layer means the initial conditioning
    is dominated by the lambda features, so the legacy Lasso behavior is
    recovered at step 0 and the model only diverges from it as the
    shape encoder learns to be useful.

    FPK v2: ``depth`` and ``hidden`` defaults raised (1 -> 2 hidden
    layers, 64 -> 128 width) per A5 decision. The 101-dim input is
    enough capacity to benefit from a second nonlinearity.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 24,
        hidden: int = 128,
        depth: int = 2,
        zero_init_last: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"PenaltyShapeEncoder depth must be >= 1, got {depth}")
        self.in_dim = in_dim
        self.out_dim = out_dim

        layers: List[nn.Module] = [nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

        if zero_init_last:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, shape_features: Tensor) -> Tensor:
        if shape_features.dim() != 3:
            raise ValueError(
                f"shape_features must be (B, Q, D), got {tuple(shape_features.shape)}"
            )
        if shape_features.shape[-1] != self.in_dim:
            raise ValueError(
                f"shape_features last dim {shape_features.shape[-1]} != in_dim {self.in_dim}"
            )
        B, Q, D = shape_features.shape
        return self.net(shape_features.reshape(B * Q, D)).view(B, Q, -1)


class PenaltyShapeAwareCondEmbed(nn.Module):
    """Patched conditioning embed: bounded lambda + penalty-shape code.

    The legacy version (kept above in Section A as `PenaltyAwareCondEmbed`)
    fed raw
    penalty parameters and a discrete family ID. That made non-Lasso
    Stage B training collapse. This version instead:

      - drops the raw `penalty_params` from the conditioning path,
      - keeps `penalty_type_id` only as an optional embedding (off by
        default; we want the model to learn from `shape_features`),
      - introduces a `threshold_ratio = lambda * d0 / lambda_max` feature
        so that Ridge (d0=0) gets a flat threshold signal and Lasso
        gets a familiar one,
      - normalizes N and P with centered logs that are stable across
        the (small N, P=50) Stage B regime and the eventual large-P regime.
    """

    def __init__(
        self,
        out_dim: int,
        hidden: int = 96,            # unused after FPK v2 Linear simplification; kept for API stability
        eps: float = 1e-12,
        use_np: bool = False,
        penalty_shape_in_dim: int = DEFAULT_SHAPE_FEATURE_DIM,
        penalty_shape_code_dim: int = 24,
        penalty_shape_hidden: int = 128,    # FPK v2: was 64
        penalty_shape_depth: int = 2,       # FPK v2: was 1
        use_penalty_type_embed: bool = False,
        penalty_embed_dim: int = 8,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.use_np = use_np
        self.use_penalty_type_embed = use_penalty_type_embed

        # Bounded lambda features (6d) Z-scaling mean & std
        self.register_buffer("_lambda_mean", torch.tensor([-0.425504, 0.208059, -0.312839, -0.675248, 0.341111, -3.300555], dtype=torch.float32))
        self.register_buffer("_lambda_scale", torch.tensor([0.436441, 0.201869, 0.141745, 0.632945, 0.213211, 1.658484], dtype=torch.float32))
        from shape_feature_normalization import SHAPE_FEATURE_MEAN, SHAPE_FEATURE_SCALE
        if len(SHAPE_FEATURE_MEAN) != penalty_shape_in_dim or len(SHAPE_FEATURE_SCALE) != penalty_shape_in_dim:
            raise ValueError(
                "shape feature normalization dimension mismatch: "
                f"{len(SHAPE_FEATURE_MEAN)}/{len(SHAPE_FEATURE_SCALE)} vs {penalty_shape_in_dim}"
            )
        self.register_buffer("_shape_mean", torch.tensor(SHAPE_FEATURE_MEAN, dtype=torch.float32))
        self.register_buffer("_shape_scale", torch.tensor(SHAPE_FEATURE_SCALE, dtype=torch.float32))

        self.penalty_shape_encoder = PenaltyShapeEncoder(
            in_dim=penalty_shape_in_dim,
            out_dim=penalty_shape_code_dim,
            hidden=penalty_shape_hidden,
            depth=penalty_shape_depth,
            zero_init_last=True,
        )

        if use_penalty_type_embed:
            self.pen_embed = nn.Embedding(NUM_PENALTY_TYPES, penalty_embed_dim)
            type_dim = penalty_embed_dim
        else:
            self.pen_embed = None
            type_dim = 0

        # 6 lambda features + (optional) 3 N/P features.
        numeric_dim = 6 + (3 if use_np else 0)
        in_dim = numeric_dim + penalty_shape_code_dim + type_dim

        # FiLM path: numeric lambda/context features modulate the penalty-shape code.
        # Zero-init keeps the initial behavior close to the concat baseline
        # (shape_code_film = shape_code) while allowing learned multiplicative
        # and additive injection of lambda-dependent context.
        self.numeric_norm = nn.LayerNorm(numeric_dim)
        self.shape_film = nn.Sequential(
            nn.Linear(numeric_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * penalty_shape_code_dim),
        )
        nn.init.zeros_(self.shape_film[-1].weight)
        nn.init.zeros_(self.shape_film[-1].bias)

        # FPK v2 A5: simplified conditioning MLP. Was
        #   LayerNorm -> Linear(in_dim, hidden) -> GELU -> Linear(hidden, out_dim)
        # The shape code already passes through a 2-layer GELU MLP inside
        # PenaltyShapeEncoder; an extra GELU here was redundant capacity.
        # Now the conditioning path is a single affine over the concat:
        #   LayerNorm -> Linear(in_dim, out_dim)
        # This is the "Linear replacement" decision (FiLM is held back for
        # a later iteration).
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
        )

    def forward(
        self,
        lambdas: Tensor,
        lambda_max: Tensor,
        N: Optional[Union[int, float, Tensor]] = None,
        P: Optional[Union[int, float, Tensor]] = None,
        penalty_type_id: Optional[Tensor] = None,
        penalty_shape_features: Optional[Tensor] = None,
        penalty_d0: Optional[Tensor] = None,
    ) -> Tensor:
        if lambdas.dim() != 2:
            raise ValueError(f"lambdas must be (B,Q), got {tuple(lambdas.shape)}")
        if lambda_max.shape != lambdas.shape:
            raise ValueError(
                "lambda_max must match lambdas shape, got "
                f"{tuple(lambda_max.shape)} vs {tuple(lambdas.shape)}"
            )
        if penalty_shape_features is None:
            raise ValueError(
                "penalty_shape_features is required; call "
                "krylov.penalty.features.make_penalty_features first."
            )

        device = self.mlp[-1].weight.device
        dtype = self.mlp[-1].weight.dtype
        B, Q = lambdas.shape
        eps = self.eps

        lam = lambdas.to(device=device, dtype=torch.float32).clamp_min(eps)
        lmx = lambda_max.to(device=device, dtype=torch.float32).clamp_min(eps)
        ratio = (lam / lmx).clamp_min(eps)

        if penalty_d0 is None:
            # Lasso-equivalent fallback. Useful for unit-tested smoke runs;
            # production training always passes penalty_d0.
            d0 = torch.ones(B, Q, device=device, dtype=torch.float32)
        else:
            d0 = _broadcast_queries_like(
                penalty_d0, B, Q, "penalty_d0",
                device=device, dtype=torch.float32,
            ).clamp_min(0.0)

        threshold_ratio = (lam * d0.clamp_min(eps) / lmx).clamp_min(eps)

        # Six bounded lambda features (6d):
        # 1. log(lambda/lambda_max)        -- normalized strength
        # 2. sqrt(lambda/lambda_max)       -- bounded linear strength
        # 3. tanh(log(lambda_max)/5)       -- bounded absolute scale
        # 4. log(lambda*d0/lambda_max)     -- threshold strength
        # 5. tanh(d0/2)                    -- penalty kink magnitude
        # 6. log(original lambda + 1e-7)   -- log absolute scale
        lambda_feat = torch.stack(
            [
                ratio.log().clamp(-8.0, 1.0) / 4.0,
                ratio.clamp(0.0, 10.0).sqrt() / math.sqrt(10.0),
                torch.tanh(lmx.log() / 5.0),
                threshold_ratio.log().clamp(-8.0, 1.0) / 4.0,
                torch.tanh(d0 / 2.0),
                (lam + 1e-7).log(),
            ],
            dim=-1,
        )

        # Z-scaling for lambda features. Add 0.1 to std to avoid amplifying tiny-variance dims.
        lambda_normed = (lambda_feat - self._lambda_mean) / (self._lambda_scale + 0.1)
        numeric_parts = [lambda_normed]

        if self.use_np:
            Nt = _broadcast_queries_like(
                1.0 if N is None else N, B, Q, "N",
                device=device, dtype=torch.float32,
            ).clamp_min(1.0)
            Pt = _broadcast_queries_like(
                1.0 if P is None else P, B, Q, "P",
                device=device, dtype=torch.float32,
            ).clamp_min(1.0)

            zN = ((Nt.log() - math.log(500.0)) / 2.0).clamp(-3.0, 3.0)
            zP = ((Pt.log() - math.log(50.0)) / 2.0).clamp(-3.0, 3.0)
            zA = (
                ((Nt / Pt).clamp_min(eps).log() - math.log(10.0)) / 2.0
            ).clamp(-3.0, 3.0)
            numeric_parts.append(torch.stack([zN, zP, zA], dim=-1))
        numeric_feat = torch.cat(numeric_parts, dim=-1)
        parts = [numeric_feat]

        # Penalty-shape code (zero-initialized last layer at boot).
        shape_features = penalty_shape_features.to(device=device, dtype=torch.float32)
        shape_features = (shape_features - self._shape_mean) / (self._shape_scale + 0.1)
        z_pen = self.penalty_shape_encoder(shape_features)
        gamma, beta = self.shape_film(self.numeric_norm(numeric_feat)).chunk(2, dim=-1)
        z_pen = z_pen * (1.0 + torch.tanh(gamma)) + beta
        parts.append(z_pen)

        if self.use_penalty_type_embed:
            if penalty_type_id is None:
                pid = torch.zeros(B, Q, device=device, dtype=torch.long)
            else:
                pid = penalty_type_id.to(device=device, dtype=torch.long)
                if pid.shape != (B, Q):
                    raise ValueError(
                        f"penalty_type_id must have shape (B, Q), got {tuple(pid.shape)}"
                    )
            parts.append(self.pen_embed(pid).to(torch.float32))

        feat = torch.cat(parts, dim=-1).to(dtype)
        return self.mlp(feat.reshape(B * Q, -1)).view(B, Q, -1)


class BoundedLearnedProxOp(nn.Module):
    """Patched penalty-conditioned shrinkage.

    Two changes from the legacy LearnedProxOp:

    1. Token features. Legacy used `[abs(u), tau, abs(u)/tau]`. The
       ratio explodes for tiny tau (Ridge with d0=0 reaches there at
       initialization). We replace with bounded substitutes:
         - mag_frac = |u| / (|u| + tau)  in [0, 1]
         - log_tau  in clamped log scale
         - log_ratio = log(|u|) - log(tau), clamped
    2. LayerNorm on the input. Cheap but stabilizes the head when
       conditioning scales drift early in training.

    The output formula is unchanged so that the zero-initialized last
    layer reproduces standard soft-thresholding at init.
    """

    def __init__(
        self,
        cond_dim: int,
        hidden: int = 64,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        in_dim = 3 + cond_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, u_hat: Tensor, tau_hat: Tensor, cond: Tensor) -> Tensor:
        if u_hat.dim() != 2 or tau_hat.dim() != 2 or cond.dim() != 2:
            raise ValueError("LearnedProxOp expects (BQ,P), (BQ,P), (BQ,D)")

        BQ, P = u_hat.shape
        tau = tau_hat.expand(BQ, P).clamp_min(1e-12)
        au = u_hat.abs()

        log_tau = tau.log().clamp(-12.0, 6.0) / 6.0
        log_ratio = ((au + 1e-12).log() - tau.log()).clamp(-8.0, 8.0) / 4.0
        mag_frac = au / (au + tau + 1e-12)  # bounded in [0, 1]

        cond_feat = cond.unsqueeze(1).expand(BQ, P, -1)
        tok = torch.cat(
            [
                mag_frac.unsqueeze(-1),
                log_tau.unsqueeze(-1),
                log_ratio.unsqueeze(-1),
                cond_feat,
            ],
            dim=-1,
        )
        raw = self.net(tok)
        scale = raw[..., 0].clamp(-2.0, 2.0)
        shift = raw[..., 1].clamp(-2.0, 2.0)

        return torch.sign(u_hat) * torch.relu(
            au * (1.0 + scale) - tau * (1.0 + shift)
        )




# ##############################################################################
# SECTION C -- Integration
# (merged from krylov/penalty/integration.py)
#
# `IntegratedRidgeKrylovDecoder` subclasses the Section-A backbone and swaps
# in the Section-B conditioning. The lazy-import / lazy-class-binding plumbing
# from the original integration.py is dropped: every symbol now lives in this
# one module, so the class is defined directly and the factory references
# local names.
# ##############################################################################
class IntegratedRidgeKrylovDecoder(KrylovRowSpaceQueryDecoder):
    """Patched decoder. Inherits the Krylov backbone unchanged.

    Behavioral differences from the legacy class:
      * `cond_embed` is the shape-aware `PenaltyAwareCondEmbed`.
      * `learned_prox` uses bounded token features.
      * `decode_queries` computes `tau_base = lambda * d0 / gram_diag`.
      * `decode_queries`, `solve_queries`, and `forward` all accept
        optional `penalty_shape_features` and `penalty_d0` so the
        caller can cache them across many lambda queries.
    """

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        patched = _read_patched_cfg(cfg)
        self._patched_cfg = patched

        if not patched.use_penalty_shape_encoding:
            # Strict legacy mode. Useful for the Phase 0 golden test.
            return

        # Replace the conditioning module. `condition_np` is read
        # from the legacy cfg field, not the patched overrides.
        self.cond_embed = PenaltyShapeAwareCondEmbed(
            out_dim=cfg.cond_dim,
            hidden=cfg.cond_hidden,
            eps=cfg.cond_eps,
            use_np=bool(getattr(cfg, "condition_np", False)),
            penalty_shape_in_dim=patched.penalty_shape_in_dim,
            penalty_shape_code_dim=patched.penalty_shape_code_dim,
            penalty_shape_hidden=patched.penalty_shape_hidden,
            use_penalty_type_embed=patched.use_penalty_type_embed,
            penalty_embed_dim=cfg.penalty_embed_dim,
        )

        if patched.prox_use_bounded_inputs:
            self.learned_prox = BoundedLearnedProxOp(
                cond_dim=cfg.cond_dim,
                hidden=cfg.prox_hidden,
                zero_init=cfg.zero_init_learned_prox,
            )

    # ------------------------------------------------------------------
    # Public API: decode_queries / solve_queries / forward all accept
    # the two new optional tensors. The base class methods don't, so
    # we have to override all three.
    # ------------------------------------------------------------------

    def decode_queries(
        self,
        encoded: Dict[str, Any],
        lambdas: Tensor,
        lambda_max: Optional[Tensor] = None,
        penalty_type_id: Optional[Union[str, int, Tensor]] = None,
        penalty_params: Optional[Tensor] = None,
        penalty_shape_features: Optional[Tensor] = None,
        penalty_d0: Optional[Tensor] = None,
        return_aux: bool = False,
        return_row_state: bool = False,
    ):
        patched = self._patched_cfg
        if not patched.use_penalty_shape_encoding:
            return super().decode_queries(
                encoded=encoded,
                lambdas=lambdas,
                lambda_max=lambda_max,
                penalty_type_id=penalty_type_id,
                penalty_params=penalty_params,
                return_aux=return_aux,
                return_row_state=return_row_state,
            )

        B = encoded["B"]
        P = encoded["P"]
        N = encoded["N"]
        device = encoded["Xty"].device

        lam = _as_2d_queries(
            lambdas, B, "lambdas", device=device, dtype=torch.float32,
        )
        _, Q = lam.shape

        lmx_source = encoded["lambda_max"] if lambda_max is None else lambda_max
        lmx = _broadcast_queries_like(
            lmx_source, B, Q, "lambda_max",
            device=device, dtype=torch.float32,
        )

        # ---- Auto-build features if caller didn't precompute --------
        # Use the legacy resolvers so scalar/string/(B,)/(Q,)/(B,Q)
        # forms all work the same way they do for the legacy decoder.
        if penalty_shape_features is None or penalty_d0 is None:
            if penalty_type_id is None:
                raise ValueError(
                    "Patched decode_queries requires either "
                    "penalty_shape_features + penalty_d0, or "
                    "penalty_type_id (+ optional penalty_params)."
                )
            pid_t = _resolve_penalty_type_grid(
                penalty_type_id, B=B, Q=Q, device=device,
            )
            pp_t = _resolve_penalty_param_grid(
                penalty_params,
                B=B, Q=Q, device=device,
                dtype=torch.float32,
                out_dim=1,
            )
            feats = make_penalty_features(
                penalty_type_id=pid_t,
                penalty_params=pp_t,
            )
            penalty_shape_features = feats["shape_features"]
            penalty_d0 = feats["d0"]

        # Normalize the optional explicit penalty_type_id for the
        # cond_embed call (used only when use_penalty_type_embed=True).
        pid_for_embed = (
            _resolve_penalty_type_grid(
                penalty_type_id, B=B, Q=Q, device=device,
            )
            if penalty_type_id is not None
            else None
        )

        # ---- Patched conditioning -----------------------------------
        cond = self.cond_embed(
            lambdas=lam,
            lambda_max=lmx,
            N=float(N),
            P=float(P),
            penalty_type_id=pid_for_embed,
            penalty_shape_features=penalty_shape_features,
            penalty_d0=penalty_d0,
        )

        A = self.query_mixer(cond)
        with _maybe_disable_autocast(device.type):
            basis_proj_f = encoded["basis_proj"].float()
            S = torch.einsum(
                "bqcm,bmp->bqcp",
                A.float(),
                basis_proj_f,
            )

        model_dtype = cond.dtype
        S = S.to(model_dtype)
        Xty = encoded["Xty"].to(model_dtype)
        gram_diag = encoded["gram_diag"].to(model_dtype)
        log_abs_Xty = encoded["log_abs_Xty"].to(model_dtype)

        u_resid, log_tau_scale = self.feature_head(
            S=S, Xty=Xty, gram_diag=gram_diag,
            log_abs_Xty=log_abs_Xty, cond=cond,
        )

        denom = gram_diag.clamp_min(self.cfg.diag_eps)
        u_diag = Xty[:, None, :] / denom[:, None, :]

        # ---- THE CRITICAL PATCH -------------------------------------
        # Legacy:  tau_base = lambda / denom       (assumes d0 = 1)
        # Patched: tau_base = lambda * d0 / denom  (Ridge: d0=0 -> 0)
        d0 = _broadcast_queries_like(
            penalty_d0, B, Q, "penalty_d0",
            device=device, dtype=torch.float32,
        ).clamp_min(0.0)
        tau_base = (
            lam.to(model_dtype).unsqueeze(-1)
            * d0.to(model_dtype).unsqueeze(-1)
            / denom[:, None, :]
        )

        # ---- Correlation-aware ridge base (unchanged) ---------------
        if self.cfg.use_krylov_ridge_base:
            base_raw = self.base_head(cond)
            log_alpha_scale = base_raw[..., 0].clamp(
                self.cfg.base_alpha_log_min,
                self.cfg.base_alpha_log_max,
            )
            mix = torch.sigmoid(base_raw[..., 1])

            alpha = lam * torch.exp(log_alpha_scale) + self.cfg.diag_eps

            with _maybe_disable_autocast(device.type):
                T = encoded["T_krylov"].float()
                Uy = encoded["Uy_krylov"].float()
                Phi = encoded["basis_proj"].float()
                M = T.shape[-1]

                eye = torch.eye(
                    M, device=device, dtype=torch.float32
                ).view(1, 1, M, M)
                rhs = Uy[:, None, :, None].expand(B, Q, M, 1)
                sys = (
                    T[:, None, :, :]
                    + alpha[..., None, None].float() * eye
                    + self.cfg.diag_eps * eye
                )

                coeff = torch.linalg.solve(sys, rhs).squeeze(-1)
                u_krylov = torch.einsum("bqm,bmp->bqp", coeff, Phi)

            u_base = (
                mix.unsqueeze(-1) * u_krylov.to(model_dtype)
                + (1.0 - mix.unsqueeze(-1)) * u_diag
            )
        else:
            mix = torch.zeros(B, Q, device=device, dtype=model_dtype)
            alpha = lam.new_full((B, Q), float("nan"))
            u_base = u_diag

        u_hat = u_base + u_resid
        tau_hat = tau_base * torch.exp(log_tau_scale)

        beta = self.learned_prox(
            u_hat.reshape(B * Q, P),
            tau_hat.reshape(B * Q, P),
            cond.reshape(B * Q, -1),
        ).view(B, Q, P)

        fm = encoded["feature_mask"]
        if fm is not None:
            beta = beta * fm[:, None, :].to(beta.dtype)
            u_hat = u_hat * fm[:, None, :].to(u_hat.dtype)

        if not return_aux:
            return beta

        out = {
            "beta_q": beta,
            "u_base_q": u_base,
            "u_hat_q": u_hat,
            "tau_hat_q": tau_hat,
            "mixing_weights": A,
            "query_embedding": cond,
            "channel_projection_q": S,
            "base_mix_q": mix,
            "base_alpha_q": alpha.to(model_dtype),
            "penalty_d0_q": d0,
        }
        if return_row_state:
            if "U" not in encoded:
                raise ValueError(
                    "return_row_state=True requires cfg.retain_U=True"
                )
            with _maybe_disable_autocast(device.type):
                Z = torch.einsum(
                    "bqcm,bmn->bqcn",
                    A.float(),
                    encoded["U"].float(),
                )
            out["row_state_q"] = Z.to(model_dtype)
        return out

    # ------------------------------------------------------------------
    # solve_queries: thread the new kwargs through to decode_queries.
    # ------------------------------------------------------------------
    def solve_queries(
        self,
        encoded: Dict[str, Any],
        lambdas: Tensor,
        lambda_max: Optional[Tensor] = None,
        penalty_type_id: Optional[Union[str, int, Tensor]] = None,
        penalty_params: Optional[Tensor] = None,
        penalty_shape_features: Optional[Tensor] = None,
        penalty_d0: Optional[Tensor] = None,
        return_aux: bool = False,
        return_row_state: bool = False,
    ):
        dec = self.decode_queries(
            encoded=encoded,
            lambdas=lambdas,
            lambda_max=lambda_max,
            penalty_type_id=penalty_type_id,
            penalty_params=penalty_params,
            penalty_shape_features=penalty_shape_features,
            penalty_d0=penalty_d0,
            return_aux=True,
            return_row_state=return_row_state,
        )

        beta_q = dec["beta_q"]
        intercept_q = recover_intercept(
            x_mean=encoded["x_mean"],
            y_mean=encoded["y_mean"],
            beta=beta_q,
        )

        beta = beta_q[:, 0, :] if beta_q.shape[1] == 1 else beta_q
        intercept = intercept_q[:, 0] if intercept_q.shape[1] == 1 else intercept_q

        if not return_aux:
            return beta

        out: Dict[str, Any] = dict(dec)
        out["beta"] = beta
        out["intercept"] = intercept
        out["intercept_q"] = intercept_q
        out["basis_diagnostics"] = encoded["basis_diagnostics"]
        out["lambda_max_dataset"] = encoded["lambda_max"]
        return out

    # ------------------------------------------------------------------
    # forward: same pattern; the only added kwargs are the two
    # precomputed-feature tensors.
    # ------------------------------------------------------------------
    def forward(
        self,
        X: Tensor,
        y: Tensor,
        lambdas: Tensor,
        lambda_max: Optional[Tensor] = None,
        feature_mask: Optional[Tensor] = None,
        penalty_type_id: Optional[Union[str, int, Tensor]] = None,
        penalty_params: Optional[Tensor] = None,
        penalty_shape_features: Optional[Tensor] = None,
        penalty_d0: Optional[Tensor] = None,
        return_aux: bool = False,
        return_row_state: bool = False,
    ):
        precomputed = self.precompute(X, y, feature_mask=feature_mask)
        encoded = self.encode_dataset(precomputed)
        return self.solve_queries(
            encoded=encoded,
            lambdas=lambdas,
            lambda_max=lambda_max,
            penalty_type_id=penalty_type_id,
            penalty_params=penalty_params,
            penalty_shape_features=penalty_shape_features,
            penalty_d0=penalty_d0,
            return_aux=return_aux,
            return_row_state=return_row_state,
        )


# ---------------------------------------------------------------------------
# Patched-only config fields
# ---------------------------------------------------------------------------
# These must NOT shadow any field on `KrylovQueryDecoderConfig`. They are
# attached onto the legacy config object by `make_integrated_model` and read
# back inside `IntegratedRidgeKrylovDecoder.__init__` via `_read_patched_cfg`.
# ---------------------------------------------------------------------------

@dataclass
class PatchedConfigOverrides:
    # Master switch. When False, the decoder behaves identically to the
    # legacy v3 model and accepts legacy checkpoints.
    use_penalty_shape_encoding: bool = True

    # Conditioning encoder.
    penalty_shape_code_dim: int = 24
    penalty_shape_hidden: int = 64
    penalty_shape_in_dim: int = DEFAULT_SHAPE_FEATURE_DIM
    use_penalty_type_embed: bool = False
    use_raw_penalty_params: bool = False  # kept for ablation only

    # Prox head.
    prox_use_bounded_inputs: bool = True


# Legacy fields whose integrated default differs from the legacy default.
# `make_integrated_model` applies these unless the caller overrides them.
_V4_LEGACY_DEFAULT_OVERRIDES: Dict[str, Any] = {
    # For the N=500, P=50 Ridge run the legacy default `condition_np=True`
    # is actively harmful (N and P are constant, so the N/P conditioning
    # input is constant noise).
    "condition_np": False,
}


def _read_patched_cfg(cfg: Any) -> PatchedConfigOverrides:
    """Pull patched-only fields off any config object, with defaults."""
    out = PatchedConfigOverrides()
    for name in PatchedConfigOverrides.__dataclass_fields__:
        if hasattr(cfg, name):
            setattr(out, name, getattr(cfg, name))
    return out


def make_integrated_model(
    legacy_cfg_dict: Optional[Dict[str, Any]] = None,
    device: Union[str, torch.device] = "cpu",
    **overrides: Any,
):
    """Build the integrated Ridge model. Returns ``(model, full_config_dict)``.

    In-module replacement for the old ``integration.make_v4_model``.

    ``overrides`` accepts any field on ``KrylovQueryDecoderConfig`` plus any
    field on ``PatchedConfigOverrides``. Unknown keys raise a clear error so
    silent typos do not sneak into training scripts.
    """
    import dataclasses as _dc

    legacy_field_names = set(KrylovQueryDecoderConfig.__dataclass_fields__.keys())
    patched_field_names = set(PatchedConfigOverrides.__dataclass_fields__.keys())

    # The two field sets must stay disjoint -- a shadowed field would create
    # a silent double-source-of-truth bug.
    overlap = legacy_field_names & patched_field_names
    if overlap:
        raise RuntimeError(
            f"PatchedConfigOverrides must not shadow legacy fields: {sorted(overlap)}."
        )

    # 1. Start from the legacy dataclass defaults.
    base: Dict[str, Any] = _dc.asdict(KrylovQueryDecoderConfig())

    # 2. Apply integrated-default overrides (e.g. condition_np=False) unless
    #    the caller is going to set them explicitly.
    user_keys = set(overrides)
    if legacy_cfg_dict:
        user_keys |= set(legacy_cfg_dict)
    for k, v in _V4_LEGACY_DEFAULT_OVERRIDES.items():
        if k not in user_keys:
            base[k] = v

    # 3. Apply the user-supplied legacy dict (e.g. a loaded checkpoint cfg).
    if legacy_cfg_dict:
        base.update(legacy_cfg_dict)

    # 4. Apply explicit overrides (highest precedence).
    base.update(overrides)

    # 5. Partition into legacy / patched and reject unknowns.
    legacy_kwargs = {k: v for k, v in base.items() if k in legacy_field_names}
    patched_kwargs = {k: v for k, v in base.items() if k in patched_field_names}
    extra_kwargs = {
        k: v for k, v in base.items()
        if k not in legacy_field_names and k not in patched_field_names
    }
    if extra_kwargs:
        raise ValueError(f"Unknown config overrides: {sorted(extra_kwargs)}")

    # 6. Build the legacy config and attach patched fields onto it.
    legacy_cfg = KrylovQueryDecoderConfig(**legacy_kwargs)
    defaults = PatchedConfigOverrides()
    for k in patched_field_names:
        setattr(legacy_cfg, k, getattr(defaults, k))
    for k, v in patched_kwargs.items():
        setattr(legacy_cfg, k, v)

    # 7. Instantiate the integrated decoder.
    model = IntegratedRidgeKrylovDecoder(legacy_cfg).to(torch.device(device))

    # 8. Return a config dict reflecting the actually-applied values.
    full_cfg = dict(legacy_kwargs)
    for k in patched_field_names:
        full_cfg[k] = getattr(legacy_cfg, k)
    return model, full_cfg


# Backwards-compatible alias for callers expecting the old factory name.
make_v4_model = make_integrated_model


# ##############################################################################
# SECTION D -- Penalty representation, scalar prox, random-theta sampler,
#              and the mid-heavy PenaltySampler.
#
# Absorbs the dg_fpk modules:
#   - penalty_segments.py  (Segment, SegmentList, family constructors,
#                           theta_to_segments)
#   - scalar_prox.py       (scalar_abs_penalty_prox + closed-form helpers)
#   - derivative_staircase_sampler.py  (sample_random_theta + curriculum stages)
#
# Adds: a new PenaltySampler with
#   * 80% random theta / 20% known families
#   * mid-heavy lambda mixture (50% [1e-2,1e-1] + 30% [1e-3,1e-2] + 20% [1e-1,1])
#   * K uniform in [3, 64] (each random sample picks K independently)
#   * stage curriculum easy -> medium -> hard with d0 bucket weights
#
# Conventions:
#   * standardized rho(0)=0, rho(1)=1
#   * objective convention: lam * sum_j rho(|beta_j|), NO division by P
#   * all internal numpy where convenient; torch tensors only where the
#     caller needs them (sampled batches return torch tensors on device).
# ##############################################################################


# -----------------------------------------------------------------------------
# D.1 Segment dataclass + generic evaluators
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Segment:
    """rho(t) = c + d * t + 0.5 * h * t^2  on  [lo, hi].

    Note (A1, DECISION_PENDING): we keep the piecewise-quadratic
    representation INTERNALLY for the reference solver / scalar prox /
    KKT residual. The conditioning encoder does NOT consume (c, d, h)
    tokens directly; it consumes the grid-evaluated penalty features
    built by ``make_penalty_features`` / ``make_penalty_features_from_segments``.
    Until the (b) path is signed off, do not surface ``Segment`` in
    encoder-facing docstrings.
    """
    lo: float
    hi: float           # may be math.inf
    c: float
    d: float
    h: float

    def eval_rho(self, t: float) -> float:
        return self.c + self.d * t + 0.5 * self.h * t * t

    def eval_drho(self, t: float) -> float:
        return self.d + self.h * t

    def contains(self, t: float, *, atol: float = 0.0) -> bool:
        return (self.lo - atol) <= t <= (self.hi + atol)


SegmentList = List[Segment]


def eval_piecewise_rho(t: float, segments: SegmentList) -> float:
    """Evaluate rho(t) at a single scalar t >= 0."""
    if t < 0.0:
        raise ValueError(f"eval_piecewise_rho requires t >= 0, got {t}")
    for seg in segments:
        if seg.lo <= t <= seg.hi:
            return seg.eval_rho(t)
    last = segments[-1]
    return last.eval_rho(t)


def eval_piecewise_drho_at(
    t: float, segments: SegmentList, *, knot_tol: float = 1e-12
) -> Tuple[float, float]:
    """Return (left, right) derivative of rho at t."""
    if t < 0.0:
        raise ValueError(f"eval_piecewise_drho_at requires t >= 0, got {t}")
    if t <= knot_tol:
        first = segments[0]
        d0 = first.eval_drho(first.lo)
        return (d0, d0)

    left_seg = None
    right_seg = None
    for seg in segments:
        if seg.hi >= t - knot_tol and seg.lo <= t + knot_tol:
            if abs(seg.hi - t) <= knot_tol and right_seg is None:
                left_seg = seg
            elif abs(seg.lo - t) <= knot_tol and right_seg is None:
                right_seg = seg
            elif seg.lo < t < seg.hi:
                left_seg = seg
                right_seg = seg

    if left_seg is None and right_seg is None:
        last = segments[-1]
        return (last.eval_drho(t), last.eval_drho(t))
    if left_seg is None:
        left_seg = right_seg
    if right_seg is None:
        right_seg = left_seg
    return (left_seg.eval_drho(t), right_seg.eval_drho(t))


def d0_from_segments(segments: SegmentList) -> float:
    """Right derivative at 0+, used everywhere as the inactive-KKT threshold."""
    return segments[0].eval_drho(segments[0].lo)


def _scale_segments_by_constant(segments: SegmentList, s: float) -> SegmentList:
    if s == 0.0:
        raise ValueError("cannot scale segments by zero")
    inv = 1.0 / float(s)
    return [Segment(seg.lo, seg.hi, seg.c * inv, seg.d * inv, seg.h * inv) for seg in segments]


# -----------------------------------------------------------------------------
# D.2 Known-family segment constructors (all standardized so rho(1) = 1)
# -----------------------------------------------------------------------------

def lasso_segments() -> SegmentList:
    """rho(t) = t, d0 = 1, rho(1) = 1."""
    return [Segment(lo=0.0, hi=math.inf, c=0.0, d=1.0, h=0.0)]


def ridge_segments() -> SegmentList:
    """rho_std(t) = t^2 (raw 0.5 t^2 standardized by rho_raw(1) = 0.5).

    d0 = 0 (this is the magic that disables soft thresholding for Ridge).
    """
    return [Segment(lo=0.0, hi=math.inf, c=0.0, d=0.0, h=2.0)]


def elastic_net_segments(alpha: float) -> SegmentList:
    """rho_raw(t) = alpha * t + 0.5 * (1 - alpha) * t^2, then rho(1)=1."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"elastic_net alpha must be in [0, 1], got {alpha}")
    if alpha == 1.0:
        return lasso_segments()
    if alpha == 0.0:
        return ridge_segments()
    s = 0.5 * (alpha + 1.0)
    d_raw = alpha
    h_raw = 1.0 - alpha
    return [Segment(lo=0.0, hi=math.inf, c=0.0, d=d_raw / s, h=h_raw / s)]


def mcp_segments(gamma: float) -> SegmentList:
    """MCP standardized to rho(1) = 1."""
    if gamma <= 1.0:
        raise ValueError(
            f"MCP gamma must be > 1 for the standardization to make sense, got {gamma}"
        )
    rho_raw_at_1 = 1.0 - 1.0 / (2.0 * gamma)
    raw = [
        Segment(lo=0.0, hi=float(gamma), c=0.0, d=1.0, h=-1.0 / gamma),
        Segment(lo=float(gamma), hi=math.inf, c=gamma / 2.0, d=0.0, h=0.0),
    ]
    return _scale_segments_by_constant(raw, rho_raw_at_1)


def scad_segments(a: float) -> SegmentList:
    """SCAD standardized to rho(1) = 1."""
    if a <= 2.0:
        raise ValueError(f"SCAD a must be > 2, got {a}")
    mid_c = -1.0 / (2.0 * (a - 1.0))
    mid_d = a / (a - 1.0)
    mid_h = -1.0 / (a - 1.0)
    raw = [
        Segment(lo=0.0, hi=1.0, c=0.0, d=1.0, h=0.0),
        Segment(lo=1.0, hi=float(a), c=mid_c, d=mid_d, h=mid_h),
        Segment(lo=float(a), hi=math.inf, c=(a + 1.0) / 2.0, d=0.0, h=0.0),
    ]
    rho_raw_at_1 = 1.0  # SCAD raw already has rho(1) = 1
    return _scale_segments_by_constant(raw, rho_raw_at_1)


# -----------------------------------------------------------------------------
# D.3 Theta-token (random hinge basis) -> SegmentList
# -----------------------------------------------------------------------------

def theta_to_segments(
    theta: Union[torch.Tensor, np.ndarray, Sequence[Sequence[float]]],
    *,
    knot_tol: float = 1e-10,
) -> SegmentList:
    """Convert hinge-basis theta tokens [kind, knot, weight, _] -> segments.

    Token kinds:
      0 = intercept       (rho += w)
      1 = linear slope    (rho += w * t)
      2 = relu hinge      (rho += w * max(0, t - knot))

    Returns piecewise-LINEAR segments (h = 0). Continuity at knots is enforced.
    """
    if isinstance(theta, torch.Tensor):
        theta_np = theta.detach().cpu().double().numpy()
    else:
        theta_np = np.asarray(theta, dtype=np.float64)
    if theta_np.ndim != 2 or theta_np.shape[1] < 3:
        raise ValueError(f"theta must have shape (K, >=3), got {theta_np.shape}")

    c0 = 0.0
    c1 = 0.0
    hinges: List[Tuple[float, float]] = []
    for row in theta_np:
        kind = int(round(float(row[0])))
        knot = float(row[1])
        weight = float(row[2])
        if kind == 0:
            c0 += weight
        elif kind == 1:
            c1 += weight
        elif kind == 2:
            hinges.append((knot, weight))

    base_extra_slope = sum(w for k, w in hinges if k <= knot_tol)
    active = sorted([(k, w) for k, w in hinges if k > knot_tol])

    boundaries: List[float] = [0.0]
    for k, _ in active:
        if not boundaries or k > boundaries[-1] + knot_tol:
            boundaries.append(k)
    boundaries.append(math.inf)

    segments: List[Segment] = []
    slope = c1 + base_extra_slope
    rho_at_left = c0
    active_idx = 0
    for i in range(len(boundaries) - 1):
        lo = boundaries[i]
        hi = boundaries[i + 1]
        seg_c = rho_at_left - slope * lo
        segments.append(Segment(lo=lo, hi=hi, c=seg_c, d=slope, h=0.0))
        if math.isfinite(hi):
            rho_at_left = rho_at_left + slope * (hi - lo)
            added = 0.0
            while active_idx < len(active) and abs(active[active_idx][0] - hi) <= knot_tol:
                added += active[active_idx][1]
                active_idx += 1
            slope += added

    return segments


# -----------------------------------------------------------------------------
# D.4 Scalar proximal operator (generic, segment-based)
# -----------------------------------------------------------------------------

def scalar_abs_penalty_prox(v: float, tau: float, segments: SegmentList) -> float:
    """Solve  argmin_b  0.5 * (b - v)^2 + tau * rho(|b|).

    Works for any segment-representable penalty (Lasso, Ridge, EN, MCP,
    SCAD, random hinge theta) by candidate enumeration. For tau = 0 it
    returns v.
    """
    if tau <= 0.0:
        return float(v)

    s = 1.0 if v >= 0.0 else -1.0
    a = abs(float(v))

    candidates: List[float] = [0.0]
    for seg in segments:
        lo, hi, c, d, h = seg.lo, seg.hi, seg.c, seg.d, seg.h
        if math.isfinite(lo):
            candidates.append(lo)
        if math.isfinite(hi):
            candidates.append(hi)
        denom = 1.0 + tau * h
        if abs(denom) > 1e-10:
            t_star = (a - tau * d) / denom
            if lo - 1e-12 <= t_star <= hi + 1e-12:
                candidates.append(t_star)
        if (not math.isfinite(hi)) and abs(d) < 1e-12 and abs(h) < 1e-12:
            candidates.append(a if a >= lo else lo)

    best_t = 0.0
    best_val = float("inf")
    for t in candidates:
        if t < 0.0 or not math.isfinite(t):
            continue
        rho_t = eval_piecewise_rho(t, segments)
        val = 0.5 * (t - a) * (t - a) + tau * rho_t
        if val < best_val:
            best_val = val
            best_t = t
    return s * best_t


def vectorized_scalar_prox(
    v: np.ndarray,
    tau: np.ndarray,
    segments: SegmentList,
) -> np.ndarray:
    """Apply scalar_abs_penalty_prox component-wise to a vector."""
    v_arr = np.asarray(v, dtype=np.float64).reshape(-1)
    tau_arr = np.broadcast_to(np.asarray(tau, dtype=np.float64), v_arr.shape).copy()
    out = np.empty_like(v_arr)
    for i in range(v_arr.size):
        out[i] = scalar_abs_penalty_prox(float(v_arr[i]), float(tau_arr[i]), segments)
    return out.reshape(np.asarray(v).shape)


def lasso_prox_closed_form(v: float, tau: float) -> float:
    return math.copysign(max(abs(v) - tau, 0.0), v)


def ridge_prox_closed_form(v: float, tau: float) -> float:
    return v / (1.0 + 2.0 * tau)


# -----------------------------------------------------------------------------
# D.5 Curriculum stages for random-theta sampling
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class CurriculumStage:
    name: str
    tv_budget: float           # total variation of derivative jumps
    neg_budget: float          # total negative derivative budget
    jump_scale: float          # Laplace scale for raw jump sampling


EASY = CurriculumStage(name="easy", tv_budget=0.75, neg_budget=0.45, jump_scale=0.05)
MEDIUM = CurriculumStage(name="medium", tv_budget=1.50, neg_budget=1.00, jump_scale=0.12)
HARD = CurriculumStage(name="hard", tv_budget=3.00, neg_budget=2.00, jump_scale=0.25)

STAGES: Dict[str, CurriculumStage] = {"easy": EASY, "medium": MEDIUM, "hard": HARD}


# d0 buckets per plan §3 (after rho_at_1 normalization).
# Weights: [0.10, 0.25, 0.45, 0.20] per fpk_decisions.md §1 (Tier 2).
D0_BUCKETS: Tuple[Tuple[str, Tuple[float, float], float], ...] = (
    ("denseish",     (0.02, 0.10), 0.10),
    ("weak_sparse",  (0.10, 0.50), 0.25),
    ("lasso_like",   (0.50, 1.50), 0.45),
    ("hard_sparse", (1.50, 3.50), 0.20),
)


def tail_floor_for_coercivity() -> float:
    """Minimum rho'(6) (coercivity floor)."""
    return 0.02


def stage_for_step(step: int, boundaries: Sequence[int] = (10000, 30000)) -> str:
    """Map a global step to a curriculum stage name.

    Defaults to the fpk_decisions.md §1 schedule (50k total):
        easy   for step <  10_000
        medium for step <  30_000
        hard   otherwise
    """
    if step < boundaries[0]:
        return "easy"
    if step < boundaries[1]:
        return "medium"
    return "hard"


# -----------------------------------------------------------------------------
# D.6 Random theta sampler — derivative-staircase, with K randomization
# -----------------------------------------------------------------------------

def _knot_strata_counts(K: int, strata: Sequence[float]) -> Tuple[int, int, int]:
    """Split K knots into 3 strata according to ratio weights `strata`.

    Default ratio = (0.60, 0.25, 0.15) per fpk_decisions.md §1 (Tier 2).
    Ensures each non-zero stratum gets at least one knot when feasible.
    """
    if len(strata) != 3:
        raise ValueError(f"knot strata must have 3 entries, got {len(strata)}")
    s = float(sum(strata))
    if s <= 0:
        raise ValueError("knot strata must sum to a positive number")
    w = [float(x) / s for x in strata]
    n_low = max(1, int(round(K * w[0])))
    n_mid = max(1, int(round(K * w[1])))
    n_high = K - n_low - n_mid
    if n_high < 0:
        # Re-balance: the rounding overshot.
        n_high = 0
        n_mid = max(0, K - n_low)
    return n_low, n_mid, n_high


def _stratified_jittered_knots(
    K: int,
    rng: np.random.Generator,
    strata: Sequence[float] = (0.60, 0.25, 0.15),
) -> np.ndarray:
    """Stratified knots in [0.05, 6.0] with jitter."""
    n_low, n_mid, n_high = _knot_strata_counts(K, strata)

    def _uniform_jittered(lo: float, hi: float, n: int) -> np.ndarray:
        if n == 0:
            return np.array([], dtype=np.float64)
        base = np.linspace(lo, hi, n + 2)[1:-1]
        jitter = rng.uniform(-0.04, 0.04, size=n)
        return np.clip(base + jitter, lo + 1e-6, hi - 1e-6)

    knots = np.concatenate([
        _uniform_jittered(0.05, 1.25, n_low),
        _uniform_jittered(1.25, 3.00, n_mid),
        _uniform_jittered(3.00, 6.00, n_high),
    ])
    knots.sort()
    return knots


def _project_derivative_path(
    deriv: np.ndarray,
    *,
    min_value: float,
    max_value: float,
    total_variation: float,
    total_negative: float,
) -> np.ndarray:
    """Clamp + rescale derivative levels to meet TV and negative budgets."""
    deriv = np.asarray(deriv, dtype=np.float64).copy()
    deriv = np.clip(deriv, min_value, max_value)
    jumps = np.diff(deriv)
    tv = float(np.sum(np.abs(jumps)))
    neg = float(np.sum(np.maximum(0.0, -jumps)))
    scale = 1.0
    if tv > total_variation > 0:
        scale = min(scale, total_variation / tv)
    if neg > total_negative > 0:
        scale = min(scale, total_negative / neg)
    if scale < 1.0:
        deriv = deriv[0] + scale * np.cumsum(np.concatenate([[0.0], jumps]))
        deriv = np.clip(deriv, min_value, max_value)
    return deriv


def _normalize_theta_rows(theta_rows: List[List[float]]) -> List[List[float]]:
    """Divide all weights by rho_raw(1) so standardized rho(1) = 1."""
    rho_at_1 = 0.0
    for row in theta_rows:
        kind = int(round(row[0]))
        knot = row[1]
        weight = row[2]
        if kind == 0:
            rho_at_1 += weight
        elif kind == 1:
            rho_at_1 += weight * 1.0
        elif kind == 2:
            if 1.0 > knot:
                rho_at_1 += weight * (1.0 - knot)
    if abs(rho_at_1) < 1e-8 or not math.isfinite(rho_at_1):
        return []
    inv = 1.0 / rho_at_1
    return [[row[0], row[1], row[2] * inv, row[3]] for row in theta_rows]


def _eval_rho_rows_at(theta_rows: List[List[float]], t: float) -> Optional[float]:
    rho = 0.0
    for row in theta_rows:
        kind = int(round(row[0]))
        knot = row[1]
        weight = row[2]
        if kind == 0:
            rho += weight
        elif kind == 1:
            rho += weight * t
        elif kind == 2:
            if t > knot:
                rho += weight * (t - knot)
    if not math.isfinite(rho):
        return None
    return rho


def _derivative_levels_from_segments(segments: SegmentList) -> np.ndarray:
    return np.array([seg.d for seg in segments], dtype=np.float64)


def _bucket_of(d0: float) -> str:
    for name, (lo, hi), _ in D0_BUCKETS:
        if lo <= d0 <= hi:
            return name
    if d0 < D0_BUCKETS[0][1][0]:
        return "below_denseish"
    return "above_hard_sparse"


def _sample_d0_target(
    rng: np.random.Generator,
    bucket_weights: Optional[Sequence[float]] = None,
) -> Tuple[str, float]:
    if bucket_weights is None:
        probs = np.array([w for _, _, w in D0_BUCKETS], dtype=np.float64)
    else:
        probs = np.asarray(bucket_weights, dtype=np.float64)
        if probs.shape != (len(D0_BUCKETS),):
            raise ValueError(
                f"bucket_weights must have length {len(D0_BUCKETS)}, "
                f"got {probs.shape}"
            )
    probs = probs / probs.sum()
    bucket_idx = int(rng.choice(len(D0_BUCKETS), p=probs))
    name, (lo, hi), _ = D0_BUCKETS[bucket_idx]
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    d0_target = math.exp(rng.uniform(log_lo, log_hi))
    return name, d0_target


def _sample_initial_slope_for_bucket(
    target_bucket: str, rng: np.random.Generator
) -> float:
    if target_bucket == "denseish":
        return float(np.exp(rng.uniform(math.log(0.02), math.log(0.10))))
    if target_bucket == "weak_sparse":
        return float(np.exp(rng.uniform(math.log(0.10), math.log(0.50))))
    if target_bucket == "lasso_like":
        return float(np.exp(rng.uniform(math.log(0.50), math.log(1.50))))
    if target_bucket == "hard_sparse":
        return float(np.exp(rng.uniform(math.log(1.50), math.log(3.50))))
    return float(np.exp(rng.uniform(math.log(0.05), math.log(3.0))))


def _trend_bias_for_target(
    target_d0: float,
    d_raw_0: float,
    knots: np.ndarray,
    *,
    tv_budget: float,
    neg_budget: float,
) -> np.ndarray:
    D = max(target_d0, 1e-3)
    s_target = 2.0 * d_raw_0 * (1.0 - D) / D
    knots_in_unit_mask = knots <= 1.0
    n_unit = int(knots_in_unit_mask.sum())
    bias = np.zeros(len(knots), dtype=np.float64)
    if n_unit == 0:
        return bias
    per_knot = s_target / n_unit
    bias[knots_in_unit_mask] = per_knot
    pos_sum = float(bias[bias > 0].sum())
    neg_sum = float(-bias[bias < 0].sum())
    pos_cap = 0.85 * tv_budget
    neg_cap = 0.85 * neg_budget
    scale = 1.0
    if pos_sum > pos_cap > 0:
        scale = min(scale, pos_cap / pos_sum)
    if neg_sum > neg_cap > 0:
        scale = min(scale, neg_cap / neg_sum)
    bias *= scale
    return bias


@dataclass(frozen=True)
class _ValidityReport:
    passed: bool
    rejection_reason: Optional[str]


def _validity_checks(
    segments: SegmentList,
    *,
    test_grid: np.ndarray,
    derivative_total_variation: float,
    tv_budget: float,
    tail_slope_min: float = 0.02,
    derivative_min: float = -1e-6,
    rho_min: float = -1e-6,
    drho_max: float = 6.0,
    d0_bounds: Tuple[float, float] = (0.02, 3.5),
    rho_at_1_pre_norm: Tuple[float, float] = (0.05, 20.0),
    pre_norm_rho_at_1: Optional[float] = None,
) -> _ValidityReport:
    """Plan §3 validity gate."""
    for t in test_grid:
        t_f = float(t)
        rho_t = eval_piecewise_rho(t_f, segments)
        drho_lo, drho_hi = eval_piecewise_drho_at(t_f, segments)
        if not (math.isfinite(rho_t) and math.isfinite(drho_lo) and math.isfinite(drho_hi)):
            return _ValidityReport(False, "nonfinite_rho_drho")
        if rho_t < rho_min:
            return _ValidityReport(False, "negative_rho")
        if drho_lo < derivative_min:
            return _ValidityReport(False, "negative_drho")
        if drho_hi > drho_max:
            return _ValidityReport(False, "drho_too_large")
    if abs(eval_piecewise_rho(0.0, segments)) > 1e-9:
        return _ValidityReport(False, "rho_at_zero_nonzero")
    rho_1 = eval_piecewise_rho(1.0, segments)
    if not math.isclose(rho_1, 1.0, abs_tol=1e-6):
        return _ValidityReport(False, f"rho_at_1_off:{rho_1}")
    d0 = d0_from_segments(segments)
    if not (d0_bounds[0] <= d0 <= d0_bounds[1]):
        return _ValidityReport(False, f"d0_out_of_range:{d0}")
    if pre_norm_rho_at_1 is not None:
        if not (rho_at_1_pre_norm[0] <= pre_norm_rho_at_1 <= rho_at_1_pre_norm[1]):
            return _ValidityReport(False, f"pre_norm_rho_at_1_out_of_range:{pre_norm_rho_at_1}")
    if derivative_total_variation > tv_budget + 1e-6:
        return _ValidityReport(False, f"tv_over_budget:{derivative_total_variation}")
    drho_at_6_lo, _ = eval_piecewise_drho_at(6.0, segments)
    if drho_at_6_lo < tail_slope_min:
        return _ValidityReport(False, f"tail_slope_too_small:{drho_at_6_lo}")
    # Light prox finiteness sweep
    for v in (-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0):
        for tau in (0.001, 0.01, 0.1, 1.0):
            try:
                b = scalar_abs_penalty_prox(v, tau, segments)
            except Exception as exc:  # noqa: BLE001
                return _ValidityReport(False, f"scalar_prox_exception:{exc}")
            if not math.isfinite(b):
                return _ValidityReport(False, f"scalar_prox_nonfinite:v={v},tau={tau}")
    return _ValidityReport(True, None)


@dataclass
class RandomThetaSample:
    """One realized random-theta penalty.

    Attributes
    ----------
    theta : (K+2, 4) torch.float64 tensor — encoder-ready token grid.
    segments : SegmentList — same penalty in the segment representation.
    d0 : standardized rho'(0+).
    rho_at_1_pre_norm : pre-standardization rho(1), for diagnostics.
    bucket : realized d0 bucket name.
    stage : the curriculum stage used.
    K : number of hinge tokens actually sampled.
    metadata : dict with attempt count and other diagnostics.
    """
    theta: torch.Tensor
    segments: SegmentList
    d0: float
    rho_at_1_pre_norm: float
    bucket: str
    stage: str
    K: int
    metadata: Dict[str, Any]


def sample_random_theta(
    stage: str,
    seed: int,
    *,
    K: int = 16,
    max_resample_attempts: int = 50,
    target_bucket: Optional[str] = None,
    bucket_weights: Optional[Sequence[float]] = None,
    knot_strata: Sequence[float] = (0.60, 0.25, 0.15),
    tail_floor: float = 0.02,
) -> RandomThetaSample:
    """Draw one valid random hinge-basis penalty for ``stage``.

    Parameters
    ----------
    stage : "easy", "medium", or "hard".
    seed : int seed for the numpy RNG.
    K : number of hinge tokens (random K is set by ``PenaltySampler``).
    target_bucket : if set, reject samples that don't realize this bucket.
    bucket_weights : override the (denseish, weak_sparse, lasso_like,
        hard_sparse) target-d0 probabilities. Defaults to fpk_decisions.md
        §1 weights (0.10, 0.25, 0.45, 0.20).
    knot_strata : (low, mid, high) knot-band ratios; default 60/25/15 per
        fpk_decisions.md §1 Tier 2.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {list(STAGES)}")
    s = STAGES[stage]
    rng = np.random.default_rng(seed)
    if K < 3:
        raise ValueError(f"K must be >= 3, got {K}")

    attempts: List[Dict[str, Any]] = []
    for attempt_idx in range(max_resample_attempts):
        chosen_bucket, target_d0 = _sample_d0_target(rng, bucket_weights=bucket_weights)
        if target_bucket is not None:
            chosen_bucket = target_bucket
            for name, (lo, hi), _ in D0_BUCKETS:
                if name == target_bucket:
                    target_d0 = math.exp(rng.uniform(math.log(lo), math.log(hi)))
                    break

        knots = _stratified_jittered_knots(K, rng, strata=knot_strata)
        d_raw_0 = _sample_initial_slope_for_bucket(chosen_bucket, rng)

        bias = _trend_bias_for_target(
            target_d0, d_raw_0, knots,
            tv_budget=s.tv_budget, neg_budget=s.neg_budget,
        )
        noise = rng.laplace(scale=s.jump_scale, size=K)
        jumps_raw = bias + noise

        deriv = np.concatenate([[d_raw_0], d_raw_0 + np.cumsum(jumps_raw)])
        deriv = _project_derivative_path(
            deriv,
            min_value=max(tail_floor, tail_floor_for_coercivity()),
            max_value=6.0,
            total_variation=s.tv_budget,
            total_negative=s.neg_budget,
        )

        weights = np.diff(deriv)
        theta_rows = [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, float(deriv[0]), 0.0],
        ]
        for k, w in zip(knots, weights):
            theta_rows.append([2.0, float(k), float(w), 0.0])

        pre_norm_rho_at_1 = _eval_rho_rows_at(theta_rows, 1.0)
        if pre_norm_rho_at_1 is None or pre_norm_rho_at_1 <= 0.0:
            attempts.append({"attempt": attempt_idx, "reason": "pre_norm_rho_at_1_nonpos"})
            continue

        normalized = _normalize_theta_rows(theta_rows)
        if not normalized:
            attempts.append({"attempt": attempt_idx, "reason": "normalize_failed"})
            continue

        theta_tensor = torch.tensor(normalized, dtype=torch.float64)
        segments = theta_to_segments(theta_tensor)

        derivative_levels = _derivative_levels_from_segments(segments)
        tv_actual = float(np.sum(np.abs(np.diff(derivative_levels))))

        report = _validity_checks(
            segments,
            test_grid=np.linspace(0.0, 6.0, 25),
            derivative_total_variation=tv_actual,
            tv_budget=s.tv_budget,
            pre_norm_rho_at_1=pre_norm_rho_at_1,
        )
        if not report.passed:
            attempts.append({"attempt": attempt_idx, "reason": report.rejection_reason})
            continue

        realized_d0 = d0_from_segments(segments)
        bucket = _bucket_of(realized_d0)
        if target_bucket is not None and bucket != target_bucket:
            attempts.append({"attempt": attempt_idx, "reason": f"wrong_bucket:got={bucket}"})
            continue

        return RandomThetaSample(
            theta=theta_tensor,
            segments=segments,
            d0=realized_d0,
            rho_at_1_pre_norm=pre_norm_rho_at_1,
            bucket=bucket,
            stage=stage,
            K=K,
            metadata={
                "attempt_index": attempt_idx,
                "K": K,
                "tv_actual": tv_actual,
                "d_raw_0": d_raw_0,
                "target_d0": target_d0,
                "target_bucket_intended": chosen_bucket,
            },
        )
    raise RuntimeError(
        f"sample_random_theta exhausted {max_resample_attempts} attempts for "
        f"stage={stage} seed={seed}; recent rejections: {attempts[-5:]}"
    )


# -----------------------------------------------------------------------------
# D.7 Known-family scaffolding (LASSO_FAMILY etc.)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class FamilyConfig:
    """One known family. ``param_low``/``param_high`` are the uniform
    sampling range for the family's first parameter slot (alpha / gamma /
    a). Lasso and Ridge have no free parameter."""
    name: str
    family_id: int
    weight: float
    param_low: Optional[float] = None
    param_high: Optional[float] = None


LASSO_FAMILY = FamilyConfig(name="lasso", family_id=PID_LASSO, weight=1.0)
RIDGE_FAMILY = FamilyConfig(name="ridge", family_id=PID_RIDGE, weight=1.0)
ENET_FAMILY = FamilyConfig(
    name="elastic_net", family_id=PID_ENET, weight=1.0,
    param_low=0.1, param_high=0.9,
)
MCP_FAMILY = FamilyConfig(
    name="mcp", family_id=PID_MCP, weight=1.0,
    param_low=1.5, param_high=8.0,
)
SCAD_FAMILY = FamilyConfig(
    name="scad", family_id=PID_SCAD, weight=1.0,
    param_low=2.5, param_high=8.0,
)


def with_weight(fam: FamilyConfig, w: float) -> FamilyConfig:
    return FamilyConfig(
        name=fam.name, family_id=fam.family_id, weight=w,
        param_low=fam.param_low, param_high=fam.param_high,
    )


def known_family_segments(fam: FamilyConfig, param: float) -> SegmentList:
    """Build a SegmentList for the given known family using ``param``
    (alpha for EN, gamma for MCP, a for SCAD; ignored for Lasso/Ridge)."""
    if fam.family_id == PID_LASSO:
        return lasso_segments()
    if fam.family_id == PID_RIDGE:
        return ridge_segments()
    if fam.family_id == PID_ENET:
        return elastic_net_segments(float(param))
    if fam.family_id == PID_MCP:
        return mcp_segments(float(param))
    if fam.family_id == PID_SCAD:
        return scad_segments(float(param))
    raise ValueError(f"unknown family_id {fam.family_id}")


# -----------------------------------------------------------------------------
# D.8 PenaltySampler  (mid-heavy lambda + K randomization + 80/20 mix)
# -----------------------------------------------------------------------------

@dataclass
class PenaltySamplerConfig:
    """Sampling settings per fpk_decisions.md §1 (Tier 2) and §2.4.

    Mix:
        known_family_ratio   = 0.20
        random_theta_ratio   = 0.80  (must sum to 1)

    Lambda (mid-heavy mixture, log-uniform within each band):
        50% in [lambda_mid_low, lambda_mid_high]   = [1e-2, 1e-1]
        30% in [lambda_low_min, lambda_mid_low]    = [1e-3, 1e-2]
        20% in [lambda_mid_high, lambda_high_max]  = [1e-1, 1.0]

    Random theta:
        K_min=3, K_max=64 (uniform)
        monotone=False (kept; documented)
        knot_strata=(0.60, 0.25, 0.15)
        d0_bucket_weights=(0.10, 0.25, 0.45, 0.20)
        tail_floor=0.02

    Curriculum:
        stages = (easy, medium, hard)
        stage_step_boundaries = (10_000, 30_000)
    """
    families: List[FamilyConfig] = field(default_factory=lambda: [
        LASSO_FAMILY, RIDGE_FAMILY, ENET_FAMILY, MCP_FAMILY, SCAD_FAMILY,
    ])
    queries_per_task: int = 8

    # Mix
    known_family_ratio: float = 0.20
    random_theta_ratio: float = 0.80

    # Lambda
    lambda_low_min: float = 1e-3
    lambda_mid_low: float = 1e-2
    lambda_mid_high: float = 1e-1
    lambda_high_max: float = 1.0
    lambda_low_weight: float = 0.30
    lambda_mid_weight: float = 0.50
    lambda_high_weight: float = 0.20

    # Random theta knobs
    K_min: int = 3
    K_max: int = 64
    monotone: bool = False
    knot_strata: Tuple[float, float, float] = (0.60, 0.25, 0.15)
    d0_bucket_weights: Tuple[float, float, float, float] = (0.10, 0.25, 0.45, 0.20)
    tail_floor: float = 0.02

    # Curriculum
    stage_step_boundaries: Tuple[int, int] = (10_000, 30_000)

    def __post_init__(self) -> None:
        if not self.families:
            raise ValueError("PenaltySamplerConfig.families must be non-empty")
        if abs(self.known_family_ratio + self.random_theta_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"known_family_ratio + random_theta_ratio must equal 1.0, "
                f"got {self.known_family_ratio} + {self.random_theta_ratio}"
            )
        if not (self.lambda_low_min < self.lambda_mid_low
                < self.lambda_mid_high < self.lambda_high_max):
            raise ValueError(
                "lambda bands must be strictly increasing: "
                f"{self.lambda_low_min} < {self.lambda_mid_low} "
                f"< {self.lambda_mid_high} < {self.lambda_high_max}"
            )
        w_sum = self.lambda_low_weight + self.lambda_mid_weight + self.lambda_high_weight
        if abs(w_sum - 1.0) > 1e-6:
            raise ValueError(
                f"lambda band weights must sum to 1.0, got {w_sum}"
            )
        if not (3 <= self.K_min <= self.K_max <= 256):
            raise ValueError(
                f"K_min/K_max must satisfy 3 <= K_min <= K_max <= 256, "
                f"got K_min={self.K_min}, K_max={self.K_max}"
            )


@dataclass
class SampledBatch:
    """One sampled batch ready to feed the model."""
    penalty_type_id: Tensor          # (B, Q) long
    penalty_params: Tensor           # (B, Q, 1) float
    lambdas: Tensor                  # (B, Q) float
    lambda_max: Tensor               # (B, Q) float (broadcast from lmx)
    shape_features: Tensor           # (B, Q, D_raw) float32
    d0: Tensor                       # (B, Q) float32
    is_random: Tensor                # (B, Q) bool — True for random theta
    segments_grid: List[List[SegmentList]]   # (B)(Q) nested list of SegmentLists
    stage: str
    metadata: Dict[str, Any]


class PenaltySampler:
    """FPK v2 sampler. Drop-in replacement for the prototype sampler.

    Per fpk_decisions.md §1 (Tier 2):
      * 80% random theta / 20% known family
      * mid-heavy lambda mixture
      * K uniform in [3, 64] per random sample
      * stage-conditioned d0 buckets

    Reproducibility:
      A torch.Generator (CPU) + a numpy default_rng both seeded from
      ``seed``. ``state_dict`` / ``load_state_dict`` persist both.
    """

    def __init__(self, cfg: PenaltySamplerConfig, seed: int = 0) -> None:
        self.cfg = cfg
        self._torch_gen = torch.Generator(device="cpu")
        self._torch_gen.manual_seed(int(seed))
        # We bump a counter every sample call so that two consecutive
        # sample() invocations with the same seed don't yield identical
        # random thetas. The actual numpy RNG seed for sample_random_theta
        # is derived from this counter.
        self._call_counter: int = 0
        # Family weights (used when the query is a "known family" query)
        total = sum(f.weight for f in cfg.families)
        if total <= 0:
            raise ValueError("Family weights must sum to a positive number")
        self._family_weights = torch.tensor(
            [f.weight / total for f in cfg.families],
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # Stage helper
    # ------------------------------------------------------------------
    def stage_for_step(self, step: int) -> str:
        return stage_for_step(step, self.cfg.stage_step_boundaries)

    # ------------------------------------------------------------------
    # Lambda sampler (mid-heavy mixture)
    # ------------------------------------------------------------------
    def _sample_lambdas(self, B: int, Q: int) -> Tensor:
        cfg = self.cfg
        band_w = torch.tensor(
            [cfg.lambda_low_weight, cfg.lambda_mid_weight, cfg.lambda_high_weight],
            dtype=torch.float32,
        )
        # Per-element band assignment
        band_idx = torch.multinomial(
            band_w, B * Q, replacement=True, generator=self._torch_gen,
        ).view(B, Q)

        u = torch.rand(B, Q, generator=self._torch_gen)

        log_lo = torch.empty(B, Q)
        log_hi = torch.empty(B, Q)

        # Band 0: [low_min, mid_low]
        m0 = band_idx == 0
        log_lo[m0] = math.log(cfg.lambda_low_min)
        log_hi[m0] = math.log(cfg.lambda_mid_low)
        # Band 1: [mid_low, mid_high]
        m1 = band_idx == 1
        log_lo[m1] = math.log(cfg.lambda_mid_low)
        log_hi[m1] = math.log(cfg.lambda_mid_high)
        # Band 2: [mid_high, high_max]
        m2 = band_idx == 2
        log_lo[m2] = math.log(cfg.lambda_mid_high)
        log_hi[m2] = math.log(cfg.lambda_high_max)

        lambdas = torch.exp(log_lo * (1.0 - u) + log_hi * u)
        return lambdas

    # ------------------------------------------------------------------
    # Known family sampling for a flat boolean mask
    # ------------------------------------------------------------------
    def _sample_known_families(
        self,
        n: int,
    ) -> Tuple[Tensor, Tensor, List[SegmentList]]:
        """Draw n known-family queries. Returns (family_ids, params, segments_list)."""
        if n == 0:
            return (
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, 1, dtype=torch.float32),
                [],
            )
        idx_flat = torch.multinomial(
            self._family_weights, n, replacement=True, generator=self._torch_gen,
        )
        family_ids = torch.zeros(n, dtype=torch.long)
        params = torch.zeros(n, 1, dtype=torch.float32)
        segments_list: List[SegmentList] = [None] * n  # type: ignore[list-item]

        for i, fam in enumerate(self.cfg.families):
            mask = idx_flat == i
            if not mask.any():
                continue
            n_i = int(mask.sum().item())
            family_ids[mask] = fam.family_id
            if fam.param_low is not None and fam.param_high is not None:
                u = torch.rand(n_i, generator=self._torch_gen)
                pvals = fam.param_low + (fam.param_high - fam.param_low) * u
                params[mask] = pvals.unsqueeze(-1)
                # Build segments for each
                pos_idx = mask.nonzero(as_tuple=False).flatten().tolist()
                for j, slot in enumerate(pos_idx):
                    segments_list[slot] = known_family_segments(fam, float(pvals[j].item()))
            else:
                # Lasso / Ridge: no parameter
                pos_idx = mask.nonzero(as_tuple=False).flatten().tolist()
                for slot in pos_idx:
                    segments_list[slot] = known_family_segments(fam, 0.0)
        # Validate: no None remained
        for slot, seg in enumerate(segments_list):
            if seg is None:
                raise RuntimeError(f"known-family slot {slot} unfilled (likely a family with bad weight)")
        return family_ids, params, segments_list

    # ------------------------------------------------------------------
    # Random theta sampling for a flat boolean mask
    # ------------------------------------------------------------------
    def _sample_random_thetas(
        self,
        n: int,
        stage: str,
        base_seed: int,
    ) -> Tuple[Tensor, Tensor, List[SegmentList], List[int]]:
        """Returns (family_ids, params, segments_list, K_list).

        Random-theta queries get family_id = PID_LASSO and params = 0 as
        a place-holder; the actual penalty is supplied via the segment
        list and via the shape features computed downstream.
        """
        if n == 0:
            return (
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, 1, dtype=torch.float32),
                [],
                [],
            )
        # Pick a K per sample uniformly in [K_min, K_max].
        K_low, K_high = self.cfg.K_min, self.cfg.K_max
        Ks = torch.randint(
            low=K_low, high=K_high + 1, size=(n,),
            generator=self._torch_gen,
        ).tolist()

        segments_list: List[SegmentList] = []
        K_used: List[int] = []
        for i in range(n):
            # Sub-seed mixes the base seed with the slot index. Using a 64-bit
            # combine avoids accidental collisions across consecutive calls.
            sub_seed = (
                (int(base_seed) * 1_000_003 + i * 999_983
                 + self._call_counter * 7_919) & 0x7FFFFFFF
            )
            sample = sample_random_theta(
                stage=stage, seed=sub_seed, K=int(Ks[i]),
                bucket_weights=self.cfg.d0_bucket_weights,
                knot_strata=self.cfg.knot_strata,
                tail_floor=self.cfg.tail_floor,
            )
            segments_list.append(sample.segments)
            K_used.append(sample.K)

        # Random-theta queries get family_id = PID_LASSO; the model
        # ignores this when use_penalty_type_embed = False (the default).
        family_ids = torch.full((n,), PID_LASSO, dtype=torch.long)
        params = torch.zeros(n, 1, dtype=torch.float32)
        return family_ids, params, segments_list, K_used

    # ------------------------------------------------------------------
    # Main sample entry point
    # ------------------------------------------------------------------
    def sample(
        self,
        batch_size: int,
        lambda_max: Tensor,
        device: Union[str, torch.device],
        *,
        step: int = 0,
        stage: Optional[str] = None,
    ) -> SampledBatch:
        """Sample one batch.

        Parameters
        ----------
        batch_size : B
        lambda_max : (B,) tensor or scalar — used only to broadcast into
            the returned ``lambda_max`` field. **Lambdas themselves are
            sampled in absolute space**, not scaled by lambda_max (the
            mid-heavy distribution is on absolute lambda values per
            fpk_decisions.md §2.4).
        device : where to place the returned tensors.
        step : global step (controls curriculum stage if ``stage`` is None).
        stage : optional explicit stage override.
        """
        cfg = self.cfg
        B = int(batch_size)
        Q = int(cfg.queries_per_task)
        dev = torch.device(device)
        self._call_counter += 1
        call_id = self._call_counter

        if stage is None:
            stage = self.stage_for_step(step)

        # ---- Lambdas (mid-heavy) -------------------------------------
        lambdas = self._sample_lambdas(B, Q)

        # ---- Decide random vs known for every (b, q) ------------------
        rand_mask_flat = torch.bernoulli(
            torch.full((B * Q,), cfg.random_theta_ratio),
            generator=self._torch_gen,
        ).bool()
        is_random = rand_mask_flat.view(B, Q)

        n_random = int(rand_mask_flat.sum().item())
        n_known = B * Q - n_random

        # ---- Known family slots ---------------------------------------
        kn_fam_ids, kn_params, kn_segments = self._sample_known_families(n_known)

        # ---- Random theta slots ---------------------------------------
        rn_fam_ids, rn_params, rn_segments, rn_K = self._sample_random_thetas(
            n_random, stage=stage, base_seed=call_id,
        )

        # ---- Stitch back into (B, Q) arrays ---------------------------
        family_ids = torch.zeros(B * Q, dtype=torch.long)
        params = torch.zeros(B * Q, 1, dtype=torch.float32)
        segments_flat: List[SegmentList] = [None] * (B * Q)  # type: ignore[list-item]

        kn_positions = (~rand_mask_flat).nonzero(as_tuple=False).flatten().tolist()
        rn_positions = rand_mask_flat.nonzero(as_tuple=False).flatten().tolist()

        for slot, pos in enumerate(kn_positions):
            family_ids[pos] = kn_fam_ids[slot]
            params[pos] = kn_params[slot]
            segments_flat[pos] = kn_segments[slot]
        for slot, pos in enumerate(rn_positions):
            family_ids[pos] = rn_fam_ids[slot]
            params[pos] = rn_params[slot]
            segments_flat[pos] = rn_segments[slot]

        family_ids = family_ids.view(B, Q)
        params = params.view(B, Q, 1)

        # Nested list (B)(Q) for downstream feature/loss code
        segments_grid: List[List[SegmentList]] = [
            [segments_flat[b * Q + q] for q in range(Q)] for b in range(B)
        ]

        # ---- Shape features -------------------------------------------
        # For known families we use make_penalty_features (PID path) on
        # the known slots. For random theta we use the segments path.
        # We compute both and stitch.
        cpu_dev = torch.device("cpu")
        feats_known = make_penalty_features(
            penalty_type_id=family_ids,
            penalty_params=params,
            detach_inputs=True,
        )  # shape_features (B, Q, D_raw), d0 (B, Q)

        # For the random slots, overwrite with segments-based features.
        # This loop is per-batch so we only do segment evaluation for the
        # random fraction.
        if n_random > 0:
            seg_feats = make_penalty_features_from_segments(
                segments_grid=segments_grid,
                device=cpu_dev,
            )
            shape_features = feats_known["shape_features"].clone()
            d0 = feats_known["d0"].clone()
            # Replace random-slot rows
            rand_mask_bq = is_random  # (B, Q) bool
            shape_features[rand_mask_bq] = seg_feats["shape_features"][rand_mask_bq]
            d0[rand_mask_bq] = seg_feats["d0"][rand_mask_bq]
        else:
            shape_features = feats_known["shape_features"]
            d0 = feats_known["d0"]

        # ---- lambda_max broadcasting ---------------------------------
        lmx_cpu = lambda_max.detach().to("cpu", dtype=torch.float32).clamp_min(1e-12)
        if lmx_cpu.dim() == 0:
            lmx_cpu = lmx_cpu.view(1).expand(B)
        elif lmx_cpu.dim() == 2 and lmx_cpu.shape[-1] == 1:
            lmx_cpu = lmx_cpu[:, 0]
        if lmx_cpu.shape != (B,):
            raise ValueError(
                f"lambda_max must broadcast to (B,)=({B},), got {tuple(lmx_cpu.shape)}"
            )
        lmx_bq = lmx_cpu.unsqueeze(-1).expand(B, Q)

        return SampledBatch(
            penalty_type_id=family_ids.to(dev),
            penalty_params=params.to(dev),
            lambdas=lambdas.to(dev),
            lambda_max=lmx_bq.to(dev),
            shape_features=shape_features.to(dev),
            d0=d0.to(dev),
            is_random=is_random.to(dev),
            segments_grid=segments_grid,
            stage=stage,
            metadata={
                "n_random": n_random,
                "n_known": n_known,
                "K_used": rn_K,
                "call_id": call_id,
                "step": step,
            },
        )

    # ------------------------------------------------------------------
    # State dict
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "torch_generator_state": self._torch_gen.get_state(),
            "call_counter": self._call_counter,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._torch_gen.set_state(state["torch_generator_state"])
        self._call_counter = int(state.get("call_counter", 0))




# ##############################################################################
# SECTION E -- Loss helpers, FpkConfig, make_fpk_model.
#
# Absorbs (with new weights per fpk_decisions.md §1 Tier 1):
#   - dg_fpk/kkt_residual.py     (numpy: KKT residual, penalty_objective)
#   - dg_fpk/loss_terms.py       (torch loss terms; compute_batch_loss)
#
# Default weights (FPK v2):
#   W_OBJ   = 1.00      (objective-first)
#   W_KKT   = 0.00      (ablation toggles to 0.25)
#   W_PROX  = 0.10
#   W_RIDGE = 0.00      (DECISION_PENDING; monitor Ridge beta-corr)
#   W_DELTA = 1.0e-4
#   w_distill_schedule = off  (returns 0.0)
#   w_z_schedule       = off  (returns 0.0)
#
# Convention: lam * sum_j rho(|beta_j|), NO division by P.
# ##############################################################################


# -----------------------------------------------------------------------------
# E.1 Default loss weights (FPK v2)
# -----------------------------------------------------------------------------

W_OBJ = 1.00
W_KKT = 0.00
W_PROX = 0.10
W_RIDGE = 0.00
W_DELTA = 1e-4


def w_distill_schedule(step: int) -> float:
    """FPK v2: distillation off. Returns 0 regardless of step."""
    return 0.0


def w_z_schedule(theta_encoder_unfrozen: bool) -> float:
    """FPK v2: z-schedule off. Returns 0 regardless."""
    return 0.0


# -----------------------------------------------------------------------------
# E.2 Numpy reference primitives (used by eval and by the L_kkt term)
# -----------------------------------------------------------------------------

def kkt_residual_per_coord_np(
    G: np.ndarray,
    a: np.ndarray,
    beta: np.ndarray,
    lam: float,
    segments: SegmentList,
) -> np.ndarray:
    """Coord-wise KKT residual r_j >= 0 (numpy, non-differentiable)."""
    P = beta.shape[0]
    g = G @ beta - a
    r = np.zeros(P, dtype=np.float64)
    d0 = d0_from_segments(segments)
    for j in range(P):
        b = float(beta[j])
        if b == 0.0:
            sub_lo, sub_hi = -d0, d0
        else:
            t = abs(b)
            d_l, d_r = eval_piecewise_drho_at(t, segments)
            if d_l > d_r:
                d_l, d_r = d_r, d_l
            if b > 0:
                sub_lo, sub_hi = d_l, d_r
            else:
                sub_lo, sub_hi = -d_r, -d_l
        scaled_lo = lam * sub_lo
        scaled_hi = lam * sub_hi
        target = -float(g[j])
        if target < scaled_lo:
            r[j] = scaled_lo - target
        elif target > scaled_hi:
            r[j] = target - scaled_hi
    return r


def kkt_rms_normalized_np(
    G: np.ndarray,
    a: np.ndarray,
    beta: np.ndarray,
    lam: float,
    segments: SegmentList,
    *,
    eps: float = 1e-6,
) -> float:
    """Plan §2 normalized RMS KKT residual."""
    g = G @ beta - a
    r = kkt_residual_per_coord_np(G, a, beta, lam, segments)
    d0 = d0_from_segments(segments)
    denom = np.abs(g) + lam * (d0 + 1.0) + eps
    r_tilde = r / denom
    return float(np.sqrt(np.mean(r_tilde * r_tilde)))


def penalty_objective_np(
    Xc: np.ndarray,
    yc: np.ndarray,
    beta: np.ndarray,
    lam: float,
    segments: SegmentList,
) -> float:
    """F(beta) = (1/2N) ||y - X beta||^2 + lam * sum_j rho(|beta_j|)."""
    N = Xc.shape[0]
    resid = yc - Xc @ beta
    quad = 0.5 / N * float(np.dot(resid, resid))
    pen = 0.0
    for j in range(beta.shape[0]):
        pen += eval_piecewise_rho(float(abs(beta[j])), segments)
    return quad + lam * pen


def penalty_objective_at_zero_np(Xc: np.ndarray, yc: np.ndarray) -> float:
    """F(0) = (1/2N) ||y||^2 (centered). Used to normalize L_obj."""
    N = Xc.shape[0]
    return 0.5 / N * float(np.dot(yc, yc))


def power_iter_largest_eigval(G: np.ndarray, n_iter: int = 5) -> float:
    """Estimate largest eigenvalue of symmetric PSD G via power iteration."""
    P = G.shape[0]
    v = np.random.default_rng(0).standard_normal(P)
    v /= np.linalg.norm(v) + 1e-12
    lam_est = 0.0
    for _ in range(n_iter):
        v_new = G @ v
        lam_est = float(np.linalg.norm(v_new) + 1e-30)
        v = v_new / lam_est
    return lam_est


def prox_stationarity_residual_np(
    G: np.ndarray,
    a: np.ndarray,
    beta: np.ndarray,
    lam: float,
    segments: SegmentList,
    *,
    eta: Optional[float] = None,
    n_power_iters: int = 5,
) -> float:
    """Prox-stationarity P = ||beta - prox(beta - eta g)||/(||beta|| + ||z||)."""
    g = G @ beta - a
    if eta is None:
        L = power_iter_largest_eigval(G, n_iter=n_power_iters)
        L = max(L, 1e-8)
        eta = 0.99 / L
    v = beta - eta * g
    tau = eta * lam
    z = vectorized_scalar_prox(v, np.full_like(v, tau), segments)
    num = float(np.linalg.norm(beta - z))
    den = float(np.linalg.norm(beta)) + float(np.linalg.norm(z)) + 1e-6
    return num / den


# -----------------------------------------------------------------------------
# E.3 Torch loss primitives (differentiable in beta_hat)
# -----------------------------------------------------------------------------

def per_query_objective_torch(
    Xc: torch.Tensor,
    yc: torch.Tensor,
    beta: torch.Tensor,
    lam: float,
    rho_at_beta: torch.Tensor,
) -> torch.Tensor:
    """F(beta) = (1/2N) ||y - X beta||^2 + lam * sum_j rho(|beta_j|).
    Gradient flows through beta and rho_at_beta."""
    resid = yc - Xc @ beta
    quad = 0.5 / Xc.shape[0] * torch.dot(resid, resid)
    pen = rho_at_beta.sum()
    return quad + lam * pen


def evaluate_rho_at_beta(
    beta: torch.Tensor,
    segments: SegmentList,
) -> torch.Tensor:
    """Compute rho(|beta_j|) for each j using piecewise quadratic eval.

    Differentiable in beta because each segment is c + d*t + 0.5*h*t^2.
    Uses a smoothed-abs near 0 to keep gradients well-defined.
    """
    abs_beta = torch.sqrt(beta * beta + 1e-20)
    P = beta.shape[0]
    rho = torch.zeros_like(beta)
    for j in range(P):
        t = abs_beta[j]
        t_f = float(t.item())
        matched = False
        for seg in segments:
            if (seg.lo - 1e-12) <= t_f <= (seg.hi + 1e-12):
                rho[j] = seg.c + seg.d * t + 0.5 * seg.h * t * t
                matched = True
                break
        if not matched:
            last = segments[-1]
            rho[j] = last.c + last.d * t + 0.5 * last.h * t * t
    return rho


@dataclass
class QueryLossInputs:
    """Container for one query's loss-relevant tensors."""
    Xc: torch.Tensor                     # (N, P) centered, no grad
    yc: torch.Tensor                     # (N,)  centered, no grad
    beta_hat: torch.Tensor               # (P,)  MODEL output, requires_grad
    lam: float                           # absolute lambda
    segments: SegmentList                # penalty
    G: Optional[torch.Tensor] = None     # cached Gram (P, P)
    a: Optional[torch.Tensor] = None     # cached X^T y / N (P,)
    L_lipschitz: Optional[float] = None  # ||X||^2 / N estimate
    beta_star: Optional[torch.Tensor] = None       # reference solution (distill; off in FPK v2)
    beta_star_reliable: bool = False


def compute_L_obj_term(inputs: QueryLossInputs) -> torch.Tensor:
    """L_obj = log(1 + F(beta_hat) / (F(0) + 1e-6))."""
    packed = _pack_query_loss_inputs([inputs])
    seg_pack = _pack_segments_for_batch(
        packed.segments,
        device=packed.beta.device,
        dtype=packed.beta.dtype,
    )
    return _compute_L_obj_batched(
        packed.Xc,
        packed.yc,
        packed.beta,
        packed.lam,
        seg_pack,
    )[0]


def compute_L_kkt_term(inputs: QueryLossInputs) -> torch.Tensor:
    """L_kkt = log(1 + (R / 0.05)^2). Differentiable through beta via g(beta)."""
    packed = _pack_query_loss_inputs([inputs])
    seg_pack = _pack_segments_for_batch(
        packed.segments,
        device=packed.beta.device,
        dtype=packed.beta.dtype,
    )
    return _compute_L_kkt_batched(
        packed.G,
        packed.a,
        packed.beta,
        packed.lam,
        seg_pack,
    )[0]


def compute_L_prox_term(inputs: QueryLossInputs) -> torch.Tensor:
    """L_prox = log(1 + (P / 0.05)^2)."""
    packed = _pack_query_loss_inputs([inputs])
    seg_pack = _pack_segments_for_batch(
        packed.segments,
        device=packed.beta.device,
        dtype=packed.beta.dtype,
    )
    return _compute_L_prox_batched(
        packed.G,
        packed.a,
        packed.beta,
        packed.lam,
        packed.L_lipschitz,
        seg_pack,
    )[0]


@dataclass
class _PackedQueryLossInputs:
    Xc: torch.Tensor
    yc: torch.Tensor
    beta: torch.Tensor
    lam: torch.Tensor
    G: torch.Tensor
    a: torch.Tensor
    L_lipschitz: torch.Tensor
    segments: List[SegmentList]


@dataclass
class _PackedSegments:
    lo: torch.Tensor
    hi: torch.Tensor
    c: torch.Tensor
    d: torch.Tensor
    h: torch.Tensor
    valid: torch.Tensor
    d0: torch.Tensor


def _batched_power_iter_largest_eigval(G: torch.Tensor, n_iter: int = 5) -> torch.Tensor:
    """Batched PSD spectral-norm estimate without host scalar synchronization."""
    with torch.no_grad():
        M, P = G.shape[0], G.shape[1]
        v = torch.ones(M, P, dtype=G.dtype, device=G.device)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-12)
        L = torch.ones(M, dtype=G.dtype, device=G.device)
        for _ in range(n_iter):
            v_new = torch.einsum("mpq,mq->mp", G, v)
            L = v_new.norm(dim=1).clamp_min(1e-30)
            v = v_new / L[:, None]
    return L.clamp_min(1e-8)


def _pack_query_loss_inputs(query_inputs: Sequence[QueryLossInputs]) -> _PackedQueryLossInputs:
    if len(query_inputs) == 0:
        raise ValueError("_pack_query_loss_inputs requires at least one query")

    beta0 = query_inputs[0].beta_hat
    device = beta0.device
    dtype = beta0.dtype
    N, P = query_inputs[0].Xc.shape
    segments: List[SegmentList] = []
    lam_parts: List[torch.Tensor] = []
    X_parts: List[torch.Tensor] = []
    y_parts: List[torch.Tensor] = []
    beta_parts: List[torch.Tensor] = []

    all_have_Ga = all(q.G is not None and q.a is not None for q in query_inputs)
    all_have_L = all(q.L_lipschitz is not None for q in query_inputs)
    G_parts: List[torch.Tensor] = []
    a_parts: List[torch.Tensor] = []
    L_parts: List[torch.Tensor] = []

    for qi, q in enumerate(query_inputs):
        if q.Xc.dim() != 2 or q.yc.dim() != 1 or q.beta_hat.dim() != 1:
            raise ValueError("QueryLossInputs expects Xc=(N,P), yc=(N,), beta_hat=(P,)")
        if q.Xc.shape != (N, P) or q.yc.shape[0] != N or q.beta_hat.shape[0] != P:
            raise ValueError(
                f"all query losses must share N/P; query 0={(N, P)}, "
                f"query {qi} Xc={tuple(q.Xc.shape)}, yc={tuple(q.yc.shape)}, "
                f"beta={tuple(q.beta_hat.shape)}"
            )
        X_parts.append(q.Xc.to(device=device, dtype=dtype))
        y_parts.append(q.yc.to(device=device, dtype=dtype))
        beta_parts.append(q.beta_hat.to(device=device, dtype=dtype))
        lam_parts.append(torch.as_tensor(q.lam, device=device, dtype=dtype).reshape(()))
        segments.append(q.segments)
        if all_have_Ga:
            assert q.G is not None and q.a is not None
            G_parts.append(q.G.to(device=device, dtype=dtype))
            a_parts.append(q.a.to(device=device, dtype=dtype))
        if all_have_L:
            L_parts.append(torch.as_tensor(q.L_lipschitz, device=device, dtype=dtype).reshape(()))

    Xc = torch.stack(X_parts, dim=0)
    yc = torch.stack(y_parts, dim=0)
    beta = torch.stack(beta_parts, dim=0)
    lam = torch.stack(lam_parts, dim=0)

    if all_have_Ga:
        G = torch.stack(G_parts, dim=0)
        a = torch.stack(a_parts, dim=0)
    else:
        Xf = Xc.float()
        yf = yc.float()
        N_f = float(N)
        G = torch.einsum("mnp,mnq->mpq", Xf, Xf) / N_f
        a = torch.einsum("mnp,mn->mp", Xf, yf) / N_f
        G = G.to(dtype=dtype)
        a = a.to(dtype=dtype)

    if all_have_L:
        L_lipschitz = torch.stack(L_parts, dim=0).clamp_min(1e-8)
    else:
        L_lipschitz = _batched_power_iter_largest_eigval(G.float()).to(dtype=dtype)

    return _PackedQueryLossInputs(
        Xc=Xc,
        yc=yc,
        beta=beta,
        lam=lam,
        G=G,
        a=a,
        L_lipschitz=L_lipschitz,
        segments=segments,
    )


def _pack_segments_for_batch(
    segments_list: Sequence[SegmentList],
    device: torch.device,
    dtype: torch.dtype,
) -> _PackedSegments:
    if len(segments_list) == 0:
        raise ValueError("_pack_segments_for_batch requires at least one segment list")
    S = max(len(segs) for segs in segments_list)
    M = len(segments_list)
    lo = torch.zeros(M, S, device=device, dtype=dtype)
    hi = torch.zeros(M, S, device=device, dtype=dtype)
    c = torch.zeros(M, S, device=device, dtype=dtype)
    d = torch.zeros(M, S, device=device, dtype=dtype)
    h = torch.zeros(M, S, device=device, dtype=dtype)
    valid = torch.zeros(M, S, device=device, dtype=torch.bool)
    d0 = torch.zeros(M, device=device, dtype=dtype)

    for m, segs in enumerate(segments_list):
        if not segs:
            raise ValueError(f"empty SegmentList at batch index {m}")
        d0[m] = float(d0_from_segments(segs))
        for s, seg in enumerate(segs):
            lo[m, s] = float(seg.lo)
            hi[m, s] = float(seg.hi)
            c[m, s] = float(seg.c)
            d[m, s] = float(seg.d)
            h[m, s] = float(seg.h)
            valid[m, s] = True
    return _PackedSegments(lo=lo, hi=hi, c=c, d=d, h=h, valid=valid, d0=d0)


def _gather_segment_values(values: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return torch.gather(values, dim=1, index=idx)


def _segment_match_indices(t: torch.Tensor, seg_pack: _PackedSegments, tol: float = 1e-12) -> torch.Tensor:
    lo = seg_pack.lo[:, None, :]
    hi = seg_pack.hi[:, None, :]
    valid = seg_pack.valid[:, None, :]
    match = valid & (lo - tol <= t[:, :, None]) & (t[:, :, None] <= hi + tol)
    first_idx = match.to(torch.long).argmax(dim=-1)
    last_idx = seg_pack.valid.to(torch.long).sum(dim=1).clamp_min(1) - 1
    return torch.where(match.any(dim=-1), first_idx, last_idx[:, None].expand_as(first_idx))


def _evaluate_rho_at_beta_batched(beta: torch.Tensor, seg_pack: _PackedSegments) -> torch.Tensor:
    abs_beta = torch.sqrt(beta * beta + 1e-20)
    idx = _segment_match_indices(abs_beta, seg_pack)
    c = _gather_segment_values(seg_pack.c, idx)
    d = _gather_segment_values(seg_pack.d, idx)
    h = _gather_segment_values(seg_pack.h, idx)
    return c + d * abs_beta + 0.5 * h * abs_beta * abs_beta


def _evaluate_subgradient_interval_batched(
    beta: torch.Tensor,
    seg_pack: _PackedSegments,
    tol: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = beta.abs()
    lo = seg_pack.lo[:, None, :]
    hi = seg_pack.hi[:, None, :]
    valid = seg_pack.valid[:, None, :]
    match = valid & (lo - tol <= t[:, :, None]) & (t[:, :, None] <= hi + tol)
    deriv = seg_pack.d[:, None, :] + seg_pack.h[:, None, :] * t[:, :, None]

    inf = torch.full_like(deriv, float("inf"))
    neg_inf = torch.full_like(deriv, -float("inf"))
    d_left = torch.where(match, deriv, inf).amin(dim=-1)
    d_right = torch.where(match, deriv, neg_inf).amax(dim=-1)

    has_match = match.any(dim=-1)
    last_idx = seg_pack.valid.to(torch.long).sum(dim=1).clamp_min(1) - 1
    last_deriv = _gather_segment_values(
        seg_pack.d + seg_pack.h * t.new_zeros(seg_pack.h.shape),
        last_idx[:, None].expand(beta.shape[0], beta.shape[1]),
    ) + _gather_segment_values(
        seg_pack.h,
        last_idx[:, None].expand(beta.shape[0], beta.shape[1]),
    ) * t
    d_left = torch.where(has_match, d_left, last_deriv)
    d_right = torch.where(has_match, d_right, last_deriv)
    d_min = torch.minimum(d_left, d_right)
    d_max = torch.maximum(d_left, d_right)

    d0 = seg_pack.d0
    zero = beta == 0
    pos = beta > 0
    sub_lo = torch.where(pos, d_min, -d_max)
    sub_hi = torch.where(pos, d_max, -d_min)
    sub_lo = torch.where(zero, -d0[:, None], sub_lo)
    sub_hi = torch.where(zero, d0[:, None], sub_hi)
    return sub_lo, sub_hi, d0


def _vectorized_scalar_prox_torch_batched(
    v: torch.Tensor,
    tau: torch.Tensor,
    seg_pack: _PackedSegments,
) -> torch.Tensor:
    tau_mp = tau.reshape(-1, 1).to(device=v.device, dtype=v.dtype).expand_as(v)
    a = v.abs()
    sign = torch.where(v >= 0, torch.ones_like(v), -torch.ones_like(v))

    lo = seg_pack.lo[:, None, :]
    hi = seg_pack.hi[:, None, :]
    c = seg_pack.c[:, None, :]
    d = seg_pack.d[:, None, :]
    h = seg_pack.h[:, None, :]
    valid = seg_pack.valid[:, None, :]
    tau_mps = tau_mp[:, :, None]
    a_mps = a[:, :, None]

    denom = 1.0 + tau_mps * h
    t_star = (a_mps - tau_mps * d) / denom.clamp(min=-1e30, max=1e30)
    finite_hi = torch.isfinite(hi)
    station_valid = (
        valid
        & (denom.abs() > 1e-10)
        & (t_star >= lo - 1e-12)
        & (t_star <= hi + 1e-12)
        & (t_star >= 0.0)
    )
    lo_valid = valid & torch.isfinite(lo) & (lo >= 0.0)
    hi_valid = valid & finite_hi & (hi >= 0.0)

    t_station = t_star
    rho_station = c + d * t_station + 0.5 * h * t_station * t_station
    obj_station = 0.5 * (t_station - a_mps).square() + tau_mps * rho_station

    t_lo = lo.expand_as(t_station)
    rho_lo = c + d * t_lo + 0.5 * h * t_lo * t_lo
    obj_lo = 0.5 * (t_lo - a_mps).square() + tau_mps * rho_lo

    t_hi = hi.expand_as(t_station)
    rho_hi = c + d * t_hi + 0.5 * h * t_hi * t_hi
    obj_hi = 0.5 * (t_hi - a_mps).square() + tau_mps * rho_hi

    t_zero = torch.zeros(v.shape[0], v.shape[1], 1, dtype=v.dtype, device=v.device)
    obj_zero = 0.5 * a[:, :, None].square()

    big = torch.full_like(obj_station, float("inf"))
    obj_station = torch.where(station_valid, obj_station, big)
    obj_lo = torch.where(lo_valid.expand_as(obj_lo), obj_lo, big)
    obj_hi = torch.where(hi_valid.expand_as(obj_hi), obj_hi, big)

    candidates = torch.cat([t_station, t_lo, t_hi, t_zero], dim=-1)
    objectives = torch.cat([obj_station, obj_lo, obj_hi, obj_zero], dim=-1)
    best_idx = objectives.argmin(dim=-1, keepdim=True)
    best_t = torch.gather(candidates, dim=-1, index=best_idx).squeeze(-1)
    return (sign * best_t).detach()


def _compute_L_obj_batched(
    Xc: torch.Tensor,
    yc: torch.Tensor,
    beta: torch.Tensor,
    lam: torch.Tensor,
    seg_pack: _PackedSegments,
) -> torch.Tensor:
    rho = _evaluate_rho_at_beta_batched(beta, seg_pack)
    Xf = Xc.float()
    yf = yc.float()
    bf = beta.float()
    pred = torch.einsum("mnp,mp->mn", Xf, bf)
    resid = yf - pred
    N = Xc.shape[1]
    F_hat = 0.5 / float(N) * resid.square().sum(dim=1) + lam.float() * rho.float().sum(dim=1)
    F_zero = 0.5 / float(N) * yf.square().sum(dim=1)
    return torch.log1p(F_hat / (F_zero + 1e-6))


def _compute_L_kkt_batched(
    G: torch.Tensor,
    a: torch.Tensor,
    beta: torch.Tensor,
    lam: torch.Tensor,
    seg_pack: _PackedSegments,
) -> torch.Tensor:
    Gf = G.float()
    af = a.float()
    bf = beta.float()
    lamf = lam.float()
    g = torch.einsum("mpq,mq->mp", Gf, bf) - af
    sub_lo, sub_hi, d0 = _evaluate_subgradient_interval_batched(bf, seg_pack)
    scaled_lo = lamf[:, None] * sub_lo.float()
    scaled_hi = lamf[:, None] * sub_hi.float()
    target = -g
    r = torch.clamp(target - scaled_hi, min=0.0) + torch.clamp(scaled_lo - target, min=0.0)
    denom = torch.abs(g) + lamf[:, None] * (d0.float()[:, None] + 1.0) + 1e-6
    R = torch.sqrt(torch.mean((r / denom).square(), dim=1) + 1e-30)
    return torch.log1p((R / 0.05).square())


def _compute_L_prox_batched(
    G: torch.Tensor,
    a: torch.Tensor,
    beta: torch.Tensor,
    lam: torch.Tensor,
    L_lipschitz: torch.Tensor,
    seg_pack: _PackedSegments,
) -> torch.Tensor:
    Gf = G.float()
    af = a.float()
    bf = beta.float()
    lamf = lam.float()
    Lf = L_lipschitz.float()
    g = torch.einsum("mpq,mq->mp", Gf, bf) - af
    eta = 0.99 / Lf.clamp_min(1e-8)
    tau = eta * lamf
    v_inner = bf - eta[:, None] * g
    z = _vectorized_scalar_prox_torch_batched(v_inner.detach(), tau.detach(), seg_pack).detach()
    diff = bf - z.float()
    num = torch.sqrt(diff.square().sum(dim=1) + 1e-30)
    den = (
        torch.sqrt(bf.square().sum(dim=1) + 1e-30)
        + torch.sqrt(z.float().square().sum(dim=1) + 1e-30)
        + 1e-6
    )
    P_resid = num / den
    return torch.log1p((P_resid / 0.05).square())


def compute_L_delta_term(beta_hat: torch.Tensor) -> torch.Tensor:
    """Tiny L2 regularizer on beta magnitude (W_DELTA = 1e-4 by default)."""
    return torch.dot(beta_hat, beta_hat)


# -----------------------------------------------------------------------------
# E.4 Batched total loss
# -----------------------------------------------------------------------------

@dataclass
class BatchLossOutputs:
    total: torch.Tensor
    L_obj: torch.Tensor
    L_kkt: torch.Tensor
    L_prox: torch.Tensor
    L_delta: torch.Tensor
    w_obj: float
    w_kkt: float
    w_prox: float
    w_delta: float


def compute_batch_loss(
    query_inputs: Sequence[QueryLossInputs],
    *,
    step: int = 0,
    w_obj: float = W_OBJ,
    w_kkt: float = W_KKT,
    w_prox: float = W_PROX,
    w_delta: float = W_DELTA,
    beta_hat_all: Optional[torch.Tensor] = None,
) -> BatchLossOutputs:
    """FPK v2 batched loss. Distillation, Ridge teacher, and z-loss are
    off by default (per fpk_decisions.md §1 Tier 1). All four reported
    weights can be overridden at call time without rebuilding the model.
    """
    if len(query_inputs) == 0:
        zero = torch.tensor(0.0, requires_grad=False)
        return BatchLossOutputs(
            total=zero, L_obj=zero, L_kkt=zero, L_prox=zero, L_delta=zero,
            w_obj=w_obj, w_kkt=w_kkt, w_prox=w_prox, w_delta=w_delta,
        )

    packed = _pack_query_loss_inputs(query_inputs)
    seg_pack = _pack_segments_for_batch(
        packed.segments,
        device=packed.beta.device,
        dtype=packed.beta.dtype,
    )

    L_obj = _compute_L_obj_batched(
        packed.Xc,
        packed.yc,
        packed.beta,
        packed.lam,
        seg_pack,
    ).mean()
    if w_kkt > 0.0:
        L_kkt = _compute_L_kkt_batched(
            packed.G,
            packed.a,
            packed.beta,
            packed.lam,
            seg_pack,
        ).mean()
    else:
        L_kkt = torch.tensor(0.0, dtype=L_obj.dtype, device=L_obj.device)
    if w_prox > 0.0:
        L_prox = _compute_L_prox_batched(
            packed.G,
            packed.a,
            packed.beta,
            packed.lam,
            packed.L_lipschitz,
            seg_pack,
        ).mean()
    else:
        L_prox = torch.tensor(0.0, dtype=L_obj.dtype, device=L_obj.device)

    if beta_hat_all is not None:
        L_delta = compute_L_delta_term(beta_hat_all.flatten())
    else:
        L_delta = packed.beta.square().sum(dim=1).mean()

    total = (
        w_obj * L_obj
        + w_kkt * L_kkt
        + w_prox * L_prox
        + w_delta * L_delta
    )
    return BatchLossOutputs(
        total=total, L_obj=L_obj, L_kkt=L_kkt, L_prox=L_prox, L_delta=L_delta,
        w_obj=w_obj, w_kkt=w_kkt, w_prox=w_prox, w_delta=w_delta,
    )


def _make_loss_smoke_queries(
    M: int = 3,
    N: int = 16,
    P: int = 9,
) -> Tuple[List[QueryLossInputs], torch.Tensor]:
    X = torch.randn(M, N, P)
    y = torch.randn(M, N)
    Xc = X - X.mean(dim=1, keepdim=True)
    yc = y - y.mean(dim=1, keepdim=True)
    beta = (0.1 * torch.randn(M, P)).requires_grad_()
    segment_bank = [
        lasso_segments(),
        ridge_segments(),
        elastic_net_segments(0.5),
        mcp_segments(3.0),
        scad_segments(3.7),
    ]
    queries: List[QueryLossInputs] = []
    for i in range(M):
        Xi = Xc[i]
        yi = yc[i]
        G = Xi.T @ Xi / N
        a = Xi.T @ yi / N
        L = float(torch.linalg.eigvalsh(G.double()).amax().clamp_min(1e-8).item())
        queries.append(
            QueryLossInputs(
                Xc=Xi,
                yc=yi,
                beta_hat=beta[i],
                lam=float(0.02 + 0.01 * i),
                segments=segment_bank[i % len(segment_bank)],
                G=G,
                a=a,
                L_lipschitz=L,
            )
        )
    return queries, beta


def _run_loss_path_smoke_tests() -> None:
    _sep("Loss path smoke: prox gating and legacy wrappers")
    torch.manual_seed(123)
    queries, beta = _make_loss_smoke_queries()

    original_prox = globals()["_compute_L_prox_batched"]

    def _raising_prox(*args, **kwargs):
        raise RuntimeError("prox should be skipped when w_prox == 0")

    globals()["_compute_L_prox_batched"] = _raising_prox
    try:
        skipped = compute_batch_loss(queries, w_kkt=0.0, w_prox=0.0)
    finally:
        globals()["_compute_L_prox_batched"] = original_prox
    assert float(skipped.L_prox.detach().item()) == 0.0
    skipped.total.backward(retain_graph=True)
    assert beta.grad is not None and torch.isfinite(beta.grad).all()
    print("w_prox=0 skips prox and backward remains finite.")

    if beta.grad is not None:
        beta.grad.zero_()
    computed = compute_batch_loss(queries, w_kkt=0.0, w_prox=0.1)
    assert torch.isfinite(computed.L_prox)
    print(f"w_prox>0 L_prox:         {float(computed.L_prox.detach()):.6f}")

    q = queries[0]
    packed = _pack_query_loss_inputs([q])
    seg_pack = _pack_segments_for_batch(
        packed.segments,
        device=packed.beta.device,
        dtype=packed.beta.dtype,
    )
    obj_direct = _compute_L_obj_batched(packed.Xc, packed.yc, packed.beta, packed.lam, seg_pack)[0]
    kkt_direct = _compute_L_kkt_batched(packed.G, packed.a, packed.beta, packed.lam, seg_pack)[0]
    prox_direct = _compute_L_prox_batched(
        packed.G,
        packed.a,
        packed.beta,
        packed.lam,
        packed.L_lipschitz,
        seg_pack,
    )[0]
    assert torch.allclose(compute_L_obj_term(q), obj_direct, rtol=1e-4, atol=1e-5)
    assert torch.allclose(compute_L_kkt_term(q), kkt_direct, rtol=1e-4, atol=1e-5)
    assert torch.allclose(compute_L_prox_term(q), prox_direct, rtol=1e-4, atol=1e-5)
    print("legacy helper wrappers agree with batched helpers.")

    src = textwrap.dedent(inspect.getsource(compute_batch_loss))
    assert "if w_prox > 0.0:" in src
    tree = ast.parse(src)
    forbidden = {"compute_L_obj_term", "compute_L_kkt_term", "compute_L_prox_term"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ListComp):
            text = ast.get_source_segment(src, node) or ""
            assert not any(name in text for name in forbidden)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden
    print("compute_batch_loss static prox-gate/per-query-helper check passed.")


if __name__ == "__main__":
    _run_loss_path_smoke_tests()


# -----------------------------------------------------------------------------
# E.5 FpkConfig + make_fpk_model factory
# -----------------------------------------------------------------------------

@dataclass
class FpkConfig:
    """Aggregator that mirrors configs/fpk_v2.yaml `model:` and
    `penalty_shape:` sections. Passed to ``make_fpk_model``.
    """
    # Krylov backbone
    krylov_dim: int = 32
    num_decoder_channels: int = 12
    krylov_reorth_passes: int = 2
    krylov_eps: float = 1e-7

    # Hidden sizes
    cond_dim: int = 24
    cond_hidden: int = 64
    query_mixer_hidden: int = 64
    feature_head_hidden: int = 512
    prox_hidden: int = 64
    base_head_hidden: int = 64

    # Numerics
    cond_eps: float = 1e-12
    diag_eps: float = 1e-8

    # Toggles
    use_penalty_shape_encoding: bool = True
    use_penalty_type_embed: bool = False
    use_raw_penalty_params: bool = False
    prox_use_bounded_inputs: bool = True
    condition_np: bool = False
    use_krylov_ridge_base: bool = True
    use_bilinear_channels: bool = True
    use_slope_prox: bool = False

    # Init
    zero_init_query_mixer: bool = True
    zero_init_feature_head: bool = True
    zero_init_learned_prox: bool = True

    # Cache
    retain_U: bool = True
    retain_centered_data_in_encoding: bool = False

    # Penalty shape (DEFAULT_META_DIM = 5 fixed per user decision)
    penalty_grid_size: int = 24
    penalty_grid_channels: int = 4
    meta_dim: int = 5
    shape_feature_dim: int = 53            # = 24 * 4 + 5
    penalty_shape_hidden: int = 128
    penalty_shape_depth: int = 2
    penalty_shape_code_dim: int = 24


def make_fpk_model(
    cfg: Optional[FpkConfig] = None,
    device: Union[str, torch.device] = "cpu",
    **overrides: Any,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Build the FPK v2 model.

    Returns ``(model, applied_config_dict)``. ``applied_config_dict`` is
    the full set of legacy + patched fields actually used (handy for
    logging / checkpoint metadata).

    Hard sanity checks per fpk_decisions.md §8:
      * meta_dim == 5
      * shape_feature_dim == 101
    """
    if cfg is None:
        cfg = FpkConfig()

    # Apply overrides
    cfg_dict = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
    for k, v in overrides.items():
        if k not in cfg_dict:
            raise ValueError(f"unknown FpkConfig override: {k!r}")
        cfg_dict[k] = v

    # Hard checks
    if cfg_dict["meta_dim"] != 5:
        raise ValueError(
            f"FPK v2 fixes meta_dim = 5 (user decision); got {cfg_dict['meta_dim']}"
        )
    if cfg_dict["shape_feature_dim"] != 5 + cfg_dict["penalty_grid_size"] * cfg_dict["penalty_grid_channels"]:
        raise ValueError(
            f"shape_feature_dim ({cfg_dict['shape_feature_dim']}) must equal "
            f"meta_dim + grid_size * grid_channels "
            f"= 5 + {cfg_dict['penalty_grid_size']} * {cfg_dict['penalty_grid_channels']}"
        )

    # Build the patched penalty-shape conditioning embed manually (so we
    # can pass penalty_shape_depth/hidden) and inject it after
    # make_integrated_model.
    legacy_overrides = dict(
        # Krylov backbone fields
        krylov_dim=cfg_dict["krylov_dim"],
        krylov_reorth_passes=cfg_dict["krylov_reorth_passes"],
        krylov_eps=cfg_dict["krylov_eps"],
        num_decoder_channels=cfg_dict["num_decoder_channels"],
        use_bilinear_channels=cfg_dict["use_bilinear_channels"],
        cond_dim=cfg_dict["cond_dim"],
        cond_hidden=cfg_dict["cond_hidden"],
        query_mixer_hidden=cfg_dict["query_mixer_hidden"],
        feature_head_hidden=cfg_dict["feature_head_hidden"],
        prox_hidden=cfg_dict["prox_hidden"],
        base_head_hidden=cfg_dict["base_head_hidden"],
        cond_eps=cfg_dict["cond_eps"],
        diag_eps=cfg_dict["diag_eps"],
        use_krylov_ridge_base=cfg_dict["use_krylov_ridge_base"],
        use_slope_prox=cfg_dict["use_slope_prox"],
        zero_init_query_mixer=cfg_dict["zero_init_query_mixer"],
        zero_init_feature_head=cfg_dict["zero_init_feature_head"],
        zero_init_learned_prox=cfg_dict["zero_init_learned_prox"],
        condition_np=cfg_dict["condition_np"],
        retain_U=cfg_dict["retain_U"],
        retain_centered_data_in_encoding=cfg_dict["retain_centered_data_in_encoding"],
        # Patched fields
        use_penalty_shape_encoding=cfg_dict["use_penalty_shape_encoding"],
        use_penalty_type_embed=cfg_dict["use_penalty_type_embed"],
        use_raw_penalty_params=cfg_dict["use_raw_penalty_params"],
        prox_use_bounded_inputs=cfg_dict["prox_use_bounded_inputs"],
        penalty_shape_in_dim=cfg_dict["shape_feature_dim"],
        penalty_shape_code_dim=cfg_dict["penalty_shape_code_dim"],
        penalty_shape_hidden=cfg_dict["penalty_shape_hidden"],
    )

    model, full_cfg = make_integrated_model(device=device, **legacy_overrides)

    # If the user requested a non-default encoder depth, rebuild the
    # cond_embed with the right depth (make_integrated_model uses default
    # depth=2 via PenaltyShapeEncoder's new default; if cfg requests
    # something else, swap it in).
    if cfg_dict["penalty_shape_depth"] != 2:
        # Read legacy fields we need from the existing cond_embed
        if not cfg_dict["use_penalty_shape_encoding"]:
            raise ValueError(
                "penalty_shape_depth override has no effect when "
                "use_penalty_shape_encoding=False"
            )
        new_embed = PenaltyShapeAwareCondEmbed(
            out_dim=cfg_dict["cond_dim"],
            hidden=cfg_dict["cond_hidden"],
            eps=cfg_dict["cond_eps"],
            use_np=cfg_dict["condition_np"],
            penalty_shape_in_dim=cfg_dict["shape_feature_dim"],
            penalty_shape_code_dim=cfg_dict["penalty_shape_code_dim"],
            penalty_shape_hidden=cfg_dict["penalty_shape_hidden"],
            penalty_shape_depth=cfg_dict["penalty_shape_depth"],
            use_penalty_type_embed=cfg_dict["use_penalty_type_embed"],
        )
        model.cond_embed = new_embed.to(torch.device(device))
        full_cfg["penalty_shape_depth"] = cfg_dict["penalty_shape_depth"]

    full_cfg["fpk_config"] = cfg_dict
    return model, full_cfg


# -----------------------------------------------------------------------------
# E.6 Public API
# -----------------------------------------------------------------------------

__all__ = [
    # Backbone config + base decoder
    "KrylovQueryDecoderConfig",
    "KrylovRowSpaceQueryDecoder",
    "IntegratedRidgeKrylovDecoder",
    "PatchedConfigOverrides",
    "make_integrated_model",
    "make_v4_model",        # backwards-compat alias

    # FPK v2 top-level factory
    "FpkConfig",
    "make_fpk_model",

    # Penalty registry
    "PID_LASSO", "PID_RIDGE", "PID_ENET", "PID_SCAD", "PID_MCP",
    "NUM_PENALTY_TYPES",
    "PENALTY_REGISTRY",

    # Penalty shape (grid-evaluated penalty features)
    "DEFAULT_GRID_SIZE",
    "DEFAULT_GRID_CHANNELS",
    "DEFAULT_META_DIM",
    "DEFAULT_SHAPE_FEATURE_DIM",
    "default_penalty_grid",
    "standardized_penalty_rho_drho",
    "make_penalty_features",
    "make_penalty_features_from_segments",
    "PenaltyShapeEncoder",
    "PenaltyShapeAwareCondEmbed",
    "BoundedLearnedProxOp",

    # Segment representation (internal; not encoder-facing)
    "Segment", "SegmentList",
    "eval_piecewise_rho", "eval_piecewise_drho_at", "d0_from_segments",
    "lasso_segments", "ridge_segments", "elastic_net_segments",
    "mcp_segments", "scad_segments", "theta_to_segments",
    "known_family_segments",

    # Scalar prox + closed forms
    "scalar_abs_penalty_prox", "vectorized_scalar_prox",
    "lasso_prox_closed_form", "ridge_prox_closed_form",

    # Random theta sampler
    "CurriculumStage", "EASY", "MEDIUM", "HARD", "STAGES",
    "D0_BUCKETS", "tail_floor_for_coercivity", "stage_for_step",
    "RandomThetaSample", "sample_random_theta",

    # Family + batch sampler
    "FamilyConfig",
    "LASSO_FAMILY", "RIDGE_FAMILY", "ENET_FAMILY", "MCP_FAMILY", "SCAD_FAMILY",
    "with_weight",
    "PenaltySamplerConfig", "PenaltySampler", "SampledBatch",

    # Loss (numpy reference)
    "kkt_residual_per_coord_np", "kkt_rms_normalized_np",
    "penalty_objective_np", "penalty_objective_at_zero_np",
    "power_iter_largest_eigval", "prox_stationarity_residual_np",

    # Loss (torch / training)
    "QueryLossInputs", "BatchLossOutputs",
    "W_OBJ", "W_KKT", "W_PROX", "W_RIDGE", "W_DELTA",
    "w_distill_schedule", "w_z_schedule",
    "per_query_objective_torch", "evaluate_rho_at_beta",
    "compute_L_obj_term", "compute_L_kkt_term",
    "compute_L_prox_term", "compute_L_delta_term",
    "compute_batch_loss",
]
