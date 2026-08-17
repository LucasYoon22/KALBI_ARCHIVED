"""
penalty_relu_mlp.py — PyTorch differentiable penalty via one-layer ReLU MLP
============================================================================

ρ(t) = b · |t| + Σ_k w_k · F.relu(|t| - t_k)

All penalty functions (known + random) → same ReLU MLP representation.
No argmax, no gather, no numpy. Pure PyTorch. Gradient flows everywhere.
"""

import math
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Any


# ═══════════════════════════════════════════════════════════════════════════════
# Core: evaluate ρ and ρ' via F.relu
# ═══════════════════════════════════════════════════════════════════════════════


def _v33_grid_list(K=96, t_max=6.0):
    import math as _m
    eps = 0.05; n_lo = K // 2; n_hi = K - n_lo
    lo = [_m.exp(_m.log(eps) + (_m.log(0.3+eps)-_m.log(eps))*i/(n_lo-1)) - eps for i in range(n_lo)]
    hf = [_m.exp(_m.log(0.3+eps) + (_m.log(t_max+eps)-_m.log(0.3+eps))*i/(n_hi)) - eps for i in range(n_hi+1)]
    return lo + hf[1:]   # 96 pts: 48 in [0,0.3], 48 in (0.3,t_max]
def _v33_grid_t(K=96, t_max=6.0, device=None, dtype=None):
    import torch as _t
    return _t.tensor(_v33_grid_list(K, t_max), device=device, dtype=(dtype or _t.float32))

def eval_rho(t: torch.Tensor, knots: torch.Tensor, b: torch.Tensor,
             weights: torch.Tensor) -> torch.Tensor:
    """ρ(|t|) = b·|t| + Σ_k w_k · F.relu(|t| - t_k)
    
    Args:
        t: (...) any shape
        knots: (K,) knot positions
        b: scalar tensor
        weights: (K,) ReLU weights
    Returns:
        (...) same shape as t
    """
    t_abs = t.abs()
    relu = F.relu(t_abs.unsqueeze(-1) - knots)  # (..., K)
    return b * t_abs + (relu * weights).sum(dim=-1)


def eval_rho_batched(t: torch.Tensor, knots: torch.Tensor, b: torch.Tensor,
                     weights: torch.Tensor) -> torch.Tensor:
    """Batched ρ. 
    
    Args:
        t: (M, P)
        knots: (M, K) or (K,)
        b: (M,) or scalar
        weights: (M, K)
    Returns:
        (M, P)
    """
    t_abs = t.abs()  # (M, P)

    if knots.dim() == 1:
        # shared knots: (M, P, 1) - (K,) → (M, P, K)
        relu = F.relu(t_abs.unsqueeze(-1) - knots)
    else:
        # per-query: (M, P, 1) - (M, 1, K) → (M, P, K)
        relu = F.relu(t_abs.unsqueeze(-1) - knots.unsqueeze(1))

    relu2 = 0.5 * relu * relu   # PWL basis: integral of relu(rho') -> 0.5*relu^2
    if weights.dim() == 1:
        penalty = (relu2 * weights).sum(dim=-1)
    else:
        penalty = (relu2 * weights.unsqueeze(1)).sum(dim=-1)

    if isinstance(b, (int, float)):
        return b * t_abs + penalty
    elif b.dim() == 0:
        return b * t_abs + penalty
    else:
        return b.unsqueeze(-1) * t_abs + penalty


# ═══════════════════════════════════════════════════════════════════════════════
# Conversion: ANY Segment list → (knots, b, weights) via uniform grid sampling
# ═══════════════════════════════════════════════════════════════════════════════

def segments_to_relu_params(segments, K: int = 96, t_max: float = 6.0,
                             device=None, dtype=torch.float32):
    """V24: LOG-UNIFORM knots = make_v4_grid (same grid the model SEES as shape features),
    so the loss rho/rho' (this basis) and the model-input rho/rho' (sampled on this grid)
    are on IDENTICAL points. Knots: m_k = exp(lin(log eps, log(t_max+eps), K)) - eps,
    m_0 = 0, m_{K-1} = t_max. Slopes use non-uniform interval widths."""
    if device is None:
        device = torch.device('cpu')
    eps = 0.05
    lo, hi = math.log(eps), math.log(t_max + eps)
    m = _v33_grid_list(K, t_max)  # V33 grid

    def _drho(t):
        for seg in segments:
            if seg.lo <= t <= seg.hi or (seg is segments[-1] and t > seg.lo):
                return seg.d + seg.h * t
        s = segments[-1]
        return s.d + s.h * t

    dv = [_drho(t) for t in m]                                  # rho'(m_k), exact from segments
    g = [(dv[k + 1] - dv[k]) / (m[k + 1] - m[k]) for k in range(K - 1)]   # interval slopes (rho'')
    b_val = dv[0]                                               # rho'(0)
    # rho'(t) = b + sum_k w_k * relu(t - m_k); w_0=g_0, w_k=g_k-g_{k-1}, last knot has no interval -> 0
    weight_list = [g[0]] + [g[k] - g[k - 1] for k in range(1, K - 1)] + [0.0]   # K weights

    knots = torch.tensor(m, device=device, dtype=dtype)
    b = torch.tensor(b_val, device=device, dtype=dtype)
    weights = torch.tensor(weight_list, device=device, dtype=dtype)
    return knots, b, weights


def batch_segments_to_relu_params(segments_list, K: int = 96, t_max: float = 6.0,
                                   device=None, dtype=torch.float32):
    """V24 batched: shared LOG-UNIFORM knots = make_v4_grid."""
    if device is None:
        device = torch.device('cpu')
    all_b, all_w = [], []
    for segs in segments_list:
        _, b, w = segments_to_relu_params(segs, K=K, t_max=t_max, device='cpu', dtype=dtype)
        all_b.append(b); all_w.append(w)
    eps = 0.05
    knots = _v33_grid_t(K, t_max, device, dtype)  # V33 grid
    return knots, torch.stack(all_b).to(device), torch.stack(all_w).to(device)


# ═══════════════════════════════════════════════════════════════════════════════
# Drop-in replacement for _fast_rho in loss computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_penalty_relu(beta_q: torch.Tensor, lambdas: torch.Tensor,
                          segments_grid: List[List[Any]],
                          K: int = 96, t_max: float = 6.0) -> torch.Tensor:
    """Drop-in replacement for _fast_rho penalty computation.
    Fully differentiable. No argmax/gather.
    
    Args:
        beta_q: (B, Q, P)
        lambdas: (B, Q)
        segments_grid: list of B lists, each with Q segment lists
    Returns:
        penalty: (B, Q)
    """
    B, Q, P = beta_q.shape
    device = beta_q.device
    dtype = beta_q.dtype

    # Flatten segments: B*Q segment lists
    flat_segs = []
    for b_segs in segments_grid:
        for q_segs in b_segs:
            flat_segs.append(q_segs)

    knots, b_vec, w_mat = batch_segments_to_relu_params(
        flat_segs, K=K, t_max=t_max, device=device, dtype=torch.float32)

    bf = beta_q.float().reshape(B * Q, P)  # (M, P)
    rho = eval_rho_batched(bf, knots, b_vec, w_mat)  # (M, P)
    rho_sum = rho.reshape(B, Q, P).sum(dim=-1)  # (B, Q)

    return lambdas.float() * rho_sum


def compute_Lobj_relu(*, Xc, yc, beta_q, lambdas, segments_grid,
                       w_obj=1.0, w_kkt=0.0, w_delta=0.0, K=48, t_max=6.0):
    """Drop-in replacement for compute_Lobj_ultrafast. Fully differentiable.
    """
    B, Q, P = beta_q.shape
    N = Xc.shape[1]
    device = beta_q.device

    bf = beta_q.float()
    Xf = Xc.float()
    yf = yc.float()
    lamf = lambdas.float()

    # Data term
    pred = torch.einsum("bnp,bqp->bqn", Xf, bf)
    resid = yf.unsqueeze(1) - pred
    data_term = 0.5 / float(N) * resid.square().sum(dim=-1)

    # Penalty term (differentiable!)
    penalty = compute_penalty_relu(beta_q, lamf, segments_grid, K=K, t_max=t_max)

    F_hat = data_term + penalty
    F_zero = 0.5 / float(N) * yf.square().sum(dim=-1)
    # log(1 + F_hat / F_zero) loss
    F_zero_3d = F_zero.unsqueeze(1).clamp_min(1e-12)
    L_obj = torch.log(1.0 + F_hat / F_zero_3d).mean()

    # KKT loss (differentiable)
    if w_kkt > 0:
        from model_fpk import _compute_L_kkt_batched, _PackedSegments
        # G = X'X/N, a = X'y/N per dataset
        Gmat = torch.einsum("bnp,bnq->bpq", Xf, Xf) / float(N)
        avec = torch.einsum("bnp,bn->bp", Xf, yf) / float(N)
        # Flatten (B,Q) -> (B*Q,)
        G_bq = Gmat.unsqueeze(1).expand(B, Q, P, P).reshape(B*Q, P, P)
        a_bq = avec.unsqueeze(1).expand(B, Q, P).reshape(B*Q, P)
        beta_bq = bf.reshape(B*Q, P)
        lam_bq = lamf.reshape(B*Q)
        # Pack segments
        seg_flat = []
        for b in range(B):
            for q in range(Q):
                seg_flat.append(segments_grid[b][q])
        from model_fpk import _pack_segments_for_batch
        seg_pack = _pack_segments_for_batch(seg_flat, device=device, dtype=torch.float32)
        L_kkt_vec = _compute_L_kkt_batched(G_bq, a_bq, beta_bq, lam_bq, seg_pack)
        L_kkt = L_kkt_vec.mean()
    else:
        L_kkt = torch.tensor(0.0, device=device)

    L_delta = bf.square().mean() if w_delta > 0 else torch.tensor(0.0, device=device)
    total = w_obj * L_obj + w_kkt * L_kkt + w_delta * L_delta

    # Separate MSE and penalty components (raw, not normalized)
    L_mse = data_term.mean().detach()
    L_pen = penalty.mean().detach()

    return {
        "total": total,
        "L_obj": L_obj,
        "L_mse": L_mse,
        "L_pen": L_pen,
        "L_kkt": L_kkt,
        "L_prox": torch.tensor(0.0, device=device),
        "L_delta": L_delta,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/workspace')
    sys.path.insert(0, '/users/minsuk/fm-lm')

    import model_fpk
    from penalty_sampler_v3 import STAGES_V3, D0_BUCKETS_V3, normalize_theta_rows_at_6
    model_fpk.STAGES = STAGES_V3
    model_fpk.D0_BUCKETS = D0_BUCKETS_V3
    model_fpk.tail_floor_for_coercivity = lambda: 0.0
    model_fpk.stage_for_step = lambda s, boundaries=None: 'hard'
    model_fpk._normalize_theta_rows = normalize_theta_rows_at_6

    from model_fpk import (sample_random_theta, lasso_segments, mcp_segments,
        scad_segments, elastic_net_segments, eval_piecewise_rho)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    penalties = [
        ('LASSO', lasso_segments()),
        ('MCP(g=3)', mcp_segments(3.0)),
        ('SCAD(a=3.7)', scad_segments(3.7)),
        ('ENet(a=0.5)', elastic_net_segments(0.5)),
    ]
    for i in range(4):
        try:
            s = sample_random_theta('hard', seed=200+i, K=32)
            penalties.append((f'Rand_{i}', s.segments))
        except:
            pass

    t_grid_np = np.linspace(0.001, 6, 500)
    t_grid = torch.tensor(t_grid_np, device=device, dtype=torch.float32)

    fig, axes = plt.subplots(len(penalties), 2, figsize=(14, 4 * len(penalties)))

    for idx, (name, segs) in enumerate(penalties):
        # Original
        rho_orig = np.array([eval_piecewise_rho(float(t), segs) for t in t_grid_np])

        # ReLU MLP
        knots, b, weights = segments_to_relu_params(segs, K=48, t_max=6.0, device=device)
        rho_mlp = eval_rho(t_grid, knots, b, weights).detach().cpu().numpy()

        rho_err = np.max(np.abs(rho_orig - rho_mlp))

        # Gradient test
        t_test = torch.tensor([0.5, 1.0, 2.0, 3.0], device=device, requires_grad=True)
        rho_test = eval_rho(t_test, knots, b, weights)
        rho_test.sum().backward()
        grads = t_test.grad.cpu().numpy()

        # Plot rho comparison
        axes[idx, 0].plot(t_grid_np, rho_orig, 'b-', linewidth=2.5, label='Original')
        axes[idx, 0].plot(t_grid_np, rho_mlp, 'r--', linewidth=1.5, label=f'ReLU K=48')
        axes[idx, 0].set_title(f'{name} — rho(t) [max err={rho_err:.2e}]', fontsize=11)
        axes[idx, 0].legend(fontsize=8); axes[idx, 0].grid(alpha=0.3)

        # Plot gradients
        t_dense = torch.linspace(0.01, 6, 200, device=device, requires_grad=False)
        t_dense.requires_grad_(True)
        rho_dense = eval_rho(t_dense, knots, b, weights)
        rho_dense.sum().backward()
        grad_dense = t_dense.grad.detach().cpu().numpy()
        
        from model_fpk import eval_piecewise_drho_at
        drho_orig = np.array([eval_piecewise_drho_at(float(t), segs)[0] for t in t_dense.detach().cpu().numpy()])
        
        axes[idx, 1].plot(t_dense.detach().cpu(), drho_orig, 'b-', linewidth=2.5, label="Original rho'")
        axes[idx, 1].plot(t_dense.detach().cpu(), grad_dense, 'r--', linewidth=1.5, label="autograd drho/dt")
        axes[idx, 1].set_title(f"{name} — rho'(t) via autograd", fontsize=11)
        axes[idx, 1].legend(fontsize=8); axes[idx, 1].grid(alpha=0.3)

        print(f"{name:20s}: err={rho_err:.2e}, grads@[0.5,1,2,3]={grads}")

    plt.suptitle('PyTorch F.relu penalty: rho + autograd gradient',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('/users/minsuk/fm-lm/relu_mlp_pytorch_v2.png', dpi=150, bbox_inches='tight')
    print('\nSaved: relu_mlp_pytorch_v2.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Fast KKT loss using ReLU basis (no segment packing needed)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_penalty_relu_cached(beta_q, lambdas, relu_knots, relu_b, relu_weights):
    """VECTORIZED F_hat penalty from PRECOMPUTED relu params (no per-step segment Python loop).
    beta_q (B,Q,P); lambdas (B,Q); relu_knots (K,); relu_b (B*Q,); relu_weights (B*Q,K)."""
    B, Q, P = beta_q.shape
    bf = beta_q.reshape(B * Q, P).float()
    rho = eval_rho_batched(bf, relu_knots, relu_b, relu_weights)   # (B*Q, P)
    rho_sum = rho.reshape(B, Q, P).sum(dim=-1)                     # (B, Q)
    return lambdas.float() * rho_sum


def compute_L_kkt_relu(*, Xc, yc, beta_q, lambdas, relu_knots, relu_b, relu_weights):
    """Fast differentiable KKT loss using ReLU basis.
    
    ρ(|t|) = b·|t| + Σ_k w_k·ReLU(|t| - t_k)
    ρ'(|t|) = b + Σ_k w_k·1(|t| > t_k)   for |t| > 0
    ρ'(0+) = b  (= d0)
    
    KKT: g_j + λ·∂ρ(|β_j|) ∋ 0
      where g_j = (Gβ - a)_j, G = X'X/N, a = X'y/N
    
    Args:
        Xc: (B, N, P) centered X
        yc: (B, N) centered y
        beta_q: (B, Q, P)
        lambdas: (B, Q)
        relu_knots: (K-1,) shared uniform knots
        relu_b: (B*Q,) initial slopes
        relu_weights: (B*Q, K-1) ReLU weights
    
    Returns:
        scalar L_kkt loss
    """
    B, Q, P = beta_q.shape
    N = Xc.shape[1]
    device = beta_q.device
    
    bf = beta_q.float()
    Xf = Xc.float()
    yf = yc.float()
    lamf = lambdas.float()
    
    # G = X'X/N, a = X'y/N per dataset
    G = torch.einsum("bnp,bnq->bpq", Xf, Xf) / float(N)  # (B, P, P)
    a = torch.einsum("bnp,bn->bp", Xf, yf) / float(N)     # (B, P)
    
    # g = Gβ - a for each query: (B, Q, P)
    g = torch.einsum("bpq,biq->bip", G, bf) - a.unsqueeze(1)
    
    # ρ'(|β|) using ReLU basis
    abs_beta = bf.abs()  # (B, Q, P)
    abs_beta_flat = abs_beta.reshape(B * Q, P)  # (BQ, P)
    
    # indicators: (BQ, P, K-1) — is |β_j| > t_k?
    relu_act = torch.relu(abs_beta_flat.unsqueeze(-1) - relu_knots.unsqueeze(0).unsqueeze(0))  # PWL
    
    # ρ'(|β|) = b + Σ w_k · 1(|β| > t_k)
    # relu_b: (BQ,), relu_weights: (BQ, K-1)
    drho = relu_b.unsqueeze(-1) + (relu_act * relu_weights.unsqueeze(1)).sum(dim=-1)  # PWL rho' (differentiable)
    drho = drho.reshape(B, Q, P)
    
    # d0 = ρ'(0+) = b (initial slope)
    d0 = relu_b.reshape(B, Q)  # (B, Q)
    
    # KKT residual
    sign_beta = bf.sign()
    is_nonzero = (abs_beta > 1e-12).float()
    
    # For β ≠ 0: r = g + λ·ρ'(|β|)·sign(β)
    r_nonzero = g + lamf.unsqueeze(-1) * drho * sign_beta
    
    # For β = 0: r = max(0, |g| - λ·d0)
    r_zero = torch.relu(g.abs() - lamf.unsqueeze(-1) * d0.unsqueeze(-1))
    
    r = is_nonzero * r_nonzero + (1.0 - is_nonzero) * r_zero
    
    # KKT = ||grad F_hat||_2 : SUM over P coords (matches F_hat lambda*sum_j rho); no per-coord denom
    R = torch.sqrt((r * r).sum(dim=-1) + 1e-30)  # (B, Q)
    
    L_kkt = R.mean()
    return L_kkt


# ── PWL shape-feature encoder: rho/rho' from the relu^2 basis (matches the loss) ──
def make_penalty_features_pwl(segments_grid, device=None, G: int = 96, t_max: float = 6.0, K: int = 96):
    """101-dim shape features [rho(G), rho'(G), meta(5)] computed from the PWL
    relu^2 basis, so the ENCODING's rho' is piecewise-linear (consistent with the
    F_hat / KKT loss). Drop-in for make_penalty_features_v4."""
    if device is None:
        device = torch.device('cpu')
    B = len(segments_grid)
    Q = len(segments_grid[0]) if B > 0 else 0
    flat = [segments_grid[b][q] for b in range(B) for q in range(Q)]
    knots, b, w = batch_segments_to_relu_params(flat, K=K, t_max=t_max, device=device)  # knots(K), b(M), w(M,K)
    M = len(flat)
    eps = 0.05
    u = torch.linspace(math.log(eps), math.log(t_max + eps), G, device=device, dtype=torch.float32)
    grid = _v33_grid_t(G, t_max, device, torch.float32)  # V33 grid
    relu_gk = torch.relu(grid[:, None] - knots[None, :])        # (G, K)
    bM = b.view(M, 1)
    rho = bM * grid[None, :] + 0.5 * (w[:, None, :] * (relu_gk[None, :, :] ** 2)).sum(-1)   # (M, G)
    drho = bM + (w[:, None, :] * relu_gk[None, :, :]).sum(-1)                                # (M, G), piecewise-linear
    # meta
    d0 = b                                                      # rho'(0)
    r1 = torch.relu(torch.tensor(1.0, device=device) - knots)   # (K,)
    rho_at_1 = b * 1.0 + 0.5 * (w * (r1[None, :] ** 2)).sum(-1)  # (M,)
    tail_slope = drho[:, -1]                                    # ~ rho'(6)
    tv = (drho[:, 1:] - drho[:, :-1]).abs().sum(-1)
    inc = (drho[:, 1:] - drho[:, :-1] >= -1e-6).all(dim=-1)
    is_convex = torch.where(inc, torch.ones(M, device=device), -torch.ones(M, device=device))
    meta = torch.stack([d0, rho_at_1, tail_slope, tv, is_convex], dim=-1)   # (M, 5)
    grid_feat = torch.stack([rho, drho], dim=-1).reshape(M, G * 2)          # (M, 96)
    shape = torch.cat([grid_feat, meta], dim=-1).reshape(B, Q, G * 2 + 5)   # (B, Q, 101)
    return {"shape_features": shape.to(device),
            "d0": d0.reshape(B, Q).to(device),
            "drho6": tail_slope.reshape(B, Q).to(device)}
