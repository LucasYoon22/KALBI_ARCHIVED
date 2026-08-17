"""
penalty_sampler_v4.py — Aggressive random penalty generation
=============================================================
Drop-in replacement stages + new bucket weights for more diverse penalties.

Problems with original:
1. Easy/Medium stages have tiny TV/neg budgets → derivative almost flat → lasso
2. d0 bucket weights favor lasso_like (45%)
3. jump_scale too small → noise is negligible after projection

Fixes:
1. ALL stages get aggressive budgets (at least medium-level)
2. Bucket weights rebalanced: more denseish + hard_sparse
3. Higher jump_scale for actual non-trivial penalty shapes
4. Known penalty ratio increased: 40% known (was 20%)
5. Known families weighted toward MCP/SCAD (non-convex)

Usage:
    from penalty_sampler_v4 import STAGES_V4, D0_BUCKETS_V4, stage_for_step_v4
    # Then replace STAGES and D0_BUCKETS in the training loop
"""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    tv_budget: float
    neg_budget: float
    jump_scale: float


# ============================================================
# V4 Stages — much more aggressive
# ============================================================
# Original:
#   EASY   = tv=0.75, neg=0.45, jump=0.05  → 100% lasso_like
#   MEDIUM = tv=1.50, neg=1.00, jump=0.12  → 70% lasso_like
#   HARD   = tv=3.00, neg=2.00, jump=0.25  → diverse

# V4: aggressive — even "easy" generates diverse penalties
EASY_V4 = CurriculumStage(name="easy", tv_budget=8.00, neg_budget=5.00, jump_scale=0.60)
MEDIUM_V4 = CurriculumStage(name="medium", tv_budget=8.00, neg_budget=5.00, jump_scale=0.60)
HARD_V4 = CurriculumStage(name="hard", tv_budget=8.00, neg_budget=5.00, jump_scale=0.60)

STAGES_V4 = {"easy": EASY_V4, "medium": MEDIUM_V4, "hard": HARD_V4}


# ============================================================
# V4 d0 buckets — rebalanced for diversity
# ============================================================
# Original: denseish=10%, weak_sparse=25%, lasso_like=45%, hard_sparse=20%
# Problem: lasso_like dominates

# V4: flatten the distribution, more extreme buckets
# Added ultra_sparse for steep initial rise (d0=3.5~10)
D0_BUCKETS_V4 = (
    ("denseish",     (0.02, 0.10), 0.15),   # 10% → 15%
    ("weak_sparse",  (0.10, 0.50), 0.20),   # 25% → 20%
    ("lasso_like",   (0.50, 1.50), 0.20),   # 45% → 20% (halved!)
    ("hard_sparse",  (1.50, 3.50), 0.25),   # 20% → 25%
    ("ultra_sparse", (3.50, 10.0), 0.20),   # NEW: steep initial rise, fast penalty
)


# ============================================================
# V4 stage schedule — faster graduation
# ============================================================
def stage_for_step_v4(step: int, boundaries: Sequence[int] = (5000, 15000)) -> str:
    """V4: always hard. No curriculum."""
    return "hard"


# ============================================================
# V4 known-family ratio + weights
# ============================================================
# Original: known_family_ratio = 0.20, random_theta_ratio = 0.80
# All 5 families equal weight

# V4: known_family_ratio = 0.40, with MCP/SCAD weighted higher
KNOWN_FAMILY_RATIO_V4 = 0.40
RANDOM_THETA_RATIO_V4 = 0.60

# Non-convex families get 2x weight
FAMILY_WEIGHTS_V4 = {
    "lasso": 1.0,
    "ridge": 1.0,
    "elastic_net": 1.5,
    "mcp": 2.5,       # was 1.0
    "scad": 2.5,       # was 1.0
}


# ============================================================
# Validation relaxation for extreme buckets
# ============================================================
# Original validity checks are too strict for denseish/hard_sparse:
#   d0_bounds = (0.02, 3.5)         → OK
#   rho_at_1_pre_norm = (0.05, 20)  → too tight for hard_sparse
#   tail_slope_min = 0.02           → BLOCKS MCP/SCAD-like penalties!

VALIDITY_OVERRIDES_V4 = {
    "rho_at_1_pre_norm": (0.01, 50.0),    # was (0.05, 20.0)
    "tail_slope_min": 0.0,                 # was 0.02 → 0.0: allow flat tails (MCP/SCAD)
    "d0_bounds": (0.01, 5.0),              # was (0.02, 3.5)
    "tail_floor": 0.0,                     # for flat-tail mode only
    "tail_floor_coercive": 0.02,           # for coercive mode (lasso/ridge-like, tail goes up)
}


# ============================================================
# Flat-tail penalty generation helpers
# ============================================================
# MCP/SCAD key property: rho'(t) drops to 0 after some threshold.
# Current staircase generator can't produce this because:
#   1. _project_derivative_path clips min_value >= tail_floor (0.02)
#   2. _validity_checks rejects tail_slope < 0.02
# Fix: explicitly allow derivative path to reach 0.

FLAT_TAIL_RATIO = 0.70  # 70% of random penalties should have flat tails


def generate_flat_tail_derivative(
    K: int,
    rng,
    knots,
    d0: float,
    *,
    flatten_start_range: tuple = (0.3, 4.0),  # t where rho' starts dropping to 0
    flatten_end_range: tuple = (1.5, 6.0),    # t where rho' reaches 0
    tv_budget: float = 5.0,
) -> 'np.ndarray':
    """Generate derivative path that drops to 0 (MCP/SCAD-like).
    
    Supports high d0 (ultra_sparse): steep initial rise then flat.
    Example shapes:
        d0=1.0, flatten at 3.0: standard MCP-like
        d0=5.0, flatten at 1.0: steep spike then flat (hard thresholding)
        d0=8.0, flatten at 0.5: nearly binary penalty
    """
    import numpy as np
    
    # For high d0, flatten earlier (steep rise + early saturation)
    if d0 > 3.0:
        flatten_start = rng.uniform(0.3, 2.0)
        flatten_end = rng.uniform(flatten_start + 0.3, min(flatten_start + 2.0, 6.0))
    else:
        flatten_start = rng.uniform(*flatten_start_range)
        flatten_end = rng.uniform(max(flatten_start + 0.5, flatten_end_range[0]), flatten_end_range[1])
    
    deriv = np.zeros(K + 1)
    deriv[0] = d0
    
    for i, knot in enumerate(knots):
        if knot < flatten_start:
            # Before flattening: small random variation around d0
            noise = rng.laplace(scale=0.05) * d0
            deriv[i + 1] = max(0.0, deriv[i] + noise)
        elif knot < flatten_end:
            # Flattening zone: linearly interpolate to 0
            progress = (knot - flatten_start) / (flatten_end - flatten_start)
            target = d0 * (1.0 - progress)
            deriv[i + 1] = max(0.0, target + rng.laplace(scale=0.02) * d0)
        else:
            # After flatten_end: stay at 0
            deriv[i + 1] = 0.0
    
    # Ensure monotone decrease in the flattening zone
    for i in range(1, len(deriv)):
        if knots[i-1] >= flatten_start:
            deriv[i] = min(deriv[i], deriv[i-1])
    
    return deriv


# ============================================================
# rho(6)=1 normalization
# ============================================================
# V4/V5 pins the penalty scale at t=6. Lambda controls the objective strength.
#
# This keeps the full [0,6] penalty shape on a comparable scale and matches
# fpk_v4_common / penalty_sampler_v5.

def normalize_theta_rows_at_6(
    theta_rows: 'list[list[float]]',
) -> 'list[list[float]]':
    """Divide all weights by rho_raw(6) so that rho(6) = 1."""
    import math
    rho_at_6 = _eval_rho_rows_at_impl(theta_rows, 6.0)
    if rho_at_6 is None or abs(rho_at_6) < 1e-8 or not math.isfinite(rho_at_6):
        return []
    inv = 1.0 / rho_at_6
    return [[row[0], row[1], row[2] * inv, row[3]] for row in theta_rows]


def _eval_rho_rows_at_impl(
    theta_rows: 'list[list[float]]', t: float
) -> 'float | None':
    """Evaluate rho(t) from hinge-basis theta rows."""
    import math
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


def normalize_known_segments_at_6(segments: 'list') -> 'list':
    """Re-scale known-family segments so rho(6) = 1.
    
    For use with lasso_segments(), mcp_segments(), scad_segments(), etc.
    Returns new segment list with scaled c, d, h values.
    """
    from model_fpk import eval_piecewise_rho, Segment
    rho_6 = eval_piecewise_rho(6.0, segments)
    if abs(rho_6) < 1e-8:
        return segments  # already ~0, can't normalize
    inv = 1.0 / rho_6
    return [
        Segment(lo=s.lo, hi=s.hi, c=s.c * inv, d=s.d * inv, h=s.h * inv)
        for s in segments
    ]


# ============================================================
# V4 Knot settings — more knots, uniform distribution
# ============================================================
# Original: K=16, strata=(0.60, 0.25, 0.15)
#   [0.05, 1.25]: 10 knots, [1.25, 3.0]: 4 knots, [3.0, 6.0]: 2 knots
#   → t>3 구간이 2개 knot으로 부드럽지 못함
#
# V4: K=32, strata=(0.30, 0.35, 0.35)
#   [0.05, 1.25]: ~10 knots, [1.25, 3.0]: ~11 knots, [3.0, 6.0]: ~11 knots
#   → t>3에서도 충분한 해상도

DEFAULT_K_V4 = 32                            # was 16
KNOT_STRATA_V4 = (0.30, 0.35, 0.35)          # was (0.60, 0.25, 0.15)


def print_comparison():
    """Print side-by-side comparison of original vs V4."""
    print("=" * 70)
    print("STAGE COMPARISON")
    print("=" * 70)
    print(f"{'Stage':<10} {'Original TV/Neg/Jump':<30} {'V4 TV/Neg/Jump':<30}")
    print("-" * 70)
    orig = [
        ("easy",   0.75, 0.45, 0.05),
        ("medium", 1.50, 1.00, 0.12),
        ("hard",   3.00, 2.00, 0.25),
    ]
    v4 = [
        ("easy",   1.50, 1.00, 0.15),
        ("medium", 3.00, 2.00, 0.25),
        ("hard",   5.00, 3.50, 0.40),
    ]
    for o, n in zip(orig, v4):
        print(f"{o[0]:<10} tv={o[1]:.2f} neg={o[2]:.2f} jump={o[3]:.2f}     "
              f"tv={n[1]:.2f} neg={n[2]:.2f} jump={n[3]:.2f}")

    print()
    print("=" * 70)
    print("BUCKET WEIGHT COMPARISON")
    print("=" * 70)
    orig_b = [("denseish", 0.10), ("weak_sparse", 0.25), ("lasso_like", 0.45), ("hard_sparse", 0.20)]
    v4_b = [("denseish", 0.20), ("weak_sparse", 0.25), ("lasso_like", 0.25), ("hard_sparse", 0.30)]
    for o, n in zip(orig_b, v4_b):
        print(f"  {o[0]:<15} {o[1]:.2f} → {n[1]:.2f}")

    print()
    print(f"Known family ratio: 0.20 → {KNOWN_FAMILY_RATIO_V4:.2f}")
    print(f"Stage boundaries: (10K, 30K) → (5K, 15K)")


if __name__ == "__main__":
    print_comparison()


# ============================================================
# V4 Feature Builder: uniform grid, no tanh, raw values
# ============================================================
# Original problems:
#   1. Grid log-spaced: 12/24 points in t<0.2, only 8 in t=1~6
#   2. tanh squashes values → information loss
#   3. t=8.0 outside penalty domain [0,6]
#
# V4: uniform grid on [0, 6], raw values, vectorized
#   shape_feature_dim = G*2 + 5 = 53 (no t_feat, no rho_prime_prime)

import math as _math
import torch as _torch
import numpy as _np

V4_GRID_SIZE = 48


def make_v4_grid(G: int = V4_GRID_SIZE, device=None):
    """Log-uniform grid on [0, 6]: extremely dense near 0, sparser toward 6."""
    if device is None:
        device = _torch.device('cpu')
    eps = 0.05
    u = _torch.linspace(_math.log(eps), _math.log(6.0 + eps), G, device=device, dtype=_torch.float32)
    return _torch.exp(u) - eps


def _segments_to_tensors(segments_grid, max_segs=40):
    """Pack segments_grid [[SegmentList]*Q]*B into padded tensors for vectorized eval.

    Returns:
        lo:    (B, Q, S) float32
        hi:    (B, Q, S) float32
        c:     (B, Q, S) float32
        d:     (B, Q, S) float32
        h:     (B, Q, S) float32
        valid: (B, Q, S) bool
        d0:    (B, Q) float32 — first segment's d value
    """
    B = len(segments_grid)
    Q = len(segments_grid[0]) if B > 0 else 0

    # Find max segment count
    S = 0
    for b in range(B):
        for q in range(Q):
            S = max(S, len(segments_grid[b][q]))
    S = min(S, max_segs)

    lo = _np.full((B, Q, S), 0.0, dtype=_np.float32)
    hi = _np.full((B, Q, S), 1e30, dtype=_np.float32)
    c  = _np.zeros((B, Q, S), dtype=_np.float32)
    d  = _np.zeros((B, Q, S), dtype=_np.float32)
    h  = _np.zeros((B, Q, S), dtype=_np.float32)
    valid = _np.zeros((B, Q, S), dtype=bool)
    d0 = _np.zeros((B, Q), dtype=_np.float32)

    for b in range(B):
        for q in range(Q):
            segs = segments_grid[b][q]
            n_s = min(len(segs), S)
            for s in range(n_s):
                seg = segs[s]
                lo[b, q, s] = seg.lo
                hi[b, q, s] = min(seg.hi, 1e30)
                c[b, q, s] = seg.c
                d[b, q, s] = seg.d
                h[b, q, s] = seg.h
                valid[b, q, s] = True
            # d0 = first segment's d
            if len(segs) > 0:
                d0[b, q] = segs[0].d

    return (
        _torch.from_numpy(lo),
        _torch.from_numpy(hi),
        _torch.from_numpy(c),
        _torch.from_numpy(d),
        _torch.from_numpy(h),
        _torch.from_numpy(valid),
        _torch.from_numpy(d0),
    )


def _vectorized_eval_rho(t_grid, lo, hi, c, d, h, valid):
    """Evaluate rho(t) for all (B, Q) penalties at all G grid points.

    Args:
        t_grid: (G,)
        lo, hi, c, d, h: (B, Q, S)
        valid: (B, Q, S) bool

    Returns:
        rho: (B, Q, G)
    """
    G = t_grid.shape[0]
    B, Q, S = lo.shape

    # (B, Q, S, G)
    t = t_grid.view(1, 1, 1, G).expand(B, Q, S, G)
    lo_e = lo.unsqueeze(-1).expand(B, Q, S, G)
    hi_e = hi.unsqueeze(-1).expand(B, Q, S, G)
    c_e = c.unsqueeze(-1).expand(B, Q, S, G)
    d_e = d.unsqueeze(-1).expand(B, Q, S, G)
    h_e = h.unsqueeze(-1).expand(B, Q, S, G)
    valid_e = valid.unsqueeze(-1).expand(B, Q, S, G)

    # rho = c + d*t + 0.5*h*t^2 for matching segment
    rho_all = c_e + d_e * t + 0.5 * h_e * t * t  # (B, Q, S, G)

    # Match: t in [lo, hi]
    match = valid_e & (t >= lo_e) & (t <= hi_e)  # (B, Q, S, G)

    # Take first matching segment (argmax on bool gives first True)
    idx = match.to(_torch.long).argmax(dim=2)  # (B, Q, G)
    rho = _torch.gather(rho_all, 2, idx.unsqueeze(2)).squeeze(2)  # (B, Q, G)

    return rho


def _vectorized_eval_drho(t_grid, lo, hi, d, h, valid):
    """Evaluate rho'(t) for all (B, Q) penalties at all G grid points.

    rho'(t) = d + h*t within matching segment.
    """
    G = t_grid.shape[0]
    B, Q, S = lo.shape

    t = t_grid.view(1, 1, 1, G).expand(B, Q, S, G)
    lo_e = lo.unsqueeze(-1).expand(B, Q, S, G)
    hi_e = hi.unsqueeze(-1).expand(B, Q, S, G)
    d_e = d.unsqueeze(-1).expand(B, Q, S, G)
    h_e = h.unsqueeze(-1).expand(B, Q, S, G)
    valid_e = valid.unsqueeze(-1).expand(B, Q, S, G)

    drho_all = d_e + h_e * t  # (B, Q, S, G)
    match = valid_e & (t >= lo_e) & (t <= hi_e)
    idx = match.to(_torch.long).argmax(dim=2)
    drho = _torch.gather(drho_all, 2, idx.unsqueeze(2)).squeeze(2)

    return drho


def make_penalty_features_v4(
    segments_grid,   # [[SegmentList]*Q]*B
    device=None,
    G: int = V4_GRID_SIZE,
):
    """V4 penalty features: uniform grid, raw values, vectorized.

    Returns shape_features: (B, Q, G*4 + 5) = (B, Q, 101)
      - t_feat:   G values (normalized t position)
      - rho(t):   G values (raw)
      - rho'(t):  G values (raw)
      - rho''(t): G values (finite difference, raw)
      - meta:     5 scalars [d0, rho(1), rho'(6), TV, is_convex]
    """
    if device is None:
        device = _torch.device('cpu')

    grid_t = make_v4_grid(G, device=_torch.device('cpu'))
    dt = float(grid_t[1] - grid_t[0])

    # Pack segments into tensors (CPU, fast numpy loop)
    lo, hi, c_t, d_t, h_t, valid, d0 = _segments_to_tensors(segments_grid)
    B, Q = d0.shape

    # Vectorized rho and drho
    rho = _vectorized_eval_rho(grid_t, lo, hi, c_t, d_t, h_t, valid)   # (B, Q, G)
    drho = _vectorized_eval_drho(grid_t, lo, hi, d_t, h_t, valid)      # (B, Q, G)

    # Finite-difference curvature
    curv = _torch.zeros_like(drho)
    curv[..., 1:] = (drho[..., 1:] - drho[..., :-1]) / dt
    curv[..., 0] = curv[..., 1]

    # t_feat: normalized position [0, 1]
    t_feat = (grid_t / 6.0).view(1, 1, G).expand(B, Q, G)

    # Meta features (vectorized)
    # rho_at_1: interpolate from grid (t=1.0 is at index 1.0/dt = 1.0/(6/23) ≈ 3.83)
    from model_fpk import eval_piecewise_rho as _eval_rho_scalar
    rho_at_1 = _torch.zeros(B, Q)
    for b in range(B):
        for q in range(Q):
            rho_at_1[b, q] = _eval_rho_scalar(1.0, segments_grid[b][q])

    tail_slope = drho[..., -1]  # rho'(6)
    tv = _torch.sum(_torch.abs(_torch.diff(drho, dim=-1)), dim=-1)  # (B, Q)
    is_convex = _torch.where(
        _torch.any(_torch.diff(drho, dim=-1) < -1e-6, dim=-1),
        _torch.tensor(-1.0), _torch.tensor(1.0)
    ).expand(B, Q)

    meta = _torch.stack([d0, rho_at_1, tail_slope, tv, is_convex], dim=-1)  # (B, Q, 5)

    # Stack: (B, Q, G, 4) then flatten
    grid_feat = _torch.stack([rho, drho], dim=-1)  # curv bounded by tanh(/8)
    shape_features = _torch.cat([grid_feat.reshape(B, Q, G * 2), meta], dim=-1)  # (B, Q, G*4+5)

    return {
        "shape_features": shape_features.to(device),
        "d0": d0.to(device),
    }


V4_SHAPE_FEATURE_DIM = V4_GRID_SIZE * 2 + 5  # 24*4 + 5 = 101


def patch_model_fpk_sampler():
    import model_fpk
    import numpy as np
    import math
    import torch

    # 1. Patch stages, buckets, validity overrides
    model_fpk.STAGES = STAGES_V4
    model_fpk.D0_BUCKETS = D0_BUCKETS_V4
    model_fpk.tail_floor_for_coercivity = lambda: 0.0
    model_fpk._normalize_theta_rows = normalize_theta_rows_at_6
    model_fpk.VALIDITY_OVERRIDES = VALIDITY_OVERRIDES_V4
    model_fpk.stage_for_step = lambda step, boundaries=None: "hard"
    model_fpk.make_penalty_features_from_segments = make_penalty_features_v4

    # 2. Patch _sample_initial_slope_for_bucket to support ultra_sparse
    orig_initial_slope = model_fpk._sample_initial_slope_for_bucket
    def patched_initial_slope(target_bucket, rng):
        if target_bucket == "ultra_sparse":
            return float(np.exp(rng.uniform(math.log(3.5), math.log(10.0))))
        return orig_initial_slope(target_bucket, rng)
    model_fpk._sample_initial_slope_for_bucket = patched_initial_slope

    # 3. Patch _validity_checks to use VALIDITY_OVERRIDES_V4 defaults
    orig_validity_checks = model_fpk._validity_checks
    def patched_validity_checks(segments, *args, **kwargs):
        if "tail_slope_min" not in kwargs:
            kwargs["tail_slope_min"] = 0.0
        if "d0_bounds" not in kwargs:
            kwargs["d0_bounds"] = (0.01, 10.0) # Allow ultra_sparse
        if "rho_at_1_pre_norm" not in kwargs:
            kwargs["rho_at_1_pre_norm"] = (0.01, 50.0)
        return orig_validity_checks(segments, *args, **kwargs)
    model_fpk._validity_checks = patched_validity_checks

    # 4. Patch sample_random_theta to support flat tails
    orig_sample_random_theta = model_fpk.sample_random_theta
    def patched_sample_random_theta(stage, seed, **kwargs):
        K = kwargs.get("K", 32)
        knot_strata = kwargs.get("knot_strata", (0.3, 0.35, 0.35))
        tail_floor = kwargs.get("tail_floor", 0.0)
        
        rng = np.random.default_rng(seed)
        s = STAGES_V4["hard"]
        
        attempts = []
        max_attempts = kwargs.get("max_resample_attempts", 100)
        for attempt_idx in range(max_attempts):
            chosen_bucket, target_d0 = model_fpk._sample_d0_target(rng)
            knots = model_fpk._stratified_jittered_knots(K, rng, strata=knot_strata)
            
            is_flat = rng.uniform() < FLAT_TAIL_RATIO
            if is_flat:
                # Override target_d0 to be closer to MCP (pre-normalized d0 around 1.2 to 2.2)
                # and restrict flattening start to be in [1.5, 4.0]
                target_d0 = rng.uniform(1.2, 2.2)
                deriv = generate_flat_tail_derivative(
                    K, rng, knots, target_d0,
                    flatten_start_range=(1.5, 4.0),
                    flatten_end_range=(2.5, 6.0),
                    tv_budget=s.tv_budget
                )
            else:
                d_raw_0 = model_fpk._sample_initial_slope_for_bucket(chosen_bucket, rng)
                bias = model_fpk._trend_bias_for_target(
                    target_d0, d_raw_0, knots,
                    tv_budget=s.tv_budget, neg_budget=s.neg_budget,
                )
                noise = rng.laplace(scale=s.jump_scale, size=K)
                jumps_raw = bias + noise
                deriv = np.concatenate([[d_raw_0], d_raw_0 + np.cumsum(jumps_raw)])
                deriv = model_fpk._project_derivative_path(
                    deriv,
                    min_value=max(tail_floor, model_fpk.tail_floor_for_coercivity()),
                    max_value=12.0,
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
                
            pre_norm_rho_at_1 = model_fpk._eval_rho_rows_at(theta_rows, 1.0)
            if pre_norm_rho_at_1 is None or pre_norm_rho_at_1 <= 0.0:
                attempts.append({"attempt": attempt_idx, "reason": "pre_norm_rho_at_1_nonpos"})
                continue
                
            normalized = normalize_theta_rows_at_6(theta_rows)
            if not normalized:
                attempts.append({"attempt": attempt_idx, "reason": "normalize_failed"})
                continue
                
            theta_tensor = torch.tensor(normalized, dtype=torch.float64)
            segments = model_fpk.theta_to_segments(theta_tensor)
            
            derivative_levels = model_fpk._derivative_levels_from_segments(segments)
            tv_actual = float(np.sum(np.abs(np.diff(derivative_levels))))
            
            report = model_fpk._validity_checks(
                segments,
                test_grid=np.linspace(0.0, 6.0, 25),
                derivative_total_variation=tv_actual,
                tv_budget=s.tv_budget,
                pre_norm_rho_at_1=pre_norm_rho_at_1,
            )
            if not report.passed:
                attempts.append({"attempt": attempt_idx, "reason": report.rejection_reason})
                continue
                
            realized_d0 = model_fpk.d0_from_segments(segments)
            bucket = model_fpk._bucket_of(realized_d0)
            
            return model_fpk.RandomThetaSample(
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
                    "target_d0": target_d0,
                    "target_bucket_intended": chosen_bucket,
                    "is_flat_tail": is_flat,
                },
            )
        raise RuntimeError(
            f"patched_sample_random_theta exhausted {max_attempts} attempts; recent rejections: {attempts[-5:]}"
        )
    model_fpk.sample_random_theta = patched_sample_random_theta
