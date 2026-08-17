#!/usr/bin/env python3
"""Single-source response contract for both Exp122 feature priors."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


TARGET_CONTRACT = "sparse_linear_on_final_stored_float16_X_plus_gaussian_noise_loguniform_snr_v3"
TARGET_SEED_CONTRACT = (
    "(worker_seed + 1000003*batch_index + 2000033*(task_index+1) + 3000017) mod (2^32-1)"
)


def derive_target_seed(worker_seed: int, batch_index: int, task_index: int) -> int:
    modulus = 2**32 - 1
    seed = int(
        (
            int(worker_seed)
            + 1_000_003 * int(batch_index)
            + 2_000_033 * (int(task_index) + 1)
            + 3_000_017
        )
        % modulus
    )
    return seed if seed != 0 else 1


def _standardize(x: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    if not np.isfinite(mu).all() or not np.isfinite(sd).all():
        raise ValueError("nonfinite standardization statistics")
    sd = np.where(sd > eps, sd, 1.0)
    return np.nan_to_num((x - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    if not 0.0 < lo <= hi:
        raise ValueError(f"invalid log-uniform bounds: {lo}, {hi}")
    return float(np.exp(rng.uniform(math.log(lo), math.log(hi))))


def make_controlled_sparse_linear_target(
    x_float16: np.ndarray,
    *,
    target_seed: int,
    support_min: int,
    support_max: int,
    snr_min: float,
    snr_max: float,
    clip_value: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create y from the final stored/permuted X under one shared contract."""
    rng = np.random.default_rng(int(target_seed))
    if x_float16.dtype != np.float16 or x_float16.ndim != 2:
        raise ValueError(f"expected a 2D final float16 X, got {x_float16.dtype}/{x_float16.shape}")
    n, p = x_float16.shape
    hi = min(int(support_max), p)
    lo = min(max(1, int(support_min)), hi)
    support_size = lo if lo == hi else int(np.clip(round(_log_uniform(rng, float(lo), float(hi))), lo, hi))
    support = np.sort(rng.choice(p, size=support_size, replace=False).astype(np.int64))
    beta = rng.normal(0.0, 1.0, size=support_size).astype(np.float32)
    beta /= max(float(np.linalg.norm(beta)), 1e-8)

    # The response uses the post-filter, post-permutation, float16-persisted X
    # that the learner will actually read.  Float64 accumulation avoids
    # backend-dependent float16 BLAS behavior.
    x_signal = x_float16[:, support].astype(np.float64, copy=False)
    signal_raw = np.matmul(x_signal, beta.astype(np.float64)).astype(np.float32)
    signal = _standardize(signal_raw.reshape(-1, 1)).reshape(-1)
    if float(np.std(signal)) <= 1e-7:
        raise ValueError("degenerate sparse-linear signal")
    noise = _standardize(rng.normal(0.0, 1.0, size=(n, 1)).astype(np.float32)).reshape(-1)
    target_snr = _log_uniform(rng, snr_min, snr_max)
    scaled_noise = noise / math.sqrt(target_snr)
    empirical_snr = float(np.var(signal, dtype=np.float64) / np.var(scaled_noise, dtype=np.float64))
    y = _standardize((signal + scaled_noise).reshape(-1, 1)).reshape(-1)
    if clip_value > 0:
        y = np.clip(y, -clip_value, clip_value)
        y = _standardize(y.reshape(-1, 1)).reshape(-1)
    if not np.isfinite(y).all() or float(np.std(y)) <= 1e-7:
        raise ValueError("invalid controlled target")
    return y.astype(np.float32, copy=False), {
        "target_contract": TARGET_CONTRACT,
        "target_seed_contract": TARGET_SEED_CONTRACT,
        "target_seed": int(target_seed),
        "support_size": int(support_size),
        "support_indices": support.tolist(),
        "support_coefficients": [float(v) for v in beta.tolist()],
        "target_snr": float(target_snr),
        "empirical_signal_to_noise_variance_ratio": empirical_snr,
        "oracle_linear_r2": float(target_snr / (1.0 + target_snr)),
    }
