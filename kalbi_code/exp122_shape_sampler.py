#!/usr/bin/env python3
"""Single-source Exp122 N/P sampler shared by both feature priors."""

from __future__ import annotations

import math

import numpy as np


SHAPE_SAMPLER_NAME = "smallN10pct_coupled_lognormal_median_N2000_P10000_maxN40K_maxP200K_v3"


def sample_shape_pair(
    rng: np.random.Generator,
    *,
    n_min: int = 32,
    n_max: int = 40_000,
    main_n_min: int = 200,
    small_n_max: int = 199,
    small_n_probability: float = 0.10,
    p_min: int = 50,
    p_max: int = 200_000,
    max_cells: int = 24_000_000,
    p_median: float = 10_000.0,
    p_log_sigma: float = 0.75,
    raw_n_median: float = 5_700.0,
    n_log_sigma: float = 0.70,
) -> tuple[int, int]:
    if not (n_min <= small_n_max < main_n_min <= n_max):
        raise ValueError("invalid small/main N bounds")
    if not 0.0 <= small_n_probability < 0.5:
        raise ValueError("small_n_probability must lie in [0,0.5)")
    if n_min * p_max > max_cells:
        raise ValueError("max_cells cannot realize p_max at n_min")

    p = int(np.clip(round(np.exp(rng.normal(math.log(p_median), p_log_sigma))), p_min, p_max))
    if float(rng.random()) < small_n_probability:
        n_hi = min(small_n_max, int(max_cells // max(1, p)))
        if n_hi < n_min:
            p = min(p, int(max_cells // n_min))
            n_hi = min(small_n_max, int(max_cells // max(1, p)))
        if n_hi < n_min:
            raise RuntimeError(f"cannot satisfy small-N max_cells contract at P={p}")
        n = int(
            np.clip(
                math.floor(np.exp(rng.uniform(math.log(n_min), math.log(n_hi + 1)))),
                n_min,
                n_hi,
            )
        )
    else:
        # Main tasks retain N>=200, so P>120k is reserved for the small-N
        # component under the 24M-cell cap.
        p = min(p, int(max_cells // main_n_min))
        raw_n = int(
            np.clip(
                round(np.exp(rng.normal(math.log(raw_n_median), n_log_sigma))),
                main_n_min,
                n_max,
            )
        )
        n_cap = max(main_n_min, int(max_cells // max(1, p)))
        n = int(np.clip(min(raw_n, n_cap), main_n_min, n_max))
    if not (n_min <= n <= n_max and p_min <= p <= p_max and n * p <= max_cells):
        raise RuntimeError(f"shape sampler contract failure N={n}, P={p}")
    return n, p
