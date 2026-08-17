#!/usr/bin/env python3
#%%
"""Generate TabICLv2 features with a 50/50 TabICL/linear outcome mixture.

Contract
--------
* X always follows the V11 SCM2048-qfilter-BNN128 feature contract.
* Even outcome-source certificates retain the official TabICLv2 response.
* Odd outcome-source certificates replace y with a controlled linear model.
* Linear tasks alternate between P-referenced and N-referenced support
  samplers, giving 25/25/50 percent of all tasks respectively.
* Every linear support satisfies the strict identifiability guard ``k < N/3``.
* Linear target SNR is log-uniform on the closed contract range [0.2, 50].
* Every SCM and BNN feature independently draws ``variance ~ Exp(rate=1)``
  and receives samplewise ``error ~ Normal(0, variance)``.
* Direct tasks have an exact full-P correlation certificate. High-P tasks
  preserve the exact certificate for the SCM2048 core and use a deterministic
  sampled audit for the final P features to avoid an O(P^2) extension pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from generate_tabiclv2_graph_highdim_npz import TabICLv2GraphGenerator
from generate_tabpfn_v1_causal_scm_mlp512_groupdrop095_sparse_linear_v1 import (
    ExactLowCorrelationAccumulator,
    activation,
    atomic_save_npz,
    atomic_write_json,
    standardize,
)
from exp122_shape_sampler import SHAPE_SAMPLER_NAME, sample_shape_pair as shared_sample_shape_pair


GENERATOR_NAME = Path(__file__).name
DATASET_CONTRACT = (
    "tabiclv2_graph_scm2048_qfilter_bnn128_expvarnoise_"
    "mixed50_tabicl_linear_supportmix_pn_maxsupp0p6n_snr0p2to50_rq_v15"
)
PRIOR_NAME = (
    "tabiclv2_graph_scm_x_expvarnoise_qfiltered_core2048_then_bnn128_"
    "mixed50_official_or_controlled_linear_supportmix_pn_maxsupp0p6n_"
    "snr_loguniform_0p2_50_y_rq_v4"
)
DATA_POLICY = (
    "v14_tabicl_x_qfilter_noisy_core2048_bnn128_expvarnoise_"
    "mixed50_outcome_linear_pn50_maxsupp0p6n_snr0p2to50_"
    "sampled_final_audit_rq_v2"
)
EXPECTED_TABICL_COMMIT = "46b91961db4f8873dd049ec09990698a435e1e29"
SCM_RESPONSE_CONTRACT = (
    "official_tabiclv2_graph_scm_y_qualityfilter_center_scale_clip10_rescale_v2"
)
LINEAR_RESPONSE_CONTRACT = (
    "controlled_linear_y_supportmix_pn50_maxsupp0p6n_"
    "snr_loguniform_0p2_50_errorfamily_"
    "qualityfilter_center_scale_clip10_rescale_rq_v4"
)
MIXED_RESPONSE_CONTRACT = (
    "mixed50_official_tabicl_or_controlled_linear_supportmix_pn50_"
    "maxsupp0p6n_snr_loguniform_0p2_50_y_"
    "qualityfilter_center_scale_clip10_rescale_rq_v4"
)
FEATURE_NOISE_CONTRACT = "all_features_variance_iid_exponential_rate1_error_gaussian_v1"
BNN_CONTRACT = "fixed_noisy_qfiltered_scm2048_core_parents_random_gaussian_weight_prior_block128_depth1or2_v1"
CORRELATION_AUDIT_CONTRACT = "deterministic_uniform_feature_sample_float16_pearson_not_full_pair_certificate_v1"
CORE_CORRELATION_CERTIFICATE = "exact incremental all-pair Pearson after featurewise noise and float16 roundtrip"
REQUIRED_CORRELATION_THRESHOLD = 0.95
TABICL_FUNCTION_TYPES = "default"
GAUSSIAN_NOISE_FAMILY_ID = 0
NOISE_VARIANCE_RATE = 1.0
OUTCOME_SOURCE_TABICL = 0
OUTCOME_SOURCE_LINEAR = 1
OUTCOME_SOURCE_NAMES = {
    OUTCOME_SOURCE_TABICL: "tabicl",
    OUTCOME_SOURCE_LINEAR: "linear",
}
SUPPORT_REFERENCE_P = 0
SUPPORT_REFERENCE_N = 1
SUPPORT_REFERENCE_NAMES = {
    SUPPORT_REFERENCE_P: "p_based",
    SUPPORT_REFERENCE_N: "n_based",
}
SUPPORT_REFERENCE_PROBABILITIES = {
    "p_based": 0.5,
    "n_based": 0.5,
}
MAX_P_SUPPORT_REJECTION_DRAWS = 256
SUPPORT_REGIME_NAMES = (
    "ultra_sparse",
    "sparse",
    "moderate",
    "large",
    "very_dense",
)
SUPPORT_REGIME_PROBABILITIES = np.asarray(
    [0.15, 0.20, 0.25, 0.25, 0.15], dtype=np.float64
)
MAX_SUPPORT_FRACTION_OF_N = 0.60
SUPPORT_SELECTION_NAMES = ("random", "correlated_block", "mixed")
SUPPORT_SELECTION_PROBABILITIES = np.asarray(
    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64
)
SUPPORT_SELECTION_CONTRACT = (
    "random_or_seed_column_top_abs_corr_block_or_half_block_half_random_v1"
)
RAYLEIGH_TARGET_PROBABILITY = 0.85
RAYLEIGH_TARGET_MIN = 0.03
RAYLEIGH_TARGET_MAX = 30.0
RAYLEIGH_EIGH_MAX_K = 768
RAYLEIGH_ISOTROPIC_MIX = 0.15
RAYLEIGH_CONTRACT = (
    "subgram_eigen_targeted_two_mode_interpolation_with_isotropic_mix_v1"
)
COEFFICIENT_FAMILY_NAMES = (
    "gaussian",
    "laplace",
    "rademacher_loguniform",
    "student_t",
    "two_scale_gaussian",
)
COEFFICIENT_FAMILY_PROBABILITIES = np.asarray(
    [0.30, 0.20, 0.20, 0.15, 0.15], dtype=np.float64
)
ERROR_FAMILY_NAMES = (
    "gaussian",
    "student_t",
    "laplace",
    "logistic",
    "centered_lognormal",
    "contaminated_gaussian",
    "heteroskedastic_gaussian",
)
ERROR_FAMILY_PROBABILITIES = np.asarray(
    [0.25, 0.20, 0.15, 0.10, 0.10, 0.10, 0.10], dtype=np.float64
)
LINEAR_SNR_MIN = 0.2
LINEAR_SNR_MAX = 50.0
LINEAR_SNR_DISTRIBUTION = "log_uniform"
LINEAR_SNR_CONTRACT = "linear_target_snr_loguniform_0p2_50_v1"
SNR_TELEMETRY_BIN_NAMES = (
    "log_quantile_1",
    "log_quantile_2",
    "log_quantile_3",
    "log_quantile_4",
    "log_quantile_5",
)
SNR_TELEMETRY_BIN_EDGES = tuple(
    float(value)
    for value in np.geomspace(
        LINEAR_SNR_MIN,
        LINEAR_SNR_MAX,
        len(SNR_TELEMETRY_BIN_NAMES) + 1,
    )
)


class ConfiguredTabICLv2GraphGenerator(TabICLv2GraphGenerator):
    """Official TabICLv2 graph-SCM wrapper with explicit function policy."""

    def generate_base(self, n: int, q: int) -> tuple[np.ndarray, np.ndarray, int]:
        config = self.PriorConfig(
            filter_unpredictable_datasets=self.filter_unpredictable,
            filter_unpredictable_graphs=self.filter_unpredictable,
            fct_types=TABICL_FUNCTION_TYPES,
        )
        ds = self.PriorDataset(
            regression=True,
            batch_size=1,
            batch_size_per_gp=1,
            min_features=q,
            max_features=q,
            min_seq_len=n,
            max_seq_len=n + 1,
            prior_type="graph_scm",
            config=config,
            n_jobs=1,
            device="cpu",
            seq_len_per_gp=False,
            log_n_features=False,
            log_seq_len=False,
            min_train_size=0.8,
            max_train_size=0.81,
        )
        x, y, d, _, _ = ds.get_batch(1)
        return (
            x[0].detach().cpu().numpy().astype(np.float32, copy=False),
            y[0].detach().cpu().numpy().astype(np.float32, copy=False),
            int(d[0].item()),
        )


def quantile_summary(values: np.ndarray) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    q = np.quantile(a, [0.0, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0])
    return {
        "min": float(q[0]),
        "q05": float(q[1]),
        "q25": float(q[2]),
        "median": float(q[3]),
        "q75": float(q[4]),
        "q90": float(q[5]),
        "q95": float(q[6]),
        "q99": float(q[7]),
        "max": float(q[8]),
        "mean": float(np.mean(a, dtype=np.float64)),
        "std": float(np.std(a, dtype=np.float64)),
    }


def standardize_scm_response(
    response: np.ndarray,
    *,
    clip_value: float,
    min_unique_count: int,
    min_unique_ratio: float,
    max_dominant_ratio: float,
    min_standardized_iqr: float,
    round_decimals: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(response, dtype=np.float32).reshape(-1)
    if not np.isfinite(raw).all():
        raise ValueError("official SCM response is nonfinite")
    raw_mean = float(np.mean(raw, dtype=np.float64))
    raw_std = float(np.std(raw, dtype=np.float64))
    if not math.isfinite(raw_std) or raw_std <= 1e-7:
        raise ValueError("official SCM response is degenerate")

    standardized = (raw.astype(np.float64) - raw_mean) / raw_std
    rounded = np.round(standardized, decimals=int(round_decimals))
    _, counts = np.unique(rounded, return_counts=True)
    effective_unique_count = int(counts.size)
    required_unique_count = max(
        int(min_unique_count), int(math.ceil(float(min_unique_ratio) * raw.size))
    )
    dominant_ratio = float(counts.max() / raw.size)
    q25, q75 = np.quantile(standardized, [0.25, 0.75])
    standardized_iqr = float(q75 - q25)
    if effective_unique_count < required_unique_count:
        raise ValueError(
            f"official SCM response unique count {effective_unique_count} < {required_unique_count}"
        )
    if dominant_ratio > max_dominant_ratio:
        raise ValueError(
            f"official SCM response dominant ratio {dominant_ratio:.6f} > {max_dominant_ratio:.6f}"
        )
    if not math.isfinite(standardized_iqr) or standardized_iqr < min_standardized_iqr:
        raise ValueError(
            f"official SCM response standardized IQR {standardized_iqr:.6f} < {min_standardized_iqr:.6f}"
        )

    clip_count = int(np.count_nonzero(np.abs(standardized) > clip_value))
    clipped = np.clip(standardized, -clip_value, clip_value)
    clipped_mean = float(np.mean(clipped, dtype=np.float64))
    clipped_std = float(np.std(clipped, dtype=np.float64))
    if not math.isfinite(clipped_std) or clipped_std <= 1e-7:
        raise ValueError("clipped official SCM response is degenerate")
    final = ((clipped - clipped_mean) / clipped_std).astype(np.float32)
    return final, {
        "target_contract": SCM_RESPONSE_CONTRACT,
        "target_source": "official TabICLv2 PriorDataset graph_scm response",
        "official_target_retained": True,
        "target_raw_mean": raw_mean,
        "target_raw_std": raw_std,
        "target_effective_unique_count": effective_unique_count,
        "target_required_unique_count": required_unique_count,
        "target_effective_unique_ratio": float(effective_unique_count / raw.size),
        "target_dominant_ratio": dominant_ratio,
        "target_standardized_iqr": standardized_iqr,
        "target_round_decimals": int(round_decimals),
        "target_pre_rescale_clip_count": clip_count,
        "target_final_mean": float(np.mean(final, dtype=np.float64)),
        "target_final_std": float(np.std(final, dtype=np.float64)),
        "target_final_min": float(np.min(final)),
        "target_final_max": float(np.max(final)),
    }


def _choice_index(
    rng: np.random.Generator, probabilities: np.ndarray
) -> int:
    return int(rng.choice(len(probabilities), p=probabilities))


def sample_target_snr(
    rng: np.random.Generator,
) -> tuple[float, int, str]:
    target_snr = math.exp(
        float(
            rng.uniform(
                math.log(LINEAR_SNR_MIN),
                math.log(LINEAR_SNR_MAX),
            )
        )
    )
    telemetry_bin_id = int(
        np.searchsorted(
            np.asarray(SNR_TELEMETRY_BIN_EDGES[1:-1]),
            target_snr,
            side="right",
        )
    )
    telemetry_bin_id = min(
        telemetry_bin_id,
        len(SNR_TELEMETRY_BIN_NAMES) - 1,
    )
    return (
        target_snr,
        telemetry_bin_id,
        SNR_TELEMETRY_BIN_NAMES[telemetry_bin_id],
    )


def _sample_support_size_for_scale(
    scale: int,
    p: int,
    rng: np.random.Generator,
) -> tuple[int, int, str]:
    regime_id = _choice_index(rng, SUPPORT_REGIME_PROBABILITIES)
    regime = SUPPORT_REGIME_NAMES[regime_id]
    if regime == "ultra_sparse":
        k = int(rng.integers(1, min(4, p) + 1))
    elif regime == "sparse":
        low = min(5, p)
        high = min(64, p)
        if low >= high:
            k = high
        else:
            k = int(round(math.exp(rng.uniform(math.log(low), math.log(high)))))
    elif regime == "moderate":
        fraction = 10.0 ** float(rng.uniform(-4.0, -2.0))
        k = int(round(fraction * scale))
    elif regime == "large":
        fraction = 10.0 ** float(
            rng.uniform(math.log10(1e-2), math.log10(0.2))
        )
        k = int(round(fraction * scale))
    else:
        k = int(round(float(rng.uniform(0.2, 1.0)) * scale))
    return min(max(k, 1), p), regime_id, regime


def _sample_n_referenced_support_size(
    n: int,
    p: int,
    max_support: int,
    rng: np.random.Generator,
) -> tuple[int, int, str]:
    regime_id = _choice_index(rng, SUPPORT_REGIME_PROBABILITIES)
    regime = SUPPORT_REGIME_NAMES[regime_id]
    if regime == "ultra_sparse":
        k = int(rng.integers(1, min(4, max_support) + 1))
    elif regime == "sparse":
        low = min(5, max_support)
        high = min(64, max_support)
        if low >= high:
            k = high
        else:
            k = int(round(math.exp(rng.uniform(math.log(low), math.log(high)))))
    elif regime == "moderate":
        fraction = 10.0 ** float(rng.uniform(-4.0, -2.0))
        k = int(round(fraction * n))
    elif regime == "large":
        fraction = 10.0 ** float(
            rng.uniform(math.log10(1e-2), math.log10(0.2))
        )
        k = int(round(fraction * n))
    else:
        low = min(max_support, max(1, int(math.ceil(0.2 * n))))
        k = int(rng.integers(low, max_support + 1))
    return min(max(k, 1), max_support), regime_id, regime


def sample_support_size(
    n: int,
    p: int,
    rng: np.random.Generator,
    *,
    support_reference_id: int,
) -> tuple[int, int, str, dict[str, Any]]:
    max_support = min(p, int(MAX_SUPPORT_FRACTION_OF_N * (n - 1)))
    if max_support < 1:
        raise ValueError(
            f"N={n} and P={p} admit no support with "
            f"k <= {MAX_SUPPORT_FRACTION_OF_N} * (N-1)"
        )

    if support_reference_id == SUPPORT_REFERENCE_N:
        k, regime_id, regime = _sample_n_referenced_support_size(
            n, p, max_support, rng
        )
        attempts = 1
        rejected = 0
    elif support_reference_id == SUPPORT_REFERENCE_P:
        for attempts in range(1, MAX_P_SUPPORT_REJECTION_DRAWS + 1):
            k, regime_id, regime = _sample_support_size_for_scale(p, p, rng)
            if k <= max_support:
                break
        else:
            raise ValueError(
                "P-referenced support sampler failed conditional rejection: "
                f"N={n}, P={p}, max_support={max_support}"
            )
        rejected = attempts - 1
    else:
        raise ValueError(
            f"unknown support_reference_id={support_reference_id!r}"
        )

    if not 1 <= k <= p or k > max_support:
        raise AssertionError(
            "linear support contract failed: "
            f"N={n}, P={p}, k={k}, reference={support_reference_id}"
        )
    metadata = {
        "linear_support_reference_id": int(support_reference_id),
        "linear_support_reference": SUPPORT_REFERENCE_NAMES[
            support_reference_id
        ],
        "linear_support_sampling_attempts": int(attempts),
        "linear_support_rejected_n_third_draws": int(rejected),
        "linear_support_max_allowed": int(max_support),
        "linear_support_max_fraction_of_n": float(MAX_SUPPORT_FRACTION_OF_N),
        "linear_support_strict_n_third_verified": bool(3 * k < n),
    }
    return k, regime_id, regime, metadata


def select_support_indices(
    x64: np.ndarray, k: int, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pick the k support columns, sometimes as a correlated block.

    A random support drawn from a wide design has a sub-Gram whose largest
    eigenvalue stays near 1 + O(sqrt(k/n)), which caps R(beta) well below
    what real benchmarks show.  Since tr(G_S) = k, reaching R = w is only
    possible when k >= w and the support shares a common factor; by
    Gershgorin, w_max <= 1 + (k-1)*max_offdiag, so the corr-0.95 generation
    filter leaves ample headroom.  Seeding on one column and taking its top
    absolute correlations produces exactly such a factor cluster.
    """
    n, p = x64.shape
    mode_id = _choice_index(rng, SUPPORT_SELECTION_PROBABILITIES)
    mode = SUPPORT_SELECTION_NAMES[mode_id]
    if mode == "random" or k < 2 or k >= p:
        support = rng.choice(p, size=k, replace=False)
        return np.sort(support.astype(np.int64)), {
            "linear_support_selection_id": 0,
            "linear_support_selection": "random",
            "linear_support_selection_contract": SUPPORT_SELECTION_CONTRACT,
        }

    block_size = k if mode == "correlated_block" else max(2, k // 2)
    seed_column = int(rng.integers(p))

    mean = x64.mean(axis=0)
    seed_centered = x64[:, seed_column] - mean[seed_column]
    cross = x64.T @ seed_centered
    norm_squared = np.maximum(
        np.einsum("ij,ij->j", x64, x64) - n * mean * mean, 1e-24
    )
    denominator = np.sqrt(norm_squared * norm_squared[seed_column])
    correlation = np.abs(cross) / denominator
    correlation[seed_column] = np.inf

    block = np.argpartition(-correlation, block_size - 1)[:block_size]
    chosen = np.zeros(p, dtype=bool)
    chosen[block] = True
    shortfall = k - int(chosen.sum())
    if shortfall > 0:
        pool = np.flatnonzero(~chosen)
        chosen[
            rng.choice(pool, size=min(shortfall, pool.size), replace=False)
        ] = True

    finite = correlation[block][np.isfinite(correlation[block])]
    return np.flatnonzero(chosen).astype(np.int64), {
        "linear_support_selection_id": int(mode_id),
        "linear_support_selection": mode,
        "linear_support_selection_contract": SUPPORT_SELECTION_CONTRACT,
        "linear_support_seed_column": seed_column,
        "linear_support_block_size": int(block_size),
        "linear_support_block_mean_abs_corr": float(
            np.mean(finite) if finite.size else 0.0
        ),
        "linear_support_block_min_abs_corr": float(
            np.min(finite) if finite.size else 0.0
        ),
    }


def sample_rayleigh_targeted_coefficients(
    x_support: np.ndarray,
    base_values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build beta on the support so that R(beta) hits a sampled target.

    With standardized columns the sub-Gram  G = X_S^T X_S / n  has unit
    diagonal, hence  mean(eig(G)) = 1  for every support.  Writing
    beta = V c in the eigenbasis,

        R(beta) = sum_i c_i^2 w_i / sum_i c_i^2 ,

    so R is a convex combination of the eigenvalues and any target in
    [w_min, w_max] is reachable exactly by mixing two eigenvectors.
    A small isotropic component is added so beta is not literally
    rank-two in the eigenbasis; R is recomputed after the mix.
    """
    n, k = x_support.shape
    metadata: dict[str, Any] = {"linear_rayleigh_targeted": False}
    if k < 2 or k > RAYLEIGH_EIGH_MAX_K:
        return base_values, metadata

    gram = (x_support.T @ x_support) / float(n)
    gram = 0.5 * (gram + gram.T)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
    except np.linalg.LinAlgError:
        return base_values, metadata
    if not np.all(np.isfinite(eigenvalues)):
        return base_values, metadata

    eigenvalues = np.clip(eigenvalues, 0.0, None)
    low, high = float(eigenvalues[0]), float(eigenvalues[-1])
    if high - low < 1e-9:
        return base_values, metadata

    target = math.exp(
        rng.uniform(math.log(RAYLEIGH_TARGET_MIN), math.log(RAYLEIGH_TARGET_MAX))
    )
    target = float(np.clip(target, low + 1e-9, high - 1e-9))

    upper = int(np.searchsorted(eigenvalues, target, side="left"))
    upper = int(np.clip(upper, 1, k - 1))
    lower = upper - 1
    w_lo, w_hi = float(eigenvalues[lower]), float(eigenvalues[upper])
    if w_hi - w_lo < 1e-12:
        weight_hi = 0.5
    else:
        weight_hi = (target - w_lo) / (w_hi - w_lo)
    weight_hi = float(np.clip(weight_hi, 0.0, 1.0))

    coefficients = np.zeros(k, dtype=np.float64)
    sign_lo = 1.0 if rng.random() < 0.5 else -1.0
    sign_hi = 1.0 if rng.random() < 0.5 else -1.0
    coefficients[lower] = sign_lo * math.sqrt(1.0 - weight_hi)
    coefficients[upper] = sign_hi * math.sqrt(weight_hi)

    isotropic = np.asarray(base_values, dtype=np.float64) @ eigenvectors
    isotropic_norm = float(np.linalg.norm(isotropic))
    if isotropic_norm > 1e-12:
        coefficients = (
            (1.0 - RAYLEIGH_ISOTROPIC_MIX) * coefficients
            + RAYLEIGH_ISOTROPIC_MIX * isotropic / isotropic_norm
        )

    beta_support = eigenvectors @ coefficients
    norm = float(np.linalg.norm(beta_support))
    if not math.isfinite(norm) or norm < 1e-12:
        return base_values, metadata
    realized = float(
        coefficients @ (eigenvalues * coefficients) / (norm * norm)
    )

    metadata.update(
        {
            "linear_rayleigh_targeted": True,
            "linear_rayleigh_contract": RAYLEIGH_CONTRACT,
            "linear_rayleigh_target": float(target),
            "linear_rayleigh_realized_presignal": realized,
            "linear_subgram_eig_min": low,
            "linear_subgram_eig_max": high,
            "linear_subgram_eig_mean": float(np.mean(eigenvalues)),
            "linear_rayleigh_eig_index_low": lower,
            "linear_rayleigh_eig_index_high": upper,
            "linear_rayleigh_isotropic_mix": float(RAYLEIGH_ISOTROPIC_MIX),
        }
    )
    return beta_support, metadata


def sample_coefficient_values(
    k: int, rng: np.random.Generator
) -> tuple[np.ndarray, int, str, dict[str, Any]]:
    family_id = _choice_index(rng, COEFFICIENT_FAMILY_PROBABILITIES)
    family = COEFFICIENT_FAMILY_NAMES[family_id]
    metadata: dict[str, Any] = {}
    if family == "gaussian":
        values = rng.standard_normal(k)
    elif family == "laplace":
        values = rng.laplace(0.0, 1.0 / math.sqrt(2.0), size=k)
    elif family == "rademacher_loguniform":
        sign = rng.choice(np.asarray([-1.0, 1.0]), size=k)
        magnitude = np.exp(
            rng.uniform(math.log(0.25), math.log(4.0), size=k)
        )
        values = sign * magnitude
    elif family == "student_t":
        degrees_of_freedom = float(rng.uniform(3.0, 10.0))
        values = rng.standard_t(degrees_of_freedom, size=k)
        metadata["coefficient_student_t_df"] = degrees_of_freedom
    else:
        strong_ratio = float(
            math.exp(rng.uniform(math.log(3.0), math.log(30.0)))
        )
        strong = rng.random(k) < 0.20
        values = rng.standard_normal(k) * np.where(strong, strong_ratio, 1.0)
        metadata.update(
            {
                "coefficient_strong_fraction": float(np.mean(strong)),
                "coefficient_strong_weak_scale_ratio": strong_ratio,
            }
        )
    return (
        np.asarray(values, dtype=np.float64),
        family_id,
        family,
        metadata,
    )


def sample_standardized_error(
    x: np.ndarray,
    signal: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, str, dict[str, Any]]:
    n, p = x.shape
    family_id = _choice_index(rng, ERROR_FAMILY_PROBABILITIES)
    family = ERROR_FAMILY_NAMES[family_id]
    metadata: dict[str, Any] = {}
    if family == "gaussian":
        error = rng.standard_normal(n)
    elif family == "student_t":
        degrees_of_freedom = float(rng.uniform(2.5, 12.0))
        error = rng.standard_t(degrees_of_freedom, size=n)
        metadata["error_student_t_df"] = degrees_of_freedom
    elif family == "laplace":
        error = rng.laplace(0.0, 1.0 / math.sqrt(2.0), size=n)
    elif family == "logistic":
        error = rng.logistic(0.0, math.sqrt(3.0) / math.pi, size=n)
    elif family == "centered_lognormal":
        log_sigma = float(rng.uniform(0.35, 1.10))
        error = np.exp(log_sigma * rng.standard_normal(n))
        metadata["error_lognormal_sigma"] = log_sigma
    elif family == "contaminated_gaussian":
        contamination = float(rng.uniform(0.01, 0.10))
        outlier_scale = float(
            math.exp(rng.uniform(math.log(5.0), math.log(30.0)))
        )
        is_outlier = rng.random(n) < contamination
        error = rng.standard_normal(n) * np.where(
            is_outlier, outlier_scale, 1.0
        )
        metadata.update(
            {
                "error_contamination_probability": contamination,
                "error_outlier_scale": outlier_scale,
                "error_realized_outlier_fraction": float(np.mean(is_outlier)),
            }
        )
    else:
        driver_feature = int(rng.integers(0, p))
        strength = float(rng.uniform(0.25, 1.25))
        driver = np.asarray(x[:, driver_feature], dtype=np.float64)
        driver = np.abs(
            (driver - float(np.mean(driver)))
            / max(float(np.std(driver)), 1e-8)
        )
        scale = np.exp(strength * np.clip(driver, 0.0, 3.0) / 3.0)
        error = rng.standard_normal(n) * scale
        metadata.update(
            {
                "heteroskedastic_driver_feature": driver_feature,
                "heteroskedastic_strength": strength,
                "heteroskedastic_scale_min": float(np.min(scale)),
                "heteroskedastic_scale_max": float(np.max(scale)),
                "heteroskedastic_conditional_mean_zero": True,
            }
        )
    error = np.asarray(error, dtype=np.float64)
    error -= float(np.mean(error))
    error_std = float(np.std(error))
    if not math.isfinite(error_std) or error_std <= 1e-10:
        raise ValueError(f"degenerate {family} error realization")
    error /= error_std
    if not np.isfinite(error).all():
        raise ValueError(f"nonfinite {family} error realization")
    metadata["standardized_error_signal_correlation"] = float(
        np.corrcoef(error, signal)[0, 1]
    )
    return error, family_id, family, metadata


def generate_controlled_linear_response(
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    support_reference_id: int,
    clip_value: float,
    min_unique_count: int,
    min_unique_ratio: float,
    max_dominant_ratio: float,
    min_standardized_iqr: float,
    round_decimals: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    x64 = np.asarray(x, dtype=np.float64)
    n, p = x64.shape
    for beta_attempt in range(20):
        (
            k,
            support_regime_id,
            support_regime,
            support_metadata,
        ) = sample_support_size(
            n,
            p,
            rng,
            support_reference_id=support_reference_id,
        )
        support, selection_metadata = select_support_indices(x64, k, rng)
        (
            coefficient_values,
            coefficient_family_id,
            coefficient_family,
            coefficient_metadata,
        ) = sample_coefficient_values(k, rng)
        if rng.random() < RAYLEIGH_TARGET_PROBABILITY:
            coefficient_values, rayleigh_metadata = (
                sample_rayleigh_targeted_coefficients(
                    x64[:, support], coefficient_values, rng
                )
            )
        else:
            rayleigh_metadata = {"linear_rayleigh_targeted": False}
        beta = np.zeros(p, dtype=np.float64)
        beta[support] = coefficient_values
        signal = x64 @ beta
        signal -= float(np.mean(signal))
        signal_std = float(np.std(signal))
        if math.isfinite(signal_std) and signal_std > 1e-10:
            beta /= signal_std
            signal /= signal_std
            break
    else:
        raise ValueError("failed to create a nondegenerate linear signal")

    target_snr, snr_regime_id, snr_regime = sample_target_snr(rng)
    (
        base_error,
        error_family_id,
        error_family,
        error_metadata,
    ) = sample_standardized_error(x64, signal, rng)
    noise = base_error / math.sqrt(target_snr)
    raw_response = signal + noise
    signal_variance = float(np.var(signal))
    noise_variance = float(np.var(noise))
    realized_snr = signal_variance / max(noise_variance, 1e-30)

    response, quality = standardize_scm_response(
        raw_response,
        clip_value=clip_value,
        min_unique_count=min_unique_count,
        min_unique_ratio=min_unique_ratio,
        max_dominant_ratio=max_dominant_ratio,
        min_standardized_iqr=min_standardized_iqr,
        round_decimals=round_decimals,
    )
    quality.update(
        {
            "target_contract": LINEAR_RESPONSE_CONTRACT,
            "target_source": "controlled linear model on final stored TabICL features",
            "official_target_retained": False,
            "outcome_source": "linear",
            "outcome_source_id": OUTCOME_SOURCE_LINEAR,
            "linear_beta_attempts": int(beta_attempt + 1),
            "linear_support_size": int(k),
            "linear_support_fraction": float(k / p),
            "linear_support_fraction_of_p": float(k / p),
            "linear_support_fraction_of_n": float(k / n),
            "linear_support_regime_id": int(support_regime_id),
            "linear_support_regime": support_regime,
            "linear_support_sha256": hashlib.sha256(support.tobytes()).hexdigest(),
            **support_metadata,
            "linear_coefficient_family_id": int(coefficient_family_id),
            "linear_coefficient_family": coefficient_family,
            "linear_beta_l2_norm": float(np.linalg.norm(beta)),
            "linear_beta_max_abs": float(np.max(np.abs(beta))),
            "linear_rayleigh_quotient": float(
                np.var(x64 @ beta) / max(float(beta @ beta), 1e-30)
            ),
            "linear_beta_nnz_over_n": float(k / n),
            **selection_metadata,
            **rayleigh_metadata,
            "linear_signal_variance": signal_variance,
            "linear_noise_variance": noise_variance,
            "linear_target_snr": float(target_snr),
            "linear_realized_snr": float(realized_snr),
            "linear_log_realized_target_snr_error": float(
                abs(math.log(realized_snr / target_snr))
            ),
            "linear_snr_contract": LINEAR_SNR_CONTRACT,
            "linear_snr_distribution": LINEAR_SNR_DISTRIBUTION,
            "linear_snr_min": LINEAR_SNR_MIN,
            "linear_snr_max": LINEAR_SNR_MAX,
            "linear_snr_regime_id": int(snr_regime_id),
            "linear_snr_regime": snr_regime,
            "linear_error_family_id": int(error_family_id),
            "linear_error_family": error_family,
            **coefficient_metadata,
            **error_metadata,
        }
    )
    return response, quality


def add_featurewise_exp_variance_gaussian_noise(
    signal: np.ndarray,
    rng: np.random.Generator,
    *,
    clip_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Add eps_ij ~ N(0, v_j), with independent v_j ~ Exp(rate=1)."""
    signal = standardize(signal.astype(np.float32, copy=False))
    variance = rng.exponential(
        scale=1.0 / NOISE_VARIANCE_RATE, size=signal.shape[1]
    ).astype(np.float32)
    sigma = np.sqrt(variance).astype(np.float32)
    error = rng.standard_normal(signal.shape).astype(np.float32) * sigma[None, :]
    noisy = standardize(signal + error)
    if clip_value > 0:
        noisy = np.clip(noisy, -clip_value, clip_value)
        noisy = standardize(noisy)
    stored = noisy.astype(np.float16)
    family_ids = np.full(
        signal.shape[1], GAUSSIAN_NOISE_FAMILY_ID, dtype=np.uint8
    )
    attenuation = 1.0 / np.sqrt(1.0 + variance.astype(np.float64))
    realized_variance = np.var(error, axis=0, dtype=np.float64)
    return stored, variance, sigma, family_ids, {
        "noise_variance_distribution": "Exponential(rate=1)",
        "error_distribution": "Normal(0, feature_noise_variance)",
        "variance_independently_sampled_per_feature": True,
        "noise_variance_summary": quantile_summary(variance),
        "noise_sigma_summary": quantile_summary(sigma),
        "realized_error_variance_summary": quantile_summary(realized_variance),
        "theoretical_signal_retention_summary": quantile_summary(attenuation),
        "noise_family": "gaussian",
    }


def random_bnn_block(
    filtered_core: np.ndarray,
    rng: np.random.Generator,
    *,
    output_dim: int,
    hidden_dim: int,
    parent_min: int,
    parent_max: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    max_parents = min(parent_max, filtered_core.shape[1])
    min_parents = min(parent_min, max_parents)
    if min_parents == max_parents:
        parent_count = max_parents
    else:
        parent_count = int(
            round(
                math.exp(
                    rng.uniform(math.log(float(min_parents)), math.log(float(max_parents)))
                )
            )
        )
        parent_count = min(max(parent_count, min_parents), max_parents)
    parent_idx = rng.choice(filtered_core.shape[1], size=parent_count, replace=False)
    inp = standardize(filtered_core[:, parent_idx].astype(np.float32, copy=False))

    activation_name = str(rng.choice(["gelu", "tanh", "relu", "sigmoid", "sin_pi"]))
    depth = int(rng.choice([1, 2], p=[0.75, 0.25]))
    prior_scale_1 = float(np.exp(rng.uniform(math.log(0.60), math.log(1.60))))
    w1 = rng.normal(
        0.0, prior_scale_1 / math.sqrt(parent_count), size=(parent_count, hidden_dim)
    ).astype(np.float32)
    b1 = rng.normal(0.0, 0.25, size=hidden_dim).astype(np.float32)
    hidden = standardize(activation(inp @ w1 + b1[None, :], activation_name))
    if depth == 2:
        prior_scale_2 = float(np.exp(rng.uniform(math.log(0.60), math.log(1.40))))
        w2 = rng.normal(
            0.0, prior_scale_2 / math.sqrt(hidden_dim), size=(hidden_dim, hidden_dim)
        ).astype(np.float32)
        b2 = rng.normal(0.0, 0.20, size=hidden_dim).astype(np.float32)
        hidden = standardize(activation(hidden @ w2 + b2[None, :], activation_name))
    else:
        prior_scale_2 = 0.0

    output_prior_scale = float(np.exp(rng.uniform(math.log(0.70), math.log(1.40))))
    w_out = rng.normal(
        0.0,
        output_prior_scale / math.sqrt(hidden_dim),
        size=(hidden_dim, output_dim),
    ).astype(np.float32)
    b_out = rng.normal(0.0, 0.10, size=output_dim).astype(np.float32)
    signal = standardize(hidden @ w_out + b_out[None, :])
    return signal, {
        "output_features": int(output_dim),
        "parent_count": int(parent_count),
        "parent_index_sha256": hashlib.sha256(
            np.asarray(parent_idx, dtype=np.int64).tobytes()
        ).hexdigest(),
        "hidden_dim": int(hidden_dim),
        "depth": int(depth),
        "activation": activation_name,
        "prior_scale_1": prior_scale_1,
        "prior_scale_2": prior_scale_2,
        "output_prior_scale": output_prior_scale,
    }


def sampled_correlation_audit(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    *,
    max_features: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    q = min(max_features, x.shape[1])
    idx = np.sort(rng.choice(x.shape[1], size=q, replace=False).astype(np.int64))
    a = x[:, idx].astype(np.float32)
    a -= a.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(a.astype(np.float64), axis=0)
    if not np.all(np.isfinite(norm) & (norm > 1e-10)):
        raise ValueError("sampled correlation audit found a constant feature")
    z = a / norm.astype(np.float32)[None, :]
    corr = np.clip(z.T @ z, -1.0, 1.0)
    off = np.abs(corr[np.triu_indices(q, 1)]) if q > 1 else np.asarray([0.0])
    yc = y.astype(np.float32) - float(np.mean(y, dtype=np.float64))
    ynorm = max(float(np.linalg.norm(yc.astype(np.float64))), 1e-12)
    xy = np.abs((z.T @ yc) / ynorm)
    return idx, {
        "correlation_audit_contract": CORRELATION_AUDIT_CONTRACT,
        "sampled_features": int(q),
        "sampled_pair_abs_corr_summary": quantile_summary(off),
        "sampled_feature_y_abs_corr_summary": quantile_summary(xy),
        "sampled_feature_index_sha256": hashlib.sha256(idx.tobytes()).hexdigest(),
        "audit_is_full_pair_certificate": False,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tabicl-src", type=Path, default=Path("/users/minsuk/fm-lm-lambda/external/tabicl/src"))
    p.add_argument("--num-batches", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--start-index", type=int, default=-1)
    p.add_argument("--end-index-exclusive", type=int, default=-1)
    p.add_argument("--stop-file", type=Path, default=None)
    p.add_argument("--target-bytes", type=int, default=0)
    p.add_argument("--seed", type=int, default=88076230)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--n-min", type=int, default=32)
    p.add_argument("--n-max", type=int, default=40_000)
    p.add_argument("--main-n-min", type=int, default=200)
    p.add_argument("--small-n-max", type=int, default=199)
    p.add_argument("--small-n-probability", type=float, default=0.10)
    p.add_argument("--p-min", type=int, default=50)
    p.add_argument("--p-max", type=int, default=200_000)
    p.add_argument("--max-cells", type=int, default=24_000_000)
    p.add_argument("--p-median", type=float, default=10_000.0)
    p.add_argument("--p-log-sigma", type=float, default=0.75)
    p.add_argument("--raw-n-median", type=float, default=5_700.0)
    p.add_argument("--n-log-sigma", type=float, default=0.70)
    p.add_argument("--scm-core-cap", type=int, default=2_048)
    p.add_argument("--core-oversample-factor", type=float, default=1.5)
    p.add_argument("--corr-threshold", type=float, default=0.95)
    p.add_argument("--corr-device", default="cpu")
    p.add_argument("--bnn-hidden-dim", type=int, default=64)
    p.add_argument("--bnn-output-block", type=int, default=128)
    p.add_argument("--bnn-parent-min", type=int, default=8)
    p.add_argument("--bnn-parent-max", type=int, default=128)
    p.add_argument("--corr-audit-features", type=int, default=256)
    p.add_argument("--clip-value", type=float, default=10.0)
    p.add_argument("--min-y-unique-count", type=int, default=10)
    p.add_argument("--min-y-unique-ratio", type=float, default=0.02)
    p.add_argument("--max-y-dominant-ratio", type=float, default=0.30)
    p.add_argument("--min-y-standardized-iqr", type=float, default=0.20)
    p.add_argument("--y-round-decimals", type=int, default=6)
    p.add_argument("--max-retries", type=int, default=30)
    p.add_argument("--filter-unpredictable", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force-n", type=int, default=None)
    p.add_argument("--force-p", type=int, default=None)
    return p.parse_args()


def load_entries(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")).get("files", []))


def write_manifest(args: argparse.Namespace, entries: list[dict[str, Any]]) -> None:
    tabicl_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.tabicl_src, text=True
    ).strip()
    if tabicl_commit != EXPECTED_TABICL_COMMIT:
        raise RuntimeError(f"expected TabICLv2 commit {EXPECTED_TABICL_COMMIT}, got {tabicl_commit}")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=args.tabicl_src, text=True
    ).strip():
        raise RuntimeError("TabICL source worktree is dirty")
    payload = {
        "generator": GENERATOR_NAME,
        "prior": PRIOR_NAME,
        "dataset_contract": DATASET_CONTRACT,
        "num_datasets": int(sum(int(e["num_tasks"]) for e in entries)),
        "settings": {
            "shape_sampler": SHAPE_SAMPLER_NAME,
            "batch_size": int(args.batch_size),
            "n_min": int(args.n_min),
            "n_max": int(args.n_max),
            "main_n_min": int(args.main_n_min),
            "small_n_max": int(args.small_n_max),
            "small_n_probability": float(args.small_n_probability),
            "p_min": int(args.p_min),
            "p_max": int(args.p_max),
            "max_cells": int(args.max_cells),
            "p_median": float(args.p_median),
            "p_log_sigma": float(args.p_log_sigma),
            "raw_n_median": float(args.raw_n_median),
            "n_log_sigma": float(args.n_log_sigma),
            "scm_core_cap": int(args.scm_core_cap),
            "core_oversample_factor": float(args.core_oversample_factor),
            "core_candidate_cap": int(math.ceil(args.scm_core_cap * args.core_oversample_factor)),
            "high_p_rule": "P>2048 uses exactly 2048 noise-injected and V7-style qfiltered official graph_scm core features; remaining features are noisy random-BNN draws in 128-column blocks",
            "core_correlation_threshold_exclusive": float(args.corr_threshold),
            "core_correlation_group_rule": "threshold-connected components keep first representative; constants and cross-conflicts are dropped",
            "core_correlation_certification": CORE_CORRELATION_CERTIFICATE,
            "core_corr_device": str(args.corr_device),
            "bnn_contract": BNN_CONTRACT,
            "bnn_hidden_dim": int(args.bnn_hidden_dim),
            "bnn_output_block": int(args.bnn_output_block),
            "bnn_parent_min": int(args.bnn_parent_min),
            "bnn_parent_max": int(args.bnn_parent_max),
            "feature_noise_contract": FEATURE_NOISE_CONTRACT,
            "feature_noise_applies_to": "all SCM core and BNN extension features",
            "feature_noise_variance_distribution": "Exponential(rate=1) independently per feature",
            "feature_error_distribution": "Normal(0, feature_noise_variance) independently per sample",
            "feature_noise_family_map": {str(GAUSSIAN_NOISE_FAMILY_ID): "gaussian"},
            "noise_variance_rate": NOISE_VARIANCE_RATE,
            "correlation_audit_contract": CORRELATION_AUDIT_CONTRACT,
            "corr_audit_features": int(args.corr_audit_features),
            "full_pair_exact_certificate_rule": "true for P<=2048; high-P certifies only the SCM2048 core and samples the final P features",
            "high_p_full_pair_exact_certificate": False,
            "high_p_full_pair_exact_certificate_reason": "an all-extension exact pass would restore quadratic scaling in P",
            "target_contract": MIXED_RESPONSE_CONTRACT,
            "outcome_mixture_contract": (
                "batch_seed_parity_exact_alternation_50pct_tabicl_"
                "50pct_controlled_linear_v2"
            ),
            "outcome_source_map": {
                str(key): value for key, value in OUTCOME_SOURCE_NAMES.items()
            },
            "tabicl_target_contract": SCM_RESPONSE_CONTRACT,
            "linear_target_contract": LINEAR_RESPONSE_CONTRACT,
            "linear_support_reference_assignment": (
                "bit1(batch_seed + task_index), exact alternation among "
                "linear tasks: 50pct P-referenced conditional on k<N/3 and "
                "50pct N-referenced"
            ),
            "linear_support_reference_probabilities": (
                SUPPORT_REFERENCE_PROBABILITIES
            ),
            "linear_support_strict_constraint": "k < N/3",
            "linear_support_integer_cap": "min(P, floor((N-1)/3))",
            "p_based_support_policy": (
                "original V12 P-scale mixture conditioned by rejection on "
                "k<N/3; no clipping or cap pile-up"
            ),
            "n_based_support_policy": (
                "same regime probabilities on N scale; very_dense is "
                "integer-uniform from ceil(0.2N) through floor((N-1)/3)"
            ),
            "linear_support_regimes": {
                name: float(probability)
                for name, probability in zip(
                    SUPPORT_REGIME_NAMES, SUPPORT_REGIME_PROBABILITIES
                )
            },
            "linear_coefficient_families": {
                name: float(probability)
                for name, probability in zip(
                    COEFFICIENT_FAMILY_NAMES,
                    COEFFICIENT_FAMILY_PROBABILITIES,
                )
            },
            "linear_snr_contract": LINEAR_SNR_CONTRACT,
            "linear_snr_distribution": LINEAR_SNR_DISTRIBUTION,
            "linear_snr_min": LINEAR_SNR_MIN,
            "linear_snr_max": LINEAR_SNR_MAX,
            "linear_snr_telemetry_bins": {
                name: [
                    SNR_TELEMETRY_BIN_EDGES[index],
                    SNR_TELEMETRY_BIN_EDGES[index + 1],
                ]
                for index, name in enumerate(SNR_TELEMETRY_BIN_NAMES)
            },
            "linear_error_families": {
                name: float(probability)
                for name, probability in zip(
                    ERROR_FAMILY_NAMES, ERROR_FAMILY_PROBABILITIES
                )
            },
            "target_quality_filter": {
                "min_unique_count": int(args.min_y_unique_count),
                "min_unique_ratio": float(args.min_y_unique_ratio),
                "max_dominant_ratio": float(args.max_y_dominant_ratio),
                "min_standardized_iqr": float(args.min_y_standardized_iqr),
                "round_decimals": int(args.y_round_decimals),
            },
            "official_tabicl_commit": tabicl_commit,
            "tabicl_function_types": TABICL_FUNCTION_TYPES,
            "data_policy": DATA_POLICY,
            "batch_rng_contract": "seed=(worker_seed+1000003*batch_index) mod (2^32-1)",
        },
        "files": sorted(entries, key=lambda e: e["path"]),
    }
    atomic_write_json(args.out_dir / "manifest.json", payload)


def main() -> None:
    args = parse_args()
    if (args.force_n is None) != (args.force_p is None):
        raise ValueError("force-n and force-p must be provided together")
    if args.batch_size != 1:
        raise ValueError("batch-size must be 1 for variable shapes")
    if args.scm_core_cap != 2_048:
        raise ValueError("V11 contract requires scm-core-cap=2048")
    if args.bnn_output_block != 128:
        raise ValueError("V11 contract requires bnn-output-block=128")
    if args.core_oversample_factor < 1.0:
        raise ValueError("core-oversample-factor must be at least 1")
    if not math.isclose(
        args.corr_threshold, REQUIRED_CORRELATION_THRESHOLD, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"corr-threshold must equal {REQUIRED_CORRELATION_THRESHOLD}; got {args.corr_threshold}"
        )
    if args.corr_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA core-correlation filtering requested but CUDA is unavailable")
    if args.bnn_parent_min <= 0 or args.bnn_parent_max < args.bnn_parent_min:
        raise ValueError("invalid BNN parent range")
    if args.corr_audit_features <= 1:
        raise ValueError("corr-audit-features must exceed 1")
    if args.n_min * args.p_max > args.max_cells:
        raise ValueError("max-cells must permit n-min at p-max")

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.manual_seed(args.seed % (2**32 - 1))
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    entries = load_entries(args.out_dir / "manifest.json")
    if args.start_index < 0:
        completed = [int(Path(e["path"]).stem.split("_")[-1]) for e in entries]
        start_index = max(completed, default=-1) + 1
    else:
        start_index = int(args.start_index)

    core_candidate_cap = int(math.ceil(args.scm_core_cap * args.core_oversample_factor))
    graph = ConfiguredTabICLv2GraphGenerator(
        args.tabicl_src, core_candidate_cap, args.filter_unpredictable, args.seed
    )
    corr_device = torch.device(args.corr_device)
    print(
        f"V14 supportmix SCM2048-qfilter-BNN128 generator start part={args.out_dir.name} "
        f"seed={args.seed} start={start_index} threads={args.threads} "
        f"corr_device={corr_device}",
        flush=True,
    )

    for offset in range(args.num_batches):
        batch_index = start_index + offset
        if args.end_index_exclusive >= 0 and batch_index >= args.end_index_exclusive:
            print(f"end index reached: {batch_index} >= {args.end_index_exclusive}", flush=True)
            break
        if args.stop_file is not None and args.stop_file.exists():
            print(f"stop file observed: {args.stop_file}", flush=True)
            break
        current_bytes = sum(path.stat().st_size for path in args.out_dir.glob("batch_*.npz"))
        if args.target_bytes > 0 and current_bytes >= args.target_bytes:
            print(f"worker byte target reached: {current_bytes} >= {args.target_bytes}", flush=True)
            break

        batch_seed = int((int(args.seed) + 1_000_003 * int(batch_index)) % (2**32 - 1))
        random.seed(batch_seed)
        np.random.seed(batch_seed)
        torch.manual_seed(batch_seed)
        rng = np.random.default_rng(batch_seed)
        n, p = (
            (int(args.force_n), int(args.force_p))
            if args.force_n is not None
            else shared_sample_shape_pair(
                rng,
                n_min=args.n_min,
                n_max=args.n_max,
                main_n_min=args.main_n_min,
                small_n_max=args.small_n_max,
                small_n_probability=args.small_n_probability,
                p_min=args.p_min,
                p_max=args.p_max,
                max_cells=args.max_cells,
                p_median=args.p_median,
                p_log_sigma=args.p_log_sigma,
                raw_n_median=args.raw_n_median,
                n_log_sigma=args.n_log_sigma,
            )
        )
        if not (args.n_min <= n <= args.n_max and args.p_min <= p <= args.p_max):
            raise ValueError(f"shape outside contract: N={n}, P={p}")
        if n * p > args.max_cells:
            raise ValueError(f"shape exceeds max cells: N={n}, P={p}")

        batch_start = time.perf_counter()
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        variances: list[np.ndarray] = []
        sigmas: list[np.ndarray] = []
        family_ids_all: list[np.ndarray] = []
        feature_sources: list[np.ndarray] = []
        block_ids_all: list[np.ndarray] = []
        audit_indices_all: list[np.ndarray] = []
        outcome_source_ids: list[int] = []
        linear_support_sizes: list[int] = []
        linear_support_fractions: list[float] = []
        linear_support_fractions_of_n: list[float] = []
        linear_support_regime_ids: list[int] = []
        linear_support_reference_ids: list[int] = []
        linear_coefficient_family_ids: list[int] = []
        linear_snr_regime_ids: list[int] = []
        linear_error_family_ids: list[int] = []
        linear_target_snrs: list[float] = []
        linear_realized_snrs: list[float] = []
        linear_rayleigh_quotients: list[float] = []
        task_meta: list[dict[str, Any]] = []

        for task_index in range(args.batch_size):
            last_error: str | None = None
            rejected_attempt_errors: list[str] = []
            for attempt in range(args.max_retries):
                try:
                    timings: dict[str, float] = {}
                    core_p = min(p, args.scm_core_cap)
                    requested_core = max(
                        core_p, int(math.ceil(core_p * args.core_oversample_factor))
                    )
                    t = time.perf_counter()
                    core_candidates, official_y, active_base = graph.generate_base(
                        n, requested_core
                    )
                    timings["scm_seconds"] = time.perf_counter() - t
                    tabicl_y, tabicl_y_meta = standardize_scm_response(
                        official_y,
                        clip_value=args.clip_value,
                        min_unique_count=args.min_y_unique_count,
                        min_unique_ratio=args.min_y_unique_ratio,
                        max_dominant_ratio=args.max_y_dominant_ratio,
                        min_standardized_iqr=args.min_y_standardized_iqr,
                        round_decimals=args.y_round_decimals,
                    )
                    t = time.perf_counter()
                    core_acc = ExactLowCorrelationAccumulator(
                        n, args.corr_threshold, corr_device, args.clip_value
                    )
                    accepted_core = core_acc.add(core_candidates, limit=core_p)
                    if accepted_core != core_p or core_acc.p != core_p:
                        raise RuntimeError(
                            "SCM qfilter retained too few core features: "
                            f"accepted={accepted_core}, target={core_p}, "
                            f"candidates={requested_core}"
                        )
                    if not core_acc.max_abs_corr < args.corr_threshold:
                        raise RuntimeError(
                            "SCM core exact correlation certificate failed: "
                            f"max={core_acc.max_abs_corr}"
                        )
                    clean_filtered_core = core_acc.stored()
                    timings["clean_core_filter_seconds"] = time.perf_counter() - t

                    t = time.perf_counter()
                    noisy_core, core_variance, core_sigma, core_family, core_noise_meta = (
                        add_featurewise_exp_variance_gaussian_noise(
                            clean_filtered_core,
                            rng,
                            clip_value=args.clip_value,
                        )
                    )
                    timings["core_noise_seconds"] = time.perf_counter() - t
                    t = time.perf_counter()
                    final_core_acc = ExactLowCorrelationAccumulator(
                        n, args.corr_threshold, corr_device, args.clip_value
                    )
                    final_core_accepted = final_core_acc.add(noisy_core, limit=core_p)
                    if final_core_accepted != core_p or final_core_acc.p != core_p:
                        raise RuntimeError(
                            "noise-injected SCM core retained too few features: "
                            f"accepted={final_core_accepted}, target={core_p}"
                        )
                    if not final_core_acc.max_abs_corr < args.corr_threshold:
                        raise RuntimeError(
                            "noise-injected SCM core exact certificate failed: "
                            f"max={final_core_acc.max_abs_corr}"
                        )
                    filtered_core = final_core_acc.stored()
                    timings["noisy_core_verify_seconds"] = time.perf_counter() - t

                    x_chunks = [filtered_core]
                    variance_chunks = [core_variance]
                    sigma_chunks = [core_sigma]
                    family_chunks = [core_family]
                    source_chunks = [np.zeros(core_p, dtype=np.uint8)]
                    block_id_chunks = [np.full(core_p, -1, dtype=np.int32)]
                    bnn_meta: list[dict[str, Any]] = []
                    bnn_seconds = 0.0
                    bnn_noise_seconds = 0.0
                    remaining = p - core_p
                    block_id = 0
                    while remaining > 0:
                        out_dim = min(args.bnn_output_block, remaining)
                        t = time.perf_counter()
                        signal, block_meta = random_bnn_block(
                            filtered_core,
                            rng,
                            output_dim=out_dim,
                            hidden_dim=args.bnn_hidden_dim,
                            parent_min=args.bnn_parent_min,
                            parent_max=args.bnn_parent_max,
                        )
                        bnn_seconds += time.perf_counter() - t
                        t = time.perf_counter()
                        (
                            noisy_block,
                            block_variance,
                            block_sigma,
                            block_family,
                            block_noise_meta,
                        ) = add_featurewise_exp_variance_gaussian_noise(
                            signal, rng, clip_value=args.clip_value
                        )
                        bnn_noise_seconds += time.perf_counter() - t
                        block_meta["block_id"] = int(block_id)
                        block_meta["noise"] = block_noise_meta
                        bnn_meta.append(block_meta)
                        x_chunks.append(noisy_block)
                        variance_chunks.append(block_variance)
                        sigma_chunks.append(block_sigma)
                        family_chunks.append(block_family)
                        source_chunks.append(np.ones(out_dim, dtype=np.uint8))
                        block_id_chunks.append(np.full(out_dim, block_id, dtype=np.int32))
                        remaining -= out_dim
                        block_id += 1
                    timings["bnn_seconds"] = bnn_seconds
                    timings["bnn_noise_seconds"] = bnn_noise_seconds

                    x = np.concatenate(x_chunks, axis=1)
                    variance = np.concatenate(variance_chunks)
                    sigma = np.concatenate(sigma_chunks)
                    family = np.concatenate(family_chunks)
                    source = np.concatenate(source_chunks)
                    block_ids = np.concatenate(block_id_chunks)
                    if x.shape != (n, p):
                        raise RuntimeError(f"final X shape mismatch: {x.shape} != {(n, p)}")
                    perm = rng.permutation(p)
                    x = x[:, perm]
                    variance = variance[perm]
                    sigma = sigma[perm]
                    family = family[perm]
                    source = source[perm]
                    block_ids = block_ids[perm]
                    outcome_source_id = int((batch_seed + task_index) & 1)
                    support_reference_id = int(
                        ((batch_seed + task_index) >> 1) & 1
                    )
                    outcome_rng = np.random.default_rng(
                        (
                            batch_seed
                            ^ 0xA5A5_5A5A
                            ^ (task_index * 0x9E37_79B1)
                        )
                        % (2**32 - 1)
                    )
                    if outcome_source_id == OUTCOME_SOURCE_LINEAR:
                        y, y_meta = generate_controlled_linear_response(
                            x,
                            outcome_rng,
                            support_reference_id=support_reference_id,
                            clip_value=args.clip_value,
                            min_unique_count=args.min_y_unique_count,
                            min_unique_ratio=args.min_y_unique_ratio,
                            max_dominant_ratio=args.max_y_dominant_ratio,
                            min_standardized_iqr=args.min_y_standardized_iqr,
                            round_decimals=args.y_round_decimals,
                        )
                    else:
                        y = tabicl_y
                        y_meta = dict(tabicl_y_meta)
                        y_meta.update(
                            {
                                "outcome_source": "tabicl",
                                "outcome_source_id": OUTCOME_SOURCE_TABICL,
                                "linear_support_size": 0,
                                "linear_support_fraction": 0.0,
                                "linear_support_fraction_of_p": 0.0,
                                "linear_support_fraction_of_n": 0.0,
                                "linear_support_regime_id": 255,
                                "linear_support_reference_id": 255,
                                "linear_support_reference": "not_applicable",
                                "linear_support_sampling_attempts": 0,
                                "linear_support_rejected_n_third_draws": 0,
                                "linear_support_max_allowed": int(
                                    min(p, (n - 1) // 3)
                                ),
                                "linear_support_strict_n_third_verified": True,
                                "linear_coefficient_family_id": 255,
                                "linear_snr_contract": "not_applicable",
                                "linear_snr_distribution": "not_applicable",
                                "linear_snr_min": -1.0,
                                "linear_snr_max": -1.0,
                                "linear_snr_regime_id": 255,
                                "linear_error_family_id": 255,
                                "linear_target_snr": -1.0,
                                "linear_realized_snr": -1.0,
                            }
                        )
                    t = time.perf_counter()
                    audit_idx, audit_meta = sampled_correlation_audit(
                        x, y, rng, max_features=args.corr_audit_features
                    )
                    timings["correlation_audit_seconds"] = time.perf_counter() - t
                    if (
                        audit_meta["sampled_pair_abs_corr_summary"]["max"]
                        >= args.corr_threshold
                    ):
                        raise RuntimeError(
                            "final sampled correlation audit failed: "
                            f"max={audit_meta['sampled_pair_abs_corr_summary']['max']}"
                        )
                    timings["task_seconds"] = time.perf_counter() - batch_start
                    if not np.isfinite(x).all() or not np.isfinite(y).all():
                        raise RuntimeError("final task contains nonfinite values")

                    xs.append(x.astype(np.float16, copy=False))
                    ys.append(y.astype(np.float32, copy=False))
                    variances.append(variance.astype(np.float32, copy=False))
                    sigmas.append(sigma.astype(np.float32, copy=False))
                    family_ids_all.append(family.astype(np.uint8, copy=False))
                    feature_sources.append(source.astype(np.uint8, copy=False))
                    block_ids_all.append(block_ids.astype(np.int32, copy=False))
                    audit_indices_all.append(audit_idx)
                    outcome_source_ids.append(outcome_source_id)
                    linear_support_sizes.append(
                        int(y_meta["linear_support_size"])
                    )
                    linear_support_fractions.append(
                        float(y_meta["linear_support_fraction"])
                    )
                    linear_support_fractions_of_n.append(
                        float(y_meta["linear_support_fraction_of_n"])
                    )
                    linear_support_regime_ids.append(
                        int(y_meta["linear_support_regime_id"])
                    )
                    linear_support_reference_ids.append(
                        int(y_meta["linear_support_reference_id"])
                    )
                    linear_coefficient_family_ids.append(
                        int(y_meta["linear_coefficient_family_id"])
                    )
                    linear_snr_regime_ids.append(
                        int(y_meta["linear_snr_regime_id"])
                    )
                    linear_error_family_ids.append(
                        int(y_meta["linear_error_family_id"])
                    )
                    linear_target_snrs.append(
                        float(y_meta["linear_target_snr"])
                    )
                    linear_realized_snrs.append(
                        float(y_meta["linear_realized_snr"])
                    )
                    linear_rayleigh_quotients.append(
                        float(
                            y_meta.get(
                                "linear_rayleigh_quotient",
                                float("nan"),
                            )
                        )
                    )
                    task_meta.append({
                        "task_index": int(task_index),
                        "attempt": int(attempt + 1),
                        "rejected_attempt_errors": rejected_attempt_errors,
                        "mode": (
                            "direct_qfiltered_scm"
                            if p <= args.scm_core_cap
                            else "noisy_qfiltered_scm2048_plus_noisy_bnn128"
                        ),
                        "n": int(n),
                        "p": int(p),
                        "scm_core_features": int(core_p),
                        "bnn_extension_features": int(p - core_p),
                        "official_core_candidates": int(requested_core),
                        "official_active_base_features": int(active_base),
                        "official_core_accepted": int(accepted_core),
                        "core_correlation_threshold_exclusive": float(args.corr_threshold),
                        "core_correlation_certification": CORE_CORRELATION_CERTIFICATE,
                        "clean_core_exact_max_abs_corr": float(core_acc.max_abs_corr),
                        "core_exact_max_abs_corr": float(final_core_acc.max_abs_corr),
                        "core_candidates": int(core_acc.total_candidates),
                        "core_dropped_constant": int(core_acc.dropped_constant),
                        "core_dropped_cross": int(core_acc.dropped_cross),
                        "core_dropped_within": int(core_acc.dropped_within),
                        "core_added_noise": True,
                        "core_noise": core_noise_meta,
                        "bnn_contract": BNN_CONTRACT,
                        "bnn_blocks": bnn_meta,
                        "feature_noise_contract": FEATURE_NOISE_CONTRACT,
                        "mixed_target_contract": MIXED_RESPONSE_CONTRACT,
                        "outcome_source": OUTCOME_SOURCE_NAMES[
                            outcome_source_id
                        ],
                        "outcome_source_id": outcome_source_id,
                        "outcome_source_assignment": (
                            "parity(batch_seed + task_index)"
                        ),
                        "linear_support_reference_assignment": (
                            "bit1(batch_seed + task_index)"
                            if outcome_source_id == OUTCOME_SOURCE_LINEAR
                            else "not_applicable"
                        ),
                        "feature_noise_variance_summary": quantile_summary(variance),
                        "feature_sigma_summary": quantile_summary(sigma),
                        "feature_source_counts": {
                            "scm_core": int(np.count_nonzero(source == 0)),
                            "bnn_extension": int(np.count_nonzero(source == 1)),
                        },
                        "full_pair_exact_certificate": bool(p <= args.scm_core_cap),
                        "final_sampled_corr_audit_passed": True,
                        "display_permutation_sha256": hashlib.sha256(
                            perm.astype(np.int64).tobytes()
                        ).hexdigest(),
                        "timings": timings,
                        **audit_meta,
                        **y_meta,
                    })
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = repr(exc)
                    rejected_attempt_errors.append(last_error)
                    print(
                        f"retry batch={batch_index} task={task_index} "
                        f"attempt={attempt + 1}/{args.max_retries}: {last_error}",
                        flush=True,
                    )
            else:
                raise RuntimeError(
                    f"task {task_index} failed after {args.max_retries} attempts: {last_error}"
                )

        out_path = args.out_dir / f"batch_{batch_index:06d}.npz"
        response_mean = np.asarray([m["target_final_mean"] for m in task_meta], dtype=np.float32)
        response_std = np.asarray([m["target_final_std"] for m in task_meta], dtype=np.float32)
        audit_max = np.asarray(
            [m["sampled_pair_abs_corr_summary"]["max"] for m in task_meta], dtype=np.float32
        )
        core_exact_max = np.asarray(
            [m["core_exact_max_abs_corr"] for m in task_meta], dtype=np.float32
        )
        atomic_save_npz(
            out_path,
            X=np.stack(xs),
            y=np.stack(ys),
            feature_noise_variance=np.stack(variances),
            feature_noise_sigma=np.stack(sigmas),
            feature_noise_family=np.stack(family_ids_all),
            feature_source=np.stack(feature_sources),
            bnn_block_id=np.stack(block_ids_all),
            corr_audit_feature_indices=np.stack(audit_indices_all),
            sampled_corr_audit_max=audit_max,
            core_exact_max_abs_corr=core_exact_max,
            scm_response_mean=response_mean,
            scm_response_std=response_std,
            batch_seed=np.full(args.batch_size, batch_seed, dtype=np.uint64),
            outcome_source_id=np.asarray(
                outcome_source_ids, dtype=np.uint8
            ),
            linear_support_size=np.asarray(
                linear_support_sizes, dtype=np.int64
            ),
            linear_support_fraction=np.asarray(
                linear_support_fractions, dtype=np.float32
            ),
            linear_support_fraction_of_n=np.asarray(
                linear_support_fractions_of_n, dtype=np.float32
            ),
            linear_support_regime_id=np.asarray(
                linear_support_regime_ids, dtype=np.uint8
            ),
            linear_support_reference_id=np.asarray(
                linear_support_reference_ids, dtype=np.uint8
            ),
            linear_coefficient_family_id=np.asarray(
                linear_coefficient_family_ids, dtype=np.uint8
            ),
            linear_snr_regime_id=np.asarray(
                linear_snr_regime_ids, dtype=np.uint8
            ),
            linear_error_family_id=np.asarray(
                linear_error_family_ids, dtype=np.uint8
            ),
            linear_target_snr=np.asarray(
                linear_target_snrs, dtype=np.float32
            ),
            linear_realized_snr=np.asarray(
                linear_realized_snrs, dtype=np.float32
            ),
            linear_rayleigh_quotient=np.asarray(
                linear_rayleigh_quotients, dtype=np.float32
            ),
        )
        entry = {
            "path": out_path.name,
            "batch_seed": int(batch_seed),
            "bytes": int(out_path.stat().st_size),
            "num_tasks": int(args.batch_size),
            "n": int(n),
            "p": int(p),
            "small_n_component": bool(n < args.main_n_min),
            "shape_sampler": SHAPE_SAMPLER_NAME,
            "prior": PRIOR_NAME,
            "dataset_contract": DATASET_CONTRACT,
            "data_policy": DATA_POLICY,
            "target_contract": MIXED_RESPONSE_CONTRACT,
            "outcome_source": OUTCOME_SOURCE_NAMES[outcome_source_ids[0]],
            "outcome_source_id": int(outcome_source_ids[0]),
            "linear_support_size": int(linear_support_sizes[0]),
            "linear_support_fraction": float(linear_support_fractions[0]),
            "linear_support_fraction_of_n": float(
                linear_support_fractions_of_n[0]
            ),
            "linear_support_reference_id": int(
                linear_support_reference_ids[0]
            ),
            "linear_support_reference": task_meta[0][
                "linear_support_reference"
            ],
            "linear_support_strict_n_third_verified": bool(
                task_meta[0]["linear_support_strict_n_third_verified"]
            ),
            "linear_snr_contract": task_meta[0][
                "linear_snr_contract"
            ],
            "linear_snr_distribution": task_meta[0][
                "linear_snr_distribution"
            ],
            "linear_snr_min": float(task_meta[0]["linear_snr_min"]),
            "linear_snr_max": float(task_meta[0]["linear_snr_max"]),
            "linear_target_snr": float(linear_target_snrs[0]),
            "linear_realized_snr": float(linear_realized_snrs[0]),
            "correlation_audit_contract": CORRELATION_AUDIT_CONTRACT,
            "full_pair_exact_certificate": bool(p <= args.scm_core_cap),
            "core_exact_correlation_certificate": True,
            "task_core_exact_max_abs_corr": float(core_exact_max.max()),
            "task_sampled_corr_audit_max": float(audit_max.max()),
            "task_metadata": task_meta,
        }
        entries = [e for e in entries if e.get("path") != out_path.name]
        entries.append(entry)
        write_manifest(args, entries)
        elapsed = time.perf_counter() - batch_start
        print(
            f"saved {out_path.name} N={n} P={p} bytes={out_path.stat().st_size} "
            f"outcome={OUTCOME_SOURCE_NAMES[outcome_source_ids[0]]} "
            f"samplemaxcorr={audit_max.max():.6f} dt={elapsed:.2f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()

# %%
