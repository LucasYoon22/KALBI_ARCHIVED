#!/usr/bin/env python3
"""Isolated counterfactual tests for the live KALBI V13 response functions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def standardized_x(n: int, p: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    x -= np.mean(x, axis=0, keepdims=True)
    x /= np.maximum(np.std(x, axis=0, keepdims=True), 1e-12)
    return x.astype(np.float32)


def captured_linear_response(
    module: Any,
    x: np.ndarray,
    rng: np.random.Generator,
    support_reference_id: int,
    target_snr_override: float | None = None,
) -> dict[str, Any]:
    x64 = np.asarray(x, dtype=np.float64)
    n, p = x64.shape
    for beta_attempt in range(20):
        k, support_regime_id, support_regime, support_metadata = (
            module.sample_support_size(
                n, p, rng, support_reference_id=support_reference_id
            )
        )
        support = np.sort(rng.choice(p, size=k, replace=False).astype(np.int64))
        values, coefficient_family_id, coefficient_family, coefficient_metadata = (
            module.sample_coefficient_values(k, rng)
        )
        beta = np.zeros(p, dtype=np.float64)
        beta[support] = values
        uncentered_signal = x64 @ beta
        signal_mean = float(np.mean(uncentered_signal))
        signal = uncentered_signal - signal_mean
        signal_std = float(np.std(signal))
        if math.isfinite(signal_std) and signal_std > 1e-10:
            beta /= signal_std
            signal /= signal_std
            signal_mean /= signal_std
            break
    else:
        raise RuntimeError("failed to create signal")

    if target_snr_override is None:
        snr_regime_id = int(rng.integers(0, len(module.SNR_REGIME_NAMES)))
        low, high = module.SNR_LOG10_BOUNDS[snr_regime_id]
        target_snr = 10.0 ** float(rng.uniform(low, high))
    else:
        snr_regime_id = -1
        target_snr = float(target_snr_override)
    base_error, error_family_id, error_family, error_metadata = (
        module.sample_standardized_error(x64, signal, rng)
    )
    noise = base_error / math.sqrt(target_snr)
    raw = signal + noise
    response, quality = module.standardize_scm_response(
        raw,
        clip_value=10.0,
        min_unique_count=10,
        min_unique_ratio=0.02,
        max_dominant_ratio=0.30,
        min_standardized_iqr=0.20,
        round_decimals=6,
    )
    raw_std = float(np.std(raw))
    transformed_beta = beta / raw_std
    transformed_intercept = -signal_mean / raw_std
    reconstructed_signal = x64 @ transformed_beta + transformed_intercept
    transformed_noise = noise / raw_std
    reconstructed_y = reconstructed_signal + transformed_noise
    return {
        "x": x64,
        "beta": beta,
        "support": support,
        "signal": signal,
        "noise": noise,
        "raw": raw,
        "response": response.astype(np.float64),
        "transformed_beta": transformed_beta,
        "transformed_intercept": transformed_intercept,
        "transformed_noise": transformed_noise,
        "reconstructed_y": reconstructed_y,
        "k": k,
        "support_regime_id": support_regime_id,
        "support_regime": support_regime,
        "support_metadata": support_metadata,
        "coefficient_family_id": coefficient_family_id,
        "coefficient_family": coefficient_family,
        "coefficient_metadata": coefficient_metadata,
        "error_family_id": error_family_id,
        "error_family": error_family,
        "error_metadata": error_metadata,
        "target_snr": target_snr,
        "realized_snr": float(np.var(signal) / np.var(noise)),
        "raw_std": raw_std,
        "quality": quality,
        "beta_attempts": beta_attempt + 1,
    }


def same_capture(left: dict[str, Any], right: dict[str, Any]) -> bool:
    array_keys = (
        "beta",
        "support",
        "signal",
        "noise",
        "raw",
        "response",
        "transformed_beta",
        "transformed_noise",
    )
    scalar_keys = (
        "k",
        "support_regime_id",
        "coefficient_family_id",
        "error_family_id",
        "target_snr",
        "realized_snr",
    )
    return all(np.array_equal(left[key], right[key]) for key in array_keys) and all(
        left[key] == right[key] for key in scalar_keys
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--support-draws", type=int, default=200_000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.project))
    import generate_tabiclv2_scm2048_qfilter_bnn128_mixedlinear_supportmix_v13 as g

    frozen = json.loads(args.manifest.read_text())["rows"]
    shape_pairs = np.asarray(
        [(int(row["n"]), int(row["p"])) for row in frozen], dtype=np.int64
    )
    rng = np.random.default_rng(2026072701)
    support_rows: list[dict[str, Any]] = []
    regime_counts: dict[str, Counter[str]] = {
        "p_based": Counter(),
        "n_based": Counter(),
    }
    retry_counts: dict[str, list[int]] = defaultdict(list)
    violations = 0
    for draw in range(args.support_draws):
        n, p = shape_pairs[int(rng.integers(0, len(shape_pairs)))]
        reference_id = draw & 1
        k, regime_id, regime, metadata = g.sample_support_size(
            int(n), int(p), rng, support_reference_id=reference_id
        )
        reference = g.SUPPORT_REFERENCE_NAMES[reference_id]
        regime_counts[reference][regime] += 1
        retry_counts[reference].append(
            int(metadata["linear_support_rejected_n_third_draws"])
        )
        violations += int(not (1 <= k <= p and 3 * k < n))
        if draw < 10_000:
            support_rows.append(
                {
                    "draw": draw,
                    "n": int(n),
                    "p": int(p),
                    "reference": reference,
                    "k": int(k),
                    "k_over_n": float(k / n),
                    "k_over_p": float(k / p),
                    "regime_id": int(regime_id),
                    "regime": regime,
                    "rejections": int(
                        metadata["linear_support_rejected_n_third_draws"]
                    ),
                }
            )

    x = standardized_x(256, 512, 2026072702)
    deterministic_a = captured_linear_response(
        g, x, np.random.default_rng(99173), g.SUPPORT_REFERENCE_P
    )
    deterministic_b = captured_linear_response(
        g, x, np.random.default_rng(99173), g.SUPPORT_REFERENCE_P
    )
    different_seed = captured_linear_response(
        g, x, np.random.default_rng(99174), g.SUPPORT_REFERENCE_P
    )

    snr_rows = []
    identity_errors = []
    transformed_identity_errors = []
    for reference_id in (g.SUPPORT_REFERENCE_P, g.SUPPORT_REFERENCE_N):
        for target_snr in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
            result = captured_linear_response(
                g,
                x,
                np.random.default_rng(
                    2026072800 + reference_id * 100 + int(round(math.log10(target_snr))) + 2
                ),
                reference_id,
                target_snr_override=target_snr,
            )
            identity_error = float(
                np.max(np.abs(result["raw"] - result["signal"] - result["noise"]))
            )
            transformed_error = float(
                np.max(
                    np.abs(
                        result["response"] - result["reconstructed_y"]
                    )
                )
            )
            identity_errors.append(identity_error)
            transformed_identity_errors.append(transformed_error)
            snr_rows.append(
                {
                    "reference": g.SUPPORT_REFERENCE_NAMES[reference_id],
                    "target_snr": target_snr,
                    "realized_snr": result["realized_snr"],
                    "log_ratio_error": abs(
                        math.log(result["realized_snr"] / target_snr)
                    ),
                    "raw_identity_max_error": identity_error,
                    "transformed_identity_max_error": transformed_error,
                    "clip_count": int(
                        result["quality"]["target_pre_rescale_clip_count"]
                    ),
                    "k": int(result["k"]),
                    "error_family": result["error_family"],
                }
            )

    noise_stats: dict[str, list[dict[str, float]]] = defaultdict(list)
    noise_rng = np.random.default_rng(2026072703)
    fixed_signal = x[:, 0].astype(np.float64)
    fixed_signal -= np.mean(fixed_signal)
    fixed_signal /= np.std(fixed_signal)
    attempts = 0
    while min((len(noise_stats[name]) for name in g.ERROR_FAMILY_NAMES), default=0) < 200:
        error, _family_id, family, metadata = g.sample_standardized_error(
            x.astype(np.float64), fixed_signal, noise_rng
        )
        if len(noise_stats[family]) < 200:
            z = np.asarray(error, dtype=np.float64)
            noise_stats[family].append(
                {
                    "mean": float(np.mean(z)),
                    "std": float(np.std(z)),
                    "signal_corr": float(np.corrcoef(z, fixed_signal)[0, 1]),
                    "tail3": float(np.mean(np.abs(z) > 3.0)),
                }
            )
        attempts += 1
        if attempts > 100_000:
            raise RuntimeError("noise-family collection failed")

    noise_summary = {}
    for family, values in noise_stats.items():
        noise_summary[family] = {
            "draws": len(values),
            "mean_abs_max": max(abs(value["mean"]) for value in values),
            "std_abs_error_max": max(abs(value["std"] - 1.0) for value in values),
            "signal_corr_mean": float(
                np.mean([value["signal_corr"] for value in values])
            ),
            "signal_corr_std": float(
                np.std([value["signal_corr"] for value in values])
            ),
            "tail3_mean": float(np.mean([value["tail3"] for value in values])),
        }

    support_summary = {}
    for reference in ("p_based", "n_based"):
        count = sum(regime_counts[reference].values())
        retries = np.asarray(retry_counts[reference], dtype=np.int64)
        support_summary[reference] = {
            "draws": count,
            "regime_fractions": {
                name: regime_counts[reference][name] / count
                for name in g.SUPPORT_REGIME_NAMES
            },
            "rejection_mean": float(np.mean(retries)),
            "rejection_p99": float(np.quantile(retries, 0.99)),
            "rejection_max": int(np.max(retries)),
        }

    results = {
        "generator_path": str(Path(g.__file__).resolve()),
        "support_draws": args.support_draws,
        "support_violation_count": violations,
        "support_summary": support_summary,
        "declared_support_regime_probabilities": {
            name: float(probability)
            for name, probability in zip(
                g.SUPPORT_REGIME_NAMES, g.SUPPORT_REGIME_PROBABILITIES
            )
        },
        "same_seed_exact_replay": same_capture(deterministic_a, deterministic_b),
        "adjacent_seed_collision": same_capture(deterministic_a, different_seed),
        "raw_linear_identity_max_error": max(identity_errors),
        "transformed_linear_identity_max_error": max(transformed_identity_errors),
        "snr_log_ratio_error_max": max(row["log_ratio_error"] for row in snr_rows),
        "snr_monotone_grid": all(
            math.isclose(row["target_snr"], row["realized_snr"], rel_tol=1e-12)
            for row in snr_rows
        ),
        "noise_summary": noise_summary,
        "production_shard_provenance_gap": (
            "true beta, support indices, signal, and noise are not serialized; "
            "these exact identity checks are counterfactual function tests only"
        ),
    }
    write_csv(args.out / "support_sampler_monte_carlo.csv", support_rows)
    write_csv(args.out / "snr_counterfactual.csv", snr_rows)
    write_json(args.out / "generator_unit_test_results.json", results)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
