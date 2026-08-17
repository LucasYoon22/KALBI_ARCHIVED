#!/usr/bin/env python3
"""Deterministic distribution audit for the V12 controlled-linear branch."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from generate_tabiclv2_scm2048_qfilter_bnn128_mixedlinear_v12 import (
    COEFFICIENT_FAMILY_NAMES,
    COEFFICIENT_FAMILY_PROBABILITIES,
    ERROR_FAMILY_NAMES,
    ERROR_FAMILY_PROBABILITIES,
    OUTCOME_SOURCE_LINEAR,
    SNR_LOG10_BOUNDS,
    SNR_REGIME_NAMES,
    SUPPORT_REGIME_NAMES,
    SUPPORT_REGIME_PROBABILITIES,
    generate_controlled_linear_response,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026072601)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--p", type=int, default=512)
    return parser.parse_args()


def normalized_counts(
    values: list[str], names: tuple[str, ...]
) -> dict[str, float]:
    counts = Counter(values)
    return {name: counts[name] / len(values) for name in names}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    common = rng.standard_normal((args.n, 8))
    loadings = rng.standard_normal((8, args.p)) / math.sqrt(8.0)
    x = 0.35 * (common @ loadings) + rng.standard_normal((args.n, args.p))
    x -= x.mean(axis=0, keepdims=True)
    x /= np.maximum(x.std(axis=0, keepdims=True), 1e-6)
    x = x.astype(np.float32)

    records: list[dict[str, object]] = []
    for _ in range(args.trials):
        y, metadata = generate_controlled_linear_response(
            x,
            rng,
            clip_value=10.0,
            min_unique_count=10,
            min_unique_ratio=0.02,
            max_dominant_ratio=0.30,
            min_standardized_iqr=0.20,
            round_decimals=6,
        )
        if int(metadata["outcome_source_id"]) != OUTCOME_SOURCE_LINEAR:
            raise RuntimeError("controlled-linear helper emitted wrong source id")
        records.append(
            {
                "support_size": int(metadata["linear_support_size"]),
                "support_fraction": float(metadata["linear_support_fraction"]),
                "support_regime": str(metadata["linear_support_regime"]),
                "coefficient_family": str(
                    metadata["linear_coefficient_family"]
                ),
                "snr_regime": str(metadata["linear_snr_regime"]),
                "error_family": str(metadata["linear_error_family"]),
                "target_snr": float(metadata["linear_target_snr"]),
                "realized_snr": float(metadata["linear_realized_snr"]),
                "y_mean": float(np.mean(y, dtype=np.float64)),
                "y_std": float(np.std(y, dtype=np.float64)),
                "y_iqr": float(
                    np.quantile(y, 0.75) - np.quantile(y, 0.25)
                ),
            }
        )

    support_frequencies = normalized_counts(
        [str(row["support_regime"]) for row in records],
        SUPPORT_REGIME_NAMES,
    )
    coefficient_frequencies = normalized_counts(
        [str(row["coefficient_family"]) for row in records],
        COEFFICIENT_FAMILY_NAMES,
    )
    snr_frequencies = normalized_counts(
        [str(row["snr_regime"]) for row in records], SNR_REGIME_NAMES
    )
    error_frequencies = normalized_counts(
        [str(row["error_family"]) for row in records], ERROR_FAMILY_NAMES
    )
    target_snr = np.asarray(
        [float(row["target_snr"]) for row in records], dtype=np.float64
    )
    realized_snr = np.asarray(
        [float(row["realized_snr"]) for row in records], dtype=np.float64
    )
    support_fraction = np.asarray(
        [float(row["support_fraction"]) for row in records], dtype=np.float64
    )
    y_mean = np.asarray(
        [float(row["y_mean"]) for row in records], dtype=np.float64
    )
    y_std = np.asarray(
        [float(row["y_std"]) for row in records], dtype=np.float64
    )
    y_iqr = np.asarray(
        [float(row["y_iqr"]) for row in records], dtype=np.float64
    )

    parity_counts = Counter(
        "linear" if ((3_600_764_300 + 1_000_003 * index) & 1) else "tabicl"
        for index in range(1_000)
    )
    if parity_counts != {"tabicl": 500, "linear": 500}:
        raise RuntimeError(f"outcome parity contract failed: {parity_counts}")
    if np.max(np.abs(np.log(realized_snr / target_snr))) > 1e-10:
        raise RuntimeError("realized SNR does not match target SNR")
    if np.max(np.abs(y_mean)) > 2e-6:
        raise RuntimeError("generated response is not centered")
    if np.max(np.abs(y_std - 1.0)) > 2e-6:
        raise RuntimeError("generated response is not unit scale")
    if np.min(y_iqr) < 0.20:
        raise RuntimeError("generated response failed IQR filter")

    summary = {
        "trials": args.trials,
        "seed": args.seed,
        "n": args.n,
        "p": args.p,
        "outcome_assignment_1000": dict(parity_counts),
        "support_frequencies": support_frequencies,
        "support_target_probabilities": dict(
            zip(SUPPORT_REGIME_NAMES, SUPPORT_REGIME_PROBABILITIES.tolist())
        ),
        "coefficient_frequencies": coefficient_frequencies,
        "coefficient_target_probabilities": dict(
            zip(
                COEFFICIENT_FAMILY_NAMES,
                COEFFICIENT_FAMILY_PROBABILITIES.tolist(),
            )
        ),
        "snr_frequencies": snr_frequencies,
        "snr_log10_ranges": dict(zip(SNR_REGIME_NAMES, SNR_LOG10_BOUNDS)),
        "error_frequencies": error_frequencies,
        "error_target_probabilities": dict(
            zip(ERROR_FAMILY_NAMES, ERROR_FAMILY_PROBABILITIES.tolist())
        ),
        "support_fraction_quantiles": {
            str(q): float(np.quantile(support_fraction, q))
            for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "target_snr_quantiles": {
            str(q): float(np.quantile(target_snr, q))
            for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "max_log_realized_target_snr_error": float(
            np.max(np.abs(np.log(realized_snr / target_snr)))
        ),
        "max_abs_y_mean": float(np.max(np.abs(y_mean))),
        "max_abs_y_std_error": float(np.max(np.abs(y_std - 1.0))),
        "min_y_iqr": float(np.min(y_iqr)),
    }
    with (args.out_dir / "mixedlinear_v12_distribution_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if plt is not None:
        figure, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].bar(
            SUPPORT_REGIME_NAMES,
            [support_frequencies[name] for name in SUPPORT_REGIME_NAMES],
            color="#0072B2",
        )
        axes[0, 0].plot(
            SUPPORT_REGIME_NAMES,
            SUPPORT_REGIME_PROBABILITIES,
            "o",
            color="#D55E00",
            label="Target",
        )
        axes[0, 0].set_ylabel("Fraction")
        axes[0, 0].set_title("Support regime")
        axes[0, 0].tick_params(axis="x", rotation=25)
        axes[0, 0].legend(frameon=False)

        axes[0, 1].hist(np.log10(target_snr), bins=30, color="#009E73")
        axes[0, 1].set_xlabel("log10(target SNR)")
        axes[0, 1].set_ylabel("Tasks")
        axes[0, 1].set_title("SNR coverage")

        axes[1, 0].bar(
            COEFFICIENT_FAMILY_NAMES,
            [
                coefficient_frequencies[name]
                for name in COEFFICIENT_FAMILY_NAMES
            ],
            color="#56B4E9",
        )
        axes[1, 0].set_ylabel("Fraction")
        axes[1, 0].set_title("Coefficient family")
        axes[1, 0].tick_params(axis="x", rotation=25)

        axes[1, 1].bar(
            ERROR_FAMILY_NAMES,
            [error_frequencies[name] for name in ERROR_FAMILY_NAMES],
            color="#CC79A7",
        )
        axes[1, 1].set_ylabel("Fraction")
        axes[1, 1].set_title("Error family")
        axes[1, 1].tick_params(axis="x", rotation=25)
        figure.tight_layout()
        figure.savefig(
            args.out_dir / "mixedlinear_v12_distribution_audit.png",
            dpi=180,
        )
        plt.close(figure)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
