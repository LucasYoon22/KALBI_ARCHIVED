#!/usr/bin/env python3
"""Validate the V14 controlled-linear SNR contract without touching rolling data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


V13_SNR_NAMES = (
    "very_low",
    "low",
    "medium",
    "high",
    "near_noiseless",
)
V13_SNR_LOG10_BOUNDS = (
    (-2.0, -1.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (1.0, 2.0),
    (2.0, 3.0),
)
ACTUAL_SHAPES = (
    (256, 96),
    (256, 192),
    (320, 160),
    (320, 320),
    (384, 256),
    (384, 512),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v13-generator", type=Path, required=True)
    parser.add_argument("--v14-generator", type=Path, required=True)
    parser.add_argument("--producer-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path("/venv/main/bin/python"))
    parser.add_argument("--simulation-draws", type=int, default=100_000)
    parser.add_argument("--simulation-seed", type=int, default=2026072701)
    parser.add_argument("--actual-tasks", type=int, default=30)
    parser.add_argument("--actual-seed-base", type=int, default=2147000001)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_module(path: Path, name: str) -> Any:
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def v13_snr_draws(draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        regime = int(rng.integers(0, len(V13_SNR_NAMES)))
        low, high = V13_SNR_LOG10_BOUNDS[regime]
        values[index] = 10.0 ** float(rng.uniform(low, high))
    return values


def quantile_dict(values: np.ndarray) -> dict[str, float]:
    probabilities = (0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0)
    quantiles = np.quantile(values, probabilities)
    return {
        f"q{int(round(probability * 100)):02d}": float(value)
        for probability, value in zip(probabilities, quantiles)
    }


def verify_static_contracts(v13: Any, v14: Any) -> dict[str, Any]:
    equality_checks = {
        "support_regime_names": tuple(v13.SUPPORT_REGIME_NAMES)
        == tuple(v14.SUPPORT_REGIME_NAMES),
        "support_regime_probabilities": bool(
            np.array_equal(
                v13.SUPPORT_REGIME_PROBABILITIES,
                v14.SUPPORT_REGIME_PROBABILITIES,
            )
        ),
        "coefficient_family_names": tuple(v13.COEFFICIENT_FAMILY_NAMES)
        == tuple(v14.COEFFICIENT_FAMILY_NAMES),
        "coefficient_family_probabilities": bool(
            np.array_equal(
                v13.COEFFICIENT_FAMILY_PROBABILITIES,
                v14.COEFFICIENT_FAMILY_PROBABILITIES,
            )
        ),
        "error_family_names": tuple(v13.ERROR_FAMILY_NAMES)
        == tuple(v14.ERROR_FAMILY_NAMES),
        "error_family_probabilities": bool(
            np.array_equal(
                v13.ERROR_FAMILY_PROBABILITIES,
                v14.ERROR_FAMILY_PROBABILITIES,
            )
        ),
        "feature_noise_variance_rate": (
            v13.NOISE_VARIANCE_RATE == v14.NOISE_VARIANCE_RATE
        ),
        "outcome_source_ids": (
            v13.OUTCOME_SOURCE_TABICL == v14.OUTCOME_SOURCE_TABICL
            and v13.OUTCOME_SOURCE_LINEAR == v14.OUTCOME_SOURCE_LINEAR
        ),
        "support_reference_ids": (
            v13.SUPPORT_REFERENCE_P == v14.SUPPORT_REFERENCE_P
            and v13.SUPPORT_REFERENCE_N == v14.SUPPORT_REFERENCE_N
        ),
    }
    if not all(equality_checks.values()):
        raise AssertionError(f"non-SNR static contract drift: {equality_checks}")
    return equality_checks


def run_simulation(
    v14: Any,
    draws: int,
    seed: int,
    output_root: Path,
    *,
    skip_plots: bool,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if draws < 100_000:
        raise ValueError("simulation-draws must be at least 100,000")
    rng = np.random.default_rng(seed)
    v14_values = np.asarray(
        [v14.sample_target_snr(rng)[0] for _ in range(draws)],
        dtype=np.float64,
    )
    v13_values = v13_snr_draws(draws, seed)
    lower = float(v14.LINEAR_SNR_MIN)
    upper = float(v14.LINEAR_SNR_MAX)
    if np.any(v14_values < lower) or np.any(v14_values > upper):
        raise AssertionError("V14 SNR draw escaped [0.2, 50]")

    normalized_log = (
        np.log(v14_values) - math.log(lower)
    ) / (math.log(upper) - math.log(lower))
    ks = stats.kstest(normalized_log, "uniform")
    if float(ks.statistic) >= 0.01:
        raise AssertionError(f"log-uniform KS statistic too large: {ks.statistic}")

    probabilities = np.asarray([0.25, 0.50, 0.75])
    expected = lower * (upper / lower) ** probabilities
    empirical = np.quantile(v14_values, probabilities)
    relative_error = np.abs(empirical / expected - 1.0)
    if np.max(relative_error) >= 0.02:
        raise AssertionError(
            f"V14 SNR quantile relative error too large: {relative_error}"
        )

    telemetry_counts, _ = np.histogram(
        v14_values,
        bins=np.asarray(v14.SNR_TELEMETRY_BIN_EDGES),
    )
    telemetry_fractions = telemetry_counts / draws
    if np.max(np.abs(telemetry_fractions - 0.20)) >= 0.01:
        raise AssertionError(
            f"telemetry bins are not equal log-mass: {telemetry_fractions}"
        )

    parity_index = np.arange(100_000, dtype=np.uint64)
    outcome = parity_index & np.uint64(1)
    linear_index = parity_index[outcome == v14.OUTCOME_SOURCE_LINEAR]
    support_reference = (linear_index >> np.uint64(1)) & np.uint64(1)
    outcome_fraction = float(np.mean(outcome == v14.OUTCOME_SOURCE_LINEAR))
    p_reference_fraction = float(
        np.mean(support_reference == v14.SUPPORT_REFERENCE_P)
    )
    n_reference_fraction = float(
        np.mean(support_reference == v14.SUPPORT_REFERENCE_N)
    )
    if (outcome_fraction, p_reference_fraction, n_reference_fraction) != (
        0.5,
        0.5,
        0.5,
    ):
        raise AssertionError("50:50 parity contracts drifted")

    v14_oracle_r2 = v14_values / (1.0 + v14_values)
    v13_oracle_r2 = v13_values / (1.0 + v13_values)
    summary = {
        "draws": draws,
        "seed": seed,
        "distribution": v14.LINEAR_SNR_DISTRIBUTION,
        "contract": v14.LINEAR_SNR_CONTRACT,
        "bounds": [lower, upper],
        "v14_snr_quantiles": quantile_dict(v14_values),
        "v13_snr_quantiles": quantile_dict(v13_values),
        "expected_quartiles": {
            "q25": float(expected[0]),
            "q50": float(expected[1]),
            "q75": float(expected[2]),
        },
        "empirical_quartile_relative_error": {
            key: float(value)
            for key, value in zip(("q25", "q50", "q75"), relative_error)
        },
        "log_uniform_ks_statistic": float(ks.statistic),
        "log_uniform_ks_pvalue": float(ks.pvalue),
        "telemetry_bin_edges": list(v14.SNR_TELEMETRY_BIN_EDGES),
        "telemetry_bin_fractions": telemetry_fractions.tolist(),
        "outcome_linear_fraction": outcome_fraction,
        "linear_p_reference_fraction": p_reference_fraction,
        "linear_n_reference_fraction": n_reference_fraction,
        "v14_oracle_r2_quantiles": quantile_dict(v14_oracle_r2),
        "v13_oracle_r2_quantiles": quantile_dict(v13_oracle_r2),
        "v14_fraction_snr_below_0p2": float(np.mean(v14_values < 0.2)),
        "v14_fraction_snr_above_50": float(np.mean(v14_values > 50.0)),
        "v13_fraction_snr_below_0p2": float(np.mean(v13_values < 0.2)),
        "v13_fraction_snr_above_50": float(np.mean(v13_values > 50.0)),
    }
    write_json(output_root / "simulation_summary.json", summary)

    if skip_plots:
        return summary, v13_values, v14_values
    if plt is None:
        raise RuntimeError(
            "matplotlib is unavailable; rerun with --skip-plots"
        )

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    log_bins = np.linspace(-2.0, 3.0, 61)
    axes[0].hist(
        np.log10(v13_values),
        bins=log_bins,
        density=True,
        alpha=0.65,
        label="V13",
        color="#7F8C8D",
    )
    axes[0].hist(
        np.log10(v14_values),
        bins=log_bins,
        density=True,
        alpha=0.75,
        label="V14",
        color="#0072B2",
    )
    axes[0].set_xlabel("log10(SNR)")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False)

    r2_bins = np.linspace(0.0, 1.0, 51)
    axes[1].hist(
        v13_oracle_r2,
        bins=r2_bins,
        density=True,
        alpha=0.65,
        label="V13",
        color="#7F8C8D",
    )
    axes[1].hist(
        v14_oracle_r2,
        bins=r2_bins,
        density=True,
        alpha=0.75,
        label="V14",
        color="#009E73",
    )
    axes[1].set_xlabel("Oracle R2 = SNR / (1 + SNR)")
    axes[1].set_ylabel("Density")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_root / "snr_oracle_r2_comparison.png", dpi=220)
    figure.savefig(output_root / "snr_oracle_r2_comparison.pdf")
    plt.close(figure)
    return summary, v13_values, v14_values


def generator_command(
    python: Path,
    generator: Path,
    config: dict[str, Any],
    output: Path,
    seed: int,
    n: int,
    p: int,
    threads: int,
) -> list[str]:
    return [
        str(python),
        str(generator),
        "--out-dir",
        str(output),
        "--num-batches",
        "1",
        "--batch-size",
        "1",
        "--start-index",
        "0",
        "--end-index-exclusive",
        "1",
        "--seed",
        str(seed),
        "--threads",
        str(threads),
        "--corr-device",
        "cpu",
        "--force-n",
        str(n),
        "--force-p",
        str(p),
        *[str(value) for value in config["generator_common_args"]],
        *[
            str(value)
            for value in config["generator_sources"]["tabicl_v2"]["args"]
        ],
    ]


def run_actual_task(
    *,
    python: Path,
    generator: Path,
    config: dict[str, Any],
    output: Path,
    seed: int,
    n: int,
    p: int,
    threads: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        command = generator_command(
            python,
            generator,
            config,
            output,
            seed,
            n,
            p,
            threads,
        )
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONHASHSEED": "0",
                "OMP_NUM_THREADS": str(threads),
                "MKL_NUM_THREADS": str(threads),
                "OPENBLAS_NUM_THREADS": str(threads),
            }
        )
        completed = subprocess.run(
            command,
            cwd=Path(config["project_dir"]),
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=1800,
        )
        output.mkdir(parents=True, exist_ok=True)
        (output / "validation_generator.log").write_text(
            completed.stdout,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"generator failed for seed={seed}; see "
                f"{output / 'validation_generator.log'}"
            )
    manifest = json.loads(manifest_path.read_text())
    if len(manifest["files"]) != 1:
        raise RuntimeError("actual-task manifest cardinality drift")
    entry = manifest["files"][0]
    data_path = output / entry["path"]
    metadata = entry["task_metadata"][0]
    with np.load(data_path, allow_pickle=False) as archive:
        x = np.asarray(archive["X"][0], dtype=np.float64)
        y = np.asarray(archive["y"][0], dtype=np.float64)
        outcome_source_id = int(archive["outcome_source_id"][0])
    if outcome_source_id != 1 or metadata["outcome_source"] != "linear":
        raise AssertionError(f"seed={seed} did not generate a linear outcome")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise AssertionError(f"seed={seed} contains nonfinite values")

    target_snr = float(metadata["linear_target_snr"])
    realized_snr = float(metadata["linear_realized_snr"])
    log_error = abs(math.log(realized_snr / target_snr))
    if not 0.2 <= target_snr <= 50.0:
        raise AssertionError(f"seed={seed} SNR out of range: {target_snr}")
    if log_error >= 1e-8:
        raise AssertionError(
            f"seed={seed} realized/target SNR mismatch: {log_error}"
        )
    if metadata["linear_snr_distribution"] != "log_uniform":
        raise AssertionError("actual-task SNR distribution metadata drift")
    if metadata["linear_snr_contract"] != "linear_target_snr_loguniform_0p2_50_v1":
        raise AssertionError("actual-task SNR contract metadata drift")
    if not bool(metadata["linear_support_strict_n_third_verified"]):
        raise AssertionError("strict support constraint was not verified")
    if not 3 * int(metadata["linear_support_size"]) < n:
        raise AssertionError("actual support violates k < N/3")

    rounded = np.round(y, decimals=6)
    _, counts = np.unique(rounded, return_counts=True)
    q25, q75 = np.quantile(y, [0.25, 0.75])
    actual = {
        "seed": seed,
        "n": n,
        "p": p,
        "data_path": str(data_path),
        "data_sha256": sha256(data_path),
        "generator_seconds": float(time.perf_counter() - started),
        "target_snr": target_snr,
        "realized_snr": realized_snr,
        "log_realized_target_snr_error": log_error,
        "oracle_r2": target_snr / (1.0 + target_snr),
        "support_size": int(metadata["linear_support_size"]),
        "support_reference": metadata["linear_support_reference"],
        "support_regime": metadata["linear_support_regime"],
        "coefficient_family": metadata["linear_coefficient_family"],
        "error_family": metadata["linear_error_family"],
        "standardized_error_signal_correlation": float(
            metadata["standardized_error_signal_correlation"]
        ),
        "x_finite": bool(np.isfinite(x).all()),
        "y_finite": bool(np.isfinite(y).all()),
        "y_mean": float(np.mean(y)),
        "y_std": float(np.std(y)),
        "y_unique_count_rounded6": int(counts.size),
        "y_dominant_ratio_rounded6": float(counts.max() / y.size),
        "y_iqr": float(q75 - q25),
        "target_pre_rescale_clip_count": int(
            metadata["target_pre_rescale_clip_count"]
        ),
        "target_quality_filter_passed": bool(
            counts.size >= max(10, int(math.ceil(0.02 * n)))
            and counts.max() / n <= 0.30
            and q75 - q25 >= 0.20
        ),
    }
    if not actual["target_quality_filter_passed"]:
        raise AssertionError(f"stored task failed y quality audit: seed={seed}")
    return actual


def run_actual_tasks(
    *,
    v14: Any,
    generator: Path,
    producer_config: Path,
    output_root: Path,
    python: Path,
    count: int,
    seed_base: int,
    threads: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if count < 30:
        raise ValueError("actual-tasks must be at least 30")
    if seed_base % 2 != 1:
        raise ValueError("actual-seed-base must be odd to force linear outcomes")
    config = json.loads(producer_config.read_text())
    records: list[dict[str, Any]] = []
    actual_root = output_root / "actual_tasks"
    actual_root.mkdir(exist_ok=True)
    for index in range(count):
        seed = seed_base + 2 * index
        n, p = ACTUAL_SHAPES[index % len(ACTUAL_SHAPES)]
        task_root = actual_root / f"task_{index + 1:03d}_seed{seed}"
        record = run_actual_task(
            python=python,
            generator=generator,
            config=config,
            output=task_root,
            seed=seed,
            n=n,
            p=p,
            threads=threads,
        )
        record["task_index"] = index + 1
        records.append(record)
        print(
            f"validated actual task {index + 1}/{count}: "
            f"N={n} P={p} SNR={record['target_snr']:.6g}",
            flush=True,
        )

    first = records[0]
    repeat_root = output_root / "determinism_repeat"
    repeat = run_actual_task(
        python=python,
        generator=generator,
        config=config,
        output=repeat_root,
        seed=int(first["seed"]),
        n=int(first["n"]),
        p=int(first["p"]),
        threads=threads,
    )
    deterministic = repeat["data_sha256"] == first["data_sha256"]
    if not deterministic:
        raise AssertionError(
            "same-seed deterministic NPZ SHA256 validation failed"
        )

    json_records = output_root / "actual_tasks.json"
    write_json(json_records, records)
    csv_path = output_root / "actual_tasks.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    snrs = np.asarray([record["target_snr"] for record in records])
    realized = np.asarray([record["realized_snr"] for record in records])
    summary = {
        "count": len(records),
        "target_snr_quantiles": quantile_dict(snrs),
        "realized_snr_quantiles": quantile_dict(realized),
        "max_log_realized_target_snr_error": float(
            max(record["log_realized_target_snr_error"] for record in records)
        ),
        "all_finite": all(
            record["x_finite"] and record["y_finite"] for record in records
        ),
        "all_quality_filters_passed": all(
            record["target_quality_filter_passed"] for record in records
        ),
        "all_support_constraints_passed": all(
            3 * record["support_size"] < record["n"] for record in records
        ),
        "support_reference_counts": {
            name: sum(record["support_reference"] == name for record in records)
            for name in ("p_based", "n_based")
        },
        "error_family_counts": {
            name: sum(record["error_family"] == name for record in records)
            for name in v14.ERROR_FAMILY_NAMES
        },
        "coefficient_family_counts": {
            name: sum(record["coefficient_family"] == name for record in records)
            for name in v14.COEFFICIENT_FAMILY_NAMES
        },
        "same_seed_npz_sha256_match": deterministic,
        "determinism_original_sha256": first["data_sha256"],
        "determinism_repeat_sha256": repeat["data_sha256"],
    }
    write_json(output_root / "actual_tasks_summary.json", summary)
    return records, summary


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and not args.resume:
        raise RuntimeError(f"refusing existing output root: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=args.resume)
    v13 = load_module(args.v13_generator.resolve(), "kalbi_v13_generator_audit")
    v14 = load_module(args.v14_generator.resolve(), "kalbi_v14_generator_audit")
    static_checks = verify_static_contracts(v13, v14)
    simulation, _, _ = run_simulation(
        v14,
        args.simulation_draws,
        args.simulation_seed,
        args.output_root,
        skip_plots=args.skip_plots,
    )
    records, actual_summary = run_actual_tasks(
        v14=v14,
        generator=args.v14_generator.resolve(),
        producer_config=args.producer_config.resolve(),
        output_root=args.output_root,
        python=args.python.resolve(),
        count=args.actual_tasks,
        seed_base=args.actual_seed_base,
        threads=args.threads,
    )
    summary = {
        "status": "PASS",
        "v13_generator": str(args.v13_generator.resolve()),
        "v13_sha256": sha256(args.v13_generator),
        "v14_generator": str(args.v14_generator.resolve()),
        "v14_sha256": sha256(args.v14_generator),
        "producer_config": str(args.producer_config.resolve()),
        "producer_config_sha256": sha256(args.producer_config),
        "static_contract_checks": static_checks,
        "simulation": simulation,
        "actual_tasks_summary": actual_summary,
        "actual_task_count": len(records),
    }
    write_json(args.output_root / "validation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
