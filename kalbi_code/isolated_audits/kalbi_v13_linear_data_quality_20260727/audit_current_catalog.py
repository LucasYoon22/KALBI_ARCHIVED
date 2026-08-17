#!/usr/bin/env python3
"""Bounded, read-only full-array audit for a frozen KALBI V13 sample."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
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


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return default
    return output


def quantiles(values: np.ndarray, prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {}
    q = np.quantile(array, [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    names = ("min", "q01", "q10", "q25", "median", "q75", "q90", "q99", "max")
    output = {f"{prefix}_{name}": float(value) for name, value in zip(names, q)}
    output[f"{prefix}_mean"] = float(np.mean(array))
    output[f"{prefix}_std"] = float(np.std(array))
    return output


def skew_kurtosis(y: np.ndarray) -> tuple[float, float]:
    centered = y.astype(np.float64) - float(np.mean(y, dtype=np.float64))
    scale = float(np.std(centered))
    if scale <= 0:
        return float("nan"), float("nan")
    z = centered / scale
    return float(np.mean(z**3)), float(np.mean(z**4) - 3.0)


def top_ridge_r2(x: np.ndarray, y: np.ndarray, indices: np.ndarray) -> float:
    if indices.size == 0:
        return float("nan")
    design = x[:, indices].astype(np.float64)
    design -= np.mean(design, axis=0, keepdims=True)
    target = y.astype(np.float64) - float(np.mean(y, dtype=np.float64))
    gram = design.T @ design / max(1, design.shape[0])
    rhs = design.T @ target / max(1, design.shape[0])
    try:
        beta = np.linalg.solve(gram + 1e-6 * np.eye(indices.size), rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(
            gram + 1e-6 * np.eye(indices.size), rhs, rcond=None
        )[0]
    prediction = design @ beta
    denominator = float(np.sum(target**2))
    return 1.0 - float(np.sum((target - prediction) ** 2)) / max(denominator, 1e-30)


def audit_task(row: dict[str, Any], feature_chunk: int, corr_features: int) -> dict[str, Any]:
    path = Path(row["absolute_path"])
    n = int(row["n"])
    p = int(row["p"])
    start = time.perf_counter()
    required = {
        "X",
        "y",
        "feature_noise_variance",
        "feature_noise_sigma",
        "feature_noise_family",
        "feature_source",
        "bnn_block_id",
        "corr_audit_feature_indices",
        "sampled_corr_audit_max",
        "core_exact_max_abs_corr",
        "scm_response_mean",
        "scm_response_std",
        "batch_seed",
        "outcome_source_id",
        "linear_support_size",
        "linear_support_fraction",
        "linear_support_fraction_of_n",
        "linear_support_regime_id",
        "linear_support_reference_id",
        "linear_coefficient_family_id",
        "linear_snr_regime_id",
        "linear_error_family_id",
        "linear_target_snr",
        "linear_realized_snr",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise RuntimeError(f"{path} missing {missing}")
        x_raw = archive["X"]
        y_raw = archive["y"]
        variance = archive["feature_noise_variance"]
        sigma = archive["feature_noise_sigma"]
        family = archive["feature_noise_family"]
        source = archive["feature_source"]
        audit_indices = archive["corr_audit_feature_indices"]
        stored_corr = float(np.asarray(archive["sampled_corr_audit_max"])[0])
        stored_core_corr = float(np.asarray(archive["core_exact_max_abs_corr"])[0])
        stored_y_mean = float(np.asarray(archive["scm_response_mean"])[0])
        stored_y_std = float(np.asarray(archive["scm_response_std"])[0])
        stored_outcome = int(np.asarray(archive["outcome_source_id"])[0])
        stored_k = int(np.asarray(archive["linear_support_size"])[0])
        stored_reference = int(np.asarray(archive["linear_support_reference_id"])[0])
        stored_target_snr = float(np.asarray(archive["linear_target_snr"])[0])
        stored_realized_snr = float(np.asarray(archive["linear_realized_snr"])[0])

        shape_pass = x_raw.shape == (1, n, p) and y_raw.shape == (1, n)
        dtype_pass = x_raw.dtype == np.float16 and y_raw.dtype == np.float32
        x = x_raw[0]
        y = y_raw[0].astype(np.float32, copy=False)
        finite_pass = bool(
            np.isfinite(x).all()
            and np.isfinite(y).all()
            and np.isfinite(variance).all()
            and np.isfinite(sigma).all()
        )
        variance_sigma_pass = bool(
            np.allclose(
                sigma.astype(np.float64) ** 2,
                variance.astype(np.float64),
                rtol=3e-6,
                atol=3e-7,
            )
        )
        source_core_count = int(np.count_nonzero(source == 0))
        source_bnn_count = int(np.count_nonzero(source == 1))
        source_pass = (
            source_core_count == min(p, 2048)
            and source_bnn_count == max(0, p - 2048)
            and bool(np.all(np.isin(source, (0, 1))))
            and bool(np.all(family == 0))
        )
        observed_outcome = int(row["observed_outcome_source_id"])
        observed_reference = int(row["observed_support_reference_id"])
        branch_array_pass = (
            stored_outcome == observed_outcome
            and stored_k == int(row["linear_support_size"])
            and (
                row["outcome_source"] != "linear"
                or stored_reference == observed_reference
            )
        )
        snr_array_pass = (
            row["outcome_source"] != "linear"
            or (
                math.isclose(
                    stored_target_snr,
                    float(row["linear_target_snr"]),
                    rel_tol=2e-6,
                    abs_tol=1e-8,
                )
                and math.isclose(
                    stored_realized_snr,
                    float(row["linear_realized_snr"]),
                    rel_tol=2e-6,
                    abs_tol=1e-8,
                )
            )
        )

        y64 = y.astype(np.float64)
        y_mean = float(np.mean(y64))
        y_std = float(np.std(y64))
        y_centered = y64 - y_mean
        y_norm = float(np.linalg.norm(y_centered))
        y_q = np.quantile(y64, [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
        y_skew, y_kurtosis = skew_kurtosis(y)
        rounded, counts = np.unique(np.round(y64, 6), return_counts=True)
        y_iqr = float(y_q[5] - y_q[3])
        y_tail_fraction_3sd = float(np.mean(np.abs(y_centered / max(y_std, 1e-30)) > 3.0))

        feature_means: list[np.ndarray] = []
        feature_stds: list[np.ndarray] = []
        marginal_abs: list[np.ndarray] = []
        top_values = np.full(32, -np.inf, dtype=np.float64)
        top_indices = np.full(32, -1, dtype=np.int64)
        for left in range(0, p, feature_chunk):
            right = min(p, left + feature_chunk)
            block = x[:, left:right].astype(np.float32)
            means = np.mean(block, axis=0, dtype=np.float64)
            centered = block.astype(np.float64) - means[None, :]
            norms = np.sqrt(np.sum(centered**2, axis=0))
            stds = norms / math.sqrt(max(n, 1))
            corr = np.abs(centered.T @ y_centered) / np.maximum(norms * y_norm, 1e-30)
            feature_means.append(means)
            feature_stds.append(stds)
            marginal_abs.append(corr)
            combined_values = np.concatenate((top_values, corr))
            combined_indices = np.concatenate(
                (top_indices, np.arange(left, right, dtype=np.int64))
            )
            chosen = np.argpartition(combined_values, -32)[-32:]
            order = chosen[np.argsort(combined_values[chosen])[::-1]]
            top_values = combined_values[order]
            top_indices = combined_indices[order]
        all_feature_means = np.concatenate(feature_means)
        all_feature_stds = np.concatenate(feature_stds)
        all_marginal_abs = np.concatenate(marginal_abs)
        top_valid = top_indices[top_indices >= 0]
        ridge_r2 = top_ridge_r2(x, y, top_valid)

        indices = np.asarray(audit_indices[0], dtype=np.int64)
        indices = np.sort(indices[: min(corr_features, indices.size)])
        if n <= 2048:
            row_indices = np.arange(n)
        else:
            row_indices = np.linspace(0, n - 1, 2048, dtype=np.int64)
        sampled = x[np.ix_(row_indices, indices)].astype(np.float64)
        sampled -= np.mean(sampled, axis=0, keepdims=True)
        sampled /= np.maximum(
            np.linalg.norm(sampled, axis=0, keepdims=True), 1e-30
        )
        corr_matrix = np.clip(sampled.T @ sampled, -1.0, 1.0)
        offdiag = (
            np.abs(corr_matrix[np.triu_indices(indices.size, 1)])
            if indices.size > 1
            else np.asarray([0.0])
        )

    y_filter_pass = (
        rounded.size >= max(10, math.ceil(0.02 * n))
        and float(counts.max() / n) <= 0.30
        and y_iqr >= 0.20 - 1e-6
    )
    y_certificate_pass = (
        abs(y_mean) <= 2e-6
        and abs(y_std - 1.0) <= 2e-6
        and abs(stored_y_mean - y_mean) <= 2e-6
        and abs(stored_y_std - y_std) <= 2e-6
    )
    output = {
        "task_hash": row["task_hash"],
        "shard_id": int(row["shard_id"]),
        "audit_stratum": row["audit_stratum"],
        "outcome_source": row["outcome_source"],
        "worker_id": row["worker_id"],
        "n": n,
        "p": p,
        "p_over_n": p / n,
        "bytes": int(row["bytes"]),
        "shape_pass": shape_pass,
        "dtype_pass": dtype_pass,
        "finite_pass": finite_pass,
        "variance_sigma_pass": variance_sigma_pass,
        "feature_source_pass": source_pass,
        "branch_array_pass": branch_array_pass,
        "snr_array_pass": snr_array_pass,
        "y_filter_pass": y_filter_pass,
        "y_certificate_pass": y_certificate_pass,
        "strict_support_pass": (
            row["outcome_source"] != "linear"
            or (1 <= int(row["linear_support_size"]) and 3 * int(row["linear_support_size"]) < n)
        ),
        "linear_support_size": int(row["linear_support_size"]),
        "linear_support_reference": row["linear_support_reference"],
        "linear_support_regime": row["linear_support_regime"],
        "linear_target_snr": finite_float(row["linear_target_snr"]),
        "linear_realized_snr": finite_float(row["linear_realized_snr"]),
        "linear_snr_regime": row["linear_snr_regime"],
        "linear_error_family": row["linear_error_family"],
        "linear_coefficient_family": row["linear_coefficient_family"],
        "linear_beta_l2_norm": finite_float(row["linear_beta_l2_norm"]),
        "linear_beta_max_abs": finite_float(row["linear_beta_max_abs"]),
        "standardized_error_signal_correlation": finite_float(
            row["standardized_error_signal_correlation"]
        ),
        "target_pre_rescale_clip_count": int(row["target_pre_rescale_clip_count"]),
        "y_mean": y_mean,
        "y_std": y_std,
        "y_min": float(y_q[0]),
        "y_q01": float(y_q[1]),
        "y_q10": float(y_q[2]),
        "y_q25": float(y_q[3]),
        "y_median": float(y_q[4]),
        "y_q75": float(y_q[5]),
        "y_q90": float(y_q[6]),
        "y_q99": float(y_q[7]),
        "y_max": float(y_q[8]),
        "y_iqr": y_iqr,
        "y_skew": y_skew,
        "y_excess_kurtosis": y_kurtosis,
        "y_tail_fraction_3sd": y_tail_fraction_3sd,
        "y_unique_count_round6": int(rounded.size),
        "y_dominant_ratio_round6": float(counts.max() / n),
        "zero_predictor_mse": float(np.mean(y64**2)),
        "top32_marginal_ridge_r2_full_rows": ridge_r2,
        "marginal_abs_corr_max": float(np.max(all_marginal_abs)),
        "marginal_abs_corr_q90": float(np.quantile(all_marginal_abs, 0.90)),
        "marginal_abs_corr_q99": float(np.quantile(all_marginal_abs, 0.99)),
        "marginal_top32_concentration": float(
            np.sum(np.sort(all_marginal_abs)[-32:])
            / max(np.sum(all_marginal_abs), 1e-30)
        ),
        "feature_mean_abs_max": float(np.max(np.abs(all_feature_means))),
        "feature_std_min": float(np.min(all_feature_stds)),
        "feature_std_median": float(np.median(all_feature_stds)),
        "feature_std_max": float(np.max(all_feature_stds)),
        "feature_near_constant_count": int(np.count_nonzero(all_feature_stds <= 1e-6)),
        "feature_core_count": source_core_count,
        "feature_bnn_count": source_bnn_count,
        "stored_sampled_corr_audit_max_full_rows_256features": stored_corr,
        "stored_core_exact_max_abs_corr": stored_core_corr,
        "recomputed_sample_corr_max_upto2048rows": float(np.max(offdiag)),
        "recomputed_sample_corr_mean_upto2048rows": float(np.mean(offdiag)),
        "recomputed_sample_corr_q99_upto2048rows": float(np.quantile(offdiag, 0.99)),
        "audit_seconds": time.perf_counter() - start,
        "true_beta_stored": False,
        "true_noise_stored": False,
        "exact_linear_identity_status": "not_testable_from_production_shard",
    }
    return output


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "n",
        "p",
        "p_over_n",
        "linear_support_size",
        "linear_target_snr",
        "linear_beta_l2_norm",
        "standardized_error_signal_correlation",
        "target_pre_rescale_clip_count",
        "y_iqr",
        "y_skew",
        "y_excess_kurtosis",
        "y_tail_fraction_3sd",
        "top32_marginal_ridge_r2_full_rows",
        "marginal_abs_corr_max",
        "marginal_abs_corr_q99",
        "feature_std_min",
        "feature_std_max",
        "stored_sampled_corr_audit_max_full_rows_256features",
        "stored_core_exact_max_abs_corr",
        "recomputed_sample_corr_max_upto2048rows",
        "audit_seconds",
    ]
    output: list[dict[str, Any]] = []
    for stratum in ("linear_p", "linear_n", "tabicl"):
        subset = [row for row in rows if row["audit_stratum"] == stratum]
        summary: dict[str, Any] = {"audit_stratum": stratum, "task_count": len(subset)}
        for metric in metrics:
            values = np.asarray(
                [finite_float(row.get(metric)) for row in subset], dtype=np.float64
            )
            summary.update(quantiles(values, metric))
        for gate in (
            "shape_pass",
            "dtype_pass",
            "finite_pass",
            "variance_sigma_pass",
            "feature_source_pass",
            "branch_array_pass",
            "snr_array_pass",
            "y_filter_pass",
            "y_certificate_pass",
            "strict_support_pass",
        ):
            summary[f"{gate}_failures"] = sum(not bool(row[gate]) for row in subset)
        output.append(summary)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--feature-chunk", type=int, default=2048)
    parser.add_argument("--corr-features", type=int, default=128)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.manifest.read_text())
    frozen = list(payload["rows"])
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, row in enumerate(frozen, 1):
        rows.append(audit_task(row, args.feature_chunk, args.corr_features))
        if index % 25 == 0 or index == len(frozen):
            elapsed = time.perf_counter() - started
            print(
                f"audited {index}/{len(frozen)} tasks elapsed={elapsed:.1f}s "
                f"rate={index/max(elapsed,1e-9):.2f}/s",
                flush=True,
            )
            write_csv(args.out / "task_metrics.partial.csv", rows)
    write_csv(args.out / "task_metrics.csv", rows)
    write_json(args.out / "task_metrics.json", rows)
    distribution = aggregate(rows)
    write_csv(args.out / "distribution_summary.csv", distribution)
    gates = {
        "sample_count": len(rows),
        "stratum_counts": dict(Counter(row["audit_stratum"] for row in rows)),
        "failures": {
            gate: [
                row["task_hash"]
                for row in rows
                if not bool(row[gate])
            ]
            for gate in (
                "shape_pass",
                "dtype_pass",
                "finite_pass",
                "variance_sigma_pass",
                "feature_source_pass",
                "branch_array_pass",
                "snr_array_pass",
                "y_filter_pass",
                "y_certificate_pass",
                "strict_support_pass",
            )
        },
        "exact_linear_identity": {
            "status": "not_testable_from_production_shards",
            "reason": (
                "production NPZ stores support size/hash and beta norms but not "
                "support indices, beta, signal, or noise"
            ),
        },
        "runtime_seconds": time.perf_counter() - started,
        "total_bytes_read_nominal": sum(int(row["bytes"]) for row in rows),
    }
    write_json(args.out / "quality_gate_results.json", gates)
    print(json.dumps(gates, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
