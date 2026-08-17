#!/usr/bin/env python3
"""Read-only metadata census and frozen sample selection for KALBI V13."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def extract_row(root: Path, entry: dict[str, Any], epoch: int) -> dict[str, Any]:
    metadata = entry.get("metadata") or {}
    semantic = metadata.get("semantic_contract") or {}
    manifest = metadata.get("generator_manifest_entry") or {}
    task_metadata = manifest.get("task_metadata") or []
    task = task_metadata[0] if task_metadata else {}
    outcome = str(semantic.get("outcome_source", task.get("outcome_source", "unknown")))
    reference = str(
        semantic.get(
            "linear_support_reference",
            task.get("linear_support_reference", "not_applicable"),
        )
    )
    if outcome == "tabicl":
        stratum = "tabicl"
    elif reference == "p_based":
        stratum = "linear_p"
    elif reference == "n_based":
        stratum = "linear_n"
    else:
        stratum = "linear_unknown"

    n = safe_int(entry.get("n"))
    p = safe_int(entry.get("p"))
    k = safe_int(
        semantic.get("linear_support_size", task.get("linear_support_size", 0)),
        0,
    )
    target_snr = safe_float(
        semantic.get("linear_target_snr", task.get("linear_target_snr", -1.0))
    )
    realized_snr = safe_float(
        semantic.get("linear_realized_snr", task.get("linear_realized_snr", -1.0))
    )
    batch_seed = safe_int(
        manifest.get("batch_seed", metadata.get("batch_seed_min", entry.get("seed")))
    )
    task_index = safe_int(task.get("task_index"), 0)
    expected_outcome_id = (batch_seed + task_index) & 1 if batch_seed >= 0 else -1
    expected_reference_id = (
        ((batch_seed + task_index) >> 1) & 1 if batch_seed >= 0 else -1
    )
    observed_outcome_id = safe_int(
        semantic.get("outcome_source_id", task.get("outcome_source_id"))
    )
    observed_reference_id = safe_int(
        semantic.get(
            "linear_support_reference_id",
            task.get("linear_support_reference_id", 255),
        )
    )
    relative_path = str(entry.get("relative_path", ""))
    absolute_path = str((root / relative_path).resolve())
    rejected_errors = task.get("rejected_attempt_errors") or []
    return {
        "catalog_epoch": epoch,
        "shard_id": safe_int(entry.get("shard_id")),
        "task_hash": str(entry.get("sha256", "")),
        "seed": safe_int(entry.get("seed")),
        "batch_seed": batch_seed,
        "task_index": task_index,
        "worker_id": str(metadata.get("worker_id", "")),
        "dataset_id": str(metadata.get("dataset_id", "")),
        "relative_path": relative_path,
        "absolute_path": absolute_path,
        "bytes": safe_int(entry.get("bytes"), 0),
        "created_at_utc": str(entry.get("created_at_utc", "")),
        "origin": str(entry.get("origin", "")),
        "n": n,
        "p": p,
        "p_over_n": p / n if n > 0 else float("nan"),
        "outcome_source": outcome,
        "linear_support_reference": reference,
        "audit_stratum": stratum,
        "observed_outcome_source_id": observed_outcome_id,
        "expected_outcome_source_id": expected_outcome_id,
        "outcome_assignment_match": observed_outcome_id == expected_outcome_id,
        "observed_support_reference_id": observed_reference_id,
        "expected_support_reference_id": expected_reference_id,
        "support_assignment_match": (
            outcome != "linear" or observed_reference_id == expected_reference_id
        ),
        "linear_support_size": k,
        "linear_support_fraction_of_p": safe_float(
            task.get(
                "linear_support_fraction_of_p",
                semantic.get("linear_support_fraction", 0.0),
            )
        ),
        "linear_support_fraction_of_n": safe_float(
            task.get(
                "linear_support_fraction_of_n",
                semantic.get("linear_support_fraction_of_n", 0.0),
            )
        ),
        "linear_support_max_allowed": safe_int(
            task.get("linear_support_max_allowed"), min(p, (n - 1) // 3)
        ),
        "linear_support_regime_id": safe_int(
            task.get(
                "linear_support_regime_id",
                semantic.get("linear_support_regime_id", 255),
            )
        ),
        "linear_support_regime": str(task.get("linear_support_regime", "")),
        "linear_support_sampling_attempts": safe_int(
            task.get("linear_support_sampling_attempts"), 0
        ),
        "linear_support_rejected_n_third_draws": safe_int(
            task.get("linear_support_rejected_n_third_draws"), 0
        ),
        "linear_support_strict_n_third_verified": bool(
            task.get(
                "linear_support_strict_n_third_verified",
                semantic.get("linear_support_strict_n_third_verified", False),
            )
        ),
        "linear_coefficient_family_id": safe_int(
            task.get(
                "linear_coefficient_family_id",
                semantic.get("linear_coefficient_family_id", 255),
            )
        ),
        "linear_coefficient_family": str(task.get("linear_coefficient_family", "")),
        "linear_beta_l2_norm": safe_float(task.get("linear_beta_l2_norm")),
        "linear_beta_max_abs": safe_float(task.get("linear_beta_max_abs")),
        "linear_beta_attempts": safe_int(task.get("linear_beta_attempts"), 0),
        "linear_snr_regime_id": safe_int(
            task.get(
                "linear_snr_regime_id",
                semantic.get("linear_snr_regime_id", 255),
            )
        ),
        "linear_snr_regime": str(task.get("linear_snr_regime", "")),
        "linear_target_snr": target_snr,
        "linear_realized_snr": realized_snr,
        "linear_log_realized_target_snr_error": (
            abs(math.log(realized_snr / target_snr))
            if outcome == "linear" and target_snr > 0 and realized_snr > 0
            else float("nan")
        ),
        "linear_signal_variance": safe_float(task.get("linear_signal_variance")),
        "linear_noise_variance": safe_float(task.get("linear_noise_variance")),
        "linear_error_family_id": safe_int(
            task.get(
                "linear_error_family_id",
                semantic.get("linear_error_family_id", 255),
            )
        ),
        "linear_error_family": str(task.get("linear_error_family", "")),
        "standardized_error_signal_correlation": safe_float(
            task.get("standardized_error_signal_correlation")
        ),
        "target_raw_mean": safe_float(task.get("target_raw_mean")),
        "target_raw_std": safe_float(task.get("target_raw_std")),
        "target_final_mean": safe_float(task.get("target_final_mean")),
        "target_final_std": safe_float(task.get("target_final_std")),
        "target_final_min": safe_float(task.get("target_final_min")),
        "target_final_max": safe_float(task.get("target_final_max")),
        "target_standardized_iqr": safe_float(task.get("target_standardized_iqr")),
        "target_effective_unique_count": safe_int(
            task.get("target_effective_unique_count"), 0
        ),
        "target_effective_unique_ratio": safe_float(
            task.get("target_effective_unique_ratio")
        ),
        "target_dominant_ratio": safe_float(task.get("target_dominant_ratio")),
        "target_pre_rescale_clip_count": safe_int(
            task.get("target_pre_rescale_clip_count"), 0
        ),
        "generator_attempt": safe_int(task.get("attempt"), 0),
        "generator_rejected_attempt_count": len(rejected_errors),
        "sampled_corr_audit_max": safe_float(
            semantic.get("sampled_corr_audit_max")
        ),
        "core_exact_max_abs_corr": safe_float(
            semantic.get("core_exact_max_abs_corr")
        ),
        "feature_noise_variance_mean": safe_float(
            (task.get("feature_noise_variance_summary") or {}).get("mean")
        ),
        "feature_noise_variance_median": safe_float(
            (task.get("feature_noise_variance_summary") or {}).get("median")
        ),
        "x_dtype": str(metadata.get("x_dtype", "")),
        "y_dtype": str(metadata.get("y_dtype", "")),
        "pipeline_config_sha256": str(entry.get("pipeline_config_sha256", "")),
        "generator_manifest_sha256": str(
            metadata.get("generator_manifest_sha256", "")
        ),
    }


def summarize(rows: list[dict[str, Any]], active_epoch: int) -> dict[str, Any]:
    active = [row for row in rows if row["catalog_epoch"] == active_epoch]
    branch_counts = Counter(row["audit_stratum"] for row in active)
    branch_bytes: dict[str, int] = defaultdict(int)
    for row in active:
        branch_bytes[row["audit_stratum"]] += int(row["bytes"])
    linear = [row for row in active if row["outcome_source"] == "linear"]
    hashes = [row["task_hash"] for row in active]
    seeds = [row["seed"] for row in active]
    workers = Counter(row["worker_id"] for row in active)
    return {
        "active_epoch": active_epoch,
        "active_tasks": len(active),
        "active_bytes": sum(int(row["bytes"]) for row in active),
        "active_branch_counts": dict(sorted(branch_counts.items())),
        "active_branch_task_fractions": {
            key: value / len(active) for key, value in sorted(branch_counts.items())
        },
        "active_branch_bytes": dict(sorted(branch_bytes.items())),
        "active_branch_byte_fractions": {
            key: value / max(1, sum(branch_bytes.values()))
            for key, value in sorted(branch_bytes.items())
        },
        "active_worker_counts": dict(sorted(workers.items())),
        "active_duplicate_task_hash_count": len(hashes) - len(set(hashes)),
        "active_duplicate_seed_count": len(seeds) - len(set(seeds)),
        "outcome_assignment_mismatch_count": sum(
            not bool(row["outcome_assignment_match"]) for row in active
        ),
        "support_assignment_mismatch_count": sum(
            not bool(row["support_assignment_match"]) for row in active
        ),
        "strict_support_violation_count": sum(
            not (1 <= int(row["linear_support_size"]) and 3 * int(row["linear_support_size"]) < int(row["n"]))
            for row in linear
        ),
        "linear_snr_log_error_max": max(
            (float(row["linear_log_realized_target_snr_error"]) for row in linear),
            default=float("nan"),
        ),
        "linear_clip_task_count": sum(
            int(row["target_pre_rescale_clip_count"]) > 0 for row in linear
        ),
        "linear_tasks": len(linear),
        "catalog_rows_scanned": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--recent-catalogs", type=int, default=5)
    parser.add_argument("--per-stratum", type=int, default=200)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    current_path = args.root / "rolling_control" / "CURRENT.json"
    pointer = json.loads(current_path.read_text())
    active_epoch = int(pointer["epoch"])
    catalog_dir = args.root / "rolling_control" / "catalogs"
    catalog_paths: dict[int, Path] = {}
    for path in catalog_dir.glob("CATALOG_EPOCH_*.json"):
        try:
            epoch = int(path.name.split("_")[2])
        except (IndexError, ValueError):
            continue
        if active_epoch - args.recent_catalogs + 1 <= epoch <= active_epoch:
            catalog_paths[epoch] = path
    expected_epochs = list(
        range(max(0, active_epoch - args.recent_catalogs + 1), active_epoch + 1)
    )
    missing_epochs = [epoch for epoch in expected_epochs if epoch not in catalog_paths]
    if missing_epochs:
        raise RuntimeError(f"missing recent catalogs: {missing_epochs}")

    rows: list[dict[str, Any]] = []
    catalog_provenance = []
    for epoch in expected_epochs:
        path = catalog_paths[epoch]
        payload = json.loads(path.read_text())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        catalog_provenance.append(
            {
                "epoch": epoch,
                "path": str(path.resolve()),
                "sha256": digest,
                "tasks": len(payload.get("files", [])),
                "bytes": int((payload.get("accounting") or {}).get("total_bytes", 0)),
            }
        )
        rows.extend(extract_row(args.root, entry, epoch) for entry in payload["files"])

    active_rows = [row for row in rows if row["catalog_epoch"] == active_epoch]
    frozen: list[dict[str, Any]] = []
    for stratum in ("linear_p", "linear_n", "tabicl"):
        candidates = sorted(
            (row for row in active_rows if row["audit_stratum"] == stratum),
            key=lambda row: (row["task_hash"], row["shard_id"]),
        )
        for rank, row in enumerate(candidates[: args.per_stratum]):
            selected = dict(row)
            selected["frozen_hash_rank"] = rank
            frozen.append(selected)

    write_csv(args.out / "metadata_census.csv", rows)
    write_csv(args.out / "frozen_audit_manifest.csv", frozen)
    write_json(
        args.out / "frozen_audit_manifest.json",
        {
            "selection_contract": (
                "active catalog only; lexicographic SHA256 then shard_id; "
                "first 200 in linear_p, linear_n, tabicl; fixed before array metrics"
            ),
            "pointer": pointer,
            "catalog_provenance": catalog_provenance,
            "rows": frozen,
            "manifest_sha256": canonical_sha256(frozen),
        },
    )
    summary = summarize(rows, active_epoch)
    summary["pointer"] = pointer
    summary["catalog_provenance"] = catalog_provenance
    summary["frozen_counts"] = dict(
        Counter(row["audit_stratum"] for row in frozen)
    )
    summary["frozen_manifest_sha256"] = canonical_sha256(frozen)
    write_json(args.out / "metadata_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
