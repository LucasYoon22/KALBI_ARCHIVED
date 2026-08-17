#!/usr/bin/env python3
"""Freeze ten dimension-selected V13 rolling tasks as N<500, P=999 slices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "kalbi_v23_current_v13_training_smallnp10_prepared_v1"
DIMENSION_REGIME = "current_v13_training_tasks_np999_10"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def exact_max_abs_correlation(x: np.ndarray) -> float:
    centered = np.asarray(x, dtype=np.float64)
    centered = centered - centered.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    valid = norms > 1e-12
    if int(valid.sum()) < 2:
        return 0.0
    standardized = centered[:, valid] / norms[valid][None, :]
    correlation = np.abs(standardized.T @ standardized)
    np.fill_diagonal(correlation, 0.0)
    return float(correlation.max(initial=0.0))


def selected_catalog_rows(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in catalog["files"]:
        metadata = item["metadata"]
        generated = metadata["generator_manifest_entry"]
        n = int(generated["n"])
        p = int(generated["p"])
        if not (100 <= n < 500):
            continue
        rows.append(
            {
                "source_shard_id": int(item["shard_id"]),
                "worker_batch_id": int(metadata["batch_id"]),
                "source_n": n,
                "source_p": p,
                "source_seed": int(item["seed"]),
                "source_relative_path": str(item["relative_path"]),
                "source_sha256": str(item["sha256"]),
                "dataset_id": str(metadata["dataset_id"]),
                "outcome_source": str(generated["outcome_source"]),
                "linear_target_snr": float(
                    generated.get("linear_target_snr", -1.0)
                ),
                "linear_realized_snr": float(
                    generated.get("linear_realized_snr", -1.0)
                ),
                "linear_support_size": int(
                    generated.get("linear_support_size", 0)
                ),
                "source_core_exact_max_abs_corr": float(
                    generated["task_core_exact_max_abs_corr"]
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            row["source_p"],
            row["source_n"],
            row["source_shard_id"],
        )
    )
    if len(rows) < 10:
        raise RuntimeError(f"only {len(rows)} dimension-eligible tasks")
    return rows[:10]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--catalog-file", required=True)
    parser.add_argument("--catalog-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--derived-p", type=int, default=999)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    catalog_path = (
        data_root / "rolling_control" / "catalogs" / args.catalog_file
    )
    if file_sha256(catalog_path) != args.catalog_sha256:
        raise RuntimeError("catalog SHA mismatch")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise RuntimeError(f"refusing existing output root: {output_root}")
    (output_root / "data").mkdir(parents=True)
    (output_root / "manifests").mkdir()
    (output_root / "reproducibility").mkdir()

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    rows = selected_catalog_rows(catalog)
    accepted: list[dict[str, Any]] = []
    for dataset_index, row in enumerate(rows, start=1):
        source_path = data_root / row["source_relative_path"]
        if file_sha256(source_path) != row["source_sha256"]:
            raise RuntimeError(f"source SHA mismatch: {source_path}")
        with np.load(source_path, allow_pickle=False) as source:
            source_x = np.asarray(source["X"])
            source_y = np.asarray(source["y"])
            source_variance = np.asarray(source["feature_noise_variance"])
        if source_x.shape != (
            1,
            row["source_n"],
            row["source_p"],
        ):
            raise RuntimeError(f"source X shape mismatch: {source_path}")
        derived_p = min(int(args.derived_p), int(row["source_p"]))
        if derived_p >= 1000:
            raise RuntimeError("derived task must satisfy P<1000")
        x = np.ascontiguousarray(source_x[:, :, :derived_p])
        y = np.ascontiguousarray(source_y)
        variance = np.ascontiguousarray(source_variance[:, :derived_p])
        exact_corr = exact_max_abs_correlation(x[0])
        data_path = output_root / "data" / f"task_{dataset_index:03d}.npz"
        np.savez_compressed(
            data_path,
            X=x,
            y=y,
            feature_noise_variance=variance,
            core_exact_max_abs_corr=np.asarray(
                [exact_corr], dtype=np.float32
            ),
            sampled_corr_audit_max=np.asarray(
                [exact_corr], dtype=np.float32
            ),
            batch_seed=np.asarray([row["source_seed"]], dtype=np.uint64),
        )

        selection_rule = (
            "active catalog tasks with 100<=N<500, sorted by "
            "(source P, source N, source shard_id); first 10; then prefix "
            "indices [0, 999) in generator display-permuted feature order"
        )
        task_manifest = {
            "version": "kalbi_v23_current_v13_training_task_slice_v2",
            "dataset_index": dataset_index,
            "dataset_id": (
                f"v13_beta_epoch{int(catalog['epoch'])}_"
                f"shard{row['source_shard_id']}_prefix{derived_p}"
            ),
            **row,
            "derived_n": int(row["source_n"]),
            "derived_p": derived_p,
            "retained_feature_fraction": derived_p / row["source_p"],
            "derived_exact_max_abs_corr": exact_corr,
            "feature_slice": (
                f"prefix indices [0, {derived_p}) in generator "
                "display-permuted feature order"
            ),
            "selection_rule": selection_rule,
            "performance_used_for_selection": False,
            "source_catalog_epoch": int(catalog["epoch"]),
            "source_catalog_file": str(catalog_path),
            "source_catalog_sha256": args.catalog_sha256,
        }
        manifest_path = (
            output_root / "manifests" / f"task_{dataset_index:03d}.json"
        )
        write_json(manifest_path, task_manifest)
        accepted.append(
            {
                "dataset_index": dataset_index,
                "dataset_id": task_manifest["dataset_id"],
                "n": int(row["source_n"]),
                "p": derived_p,
                "relative_data_path": str(data_path.relative_to(output_root)),
                "relative_manifest_path": str(
                    manifest_path.relative_to(output_root)
                ),
                "task_sha256": file_sha256(data_path),
                "manifest_sha256": file_sha256(manifest_path),
            }
        )

    reproducibility = {
        "version": VERSION,
        "dimension_regime": DIMENSION_REGIME,
        "accepted_tasks": accepted,
        "selection_rule": (
            "dimension-only selection: active catalog tasks with "
            "100<=N<500, sorted by (source P, source N, source shard_id); "
            "first 10; no outcome, SNR, correlation, sparsity, or model "
            "metric used"
        ),
        "feature_slice": {
            "derived_p": int(args.derived_p),
            "indices": f"[0, {int(args.derived_p)})",
            "ordering": "generator display-permuted feature order",
        },
        "source_catalog": {
            "data_root": str(data_root),
            "catalog_file": args.catalog_file,
            "sha256": args.catalog_sha256,
            "epoch": int(catalog["epoch"]),
            "server": "ssh -p 4597 root@129.204.7.16",
        },
    }
    reproducibility_path = (
        output_root / "reproducibility" / "reproducibility_manifest.json"
    )
    write_json(reproducibility_path, reproducibility)
    print(
        json.dumps(
            {
                "prepared_root": str(output_root),
                "manifest": str(reproducibility_path),
                "manifest_sha256": file_sha256(reproducibility_path),
                "tasks": accepted,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
