#!/usr/bin/env python3
"""Freeze the 41-task registry and create one deterministic outer split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


EXPECTED_REGISTRY_SHA256 = (
    "2b444127db5eb46f0d4e8489e15ad76b6bca6b471e4459f1f6958ab0f2739499"
)
MASTER_SEED = 20260728
SPLIT_CONTRACT = "KALBI_REAL_PGTN_ONESPLIT_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def task_seed(task_id: str) -> int:
    payload = (
        SPLIT_CONTRACT
        + EXPECTED_REGISTRY_SHA256
        + task_id
        + str(MASTER_SEED)
    ).encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    args = parser.parse_args()

    registry_path = args.data_root / "manifests" / "screen_accepted_highdim_datasets.json"
    actual_registry_sha = sha256(registry_path)
    if actual_registry_sha != EXPECTED_REGISTRY_SHA256:
        raise RuntimeError(
            f"registry SHA mismatch: {actual_registry_sha}"
        )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if len(registry) != 41:
        raise RuntimeError(f"expected 41 tasks, found {len(registry)}")
    if len({row["task_id"] for row in registry}) != len(registry):
        raise RuntimeError("duplicate task IDs")
    if not all(int(row["P_raw"]) > int(row["N"]) for row in registry):
        raise RuntimeError("registry contains a non-P>N task")

    split_root = args.evaluation_root / "splits"
    split_root.mkdir(parents=True, exist_ok=True)
    enriched: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for index, row in enumerate(registry, start=1):
        task_id = row["task_id"]
        task_root = args.data_root / "processed" / task_id
        dense_x_path = task_root / "X.npy"
        sparse_x_path = task_root / "X.npz"
        x_path = dense_x_path if dense_x_path.is_file() else sparse_x_path
        y_path = task_root / "y.npy"
        metadata_path = task_root / "metadata.json"
        if not x_path.is_file() or not y_path.is_file() or not metadata_path.is_file():
            raise RuntimeError(f"missing prepared artifact: {task_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        x = (
            np.load(x_path, mmap_mode="r")
            if x_path.suffix == ".npy"
            else sparse.load_npz(x_path)
        )
        y = np.load(y_path, mmap_mode="r")
        if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
            raise RuntimeError(f"shape contract failed: {task_id}")
        n, p = map(int, x.shape)
        if n != int(row["N"]) or p != int(row["P_raw"]):
            raise RuntimeError(f"registry/array dimension mismatch: {task_id}")
        if p <= n:
            raise RuntimeError(f"P>N failed from array: {task_id}")
        x_finite = (
            np.isfinite(x.data).all()
            if sparse.issparse(x)
            else np.isfinite(np.asarray(x)).all()
        )
        if not x_finite or not np.isfinite(np.asarray(y)).all():
            raise RuntimeError(f"non-finite prepared data: {task_id}")

        seed = task_seed(task_id)
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(n).astype(np.int64)
        test_count = int(math.ceil(0.20 * n))
        test_index = np.sort(permutation[:test_count])
        train_index = np.sort(permutation[test_count:])
        if train_index.size + test_index.size != n:
            raise RuntimeError(f"split coverage failed: {task_id}")
        if np.intersect1d(train_index, test_index).size:
            raise RuntimeError(f"split overlap: {task_id}")

        split_path = split_root / f"{task_id}.npz"
        if split_path.exists():
            with np.load(split_path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(
                    archive["train_indices"], train_index
                )
                np.testing.assert_array_equal(
                    archive["test_indices"], test_index
                )
                if int(archive["seed"]) != seed:
                    raise RuntimeError(f"existing seed mismatch: {task_id}")
        else:
            temporary = split_path.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    train_indices=train_index,
                    test_indices=test_index,
                    seed=np.asarray(seed, dtype=np.uint32),
                )
            temporary.replace(split_path)

        entry = {
            **row,
            "registry_index": index,
            "task_root": str(task_root),
            "X_path": str(x_path),
            "X_storage": (
                "scipy_sparse_npz" if sparse.issparse(x) else "dense_npy"
            ),
            "y_path": str(y_path),
            "metadata_path": str(metadata_path),
            "X_sha256": sha256(x_path),
            "y_sha256": sha256(y_path),
            "metadata_sha256": sha256(metadata_path),
            "prepared_source_file_sha256": metadata.get(
                "source_file_sha256", []
            ),
            "source_revision": metadata.get("source_revision"),
            "license": metadata.get("license", "unknown"),
            "split_provenance_original": metadata.get("split_provenance"),
            "outer_split_contract": SPLIT_CONTRACT,
            "outer_split_master_seed": MASTER_SEED,
            "outer_split_seed": seed,
            "outer_train_rows": int(train_index.size),
            "outer_test_rows": int(test_index.size),
            "outer_split_path": str(split_path),
            "outer_split_sha256": sha256(split_path),
            "outer_split_group_risk": (
                "row-random split requested; source groups may overlap"
            ),
            "test_y_numeric_values_accessed_for_selection": False,
        }
        enriched.append(entry)
        csv_rows.append(
            {
                "registry_index": index,
                "task_id": task_id,
                "source": row["source"],
                "target": row["target"],
                "evidence_tier": row["evidence_tier"],
                "N": n,
                "P": p,
                "P_over_N": p / n,
                "train_rows": train_index.size,
                "test_rows": test_index.size,
                "seed": seed,
                "split_sha256": entry["outer_split_sha256"],
            }
        )
        print(
            json.dumps(
                {
                    "index": index,
                    "task_id": task_id,
                    "N": n,
                    "P": p,
                    "train": int(train_index.size),
                    "test": int(test_index.size),
                    "seed": seed,
                }
            ),
            flush=True,
        )

    standard = sum(row["evidence_tier"] == "standard" for row in enriched)
    exploratory = len(enriched) - standard
    if (standard, exploratory) != (19, 22):
        raise RuntimeError(
            f"evidence-tier drift: standard={standard}, exploratory={exploratory}"
        )

    manifest = {
        "status": "PASS",
        "registry_path": str(registry_path),
        "registry_sha256": actual_registry_sha,
        "task_count": len(enriched),
        "standard_task_count": standard,
        "exploratory_task_count": exploratory,
        "all_raw_p_gt_n": True,
        "outer_split": {
            "contract": SPLIT_CONTRACT,
            "master_seed": MASTER_SEED,
            "train_fraction": 0.80,
            "test_fraction": 0.20,
            "split_count_per_task": 1,
            "seed_uses_X_or_y": False,
            "test_y_used_for_task_selection": False,
        },
        "tasks": enriched,
    }
    manifest_path = args.evaluation_root / "manifests" / "split_manifest.json"
    atomic_json(manifest_path, manifest)

    csv_path = args.evaluation_root / "manifests" / "split_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(
        json.dumps(
            {
                "status": "PASS",
                "task_count": len(enriched),
                "manifest": str(manifest_path),
                "manifest_sha256": sha256(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
