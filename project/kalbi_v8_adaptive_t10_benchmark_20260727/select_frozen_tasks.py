#!/usr/bin/env python3
"""Freeze the predeclared six-task benchmark sample and query bank."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any


SAMPLING_SEED = 202607270256
EXPECTED_CATALOG_EPOCH = 66
EXPECTED_CATALOG_SHA256 = (
    "083d61be54c88d86fe281dfe983f5cacb36190ad791c2b0cf6b719ad515bf30d"
)
P_BINS = (
    ("low", 500, 2_000),
    ("mid", 2_000, 20_000),
    ("high", 20_000, 100_001),
)
PENALTY_CACHE_COUNT = 300_000
PENALTY_CACHE_PATH = Path(
    "/users/minsuk/fm-lm-lambda/penalty_cache_300k_v12_canonical_final.pt"
)
PENALTY_CACHE_SHA256 = (
    "6a3fda8201de87d5e2078e6e56ad0c67e5ad2a27248865910cac8f0f36bfe1dc"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def select_tasks(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for label, lower, upper in P_BINS:
        candidates: list[tuple[str, dict[str, Any]]] = []
        for entry in catalog["files"]:
            feature_count = int(entry["p"])
            if lower <= feature_count < upper:
                permutation_key = hashlib.sha256(
                    f"{SAMPLING_SEED}:{entry['sha256']}".encode()
                ).hexdigest()
                candidates.append((permutation_key, entry))
        candidates.sort(key=lambda item: item[0])
        if len(candidates) < 2:
            raise RuntimeError(
                f"insufficient candidates in {label}: {len(candidates)}"
            )
        for rank, (permutation_key, entry) in enumerate(candidates[:2], 1):
            selected.append(
                {
                    "entry": entry,
                    "p_bin": label,
                    "bin_bounds": {
                        "lower_inclusive": lower,
                        "upper_exclusive": upper,
                    },
                    "selection_rank_within_bin": rank,
                    "deterministic_permutation_key": permutation_key,
                }
            )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rolling-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise RuntimeError(
            f"refusing to overwrite existing output: {args.output_dir}"
        )

    current_path = args.rolling_root / "rolling_control" / "CURRENT.json"
    current = json.loads(current_path.read_text())
    if (
        int(current["epoch"]) != EXPECTED_CATALOG_EPOCH
        or current["catalog_sha256"] != EXPECTED_CATALOG_SHA256
    ):
        raise RuntimeError(
            "CURRENT changed after preflight; do not silently resample"
        )
    catalog_path = (
        args.rolling_root
        / "rolling_control"
        / "catalogs"
        / current["catalog_file"]
    )
    if sha256_file(catalog_path) != EXPECTED_CATALOG_SHA256:
        raise RuntimeError("catalog file SHA256 does not match CURRENT")
    catalog = json.loads(catalog_path.read_text())
    selected = select_tasks(catalog)

    frozen_dir = args.output_dir / "frozen_tasks"
    frozen_dir.mkdir(parents=True)
    manifest_tasks: list[dict[str, Any]] = []
    for dataset_index, selected_item in enumerate(selected, 1):
        entry = selected_item["entry"]
        source = args.rolling_root / entry["relative_path"]
        if not source.is_file():
            raise RuntimeError(f"missing selected task: {source}")
        source_sha = sha256_file(source)
        if source_sha != entry["sha256"]:
            raise RuntimeError(f"selected task SHA mismatch: {source}")

        target = frozen_dir / (
            f"{selected_item['p_bin']}_"
            f"{selected_item['selection_rank_within_bin']}_"
            f"{entry['sha256']}.npz"
        )
        shutil.copy2(source, target)
        if sha256_file(target) != source_sha:
            raise RuntimeError(f"frozen-copy SHA mismatch: {target}")

        metadata = entry.get("metadata", {})
        semantic = metadata.get("semantic_contract", {})
        manifest_tasks.append(
            {
                "dataset_index": dataset_index,
                "p_bin": selected_item["p_bin"],
                "bin_bounds": selected_item["bin_bounds"],
                "selection_rank_within_bin": selected_item[
                    "selection_rank_within_bin"
                ],
                "deterministic_permutation_key": selected_item[
                    "deterministic_permutation_key"
                ],
                "shard_id": int(entry["shard_id"]),
                "seed": int(entry["seed"]),
                "task_sha256": entry["sha256"],
                "source_relative_path": entry["relative_path"],
                "frozen_path": str(target),
                "n": int(entry["n"]),
                "p": int(entry["p"]),
                "p_over_n": float(entry["p"]) / float(entry["n"]),
                "outcome_source": semantic.get(
                    "outcome_source", metadata.get("outcome_source")
                ),
                "stored_x_dtype": metadata.get("x_dtype"),
                "stored_y_dtype": metadata.get("y_dtype"),
                "bytes": int(entry["bytes"]),
                "source_sha_verified": True,
                "frozen_copy_sha_verified": True,
            }
        )

    manifest = {
        "version": "kalbi_v7_blocksize_microbenchmark_sampling_v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sampling_seed": SAMPLING_SEED,
        "sampling_rule": (
            "within each predeclared P bin, sort by "
            "SHA256(f'{seed}:{task_sha256}') and take first two; "
            "no outcome or performance inspection"
        ),
        "catalog": {
            "epoch": int(current["epoch"]),
            "sha256": current["catalog_sha256"],
            "file": str(catalog_path),
            "file_sha256_verified": True,
            "num_files": len(catalog["files"]),
            "total_bytes": int(catalog["accounting"]["total_bytes"]),
        },
        "tasks": manifest_tasks,
    }
    (args.output_dir / "sampling_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    random_generator = random.Random(SAMPLING_SEED)
    penalty_indices = random_generator.sample(
        range(PENALTY_CACHE_COUNT), 8
    )
    raw_lambdas = [10.0 ** (-6.0 + index) for index in range(8)]
    queries = [
        {
            "query_index": query_index,
            "raw_lambda": raw_lambda,
            "penalty_cache_index": penalty_index,
        }
        for query_index, (raw_lambda, penalty_index) in enumerate(
            zip(raw_lambdas, penalty_indices)
        )
    ]
    query_bank = {
        "version": "kalbi_v7_blocksize_query_bank_v1",
        "seed": SAMPLING_SEED,
        "raw_lambda_contract": "raw lambda passed unchanged",
        "penalty_cache_count": PENALTY_CACHE_COUNT,
        "penalty_cache_path": str(PENALTY_CACHE_PATH),
        "penalty_cache_sha256": PENALTY_CACHE_SHA256,
        "queries": queries,
    }
    (args.output_dir / "query_bank.json").write_text(
        json.dumps(query_bank, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "catalog": manifest["catalog"],
                "tasks": manifest_tasks,
                "queries": queries,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
