#!/usr/bin/env python3
"""Convert immutable Exp123 catalog entries into rank-local training buckets."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


class CatalogTrainingViewError(RuntimeError):
    pass


def stable_shard_owner(shard_id: str, seed: int, world_size: int) -> int:
    if int(world_size) <= 0:
        raise CatalogTrainingViewError("world_size must be positive")
    digest = hashlib.sha256(
        f"exp123-catalog-owner-v1\0{int(seed)}\0{shard_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % int(world_size)


def _task_metadata(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = entry.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise CatalogTrainingViewError("catalog entry metadata must be an object")
    candidates = (
        entry,
        metadata.get("task"),
        metadata.get("shard_manifest", {}).get("task")
        if isinstance(metadata.get("shard_manifest"), Mapping)
        else None,
        metadata.get("producer_manifest", {}).get("task")
        if isinstance(metadata.get("producer_manifest"), Mapping)
        else None,
        metadata,
    )
    for candidate in candidates:
        if isinstance(candidate, Mapping) and all(
            key in candidate for key in ("n", "p", "num_tasks")
        ):
            return candidate
    raise CatalogTrainingViewError(
        f"catalog shard {entry.get('shard_id')!r} lacks n/p/num_tasks metadata"
    )


def build_rank_training_groups(
    entries: Sequence[Mapping[str, Any]],
    *,
    resolve_path: Callable[[Mapping[str, Any]], str | Path],
    rank: int,
    world_size: int,
    seed: int,
    prior_contracts: Mapping[str, str],
    prior_tasks_per_file: Mapping[str, int],
    expected_n_min: int,
    expected_n_max: int,
    expected_p_min: int,
    expected_p_max: int,
    expected_max_cells: int,
    corr_threshold: float,
    p_bucket_subdiv: int,
    n_bucket_subdiv: int,
) -> tuple[list[dict[str, Any]], dict[str, tuple[int, int, int]]]:
    """Build exact catalog-driven groups without scanning part_*/manifest.json."""
    if not entries:
        raise CatalogTrainingViewError("active catalog has no entries")
    if not (0 <= int(rank) < int(world_size)):
        raise CatalogTrainingViewError("rank is outside world_size")
    p_bucket_subdiv = max(1, int(p_bucket_subdiv))
    n_bucket_subdiv = max(0, int(n_bucket_subdiv))
    by_prior: dict[str, dict[Any, list[tuple[str, str]]]] = {}
    file_shapes: dict[str, tuple[int, int, int]] = {}
    seen_paths: set[str] = set()
    seen_ids: set[str] = set()
    for entry in entries:
        prior = str(entry.get("prior", ""))
        if prior not in prior_contracts or prior not in prior_tasks_per_file:
            raise CatalogTrainingViewError(f"unsupported catalog prior {prior!r}")
        shard_id = str(entry.get("shard_id", ""))
        if not shard_id or shard_id in seen_ids:
            raise CatalogTrainingViewError(f"missing/duplicate shard_id {shard_id!r}")
        seen_ids.add(shard_id)
        path = Path(resolve_path(entry)).resolve()
        path_str = str(path)
        if path_str in seen_paths:
            raise CatalogTrainingViewError(f"duplicate resolved path {path}")
        seen_paths.add(path_str)
        if not path.is_file():
            raise CatalogTrainingViewError(f"catalog shard is missing: {path}")
        expected_bytes = int(entry.get("bytes", -1))
        if expected_bytes <= 0 or path.stat().st_size != expected_bytes:
            raise CatalogTrainingViewError(
                f"catalog shard byte mismatch: {path}"
            )
        digest = str(entry.get("sha256", ""))
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise CatalogTrainingViewError(f"invalid shard SHA256 for {path}")
        task = _task_metadata(entry)
        n = int(task["n"])
        p = int(task["p"])
        num_tasks = int(task["num_tasks"])
        if not (int(expected_n_min) <= n <= int(expected_n_max)):
            raise CatalogTrainingViewError(f"N={n} outside contract in {path}")
        if not (int(expected_p_min) <= p <= int(expected_p_max)):
            raise CatalogTrainingViewError(f"P={p} outside contract in {path}")
        if n * p > int(expected_max_cells):
            raise CatalogTrainingViewError(f"N*P exceeds contract in {path}")
        expected_tasks = int(prior_tasks_per_file[prior])
        if num_tasks != expected_tasks:
            raise CatalogTrainingViewError(
                f"num_tasks={num_tasks} != {expected_tasks} in {path}"
            )
        measured_corr = float(entry.get("task_max_abs_corr_max", float("nan")))
        if not math.isfinite(measured_corr) or measured_corr >= float(corr_threshold):
            raise CatalogTrainingViewError(
                f"max |corr| certificate failed in {path}: {measured_corr}"
            )
        file_shapes[path_str] = (n, p, num_tasks)
        p_bin = int(math.floor(math.log2(p) * p_bucket_subdiv))
        bucket = (
            (p_bin, int(math.floor(math.log2(n) * n_bucket_subdiv)))
            if n_bucket_subdiv > 0
            else p_bin
        )
        by_prior.setdefault(prior, {}).setdefault(bucket, []).append(
            (path_str, shard_id)
        )

    groups = []
    for prior in sorted(by_prior):
        buckets = {}
        for bucket, paths_and_ids in sorted(by_prior[prior].items()):
            # CatalogStore.entries_for_rank already applied the immutable
            # catalog assignment. Do not independently rehash/reassign here.
            owned = sorted(path for path, _shard_id in paths_and_ids)
            if owned:
                buckets[bucket] = {"files": owned, "fi": 0}
        if buckets:
            groups.append(
                {
                    "root": f"catalog:{prior}",
                    "dataset_contract": prior_contracts[prior],
                    "buckets": buckets,
                    "bucket_ids": sorted(buckets),
                }
            )
    if not groups:
        raise CatalogTrainingViewError(
            f"rank {rank} owns no shards in active catalog"
        )
    return groups, file_shapes
