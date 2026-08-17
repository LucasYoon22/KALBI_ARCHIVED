#!/usr/bin/env python3
"""Bootstrap and continuously rotate the canonical 100GB active catalog."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import time
from typing import Any

from .rolling_catalog import (
    CatalogError,
    CatalogStore,
    GCBlocked,
    PublishDeferred,
)
from .rolling_producer import (
    PRIOR_KEYS,
    ensure_immutable_root_lock,
    load_and_validate_config,
)


CURATOR_VERSION = "v58_v17_n200p500_50_50_rolling100gb_curator_v1"


def read_inbox(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted((root / "staging" / "inbox").rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_inbox_path"] = str(path)
        entries.append(payload)
    return entries


def project_npz_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*.npz") if path.is_file())


def bootstrap_if_ready(
    store: CatalogStore,
    root: Path,
    root_lock: dict[str, Any],
    *,
    target_bytes: int,
) -> bool:
    if store.current_path.exists():
        return False
    target_per_prior = int(target_bytes) // len(PRIOR_KEYS)
    grouped: dict[str, list[dict[str, Any]]] = {prior: [] for prior in PRIOR_KEYS}
    for raw in read_inbox(root):
        prior = str(raw.get("prior"))
        if prior in grouped:
            grouped[prior].append(raw)
    candidates: list[dict[str, Any]] = []
    per_prior_bytes: dict[str, int] = {}
    for prior in PRIOR_KEYS:
        subtotal = 0
        for raw in grouped[prior]:
            entry = dict(raw)
            entry.pop("_inbox_path", None)
            # Polling remains metadata-only; epoch-0 publication hashes every
            # selected byte exactly once.
            normalized = store.normalise_staged_entry(entry, verify_file=False)
            candidates.append(normalized)
            subtotal += int(normalized["bytes"])
            if subtotal >= target_per_prior:
                break
        per_prior_bytes[prior] = subtotal
    total = sum(per_prior_bytes.values())
    if any(value < target_per_prior for value in per_prior_bytes.values()):
        print(
            json.dumps(
                {
                    "kind": "bootstrap_wait",
                    "validated_files": len(candidates),
                    "validated_bytes": total,
                    "target_bytes": int(target_bytes),
                    "target_bytes_per_prior": target_per_prior,
                    "validated_bytes_per_prior": per_prior_bytes,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return False
    pointer, catalog = store.bootstrap_from_staged_entries(
        candidates,
        pipeline_config_sha256=root_lock["pipeline_config_sha256"],
        source_hashes=root_lock["source_hashes"],
        verify_files=True,
    )
    print(
        json.dumps(
            {
                "kind": "bootstrap_published",
                "epoch": pointer["epoch"],
                "catalog_sha256": pointer["catalog_sha256"],
                "files": len(catalog["files"]),
                "bytes": catalog["accounting"]["total_bytes"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return True


def gc_ready_retirements(store: CatalogStore, trainer_id: str, world_size: int) -> int:
    try:
        tombstone = store.create_gc_tombstone(
            trainer_id, world_size=int(world_size)
        )
    except GCBlocked:
        return 0
    if tombstone is None:
        return 0
    removed = store.execute_gc_tombstone(tombstone)
    print(
        json.dumps(
            {
                "kind": "gc_complete",
                "tombstone": str(tombstone),
                "removed_files": len(removed),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return len(removed)


def publish_if_ready(
    store: CatalogStore,
    root: Path,
    *,
    trainer_id: str,
    world_size: int,
    refresh_bytes: int,
    target_bytes: int,
) -> bool:
    pointer, current = store.load_current(verify_files=False)
    digest = str(pointer["catalog_sha256"])
    if not store.all_ranks_acked(trainer_id, digest, int(world_size)):
        return False
    if not store._has_verified_resumable_checkpoint(digest):
        return False
    historical_ids = store._historical_identity_sets()[0]
    refresh_per_prior = max(1, int(refresh_bytes) // len(PRIOR_KEYS))
    grouped: dict[str, list[dict[str, Any]]] = {prior: [] for prior in PRIOR_KEYS}
    for raw in read_inbox(root):
        prior = str(raw.get("prior"))
        if prior in grouped and int(raw.get("shard_id", -1)) not in historical_ids:
            grouped[prior].append(raw)
    additions: list[dict[str, Any]] = []
    addition_bytes_by_prior: dict[str, int] = {}
    for prior in PRIOR_KEYS:
        subtotal = 0
        for raw in grouped[prior]:
            entry = dict(raw)
            entry.pop("_inbox_path", None)
            normalized = store.normalise_staged_entry(entry, verify_file=True)
            additions.append(normalized)
            subtotal += int(normalized["bytes"])
            if subtotal >= refresh_per_prior:
                break
        addition_bytes_by_prior[prior] = subtotal
    if any(value < refresh_per_prior for value in addition_bytes_by_prior.values()):
        return False
    addition_bytes = sum(addition_bytes_by_prior.values())
    if not additions:
        return False
    retirements: list[int] = []
    target_per_prior = int(target_bytes) // len(PRIOR_KEYS)
    current_by_prior = current["accounting"]["per_prior"]
    for prior in PRIOR_KEYS:
        projected = int(current_by_prior[prior]["total_bytes"]) + addition_bytes_by_prior[prior]
        candidates = sorted(
            (entry for entry in current["files"] if entry["prior"] == prior),
            key=lambda item: int(item["shard_id"]),
        )
        for entry in candidates:
            if projected <= target_per_prior:
                break
            next_projected = projected - int(entry["bytes"])
            if next_projected < int(store.policy.prior_min_bytes):
                break
            retirements.append(int(entry["shard_id"]))
            projected = next_projected
    pointer_new, catalog_new = store.publish_transition(
        additions,
        retirements,
        expected_parent_sha256=digest,
        observed_project_bytes=project_npz_bytes(root),
        verify_files=True,
        check_free_space=True,
    )
    print(
        json.dumps(
            {
                "kind": "transition_published",
                "epoch": pointer_new["epoch"],
                "catalog_sha256": pointer_new["catalog_sha256"],
                "additions": len(additions),
                "addition_bytes": addition_bytes,
                "addition_bytes_by_prior": addition_bytes_by_prior,
                "retirements": len(retirements),
                "active_files": len(catalog_new["files"]),
                "active_bytes": catalog_new["accounting"]["total_bytes"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trainer-id", required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--target-bytes", type=int, default=100_000_000_000)
    parser.add_argument("--refresh-bytes", type=int, default=2_000_000_000)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    config, contract_sha = load_and_validate_config(args.config)
    root = args.root.resolve()
    root_lock = ensure_immutable_root_lock(root, config, contract_sha)
    store = CatalogStore(
        config["control_root"], root, priors=PRIOR_KEYS
    )
    stopped = False

    def request_stop(_signal: int, _frame: Any) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    while not stopped:
        try:
            if not store.current_path.exists():
                bootstrap_if_ready(
                    store, root, root_lock, target_bytes=args.target_bytes
                )
            else:
                gc_ready_retirements(store, args.trainer_id, args.world_size)
                publish_if_ready(
                    store,
                    root,
                    trainer_id=args.trainer_id,
                    world_size=args.world_size,
                    refresh_bytes=args.refresh_bytes,
                    target_bytes=args.target_bytes,
                )
        except PublishDeferred as exc:
            print(json.dumps({"kind": "publish_deferred", "reason": str(exc)}), flush=True)
        except (CatalogError, OSError, ValueError, json.JSONDecodeError) as exc:
            print(json.dumps({"kind": "curator_error", "error": repr(exc)}), flush=True)
            raise
        time.sleep(max(1.0, float(args.poll_seconds)))


if __name__ == "__main__":
    main()
