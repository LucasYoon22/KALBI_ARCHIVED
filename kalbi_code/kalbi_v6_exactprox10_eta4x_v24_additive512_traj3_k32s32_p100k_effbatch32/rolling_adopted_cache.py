"""Small process-shared cache for catalog-adopted shard IDs."""

from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_v58_chebyshev_v14_scmmixture.rolling_catalog import (
    CatalogError,
    CatalogStore,
    atomic_write_json,
)


CACHE_VERSION = "v58_v17_p100k_adopted_shard_ids_cache_v1"
CACHE_FILENAME = "ADOPTED_SHARD_IDS_CACHE.json"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CatalogError(f"expected a JSON object: {path}")
    return payload


def _current_identity(store: CatalogStore) -> tuple[int, str]:
    pointer = _read_json(store.current_path)
    return int(pointer["epoch"]), str(pointer["catalog_sha256"])


def _cache_ids_if_current(
    cache_path: Path,
    *,
    epoch: int,
    digest: str,
) -> set[int] | None:
    if not cache_path.is_file():
        return None
    try:
        payload = _read_json(cache_path)
        if payload.get("version") != CACHE_VERSION:
            return None
        if int(payload.get("catalog_epoch", -1)) != epoch:
            return None
        if str(payload.get("catalog_sha256", "")) != digest:
            return None
        raw_ids = payload.get("adopted_shard_ids")
        if not isinstance(raw_ids, list):
            return None
        return {int(value) for value in raw_ids}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def load_cached_adopted_ids(store: CatalogStore) -> set[int]:
    """Return adopted IDs, rebuilding the cache only when CURRENT advances."""

    epoch, digest = _current_identity(store)
    cache_path = store.control_root / CACHE_FILENAME
    cached = _cache_ids_if_current(cache_path, epoch=epoch, digest=digest)
    if cached is not None:
        return cached

    lock_path = store.lock_dir / "adopted_shard_ids_cache.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        epoch, digest = _current_identity(store)
        cached = _cache_ids_if_current(cache_path, epoch=epoch, digest=digest)
        if cached is not None:
            return cached

        _pointer, current = store.load_current(verify_files=False)
        adopted = {int(entry["shard_id"]) for entry in current["files"]}
        # The original P<=100K catalog predates adopted_shard_ids(). Its
        # immutable parent chain still records every retired shard, so include
        # those IDs until the curator's GC has actually removed their files.
        adopted.update(int(shard_id) for shard_id in store._retirement_records())
        atomic_write_json(
            cache_path,
            {
                "version": CACHE_VERSION,
                "catalog_epoch": epoch,
                "catalog_sha256": digest,
                "adopted_shard_ids": sorted(adopted),
                "count": len(adopted),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return adopted
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
