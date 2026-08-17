"""Start the immutable V11 generator with cached producer history accounting."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
from typing import Any

from model_v58_chebyshev_v14_scmmixture import rolling_producer as producer
from model_v58_chebyshev_v14_scmmixture.rolling_catalog import atomic_write_json
from model_v58_chebyshev_v17_scmmixture_p100k import (
    rolling_producer_backpressure_fixed as fixed,
)
from model_v58_chebyshev_v17_scmmixture_p100k import (
    rolling_producer_tabicl_v11_expvarnoise as v11,
)
from model_v58_chebyshev_v17_scmmixture_p100k import (
    rolling_adopted_cache as adopted_cache,
)


_SHARD_ID_PATTERN = re.compile(rb'"shard_id"\s*:\s*(\d+)')
_PARENT_FILE_PATTERN = re.compile(rb'"catalog_file"\s*:\s*"([^"]+)"')


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return payload


def _stream_catalog_metadata(path: Path) -> tuple[set[int], str | None]:
    """Extract shard IDs and the parent filename without materializing JSON."""

    shard_ids: set[int] = set()
    parent_files: set[str] = set()
    carry = b""
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            data = carry + chunk
            shard_ids.update(int(match.group(1)) for match in _SHARD_ID_PATTERN.finditer(data))
            parent_files.update(
                match.group(1).decode("ascii")
                for match in _PARENT_FILE_PATTERN.finditer(data)
            )
            carry = data[-256:]
    if not shard_ids:
        raise RuntimeError(f"current catalog contains no shard IDs: {path}")
    if len(parent_files) > 1:
        raise RuntimeError(f"catalog contains multiple parent files: {path}")
    return shard_ids, next(iter(parent_files), None)


def _stream_catalog_history_shard_ids(
    store: v11.TabICLV11CatalogStore, current_filename: Path
) -> set[int]:
    shard_ids: set[int] = set()
    seen: set[str] = set()
    filename: str | None = str(current_filename)
    while filename is not None:
        candidate = Path(filename)
        if candidate.name != filename or filename in seen:
            raise RuntimeError(f"invalid catalog parent chain at {filename!r}")
        seen.add(filename)
        path = store.control_root / "catalogs" / candidate
        current_ids, filename = _stream_catalog_metadata(path)
        shard_ids.update(current_ids)
    return shard_ids


def _fast_adopted_ids(store: v11.TabICLV11CatalogStore) -> set[int]:
    """Advance an existing monotone ID cache using only the current catalog."""

    pointer = _read_object(store.current_path)
    epoch = int(pointer["epoch"])
    digest = str(pointer["catalog_sha256"])
    cache_path = store.control_root / adopted_cache.CACHE_FILENAME

    def read_cache() -> tuple[dict[str, Any], set[int]] | None:
        if not cache_path.is_file():
            return None
        payload = _read_object(cache_path)
        if payload.get("version") != adopted_cache.CACHE_VERSION:
            return None
        raw_ids = payload.get("adopted_shard_ids")
        if not isinstance(raw_ids, list):
            return None
        return payload, {int(value) for value in raw_ids}

    cached = read_cache()
    if cached is not None:
        payload, shard_ids = cached
        if (
            shard_ids
            and int(payload.get("catalog_epoch", -1)) == epoch
            and str(payload.get("catalog_sha256", "")) == digest
        ):
            return shard_ids

    lock_path = store.lock_dir / "adopted_shard_ids_cache.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        pointer = _read_object(store.current_path)
        epoch = int(pointer["epoch"])
        digest = str(pointer["catalog_sha256"])
        cached = read_cache()
        rebuild_history = cached is None or not cached[1]
        if rebuild_history:
            prior_epoch = -1
            shard_ids: set[int] = set()
        else:
            payload, shard_ids = cached
            prior_epoch = int(payload.get("catalog_epoch", -1))
            if prior_epoch > epoch:
                raise RuntimeError(
                    f"adopted-ID cache epoch {prior_epoch} exceeds CURRENT epoch {epoch}"
                )
            if (
                shard_ids
                and prior_epoch == epoch
                and str(payload.get("catalog_sha256", "")) == digest
            ):
                return shard_ids

        filename = Path(str(pointer["catalog_file"]))
        if filename.name != str(filename):
            raise RuntimeError(f"unsafe current catalog filename: {filename}")
        if rebuild_history:
            shard_ids.update(_stream_catalog_history_shard_ids(store, filename))
        else:
            catalog_path = store.control_root / "catalogs" / filename
            current_ids, _parent = _stream_catalog_metadata(catalog_path)
            shard_ids.update(current_ids)
        atomic_write_json(
            cache_path,
            {
                "version": adopted_cache.CACHE_VERSION,
                "catalog_epoch": epoch,
                "catalog_sha256": digest,
                "adopted_shard_ids": sorted(shard_ids),
                "count": len(shard_ids),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "advanced_from_epoch": prior_epoch,
                "producer_fast_path": "stale_cache_union_current_catalog_shard_ids_v1",
                "producer_history_complete": True,
            },
        )
        return shard_ids
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


class ProducerCatalogStore(v11.TabICLV11CatalogStore):
    """Avoid rebuilding the multi-GB catalog chain for producer backpressure."""

    def load_current(
        self, *, verify_files: bool = True
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # The producer's two callers use this only as a CURRENT-exists guard.
        # Full catalog validation and transition publication remain curator-only.
        if verify_files:
            raise RuntimeError("producer cannot request full current-catalog validation")
        pointer = _read_object(self.current_path)
        filename = Path(str(pointer["catalog_file"]))
        digest = str(pointer["catalog_sha256"])
        if filename.name != str(filename) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError("invalid CURRENT catalog pointer")
        catalog_path = self.control_root / "catalogs" / filename
        if not catalog_path.is_file():
            raise RuntimeError(f"CURRENT catalog file is missing: {catalog_path}")
        return pointer, {
            "epoch": int(pointer["epoch"]),
            "data_root": str(self.data_root),
            "files": [],
        }

    def _historical_identity_sets(self) -> tuple[set[int], set[str], set[int]]:
        # rolling_producer.produce_one consumes only element zero here. The
        # curator continues to use TabICLV11CatalogStore and performs the full
        # hash/seed history validation before publishing a catalog transition.
        return _fast_adopted_ids(self), set(), set()

    def publish_transition(self, *args: object, **kwargs: object) -> dict:
        raise RuntimeError("ProducerCatalogStore cannot publish catalog transitions")


def main(argv: Sequence[str] | None = None) -> int:
    v11.configure_v11_runtime()
    producer.CatalogStore = ProducerCatalogStore
    fixed.CatalogStore = ProducerCatalogStore
    fixed.load_cached_adopted_ids = _fast_adopted_ids
    return fixed.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
