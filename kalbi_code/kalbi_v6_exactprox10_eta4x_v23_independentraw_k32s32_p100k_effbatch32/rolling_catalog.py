#!/usr/bin/env python3
"""Crash-safe rolling-corpus catalog and retention primitives for Exp123.

This module intentionally does not know about torch.  Rank 0 may poll CURRENT,
broadcast the loaded immutable catalog, and let every rank verify/ACK it at a
safe optimizer/selector boundary.  Publication and physical deletion are kept
out of the trainer process.

The implementation is fail closed:

* catalogs are immutable, content-hashed JSON artifacts with a parent hash;
* CURRENT is the only mutable publication pointer and is replaced atomically;
* surviving shard metadata may not change across a transition;
* additions and retirements must exactly describe the parent/child delta;
* all paths remain relative to one pinned data root and are resolved safely;
* GC needs a complete rank ACK set and protects CURRENT, every live lease, and
  every verified resumable checkpoint before it creates or executes a
  tombstone.

The initial Exp122 frozen inventory is supported as a one-time epoch-0 import.
That import is a semantic migration: it preserves relative paths, sizes, and
content hashes but adds Exp123 shard IDs and catalog metadata.
"""

from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


CATALOG_VERSION = "v12_canonical_rolling_catalog_v1"
POINTER_VERSION = "v12_canonical_current_pointer_v1"
SHARD_VERSION = "v12_canonical_shard_v1"
LEASE_VERSION = "v12_canonical_catalog_lease_v1"
ACK_VERSION = "v12_canonical_catalog_ack_v1"
CHECKPOINT_PIN_VERSION = "v12_canonical_checkpoint_pin_v1"
TOMBSTONE_VERSION = "v12_canonical_gc_tombstone_v1"
EXPECTED_PRIORS = ("tabpfn_v1", "tabicl_v2")
DEFAULT_PIPELINE_CONFIG_SHA256 = "7196a1a59ebe2f4a79efa2bb4569b141f3f7e0d3efd0ded08498620dc5df4f8a"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CatalogError(RuntimeError):
    """The catalog, pointer, or requested transition is invalid."""


class PublishDeferred(CatalogError):
    """A valid publication cannot happen yet; CURRENT remains unchanged."""


class GCBlocked(CatalogError):
    """GC proof is incomplete; no data file was removed."""


@dataclasses.dataclass(frozen=True)
class CapacityPolicy:
    """Configurable resident-corpus and retention limits.

    Defaults use decimal GB because the Exp122 generation contract used exact
    decimal byte targets.
    """

    active_min_bytes: int = 98_000_000_000
    active_max_bytes: int = 102_000_000_000
    prior_min_bytes: int = 49_000_000_000
    prior_max_bytes: int = 51_000_000_000
    hard_cap_bytes: int = 130_000_000_000
    min_free_bytes: int = 300_000_000_000
    world_size: int = 4
    minimum_publish_interval_seconds: int = 60
    retire_grace_epochs: int = 1
    retire_grace_seconds: int = 0
    tombstone_grace_seconds: int = 0

    def __post_init__(self) -> None:
        integer_fields = [
            "active_min_bytes",
            "active_max_bytes",
            "prior_min_bytes",
            "prior_max_bytes",
            "hard_cap_bytes",
            "min_free_bytes",
            "world_size",
            "minimum_publish_interval_seconds",
            "retire_grace_epochs",
            "retire_grace_seconds",
            "tombstone_grace_seconds",
        ]
        for name in integer_fields:
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if self.active_min_bytes > self.active_max_bytes:
            raise ValueError("active byte band is inverted")
        if self.prior_min_bytes > self.prior_max_bytes:
            raise ValueError("prior byte band is inverted")
        if self.active_max_bytes > self.hard_cap_bytes:
            raise ValueError("active_max_bytes exceeds hard_cap_bytes")

    def to_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime | str | None) -> datetime:
    if value is None:
        return utc_now()
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone aware")
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return _as_utc(value).isoformat().replace("+00:00", "Z")


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: Any, label: str) -> str:
    value = str(value).lower()
    if not SHA256_RE.fullmatch(value):
        raise CatalogError(f"{label} is not a lowercase SHA256 digest: {value!r}")
    return value


def atomic_write_json(path: str | Path, payload: Any, *, overwrite: bool = True) -> None:
    """Write one JSON artifact durably and atomically."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{os.urandom(6).hex()}")
    encoded = _json_bytes(payload)
    try:
        with tmp.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if not overwrite and path.exists():
            raise FileExistsError(path)
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if tmp.exists():
            tmp.unlink()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"cannot read valid JSON from {path}: {exc}") from exc


class CatalogStore:
    """Own Exp123 immutable catalogs, CURRENT, leases, ACKs, and tombstones."""

    def __init__(
        self,
        control_root: str | Path,
        data_root: str | Path,
        policy: CapacityPolicy | None = None,
        *,
        priors: Sequence[str] = EXPECTED_PRIORS,
    ) -> None:
        self.control_root = Path(control_root).expanduser().resolve()
        self.data_root = Path(data_root).expanduser().resolve()
        self.policy = policy or CapacityPolicy()
        self.priors = tuple(str(item) for item in priors)
        if not self.priors or len(set(self.priors)) != len(self.priors):
            raise ValueError("priors must be unique and nonempty")
        self.catalog_dir = self.control_root / "catalogs"
        self.lease_dir = self.control_root / "leases"
        self.ack_dir = self.control_root / "acks"
        self.checkpoint_dir = self.control_root / "checkpoints"
        self.tombstone_dir = self.control_root / "tombstones"
        self.lock_dir = self.control_root / "locks"
        self.current_path = self.control_root / "CURRENT.json"
        self.next_shard_id_path = self.control_root / "NEXT_SHARD_ID.json"
        self.source_lock_path = self.control_root / "SOURCE_LOCK.json"
        for directory in (
            self.catalog_dir,
            self.lease_dir,
            self.ack_dir,
            self.checkpoint_dir,
            self.tombstone_dir,
            self.lock_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def _lock(self, name: str, *, nonblocking: bool = False) -> Iterator[None]:
        path = self.lock_dir / f"{name}.lock"
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
        try:
            try:
                fcntl.flock(fd, flags)
            except BlockingIOError as exc:
                raise PublishDeferred(f"{name} lock is already held") from exc
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _safe_relative_path(self, value: Any) -> str:
        relative = Path(str(value))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise CatalogError(f"unsafe data-root-relative path: {value!r}")
        if relative.suffix != ".npz":
            raise CatalogError(f"catalog shard is not an NPZ: {relative}")
        absolute = (self.data_root / relative).resolve()
        if not absolute.is_relative_to(self.data_root):
            raise CatalogError(f"catalog shard escapes data root: {relative}")
        return relative.as_posix()

    def resolve_entry_path(self, entry: Mapping[str, Any]) -> Path:
        relative = self._safe_relative_path(entry.get("relative_path", entry.get("path")))
        result = (self.data_root / relative).resolve()
        if not result.is_relative_to(self.data_root):
            raise CatalogError(f"resolved shard escapes data root: {relative}")
        return result

    def entries_for_rank(
        self,
        catalog: Mapping[str, Any],
        rank: int,
    ) -> list[dict[str, Any]]:
        """Return direct, manifest-free training entries assigned to one rank."""

        if not 0 <= rank < self.policy.world_size:
            raise ValueError("rank is outside catalog world size")
        self._validate_catalog_payload(catalog)
        by_id = {int(entry["shard_id"]): entry for entry in catalog["files"]}
        selected: list[dict[str, Any]] = []
        for prior in self.priors:
            for shard_id in catalog["assignments"][prior][str(rank)]:
                entry = dict(by_id[int(shard_id)])
                entry["absolute_path"] = str(self.resolve_entry_path(entry))
                selected.append(entry)
        return selected

    def _normalise_entry(
        self,
        entry: Mapping[str, Any],
        *,
        frozen_import: bool = False,
        assigned_id: int | None = None,
    ) -> dict[str, Any]:
        relative_path = self._safe_relative_path(
            entry.get("relative_path", entry.get("path", ""))
        )
        shard_id_raw = assigned_id if assigned_id is not None else entry.get("shard_id")
        if not isinstance(shard_id_raw, int) or isinstance(shard_id_raw, bool) or shard_id_raw < 0:
            raise CatalogError(f"invalid shard_id for {relative_path}: {shard_id_raw!r}")
        prior = str(entry.get("prior") or Path(relative_path).parts[0])
        if prior not in self.priors:
            raise CatalogError(f"invalid prior for {relative_path}: {prior!r}")
        size = int(entry.get("bytes", -1))
        if size <= 0:
            raise CatalogError(f"invalid byte count for {relative_path}: {size}")
        content_hash = _validate_sha256(entry.get("sha256", ""), f"shard {relative_path}")
        seed_value = entry.get("seed", entry.get("batch_seed"))
        if seed_value is None and not frozen_import:
            raise CatalogError(f"new shard lacks a durable seed: {relative_path}")
        seed = None if seed_value is None else int(seed_value)
        if seed is not None and seed < 0:
            raise CatalogError(f"invalid seed for {relative_path}: {seed}")

        metadata = dict(entry.get("metadata") or {})
        dimensions: dict[str, int] = {}
        for key in ("n", "p", "num_tasks"):
            raw = entry.get(key, metadata.pop(key, -1))
            value = int(raw)
            if value <= 0:
                raise CatalogError(f"shard has invalid {key}: {relative_path}")
            dimensions[key] = value
        corr = float(
            entry.get(
                "task_max_abs_corr_max",
                metadata.pop("task_max_abs_corr_max", float("nan")),
            )
        )
        if not math.isfinite(corr) or not 0.0 <= corr < 0.95:
            raise CatalogError(f"shard has invalid correlation certificate: {relative_path}")
        pipeline_raw = entry.get(
            "pipeline_config_sha256", metadata.pop("pipeline_config_sha256", None)
        )
        pipeline_hash = (
            None
            if pipeline_raw is None and frozen_import
            else _validate_sha256(pipeline_raw, f"pipeline config for {relative_path}")
        )
        sources_raw = entry.get("source_hashes", metadata.pop("source_hashes", {}))
        if not isinstance(sources_raw, Mapping):
            raise CatalogError(f"source_hashes is not a mapping: {relative_path}")
        source_hashes = {
            str(key): _validate_sha256(value, f"source hash {key} for {relative_path}")
            for key, value in sorted(sources_raw.items())
        }
        created_raw = entry.get("created_at_utc", metadata.pop("created_at_utc", None))
        created_at = None if created_raw is None else _iso(_as_utc(created_raw))
        for key in ("worker_id", "batch_id", "stratum"):
            if key in entry and key not in metadata:
                metadata[key] = entry[key]

        return {
            "schema_version": SHARD_VERSION,
            "shard_id": shard_id_raw,
            "prior": prior,
            "relative_path": relative_path,
            "bytes": size,
            "sha256": content_hash,
            "seed": seed,
            "n": dimensions["n"],
            "p": dimensions["p"],
            "num_tasks": dimensions["num_tasks"],
            "task_max_abs_corr_max": corr,
            "pipeline_config_sha256": pipeline_hash,
            "source_hashes": source_hashes,
            "created_at_utc": created_at,
            "origin": "frozen_import" if frozen_import else "validated_staging",
            "metadata": metadata,
        }

    @staticmethod
    def _entry_sort_key(entry: Mapping[str, Any]) -> tuple[int, str]:
        return int(entry["shard_id"]), str(entry["relative_path"])

    def _validate_unique_entries(self, entries: Sequence[Mapping[str, Any]]) -> None:
        seen_ids: set[int] = set()
        seen_paths: set[str] = set()
        seen_hashes: set[str] = set()
        seen_seeds: set[int] = set()
        for entry in entries:
            shard_id = int(entry["shard_id"])
            path = str(entry["relative_path"])
            content_hash = str(entry["sha256"])
            seed = entry.get("seed")
            if shard_id in seen_ids:
                raise CatalogError(f"duplicate shard_id in catalog: {shard_id}")
            if path in seen_paths:
                raise CatalogError(f"duplicate relative_path in catalog: {path}")
            if content_hash in seen_hashes:
                raise CatalogError(f"duplicate content hash in catalog: {content_hash}")
            if seed is not None and int(seed) in seen_seeds:
                raise CatalogError(f"duplicate seed in catalog: {seed}")
            seen_ids.add(shard_id)
            seen_paths.add(path)
            seen_hashes.add(content_hash)
            if seed is not None:
                seen_seeds.add(int(seed))

    def _accounting(self, entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        per_prior = {
            prior: {"num_files": 0, "total_bytes": 0, "num_tasks": 0}
            for prior in self.priors
        }
        for entry in entries:
            stats = per_prior[str(entry["prior"])]
            stats["num_files"] += 1
            stats["total_bytes"] += int(entry["bytes"])
            stats["num_tasks"] += int(entry["num_tasks"])
        return {
            "num_files": len(entries),
            "total_bytes": sum(int(entry["bytes"]) for entry in entries),
            "per_prior": per_prior,
        }

    def _assignments(self, entries: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, list[int]]]:
        assignments: dict[str, dict[str, list[int]]] = {}
        for prior in self.priors:
            ranks = {str(rank): [] for rank in range(self.policy.world_size)}
            candidates = [entry for entry in entries if entry["prior"] == prior]
            candidates.sort(
                key=lambda item: hashlib.sha256(
                    f"{item['sha256']}\0{item['shard_id']}".encode("utf-8")
                ).digest()
            )
            for index, entry in enumerate(candidates):
                ranks[str(index % self.policy.world_size)].append(int(entry["shard_id"]))
            assignments[prior] = ranks
        return assignments

    def _validate_capacity(
        self,
        entries: Sequence[Mapping[str, Any]],
        *,
        observed_project_bytes: int | None = None,
        check_free_space: bool = True,
    ) -> dict[str, Any]:
        accounting = self._accounting(entries)
        total = int(accounting["total_bytes"])
        if not self.policy.active_min_bytes <= total <= self.policy.active_max_bytes:
            raise PublishDeferred(
                f"active bytes outside band: {total} not in "
                f"[{self.policy.active_min_bytes}, {self.policy.active_max_bytes}]"
            )
        for prior, stats in accounting["per_prior"].items():
            prior_bytes = int(stats["total_bytes"])
            if not self.policy.prior_min_bytes <= prior_bytes <= self.policy.prior_max_bytes:
                raise PublishDeferred(
                    f"{prior} bytes outside band: {prior_bytes} not in "
                    f"[{self.policy.prior_min_bytes}, {self.policy.prior_max_bytes}]"
                )
            if int(stats["num_files"]) < self.policy.world_size:
                raise PublishDeferred(
                    f"{prior} has {stats['num_files']} files for {self.policy.world_size} ranks"
                )
        if observed_project_bytes is not None:
            observed_project_bytes = int(observed_project_bytes)
            if observed_project_bytes > self.policy.hard_cap_bytes:
                raise PublishDeferred(
                    f"project high-water exceeded: {observed_project_bytes} > "
                    f"{self.policy.hard_cap_bytes}"
                )
        if check_free_space and self.policy.min_free_bytes:
            free_bytes = shutil.disk_usage(self.data_root).free
            if free_bytes < self.policy.min_free_bytes:
                raise PublishDeferred(
                    f"free-space reserve violated: {free_bytes} < {self.policy.min_free_bytes}"
                )
        return accounting

    def verify_catalog_files(
        self,
        catalog: Mapping[str, Any],
        *,
        hashes: bool = True,
    ) -> None:
        if str(catalog.get("data_root")) != str(self.data_root):
            raise CatalogError(
                f"catalog data_root mismatch: {catalog.get('data_root')!r} != {self.data_root}"
            )
        for entry in catalog.get("files", []):
            path = self.resolve_entry_path(entry)
            expected_bytes = int(entry["bytes"])
            if not path.is_file() or path.stat().st_size != expected_bytes:
                raise CatalogError(
                    f"catalog shard missing/size mismatch: {path} expected={expected_bytes}"
                )
            if hashes:
                observed = sha256_file(path)
                if observed != entry["sha256"]:
                    raise CatalogError(
                        f"catalog shard hash mismatch: {path} {observed} != {entry['sha256']}"
                    )

    def _validate_catalog_payload(self, catalog: Mapping[str, Any]) -> None:
        if catalog.get("version") != CATALOG_VERSION:
            raise CatalogError(f"unsupported catalog version: {catalog.get('version')!r}")
        epoch = catalog.get("epoch")
        if not isinstance(epoch, int) or epoch < 0:
            raise CatalogError(f"invalid catalog epoch: {epoch!r}")
        if str(catalog.get("data_root")) != str(self.data_root):
            raise CatalogError("catalog is bound to a different data_root")
        entries = catalog.get("files")
        if not isinstance(entries, list) or not entries:
            raise CatalogError("catalog has no files")
        normalised = [self._normalise_entry(entry, frozen_import=entry.get("origin") == "frozen_import") for entry in entries]
        if normalised != entries:
            raise CatalogError("catalog entries are not canonical")
        if entries != sorted(entries, key=self._entry_sort_key):
            raise CatalogError("catalog entries are not sorted canonically")
        self._validate_unique_entries(entries)
        if catalog.get("accounting") != self._accounting(entries):
            raise CatalogError("catalog accounting does not match files")
        if catalog.get("assignments") != self._assignments(entries):
            raise CatalogError("catalog assignments do not match files/policy")
        if catalog.get("capacity_policy") != self.policy.to_dict():
            raise CatalogError("catalog capacity policy differs from this store")

    def _catalog_path(self, filename: str) -> Path:
        relative = Path(filename)
        if relative.is_absolute() or len(relative.parts) != 1 or relative.suffix != ".json":
            raise CatalogError(f"unsafe catalog filename: {filename!r}")
        path = (self.catalog_dir / relative).resolve()
        if not path.is_relative_to(self.catalog_dir.resolve()):
            raise CatalogError(f"catalog filename escapes catalog dir: {filename!r}")
        return path

    def _load_catalog_file(self, filename: str, expected_hash: str) -> dict[str, Any]:
        expected_hash = _validate_sha256(expected_hash, "catalog hash")
        path = self._catalog_path(filename)
        if not path.is_file():
            raise CatalogError(f"catalog artifact is missing: {path}")
        observed = sha256_file(path)
        if observed != expected_hash:
            raise CatalogError(f"catalog hash mismatch: {observed} != {expected_hash}")
        catalog = _read_json(path)
        self._validate_catalog_payload(catalog)
        filename_match = re.fullmatch(
            r"CATALOG_EPOCH_(\d{8})_([0-9a-f]{16})\.json", path.name
        )
        if not filename_match:
            raise CatalogError(f"noncanonical catalog filename: {path.name}")
        if int(filename_match.group(1)) != int(catalog["epoch"]):
            raise CatalogError("catalog filename epoch mismatch")
        if filename_match.group(2) != observed[:16]:
            raise CatalogError("catalog filename hash prefix mismatch")
        return catalog

    def load_current(self, *, verify_files: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.current_path.is_file():
            raise CatalogError(f"CURRENT is missing: {self.current_path}")
        pointer = _read_json(self.current_path)
        if pointer.get("version") != POINTER_VERSION:
            raise CatalogError(f"unsupported CURRENT version: {pointer.get('version')!r}")
        catalog = self._load_catalog_file(
            str(pointer.get("catalog_file", "")),
            str(pointer.get("catalog_sha256", "")),
        )
        if int(pointer.get("epoch", -1)) != int(catalog["epoch"]):
            raise CatalogError("CURRENT epoch does not match catalog")
        if verify_files:
            self.verify_catalog_files(catalog, hashes=True)
        return pointer, catalog

    def _write_catalog(self, body: Mapping[str, Any]) -> tuple[Path, str]:
        encoded = _json_bytes(body)
        digest = sha256_bytes(encoded)
        filename = f"CATALOG_EPOCH_{int(body['epoch']):08d}_{digest[:16]}.json"
        path = self.catalog_dir / filename
        if path.exists():
            if path.read_bytes() != encoded:
                raise CatalogError(f"immutable catalog filename collision: {path}")
            return path, digest
        tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{os.urandom(6).hex()}")
        try:
            with tmp.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            if path.exists():
                if path.read_bytes() != encoded:
                    raise CatalogError(f"immutable catalog filename collision: {path}")
                tmp.unlink()
            else:
                os.replace(tmp, path)
                dir_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
        finally:
            if tmp.exists():
                tmp.unlink()
        return path, digest

    def _pointer_for(self, catalog_path: Path, digest: str, now: datetime) -> dict[str, Any]:
        next_time = now + timedelta(seconds=self.policy.minimum_publish_interval_seconds)
        return {
            "version": POINTER_VERSION,
            "epoch": int(re.search(r"EPOCH_(\d{8})", catalog_path.name).group(1)),
            "catalog_file": catalog_path.name,
            "catalog_sha256": digest,
            "published_at_utc": _iso(now),
            "next_publish_not_before_utc": _iso(next_time),
            "trainer_reload_interval_updates": 500,
        }

    def reserve_shard_ids(self, count: int) -> range:
        if not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer")
        with self._lock("shard_ids"):
            if self.next_shard_id_path.exists():
                payload = _read_json(self.next_shard_id_path)
                next_id = int(payload.get("next_shard_id", -1))
            else:
                next_id = 0
            if next_id < 0:
                raise CatalogError("NEXT_SHARD_ID is invalid")
            stop = next_id + count
            atomic_write_json(
                self.next_shard_id_path,
                {"version": "exp123_next_shard_id_v1", "next_shard_id": stop},
            )
            return range(next_id, stop)

    def bootstrap_from_staged_entries(
        self,
        raw_entries: Sequence[Mapping[str, Any]],
        *,
        pipeline_config_sha256: str,
        source_hashes: Mapping[str, str],
        now: datetime | str | None = None,
        verify_files: bool = True,
        check_free_space: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Create epoch 0 directly from validated fresh producer staging."""
        now_dt = _as_utc(now)
        pipeline_hash = _validate_sha256(
            pipeline_config_sha256, "canonical rolling pipeline config"
        )
        pinned_sources = {
            str(key): _validate_sha256(value, f"source hash {key}")
            for key, value in sorted(source_hashes.items())
        }
        if not pinned_sources:
            raise CatalogError("bootstrap requires pinned source hashes")
        with self._lock("publisher", nonblocking=True):
            if self.current_path.exists():
                raise CatalogError("refusing bootstrap because CURRENT already exists")
            entries = [self._normalise_entry(entry) for entry in raw_entries]
            entries.sort(key=self._entry_sort_key)
            self._validate_unique_entries(entries)
            observed_pipeline = {
                entry.get("pipeline_config_sha256") for entry in entries
            }
            observed_sources = {
                json.dumps(entry.get("source_hashes", {}), sort_keys=True)
                for entry in entries
            }
            if observed_pipeline != {pipeline_hash}:
                raise CatalogError("bootstrap entries have pipeline hash drift")
            if observed_sources != {json.dumps(pinned_sources, sort_keys=True)}:
                raise CatalogError("bootstrap entries have source hash drift")
            if verify_files:
                self.verify_catalog_files(
                    {"data_root": str(self.data_root), "files": entries},
                    hashes=True,
                )
            accounting = self._validate_capacity(
                entries, check_free_space=check_free_space
            )
            self._ensure_source_lock(entries)
            body = {
                "version": CATALOG_VERSION,
                "epoch": 0,
                "parent": None,
                "created_at_utc": _iso(now_dt),
                "data_root": str(self.data_root),
                "contract": {
                    "migration": "fresh_canonical_staging_bootstrap_v1",
                    "pipeline_config_sha256": pipeline_hash,
                    "source_hashes": pinned_sources,
                },
                "files": entries,
                "additions": [int(entry["shard_id"]) for entry in entries],
                "retirements": [],
                "accounting": accounting,
                "assignments": self._assignments(entries),
                "capacity_policy": self.policy.to_dict(),
            }
            self._validate_catalog_payload(body)
            catalog_path, digest = self._write_catalog(body)
            pointer = self._pointer_for(catalog_path, digest, now_dt)
            atomic_write_json(self.current_path, pointer)
            next_id = 1 + max(int(entry["shard_id"]) for entry in entries)
            current_next = 0
            if self.next_shard_id_path.exists():
                current_next = int(
                    _read_json(self.next_shard_id_path).get("next_shard_id", 0)
                )
            atomic_write_json(
                self.next_shard_id_path,
                {
                    "version": "v12_canonical_next_shard_id_v1",
                    "next_shard_id": max(next_id, current_next),
                },
            )
            return pointer, body

    def _frozen_manifest_metadata(self) -> dict[str, dict[str, Any]]:
        """Index current Exp122 manifests by data-root-relative NPZ path."""

        result: dict[str, dict[str, Any]] = {}
        for prior in self.priors:
            prior_root = self.data_root / prior
            for manifest_path in sorted(prior_root.glob("part_*/manifest.json")):
                manifest = _read_json(manifest_path)
                for raw in manifest.get("files", []):
                    relative_name = Path(str(raw.get("path", "")))
                    if relative_name.is_absolute() or ".." in relative_name.parts:
                        raise CatalogError(f"unsafe Exp122 manifest path: {relative_name}")
                    absolute = (manifest_path.parent / relative_name).resolve()
                    if not absolute.is_relative_to(prior_root.resolve()):
                        raise CatalogError(f"Exp122 manifest path escapes prior root: {absolute}")
                    relative = absolute.relative_to(self.data_root).as_posix()
                    if relative in result:
                        raise CatalogError(f"duplicate Exp122 manifest entry: {relative}")
                    result[relative] = dict(raw)
        return result

    def import_frozen_inventory(
        self,
        inventory_path: str | Path,
        *,
        expected_sha256: str | None = None,
        pipeline_config_sha256: str | None = None,
        source_hashes: Mapping[str, str] | None = None,
        now: datetime | str | None = None,
        verify_files: bool = True,
        check_free_space: bool = True,
        failpoint: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Create epoch 0 from the current Exp122 FROZEN_INVENTORY.json."""

        now_dt = _as_utc(now)
        if pipeline_config_sha256 is None:
            default_config = Path(__file__).resolve().with_name(
                "exp123_rolling_pipeline_config.json"
            )
            if not default_config.is_file():
                raise CatalogError(
                    "pipeline_config_sha256 is required when the canonical config is absent"
                )
            observed_default = sha256_file(default_config)
            if observed_default != DEFAULT_PIPELINE_CONFIG_SHA256:
                raise CatalogError(
                    "canonical Exp123 pipeline config hash changed: "
                    f"{observed_default} != {DEFAULT_PIPELINE_CONFIG_SHA256}"
                )
            pipeline_config_sha256 = observed_default
        pipeline_config_sha256 = _validate_sha256(
            pipeline_config_sha256, "Exp123 pipeline config hash"
        )
        inventory_path = Path(inventory_path).expanduser().resolve()
        if inventory_path.parent != self.data_root:
            raise CatalogError(
                f"frozen inventory parent must equal data_root: {inventory_path.parent} != {self.data_root}"
            )
        if not inventory_path.is_file():
            raise FileNotFoundError(inventory_path)
        inventory_sha = sha256_file(inventory_path)
        if expected_sha256 is not None and inventory_sha != _validate_sha256(
            expected_sha256, "expected frozen inventory hash"
        ):
            raise CatalogError("frozen inventory hash mismatch")
        frozen = _read_json(inventory_path)
        raw_entries = frozen.get("files")
        if not isinstance(raw_entries, list) or not raw_entries:
            raise CatalogError("frozen inventory has no files")

        with self._lock("publisher", nonblocking=True):
            if self.current_path.exists():
                raise CatalogError("refusing frozen import because CURRENT already exists")
            manifest_metadata = self._frozen_manifest_metadata()
            enriched: list[dict[str, Any]] = []
            for frozen_entry in sorted(raw_entries, key=lambda item: str(item.get("path", ""))):
                relative = self._safe_relative_path(
                    frozen_entry.get("relative_path", frozen_entry.get("path", ""))
                )
                manifest_entry = manifest_metadata.get(relative)
                if manifest_entry is None:
                    raise CatalogError(
                        f"frozen inventory path lacks manifest metadata needed by Exp123: {relative}"
                    )
                merged = dict(manifest_entry)
                merged.update(dict(frozen_entry))
                merged["relative_path"] = relative
                merged["prior"] = Path(relative).parts[0]
                merged["pipeline_config_sha256"] = pipeline_config_sha256
                merged["source_hashes"] = dict(source_hashes or {})
                enriched.append(merged)
            entries = [
                self._normalise_entry(entry, frozen_import=True, assigned_id=index)
                for index, entry in enumerate(enriched)
            ]
            self._validate_unique_entries(entries)
            if verify_files:
                self.verify_catalog_files(
                    {"data_root": str(self.data_root), "files": entries}, hashes=True
                )
            accounting = self._validate_capacity(
                entries,
                check_free_space=check_free_space,
            )
            body = {
                "version": CATALOG_VERSION,
                "epoch": 0,
                "parent": None,
                "created_at_utc": _iso(now_dt),
                "data_root": str(self.data_root),
                "contract": {
                    "migration": "exp122_frozen_inventory_to_exp123_semantic_v1",
                    "frozen_inventory_sha256": inventory_sha,
                    "pipeline_config_sha256": pipeline_config_sha256,
                    "source_hashes": dict(source_hashes or {}),
                },
                "files": entries,
                "additions": [int(entry["shard_id"]) for entry in entries],
                "retirements": [],
                "accounting": accounting,
                "assignments": self._assignments(entries),
                "capacity_policy": self.policy.to_dict(),
            }
            self._validate_catalog_payload(body)
            catalog_path, digest = self._write_catalog(body)
            if failpoint == "after_catalog_write":
                raise RuntimeError("injected failure after catalog write")
            pointer = self._pointer_for(catalog_path, digest, now_dt)
            atomic_write_json(self.current_path, pointer)
            if failpoint == "after_pointer_write":
                raise RuntimeError("injected failure after pointer write")
            atomic_write_json(
                self.next_shard_id_path,
                {"version": "exp123_next_shard_id_v1", "next_shard_id": len(entries)},
            )
            return pointer, body

    def _chain_from_current(self) -> list[tuple[str, dict[str, Any]]]:
        pointer, catalog = self.load_current(verify_files=False)
        chain: list[tuple[str, dict[str, Any]]] = []
        digest = str(pointer["catalog_sha256"])
        seen: set[str] = set()
        while True:
            if digest in seen:
                raise CatalogError("catalog parent cycle detected")
            seen.add(digest)
            chain.append((digest, catalog))
            parent = catalog.get("parent")
            if parent is None:
                break
            digest = _validate_sha256(parent.get("catalog_sha256"), "parent hash")
            filename = str(parent.get("catalog_file", ""))
            catalog = self._load_catalog_file(filename, digest)
            if int(parent.get("epoch", -1)) != int(catalog["epoch"]):
                raise CatalogError("parent epoch mismatch")
        return chain

    def _historical_identity_sets(self) -> tuple[set[int], set[str], set[int]]:
        ids: set[int] = set()
        hashes: set[str] = set()
        seeds: set[int] = set()
        for _digest, catalog in self._chain_from_current():
            for entry in catalog["files"]:
                ids.add(int(entry["shard_id"]))
                hashes.add(str(entry["sha256"]))
                if entry.get("seed") is not None:
                    seeds.add(int(entry["seed"]))
        return ids, hashes, seeds

    def normalise_staged_entry(
        self,
        entry: Mapping[str, Any],
        *,
        verify_file: bool = True,
    ) -> dict[str, Any]:
        """Validate/canonicalise one producer inbox entry without publishing it."""

        result = self._normalise_entry(entry)
        if verify_file:
            self.verify_catalog_files(
                {"data_root": str(self.data_root), "files": [result]}, hashes=True
            )
        return result

    def _ensure_source_lock(self, additions: Sequence[Mapping[str, Any]]) -> None:
        if not additions:
            return
        pipeline_hashes = {entry.get("pipeline_config_sha256") for entry in additions}
        source_sets = {
            json.dumps(entry.get("source_hashes", {}), sort_keys=True, separators=(",", ":"))
            for entry in additions
        }
        if len(pipeline_hashes) != 1 or None in pipeline_hashes:
            raise CatalogError("one publication contains multiple/missing pipeline config hashes")
        if len(source_sets) != 1 or source_sets == {"{}"}:
            raise CatalogError("one publication contains multiple/missing source hash sets")
        payload = {
            "version": "exp123_source_lock_v1",
            "pipeline_config_sha256": next(iter(pipeline_hashes)),
            "source_hashes": json.loads(next(iter(source_sets))),
        }
        if self.source_lock_path.exists():
            if _read_json(self.source_lock_path) != payload:
                raise CatalogError("producer source lock drift")
        else:
            atomic_write_json(self.source_lock_path, payload, overwrite=False)

    def publish_transition(
        self,
        additions: Iterable[Mapping[str, Any]],
        retirements: Iterable[int | str],
        *,
        expected_parent_sha256: str | None = None,
        now: datetime | str | None = None,
        observed_project_bytes: int | None = None,
        verify_files: bool = True,
        check_free_space: bool = True,
        failpoint: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Publish one explicit add/retire transition and atomically advance CURRENT."""

        now_dt = _as_utc(now)
        with self._lock("publisher", nonblocking=True):
            pointer, parent = self.load_current(verify_files=False)
            parent_hash = str(pointer["catalog_sha256"])
            if expected_parent_sha256 is not None and parent_hash != _validate_sha256(
                expected_parent_sha256, "expected parent hash"
            ):
                raise CatalogError("CURRENT changed since transition planning")
            next_not_before = _as_utc(pointer["next_publish_not_before_utc"])
            if now_dt < next_not_before:
                raise PublishDeferred(
                    f"hourly publication is not due until {pointer['next_publish_not_before_utc']}"
                )

            parent_by_id = {int(entry["shard_id"]): entry for entry in parent["files"]}
            parent_by_path = {str(entry["relative_path"]): entry for entry in parent["files"]}
            retire_ids: set[int] = set()
            for value in retirements:
                if isinstance(value, int) and not isinstance(value, bool):
                    shard_id = value
                else:
                    relative = self._safe_relative_path(value)
                    if relative not in parent_by_path:
                        raise CatalogError(f"retirement is not in parent catalog: {relative}")
                    shard_id = int(parent_by_path[relative]["shard_id"])
                if shard_id not in parent_by_id:
                    raise CatalogError(f"retirement shard_id is not in parent: {shard_id}")
                if shard_id in retire_ids:
                    raise CatalogError(f"duplicate retirement: {shard_id}")
                retire_ids.add(shard_id)

            additions_normalised = [self._normalise_entry(entry) for entry in additions]
            if not additions_normalised and not retire_ids:
                raise PublishDeferred("no validated catalog delta")
            self._validate_unique_entries(additions_normalised)
            historical_ids, historical_hashes, historical_seeds = self._historical_identity_sets()
            for entry in additions_normalised:
                if int(entry["shard_id"]) in historical_ids:
                    raise CatalogError(f"shard_id was already used: {entry['shard_id']}")
                if str(entry["sha256"]) in historical_hashes:
                    raise CatalogError(f"content hash was already used: {entry['sha256']}")
                if entry.get("seed") is not None and int(entry["seed"]) in historical_seeds:
                    raise CatalogError(f"seed was already used: {entry['seed']}")
                if str(entry["relative_path"]) in parent_by_path:
                    raise CatalogError(
                        f"addition path already exists in parent: {entry['relative_path']}"
                    )
                expected_pipeline = parent.get("contract", {}).get("pipeline_config_sha256")
                observed_pipeline = entry.get("pipeline_config_sha256")
                if expected_pipeline and observed_pipeline != expected_pipeline:
                    raise CatalogError("addition pipeline_config_sha256 drift")
                expected_sources = parent.get("contract", {}).get("source_hashes", {})
                observed_sources = entry.get("source_hashes", {})
                if expected_sources and observed_sources != expected_sources:
                    raise CatalogError("addition source_hashes drift")

            survivors = [
                dict(entry)
                for entry in parent["files"]
                if int(entry["shard_id"]) not in retire_ids
            ]
            candidate = sorted(survivors + additions_normalised, key=self._entry_sort_key)
            self._validate_unique_entries(candidate)
            if verify_files:
                self.verify_catalog_files(
                    {"data_root": str(self.data_root), "files": additions_normalised},
                    hashes=True,
                )
            accounting = self._validate_capacity(
                candidate,
                observed_project_bytes=observed_project_bytes,
                check_free_space=check_free_space,
            )
            self._ensure_source_lock(additions_normalised)
            child_epoch = int(parent["epoch"]) + 1
            body = {
                "version": CATALOG_VERSION,
                "epoch": child_epoch,
                "parent": {
                    "epoch": int(parent["epoch"]),
                    "catalog_file": str(pointer["catalog_file"]),
                    "catalog_sha256": parent_hash,
                },
                "created_at_utc": _iso(now_dt),
                "data_root": str(self.data_root),
                "contract": dict(parent["contract"]),
                "files": candidate,
                "additions": sorted(int(entry["shard_id"]) for entry in additions_normalised),
                "retirements": sorted(retire_ids),
                "accounting": accounting,
                "assignments": self._assignments(candidate),
                "capacity_policy": self.policy.to_dict(),
            }
            self._validate_transition(parent, body)
            catalog_path, digest = self._write_catalog(body)
            if failpoint == "after_catalog_write":
                raise RuntimeError("injected failure after catalog write")
            new_pointer = self._pointer_for(catalog_path, digest, now_dt)
            atomic_write_json(self.current_path, new_pointer)
            if failpoint == "after_pointer_write":
                raise RuntimeError("injected failure after pointer write")
            return new_pointer, body

    def _validate_transition(
        self,
        parent: Mapping[str, Any],
        child: Mapping[str, Any],
    ) -> None:
        self._validate_catalog_payload(child)
        if int(child["epoch"]) != int(parent["epoch"]) + 1:
            raise CatalogError("child epoch is not parent epoch + 1")
        if child.get("contract") != parent.get("contract"):
            raise CatalogError("catalog contract changed within one semantic version")
        parent_by_id = {int(entry["shard_id"]): entry for entry in parent["files"]}
        child_by_id = {int(entry["shard_id"]): entry for entry in child["files"]}
        additions = sorted(set(child_by_id) - set(parent_by_id))
        retirements = sorted(set(parent_by_id) - set(child_by_id))
        if additions != child.get("additions"):
            raise CatalogError("declared additions do not equal the catalog delta")
        if retirements != child.get("retirements"):
            raise CatalogError("declared retirements do not equal the catalog delta")
        mutated = [
            shard_id
            for shard_id in set(parent_by_id) & set(child_by_id)
            if parent_by_id[shard_id] != child_by_id[shard_id]
        ]
        if mutated:
            raise CatalogError(f"surviving shard metadata mutated: {mutated[:5]}")

    @staticmethod
    def _safe_token(value: str, label: str) -> str:
        value = str(value)
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value):
            raise CatalogError(f"unsafe {label}: {value!r}")
        return value

    def _catalog_epoch_from_current_or_chain(
        self, catalog_sha256: str
    ) -> int | None:
        """Resolve CURRENT from its atomic pointer before parsing history."""

        if not self.current_path.is_file():
            raise CatalogError(f"CURRENT is missing: {self.current_path}")
        pointer = _read_json(self.current_path)
        if pointer.get("version") != POINTER_VERSION:
            raise CatalogError(f"unsupported CURRENT version: {pointer.get('version')!r}")
        current_sha256 = _validate_sha256(
            pointer.get("catalog_sha256"), "CURRENT catalog hash"
        )
        epoch = pointer.get("epoch")
        if not isinstance(epoch, int) or epoch < 0:
            raise CatalogError(f"invalid CURRENT epoch: {epoch!r}")
        expected_file = f"CATALOG_EPOCH_{epoch:08d}_{current_sha256[:16]}.json"
        if pointer.get("catalog_file") != expected_file:
            raise CatalogError("CURRENT catalog filename is not canonical")
        if current_sha256 == catalog_sha256:
            return epoch
        historical = dict(self._chain_from_current()).get(catalog_sha256)
        return None if historical is None else int(historical["epoch"])

    def acquire_lease(
        self,
        trainer_id: str,
        catalog_sha256: str,
        *,
        now: datetime | str | None = None,
    ) -> Path:
        trainer_id = self._safe_token(trainer_id, "trainer_id")
        catalog_sha256 = _validate_sha256(catalog_sha256, "lease catalog hash")
        catalog_epoch = self._catalog_epoch_from_current_or_chain(catalog_sha256)
        if catalog_epoch is None:
            raise CatalogError("cannot lease a catalog outside the adopted chain")
        payload = {
            "version": LEASE_VERSION,
            "trainer_id": trainer_id,
            "catalog_sha256": catalog_sha256,
            "catalog_epoch": catalog_epoch,
            "acquired_at_utc": _iso(_as_utc(now)),
            "released_at_utc": None,
        }
        path = self.lease_dir / f"{trainer_id}_{catalog_sha256[:16]}.json"
        if path.exists():
            old = _read_json(path)
            if old.get("released_at_utc") is None:
                return path
        atomic_write_json(path, payload)
        return path

    def release_lease(
        self,
        trainer_id: str,
        *,
        expected_catalog_sha256: str,
        now: datetime | str | None = None,
    ) -> None:
        trainer_id = self._safe_token(trainer_id, "trainer_id")
        expected = _validate_sha256(expected_catalog_sha256, "expected lease hash")
        path = self.lease_dir / f"{trainer_id}_{expected[:16]}.json"
        if not path.is_file():
            raise CatalogError(f"lease is missing: {path}")
        payload = _read_json(path)
        if payload.get("catalog_sha256") != expected:
            raise CatalogError("lease hash changed before release")
        if payload.get("released_at_utc") is None:
            payload["released_at_utc"] = _iso(_as_utc(now))
            atomic_write_json(path, payload)

    def write_ack(
        self,
        trainer_id: str,
        rank: int,
        world_size: int,
        catalog_sha256: str,
        *,
        now: datetime | str | None = None,
    ) -> Path:
        trainer_id = self._safe_token(trainer_id, "trainer_id")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("invalid rank/world_size")
        if world_size != self.policy.world_size:
            raise CatalogError(
                f"ACK world size differs from catalog policy: {world_size} != {self.policy.world_size}"
            )
        catalog_sha256 = _validate_sha256(catalog_sha256, "ACK catalog hash")
        catalog_epoch = self._catalog_epoch_from_current_or_chain(catalog_sha256)
        if catalog_epoch is None:
            raise CatalogError("cannot ACK a catalog outside the adopted chain")
        ack_root = self.ack_dir / f"epoch_{catalog_epoch:08d}_{catalog_sha256[:16]}" / trainer_id
        payload = {
            "version": ACK_VERSION,
            "trainer_id": trainer_id,
            "rank": int(rank),
            "world_size": int(world_size),
            "catalog_epoch": catalog_epoch,
            "catalog_sha256": catalog_sha256,
            "acked_at_utc": _iso(_as_utc(now)),
        }
        path = ack_root / f"rank_{rank:04d}.json"
        if path.exists():
            existing = _read_json(path)
            for key in ("trainer_id", "rank", "world_size", "catalog_epoch", "catalog_sha256"):
                if existing.get(key) != payload[key]:
                    raise CatalogError(f"conflicting ACK already exists: {path}")
            return path
        atomic_write_json(path, payload, overwrite=False)
        return path

    def all_ranks_acked(
        self,
        trainer_id: str,
        catalog_sha256: str,
        world_size: int,
    ) -> bool:
        trainer_id = self._safe_token(trainer_id, "trainer_id")
        catalog_sha256 = _validate_sha256(catalog_sha256, "ACK catalog hash")
        catalog_epoch = self._catalog_epoch_from_current_or_chain(catalog_sha256)
        if catalog_epoch is None:
            return False
        ack_root = self.ack_dir / f"epoch_{catalog_epoch:08d}_{catalog_sha256[:16]}" / trainer_id
        for rank in range(world_size):
            path = ack_root / f"rank_{rank:04d}.json"
            if not path.is_file():
                return False
            payload = _read_json(path)
            expected = {
                "version": ACK_VERSION,
                "trainer_id": trainer_id,
                "rank": rank,
                "world_size": world_size,
                "catalog_epoch": catalog_epoch,
                "catalog_sha256": catalog_sha256,
            }
            if any(payload.get(key) != value for key, value in expected.items()):
                return False
        return True

    def register_checkpoint_pin(
        self,
        checkpoint_id: str,
        catalog_sha256: str,
        *,
        checkpoint_sha256: str,
        resumable: bool = True,
        reload_verified: bool = True,
        now: datetime | str | None = None,
    ) -> Path:
        checkpoint_id = self._safe_token(checkpoint_id, "checkpoint_id")
        catalog_sha256 = _validate_sha256(catalog_sha256, "checkpoint catalog hash")
        checkpoint_sha256 = _validate_sha256(checkpoint_sha256, "checkpoint file hash")
        catalog_epoch = self._catalog_epoch_from_current_or_chain(catalog_sha256)
        if catalog_epoch is None:
            raise CatalogError("checkpoint pins a catalog outside the adopted chain")
        payload = {
            "version": CHECKPOINT_PIN_VERSION,
            "checkpoint_id": checkpoint_id,
            "checkpoint_sha256": checkpoint_sha256,
            "catalog_epoch": catalog_epoch,
            "catalog_sha256": catalog_sha256,
            "resumable": bool(resumable),
            "reload_verified": bool(reload_verified),
            "created_at_utc": _iso(_as_utc(now)),
        }
        if resumable and not reload_verified:
            raise CatalogError("a resumable checkpoint must be reload verified")
        path = self.checkpoint_dir / f"{checkpoint_id}.json"
        if path.exists():
            existing = _read_json(path)
            if existing != payload:
                raise CatalogError(f"checkpoint pin is immutable: {path}")
            return path
        atomic_write_json(path, payload, overwrite=False)
        return path

    def mark_checkpoint_nonresumable(self, checkpoint_id: str) -> None:
        checkpoint_id = self._safe_token(checkpoint_id, "checkpoint_id")
        path = self.checkpoint_dir / f"{checkpoint_id}.json"
        if not path.is_file():
            raise CatalogError(f"checkpoint pin is missing: {path}")
        payload = _read_json(path)
        payload["resumable"] = False
        atomic_write_json(path, payload)

    def _protected_catalog_hashes(self) -> set[str]:
        pointer, _catalog = self.load_current(verify_files=False)
        protected = {str(pointer["catalog_sha256"])}
        for path in self.lease_dir.glob("*.json"):
            payload = _read_json(path)
            if payload.get("version") == LEASE_VERSION and payload.get("released_at_utc") is None:
                protected.add(_validate_sha256(payload.get("catalog_sha256"), "lease hash"))
        for path in self.checkpoint_dir.glob("*.json"):
            payload = _read_json(path)
            if (
                payload.get("version") == CHECKPOINT_PIN_VERSION
                and payload.get("resumable") is True
                and payload.get("reload_verified") is True
            ):
                protected.add(_validate_sha256(payload.get("catalog_sha256"), "checkpoint hash"))
        return protected

    def _has_verified_resumable_checkpoint(self, catalog_sha256: str) -> bool:
        for path in self.checkpoint_dir.glob("*.json"):
            payload = _read_json(path)
            if (
                payload.get("version") == CHECKPOINT_PIN_VERSION
                and payload.get("catalog_sha256") == catalog_sha256
                and payload.get("resumable") is True
                and payload.get("reload_verified") is True
            ):
                return True
        return False

    def _retirement_records(self) -> dict[int, dict[str, Any]]:
        chain = list(reversed(self._chain_from_current()))
        records: dict[int, dict[str, Any]] = {}
        previous: dict[str, Any] | None = None
        for digest, catalog in chain:
            if previous is not None:
                previous_by_id = {int(entry["shard_id"]): entry for entry in previous["files"]}
                for shard_id in catalog["retirements"]:
                    records[int(shard_id)] = {
                        "entry": previous_by_id[int(shard_id)],
                        "retired_at_epoch": int(catalog["epoch"]),
                        "retired_at_utc": str(catalog["created_at_utc"]),
                        "retiring_catalog_sha256": digest,
                    }
            previous = catalog
        return records

    def create_gc_tombstone(
        self,
        trainer_id: str,
        *,
        world_size: int,
        now: datetime | str | None = None,
    ) -> Path | None:
        """Create a durable deletion plan; never unlinks an NPZ."""

        now_dt = _as_utc(now)
        with self._lock("gc"):
            pointer, current = self.load_current(verify_files=False)
            current_hash = str(pointer["catalog_sha256"])
            if not self.all_ranks_acked(trainer_id, current_hash, world_size):
                raise GCBlocked("current catalog lacks a complete matching rank ACK set")
            if not self._has_verified_resumable_checkpoint(current_hash):
                raise GCBlocked("CURRENT lacks a reload-verified resumable checkpoint")
            chain = dict(self._chain_from_current())
            protected_hashes = self._protected_catalog_hashes()
            unknown = protected_hashes - set(chain)
            if unknown:
                raise GCBlocked(f"protected catalog is outside current chain: {sorted(unknown)}")
            protected_ids: set[int] = set()
            for digest in protected_hashes:
                protected_ids.update(int(entry["shard_id"]) for entry in chain[digest]["files"])
            current_ids = {int(entry["shard_id"]) for entry in current["files"]}
            records = self._retirement_records()
            candidates: list[dict[str, Any]] = []
            for shard_id, record in sorted(records.items()):
                if shard_id in current_ids or shard_id in protected_ids:
                    continue
                age_epochs = int(current["epoch"]) - int(record["retired_at_epoch"])
                age_seconds = (now_dt - _as_utc(record["retired_at_utc"])).total_seconds()
                if age_epochs < self.policy.retire_grace_epochs:
                    continue
                if age_seconds < self.policy.retire_grace_seconds:
                    continue
                candidates.append(
                    {
                        "shard_id": shard_id,
                        "relative_path": record["entry"]["relative_path"],
                        "bytes": int(record["entry"]["bytes"]),
                        "sha256": record["entry"]["sha256"],
                        "retired_at_epoch": int(record["retired_at_epoch"]),
                        "retired_at_utc": record["retired_at_utc"],
                    }
                )
            if not candidates:
                return None
            tombstone_id = hashlib.sha256(
                _json_bytes(
                    {
                        "current": current_hash,
                        "created_at": _iso(now_dt),
                        "candidates": candidates,
                    }
                )
            ).hexdigest()[:24]
            payload = {
                "version": TOMBSTONE_VERSION,
                "tombstone_id": tombstone_id,
                "created_at_utc": _iso(now_dt),
                "execute_not_before_utc": _iso(
                    now_dt + timedelta(seconds=self.policy.tombstone_grace_seconds)
                ),
                "proof": {
                    "trainer_id": self._safe_token(trainer_id, "trainer_id"),
                    "world_size": world_size,
                    "current_epoch": int(current["epoch"]),
                    "current_catalog_sha256": current_hash,
                    "protected_catalog_sha256": sorted(protected_hashes),
                },
                "files": candidates,
                "status": "pending",
                "completed_at_utc": None,
            }
            path = self.tombstone_dir / f"TOMBSTONE_{tombstone_id}.json"
            if path.exists():
                existing = _read_json(path)
                if existing != payload:
                    raise CatalogError(f"tombstone ID collision: {path}")
                return path
            atomic_write_json(path, payload, overwrite=False)
            return path

    def execute_gc_tombstone(
        self,
        tombstone_path: str | Path,
        *,
        now: datetime | str | None = None,
        fail_after_unlinks: int | None = None,
    ) -> list[str]:
        """Re-prove safety and idempotently unlink files named by a tombstone."""

        now_dt = _as_utc(now)
        path = Path(tombstone_path).expanduser().resolve()
        if not path.is_relative_to(self.tombstone_dir.resolve()) or path.parent != self.tombstone_dir.resolve():
            raise CatalogError(f"unsafe tombstone path: {path}")
        with self._lock("gc"):
            payload = _read_json(path)
            if payload.get("version") != TOMBSTONE_VERSION:
                raise CatalogError("unsupported tombstone version")
            if payload.get("status") == "complete":
                return []
            if now_dt < _as_utc(payload["execute_not_before_utc"]):
                raise GCBlocked("tombstone grace period has not elapsed")
            proof = payload.get("proof", {})
            trainer_id = str(proof.get("trainer_id", ""))
            world_size = int(proof.get("world_size", -1))
            pointer, current = self.load_current(verify_files=False)
            current_hash = str(pointer["catalog_sha256"])
            if not self.all_ranks_acked(trainer_id, current_hash, world_size):
                raise GCBlocked("latest CURRENT lacks complete ACKs at unlink time")
            if not self._has_verified_resumable_checkpoint(current_hash):
                raise GCBlocked("latest CURRENT lacks a verified resumable checkpoint")
            chain = dict(self._chain_from_current())
            protected_hashes = self._protected_catalog_hashes()
            protected_ids: set[int] = set()
            for digest in protected_hashes:
                if digest not in chain:
                    raise GCBlocked("a protected catalog is outside the current chain")
                protected_ids.update(int(entry["shard_id"]) for entry in chain[digest]["files"])
            current_ids = {int(entry["shard_id"]) for entry in current["files"]}
            removed: list[str] = []
            for entry in payload.get("files", []):
                shard_id = int(entry["shard_id"])
                if shard_id in current_ids or shard_id in protected_ids:
                    raise GCBlocked(f"tombstoned shard became protected: {shard_id}")
                target = self.resolve_entry_path(entry)
                if target.exists():
                    if not target.is_file() or target.stat().st_size != int(entry["bytes"]):
                        raise GCBlocked(f"tombstoned file changed size/type: {target}")
                    observed = sha256_file(target)
                    if observed != entry["sha256"]:
                        raise GCBlocked(f"tombstoned file changed hash: {target}")
                    target.unlink()
                    dir_fd = os.open(target.parent, os.O_RDONLY)
                    try:
                        os.fsync(dir_fd)
                    finally:
                        os.close(dir_fd)
                    removed.append(str(target))
                    if fail_after_unlinks is not None and len(removed) >= fail_after_unlinks:
                        raise RuntimeError("injected failure after physical unlink")
            payload["status"] = "complete"
            payload["completed_at_utc"] = _iso(now_dt)
            atomic_write_json(path, payload)
            return removed


__all__ = [
    "ACK_VERSION",
    "CATALOG_VERSION",
    "CatalogError",
    "CatalogStore",
    "CapacityPolicy",
    "EXPECTED_PRIORS",
    "DEFAULT_PIPELINE_CONFIG_SHA256",
    "GCBlocked",
    "POINTER_VERSION",
    "PublishDeferred",
    "SHARD_VERSION",
    "atomic_write_json",
    "sha256_file",
]
