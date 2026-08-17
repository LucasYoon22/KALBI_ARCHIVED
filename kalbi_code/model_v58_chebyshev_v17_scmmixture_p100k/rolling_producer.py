#!/usr/bin/env python3
"""Crash-safe CPU producer for the 50:50 KALBI paired-SCM corpus.

This program wraps one pinned TabICLv2 graph-SCM X/y generator.  Each
invocation owns exactly one configured worker.  The
worker reserves its next monotonically increasing batch id in a durable state
file *before* generation, runs the legacy generator for exactly one batch in a
private scratch directory, validates the resulting NPZ, and publishes:

  1. an immutable content-addressed NPZ under ``staging/shards``;
  2. an immutable SHA-bearing shard manifest under ``staging/inbox``.

The durable worker state is independent of both the generator's deletable
manifest and Exp123 catalog/GC manifests.  A crash may leave a gap or an
unreferenced content-addressed shard, but it cannot reuse a dataset id/seed or
make a partially written NPZ visible to the catalog publisher.

Production is CPU-only by construction: CUDA is hidden, correlation checks use
CPU, BLAS/PyTorch thread counts are fixed, and Linux ``taskset`` pins every
generator process to the exact cores declared in the immutable config.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import errno
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .rolling_catalog import CatalogStore, SHARD_VERSION


CONFIG_SCHEMA_VERSION = "v14.scmmixture.rolling.producer.config.v1"
PIPELINE_VERSION = "v14_kalbi_tabpfnv1_tabiclv2_scm_xy_50_50_rolling100gb_v1"
SOURCE_VERSION = "v14_scmmixture_cpu_rolling_producer_v1"
STATE_SCHEMA_VERSION = "v14.scmmixture.rolling.producer.worker-state.v1"
SHARD_MANIFEST_SCHEMA_VERSION = SHARD_VERSION
ROOT_LOCK_SCHEMA_VERSION = "v14.scmmixture.rolling.producer.root-lock.v1"

PRIOR_KEYS = ("tabpfn_v1", "tabicl_v2")
EXP122_BASE_SEEDS = frozenset(
    (1_220_100_001, 1_220_100_002, 1_220_200_001, 1_220_200_002)
)
EXP123_DEFAULT_BASE_SEEDS = {
    "tabpfn_00": 1_230_100_001,
    "tabpfn_01": 1_230_100_002,
    "tabicl_00": 1_230_200_001,
    "tabicl_01": 1_230_200_002,
}
BATCH_SEED_MODULUS = 2**32 - 1
BATCH_SEED_MULTIPLIER = 1_000_003
DEFAULT_ACTIVE_TARGET_BYTES = 100_000_000_000
DEFAULT_BOOTSTRAP_STAGING_HIGH_WATER_BYTES = 102_000_000_000
DEFAULT_STAGING_HIGH_WATER_BYTES = 4_000_000_000
DEFAULT_RETIREMENT_GRACE_BYTES = 10_000_000_000
DEFAULT_PROJECT_HARD_CAP_BYTES = 130_000_000_000
DEFAULT_MIN_FREE_BYTES = 300_000_000_000
DEFAULT_MAX_SHARD_BYTES = 64_000_000
DEFAULT_IDLE_SECONDS = 60.0
EXPECTED_SHAPE_SAMPLER = "smallN10pct_coupled_lognormal_median_N2000_P10000_maxN40K_maxP200K_v3"
EXPECTED_CORRELATION_CERTIFICATE = "exact incremental all-pair Pearson after float16 roundtrip"
EXPECTED_MLP_NOISE_CONTRACT = (
    "parents_zscored_plus_gaussian16_zscored_then_scaled; "
    "output_signal_zscored_plus_scaled_noise_v2"
)
EXPECTED_MLP_ACTIVATIONS = frozenset(("gelu", "tanh", "relu", "sigmoid", "sin_pi"))
PRIOR_CONTRACTS = {
    "tabpfn_v1": {
        "generator": "generate_tabpfn_v1_causal_scm_mlp512_groupdrop095_scm_y_v2.py",
        "prior_name": "official_tabpfn_v1_causal_scm_xy_groupdrop095_then_random_mlp512_v1",
        "dataset_contract": "tabpfn_v1_causal_scm_xy_groupdrop095_v2",
        "data_policy": "tabpfn_v1_causal_scm_xy_exact_corr_lt_0p95_v1",
        "target_contract": "official_tabpfn_v1_causal_scm_y_center_scale_clip10_rescale_v1",
        "official_core_prior": "PriorLabs/TabPFN v1.0.0 causal SCM regression branch",
        "target_source": "official TabPFN v1 causal MLP-SCM continuous response",
        "direct_mode": "direct_tabpfn_v1_causal_scm",
        "highp_mode": "tabpfn_v1_causal_scm_plus_random_mlp512",
        "commit": "a9298ae0fe2ada199a113dffa4afd53c7ba0edce",
        "checkpoint_sha256": "3c9aadaeddbf51462af8c0ee4b3ca3c697890f77e92318abbb0821b75261c392",
    },
    "tabicl_v2": {
        "generator": "generate_tabiclv2_graph_scm_mlp512_groupdrop095_scm_y_v5.py",
        "prior_name": "official_tabiclv2_graph_scm_xy_groupdrop095_then_random_mlp512_v1",
        "dataset_contract": "tabiclv2_graph_scm_xy_groupdrop095_v5",
        "data_policy": "tabiclv2_graph_scm_xy_exact_corr_lt_0p95_v1",
        "target_contract": "official_tabiclv2_graph_scm_y_center_scale_clip10_rescale_v1",
        "official_core_prior": "TabICLv2 PriorDataset prior_type=graph_scm",
        "target_source": "official TabICLv2 PriorDataset graph_scm response",
        "direct_mode": "direct_tabiclv2",
        "highp_mode": "tabiclv2_plus_random_mlp512",
        "commit": "46b91961db4f8873dd049ec09990698a435e1e29",
    },
}
RESERVED_GENERATOR_ARGS = frozenset(
    (
        "--out-dir",
        "--num-batches",
        "--batch-size",
        "--start-index",
        "--seed",
        "--threads",
        "--corr-device",
        "--target-bytes",
        "--stop-file",
    )
)


class ProducerError(RuntimeError):
    """Base class for fail-closed producer errors."""


class Backpressure(ProducerError):
    """The producer must wait without reserving a new dataset id."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_dir(path: str | Path) -> None:
    fd = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(payload))


def publish_immutable_bytes(path: str | Path, payload: bytes) -> None:
    """Create ``path`` once; an identical retry is idempotent."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        current = path.read_bytes()
        if current != payload:
            raise ProducerError(f"immutable artifact changed: {path}")
        return
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(tmp, path)
    except FileExistsError:
        current = path.read_bytes()
        if current != payload:
            raise ProducerError(f"immutable artifact publication raced with different bytes: {path}")
    finally:
        tmp.unlink(missing_ok=True)
    fsync_dir(path.parent)


def publish_immutable_file(source: str | Path, destination: str | Path, expected_sha256: str) -> None:
    """Hard-link a complete file into the content store without overwrite."""

    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if sha256_file(source) != expected_sha256:
        raise ProducerError(f"source SHA256 changed before publication: {source}")
    if destination.exists():
        if sha256_file(destination) != expected_sha256:
            raise ProducerError(f"content-addressed destination is corrupt: {destination}")
        return
    try:
        if source.stat().st_dev != destination.parent.stat().st_dev:
            raise ProducerError("scratch and staging must be on the same filesystem for atomic publication")
        os.link(source, destination)
    except FileExistsError:
        if sha256_file(destination) != expected_sha256:
            raise ProducerError(f"content-addressed publication collision: {destination}")
    fsync_dir(destination.parent)


def _require_int(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProducerError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _canonical_source_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": config["schema_version"],
        "pipeline_version": config["pipeline_version"],
        "source_version": config["source_version"],
        "project_dir": config["project_dir"],
        "control_root": config["control_root"],
        "pipeline_config_path": config["pipeline_config_path"],
        "pipeline_config_sha256": config["pipeline_config_sha256"],
        "generator_sources": config["generator_sources"],
        "source_files": config.get("source_files", []),
        "generator_common_args": config.get("generator_common_args", []),
        "workers": config["workers"],
        "bootstrap_staging_high_water_bytes": config[
            "bootstrap_staging_high_water_bytes"
        ],
        "staging_high_water_bytes": config["staging_high_water_bytes"],
        "active_target_bytes": config["active_target_bytes"],
        "retirement_grace_bytes": config["retirement_grace_bytes"],
        "project_hard_cap_bytes": config["project_hard_cap_bytes"],
        "min_free_bytes": config["min_free_bytes"],
        "max_shard_bytes": config["max_shard_bytes"],
    }


def _validate_arg_list(values: Any, name: str) -> list[str]:
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ProducerError(f"{name} must be a list of strings")
    for value in values:
        if value in RESERVED_GENERATOR_ARGS:
            raise ProducerError(f"{name} may not override producer-owned argument {value}")
    return list(values)


def load_and_validate_config(path: str | Path) -> tuple[dict[str, Any], str]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ProducerError(f"unexpected producer config schema: {config.get('schema_version')!r}")
    if config.get("pipeline_version") != PIPELINE_VERSION:
        raise ProducerError(f"unexpected pipeline version: {config.get('pipeline_version')!r}")
    if config.get("source_version") != SOURCE_VERSION:
        raise ProducerError(f"unexpected source version: {config.get('source_version')!r}")
    for key, default in (
        (
            "bootstrap_staging_high_water_bytes",
            DEFAULT_BOOTSTRAP_STAGING_HIGH_WATER_BYTES,
        ),
        ("staging_high_water_bytes", DEFAULT_STAGING_HIGH_WATER_BYTES),
        ("active_target_bytes", DEFAULT_ACTIVE_TARGET_BYTES),
        ("retirement_grace_bytes", DEFAULT_RETIREMENT_GRACE_BYTES),
        ("project_hard_cap_bytes", DEFAULT_PROJECT_HARD_CAP_BYTES),
        ("min_free_bytes", DEFAULT_MIN_FREE_BYTES),
        ("max_shard_bytes", DEFAULT_MAX_SHARD_BYTES),
    ):
        if key not in config:
            config[key] = default
        _require_int(config[key], key, 1)
    if config["max_shard_bytes"] >= min(
        config["bootstrap_staging_high_water_bytes"],
        config["staging_high_water_bytes"],
    ):
        raise ProducerError("max_shard_bytes must be below both staging watermarks")
    if config["bootstrap_staging_high_water_bytes"] < config["active_target_bytes"]:
        raise ProducerError(
            "bootstrap staging watermark must hold the complete initial active corpus"
        )
    if config["bootstrap_staging_high_water_bytes"] > config["project_hard_cap_bytes"]:
        raise ProducerError("bootstrap staging watermark exceeds project hard cap")
    budgeted = (
        int(config["active_target_bytes"])
        + int(config["staging_high_water_bytes"])
        + int(config["retirement_grace_bytes"])
    )
    if budgeted > int(config["project_hard_cap_bytes"]):
        raise ProducerError(
            "active_target_bytes + rolling staging_high_water_bytes + retirement_grace_bytes "
            "must not exceed project_hard_cap_bytes"
        )

    project_dir = Path(config.get("project_dir", "")).resolve()
    if not project_dir.is_dir():
        raise ProducerError(f"project_dir is missing: {project_dir}")
    config["project_dir"] = str(project_dir)
    control_root = Path(config.get("control_root", "")).expanduser().resolve()
    if not str(config.get("control_root", "")):
        raise ProducerError("control_root must name the shared durable catalog control root")
    config["control_root"] = str(control_root)
    pipeline_config_path = Path(config.get("pipeline_config_path", "")).resolve()
    pipeline_config_sha256 = str(config.get("pipeline_config_sha256", "")).lower()
    if not pipeline_config_path.is_file() or len(pipeline_config_sha256) != 64:
        raise ProducerError("pipeline_config_path/pipeline_config_sha256 is invalid")
    actual_pipeline_sha256 = sha256_file(pipeline_config_path)
    if actual_pipeline_sha256 != pipeline_config_sha256:
        raise ProducerError(
            f"pipeline config SHA256 mismatch: {actual_pipeline_sha256} != {pipeline_config_sha256}"
        )
    pipeline_payload = json.loads(pipeline_config_path.read_text(encoding="utf-8"))
    if pipeline_payload.get("pipeline_version") != PIPELINE_VERSION:
        raise ProducerError("pipeline config has a different pipeline_version")
    config["pipeline_config_path"] = str(pipeline_config_path)
    config["pipeline_config_sha256"] = pipeline_config_sha256

    generator_sources = config.get("generator_sources")
    if not isinstance(generator_sources, dict) or set(generator_sources) != set(PRIOR_KEYS):
        raise ProducerError(f"generator_sources must contain exactly {PRIOR_KEYS}")
    for prior in PRIOR_KEYS:
        source = generator_sources[prior]
        if not isinstance(source, dict):
            raise ProducerError(f"generator source for {prior} must be an object")
        generator_path = Path(source.get("path", "")).resolve()
        expected = str(source.get("sha256", "")).lower()
        if len(expected) != 64 or not generator_path.is_file():
            raise ProducerError(f"invalid generator path/hash for {prior}")
        actual = sha256_file(generator_path)
        if actual != expected:
            raise ProducerError(f"generator SHA256 mismatch for {prior}: {actual} != {expected}")
        source["path"] = str(generator_path)
        source["args"] = _validate_arg_list(source.get("args", []), f"generator_sources.{prior}.args")

    source_files = config.get("source_files", [])
    if not isinstance(source_files, list):
        raise ProducerError("source_files must be a list")
    seen_labels: set[str] = set()
    for entry in source_files:
        if not isinstance(entry, dict) or not isinstance(entry.get("label"), str):
            raise ProducerError("every source_files entry needs a string label")
        label = entry["label"]
        if label in seen_labels:
            raise ProducerError(f"duplicate source file label: {label}")
        seen_labels.add(label)
        source_path = Path(entry.get("path", "")).resolve()
        expected = str(entry.get("sha256", "")).lower()
        if not source_path.is_file() or len(expected) != 64:
            raise ProducerError(f"invalid source file path/hash for {label}")
        actual = sha256_file(source_path)
        if actual != expected:
            raise ProducerError(f"source file SHA256 mismatch for {label}: {actual} != {expected}")
        entry["path"] = str(source_path)

    config["generator_common_args"] = _validate_arg_list(
        config.get("generator_common_args", []), "generator_common_args"
    )
    workers = config.get("workers")
    if not isinstance(workers, list) or not workers:
        raise ProducerError("workers must be a nonempty list")
    seen_ids: set[str] = set()
    seen_seeds: set[int] = set()
    seen_cores: set[int] = set()
    for worker in workers:
        if not isinstance(worker, dict):
            raise ProducerError("worker entries must be objects")
        worker_id = worker.get("worker_id")
        prior = worker.get("prior")
        if not isinstance(worker_id, str) or not worker_id or worker_id in seen_ids:
            raise ProducerError(f"invalid or duplicate worker_id: {worker_id!r}")
        if prior not in PRIOR_KEYS:
            raise ProducerError(f"invalid prior for {worker_id}: {prior!r}")
        seen_ids.add(worker_id)
        seed = _require_int(worker.get("base_seed"), f"{worker_id}.base_seed", 1)
        if seed in seen_seeds or seed in EXP122_BASE_SEEDS:
            raise ProducerError(f"colliding/reused base seed for {worker_id}: {seed}")
        seen_seeds.add(seed)
        cores = worker.get("cpu_cores")
        if not isinstance(cores, list) or not cores:
            raise ProducerError(f"{worker_id}.cpu_cores must explicitly list at least one core")
        cores = [_require_int(core, f"{worker_id}.cpu_core", 0) for core in cores]
        if len(cores) != len(set(cores)) or seen_cores.intersection(cores):
            raise ProducerError(f"CPU cores must be distinct within/across workers: {worker_id}")
        seen_cores.update(cores)
        worker["cpu_cores"] = cores
        threads = _require_int(worker.get("threads", len(cores)), f"{worker_id}.threads", 1)
        if threads != len(cores):
            raise ProducerError(f"{worker_id}.threads must equal its explicitly pinned core count")
    contract = _canonical_source_contract(config)
    return config, sha256_bytes(canonical_json_bytes(contract))


def ensure_immutable_root_lock(root: str | Path, config: Mapping[str, Any], contract_sha256: str) -> dict[str, Any]:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    producer_path = Path(__file__).resolve()
    payload = {
        "schema_version": ROOT_LOCK_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "source_version": SOURCE_VERSION,
        "source_contract_sha256": contract_sha256,
        "producer_sha256": sha256_file(producer_path),
        "pipeline_config_sha256": config["pipeline_config_sha256"],
        "source_hashes": {
            "exp123_rolling_producer": sha256_file(producer_path),
            "pipeline_config": config["pipeline_config_sha256"],
            **{
                f"generator_{prior}": config["generator_sources"][prior]["sha256"]
                for prior in PRIOR_KEYS
            },
            **{
                str(item["label"]): str(item["sha256"])
                for item in config.get("source_files", [])
            },
        },
        "source_contract": _canonical_source_contract(config),
    }
    path = root / "producer" / "PRODUCER_CONFIG_LOCK.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ProducerError(
                "producer source/config changed under an existing Exp123 root; use a new source version/root"
            )
    else:
        atomic_write_json(path, payload)
    return payload


@dataclasses.dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    prior: str
    base_seed: int
    cpu_cores: tuple[int, ...]
    threads: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any], worker_id: str) -> "WorkerSpec":
        matches = [item for item in config["workers"] if item["worker_id"] == worker_id]
        if len(matches) != 1:
            raise ProducerError(f"worker_id not found exactly once: {worker_id}")
        item = matches[0]
        return cls(
            worker_id=str(item["worker_id"]),
            prior=str(item["prior"]),
            base_seed=int(item["base_seed"]),
            cpu_cores=tuple(int(core) for core in item["cpu_cores"]),
            threads=int(item["threads"]),
        )


def worker_state_path(root: str | Path, worker: WorkerSpec) -> Path:
    return Path(root).resolve() / "producer" / "workers" / worker.worker_id / "STATE.json"


def initial_worker_state(worker: WorkerSpec, source_contract_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "source_version": SOURCE_VERSION,
        "source_contract_sha256": source_contract_sha256,
        "worker_id": worker.worker_id,
        "prior": worker.prior,
        "base_seed": worker.base_seed,
        "next_batch_id": 0,
        "published_count": 0,
        "inflight": None,
        "last_published": None,
    }


def load_worker_state(root: str | Path, worker: WorkerSpec, source_contract_sha256: str) -> dict[str, Any]:
    path = worker_state_path(root, worker)
    if not path.exists():
        state = initial_worker_state(worker, source_contract_sha256)
        atomic_write_json(path, state)
        return state
    state = json.loads(path.read_text(encoding="utf-8"))
    expected = initial_worker_state(worker, source_contract_sha256)
    for key in (
        "schema_version",
        "pipeline_version",
        "source_version",
        "source_contract_sha256",
        "worker_id",
        "prior",
        "base_seed",
    ):
        if state.get(key) != expected[key]:
            raise ProducerError(f"worker state contract mismatch for {worker.worker_id}: {key}")
    _require_int(state.get("next_batch_id"), "next_batch_id", 0)
    _require_int(state.get("published_count"), "published_count", 0)
    inflight = state.get("inflight")
    if inflight is not None:
        if not isinstance(inflight, dict):
            raise ProducerError("inflight state must be null or an object")
        batch_id = _require_int(inflight.get("batch_id"), "inflight.batch_id", 0)
        if batch_id >= int(state["next_batch_id"]):
            raise ProducerError("inflight batch id was not durably consumed")
    return state


def reserve_or_resume_batch(
    root: str | Path,
    worker: WorkerSpec,
    source_contract_sha256: str,
    catalog_store: CatalogStore,
    *,
    require_catalog_initialized: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = load_worker_state(root, worker, source_contract_sha256)
    if state["inflight"] is not None:
        return state, dict(state["inflight"])
    if require_catalog_initialized:
        # The epoch-0 frozen import establishes NEXT_SHARD_ID above every
        # imported Exp122 shard. Reserving before that import could collide
        # with the migration IDs, so production fails closed here.
        catalog_store.load_current(verify_files=False)
    batch_id = int(state["next_batch_id"])
    shard_id = next(iter(catalog_store.reserve_shard_ids(1)))
    reserved = utc_now()
    dataset_id = f"{SOURCE_VERSION}:{worker.prior}:{worker.worker_id}:{batch_id:012d}"
    inflight = {
        "batch_id": batch_id,
        "shard_id": int(shard_id),
        "dataset_id": dataset_id,
        "reserved_at_utc": reserved,
        "prepared_sha256": None,
        "prepared_bytes": None,
    }
    state["next_batch_id"] = batch_id + 1
    state["inflight"] = inflight
    atomic_write_json(worker_state_path(root, worker), state)
    return state, inflight


def predicted_batch_seed(base_seed: int, batch_id: int) -> int:
    return int((int(base_seed) + BATCH_SEED_MULTIPLIER * int(batch_id)) % BATCH_SEED_MODULUS)


def staging_npz_bytes(
    root: str | Path,
    *,
    adopted_shard_ids: set[int] | None = None,
    prior: str | None = None,
) -> int:
    shards = Path(root).resolve() / "staging" / "shards"
    if prior is not None:
        shards = shards / prior
    if not shards.exists():
        return 0
    if adopted_shard_ids is None:
        return sum(path.stat().st_size for path in shards.rglob("*.npz") if path.is_file())
    # Active/adopted shards remain at their immutable data-root-relative paths,
    # so staging capacity is the physical shard store minus paths already in
    # catalog history. Orphans with no manifest remain conservatively counted.
    adopted_paths: set[Path] = set()
    inbox = Path(root).resolve() / "staging" / "inbox"
    for manifest_path in inbox.rglob("*.json") if inbox.exists() else ():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if int(manifest.get("shard_id", -1)) in adopted_shard_ids:
                relative = Path(str(manifest.get("relative_path", "")))
                candidate = (Path(root).resolve() / relative).resolve()
                if candidate.is_relative_to(Path(root).resolve()) and candidate.suffix == ".npz":
                    adopted_paths.add(candidate)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            # A malformed inbox artifact is not proof of adoption and cannot
            # reduce the producer's observed staging bytes.
            continue
    return sum(
        path.stat().st_size
        for path in shards.rglob("*.npz")
        if path.is_file() and path.resolve() not in adopted_paths
    )


def check_backpressure(
    root: str | Path,
    *,
    staging_high_water_bytes: int,
    min_free_bytes: int,
    reserve_bytes: int,
    adopted_shard_ids: set[int] | None = None,
    prior: str | None = None,
    prior_high_water_bytes: int | None = None,
) -> dict[str, int]:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    used = staging_npz_bytes(root, adopted_shard_ids=adopted_shard_ids)
    free = int(shutil.disk_usage(root).free)
    if used + reserve_bytes > int(staging_high_water_bytes):
        raise Backpressure(
            f"staging high-water backpressure: {used}+{reserve_bytes}>{staging_high_water_bytes}"
        )
    prior_used = 0
    if prior is not None and prior_high_water_bytes is not None:
        prior_used = staging_npz_bytes(
            root, adopted_shard_ids=adopted_shard_ids, prior=prior
        )
        if prior_used + reserve_bytes > int(prior_high_water_bytes):
            raise Backpressure(
                f"{prior} staging high-water backpressure: "
                f"{prior_used}+{reserve_bytes}>{prior_high_water_bytes}"
            )
    if free - reserve_bytes < int(min_free_bytes):
        raise Backpressure(f"free-space backpressure: {free}-{reserve_bytes}<{min_free_bytes}")
    return {
        "staging_bytes": used,
        "prior_staging_bytes": prior_used,
        "free_bytes": free,
        "reserve_bytes": int(reserve_bytes),
    }


def scratch_dir(root: str | Path, worker: WorkerSpec, batch_id: int) -> Path:
    return (
        Path(root).resolve()
        / "producer"
        / "scratch"
        / worker.worker_id
        / f"batch_{int(batch_id):012d}"
    )


def _assert_linux_cpu_affinity(cpu_cores: Sequence[int]) -> None:
    if not sys.platform.startswith("linux"):
        raise ProducerError("production generation requires Linux taskset CPU affinity")
    if shutil.which("taskset") is None:
        raise ProducerError("taskset is required for production CPU affinity")
    if not hasattr(os, "sched_getaffinity"):
        raise ProducerError("Linux sched_getaffinity is required")
    allowed = set(os.sched_getaffinity(0))
    missing = sorted(set(cpu_cores) - allowed)
    if missing:
        raise ProducerError(f"configured CPU cores are outside current affinity: {missing}")


def build_generator_command(
    config: Mapping[str, Any],
    worker: WorkerSpec,
    batch_id: int,
    out_dir: str | Path,
    *,
    python: str | Path = sys.executable,
    enforce_platform: bool = True,
) -> tuple[list[str], dict[str, str]]:
    if enforce_platform:
        _assert_linux_cpu_affinity(worker.cpu_cores)
    source = config["generator_sources"][worker.prior]
    command = [
        "taskset",
        "-c",
        ",".join(str(core) for core in worker.cpu_cores),
        str(python),
        str(source["path"]),
        "--out-dir",
        str(Path(out_dir).resolve()),
        "--num-batches",
        "1",
        "--batch-size",
        "1",
        "--start-index",
        str(int(batch_id)),
        "--seed",
        str(worker.base_seed),
        "--threads",
        str(worker.threads),
        "--corr-device",
        "cpu",
        *config.get("generator_common_args", []),
        *source.get("args", []),
    ]
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONHASHSEED": str(worker.base_seed),
            "OMP_NUM_THREADS": str(worker.threads),
            "MKL_NUM_THREADS": str(worker.threads),
            "OPENBLAS_NUM_THREADS": str(worker.threads),
            "NUMEXPR_NUM_THREADS": str(worker.threads),
            "VECLIB_MAXIMUM_THREADS": str(worker.threads),
            "BLIS_NUM_THREADS": str(worker.threads),
        }
    )
    return command, env


def run_generator_once(
    config: Mapping[str, Any],
    worker: WorkerSpec,
    batch_id: int,
    out_dir: str | Path,
    *,
    python: str | Path = sys.executable,
) -> None:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    command, env = build_generator_command(
        config, worker, batch_id, out_dir, python=python, enforce_platform=True
    )
    subprocess.run(command, cwd=config["project_dir"], env=env, check=True)


def _find_generator_output(out_dir: str | Path, batch_id: int) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    out_dir = Path(out_dir).resolve()
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ProducerError(f"generator returned without manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ProducerError("generator manifest files must be a list")
    expected_name = f"batch_{int(batch_id):06d}.npz"
    matches = [entry for entry in entries if entry.get("path") == expected_name]
    if len(matches) != 1:
        raise ProducerError(f"generator manifest does not contain exactly one {expected_name}")
    entry = matches[0]
    npz_path = (out_dir / expected_name).resolve()
    if not npz_path.is_relative_to(out_dir) or not npz_path.is_file():
        raise ProducerError(f"generator NPZ is missing or escaped scratch: {npz_path}")
    if int(entry.get("bytes", -1)) != int(npz_path.stat().st_size):
        raise ProducerError(f"generator NPZ byte count mismatch: {npz_path}")
    return npz_path, manifest, entry


def validate_generated_npz(npz_path: str | Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as archive:
        required = {
            "X",
            "y",
            "exact_max_abs_corr",
            "corr_threshold",
            "scm_response_mean",
            "scm_response_std",
            "batch_seed",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ProducerError(f"generated NPZ lacks required arrays: {missing}")
        x = archive["X"]
        y = archive["y"]
        corr = archive["exact_max_abs_corr"]
        threshold = archive["corr_threshold"]
        response_mean = archive["scm_response_mean"]
        response_std = archive["scm_response_std"]
        batch_seed = archive["batch_seed"]
        if x.ndim != 3 or y.ndim != 2 or x.shape[:2] != y.shape:
            raise ProducerError(f"invalid X/y shapes: X={x.shape} y={y.shape}")
        if x.dtype != np.float16 or y.dtype != np.float32:
            raise ProducerError(f"invalid X/y dtypes: X={x.dtype} y={y.dtype}")
        if int(entry.get("num_tasks", -1)) != x.shape[0]:
            raise ProducerError("manifest num_tasks disagrees with NPZ")
        if int(entry.get("n", -1)) != x.shape[1] or int(entry.get("p", -1)) != x.shape[2]:
            raise ProducerError("manifest N/P disagrees with NPZ")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ProducerError("generated X/y contains nonfinite values")
        for name, value in (
            ("exact_max_abs_corr", corr),
            ("corr_threshold", threshold),
            ("scm_response_mean", response_mean),
            ("scm_response_std", response_std),
        ):
            if value.shape != (x.shape[0],) or not np.isfinite(value).all():
                raise ProducerError(f"invalid {name} certificate")
        if batch_seed.shape != (x.shape[0],) or batch_seed.dtype != np.uint64:
            raise ProducerError("invalid batch_seed certificate")
        if not bool(np.all(corr >= 0.0)) or not bool(np.all(corr < threshold)):
            raise ProducerError("stored-X correlation certificate failed")
        if not bool(np.all(threshold > 0.0)) or not bool(np.all(threshold < 1.0)):
            raise ProducerError("invalid correlation threshold")
        realized_mean = np.mean(y.astype(np.float64), axis=1)
        realized_std = np.std(y.astype(np.float64), axis=1)
        if not np.allclose(response_mean, realized_mean, rtol=0.0, atol=2e-6):
            raise ProducerError("stored SCM response mean certificate disagrees with y")
        if not np.allclose(response_std, realized_std, rtol=0.0, atol=2e-6):
            raise ProducerError("stored SCM response std certificate disagrees with y")
        if not bool(np.all(np.abs(realized_mean) <= 2e-6)):
            raise ProducerError("stored SCM response is not centered")
        if not bool(np.all(np.abs(realized_std - 1.0) <= 2e-6)):
            raise ProducerError("stored SCM response is not unit scale")
        summary = {
            "num_tasks": int(x.shape[0]),
            "n": int(x.shape[1]),
            "p": int(x.shape[2]),
            "x_dtype": str(x.dtype),
            "y_dtype": str(y.dtype),
            "max_abs_corr": float(np.max(corr)),
            "corr_threshold_min": float(np.min(threshold)),
            "scm_response_mean_max_abs": float(np.max(np.abs(realized_mean))),
            "scm_response_std_min": float(np.min(realized_std)),
            "scm_response_std_max": float(np.max(realized_std)),
            "batch_seed_min": int(np.min(batch_seed)),
            "batch_seed_max": int(np.max(batch_seed)),
        }
    max_from_entry = float(entry.get("task_max_abs_corr_max", float("nan")))
    if not math.isfinite(max_from_entry) or not math.isclose(
        max_from_entry, summary["max_abs_corr"], rel_tol=1e-5, abs_tol=1e-6
    ):
        raise ProducerError("manifest max-correlation certificate disagrees with NPZ")
    return summary


def _require_exact(mapping: Mapping[str, Any], key: str, expected: Any, *, where: str) -> Any:
    actual = mapping.get(key)
    if actual != expected:
        raise ProducerError(f"{where}.{key} contract mismatch: {actual!r} != {expected!r}")
    return actual


def _require_close(
    mapping: Mapping[str, Any],
    key: str,
    expected: float,
    *,
    where: str,
    atol: float = 1e-8,
) -> float:
    try:
        actual = float(mapping.get(key))
    except (TypeError, ValueError) as exc:
        raise ProducerError(f"{where}.{key} is not numeric") from exc
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=0.0, abs_tol=atol):
        raise ProducerError(f"{where}.{key} contract mismatch: {actual!r} != {expected!r}")
    return actual


def validate_generator_contract(
    prior: str,
    manifest: Mapping[str, Any],
    entry: Mapping[str, Any],
    npz_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless a shard is an exact paired-SCM X/y task."""
    if prior not in PRIOR_CONTRACTS:
        raise ProducerError(f"unsupported prior: {prior!r}")
    contract = PRIOR_CONTRACTS[prior]
    _require_exact(manifest, "generator", contract["generator"], where="manifest")
    _require_exact(manifest, "prior", contract["prior_name"], where="manifest")
    _require_exact(manifest, "dataset_contract", contract["dataset_contract"], where="manifest")
    _require_exact(manifest, "num_datasets", 1, where="manifest")
    settings = manifest.get("settings")
    if not isinstance(settings, Mapping):
        raise ProducerError("manifest.settings must be an object")
    settings_exact = {
        "shape_sampler": EXPECTED_SHAPE_SAMPLER,
        "target_median_n": 2000,
        "target_median_p": 10000,
        "fixed_batch_size": True,
        "batch_size": 1,
        "n_min": 32,
        "n_max": 20_000,
        "main_n_min": 200,
        "small_n_max": 199,
        "small_n_probability": 0.10,
        "p_min": 50,
        "p_max": 100_000,
        "max_cells": 24_000_000,
        "official_feature_prior": contract["official_core_prior"],
        "data_policy": contract["data_policy"],
        "direct_p_cap": 4096,
        "core_oversample_factor": 1.5,
        "correlation_threshold_exclusive": 0.95,
        "correlation_certification": EXPECTED_CORRELATION_CERTIFICATE,
        "mlp_output_features": 512,
        "mlp_hidden_dim": 256,
        "gaussian_input_dims": 16,
        "mlp_noise_scale_contract": EXPECTED_MLP_NOISE_CONTRACT,
        "target_contract": contract["target_contract"],
        "target_transformation": "center, divide by population standard deviation, clip to +/-clip_value, center and scale again",
        "target_clip_value": 10.0,
    }
    if prior == "tabpfn_v1":
        settings_exact.update({
            "official_tabpfn_tag": "v1.0.0",
            "official_tabpfn_commit": contract["commit"],
            "official_tabpfn_worktree_clean": True,
            "official_checkpoint_sha256": contract["checkpoint_sha256"],
            "official_prior_conditioning": "causal SCM branch only (is_causal=True)",
            "causal_policy": "tabpfn_v1_is_causal_true_only_v1",
            "noncausal_bnn_used": False,
            "official_target_usage": "retained from the same official TabPFN v1 causal SCM draw, then standardized",
        })
    else:
        settings_exact.update({
            "official_tabicl_commit": contract["commit"],
            "official_tabicl_worktree_clean": True,
            "official_target_usage": "retained from the same official TabICLv2 graph_scm draw, then standardized",
            "filter_unpredictable": True,
        })
    for key, expected in settings_exact.items():
        actual = settings.get(key)
        if isinstance(expected, float):
            _require_close(settings, key, expected, where="manifest.settings")
        elif actual != expected:
            raise ProducerError(
                f"manifest.settings.{key} contract mismatch: {actual!r} != {expected!r}"
            )
    for key, expected in (
        ("input_noise_variance_range", [0.001, 1.0]),
        ("output_noise_variance_range", [0.001, 0.25]),
    ):
        if settings.get(key) != expected:
            raise ProducerError(f"manifest.settings.{key} contract mismatch")

    entry_exact = {
        "num_tasks": 1,
        "shape_sampler": EXPECTED_SHAPE_SAMPLER,
        "prior": contract["prior_name"],
        "official_core_prior": contract["official_core_prior"],
        "data_policy": contract["data_policy"],
        "target_contract": contract["target_contract"],
        "correlation_certification": EXPECTED_CORRELATION_CERTIFICATE,
    }
    for key, expected in entry_exact.items():
        _require_exact(entry, key, expected, where="manifest.entry")
    _require_close(entry, "correlation_threshold_exclusive", 0.95, where="manifest.entry")
    if prior == "tabpfn_v1":
        _require_exact(entry, "causal_policy", "tabpfn_v1_is_causal_true_only_v1", where="manifest.entry")
        _require_exact(entry, "noncausal_bnn_used", False, where="manifest.entry")

    n = int(entry.get("n", -1))
    p = int(entry.get("p", -1))
    requested_p = int(entry.get("requested_p", -1))
    if not (32 <= n <= 20_000 and 50 <= p <= 100_000 and 50 <= requested_p <= 100_000):
        raise ProducerError(f"manifest.entry shape outside contract: N={n} P={p} requestedP={requested_p}")
    if n * requested_p > 24_000_000 or n * p > 24_000_000:
        raise ProducerError(f"manifest.entry shape exceeds 24M cells: N={n} P={p} requestedP={requested_p}")
    if (n, p) != (int(npz_summary["n"]), int(npz_summary["p"])):
        raise ProducerError("semantic manifest shape disagrees with NPZ summary")
    for key in ("task_response_std_min", "task_response_std_median", "task_response_std_max"):
        value = float(entry.get(key, float("nan")))
        if not math.isfinite(value) or not math.isclose(value, 1.0, rel_tol=0.0, abs_tol=2e-6):
            raise ProducerError(f"manifest.entry.{key} is not unit scale")

    task_meta = entry.get("task_metadata")
    if not isinstance(task_meta, list) or len(task_meta) != 1 or not isinstance(task_meta[0], Mapping):
        raise ProducerError("manifest.entry.task_metadata must contain exactly one object")
    task = task_meta[0]
    _require_exact(task, "task_index", 0, where="task")
    _require_exact(task, "official_core_prior", contract["official_core_prior"], where="task")
    _require_exact(task, "data_policy", contract["data_policy"], where="task")
    _require_exact(task, "target_contract", contract["target_contract"], where="task")
    _require_exact(task, "target_source", contract["target_source"], where="task")
    _require_exact(task, "official_target_retained", True, where="task")
    if prior == "tabpfn_v1":
        _require_exact(task, "causal_policy", "tabpfn_v1_is_causal_true_only_v1", where="task")
        _require_exact(task, "noncausal_bnn_used", False, where="task")
        hp = task.get("official_sampled_hyperparameters")
        if not isinstance(hp, Mapping) or hp.get("is_causal") is not True:
            raise ProducerError("TabPFN task is not certified causal")
    _require_close(task, "exact_max_abs_corr", float(npz_summary["max_abs_corr"]), where="task", atol=1e-6)
    _require_close(task, "target_final_mean", 0.0, where="task", atol=2e-6)
    _require_close(task, "target_final_std", 1.0, where="task", atol=2e-6)
    raw_std = float(task.get("target_raw_std", float("nan")))
    clip_count = int(task.get("target_pre_rescale_clip_count", -1))
    if not math.isfinite(raw_std) or raw_std <= 1e-7:
        raise ProducerError("task target_raw_std is invalid")
    if not (0 <= clip_count <= n):
        raise ProducerError("task target clip count is invalid")
    if int(npz_summary["batch_seed_min"]) != int(npz_summary["batch_seed_max"]):
        raise ProducerError("one-task shard has inconsistent batch seed")

    mlp_chunks = task.get("mlp_chunks")
    if not isinstance(mlp_chunks, list):
        raise ProducerError("task.mlp_chunks must be a list")
    if requested_p <= 4096:
        _require_exact(task, "mode", contract["direct_mode"], where="task")
        if p > requested_p or mlp_chunks or int(task.get("direct_mode_mlp_chunks", -1)) != 0:
            raise ProducerError("direct P<=4096 task violated no-MLP/no-replenishment contract")
    else:
        _require_exact(task, "mode", contract["highp_mode"], where="task")
        if p != requested_p or not mlp_chunks:
            raise ProducerError("high-P task failed exact MLP replenishment contract")
        for index, chunk in enumerate(mlp_chunks):
            if not isinstance(chunk, Mapping):
                raise ProducerError(f"task.mlp_chunks[{index}] is not an object")
            where = f"task.mlp_chunks[{index}]"
            _require_exact(chunk, "gaussian_input_dims", 16, where=where)
            _require_exact(chunk, "hidden_dim", 256, where=where)
            _require_exact(chunk, "candidate_features", 512, where=where)
            _require_exact(chunk, "noise_scale_contract", EXPECTED_MLP_NOISE_CONTRACT, where=where)
            if chunk.get("activation") not in EXPECTED_MLP_ACTIVATIONS:
                raise ProducerError(f"{where}.activation is outside the contract")
            parents = int(chunk.get("parents", 0))
            accepted = int(chunk.get("accepted_features", -1))
            retained_before = int(chunk.get("retained_before", -1))
            if not (1 <= parents <= min(512, retained_before)) or not (0 <= accepted <= 512):
                raise ProducerError(f"{where} parent/accepted count is invalid")
            input_variance = float(chunk.get("gaussian_input_variance", float("nan")))
            output_variance = float(chunk.get("output_noise_variance", float("nan")))
            if not (0.001 <= input_variance <= 1.0) or not (0.001 <= output_variance <= 0.25):
                raise ProducerError(f"{where} sampled noise variance is outside the contract")
            for value_key in (
                "gaussian_input_realized_variance",
                "parent_input_realized_variance",
                "output_noise_realized_variance",
                "output_signal_realized_variance",
            ):
                value = float(chunk.get(value_key, float("nan")))
                if not math.isfinite(value) or value < 0.0:
                    raise ProducerError(f"{where}.{value_key} is invalid")
    return {
        "contract": contract["dataset_contract"],
        "prior": prior,
        "n": n,
        "p": p,
        "requested_p": requested_p,
        "mode": str(task["mode"]),
        "target_contract": contract["target_contract"],
        "scm_response_mean_max_abs": float(npz_summary["scm_response_mean_max_abs"]),
        "scm_response_std_min": float(npz_summary["scm_response_std_min"]),
        "scm_response_std_max": float(npz_summary["scm_response_std_max"]),
        "mlp_chunks": len(mlp_chunks),
    }


def _bucket(value: int, cuts: Sequence[int]) -> str:
    lower = 0
    for upper in cuts:
        if value <= upper:
            return f"{lower:06d}-{upper:06d}"
        lower = upper + 1
    return f"{lower:06d}+"


def build_shard_manifest(
    *,
    config: Mapping[str, Any],
    root_lock: Mapping[str, Any],
    worker: WorkerSpec,
    inflight: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    source_entry: Mapping[str, Any],
    npz_summary: Mapping[str, Any],
    content_sha256: str,
    content_bytes: int,
    relative_shard_path: str,
) -> dict[str, Any]:
    batch_id = int(inflight["batch_id"])
    n, p = int(npz_summary["n"]), int(npz_summary["p"])
    source_hashes = dict(root_lock["source_hashes"])
    stratum = {
        "prior": worker.prior,
        "n_bucket": _bucket(n, (199, 999, 4_999, 9_999, 19_999, 40_000)),
        "p_bucket": _bucket(p, (99, 999, 4_095, 9_999, 49_999, 200_000)),
        "small_n_component": bool(source_entry.get("small_n_component", n < 200)),
        "target_contract": PRIOR_CONTRACTS[worker.prior]["target_contract"],
    }
    # These top-level names are the canonical CatalogStore addition schema.
    # Rich producer-only details remain under metadata and are preserved when
    # CatalogStore normalises the entry.
    return {
        "schema_version": SHARD_MANIFEST_SCHEMA_VERSION,
        "shard_id": int(inflight["shard_id"]),
        "prior": worker.prior,
        "relative_path": relative_shard_path,
        "bytes": int(content_bytes),
        "sha256": content_sha256,
        "seed": predicted_batch_seed(worker.base_seed, batch_id),
        "n": n,
        "p": p,
        "num_tasks": int(npz_summary["num_tasks"]),
        "task_max_abs_corr_max": float(npz_summary["max_abs_corr"]),
        "created_at_utc": inflight["reserved_at_utc"],
        "pipeline_config_sha256": root_lock["pipeline_config_sha256"],
        "source_hashes": source_hashes,
        "stratum": stratum,
        "metadata": {
            "pipeline_version": PIPELINE_VERSION,
            "source_version": SOURCE_VERSION,
            "dataset_id": inflight["dataset_id"],
            "worker_id": worker.worker_id,
            "batch_id": batch_id,
            "base_seed": worker.base_seed,
            "requested_p": int(source_entry.get("requested_p", p)),
            "x_dtype": npz_summary["x_dtype"],
            "y_dtype": npz_summary["y_dtype"],
            "finite": True,
            "corr_threshold_min": float(npz_summary["corr_threshold_min"]),
            "batch_seed_min": int(npz_summary["batch_seed_min"]),
            "batch_seed_max": int(npz_summary["batch_seed_max"]),
            "scm_response_mean_max_abs": float(npz_summary["scm_response_mean_max_abs"]),
            "scm_response_std_min": float(npz_summary["scm_response_std_min"]),
            "scm_response_std_max": float(npz_summary["scm_response_std_max"]),
            "source_contract_sha256": root_lock["source_contract_sha256"],
            "generator_manifest_sha256": sha256_bytes(canonical_json_bytes(source_manifest)),
            "generator_manifest_entry": dict(source_entry),
            # Preserve the producer-validated semantic certificate verbatim.
            # Catalog consumers can then enforce the generator contract
            # without consulting mutable generator-side files.
            "semantic_contract": dict(npz_summary["semantic_contract"]),
        },
    }


def _published_manifests_for_batch(root: str | Path, worker: WorkerSpec, batch_id: int) -> list[Path]:
    directory = Path(root).resolve() / "staging" / "inbox" / worker.prior / worker.worker_id
    return sorted(directory.glob(f"batch_{int(batch_id):012d}_*.json")) if directory.exists() else []


def recover_committed_inflight(
    root: str | Path,
    worker: WorkerSpec,
    source_contract_sha256: str,
) -> bool:
    state = load_worker_state(root, worker, source_contract_sha256)
    inflight = state.get("inflight")
    if inflight is None:
        return False
    paths = _published_manifests_for_batch(root, worker, int(inflight["batch_id"]))
    if not paths:
        return False
    if len(paths) != 1:
        raise ProducerError(f"multiple published manifests for one dataset id: {paths}")
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    if metadata.get("dataset_id") != inflight["dataset_id"]:
        raise ProducerError("published manifest dataset id disagrees with inflight state")
    if int(payload.get("shard_id", -1)) != int(inflight["shard_id"]):
        raise ProducerError("published manifest shard id disagrees with inflight state")
    shard = Path(root).resolve() / str(payload.get("relative_path", ""))
    if not shard.is_file() or sha256_file(shard) != payload.get("sha256"):
        raise ProducerError("published manifest references a missing/corrupt shard")
    state["published_count"] = int(state["published_count"]) + 1
    state["last_published"] = {
        "dataset_id": metadata["dataset_id"],
        "shard_id": int(payload["shard_id"]),
        "manifest": str(paths[0].relative_to(Path(root).resolve())),
        "sha256": payload["sha256"],
        "published_at_utc": utc_now(),
    }
    state["inflight"] = None
    atomic_write_json(worker_state_path(root, worker), state)
    work = scratch_dir(root, worker, int(inflight["batch_id"]))
    if work.exists() and work.is_relative_to(Path(root).resolve() / "producer" / "scratch"):
        shutil.rmtree(work)
    return True


def produce_one(
    config: Mapping[str, Any],
    root: str | Path,
    worker: WorkerSpec,
    source_contract_sha256: str,
    root_lock: Mapping[str, Any],
    catalog_store: CatalogStore,
    *,
    python: str | Path = sys.executable,
    runner=run_generator_once,
    failure_point: str | None = None,
    require_catalog_initialized: bool = True,
) -> Path:
    root = Path(root).resolve()
    if catalog_store.data_root != root:
        raise ProducerError(
            f"catalog data_root differs from producer root: {catalog_store.data_root} != {root}"
        )
    if recover_committed_inflight(root, worker, source_contract_sha256):
        # Recovery completed a prior commit; reserve a fresh id below.
        pass
    adopted_ids: set[int] = set()
    if require_catalog_initialized:
        # CURRENT was already required by reserve_or_resume_batch; read the
        # immutable catalog chain to distinguish pending producer staging from
        # files that are now part of active/history accounting.
        catalog_store.load_current(verify_files=False)
        adopted_ids = set(catalog_store._historical_identity_sets()[0])
    check_backpressure(
        root,
        staging_high_water_bytes=int(
            config["staging_high_water_bytes"]
            if require_catalog_initialized
            else config["bootstrap_staging_high_water_bytes"]
        ),
        min_free_bytes=int(config["min_free_bytes"]),
        reserve_bytes=int(config["max_shard_bytes"]),
        adopted_shard_ids=adopted_ids,
        prior=worker.prior,
        prior_high_water_bytes=int(
            (config["staging_high_water_bytes"] if require_catalog_initialized else config["bootstrap_staging_high_water_bytes"])
            // len(PRIOR_KEYS)
        ),
    )
    state, inflight = reserve_or_resume_batch(
        root,
        worker,
        source_contract_sha256,
        catalog_store,
        require_catalog_initialized=require_catalog_initialized,
    )
    batch_id = int(inflight["batch_id"])
    work = scratch_dir(root, worker, batch_id)
    work.mkdir(parents=True, exist_ok=True)

    try:
        npz_path, source_manifest, source_entry = _find_generator_output(work, batch_id)
        npz_summary = validate_generated_npz(npz_path, source_entry)
    except (FileNotFoundError, ProducerError, json.JSONDecodeError, ValueError):
        runner(config, worker, batch_id, work, python=python)
        npz_path, source_manifest, source_entry = _find_generator_output(work, batch_id)
        npz_summary = validate_generated_npz(npz_path, source_entry)
    npz_summary = dict(npz_summary)
    npz_summary["semantic_contract"] = validate_generator_contract(
        worker.prior, source_manifest, source_entry, npz_summary
    )

    content_bytes = int(npz_path.stat().st_size)
    if content_bytes <= 0 or content_bytes > int(config["max_shard_bytes"]):
        raise ProducerError(
            f"generated shard size {content_bytes} is outside (0,{config['max_shard_bytes']}]"
        )
    check_backpressure(
        root,
        staging_high_water_bytes=int(
            config["staging_high_water_bytes"]
            if require_catalog_initialized
            else config["bootstrap_staging_high_water_bytes"]
        ),
        min_free_bytes=int(config["min_free_bytes"]),
        reserve_bytes=content_bytes,
        adopted_shard_ids=adopted_ids,
        prior=worker.prior,
        prior_high_water_bytes=int(
            (config["staging_high_water_bytes"] if require_catalog_initialized else config["bootstrap_staging_high_water_bytes"])
            // len(PRIOR_KEYS)
        ),
    )
    content_sha256 = sha256_file(npz_path)
    state = load_worker_state(root, worker, source_contract_sha256)
    current = state.get("inflight")
    if current is None or current.get("dataset_id") != inflight["dataset_id"]:
        raise ProducerError("worker inflight state changed during generation")
    prepared = current.get("prepared_sha256")
    if prepared not in (None, content_sha256):
        raise ProducerError(
            f"same reserved dataset id generated different bytes: {prepared} != {content_sha256}"
        )
    current["prepared_sha256"] = content_sha256
    current["prepared_bytes"] = content_bytes
    state["inflight"] = current
    atomic_write_json(worker_state_path(root, worker), state)
    if failure_point == "after_prepare":
        raise ProducerError("injected failure after durable preparation")

    shard_rel = (
        Path("staging")
        / "shards"
        / worker.prior
        / content_sha256[:2]
        / f"{content_sha256}.npz"
    )
    shard_path = root / shard_rel
    publish_immutable_file(npz_path, shard_path, content_sha256)
    if failure_point == "after_shard":
        raise ProducerError("injected failure after shard publication")

    manifest = build_shard_manifest(
        config=config,
        root_lock=root_lock,
        worker=worker,
        inflight=current,
        source_manifest=source_manifest,
        source_entry=source_entry,
        npz_summary=npz_summary,
        content_sha256=content_sha256,
        content_bytes=content_bytes,
        relative_shard_path=str(shard_rel),
    )
    manifest_name = f"batch_{batch_id:012d}_{content_sha256}.json"
    manifest_path = root / "staging" / "inbox" / worker.prior / worker.worker_id / manifest_name
    publish_immutable_bytes(manifest_path, canonical_json_bytes(manifest))
    if failure_point == "after_manifest":
        raise ProducerError("injected failure after manifest publication")

    recovered = recover_committed_inflight(root, worker, source_contract_sha256)
    if not recovered:
        raise ProducerError("manifest published but worker state did not commit")
    return manifest_path


@contextlib.contextmanager
def exclusive_worker_lock(root: str | Path, worker: WorkerSpec):
    path = Path(root).resolve() / "producer" / "workers" / worker.worker_id / "LOCK"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProducerError(f"another process owns worker {worker.worker_id}") from exc
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def run_worker_loop(
    config: Mapping[str, Any],
    root: str | Path,
    worker: WorkerSpec,
    source_contract_sha256: str,
    root_lock: Mapping[str, Any],
    catalog_store: CatalogStore,
    *,
    python: str | Path,
    max_batches: int,
    idle_seconds: float,
    require_catalog_initialized: bool,
) -> int:
    stop = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    old_handlers = {
        signal.SIGINT: signal.signal(signal.SIGINT, request_stop),
        signal.SIGTERM: signal.signal(signal.SIGTERM, request_stop),
    }
    published = 0
    try:
        with exclusive_worker_lock(root, worker):
            while not stop and (max_batches <= 0 or published < max_batches):
                try:
                    path = produce_one(
                        config,
                        root,
                        worker,
                        source_contract_sha256,
                        root_lock,
                        catalog_store,
                        python=python,
                        require_catalog_initialized=require_catalog_initialized,
                    )
                    published += 1
                    print(f"published {path}", flush=True)
                except Backpressure as exc:
                    print(f"BACKPRESSURE worker={worker.worker_id}: {exc}", flush=True)
                    if max_batches == 1:
                        return 75
                    time.sleep(max(1.0, idle_seconds))
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="0 means continuous; 1 is useful for a service/cron tick",
    )
    parser.add_argument("--idle-seconds", type=float, default=DEFAULT_IDLE_SECONDS)
    parser.add_argument(
        "--allow-before-current",
        action="store_true",
        help="bootstrap workers may stage fresh shards before epoch-0 CURRENT exists",
    )
    args = parser.parse_args(argv)
    if args.max_batches < 0:
        parser.error("--max-batches must be >= 0")
    if args.idle_seconds <= 0:
        parser.error("--idle-seconds must be > 0")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config, contract_sha256 = load_and_validate_config(args.config)
    worker = WorkerSpec.from_config(config, args.worker_id)
    root_lock = ensure_immutable_root_lock(args.root, config, contract_sha256)
    catalog_store = CatalogStore(
        config["control_root"], args.root, priors=PRIOR_KEYS
    )
    return run_worker_loop(
        config,
        args.root,
        worker,
        contract_sha256,
        root_lock,
        catalog_store,
        python=args.python,
        max_batches=args.max_batches,
        idle_seconds=args.idle_seconds,
        require_catalog_initialized=not args.allow_before_current,
    )


if __name__ == "__main__":
    raise SystemExit(main())
