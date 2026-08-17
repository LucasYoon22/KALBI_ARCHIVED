"""Run the P<=100K producer with bounded generation and staging accounting."""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

from model_v58_chebyshev_v14_scmmixture import rolling_producer as producer
from model_v58_chebyshev_v14_scmmixture.rolling_catalog import CatalogStore

from .rolling_adopted_cache import load_cached_adopted_ids


_ORIGINAL_STAGING_NPZ_BYTES = producer.staging_npz_bytes
_ORIGINAL_PRODUCE_ONE = producer.produce_one
GENERATOR_TIMEOUT_ENV = "ROLLING_GENERATOR_TIMEOUT_SECONDS"
DEFAULT_GENERATOR_TIMEOUT_SECONDS = 900.0


class GeneratorTimeout(producer.ProducerError):
    pass


def _generator_timeout_seconds() -> float:
    value = float(os.environ.get(
        GENERATOR_TIMEOUT_ENV,
        str(DEFAULT_GENERATOR_TIMEOUT_SECONDS),
    ))
    if not value > 0.0:
        raise producer.ProducerError(
            f"{GENERATOR_TIMEOUT_ENV} must be positive"
        )
    return value


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise producer.ProducerError("inflight timestamp is not timezone-aware")
    return parsed.astimezone(timezone.utc)


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    with contextlib.suppress(ProcessLookupError):
        process.terminate()
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        process.wait()


def _run_generator_once_with_timeout(
    config: Mapping[str, Any],
    worker: producer.WorkerSpec,
    batch_id: int,
    out_dir: str | Path,
    *,
    python: str | Path = sys.executable,
) -> None:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    command, env = producer.build_generator_command(
        config,
        worker,
        batch_id,
        out_dir,
        python=python,
        enforce_platform=True,
    )
    timeout = _generator_timeout_seconds()
    process = subprocess.Popen(
        command,
        cwd=config["project_dir"],
        env=env,
    )
    try:
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_process(process)
        raise GeneratorTimeout(
            f"generator exceeded {timeout:.0f}s for "
            f"worker={worker.worker_id} batch={batch_id}"
        ) from exc
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _abandon_inflight(
    root: str | Path,
    worker: producer.WorkerSpec,
    source_contract_sha256: str,
    *,
    reason: str,
) -> bool:
    root_path = Path(root).expanduser().resolve()
    state = producer.load_worker_state(
        root_path,
        worker,
        source_contract_sha256,
    )
    inflight = state.get("inflight")
    if inflight is None:
        return False
    if inflight.get("prepared_sha256") is not None:
        raise producer.ProducerError(
            "refusing to abandon a durably prepared inflight shard"
        )
    published = producer._published_manifests_for_batch(
        root_path,
        worker,
        int(inflight["batch_id"]),
    )
    if published:
        raise producer.ProducerError(
            "refusing to abandon an inflight shard with a published manifest"
        )

    abandoned_at = producer.utc_now()
    event = {
        "version": "v58v17_p100k_abandoned_inflight_v1",
        "worker_id": worker.worker_id,
        "prior": worker.prior,
        "batch_id": int(inflight["batch_id"]),
        "shard_id": int(inflight["shard_id"]),
        "dataset_id": str(inflight["dataset_id"]),
        "reserved_at_utc": str(inflight["reserved_at_utc"]),
        "abandoned_at_utc": abandoned_at,
        "reason": reason,
    }
    event_path = (
        root_path
        / "producer"
        / "workers"
        / worker.worker_id
        / "abandoned"
        / f"batch_{int(inflight['batch_id']):012d}.json"
    )
    producer.atomic_write_json(event_path, event)
    state["abandoned_count"] = int(state.get("abandoned_count", 0)) + 1
    state["last_abandoned"] = event
    state["inflight"] = None
    producer.atomic_write_json(producer.worker_state_path(root_path, worker), state)

    work = producer.scratch_dir(root_path, worker, int(inflight["batch_id"]))
    scratch_root = root_path / "producer" / "scratch"
    if work.exists() and work.is_relative_to(scratch_root):
        shutil.rmtree(work)
    print(
        f"ABANDONED worker={worker.worker_id} "
        f"batch={int(inflight['batch_id'])}: {reason}",
        flush=True,
    )
    return True


def _produce_one_with_timeout(
    config: Mapping[str, Any],
    root: str | Path,
    worker: producer.WorkerSpec,
    source_contract_sha256: str,
    root_lock: Mapping[str, Any],
    catalog_store: CatalogStore,
    *,
    python: str | Path = sys.executable,
    runner=None,
    failure_point: str | None = None,
    require_catalog_initialized: bool = True,
) -> Path:
    timeout = _generator_timeout_seconds()
    state = producer.load_worker_state(root, worker, source_contract_sha256)
    inflight = state.get("inflight")
    if inflight is not None and inflight.get("prepared_sha256") is None:
        reserved_at = _parse_utc(str(inflight["reserved_at_utc"]))
        age = (datetime.now(timezone.utc) - reserved_at).total_seconds()
        if age > timeout:
            _abandon_inflight(
                root,
                worker,
                source_contract_sha256,
                reason=f"stale_inflight_age_seconds={age:.0f}",
            )

    selected_runner = runner or _run_generator_once_with_timeout
    try:
        return _ORIGINAL_PRODUCE_ONE(
            config,
            root,
            worker,
            source_contract_sha256,
            root_lock,
            catalog_store,
            python=python,
            runner=selected_runner,
            failure_point=failure_point,
            require_catalog_initialized=require_catalog_initialized,
        )
    except GeneratorTimeout as exc:
        _abandon_inflight(
            root,
            worker,
            source_contract_sha256,
            reason=str(exc),
        )
        raise producer.Backpressure(str(exc)) from exc


def _unadopted_staging_npz_bytes(
    root: str | Path,
    *,
    adopted_shard_ids: set[int] | None = None,
    prior: str | None = None,
) -> int:
    del adopted_shard_ids
    root_path = Path(root).expanduser().resolve()
    store = CatalogStore(
        root_path / "rolling_control",
        root_path,
        priors=producer.PRIOR_KEYS,
    )
    adopted = load_cached_adopted_ids(store)
    return _ORIGINAL_STAGING_NPZ_BYTES(
        root_path,
        adopted_shard_ids=adopted,
        prior=prior,
    )


def main(argv: Sequence[str] | None = None) -> int:
    producer.staging_npz_bytes = _unadopted_staging_npz_bytes
    producer.produce_one = _produce_one_with_timeout
    return producer.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
