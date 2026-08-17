#!/usr/bin/env python3
"""Write a read-only KALBI V8 server and rolling-pipeline audit snapshot."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


DATA_ROOT = Path(
    "/users/sync_data/"
    "kalbi_tabiclv2_v13_mixed50_linear_pn50_strictn3_p100k_"
    "100GB_target3080_seed160076430"
)
CONTROL_ROOT = DATA_ROOT / "rolling_control"
EXPECTED_PIPELINE_SHA256 = (
    "99f81c6a5bf5083a56f3268a56a0700cd64865b1ff0bba9b23386088d11a9559"
)
LOG_ROOT = Path("/users/minsuk/fm-lm-lambda/logs/kalbi_v13_beta_rolling")


def command(*arguments: str, allowed_returncodes: tuple[int, ...] = (0,)) -> str:
    completed = subprocess.run(
        arguments,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode not in allowed_returncodes:
        raise RuntimeError(
            f"command failed with exit {completed.returncode}: {arguments}\n"
            f"{completed.stdout}"
        )
    return completed.stdout


def directory_digest(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    size = 0
    for candidate in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = candidate.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        with candidate.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
                size += len(block)
        count += 1
    return {"sha256": digest.hexdigest(), "files": count, "bytes": size}


def file_state(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def parse_catalog_summary(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        prefix = handle.read(8192).decode("utf-8", errors="replace")
        handle.seek(max(0, path.stat().st_size - 131072))
        suffix = handle.read().decode("utf-8", errors="replace")
    total = re.search(r'"total_bytes":\s*(\d+)', prefix)
    files = re.search(r'"num_files":\s*(\d+)', prefix)
    pipeline = re.search(
        r'"pipeline_config_sha256":\s*"([0-9a-f]{64})"', suffix
    )
    if total is None or files is None or pipeline is None:
        raise RuntimeError("could not parse immutable catalog summary")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "num_files": int(files.group(1)),
        "active_bytes": int(total.group(1)),
        "pipeline_config_sha256": pipeline.group(1),
        "pipeline_sha_matches": pipeline.group(1) == EXPECTED_PIPELINE_SHA256,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pointer = json.loads((CONTROL_ROOT / "CURRENT.json").read_text())
    catalog = CONTROL_ROOT / "catalogs" / pointer["catalog_file"]
    supervisor = command(
        "supervisorctl", "status", allowed_returncodes=(0, 3)
    )
    relevant = [
        line
        for line in supervisor.splitlines()
        if line.startswith(("kalbi_v13_", "kalbi_v24_", "kalbi_v8_"))
    ]
    producer_logs = {
        f"producer_{index:02d}": {
            stream: file_state(
                LOG_ROOT / f"producer_{index:02d}.{stream}.log"
            )
            for stream in ("stdout", "stderr")
        }
        for index in range(8)
    }
    tombstones = sorted(
        (CONTROL_ROOT / "tombstones").glob("*.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    disk = shutil.disk_usage(DATA_ROOT)
    gpu_lines = command(
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ).splitlines()
    payload = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rolling_root": str(DATA_ROOT),
        "control_root": str(CONTROL_ROOT),
        "expected_pipeline_sha256": EXPECTED_PIPELINE_SHA256,
        "current_pointer": pointer,
        "current_pointer_state": file_state(CONTROL_ROOT / "CURRENT.json"),
        "current_catalog": parse_catalog_summary(catalog),
        "next_shard_id_state": file_state(
            CONTROL_ROOT / "NEXT_SHARD_ID.json"
        ),
        "newest_tombstone": (
            None if not tombstones else file_state(tombstones[-1])
        ),
        "producer_logs": producer_logs,
        "supervisor_relevant": relevant,
        "curator_running": any(
            line.startswith("kalbi_v13_beta_curator")
            and "RUNNING" in line
            for line in relevant
        ),
        "producer_running_count": sum(
            line.startswith("kalbi_v13_beta_producer_")
            and "RUNNING" in line
            for line in relevant
        ),
        "production_control_mutable_digests": {
            name: directory_digest(CONTROL_ROOT / name)
            for name in ("leases", "acks", "checkpoints")
        },
        "gpu": gpu_lines,
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
    }
    if not payload["curator_running"]:
        raise RuntimeError("V13 curator is not RUNNING")
    if payload["producer_running_count"] != 8:
        raise RuntimeError("not all eight V13 producers are RUNNING")
    if not payload["current_catalog"]["pipeline_sha_matches"]:
        raise RuntimeError("rolling pipeline SHA mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
