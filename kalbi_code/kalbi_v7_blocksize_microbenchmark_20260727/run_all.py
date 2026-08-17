#!/usr/bin/env python3
"""Deterministically interleave isolated benchmark cell processes."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from .common import METHODS, OUTPUT, atomic_json_dump


SEED = 202607270256


def append_command_log(command: list[str], status: str, returncode: int | None) -> None:
    record = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": command,
        "shell_command": shlex.join(command),
        "status": status,
        "returncode": returncode,
    }
    path = OUTPUT / "exact_command_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    with (OUTPUT / "exact_commands.txt").open("a", encoding="utf-8") as handle:
        handle.write(
            f"{record['timestamp_utc']}\t{status}\t"
            f"{'' if returncode is None else returncode}\t"
            f"{record['shell_command']}\n"
        )


def run_cell(
    *,
    task_index: int,
    method: str,
    repetition: int,
    warmups: int,
    query_limit: int | None,
    output: Path,
    device: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "kalbi_v7_blocksize_microbenchmark_20260727.benchmark_cell",
        "--task-index",
        str(task_index),
        "--method",
        method,
        "--repetition",
        str(repetition),
        "--warmups",
        str(warmups),
        "--device",
        device,
        "--output",
        str(output),
    ]
    if query_limit is not None:
        command.extend(["--query-limit", str(query_limit)])
    append_command_log(command, "STARTED", None)
    try:
        completed = subprocess.run(
            command,
            cwd="/users/minsuk/fm-lm-lambda",
            env=os.environ.copy(),
            check=False,
            timeout=1800,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.TimeoutExpired:
        result = {
            "status": "TIMEOUT",
            "task_index": task_index,
            "method": method,
            "repetition": repetition,
            "error": "cell process exceeded 1800 seconds",
        }
        atomic_json_dump(result, output)
        append_command_log(command, "PROCESS_TIMEOUT", None)
        return result
    append_command_log(
        command,
        "FINISHED" if completed.returncode == 0 else "FAILED",
        completed.returncode,
    )
    if not output.is_file():
        raise RuntimeError(f"cell did not write output: {output}")
    result = json.loads(output.read_text())
    print(
        f"task={task_index} method={method} repetition={repetition} "
        f"status={result.get('status')}",
        flush=True,
    )
    if completed.returncode != 0:
        print(completed.stdout, flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.mode == "smoke":
        root = OUTPUT / "smoke"
        tasks = (1,)
        repetitions = range(1)
        warmups = 0
        query_limit = 2
    else:
        root = OUTPUT / "fragments"
        tasks = range(1, 7)
        repetitions = range(5)
        warmups = 2
        query_limit = None
    root.mkdir(parents=True, exist_ok=True)
    statuses: list[dict[str, Any]] = []
    for task_index in tasks:
        for repetition in repetitions:
            methods = list(METHODS)
            random.Random(
                SEED + task_index * 10_000 + repetition
            ).shuffle(methods)
            for method in methods:
                output = root / (
                    f"task{task_index:02d}_{method}_rep{repetition:02d}.json"
                )
                if output.exists():
                    existing = json.loads(output.read_text())
                    if existing.get("status") in {"OK", "TIMEOUT", "OOM"}:
                        statuses.append(existing)
                        continue
                    raise RuntimeError(
                        f"existing failed fragment requires audit: {output}"
                    )
                result = run_cell(
                    task_index=task_index,
                    method=method,
                    repetition=repetition,
                    warmups=warmups,
                    query_limit=query_limit,
                    output=output,
                    device=args.device,
                )
                statuses.append(result)
                if result.get("status") not in {"OK", "TIMEOUT", "OOM"}:
                    raise RuntimeError(
                        f"benchmark cell failed: task={task_index}, "
                        f"method={method}, rep={repetition}"
                    )
    summary = {
        "mode": args.mode,
        "cell_count": len(statuses),
        "status_counts": {
            status: sum(item.get("status") == status for item in statuses)
            for status in sorted({item.get("status") for item in statuses})
        },
    }
    atomic_json_dump(summary, root / "run_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
