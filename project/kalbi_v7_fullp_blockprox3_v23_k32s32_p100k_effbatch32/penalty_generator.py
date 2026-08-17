#!/usr/bin/env python3
"""Generate the random-penalty segment cache used by standalone V58.

The serialized cache stores continuous penalty segments, not an encoded
convexity flag.  V58 training reconstructs the PWL grid and emits its native
196-D/meta4 public tensor at load time.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

SOURCE_DIR = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get("FM_PROJECT", str(SOURCE_DIR.parent))).resolve()
sys.path.insert(0, str(PROJECT))

from penalty_sampler_v9 import make_v9


GENERATOR_VERSION = "v58_penalty_segments_v9_native196_loader_v1"
DEFAULT_COUNT = 100_000
DEFAULT_SEED = 74_070_622


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_segments(count: int, seed: int) -> tuple[list, int]:
    if int(count) <= 0:
        raise ValueError("count must be positive")
    sampler = make_v9(seed=int(seed))
    rng = np.random.default_rng(int(seed) + 17)
    segments: list = []
    attempts = 0
    started = time.time()
    while len(segments) < int(count):
        attempts += 1
        result = sampler.sample_random_penalty(rng)
        if result is None:
            continue
        segment = result.get("segments")
        if segment is None:
            raise RuntimeError("V9 sampler returned a penalty without segments")
        segments.append(segment)
        if len(segments) % 5_000 == 0:
            print(
                f"generated {len(segments)}/{count} attempts={attempts} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    return segments, attempts


def save_cache(path: Path, *, segments: list, attempts: int, seed: int) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing cache: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "segments": segments,
            "sampler_version": "V9",
            "generator_version": GENERATOR_VERSION,
            "num_penalties": len(segments),
            "attempts": int(attempts),
            "seed": int(seed),
            "public_penalty_contract": (
                "native196_interleaved_rho_drho_meta4_expanded388_v1"
            ),
            "convexity_flag_serialized": False,
        },
        temporary,
    )
    os.replace(temporary, path)


def validate_cache(path: Path, expected_count: int | None = None) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    segments = payload.get("segments")
    if not isinstance(segments, list) or not segments:
        raise RuntimeError("cache does not contain a nonempty segment list")
    if expected_count is not None and len(segments) != int(expected_count):
        raise RuntimeError(
            f"cache count {len(segments)} != expected {expected_count}"
        )
    if payload.get("convexity_flag_serialized", False):
        raise RuntimeError("V58 cache must not serialize a convexity flag")
    return {
        "status": "PASS",
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "count": len(segments),
        "sampler_version": payload.get("sampler_version"),
        "generator_version": payload.get("generator_version"),
        "seed": payload.get("seed"),
        "attempts": payload.get("attempts"),
        "convexity_flag_serialized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", "--n", dest="count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out", type=Path, default=Path("penalty_cache_100k_v9_v58.pt")
    )
    parser.add_argument("--validate-only", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only is not None:
        print(validate_cache(args.validate_only, args.count))
        return
    started = time.time()
    segments, attempts = generate_segments(args.count, args.seed)
    save_cache(args.out, segments=segments, attempts=attempts, seed=args.seed)
    report = validate_cache(args.out, args.count)
    report["elapsed_seconds"] = time.time() - started
    print(report, flush=True)


if __name__ == "__main__":
    main()
