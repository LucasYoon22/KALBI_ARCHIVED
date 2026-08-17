#!/usr/bin/env python3
"""Generate the compact 300K cache of identifiable Direct96 proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

SOURCE_DIR = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get("FM_PROJECT", str(SOURCE_DIR.parent))).resolve()
sys.path.insert(0, str(PROJECT))

from penalty_relu_mlp_pwl_v33 import batch_segments_to_relu_params
from penalty_sampler_v9 import make_v9

from .canonical_penalty import (
    CONTRACT_VERSION,
    DIRECT_PARAMETER_DIM,
    build_penalty,
    canonical_grid,
)
from .canonical_penalty_cache import CACHE_VERSION, file_sha256


GENERATOR_VERSION = "v12_canonical_direct96_from_v9_segments_300k_v1"


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_direct96(segments: list) -> torch.Tensor:
    knots, b, weights = batch_segments_to_relu_params(
        segments,
        K=96,
        t_max=6.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    if not torch.equal(knots, canonical_grid()):
        raise RuntimeError("converted random penalties use a different V33 grid")
    if weights.shape[1] != 96 or not torch.equal(
        weights[:, -1], torch.zeros_like(weights[:, -1])
    ):
        raise RuntimeError("converted random penalty lacks exact-zero dead weight")
    effective = weights[:, :-1]
    mass_coeff = 0.5 * (6.0 - knots[:-1]).clamp_min(0.0).square()
    mass = 6.0 * b + (effective * mass_coeff).sum(-1)
    if bool((mass <= 1e-8).any().item()):
        raise RuntimeError("converted random penalty has nonpositive rho(6) mass")
    direct = torch.cat(
        [(b / mass)[:, None], effective / mass[:, None]], dim=-1
    ).contiguous()
    if direct.shape[1] != DIRECT_PARAMETER_DIM:
        raise RuntimeError("Direct96 conversion width drift")
    return direct


def generate(count: int, seed: int, chunk_size: int) -> tuple[torch.Tensor, int]:
    sampler = make_v9(seed=int(seed))
    rng = np.random.default_rng(int(seed) + 17)
    output = torch.empty((int(count), DIRECT_PARAMETER_DIM), dtype=torch.float32)
    attempts = 0
    cursor = 0
    started = time.time()
    while cursor < int(count):
        target = min(int(chunk_size), int(count) - cursor)
        segments = []
        while len(segments) < target:
            attempts += 1
            result = sampler.sample_random_penalty(rng)
            if result is not None and result.get("segments") is not None:
                segments.append(result["segments"])
        output[cursor : cursor + target].copy_(normalized_direct96(segments))
        cursor += target
        print(
            json.dumps(
                {
                    "generated": cursor,
                    "count": int(count),
                    "attempts": attempts,
                    "elapsed_seconds": time.time() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return output, attempts


def validate(parameters: torch.Tensor) -> dict:
    indices = torch.linspace(0, parameters.shape[0] - 1, 4096).round().long()
    direct = parameters.index_select(0, indices)
    encoded_a = build_penalty(direct)
    encoded_b = build_penalty(direct.clone())
    exact_fields = (
        "shape_features",
        "d0",
        "rho",
        "drho",
        "relu_b",
        "relu_weights",
    )
    unequal = [key for key in exact_fields if not torch.equal(encoded_a[key], encoded_b[key])]
    if unequal:
        raise RuntimeError(f"canonical builder repeatability failed: {unequal}")
    return {
        "status": "PASS",
        "sampled_rows": int(indices.numel()),
        "exact_repeatability_fields": list(exact_fields),
        "min_drho": float(encoded_a["drho"].amin()),
        "rho6_error_max": float(encoded_a["rho6_error"].amax()),
        "dead_last_weight_max_abs": float(
            encoded_a["relu_weights"][:, -1].abs().amax()
        ),
        "direct_coordinate_std_min": float(parameters.std(0).amin()),
        "direct_coordinate_std_max": float(parameters.std(0).amax()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=74_070_622)
    parser.add_argument("--chunk-size", type=int, default=4096)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    parameters, attempts = generate(args.count, args.seed, args.chunk_size)
    report = validate(parameters)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    torch.save(
        {
            "cache_version": CACHE_VERSION,
            "canonical_contract_version": CONTRACT_VERSION,
            "generator_version": GENERATOR_VERSION,
            "count": int(args.count),
            "seed": int(args.seed),
            "attempts": int(attempts),
            "direct_parameters": parameters,
            "validation": report,
            "source_hashes": {
                "generator": source_sha256(Path(__file__)),
                "canonical_builder": source_sha256(
                    Path(__file__).with_name("canonical_penalty.py")
                ),
                "penalty_sampler_v9": source_sha256(PROJECT / "penalty_sampler_v9.py"),
                "penalty_relu_converter": source_sha256(
                    PROJECT / "penalty_relu_mlp_pwl_v33.py"
                ),
            },
        },
        temporary,
    )
    os.replace(temporary, args.out)
    print(
        json.dumps(
            {
                **report,
                "path": str(args.out.resolve()),
                "sha256": file_sha256(args.out),
                "bytes": args.out.stat().st_size,
                "count": int(args.count),
                "attempts": int(attempts),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
