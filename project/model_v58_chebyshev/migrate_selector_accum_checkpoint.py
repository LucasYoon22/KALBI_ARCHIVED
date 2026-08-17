#!/usr/bin/env python3
"""Create a state-exact V10 selector-accumulation contract checkpoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


NEW_TRAINER_VERSION = "v58cheb_v10_joint_small_n200_p400_selector_accum4_v2"
NEW_CHECKPOINT_VERSION = "v58cheb_v10_joint_checkpoint_selector_accum4_v2"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_exact(left: Any, right: Any, path: str = "root") -> None:
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or not torch.equal(left, right):
            raise RuntimeError(f"tensor drift at {path}")
        return
    if isinstance(left, np.ndarray):
        if not isinstance(right, np.ndarray) or not np.array_equal(left, right):
            raise RuntimeError(f"array drift at {path}")
        return
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise RuntimeError(f"mapping schema drift at {path}")
        for key in left:
            assert_exact(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            raise RuntimeError(f"sequence schema drift at {path}")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            assert_exact(left_item, right_item, f"{path}[{index}]")
        return
    if left != right:
        raise RuntimeError(f"value drift at {path}: {left!r} != {right!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--new-trainer", type=Path, required=True)
    parser.add_argument("--selector-grad-accum-steps", type=int, default=4)
    args = parser.parse_args()

    if args.selector_grad_accum_steps != 4:
        parser.error("this migration is pinned to four selector accumulation steps")
    actual_source_sha256 = file_sha256(args.source)
    if actual_source_sha256 != args.source_sha256:
        raise RuntimeError("source checkpoint SHA256 mismatch")
    if args.destination.exists():
        raise RuntimeError(f"destination already exists: {args.destination}")

    original = torch.load(args.source, map_location="cpu", weights_only=False)
    migrated = copy.deepcopy(original)
    metadata = migrated["metadata"]
    exact_arguments = metadata["exact_training_arguments"]
    if "selector_grad_accum_steps" in exact_arguments:
        raise RuntimeError("source checkpoint already has selector accumulation")
    if int(exact_arguments["batch_per_rank"]) != 1:
        raise RuntimeError("migration expects batch_per_rank=1")

    trainer_sha256 = file_sha256(args.new_trainer)
    old_trainer_version = metadata["trainer_version"]
    old_checkpoint_version = metadata["checkpoint_version"]
    metadata["trainer_version"] = NEW_TRAINER_VERSION
    metadata["checkpoint_version"] = NEW_CHECKPOINT_VERSION
    exact_arguments["selector_grad_accum_steps"] = args.selector_grad_accum_steps
    metadata["source_hashes"]["trainer"] = trainer_sha256
    metadata["selector_gradient_accumulation_migration"] = {
        "source_checkpoint": str(args.source.resolve()),
        "source_checkpoint_sha256": actual_source_sha256,
        "old_global_selector_task_batch": 8,
        "new_global_selector_task_batch": 32,
        "old_trainer_version": old_trainer_version,
        "new_trainer_version": NEW_TRAINER_VERSION,
        "old_checkpoint_version": old_checkpoint_version,
        "new_checkpoint_version": NEW_CHECKPOINT_VERSION,
        "selector_grad_accum_steps": args.selector_grad_accum_steps,
        "all_nonmetadata_state_preserved_exact": True,
    }

    temporary = args.destination.with_suffix(args.destination.suffix + ".tmp")
    torch.save(migrated, temporary)
    os.replace(temporary, args.destination)
    reloaded = torch.load(args.destination, map_location="cpu", weights_only=False)

    original_nonmetadata = {
        key: value for key, value in original.items() if key != "metadata"
    }
    migrated_nonmetadata = {
        key: value for key, value in reloaded.items() if key != "metadata"
    }
    assert_exact(original_nonmetadata, migrated_nonmetadata, "nonmetadata")
    assert_exact(migrated["metadata"], reloaded["metadata"], "metadata")

    print(
        json.dumps(
            {
                "destination": str(args.destination),
                "destination_sha256": file_sha256(args.destination),
                "new_trainer_sha256": trainer_sha256,
                "nonmetadata_exact": True,
                "selector_grad_accum_steps": args.selector_grad_accum_steps,
                "source": str(args.source),
                "source_sha256": actual_source_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
