#!/usr/bin/env python3
"""Extend a completed Kalbi checkpoint to a larger cosine-schedule target."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cosine_multiplier(step: int, warmup_steps: int, target_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(
        max(1, target_steps - warmup_steps)
    )
    return 0.5 * (
        1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--target-steps", type=int, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    observed_sha256 = sha256(args.input)
    if observed_sha256 != args.expected_input_sha256:
        raise RuntimeError("input checkpoint SHA256 mismatch")

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    completed_steps = int(checkpoint["completed_steps"])
    metadata = checkpoint["metadata"]
    exact = metadata["exact_training_arguments"]
    old_target_steps = int(exact["steps"])
    warmup_steps = int(exact["warmup_steps"])
    if completed_steps != old_target_steps:
        raise RuntimeError(
            "target extension requires a checkpoint completed at its old target"
        )
    if args.target_steps <= old_target_steps:
        raise RuntimeError("new target must be larger than the old target")

    scheduler = checkpoint["beta_scheduler_state_dict"]
    if int(scheduler["last_epoch"]) != completed_steps:
        raise RuntimeError("scheduler epoch does not match completed steps")
    base_lrs = [float(value) for value in scheduler["base_lrs"]]
    optimizer_groups = checkpoint["beta_optimizer_state_dict"]["param_groups"]
    if len(base_lrs) != len(optimizer_groups):
        raise RuntimeError("optimizer/scheduler parameter-group mismatch")

    multiplier = cosine_multiplier(
        completed_steps, warmup_steps, args.target_steps
    )
    resumed_lrs = [base_lr * multiplier for base_lr in base_lrs]
    for group, lr in zip(optimizer_groups, resumed_lrs, strict=True):
        group["lr"] = lr
    scheduler["_last_lr"] = resumed_lrs

    exact["steps"] = int(args.target_steps)
    metadata["target_steps_extension"] = {
        "completed_steps": completed_steps,
        "old_target_steps": old_target_steps,
        "new_target_steps": int(args.target_steps),
        "source_checkpoint": str(args.input.resolve()),
        "source_checkpoint_sha256": observed_sha256,
        "scheduler_multiplier_at_resume": multiplier,
        "resumed_beta_lrs": resumed_lrs,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, args.output)
    print(f"output={args.output}")
    print(f"sha256={sha256(args.output)}")
    print(f"completed_steps={completed_steps}")
    print(f"target_steps={args.target_steps}")
    print(f"scheduler_multiplier={multiplier:.12g}")
    print(f"beta_lrs={resumed_lrs}")


if __name__ == "__main__":
    main()
