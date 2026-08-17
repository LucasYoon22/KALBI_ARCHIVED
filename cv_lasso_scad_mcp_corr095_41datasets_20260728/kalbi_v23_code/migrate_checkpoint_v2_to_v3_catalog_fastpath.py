#!/usr/bin/env python3
"""Migrate a V2 checkpoint after the catalog-runtime-only optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import train_joint_v17_stagegain_rolling_8gpu as trainer
from .train import atomic_torch_save, file_sha256, state_dict_sha256


OLD_TRAINER_VERSION = (
    "kalbi_v2_v17_stagegain_scratch_p100k_world8_effbatch32_"
    "abfilm_k32s32_ridge1e-6_rolling100gb_reload100_v2"
)
OLD_CHECKPOINT_VERSION = "kalbi_v2_v17_stagegain_p100k_effbatch32_checkpoint_v2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if file_sha256(args.input) != args.input_sha256:
        raise RuntimeError("input checkpoint SHA256 mismatch")
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite migration output: {args.output}")

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    if metadata.get("trainer_version") != OLD_TRAINER_VERSION:
        raise RuntimeError("input trainer version is not V2")
    if metadata.get("checkpoint_version") != OLD_CHECKPOINT_VERSION:
        raise RuntimeError("input checkpoint version is not V2")

    model_hash = state_dict_sha256(payload["model_state_dict"])
    selector_hash = state_dict_sha256(payload["selector_state_dict"])
    source_hashes = dict(metadata["source_hashes"])
    trainer_path = Path(trainer.__file__).resolve()
    source_hashes["trainer"] = file_sha256(trainer_path)
    source_hashes["rolling_catalog"] = file_sha256(
        trainer_path.with_name("rolling_catalog.py")
    )

    metadata["trainer_version"] = trainer.TRAINER_VERSION
    metadata["checkpoint_version"] = trainer.CHECKPOINT_VERSION
    metadata["source_hashes"] = source_hashes
    metadata["runtime_migration"] = {
        "from_checkpoint_version": OLD_CHECKPOINT_VERSION,
        "from_trainer_version": OLD_TRAINER_VERSION,
        "input_checkpoint": str(args.input.resolve()),
        "input_checkpoint_sha256": args.input_sha256,
        "reason": (
            "CURRENT catalog O(1) ACK/lease lookup and rank-local path checks; "
            "model, optimizer, scheduler, sampler, and RNG states unchanged"
        ),
    }

    atomic_torch_save(payload, args.output)
    reloaded = torch.load(args.output, map_location="cpu", weights_only=False)
    if int(reloaded["completed_steps"]) != int(payload["completed_steps"]):
        raise RuntimeError("completed_steps changed during migration")
    if state_dict_sha256(reloaded["model_state_dict"]) != model_hash:
        raise RuntimeError("model state changed during migration")
    if state_dict_sha256(reloaded["selector_state_dict"]) != selector_hash:
        raise RuntimeError("selector state changed during migration")
    if reloaded["metadata"]["source_hashes"] != source_hashes:
        raise RuntimeError("migrated source hash contract failed strict reload")

    print(
        json.dumps(
            {
                "completed_steps": int(reloaded["completed_steps"]),
                "input": str(args.input.resolve()),
                "input_sha256": args.input_sha256,
                "model_state_sha256": model_hash,
                "output": str(args.output.resolve()),
                "output_sha256": file_sha256(args.output),
                "selector_state_sha256": selector_hash,
                "target_checkpoint_version": trainer.CHECKPOINT_VERSION,
                "target_trainer_version": trainer.TRAINER_VERSION,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
