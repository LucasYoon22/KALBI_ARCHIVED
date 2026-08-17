#!/usr/bin/env python3
"""Audit the frozen-beta contract of a selector-only checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .train import state_dict_sha256
from .train_selector_only_v23_frozen_kalbi_v6_8gpu import (
    CHECKPOINT_VERSION,
    TRAINER_VERSION,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    source = torch.load(
        args.source, map_location="cpu", weights_only=False
    )
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    metadata = checkpoint["metadata"]
    assert metadata["trainer_version"] == TRAINER_VERSION
    assert metadata["checkpoint_version"] == CHECKPOINT_VERSION
    assert metadata["training_mode"] == "selector_only_frozen_beta"
    assert "beta_optimizer_state_dict" not in checkpoint
    assert "beta_scheduler_state_dict" not in checkpoint
    assert "selector_optimizer_state_dict" in checkpoint
    assert len(checkpoint["rank_states"]) == 8

    source_beta_hash = state_dict_sha256(source["model_state_dict"])
    checkpoint_beta_hash = state_dict_sha256(
        checkpoint["model_state_dict"]
    )
    source_selector_hash = state_dict_sha256(
        source["selector_state_dict"]
    )
    checkpoint_selector_hash = state_dict_sha256(
        checkpoint["selector_state_dict"]
    )
    assert source_beta_hash == checkpoint_beta_hash
    assert source_selector_hash != checkpoint_selector_hash
    assert metadata["frozen_beta_state_sha256"] == checkpoint_beta_hash

    print(json.dumps({
        "status": "PASS",
        "selector_step": int(checkpoint["completed_steps"]),
        "source_joint_step": int(metadata["source_joint_step"]),
        "beta_state_sha256": checkpoint_beta_hash,
        "selector_changed_from_source": True,
        "beta_optimizer_present": False,
        "beta_scheduler_present": False,
        "selector_optimizer_present": True,
        "rank_state_count": len(checkpoint["rank_states"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
