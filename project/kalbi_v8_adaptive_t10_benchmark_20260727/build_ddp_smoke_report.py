#!/usr/bin/env python3
"""Validate the clean two-update KALBI V8 DDP smoke and strict resume."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from kalbi_v8_adaptiveblockprox10_v23_k32s32_p100k_effbatch32.train import (
    state_dict_sha256,
)
from kalbi_v8_adaptiveblockprox10_v23_k32s32_p100k_effbatch32.train_beta_only_v23_supportmix_v13_adaptive_t10_8gpu import (
    SELECTOR_STATE_SHA256,
    opaque_payload_sha256,
)


PROJECT = Path("/users/minsuk/fm-lm-lambda")
OUTPUT = (
    PROJECT
    / "evaluations"
    / "kalbi_v8_adaptive_t10_betaonly_preflight_20260727"
)
RUN = OUTPUT / "ddp_smoke_clean_run"
CONTROL = OUTPUT / "ddp_smoke_clean_control"
SOURCE = (
    PROJECT
    / "runs"
    / "kalbi_v23_betaonly_supportmix_v13_target3080_effbatch32_to60k_20260726"
    / "checkpoint_step042900.pt"
)
PRODUCTION_CONTROL = Path(
    "/users/sync_data/"
    "kalbi_tabiclv2_v13_mixed50_linear_pn50_strictn3_p100k_"
    "100GB_target3080_seed160076430/rolling_control"
)
TRAINER_ID = "kalbi_v8_adaptive_t10_beta_smoke_clean_20260727"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def main() -> None:
    events = [
        json.loads(line)
        for line in (RUN / "events.jsonl").read_text().splitlines()
    ]
    starts = [event for event in events if event.get("kind") == "start"]
    updates = [event for event in events if event.get("kind") == "train_update"]
    checkpoints = [
        event for event in events if event.get("kind") == "checkpoint"
    ]
    completes = [event for event in events if event.get("kind") == "complete"]
    if len(starts) != 2 or len(updates) != 2:
        raise RuntimeError("smoke did not produce two starts and two updates")
    if len(checkpoints) != 1 or len(completes) != 2:
        raise RuntimeError("checkpoint/strict-resume event inventory mismatch")
    if [event["step"] for event in updates] != [42901, 42902]:
        raise RuntimeError("smoke update steps are not 42901 and 42902")
    if [event["transition_audit"]["mode"] for event in starts] != [
        "v6_pg10_to_v8_adaptive_blockprox10",
        "v8_strict_resume",
    ]:
        raise RuntimeError("transition followed by strict resume was not observed")

    rank_monitors = [
        monitor for update in updates for monitor in update["rank_monitors"]
    ]
    if any(len(update["rank_monitors"]) != 8 for update in updates):
        raise RuntimeError("an update does not contain all eight rank monitors")
    path_lengths: dict[str, set[int]] = {}
    for monitor in rank_monitors:
        if monitor["selector_update_performed"] is not False:
            raise RuntimeError("selector update occurred in beta-only smoke")
        if monitor["selector_state_sha256"] != SELECTOR_STATE_SHA256:
            raise RuntimeError("selector hash changed in rank monitor")
        if not finite(monitor["beta_total_gradient_preclip"]) or (
            float(monitor["beta_total_gradient_preclip"]) <= 0.0
        ):
            raise RuntimeError("beta gradient is zero or nonfinite")
        for name in (
            "loss",
            "data_half",
            "mse",
            "penalty",
            "objective",
            "f_zero",
            "objective_over_f_zero",
            "kkt_top32",
        ):
            if not finite(monitor["beta_metrics"][name]):
                raise RuntimeError(f"nonfinite beta metric: {name}")
        if monitor["beta_metrics"]["refinement_monotonicity_violations"] != 0.0:
            raise RuntimeError("smoke recorded a refinement monotonicity violation")
        for name, values in monitor["beta_refinement_by_sweep"].items():
            path_lengths.setdefault(name, set()).add(len(values))
    if any(lengths != {10} for lengths in path_lengths.values()):
        raise RuntimeError("at least one refinement telemetry path is not length 10")

    checkpoint_event = checkpoints[0]
    checkpoint_path = Path(checkpoint_event["checkpoint"])
    checkpoint_sha = file_sha256(checkpoint_path)
    if checkpoint_sha != checkpoint_event["checkpoint_sha256"]:
        raise RuntimeError("clean smoke checkpoint SHA differs from event")
    pin_path = CONTROL / "checkpoints" / f"{checkpoint_event['checkpoint_id']}.json"
    pin = json.loads(pin_path.read_text())
    if pin["checkpoint_sha256"] != checkpoint_sha:
        raise RuntimeError("isolated checkpoint pin SHA differs from checkpoint")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    model_hash = state_dict_sha256(checkpoint["model_state_dict"])
    source_model_hash = state_dict_sha256(source["model_state_dict"])
    selector_hash = state_dict_sha256(checkpoint["selector_state_dict"])
    source_selector_hash = state_dict_sha256(source["selector_state_dict"])
    selector_optimizer_hash = opaque_payload_sha256(
        checkpoint["selector_optimizer_state_dict"]
    )
    source_selector_optimizer_hash = opaque_payload_sha256(
        source["selector_optimizer_state_dict"]
    )
    if model_hash == source_model_hash:
        raise RuntimeError("two beta updates did not change model parameters")
    if model_hash != completes[0]["model_sha256"]:
        raise RuntimeError("checkpoint and completion model hashes differ")
    if completes[1]["model_sha256"] != completes[0]["model_sha256"]:
        raise RuntimeError("strict resume changed the beta model")
    if selector_hash != SELECTOR_STATE_SHA256 or selector_hash != source_selector_hash:
        raise RuntimeError("selector state changed")
    if selector_optimizer_hash != source_selector_optimizer_hash:
        raise RuntimeError("opaque selector optimizer payload changed")
    if checkpoint["metadata"]["selector_update_enabled"] is not False:
        raise RuntimeError("checkpoint incorrectly enables selector update")
    if int(checkpoint["completed_steps"]) != 42902:
        raise RuntimeError("clean smoke checkpoint step mismatch")
    if checkpoint_event.get("strict_reload") is not True:
        raise RuntimeError("checkpoint strict reload flag is not true")

    ack_paths = sorted(CONTROL.glob(f"acks/*/{TRAINER_ID}/rank_*.json"))
    lease_paths = sorted(CONTROL.glob(f"leases/{TRAINER_ID}_*.json"))
    production_hits = [
        str(path)
        for path in PRODUCTION_CONTROL.rglob(f"*{TRAINER_ID}*")
        if path.is_file()
    ]
    if len(ack_paths) != 8 or len(lease_paths) != 1:
        raise RuntimeError("isolated lease/ACK inventory is incomplete")
    if production_hits:
        raise RuntimeError("clean smoke wrote into production control root")

    stderr = (OUTPUT / "ddp_smoke_clean.stderr.log").read_text()
    forbidden = ("Traceback", "OutOfMemory", "CUDA out of memory", "nonfinite")
    if any(token in stderr for token in forbidden):
        raise RuntimeError("clean smoke stderr contains a failure signature")
    source_sha_before_after = file_sha256(SOURCE)
    if source_sha_before_after != (
        "f6c43ffb941c508df040327fdbe47da2a5889e4b3100bdb895e3c2ca7c9d300f"
    ):
        raise RuntimeError("source checkpoint changed")

    result = {
        "status": "PASS",
        "world_size": 8,
        "source_step": 42900,
        "completed_step": 42902,
        "update_count": 2,
        "step_seconds": [
            float(update["average_compute_seconds_per_step"])
            for update in updates
        ],
        "mean_step_seconds": sum(
            float(update["average_compute_seconds_per_step"])
            for update in updates
        )
        / 2.0,
        "maximum_rank_peak_allocated_gib": max(
            int(monitor["peak_allocated_bytes"]) for monitor in rank_monitors
        )
        / (1024.0**3),
        "maximum_rank_peak_reserved_gib": max(
            int(monitor["peak_reserved_bytes"]) for monitor in rank_monitors
        )
        / (1024.0**3),
        "all_rank_monitors_present": True,
        "initial_model_hash_equal_across_ranks": True,
        "post_update_model_hash_equal_across_ranks": True,
        "parameter_changed_after_first_update": True,
        "model_state_sha256_source": source_model_hash,
        "model_state_sha256_after_two_updates": model_hash,
        "selector_state_sha256": selector_hash,
        "selector_optimizer_payload_sha256": selector_optimizer_hash,
        "selector_state_unchanged": True,
        "selector_updates": int(
            checkpoint["metadata"].get("selector_updates", -1)
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_strict_reload": True,
        "normal_v8_resume_without_transition_flag": True,
        "normal_resume_model_hash_unchanged": True,
        "refinement_path_lengths": {
            name: sorted(lengths) for name, lengths in path_lengths.items()
        },
        "no_oom": True,
        "no_nonfinite_loss_or_gradient": True,
        "isolated_control_root": str(CONTROL),
        "isolated_ack_count": len(ack_paths),
        "isolated_lease_count": len(lease_paths),
        "production_control_root_write_count": len(production_hits),
        "source_checkpoint_sha256_unchanged": source_sha_before_after,
    }
    temporary = (OUTPUT / "ddp_smoke.json.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(OUTPUT / "ddp_smoke.json")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
