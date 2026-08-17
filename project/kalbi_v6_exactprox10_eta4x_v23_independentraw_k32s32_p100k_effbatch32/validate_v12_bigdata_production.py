#!/usr/bin/env python3
"""Fail-closed audit for the V12 start, step-100 telemetry, and step-500 reload."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from .canonical_penalty import CONTRACT_VERSION
from .canonical_penalty_cache import CACHE_VERSION
from .selector_v12_canonical import (
    DirectPenaltyMLPQuerySelectorV12Canonical,
    V12_HIDDEN_DIM,
    V12_MLP_DEPTH,
    V12_SELECTOR_VERSION,
)
from .train_joint_v12_canonical_rolling import (
    CHECKPOINT_VERSION,
    SELECTOR_PATH_FEATURES,
    TRAINER_VERSION,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_events(path: Path) -> list[dict[str, Any]]:
    events = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid event JSON at line {line_number}") from exc
        require(isinstance(value, dict), f"event line {line_number} is not an object")
        events.append(value)
    return events


def one_event(events: list[dict[str, Any]], kind: str, step: int | None = None) -> dict[str, Any]:
    matches = [
        event
        for event in events
        if event.get("kind") == kind and (step is None or int(event.get("step", -1)) == step)
    ]
    require(len(matches) == 1, f"expected one {kind!r} event at step={step}, got {len(matches)}")
    return matches[0]


def positive(value: Any, label: str) -> float:
    value = float(value)
    require(math.isfinite(value) and value > 0.0, f"{label} is not finite-positive")
    return value


def validate_monitor(monitor: Mapping[str, Any], expected_catalog: str) -> dict[str, Any]:
    rank = int(monitor["rank"])
    require(monitor["catalog_sha256"] == expected_catalog, f"rank {rank} catalog drift")
    require(float(monitor["sampled_lambda_min"]) >= 1e-6, f"rank {rank} lambda floor failed")
    require(float(monitor["penalty_min_drho"]) > 0.0, f"rank {rank} penalty lost monotonicity")
    require(float(monitor["penalty_rho6_error_max"]) <= 1e-5, f"rank {rank} rho(6) drift")
    positive(monitor["positivity_jacobian_min"], f"rank {rank} positivity Jacobian")
    positive(monitor["penalty_rho_batch_std_mean"], f"rank {rank} penalty diversity")
    direct = monitor["direct96_gradient"]
    require(int(direct["active_coordinates"]) == 96, f"rank {rank} Direct96 coverage")
    require(int(direct["total_coordinates"]) == 96, f"rank {rank} Direct96 width")
    positive(direct["norm"], f"rank {rank} Direct96 gradient")
    for name in ("native196_gradient", "d0_gradient", "lambda_gradient"):
        positive(monitor[name]["norm"], f"rank {rank} {name}")
    for name in ("penalty_head", "lambda_head", "penalty_trunk", "lambda_trunk"):
        positive(monitor["selector_group_gradients"][name]["norm"], f"rank {rank} {name}")
    for name in ("feature", "local", "query", "shape"):
        positive(monitor["beta_group_gradients"][name], f"rank {rank} beta {name}")
    return {
        "rank": rank,
        "direct96_active": int(direct["active_coordinates"]),
        "positivity_jacobian_min": float(monitor["positivity_jacobian_min"]),
        "penalty_min_drho": float(monitor["penalty_min_drho"]),
        "sampled_lambda_min": float(monitor["sampled_lambda_min"]),
        "penalty_rho_batch_std_mean": float(monitor["penalty_rho_batch_std_mean"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--trainer-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    events = load_events(args.run_dir / "events.jsonl")
    start_candidates = [event for event in events if event.get("kind") in ("start", "resume")]
    require(start_candidates and start_candidates[0].get("kind") == "start", "missing initial start event")
    start = start_candidates[0]
    require(start["trainer_version"] == TRAINER_VERSION, "trainer version drift")
    require(start["checkpoint_version"] == CHECKPOINT_VERSION, "checkpoint version drift")
    require(start["selector_version"] == V12_SELECTOR_VERSION, "selector version drift")
    require(start["canonical_penalty_contract"] == CONTRACT_VERSION, "penalty contract drift")
    require(start["canonical_cache"]["cache_version"] == CACHE_VERSION, "cache version drift")
    require(int(start["canonical_cache"]["count"]) == 300_000, "cache count drift")
    require(int(start["world_size"]) == 8, "world size drift")
    require(int(start["global_beta_task_batch"]) == 32, "beta batch drift")
    require(int(start["global_selector_task_batch"]) == 32, "selector batch drift")
    require(int(start["reload_interval_updates"]) == 500, "reload interval drift")
    require(bool(start["selector_uses_actual_hard_beta_q"]), "actual hard beta_q is disabled")
    require(not bool(start["selector_inactive_ste_used"]), "inactive STE was re-enabled")
    require(bool(start["warm_beta_optimizer_resumed"]), "warm beta optimizer was not resumed")
    require(bool(start["selector_optimizer_reset_for_new_contract"]), "V12 selector optimizer was not reset")
    init = start["selector_initialization"]
    require(not bool(init["warm_v11_selector_loaded"]), "incompatible V11 selector was loaded")
    architecture = init["architecture"]
    require(int(architecture["hidden_dim"]) == V12_HIDDEN_DIM, "hidden width drift")
    require(int(architecture["all_mlp_depth"]) == V12_MLP_DEPTH, "MLP depth drift")
    require(int(architecture["total_parameters"]) == 5_161_389, "selector parameter-count drift")
    require(98_000_000_000 <= int(start["active_catalog_bytes"]) <= 102_000_000_000, "start catalog size drift")

    step100 = one_event(events, "train_update", 100)
    monitors = step100["rank_monitors"]
    require(len(monitors) == 8, "step-100 rank monitor count drift")
    monitor_report = [validate_monitor(item, start["active_catalog_sha256"]) for item in monitors]
    require(sorted(item["rank"] for item in monitor_report) == list(range(8)), "rank inventory drift")

    reload_event = one_event(events, "catalog_reload_checkpoint", 500)
    require(bool(reload_event["stream_cache_cleared_all_ranks"]), "stream caches were not cleared")
    require(bool(reload_event["reload_verified"]), "checkpoint reload verification missing")
    require(98_000_000_000 <= int(reload_event["catalog_bytes"]) <= 102_000_000_000, "step-500 catalog size drift")
    checkpoint_path = Path(reload_event["checkpoint"])
    require(checkpoint_path.is_file(), "step-500 checkpoint is missing")
    checkpoint_sha = sha256_file(checkpoint_path)
    require(checkpoint_sha == reload_event["checkpoint_sha256"], "step-500 checkpoint hash drift")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(int(checkpoint["completed_steps"]) == 500, "checkpoint completed-step drift")
    metadata = checkpoint["metadata"]
    require(metadata["checkpoint_version"] == CHECKPOINT_VERSION, "checkpoint metadata version drift")
    require(metadata["trainer_version"] == TRAINER_VERSION, "checkpoint trainer version drift")
    require(metadata["selector_version"] == V12_SELECTOR_VERSION, "checkpoint selector version drift")
    require(bool(metadata["catalog_reloaded_before_checkpoint"]), "catalog was not reloaded before checkpoint")
    require(bool(metadata["dataloader_cache_cleared_before_checkpoint"]), "dataloader cache was not cleared")
    require(int(metadata["reload_interval_updates"]) == 500, "checkpoint reload interval drift")
    require(int(metadata["world_size"]) == 8, "checkpoint world-size drift")
    require(len(checkpoint["rank_states"]) == 8, "checkpoint rank-state count drift")
    require(sorted(int(item["rank"]) for item in checkpoint["rank_states"]) == list(range(8)), "checkpoint rank-state inventory drift")
    require(checkpoint["catalog_pointer"]["catalog_sha256"] == reload_event["catalog_sha256"], "checkpoint catalog pointer drift")

    selector = DirectPenaltyMLPQuerySelectorV12Canonical(
        ds_feat_dim=9,
        t_feat_dim=32 * 33 // 2,
        krylov_dim=32,
        shape_dim=196,
        path_feat_dim=SELECTOR_PATH_FEATURES,
    )
    selector.load_state_dict(checkpoint["selector_state_dict"], strict=True)
    require(sum(parameter.numel() for parameter in selector.parameters()) == 5_161_389, "strict V12 selector size drift")
    optimizer_groups = checkpoint["selector_optimizer_state_dict"]["param_groups"]
    require(len(optimizer_groups) == 2, "selector optimizer group-count drift")
    group_lrs = {str(group.get("name")): float(group["lr"]) for group in optimizer_groups}
    require(group_lrs == {"selector_context_lambda_trunks": 3e-4, "direct96_penalty_head": 1e-5}, "selector optimizer LR drift")

    pin_path = args.control_root / "checkpoints" / f"{reload_event['checkpoint_id']}.json"
    require(pin_path.is_file(), "step-500 checkpoint pin is missing")
    pin = json.loads(pin_path.read_text())
    require(pin["checkpoint_sha256"] == checkpoint_sha, "checkpoint pin hash drift")
    require(pin["catalog_sha256"] == reload_event["catalog_sha256"], "checkpoint pin catalog drift")
    require(bool(pin["resumable"]) and bool(pin["reload_verified"]), "checkpoint pin is not verified-resumable")

    ack_root = (
        args.control_root
        / "acks"
        / f"epoch_{int(reload_event['catalog_epoch']):08d}_{reload_event['catalog_sha256'][:16]}"
        / args.trainer_id
    )
    ack_files = sorted(ack_root.glob("rank_*.json"))
    require(len(ack_files) == 8, "step-500 ACK count drift")
    require(
        sorted(int(json.loads(path.read_text())["rank"]) for path in ack_files) == list(range(8)),
        "step-500 ACK rank inventory drift",
    )

    report = {
        "status": "PASS",
        "start": {
            "selector_version": start["selector_version"],
            "selector_parameters": architecture["total_parameters"],
            "hidden_dim": architecture["hidden_dim"],
            "mlp_depth_contract": architecture["all_mlp_depth"],
            "warm_start_sha256": start["source_hashes"]["warm_start"],
            "active_catalog_epoch": start["active_catalog_epoch"],
            "active_catalog_sha256": start["active_catalog_sha256"],
            "active_catalog_files": start["active_catalog_files"],
            "active_catalog_bytes": start["active_catalog_bytes"],
        },
        "step100": {
            "average_compute_seconds_per_step": step100["average_compute_seconds_per_step"],
            "rank_monitors": monitor_report,
        },
        "step500": {
            "catalog_changed": reload_event["catalog_changed"],
            "catalog_epoch": reload_event["catalog_epoch"],
            "catalog_sha256": reload_event["catalog_sha256"],
            "catalog_files": reload_event["catalog_files"],
            "catalog_bytes": reload_event["catalog_bytes"],
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_id": reload_event["checkpoint_id"],
            "rank_acks": len(ack_files),
            "strict_selector_load": True,
            "selector_optimizer_group_lrs": group_lrs,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
