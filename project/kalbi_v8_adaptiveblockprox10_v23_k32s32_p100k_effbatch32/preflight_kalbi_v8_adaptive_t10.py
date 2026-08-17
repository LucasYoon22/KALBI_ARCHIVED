#!/usr/bin/env python3
"""Static source-checkpoint and resume audit for KALBI V8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

import torch

from .adaptive_block_prox_v8 import (
    ADAPTIVE_BLOCK_FORMULA_VERSION,
    AdaptiveBlockProxConfig,
    adaptive_block_size,
    feature_blocks,
    sweep_reverse_flags,
)
from .model import (
    BASIS_VERSION,
    KALBI_REFINEMENT_MODE,
    KALBI_REFINEMENT_SWEEPS,
    MODEL_VERSION,
    SHARED_SUPPORT_CONTRACT,
    build_model,
)
from .train import (
    atomic_torch_save,
    capture_rng,
    restore_rng,
    state_dict_sha256,
)
from .train_beta_only_v23_supportmix_v13_adaptive_t10_8gpu import (
    CERTIFIED_PIPELINE_SHA256,
    CHECKPOINT_VERSION,
    EXPECTED_MODEL_PARAMETER_COUNT,
    SELECTOR_STATE_SHA256,
    SOURCE_BASIS_VERSION,
    SOURCE_CHECKPOINT_SHA256,
    SOURCE_CHECKPOINT_VERSION,
    SOURCE_COMPLETED_STEPS,
    SOURCE_MODEL_VERSION,
    SOURCE_SELECTOR_VERSION,
    SOURCE_SHARED_SUPPORT_CONTRACT,
    SOURCE_TRAINER_VERSION,
    TRAINER_VERSION,
    audit_optimizer_state_by_parameter_name,
    audit_state_dict_compatibility,
    build_beta_parameter_groups,
    opaque_payload_sha256,
)
from .train_joint_v11_direct import file_sha256, module_sha256
from train_fpk import _load_yaml


PARTITION_COUNTS = {
    500: (256, 2),
    904: (256, 4),
    1_601: (256, 7),
    4_497: (512, 9),
    10_000: (1_024, 10),
    13_084: (1_024, 13),
    24_416: (2_048, 12),
    36_241: (4_096, 9),
    100_000: (4_096, 25),
}


def dump_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if file_sha256(args.checkpoint) != args.checkpoint_sha256:
        raise RuntimeError("source checkpoint SHA mismatch")
    if args.checkpoint_sha256 != SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("source checkpoint is not certified")

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    metadata = checkpoint["metadata"]
    expected_metadata = {
        "checkpoint_version": SOURCE_CHECKPOINT_VERSION,
        "trainer_version": SOURCE_TRAINER_VERSION,
        "model_version": SOURCE_MODEL_VERSION,
        "basis_version": SOURCE_BASIS_VERSION,
        "shared_support_contract": SOURCE_SHARED_SUPPORT_CONTRACT,
        "training_mode": "beta_only",
        "selector_version": SOURCE_SELECTOR_VERSION,
        "world_size": 8,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise RuntimeError(f"source metadata mismatch for {key}")
    if int(checkpoint["completed_steps"]) != SOURCE_COMPLETED_STEPS:
        raise RuntimeError("source completed-step mismatch")
    pipeline_sha = metadata["exact_training_arguments"][
        "expected_pipeline_sha256"
    ]
    if pipeline_sha != CERTIFIED_PIPELINE_SHA256:
        raise RuntimeError("source pipeline SHA mismatch")
    source_control_root = Path(
        metadata["exact_training_arguments"]["rolling_control_root"]
    )
    catalog_path = (
        source_control_root
        / "catalogs"
        / checkpoint["catalog_pointer"]["catalog_file"]
    )
    if (
        not catalog_path.is_file()
        or file_sha256(catalog_path)
        != checkpoint["catalog_pointer"]["catalog_sha256"]
        or json.loads(catalog_path.read_text(encoding="utf-8"))
        != checkpoint["catalog_payload"]
    ):
        raise RuntimeError("source catalog file/payload verification failed")

    model = build_model(
        _load_yaml(str(args.config)),
        torch.device("cpu"),
    )
    model_audit = audit_state_dict_compatibility(
        model,
        checkpoint["model_state_dict"],
    )
    if model_audit["trainable_parameter_count"] != (
        EXPECTED_MODEL_PARAMETER_COUNT
    ):
        raise RuntimeError("model parameter count mismatch")
    source_model_hash = state_dict_sha256(checkpoint["model_state_dict"])
    if module_sha256(model) != source_model_hash:
        raise RuntimeError("strict model load output hash mismatch")

    groups = build_beta_parameter_groups(model)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": groups[name],
                "lr": 1e-4,
                "name": name,
            }
            for name in groups
            if groups[name]
        ],
        lr=1e-4,
        weight_decay=1e-4,
        betas=(0.9, 0.95),
    )
    optimizer_audit = audit_optimizer_state_by_parameter_name(
        module=model,
        optimizer=optimizer,
        source_state_dict=checkpoint["beta_optimizer_state_dict"],
    )
    optimizer.load_state_dict(checkpoint["beta_optimizer_state_dict"])

    def multiplier(step: int) -> float:
        if step < 500:
            return float(step + 1) / 500.0
        progress = float(step - 500) / float(30_000 - 500)
        return 0.5 * (
            1.0 + torch.cos(
                torch.tensor(
                    torch.pi * min(max(progress, 0.0), 1.0)
                )
            ).item()
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    scheduler.load_state_dict(checkpoint["beta_scheduler_state_dict"])
    if scheduler.last_epoch != int(
        checkpoint["beta_scheduler_state_dict"]["last_epoch"]
    ):
        raise RuntimeError("scheduler strict load failed")

    selector_hash = state_dict_sha256(checkpoint["selector_state_dict"])
    if selector_hash != SELECTOR_STATE_SHA256:
        raise RuntimeError("opaque selector hash mismatch")
    selector_optimizer_hash = opaque_payload_sha256(
        checkpoint["selector_optimizer_state_dict"]
    )
    selector_updates = int(
        checkpoint["trainer_state"].get("selector_updates", -1)
    )

    original_rng = capture_rng()
    rank_zero = next(
        state
        for state in checkpoint["rank_states"]
        if int(state["rank"]) == 0
    )
    restore_rng(rank_zero["rng"])
    first = {
        "torch": torch.rand(4),
        "numpy": __import__("numpy").random.random(4),
        "python": [__import__("random").random() for _ in range(4)],
    }
    restore_rng(rank_zero["rng"])
    second = {
        "torch": torch.rand(4),
        "numpy": __import__("numpy").random.random(4),
        "python": [__import__("random").random() for _ in range(4)],
    }
    restore_rng(original_rng)
    rng_reproducible = (
        torch.equal(first["torch"], second["torch"])
        and bool((first["numpy"] == second["numpy"]).all())
        and first["python"] == second["python"]
    )
    if not rng_reproducible:
        raise RuntimeError("rank RNG state is not reproducible")

    roundtrip_payload = {
        "completed_steps": checkpoint["completed_steps"],
        "model_state_dict": model.state_dict(),
        "beta_optimizer_state_dict": optimizer.state_dict(),
        "beta_scheduler_state_dict": scheduler.state_dict(),
        "selector_state_dict": checkpoint["selector_state_dict"],
        "selector_optimizer_state_dict": (
            checkpoint["selector_optimizer_state_dict"]
        ),
    }
    with tempfile.TemporaryDirectory(dir=args.output_dir) as temporary:
        roundtrip_path = Path(temporary) / "roundtrip.pt"
        atomic_torch_save(roundtrip_payload, roundtrip_path)
        reloaded = torch.load(
            roundtrip_path,
            map_location="cpu",
            weights_only=False,
        )
        roundtrip_ok = (
            state_dict_sha256(reloaded["model_state_dict"])
            == source_model_hash
            and state_dict_sha256(reloaded["selector_state_dict"])
            == selector_hash
            and opaque_payload_sha256(
                reloaded["selector_optimizer_state_dict"]
            )
            == selector_optimizer_hash
            and int(reloaded["completed_steps"]) == SOURCE_COMPLETED_STEPS
        )
    if not roundtrip_ok:
        raise RuntimeError("checkpoint roundtrip failed")

    partition_contract = {
        str(feature_count): {
            "block_size": adaptive_block_size(feature_count),
            "blocks_per_sweep": len(
                feature_blocks(
                    feature_count,
                    adaptive_block_size(feature_count),
                )
            ),
        }
        for feature_count in PARTITION_COUNTS
    }
    for feature_count, expected in PARTITION_COUNTS.items():
        observed = partition_contract[str(feature_count)]
        if (observed["block_size"], observed["blocks_per_sweep"]) != expected:
            raise RuntimeError(f"adaptive partition drift for P={feature_count}")
    adaptive_manifest = {
        "formula_version": ADAPTIVE_BLOCK_FORMULA_VERSION,
        "config": AdaptiveBlockProxConfig().__dict__,
        "sweep_reverse_flags": list(sweep_reverse_flags()),
        "partitions": partition_contract,
    }
    source_manifest = {
        "path": str(args.checkpoint.resolve()),
        "sha256": args.checkpoint_sha256,
        "completed_steps": int(checkpoint["completed_steps"]),
        "metadata_contract": expected_metadata,
        "pipeline_sha256": pipeline_sha,
        "catalog_pointer": checkpoint["catalog_pointer"],
        "rank_state_count": len(checkpoint["rank_states"]),
        "selector_state_sha256": selector_hash,
        "selector_optimizer_payload_sha256": selector_optimizer_hash,
        "selector_updates": selector_updates,
    }
    resume_audit = {
        "status": "PASS",
        "model": model_audit,
        "model_state_sha256": source_model_hash,
        "optimizer": optimizer_audit,
        "optimizer_state_loaded": True,
        "scheduler_state_loaded": True,
        "scheduler_last_epoch": scheduler.last_epoch,
        "rank_rng_reproducible": rng_reproducible,
        "rank_stream_states_present": all(
            "stream" in state for state in checkpoint["rank_states"]
        ),
        "rank_sampler_states_present": all(
            "sampler" in state for state in checkpoint["rank_states"]
        ),
        "rank_penalty_cache_states_present": all(
            "penalty_cache" in state for state in checkpoint["rank_states"]
        ),
        "checkpoint_roundtrip": roundtrip_ok,
        "selector_payload_carried_opaque": True,
        "selector_state_unchanged": True,
        "source_checkpoint_unchanged": (
            file_sha256(args.checkpoint) == args.checkpoint_sha256
        ),
    }
    source_files = (
        sorted(args.package_dir.glob("*.py"))
        + sorted(args.package_dir.glob("*.sh"))
        + sorted(args.package_dir.glob("*.conf"))
    )
    source_hashes = {
        str(path.relative_to(args.package_dir)): file_sha256(path)
        for path in source_files
        if path.is_file()
    }
    implementation = {
        "status": "IMPLEMENTED_PREFLIGHT_ONLY",
        "trainer_version": TRAINER_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_version": MODEL_VERSION,
        "basis_version": BASIS_VERSION,
        "shared_support_contract": SHARED_SUPPORT_CONTRACT,
        "refinement_mode": KALBI_REFINEMENT_MODE,
        "refinement_sweeps": KALBI_REFINEMENT_SWEEPS,
        "full_training_started": False,
        "selector_model_instantiated_by_trainer": False,
        "selector_optimizer_instantiated_by_trainer": False,
        "selector_update_enabled": False,
    }
    dump_json(args.output_dir / "implementation_manifest.json", implementation)
    dump_json(args.output_dir / "source_checkpoint_manifest.json", source_manifest)
    dump_json(args.output_dir / "adaptive_block_contract.json", adaptive_manifest)
    dump_json(args.output_dir / "resume_audit.json", resume_audit)
    dump_json(
        args.output_dir / "source_sha256_manifest.json",
        source_hashes,
    )
    print(json.dumps(resume_audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
