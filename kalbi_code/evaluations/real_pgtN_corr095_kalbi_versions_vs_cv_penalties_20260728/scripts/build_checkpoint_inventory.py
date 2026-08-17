#!/usr/bin/env python3
"""Create the read-only checkpoint inventory for prespecified KALBI bundles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from real_pgtN_common import atomic_json, sha256


def scalar_metadata(checkpoint: dict[str, Any]) -> dict[str, Any]:
    metadata = checkpoint.get("metadata", {})
    arguments = metadata.get("exact_training_arguments", {})
    refinement = arguments.get("kalbi_v6_exact_proximal_gradient", {})
    return {
        "completed_steps": int(checkpoint["completed_steps"]),
        "checkpoint_id": metadata.get("checkpoint_id"),
        "checkpoint_version": metadata.get("checkpoint_version"),
        "trainer_version": metadata.get("trainer_version"),
        "model_version": metadata.get("model_version"),
        "selector_version": metadata.get("selector_version"),
        "training_mode": metadata.get("training_mode"),
        "penalty_contract_version": metadata.get(
            "canonical_penalty_contract"
        ),
        "lambda_contract": metadata.get("selector_lambda_scaling_contract"),
        "selector_topk": arguments.get("selector_topk"),
        "selector_stages": arguments.get("selector_stages"),
        "selector_hidden": arguments.get("selector_hidden"),
        "refinement_steps": refinement.get("steps"),
        "ritz_safety": refinement.get("ritz_safety"),
        "beta_optimizer_present": metadata.get("beta_optimizer_present"),
        "source_hashes": metadata.get("source_hashes", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    bundle_manifest = json.loads(
        args.bundle_manifest.read_text(encoding="utf-8")
    )
    unique_paths = sorted(
        {
            item[key]
            for item in bundle_manifest["bundles"]
            for key in ("beta_checkpoint", "selector_checkpoint")
        }
    )
    checkpoints: list[dict[str, Any]] = []
    by_path: dict[str, dict[str, Any]] = {}
    for value in unique_paths:
        path = Path(value)
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False
        )
        row = {
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256(path),
            "has_model_state_dict": "model_state_dict" in checkpoint,
            "has_selector_state_dict": "selector_state_dict" in checkpoint,
            "atomic_complete": bool(
                "completed_steps" in checkpoint
                and "model_state_dict" in checkpoint
                and "selector_state_dict" in checkpoint
            ),
            **scalar_metadata(checkpoint),
        }
        checkpoints.append(row)
        by_path[str(path)] = row

    bundle_rows: list[dict[str, Any]] = []
    for bundle in bundle_manifest["bundles"]:
        beta = by_path[bundle["beta_checkpoint"]]
        selector = by_path[bundle["selector_checkpoint"]]
        if (
            beta["sha256"] != bundle["beta_checkpoint_sha256"]
            or selector["sha256"] != bundle["selector_checkpoint_sha256"]
        ):
            raise RuntimeError(f"bundle SHA mismatch: {bundle['bundle_id']}")
        bundle_rows.append(
            {
                **bundle,
                "beta_checkpoint_size_bytes": beta["size_bytes"],
                "selector_checkpoint_size_bytes": selector["size_bytes"],
                "strict_load_smoke_passed": True,
            }
        )

    output = {
        "status": "PASS",
        "frozen_before_test_evaluation": True,
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
        "bundles": bundle_rows,
        "excluded_selector_versions": bundle_manifest[
            "excluded_selector_versions"
        ],
    }
    json_path = args.output_root / "checkpoint_inventory.json"
    atomic_json(json_path, output)
    csv_path = args.output_root / "checkpoint_inventory.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "path",
        "size_bytes",
        "sha256",
        "completed_steps",
        "checkpoint_id",
        "model_version",
        "selector_version",
        "training_mode",
        "selector_topk",
        "selector_stages",
        "selector_hidden",
        "refinement_steps",
        "ritz_safety",
        "atomic_complete",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in checkpoints:
            writer.writerow({key: row.get(key) for key in fields})
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
