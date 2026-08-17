#!/usr/bin/env python3
"""Read-only strict validation of pinned KALBI evaluation checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kalbi_v6_exactprox10_eta4x_v24_additive512_traj3_k32s32_p100k_effbatch32.model import (
    build_model as build_v6_model,
)
from kalbi_v6_exactprox10_eta4x_v24_additive512_traj3_k32s32_p100k_effbatch32.selector_v24_additive512_trajectory import (
    DirectPenaltyMLPQuerySelectorV24Additive512,
)
from kalbi_v6_exactprox10_eta4x_v24_additive512_traj3_k32s32_p100k_effbatch32.train import (
    state_dict_sha256,
)
from kalbi_v8_adaptiveblockprox10_v23_k32s32_p100k_effbatch32.model import (
    build_model as build_v8_model,
)
from train_fpk import _load_yaml


EXPECTED_BETA_MODEL = (
    "kalbi_v8_krylov_abfilm_threshold_adaptiveblockprox10_k32_stages32_v1"
)
EXPECTED_SELECTOR = (
    "v24_v23_additive512_global_solvertrajectory3_proposal_residual4_"
    "independent_rawlambda_v1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def creation_time_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def finite_audit(value: Any, path: str = "root") -> tuple[int, list[str]]:
    count = 0
    failures: list[str] = []
    if torch.is_tensor(value):
        count += 1
        if value.is_floating_point() or value.is_complex():
            if not bool(torch.isfinite(value).all()):
                failures.append(path)
    elif isinstance(value, np.ndarray):
        count += 1
        if np.issubdtype(value.dtype, np.inexact) and not np.isfinite(value).all():
            failures.append(path)
    elif isinstance(value, dict):
        for key, child in value.items():
            child_count, child_failures = finite_audit(child, f"{path}.{key}")
            count += child_count
            failures.extend(child_failures)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_count, child_failures = finite_audit(child, f"{path}[{index}]")
            count += child_count
            failures.extend(child_failures)
    return count, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pin-time-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.beta, args.selector):
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"missing or empty checkpoint: {path}")
        if any(token in path.name.lower() for token in (".tmp", "partial", "incomplete")):
            raise RuntimeError(f"temporary/partial checkpoint rejected: {path}")

    beta = torch.load(args.beta, map_location="cpu", weights_only=False, mmap=True)
    selector_checkpoint = torch.load(
        args.selector, map_location="cpu", weights_only=False, mmap=True
    )
    beta_metadata = beta["metadata"]
    selector_metadata = selector_checkpoint["metadata"]
    if beta_metadata["model_version"] != EXPECTED_BETA_MODEL:
        raise RuntimeError("beta model-version mismatch")
    if beta_metadata.get("refinement_mode") != "full_p_adaptive_block_prox":
        raise RuntimeError("beta refinement-mode mismatch")
    if beta_metadata.get("refinement_sweeps") != 10:
        raise RuntimeError("beta T mismatch")
    if (
        beta_metadata.get("adaptive_block_formula")
        != "pow2_ceil_P_over_16_clamp_256_4096_v1"
    ):
        raise RuntimeError("beta adaptive-block formula mismatch")
    if selector_metadata["selector_version"] != EXPECTED_SELECTOR:
        raise RuntimeError("selector-version mismatch")

    config = _load_yaml(str(args.config))
    v8_model = build_v8_model(config, torch.device("cpu"))
    v8_model.load_state_dict(beta["model_state_dict"], strict=True)
    v6_model = build_v6_model(config, torch.device("cpu"))
    v6_model.load_state_dict(selector_checkpoint["model_state_dict"], strict=True)
    selector = DirectPenaltyMLPQuerySelectorV24Additive512(
        hidden=512,
        derivative_floor=1e-6,
        squareplus_scale=0.02,
        direct_head_init_std=1e-3,
    )
    selector.load_state_dict(selector_checkpoint["selector_state_dict"], strict=True)

    beta_tensor_count, beta_failures = finite_audit(beta)
    selector_tensor_count, selector_failures = finite_audit(selector_checkpoint)
    if beta_failures or selector_failures:
        raise RuntimeError(
            f"nonfinite checkpoint tensors: beta={beta_failures[:5]} "
            f"selector={selector_failures[:5]}"
        )

    beta_file_sha = sha256(args.beta)
    selector_file_sha = sha256(args.selector)
    beta_state_sha = state_dict_sha256(beta["model_state_dict"])
    selector_frozen_sha = selector_metadata["frozen_beta_state_sha256"]
    result = {
        "status": "PASS",
        "audit_time_utc": datetime.now(timezone.utc).isoformat(),
        "pin_time_utc": args.pin_time_utc,
        "candidate_policy": (
            "latest complete candidate existing at pin time; later-created files excluded"
        ),
        "beta": {
            "path": str(args.beta.resolve()),
            "bytes": args.beta.stat().st_size,
            "creation_time_utc": creation_time_utc(args.beta),
            "completed_steps": int(beta["completed_steps"]),
            "model_version": beta_metadata["model_version"],
            "checkpoint_version": beta_metadata["checkpoint_version"],
            "basis_version": beta_metadata["basis_version"],
            "refinement_mode": beta_metadata["refinement_mode"],
            "refinement_sweeps": beta_metadata["refinement_sweeps"],
            "adaptive_block_formula": beta_metadata["adaptive_block_formula"],
            "sweep_order": beta_metadata["refinement_sweep_order"],
            "file_sha256": beta_file_sha,
            "model_state_sha256": beta_state_sha,
            "strict_cpu_load": True,
            "finite_tensor_count": beta_tensor_count,
        },
        "selector": {
            "path": str(args.selector.resolve()),
            "bytes": args.selector.stat().st_size,
            "creation_time_utc": creation_time_utc(args.selector),
            "completed_steps": int(selector_checkpoint["completed_steps"]),
            "model_version": selector_metadata["model_version"],
            "selector_version": selector_metadata["selector_version"],
            "checkpoint_version": selector_metadata["checkpoint_version"],
            "solver_contract": selector_metadata["shared_support_contract"],
            "probe_contract": selector_metadata["exact_training_arguments"],
            "file_sha256": selector_file_sha,
            "frozen_beta_state_sha256": selector_frozen_sha,
            "strict_cpu_model_load": True,
            "strict_cpu_selector_load": True,
            "finite_tensor_count": selector_tensor_count,
        },
        "pairing": {
            "selector_frozen_beta_matches_latest_beta": (
                selector_frozen_sha == beta_state_sha
            ),
            "latest_beta_model_state_sha256": beta_state_sha,
            "selector_frozen_beta_state_sha256": selector_frozen_sha,
            "v24_on_v8_transition_completed": False,
            "actual_selector_solver_version": selector_metadata["model_version"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
