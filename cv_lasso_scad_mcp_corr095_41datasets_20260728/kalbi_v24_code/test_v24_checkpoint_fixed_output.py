#!/usr/bin/env python3
"""Strict-load a V24 checkpoint twice and compare fixed-task outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .model import build_model
from .selector_v24_additive512_trajectory import (
    DirectPenaltyMLPQuerySelectorV24Additive512,
)
from .train import state_dict_sha256
from train_fpk import _load_yaml


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_selector() -> DirectPenaltyMLPQuerySelectorV24Additive512:
    return DirectPenaltyMLPQuerySelectorV24Additive512(
        hidden=512,
        derivative_floor=1e-6,
        squareplus_scale=0.02,
        direct_head_init_std=1e-3,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    observed_sha = file_sha256(args.checkpoint)
    if observed_sha != args.checkpoint_sha256:
        raise RuntimeError("checkpoint SHA256 mismatch")
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if "beta_optimizer_state_dict" in checkpoint:
        raise RuntimeError("selector-only checkpoint contains a beta optimizer")
    if "beta_scheduler_state_dict" in checkpoint:
        raise RuntimeError("selector-only checkpoint contains a beta scheduler")

    torch.set_num_threads(4)
    torch.manual_seed(202607271120)
    device = torch.device(args.device)
    beta = build_model(_load_yaml(str(args.config)), device)
    beta.load_state_dict(checkpoint["model_state_dict"], strict=True)
    beta.eval()
    for parameter in beta.parameters():
        parameter.requires_grad_(False)

    selectors = [make_selector(), make_selector()]
    for selector in selectors:
        selector.load_state_dict(
            checkpoint["selector_state_dict"], strict=True
        )
        selector.to(device).eval()

    x = torch.randn(2, 80, 128, device=device)
    y = torch.randn(2, 80, device=device)
    x = (x - x.mean(1, keepdim=True)) / x.std(
        1, keepdim=True, correction=0
    ).clamp_min(1e-6)
    y = (y - y.mean(1, keepdim=True)) / y.std(
        1, keepdim=True, correction=0
    ).clamp_min(1e-6)
    with torch.no_grad():
        encoded = beta.encode_dataset(beta.precompute(x, y))
        outputs = [
            selector(
                x,
                y,
                encoded["lambda_max"].float(),
                solver_context=encoded,
                return_representation=True,
            )
            for selector in selectors
        ]
        betas = [
            beta.decode_queries(
                encoded,
                lambdas=output["lambdas"],
                lambda_max=encoded["lambda_max"].float()[:, None],
                penalty_shape_features=output["penalty_shape_features"],
                penalty_d0=output["penalty_d0"],
                return_aux=True,
            )["beta_q"]
            for output in outputs
        ]

    errors = {
        "raw_lambda_max_abs": float(
            (outputs[0]["raw_lambda"] - outputs[1]["raw_lambda"])
            .abs()
            .max()
        ),
        "penalty_shape_max_abs": float(
            (
                outputs[0]["penalty_shape_features"]
                - outputs[1]["penalty_shape_features"]
            )
            .abs()
            .max()
        ),
        "final_beta_max_abs": float((betas[0] - betas[1]).abs().max()),
    }
    if any(value != 0.0 for value in errors.values()):
        raise RuntimeError(f"fixed-task strict-load reproduction failed: {errors}")
    scaler_keys = sorted(
        key
        for key in checkpoint["selector_state_dict"]
        if key.startswith("v24_scaler.")
    )
    expected_scaler_keys = {
        "v24_scaler.global_center",
        "v24_scaler.global_scale",
        "v24_scaler.trajectory_center",
        "v24_scaler.trajectory_scale",
        "v24_scaler.proposal_center",
        "v24_scaler.proposal_scale",
        "v24_scaler.cache_loaded",
        "v24_scaler.cache_size",
        "v24_scaler.cache_sha256_bytes",
    }
    if not expected_scaler_keys.issubset(scaler_keys):
        raise RuntimeError("V24 scaler buffers are incomplete")
    beta_hash = state_dict_sha256(checkpoint["model_state_dict"])
    if beta_hash != checkpoint["metadata"]["frozen_beta_state_sha256"]:
        raise RuntimeError("frozen beta metadata hash mismatch")

    report = {
        "status": "PASS",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": observed_sha,
        "completed_steps": int(checkpoint["completed_steps"]),
        "catalog_epoch": int(checkpoint["metadata"]["catalog_epoch"]),
        "catalog_sha256": checkpoint["metadata"]["catalog_sha256"],
        "frozen_beta_state_sha256": beta_hash,
        "selector_state_key_count": len(checkpoint["selector_state_dict"]),
        "scaler_buffer_keys": scaler_keys,
        "fixed_task_shape": [2, 80, 128],
        "output_errors": errors,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
