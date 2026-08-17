#!/usr/bin/env python3
"""Read-only compatibility audit for the pinned V8 beta and V24 selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    array = value.detach().float()
    return {
        "shape": list(array.shape),
        "finite": bool(torch.isfinite(array).all().item()),
        "minimum": float(array.min().item()),
        "maximum": float(array.max().item()),
        "mean": float(array.mean().item()),
    }


def synthetic_task(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260728)
    rows, features = 128, 1200
    latent = torch.randn(rows, 24, generator=generator)
    mixing = torch.randn(24, features, generator=generator)
    independent = torch.randn(rows, features, generator=generator)
    x = 0.55 * (latent @ mixing) / (24.0**0.5) + 0.835 * independent
    beta = torch.zeros(features)
    beta[:24] = torch.linspace(1.0, 0.15, 24)
    y = x @ beta + 0.35 * torch.randn(rows, generator=generator)
    x = (x - x.mean(0, keepdim=True)) / x.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    y = (y - y.mean()) / y.std(unbiased=False).clamp_min(1e-6)
    return x[None].to(device), y[None].to(device)


def decode(
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    selected: dict[str, Any],
    raw_lambdas: torch.Tensor,
) -> dict[str, Any]:
    query_count = raw_lambdas.shape[1]
    return model.decode_queries(
        encoded,
        lambdas=raw_lambdas,
        lambda_max=encoded["lambda_max"].reshape(-1, 1).expand(-1, query_count),
        penalty_shape_features=selected["penalty_shape_features"].reshape(1, 1, -1).expand(
            1, query_count, -1
        ),
        penalty_d0=selected["penalty_d0"].reshape(1, 1).expand(1, query_count),
        return_aux=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--beta-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    args = parser.parse_args()

    started = time.perf_counter()
    device = torch.device(args.device)
    selector_checkpoint = torch.load(
        args.selector_checkpoint, map_location="cpu", weights_only=False
    )
    beta_checkpoint = torch.load(
        args.beta_checkpoint, map_location="cpu", weights_only=False
    )

    v6 = build_v6_model(_load_yaml(str(args.config)), device)
    v8 = build_v8_model(_load_yaml(str(args.config)), device)
    v6.load_state_dict(selector_checkpoint["model_state_dict"], strict=True)
    v8.load_state_dict(beta_checkpoint["model_state_dict"], strict=True)
    v6.to(device).eval()
    v8.to(device).eval()
    for model in (v6, v8):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    selector = DirectPenaltyMLPQuerySelectorV24Additive512(
        hidden=512,
        derivative_floor=1e-6,
        squareplus_scale=0.02,
        direct_head_init_std=1e-3,
    )
    selector.load_state_dict(
        selector_checkpoint["selector_state_dict"], strict=True
    )
    selector.to(device).eval()
    for parameter in selector.parameters():
        parameter.requires_grad_(False)

    frozen_hash = state_dict_sha256(selector_checkpoint["model_state_dict"])
    expected_frozen_hash = selector_checkpoint["metadata"][
        "frozen_beta_state_sha256"
    ]
    if frozen_hash != expected_frozen_hash:
        raise RuntimeError("V24 coherent frozen-beta hash mismatch")

    v6_keys = set(selector_checkpoint["model_state_dict"])
    v8_keys = set(beta_checkpoint["model_state_dict"])
    common_keys = v6_keys & v8_keys
    shape_mismatches = [
        key
        for key in sorted(common_keys)
        if selector_checkpoint["model_state_dict"][key].shape
        != beta_checkpoint["model_state_dict"][key].shape
    ]

    x, y = synthetic_task(device)
    with torch.inference_mode():
        encoded_v6 = v6.encode_dataset(v6.precompute(x, y))
        encoded_v8 = v8.encode_dataset(v8.precompute(x, y))
        selected_v6_context = selector(
            x,
            y,
            encoded_v6["lambda_max"].reshape(-1),
            solver_context=encoded_v6,
            return_representation=True,
        )
        selected_v8_context = selector(
            x,
            y,
            encoded_v8["lambda_max"].reshape(-1),
            solver_context=encoded_v8,
            return_representation=True,
        )

        selected_lambda = selected_v8_context["raw_lambda"].reshape(-1)
        query_lambda = torch.stack(
            [
                torch.tensor(0.02, device=device),
                torch.tensor(0.2, device=device),
                selected_lambda[0],
            ]
        )[None]
        decoded_v8 = decode(v8, encoded_v8, selected_v8_context, query_lambda)
        decoded_v6 = decode(v6, encoded_v6, selected_v6_context, query_lambda)

    for name, value in {
        "selected_raw_lambda": selected_v8_context["raw_lambda"],
        "selected_penalty_shape": selected_v8_context["penalty_shape_features"],
        "selected_penalty_d0": selected_v8_context["penalty_d0"],
        "v8_beta": decoded_v8["beta_q"],
        "v6_beta": decoded_v6["beta_q"],
    }.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite tensor: {name}")

    eta = decoded_v8["kalbi_v8_refinement_step_size_path_q"].float()
    gamma = decoded_v8["kalbi_v8_refinement_gamma_path_q"].float()
    ratio = gamma / eta.clamp_min(1e-30)
    expected = query_lambda[..., None].expand_as(ratio)
    gamma_error = (ratio - expected).abs()
    if float(gamma_error.max()) > 2e-5:
        raise RuntimeError("V8 gamma/eta does not equal raw lambda")

    selected_probe_error = selected_v8_context[
        "v24_probe_gamma_contract_error"
    ].float()
    if float(selected_probe_error.max()) > 2e-5:
        raise RuntimeError("V24 probe gamma contract failed")

    penalty_shape_width = int(
        selected_v8_context["penalty_shape_features"].shape[-1]
    )
    if penalty_shape_width != int(
        selected_v6_context["penalty_shape_features"].shape[-1]
    ):
        raise RuntimeError("V6/V8 penalty-shape width mismatch")

    result = {
        "status": "PASS",
        "pairing_decision": (
            "V24 selector is executable with V8 encoded context and its raw-lambda/"
            "Direct96 outputs pass unchanged into the V8 public decode interface."
        ),
        "selector_checkpoint": str(args.selector_checkpoint),
        "selector_checkpoint_sha256": file_sha256(args.selector_checkpoint),
        "selector_completed_steps": int(
            selector_checkpoint["completed_steps"]
        ),
        "selector_version": selector_checkpoint["metadata"]["selector_version"],
        "selector_coherent_beta_model_version": selector_checkpoint["metadata"][
            "model_version"
        ],
        "selector_frozen_beta_state_sha256": frozen_hash,
        "beta_checkpoint": str(args.beta_checkpoint),
        "beta_checkpoint_sha256": file_sha256(args.beta_checkpoint),
        "beta_completed_steps": int(beta_checkpoint["completed_steps"]),
        "beta_model_version": beta_checkpoint["metadata"]["model_version"],
        "beta_refinement_mode": beta_checkpoint["metadata"]["refinement_mode"],
        "beta_refinement_sweeps": int(
            beta_checkpoint["metadata"]["refinement_sweeps"]
        ),
        "state_dict": {
            "v6_keys": len(v6_keys),
            "v8_keys": len(v8_keys),
            "common_keys": len(common_keys),
            "v6_only_keys": sorted(v6_keys - v8_keys),
            "v8_only_keys": sorted(v8_keys - v6_keys),
            "common_shape_mismatches": shape_mismatches,
        },
        "runtime": {
            "device": str(device),
            "seconds": time.perf_counter() - started,
        },
        "sentinel": {
            "x_shape": list(x.shape),
            "y_shape": list(y.shape),
            "query_raw_lambdas": query_lambda.detach().cpu().tolist(),
            "penalty_shape_width": penalty_shape_width,
            "selected_v8_context_raw_lambda": float(
                selected_v8_context["raw_lambda"].item()
            ),
            "selected_v6_context_raw_lambda": float(
                selected_v6_context["raw_lambda"].item()
            ),
            "selected_raw_lambda_absolute_difference": float(
                (
                    selected_v8_context["raw_lambda"]
                    - selected_v6_context["raw_lambda"]
                )
                .abs()
                .item()
            ),
            "selected_penalty_max_absolute_difference": float(
                (
                    selected_v8_context["penalty_rho"]
                    - selected_v6_context["penalty_rho"]
                )
                .abs()
                .max()
                .item()
            ),
            "v8_gamma_over_eta_max_absolute_error": float(
                gamma_error.max().item()
            ),
            "v24_probe_gamma_max_absolute_error": float(
                selected_probe_error.max().item()
            ),
            "v8_refinement_eta": tensor_summary(eta),
            "v8_refinement_gamma": tensor_summary(gamma),
            "v8_beta": tensor_summary(decoded_v8["beta_q"]),
            "v6_beta": tensor_summary(decoded_v6["beta_q"]),
            "v8_vs_v6_beta_rmse": float(
                (
                    decoded_v8["beta_q"].float()
                    - decoded_v6["beta_q"].float()
                )
                .square()
                .mean()
                .sqrt()
                .item()
            ),
        },
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
