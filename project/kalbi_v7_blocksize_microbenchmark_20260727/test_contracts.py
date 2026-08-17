#!/usr/bin/env python3
"""Fail-closed gradient and partition tests for experimental block sizes."""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch

from .benchmark_cell import run_method
from .common import OUTPUT, atomic_json_dump, load_model_and_checkpoint, load_precomputed
from .configurable_block_prox import (
    SWEEP_REVERSE,
    adaptive_block_size,
    feature_blocks,
)
from .metrics import objective_components


def query_slice(payload: dict[str, Any], query_index: int) -> dict[str, Any]:
    result = dict(payload)
    for key in (
        "lambdas",
        "penalty_shape_features",
        "penalty_d0",
        "rho",
        "drho",
        "beta_initial",
        "prethreshold",
        "log_threshold_multiplier",
    ):
        result[key] = payload[key][:, query_index : query_index + 1]
    return result


def reconstruct_beta(
    payload: dict[str, Any],
    prethreshold: torch.Tensor,
    lambdas: torch.Tensor,
    d0: torch.Tensor,
) -> torch.Tensor:
    threshold = (
        lambdas.float()[..., None]
        * d0.float()[..., None]
        / payload["gram_diagonal"].float()[:, None, :].clamp_min(1e-4)
        * torch.exp(payload["log_threshold_multiplier"].float())
    )
    return (
        torch.sign(prethreshold)
        * torch.relu(prethreshold.abs() - threshold)
    ).clamp(-6.0, 6.0).to(torch.bfloat16).float()


def scalar_loss(
    *,
    model: torch.nn.Module,
    payload: dict[str, Any],
    beta: torch.Tensor,
    lambdas: torch.Tensor,
    shape: torch.Tensor,
) -> torch.Tensor:
    metrics = objective_components(
        x=payload["x"],
        y=payload["y"],
        beta=beta,
        lambdas=lambdas,
        penalty_shape_features=shape,
        grid=model._kalbi_v7_penalty_grid,
    )
    return torch.log1p(
        metrics["objective"]
        / metrics["f_zero"].unsqueeze(1).clamp_min(1e-12)
    ).mean()


def audit_method(
    *,
    model: torch.nn.Module,
    base_payload: dict[str, Any],
    method: str,
    query_index: int,
    seed: int,
) -> dict[str, Any]:
    payload = query_slice(base_payload, query_index)
    model.eval()
    with torch.no_grad():
        encoded = model.encode_dataset(
            model.precompute(payload["x"], payload["y"])
        )
    prethreshold = (
        payload["prethreshold"].detach().clone().requires_grad_(True)
    )
    lambdas = payload["lambdas"].detach().clone().requires_grad_(True)
    shape = (
        payload["penalty_shape_features"]
        .detach()
        .clone()
        .requires_grad_(True)
    )
    d0 = payload["penalty_d0"].detach().clone().requires_grad_(True)
    beta0 = reconstruct_beta(payload, prethreshold, lambdas, d0)
    beta0.retain_grad()
    reconstruction_error = float(
        (beta0.detach() - payload["beta_initial"]).abs().max().item()
    )
    if reconstruction_error > 1e-6:
        raise AssertionError(
            f"{method} terminal reconstruction error {reconstruction_error}"
        )
    beta_checkpointed, _ = run_method(
        method=method,
        model=model,
        encoded=encoded,
        beta_initial=beta0,
        lambdas=lambdas,
        penalty_d0=d0,
        penalty_shape_features=shape,
        activation_checkpointing=True,
    )
    beta_uncheckpointed, _ = run_method(
        method=method,
        model=model,
        encoded=encoded,
        beta_initial=beta0,
        lambdas=lambdas,
        penalty_d0=d0,
        penalty_shape_features=shape,
        activation_checkpointing=False,
    )
    output_difference = float(
        (beta_checkpointed - beta_uncheckpointed).abs().max().item()
    )
    if output_difference >= 1e-6:
        raise AssertionError(
            f"{method} checkpoint output difference {output_difference}"
        )
    loss = scalar_loss(
        model=model,
        payload=payload,
        beta=beta_checkpointed,
        lambdas=lambdas,
        shape=shape,
    )
    loss.backward()
    gradients = {
        "beta_initial": beta0.grad,
        "raw_lambda": lambdas.grad,
        "penalty_shape": shape.grad,
        "d0": d0.grad,
        "prethreshold": prethreshold.grad,
    }
    gradient_norms: dict[str, float] = {}
    for name, gradient in gradients.items():
        if gradient is None or not bool(torch.isfinite(gradient).all().item()):
            raise AssertionError(f"{method} {name} gradient missing/nonfinite")
        norm = float(gradient.float().norm().item())
        if norm <= 0.0:
            raise AssertionError(f"{method} {name} gradient is zero")
        gradient_norms[name] = norm

    generator = torch.Generator(device=prethreshold.device).manual_seed(seed)
    direction = torch.randn(
        beta0.shape,
        generator=generator,
        device=beta0.device,
    )
    direction = direction / direction.norm().clamp_min(1e-12)
    autodiff = float((beta0.grad * direction).sum().item())
    epsilon = 1e-3

    def perturbed_loss(sign: float) -> float:
        with torch.no_grad():
            candidate_beta0 = beta0.detach() + sign * epsilon * direction
            candidate, _ = run_method(
                method=method,
                model=model,
                encoded=encoded,
                beta_initial=candidate_beta0,
                lambdas=lambdas.detach(),
                penalty_d0=d0.detach(),
                penalty_shape_features=shape.detach(),
                activation_checkpointing=False,
            )
            return float(
                scalar_loss(
                    model=model,
                    payload=payload,
                    beta=candidate,
                    lambdas=lambdas.detach(),
                    shape=shape.detach(),
                ).item()
            )

    plus = perturbed_loss(1.0)
    minus = perturbed_loss(-1.0)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    relative_error = abs(autodiff - finite_difference) / max(
        1e-8, abs(autodiff), abs(finite_difference)
    )
    if relative_error >= 0.05:
        raise AssertionError(
            f"{method} directional-gradient error {relative_error}: "
            f"autodiff={autodiff}, finite_difference={finite_difference}"
        )
    return {
        "task_index": int(payload["task"]["dataset_index"]),
        "query_index": query_index,
        "method": method,
        "terminal_reconstruction_max_abs_error": reconstruction_error,
        "activation_checkpoint_max_abs_difference": output_difference,
        "gradient_norms": gradient_norms,
        "directional_autodiff": autodiff,
        "directional_finite_difference": finite_difference,
        "directional_relative_error": relative_error,
        "loss": float(loss.detach().item()),
    }


def visit_audit() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for feature_count in (500, 10_000, 100_000):
        policies = {
            "block256": 256,
            "block4096": 4096,
            "adaptive": adaptive_block_size(feature_count),
        }
        output[str(feature_count)] = {}
        for name, block_size in policies.items():
            blocks = feature_blocks(feature_count, block_size)
            visits = torch.zeros(feature_count, dtype=torch.int16)
            for reverse in SWEEP_REVERSE:
                order = (
                    range(len(blocks) - 1, -1, -1)
                    if reverse
                    else range(len(blocks))
                )
                for block_index in order:
                    start, stop = blocks[block_index]
                    visits[start:stop] += 1
            if not bool((visits == 3).all().item()):
                raise AssertionError(
                    f"{name}, P={feature_count} visit contract failed"
                )
            output[str(feature_count)][name] = {
                "block_size": block_size,
                "block_count": len(blocks),
                "visit_min": int(visits.min().item()),
                "visit_max": int(visits.max().item()),
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output", default=str(OUTPUT / "gradient_audit.json")
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    model, checkpoint = load_model_and_checkpoint(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    audits = [
        audit_method(
            model=model,
            base_payload=load_precomputed(1, device),
            method="block256",
            query_index=4,
            seed=202607270401,
        ),
        audit_method(
            model=model,
            base_payload=load_precomputed(3, device),
            method="adaptive",
            query_index=4,
            seed=202607270403,
        ),
    ]
    result = {
        "status": "PASS",
        "checkpoint_step": int(checkpoint["completed_steps"]),
        "visit_audit": visit_audit(),
        "gradient_audits": audits,
    }
    atomic_json_dump(result, OUTPUT / "gradient_audit.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
