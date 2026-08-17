#!/usr/bin/env python3
"""End-to-end Kalbi V5 exact-prox forward, descent, and gradient preflight."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from train_fpk import _load_yaml

from .canonical_penalty import build_penalty
from .exact_direct96_prox import (
    interpolate_direct96_penalty,
    unpack_direct96_penalty,
)
from .model import MODEL_VERSION, build_model
from .train import objective_kalbi_v5_exactprox


PROJECT = Path("/users/minsuk/fm-lm-lambda")


def gradient_norm(parameters) -> float:
    total = torch.zeros((), device="cuda")
    for parameter in parameters:
        if parameter.grad is not None:
            total = total + parameter.grad.float().square().sum()
    return float(total.sqrt())


def make_task(
    batch: int,
    rows: int,
    features: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    latent_width = min(12, max(4, features // 8))
    latent = torch.randn(batch, rows, latent_width, device=device)
    loading = (
        torch.randn(batch, latent_width, features, device=device)
        / latent_width**0.5
    )
    x = torch.matmul(latent, loading) + 0.4 * torch.randn(
        batch, rows, features, device=device
    )
    beta_true = torch.zeros(batch, features, device=device)
    signal_width = min(10, features)
    beta_true[:, :signal_width] = torch.linspace(
        1.0, -0.6, signal_width, device=device
    )
    y = torch.einsum("bnp,bp->bn", x, beta_true)
    y = y + 0.3 * torch.randn(batch, rows, device=device)
    return x, y, beta_true


def composite_objective(
    beta: torch.Tensor,
    x_centered: torch.Tensor,
    y_centered: torch.Tensor,
    lambdas: torch.Tensor,
    native196: torch.Tensor,
    grid: torch.Tensor,
) -> torch.Tensor:
    prediction = torch.matmul(beta.float(), x_centered.float().transpose(1, 2))
    data_half = 0.5 * (
        prediction - y_centered.float()[:, None, :]
    ).square().mean(dim=-1)
    pieces = unpack_direct96_penalty(native196.float(), grid)
    rho, _, _ = interpolate_direct96_penalty(beta.float().abs(), pieces, grid)
    return data_half + lambdas.float() * rho.sum(dim=-1)


def run_shape_smoke(
    model: torch.nn.Module,
    *,
    rows: int,
    features: int,
    device: torch.device,
) -> dict[str, float]:
    x, y, _ = make_task(1, rows, features, device=device)
    encoded = model.encode_dataset(model.precompute(x, y))
    lambda_max = encoded["lambda_max"].float()
    lambdas = lambda_max[:, None] * torch.tensor(
        [[0.03, 0.15]], device=device
    )
    direct = torch.zeros(2, 96, device=device)
    penalty = build_penalty(direct)
    aux = model.decode_queries(
        encoded,
        lambdas=lambdas,
        lambda_max=lambda_max[:, None],
        penalty_shape_features=penalty["shape_features"].view(1, 2, 196),
        penalty_d0=penalty["d0"].view(1, 2),
        return_aux=True,
    )
    beta = aux["beta_q"].float()
    return {
        "rows": float(rows),
        "features": float(features),
        "beta_abs_max": float(beta.abs().max()),
        "majorization_ratio_max": float(
            aux["kalbi_v5_refinement_majorization_ratio_q"].max()
        ),
        "minimum_curvature_denominator": float(
            aux["kalbi_v5_refinement_min_curvature_denominator_q"].min()
        ),
    }


def run_training_objective_smoke(
    model: torch.nn.Module,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.zero_grad(set_to_none=True)
    x, y, _ = make_task(1, 52, 80, device=device)
    with torch.no_grad():
        encoded = model.encode_dataset(model.precompute(x, y))
        lambda_max = encoded["lambda_max"].float()
    lambdas = lambda_max[:, None] * torch.tensor(
        [[0.035, 0.14]], device=device
    )
    penalty = build_penalty(
        0.1 * torch.randn(2, 96, device=device)
    )
    payload = {
        "X": x,
        "y": y,
        "lambdas": lambdas,
        "lambda_max": lambda_max[:, None].expand(1, 2),
        "shape": penalty["shape_features"].view(1, 2, 196),
        "d0": penalty["d0"].view(1, 2),
        "relu_b": penalty["relu_b"],
        "relu_weights": penalty["relu_weights"],
    }
    result = objective_kalbi_v5_exactprox(
        model, payload, model._kalbi_v5_penalty_grid
    )
    result["loss"].backward()
    scalar_names = (
        "loss",
        "data_half",
        "penalty",
        "objective",
        "kkt_top32",
        "refinement_step_size",
        "refinement_min_curvature_denominator",
        "refinement_majorization_ratio",
        "refinement_change_rms",
    )
    if any(not bool(torch.isfinite(result[name])) for name in scalar_names):
        raise RuntimeError("Kalbi V5 trainer objective produced nonfinite metrics")
    return {
        "loss": float(result["loss"].detach()),
        "kkt_top32": float(result["kkt_top32"].detach()),
        "majorization_ratio": float(
            result["refinement_majorization_ratio"].detach()
        ),
        "model_gradient": gradient_norm(model.parameters()),
    }


def main() -> None:
    torch.manual_seed(202607232355)
    device = torch.device("cuda:0")
    model = build_model(_load_yaml(str(PROJECT / "fpk_v33_kkt0.yaml")), device)
    model.train()

    batch, rows, features, queries = 1, 64, 96, 3
    x, y, beta_true = make_task(batch, rows, features, device=device)
    encoded = model.encode_dataset(model.precompute(x, y))
    lambda_max = encoded["lambda_max"].float()
    lambdas = (
        lambda_max[:, None]
        * torch.tensor([[0.015, 0.06, 0.2]], device=device)
    ).detach().requires_grad_(True)
    direct = (
        0.15 * torch.randn(batch, queries, 96, device=device)
    ).requires_grad_(True)
    penalty = build_penalty(direct.reshape(-1, 96))
    native = penalty["shape_features"].reshape(batch, queries, 196)
    d0 = penalty["d0"].reshape(batch, queries)

    aux = model.decode_queries(
        encoded,
        lambdas=lambdas,
        lambda_max=lambda_max[:, None],
        penalty_shape_features=native,
        penalty_d0=d0,
        return_aux=True,
    )
    beta0 = aux["kalbi_v5_beta0_q"].float()
    beta = aux["beta_q"].float()
    grid = model._kalbi_v5_penalty_grid.float()
    objective0 = composite_objective(
        beta0, encoded["Xc"], encoded["yc"], lambdas, native, grid
    )
    objective1 = composite_objective(
        beta, encoded["Xc"], encoded["yc"], lambdas, native, grid
    )
    target = beta_true[:, None, :].expand_as(beta)
    supervised_loss = (beta - target).square().mean()
    supervised_loss.backward()

    checks = {
        "beta_finite": bool(torch.isfinite(beta).all()),
        "loss_finite": bool(torch.isfinite(supervised_loss)),
        "composite_descent_max": float(
            (objective1 - objective0).max().detach()
        ),
        "lambda_gradient": float(lambdas.grad.float().norm()),
        "direct96_gradient": float(direct.grad.float().norm()),
        "v3_ab_head_gradient": gradient_norm(model.kalbi_ab_head.parameters()),
        "base_head_gradient": gradient_norm(model.base_head.parameters()),
        "majorization_ratio_max": float(
            aux["kalbi_v5_refinement_majorization_ratio_q"].max()
        ),
        "minimum_curvature_denominator": float(
            aux["kalbi_v5_refinement_min_curvature_denominator_q"].min()
        ),
        "refinement_change_rms": float(
            aux["kalbi_v5_refinement_change_rms_q"].mean()
        ),
        "trace_fallback_fraction": float(
            aux["kalbi_v5_refinement_trace_fallback_q"].mean()
        ),
        "d0_contract_error_max": float(
            aux["kalbi_v5_refinement_d0_contract_error_q"].max()
        ),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "model_version": MODEL_VERSION,
        "shape_smokes": [
            run_shape_smoke(model, rows=48, features=73, device=device),
            run_shape_smoke(model, rows=91, features=129, device=device),
        ],
        "training_objective_smoke": run_training_objective_smoke(
            model, device=device
        ),
    }
    positive_gradient_keys = (
        "lambda_gradient",
        "direct96_gradient",
        "v3_ab_head_gradient",
        "base_head_gradient",
    )
    if not checks["beta_finite"] or not checks["loss_finite"]:
        raise RuntimeError(f"nonfinite V5 preflight: {checks}")
    if checks["composite_descent_max"] > 2e-5:
        raise RuntimeError(f"V5 composite objective did not descend: {checks}")
    if checks["majorization_ratio_max"] > 0.951:
        raise RuntimeError(f"V5 data majorization contract failed: {checks}")
    if checks["minimum_curvature_denominator"] < 0.099:
        raise RuntimeError(f"V5 curvature contract failed: {checks}")
    if checks["d0_contract_error_max"] > 2e-6:
        raise RuntimeError(f"V5 Direct96 d0 contract failed: {checks}")
    if any(checks[key] <= 0.0 for key in positive_gradient_keys):
        raise RuntimeError(f"missing V5 gradient path: {checks}")
    if checks["training_objective_smoke"]["model_gradient"] <= 0.0:
        raise RuntimeError(f"V5 trainer objective lost model gradients: {checks}")
    print(json.dumps(checks, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
