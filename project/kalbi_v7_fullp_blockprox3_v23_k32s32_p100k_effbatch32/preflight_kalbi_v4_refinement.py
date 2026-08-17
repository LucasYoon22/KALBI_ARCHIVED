#!/usr/bin/env python3
"""Focused V4 forward/backward and gradient-contract preflight."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from train_fpk import _load_yaml

from .model import MODEL_VERSION, build_model
from .utils import GRID_SIZE, NATIVE_DIM, make_penalty_grid


PROJECT = Path("/users/minsuk/fm-lm-lambda")


def gradient_norm(parameters) -> float:
    total = torch.zeros((), device="cuda")
    for parameter in parameters:
        if parameter.grad is not None:
            total = total + parameter.grad.float().square().sum()
    return float(total.sqrt())


def main() -> None:
    torch.manual_seed(202607232131)
    device = torch.device("cuda:0")
    model = build_model(_load_yaml(str(PROJECT / "fpk_v33_kkt0.yaml")), device)
    model.train()

    batch, rows, features, queries = 1, 64, 96, 3
    latent = torch.randn(batch, rows, 8, device=device)
    loading = torch.randn(batch, 8, features, device=device) / 8**0.5
    x = torch.matmul(latent, loading) + 0.35 * torch.randn(
        batch, rows, features, device=device
    )
    beta_true = torch.zeros(batch, features, device=device)
    beta_true[:, :8] = torch.linspace(1.0, -0.7, 8, device=device)
    y = torch.einsum("bnp,bp->bn", x, beta_true) + 0.25 * torch.randn(
        batch, rows, device=device
    )

    with torch.no_grad():
        encoded = model.encode_dataset(model.precompute(x, y))
        lambda_max = encoded["lambda_max"].float()
    ratios = torch.tensor([[0.01, 0.05, 0.2]], device=device)
    lambdas = (ratios * lambda_max[:, None]).detach().requires_grad_(True)

    grid = make_penalty_grid().to(device)
    rho = (grid / grid[-1]).clamp(0.0, 1.0)
    drho = torch.full_like(grid, 1.0 / float(grid[-1]))
    native = torch.stack([rho, drho], dim=-1).reshape(2 * GRID_SIZE)
    native = torch.cat([native, torch.zeros(4, device=device)])
    shape = native.view(1, 1, NATIVE_DIM).expand(
        batch, queries, NATIVE_DIM
    ).clone().detach().requires_grad_(True)
    d0 = torch.full(
        (batch, queries), 1.0 / float(grid[-1]), device=device
    ).requires_grad_(True)

    aux = model(
        x,
        y,
        lambdas=lambdas,
        lambda_max=lambda_max[:, None],
        penalty_shape_features=shape,
        penalty_d0=d0,
        return_aux=True,
    )
    beta = aux["beta_q"].float()
    target = beta_true[:, None, :].expand_as(beta)
    loss = (beta - target).square().mean()
    loss.backward()

    checks = {
        "beta_finite": bool(torch.isfinite(beta).all()),
        "loss_finite": bool(torch.isfinite(loss)),
        "lambda_gradient": float(lambdas.grad.float().norm()),
        "shape_gradient": float(shape.grad.float().norm()),
        "d0_gradient": float(d0.grad.float().norm()),
        "v3_ab_head_gradient": gradient_norm(model.kalbi_ab_head.parameters()),
        "v4_refinement_gradient": gradient_norm(
            model.kalbi_refinement_control.parameters()
        ),
        "step_fraction_mean": float(
            aux["kalbi_v4_refinement_step_fraction_q"].float().mean().detach()
        ),
        "refinement_change_rms": float(
            aux["kalbi_v4_refinement_change_rms_q"].float().mean().detach()
        ),
        "active_fraction": float(
            aux["kalbi_v4_refined_active_fraction_q"].float().mean().detach()
        ),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_version": MODEL_VERSION,
    }
    positive_gradient_keys = (
        "lambda_gradient",
        "shape_gradient",
        "d0_gradient",
        "v3_ab_head_gradient",
        "v4_refinement_gradient",
    )
    if not checks["beta_finite"] or not checks["loss_finite"]:
        raise RuntimeError(f"nonfinite V4 preflight: {checks}")
    if any(checks[key] <= 0.0 for key in positive_gradient_keys):
        raise RuntimeError(f"missing V4 gradient path: {checks}")
    print(json.dumps(checks, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
