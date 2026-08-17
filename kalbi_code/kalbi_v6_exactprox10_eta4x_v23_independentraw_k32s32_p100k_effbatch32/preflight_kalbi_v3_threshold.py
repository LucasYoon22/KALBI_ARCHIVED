#!/usr/bin/env python3
"""Focused Kalbi V3 terminal-threshold architecture and gradient preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .canonical_penalty_cache import CanonicalPenaltyCache
from .model import (
    KALBI_BETA_BOUND,
    KALBI_STAGES,
    KALBI_TOPK,
    MODEL_VERSION,
    build_model,
)
from .train import standardize_tasks
from train_fpk import _build_X_y_tensors, _load_yaml


EXPECTED_MODEL_PARAMETER_COUNT = 3_588_547


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--penalty-cache", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=202607220931)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32 - 1))
    rows, features = 96, 1_200
    latent = torch.randn(rows, 24)
    loadings = torch.randn(24, features) / np.sqrt(24.0)
    x = 0.55 * (latent @ loadings) + 0.45 * torch.randn(rows, features)
    true_beta = torch.zeros(features)
    true_beta[torch.randperm(features)[:24]] = torch.randn(24)
    y = x @ true_beta + 0.2 * torch.randn(rows)
    xs, ys = standardize_tasks([x.numpy()], [y.numpy()])
    x_tensor, y_tensor = _build_X_y_tensors(
        xs, ys, device=device, dtype=torch.float32
    )

    model = build_model(_load_yaml(str(args.config)), device).train()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    require(
        parameter_count == EXPECTED_MODEL_PARAMETER_COUNT,
        f"model parameter count drift: {parameter_count}",
    )
    cache = CanonicalPenaltyCache(args.penalty_cache, seed=int(args.seed) + 1)
    cache.load(expected_count=300_000)
    penalty = cache.encoded_indices(np.asarray([123_456], dtype=np.int64), device)

    encoded = model.encode_dataset(model.precompute(x_tensor, y_tensor))
    ratios = torch.tensor([1e-6, 1e-3, 1.0], device=device)
    lambda_max = encoded["lambda_max"].float().reshape(1)
    d0_scalar = penalty["d0"].reshape(1)
    lambdas = ratios * lambda_max[0] / d0_scalar[0].clamp_min(1e-12)
    shape = penalty["shape_features"].expand(ratios.numel(), -1)
    d0 = penalty["d0"].expand(ratios.numel())
    aux = model.decode_queries(
        encoded,
        lambdas=lambdas[None],
        lambda_max=lambda_max[:, None].expand(1, ratios.numel()),
        penalty_shape_features=shape[None],
        penalty_d0=d0[None],
        return_aux=True,
    )

    beta = aux["beta_q"].float()
    threshold = aux["kalbi_v3_threshold_q"].float()
    log_multiplier = aux["kalbi_v3_threshold_log_multiplier_q"].float()
    prethreshold = aux["kalbi_v3_prethreshold_beta_q"].float()
    support = aux["kalbi_v3_support_mask"].bool()
    expected_shape = (1, ratios.numel(), features)
    for name, value in (
        ("beta", beta),
        ("threshold", threshold),
        ("log_multiplier", log_multiplier),
        ("prethreshold", prethreshold),
    ):
        require(tuple(value.shape) == expected_shape, f"{name} shape drift")
        require(bool(torch.isfinite(value).all()), f"{name} is nonfinite")
    require(bool((threshold >= 0.0).all()), "threshold became negative")
    require(
        float(beta.detach().abs().max()) <= KALBI_BETA_BOUND + 1e-6,
        "beta left the canonical domain",
    )
    require(
        tuple(support.shape) == (1, features), "support-mask shape drift"
    )
    outside = ~support[0]
    require(bool(outside.any()), "preflight has no outside-support features")
    require(
        float(threshold.detach()[0, :, outside].min()) > 0.0,
        "outside-support threshold was not emitted",
    )
    nonzero_fraction = (beta != 0.0).float().mean(dim=-1)[0]
    require(
        float(nonzero_fraction[-1]) < float(nonzero_fraction[0]),
        "high lambda did not increase exact sparsity",
    )
    require(
        float(log_multiplier.detach().std()) > 1e-7,
        "learned threshold multiplier is feature-constant",
    )

    prediction = torch.einsum("bnp,bqp->bqn", x_tensor.float(), beta)
    loss = 0.5 * (prediction - y_tensor[:, None].float()).square().mean()
    loss.backward()
    output_gradient = model.kalbi_ab_head.out.weight.grad
    require(output_gradient is not None, "joint output head gradient is missing")
    correction_gradient = float(output_gradient[0].norm())
    threshold_gradient = float(output_gradient[1].norm())
    require(correction_gradient > 0.0, "correction output gradient is zero")
    require(threshold_gradient > 0.0, "threshold output gradient is zero")

    report = {
        "model_version": MODEL_VERSION,
        "parameter_count": parameter_count,
        "N": rows,
        "P": features,
        "K": KALBI_TOPK,
        "S": KALBI_STAGES,
        "support_size": int(support.sum()),
        "outside_support_size": int(outside.sum()),
        "effective_lambda_ratios": ratios.cpu().tolist(),
        "nonzero_fraction": nonzero_fraction.detach().cpu().tolist(),
        "threshold_mean": threshold.mean(dim=-1)[0].detach().cpu().tolist(),
        "threshold_min": float(threshold.detach().min()),
        "threshold_max": float(threshold.detach().max()),
        "log_multiplier_std": float(log_multiplier.detach().std()),
        "prethreshold_max_abs": float(prethreshold.detach().abs().max()),
        "beta_max_abs": float(beta.detach().abs().max()),
        "correction_output_gradient_norm": correction_gradient,
        "threshold_output_gradient_norm": threshold_gradient,
        "passed": True,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
