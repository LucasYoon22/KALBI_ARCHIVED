#!/usr/bin/env python3
"""One-GPU max-cell beta/selector gradient and memory preflight for V12."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .canonical_penalty import DIRECT_PARAMETER_DIM, canonical_grid
from .canonical_penalty_cache import CanonicalPenaltyCache
from .model import build_model
from .selector_v10 import summarize_encoded
from .selector_v12_canonical import DirectPenaltyMLPQuerySelectorV12Canonical
from .train import make_sampler, module_grad_norm, objective_exp125_v58
from .train_joint_v11_direct import grad_stats, selector_path_features, tensor_gradient_stats
from .train_joint_v12_canonical_rolling import (
    AUDITED_REFERENCE_INDICES,
    SELECTOR_PATH_FEATURES,
    SELECTOR_RATIO_MAX,
    SELECTOR_RATIO_MIN,
    make_canonical_query_payload,
    refit_canonical_shape_normalizers,
    split_and_standardize_variable,
)
from train_fpk import _load_yaml


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--penalty-cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202607170301)
    args = parser.parse_args()

    require(torch.cuda.is_available(), "CUDA is required")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    # This Vast image's torch build accepts an integer CUDA index here but
    # rejects a torch.device object even though tensor placement accepts it.
    torch.cuda.reset_peak_memory_stats(0)

    with np.load(args.task, allow_pickle=False) as payload:
        x_np = np.asarray(payload["X"][0], dtype=np.float32)
        y_np = np.asarray(payload["y"][0], dtype=np.float32)
    xs, ys = [x_np], [y_np]
    n, p = map(int, x_np.shape)

    config = _load_yaml(str(args.config))
    warm = torch.load(args.warm_start, map_location="cpu", weights_only=False)
    model = build_model(config, device)
    model.load_state_dict(warm["model_state_dict"], strict=True)
    require(sum(parameter.numel() for parameter in model.parameters()) == 4_909_208, "beta size drift")

    cache = CanonicalPenaltyCache(args.penalty_cache, args.seed + 101)
    cache_report = cache.load(expected_count=300_000)
    sampler = make_sampler(config, args.seed + 211)
    normalizer_report = refit_canonical_shape_normalizers(model, cache, sampler)

    selector = DirectPenaltyMLPQuerySelectorV12Canonical(
        ds_feat_dim=9,
        t_feat_dim=32 * 33 // 2,
        krylov_dim=32,
        shape_dim=196,
        path_feat_dim=SELECTOR_PATH_FEATURES,
    )
    calibration = selector.fit_penalty_parameter_cache(cache.direct_parameters)
    selector.to(device)
    require(sum(parameter.numel() for parameter in selector.parameters()) == 5_161_389, "selector size drift")

    direct_parameters = list(selector.penalty_parameter_head.parameters())
    direct_ids = {id(parameter) for parameter in direct_parameters}
    other_parameters = [parameter for parameter in selector.parameters() if id(parameter) not in direct_ids]
    selector_optimizer = torch.optim.AdamW(
        [
            {"params": other_parameters, "lr": 3e-4},
            {"params": direct_parameters, "lr": 1e-5},
        ],
        weight_decay=0.0,
    )

    knots = canonical_grid(device=device)
    fixed = cache.encoded_indices(np.asarray(AUDITED_REFERENCE_INDICES), device)
    path_ratios = torch.exp(
        torch.linspace(math.log(SELECTOR_RATIO_MIN), math.log(SELECTOR_RATIO_MAX), 50, device=device)
    )
    split_generator = torch.Generator(device="cpu").manual_seed(args.seed + 301)

    beta_started = time.perf_counter()
    model.train()
    model.zero_grad(set_to_none=True)
    beta_payload = make_canonical_query_payload(xs, ys, cache, sampler, device)
    beta_result = objective_exp125_v58(model, beta_payload, knots)
    require(bool(torch.isfinite(beta_result["loss"])), "nonfinite max-cell beta loss")
    beta_result["loss"].backward()
    beta_group_gradients = {
        "shape": module_grad_norm(model.cond_embed.penalty_shape_encoder),
        "query": module_grad_norm(model.query_mixer),
        "feature": module_grad_norm(model.feature_head),
        "local": module_grad_norm(model.v46_local_mixer),
    }
    require(all(math.isfinite(value) and value > 0.0 for value in beta_group_gradients.values()), "beta group gradient failed")
    beta_seconds = time.perf_counter() - beta_started
    model.zero_grad(set_to_none=True)

    was_training = model.training
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    steps = []
    for update in (1, 2):
        selector_optimizer.zero_grad(set_to_none=True)
        x_train, y_train, x_test, y_test = split_and_standardize_variable(
            xs, ys, generator=split_generator, device=device
        )
        started = time.perf_counter()
        with torch.no_grad():
            encoded = model.encode_dataset(model.precompute(x_train, y_train))
            dataset_features, krylov_features = summarize_encoded(encoded, 32 * 33 // 2)
            path_features = selector_path_features(
                model,
                encoded,
                x_train,
                y_train,
                fixed["shape_features"],
                fixed["d0"],
                path_ratios,
                25,
            )
        query = selector(
            dataset_features,
            krylov_features,
            encoded["lambda_max"].float(),
            path_features,
            U=encoded.get("U"),
            y_context=y_train,
        )
        for name in ("penalty_shape_features", "penalty_d0", "lambdas"):
            query[name].retain_grad()
        aux = model.decode_queries(
            encoded,
            lambdas=query["lambdas"],
            lambda_max=encoded["lambda_max"].float()[:, None],
            penalty_shape_features=query["penalty_shape_features"],
            penalty_d0=query["penalty_d0"],
            return_aux=True,
        )
        require("beta_q" in aux, "actual hard beta_q missing")
        beta = aux["beta_q"][:, 0].float()
        prediction = torch.einsum("bnp,bp->bn", x_test.float(), beta)
        mse = (prediction - y_test.float()).square().mean(-1)
        loss = torch.log1p(0.5 * mse).mean()
        require(bool(torch.isfinite(loss)), "nonfinite max-cell selector loss")
        proposal = selector.penalty_parameter_head._last_parameter_proposal
        require(proposal is not None, "Direct96 proposal missing")
        loss.backward()
        require(proposal.grad is not None, "Direct96 gradient missing")
        direct_matrix = proposal.grad.detach().float().reshape(-1, DIRECT_PARAMETER_DIM)
        active = int((direct_matrix.abs().sum(0) != 0.0).sum())
        require(active == DIRECT_PARAMETER_DIM, f"Direct96 coverage failed: {active}/96")
        gradients = {}
        for name in ("penalty_shape_features", "penalty_d0", "lambdas"):
            require(query[name].grad is not None, f"{name} output gradient missing")
            stats = tensor_gradient_stats([query[name].grad.detach().float()])
            require(math.isfinite(float(stats["norm"])) and float(stats["norm"]) > 0.0, f"{name} gradient failed")
            gradients[name] = stats
        groups = {
            "penalty_head": grad_stats(selector.penalty_parameter_head.parameters()),
            "lambda_head": grad_stats(selector.lambda_head.parameters()),
            "penalty_trunk": grad_stats(selector.penalty_net.parameters()),
            "lambda_trunk": grad_stats(selector.lambda_net.parameters()),
            "y_token_mlp": grad_stats(selector.y_token_mlp.parameters()),
        }
        if update >= 2:
            require(all(float(groups[name]["norm"]) > 0.0 for name in groups), "step-two group gradient failed")
        positivity = float(query["penalty_positivity_jacobian_min"].min().detach())
        min_drho = float(query["penalty_drho"].min().detach())
        rho6_error = float(query["penalty_rho6_error"].abs().max().detach())
        require(positivity > 0.0 and min_drho > 0.0 and rho6_error <= 1e-5, "penalty feasibility failed")
        selector_optimizer.step()
        steps.append(
            {
                "update": update,
                "seconds": time.perf_counter() - started,
                "loss": float(loss.detach()),
                "heldout_mse": float(mse.mean().detach()),
                "direct96_active_coordinates": active,
                "direct96_gradient_norm": float(direct_matrix.norm()),
                "output_gradients": gradients,
                "group_gradients": groups,
                "positivity_jacobian_min": positivity,
                "min_drho": min_drho,
                "rho6_error": rho6_error,
            }
        )

    require(not any(parameter.grad is not None for parameter in model.parameters()), "frozen beta received selector gradients")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.train(was_training)

    report = {
        "status": "PASS",
        "task": str(args.task),
        "shape": [n, p],
        "cells": n * p,
        "warm_start": str(args.warm_start),
        "cache": cache_report,
        "normalizer_report": normalizer_report,
        "calibration": calibration,
        "architecture": selector.architecture_report(),
        "beta": {
            "seconds": beta_seconds,
            "loss": float(beta_result["loss"].detach()),
            "group_gradients": beta_group_gradients,
        },
        "selector_updates": steps,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "total_cuda_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "frozen_beta_received_selector_gradients": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
