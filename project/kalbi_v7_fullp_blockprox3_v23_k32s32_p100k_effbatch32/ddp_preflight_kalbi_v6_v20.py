#!/usr/bin/env python3
"""Eight-rank NCCL forward/backward smoke for Kalbi V6 exact prox."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from train_fpk import _load_yaml

from .canonical_penalty import build_penalty
from .model import KALBI_EXACT_PG_STEPS, MODEL_VERSION, build_model
from .selector_v20_yconditioned import (
    DirectPenaltyMLPQuerySelectorV20YConditioned,
    V20_SELECTOR_VERSION,
)
from .train_joint_v11_direct import assert_ddp_synced


PROJECT = Path("/users/minsuk/fm-lm-lambda")
EXPECTED_WORLD_SIZE = 8
EXPECTED_PARAMETER_COUNT = 3_588_547
EXPECTED_SELECTOR_PARAMETER_COUNT = 21_749_345


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != EXPECTED_WORLD_SIZE:
        raise RuntimeError(f"expected {EXPECTED_WORLD_SIZE} ranks, got {world}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(202607232359)
    model = build_model(
        _load_yaml(str(PROJECT / "fpk_v33_kkt0.yaml")), device
    )
    if sum(parameter.numel() for parameter in model.parameters()) != (
        EXPECTED_PARAMETER_COUNT
    ):
        raise RuntimeError("Kalbi V6 parameter-count drift")
    wrapped = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=True,
        find_unused_parameters=True,
    )
    before = assert_ddp_synced(wrapped.module, world, "v6_ddp_initial")

    generator = torch.Generator(device=device).manual_seed(202607240000 + rank)
    rows, features, queries = 48, 72, 2
    x = torch.randn(1, rows, features, generator=generator, device=device)
    beta_true = torch.zeros(1, features, device=device)
    beta_true[:, :8] = torch.linspace(0.9, -0.5, 8, device=device)
    y = torch.einsum("bnp,bp->bn", x, beta_true)
    y = y + 0.2 * torch.randn(
        1, rows, generator=generator, device=device
    )
    encoded = wrapped.module.encode_dataset(wrapped.module.precompute(x, y))
    lambda_max = encoded["lambda_max"].float()
    lambdas = lambda_max[:, None] * torch.tensor(
        [[0.025, 0.12]], device=device
    )
    penalty = build_penalty(torch.zeros(queries, 96, device=device))
    aux = wrapped(
        x,
        y,
        lambdas=lambdas,
        lambda_max=lambda_max[:, None],
        penalty_shape_features=penalty["shape_features"].view(
            1, queries, 196
        ),
        penalty_d0=penalty["d0"].view(1, queries),
        return_aux=True,
    )
    beta = aux["beta_q"].float()
    if aux["kalbi_v6_refinement_step_size_path_q"].shape != (
        1,
        queries,
        KALBI_EXACT_PG_STEPS,
    ):
        raise RuntimeError("Kalbi V6 DDP refinement path shape drift")
    target = beta_true[:, None, :].expand_as(beta)
    loss = (beta - target).square().mean()
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        wrapped.parameters(), 20.0, error_if_nonfinite=True
    )
    if not bool(torch.isfinite(loss)) or float(gradient_norm) <= 0.0:
        raise RuntimeError("Kalbi V6 DDP loss/gradient contract failed")
    optimizer = torch.optim.SGD(wrapped.parameters(), lr=1e-5)
    optimizer.step()
    after = assert_ddp_synced(wrapped.module, world, "v6_ddp_post_step")
    if after == before:
        raise RuntimeError("Kalbi V6 parameters did not change")

    for parameter in wrapped.module.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)
    torch.manual_seed(202607240031)
    selector = DirectPenaltyMLPQuerySelectorV20YConditioned().to(device)
    selector.fit_penalty_parameter_cache(
        torch.randn(512, 96, device=device)
    )
    if sum(parameter.numel() for parameter in selector.parameters()) != (
        EXPECTED_SELECTOR_PARAMETER_COUNT
    ):
        raise RuntimeError("Selector V20 parameter-count drift")
    selector_wrapped = DDP(
        selector,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=True,
        find_unused_parameters=False,
    )
    selector_before = assert_ddp_synced(
        selector_wrapped.module, world, "v20_ddp_initial"
    )
    x_centered = x - x.mean(dim=1, keepdim=True)
    x_scaled = x_centered / x_centered.square().mean(
        dim=1, keepdim=True
    ).sqrt().clamp_min(1e-6)
    y_centered = y - y.mean(dim=1, keepdim=True)
    y_scaled = y_centered / y_centered.square().mean(
        dim=1, keepdim=True
    ).sqrt().clamp_min(1e-6)
    selector_encoded = wrapped.module.encode_dataset(
        wrapped.module.precompute(x_scaled, y_scaled)
    )
    query = selector_wrapped(
        x_scaled,
        y_scaled,
        selector_encoded["lambda_max"].float(),
        return_representation=True,
    )
    selector_aux = wrapped.module.decode_queries(
        selector_encoded,
        lambdas=query["lambdas"],
        lambda_max=selector_encoded["lambda_max"].float()[:, None],
        penalty_shape_features=query["penalty_shape_features"],
        penalty_d0=query["penalty_d0"],
        return_aux=True,
    )
    selector_prediction = torch.einsum(
        "bnp,bp->bn", x_scaled.float(), selector_aux["beta_q"][:, 0].float()
    )
    selector_loss = torch.log1p(
        0.5 * (selector_prediction - y_scaled.float()).square().mean(-1)
    ).mean()
    selector_loss.backward()
    response_token_gradient = torch.nn.utils.clip_grad_norm_(
        selector_wrapped.module.response_token_projection.parameters(),
        1e9,
        error_if_nonfinite=True,
    )
    response_pool_gradient = torch.nn.utils.clip_grad_norm_(
        selector_wrapped.module.response_pool_projection.parameters(),
        1e9,
        error_if_nonfinite=True,
    )
    selector_gradient_norm = torch.nn.utils.clip_grad_norm_(
        selector_wrapped.parameters(), 20.0, error_if_nonfinite=True
    )
    if (
        float(response_token_gradient) <= 0.0
        or float(response_pool_gradient) <= 0.0
        or float(selector_gradient_norm) <= 0.0
    ):
        raise RuntimeError("Selector V20 DDP response gradient contract failed")
    selector_optimizer = torch.optim.SGD(selector_wrapped.parameters(), lr=1e-5)
    selector_optimizer.step()
    selector_after = assert_ddp_synced(
        selector_wrapped.module, world, "v20_ddp_post_step"
    )
    if selector_after == selector_before:
        raise RuntimeError("Selector V20 parameters did not change")

    local_metrics = torch.tensor(
        [
            float(loss.detach()),
            float(gradient_norm.detach()),
            float(
                aux["kalbi_v6_refinement_majorization_ratio_q"].max().detach()
            ),
            float(
                aux[
                    "kalbi_v6_refinement_min_curvature_denominator_q"
                ].min().detach()
            ),
            float(selector_loss.detach()),
            float(response_token_gradient.detach()),
            float(response_pool_gradient.detach()),
            float(selector_gradient_norm.detach()),
        ],
        device=device,
    )
    dist.all_reduce(local_metrics, op=dist.ReduceOp.SUM)
    local_metrics /= float(world)
    if rank == 0:
        print(
            json.dumps(
                {
                    "model_version": MODEL_VERSION,
                    "world_size": world,
                    "parameter_count": EXPECTED_PARAMETER_COUNT,
                    "mean_loss": float(local_metrics[0]),
                    "mean_gradient_norm": float(local_metrics[1]),
                    "mean_rank_majorization_ratio_max": float(
                        local_metrics[2]
                    ),
                    "mean_rank_min_curvature_denominator": float(
                        local_metrics[3]
                    ),
                    "selector_version": V20_SELECTOR_VERSION,
                    "selector_parameter_count": EXPECTED_SELECTOR_PARAMETER_COUNT,
                    "mean_selector_loss": float(local_metrics[4]),
                    "mean_response_token_gradient": float(local_metrics[5]),
                    "mean_response_pool_gradient": float(local_metrics[6]),
                    "mean_selector_gradient_norm": float(local_metrics[7]),
                    "ddp_state_synchronized": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
