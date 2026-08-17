#!/usr/bin/env python3
"""Two-rank DDP smoke test for V24 forward/backward and state sync."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .model import build_model
from .selector_v24_additive512_trajectory import (
    DirectPenaltyMLPQuerySelectorV24Additive512,
    load_v23_state_into_v24,
)
from train_fpk import _load_yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scaler-cache", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 2:
        raise RuntimeError("V24 DDP smoke fixes world size 2")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}")
    if sha256(args.checkpoint) != args.checkpoint_sha256:
        raise RuntimeError("DDP smoke checkpoint hash mismatch")
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    beta = build_model(_load_yaml(str(args.config)), device)
    beta.load_state_dict(checkpoint["model_state_dict"], strict=True)
    beta.eval()
    for parameter in beta.parameters():
        parameter.requires_grad_(False)
    torch.manual_seed(202607270930)
    selector = DirectPenaltyMLPQuerySelectorV24Additive512(
        hidden=512,
        derivative_floor=1e-6,
        squareplus_scale=0.02,
        direct_head_init_std=1e-3,
    )
    load_v23_state_into_v24(selector, checkpoint["selector_state_dict"])
    selector.load_v24_scaler_cache(
        torch.load(args.scaler_cache, map_location="cpu", weights_only=False)
    )
    selector.to(device)
    wrapped = DDP(
        selector,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=True,
        find_unused_parameters=False,
    )
    generator = torch.Generator(device=device).manual_seed(202607270931 + rank)
    x = torch.randn(1, 96 + 8 * rank, 192 + 32 * rank, generator=generator, device=device)
    y = torch.randn(1, x.shape[1], generator=generator, device=device)
    x = (x - x.mean(1, keepdim=True)) / x.std(
        1, keepdim=True, correction=0
    ).clamp_min(1e-6)
    y = (y - y.mean(1, keepdim=True)) / y.std(
        1, keepdim=True, correction=0
    ).clamp_min(1e-6)
    with torch.no_grad():
        encoded = beta.encode_dataset(beta.precompute(x, y))
    result = wrapped(
        x,
        y,
        encoded["lambda_max"].float(),
        solver_context=encoded,
        return_representation=True,
    )
    output = beta.decode_queries(
        encoded,
        lambdas=result["lambdas"],
        lambda_max=encoded["lambda_max"].float()[:, None],
        penalty_shape_features=result["penalty_shape_features"],
        penalty_d0=result["penalty_d0"],
        return_aux=True,
    )
    prediction = torch.einsum("bnp,bp->bn", x, output["beta_q"][:, 0])
    loss = (prediction - y).square().mean()
    loss.backward()
    if any(parameter.grad is not None for parameter in beta.parameters()):
        raise RuntimeError("frozen beta received a parameter gradient")
    if selector.delta_loglambda_head.weight.grad is None:
        raise RuntimeError("V24 DDP correction head gradient missing")
    checksum = torch.cat(
        [parameter.detach().reshape(-1)[:1] for parameter in selector.parameters()]
    ).sum()
    gathered = [torch.zeros_like(checksum) for _ in range(world)]
    dist.all_gather(gathered, checksum)
    if not all(torch.equal(gathered[0], value) for value in gathered[1:]):
        raise RuntimeError("V24 DDP initial state is not synchronized")
    if rank == 0:
        print(
            {
                "status": "PASS",
                "world_size": world,
                "loss_rank0": float(loss.detach()),
                "gamma_contract_error": float(
                    result["v24_probe_gamma_contract_error"].max().detach()
                ),
                "beta_parameters_with_grad": 0,
            },
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
