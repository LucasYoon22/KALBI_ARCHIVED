from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from train_fpk import _load_yaml

from .model import build_model
from .preflight_kalbi_v6_v20 import build_penalty


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model = build_model(_load_yaml(str(args.config)), device)
    incompatible = model.load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict load failed: {incompatible}")
    model.eval()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(314159)
    rows = 48
    features = 64
    x = torch.randn(
        1, rows, features, generator=generator, dtype=torch.float32
    ).to(device)
    beta_true = torch.zeros(1, features, dtype=torch.float32, device=device)
    beta_true[0, :8] = torch.linspace(
        1.0, -0.5, 8, dtype=torch.float32, device=device
    )
    y = torch.einsum("bnp,bp->bn", x, beta_true)
    y = y + 0.125 * torch.randn(
        y.shape, generator=generator, dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        encoded = model.encode_dataset(model.precompute(x, y))
        lambda_max = encoded["lambda_max"].float()
        lambdas = lambda_max[:, None] * torch.tensor(
            [[0.025, 0.2]], dtype=torch.float32, device=device
        )
        penalty = build_penalty(
            torch.zeros(2, 96, dtype=torch.float32, device=device)
        )
        result = model.decode_queries(
            encoded,
            lambdas=lambdas,
            lambda_max=lambda_max[:, None],
            penalty_shape_features=penalty["shape_features"].view(
                1, 2, 196
            ),
            penalty_d0=penalty["d0"].view(1, 2),
            return_aux=True,
        )
        beta = result["beta_q"].float().contiguous()

    if not bool(torch.isfinite(beta).all()):
        raise RuntimeError("nonfinite beta output")
    gathered = [torch.empty_like(beta) for _ in range(world)]
    dist.all_gather(gathered, beta)
    rank_max_error = max(
        float((other - gathered[0]).abs().max().item())
        for other in gathered
    )
    local_report = {
        "rank": rank,
        "device": torch.cuda.get_device_name(device),
        "peak_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "peak_reserved_bytes": int(
            torch.cuda.max_memory_reserved(device)
        ),
        "output_max_abs": float(beta.abs().max().item()),
    }
    reports: list[dict[str, object] | None] = [None] * world
    dist.all_gather_object(reports, local_report)
    if rank == 0:
        payload = {
            "kind": "migration_ddp_forward_audit",
            "completed_steps": int(checkpoint["completed_steps"]),
            "world_size": world,
            "strict_missing_keys": len(incompatible.missing_keys),
            "strict_unexpected_keys": len(incompatible.unexpected_keys),
            "rank_output_max_absolute_error": rank_max_error,
            "all_finite": True,
            "ranks": reports,
        }
        print("MIGRATION_DDP_FORWARD_JSON=" + json.dumps(payload, sort_keys=True))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
