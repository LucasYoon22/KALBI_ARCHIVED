#!/usr/bin/env python3
"""Minimal four-rank NCCL and object-collective preflight."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist


def emit(rank: int, stage: str, **values: object) -> None:
    print(json.dumps({"rank": rank, "stage": stage, **values}, sort_keys=True), flush=True)


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    emit(rank, "initialized")
    value = torch.tensor([float(rank + 1)], device=f"cuda:{local_rank}")
    dist.all_reduce(value)
    emit(rank, "all_reduce", value=float(value.item()))
    payload = [{"message": "catalog-broadcast-smoke", "blob": "x" * 1_000_000} if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    emit(rank, "object_broadcast", bytes=len(payload[0]["blob"]))
    dist.barrier()
    emit(rank, "barrier")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
