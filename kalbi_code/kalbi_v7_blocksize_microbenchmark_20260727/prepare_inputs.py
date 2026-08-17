#!/usr/bin/env python3
"""Compute and freeze the shared beta_initial once for every selected task."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import torch

from .common import (
    CHECKPOINT,
    CHECKPOINT_SHA256,
    OUTPUT,
    atomic_json_dump,
    atomic_torch_save,
    lambda_maximum,
    load_model_and_checkpoint,
    load_query_payload,
    sha256_file,
    standardized_task,
)
from .metrics import tensor_sha256
from .model_adapter import compute_beta_initial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    manifest = json.loads((OUTPUT / "sampling_manifest.json").read_text())
    query = load_query_payload(device)
    model, checkpoint = load_model_and_checkpoint(device)
    prepared: list[dict[str, object]] = []

    for task in manifest["tasks"]:
        task_index = int(task["dataset_index"])
        output_path = OUTPUT / "precomputed" / f"task_{task_index:02d}.pt"
        if output_path.exists():
            raise RuntimeError(f"refusing to overwrite {output_path}")
        x_cpu, y_cpu, preprocessing = standardized_task(task)
        x = x_cpu.to(device)
        y = y_cpu.to(device)
        lambda_max = lambda_maximum(x, y)
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            encoded = model.encode_dataset(model.precompute(x, y))
            initial = compute_beta_initial(
                model,
                encoded,
                lambdas=query["lambdas"],
                lambda_max=lambda_max,
                penalty_shape_features=query["shape_features"],
                penalty_d0=query["d0"],
            )
        feature_mask = encoded.get("feature_mask")
        if feature_mask is None:
            feature_mask = torch.ones(
                1,
                x.shape[-1],
                dtype=torch.bool,
                device=device,
            )
        tensors = {
            "x": x_cpu.contiguous(),
            "y": y_cpu.contiguous(),
            "lambda_max": lambda_max.detach().cpu().contiguous(),
            "lambdas": query["lambdas"].detach().cpu().contiguous(),
            "penalty_shape_features": query["shape_features"]
            .detach()
            .cpu()
            .contiguous(),
            "penalty_d0": query["d0"].detach().cpu().contiguous(),
            "rho": query["rho"].detach().cpu().contiguous(),
            "drho": query["drho"].detach().cpu().contiguous(),
            "feature_mask": feature_mask.detach().bool().cpu().contiguous(),
            "beta_initial": initial["beta_initial"]
            .detach()
            .float()
            .cpu()
            .contiguous(),
            "prethreshold": initial["prethreshold"]
            .detach()
            .float()
            .cpu()
            .contiguous(),
            "log_threshold_multiplier": initial[
                "log_threshold_multiplier"
            ]
            .detach()
            .float()
            .cpu()
            .contiguous(),
            "gram_diagonal": initial["gram_diagonal"]
            .detach()
            .float()
            .cpu()
            .contiguous(),
        }
        hashes = {name: tensor_sha256(value) for name, value in tensors.items()}
        payload = {
            **tensors,
            "tensor_hashes": hashes,
            "task": task,
            "preprocessing": preprocessing,
            "checkpoint_path": str(CHECKPOINT),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_step": int(checkpoint["completed_steps"]),
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        atomic_torch_save(payload, output_path)
        prepared.append(
            {
                "dataset_index": task_index,
                "path": str(output_path),
                "file_sha256": sha256_file(output_path),
                "tensor_hashes": hashes,
                "n": int(task["n"]),
                "p": int(task["p"]),
            }
        )
        del x, y, encoded, initial
        torch.cuda.empty_cache()

    atomic_json_dump(
        {
            "version": "kalbi_v7_blocksize_precomputed_inputs_v1",
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_step": int(checkpoint["completed_steps"]),
            "tasks": prepared,
        },
        OUTPUT / "precomputed_manifest.json",
    )
    print(json.dumps(prepared, indent=2))


if __name__ == "__main__":
    main()
