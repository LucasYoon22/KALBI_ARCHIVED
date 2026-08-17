#!/usr/bin/env python3
"""Benchmark V24 T3/T10 and audit branch diversity on 32 fixed cache tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .model import build_model
from .selector_v24_additive512_trajectory import (
    DirectPenaltyMLPQuerySelectorV24Additive512,
)
from .train_selector_only_v24_frozen_kalbi_v6_cachedtraj_8gpu import (
    split_and_standardize_variable,
)
from train_fpk import _load_yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cosine_report(values: torch.Tensor) -> dict[str, float]:
    unit = values / values.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    matrix = unit @ unit.T
    mask = torch.triu(
        torch.ones_like(matrix, dtype=torch.bool), diagonal=1
    )
    off = matrix[mask]
    return {
        "coordinate_std_mean": float(values.std(0, correction=0).mean()),
        "pairwise_cosine_mean": float(off.mean()),
        "pairwise_cosine_std": float(off.std(correction=0)),
        "pairwise_cosine_min": float(off.min()),
        "pairwise_cosine_max": float(off.max()),
        "pairwise_abs_cosine_max": float(off.abs().max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rolling-control-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tasks", type=int, default=32)
    args = parser.parse_args()

    if sha256(args.checkpoint) != args.checkpoint_sha256:
        raise RuntimeError("checkpoint SHA256 mismatch")
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    manifest = json.loads(args.manifest.read_text())
    selected = manifest["tasks"][: args.tasks]
    if len(selected) != args.tasks:
        raise RuntimeError("manifest has too few fixed tasks")
    pointer = manifest["catalog_pointer"]
    catalog_path = (
        args.rolling_control_root
        / "catalogs"
        / pointer["catalog_file"]
    )
    catalog = json.loads(catalog_path.read_text())
    entries = {
        int(entry["shard_id"]): entry for entry in catalog["files"]
    }
    if any(int(task["shard_id"]) not in entries for task in selected):
        raise RuntimeError("fixed scaler task is absent from its source catalog")

    device = torch.device(args.device)
    beta = build_model(_load_yaml(str(args.config)), device)
    beta.load_state_dict(checkpoint["model_state_dict"], strict=True)
    beta.eval()
    for parameter in beta.parameters():
        parameter.requires_grad_(False)
    selector = DirectPenaltyMLPQuerySelectorV24Additive512(
        hidden=512,
        derivative_floor=1e-6,
        squareplus_scale=0.02,
        direct_head_init_std=1e-3,
    )
    selector.load_state_dict(
        checkpoint["selector_state_dict"], strict=True
    )
    selector.to(device).eval()

    probe_seconds: list[float] = []
    original_probe = selector._probe

    def timed_probe(*probe_args: object, **probe_kwargs: object):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = original_probe(*probe_args, **probe_kwargs)
        torch.cuda.synchronize(device)
        probe_seconds.append(time.perf_counter() - started)
        return result

    selector._probe = timed_probe  # type: ignore[method-assign]
    task_codes: list[torch.Tensor] = []
    global_codes: list[torch.Tensor] = []
    trajectory_codes: list[torch.Tensor] = []
    proposal_codes: list[torch.Tensor] = []
    corrected_codes: list[torch.Tensor] = []
    penalties: list[torch.Tensor] = []
    raw_lambdas: list[torch.Tensor] = []
    final_t10_seconds: list[float] = []
    selector_total_seconds: list[float] = []
    peak_memory_bytes: list[int] = []
    task_records: list[dict[str, object]] = []

    for position, task in enumerate(selected):
        entry = entries[int(task["shard_id"])]
        path = Path(catalog["data_root"]) / entry["relative_path"]
        if sha256(path) != task["sha256"]:
            raise RuntimeError(f"fixed task hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            x_np = np.asarray(archive["X"][0], dtype=np.float32)
            y_np = np.asarray(archive["y"][0], dtype=np.float32)
        generator = torch.Generator(device="cpu").manual_seed(
            int(task["split_seed"])
        )
        x_train, y_train, _x_test, _y_test = split_and_standardize_variable(
            [x_np],
            [y_np],
            generator=generator,
            device=device,
        )
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            encoded = beta.encode_dataset(beta.precompute(x_train, y_train))
            torch.cuda.synchronize(device)
            selector_started = time.perf_counter()
            query = selector(
                x_train,
                y_train,
                encoded["lambda_max"].float(),
                solver_context=encoded,
                return_representation=True,
            )
            torch.cuda.synchronize(device)
            selector_total_seconds.append(
                time.perf_counter() - selector_started
            )
            torch.cuda.synchronize(device)
            final_started = time.perf_counter()
            beta.decode_queries(
                encoded,
                lambdas=query["lambdas"],
                lambda_max=encoded["lambda_max"].float()[:, None],
                penalty_shape_features=query["penalty_shape_features"],
                penalty_d0=query["penalty_d0"],
                return_aux=True,
            )
            torch.cuda.synchronize(device)
            final_t10_seconds.append(time.perf_counter() - final_started)
        peak_memory_bytes.append(torch.cuda.max_memory_allocated(device))
        task_codes.append(query["v24_task_code"][0].detach().cpu())
        global_codes.append(query["v24_global_code"][0].detach().cpu())
        trajectory_codes.append(
            query["v24_trajectory_code"][0].detach().cpu()
        )
        proposal_codes.append(query["v24_proposal_code"][0].detach().cpu())
        corrected_codes.append(
            query["v24_corrected_task_code"][0].detach().cpu()
        )
        penalties.append(query["penalty_rho"][0].detach().cpu())
        raw_lambdas.append(query["raw_lambda"][0].detach().cpu())
        task_records.append(
            {
                "position": position,
                "shard_id": int(task["shard_id"]),
                "seed": int(task["seed"]),
                "n": int(task["n"]),
                "p": int(task["p"]),
                "stratum": task["stratum"],
                "probe_t3_seconds": probe_seconds[-1],
                "selector_total_seconds": selector_total_seconds[-1],
                "final_t10_seconds": final_t10_seconds[-1],
                "peak_gpu_memory_bytes": peak_memory_bytes[-1],
            }
        )

    branches = {
        "task": torch.stack(task_codes),
        "global": torch.stack(global_codes),
        "trajectory": torch.stack(trajectory_codes),
        "proposal": torch.stack(proposal_codes),
        "corrected": torch.stack(corrected_codes),
    }
    branch_report = {
        name: cosine_report(values) for name, values in branches.items()
    }
    penalty = torch.stack(penalties)
    lambda_values = torch.stack(raw_lambdas).reshape(-1)
    penalty_pairwise = (
        penalty[:, None, :] - penalty[None, :, :]
    ).square().mean(-1).sqrt()
    pair_mask = torch.triu(
        torch.ones_like(penalty_pairwise, dtype=torch.bool), diagonal=1
    )

    def summary(values: list[float]) -> dict[str, float]:
        tensor = torch.tensor(values, dtype=torch.float64)
        return {
            "mean": float(tensor.mean()),
            "median": float(tensor.median()),
            "min": float(tensor.min()),
            "max": float(tensor.max()),
        }

    report = {
        "status": "PASS",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "fixed_manifest": str(args.manifest),
        "fixed_manifest_sha256": sha256(args.manifest),
        "task_count": args.tasks,
        "selection_rule": "first_32_tasks_of_precommitted_v24_scaler_manifest",
        "runtime": {
            "probe_t3_seconds": summary(probe_seconds),
            "selector_total_including_t3_seconds": summary(
                selector_total_seconds
            ),
            "final_t10_seconds": summary(final_t10_seconds),
            "peak_gpu_memory_bytes": summary(
                [float(value) for value in peak_memory_bytes]
            ),
            "concurrent_training_warning": (
                "Measured on cuda:0 while the production 8-GPU trainer was "
                "running; timings are conservative and contention-affected."
            ),
        },
        "branch_diversity": branch_report,
        "penalty_diversity": {
            "rho_coordinate_std_mean": float(
                penalty.std(0, correction=0).mean()
            ),
            "pairwise_rho_rmse_mean": float(
                penalty_pairwise[pair_mask].mean()
            ),
            "pairwise_rho_rmse_max": float(
                penalty_pairwise[pair_mask].max()
            ),
        },
        "raw_lambda_diversity": {
            "min": float(lambda_values.min()),
            "max": float(lambda_values.max()),
            "mean": float(lambda_values.mean()),
            "log_std": float(
                lambda_values.clamp_min(1e-30).log().std(correction=0)
            ),
            "unique_1e8": int(
                torch.unique(torch.round(lambda_values * 1e8)).numel()
            ),
        },
        "task_records": task_records,
    }
    if not all(
        math.isfinite(value)
        for group in branch_report.values()
        for value in group.values()
    ):
        raise RuntimeError("branch diversity report is nonfinite")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({k: report[k] for k in (
        "status", "task_count", "runtime", "branch_diversity",
        "penalty_diversity", "raw_lambda_diversity",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
