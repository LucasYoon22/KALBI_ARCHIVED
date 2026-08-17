#!/usr/bin/env python3
"""Frozen-task PG10 versus KALBI V7 block-prox preflight."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np
import torch

from . import block_prox_v7
from .canonical_penalty import build_penalty, canonical_grid
from .model import build_model as build_v7_model
from .train import (
    compute_penalty_relu_cached,
    compute_topk_kkt_relu_metric,
)
from train_fpk import _load_yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_task(
    root: Path, entry: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    path = root / entry["file"]
    if _sha256(path) != entry["sha256"]:
        raise RuntimeError(f"frozen task SHA256 mismatch: {path}")
    with np.load(path, allow_pickle=False) as payload:
        x = np.asarray(payload["X"][0], dtype=np.float32)
        y = np.asarray(payload["y"][0], dtype=np.float32)
    if x.shape != (int(entry["n"]), int(entry["p"])):
        raise RuntimeError(f"frozen task shape mismatch: {path}")
    x = (x - x.mean(0, keepdims=True)) / (
        x.std(0, keepdims=True) + 1e-8
    )
    y = (y - y.mean()) / (y.std() + 1e-8)
    return (
        torch.from_numpy(x).unsqueeze(0).to(device),
        torch.from_numpy(y).unsqueeze(0).to(device),
    )


def _make_queries(
    *,
    device: torch.device,
    query_count: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    direct = 0.15 * torch.randn(
        1, query_count, 96, generator=generator, device=device
    )
    penalty = build_penalty(direct)
    return {
        "lambdas": torch.logspace(
            -4.0, 0.0, query_count, device=device
        ).unsqueeze(0),
        "shape": penalty["shape_features"],
        "d0": penalty["d0"],
        "relu_b": penalty["relu_b"],
        "relu_weights": penalty["relu_weights"],
        "knots": penalty["knots"],
    }


def _metrics(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    beta: torch.Tensor,
    query: dict[str, torch.Tensor],
) -> dict[str, list[float]]:
    sample_count = int(x.shape[1])
    prediction = torch.einsum("bnp,bqp->bqn", x.float(), beta.float())
    residual = y.float().unsqueeze(1) - prediction
    data_half = 0.5 * residual.square().mean(-1)
    penalty = compute_penalty_relu_cached(
        beta.float(),
        query["lambdas"].float(),
        query["knots"],
        query["relu_b"].reshape(-1),
        query["relu_weights"].reshape(-1, query["relu_weights"].shape[-1]),
    )
    objective = data_half + penalty
    kkt_values = []
    for query_index in range(int(beta.shape[1])):
        kkt_values.append(
            float(
                compute_topk_kkt_relu_metric(
                    X=x,
                    y=y,
                    beta=beta[:, query_index : query_index + 1],
                    residual=residual[:, query_index : query_index + 1],
                    lambdas=query["lambdas"][
                        :, query_index : query_index + 1
                    ],
                    relu_knots=query["knots"],
                    relu_b=query["relu_b"][
                        :, query_index : query_index + 1
                    ],
                    relu_weights=query["relu_weights"][
                        :, query_index : query_index + 1
                    ],
                    topk=32,
                )
            )
        )
    return {
        "objective": objective.flatten().detach().cpu().tolist(),
        "kkt_top32": kkt_values,
        "nnz": (
            (beta.abs() > 1e-6).sum(-1).flatten().detach().cpu().tolist()
        ),
    }


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile input is empty")
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def run_quality(
    *,
    config_path: Path,
    checkpoint_path: Path,
    manifest_path: Path,
    task_root: Path,
    query_count: int,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config = _load_yaml(str(config_path))
    old_module = importlib.import_module(
        "kalbi_v6_exactprox10_eta4x_v23_independentraw_"
        "k32s32_p100k_effbatch32.model"
    )
    pg10_model = old_module.build_model(config, device)
    v7_model = build_v7_model(config, device)
    pg10_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    v7_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    pg10_model.eval()
    v7_model.eval()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    all_pg_objective: list[float] = []
    all_b3_objective: list[float] = []
    all_pg_kkt: list[float] = []
    all_b3_kkt: list[float] = []
    normalized_gaps: list[float] = []
    with torch.no_grad():
        for task_index, entry in enumerate(manifest["tasks"]):
            x, y = _load_task(task_root, entry, device)
            query = _make_queries(
                device=device,
                query_count=query_count,
                seed=2026072700 + task_index,
            )
            lambda_max = (
                torch.einsum("bnp,bn->bp", x.float(), y.float())
                .abs()
                .amax(-1)
                / float(x.shape[1])
            ).unsqueeze(1)

            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                pg_aux = pg10_model(
                    x,
                    y,
                    lambdas=query["lambdas"],
                    lambda_max=lambda_max,
                    penalty_shape_features=query["shape"],
                    penalty_d0=query["d0"],
                    return_aux=True,
                )
                encoded = v7_model.encode_dataset(
                    v7_model.precompute(x, y)
                )
                v7_aux = v7_model.decode_queries(
                    encoded,
                    lambdas=query["lambdas"],
                    lambda_max=lambda_max,
                    penalty_shape_features=query["shape"],
                    penalty_d0=query["d0"],
                    return_aux=True,
                )
            beta0_pg = pg_aux["kalbi_v6_beta0_q"].float()
            beta0_v7 = v7_aux["kalbi_v7_beta0_q"].float()
            beta0_max_abs_difference = float(
                (beta0_pg - beta0_v7).abs().max()
            )
            if beta0_max_abs_difference > 1e-6:
                raise RuntimeError(
                    "PG10 and V7 do not start from the same neural beta: "
                    f"{beta0_max_abs_difference}"
                )

            initial_residual = torch.matmul(
                beta0_v7, encoded["Xc"].float().transpose(1, 2)
            ) - encoded["yc"].float()[:, None, :]
            one_sweep = block_prox_v7._run_sweep(
                v7_model,
                encoded,
                beta0_v7,
                initial_residual,
                query["lambdas"],
                query["d0"],
                query["shape"],
                reverse=False,
            )
            methods = {
                "pg10": pg_aux["beta_q"].float(),
                "block1": one_sweep["beta"].float(),
                "block3": v7_aux["beta_q"].float(),
            }
            method_metrics = {
                name: _metrics(x=x, y=y, beta=beta, query=query)
                for name, beta in methods.items()
            }
            pg_objective = method_metrics["pg10"]["objective"]
            b3_objective = method_metrics["block3"]["objective"]
            for pg_value, b3_value in zip(pg_objective, b3_objective):
                normalized_gaps.append(
                    (b3_value - pg_value) / max(1e-12, abs(pg_value))
                )
            all_pg_objective.extend(pg_objective)
            all_b3_objective.extend(b3_objective)
            all_pg_kkt.extend(method_metrics["pg10"]["kkt_top32"])
            all_b3_kkt.extend(method_metrics["block3"]["kkt_top32"])
            results.append(
                {
                    **entry,
                    "beta0_max_abs_difference": (
                        beta0_max_abs_difference
                    ),
                    "lambda_raw": query["lambdas"].flatten().cpu().tolist(),
                    "methods": method_metrics,
                    "block_sweep_objective_delta": (
                        v7_aux[
                            "kalbi_v7_refinement_objective_delta_path_q"
                        ]
                        .flatten()
                        .cpu()
                        .tolist()
                    ),
                }
            )
            del x, y, encoded, pg_aux, v7_aux
            torch.cuda.empty_cache()

    aggregate = {
        "pg10_median_objective": statistics.median(all_pg_objective),
        "block3_median_objective": statistics.median(all_b3_objective),
        "pg10_median_kkt_top32": statistics.median(all_pg_kkt),
        "block3_median_kkt_top32": statistics.median(all_b3_kkt),
        "p90_normalized_objective_gap_vs_pg10": _percentile(
            normalized_gaps, 0.9
        ),
    }
    gates = {
        "block3_median_objective_better_than_pg10": (
            aggregate["block3_median_objective"]
            < aggregate["pg10_median_objective"]
        ),
        "block3_median_kkt_better_than_pg10": (
            aggregate["block3_median_kkt_top32"]
            < aggregate["pg10_median_kkt_top32"]
        ),
        "p90_normalized_objective_gap_within_1pct": (
            aggregate["p90_normalized_objective_gap_vs_pg10"] <= 0.01
        ),
    }
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["completed_steps"]),
        "query_count": query_count,
        "manifest": manifest,
        "tasks": results,
        "aggregate": aggregate,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _training_microstep(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    query: dict[str, torch.Tensor],
) -> tuple[float, int, float]:
    model.train()
    model.zero_grad(set_to_none=True)
    lambda_max = (
        torch.einsum("bnp,bn->bp", x.float(), y.float())
        .abs()
        .amax(-1)
        / float(x.shape[1])
    ).unsqueeze(1)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        beta = model(
            x,
            y,
            lambdas=query["lambdas"],
            lambda_max=lambda_max,
            penalty_shape_features=query["shape"],
            penalty_d0=query["d0"],
        )
    prediction = torch.einsum("bnp,bqp->bqn", x.float(), beta.float())
    residual = y.float().unsqueeze(1) - prediction
    data_half = 0.5 * residual.square().mean(-1)
    penalty = compute_penalty_relu_cached(
        beta.float(),
        query["lambdas"].float(),
        query["knots"],
        query["relu_b"].reshape(-1),
        query["relu_weights"].reshape(-1, query["relu_weights"].shape[-1]),
    )
    objective = data_half + penalty
    f_zero = 0.5 * y.float().square().mean(-1)
    loss = torch.log1p(
        objective / f_zero.unsqueeze(1).clamp_min(1e-12)
    ).mean()
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated())
    return elapsed, peak, float(loss.detach())


def run_benchmark(
    *,
    config_path: Path,
    checkpoint_path: Path,
    manifest_path: Path,
    task_root: Path,
    device: torch.device,
    production_seconds_per_step: float,
) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config = _load_yaml(str(config_path))
    old_module = importlib.import_module(
        "kalbi_v6_exactprox10_eta4x_v23_independentraw_"
        "k32s32_p100k_effbatch32.model"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_results: list[dict[str, Any]] = []
    for task_index, entry in enumerate(manifest["tasks"]):
        x, y = _load_task(task_root, entry, device)
        query = _make_queries(
            device=device,
            query_count=32,
            seed=2026072800 + task_index,
        )
        method_results: dict[str, Any] = {}
        for method_name, builder in (
            ("pg10", old_module.build_model),
            ("block3", build_v7_model),
        ):
            model = builder(config, device)
            model.load_state_dict(
                checkpoint["model_state_dict"], strict=True
            )
            warmup_elapsed, _, _ = _training_microstep(
                model=model, x=x, y=y, query=query
            )
            elapsed, peak, loss = _training_microstep(
                model=model, x=x, y=y, query=query
            )
            method_results[method_name] = {
                "discarded_warmup_seconds": warmup_elapsed,
                "seconds_per_microstep": elapsed,
                "peak_allocated_bytes": peak,
                "peak_allocated_gib": peak / (1024.0**3),
                "loss": loss,
            }
            del model
            gc.collect()
            torch.cuda.empty_cache()
        task_results.append({**entry, "methods": method_results})
        del x, y, query
        gc.collect()
        torch.cuda.empty_cache()

    pg_seconds = [
        task["methods"]["pg10"]["seconds_per_microstep"]
        for task in task_results
    ]
    block_seconds = [
        task["methods"]["block3"]["seconds_per_microstep"]
        for task in task_results
    ]
    ratios = [
        block / pg for block, pg in zip(block_seconds, pg_seconds)
    ]
    projected_seconds = production_seconds_per_step * statistics.mean(ratios)
    peak_gib = max(
        task["methods"]["block3"]["peak_allocated_gib"]
        for task in task_results
    )
    gates = {
        "projected_beta_step_at_most_8p6_seconds": (
            projected_seconds <= 8.6
        ),
        "peak_allocated_memory_below_18_gib": peak_gib < 18.0,
    }
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["completed_steps"]),
        "production_baseline_seconds_per_step": production_seconds_per_step,
        "tasks": task_results,
        "mean_block3_over_pg10_time_ratio": statistics.mean(ratios),
        "projected_v7_seconds_per_beta_step": projected_seconds,
        "block3_peak_allocated_gib": peak_gib,
        "gates": gates,
        "passed": all(gates.values()),
        "note": (
            "The isolated matched-replay ratio is applied to the observed "
            "5.7-second eight-rank production baseline. Raw timings include "
            "contention from the still-running production trainer."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("quality", "benchmark"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--production-seconds-per-step", type=float, default=5.7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("preflight requires CUDA")
    device = torch.device("cuda:0")
    if args.mode == "quality":
        result = run_quality(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            manifest_path=args.manifest,
            task_root=args.task_root,
            query_count=args.queries,
            device=device,
        )
    else:
        result = run_benchmark(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            manifest_path=args.manifest,
            task_root=args.task_root,
            device=device,
            production_seconds_per_step=args.production_seconds_per_step,
        )
    serialized = json.dumps(result, indent=2, sort_keys=True)
    print(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
