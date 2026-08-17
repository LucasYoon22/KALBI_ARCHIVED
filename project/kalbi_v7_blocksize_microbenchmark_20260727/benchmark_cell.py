#!/usr/bin/env python3
"""Run one isolated task/method/repetition benchmark cell."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import math
import os
import platform
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import torch

from .common import (
    CHECKPOINT,
    CHECKPOINT_SHA256,
    METHOD_LABELS,
    METHODS,
    OUTPUT,
    atomic_json_dump,
    load_model_and_checkpoint,
    load_precomputed,
)
from .configurable_block_prox import (
    adaptive_block_size,
    clear_curvature_cache,
    feature_blocks,
    run_block_refinement,
)
from .metrics import (
    kkt_metrics,
    objective_components,
    tensor_sha256,
    tensors_to_lists,
)
from .model_adapter import compute_beta_initial
from .pg10_refinement import pg10_call_counts, run_pg10_refinement


Tensor = torch.Tensor
TIMEOUT_SECONDS = 120.0


def encoded_dataset(
    model: torch.nn.Module, x: Tensor, y: Tensor
) -> dict[str, Any]:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return model.encode_dataset(model.precompute(x, y))


def method_block_size(method: str, feature_count: int) -> int | None:
    if method == "pg10":
        return None
    if method == "block4096":
        return 4096
    if method == "block256":
        return 256
    if method == "adaptive":
        return adaptive_block_size(feature_count)
    raise ValueError(f"unknown method {method}")


def limit_queries(payload: dict[str, Any], query_limit: int | None) -> None:
    if query_limit is None:
        return
    available = int(payload["lambdas"].shape[1])
    if query_limit <= 0 or query_limit > available:
        raise ValueError(
            f"query_limit must be in [1,{available}], got {query_limit}"
        )
    for key in (
        "lambdas",
        "penalty_shape_features",
        "penalty_d0",
        "rho",
        "drho",
        "beta_initial",
        "prethreshold",
        "log_threshold_multiplier",
    ):
        payload[key] = payload[key][:, :query_limit].contiguous()


def run_method(
    *,
    method: str,
    model: torch.nn.Module,
    encoded: dict[str, Any],
    beta_initial: Tensor,
    lambdas: Tensor,
    penalty_d0: Tensor,
    penalty_shape_features: Tensor,
    activation_checkpointing: bool,
) -> tuple[Tensor, list[dict[str, Tensor]]]:
    clear_curvature_cache(encoded)
    if method == "pg10":
        return run_pg10_refinement(
            model,
            encoded,
            beta_initial,
            lambdas,
            penalty_d0,
            penalty_shape_features,
        )
    block_size = method_block_size(method, int(beta_initial.shape[-1]))
    if block_size is None:
        raise AssertionError("block method lacks block size")
    return run_block_refinement(
        model,
        encoded,
        beta_initial,
        lambdas,
        penalty_d0,
        penalty_shape_features,
        block_size=block_size,
        activation_checkpointing=activation_checkpointing,
    )


def terminal_threshold_beta(
    payload: dict[str, Any],
    lambdas: Tensor,
    penalty_d0: Tensor,
) -> Tensor:
    prethreshold = payload["prethreshold"].detach()
    gram = payload["gram_diagonal"].detach().clamp_min(1e-4)
    log_multiplier = payload["log_threshold_multiplier"].detach()
    threshold = (
        lambdas.float()[..., None]
        * penalty_d0.float()[..., None]
        / gram[:, None, :]
        * torch.exp(log_multiplier.float())
    )
    beta = torch.sign(prethreshold) * torch.relu(
        prethreshold.abs() - threshold
    )
    beta = beta.clamp(-6.0, 6.0).to(torch.bfloat16).float()
    beta.retain_grad()
    return beta


def loss_from_beta(
    *, payload: dict[str, Any], beta: Tensor, lambdas: Tensor, shape: Tensor, grid: Tensor
) -> Tensor:
    components = objective_components(
        x=payload["x"],
        y=payload["y"],
        beta=beta,
        lambdas=lambdas,
        penalty_shape_features=shape,
        grid=grid,
    )
    return torch.log1p(
        components["objective"]
        / components["f_zero"].unsqueeze(1).clamp_min(1e-12)
    ).mean()


def synchronize_measure(
    function: Callable[[], Any],
) -> tuple[Any, float, int, int]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    allocated = int(torch.cuda.max_memory_allocated())
    reserved = int(torch.cuda.max_memory_reserved())
    return result, elapsed, allocated, reserved


def quality_metrics(
    *,
    method: str,
    model: torch.nn.Module,
    encoded: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    beta_initial = payload["beta_initial"]
    lambdas = payload["lambdas"]
    shape = payload["penalty_shape_features"]
    d0 = payload["penalty_d0"]
    grid = model._kalbi_v7_penalty_grid
    with torch.no_grad():
        beta, history = run_method(
            method=method,
            model=model,
            encoded=encoded,
            beta_initial=beta_initial,
            lambdas=lambdas,
            penalty_d0=d0,
            penalty_shape_features=shape,
            activation_checkpointing=False,
        )
        states = [beta_initial] + [item["beta"] for item in history]
        paths = [
            objective_components(
                x=payload["x"],
                y=payload["y"],
                beta=state,
                lambdas=lambdas,
                penalty_shape_features=shape,
                grid=grid,
            )
            for state in states
        ]
        final = paths[-1]
        kkt = kkt_metrics(
            x=payload["x"],
            beta=beta,
            residual=final["residual"],
            lambdas=lambdas,
            penalty_shape_features=shape,
            grid=grid,
        )

    objective_path = torch.stack(
        [item["objective"] for item in paths], dim=-1
    )
    initial = objective_path[..., 0]
    final_objective = objective_path[..., -1]
    tolerance = 1e-6 * torch.maximum(
        torch.ones_like(objective_path[..., :-1]),
        objective_path[..., :-1].abs(),
    )
    external_violations = (
        objective_path[..., 1:]
        > objective_path[..., :-1] + tolerance
    )
    if bool(external_violations.any().item()):
        raise RuntimeError(
            f"{method} objective trajectory violates monotonicity"
        )
    relative_reduction = (initial - final_objective) / initial.abs().clamp_min(
        1e-12
    )
    beta_delta_rms = (
        (beta.float() - beta_initial.float()).square().mean(-1).sqrt()
    )
    query_count = int(beta.shape[1])
    backtracks = torch.zeros_like(initial)
    rejected = torch.zeros_like(initial)
    monotonic_internal = torch.zeros_like(initial)
    block_updates = torch.zeros_like(initial)
    backtracking_evaluations: int
    exact_prox_calls: int
    if method == "pg10":
        for item in history:
            backtracks = backtracks + item["backtracks"].float()
        exact_prox_calls, backtracking_evaluations = pg10_call_counts(history)
    else:
        for item in history:
            backtracks = backtracks + item["backtracks"].float()
            rejected = rejected + item["rejected_fraction"].float()
            monotonic_internal = (
                monotonic_internal
                + item["monotonicity_violation_count"].float()
            )
            block_updates = (
                block_updates + item["block_update_count"].float()
            )
        rejected = rejected / float(len(history))
        exact_prox_calls = int(
            sum(
                float(item["exact_prox_call_count_forward"].max().item())
                for item in history
            )
        )
        backtracking_evaluations = int(
            sum(
                float(item["backtracking_evaluation_count"].max().item())
                for item in history
            )
        )
    return {
        "objective_path": objective_path.squeeze(0).cpu().tolist(),
        "data_half_initial": paths[0]["data_half"].squeeze(0).cpu().tolist(),
        "data_half_final": final["data_half"].squeeze(0).cpu().tolist(),
        "penalty_initial": paths[0]["penalty"].squeeze(0).cpu().tolist(),
        "penalty_final": final["penalty"].squeeze(0).cpu().tolist(),
        "final_objective": final_objective.squeeze(0).cpu().tolist(),
        "absolute_objective_reduction": (
            initial - final_objective
        ).squeeze(0).cpu().tolist(),
        "relative_objective_reduction": relative_reduction.squeeze(
            0
        ).cpu().tolist(),
        "kkt": tensors_to_lists(kkt),
        "nnz_1e_6": (beta.abs() > 1e-6)
        .sum(-1)
        .squeeze(0)
        .cpu()
        .tolist(),
        "nnz_1e_4": (beta.abs() > 1e-4)
        .sum(-1)
        .squeeze(0)
        .cpu()
        .tolist(),
        "coefficient_rms_change": beta_delta_rms.squeeze(0).cpu().tolist(),
        "backtrack_count": backtracks.squeeze(0).cpu().tolist(),
        "rejected_block_fraction": rejected.squeeze(0).cpu().tolist(),
        "monotonicity_violation_count": (
            monotonic_internal
            + external_violations.float().sum(-1)
        )
        .squeeze(0)
        .cpu()
        .tolist(),
        "block_update_count": block_updates.squeeze(0).cpu().tolist(),
        "exact_prox_call_count": exact_prox_calls,
        "backtracking_evaluation_count": backtracking_evaluations,
        "final_beta_sha256": tensor_sha256(beta),
        "query_count": query_count,
    }


def refinement_forward(
    *,
    method: str,
    model: torch.nn.Module,
    encoded: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    with torch.no_grad():
        run_method(
            method=method,
            model=model,
            encoded=encoded,
            beta_initial=payload["beta_initial"],
            lambdas=payload["lambdas"],
            penalty_d0=payload["penalty_d0"],
            penalty_shape_features=payload["penalty_shape_features"],
            activation_checkpointing=False,
        )


def refinement_backward(
    *,
    method: str,
    model: torch.nn.Module,
    encoded: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, float]:
    lambdas = payload["lambdas"].detach().clone().requires_grad_(True)
    shape = (
        payload["penalty_shape_features"]
        .detach()
        .clone()
        .requires_grad_(True)
    )
    d0 = payload["penalty_d0"].detach().clone().requires_grad_(True)
    beta_initial = terminal_threshold_beta(payload, lambdas, d0)
    reconstructed_error = float(
        (beta_initial.detach() - payload["beta_initial"]).abs().max().item()
    )
    if reconstructed_error > 1e-6:
        raise RuntimeError(
            f"terminal threshold reconstruction error {reconstructed_error}"
        )
    beta, _ = run_method(
        method=method,
        model=model,
        encoded=encoded,
        beta_initial=beta_initial,
        lambdas=lambdas,
        penalty_d0=d0,
        penalty_shape_features=shape,
        activation_checkpointing=True,
    )
    loss = loss_from_beta(
        payload=payload,
        beta=beta,
        lambdas=lambdas,
        shape=shape,
        grid=model._kalbi_v7_penalty_grid,
    )
    loss.backward()

    def norm(value: Tensor | None) -> float:
        return 0.0 if value is None else float(value.float().norm().item())

    return {
        "loss": float(loss.detach().item()),
        "beta_initial_gradient_norm": norm(beta_initial.grad),
        "raw_lambda_gradient_norm": norm(lambdas.grad),
        "penalty_shape_gradient_norm": norm(shape.grad),
        "d0_gradient_norm": norm(d0.grad),
        "terminal_threshold_reconstruction_max_abs_error": reconstructed_error,
    }


def end_to_end_backward(
    *,
    method: str,
    model: torch.nn.Module,
    payload: dict[str, Any],
) -> dict[str, float]:
    model.train()
    model.zero_grad(set_to_none=True)
    encoded = encoded_dataset(model, payload["x"], payload["y"])
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        initial = compute_beta_initial(
            model,
            encoded,
            lambdas=payload["lambdas"],
            lambda_max=payload["lambda_max"],
            penalty_shape_features=payload["penalty_shape_features"],
            penalty_d0=payload["penalty_d0"],
        )
    beta, _ = run_method(
        method=method,
        model=model,
        encoded=encoded,
        beta_initial=initial["beta_initial"],
        lambdas=initial["lambdas"],
        penalty_d0=initial["penalty_d0"],
        penalty_shape_features=initial["penalty_shape_features"],
        activation_checkpointing=True,
    )
    loss = loss_from_beta(
        payload=payload,
        beta=beta,
        lambdas=initial["lambdas"],
        shape=initial["penalty_shape_features"],
        grid=model._kalbi_v7_penalty_grid,
    )
    loss.backward()
    gradient_square_sum = 0.0
    finite = True
    for parameter in model.parameters():
        if parameter.grad is not None:
            gradient = parameter.grad.detach().float()
            finite = finite and bool(torch.isfinite(gradient).all().item())
            gradient_square_sum += float(gradient.square().sum().item())
    if not finite:
        raise RuntimeError("end-to-end model gradient is nonfinite")
    return {
        "loss": float(loss.detach().item()),
        "model_gradient_norm": math.sqrt(gradient_square_sum),
    }


def warmup(
    function: Callable[[], Any], count: int
) -> None:
    for _ in range(count):
        function()
        torch.cuda.synchronize()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(202607270256 + args.task_index * 100 + args.repetition)
    device = torch.device(args.device)
    payload = load_precomputed(args.task_index, device)
    model, checkpoint = load_model_and_checkpoint(device)
    model.eval()
    with torch.no_grad():
        encoded = encoded_dataset(model, payload["x"], payload["y"])
    observed_feature_mask = encoded.get("feature_mask")
    if observed_feature_mask is None:
        observed_feature_mask = torch.ones_like(payload["feature_mask"])
    observed_feature_mask = observed_feature_mask.bool()
    if tensor_sha256(observed_feature_mask) != tensor_sha256(
        payload["feature_mask"]
    ):
        raise RuntimeError("feature-mask contract changed")
    if int(observed_feature_mask.sum().item()) != int(payload["x"].shape[-1]):
        raise RuntimeError("single-task benchmark has invalid padded features")
    feature_count = int(payload["x"].shape[-1])
    block_size = method_block_size(args.method, feature_count)
    block_count = (
        0
        if block_size is None
        else len(feature_blocks(feature_count, block_size))
    )
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        fresh = compute_beta_initial(
            model,
            encoded,
            lambdas=payload["lambdas"],
            lambda_max=payload["lambda_max"],
            penalty_shape_features=payload["penalty_shape_features"],
            penalty_d0=payload["penalty_d0"],
        )
    beta_initial_error = float(
        (
            fresh["beta_initial"].detach().float()
            - payload["beta_initial"].float()
        )
        .abs()
        .max()
        .item()
    )
    if beta_initial_error != 0.0:
        raise RuntimeError(
            "method did not reproduce the exact frozen beta_initial: "
            f"{beta_initial_error}"
        )
    limit_queries(payload, args.query_limit)

    quality = (
        quality_metrics(
            method=args.method,
            model=model,
            encoded=encoded,
            payload=payload,
        )
        if args.repetition == 0
        else None
    )

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    warmup(
        lambda: refinement_forward(
            method=args.method,
            model=model,
            encoded=encoded,
            payload=payload,
        ),
        args.warmups,
    )
    _, forward_seconds, forward_allocated, forward_reserved = (
        synchronize_measure(
            lambda: refinement_forward(
                method=args.method,
                model=model,
                encoded=encoded,
                payload=payload,
            )
        )
    )

    warmup(
        lambda: refinement_backward(
            method=args.method,
            model=model,
            encoded=encoded,
            payload=payload,
        ),
        args.warmups,
    )
    backward_audit, backward_seconds, backward_allocated, backward_reserved = (
        synchronize_measure(
            lambda: refinement_backward(
                method=args.method,
                model=model,
                encoded=encoded,
                payload=payload,
            )
        )
    )

    for parameter in model.parameters():
        parameter.requires_grad_(True)
    warmup(
        lambda: end_to_end_backward(
            method=args.method,
            model=model,
            payload=payload,
        ),
        args.warmups,
    )
    end_to_end_audit, e2e_seconds, e2e_allocated, e2e_reserved = (
        synchronize_measure(
            lambda: end_to_end_backward(
                method=args.method,
                model=model,
                payload=payload,
            )
        )
    )
    timings = {
        "refinement_forward_seconds": forward_seconds,
        "refinement_forward_backward_seconds": backward_seconds,
        "end_to_end_forward_backward_seconds": e2e_seconds,
        "refinement_forward_peak_allocated_bytes": forward_allocated,
        "refinement_forward_peak_reserved_bytes": forward_reserved,
        "refinement_backward_peak_allocated_bytes": backward_allocated,
        "refinement_backward_peak_reserved_bytes": backward_reserved,
        "end_to_end_peak_allocated_bytes": e2e_allocated,
        "end_to_end_peak_reserved_bytes": e2e_reserved,
    }
    timed_out_modes = [
        name
        for name, seconds in (
            ("refinement_forward", forward_seconds),
            ("refinement_forward_backward", backward_seconds),
            ("end_to_end_forward_backward", e2e_seconds),
        )
        if seconds > TIMEOUT_SECONDS
    ]
    return {
        "status": "TIMEOUT" if timed_out_modes else "OK",
        "timed_out_modes": timed_out_modes,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "task_index": args.task_index,
        "task": payload["task"],
        "method": args.method,
        "method_label": METHOD_LABELS[args.method],
        "repetition": args.repetition,
        "warmups": args.warmups,
        "query_limit": args.query_limit,
        "query_count": int(payload["lambdas"].shape[1]),
        "device_argument": args.device,
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_step": int(checkpoint["completed_steps"]),
        "block_size": block_size,
        "block_count_per_sweep": block_count,
        "sweeps": 0 if args.method == "pg10" else 3,
        "iterations": 10 if args.method == "pg10" else 0,
        "beta_initial_max_abs_difference": beta_initial_error,
        "input_tensor_hashes": payload["tensor_hashes"],
        "timings": timings,
        "refinement_backward_gradient_audit": backward_audit,
        "end_to_end_gradient_audit": end_to_end_audit,
        "quality": quality,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args)
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        result = {
            "status": "OOM",
            "task_index": args.task_index,
            "method": args.method,
            "repetition": args.repetition,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    except Exception as error:
        result = {
            "status": "FAIL",
            "task_index": args.task_index,
            "method": args.method,
            "repetition": args.repetition,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2))
    if result["status"] not in {"OK", "TIMEOUT", "OOM"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
