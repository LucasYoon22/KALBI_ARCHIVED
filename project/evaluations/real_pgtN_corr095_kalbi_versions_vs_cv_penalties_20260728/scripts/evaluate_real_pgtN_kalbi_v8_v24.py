#!/usr/bin/env python3
"""Evaluate pinned V8 beta plus V24 selector on registered real P>N tasks."""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kalbi_v6_exactprox10_eta4x_v24_additive512_traj3_k32s32_p100k_effbatch32.exact_canonical_cd import (
    objective_components,
)
from kalbi_v6_exactprox10_eta4x_v24_additive512_traj3_k32s32_p100k_effbatch32.exact_canonical_cd_parallel import (
    solve_multistart_path_parallel,
)
from kalbi_v6_exactprox10_eta4x_v24_additive512_traj3_k32s32_p100k_effbatch32.selector_v24_additive512_trajectory import (
    DirectPenaltyMLPQuerySelectorV24Additive512,
)
from kalbi_v8_adaptiveblockprox10_v23_k32s32_p100k_effbatch32.model import (
    build_model,
)
from train_fpk import _load_yaml

from real_pgtN_common import (
    apply_x_preprocessing,
    atomic_json,
    atomic_npz,
    fit_preprocessing,
    iter_selected_tasks,
    load_split_manifest,
    load_x,
    parse_task_indices,
    raw_prediction_metrics,
    save_preprocessing,
    select_rows,
    sha256,
    task_family,
    zero_predictor_metrics,
)


def vector_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= 1e-15:
        return 1.0 if np.max(np.abs(x - y), initial=0.0) <= 1e-12 else 0.0
    return float(np.dot(x, y) / denominator)


def objective_path(
    x: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    raw_lambdas: np.ndarray,
    knots: np.ndarray,
    relu_b: float,
    relu_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    components = [
        objective_components(
            x,
            y,
            coefficient,
            float(raw_lambda),
            knots,
            relu_b,
            relu_weights,
        )
        for coefficient, raw_lambda in zip(beta, raw_lambdas, strict=True)
    ]
    data = np.asarray([value[0] for value in components], dtype=np.float64)
    penalty = np.asarray([value[1] for value in components], dtype=np.float64)
    objective = np.asarray([value[2] for value in components], dtype=np.float64)
    return data, penalty, objective


def top_features(reference: np.ndarray, count: int = 12) -> np.ndarray:
    score = np.max(np.abs(np.asarray(reference)), axis=0)
    return np.lexsort((np.arange(score.size), -score))[: min(count, score.size)]


def build_components(
    beta_checkpoint_path: Path,
    selector_checkpoint_path: Path,
    config_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], torch.nn.Module, torch.nn.Module]:
    beta_checkpoint = torch.load(
        beta_checkpoint_path, map_location="cpu", weights_only=False
    )
    selector_checkpoint = torch.load(
        selector_checkpoint_path, map_location="cpu", weights_only=False
    )
    model = build_model(_load_yaml(str(config_path)), device)
    model.load_state_dict(beta_checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    selector = DirectPenaltyMLPQuerySelectorV24Additive512(
        hidden=512,
        derivative_floor=1e-6,
        squareplus_scale=0.02,
        direct_head_init_std=1e-3,
    )
    selector.load_state_dict(
        selector_checkpoint["selector_state_dict"], strict=True
    )
    selector.to(device).eval()
    for parameter in selector.parameters():
        parameter.requires_grad_(False)
    return beta_checkpoint, selector_checkpoint, model, selector


def decode_queries(
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    selected: dict[str, Any],
    raw_tensor: torch.Tensor,
) -> dict[str, Any]:
    query_count = raw_tensor.shape[1]
    return model.decode_queries(
        encoded,
        lambdas=raw_tensor,
        lambda_max=encoded["lambda_max"].float().reshape(-1, 1).expand(
            -1, query_count
        ),
        penalty_shape_features=selected["penalty_shape_features"]
        .reshape(1, 1, -1)
        .expand(1, query_count, -1),
        penalty_d0=selected["penalty_d0"]
        .reshape(1, 1)
        .expand(1, query_count),
        return_aux=True,
    )


def evaluate_task(
    *,
    task: dict[str, Any],
    model: torch.nn.Module,
    selector: torch.nn.Module,
    device: torch.device,
    raw_lambdas: np.ndarray,
    evaluation_root: Path,
    cd_library: Path,
    cd_workers: int,
    cd_max_sweeps: int,
    cd_retries: int,
    cd_tolerance: float,
    cd_zero_tolerance: float,
    cd_audit_tolerance: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    task_id = task["task_id"]
    x_path = Path(task["X_path"])
    y_path = Path(task["y_path"])
    split_path = Path(task["outer_split_path"])
    for path, expected in (
        (x_path, task["X_sha256"]),
        (y_path, task["y_sha256"]),
        (split_path, task["outer_split_sha256"]),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"artifact hash mismatch: {path}")

    x_store = load_x(x_path)
    y_store = np.load(y_path, mmap_mode="r")
    with np.load(split_path, allow_pickle=False) as archive:
        train_index = np.asarray(archive["train_indices"], dtype=np.int64)
        test_index = np.asarray(archive["test_indices"], dtype=np.int64)
    if np.intersect1d(train_index, test_index).size:
        raise RuntimeError("outer split overlap")
    if np.union1d(train_index, test_index).size != int(task["N"]):
        raise RuntimeError("outer split coverage failure")

    # Test y is not numerically read before selector/path/reference choices freeze.
    x_train_raw = select_rows(x_store, train_index)
    y_train_raw = np.asarray(y_store[train_index], dtype=np.float64)
    x_train, y_train, scaler = fit_preprocessing(x_train_raw, y_train_raw)
    preprocessing_path, preprocessing_sha = save_preprocessing(
        evaluation_root, task, scaler
    )
    active = np.asarray(scaler["active"], dtype=bool)
    active_original_indices = np.flatnonzero(active).astype(np.int64)
    transformed_p = int(active.sum())
    if transformed_p <= 0:
        raise RuntimeError("no active training features")

    x_device = torch.from_numpy(x_train)[None].to(device)
    y_device = torch.from_numpy(y_train)[None].to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    selector_started = time.perf_counter()
    with torch.inference_mode():
        encoded = model.encode_dataset(model.precompute(x_device, y_device))
        lambda_max = encoded["lambda_max"].float().reshape(-1)
        selected = selector(
            x_device,
            y_device,
            lambda_max,
            solver_context=encoded,
            return_representation=True,
        )
    selector_seconds = time.perf_counter() - selector_started
    direct_lambda_max = float(
        np.max(
            np.abs(
                np.asarray(x_train, dtype=np.float64).T
                @ np.asarray(y_train, dtype=np.float64)
            )
        )
        / x_train.shape[0]
    )
    if abs(float(lambda_max[0]) - direct_lambda_max) / max(
        direct_lambda_max, 1e-12
    ) > 2e-5:
        raise RuntimeError("encoded/direct lambda_max mismatch")

    d0 = float(selected["penalty_d0"].reshape(-1)[0])
    max_drho = float(selected["penalty_drho"].max())
    selected_raw_lambda = float(selected["raw_lambda"].reshape(-1)[0])
    if (
        not math.isfinite(selected_raw_lambda)
        or selected_raw_lambda < 1e-6
        or selected_raw_lambda > 10.0
    ):
        raise RuntimeError("selected raw lambda leaves training support")
    if not math.isfinite(d0) or d0 <= 1e-8:
        raise RuntimeError("invalid selected d0")

    raw_grid = np.asarray(raw_lambdas, dtype=np.float64)
    if np.any(raw_grid < 1e-6) or np.any(raw_grid > 10.0):
        raise RuntimeError("raw lambda grid leaves training support")
    raw_tensor = torch.as_tensor(
        raw_grid[None], dtype=torch.float32, device=device
    )
    np.testing.assert_array_equal(
        raw_tensor[0].detach().cpu().numpy(),
        raw_grid.astype(np.float32),
    )

    beta_started = time.perf_counter()
    with torch.inference_mode():
        decoded = decode_queries(model, encoded, selected, raw_tensor)
        selected_decoded = decode_queries(
            model,
            encoded,
            selected,
            torch.as_tensor(
                [[selected_raw_lambda]], dtype=torch.float32, device=device
            ),
        )
    beta_seconds = time.perf_counter() - beta_started
    kalbi_beta = (
        decoded["beta_q"][0].float().cpu().numpy().astype(np.float64)
    )
    selected_beta = (
        selected_decoded["beta_q"][0, 0]
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    if kalbi_beta.shape != (raw_grid.size, transformed_p):
        raise RuntimeError("KALBI beta shape mismatch")

    eta_path = (
        decoded["kalbi_v8_refinement_step_size_path_q"][0]
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    gamma_path = (
        decoded["kalbi_v8_refinement_gamma_path_q"][0]
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    gamma_contract_error = np.max(
        np.abs(gamma_path / np.maximum(eta_path, 1e-30) - raw_grid[:, None])
    )
    if gamma_contract_error > 2e-5:
        raise RuntimeError("V8 gamma/eta raw-lambda contract failed")
    probe_contract_error = float(
        selected["v24_probe_gamma_contract_error"].float().max().item()
    )
    if probe_contract_error > 2e-5:
        raise RuntimeError("V24 probe gamma contract failed")

    knots = (
        selector.penalty_parameter_head.penalty_grid.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    penalty_rho = (
        selected["penalty_rho"].reshape(-1).cpu().numpy().astype(np.float64)
    )
    penalty_drho = (
        selected["penalty_drho"].reshape(-1).cpu().numpy().astype(np.float64)
    )
    relu_b = float(selected["penalty_relu_b"].reshape(-1)[0])
    relu_weights = (
        selected["penalty_relu_weights"]
        .reshape(-1, knots.size)[0]
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    if abs(float(penalty_rho[0])) > 1e-8:
        raise RuntimeError("rho(0) contract failed")
    if abs(float(penalty_rho[-1]) - 1.0) > 2e-6:
        raise RuntimeError("rho(6) contract failed")
    if np.min(penalty_drho) < -1e-8:
        raise RuntimeError("negative penalty derivative")

    cd_started = time.perf_counter()
    cd = solve_multistart_path_parallel(
        library_path=cd_library,
        x=x_train,
        y=y_train,
        raw_lambdas=raw_grid,
        knots=knots,
        relu_b=relu_b,
        relu_weights=relu_weights,
        model_beta=kalbi_beta,
        max_sweeps=cd_max_sweeps,
        retries=cd_retries,
        tolerance=cd_tolerance,
        zero_tolerance=cd_zero_tolerance,
        workers=cd_workers,
        include_warm=True,
    )
    cd_seconds = time.perf_counter() - cd_started
    cd_beta = np.asarray(cd["reference_beta"], dtype=np.float64)
    if cd_beta.shape != kalbi_beta.shape:
        raise RuntimeError("KALBI/CD feature-order shape mismatch")
    cd_max_fixed = float(np.max(cd["reference_fixed_point_max"]))
    cd_max_kkt = float(np.max(cd["reference_projected_kkt_max"]))
    cd_audit_pass = (
        cd_max_fixed <= cd_audit_tolerance
        and cd_max_kkt <= cd_audit_tolerance
    )

    kalbi_data, kalbi_penalty, kalbi_objective = objective_path(
        x_train,
        y_train,
        kalbi_beta,
        raw_grid,
        knots,
        relu_b,
        relu_weights,
    )
    cd_data, cd_penalty, cd_objective = objective_path(
        x_train,
        y_train,
        cd_beta,
        raw_grid,
        knots,
        relu_b,
        relu_weights,
    )
    method_freeze_time = time.time()

    # Test y is first materialized here, after selector, path and CD are frozen.
    x_test_raw = select_rows(x_store, test_index)
    x_test = apply_x_preprocessing(x_test_raw, scaler)
    y_test_raw = np.asarray(y_store[test_index], dtype=np.float64)
    kalbi_test = raw_prediction_metrics(
        x_test,
        y_test_raw,
        kalbi_beta,
        y_mean=float(scaler["y_mean"]),
        y_scale=float(scaler["y_scale"]),
    )
    cd_test = raw_prediction_metrics(
        x_test,
        y_test_raw,
        cd_beta,
        y_mean=float(scaler["y_mean"]),
        y_scale=float(scaler["y_scale"]),
    )
    selected_test = raw_prediction_metrics(
        x_test,
        y_test_raw,
        selected_beta,
        y_mean=float(scaler["y_mean"]),
        y_scale=float(scaler["y_scale"]),
    )
    zero = zero_predictor_metrics(y_test_raw, float(scaler["y_mean"]))
    oracle_index = int(np.argmin(kalbi_test["mse"]))

    correlations = np.asarray(
        [
            vector_correlation(left, right)
            for left, right in zip(kalbi_beta, cd_beta, strict=True)
        ],
        dtype=np.float64,
    )
    top = top_features(cd_beta, count=12)
    derivative_difference = np.diff(penalty_drho)
    saturation_candidates = np.flatnonzero(
        penalty_drho <= 0.01 * max(d0, 1e-12)
    )
    saturation_location = (
        float(knots[saturation_candidates[0]])
        if saturation_candidates.size
        else None
    )
    peak_gpu_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )

    path_path = evaluation_root / "paths" / f"{task_id}.npz"
    atomic_npz(
        path_path,
        raw_lambdas=raw_grid,
        zero_kkt_ratio=raw_grid * d0 / direct_lambda_max,
        force_ratio=raw_grid * max_drho / direct_lambda_max,
        active_original_indices=active_original_indices,
        kalbi_beta=kalbi_beta.astype(np.float32),
        coordinate_descent_beta=cd_beta.astype(np.float32),
        kalbi_data_term=kalbi_data,
        kalbi_penalty_term=kalbi_penalty,
        kalbi_objective=kalbi_objective,
        coordinate_descent_data_term=cd_data,
        coordinate_descent_penalty_term=cd_penalty,
        coordinate_descent_objective=cd_objective,
        kalbi_test_prediction=kalbi_test["prediction"].astype(np.float32),
        coordinate_descent_test_prediction=cd_test["prediction"].astype(np.float32),
        kalbi_test_mse=kalbi_test["mse"],
        kalbi_test_rmse=kalbi_test["rmse"],
        kalbi_test_mae=kalbi_test["mae"],
        kalbi_test_r2=kalbi_test["r2"],
        coordinate_descent_test_mse=cd_test["mse"],
        coordinate_descent_test_r2=cd_test["r2"],
        selected_raw_lambda=np.asarray(selected_raw_lambda),
        selected_beta=selected_beta.astype(np.float32),
        selected_test_prediction=selected_test["prediction"][0].astype(np.float32),
        selected_test_mse=selected_test["mse"],
        selected_test_r2=selected_test["r2"],
        oracle_index=np.asarray(oracle_index),
        penalty_grid=knots,
        penalty_rho=penalty_rho,
        penalty_drho=penalty_drho,
        top_features=top,
        top_original_features=active_original_indices[top],
        v8_eta_path=eta_path,
        v8_gamma_path=gamma_path,
        cd_projected_kkt=np.asarray(cd["reference_projected_kkt_max"]),
        cd_fixed_point=np.asarray(cd["reference_fixed_point_max"]),
        cd_sweeps=np.asarray(cd["reference_sweeps"]),
        cd_selected_start=np.asarray(cd["reference_selected_start"]),
        train_indices=train_index,
        test_indices=test_index,
    )

    metric = {
        "status": "PASS",
        "registry_index": int(task["registry_index"]),
        "task_id": task_id,
        "source": task["source"],
        "source_family": task_family(task["source"], task_id),
        "target": task["target"],
        "evidence_tier": task["evidence_tier"],
        "N": int(task["N"]),
        "P_raw": int(task["P_raw"]),
        "P_transformed": transformed_p,
        "P_transformed_over_train_N": transformed_p / train_index.size,
        "outer_train_rows": int(train_index.size),
        "outer_test_rows": int(test_index.size),
        "outer_split_seed": int(task["outer_split_seed"]),
        "outer_split_sha256": task["outer_split_sha256"],
        "preprocessing_path": str(preprocessing_path),
        "preprocessing_sha256": preprocessing_sha,
        "scaler_sha256": scaler["scaler_sha256"],
        "train_only_preprocessing": True,
        "test_y_accessed_after_method_freeze": True,
        "method_freeze_unix": method_freeze_time,
        "lambda_max": direct_lambda_max,
        "selected_raw_lambda": selected_raw_lambda,
        "selected_raw_lambda_over_lambda_max": (
            selected_raw_lambda / direct_lambda_max
        ),
        "selected_zero_kkt_ratio": (
            selected_raw_lambda * d0 / direct_lambda_max
        ),
        "selected_force_ratio": (
            selected_raw_lambda * max_drho / direct_lambda_max
        ),
        "selected_test_mse": float(selected_test["mse"][0]),
        "selected_test_rmse": float(selected_test["rmse"][0]),
        "selected_test_mae": float(selected_test["mae"][0]),
        "selected_test_r2": float(selected_test["r2"][0]),
        "selected_normalized_test_mse": float(
            selected_test["mse"][0] / max(zero["mse"], 1e-30)
        ),
        "selected_nnz_1e8": int(
            np.count_nonzero(np.abs(selected_beta) > 1e-8)
        ),
        "oracle_raw_lambda": float(raw_grid[oracle_index]),
        "oracle_test_mse": float(kalbi_test["mse"][oracle_index]),
        "oracle_test_r2": float(kalbi_test["r2"][oracle_index]),
        "selected_over_oracle_mse": float(
            selected_test["mse"][0]
            / max(kalbi_test["mse"][oracle_index], 1e-30)
        ),
        "zero_predictor": zero,
        "path_correlation_mean": float(correlations.mean()),
        "path_correlation_min": float(correlations.min()),
        "coefficient_rmse": float(
            np.sqrt(np.mean((kalbi_beta - cd_beta) ** 2))
        ),
        "objective_gap_mean": float(
            np.mean(kalbi_objective - cd_objective)
        ),
        "objective_gap_max": float(np.max(kalbi_objective - cd_objective)),
        "gamma_over_eta_max_absolute_error": float(gamma_contract_error),
        "selector_probe_gamma_max_absolute_error": probe_contract_error,
        "penalty": {
            "rho0_error": abs(float(penalty_rho[0])),
            "rho6": float(penalty_rho[-1]),
            "d0": d0,
            "min_drho": float(penalty_drho.min()),
            "max_drho": max_drho,
            "derivative_total_variation": float(
                np.sum(np.abs(derivative_difference))
            ),
            "concavity_violation_fraction": float(
                np.mean(derivative_difference > 1e-8)
            ),
            "saturation_location": saturation_location,
        },
        "coordinate_descent": {
            "status": "PASS" if cd_audit_pass else "AUDIT_FAIL",
            "audit_pass": cd_audit_pass,
            "full_feature_count": transformed_p,
            "max_projected_kkt": cd_max_kkt,
            "max_fixed_point": cd_max_fixed,
            "audit_tolerance": cd_audit_tolerance,
            "solver_tolerance": cd_tolerance,
            "median_sweeps": float(np.median(cd["reference_sweeps"])),
            "runtime_seconds": cd_seconds,
        },
        "runtime": {
            "selector_seconds": selector_seconds,
            "beta_path_and_selected_seconds": beta_seconds,
            "coordinate_descent_seconds": cd_seconds,
            "total_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": peak_gpu_memory,
        },
        "path_path": str(path_path),
        "path_sha256": sha256(path_path),
    }
    metric_path = evaluation_root / "metrics" / f"{task_id}_kalbi.json"
    atomic_json(metric_path, metric)
    return metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--beta-checkpoint", type=Path, required=True)
    parser.add_argument("--beta-checkpoint-sha256", required=True)
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--selector-checkpoint-sha256", required=True)
    parser.add_argument("--compatibility-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--task-indices", default="all")
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--cd-workers", type=int, default=4)
    parser.add_argument("--cd-max-sweeps", type=int, default=3000)
    parser.add_argument("--cd-retries", type=int, default=4)
    parser.add_argument("--cd-tolerance", type=float, default=1e-6)
    parser.add_argument("--cd-zero-tolerance", type=float, default=1e-10)
    parser.add_argument("--cd-audit-tolerance", type=float, default=5e-3)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()

    torch.set_num_threads(args.torch_threads)
    if sha256(args.beta_checkpoint) != args.beta_checkpoint_sha256:
        raise RuntimeError("beta checkpoint SHA mismatch")
    if sha256(args.selector_checkpoint) != args.selector_checkpoint_sha256:
        raise RuntimeError("selector checkpoint SHA mismatch")
    compatibility = json.loads(
        args.compatibility_manifest.read_text(encoding="utf-8")
    )
    if compatibility.get("status") != "PASS":
        raise RuntimeError("checkpoint compatibility did not pass")
    if (
        compatibility["beta_checkpoint_sha256"]
        != args.beta_checkpoint_sha256
        or compatibility["selector_checkpoint_sha256"]
        != args.selector_checkpoint_sha256
    ):
        raise RuntimeError("compatibility/checkpoint hash mismatch")

    split_manifest = load_split_manifest(args.split_manifest)
    indices = parse_task_indices(
        args.task_indices, int(split_manifest["task_count"])
    )
    device = torch.device(args.device)
    beta_checkpoint, selector_checkpoint, model, selector = build_components(
        args.beta_checkpoint,
        args.selector_checkpoint,
        args.config,
        device,
    )
    raw_lambdas = np.logspace(
        -4.0, 0.0, args.grid_size, dtype=np.float64
    )
    cd_library = Path(
        "/users/minsuk/fm-lm-lambda/"
        "kalbi_v6_exactprox10_eta4x_v24_additive512_traj3_k32s32_p100k_"
        "effbatch32/canonical_cd_kernel.so"
    )
    if not cd_library.is_file():
        raise RuntimeError("canonical CD shared library missing")

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for task in iter_selected_tasks(split_manifest, indices):
        task_id = task["task_id"]
        metric_path = args.evaluation_root / "metrics" / f"{task_id}_kalbi.json"
        if metric_path.exists():
            existing = json.loads(metric_path.read_text(encoding="utf-8"))
            path_path = Path(existing["path_path"])
            if (
                existing.get("status") == "PASS"
                and path_path.is_file()
                and sha256(path_path) == existing["path_sha256"]
            ):
                print(
                    json.dumps(
                        {"task_id": task_id, "status": "SKIP_VERIFIED"}
                    ),
                    flush=True,
                )
                successes.append(existing)
                continue
            raise RuntimeError(f"invalid existing task output: {task_id}")
        print(
            json.dumps(
                {
                    "task_id": task_id,
                    "registry_index": task["registry_index"],
                    "status": "START",
                }
            ),
            flush=True,
        )
        try:
            metric = evaluate_task(
                task=task,
                model=model,
                selector=selector,
                device=device,
                raw_lambdas=raw_lambdas,
                evaluation_root=args.evaluation_root,
                cd_library=cd_library,
                cd_workers=args.cd_workers,
                cd_max_sweeps=args.cd_max_sweeps,
                cd_retries=args.cd_retries,
                cd_tolerance=args.cd_tolerance,
                cd_zero_tolerance=args.cd_zero_tolerance,
                cd_audit_tolerance=args.cd_audit_tolerance,
            )
            successes.append(metric)
            print(
                json.dumps(
                    {
                        "task_id": task_id,
                        "status": "PASS",
                        "selected_r2": metric["selected_test_r2"],
                        "selected_normalized_mse": metric[
                            "selected_normalized_test_mse"
                        ],
                        "seconds": metric["runtime"]["total_seconds"],
                    }
                ),
                flush=True,
            )
        except Exception as error:
            failure = {
                "status": "FAIL",
                "task_id": task_id,
                "registry_index": int(task["registry_index"]),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            atomic_json(
                args.evaluation_root
                / "metrics"
                / f"{task_id}_kalbi_failure.json",
                failure,
            )
            print(json.dumps(failure), flush=True)

    worker = {
        "status": "PASS" if not failures else "PARTIAL",
        "task_indices": indices,
        "success_count": len(successes),
        "failure_count": len(failures),
        "beta_checkpoint": str(args.beta_checkpoint),
        "beta_checkpoint_sha256": args.beta_checkpoint_sha256,
        "beta_completed_steps": int(beta_checkpoint["completed_steps"]),
        "selector_checkpoint": str(args.selector_checkpoint),
        "selector_checkpoint_sha256": args.selector_checkpoint_sha256,
        "selector_completed_steps": int(selector_checkpoint["completed_steps"]),
        "raw_lambda_grid": raw_lambdas.tolist(),
        "failures": failures,
    }
    tag = "_".join(map(str, indices))
    atomic_json(
        args.evaluation_root / "reports" / f"kalbi_worker_{tag}.json",
        worker,
    )
    print(json.dumps(worker, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
