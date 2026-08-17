#!/usr/bin/env python3
"""Evaluate independently pinned KALBI beta and selector checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kalbi_v6_exactprox10_eta4x_v24_additive512_traj3_k32s32_p100k_effbatch32.exact_canonical_cd import (
    canonical_penalty_values,
    objective_components,
)
from kalbi_v6_exactprox10_eta4x_v24_additive512_traj3_k32s32_p100k_effbatch32.exact_canonical_cd_parallel import (
    solve_multistart_path_parallel,
)
from kalbi_v6_exactprox10_eta4x_v24_additive512_traj3_k32s32_p100k_effbatch32.selector_v24_additive512_trajectory import (
    DirectPenaltyMLPQuerySelectorV24Additive512,
)
from kalbi_v6_exactprox10_eta4x_v24_additive512_traj3_k32s32_p100k_effbatch32.train import (
    state_dict_sha256,
)
from kalbi_v8_adaptiveblockprox10_v23_k32s32_p100k_effbatch32.model import (
    build_model,
)
from train_fpk import _load_yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def vector_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= 1e-15:
        return 1.0 if np.max(np.abs(x - y), initial=0.0) <= 1e-12 else 0.0
    return float(np.dot(x, y) / denominator)


def load_task(
    entry: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    data_path = Path(entry["accepted_data_path"])
    split_path = Path(entry["split_path"])
    if sha256(data_path) != entry["accepted_data_sha256"]:
        raise RuntimeError("task data SHA256 mismatch")
    if sha256(split_path) != entry["split_sha256"]:
        raise RuntimeError("task split SHA256 mismatch")
    with np.load(data_path, allow_pickle=False) as archive:
        x = np.asarray(archive["X"][0], dtype=np.float32)
        y = np.asarray(archive["y"][0], dtype=np.float32)
    with np.load(split_path, allow_pickle=False) as archive:
        split = {key: np.asarray(archive[key]) for key in archive.files}
    groups = [
        set(split[name].astype(np.int64).tolist())
        for name in ("context_indices", "validation_indices", "test_indices")
    ]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise RuntimeError("row-split leakage")
    if len(groups[0] | groups[1] | groups[2]) != x.shape[0]:
        raise RuntimeError("row splits do not cover task")
    return x, y, split


def transform_with_frozen_context(
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    split: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    x_scaled = (
        np.asarray(x[indices], dtype=np.float64)
        - np.asarray(split["x_mean"], dtype=np.float64)
    ) / np.asarray(split["x_scale"], dtype=np.float64)
    y_scaled = (
        np.asarray(y[indices], dtype=np.float64) - float(split["y_mean"])
    ) / float(split["y_scale"])
    active = np.asarray(split["active"], dtype=bool)
    x_scaled[:, ~active] = 0.0
    if not np.isfinite(x_scaled).all() or not np.isfinite(y_scaled).all():
        raise RuntimeError("nonfinite frozen transform")
    return x_scaled.astype(np.float32), y_scaled.astype(np.float32)


def direct_objective(
    x: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    raw_lambda: float,
    knots: np.ndarray,
    relu_b: float,
    relu_weights: np.ndarray,
) -> float:
    residual = np.asarray(y, dtype=np.float64) - np.asarray(
        x, dtype=np.float64
    ) @ np.asarray(beta, dtype=np.float64)
    rho = float(relu_b) * np.abs(beta)
    for knot, weight in zip(knots, relu_weights, strict=True):
        rho = rho + 0.5 * float(weight) * np.maximum(
            np.abs(beta) - float(knot), 0.0
        ) ** 2
    return float(0.5 * np.mean(residual**2) + raw_lambda * np.sum(rho))


def objective_path(
    x: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    raw_lambdas: np.ndarray,
    knots: np.ndarray,
    relu_b: float,
    relu_weights: np.ndarray,
) -> np.ndarray:
    primary = np.asarray(
        [
            objective_components(
                x,
                y,
                coefficient,
                float(raw_lambda),
                knots,
                relu_b,
                relu_weights,
            )[2]
            for coefficient, raw_lambda in zip(
                beta, raw_lambdas, strict=True
            )
        ],
        dtype=np.float64,
    )
    independent = np.asarray(
        [
            direct_objective(
                x,
                y,
                coefficient,
                float(raw_lambda),
                knots,
                relu_b,
                relu_weights,
            )
            for coefficient, raw_lambda in zip(
                beta, raw_lambdas, strict=True
            )
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(primary, independent, rtol=1e-10, atol=1e-12)
    return primary


def mse_path(
    x_standardized: np.ndarray,
    y_raw: np.ndarray,
    beta_path: np.ndarray,
    *,
    y_mean: float,
    y_scale: float,
) -> np.ndarray:
    target = np.asarray(y_raw, dtype=np.float64)
    values = []
    for coefficient in np.asarray(beta_path, dtype=np.float64):
        prediction = y_mean + y_scale * (
            np.asarray(x_standardized, dtype=np.float64) @ coefficient
        )
        values.append(float(np.mean((target - prediction) ** 2)))
    return np.asarray(values, dtype=np.float64)


def top_features(reference: np.ndarray, count: int = 12) -> tuple[np.ndarray, np.ndarray]:
    score = np.max(np.abs(reference), axis=0)
    order = np.lexsort((np.arange(score.size), -score))[:count]
    return order.astype(np.int64), score[order]


def state_snapshot(module: torch.nn.Module) -> dict[str, str]:
    return {
        "state_sha256": state_dict_sha256(module.state_dict()),
        "parameter_sha256": state_dict_sha256(
            {name: value for name, value in module.named_parameters()}
        ),
        "buffer_sha256": state_dict_sha256(
            {name: value for name, value in module.named_buffers()}
        ),
    }


def build_components(
    beta_path: Path,
    selector_path: Path,
    config_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    beta_checkpoint = torch.load(
        beta_path, map_location="cpu", weights_only=False, mmap=True
    )
    selector_checkpoint = torch.load(
        selector_path, map_location="cpu", weights_only=False, mmap=True
    )
    model = build_model(_load_yaml(str(config_path)), torch.device("cpu"))
    model.load_state_dict(beta_checkpoint["model_state_dict"], strict=True)
    selector = DirectPenaltyMLPQuerySelectorV24Additive512(
        hidden=512,
        derivative_floor=1e-6,
        squareplus_scale=0.02,
        direct_head_init_std=1e-3,
    )
    selector.load_state_dict(
        selector_checkpoint["selector_state_dict"], strict=True
    )
    selector_before = state_snapshot(selector)
    model.load_state_dict(beta_checkpoint["model_state_dict"], strict=True)
    selector_after = state_snapshot(selector)
    if selector_before != selector_after:
        raise RuntimeError("beta state loading modified selector state")
    model.to(device).eval()
    selector.to(device).eval()
    for module in (model, selector):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    pairing = {
        "selector_state_before_beta_reload": selector_before,
        "selector_state_after_beta_reload": selector_after,
        "selector_state_preserved": True,
        "latest_beta_state_sha256": state_dict_sha256(
            beta_checkpoint["model_state_dict"]
        ),
        "selector_pinned_beta_state_sha256": selector_checkpoint["metadata"][
            "frozen_beta_state_sha256"
        ],
    }
    pairing["sha_mismatch"] = (
        pairing["latest_beta_state_sha256"]
        != pairing["selector_pinned_beta_state_sha256"]
    )
    return beta_checkpoint, selector_checkpoint, model, selector, pairing


def coefficient_row_hashes(beta: np.ndarray) -> list[str]:
    return [
        hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        for row in np.asarray(beta)
    ]


def evaluate_task(
    *,
    entry: dict[str, Any],
    model: torch.nn.Module,
    selector: torch.nn.Module,
    device: torch.device,
    raw_lambdas: np.ndarray,
    cd_library: Path,
    cd_workers: int,
    cd_max_sweeps: int,
    cd_retries: int,
    cd_tolerance: float,
    cd_zero_tolerance: float,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    x_raw, y_raw, split = load_task(entry)
    context_index = split["context_indices"].astype(np.int64)
    validation_index = split["validation_indices"].astype(np.int64)
    test_index = split["test_indices"].astype(np.int64)
    x_context, y_context = transform_with_frozen_context(
        x_raw, y_raw, context_index, split
    )
    x_validation, _ = transform_with_frozen_context(
        x_raw, y_raw, validation_index, split
    )
    x_test, _ = transform_with_frozen_context(
        x_raw, y_raw, test_index, split
    )
    y_validation_raw = np.asarray(y_raw[validation_index], dtype=np.float64)
    y_test_raw = np.asarray(y_raw[test_index], dtype=np.float64)
    x_device = torch.from_numpy(x_context)[None].to(device)
    y_device = torch.from_numpy(y_context)[None].to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
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
    direct_lambda_max = float(
        np.max(
            np.abs(
                np.asarray(x_context, dtype=np.float64).T
                @ np.asarray(y_context, dtype=np.float64)
            )
        )
        / x_context.shape[0]
    )
    if abs(float(lambda_max[0]) - direct_lambda_max) / max(
        direct_lambda_max, 1e-12
    ) > 1e-5:
        raise RuntimeError("lambda_max audit failed")

    raw_grid = np.asarray(raw_lambdas, dtype=np.float64)
    raw_tensor = torch.as_tensor(
        raw_grid[None, :], device=device, dtype=torch.float32
    )
    np.testing.assert_array_equal(
        raw_tensor[0].cpu().numpy(), raw_grid.astype(np.float32)
    )
    query_count = raw_grid.size
    shape = selected["penalty_shape_features"].reshape(1, 1, -1).expand(
        1, query_count, -1
    )
    d0_tensor = selected["penalty_d0"].reshape(1, 1).expand(1, query_count)
    with torch.inference_mode():
        decoded = model.decode_queries(
            encoded,
            lambdas=raw_tensor,
            lambda_max=lambda_max[:, None].expand(1, query_count),
            penalty_shape_features=shape,
            penalty_d0=d0_tensor,
            return_aux=True,
        )
    kalbi_beta = (
        decoded["beta_q"][0].float().cpu().numpy().astype(np.float64)
    )
    if kalbi_beta.shape != (query_count, x_context.shape[1]):
        raise RuntimeError("coefficient-path shape mismatch")

    crosscheck_indices = sorted({0, query_count // 2, query_count - 1})
    for index in crosscheck_indices:
        with torch.inference_mode():
            single = model.decode_queries(
                encoded,
                lambdas=raw_tensor[:, index : index + 1],
                lambda_max=lambda_max[:, None],
                penalty_shape_features=selected[
                    "penalty_shape_features"
                ].reshape(1, 1, -1),
                penalty_d0=selected["penalty_d0"].reshape(1, 1),
                return_aux=True,
            )["beta_q"][0, 0]
        np.testing.assert_allclose(
            kalbi_beta[index],
            single.float().cpu().numpy(),
            rtol=2e-6,
            atol=2e-7,
        )
    row_hashes = coefficient_row_hashes(kalbi_beta)
    if len(set(row_hashes)) < 2:
        raise RuntimeError("all lambda-grid coefficient vectors were reused")

    selected_raw_lambda = float(selected["raw_lambda"].reshape(-1)[0])
    with torch.inference_mode():
        selected_beta = (
            model.decode_queries(
                encoded,
                lambdas=torch.as_tensor(
                    [[selected_raw_lambda]], device=device, dtype=torch.float32
                ),
                lambda_max=lambda_max[:, None],
                penalty_shape_features=selected[
                    "penalty_shape_features"
                ].reshape(1, 1, -1),
                penalty_d0=selected["penalty_d0"].reshape(1, 1),
                return_aux=True,
            )["beta_q"][0, 0]
            .float()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    knots = (
        selector.penalty_parameter_head.penalty_grid.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    rho = selected["penalty_rho"].reshape(-1).cpu().numpy().astype(np.float64)
    drho = selected["penalty_drho"].reshape(-1).cpu().numpy().astype(np.float64)
    d0 = float(selected["penalty_d0"].reshape(-1)[0])
    relu_b = float(selected["penalty_relu_b"].reshape(-1)[0])
    relu_weights = (
        selected["penalty_relu_weights"]
        .reshape(-1, knots.size)[0]
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    for name, values in {
        "knots": knots,
        "rho": rho,
        "drho": drho,
        "relu_weights": relu_weights,
    }.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"nonfinite penalty {name}")
    if not all(math.isfinite(value) for value in (d0, relu_b)):
        raise RuntimeError("nonfinite penalty scalar")
    if abs(float(rho[0])) > 1e-8:
        raise RuntimeError("rho(0) audit failed")
    if abs(relu_b - d0) > 2e-6 * max(1.0, abs(d0)):
        raise RuntimeError("d0/exact-prox intercept mismatch")
    rho_exact = canonical_penalty_values(knots, knots, relu_b, relu_weights)
    rho_exact_max_error = float(np.max(np.abs(rho - rho_exact)))
    np.testing.assert_allclose(rho, rho_exact, rtol=1e-4, atol=1e-5)

    cd_started = time.perf_counter()
    cd = solve_multistart_path_parallel(
        library_path=cd_library,
        x=x_context,
        y=y_context,
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
    kalbi_objective = objective_path(
        x_context,
        y_context,
        kalbi_beta,
        raw_grid,
        knots,
        relu_b,
        relu_weights,
    )
    cd_objective = objective_path(
        x_context,
        y_context,
        cd_beta,
        raw_grid,
        knots,
        relu_b,
        relu_weights,
    )
    np.testing.assert_allclose(
        cd_objective,
        np.asarray(cd["reference_objective"]),
        rtol=1e-10,
        atol=1e-12,
    )
    if np.max(cd["reference_fixed_point_max"]) > max(
        2e-3, 5 * cd_tolerance
    ):
        raise RuntimeError("CD fixed-point audit failed")
    if np.max(cd["reference_projected_kkt_max"]) > max(
        2e-3, 5 * cd_tolerance
    ):
        raise RuntimeError("CD KKT audit failed")

    y_mean = float(split["y_mean"])
    y_scale = float(split["y_scale"])
    kalbi_validation_mse = mse_path(
        x_validation,
        y_validation_raw,
        kalbi_beta,
        y_mean=y_mean,
        y_scale=y_scale,
    )
    kalbi_test_mse = mse_path(
        x_test,
        y_test_raw,
        kalbi_beta,
        y_mean=y_mean,
        y_scale=y_scale,
    )
    cd_validation_mse = mse_path(
        x_validation,
        y_validation_raw,
        cd_beta,
        y_mean=y_mean,
        y_scale=y_scale,
    )
    cd_test_mse = mse_path(
        x_test,
        y_test_raw,
        cd_beta,
        y_mean=y_mean,
        y_scale=y_scale,
    )
    selected_validation_mse = float(
        mse_path(
            x_validation,
            y_validation_raw,
            selected_beta[None, :],
            y_mean=y_mean,
            y_scale=y_scale,
        )[0]
    )
    selected_test_mse = float(
        mse_path(
            x_test,
            y_test_raw,
            selected_beta[None, :],
            y_mean=y_mean,
            y_scale=y_scale,
        )[0]
    )
    oracle_index = int(np.argmin(kalbi_validation_mse))
    oracle_validation_mse = float(kalbi_validation_mse[oracle_index])
    oracle_test_mse = float(kalbi_test_mse[oracle_index])
    zero_validation = float(entry["zero_predictor_validation_mse"])
    zero_test = float(entry["zero_predictor_test_mse"])
    correlations = np.asarray(
        [
            vector_correlation(left, right)
            for left, right in zip(kalbi_beta, cd_beta, strict=True)
        ]
    )
    top, top_scores = top_features(cd_beta, count=12)
    derivative_difference = np.diff(drho)
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    task_dir = output_dir / f"dataset_{int(entry['dataset_index']):02d}"
    task_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        task_dir / "paths.npz",
        raw_lambdas=raw_grid,
        kalbi_beta=kalbi_beta.astype(np.float32),
        coordinate_descent_beta=cd_beta.astype(np.float32),
        kalbi_objective=kalbi_objective,
        coordinate_descent_objective=cd_objective,
        objective_gap=kalbi_objective - cd_objective,
        kalbi_validation_mse=kalbi_validation_mse,
        kalbi_test_mse=kalbi_test_mse,
        coordinate_descent_validation_mse=cd_validation_mse,
        coordinate_descent_test_mse=cd_test_mse,
        kalbi_validation_mse_normalized=kalbi_validation_mse / zero_validation,
        kalbi_test_mse_normalized=kalbi_test_mse / zero_test,
        coordinate_descent_validation_mse_normalized=cd_validation_mse
        / zero_validation,
        coordinate_descent_test_mse_normalized=cd_test_mse / zero_test,
        selected_raw_lambda=np.asarray(selected_raw_lambda),
        selected_beta=selected_beta.astype(np.float32),
        oracle_index=np.asarray(oracle_index),
        penalty_grid=knots,
        penalty_rho=rho,
        penalty_drho=drho,
        penalty_relu_b=np.asarray(relu_b),
        penalty_relu_weights=relu_weights,
        top_features=top,
        top_feature_scores=top_scores,
        kalbi_support_size=np.sum(np.abs(kalbi_beta) > 1e-6, axis=1),
        cd_support_size=np.sum(np.abs(cd_beta) > 1e-6, axis=1),
        kalbi_coefficient_rms=np.sqrt(np.mean(kalbi_beta**2, axis=1)),
        cd_coefficient_rms=np.sqrt(np.mean(cd_beta**2, axis=1)),
        cd_projected_kkt=cd["reference_projected_kkt_max"],
        cd_fixed_point=cd["reference_fixed_point_max"],
        cd_sweeps=cd["reference_sweeps"],
        cd_converged=np.ones(raw_grid.size, dtype=bool),
        cd_selected_start=cd["reference_selected_start"],
        coefficient_path_correlation=correlations,
    )
    metric = {
        "dataset_index": int(entry["dataset_index"]),
        "task_id": entry["task_id"],
        "outcome_branch": entry["outcome_branch"],
        "seed": int(entry["generator_seed"]),
        "task_hash": entry["task_hash"],
        "n": int(entry["n"]),
        "p": int(entry["p"]),
        "p_over_n": float(entry["p_over_n"]),
        "split_sizes": entry["row_counts"],
        "lambda_max": direct_lambda_max,
        "selected_raw_lambda": selected_raw_lambda,
        "selected_lambda_out_of_grid": not (
            raw_grid[0] <= selected_raw_lambda <= raw_grid[-1]
        ),
        "oracle_raw_lambda": float(raw_grid[oracle_index]),
        "selected_validation_mse": selected_validation_mse,
        "selected_test_mse": selected_test_mse,
        "oracle_validation_mse": oracle_validation_mse,
        "oracle_test_mse": oracle_test_mse,
        "selected_over_oracle_validation_mse": selected_validation_mse
        / max(oracle_validation_mse, 1e-15),
        "selected_over_oracle_test_mse": selected_test_mse
        / max(oracle_test_mse, 1e-15),
        "zero_predictor_validation_mse": zero_validation,
        "zero_predictor_test_mse": zero_test,
        "path_correlation_mean": float(correlations.mean()),
        "path_correlation_min": float(correlations.min()),
        "coefficient_rmse": float(
            np.sqrt(np.mean((kalbi_beta - cd_beta) ** 2))
        ),
        "objective_gap_mean": float(
            np.mean(kalbi_objective - cd_objective)
        ),
        "objective_gap_max": float(
            np.max(kalbi_objective - cd_objective)
        ),
        "selected_support_size": int(
            np.count_nonzero(np.abs(selected_beta) > 1e-6)
        ),
        "oracle_support_size": int(
            np.count_nonzero(np.abs(kalbi_beta[oracle_index]) > 1e-6)
        ),
        "endpoint_nnz_low_lambda": int(
            np.count_nonzero(np.abs(kalbi_beta[0]) > 1e-6)
        ),
        "endpoint_nnz_high_lambda": int(
            np.count_nonzero(np.abs(kalbi_beta[-1]) > 1e-6)
        ),
        "penalty": {
            "rho0_error": abs(float(rho[0])),
            "exact_representation_max_abs_error": rho_exact_max_error,
            "d0": d0,
            "max_rho_prime": float(drho.max()),
            "min_rho_prime": float(drho.min()),
            "derivative_total_variation": float(
                np.sum(np.abs(derivative_difference))
            ),
            "concavity_violation_count": int(
                np.count_nonzero(derivative_difference > 1e-8)
            ),
            "concavity_violation_fraction": float(
                np.mean(derivative_difference > 1e-8)
            ),
        },
        "cd": {
            "full_p": True,
            "starts": ["zero", "KALBI", "previous path point"],
            "coefficient_bound": float(knots[-1]),
            "max_sweeps_per_retry": cd_max_sweeps,
            "retries": cd_retries,
            "tolerance": cd_tolerance,
            "zero_tolerance": cd_zero_tolerance,
            "max_projected_kkt": float(
                np.max(cd["reference_projected_kkt_max"])
            ),
            "max_fixed_point": float(
                np.max(cd["reference_fixed_point_max"])
            ),
            "median_sweeps": float(np.median(cd["reference_sweeps"])),
            "runtime_seconds": cd_seconds,
            "all_converged": True,
            "independent_objective_verified": True,
        },
        "leakage_audit": {
            "preprocessing_fit_rows": "context only",
            "selector_inputs": ["context X", "context y"],
            "validation_y_selector_access": False,
            "test_y_selector_access": False,
            "validation_y_use": "diagnostic oracle and metrics only",
            "test_y_use": "final metrics only",
            "split_disjoint": True,
        },
        "lambda_reuse_test": {
            "single_query_crosscheck_indices": crosscheck_indices,
            "unique_coefficient_vector_hashes": len(set(row_hashes)),
            "grid_points": len(row_hashes),
            "passed": True,
        },
        "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": peak_memory,
        "array_path": str((task_dir / "paths.npz").resolve()),
    }
    metric["array_sha256"] = sha256(task_dir / "paths.npz")
    write_json(task_dir / "metrics.json", metric)
    return metric


def write_summary_csv(metrics: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "dataset_index",
        "task_id",
        "outcome_branch",
        "seed",
        "n",
        "p",
        "p_over_n",
        "path_correlation_mean",
        "path_correlation_min",
        "coefficient_rmse",
        "objective_gap_mean",
        "objective_gap_max",
        "selected_raw_lambda",
        "oracle_raw_lambda",
        "selected_validation_mse",
        "oracle_validation_mse",
        "selected_test_mse",
        "oracle_test_mse",
        "selected_over_oracle_validation_mse",
        "selected_over_oracle_test_mse",
        "selected_support_size",
        "oracle_support_size",
        "endpoint_nnz_low_lambda",
        "endpoint_nnz_high_lambda",
        "penalty_d0",
        "penalty_max_rho_prime",
        "penalty_derivative_total_variation",
        "penalty_concavity_violation_count",
        "cd_max_projected_kkt",
        "cd_max_fixed_point",
        "runtime_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric in metrics:
            row = {key: metric.get(key) for key in fields}
            row.update(
                {
                    "penalty_d0": metric["penalty"]["d0"],
                    "penalty_max_rho_prime": metric["penalty"]["max_rho_prime"],
                    "penalty_derivative_total_variation": metric["penalty"][
                        "derivative_total_variation"
                    ],
                    "penalty_concavity_violation_count": metric["penalty"][
                        "concavity_violation_count"
                    ],
                    "cd_max_projected_kkt": metric["cd"][
                        "max_projected_kkt"
                    ],
                    "cd_max_fixed_point": metric["cd"]["max_fixed_point"],
                }
            )
            writer.writerow(row)


def penalty_diversity(
    metrics: list[dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    arrays = []
    for metric in metrics:
        task_dir = output_dir / f"dataset_{metric['dataset_index']:02d}"
        with np.load(task_dir / "paths.npz", allow_pickle=False) as archive:
            grid = np.asarray(archive["penalty_grid"], dtype=np.float64)
            keep = grid <= 1.0 + 1e-12
            arrays.append(
                (
                    np.asarray(archive["penalty_rho"], dtype=np.float64)[keep],
                    np.asarray(archive["penalty_drho"], dtype=np.float64)[keep],
                )
            )
    rho_distance = []
    drho_distance = []
    for left, right in combinations(range(len(arrays)), 2):
        rho_distance.append(
            float(np.sqrt(np.mean((arrays[left][0] - arrays[right][0]) ** 2)))
        )
        drho_distance.append(
            float(np.sqrt(np.mean((arrays[left][1] - arrays[right][1]) ** 2)))
        )
    d0 = np.asarray([metric["penalty"]["d0"] for metric in metrics])
    if not rho_distance:
        return {
            "task_count": len(metrics),
            "pair_count": 0,
            "rho_pairwise_rmse_mean_on_0_1": None,
            "rho_pairwise_rmse_min_on_0_1": None,
            "rho_pairwise_rmse_max_on_0_1": None,
            "rho_prime_pairwise_rmse_mean_on_0_1": None,
            "rho_prime_pairwise_rmse_min_on_0_1": None,
            "rho_prime_pairwise_rmse_max_on_0_1": None,
            "d0_mean": float(d0.mean()),
            "d0_std": float(d0.std()),
            "d0_coefficient_of_variation": 0.0,
            "distinct_penalty_sha256_count": 1,
        }
    return {
        "task_count": len(metrics),
        "pair_count": len(rho_distance),
        "rho_pairwise_rmse_mean_on_0_1": float(np.mean(rho_distance)),
        "rho_pairwise_rmse_min_on_0_1": float(np.min(rho_distance)),
        "rho_pairwise_rmse_max_on_0_1": float(np.max(rho_distance)),
        "rho_prime_pairwise_rmse_mean_on_0_1": float(
            np.mean(drho_distance)
        ),
        "rho_prime_pairwise_rmse_min_on_0_1": float(np.min(drho_distance)),
        "rho_prime_pairwise_rmse_max_on_0_1": float(np.max(drho_distance)),
        "d0_mean": float(d0.mean()),
        "d0_std": float(d0.std()),
        "d0_coefficient_of_variation": float(
            d0.std() / max(abs(d0.mean()), 1e-15)
        ),
        "distinct_penalty_sha256_count": len(
            {
                hashlib.sha256(
                    np.ascontiguousarray(np.concatenate(pair)).tobytes()
                ).hexdigest()
                for pair in arrays
            }
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--beta-checkpoint", type=Path, required=True)
    parser.add_argument("--beta-sha256", required=True)
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--selector-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--task-limit", type=int, default=4)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--cd-workers", type=int, default=4)
    parser.add_argument("--cd-max-sweeps", type=int, default=4000)
    parser.add_argument("--cd-retries", type=int, default=4)
    parser.add_argument("--cd-tolerance", type=float, default=1e-7)
    parser.add_argument("--cd-zero-tolerance", type=float, default=1e-11)
    args = parser.parse_args()

    if sha256(args.beta_checkpoint) != args.beta_sha256:
        raise RuntimeError("beta checkpoint SHA256 mismatch")
    if sha256(args.selector_checkpoint) != args.selector_sha256:
        raise RuntimeError("selector checkpoint SHA256 mismatch")
    preparation = json.loads(
        (
            args.evaluation_root
            / "data_manifest"
            / "metrics"
            / "preparation_summary.json"
        ).read_text()
    )
    accepted = list(preparation["accepted"])[: args.task_limit]
    if len(accepted) != args.task_limit:
        raise RuntimeError("insufficient frozen tasks")
    output_dir = args.evaluation_root / "metrics" / args.tag
    if output_dir.exists():
        raise RuntimeError(f"refusing existing output: {output_dir}")
    output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    beta_checkpoint, selector_checkpoint, model, selector, pairing = build_components(
        args.beta_checkpoint,
        args.selector_checkpoint,
        args.config,
        device,
    )
    raw_lambdas = np.logspace(-4.0, 0.0, args.grid_size, dtype=np.float64)
    cd_library = (
        Path(
            "/users/minsuk/fm-lm-lambda/"
            "kalbi_v6_exactprox10_eta4x_v24_additive512_traj3_k32s32_"
            "p100k_effbatch32/canonical_cd_kernel.so"
        )
    )
    if not cd_library.is_file():
        raise RuntimeError("CD kernel is missing")
    metrics: list[dict[str, Any]] = []
    for entry in accepted:
        metrics.append(
            evaluate_task(
                entry=entry,
                model=model,
                selector=selector,
                device=device,
                raw_lambdas=raw_lambdas,
                cd_library=cd_library,
                cd_workers=args.cd_workers,
                cd_max_sweeps=args.cd_max_sweeps,
                cd_retries=args.cd_retries,
                cd_tolerance=args.cd_tolerance,
                cd_zero_tolerance=args.cd_zero_tolerance,
                output_dir=output_dir,
            )
        )
    diversity = penalty_diversity(metrics, output_dir)
    write_json(output_dir / "results.json", metrics)
    write_json(output_dir / "penalty_diversity.json", diversity)
    write_summary_csv(metrics, output_dir / "metrics_summary.csv")
    manifest = {
        "status": "PASS",
        "beta_checkpoint": str(args.beta_checkpoint.resolve()),
        "beta_checkpoint_sha256": args.beta_sha256,
        "beta_completed_steps": int(beta_checkpoint["completed_steps"]),
        "beta_model_version": beta_checkpoint["metadata"]["model_version"],
        "selector_checkpoint": str(args.selector_checkpoint.resolve()),
        "selector_checkpoint_sha256": args.selector_sha256,
        "selector_completed_steps": int(selector_checkpoint["completed_steps"]),
        "selector_model_version": selector_checkpoint["metadata"][
            "model_version"
        ],
        "selector_version": selector_checkpoint["metadata"]["selector_version"],
        "pairing": pairing,
        "raw_lambda_grid": raw_lambdas.tolist(),
        "objective_contract": (
            "0.5/N * ||y-X beta||^2 + raw_lambda * sum_j rho(|beta_j|)"
        ),
        "raw_lambda_contract": (
            "same unmodified raw-lambda grid passed to KALBI and CD"
        ),
        "task_count": len(metrics),
        "device": str(device),
        "cd_contract": {
            "full_p": True,
            "starts": ["zero", "KALBI", "previous path point"],
            "max_sweeps": args.cd_max_sweeps,
            "retries": args.cd_retries,
            "tolerance": args.cd_tolerance,
            "zero_tolerance": args.cd_zero_tolerance,
        },
        "numerical_checks": {
            "strict_checkpoint_load": True,
            "selector_state_preserved_during_beta_load": True,
            "context_only_preprocessing": True,
            "row_split_disjointness": True,
            "heldout_y_excluded_from_inference": True,
            "raw_lambda_grid_equality": True,
            "rho0": True,
            "penalty_exact_representation": True,
            "full_p_coordinate_descent": True,
            "independent_objective": True,
            "independent_prediction_per_lambda": True,
        },
    }
    write_json(output_dir / "evaluation_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "tag": args.tag,
                "task_count": len(metrics),
                "device": str(device),
                "pairing_sha_mismatch": pairing["sha_mismatch"],
                "penalty_diversity": diversity,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
