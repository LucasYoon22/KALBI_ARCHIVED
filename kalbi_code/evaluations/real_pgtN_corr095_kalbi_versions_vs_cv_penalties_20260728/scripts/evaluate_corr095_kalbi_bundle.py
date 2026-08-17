#!/usr/bin/env python3
"""Evaluate one frozen KALBI bundle after train-only corr095 filtering."""

from __future__ import annotations

import argparse
import hashlib
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
    projected_kkt_max,
)
from kalbi_v6_exactprox10_eta4x_v23_independentraw_k32s32_p100k_effbatch32.model import (
    build_model as build_v6_model,
)
from kalbi_v6_exactprox10_eta4x_v23_independentraw_k32s32_p100k_effbatch32.selector_v23_independent_rawlambda import (
    DirectPenaltyMLPQuerySelectorV23IndependentRawLambda,
)
from kalbi_v6_exactprox10_eta4x_v24_additive512_traj3_k32s32_p100k_effbatch32.selector_v24_additive512_trajectory import (
    DirectPenaltyMLPQuerySelectorV24Additive512,
)
from kalbi_v8_adaptiveblockprox10_v23_k32s32_p100k_effbatch32.model import (
    build_model as build_v8_model,
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


def retained_index_sha256(indices: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(indices, dtype=np.int64).tobytes()
    ).hexdigest()


def build_components(
    bundle: dict[str, Any],
    config_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], torch.nn.Module, torch.nn.Module]:
    beta_path = Path(bundle["beta_checkpoint"])
    selector_path = Path(bundle["selector_checkpoint"])
    if sha256(beta_path) != bundle["beta_checkpoint_sha256"]:
        raise RuntimeError("beta checkpoint SHA mismatch")
    if sha256(selector_path) != bundle["selector_checkpoint_sha256"]:
        raise RuntimeError("selector checkpoint SHA mismatch")

    beta_checkpoint = torch.load(
        beta_path, map_location="cpu", weights_only=False
    )
    selector_checkpoint = torch.load(
        selector_path, map_location="cpu", weights_only=False
    )
    kind = bundle["kind"]
    if kind == "v6_v23":
        model = build_v6_model(_load_yaml(str(config_path)), device)
        selector: torch.nn.Module = (
            DirectPenaltyMLPQuerySelectorV23IndependentRawLambda()
        )
    elif kind == "v8_v23":
        model = build_v8_model(_load_yaml(str(config_path)), device)
        selector = DirectPenaltyMLPQuerySelectorV23IndependentRawLambda()
    elif kind == "v8_v24":
        model = build_v8_model(_load_yaml(str(config_path)), device)
        selector = DirectPenaltyMLPQuerySelectorV24Additive512(
            hidden=512,
            derivative_floor=1e-6,
            squareplus_scale=0.02,
            direct_head_init_std=1e-3,
        )
    else:
        raise ValueError(f"unknown bundle kind: {kind}")

    model.load_state_dict(beta_checkpoint["model_state_dict"], strict=True)
    selector.load_state_dict(
        selector_checkpoint["selector_state_dict"], strict=True
    )
    model.to(device).eval()
    selector.to(device).eval()
    for module in (model, selector):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return beta_checkpoint, selector_checkpoint, model, selector


def run_selector(
    bundle: dict[str, Any],
    selector: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    encoded: dict[str, torch.Tensor],
) -> dict[str, Any]:
    lambda_max = encoded["lambda_max"].float().reshape(-1)
    if bundle["kind"] == "v8_v24":
        return selector(
            x,
            y,
            lambda_max,
            solver_context=encoded,
            return_representation=True,
        )
    return selector(
        x,
        y,
        lambda_max,
        return_representation=True,
    )


def decode_queries(
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    selected: dict[str, Any],
    raw_lambdas: torch.Tensor,
) -> dict[str, Any]:
    query_count = int(raw_lambdas.shape[1])
    return model.decode_queries(
        encoded,
        lambdas=raw_lambdas,
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
    task: dict[str, Any],
    bundle: dict[str, Any],
    model: torch.nn.Module,
    selector: torch.nn.Module,
    device: torch.device,
    raw_lambda_grid: np.ndarray,
    evaluation_root: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    task_id = task["task_id"]
    for key, hash_key in (
        ("X_path", "X_sha256"),
        ("y_path", "y_sha256"),
        ("outer_split_path", "outer_split_sha256"),
    ):
        path = Path(task[key])
        if sha256(path) != task[hash_key]:
            raise RuntimeError(f"artifact hash mismatch: {path}")

    x_store = load_x(Path(task["X_path"]))
    y_store = np.load(Path(task["y_path"]), mmap_mode="r")
    with np.load(Path(task["outer_split_path"]), allow_pickle=False) as split:
        train_indices = np.asarray(split["train_indices"], dtype=np.int64)
        test_indices = np.asarray(split["test_indices"], dtype=np.int64)
    if np.intersect1d(train_indices, test_indices).size:
        raise RuntimeError("outer split overlap")

    x_train_raw = select_rows(x_store, train_indices)
    y_train_raw = np.asarray(y_store[train_indices], dtype=np.float64)
    x_train, y_train, scaler = fit_preprocessing(
        x_train_raw, y_train_raw
    )
    preprocessing_path, preprocessing_sha = save_preprocessing(
        evaluation_root, task, scaler
    )
    retained_original_indices = np.asarray(
        scaler["retained_original_indices"], dtype=np.int64
    )

    x_device = torch.from_numpy(x_train)[None].to(device)
    y_device = torch.from_numpy(y_train)[None].to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    selector_started = time.perf_counter()
    with torch.inference_mode():
        encoded = model.encode_dataset(model.precompute(x_device, y_device))
        selected = run_selector(
            bundle, selector, x_device, y_device, encoded
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
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
    encoded_lambda_max = float(
        encoded["lambda_max"].float().reshape(-1)[0]
    )
    relative_lambda_max_error = abs(
        encoded_lambda_max - direct_lambda_max
    ) / max(direct_lambda_max, 1e-12)
    if relative_lambda_max_error > 2e-5:
        raise RuntimeError("encoded/direct lambda_max mismatch")

    selected_raw_lambda = float(selected["raw_lambda"].reshape(-1)[0])
    d0 = float(selected["penalty_d0"].reshape(-1)[0])
    max_drho = float(selected["penalty_drho"].float().max())
    if (
        not math.isfinite(selected_raw_lambda)
        or not 1e-6 <= selected_raw_lambda <= 10.0
    ):
        raise RuntimeError("selected raw lambda leaves training support")
    if not math.isfinite(d0) or d0 <= 1e-8:
        raise RuntimeError("invalid penalty d0")

    all_raw_lambdas = np.concatenate(
        [
            np.asarray([selected_raw_lambda], dtype=np.float64),
            np.asarray(raw_lambda_grid, dtype=np.float64),
        ]
    )
    raw_tensor = torch.as_tensor(
        all_raw_lambdas[None],
        dtype=torch.float32,
        device=device,
    )
    beta_started = time.perf_counter()
    with torch.inference_mode():
        decoded = decode_queries(model, encoded, selected, raw_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    beta_seconds = time.perf_counter() - beta_started
    beta = (
        decoded["beta_q"][0]
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    if beta.shape != (all_raw_lambdas.size, x_train.shape[1]):
        raise RuntimeError("decoded beta shape mismatch")
    selected_beta = beta[0]
    path_beta = beta[1:]

    refinement_prefix = (
        "kalbi_v6" if bundle["kind"] == "v6_v23" else "kalbi_v8"
    )
    eta_key = f"{refinement_prefix}_refinement_step_size_path_q"
    gamma_key = f"{refinement_prefix}_refinement_gamma_path_q"
    if eta_key not in decoded or gamma_key not in decoded:
        raise RuntimeError("refinement eta/gamma telemetry is missing")
    eta_path = decoded[eta_key][0].float().cpu().numpy().astype(np.float64)
    gamma_path = (
        decoded[gamma_key][0].float().cpu().numpy().astype(np.float64)
    )
    gamma_contract_error = float(
        np.max(
            np.abs(
                gamma_path / np.maximum(eta_path, 1e-30)
                - all_raw_lambdas[:, None]
            )
        )
    )
    if gamma_contract_error > 2e-5:
        raise RuntimeError("gamma/eta raw-lambda contract failed")

    knots = (
        selector.penalty_parameter_head.penalty_grid.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    penalty_rho = (
        selected["penalty_rho"].reshape(-1).float().cpu().numpy()
    )
    penalty_drho = (
        selected["penalty_drho"].reshape(-1).float().cpu().numpy()
    )
    if abs(float(penalty_rho[0])) > 1e-8:
        raise RuntimeError("rho(0) contract failed")
    if abs(float(penalty_rho[-1]) - 1.0) > 2e-6:
        raise RuntimeError("rho(6) contract failed")
    if not np.isfinite(penalty_rho).all() or not np.isfinite(
        penalty_drho
    ).all():
        raise RuntimeError("non-finite selected penalty")
    if float(np.min(penalty_drho)) < -1e-8:
        raise RuntimeError("negative selected penalty derivative")
    relu_b = float(selected["penalty_relu_b"].reshape(-1)[0])
    relu_weights = (
        selected["penalty_relu_weights"]
        .reshape(-1, knots.size)[0]
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    objective = np.asarray(
        [
            objective_components(
                x_train,
                y_train,
                coefficient,
                float(raw_lambda),
                knots,
                relu_b,
                relu_weights,
            )
            for coefficient, raw_lambda in zip(
                beta, all_raw_lambdas, strict=True
            )
        ],
        dtype=np.float64,
    )
    projected_kkt = np.asarray(
        [
            projected_kkt_max(
                x_train,
                y_train,
                coefficient,
                float(raw_lambda),
                knots,
                relu_b,
                relu_weights,
                6.0,
            )
            for coefficient, raw_lambda in zip(
                beta, all_raw_lambdas, strict=True
            )
        ],
        dtype=np.float64,
    )
    if not np.isfinite(objective).all() or not np.isfinite(
        projected_kkt
    ).all():
        raise RuntimeError("non-finite objective or projected KKT")

    # Test y is materialized only after selector and coefficient paths freeze.
    x_test_raw = select_rows(x_store, test_indices)
    x_test = apply_x_preprocessing(x_test_raw, scaler)
    y_test_raw = np.asarray(y_store[test_indices], dtype=np.float64)
    selected_test = raw_prediction_metrics(
        x_test,
        y_test_raw,
        selected_beta,
        y_mean=float(scaler["y_mean"]),
        y_scale=float(scaler["y_scale"]),
    )
    path_test = raw_prediction_metrics(
        x_test,
        y_test_raw,
        path_beta,
        y_mean=float(scaler["y_mean"]),
        y_scale=float(scaler["y_scale"]),
    )
    zero = zero_predictor_metrics(y_test_raw, float(scaler["y_mean"]))
    selected_nmse = float(
        selected_test["mse"][0] / max(float(zero["mse"]), 1e-30)
    )

    bundle_root = evaluation_root / "bundles" / bundle["bundle_id"]
    path_output = bundle_root / "paths" / f"{task_id}.npz"
    atomic_npz(
        path_output,
        raw_lambdas=np.asarray(raw_lambda_grid, dtype=np.float64),
        selected_raw_lambda=np.asarray(selected_raw_lambda),
        selected_beta=selected_beta.astype(np.float32),
        path_beta=path_beta.astype(np.float32),
        selected_test_prediction=selected_test["prediction"][0].astype(
            np.float32
        ),
        path_test_mse=np.asarray(path_test["mse"], dtype=np.float64),
        path_normalized_test_mse=(
            np.asarray(path_test["mse"], dtype=np.float64)
            / max(float(zero["mse"]), 1e-30)
        ),
        selected_train_data_half=objective[0, 0],
        selected_train_penalty=objective[0, 1],
        selected_train_objective=objective[0, 2],
        selected_projected_kkt_max=projected_kkt[0],
        path_train_data_half=objective[1:, 0],
        path_train_penalty=objective[1:, 1],
        path_train_objective=objective[1:, 2],
        path_projected_kkt_max=projected_kkt[1:],
        selected_refinement_eta=eta_path[0].astype(np.float32),
        selected_refinement_gamma=gamma_path[0].astype(np.float32),
        path_refinement_eta=eta_path[1:].astype(np.float32),
        path_refinement_gamma=gamma_path[1:].astype(np.float32),
        penalty_rho=penalty_rho.astype(np.float32),
        penalty_drho=penalty_drho.astype(np.float32),
        penalty_d0=np.asarray(d0),
        retained_original_indices=retained_original_indices,
        train_indices=train_indices,
        test_indices=test_indices,
    )

    metric = {
        "status": "PASS",
        "bundle_id": bundle["bundle_id"],
        "bundle_kind": bundle["kind"],
        "registry_index": int(task["registry_index"]),
        "task_id": task_id,
        "source": task["source"],
        "source_family": task_family(task["source"], task_id),
        "target": task["target"],
        "N": int(task["N"]),
        "outer_train_rows": int(train_indices.size),
        "outer_test_rows": int(test_indices.size),
        "P_raw": int(task["P_raw"]),
        "P_after_constant_filter": int(
            np.asarray(scaler["active"]).sum()
        ),
        "P_after_corr095": int(x_train.shape[1]),
        "P_after_corr095_over_N": float(x_train.shape[1] / int(task["N"])),
        "remained_high_dimensional": bool(
            x_train.shape[1] > int(task["N"])
        ),
        "correlation_threshold": float(scaler["correlation_threshold"]),
        "removed_correlated_features": int(
            scaler["removed_correlated_feature_count"]
        ),
        "retained_max_abs_correlation": float(
            scaler["retained_max_abs_correlation"]
        ),
        "retained_max_abs_correlation_pair_local": np.asarray(
            scaler["retained_max_abs_correlation_pair_local"],
            dtype=np.int64,
        ).tolist(),
        "retained_original_indices_sha256": retained_index_sha256(
            retained_original_indices
        ),
        "preprocessing_path": str(preprocessing_path),
        "preprocessing_sha256": preprocessing_sha,
        "scaler_sha256": scaler["scaler_sha256"],
        "selected_raw_lambda": selected_raw_lambda,
        "lambda_max": direct_lambda_max,
        "lambda_max_relative_audit_error": relative_lambda_max_error,
        "penalty_d0": d0,
        "penalty_max_drho": max_drho,
        "selected_raw_lambda_over_lambda_max": (
            selected_raw_lambda / max(direct_lambda_max, 1e-30)
        ),
        "selected_zero_kkt_ratio": (
            selected_raw_lambda * d0 / max(direct_lambda_max, 1e-30)
        ),
        "selected_force_ratio": (
            selected_raw_lambda * max_drho
            / max(direct_lambda_max, 1e-30)
        ),
        "selected_test_mse": float(selected_test["mse"][0]),
        "selected_test_rmse": float(selected_test["rmse"][0]),
        "selected_test_mae": float(selected_test["mae"][0]),
        "selected_test_r2": float(selected_test["r2"][0]),
        "selected_normalized_test_mse": selected_nmse,
        "selected_standardized_test_mse": float(
            np.mean(
                (
                    (
                        y_test_raw - float(scaler["y_mean"])
                    )
                    / float(scaler["y_scale"])
                    - x_test @ selected_beta
                )
                ** 2
            )
        ),
        "zero_predictor_test_mse": float(zero["mse"]),
        "nnz_1e8": int(np.count_nonzero(np.abs(selected_beta) > 1e-8)),
        "nnz_1e6": int(np.count_nonzero(np.abs(selected_beta) > 1e-6)),
        "nnz_1e4": int(np.count_nonzero(np.abs(selected_beta) > 1e-4)),
        "coefficient_l1": float(np.sum(np.abs(selected_beta))),
        "coefficient_l2": float(np.linalg.norm(selected_beta)),
        "selected_train_data_half": float(objective[0, 0]),
        "selected_train_penalty": float(objective[0, 1]),
        "selected_train_objective": float(objective[0, 2]),
        "selected_projected_kkt_max": float(projected_kkt[0]),
        "path_projected_kkt_max": float(np.max(projected_kkt[1:])),
        "gamma_over_eta_raw_lambda_max_abs_error": gamma_contract_error,
        "path_oracle_raw_lambda_telemetry_only": float(
            raw_lambda_grid[int(np.argmin(path_test["mse"]))]
        ),
        "path_oracle_test_mse_telemetry_only": float(
            np.min(path_test["mse"])
        ),
        "path_output": str(path_output),
        "path_sha256": sha256(path_output),
        "test_y_accessed_after_selector_and_path_freeze": True,
        "runtime": {
            "selector_seconds": selector_seconds,
            "beta_seconds": beta_seconds,
            "total_seconds": time.perf_counter() - started,
            "cuda_peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
    }
    metric_path = bundle_root / "metrics" / f"{task_id}.json"
    atomic_json(metric_path, metric)
    return metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task-indices", default="all")
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()

    torch.set_num_threads(int(args.torch_threads))
    manifest = json.loads(
        args.bundle_manifest.read_text(encoding="utf-8")
    )
    bundles = {
        item["bundle_id"]: item for item in manifest["bundles"]
    }
    if args.bundle_id not in bundles:
        raise KeyError(f"unknown bundle id: {args.bundle_id}")
    bundle = bundles[args.bundle_id]

    split_manifest = load_split_manifest(args.split_manifest)
    indices = parse_task_indices(
        args.task_indices, int(split_manifest["task_count"])
    )
    device = torch.device(args.device)
    beta_checkpoint, selector_checkpoint, model, selector = build_components(
        bundle, args.config, device
    )
    raw_lambdas = np.logspace(
        -4.0, 0.0, int(args.grid_size), dtype=np.float64
    )

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for task in iter_selected_tasks(split_manifest, indices):
        task_id = task["task_id"]
        metric_path = (
            args.evaluation_root
            / "bundles"
            / bundle["bundle_id"]
            / "metrics"
            / f"{task_id}.json"
        )
        if metric_path.exists():
            existing = json.loads(metric_path.read_text(encoding="utf-8"))
            output_path = Path(existing["path_output"])
            if (
                existing.get("status") == "PASS"
                and output_path.is_file()
                and sha256(output_path) == existing["path_sha256"]
            ):
                successes.append(existing)
                print(
                    json.dumps(
                        {
                            "status": "SKIP_VERIFIED",
                            "bundle_id": bundle["bundle_id"],
                            "task_id": task_id,
                        }
                    ),
                    flush=True,
                )
                continue
            raise RuntimeError(f"invalid existing output: {metric_path}")
        try:
            result = evaluate_task(
                task,
                bundle,
                model,
                selector,
                device,
                raw_lambdas,
                args.evaluation_root,
            )
            successes.append(result)
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "bundle_id": bundle["bundle_id"],
                        "registry_index": result["registry_index"],
                        "task_id": task_id,
                        "P_after_corr095": result["P_after_corr095"],
                        "nmse": result["selected_normalized_test_mse"],
                        "seconds": result["runtime"]["total_seconds"],
                    }
                ),
                flush=True,
            )
        except Exception as error:
            failure = {
                "status": "FAIL",
                "bundle_id": bundle["bundle_id"],
                "task_id": task_id,
                "registry_index": int(task["registry_index"]),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            atomic_json(
                args.evaluation_root
                / "bundles"
                / bundle["bundle_id"]
                / "metrics"
                / f"{task_id}_failure.json",
                failure,
            )
            print(json.dumps(failure), flush=True)

    report = {
        "status": "PASS" if not failures else "PARTIAL",
        "bundle": bundle,
        "beta_completed_steps": int(beta_checkpoint["completed_steps"]),
        "selector_completed_steps": int(
            selector_checkpoint["completed_steps"]
        ),
        "success_count": len(successes),
        "failure_count": len(failures),
        "task_indices": indices,
        "raw_lambda_grid": raw_lambdas.tolist(),
        "failures": failures,
    }
    tag = "_".join(map(str, indices))
    atomic_json(
        args.evaluation_root
        / "reports"
        / f"{bundle['bundle_id']}_{tag}.json",
        report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
