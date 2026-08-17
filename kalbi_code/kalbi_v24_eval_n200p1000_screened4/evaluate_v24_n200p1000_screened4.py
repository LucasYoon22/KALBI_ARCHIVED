#!/usr/bin/env python3
"""Evaluate pinned KALBI V24 and full-P exact CD on four frozen tasks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
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
from kalbi_v6_exactprox10_eta4x_v24_additive512_traj3_k32s32_p100k_effbatch32.model import (
    build_model,
)
from kalbi_v6_exactprox10_eta4x_v24_additive512_traj3_k32s32_p100k_effbatch32.selector_v24_additive512_trajectory import (
    DirectPenaltyMLPQuerySelectorV24Additive512,
)
from kalbi_v6_exactprox10_eta4x_v24_additive512_traj3_k32s32_p100k_effbatch32.train import (
    state_dict_sha256,
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
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def vector_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= 1e-15:
        return 1.0 if np.max(np.abs(x - y), initial=0.0) <= 1e-12 else 0.0
    return float(np.dot(x, y) / denominator)


def r2_path(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    estimate = np.asarray(prediction, dtype=np.float64)
    denominator = float(np.sum((target - target.mean()) ** 2))
    if denominator <= 1e-12:
        raise RuntimeError("R2 target is degenerate")
    return 1.0 - np.sum((estimate - target[None, :]) ** 2, axis=1) / denominator


def prediction_metrics(
    x_standardized: np.ndarray,
    y_raw: np.ndarray,
    beta_path: np.ndarray,
    *,
    y_mean: float,
    y_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction_standardized = (
        np.asarray(beta_path, dtype=np.float64)
        @ np.asarray(x_standardized, dtype=np.float64).T
    )
    prediction_raw = y_mean + y_scale * prediction_standardized
    target = np.asarray(y_raw, dtype=np.float64).reshape(-1)
    mse = np.mean((prediction_raw - target[None, :]) ** 2, axis=1)
    r2 = r2_path(target, prediction_raw)
    expected = np.asarray(
        [
            np.mean((row - target) ** 2)
            for row in prediction_raw
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(mse, expected, rtol=1e-12, atol=1e-14)
    return prediction_raw, mse, r2


def objective_path(
    x: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    raw_lambdas: np.ndarray,
    knots: np.ndarray,
    relu_b: float,
    relu_weights: np.ndarray,
) -> np.ndarray:
    values = [
        objective_components(
            x,
            y,
            coefficient,
            float(raw_lambda),
            knots,
            relu_b,
            relu_weights,
        )[2]
        for coefficient, raw_lambda in zip(beta, raw_lambdas, strict=True)
    ]
    return np.asarray(values, dtype=np.float64)


def top_features(reference: np.ndarray, count: int = 12) -> np.ndarray:
    score = np.max(np.abs(reference), axis=0)
    return np.lexsort((np.arange(score.size), -score))[:count]


def load_task(
    entry: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    data_path = Path(entry["accepted_data_path"])
    split_path = Path(entry["split_path"])
    if sha256(data_path) != entry["accepted_data_sha256"]:
        raise RuntimeError("accepted data SHA256 mismatch")
    if sha256(split_path) != entry["split_sha256"]:
        raise RuntimeError("split SHA256 mismatch")
    with np.load(data_path, allow_pickle=False) as archive:
        x = np.asarray(archive["X"][0], dtype=np.float32)
        y = np.asarray(archive["y"][0], dtype=np.float32)
    with np.load(split_path, allow_pickle=False) as archive:
        split = {key: np.asarray(archive[key]) for key in archive.files}
    indices = [
        set(split[name].astype(np.int64).tolist())
        for name in ("context_indices", "screening_indices", "final_indices")
    ]
    if indices[0] & indices[1] or indices[0] & indices[2] or indices[1] & indices[2]:
        raise RuntimeError("context/screening/final split leakage")
    if len(indices[0] | indices[1] | indices[2]) != x.shape[0]:
        raise RuntimeError("split indices do not cover all rows")
    return x, y, split


def standardize_from_frozen(
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
        np.asarray(y[indices], dtype=np.float64)
        - float(split["y_mean"])
    ) / float(split["y_scale"])
    active = np.asarray(split["active"], dtype=bool)
    x_scaled[:, ~active] = 0.0
    if not np.isfinite(x_scaled).all() or not np.isfinite(y_scaled).all():
        raise RuntimeError("frozen preprocessing produced nonfinite values")
    return x_scaled.astype(np.float32), y_scaled.astype(np.float32)


def build_components(
    checkpoint_path: Path,
    config_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], torch.nn.Module, torch.nn.Module]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model = build_model(_load_yaml(str(config_path)), device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
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
        checkpoint["selector_state_dict"], strict=True
    )
    selector.to(device).eval()
    if state_dict_sha256(checkpoint["model_state_dict"]) != checkpoint[
        "metadata"
    ]["frozen_beta_state_sha256"]:
        raise RuntimeError("frozen beta state SHA256 mismatch")
    return checkpoint, model, selector


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
    final_index = split["final_indices"].astype(np.int64)
    x_context, y_context = standardize_from_frozen(
        x_raw, y_raw, context_index, split
    )
    x_final, _ = standardize_from_frozen(
        x_raw, y_raw, final_index, split
    )
    y_final_raw = np.asarray(y_raw[final_index], dtype=np.float64)
    x_device = torch.from_numpy(x_context)[None].to(device)
    y_device = torch.from_numpy(y_context)[None].to(device)

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
        raise RuntimeError("encoded/direct lambda_max mismatch")
    d0 = float(selected["penalty_d0"].reshape(-1)[0])
    max_drho = float(selected["penalty_drho"].max())
    if not math.isfinite(d0) or d0 <= 1e-8:
        raise RuntimeError("invalid selected penalty d0")
    raw_grid = np.asarray(raw_lambdas, dtype=np.float64)
    if np.any(raw_grid < 1e-6) or np.any(raw_grid > 10.0):
        raise RuntimeError("raw-lambda grid leaves beta support")
    zero_kkt_ratio = raw_grid * d0 / direct_lambda_max
    force_ratio = raw_grid * max_drho / direct_lambda_max
    recovered = zero_kkt_ratio * direct_lambda_max / d0
    np.testing.assert_allclose(recovered, raw_grid, rtol=1e-12, atol=1e-14)

    query_count = raw_grid.size
    shape = selected["penalty_shape_features"].reshape(1, 1, -1).expand(
        1, query_count, -1
    )
    d0_tensor = selected["penalty_d0"].reshape(1, 1).expand(
        1, query_count
    )
    raw_tensor = torch.as_tensor(
        raw_grid[None, :], device=device, dtype=torch.float32
    )
    if not np.array_equal(
        raw_tensor[0].detach().cpu().numpy().astype(np.float64),
        raw_grid.astype(np.float32).astype(np.float64),
    ):
        raise RuntimeError("KALBI raw-lambda tensor mismatch")
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
    selected_raw_lambda = float(selected["raw_lambda"].reshape(-1)[0])
    with torch.inference_mode():
        selected_decoded = model.decode_queries(
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
        )
    selected_beta = (
        selected_decoded["beta_q"][0, 0]
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
    if cd_beta.shape != kalbi_beta.shape:
        raise RuntimeError("KALBI/CD feature-order shape mismatch")
    if np.max(cd["reference_fixed_point_max"]) > max(
        2e-3, 5.0 * cd_tolerance
    ):
        raise RuntimeError("CD fixed-point audit failed")
    if np.max(cd["reference_projected_kkt_max"]) > max(
        2e-3, 5.0 * cd_tolerance
    ):
        raise RuntimeError("CD projected-KKT audit failed")

    kalbi_objective = objective_path(
        x_context,
        y_context,
        kalbi_beta,
        raw_grid,
        knots,
        relu_b,
        relu_weights,
    )
    cd_objective = np.asarray(cd["reference_objective"], dtype=np.float64)
    y_mean = float(split["y_mean"])
    y_scale = float(split["y_scale"])
    _, kalbi_mse, kalbi_r2 = prediction_metrics(
        x_final,
        y_final_raw,
        kalbi_beta,
        y_mean=y_mean,
        y_scale=y_scale,
    )
    _, cd_mse, cd_r2 = prediction_metrics(
        x_final,
        y_final_raw,
        cd_beta,
        y_mean=y_mean,
        y_scale=y_scale,
    )
    _, selected_mse_array, selected_r2_array = prediction_metrics(
        x_final,
        y_final_raw,
        selected_beta[None, :],
        y_mean=y_mean,
        y_scale=y_scale,
    )
    oracle_index = int(np.argmin(kalbi_mse))
    correlation = np.asarray(
        [
            vector_correlation(left, right)
            for left, right in zip(kalbi_beta, cd_beta, strict=True)
        ]
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
    peak_memory = int(torch.cuda.max_memory_allocated(device))
    task_dir = output_dir / f"dataset_{int(entry['dataset_index']):02d}_{entry['label']}"
    task_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        task_dir / "paths.npz",
        raw_lambdas=raw_grid,
        zero_kkt_ratio=zero_kkt_ratio,
        force_ratio=force_ratio,
        kalbi_beta=kalbi_beta.astype(np.float32),
        coordinate_descent_beta=cd_beta.astype(np.float32),
        kalbi_objective=kalbi_objective,
        coordinate_descent_objective=cd_objective,
        kalbi_final_mse=kalbi_mse,
        coordinate_descent_final_mse=cd_mse,
        kalbi_final_r2=kalbi_r2,
        coordinate_descent_final_r2=cd_r2,
        selected_raw_lambda=np.asarray(selected_raw_lambda),
        selected_beta=selected_beta.astype(np.float32),
        selected_final_mse=selected_mse_array,
        selected_final_r2=selected_r2_array,
        oracle_index=np.asarray(oracle_index),
        penalty_grid=knots,
        penalty_rho=penalty_rho,
        penalty_drho=penalty_drho,
        top_features=top,
        cd_projected_kkt=cd["reference_projected_kkt_max"],
        cd_fixed_point=cd["reference_fixed_point_max"],
        cd_sweeps=cd["reference_sweeps"],
        cd_selected_start=cd["reference_selected_start"],
    )
    metric = {
        "dataset_index": int(entry["dataset_index"]),
        "label": entry["label"],
        "source": entry["source"],
        "seed": int(entry["generator_seed"]),
        "n": int(entry["n"]),
        "p": int(entry["p"]),
        "p_over_n": float(entry["p"] / entry["n"]),
        "lambda_max": direct_lambda_max,
        "selected_raw_lambda": selected_raw_lambda,
        "selected_zero_kkt_ratio": selected_raw_lambda * d0 / direct_lambda_max,
        "selected_force_ratio": selected_raw_lambda
        * max_drho
        / direct_lambda_max,
        "selected_final_mse": float(selected_mse_array[0]),
        "selected_final_r2": float(selected_r2_array[0]),
        "oracle_raw_lambda": float(raw_grid[oracle_index]),
        "oracle_final_mse": float(kalbi_mse[oracle_index]),
        "oracle_final_r2": float(kalbi_r2[oracle_index]),
        "selected_over_oracle_mse": float(
            selected_mse_array[0] / max(kalbi_mse[oracle_index], 1e-12)
        ),
        "cv_lasso_screening_r2": float(
            entry.get("screening_lasso_r2", float("nan"))
        ),
        "cv_lasso_final_r2": float(
            entry.get("final_lasso_r2", float("nan"))
        ),
        "cv_lasso_final_mse": float(
            entry.get("final_lasso_mse", float("nan"))
        ),
        "cv_lasso_selected_ratio": float(
            entry.get("screening_lasso_selected_ratio", float("nan"))
        ),
        "cv_lasso_nnz": int(
            np.count_nonzero(
                np.abs(
                    np.asarray(
                        split.get(
                            "screening_lasso_beta",
                            np.zeros(x_raw.shape[1], dtype=np.float32),
                        )
                    )
                )
                > 1e-4
            )
        ),
        "zero_predictor_final_mse": float(entry["zero_predictor_final_mse"]),
        "path_correlation_mean": float(correlation.mean()),
        "path_correlation_min": float(correlation.min()),
        "coefficient_rmse": float(
            np.sqrt(np.mean((kalbi_beta - cd_beta) ** 2))
        ),
        "objective_gap_mean": float(
            np.mean(kalbi_objective - cd_objective)
        ),
        "objective_gap_max": float(
            np.max(kalbi_objective - cd_objective)
        ),
        "selected_nnz_1e6": int(
            np.count_nonzero(np.abs(selected_beta) > 1e-6)
        ),
        "oracle_nnz_1e6": int(
            np.count_nonzero(np.abs(kalbi_beta[oracle_index]) > 1e-6)
        ),
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
        "cd": {
            "max_projected_kkt": float(
                np.max(cd["reference_projected_kkt_max"])
            ),
            "max_fixed_point": float(
                np.max(cd["reference_fixed_point_max"])
            ),
            "median_sweeps": float(np.median(cd["reference_sweeps"])),
            "runtime_seconds": cd_seconds,
        },
        "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": peak_memory,
        "array_path": str(task_dir / "paths.npz"),
        "array_sha256": sha256(task_dir / "paths.npz"),
    }
    write_json(task_dir / "metrics.json", metric)
    return metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--task-limit", type=int, default=4)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--cd-workers", type=int, default=4)
    parser.add_argument("--cd-max-sweeps", type=int, default=2000)
    parser.add_argument("--cd-retries", type=int, default=4)
    parser.add_argument("--cd-tolerance", type=float, default=1e-6)
    parser.add_argument("--cd-zero-tolerance", type=float, default=1e-10)
    args = parser.parse_args()

    if sha256(args.checkpoint) != args.checkpoint_sha256:
        raise RuntimeError("checkpoint SHA256 mismatch")
    preparation = json.loads(
        (args.evaluation_root / "metrics" / "preparation_summary.json").read_text()
    )
    accepted = list(preparation["accepted"])[: args.task_limit]
    if len(accepted) != args.task_limit:
        raise RuntimeError("insufficient frozen tasks")
    output_dir = args.evaluation_root / args.tag
    if output_dir.exists():
        raise RuntimeError(f"refusing existing evaluation output: {output_dir}")
    output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    checkpoint, model, selector = build_components(
        args.checkpoint, args.config, device
    )
    raw_lambdas = np.logspace(
        -4.0, 0.0, args.grid_size, dtype=np.float64
    )
    package = Path(
        "/users/minsuk/fm-lm-lambda/"
        "kalbi_v6_exactprox10_eta4x_v24_additive512_traj3_k32s32_p100k_"
        "effbatch32"
    )
    cd_library = package / "canonical_cd_kernel.so"
    if not cd_library.is_file():
        raise RuntimeError("canonical CD shared library is missing")
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
    write_json(output_dir / "results.json", metrics)
    fields = [
        "dataset_index",
        "label",
        "source",
        "seed",
        "n",
        "p",
        "p_over_n",
        "selected_raw_lambda",
        "selected_final_mse",
        "selected_final_r2",
        "oracle_raw_lambda",
        "oracle_final_mse",
        "oracle_final_r2",
        "selected_over_oracle_mse",
        "cv_lasso_screening_r2",
        "cv_lasso_final_mse",
        "cv_lasso_final_r2",
        "path_correlation_mean",
        "coefficient_rmse",
        "objective_gap_mean",
        "selected_nnz_1e6",
        "runtime_seconds",
        "peak_gpu_memory_bytes",
    ]
    with (output_dir / "dataset_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)
    manifest = {
        "status": "PASS",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": args.checkpoint_sha256,
        "completed_steps": int(checkpoint["completed_steps"]),
        "frozen_beta_state_sha256": checkpoint["metadata"][
            "frozen_beta_state_sha256"
        ],
        "selector_source_sha256": sha256(
            package / "selector_v24_additive512_trajectory.py"
        ),
        "raw_lambda_contract": "raw lambda passed unchanged to KALBI and CD",
        "raw_lambda_grid": raw_lambdas.tolist(),
        "grid_size": args.grid_size,
        "task_count": len(metrics),
        "preparation_summary_sha256": sha256(
            args.evaluation_root / "metrics" / "preparation_summary.json"
        ),
        "numerical_checks": {
            "strict_checkpoint_load": True,
            "raw_lambda_grid_equality": True,
            "context_only_preprocessing": True,
            "row_split_disjointness": True,
            "independent_prediction_per_lambda": True,
            "rho0_and_direct96_coherence": True,
            "full_p_coordinate_descent": True,
        },
    }
    write_json(output_dir / "reproducibility_manifest.json", manifest)
    print(json.dumps({"manifest": manifest, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
