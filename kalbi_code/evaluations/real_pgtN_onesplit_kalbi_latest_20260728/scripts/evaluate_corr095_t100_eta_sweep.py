#!/usr/bin/env python3
"""KALBI-only T=100 eta-candidate multiplier sweep on filtered real tasks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


PROJECT = Path("/users/minsuk/fm-lm-lambda")
SOURCE_ROOT = PROJECT / "evaluations/real_pgtN_onesplit_kalbi_latest_20260728"
BASE_ROOT = SOURCE_ROOT / "kalbi_lasso_corr095_t100_20260728"
SCRIPT_ROOT = SOURCE_ROOT / "scripts"
for entry in (str(PROJECT), str(SCRIPT_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from evaluate_kalbi_t100_reuse_baselines import (  # noqa: E402
    REFINEMENT_SWEEPS,
    build_components,
    decode,
)
from kalbi_v8_adaptiveblockprox10_v23_k32s32_p100k_effbatch32 import (  # noqa: E402
    adaptive_block_prox_v8 as adaptive_v8,
)
from real_pgtN_common import (  # noqa: E402
    apply_x_preprocessing,
    atomic_json,
    atomic_npz,
    fit_preprocessing,
    iter_selected_tasks,
    load_split_manifest,
    load_x,
    parse_task_indices,
    raw_prediction_metrics,
    select_rows,
    sha256,
    zero_predictor_metrics,
)


ETA_FACTORS = (1.0, 2.0, 5.0, 10.0)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def decode_with_eta_candidate_factor(
    model: torch.nn.Module,
    encoded: dict[str, Any],
    selected: dict[str, Any],
    raw_lambdas: torch.Tensor,
    factor: float,
) -> dict[str, Any]:
    """Scale only the initial Lipschitz eta candidate; retain all safety checks."""
    original_get_curvature = adaptive_v8._get_block_curvature

    def scaled_get_curvature(
        encoded_arg: dict[str, Any],
        blocks: tuple[tuple[int, int], ...],
        feature_count: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        estimates, traces = original_get_curvature(
            encoded_arg, blocks, feature_count, block_size
        )
        return estimates / float(factor), traces

    adaptive_v8._get_block_curvature = scaled_get_curvature
    try:
        return decode(model, encoded, selected, raw_lambdas)
    finally:
        adaptive_v8._get_block_curvature = original_get_curvature


def tensor_numpy(decoded: dict[str, Any], key: str) -> np.ndarray:
    return (
        decoded[key][0].detach().float().cpu().numpy().astype(np.float64)
    )


def evaluate_task(
    task: dict[str, Any],
    *,
    model: torch.nn.Module,
    selector: torch.nn.Module,
    device: torch.device,
    output_root: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    task_id = task["task_id"]
    metric_path = output_root / "metrics" / f"{task_id}.json"
    if metric_path.exists():
        existing = load_json(metric_path)
        output_path = Path(existing["path_output"])
        if (
            existing.get("status") == "PASS"
            and output_path.is_file()
            and sha256(output_path) == existing["path_sha256"]
        ):
            return existing
        raise RuntimeError(f"invalid existing result for {task_id}")

    base_metric = load_json(BASE_ROOT / "metrics" / f"{task_id}.json")
    with np.load(BASE_ROOT / "paths" / f"{task_id}.npz") as archive:
        base = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(task["outer_split_path"]) as archive:
        train_index = np.asarray(archive["train_indices"], dtype=np.int64)
        test_index = np.asarray(archive["test_indices"], dtype=np.int64)
    np.testing.assert_array_equal(train_index, base["train_indices"])
    np.testing.assert_array_equal(test_index, base["test_indices"])

    x_store = load_x(Path(task["X_path"]))
    y_store = np.load(task["y_path"], mmap_mode="r")
    x_train_raw = select_rows(x_store, train_index)
    y_train_raw = np.asarray(y_store[train_index], dtype=np.float64)
    x_train_all, y_train, scaler = fit_preprocessing(
        x_train_raw, y_train_raw
    )
    if scaler["scaler_sha256"] != base_metric["scaler_sha256"]:
        raise RuntimeError("preprocessing scaler SHA mismatch")
    retained_local = np.asarray(
        base["retained_local_indices"], dtype=np.int64
    )
    x_train = np.ascontiguousarray(x_train_all[:, retained_local])
    dummy_count = max(0, 32 - x_train.shape[1])
    x_model = (
        np.pad(x_train, ((0, 0), (0, dummy_count)), mode="constant")
        if dummy_count
        else x_train
    )
    x_device = torch.from_numpy(x_model)[None].to(device)
    y_device = torch.from_numpy(y_train)[None].to(device)
    with torch.inference_mode():
        encoded = model.encode_dataset(model.precompute(x_device, y_device))
        selected = selector(
            x_device,
            y_device,
            encoded["lambda_max"].float().reshape(-1),
            solver_context=encoded,
            return_representation=True,
        )
    selected_raw_lambda = float(selected["raw_lambda"].reshape(-1)[0])
    if not math.isclose(
        selected_raw_lambda,
        float(base_metric["selected_raw_lambda"]),
        rel_tol=2e-6,
        abs_tol=2e-8,
    ):
        raise RuntimeError("selector raw-lambda reproduction failed")
    np.testing.assert_allclose(
        selected["penalty_rho"].reshape(-1).detach().cpu().numpy(),
        base["penalty_rho"].reshape(-1),
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        selected["penalty_drho"].reshape(-1).detach().cpu().numpy(),
        base["penalty_drho"].reshape(-1),
        rtol=2e-6,
        atol=2e-7,
    )

    raw_lambdas = np.asarray(base["raw_lambdas"], dtype=np.float64)
    evaluation_lambdas = np.concatenate(
        [raw_lambdas, np.asarray([selected_raw_lambda])]
    )
    lambda_tensor = torch.as_tensor(
        evaluation_lambdas[None], dtype=torch.float32, device=device
    )
    factor_outputs: dict[float, dict[str, np.ndarray]] = {}
    for factor in ETA_FACTORS:
        with torch.inference_mode():
            decoded = decode_with_eta_candidate_factor(
                model, encoded, selected, lambda_tensor, factor
            )
        beta_model = tensor_numpy(decoded, "beta_q")
        beta = beta_model[:, : x_train.shape[1]]
        eta = tensor_numpy(
            decoded, "kalbi_v8_refinement_step_size_path_q"
        )
        gamma = tensor_numpy(
            decoded, "kalbi_v8_refinement_gamma_path_q"
        )
        gamma_error = float(
            np.max(
                np.abs(
                    gamma / np.maximum(eta, 1e-30)
                    - evaluation_lambdas[:, None]
                )
            )
        )
        if gamma_error > 2e-5:
            raise RuntimeError(f"factor {factor:g} gamma/eta contract failed")
        factor_outputs[factor] = {
            "beta": beta,
            "eta": eta,
            "gamma": gamma,
            "majorization": tensor_numpy(
                decoded, "kalbi_v8_refinement_majorization_ratio_path_q"
            ),
            "backtracks": tensor_numpy(
                decoded, "kalbi_v8_refinement_backtracks_path_q"
            ),
            "trace_fallback": tensor_numpy(
                decoded, "kalbi_v8_refinement_trace_fallback_path_q"
            ),
            "rejected_fraction": tensor_numpy(
                decoded, "kalbi_v8_refinement_rejected_fraction_path_q"
            ),
            "objective_delta": tensor_numpy(
                decoded, "kalbi_v8_refinement_objective_delta_path_q"
            ),
            "monotonicity_violations": tensor_numpy(
                decoded,
                "kalbi_v8_refinement_monotonicity_violations_path_q",
            ),
            "dummy_maximum_abs_beta": np.asarray(
                float(
                    np.max(
                        np.abs(beta_model[:, x_train.shape[1] :]),
                        initial=0.0,
                    )
                )
            ),
            "gamma_error": np.asarray(gamma_error),
        }
    reproduction_error = float(
        np.max(
            np.abs(
                factor_outputs[1.0]["beta"][:-1]
                - np.asarray(base["kalbi_beta"], dtype=np.float64)
            )
        )
    )
    if reproduction_error > 3e-5:
        raise RuntimeError(
            f"eta 1x path reproduction failed: {reproduction_error}"
        )

    # All paths and the selected coefficients are frozen before reading test y.
    x_test = apply_x_preprocessing(
        select_rows(x_store, test_index), scaler
    )[:, retained_local]
    y_test_raw = np.asarray(y_store[test_index], dtype=np.float64)
    zero = zero_predictor_metrics(y_test_raw, float(scaler["y_mean"]))
    denominator = max(float(zero["mse"]), 1e-30)
    saved: dict[str, np.ndarray] = {
        "raw_lambdas": raw_lambdas,
        "evaluation_lambdas": evaluation_lambdas,
        "train_indices": train_index,
        "test_indices": test_index,
        "retained_local_indices": retained_local,
        "lasso_beta": np.asarray(base["lasso_beta"]),
        "lasso_normalized_test_mse": np.asarray(
            base["lasso_normalized_test_mse"]
        ),
        "cv_lasso_raw_lambda": np.asarray(base["cv_lasso_raw_lambda"]),
        "cv_lasso_normalized_test_mse": np.asarray(
            base["cv_lasso_normalized_test_mse"]
        ),
        "selected_raw_lambda": np.asarray(selected_raw_lambda),
        "penalty_grid": np.asarray(base["penalty_grid"]),
        "penalty_rho": np.asarray(base["penalty_rho"]),
        "penalty_drho": np.asarray(base["penalty_drho"]),
        "top_features": np.asarray(base["top_features"]),
        "top_original_features": np.asarray(base["top_original_features"]),
    }
    factors_metric: dict[str, Any] = {}
    base_eta = factor_outputs[1.0]["eta"]
    for factor, output in factor_outputs.items():
        prediction = raw_prediction_metrics(
            x_test,
            y_test_raw,
            output["beta"],
            y_mean=float(scaler["y_mean"]),
            y_scale=float(scaler["y_scale"]),
        )
        normalized = np.asarray(prediction["mse"]) / denominator
        key = f"eta{factor:g}x"
        saved[f"{key}_beta"] = output["beta"].astype(np.float32)
        saved[f"{key}_normalized_test_mse"] = normalized
        saved[f"{key}_eta"] = output["eta"].astype(np.float32)
        saved[f"{key}_majorization"] = output["majorization"].astype(
            np.float32
        )
        saved[f"{key}_backtracks"] = output["backtracks"].astype(np.float32)
        saved[f"{key}_rejected_fraction"] = output[
            "rejected_fraction"
        ].astype(np.float32)
        oracle_index = int(np.argmin(normalized[:-1]))
        first_eta_ratio = output["eta"][:, 0] / np.maximum(
            base_eta[:, 0], 1e-30
        )
        factors_metric[key] = {
            "requested_initial_eta_multiplier": factor,
            "accepted_first_sweep_eta_ratio_median": float(
                np.median(first_eta_ratio)
            ),
            "accepted_first_sweep_eta_ratio_max": float(
                np.max(first_eta_ratio)
            ),
            "curvature_or_backtracking_binding_fraction": float(
                np.mean(first_eta_ratio < factor * (1.0 - 1e-6))
            ),
            "mean_backtracks_per_query_sweep": float(
                np.mean(output["backtracks"])
            ),
            "trace_fallback_fraction": float(
                np.mean(output["trace_fallback"] > 0.5)
            ),
            "rejected_fraction": float(
                np.mean(output["rejected_fraction"])
            ),
            "maximum_majorization_ratio": float(
                np.max(output["majorization"])
            ),
            "maximum_objective_increase": float(
                np.max(output["objective_delta"])
            ),
            "monotonicity_violation_count": int(
                np.sum(output["monotonicity_violations"])
            ),
            "gamma_over_eta_max_absolute_error": float(
                output["gamma_error"]
            ),
            "path_oracle_raw_lambda": float(raw_lambdas[oracle_index]),
            "path_oracle_normalized_test_mse": float(
                normalized[oracle_index]
            ),
            "selected_normalized_test_mse": float(normalized[-1]),
            "maximum_abs_beta": float(np.max(np.abs(output["beta"]))),
            "dummy_maximum_abs_beta": float(
                output["dummy_maximum_abs_beta"]
            ),
        }
    output_path = output_root / "paths" / f"{task_id}.npz"
    atomic_npz(output_path, **saved)
    metric = {
        "status": "PASS",
        "registry_index": int(task["registry_index"]),
        "task_id": task_id,
        "N": int(task["N"]),
        "P_after_correlation_filter": int(x_train.shape[1]),
        "refinement_sweeps": REFINEMENT_SWEEPS,
        "eta_multiplier_contract": (
            "initial_lipschitz_candidate_only; curvature cap, "
            "majorization<=0.95, objective monotonicity and backtracking retained"
        ),
        "eta_1x_reproduction_max_abs_error": reproduction_error,
        "factors": factors_metric,
        "test_y_accessed_after_all_paths_freeze": True,
        "path_output": str(output_path),
        "path_sha256": sha256(output_path),
        "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(metric_path, metric)
    return metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-indices", default="all")
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument(
        "--beta-checkpoint",
        type=Path,
        default=PROJECT
        / "runs/kalbi_v8_adaptive_t10_betaonly_supportmix_v14_target3080_"
        "effbatch32_to60k_20260728/checkpoint_step043000.pt",
    )
    parser.add_argument(
        "--selector-checkpoint",
        type=Path,
        default=PROJECT
        / "runs/kalbi_v24_selectoronly_v14_snr0p2to50_frozenbeta_step42700_"
        "effbatch32_20260728/checkpoint_selector_step009400.pt",
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "fpk_v33_kkt0.yaml"
    )
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    manifest = load_split_manifest(
        SOURCE_ROOT / "manifests/split_manifest.json"
    )
    indices = parse_task_indices(args.task_indices, manifest["task_count"])
    device = torch.device(args.device)
    beta_checkpoint, selector_checkpoint, model, selector = build_components(
        args.beta_checkpoint, args.selector_checkpoint, args.config, device
    )
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for task in iter_selected_tasks(manifest, indices):
        try:
            metric = evaluate_task(
                task,
                model=model,
                selector=selector,
                device=device,
                output_root=args.output_root,
            )
            results.append(metric)
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "registry_index": metric["registry_index"],
                        "task_id": metric["task_id"],
                        "selected": {
                            key: value["selected_normalized_test_mse"]
                            for key, value in metric["factors"].items()
                        },
                        "seconds": metric["runtime_seconds"],
                    }
                ),
                flush=True,
            )
        except Exception as error:
            failure = {
                "status": "FAIL",
                "registry_index": int(task["registry_index"]),
                "task_id": task["task_id"],
                "error_type": type(error).__name__,
                "error": str(error),
            }
            failures.append(failure)
            print(json.dumps(failure), flush=True)
    report = {
        "status": "PASS" if not failures else "PARTIAL",
        "eta_factors": ETA_FACTORS,
        "refinement_sweeps": REFINEMENT_SWEEPS,
        "task_indices": indices,
        "success_count": len(results),
        "failure_count": len(failures),
        "failures": failures,
        "beta_checkpoint": str(args.beta_checkpoint),
        "beta_checkpoint_sha256": sha256(args.beta_checkpoint),
        "beta_completed_steps": int(beta_checkpoint["completed_steps"]),
        "selector_checkpoint": str(args.selector_checkpoint),
        "selector_checkpoint_sha256": sha256(args.selector_checkpoint),
        "selector_completed_steps": int(selector_checkpoint["completed_steps"]),
    }
    tag = "_".join(map(str, indices))
    atomic_json(args.output_root / "reports" / f"worker_{tag}.json", report)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
