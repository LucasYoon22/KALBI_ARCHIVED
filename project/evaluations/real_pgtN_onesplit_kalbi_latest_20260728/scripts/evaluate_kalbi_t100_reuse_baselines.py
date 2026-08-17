#!/usr/bin/env python3
"""Re-evaluate only KALBI with 100 V8 block-prox sweeps.

The frozen split, train-only preprocessing, selector output, raw-lambda grid,
and previously computed LASSO/SCAD/MCP results are reused and audited.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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
SCRIPT_ROOT = SOURCE_ROOT / "scripts"
for entry in (str(PROJECT), str(SCRIPT_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from kalbi_v6_exactprox10_eta4x_v24_additive512_traj3_k32s32_p100k_effbatch32.selector_v24_additive512_trajectory import (  # noqa: E402
    DirectPenaltyMLPQuerySelectorV24Additive512,
)
from kalbi_v8_adaptiveblockprox10_v23_k32s32_p100k_effbatch32 import (  # noqa: E402
    model as v8_model,
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
    task_family,
    zero_predictor_metrics,
)
from train_fpk import _load_yaml  # noqa: E402


REFINEMENT_SWEEPS = 100


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_components(
    beta_checkpoint_path: Path,
    selector_checkpoint_path: Path,
    config_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], torch.nn.Module, torch.nn.Module]:
    v8_model.KALBI_ADAPTIVE_BLOCK_CONFIG = replace(
        v8_model.KALBI_ADAPTIVE_BLOCK_CONFIG,
        sweeps=REFINEMENT_SWEEPS,
        activation_checkpointing=False,
    )
    beta_checkpoint = torch.load(
        beta_checkpoint_path, map_location="cpu", weights_only=False
    )
    selector_checkpoint = torch.load(
        selector_checkpoint_path, map_location="cpu", weights_only=False
    )
    model = v8_model.build_model(_load_yaml(str(config_path)), device)
    model.load_state_dict(beta_checkpoint["model_state_dict"], strict=True)
    model.kalbi_refinement_sweeps = REFINEMENT_SWEEPS
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


def decode(
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


def method_snapshot(method: dict[str, Any] | None) -> dict[str, Any]:
    if method is None:
        return {
            "status": "MISSING",
            "selected_raw_lambda": None,
            "normalized_test_mse": None,
            "test_mse": None,
        }
    return {
        "status": method.get("status", "PASS"),
        "selected_raw_lambda": method.get("selected_raw_lambda"),
        "normalized_test_mse": method.get("normalized_test_mse"),
        "test_mse": method.get("test_mse"),
    }


def deterministic_top_features(
    kalbi_beta: np.ndarray,
    lasso_beta: np.ndarray,
    original_indices: np.ndarray,
    count: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    importance = np.maximum(
        np.max(np.abs(kalbi_beta), axis=0),
        np.max(np.abs(lasso_beta), axis=0),
    )
    order = np.lexsort((original_indices, -importance))
    selected = order[: min(count, order.size)].astype(np.int64)
    return selected, original_indices[selected].astype(np.int64)


def evaluate_task(
    task: dict[str, Any],
    *,
    model: torch.nn.Module,
    selector: torch.nn.Module,
    device: torch.device,
    source_root: Path,
    old_comparison_root: Path,
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
        raise RuntimeError(f"invalid existing output for {task_id}")

    with np.load(task["outer_split_path"], allow_pickle=False) as archive:
        train_index = np.asarray(archive["train_indices"], dtype=np.int64)
        test_index = np.asarray(archive["test_indices"], dtype=np.int64)
    x_store = load_x(Path(task["X_path"]))
    y_store = np.load(task["y_path"], mmap_mode="r")
    x_train_raw = select_rows(x_store, train_index)
    y_train_raw = np.asarray(y_store[train_index], dtype=np.float64)
    x_train, y_train, scaler = fit_preprocessing(x_train_raw, y_train_raw)

    old_kalbi_metric = load_json(
        source_root / "metrics" / f"{task_id}_kalbi.json"
    )
    old_baseline_metric = load_json(
        source_root / "metrics" / f"{task_id}_baselines.json"
    )
    if old_kalbi_metric["scaler_sha256"] != scaler["scaler_sha256"]:
        raise RuntimeError("preprocessing scaler SHA mismatch")

    with np.load(
        source_root / "paths" / f"{task_id}.npz", allow_pickle=False
    ) as archive:
        old_penalty_rho = np.asarray(archive["penalty_rho"], dtype=np.float64)
        old_penalty_drho = np.asarray(
            archive["penalty_drho"], dtype=np.float64
        )
    old_path_path = old_comparison_root / "paths" / f"{task_id}.npz"
    with np.load(old_path_path, allow_pickle=False) as archive:
        reused = {key: np.asarray(archive[key]) for key in archive.files}
    raw_lambdas = np.asarray(reused["raw_lambdas"], dtype=np.float64)
    np.testing.assert_array_equal(reused["train_indices"], train_index)
    np.testing.assert_array_equal(reused["test_indices"], test_index)

    x_device = torch.from_numpy(x_train)[None].to(device)
    y_device = torch.from_numpy(y_train)[None].to(device)
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
    selected_raw_lambda = float(selected["raw_lambda"].reshape(-1)[0])
    old_selected_raw_lambda = float(old_kalbi_metric["selected_raw_lambda"])
    if not math.isclose(
        selected_raw_lambda,
        old_selected_raw_lambda,
        rel_tol=2e-6,
        abs_tol=2e-8,
    ):
        raise RuntimeError("selector raw-lambda output changed")
    np.testing.assert_allclose(
        selected["penalty_rho"].reshape(-1).cpu().numpy(),
        old_penalty_rho,
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        selected["penalty_drho"].reshape(-1).cpu().numpy(),
        old_penalty_drho,
        rtol=2e-6,
        atol=2e-7,
    )

    raw_tensor = torch.as_tensor(
        raw_lambdas[None], dtype=torch.float32, device=device
    )
    selected_tensor = torch.as_tensor(
        [[selected_raw_lambda]], dtype=torch.float32, device=device
    )
    with torch.inference_mode():
        decoded = decode(model, encoded, selected, raw_tensor)
        selected_decoded = decode(model, encoded, selected, selected_tensor)
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
    if eta_path.shape != (raw_lambdas.size, REFINEMENT_SWEEPS):
        raise RuntimeError(f"unexpected T=100 eta shape {eta_path.shape}")
    gamma_error = float(
        np.max(
            np.abs(
                gamma_path / np.maximum(eta_path, 1e-30)
                - raw_lambdas[:, None]
            )
        )
    )
    if gamma_error > 2e-5:
        raise RuntimeError("gamma/eta raw-lambda contract failed")

    # Test y is read only after the selector and both coefficient outputs freeze.
    x_test = apply_x_preprocessing(select_rows(x_store, test_index), scaler)
    y_test_raw = np.asarray(y_store[test_index], dtype=np.float64)
    kalbi_test = raw_prediction_metrics(
        x_test,
        y_test_raw,
        kalbi_beta,
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
    normalized = kalbi_test["mse"] / max(zero["mse"], 1e-30)
    selected_normalized = float(
        selected_test["mse"][0] / max(zero["mse"], 1e-30)
    )

    active_original_indices = np.asarray(
        reused["active_original_indices"], dtype=np.int64
    )
    top_local, top_original = deterministic_top_features(
        kalbi_beta,
        np.asarray(reused["lasso_beta"], dtype=np.float64),
        active_original_indices,
    )
    reused.update(
        {
            "kalbi_beta": kalbi_beta.astype(np.float32),
            "kalbi_test_mse": np.asarray(kalbi_test["mse"]),
            "kalbi_normalized_test_mse": np.asarray(normalized),
            "kalbi_nnz": np.count_nonzero(
                np.abs(kalbi_beta) > 1e-8, axis=1
            ),
            "kalbi_l1": np.sum(np.abs(kalbi_beta), axis=1),
            "kalbi_l2": np.linalg.norm(kalbi_beta, axis=1),
            "kalbi_t10_beta": np.asarray(
                reused["kalbi_beta"], dtype=np.float32
            ),
            "kalbi_t10_test_mse": np.asarray(
                reused["kalbi_test_mse"], dtype=np.float64
            ),
            "kalbi_t10_normalized_test_mse": np.asarray(
                reused["kalbi_normalized_test_mse"], dtype=np.float64
            ),
            "selected_raw_lambda": np.asarray(selected_raw_lambda),
            "selected_beta": selected_beta.astype(np.float32),
            "selected_test_mse": np.asarray(selected_test["mse"][0]),
            "selected_normalized_test_mse": np.asarray(selected_normalized),
            "v8_t100_eta_path": eta_path.astype(np.float32),
            "v8_t100_gamma_path": gamma_path.astype(np.float32),
            "top_features": top_local,
            "top_original_features": top_original,
        }
    )
    # The update above needs the original T=10 arrays, captured before replacement.
    with np.load(old_path_path, allow_pickle=False) as archive:
        reused["kalbi_t10_beta"] = np.asarray(
            archive["kalbi_beta"], dtype=np.float32
        )
        reused["kalbi_t10_test_mse"] = np.asarray(
            archive["kalbi_test_mse"], dtype=np.float64
        )
        reused["kalbi_t10_normalized_test_mse"] = np.asarray(
            archive["kalbi_normalized_test_mse"], dtype=np.float64
        )

    output_path = output_root / "paths" / f"{task_id}.npz"
    atomic_npz(output_path, **reused)
    baseline_methods = old_baseline_metric.get("methods", {})
    oracle_index = int(np.argmin(normalized))
    metric = {
        "status": "PASS",
        "registry_index": int(task["registry_index"]),
        "task_id": task_id,
        "source_family": task_family(task["source"], task_id),
        "N": int(task["N"]),
        "P_transformed": int(x_train.shape[1]),
        "outer_train_rows": int(train_index.size),
        "outer_test_rows": int(test_index.size),
        "scaler_sha256": scaler["scaler_sha256"],
        "split_sha256": task["outer_split_sha256"],
        "refinement_sweeps": REFINEMENT_SWEEPS,
        "raw_lambda_count": int(raw_lambdas.size),
        "raw_lambda_min": float(raw_lambdas[0]),
        "raw_lambda_max": float(raw_lambdas[-1]),
        "selector_output_reused_exactly": True,
        "selected_raw_lambda": selected_raw_lambda,
        "selected_normalized_test_mse": selected_normalized,
        "selected_test_mse": float(selected_test["mse"][0]),
        "path_oracle_raw_lambda": float(raw_lambdas[oracle_index]),
        "path_oracle_normalized_test_mse": float(normalized[oracle_index]),
        "path_oracle_test_mse": float(kalbi_test["mse"][oracle_index]),
        "t10_selected_normalized_test_mse": float(
            old_kalbi_metric["selected_normalized_test_mse"]
        ),
        "t10_path_oracle_normalized_test_mse": float(
            np.min(reused["kalbi_t10_normalized_test_mse"])
        ),
        "maximum_abs_beta": float(np.max(np.abs(kalbi_beta))),
        "selected_maximum_abs_beta": float(np.max(np.abs(selected_beta))),
        "gamma_over_eta_max_absolute_error": gamma_error,
        "baselines_reused_without_recomputation": {
            name: method_snapshot(baseline_methods.get(name))
            for name in ("lasso", "scad", "mcp")
        },
        "old_lasso_path_sha256": sha256(old_path_path),
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
    parser.add_argument(
        "--source-root", type=Path, default=SOURCE_ROOT
    )
    parser.add_argument(
        "--old-comparison-root",
        type=Path,
        default=SOURCE_ROOT / "lasso_path_comparison_20260728",
    )
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)

    compatibility = load_json(
        args.source_root / "manifests/checkpoint_compatibility.json"
    )
    if sha256(args.beta_checkpoint) != compatibility["beta_checkpoint_sha256"]:
        raise RuntimeError("beta checkpoint SHA mismatch")
    if (
        sha256(args.selector_checkpoint)
        != compatibility["selector_checkpoint_sha256"]
    ):
        raise RuntimeError("selector checkpoint SHA mismatch")
    manifest = load_split_manifest(
        args.source_root / "manifests/split_manifest.json"
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
                source_root=args.source_root,
                old_comparison_root=args.old_comparison_root,
                output_root=args.output_root,
            )
            results.append(metric)
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "registry_index": metric["registry_index"],
                        "task_id": metric["task_id"],
                        "selected_normalized_mse": metric[
                            "selected_normalized_test_mse"
                        ],
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
        "baselines_recomputed": False,
    }
    tag = "_".join(map(str, indices))
    atomic_json(args.output_root / "reports" / f"worker_{tag}.json", report)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
