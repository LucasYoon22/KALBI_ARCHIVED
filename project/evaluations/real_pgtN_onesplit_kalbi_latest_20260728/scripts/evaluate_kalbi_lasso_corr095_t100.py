#!/usr/bin/env python3
"""Re-evaluate frozen KALBI and LASSO after train-only correlation filtering."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import KFold


PROJECT = Path("/users/minsuk/fm-lm-lambda")
SOURCE_ROOT = PROJECT / "evaluations/real_pgtN_onesplit_kalbi_latest_20260728"
SCRIPT_ROOT = SOURCE_ROOT / "scripts"
for entry in (str(PROJECT), str(SCRIPT_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from evaluate_kalbi_lasso_rawlambda_paths import (  # noqa: E402
    fit_lasso_raw_path,
    lasso_objective_and_kkt,
)
from evaluate_kalbi_t100_reuse_baselines import (  # noqa: E402
    REFINEMENT_SWEEPS,
    build_components,
    decode,
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


CORRELATION_THRESHOLD = 0.95
FILTER_CONTRACT = "TRAIN_ONLY_GREEDY_ABS_CORR_GE_0P95_ORIGINAL_ORDER_V1"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def greedy_absolute_correlation_filter(
    x_standardized: np.ndarray,
    *,
    threshold: float = CORRELATION_THRESHOLD,
    block_size: int = 256,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep a feature iff its correlation with all earlier kept features is < threshold."""
    x = np.ascontiguousarray(x_standardized, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError(f"invalid standardized X shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("standardized X contains non-finite values")
    n, p = x.shape
    # P is at most 5,472 in the frozen manifest. A single float32 Gram
    # (about 120 MB at the maximum) is substantially faster than repeated
    # skinny GEMMs and keeps the deterministic greedy rule unchanged.
    gram = np.abs((x.T @ x) / float(n))
    np.fill_diagonal(gram, 0.0)
    kept: list[int] = []
    pre_max = float(gram.max(initial=0.0))
    post_max = 0.0
    nearest_rejected: list[float] = []
    for feature in range(p):
        if kept:
            prior = gram[feature, np.asarray(kept, dtype=np.int64)]
            maximum = float(prior.max(initial=0.0))
        else:
            maximum = 0.0
        if maximum >= float(threshold):
            nearest_rejected.append(maximum)
            continue
        kept.append(feature)
        post_max = max(post_max, maximum)
    del gram
    retained = np.asarray(kept, dtype=np.int64)
    if retained.size < 1:
        raise RuntimeError("correlation filter removed every feature")
    if post_max >= float(threshold) + 2e-6:
        raise RuntimeError(
            f"post-filter maximum correlation {post_max} violates threshold"
        )
    return retained, {
        "filter_contract": FILTER_CONTRACT,
        "threshold": float(threshold),
        "comparison_dtype": "float32",
        "population_denominator": int(n),
        "ordering": "ascending post-constant-filter feature index",
        "P_before": int(p),
        "P_after": int(retained.size),
        "removed_count": int(p - retained.size),
        "removed_fraction": float(1.0 - retained.size / p),
        "pre_filter_max_abs_corr": pre_max,
        "post_filter_max_abs_corr": post_max,
        "minimum_rejected_abs_corr": (
            float(min(nearest_rejected)) if nearest_rejected else None
        ),
        "retained_local_indices_sha256": array_sha256(retained),
    }


def lasso_cv_train_only_filtered(
    x_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    *,
    split_seed: int,
    ratios_descending: np.ndarray,
    tolerance: float,
    max_iterations: int,
    filter_block_size: int,
    cv_workers: int,
) -> dict[str, Any]:
    fold_count = 5 if x_train_raw.shape[0] >= 100 else 3
    splitter = KFold(
        n_splits=fold_count,
        shuffle=True,
        random_state=int(split_seed ^ 0x00C0FFEE),
    )
    splits = list(splitter.split(np.arange(x_train_raw.shape[0])))

    def evaluate_fold(
        payload: tuple[int, tuple[np.ndarray, np.ndarray]],
    ) -> tuple[int, np.ndarray, dict[str, Any]]:
        fold_index, (fit_index, valid_index) = payload
        x_fit, y_fit, scaler = fit_preprocessing(
            x_train_raw[fit_index], y_train_raw[fit_index]
        )
        retained, filter_metric = greedy_absolute_correlation_filter(
            x_fit, block_size=filter_block_size
        )
        x_fit = x_fit[:, retained]
        x_valid = apply_x_preprocessing(
            x_train_raw[valid_index], scaler
        )[:, retained]
        lambda_max = float(
            np.max(np.abs(x_fit.astype(np.float64).T @ y_fit.astype(np.float64)))
            / x_fit.shape[0]
        )
        raw_ascending = lambda_max * ratios_descending[::-1]
        beta_ascending, diagnostic = fit_lasso_raw_path(
            x_fit,
            y_fit,
            raw_ascending,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        prediction = raw_prediction_metrics(
            x_valid,
            y_train_raw[valid_index],
            beta_ascending,
            y_mean=float(scaler["y_mean"]),
            y_scale=float(scaler["y_scale"]),
        )
        return (
            fold_index,
            np.asarray(prediction["mse"], dtype=np.float64)[::-1],
            {
                "fold": fold_index,
                "fit_rows": int(fit_index.size),
                "valid_rows": int(valid_index.size),
                "lambda_max": lambda_max,
                "filter": filter_metric,
                "convergence_warning_count": int(
                    np.asarray(diagnostic["convergence_warnings"]).sum()
                ),
                "maximum_dual_gap": float(
                    np.asarray(diagnostic["dual_gaps"]).max(initial=0.0)
                ),
            },
        )

    payloads = list(enumerate(splits))
    worker_count = min(max(1, int(cv_workers)), fold_count)
    if worker_count == 1:
        results = [evaluate_fold(payload) for payload in payloads]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(evaluate_fold, payloads))
    results.sort(key=lambda item: item[0])
    fold_mse = [item[1] for item in results]
    fold_diagnostics = [item[2] for item in results]
    scores = np.asarray(fold_mse, dtype=np.float64)
    mean = scores.mean(axis=0)
    standard_error = (
        scores.std(axis=0, ddof=1) / math.sqrt(scores.shape[0])
    )
    selected_index = int(np.argmin(mean))
    return {
        "fold_count": fold_count,
        "ratios_descending": ratios_descending,
        "fold_mse": scores,
        "mean_mse": mean,
        "standard_error": standard_error,
        "selected_index": selected_index,
        "selected_ratio": float(ratios_descending[selected_index]),
        "fold_diagnostics": fold_diagnostics,
    }


def deterministic_top_features(
    kalbi_beta: np.ndarray,
    lasso_beta: np.ndarray,
    original_indices: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    importance = np.maximum(
        np.max(np.abs(kalbi_beta), axis=0),
        np.max(np.abs(lasso_beta), axis=0),
    )
    order = np.lexsort((original_indices, -importance))
    selected = order[: min(int(count), order.size)].astype(np.int64)
    return selected, original_indices[selected].astype(np.int64)


def evaluate_task(
    task: dict[str, Any],
    *,
    model: torch.nn.Module,
    selector: torch.nn.Module,
    device: torch.device,
    output_root: Path,
    args: argparse.Namespace,
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

    with np.load(task["outer_split_path"], allow_pickle=False) as archive:
        train_index = np.asarray(archive["train_indices"], dtype=np.int64)
        test_index = np.asarray(archive["test_indices"], dtype=np.int64)
    x_store = load_x(Path(task["X_path"]))
    y_store = np.load(task["y_path"], mmap_mode="r")
    x_train_raw = select_rows(x_store, train_index)
    y_train_raw = np.asarray(y_store[train_index], dtype=np.float64)
    x_train_all, y_train, scaler = fit_preprocessing(
        x_train_raw, y_train_raw
    )
    retained_local, filter_metric = greedy_absolute_correlation_filter(
        x_train_all, block_size=args.filter_block_size
    )
    active_original = np.flatnonzero(
        np.asarray(scaler["active"], dtype=bool)
    ).astype(np.int64)
    retained_original = active_original[retained_local]
    filter_metric["retained_original_indices_sha256"] = array_sha256(
        retained_original
    )
    x_train = np.ascontiguousarray(x_train_all[:, retained_local])
    model_feature_count = max(32, x_train.shape[1])
    model_dummy_count = model_feature_count - x_train.shape[1]
    if model_dummy_count:
        x_train_model = np.pad(
            x_train,
            ((0, 0), (0, model_dummy_count)),
            mode="constant",
            constant_values=0.0,
        )
    else:
        x_train_model = x_train
    lambda_max = float(
        np.max(np.abs(x_train.astype(np.float64).T @ y_train.astype(np.float64)))
        / x_train.shape[0]
    )

    cv_ratios = np.logspace(0.0, -4.0, args.cv_lambda_count)
    cv = lasso_cv_train_only_filtered(
        x_train_raw,
        y_train_raw,
        split_seed=int(task["outer_split_seed"]),
        ratios_descending=cv_ratios,
        tolerance=args.cv_lasso_tolerance,
        max_iterations=args.cv_lasso_max_iterations,
        filter_block_size=args.filter_block_size,
        cv_workers=args.cv_workers,
    )

    raw_lambdas = np.logspace(
        math.log10(args.raw_lambda_min),
        math.log10(args.raw_lambda_max),
        args.raw_lambda_count,
    )
    lasso_beta, lasso_diagnostic = fit_lasso_raw_path(
        x_train,
        y_train,
        raw_lambdas,
        tolerance=args.lasso_tolerance,
        max_iterations=args.lasso_max_iterations,
    )
    lasso_objective, lasso_data_term, lasso_kkt = lasso_objective_and_kkt(
        x_train, y_train, lasso_beta, raw_lambdas
    )

    selected_ratio_index = int(cv["selected_index"])
    outer_cv_raw_ascending = lambda_max * cv_ratios[::-1]
    outer_cv_beta_ascending, outer_cv_diagnostic = fit_lasso_raw_path(
        x_train,
        y_train,
        outer_cv_raw_ascending,
        tolerance=args.lasso_tolerance,
        max_iterations=args.lasso_max_iterations,
    )
    cv_lasso_beta = outer_cv_beta_ascending[
        cv_ratios.size - 1 - selected_ratio_index
    ]
    cv_lasso_raw_lambda = lambda_max * float(cv["selected_ratio"])

    x_device = torch.from_numpy(x_train_model)[None].to(device)
    y_device = torch.from_numpy(y_train)[None].to(device)
    with torch.inference_mode():
        encoded = model.encode_dataset(model.precompute(x_device, y_device))
        encoded_lambda_max = encoded["lambda_max"].float().reshape(-1)
        selected = selector(
            x_device,
            y_device,
            encoded_lambda_max,
            solver_context=encoded,
            return_representation=True,
        )
        raw_tensor = torch.as_tensor(
            raw_lambdas[None], dtype=torch.float32, device=device
        )
        decoded = decode(model, encoded, selected, raw_tensor)
        selected_raw_lambda = float(selected["raw_lambda"].reshape(-1)[0])
        selected_decoded = decode(
            model,
            encoded,
            selected,
            torch.as_tensor(
                [[selected_raw_lambda]], dtype=torch.float32, device=device
            ),
        )
    if not math.isclose(
        lambda_max,
        float(encoded_lambda_max[0].cpu()),
        rel_tol=3e-5,
        abs_tol=3e-7,
    ):
        raise RuntimeError("manual/model lambda_max mismatch")
    kalbi_beta_model = (
        decoded["beta_q"][0].float().cpu().numpy().astype(np.float64)
    )
    selected_beta_model = (
        selected_decoded["beta_q"][0, 0]
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    kalbi_beta = kalbi_beta_model[:, : x_train.shape[1]]
    selected_beta = selected_beta_model[: x_train.shape[1]]
    dummy_maximum_abs_beta = (
        float(
            max(
                np.max(
                    np.abs(kalbi_beta_model[:, x_train.shape[1] :]),
                    initial=0.0,
                ),
                np.max(
                    np.abs(selected_beta_model[x_train.shape[1] :]),
                    initial=0.0,
                ),
            )
        )
        if model_dummy_count
        else 0.0
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
    gamma_error = float(
        np.max(
            np.abs(
                gamma_path / np.maximum(eta_path, 1e-30)
                - raw_lambdas[:, None]
            )
        )
    )
    if eta_path.shape != (raw_lambdas.size, REFINEMENT_SWEEPS):
        raise RuntimeError(f"unexpected eta path shape {eta_path.shape}")
    if gamma_error > 2e-5:
        raise RuntimeError("gamma/eta raw-lambda contract failed")

    # No test y is accessed until filtering, CV selection, KALBI selection,
    # and every coefficient path have been frozen.
    x_test = apply_x_preprocessing(
        select_rows(x_store, test_index), scaler
    )[:, retained_local]
    y_test_raw = np.asarray(y_store[test_index], dtype=np.float64)
    zero = zero_predictor_metrics(y_test_raw, float(scaler["y_mean"]))
    denominator = max(float(zero["mse"]), 1e-30)
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
    lasso_test = raw_prediction_metrics(
        x_test,
        y_test_raw,
        lasso_beta,
        y_mean=float(scaler["y_mean"]),
        y_scale=float(scaler["y_scale"]),
    )
    cv_lasso_test = raw_prediction_metrics(
        x_test,
        y_test_raw,
        cv_lasso_beta,
        y_mean=float(scaler["y_mean"]),
        y_scale=float(scaler["y_scale"]),
    )
    kalbi_normalized = np.asarray(kalbi_test["mse"]) / denominator
    lasso_normalized = np.asarray(lasso_test["mse"]) / denominator
    selected_normalized = float(selected_test["mse"][0] / denominator)
    cv_lasso_normalized = float(cv_lasso_test["mse"][0] / denominator)

    top_local, top_original = deterministic_top_features(
        kalbi_beta,
        lasso_beta,
        retained_original,
        args.top_features,
    )
    path_output = output_root / "paths" / f"{task_id}.npz"
    atomic_npz(
        path_output,
        raw_lambdas=raw_lambdas,
        train_indices=train_index,
        test_indices=test_index,
        active_original_indices=retained_original,
        retained_local_indices=retained_local,
        kalbi_beta=kalbi_beta.astype(np.float32),
        lasso_beta=lasso_beta.astype(np.float32),
        selected_beta=selected_beta.astype(np.float32),
        cv_lasso_beta=cv_lasso_beta.astype(np.float32),
        kalbi_test_mse=np.asarray(kalbi_test["mse"], dtype=np.float64),
        lasso_test_mse=np.asarray(lasso_test["mse"], dtype=np.float64),
        kalbi_normalized_test_mse=kalbi_normalized,
        lasso_normalized_test_mse=lasso_normalized,
        kalbi_nnz=np.count_nonzero(np.abs(kalbi_beta) > 1e-8, axis=1),
        lasso_nnz=np.count_nonzero(np.abs(lasso_beta) > 1e-8, axis=1),
        lasso_objective=lasso_objective,
        lasso_data_term=lasso_data_term,
        lasso_kkt=lasso_kkt,
        selected_raw_lambda=np.asarray(selected_raw_lambda),
        selected_normalized_test_mse=np.asarray(selected_normalized),
        cv_lasso_raw_lambda=np.asarray(cv_lasso_raw_lambda),
        cv_lasso_selected_ratio=np.asarray(cv["selected_ratio"]),
        cv_lasso_normalized_test_mse=np.asarray(cv_lasso_normalized),
        cv_lasso_fold_mse=np.asarray(cv["fold_mse"]),
        cv_lasso_mean_mse=np.asarray(cv["mean_mse"]),
        cv_lasso_standard_error=np.asarray(cv["standard_error"]),
        cv_lasso_ratios=np.asarray(cv_ratios),
        penalty_grid=selector.penalty_parameter_head.penalty_grid.detach()
        .cpu()
        .numpy(),
        penalty_rho=selected["penalty_rho"]
        .reshape(-1)
        .detach()
        .cpu()
        .numpy(),
        penalty_drho=selected["penalty_drho"]
        .reshape(-1)
        .detach()
        .cpu()
        .numpy(),
        v8_t100_eta_path=eta_path.astype(np.float32),
        v8_t100_gamma_path=gamma_path.astype(np.float32),
        top_features=top_local,
        top_original_features=top_original,
    )
    kalbi_oracle = int(np.argmin(kalbi_normalized))
    lasso_oracle = int(np.argmin(lasso_normalized))
    metric = {
        "status": "PASS",
        "registry_index": int(task["registry_index"]),
        "task_id": task_id,
        "source_family": task_family(task["source"], task_id),
        "N": int(task["N"]),
        "outer_train_rows": int(train_index.size),
        "outer_test_rows": int(test_index.size),
        "P_before_constant_filter": int(np.asarray(scaler["active"]).size),
        "P_after_constant_filter": int(x_train_all.shape[1]),
        "P_after_correlation_filter": int(x_train.shape[1]),
        "P_kalbi_model_input": int(model_feature_count),
        "kalbi_exact_zero_dummy_feature_count": int(model_dummy_count),
        "kalbi_dummy_maximum_abs_beta": dummy_maximum_abs_beta,
        "filter": filter_metric,
        "scaler_sha256": scaler["scaler_sha256"],
        "split_sha256": task["outer_split_sha256"],
        "refinement_sweeps": REFINEMENT_SWEEPS,
        "raw_lambda_count": int(raw_lambdas.size),
        "raw_lambda_min": float(raw_lambdas[0]),
        "raw_lambda_max": float(raw_lambdas[-1]),
        "lambda_max": lambda_max,
        "selected_raw_lambda": selected_raw_lambda,
        "selected_normalized_test_mse": selected_normalized,
        "kalbi_path_oracle_raw_lambda": float(raw_lambdas[kalbi_oracle]),
        "kalbi_path_oracle_normalized_test_mse": float(
            kalbi_normalized[kalbi_oracle]
        ),
        "kalbi_maximum_abs_beta": float(np.max(np.abs(kalbi_beta))),
        "lasso_path_oracle_raw_lambda": float(raw_lambdas[lasso_oracle]),
        "lasso_path_oracle_normalized_test_mse": float(
            lasso_normalized[lasso_oracle]
        ),
        "lasso_maximum_abs_beta": float(np.max(np.abs(lasso_beta))),
        "lasso_maximum_kkt": float(np.max(lasso_kkt)),
        "lasso_convergence_warning_count": int(
            np.asarray(lasso_diagnostic["convergence_warnings"]).sum()
        ),
        "cv_lasso_selected_ratio": float(cv["selected_ratio"]),
        "cv_lasso_selected_raw_lambda": cv_lasso_raw_lambda,
        "cv_lasso_normalized_test_mse": cv_lasso_normalized,
        "cv_lasso_maximum_abs_beta": float(np.max(np.abs(cv_lasso_beta))),
        "cv_lasso_fold_diagnostics": cv["fold_diagnostics"],
        "cv_lasso_outer_convergence_warning_count": int(
            np.asarray(
                outer_cv_diagnostic["convergence_warnings"]
            ).sum()
        ),
        "zero_predictor_test_mse": float(zero["mse"]),
        "gamma_over_eta_max_absolute_error": gamma_error,
        "test_y_accessed_after_all_method_freeze": True,
        "path_output": str(path_output),
        "path_sha256": sha256(path_output),
        "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(metric_path, metric)
    return metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-indices", default="all")
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--filter-block-size", type=int, default=256)
    parser.add_argument("--raw-lambda-min", type=float, default=1e-4)
    parser.add_argument("--raw-lambda-max", type=float, default=1.0)
    parser.add_argument("--raw-lambda-count", type=int, default=41)
    parser.add_argument("--cv-lambda-count", type=int, default=100)
    parser.add_argument("--cv-workers", type=int, default=5)
    parser.add_argument("--lasso-tolerance", type=float, default=1e-7)
    parser.add_argument("--lasso-max-iterations", type=int, default=50000)
    parser.add_argument("--cv-lasso-tolerance", type=float, default=1e-5)
    parser.add_argument("--cv-lasso-max-iterations", type=int, default=10000)
    parser.add_argument("--top-features", type=int, default=10)
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
                args=args,
            )
            results.append(metric)
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "registry_index": metric["registry_index"],
                        "task_id": metric["task_id"],
                        "P_before": metric["P_after_constant_filter"],
                        "P_after": metric["P_after_correlation_filter"],
                        "kalbi": metric["selected_normalized_test_mse"],
                        "cv_lasso": metric["cv_lasso_normalized_test_mse"],
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
        "filter_contract": FILTER_CONTRACT,
        "correlation_threshold": CORRELATION_THRESHOLD,
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
        "scad_mcp_reused": False,
        "test_y_access_policy": "after filter/CV/selector/all paths freeze",
    }
    tag = "_".join(map(str, indices))
    atomic_json(args.output_root / "reports" / f"worker_{tag}.json", report)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
