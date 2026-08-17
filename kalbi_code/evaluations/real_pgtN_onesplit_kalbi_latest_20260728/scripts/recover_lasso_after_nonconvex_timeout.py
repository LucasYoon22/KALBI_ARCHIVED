#!/usr/bin/env python3
"""Recover frozen CV-LASSO choices after SCAD/MCP numerical timeouts."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from evaluate_real_pgtN_cv_baselines import fit_outer_training
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


WORKER_BY_INDEX = {13: 12, 14: 13, 15: 14, 18: 17}


def frozen_lasso_choice(log_path: Path, task_id: str) -> tuple[float, dict]:
    matches = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            event.get("task_id") == task_id
            and event.get("status") == "METHOD_PASS"
            and event.get("method") == "lasso"
        ):
            matches.append(event)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one frozen LASSO choice for {task_id}, "
            f"found {len(matches)}"
        )
    return float(matches[0]["selected_ratio"]), matches[0]


def recover_task(
    task: dict,
    evaluation_root: Path,
    ratios: np.ndarray,
) -> dict:
    started = time.perf_counter()
    index = int(task["registry_index"])
    task_id = task["task_id"]
    worker = WORKER_BY_INDEX[index]
    archived_log = (
        evaluation_root
        / "reports"
        / "nonconvex_timeout_archive"
        / "main_logs"
        / f"baseline_worker_{worker:02d}.log"
    )
    selected_ratio, frozen_event = frozen_lasso_choice(
        archived_log, task_id
    )
    selected_index = int(np.argmin(np.abs(ratios - selected_ratio)))
    if not math.isclose(
        float(ratios[selected_index]),
        selected_ratio,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError(f"selected ratio is not on the fixed grid: {task_id}")

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
    x_train_raw = select_rows(x_store, train_index)
    y_train_raw = np.asarray(y_store[train_index], dtype=np.float64)
    x_train, y_train, scaler = fit_preprocessing(
        x_train_raw, y_train_raw
    )
    preprocessing_path, preprocessing_sha = save_preprocessing(
        evaluation_root, task, scaler
    )

    fit_args = SimpleNamespace(
        lasso_tolerance=1e-5,
        lasso_max_iter=10000,
        scad_gamma=3.7,
        mcp_gamma=3.0,
        nonconvex_tolerance=1e-5,
        nonconvex_max_iter=100,
        nonconvex_max_epochs=10000,
    )
    refit = fit_outer_training(
        "lasso",
        x_train,
        y_train,
        ratios,
        selected_index,
        fit_args,
    )
    method_freeze_time = time.time()

    x_test_raw = select_rows(x_store, test_index)
    x_test = apply_x_preprocessing(x_test_raw, scaler)
    y_test = np.asarray(y_store[test_index], dtype=np.float64)
    zero = zero_predictor_metrics(y_test, float(scaler["y_mean"]))
    test = raw_prediction_metrics(
        x_test,
        y_test,
        refit["beta"],
        y_mean=float(scaler["y_mean"]),
        y_scale=float(scaler["y_scale"]),
    )
    lasso = {
        "status": "PASS_RECOVERED_FROM_FROZEN_CV_CHOICE",
        "selected_index": selected_index,
        "selected_ratio": selected_ratio,
        "selected_raw_lambda": float(refit["selected_raw_lambda"]),
        "outer_train_lambda_max": float(refit["lambda_max"]),
        "train_objective": float(refit["objective"]),
        "train_kkt_max": float(refit["kkt_max"]),
        "cv_folds": 5,
        "cv_selection_source": str(archived_log),
        "cv_selection_source_sha256": sha256(archived_log),
        "frozen_cv_event": frozen_event,
        "refit_diagnostics": refit["diagnostics"],
        "test_mse": float(test["mse"][0]),
        "test_rmse": float(test["rmse"][0]),
        "test_mae": float(test["mae"][0]),
        "test_r2": float(test["r2"][0]),
        "normalized_test_mse": float(
            test["mse"][0] / max(zero["mse"], 1e-30)
        ),
        "nnz_1e8": int(np.count_nonzero(np.abs(refit["beta"]) > 1e-8)),
    }
    timeout = {
        "status": "TIMEOUT",
        "included_in_aggregate": False,
        "solver": "skglm.AndersonCD",
        "grid_size": int(ratios.size),
        "tolerance": 1e-5,
        "max_iter": 100,
        "max_epochs": 10000,
        "strict_worker_wall_time_lower_bound_seconds": 7200,
        "reason": "full 100-point five-fold path did not complete",
    }
    path = evaluation_root / "baselines" / f"{task_id}.npz"
    atomic_npz(
        path,
        lambda_ratios=ratios,
        lasso_beta=refit["beta"].astype(np.float32),
        lasso_test_prediction=test["prediction"][0].astype(np.float32),
        train_indices=train_index,
        test_indices=test_index,
    )
    result = {
        "status": "PARTIAL_NONCONVEX_TIMEOUT",
        "registry_index": index,
        "task_id": task_id,
        "source": task["source"],
        "source_family": task_family(task["source"], task_id),
        "target": task["target"],
        "evidence_tier": task["evidence_tier"],
        "N": int(task["N"]),
        "P_raw": int(task["P_raw"]),
        "P_transformed": int(np.asarray(scaler["active"]).sum()),
        "outer_train_rows": int(train_index.size),
        "outer_test_rows": int(test_index.size),
        "outer_split_seed": int(task["outer_split_seed"]),
        "outer_split_sha256": task["outer_split_sha256"],
        "preprocessing_path": str(preprocessing_path),
        "preprocessing_sha256": preprocessing_sha,
        "scaler_sha256": scaler["scaler_sha256"],
        "test_y_accessed_after_method_freeze": True,
        "method_freeze_unix": method_freeze_time,
        "zero_predictor": zero,
        "methods": {
            "lasso": lasso,
            "scad": timeout,
            "mcp": timeout,
        },
        "lambda_ratio_grid": ratios.tolist(),
        "baseline_path": str(path),
        "baseline_path_sha256": sha256(path),
        "runtime_seconds": time.perf_counter() - started,
        "nonconvex_timeout_archive": str(
            evaluation_root / "reports" / "nonconvex_timeout_archive"
        ),
    }
    atomic_json(
        evaluation_root / "metrics" / f"{task_id}_baselines.json",
        result,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--task-indices", required=True)
    args = parser.parse_args()
    manifest = load_split_manifest(args.split_manifest)
    indices = parse_task_indices(
        args.task_indices, int(manifest["task_count"])
    )
    ratios = np.logspace(0.0, -4.0, 100, dtype=np.float64)
    for task in iter_selected_tasks(manifest, indices):
        if int(task["registry_index"]) not in WORKER_BY_INDEX:
            raise RuntimeError(f"task is not a timeout task: {task['task_id']}")
        result = recover_task(task, args.evaluation_root, ratios)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "task_id": result["task_id"],
                    "lasso_normalized_test_mse": result["methods"]["lasso"][
                        "normalized_test_mse"
                    ],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
