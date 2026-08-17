#!/usr/bin/env python3
"""Independently re-audit preprocessing for the frozen real P>N evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from real_pgtN_common import (
    apply_x_preprocessing,
    atomic_json,
    fit_preprocessing,
    load_split_manifest,
    load_x,
    raw_prediction_metrics,
    select_rows,
    sha256,
)


def scaler_hash(scaler: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for value in (
        np.asarray(scaler["x_mean"], dtype=np.float64),
        np.asarray(scaler["x_scale"], dtype=np.float64),
        np.asarray(scaler["active"], dtype=np.uint8),
        np.asarray([scaler["y_mean"], scaler["y_scale"]], dtype=np.float64),
    ):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-data-hashes", action="store_true")
    args = parser.parse_args()

    manifest = load_split_manifest(args.split_manifest)
    preprocessing_manifest = load_json(
        args.evaluation_root / "manifests" / "preprocessing_manifest.json"
    )
    preprocessing_lookup = {
        row["task_id"]: row for row in preprocessing_manifest["tasks"]
    }
    task_results: list[dict[str, Any]] = []
    failures: list[str] = []

    for task in manifest["tasks"]:
        task_id = task["task_id"]
        try:
            x_path = Path(task["X_path"])
            y_path = Path(task["y_path"])
            split_path = Path(task["outer_split_path"])
            if args.verify_data_hashes:
                for path, expected in (
                    (x_path, task["X_sha256"]),
                    (y_path, task["y_sha256"]),
                    (split_path, task["outer_split_sha256"]),
                ):
                    observed = sha256(path)
                    if observed != expected:
                        raise RuntimeError(
                            f"artifact hash mismatch: {path}: {observed}"
                        )

            with np.load(split_path, allow_pickle=False) as archive:
                train_index = np.asarray(
                    archive["train_indices"], dtype=np.int64
                )
                test_index = np.asarray(
                    archive["test_indices"], dtype=np.int64
                )
            if np.intersect1d(train_index, test_index).size:
                raise RuntimeError("outer split overlap")
            if np.union1d(train_index, test_index).size != int(task["N"]):
                raise RuntimeError("outer split does not cover every row")

            x_store = load_x(x_path)
            y_store = np.load(y_path, mmap_mode="r")
            x_train_raw = select_rows(x_store, train_index)
            y_train_raw = np.asarray(y_store[train_index], dtype=np.float64)
            x_train, y_train, scaler = fit_preprocessing(
                x_train_raw, y_train_raw
            )
            observed_scaler_hash = scaler_hash(scaler)
            if observed_scaler_hash != scaler["scaler_sha256"]:
                raise RuntimeError("recomputed scaler hash is internally invalid")

            saved_path = (
                args.evaluation_root
                / "preprocessing"
                / f"{task_id}.npz"
            )
            with np.load(saved_path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(
                    saved["x_mean"], scaler["x_mean"]
                )
                np.testing.assert_array_equal(
                    saved["x_scale"], scaler["x_scale"]
                )
                np.testing.assert_array_equal(saved["active"], scaler["active"])
                np.testing.assert_array_equal(
                    saved["y_mean"], np.asarray(scaler["y_mean"])
                )
                np.testing.assert_array_equal(
                    saved["y_scale"], np.asarray(scaler["y_scale"])
                )
                if str(saved["scaler_sha256"]) != observed_scaler_hash:
                    raise RuntimeError("saved scaler hash mismatch")

            expected_preprocessing = preprocessing_lookup[task_id]
            if (
                expected_preprocessing["scaler_sha256"]
                != observed_scaler_hash
            ):
                raise RuntimeError("preprocessing manifest scaler mismatch")

            kalbi_metric = load_json(
                args.evaluation_root
                / "metrics"
                / f"{task_id}_kalbi.json"
            )
            baseline_metric = load_json(
                args.evaluation_root
                / "metrics"
                / f"{task_id}_baselines.json"
            )
            for label, metric in (
                ("KALBI", kalbi_metric),
                ("baseline", baseline_metric),
            ):
                if metric["scaler_sha256"] != observed_scaler_hash:
                    raise RuntimeError(f"{label} scaler mismatch")

            path_path = args.evaluation_root / "paths" / f"{task_id}.npz"
            baseline_path = (
                args.evaluation_root / "baselines" / f"{task_id}.npz"
            )
            with np.load(path_path, allow_pickle=False) as path_arrays:
                np.testing.assert_array_equal(
                    path_arrays["train_indices"], train_index
                )
                np.testing.assert_array_equal(
                    path_arrays["test_indices"], test_index
                )
                kalbi_beta = np.asarray(
                    path_arrays["kalbi_beta"], dtype=np.float64
                )
                saved_kalbi_test_mse = np.asarray(
                    path_arrays["kalbi_test_mse"], dtype=np.float64
                )
            with np.load(baseline_path, allow_pickle=False) as base_arrays:
                np.testing.assert_array_equal(
                    base_arrays["train_indices"], train_index
                )
                np.testing.assert_array_equal(
                    base_arrays["test_indices"], test_index
                )

            train_mean_error = float(
                np.max(np.abs(np.asarray(x_train, dtype=np.float64).mean(0)))
            )
            train_std_error = float(
                np.max(
                    np.abs(
                        np.asarray(x_train, dtype=np.float64).std(0, ddof=0)
                        - 1.0
                    )
                )
            )
            y_train_mean_error = abs(float(np.mean(y_train)))
            y_train_std_error = abs(float(np.std(y_train, ddof=0)) - 1.0)
            if max(
                train_mean_error,
                train_std_error,
                y_train_mean_error,
                y_train_std_error,
            ) > 5e-5:
                raise RuntimeError("standardized outer-training moments failed")

            x_test_raw = select_rows(x_store, test_index)
            x_test = apply_x_preprocessing(x_test_raw, scaler)
            y_test_raw = np.asarray(y_store[test_index], dtype=np.float64)
            replay = raw_prediction_metrics(
                x_test,
                y_test_raw,
                kalbi_beta,
                y_mean=float(scaler["y_mean"]),
                y_scale=float(scaler["y_scale"]),
            )
            np.testing.assert_allclose(
                replay["mse"],
                saved_kalbi_test_mse,
                rtol=5e-6,
                atol=1e-9,
            )

            result = {
                "registry_index": int(task["registry_index"]),
                "task_id": task_id,
                "status": "PASS",
                "N": int(task["N"]),
                "P_raw": int(task["P_raw"]),
                "P_transformed": int(np.asarray(scaler["active"]).sum()),
                "removed_train_constant_features": int(
                    np.count_nonzero(~np.asarray(scaler["active"], dtype=bool))
                ),
                "train_rows": int(train_index.size),
                "test_rows": int(test_index.size),
                "split_overlap": 0,
                "scaler_sha256": observed_scaler_hash,
                "kalbi_baseline_shared_scaler": True,
                "train_feature_mean_max_abs": train_mean_error,
                "train_feature_std_max_abs_error": train_std_error,
                "train_y_mean_abs": y_train_mean_error,
                "train_y_std_abs_error": y_train_std_error,
                "kalbi_test_mse_replay_max_abs_error": float(
                    np.max(np.abs(replay["mse"] - saved_kalbi_test_mse))
                ),
                "test_transform_finite": bool(np.isfinite(x_test).all()),
                "test_rows_not_used_to_fit_scaler": True,
            }
            task_results.append(result)
            print(json.dumps(result), flush=True)
        except Exception as error:
            failure = f"{task_id}: {type(error).__name__}: {error}"
            failures.append(failure)
            print(json.dumps({"status": "FAIL", "failure": failure}), flush=True)

    summary = {
        "status": "PASS" if not failures else "FAIL",
        "task_count": len(task_results),
        "expected_task_count": int(manifest["task_count"]),
        "failure_count": len(failures),
        "failures": failures,
        "contracts": {
            "outer_split": "one deterministic row-random 80/20 split",
            "x_preprocessing": (
                "training-only per-feature centering and ddof=0 scaling"
            ),
            "constant_feature_rule": "remove only train std <= 1e-12",
            "y_preprocessing": "training-only centering and ddof=0 scaling",
            "test_prediction_inverse_transform": (
                "training y_mean + training y_scale * standardized prediction"
            ),
            "same_scaler_for_kalbi_and_lasso": True,
        },
        "aggregate": {
            "removed_train_constant_features": int(
                sum(
                    row["removed_train_constant_features"]
                    for row in task_results
                )
            ),
            "tasks_with_removed_features": int(
                sum(
                    row["removed_train_constant_features"] > 0
                    for row in task_results
                )
            ),
            "max_train_feature_mean_abs": float(
                max(
                    (
                        row["train_feature_mean_max_abs"]
                        for row in task_results
                    ),
                    default=float("nan"),
                )
            ),
            "max_train_feature_std_abs_error": float(
                max(
                    (
                        row["train_feature_std_max_abs_error"]
                        for row in task_results
                    ),
                    default=float("nan"),
                )
            ),
            "max_train_y_mean_abs": float(
                max(
                    (row["train_y_mean_abs"] for row in task_results),
                    default=float("nan"),
                )
            ),
            "max_train_y_std_abs_error": float(
                max(
                    (row["train_y_std_abs_error"] for row in task_results),
                    default=float("nan"),
                )
            ),
            "max_kalbi_test_mse_replay_abs_error": float(
                max(
                    (
                        row["kalbi_test_mse_replay_max_abs_error"]
                        for row in task_results
                    ),
                    default=float("nan"),
                )
            ),
        },
        "tasks": task_results,
    }
    atomic_json(args.output, summary)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
