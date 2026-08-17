#!/usr/bin/env python3
"""Shared contracts for the real P>N one-split evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import sparse

CORRELATION_THRESHOLD = 0.95
CORRELATION_BLOCK_SIZE = 256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_task_indices(value: str, task_count: int) -> list[int]:
    if value.strip().lower() in {"all", "*"}:
        return list(range(1, task_count + 1))
    selected: set[int] = set()
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            selected.update(range(int(left), int(right) + 1))
        else:
            selected.add(int(token))
    result = sorted(selected)
    if not result or result[0] < 1 or result[-1] > task_count:
        raise ValueError(f"invalid task indices: {value}")
    return result


def load_split_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or manifest.get("task_count") != 41:
        raise RuntimeError("invalid split manifest")
    tasks = manifest.get("tasks", [])
    if len(tasks) != 41:
        raise RuntimeError("split manifest task-count mismatch")
    return manifest


def load_x(path: Path) -> np.ndarray | sparse.spmatrix:
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    if path.suffix == ".npz":
        return sparse.load_npz(path)
    raise ValueError(f"unsupported X storage: {path}")


def select_rows(
    x: np.ndarray | sparse.spmatrix, indices: np.ndarray
) -> np.ndarray:
    selected = x[indices]
    if sparse.issparse(selected):
        selected = selected.toarray()
    result = np.asarray(selected, dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise RuntimeError("selected X rows are invalid")
    return result


def _correlation_block(
    x: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    if left.size == 0 or right.size == 0:
        return np.empty((left.size, right.size), dtype=np.float64)
    return (
        np.asarray(x[:, left], dtype=np.float64).T
        @ np.asarray(x[:, right], dtype=np.float64)
        / float(x.shape[0])
    )


def verify_retained_correlations(
    x_standardized: np.ndarray,
    retained: np.ndarray,
    *,
    block_size: int = CORRELATION_BLOCK_SIZE,
) -> tuple[float, tuple[int, int]]:
    x = np.asarray(x_standardized, dtype=np.float64)
    keep = np.asarray(retained, dtype=np.int64)
    if keep.ndim != 1 or keep.size == 0:
        raise RuntimeError("correlation filter retained no features")
    if keep.size == 1:
        return 0.0, (-1, -1)

    maximum = -np.inf
    maximum_pair = (-1, -1)
    for start in range(0, keep.size, int(block_size)):
        stop = min(start + int(block_size), keep.size)
        left_position = np.arange(start, stop, dtype=np.int64)
        right_position = np.arange(keep.size, dtype=np.int64)
        block = np.abs(
            _correlation_block(
                x,
                keep[left_position],
                keep[right_position],
            )
        )
        block[
            right_position[None, :] <= left_position[:, None]
        ] = -np.inf
        if not np.isfinite(block).any():
            continue
        flat_index = int(np.argmax(block))
        value = float(block.reshape(-1)[flat_index])
        if value > maximum:
            row, column = np.unravel_index(flat_index, block.shape)
            maximum = value
            maximum_pair = (
                int(keep[left_position[row]]),
                int(keep[right_position[column]]),
            )
    if not np.isfinite(maximum):
        return 0.0, (-1, -1)
    return maximum, maximum_pair


def deterministic_correlation_filter(
    x_standardized: np.ndarray,
    *,
    threshold: float = CORRELATION_THRESHOLD,
    block_size: int = CORRELATION_BLOCK_SIZE,
) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(x_standardized, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise RuntimeError("invalid standardized matrix for correlation filtering")
    if not np.isfinite(x).all():
        raise RuntimeError("non-finite standardized matrix")
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("correlation threshold must lie in (0, 1)")

    kept: list[int] = []
    p = int(x.shape[1])
    for start in range(0, p, int(block_size)):
        stop = min(start + int(block_size), p)
        block_indices = np.arange(start, stop, dtype=np.int64)
        if kept:
            previous = np.asarray(kept, dtype=np.int64)
            cross = np.abs(_correlation_block(x, block_indices, previous))
            eligible = np.max(cross, axis=1) < float(threshold)
            candidates = block_indices[eligible]
        else:
            candidates = block_indices

        selected_in_block: list[int] = []
        for candidate in candidates:
            if selected_in_block:
                local = np.asarray(selected_in_block, dtype=np.int64)
                local_correlation = np.abs(
                    _correlation_block(
                        x,
                        np.asarray([candidate], dtype=np.int64),
                        local,
                    )
                )
                if float(np.max(local_correlation)) >= float(threshold):
                    continue
            selected_in_block.append(int(candidate))
        kept.extend(selected_in_block)

    retained = np.asarray(kept, dtype=np.int64)
    maximum, pair = verify_retained_correlations(
        x,
        retained,
        block_size=block_size,
    )
    if maximum >= float(threshold) + 1e-10:
        raise RuntimeError(
            "retained feature correlation exceeds threshold: "
            f"{maximum:.12g} for pair {pair}"
        )
    if retained.size > 1 and maximum >= float(threshold):
        raise RuntimeError(
            "retained feature correlation reaches threshold within tolerance: "
            f"{maximum:.12g} for pair {pair}"
        )
    return retained, {
        "correlation_threshold": float(threshold),
        "correlation_block_size": int(block_size),
        "retained_feature_count": int(retained.size),
        "removed_correlated_feature_count": int(p - retained.size),
        "retained_max_abs_correlation": float(maximum),
        "retained_max_abs_correlation_pair_local": np.asarray(
            pair, dtype=np.int64
        ),
    }


def fit_preprocessing(
    x_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | float | str]]:
    x_train = np.asarray(x_train_raw, dtype=np.float64)
    y_train = np.asarray(y_train_raw, dtype=np.float64).reshape(-1)
    if x_train.shape[0] != y_train.size:
        raise RuntimeError("training X/y row mismatch")
    if not np.isfinite(x_train).all() or not np.isfinite(y_train).all():
        raise RuntimeError("non-finite training data")

    x_mean = x_train.mean(axis=0)
    x_scale = x_train.std(axis=0, ddof=0)
    active = np.isfinite(x_scale) & (x_scale > 1e-12)
    if not np.any(active):
        raise RuntimeError("all features are train-constant")
    x_active_mean = x_mean[active]
    x_active_scale = x_scale[active]
    x_standardized = (
        x_train[:, active] - x_active_mean[None, :]
    ) / x_active_scale[None, :]
    correlation_keep, correlation_audit = deterministic_correlation_filter(
        x_standardized
    )
    x_filtered = x_standardized[:, correlation_keep]
    active_original_indices = np.flatnonzero(active).astype(np.int64)
    retained_original_indices = active_original_indices[correlation_keep]

    y_mean = float(y_train.mean())
    y_scale = float(y_train.std(ddof=0))
    if not np.isfinite(y_scale) or y_scale <= 1e-12:
        raise RuntimeError("training response is degenerate")
    y_standardized = (y_train - y_mean) / y_scale
    if (
        not np.isfinite(x_filtered).all()
        or not np.isfinite(y_standardized).all()
    ):
        raise RuntimeError("standardization produced non-finite values")

    scaler_hash = hashlib.sha256()
    for value in (
        x_mean.astype(np.float64),
        x_scale.astype(np.float64),
        active.astype(np.uint8),
        correlation_keep.astype(np.int64),
        retained_original_indices.astype(np.int64),
        np.asarray(
            [
                correlation_audit["correlation_threshold"],
                correlation_audit["retained_max_abs_correlation"],
            ],
            dtype=np.float64,
        ),
        np.asarray([y_mean, y_scale], dtype=np.float64),
    ):
        scaler_hash.update(np.ascontiguousarray(value).tobytes())
    scaler = {
        "x_mean": x_mean,
        "x_scale": x_scale,
        "active": active,
        "correlation_keep": correlation_keep,
        "retained_original_indices": retained_original_indices,
        **correlation_audit,
        "y_mean": y_mean,
        "y_scale": y_scale,
        "scaler_sha256": scaler_hash.hexdigest(),
    }
    return (
        x_filtered.astype(np.float32),
        y_standardized.astype(np.float32),
        scaler,
    )


def apply_x_preprocessing(
    x_raw: np.ndarray, scaler: dict[str, Any]
) -> np.ndarray:
    x = np.asarray(x_raw, dtype=np.float64)
    active = np.asarray(scaler["active"], dtype=bool)
    x_mean = np.asarray(scaler["x_mean"], dtype=np.float64)[active]
    x_scale = np.asarray(scaler["x_scale"], dtype=np.float64)[active]
    transformed = (x[:, active] - x_mean[None, :]) / x_scale[None, :]
    transformed = transformed[
        :, np.asarray(scaler["correlation_keep"], dtype=np.int64)
    ]
    if not np.isfinite(transformed).all():
        raise RuntimeError("test feature transformation produced non-finite values")
    return transformed.astype(np.float32)


def save_preprocessing(
    evaluation_root: Path,
    task: dict[str, Any],
    scaler: dict[str, Any],
) -> tuple[Path, str]:
    path = evaluation_root / "preprocessing" / f"{task['task_id']}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["x_mean"], scaler["x_mean"])
            np.testing.assert_array_equal(archive["x_scale"], scaler["x_scale"])
            np.testing.assert_array_equal(archive["active"], scaler["active"])
            np.testing.assert_array_equal(
                archive["correlation_keep"], scaler["correlation_keep"]
            )
            np.testing.assert_array_equal(
                archive["retained_original_indices"],
                scaler["retained_original_indices"],
            )
            np.testing.assert_array_equal(
                archive["y_mean"], np.asarray(scaler["y_mean"])
            )
            np.testing.assert_array_equal(
                archive["y_scale"], np.asarray(scaler["y_scale"])
            )
            if str(archive["scaler_sha256"]) != scaler["scaler_sha256"]:
                # The retained maximum correlation is diagnostic only. Its
                # final ulp can vary with BLAS reduction order even when every
                # transformation-defining array above is bitwise identical.
                np.testing.assert_array_equal(
                    archive["correlation_threshold"],
                    np.asarray(scaler["correlation_threshold"]),
                )
                np.testing.assert_allclose(
                    archive["retained_max_abs_correlation"],
                    np.asarray(scaler["retained_max_abs_correlation"]),
                    rtol=1e-14,
                    atol=1e-15,
                )
    else:
        temporary = path.with_suffix(f".npz.tmp.{os.getpid()}")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                x_mean=np.asarray(scaler["x_mean"], dtype=np.float64),
                x_scale=np.asarray(scaler["x_scale"], dtype=np.float64),
                active=np.asarray(scaler["active"], dtype=bool),
                correlation_keep=np.asarray(
                    scaler["correlation_keep"], dtype=np.int64
                ),
                retained_original_indices=np.asarray(
                    scaler["retained_original_indices"], dtype=np.int64
                ),
                correlation_threshold=np.asarray(
                    scaler["correlation_threshold"], dtype=np.float64
                ),
                retained_max_abs_correlation=np.asarray(
                    scaler["retained_max_abs_correlation"],
                    dtype=np.float64,
                ),
                retained_max_abs_correlation_pair_local=np.asarray(
                    scaler["retained_max_abs_correlation_pair_local"],
                    dtype=np.int64,
                ),
                y_mean=np.asarray(scaler["y_mean"], dtype=np.float64),
                y_scale=np.asarray(scaler["y_scale"], dtype=np.float64),
                scaler_sha256=np.asarray(scaler["scaler_sha256"]),
            )
        if path.exists():
            temporary.unlink()
            return save_preprocessing(evaluation_root, task, scaler)
        temporary.replace(path)
    return path, sha256(path)


def raw_prediction_metrics(
    x_standardized: np.ndarray,
    y_raw: np.ndarray,
    beta: np.ndarray,
    *,
    y_mean: float,
    y_scale: float,
) -> dict[str, np.ndarray]:
    coefficient = np.asarray(beta, dtype=np.float64)
    if coefficient.ndim == 1:
        coefficient = coefficient[None, :]
    prediction_standardized = coefficient @ np.asarray(
        x_standardized, dtype=np.float64
    ).T
    prediction = y_mean + y_scale * prediction_standardized
    target = np.asarray(y_raw, dtype=np.float64).reshape(-1)
    residual = prediction - target[None, :]
    mse = np.mean(residual**2, axis=1)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residual), axis=1)
    denominator = float(np.sum((target - target.mean()) ** 2))
    r2 = (
        np.full(mse.shape, np.nan)
        if denominator <= 1e-15
        else 1.0 - np.sum(residual**2, axis=1) / denominator
    )
    expected = np.asarray(
        [np.mean((row - target) ** 2) for row in prediction],
        dtype=np.float64,
    )
    np.testing.assert_allclose(mse, expected, rtol=1e-12, atol=1e-14)
    return {
        "prediction": prediction,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def zero_predictor_metrics(
    y_test: np.ndarray, training_mean: float
) -> dict[str, float]:
    target = np.asarray(y_test, dtype=np.float64).reshape(-1)
    residual = target - float(training_mean)
    mse = float(np.mean(residual**2))
    denominator = float(np.sum((target - target.mean()) ** 2))
    r2 = (
        float("nan")
        if denominator <= 1e-15
        else float(1.0 - np.sum(residual**2) / denominator)
    )
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(residual))),
        "r2": r2,
    }


def task_family(source: str, task_id: str) -> str:
    source_lower = source.lower()
    if "ramanbench" in source_lower:
        if "fuel_benchtop" in task_id:
            return "RamanBench Fuel Benchtop"
        if "fuel_handheld" in task_id:
            return "RamanBench Fuel Handheld"
        if "ecoli_metabolites_dig4bio" in task_id:
            return "RamanBench E. coli DIG4BIO"
        if "ht_raman_bio_catalysis" in task_id:
            return "RamanBench HT Bio-catalysis"
        return "RamanBench other"
    if "greenhyperspectra" in source_lower:
        return "GreenHyperSpectra"
    if "moleculenet" in source_lower:
        return "MoleculeNet fingerprints"
    if "ramanspectrabioprocess" in source_lower:
        return "Standalone bioprocess Raman"
    return source


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def iter_selected_tasks(
    manifest: dict[str, Any], indices: Iterable[int]
) -> Iterable[dict[str, Any]]:
    lookup = {int(task["registry_index"]): task for task in manifest["tasks"]}
    for index in indices:
        if index not in lookup:
            raise RuntimeError(f"missing registry index {index}")
        yield lookup[index]
