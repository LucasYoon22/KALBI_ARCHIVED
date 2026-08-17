#!/usr/bin/env python3
"""Shared strict-loading and numerical helpers for Kalbi V2 reports."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .model import KALBI_RIDGE, KALBI_STAGES, KALBI_TOPK, build_model
from .selector_v17_k32s32 import DirectPenaltyMLPQuerySelectorV17StageGain
from .train import standardize_tasks
from train_fpk import _build_X_y_tensors, _load_yaml


@dataclass(frozen=True)
class QuadraticSegment:
    lo: float
    hi: float
    c: float
    d: float
    h: float


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_components(
    checkpoint_path: Path,
    config_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], torch.nn.Module, torch.nn.Module]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    metadata = checkpoint.get("metadata", {})
    exact = metadata.get("exact_training_arguments", {})
    expected = {
        "kalbi_isis_topk": KALBI_TOPK,
        "kalbi_isis_stages": KALBI_STAGES,
        "kalbi_isis_ridge": KALBI_RIDGE,
    }
    for key, value in expected.items():
        if exact.get(key) != value:
            raise RuntimeError(f"checkpoint architecture mismatch for {key}")

    model = build_model(_load_yaml(str(config_path)), device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    selector = DirectPenaltyMLPQuerySelectorV17StageGain(
        hidden=512,
        lambda_min_ratio=1e-6,
        lambda_max_ratio=1.0,
        initial_lambda_ratio=0.1,
        derivative_floor=1e-6,
        squareplus_scale=0.02,
        direct_head_init_std=1e-3,
    )
    selector.load_state_dict(checkpoint["selector_state_dict"], strict=True)
    selector.to(device).eval()
    return checkpoint, model, selector


def select_balanced_entries(
    catalog: Mapping[str, Any], count: int
) -> list[dict[str, Any]]:
    if count <= 0 or count % 2:
        raise ValueError("count must be a positive even number")
    selected: dict[str, list[dict[str, Any]]] = {}
    for prior in ("tabpfn_v1", "tabicl_v2"):
        entries = [
            dict(entry) for entry in catalog["files"]
            if entry["prior"] == prior
        ]
        entries.sort(
            key=lambda entry: (int(entry["shard_id"]), entry["relative_path"])
        )
        if len(entries) < count // 2:
            raise RuntimeError(f"not enough {prior} entries")
        positions = np.linspace(0, len(entries) - 1, count // 2)
        selected[prior] = [entries[int(round(position))] for position in positions]
    result: list[dict[str, Any]] = []
    for index in range(count // 2):
        result.extend((selected["tabpfn_v1"][index], selected["tabicl_v2"][index]))
    return result


def select_smallest_p_regime_entries(
    catalog: Mapping[str, Any], count: int
) -> list[dict[str, Any]]:
    """Select tractable full-shape tasks across both priors and N/P regimes."""
    if count <= 0 or count % 4:
        raise ValueError("count must be a positive multiple of four")
    per_group = count // 4
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for regime, predicate in (
        ("n_gt_p", lambda entry: int(entry["n"]) > int(entry["p"])),
        ("p_gt_n", lambda entry: int(entry["p"]) > int(entry["n"])),
    ):
        for prior in ("tabpfn_v1", "tabicl_v2"):
            entries = [
                dict(entry)
                for entry in catalog["files"]
                if entry["prior"] == prior and predicate(entry)
            ]
            entries.sort(
                key=lambda entry: (
                    int(entry["p"]),
                    int(entry["n"]),
                    int(entry["shard_id"]),
                    entry["relative_path"],
                )
            )
            if len(entries) < per_group:
                raise RuntimeError(f"not enough {prior}/{regime} entries")
            groups[(regime, prior)] = entries[:per_group]
    result: list[dict[str, Any]] = []
    for index in range(per_group):
        for regime in ("n_gt_p", "p_gt_n"):
            for prior in ("tabpfn_v1", "tabicl_v2"):
                result.append(groups[(regime, prior)][index])
    return result


def load_task(
    data_root: Path, entry: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, float]:
    path = (data_root / str(entry["relative_path"])).resolve()
    if data_root not in path.parents or not path.is_file():
        raise RuntimeError(f"missing or outside task file: {path}")
    if file_sha256(path) != str(entry["sha256"]):
        raise RuntimeError(f"task SHA256 mismatch: {path}")
    with np.load(path, allow_pickle=False) as payload:
        X = np.asarray(payload["X"])
        y = np.asarray(payload["y"])
        if "exact_max_abs_corr" in payload:
            max_corr = float(np.asarray(payload["exact_max_abs_corr"]).reshape(-1)[0])
        else:
            max_corr = float(entry.get("task_max_abs_corr_max", np.nan))
    if X.ndim == 3 and X.shape[0] == 1:
        X = X[0]
    if y.ndim == 2 and y.shape[0] == 1:
        y = y[0]
    y = y.reshape(-1)
    if X.shape != (int(entry["n"]), int(entry["p"])) or y.shape != (X.shape[0],):
        raise RuntimeError(f"task shape mismatch: X={X.shape}, y={y.shape}")
    return X, y, max_corr


def standardize_full(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = standardize_tasks([X], [y])
    return np.asarray(xs[0], dtype=np.float32), np.asarray(ys[0], dtype=np.float32)


def split_standardize(
    X: np.ndarray, y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    rows = int(X.shape[0])
    order = rng.permutation(rows)
    train_index = order[: rows // 2]
    test_index = order[rows // 2 :]
    x_train_raw = np.asarray(X[train_index], dtype=np.float32)
    x_test_raw = np.asarray(X[test_index], dtype=np.float32)
    y_train_raw = np.asarray(y[train_index], dtype=np.float32)
    y_test_raw = np.asarray(y[test_index], dtype=np.float32)
    x_mean = x_train_raw.mean(0, keepdims=True)
    x_std = x_train_raw.std(0, keepdims=True)
    active = x_std > 1e-6
    x_train = np.where(active, (x_train_raw - x_mean) / np.maximum(x_std, 1e-8), 0.0)
    x_test = np.where(active, (x_test_raw - x_mean) / np.maximum(x_std, 1e-8), 0.0)
    y_mean = float(y_train_raw.mean())
    y_std = max(float(y_train_raw.std()), 1e-8)
    y_train = (y_train_raw - y_mean) / y_std
    y_test = (y_test_raw - y_mean) / y_std
    values = tuple(
        np.asarray(value, dtype=np.float32)
        for value in (x_train, y_train, x_test, y_test)
    )
    if not all(np.isfinite(value).all() for value in values):
        raise RuntimeError("split standardization produced nonfinite values")
    return values


def tensors_from_arrays(
    X: np.ndarray, y: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    return _build_X_y_tensors(
        [X], [y], device=device, dtype=torch.float32
    )


def encode_dataset(
    model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor
) -> dict[str, Any]:
    with torch.inference_mode():
        return model.encode_dataset(model.precompute(X, y))


def predict_from_encoded(
    model: torch.nn.Module,
    encoded: dict[str, Any],
    lambdas: torch.Tensor,
    shape: torch.Tensor,
    d0: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> torch.Tensor:
    lambdas = lambdas.reshape(-1).float()
    shape = shape.reshape(lambdas.numel(), -1).float()
    d0 = d0.reshape(-1).float()
    if shape.shape[0] != lambdas.numel() or d0.numel() != lambdas.numel():
        raise ValueError("query tensor count mismatch")
    lambda_max = encoded["lambda_max"].float().reshape(-1)[0]
    outputs: list[torch.Tensor] = []
    with torch.inference_mode(), torch.autocast(
        device_type=lambdas.device.type,
        dtype=torch.bfloat16,
        enabled=lambdas.device.type == "cuda",
    ):
        for start in range(0, lambdas.numel(), int(chunk_size)):
            stop = min(start + int(chunk_size), lambdas.numel())
            beta = model.decode_queries(
                encoded,
                lambdas=lambdas[start:stop][None],
                lambda_max=lambda_max.expand(1, stop - start),
                penalty_shape_features=shape[start:stop][None],
                penalty_d0=d0[start:stop][None],
            )
            outputs.append(beta[0].float().cpu())
    return torch.cat(outputs, dim=0)


def segments_from_knots(
    knots: Sequence[float],
    rho: Sequence[float],
    drho: Sequence[float],
    *,
    tail_hi: float = 20.0,
) -> list[QuadraticSegment]:
    knots_array = np.asarray(knots, dtype=np.float64)
    rho_array = np.asarray(rho, dtype=np.float64)
    drho_array = np.asarray(drho, dtype=np.float64)
    if not (
        knots_array.ndim == rho_array.ndim == drho_array.ndim == 1
        and len(knots_array) == len(rho_array) == len(drho_array)
    ):
        raise ValueError("canonical knot payload shape mismatch")
    result: list[QuadraticSegment] = []
    for index in range(len(knots_array) - 1):
        lo = float(knots_array[index])
        hi = float(knots_array[index + 1])
        curvature = float(
            (drho_array[index + 1] - drho_array[index]) / (hi - lo)
        )
        derivative_intercept = float(drho_array[index] - curvature * lo)
        penalty_intercept = float(
            rho_array[index]
            - derivative_intercept * lo
            - 0.5 * curvature * lo * lo
        )
        result.append(
            QuadraticSegment(
                lo=lo,
                hi=float(tail_hi) if index == len(knots_array) - 2 else hi,
                c=penalty_intercept,
                d=derivative_intercept,
                h=curvature,
            )
        )
    return result


def rho_abs(value: float, segments: Sequence[QuadraticSegment]) -> float:
    magnitude = abs(float(value))
    segment = segments[-1]
    for candidate in segments:
        if candidate.lo <= magnitude <= candidate.hi:
            segment = candidate
            break
    return float(
        segment.c
        + segment.d * magnitude
        + 0.5 * segment.h * magnitude * magnitude
    )


def objective_path(
    X: np.ndarray,
    y: np.ndarray,
    betas: np.ndarray,
    lambdas: np.ndarray,
    segments: Sequence[QuadraticSegment],
) -> np.ndarray:
    prediction = np.asarray(betas, dtype=np.float64) @ np.asarray(X, dtype=np.float64).T
    residual = np.asarray(y, dtype=np.float64)[None] - prediction
    data_half = 0.5 * np.mean(residual * residual, axis=1)
    penalty = np.asarray([
        float(lam) * sum(rho_abs(value, segments) for value in beta)
        for beta, lam in zip(betas, lambdas, strict=True)
    ])
    return data_half + penalty


def _coordinate_minimum(
    diagonal: float,
    center: float,
    lam: float,
    segments: Sequence[QuadraticSegment],
    current: float,
) -> float:
    best_x = float(current)
    best_value = (
        0.5 * diagonal * (best_x - center) ** 2
        + lam * rho_abs(best_x, segments)
    )
    zero_value = 0.5 * diagonal * center * center + lam * rho_abs(0.0, segments)
    if zero_value < best_value:
        best_x = 0.0
        best_value = zero_value
    for sign in (-1.0, 1.0):
        for segment in segments:
            candidates = [segment.lo, segment.hi]
            denominator = diagonal + lam * segment.h
            if denominator > 1e-12:
                stationary = (
                    diagonal * sign * center - lam * segment.d
                ) / denominator
                candidates.append(min(max(stationary, segment.lo), segment.hi))
            for magnitude in candidates:
                if not np.isfinite(magnitude) or magnitude < 0.0:
                    continue
                value = sign * float(magnitude)
                objective = (
                    0.5 * diagonal * (value - center) ** 2
                    + lam * rho_abs(value, segments)
                )
                if objective < best_value:
                    best_value = objective
                    best_x = value
    return 0.0 if abs(best_x) <= 1e-8 else best_x


def _coordinate_minimum_vectorized(
    diagonal: float,
    center: float,
    lam: float,
    segments: Sequence[QuadraticSegment],
    segment_arrays: tuple[np.ndarray, ...],
    current: float,
) -> float:
    """Equivalent segment search without the per-segment Python loop."""
    lo, hi, c, d, h = segment_arrays
    best_x = float(current)
    best_value = (
        0.5 * diagonal * (best_x - center) ** 2
        + lam * rho_abs(best_x, segments)
    )
    zero_value = 0.5 * diagonal * center * center + lam * rho_abs(0.0, segments)
    if zero_value < best_value:
        best_x = 0.0
        best_value = zero_value

    denominator = diagonal + lam * h
    valid_stationary = denominator > 1e-12
    for sign in (-1.0, 1.0):
        stationary = lo.copy()
        stationary[valid_stationary] = (
            diagonal * sign * center - lam * d[valid_stationary]
        ) / denominator[valid_stationary]
        stationary = np.minimum(np.maximum(stationary, lo), hi)
        magnitudes = np.stack((lo, hi, stationary), axis=1)
        values = sign * magnitudes

        # rho_abs selects the first segment containing a shared boundary.
        # searchsorted(side="left") preserves that exact convention.
        penalty_segment = np.searchsorted(hi, magnitudes, side="left")
        penalty_segment = np.clip(penalty_segment, 0, hi.size - 1)
        penalty = (
            c[penalty_segment]
            + d[penalty_segment] * magnitudes
            + 0.5 * h[penalty_segment] * magnitudes * magnitudes
        )
        objectives = (
            0.5 * diagonal * (values - center) * (values - center)
            + lam * penalty
        )
        flat_index = int(np.argmin(objectives))
        candidate_value = float(objectives.flat[flat_index])
        if candidate_value < best_value:
            best_value = candidate_value
            best_x = float(values.flat[flat_index])
    return 0.0 if abs(best_x) <= 1e-8 else best_x


def coordinate_descent_path(
    gram: np.ndarray,
    linear: np.ndarray,
    lambdas: np.ndarray,
    segments: Sequence[QuadraticSegment],
    initial: np.ndarray,
    *,
    sweeps: int,
    tolerance: float,
) -> np.ndarray:
    diagonal = np.diag(gram).astype(np.float64).clip(1e-12)
    segment_arrays = tuple(
        np.asarray([getattr(segment, field) for segment in segments], dtype=np.float64)
        for field in ("lo", "hi", "c", "d", "h")
    )
    output: list[np.ndarray] = []
    for query_index, lam in enumerate(lambdas):
        beta = np.asarray(initial[query_index], dtype=np.float64).copy()
        for _ in range(int(sweeps)):
            previous = beta.copy()
            for feature in range(beta.size):
                residual_coordinate = (
                    linear[feature]
                    - gram[feature].dot(beta)
                    + diagonal[feature] * beta[feature]
                )
                center = residual_coordinate / diagonal[feature]
                beta[feature] = _coordinate_minimum_vectorized(
                    float(diagonal[feature]),
                    float(center),
                    float(lam),
                    segments,
                    segment_arrays,
                    float(beta[feature]),
                )
            if float(np.max(np.abs(beta - previous))) < float(tolerance):
                break
        output.append(beta.copy())
    return np.asarray(output)


def vector_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x.dot(y) / denominator) if denominator > 1e-12 else 0.0


def validation_mse(
    X_test: np.ndarray, y_test: np.ndarray, betas: torch.Tensor
) -> np.ndarray:
    beta_array = betas.detach().cpu().float().numpy().astype(np.float64)
    prediction = beta_array @ np.asarray(X_test, dtype=np.float64).T
    return np.mean(
        (prediction - np.asarray(y_test, dtype=np.float64)[None]) ** 2,
        axis=1,
    )
