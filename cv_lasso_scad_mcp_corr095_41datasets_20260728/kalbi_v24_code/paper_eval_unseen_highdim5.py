#!/usr/bin/env python3
"""Paper evaluation of KALBI on five frozen unseen high-dimensional tasks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text

from .eval_kalbi_v2_common import (
    encode_dataset,
    file_sha256,
    tensors_from_arrays,
    vector_correlation,
)
from .exact_canonical_cd import (
    CanonicalCDKernel,
    canonical_penalty_values,
    objective_components,
    solve_multistart_path,
)
from .model import MODEL_VERSION
from .report_kalbi_v6_v20_training_task import load_components_v20


EVALUATION_VERSION = "kalbi_paper_unseen_highdim5_eval_v1"
SEED_STRIDE = 1_000_003
INK = "#202124"
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
YELLOW = "#F0E442"
GREY = "#666666"
LIGHT_GREY = "#D0D0D0"
FEATURE_COLORS = (BLUE, ORANGE, GREEN, RED, PURPLE, SKY, "#8A6D3B", "#4C566A")
FORBIDDEN_VISIBLE_TEXT = (
    "kalbi v6",
    "v20",
    "beta solver",
    "lambda selector",
    "selector model",
)


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def split_standardize_three_way(
    x: np.ndarray, y: np.ndarray, seed: int
) -> dict[str, np.ndarray | float]:
    rows = int(x.shape[0])
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(rows)
    n_train = rows // 2
    remaining = rows - n_train
    n_validation = remaining // 2
    train_index = order[:n_train]
    validation_index = order[n_train:n_train + n_validation]
    test_index = order[n_train + n_validation:]
    x_train_raw = np.asarray(x[train_index], dtype=np.float32)
    x_validation_raw = np.asarray(x[validation_index], dtype=np.float32)
    x_test_raw = np.asarray(x[test_index], dtype=np.float32)
    y_train_raw = np.asarray(y[train_index], dtype=np.float32)
    y_validation_raw = np.asarray(y[validation_index], dtype=np.float32)
    y_test_raw = np.asarray(y[test_index], dtype=np.float32)
    x_mean = x_train_raw.mean(0, keepdims=True)
    x_std_raw = x_train_raw.std(0, keepdims=True)
    active = x_std_raw > 1e-6
    x_std = np.maximum(x_std_raw, 1e-8)
    y_mean = float(y_train_raw.mean())
    y_std = max(float(y_train_raw.std()), 1e-8)

    def x_transform(value: np.ndarray) -> np.ndarray:
        return np.asarray(
            np.where(active, (value - x_mean) / x_std, 0.0), dtype=np.float32
        )

    result: dict[str, np.ndarray | float] = {
        "x_train": x_transform(x_train_raw),
        "x_validation": x_transform(x_validation_raw),
        "x_test": x_transform(x_test_raw),
        "y_train": np.asarray((y_train_raw - y_mean) / y_std, dtype=np.float32),
        "y_validation": np.asarray(
            (y_validation_raw - y_mean) / y_std, dtype=np.float32
        ),
        "y_test": np.asarray((y_test_raw - y_mean) / y_std, dtype=np.float32),
        "train_index": train_index.astype(np.int64),
        "validation_index": validation_index.astype(np.int64),
        "test_index": test_index.astype(np.int64),
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "active": active.astype(np.uint8),
        "y_mean": y_mean,
        "y_std": y_std,
    }
    arrays = [value for value in result.values() if isinstance(value, np.ndarray)]
    if not all(np.isfinite(value).all() for value in arrays):
        raise RuntimeError("three-way split standardization produced nonfinite values")
    if len(set(train_index) & set(validation_index)):
        raise RuntimeError("train/validation split overlap")
    if len(set(train_index) & set(test_index)):
        raise RuntimeError("train/test split overlap")
    if len(set(validation_index) & set(test_index)):
        raise RuntimeError("validation/test split overlap")
    return result


def load_frozen_task(
    prepared_root: Path, entry: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data_path = prepared_root / str(entry["relative_data_path"])
    manifest_path = prepared_root / str(entry["relative_manifest_path"])
    if file_sha256(data_path) != str(entry["task_sha256"]):
        raise RuntimeError(f"task SHA mismatch: {data_path}")
    if file_sha256(manifest_path) != str(entry["manifest_sha256"]):
        raise RuntimeError(f"manifest SHA mismatch: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(data_path, allow_pickle=False) as payload:
        x = np.asarray(payload["X"][0], dtype=np.float32)
        y = np.asarray(payload["y"][0], dtype=np.float32)
        feature_noise_variance = np.asarray(
            payload["feature_noise_variance"][0], dtype=np.float32
        )
        core_corr = float(np.asarray(payload["core_exact_max_abs_corr"]).reshape(-1)[0])
        sampled_corr = float(
            np.asarray(payload["sampled_corr_audit_max"]).reshape(-1)[0]
        )
    if x.shape != (int(entry["n"]), int(entry["p"])) or y.shape != (x.shape[0],):
        raise RuntimeError("frozen task shape mismatch")
    return x, y, {
        "manifest": manifest,
        "data_path": str(data_path),
        "manifest_path": str(manifest_path),
        "core_exact_max_abs_corr": core_corr,
        "sampled_final_max_abs_corr": sampled_corr,
        "feature_noise_variance": feature_noise_variance,
    }


def predict_path_fp32(
    model: torch.nn.Module,
    encoded: dict[str, Any],
    lambdas: torch.Tensor,
    shape: torch.Tensor,
    d0: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    lambdas = lambdas.reshape(-1).float()
    shape = shape.reshape(lambdas.numel(), -1).float()
    d0 = d0.reshape(-1).float()
    lambda_max = encoded["lambda_max"].float().reshape(-1)[0]
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
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


def objective_paths(
    x: np.ndarray,
    y: np.ndarray,
    betas: np.ndarray,
    lambdas: np.ndarray,
    knots: np.ndarray,
    relu_b: float,
    relu_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray([
        objective_components(
            x, y, beta, raw_lambda, knots, relu_b, relu_weights
        )
        for beta, raw_lambda in zip(betas, lambdas, strict=True)
    ])
    return values[:, 0], values[:, 1], values[:, 2]


def prediction_mse(
    x: np.ndarray, y: np.ndarray, betas: np.ndarray
) -> np.ndarray:
    prediction = np.asarray(betas, dtype=np.float64) @ np.asarray(
        x, dtype=np.float64
    ).T
    return np.mean(
        (prediction - np.asarray(y, dtype=np.float64)[None, :]) ** 2, axis=1
    )


def signal_strength_proxy(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    correlations = np.abs(
        np.asarray(x_train, dtype=np.float64).T
        @ np.asarray(y_train, dtype=np.float64)
        / float(x_train.shape[0])
    )
    top = np.lexsort((np.arange(correlations.size), -correlations))[:32]
    design = np.asarray(x_train[:, top], dtype=np.float64)
    gram = design.T @ design / float(design.shape[0])
    rhs = design.T @ np.asarray(y_train, dtype=np.float64) / float(design.shape[0])
    beta = np.linalg.solve(gram + 1e-6 * np.eye(top.size), rhs)

    def r2(x: np.ndarray, y: np.ndarray) -> float:
        residual = np.asarray(y, dtype=np.float64) - np.asarray(
            x[:, top], dtype=np.float64
        ) @ beta
        denominator = float(np.sum((y - np.mean(y)) ** 2))
        return 1.0 - float(np.sum(residual**2)) / max(denominator, 1e-12)

    return {
        "metric": "top32_train_marginal_ridge1e-6_r2",
        "max_abs_train_xy_correlation": float(correlations.max(initial=0.0)),
        "top32_feature_indices": top.tolist(),
        "validation_r2": r2(x_validation, y_validation),
        "test_r2": r2(x_test, y_test),
    }


def penalty_diagnostics(
    knots: np.ndarray, relu_b: float, weights: np.ndarray
) -> dict[str, Any]:
    domain = np.linspace(0.0, float(knots[-1]), 4097, dtype=np.float64)
    rho = canonical_penalty_values(domain, knots, relu_b, weights)
    drho = np.full(domain.shape, float(relu_b), dtype=np.float64)
    for knot, weight in zip(knots, weights, strict=True):
        drho += float(weight) * np.maximum(domain - float(knot), 0.0)
    delta = np.diff(drho)
    d0 = float(drho[0])
    threshold = max(0.01 * d0, 1e-12)
    candidates = np.flatnonzero((domain > 0) & (drho <= threshold))
    saturation = None if candidates.size == 0 else float(domain[candidates[0]])
    return {
        "domain": domain,
        "rho": rho,
        "drho": drho,
        "rho_at_zero": float(rho[0]),
        "d0": d0,
        "max_drho": float(drho.max(initial=-np.inf)),
        "min_drho": float(drho.min(initial=np.inf)),
        "max_positive_drho_increment": float(np.maximum(delta, 0).max(initial=0.0)),
        "concavity_violation": bool((delta > 1e-8).any()),
        "monotonicity_violation": bool((drho < -1e-8).any()),
        "saturation_location_1pct_d0": saturation,
    }


def style_axis(axis: plt.Axes, *, fontsize: float) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=fontsize, colors=INK, length=3)
    axis.xaxis.label.set_size(fontsize + 0.5)
    axis.yaxis.label.set_size(fontsize + 0.5)
    axis.title.set_size(fontsize + 1.0)


def top_reference_features(reference_beta: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
    score = np.max(np.abs(reference_beta[base_mask]), axis=0)
    return np.lexsort((np.arange(score.size), -score))[:8]


def plot_task(
    output: Path,
    dataset_id: str,
    n: int,
    p: int,
    x_axis: np.ndarray,
    model_beta: np.ndarray,
    reference_beta: np.ndarray,
    validation_model_mse: np.ndarray,
    validation_reference_mse: np.ndarray,
    selected_index: int,
    oracle_index: int,
    top_features: np.ndarray,
    penalty: dict[str, Any],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 3.7))
    for color_index, feature in enumerate(top_features):
        color = FEATURE_COLORS[color_index % len(FEATURE_COLORS)]
        axes[0].plot(x_axis, model_beta[:, feature], color=color, lw=1.3)
        axes[0].plot(
            x_axis, reference_beta[:, feature], color=color, lw=1.0, ls=":"
        )
    axes[0].axhline(0, color=LIGHT_GREY, lw=0.6)
    axes[0].set_yscale("symlog", linthresh=0.02)
    axes[0].set_xlabel(r"$\log_{10}$ effective lambda ratio")
    axes[0].set_ylabel("Coefficient")
    axes[0].set_title(f"{dataset_id}  N={n:,}, P={p:,}")
    style_axis(axes[0], fontsize=9)

    log_model = np.log10(np.maximum(validation_model_mse, 1e-12))
    log_reference = np.log10(np.maximum(validation_reference_mse, 1e-12))
    axes[1].plot(x_axis, log_model, color=BLUE, lw=1.8)
    axes[1].plot(x_axis, log_reference, color=GREY, lw=1.3, ls=":")
    axes[1].axvline(x_axis[selected_index], color=ORANGE, lw=1.2, ls="--")
    axes[1].scatter(
        [x_axis[selected_index]], [log_model[selected_index]],
        marker="D", s=35, color=ORANGE, edgecolor=INK, linewidth=0.4, zorder=5,
    )
    axes[1].scatter(
        [x_axis[oracle_index]], [log_model[oracle_index]],
        marker="*", s=80, color="white", edgecolor=BLUE, linewidth=1.0, zorder=5,
    )
    axes[1].set_xlabel(r"$\log_{10}$ effective lambda ratio")
    axes[1].set_ylabel(r"$\log_{10}$ validation MSE")
    axes[1].set_title("Prediction-error path")
    style_axis(axes[1], fontsize=9)

    axes[2].plot(penalty["domain"], penalty["rho"], color=GREEN, lw=1.8)
    derivative = axes[2].twinx()
    derivative.plot(
        penalty["domain"], penalty["drho"], color=ORANGE, lw=1.3, ls="--"
    )
    axes[2].set_xlabel(r"$|\beta|$")
    axes[2].set_ylabel(r"$\rho(|\beta|)$", color=GREEN)
    derivative.set_ylabel(r"$\rho'(|\beta|)$", color=ORANGE)
    axes[2].set_title("KALBI penalty")
    style_axis(axes[2], fontsize=9)
    derivative.spines["top"].set_visible(False)
    derivative.tick_params(labelsize=9, colors=ORANGE, length=3)
    figure.tight_layout(w_pad=2.0)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(figure)


def evaluate_task(
    *,
    prepared_root: Path,
    entry: dict[str, Any],
    split_seed: int,
    model: torch.nn.Module,
    selector: torch.nn.Module,
    device: torch.device,
    kernel: CanonicalCDKernel,
    grid_size: int,
    query_chunk: int,
    max_sweeps: int,
    retries: int,
    tolerance: float,
    zero_tolerance: float,
    output_dir: Path,
) -> dict[str, Any]:
    x_raw, y_raw, provenance = load_frozen_task(prepared_root, entry)
    split = split_standardize_three_way(x_raw, y_raw, split_seed)
    x_train = split["x_train"]
    y_train = split["y_train"]
    x_validation = split["x_validation"]
    y_validation = split["y_validation"]
    x_test = split["x_test"]
    y_test = split["y_test"]
    assert isinstance(x_train, np.ndarray) and isinstance(y_train, np.ndarray)
    assert isinstance(x_validation, np.ndarray) and isinstance(y_validation, np.ndarray)
    assert isinstance(x_test, np.ndarray) and isinstance(y_test, np.ndarray)

    x_device, y_device = tensors_from_arrays(x_train, y_train, device)
    encoded = encode_dataset(model, x_device, y_device)
    lambda_max_tensor = encoded["lambda_max"].float().reshape(-1)
    lambda_max = float(lambda_max_tensor[0])
    with torch.inference_mode():
        selected = selector(
            x_device, y_device, lambda_max_tensor, return_representation=True
        )
        repeated = selector(
            x_device, y_device, lambda_max_tensor, return_representation=True
        )
    repeat_deltas = {
        key: float(
            torch.max(torch.abs(selected[key].float() - repeated[key].float())).cpu()
        )
        for key in ("effective_lambda_ratio", "lambdas", "penalty_rho", "penalty_drho")
    }
    if max(repeat_deltas.values()) > 1e-7:
        raise RuntimeError(f"selector repeatability failure: {repeat_deltas}")

    selected_ratio = float(selected["effective_lambda_ratio"].reshape(-1)[0])
    selected_raw_lambda = float(selected["lambdas"].reshape(-1)[0])
    reference_slope = float(
        selected["penalty_reference_slope"].reshape(-1)[0]
    )
    base_ratios = np.logspace(-6.0, 0.0, int(grid_size), dtype=np.float64)
    ratios = np.unique(np.concatenate((base_ratios, [selected_ratio])))
    ratios.sort()
    base_mask = np.isclose(
        ratios[:, None], base_ratios[None, :], rtol=1e-12, atol=1e-15
    ).any(axis=1)
    raw_lambdas = ratios * lambda_max / max(reference_slope, 1e-12)
    reconstructed = selected_ratio * lambda_max / max(reference_slope, 1e-12)
    if not math.isclose(
        reconstructed, selected_raw_lambda, rel_tol=2e-5, abs_tol=1e-8
    ):
        raise RuntimeError("effective lambda ratio reconstruction mismatch")
    selected_index = int(np.argmin(np.abs(ratios - selected_ratio)))
    shape = selected["penalty_shape_features"].reshape(1, -1).expand(
        ratios.size, -1
    )
    d0 = selected["penalty_d0"].reshape(-1).expand(ratios.size)
    model_beta_raw = predict_path_fp32(
        model,
        encoded,
        torch.as_tensor(raw_lambdas, device=device, dtype=torch.float32),
        shape,
        d0,
        chunk_size=query_chunk,
    ).numpy().astype(np.float64)

    knots = (
        selector.penalty_parameter_head.penalty_grid.detach()
        .cpu().numpy().astype(np.float64)
    )
    relu_b = float(selected["penalty_relu_b"].reshape(-1)[0])
    weights = (
        selected["penalty_relu_weights"].reshape(-1, knots.size)[0]
        .detach().cpu().numpy().astype(np.float64)
    )
    penalty = penalty_diagnostics(knots, relu_b, weights)
    if abs(float(penalty["rho_at_zero"])) > 1e-8:
        raise RuntimeError("KALBI penalty violates rho(0)=0")

    solved = solve_multistart_path(
        kernel=kernel,
        x=x_train,
        y=y_train,
        raw_lambdas=raw_lambdas,
        knots=knots,
        relu_b=relu_b,
        relu_weights=weights,
        model_beta=model_beta_raw,
        max_sweeps=max_sweeps,
        retries=retries,
        tolerance=tolerance,
        zero_tolerance=zero_tolerance,
    )
    model_beta = solved["clipped_model_beta"]
    reference_beta = solved["reference_beta"]
    fixed_point_ceiling = max(2e-3, 5.0 * tolerance)
    if float(np.max(solved["reference_fixed_point_max"])) > fixed_point_ceiling:
        raise RuntimeError("exact-CD fixed-point residual exceeded audit ceiling")

    model_data, model_penalty, model_objective = objective_paths(
        x_train, y_train, model_beta, raw_lambdas, knots, relu_b, weights
    )
    cd_data, cd_penalty, cd_objective = objective_paths(
        x_train, y_train, reference_beta, raw_lambdas, knots, relu_b, weights
    )
    if bool((cd_objective > model_objective + 2e-8).any()):
        raise RuntimeError("CD reference objective exceeded its KALBI start")
    validation_model_mse = prediction_mse(x_validation, y_validation, model_beta)
    validation_cd_mse = prediction_mse(x_validation, y_validation, reference_beta)
    test_model_mse = prediction_mse(x_test, y_test, model_beta)
    test_cd_mse = prediction_mse(x_test, y_test, reference_beta)
    base_indices = np.flatnonzero(base_mask)
    model_base_mse = validation_model_mse[base_indices]
    model_minimum = float(np.min(model_base_mse))
    model_ties = base_indices[
        np.isclose(model_base_mse, model_minimum, rtol=1e-12, atol=1e-14)
    ]
    oracle_index = int(model_ties[np.argmax(ratios[model_ties])])
    cd_base_mse = validation_cd_mse[base_indices]
    cd_minimum = float(np.min(cd_base_mse))
    cd_ties = base_indices[
        np.isclose(cd_base_mse, cd_minimum, rtol=1e-12, atol=1e-14)
    ]
    cd_oracle_index = int(cd_ties[np.argmax(ratios[cd_ties])])
    top_features = top_reference_features(reference_beta, base_mask)
    correlations = np.asarray([
        vector_correlation(model_beta[index], reference_beta[index])
        for index in base_indices
    ])
    coefficient_delta = model_beta[base_mask] - reference_beta[base_mask]
    objective_gap = model_objective[base_mask] - cd_objective[base_mask]
    signal = signal_strength_proxy(
        x_train, y_train, x_validation, y_validation, x_test, y_test
    )
    task_meta = dict(
        provenance["manifest"]["files"][0].get("task_metadata", [{}])[0]
    )
    zero_test_mse = float(np.mean(np.asarray(y_test, dtype=np.float64) ** 2))
    metrics = {
        "evaluation_version": EVALUATION_VERSION,
        "dataset_id": entry["dataset_id"],
        "dataset_index": int(entry["dataset_index"]),
        "seed": int(entry["seed"]),
        "task_sha256": entry["task_sha256"],
        "n": int(entry["n"]),
        "p": int(entry["p"]),
        "p_over_n": float(entry["p_over_n"]),
        "split_seed": int(split_seed),
        "train_rows": int(x_train.shape[0]),
        "validation_rows": int(x_validation.shape[0]),
        "test_rows": int(x_test.shape[0]),
        "split_index_sha256": {
            name: array_sha256(split[name])
            for name in ("train_index", "validation_index", "test_index")
        },
        "preprocessing": {
            "fit_scope": "training_rows_only",
            "x_mean_sha256": array_sha256(split["x_mean"]),
            "x_std_sha256": array_sha256(split["x_std"]),
            "y_mean": float(split["y_mean"]),
            "y_std": float(split["y_std"]),
            "inactive_features": int(
                np.count_nonzero(np.asarray(split["active"]) == 0)
            ),
        },
        "leakage_audit": {
            "model_input_fields": ["standardized_training_X", "standardized_training_y"],
            "validation_y_used_as_model_input": False,
            "test_y_used_as_model_input": False,
            "repeat_output_max_deltas": repeat_deltas,
            "selector_input_sha256": {
                "x_train": array_sha256(x_train),
                "y_train": array_sha256(y_train),
            },
        },
        "correlation": {
            "core_exact_max_abs_corr": provenance["core_exact_max_abs_corr"],
            "sampled_final_max_abs_corr": provenance["sampled_final_max_abs_corr"],
            "full_pair_exact_certificate": bool(entry["full_pair_exact_certificate"]),
            "note": (
                "P>2048 V11 tasks have an exact SCM-core certificate and a "
                "deterministic sampled full-P audit, not an exact all-P-pair maximum."
            ),
        },
        "signal_strength": signal,
        "zero_predictor_test_mse": zero_test_mse,
        "target_sparsity": {
            "true_linear_support_defined": False,
            "official_active_base_features": task_meta.get(
                "official_active_base_features"
            ),
            "scm_core_features": task_meta.get("scm_core_features"),
            "bnn_extension_features": task_meta.get("bnn_extension_features"),
        },
        "feature_noise_variance_summary": task_meta.get(
            "feature_noise_variance_summary"
        ),
        "lambda_max": lambda_max,
        "effective_lambda_ratio_definition": (
            "raw_lambda_times_max_penalty_derivative_on_0_6_divided_by_lambda_max"
        ),
        "lambda_grid_min": float(base_ratios.min()),
        "lambda_grid_max": float(base_ratios.max()),
        "lambda_grid_size": int(base_ratios.size),
        "selected_effective_lambda_ratio": selected_ratio,
        "selected_raw_lambda": selected_raw_lambda,
        "selected_validation_mse": float(validation_model_mse[selected_index]),
        "oracle_effective_lambda_ratio": float(ratios[oracle_index]),
        "oracle_validation_mse": float(validation_model_mse[oracle_index]),
        "selected_over_oracle_validation_mse": float(
            validation_model_mse[selected_index]
            / max(float(validation_model_mse[oracle_index]), 1e-12)
        ),
        "selected_test_mse": float(test_model_mse[selected_index]),
        "validation_oracle_test_mse": float(test_model_mse[oracle_index]),
        "cd_validation_oracle_ratio": float(ratios[cd_oracle_index]),
        "cd_validation_oracle_test_mse": float(test_cd_mse[cd_oracle_index]),
        "selected_nnz_1e4": int(
            np.count_nonzero(np.abs(model_beta[selected_index]) > 1e-4)
        ),
        "path_metrics_full_p": {
            "mean_per_lambda_correlation": float(np.nanmean(correlations)),
            "coefficient_rmse": float(np.sqrt(np.mean(coefficient_delta**2))),
            "objective_gap_mean": float(np.mean(objective_gap)),
            "objective_gap_max": float(np.max(objective_gap)),
            "objective_gap_rmse": float(np.sqrt(np.mean(objective_gap**2))),
        },
        "cd_diagnostics": {
            "fixed_point_max": float(
                np.max(solved["reference_fixed_point_max"])
            ),
            "projected_kkt_max": float(
                np.max(solved["reference_projected_kkt_max"])
            ),
            "sweeps_max": int(np.max(solved["reference_sweeps"])),
            "sweeps_total": int(np.sum(solved["reference_sweeps"])),
        },
        "penalty": {
            key: value
            for key, value in penalty.items()
            if key not in {"domain", "rho", "drho"}
        },
        "top_reference_path_features": top_features.tolist(),
    }
    task_output = output_dir / f"dataset_{int(entry['dataset_index']):02d}"
    task_output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        task_output / "paths.npz",
        effective_lambda_ratios=ratios,
        base_grid_mask=base_mask,
        raw_lambdas=raw_lambdas,
        kalbi_beta=model_beta.astype(np.float32),
        coordinate_descent_beta=reference_beta.astype(np.float32),
        kalbi_data_half=model_data,
        kalbi_penalty=model_penalty,
        kalbi_objective=model_objective,
        coordinate_descent_data_half=cd_data,
        coordinate_descent_penalty=cd_penalty,
        coordinate_descent_objective=cd_objective,
        kalbi_validation_mse=validation_model_mse,
        coordinate_descent_validation_mse=validation_cd_mse,
        kalbi_test_mse=test_model_mse,
        coordinate_descent_test_mse=test_cd_mse,
        selected_index=np.asarray(selected_index),
        oracle_index=np.asarray(oracle_index),
        cd_oracle_index=np.asarray(cd_oracle_index),
        top_features=top_features,
        penalty_domain=penalty["domain"],
        penalty_rho=penalty["rho"],
        penalty_drho=penalty["drho"],
    )
    atomic_json(task_output / "metrics.json", metrics)
    plot_task(
        task_output / "detail",
        entry["dataset_id"],
        int(entry["n"]),
        int(entry["p"]),
        np.log10(ratios),
        model_beta,
        reference_beta,
        validation_model_mse,
        validation_cd_mse,
        selected_index,
        oracle_index,
        top_features,
        penalty,
    )
    return metrics


def load_plot_payload(output_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    task_dir = output_dir / f"dataset_{int(metrics['dataset_index']):02d}"
    with np.load(task_dir / "paths.npz", allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def create_composite(
    output_dir: Path, metrics_all: list[dict[str, Any]]
) -> tuple[Path, Path]:
    figure, axes = plt.subplots(5, 3, figsize=(7.2, 10.2))
    for row, metrics in enumerate(metrics_all):
        data = load_plot_payload(output_dir, metrics)
        ratios = data["effective_lambda_ratios"]
        x_axis = np.log10(ratios)
        model_beta = data["kalbi_beta"]
        reference_beta = data["coordinate_descent_beta"]
        top = data["top_features"].astype(int)
        selected_index = int(data["selected_index"])
        oracle_index = int(data["oracle_index"])
        for color_index, feature in enumerate(top):
            color = FEATURE_COLORS[color_index % len(FEATURE_COLORS)]
            axes[row, 0].plot(x_axis, model_beta[:, feature], color=color, lw=0.8)
            axes[row, 0].plot(
                x_axis, reference_beta[:, feature], color=color, lw=0.65, ls=":"
            )
        axes[row, 0].axhline(0, color=LIGHT_GREY, lw=0.4)
        axes[row, 0].set_yscale("symlog", linthresh=0.02)
        axes[row, 0].text(
            0.02, 0.97,
            f"Dataset {row + 1}\nN={metrics['n']:,}, P={metrics['p']:,}",
            transform=axes[row, 0].transAxes,
            ha="left", va="top", fontsize=6.5,
        )

        model_mse = data["kalbi_validation_mse"]
        reference_mse = data["coordinate_descent_validation_mse"]
        log_model = np.log10(np.maximum(model_mse, 1e-12))
        log_reference = np.log10(np.maximum(reference_mse, 1e-12))
        axes[row, 1].plot(x_axis, log_model, color=BLUE, lw=1.1)
        axes[row, 1].plot(x_axis, log_reference, color=GREY, lw=0.9, ls=":")
        axes[row, 1].axvline(
            x_axis[selected_index], color=ORANGE, lw=0.8, ls="--"
        )
        axes[row, 1].scatter(
            [x_axis[selected_index]], [log_model[selected_index]],
            marker="D", s=15, color=ORANGE, edgecolor=INK, linewidth=0.25, zorder=4,
        )
        axes[row, 1].scatter(
            [x_axis[oracle_index]], [log_model[oracle_index]],
            marker="*", s=35, color="white", edgecolor=BLUE, linewidth=0.7, zorder=4,
        )

        axes[row, 2].plot(
            data["penalty_domain"], data["penalty_rho"], color=GREEN, lw=1.1
        )
        derivative = axes[row, 2].twinx()
        derivative.plot(
            data["penalty_domain"], data["penalty_drho"],
            color=ORANGE, lw=0.8, ls="--",
        )
        derivative.spines["top"].set_visible(False)
        derivative.tick_params(labelsize=5.5, colors=ORANGE, length=2)

        for column in range(3):
            style_axis(axes[row, column], fontsize=5.8)
        axes[row, 2].tick_params(axis="y", colors=GREEN)
        if row == 0:
            axes[row, 0].set_title("Coefficient paths", pad=5)
            axes[row, 1].set_title("Validation error", pad=5)
            axes[row, 2].set_title("Selected penalty", pad=5)
        if row == 4:
            axes[row, 0].set_xlabel(r"$\log_{10}$ effective lambda ratio")
            axes[row, 1].set_xlabel(r"$\log_{10}$ effective lambda ratio")
            axes[row, 2].set_xlabel(r"$|\beta|$")
        axes[row, 0].set_ylabel("Coefficient")
        axes[row, 1].set_ylabel(r"$\log_{10}$ MSE")
        axes[row, 2].set_ylabel(r"$\rho$", color=GREEN)
        derivative.set_ylabel(r"$\rho'$", color=ORANGE, fontsize=6.3)

    handles = [
        Line2D([0], [0], color=INK, lw=1.2, label="KALBI"),
        Line2D(
            [0], [0], color=INK, lw=1.0, ls=":",
            label="Coordinate descent reference",
        ),
        Line2D(
            [0], [0], color=ORANGE, lw=1.0, ls="--",
            marker="D", markersize=3.5, label="KALBI-selected lambda",
        ),
        Line2D(
            [0], [0], color=BLUE, lw=0, marker="*", markersize=7,
            markerfacecolor="white", label="Oracle lambda",
        ),
        Line2D([0], [0], color=GREEN, lw=1.2, label=r"$\rho$"),
        Line2D([0], [0], color=ORANGE, lw=1.0, ls="--", label=r"$\rho'$"),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=6.5,
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.subplots_adjust(
        left=0.075, right=0.94, top=0.975, bottom=0.065, wspace=0.45, hspace=0.34
    )
    visible = " ".join(text.get_text() for text in figure.findobj(Text)).lower()
    for forbidden in FORBIDDEN_VISIBLE_TEXT:
        if forbidden in visible:
            raise RuntimeError(f"forbidden visible paper label: {forbidden}")
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=False)
    pdf = figure_dir / "kalbi_unseen_highdim5_composite.pdf"
    png = figure_dir / "kalbi_unseen_highdim5_composite.png"
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=600, bbox_inches="tight")
    plt.close(figure)
    return pdf, png


def write_summary(
    output_dir: Path,
    checkpoint: Path,
    checkpoint_step: int,
    config_path: Path,
    cd_library: Path,
    metrics_all: list[dict[str, Any]],
    pdf: Path,
    png: Path,
) -> None:
    rows = []
    for metric in metrics_all:
        rows.append({
            "dataset_id": metric["dataset_id"],
            "seed": metric["seed"],
            "n": metric["n"],
            "p": metric["p"],
            "p_over_n": metric["p_over_n"],
            "train_rows": metric["train_rows"],
            "validation_rows": metric["validation_rows"],
            "test_rows": metric["test_rows"],
            "core_exact_max_abs_corr": metric["correlation"][
                "core_exact_max_abs_corr"
            ],
            "sampled_final_max_abs_corr": metric["correlation"][
                "sampled_final_max_abs_corr"
            ],
            "signal_proxy_validation_r2": metric["signal_strength"]["validation_r2"],
            "zero_predictor_test_mse": metric["zero_predictor_test_mse"],
            "selected_effective_lambda_ratio": metric[
                "selected_effective_lambda_ratio"
            ],
            "selected_validation_mse": metric["selected_validation_mse"],
            "oracle_validation_mse": metric["oracle_validation_mse"],
            "selected_over_oracle_validation_mse": metric[
                "selected_over_oracle_validation_mse"
            ],
            "selected_test_mse": metric["selected_test_mse"],
            "path_correlation": metric["path_metrics_full_p"][
                "mean_per_lambda_correlation"
            ],
            "coefficient_rmse": metric["path_metrics_full_p"]["coefficient_rmse"],
            "objective_gap_mean": metric["path_metrics_full_p"][
                "objective_gap_mean"
            ],
            "rho_at_zero": metric["penalty"]["rho_at_zero"],
            "penalty_concavity_violation": metric["penalty"][
                "concavity_violation"
            ],
        })
    with (output_dir / "metrics_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(output_dir / "metrics_summary.json", rows)
    caption = (
        "KALBI on five pre-registered unseen high-dimensional datasets. "
        "Left: coefficient paths for the eight coefficients with the largest "
        "coordinate-descent reference magnitude. Center: validation prediction "
        "error over a common effective-lambda grid; diamonds mark the lambda "
        "selected by KALBI from training data alone and stars mark the descriptive "
        "validation oracle. Right: the penalty and its derivative selected by "
        "KALBI. All preprocessing statistics were estimated from the training "
        "partition only."
    )
    (output_dir / "figure_caption.txt").write_text(caption + "\n", encoding="utf-8")
    manifest = {
        "evaluation_version": EVALUATION_VERSION,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_step": int(checkpoint_step),
        "model_config": str(config_path.resolve()),
        "model_config_sha256": file_sha256(config_path.resolve()),
        "cd_library": str(cd_library.resolve()),
        "cd_library_sha256": file_sha256(cd_library.resolve()),
        "composite_pdf": str(pdf),
        "composite_pdf_sha256": file_sha256(pdf),
        "composite_png": str(png),
        "composite_png_sha256": file_sha256(png),
        "datasets": [
            {
                "dataset_id": metric["dataset_id"],
                "seed": metric["seed"],
                "task_sha256": metric["task_sha256"],
            }
            for metric in metrics_all
        ],
        "leakage_checks_passed": all(
            not metric["leakage_audit"]["validation_y_used_as_model_input"]
            and not metric["leakage_audit"]["test_y_used_as_model_input"]
            for metric in metrics_all
        ),
    }
    atomic_json(output_dir / "reproducibility_manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cd-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grid-size", type=int, default=81)
    parser.add_argument("--query-chunk", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=2_026_072_407)
    parser.add_argument("--max-sweeps", type=int, default=2000)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--zero-tolerance", type=float, default=1e-10)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--dataset-indices", type=int, nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    prepared_root = args.prepared_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.resume:
        raise RuntimeError(f"refusing existing output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    frozen = json.loads(
        (prepared_root / "reproducibility" / "reproducibility_manifest.json")
        .read_text(encoding="utf-8")
    )
    accepted_entries = list(frozen["accepted_tasks"])
    if args.dataset_indices:
        requested_indices = set(args.dataset_indices)
        entries = [
            entry
            for entry in accepted_entries
            if int(entry["dataset_index"]) in requested_indices
        ]
        found_indices = {int(entry["dataset_index"]) for entry in entries}
        if found_indices != requested_indices:
            raise RuntimeError(
                "requested dataset indices are not present in the frozen manifest"
            )
    else:
        entries = accepted_entries[: int(args.limit)]
        if len(entries) != int(args.limit):
            raise RuntimeError("prepared task count is smaller than requested limit")
    checkpoint = Path(frozen["checkpoint_frozen"]).resolve()
    if file_sha256(checkpoint) != frozen["source_hashes"]["checkpoint"]:
        raise RuntimeError("frozen checkpoint SHA mismatch")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint_payload, model, selector = load_components_v20(
        checkpoint, args.config.resolve(), device
    )
    checkpoint_step = int(checkpoint_payload["completed_steps"])
    if checkpoint_payload.get("metadata", {}).get("model_version") != MODEL_VERSION:
        raise RuntimeError("strict checkpoint model version mismatch")
    for state_name in ("model_state_dict", "selector_state_dict"):
        for name, value in checkpoint_payload[state_name].items():
            if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"nonfinite checkpoint tensor: {state_name}.{name}")
    kernel = CanonicalCDKernel(args.cd_library.resolve())
    metrics_all: list[dict[str, Any]] = []
    for entry in entries:
        dataset_index = int(entry["dataset_index"])
        task_dir = output_dir / f"dataset_{dataset_index:02d}"
        metrics_path = task_dir / "metrics.json"
        if args.resume and metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics.get("task_sha256") != entry["task_sha256"]:
                raise RuntimeError(f"resume task SHA mismatch for {task_dir}")
            metrics_all.append(metrics)
            print(json.dumps({
                "dataset": entry["dataset_id"],
                "status": "resume_existing",
            }), flush=True)
            continue
        print(json.dumps({
            "dataset": entry["dataset_id"],
            "status": "starting",
            "n": entry["n"],
            "p": entry["p"],
        }), flush=True)
        metrics = evaluate_task(
            prepared_root=prepared_root,
            entry=entry,
            split_seed=int(args.split_seed) + SEED_STRIDE * (dataset_index - 1),
            model=model,
            selector=selector,
            device=device,
            kernel=kernel,
            grid_size=int(args.grid_size),
            query_chunk=int(args.query_chunk),
            max_sweeps=int(args.max_sweeps),
            retries=int(args.retries),
            tolerance=float(args.tolerance),
            zero_tolerance=float(args.zero_tolerance),
            output_dir=output_dir,
        )
        metrics_all.append(metrics)
        print(json.dumps({
            "dataset": metrics["dataset_id"],
            "n": metrics["n"],
            "p": metrics["p"],
            "selected_over_oracle": metrics["selected_over_oracle_validation_mse"],
            "path_correlation": metrics["path_metrics_full_p"][
                "mean_per_lambda_correlation"
            ],
        }), flush=True)
    if not args.dataset_indices and len(metrics_all) == 5:
        pdf, png = create_composite(output_dir, metrics_all)
        write_summary(
            output_dir,
            checkpoint,
            checkpoint_step,
            args.config,
            args.cd_library,
            metrics_all,
            pdf,
            png,
        )
        print(json.dumps({
            "composite_pdf": str(pdf),
            "composite_png": str(png),
        }), flush=True)


if __name__ == "__main__":
    main()
