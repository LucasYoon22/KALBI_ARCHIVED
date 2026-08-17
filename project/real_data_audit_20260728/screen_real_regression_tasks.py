#!/usr/bin/env python3
"""Quality-audit and linear-signal screen prepared real regression tasks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 20260728
MAX_CONTEXT = 12000
MAX_SCREEN = 5000
MAX_TEST = 5000
ACCEPT_SCREEN_R2 = 0.20
ACCEPT_REPEAT_MEDIAN_R2 = 0.15

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def dump_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def finite_number(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def cap_indices(indices: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if indices.size <= limit:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=limit, replace=False))


def load_task(task_root: Path) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any]]:
    metadata = json.loads((task_root / "metadata.json").read_text())
    if metadata["X_storage"] == "sparse_csr_npz":
        X = sparse.load_npz(task_root / "X.npz").tocsr()
    else:
        X = np.load(task_root / "X.npy", mmap_mode="r")
    # Memory-map y so locked-test values are not paged in before acceptance.
    y = np.load(task_root / "y.npy", mmap_mode="r")
    split = np.load(task_root / "split.npy")
    return X, y, split, metadata


def build_transformer(
    X_context: Any, metadata: dict[str, Any]
) -> tuple[Any, bool]:
    categorical = metadata.get("categorical_feature_indices") or []
    if categorical and not sparse.issparse(X_context):
        categorical = sorted(set(int(index) for index in categorical))
        numeric = [index for index in range(X_context.shape[1]) if index not in categorical]
        transformer = ColumnTransformer(
            [
                (
                    "categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "onehot",
                                OneHotEncoder(
                                    handle_unknown="ignore",
                                    sparse_output=True,
                                    dtype=np.float32,
                                ),
                            ),
                        ]
                    ),
                    categorical,
                ),
                (
                    "numeric",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="mean")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    numeric,
                ),
            ],
            sparse_threshold=0.2,
        )
        transformer.fit(X_context)
        return transformer, True
    if sparse.issparse(X_context):
        scaler = StandardScaler(with_mean=False)
        scaler.fit(X_context)
        return scaler, False
    if np.isnan(np.asarray(X_context)).any():
        transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scale", StandardScaler()),
            ]
        )
    else:
        transformer = StandardScaler()
    transformer.fit(X_context)
    return transformer, False


def transform(transformer: Any, X: Any) -> Any:
    result = transformer.transform(X)
    if sparse.issparse(result):
        return result.tocsr().astype(np.float32)
    return np.asarray(result, dtype=np.float32)


def predict_centered(model: Any, X: Any, y_mean: float) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=float) + y_mean


def model_grid(
    name: str,
    X_train: Any,
    y_train_centered: np.ndarray,
    p: int,
) -> list[dict[str, float]]:
    if name == "ridge":
        return [{"alpha": float(value)} for value in np.logspace(-4, 6, 6)]
    correlation = np.asarray(X_train.T @ y_train_centered).reshape(-1)
    lambda_max = max(float(np.max(np.abs(correlation))) / len(y_train_centered), 1e-8)
    points = 5 if p > 100000 else 6
    ratios = np.logspace(0, -3, points)
    if name == "lasso":
        return [{"alpha": float(lambda_max * ratio)} for ratio in ratios]
    return [
        {"alpha": float(lambda_max * ratio), "l1_ratio": float(l1_ratio)}
        for l1_ratio in (0.2, 0.8)
        for ratio in ratios[:5]
    ]


def make_model(name: str, parameters: dict[str, float], seed: int) -> Any:
    if name == "ridge":
        return Ridge(
            alpha=parameters["alpha"],
            fit_intercept=False,
            solver="lsqr",
            tol=1e-4,
            max_iter=2000,
        )
    if name == "lasso":
        return Lasso(
            alpha=parameters["alpha"],
            fit_intercept=False,
            max_iter=1000,
            tol=2e-3,
            selection="random",
            random_state=seed,
        )
    return ElasticNet(
        alpha=parameters["alpha"],
        l1_ratio=parameters["l1_ratio"],
        fit_intercept=False,
        max_iter=1000,
        tol=2e-3,
        selection="random",
        random_state=seed,
    )


def tune_model(
    name: str,
    X_context: Any,
    y_context: np.ndarray,
    seed: int,
) -> tuple[dict[str, float], float]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y_context))
    cut = max(2, int(0.8 * len(order)))
    train_idx = order[:cut]
    valid_idx = order[cut:]
    y_mean = float(np.mean(y_context[train_idx]))
    y_centered = y_context[train_idx] - y_mean
    grid = model_grid(name, X_context[train_idx], y_centered, X_context.shape[1])
    best_parameters: dict[str, float] | None = None
    best_r2 = -np.inf
    for parameters in grid:
        model = make_model(name, parameters, seed)
        model.fit(X_context[train_idx], y_centered)
        prediction = predict_centered(model, X_context[valid_idx], y_mean)
        score = r2_score(y_context[valid_idx], prediction)
        if np.isfinite(score) and score > best_r2:
            best_r2 = float(score)
            best_parameters = parameters
    if best_parameters is None:
        raise RuntimeError(f"{name} produced no finite tuning score")
    return best_parameters, best_r2


def repeated_cv(
    name: str,
    parameters: dict[str, float],
    X_context: Any,
    y_context: np.ndarray,
) -> list[float]:
    scores: list[float] = []
    for repeat in range(5):
        seed = SEED + 1009 * (repeat + 1)
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(y_context))
        cut = max(2, int(0.8 * len(order)))
        train_idx = order[:cut]
        valid_idx = order[cut:]
        y_mean = float(np.mean(y_context[train_idx]))
        model = make_model(name, parameters, seed)
        model.fit(X_context[train_idx], y_context[train_idx] - y_mean)
        prediction = predict_centered(model, X_context[valid_idx], y_mean)
        scores.append(float(r2_score(y_context[valid_idx], prediction)))
    return scores


def sampled_feature_correlation(X: Any, seed: int) -> float | None:
    n, p = X.shape
    if p < 2 or n < 3:
        return None
    rng = np.random.default_rng(seed)
    rows = np.arange(n) if n <= 5000 else np.sort(rng.choice(n, 5000, replace=False))
    columns = np.arange(p) if p <= 256 else np.sort(rng.choice(p, 256, replace=False))
    sample = X[rows][:, columns]
    if sparse.issparse(sample):
        sample = sample.toarray()
    sample = np.asarray(sample, dtype=np.float64)
    std = np.nanstd(sample, axis=0)
    sample = sample[:, std > 1e-12]
    if sample.shape[1] < 2:
        return None
    correlation = np.corrcoef(sample, rowvar=False)
    np.fill_diagonal(correlation, 0.0)
    return finite_number(np.nanmax(np.abs(correlation)))


def marginal_correlation(X: Any, y: np.ndarray) -> float:
    centered_y = y - np.mean(y)
    y_norm = max(float(np.linalg.norm(centered_y)), 1e-12)
    if sparse.issparse(X):
        values = np.asarray(X.T @ centered_y).reshape(-1)
        column_norm = np.sqrt(np.asarray(X.power(2).sum(axis=0))).reshape(-1)
    else:
        centered_x = np.asarray(X, dtype=np.float64) - np.mean(X, axis=0)
        values = centered_x.T @ centered_y
        column_norm = np.linalg.norm(centered_x, axis=0)
    correlation = np.abs(values) / np.maximum(column_norm * y_norm, 1e-12)
    return float(np.nanmax(correlation))


def quality_gate(y_context: np.ndarray, y_screen: np.ndarray, metadata: dict[str, Any]) -> tuple[bool, list[str]]:
    y = np.concatenate([y_context, y_screen])
    finite = y[np.isfinite(y)]
    reasons: list[str] = []
    minimum_finite = int(metadata.get("screening_min_finite_context_screen", 200))
    if finite.size < minimum_finite:
        reasons.append(
            f"fewer than {minimum_finite} finite context+screening rows"
        )
    if finite.size == 0 or np.std(finite) <= 1e-8:
        reasons.append("zero or near-zero target variance")
    unique, counts = np.unique(finite, return_counts=True)
    if unique.size < 10:
        reasons.append("fewer than 10 unique target values")
    if counts.size and counts.max() / counts.sum() > 0.30:
        reasons.append("one target value occupies more than 30 percent")
    if "synthetic" in str(metadata.get("reality_status", "")).lower():
        reasons.append("synthetic benchmark, excluded from real-data acceptance")
    return not reasons, reasons


def evaluate_task(task_root: Path, metrics_root: Path) -> dict[str, Any]:
    started = time.time()
    X, y, split, metadata = load_task(task_root)
    context_idx = cap_indices(np.flatnonzero(split == 0), MAX_CONTEXT, SEED)
    screen_idx = cap_indices(np.flatnonzero(split == 1), MAX_SCREEN, SEED + 1)
    test_idx = cap_indices(np.flatnonzero(split == 2), MAX_TEST, SEED + 2)
    y_context = y[context_idx]
    y_screen = y[screen_idx]
    quality_ok, quality_reasons = quality_gate(y_context, y_screen, metadata)
    result: dict[str, Any] = {
        "screening_protocol": "real_regression_signal_screen_v2",
        "screening_seed": SEED,
        "context_row_cap": MAX_CONTEXT,
        "screening_row_cap": MAX_SCREEN,
        "locked_test_row_cap": MAX_TEST,
        "model_family_selection": "highest context-only inner-tuning R2",
        "acceptance_rule": (
            f"screening R2 >= {ACCEPT_SCREEN_R2} and repeated context median "
            f"R2 >= {ACCEPT_REPEAT_MEDIAN_R2}"
        ),
        "evidence_tier": metadata.get("evidence_tier", "standard"),
        "minimum_finite_context_screen": int(
            metadata.get("screening_min_finite_context_screen", 200)
        ),
        "task_id": task_root.name,
        "source": metadata.get("source"),
        "target": metadata.get("target"),
        "reality_status": metadata.get("reality_status"),
        "N": metadata["N"],
        "P_raw": metadata["P"],
        "P_over_N": metadata["P_over_N"],
        "context_rows_used": len(context_idx),
        "screen_rows_used": len(screen_idx),
        "test_rows_reserved": len(test_idx),
        "quality_ok": quality_ok,
        "quality_reasons": quality_reasons,
        "test_y_accessed_before_acceptance": False,
        "status": "quality_reject" if not quality_ok else "running",
    }
    if not quality_ok:
        result["elapsed_seconds"] = time.time() - started
        dump_json(metrics_root / f"{task_root.name}.json", result)
        return result

    X_context_raw = X[context_idx]
    X_screen_raw = X[screen_idx]
    transformer, categorical_transform = build_transformer(X_context_raw, metadata)
    X_context = transform(transformer, X_context_raw)
    X_screen = transform(transformer, X_screen_raw)
    result["P_transformed"] = int(X_context.shape[1])
    result["categorical_train_only_onehot"] = categorical_transform
    result["sampled_max_abs_feature_corr"] = sampled_feature_correlation(
        X_context, SEED
    )
    result["max_abs_marginal_corr_y"] = marginal_correlation(X_context, y_context)
    result["target_context_mean"] = float(np.mean(y_context))
    result["target_context_std"] = float(np.std(y_context))
    result["target_context_iqr"] = float(
        np.quantile(y_context, 0.75) - np.quantile(y_context, 0.25)
    )
    if result["max_abs_marginal_corr_y"] > 0.9999:
        result["status"] = "leakage_reject"
        result["quality_reasons"].append(
            "max marginal correlation exceeds 0.9999; target leakage suspected"
        )
        result["elapsed_seconds"] = time.time() - started
        dump_json(metrics_root / f"{task_root.name}.json", result)
        return result

    model_columns: np.ndarray | None = None
    if task_root.name == "uci_twin_gas_concentration_raw":
        time_indices = np.linspace(0, 59999, 512, dtype=np.int64)
        model_columns = np.concatenate(
            [channel * 60000 + time_indices for channel in range(8)]
        )
        result["screening_feature_selection"] = (
            "label-independent 512 evenly spaced time points per sensor"
        )
        result["P_screening_model"] = int(model_columns.size)
        X_context_model = X_context[:, model_columns]
        X_screen_model = X_screen[:, model_columns]
    else:
        result["screening_feature_selection"] = "all transformed features"
        result["P_screening_model"] = int(X_context.shape[1])
        X_context_model = X_context
        X_screen_model = X_screen

    model_results: dict[str, Any] = {}
    for model_name in ("ridge", "lasso", "elastic_net"):
        model_started = time.time()
        parameters, tuning_r2 = tune_model(
            model_name, X_context_model, y_context, SEED + len(model_results)
        )
        y_mean = float(np.mean(y_context))
        model = make_model(model_name, parameters, SEED)
        model.fit(X_context_model, y_context - y_mean)
        prediction = predict_centered(model, X_screen_model, y_mean)
        screen_r2 = float(r2_score(y_screen, prediction))
        screen_mse = float(mean_squared_error(y_screen, prediction))
        model_results[model_name] = {
            "parameters": parameters,
            "inner_tuning_r2": tuning_r2,
            "screening_r2": screen_r2,
            "screening_mse": screen_mse,
            "fit_seconds": time.time() - model_started,
        }
    result["models"] = model_results
    # Select the model family without consulting the screening holdout.
    best_name = max(model_results, key=lambda name: model_results[name]["inner_tuning_r2"])
    best = model_results[best_name]
    repeat_scores = repeated_cv(
        best_name, best["parameters"], X_context_model, y_context
    )
    best["repeated_context_r2"] = repeat_scores
    best["repeated_context_median_r2"] = float(np.median(repeat_scores))
    accepted = (
        best["screening_r2"] >= ACCEPT_SCREEN_R2
        and best["repeated_context_median_r2"] >= ACCEPT_REPEAT_MEDIAN_R2
    )
    result["best_model"] = best_name
    result["best_screening_r2"] = best["screening_r2"]
    result["best_repeated_context_median_r2"] = best[
        "repeated_context_median_r2"
    ]
    result["accepted"] = accepted
    if accepted:
        result["test_y_accessed_before_acceptance"] = False
        X_test = transform(transformer, X[test_idx])
        if model_columns is not None:
            X_test = X_test[:, model_columns]
        y_test = y[test_idx]
        y_mean = float(np.mean(y_context))
        model = make_model(best_name, best["parameters"], SEED)
        model.fit(X_context_model, y_context - y_mean)
        prediction = predict_centered(model, X_test, y_mean)
        test_mse = float(mean_squared_error(y_test, prediction))
        zero_mse = float(mean_squared_error(y_test, np.full_like(y_test, y_mean)))
        test_r2 = float(r2_score(y_test, prediction))
        clipped_r2 = max(test_r2, 0.0)
        result.update(
            {
                "status": "accepted",
                "test_r2": test_r2,
                "test_mse": test_mse,
                "zero_predictor_test_mse": zero_mse,
                "snr_proxy_from_test_r2": clipped_r2
                / max(1.0 - clipped_r2, 1e-8),
                "snr_proxy_note": (
                    "predictive lower-bound proxy R2/(1-R2), not generative SNR"
                ),
            }
        )
    else:
        result["status"] = "low_signal_reject"
    result["elapsed_seconds"] = time.time() - started
    dump_json(metrics_root / f"{task_root.name}.json", result)
    return result


def write_summary(root: Path, results: list[dict[str, Any]]) -> None:
    summary_fields = [
        "task_id",
        "source",
        "target",
        "reality_status",
        "evidence_tier",
        "status",
        "N",
        "P_raw",
        "P_transformed",
        "P_over_N",
        "context_rows_used",
        "screen_rows_used",
        "sampled_max_abs_feature_corr",
        "max_abs_marginal_corr_y",
        "best_model",
        "best_screening_r2",
        "best_repeated_context_median_r2",
        "test_r2",
        "test_mse",
        "zero_predictor_test_mse",
        "snr_proxy_from_test_r2",
        "elapsed_seconds",
        "quality_reasons",
    ]
    metrics_root = root / "metrics"
    with (root / "all_screening_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            row = result.copy()
            row["quality_reasons"] = "|".join(result.get("quality_reasons") or [])
            writer.writerow(row)
    accepted = [result for result in results if result.get("status") == "accepted"]
    rejected = [result for result in results if result.get("status") != "accepted"]
    dump_json(root / "accepted_high_signal_datasets.json", accepted)
    dump_json(root / "rejected_datasets.json", rejected)
    with (root / "accepted_high_signal_datasets.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        for result in accepted:
            row = result.copy()
            row["quality_reasons"] = "|".join(result.get("quality_reasons") or [])
            writer.writerow(row)

    if results:
        ordered = sorted(
            results,
            key=lambda item: item.get("best_screening_r2", -10),
            reverse=True,
        )
        labels = [item["task_id"].replace("hf_tabbench_", "") for item in ordered]
        values = [item.get("best_screening_r2", np.nan) for item in ordered]
        colors = [
            (
                "#E69F00"
                if item.get("status") == "accepted"
                and item.get("evidence_tier") == "exploratory_small_n_highdim"
                else "#0072B2"
                if item.get("status") == "accepted"
                else "#999999"
            )
            for item in ordered
        ]
        figure, axis = plt.subplots(figsize=(10, max(6, 0.25 * len(ordered))))
        y_positions = np.arange(len(ordered))
        axis.barh(y_positions, values, color=colors)
        axis.set_yticks(y_positions, labels, fontsize=7)
        axis.invert_yaxis()
        axis.axvline(ACCEPT_SCREEN_R2, color="#D55E00", linestyle="--", linewidth=1)
        axis.set_xlabel("Best screening-set $R^2$")
        finite_values = [value for value in values if np.isfinite(value)]
        left_limit = min(-0.25, min(finite_values) - 0.05) if finite_values else -0.25
        axis.set_xlim(left_limit, 1.02)
        axis.grid(axis="x", alpha=0.2)
        figure.tight_layout()
        figure.savefig(root / "linear_signal_screening_r2.png", dpi=220)
        figure.savefig(root / "linear_signal_screening_r2.pdf")
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(7, 5))
        for item in results:
            if item.get("best_screening_r2") is None:
                continue
            color = (
                "#E69F00"
                if item.get("status") == "accepted"
                and item.get("evidence_tier") == "exploratory_small_n_highdim"
                else "#0072B2"
                if item.get("status") == "accepted"
                else "#999999"
            )
            axis.scatter(
                item.get("P_raw", np.nan),
                item.get("N", np.nan),
                c=color,
                s=28,
                alpha=0.8,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Feature dimension P")
        axis.set_ylabel("Sample size N")
        minimum = 100
        maximum = max(max(item.get("P_raw", 0), item.get("N", 0)) for item in results)
        axis.plot([minimum, maximum], [minimum, maximum], color="black", linestyle=":", linewidth=1)
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(root / "dataset_dimension_map.png", dpi=220)
        figure.savefig(root / "dataset_dimension_map.pdf")
        plt.close(figure)


def load_all_metric_results(metrics_root: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(metrics_root.glob("*.json")):
        with path.open() as handle:
            result = json.load(handle)
        if isinstance(result, dict) and result.get("task_id"):
            result.setdefault("evidence_tier", "standard")
            results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task-regex", default=".*")
    args = parser.parse_args()
    root = args.root.resolve()
    processed_root = root / "processed"
    metrics_root = root / "metrics"
    metrics_root.mkdir(exist_ok=True)
    pattern = re.compile(args.task_regex)
    task_roots = sorted(
        path.parent
        for path in processed_root.glob("*/metadata.json")
        if pattern.search(path.parent.name)
    )
    for index, task_root in enumerate(task_roots, start=1):
        print(f"[{index}/{len(task_roots)}] screening {task_root.name}", flush=True)
        result = evaluate_task(task_root, metrics_root)
        print(
            f"  {result['status']} best_R2={result.get('best_screening_r2')} "
            f"elapsed={result['elapsed_seconds']:.1f}s",
            flush=True,
        )
        write_summary(root, load_all_metric_results(metrics_root))
    results = load_all_metric_results(metrics_root)
    write_summary(root, results)
    print(
        json.dumps(
            {
                "tasks": len(results),
                "accepted": sum(result.get("status") == "accepted" for result in results),
                "quality_reject": sum(
                    result.get("status") == "quality_reject" for result in results
                ),
                "low_signal_reject": sum(
                    result.get("status") == "low_signal_reject" for result in results
                ),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
