#!/usr/bin/env python3
"""Compute train-only pre-filter absolute-correlation heatmap inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


PROJECT = Path("/users/minsuk/fm-lm-lambda")
SOURCE_ROOT = PROJECT / "evaluations/real_pgtN_onesplit_kalbi_latest_20260728"
FILTERED_ROOT = SOURCE_ROOT / "kalbi_lasso_corr095_t100_20260728"
SCRIPT_ROOT = SOURCE_ROOT / "scripts"
for entry in (str(PROJECT), str(SCRIPT_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from real_pgtN_common import (  # noqa: E402
    atomic_json,
    atomic_npz,
    fit_preprocessing,
    iter_selected_tasks,
    load_split_manifest,
    load_x,
    parse_task_indices,
    select_rows,
    sha256,
)


THRESHOLDS = (0.90, 0.95, 0.99)
CONTRACT = "OUTER_TRAIN_PRE_FILTER_ABS_CORR_GREEDY_VIEW_V1"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def greedy_order(corr: np.ndarray) -> np.ndarray:
    absolute = np.abs(np.asarray(corr, dtype=np.float64))
    count = int(absolute.shape[0])
    start = int(np.argmax(absolute.sum(axis=1)))
    order = [start]
    used = np.zeros(count, dtype=bool)
    used[start] = True
    current = start
    for _ in range(1, count):
        scores = absolute[current].copy()
        scores[used] = -np.inf
        current = int(np.argmax(scores))
        order.append(current)
        used[current] = True
    return np.asarray(order, dtype=np.int64)


def greedy_chain(corr_abs: np.ndarray, count: int) -> np.ndarray:
    p = int(corr_abs.shape[0])
    count = min(int(count), p)
    start = int(np.argmax(corr_abs.sum(axis=1)))
    selected = [start]
    used = np.zeros(p, dtype=bool)
    used[start] = True
    current = start
    for _ in range(1, count):
        scores = corr_abs[current].copy()
        scores[used] = -np.inf
        current = int(np.argmax(scores))
        selected.append(current)
        used[current] = True
    return np.asarray(selected, dtype=np.int64)


def pair_statistics(
    corr_abs: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    p = int(corr_abs.shape[0])
    pair_count = p * (p - 1) // 2
    counts = {threshold: 0 for threshold in THRESHOLDS}
    degree_095 = np.zeros(p, dtype=np.int64)
    total = 0.0
    maximum = 0.0
    for index in range(p - 1):
        row = corr_abs[index, index + 1 :]
        total += float(row.sum(dtype=np.float64))
        maximum = max(maximum, float(row.max(initial=0.0)))
        for threshold in THRESHOLDS:
            count = int(np.count_nonzero(row >= threshold))
            counts[threshold] += count
            if threshold == 0.95 and count:
                neighbors = np.flatnonzero(row >= threshold) + index + 1
                degree_095[index] += neighbors.size
                degree_095[neighbors] += 1
    sample_count = min(1024, p)
    sample_indices = np.unique(
        np.rint(np.linspace(0, p - 1, sample_count)).astype(np.int64)
    )
    sampled = corr_abs[np.ix_(sample_indices, sample_indices)]
    sampled_off = sampled[np.triu_indices(sample_indices.size, 1)]
    statistics = {
        "pair_count": int(pair_count),
        "mean_abs_corr": float(total / max(pair_count, 1)),
        "max_abs_corr": maximum,
        "sample_quantile_feature_count": int(sample_indices.size),
        "sample_abs_corr_quantiles": {
            "q50": float(np.quantile(sampled_off, 0.50)),
            "q90": float(np.quantile(sampled_off, 0.90)),
            "q95": float(np.quantile(sampled_off, 0.95)),
            "q99": float(np.quantile(sampled_off, 0.99)),
        },
        "threshold_pair_counts": {
            f"ge_{threshold:.2f}": int(counts[threshold])
            for threshold in THRESHOLDS
        },
        "threshold_pair_fractions": {
            f"ge_{threshold:.2f}": float(
                counts[threshold] / max(pair_count, 1)
            )
            for threshold in THRESHOLDS
        },
        "degree_ge_0p95": {
            "isolated_feature_count": int(np.count_nonzero(degree_095 == 0)),
            "nonisolated_feature_count": int(np.count_nonzero(degree_095 > 0)),
            "median": float(np.median(degree_095)),
            "p90": float(np.quantile(degree_095, 0.90)),
            "p99": float(np.quantile(degree_095, 0.99)),
            "max": int(degree_095.max(initial=0)),
        },
    }
    return statistics, sample_indices


def evaluate_task(
    task: dict[str, Any],
    *,
    output_root: Path,
    max_heatmap_features: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    task_id = task["task_id"]
    metric_path = output_root / "metrics" / f"{task_id}.json"
    if metric_path.exists():
        metric = load_json(metric_path)
        array_path = Path(metric["array_output"])
        if (
            metric.get("status") == "PASS"
            and array_path.is_file()
            and sha256(array_path) == metric["array_sha256"]
        ):
            return metric
        raise RuntimeError(f"invalid existing result for {task_id}")

    with np.load(task["outer_split_path"], allow_pickle=False) as archive:
        train_index = np.asarray(archive["train_indices"], dtype=np.int64)
    x_store = load_x(Path(task["X_path"]))
    y_store = np.load(task["y_path"], mmap_mode="r")
    x_train_raw = select_rows(x_store, train_index)
    y_train_raw = np.asarray(y_store[train_index], dtype=np.float64)
    x_train, _, scaler = fit_preprocessing(x_train_raw, y_train_raw)
    x = np.ascontiguousarray(x_train, dtype=np.float32)
    n, p = x.shape
    corr_abs = np.abs((x.T @ x) / float(n)).astype(np.float32, copy=False)
    np.fill_diagonal(corr_abs, 1.0)
    pair_stats, global_indices = pair_statistics(corr_abs)

    greedy_indices = greedy_chain(corr_abs, max_heatmap_features)
    global_indices = global_indices[
        np.unique(
            np.rint(
                np.linspace(
                    0,
                    global_indices.size - 1,
                    min(max_heatmap_features, global_indices.size),
                )
            ).astype(np.int64)
        )
    ]
    global_corr = corr_abs[np.ix_(global_indices, global_indices)]
    global_order = greedy_order(global_corr)
    global_indices = global_indices[global_order]
    greedy_corr = corr_abs[np.ix_(greedy_indices, greedy_indices)]
    global_corr = corr_abs[np.ix_(global_indices, global_indices)]

    filtered_metric = load_json(
        FILTERED_ROOT / "metrics" / f"{task_id}.json"
    )
    if filtered_metric["scaler_sha256"] != scaler["scaler_sha256"]:
        raise RuntimeError("filtered/pre-filter scaler mismatch")
    array_output = output_root / "arrays" / f"{task_id}.npz"
    atomic_npz(
        array_output,
        greedy_feature_indices=greedy_indices,
        greedy_abs_corr=greedy_corr,
        global_feature_indices=global_indices,
        global_abs_corr=global_corr,
        train_indices=train_index,
    )
    metric = {
        "status": "PASS",
        "contract": CONTRACT,
        "registry_index": int(task["registry_index"]),
        "task_id": task_id,
        "N": int(task["N"]),
        "outer_train_rows": int(n),
        "P_before_constant_filter": int(np.asarray(scaler["active"]).size),
        "P_before_correlation_filter": int(p),
        "P_after_correlation_filter": int(
            filtered_metric["P_after_correlation_filter"]
        ),
        "removed_fraction": float(
            1.0
            - filtered_metric["P_after_correlation_filter"] / max(p, 1)
        ),
        "scaler_sha256": scaler["scaler_sha256"],
        "split_sha256": task["outer_split_sha256"],
        "heatmap_feature_count": int(greedy_indices.size),
        "greedy_contract": (
            "start=max row-sum |corr|; repeatedly choose most correlated "
            "unused feature to current feature"
        ),
        "global_sample_contract": (
            "deterministic evenly-spaced original feature indices followed "
            "by the existing greedy |corr| ordering"
        ),
        **pair_stats,
        "array_output": str(array_output),
        "array_sha256": sha256(array_output),
        "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(metric_path, metric)
    return metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-indices", default="all")
    parser.add_argument("--max-heatmap-features", type=int, default=128)
    args = parser.parse_args()
    manifest = load_split_manifest(
        SOURCE_ROOT / "manifests/split_manifest.json"
    )
    indices = parse_task_indices(args.task_indices, manifest["task_count"])
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for task in iter_selected_tasks(manifest, indices):
        try:
            metric = evaluate_task(
                task,
                output_root=args.output_root,
                max_heatmap_features=args.max_heatmap_features,
            )
            results.append(metric)
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "registry_index": metric["registry_index"],
                        "task_id": metric["task_id"],
                        "P_before": metric["P_before_correlation_filter"],
                        "P_after": metric["P_after_correlation_filter"],
                        "max_abs_corr": metric["max_abs_corr"],
                        "pair_fraction_ge_0p95": metric[
                            "threshold_pair_fractions"
                        ]["ge_0.95"],
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
        "contract": CONTRACT,
        "task_indices": indices,
        "success_count": len(results),
        "failure_count": len(failures),
        "failures": failures,
    }
    tag = "_".join(map(str, indices))
    atomic_json(output_root := args.output_root / "reports" / f"worker_{tag}.json", report)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
