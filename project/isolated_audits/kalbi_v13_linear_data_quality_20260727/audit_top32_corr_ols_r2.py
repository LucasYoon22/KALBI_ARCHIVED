#!/usr/bin/env python3
"""Top-32 Pearson-correlation feature selection followed by exact OLS."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np


def select_top_correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    top_k: int,
    feature_chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    y64 = np.asarray(y, dtype=np.float64)
    y_centered = y64 - y64.mean()
    y_norm = np.linalg.norm(y_centered)
    top_values = np.full(top_k, -np.inf, dtype=np.float64)
    top_indices = np.full(top_k, -1, dtype=np.int64)
    for left in range(0, x.shape[1], feature_chunk):
        right = min(x.shape[1], left + feature_chunk)
        block = x[:, left:right].astype(np.float64)
        block -= block.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(block, axis=0)
        values = np.abs(block.T @ y_centered) / np.maximum(norms * y_norm, 1e-30)
        indices = np.arange(left, right, dtype=np.int64)
        combined_values = np.concatenate((top_values, values))
        combined_indices = np.concatenate((top_indices, indices))
        chosen = np.argpartition(combined_values, -top_k)[-top_k:]
        order = chosen[
            np.lexsort((combined_indices[chosen], -combined_values[chosen]))
        ]
        top_values = combined_values[order]
        top_indices = combined_indices[order]
    valid = top_indices >= 0
    return top_indices[valid], top_values[valid]


def fit_ols(
    x_train: np.ndarray,
    y_train: np.ndarray,
    indices: np.ndarray,
    *,
    x_eval: np.ndarray | None = None,
    y_eval: np.ndarray | None = None,
) -> dict[str, float | int]:
    x_train64 = x_train[:, indices].astype(np.float64)
    y_train64 = np.asarray(y_train, dtype=np.float64)
    x_mean = x_train64.mean(axis=0, keepdims=True)
    y_mean = y_train64.mean()
    design = x_train64 - x_mean
    target = y_train64 - y_mean
    beta, _, rank, singular = np.linalg.lstsq(design, target, rcond=None)
    train_prediction = design @ beta
    train_sse = float(np.sum((target - train_prediction) ** 2))
    train_sst = float(np.sum(target**2))
    result: dict[str, float | int] = {
        "ols_rank": int(rank),
        "ols_condition_number": (
            float(singular[0] / singular[-1])
            if singular.size and singular[-1] > 0
            else float("inf")
        ),
        "ols_train_r2": 1.0 - train_sse / max(train_sst, 1e-30),
        "ols_train_mse": train_sse / max(len(y_train64), 1),
        "ols_beta_l2": float(np.linalg.norm(beta)),
        "ols_beta_max_abs": float(np.max(np.abs(beta))),
    }
    if x_eval is not None and y_eval is not None:
        eval_target = np.asarray(y_eval, dtype=np.float64)
        eval_prediction = (x_eval[:, indices].astype(np.float64) - x_mean) @ beta + y_mean
        eval_sse = float(np.sum((eval_target - eval_prediction) ** 2))
        eval_sst = float(np.sum((eval_target - eval_target.mean()) ** 2))
        result["ols_eval_r2"] = 1.0 - eval_sse / max(eval_sst, 1e-30)
        result["ols_eval_mse"] = eval_sse / max(len(eval_target), 1)
    return result


def deterministic_split(task_hash: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    seed = int.from_bytes(
        hashlib.sha256((task_hash + ":top32-ols-split-v1").encode()).digest()[:8],
        "little",
    )
    permutation = np.random.default_rng(seed).permutation(n)
    train_n = n // 2
    return permutation[:train_n], permutation[train_n:]


def summarize(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    quantile_values = np.quantile(
        finite, [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0]
    )
    names = ("min", "q05", "q10", "q25", "median", "q75", "q90", "q95", "max")
    result: dict[str, float | int] = {
        name: float(value) for name, value in zip(names, quantile_values)
    }
    result.update(
        {
            "count": int(values.size),
            "finite_count": int(finite.size),
            "mean": float(finite.mean()),
            "std": float(finite.std()),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--feature-chunk", type=int, default=4096)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(args.manifest).open(newline="") as handle:
        manifest = list(csv.DictReader(handle))

    partial_path = output_dir / "top32_corr_ols_task_metrics.partial.csv"
    rows: list[dict[str, object]] = []
    if partial_path.exists():
        with partial_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        print(f"resuming from {len(rows)} completed tasks", flush=True)
    completed_hashes = {str(row["task_hash"]) for row in rows}
    partial_handle = partial_path.open("a", newline="")
    partial_writer: csv.DictWriter[str] | None = None
    started = time.perf_counter()
    for task_number, task in enumerate(manifest, start=1):
        if task["task_hash"] in completed_hashes:
            continue
        with np.load(task["absolute_path"], allow_pickle=False) as archive:
            x = archive["X"][0]
            y = archive["y"][0].astype(np.float32)
        n, p = x.shape
        selected, abs_corr = select_top_correlation(
            x, y, top_k=min(args.top_k, p), feature_chunk=args.feature_chunk
        )
        full = fit_ols(x, y, selected)
        train_index, test_index = deterministic_split(task["task_hash"], n)
        selected_split, _ = select_top_correlation(
            x[train_index],
            y[train_index],
            top_k=min(args.top_k, p),
            feature_chunk=args.feature_chunk,
        )
        split = fit_ols(
            x[train_index],
            y[train_index],
            selected_split,
            x_eval=x[test_index],
            y_eval=y[test_index],
        )
        row: dict[str, object] = {
            "task_hash": task["task_hash"],
            "audit_stratum": task["audit_stratum"],
            "outcome_source": task["outcome_source"],
            "n": n,
            "p": p,
            "top_k": int(selected.size),
            "top_abs_corr_max": float(abs_corr[0]),
            "top_abs_corr_min": float(abs_corr[-1]),
            "full_ols_r2": float(full["ols_train_r2"]),
            "full_ols_rank": int(full["ols_rank"]),
            "full_ols_condition_number": float(full["ols_condition_number"]),
            "full_ols_beta_l2": float(full["ols_beta_l2"]),
            "split_train_ols_r2": float(split["ols_train_r2"]),
            "split_test_ols_r2": float(split["ols_eval_r2"]),
            "split_ols_rank": int(split["ols_rank"]),
            "split_ols_condition_number": float(split["ols_condition_number"]),
        }
        rows.append(row)
        if partial_writer is None:
            partial_writer = csv.DictWriter(partial_handle, fieldnames=list(row))
            if partial_path.stat().st_size == 0:
                partial_writer.writeheader()
        partial_writer.writerow(row)
        partial_handle.flush()
        if task_number % 25 == 0:
            print(
                f"audited {task_number}/{len(manifest)} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    partial_handle.close()

    rows = []
    with partial_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(manifest):
        raise RuntimeError(
            f"partial result count mismatch: {len(rows)} != {len(manifest)}"
        )

    csv_path = output_dir / "top32_corr_ols_task_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {
        "method": {
            "selection": "top-32 absolute Pearson correlation",
            "fit": "centered float64 numpy.linalg.lstsq OLS with intercept-equivalent centering",
            "primary_metric": "full-task in-sample R2",
            "auxiliary_metric": "deterministic 50/50 split, train-only selection and OLS, held-out R2",
            "top_k": args.top_k,
            "sample_selection": "frozen 600-task audit manifest; no task replacement",
        },
        "runtime_seconds": time.perf_counter() - started,
        "strata": {},
    }
    for stratum in ("linear_p", "linear_n", "tabicl"):
        selected_rows = [row for row in rows if row["audit_stratum"] == stratum]
        summary["strata"][stratum] = {
            "full_ols_r2": summarize(
                np.asarray([row["full_ols_r2"] for row in selected_rows])
            ),
            "split_train_ols_r2": summarize(
                np.asarray([row["split_train_ols_r2"] for row in selected_rows])
            ),
            "split_test_ols_r2": summarize(
                np.asarray([row["split_test_ols_r2"] for row in selected_rows])
            ),
            "rank_deficient_count": int(
                sum(int(row["full_ols_rank"]) < int(row["top_k"]) for row in selected_rows)
            ),
        }
    (output_dir / "top32_corr_ols_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
