#!/usr/bin/env python3
"""Recompute one conditional path per task and rank summary families."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_sure_divergence_study as study


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64)
    output[~np.isfinite(output)] = 0.0
    return output


def evaluate(
    baseline: np.ndarray,
    candidate: np.ndarray,
    addition: np.ndarray | None,
    target: np.ndarray,
    task_ids: np.ndarray,
    strata: dict[str, np.ndarray],
) -> dict[str, object]:
    train, test = study.split_indices(task_ids)
    blocks = [baseline, candidate]
    if addition is not None:
        blocks.append(addition)
    features = np.concatenate(blocks, axis=1)
    model = ExtraTreesRegressor(
        n_estimators=400,
        min_samples_leaf=3,
        max_features=0.5,
        n_jobs=8,
        random_state=20260727,
    )
    model.fit(features[train], target[train])
    prediction = model.predict(features[test])
    observed = target[test]
    slope, intercept = np.polyfit(prediction, observed, deg=1)
    rng = np.random.default_rng(20260727)
    bootstrap_r2 = []
    for _ in range(500):
        positions = rng.integers(0, observed.size, observed.size)
        if np.var(observed[positions]) > 1e-12:
            bootstrap_r2.append(
                r2_score(observed[positions], prediction[positions])
            )
    subgroup_rmse = {}
    for name, labels in strata.items():
        test_labels = labels[test]
        for level in np.unique(test_labels):
            selected = test_labels == level
            if selected.sum() >= 5:
                subgroup_rmse[f"{name}={level}"] = float(
                    math.sqrt(
                        mean_squared_error(
                            observed[selected], prediction[selected]
                        )
                    )
                )
    return {
        "r2": float(r2_score(observed, prediction)),
        "r2_bootstrap_low": float(np.quantile(bootstrap_r2, 0.025)),
        "r2_bootstrap_high": float(np.quantile(bootstrap_r2, 0.975)),
        "rmse": float(math.sqrt(mean_squared_error(observed, prediction))),
        "mae": float(np.mean(np.abs(observed - prediction))),
        "spearman": float(spearmanr(observed, prediction).statistic),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "worst_stratum_rmse": float(max(subgroup_rmse.values())),
        "subgroup_rmse": subgroup_rmse,
        "test_count": int(test.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tasks", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026072701)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    study_dir = Path(args.study_dir)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    target_rows = read_csv(study_dir / "divergence_targets.csv")
    raw_target_by_task = {
        int(row["task_id"]): float(row["conditional_divergence_target"])
        for row in target_rows
    }
    tasks = [
        study.make_task(
            task_id,
            args.seed + task_id * 1009,
            non_gaussian=task_id >= int(round(args.tasks * 0.75)),
        )
        for task_id in range(args.tasks)
    ]
    selector = study.load_v23_selector(Path(args.checkpoint))
    derivatives = study.penalty_derivatives(torch.device("cpu"))
    baseline_rows: list[np.ndarray] = []
    candidate_rows: list[np.ndarray] = []
    cheap_rows: list[dict[str, float]] = []
    dynamic_rows: list[dict[str, float]] = []
    target = []
    task_ids = []
    for task in tasks:
        family_index = task.task_id % len(study.PENALTY_FAMILIES)
        ratio_index = (
            task.task_id // len(study.PENALTY_FAMILIES)
        ) % len(study.LAMBDA_RATIOS)
        family = study.PENALTY_FAMILIES[family_index]
        derivative = derivatives[family]
        lambda_max = float(
            np.max(np.abs(task.x_train.T @ task.y_train / task.n))
        )
        raw_lambda = float(
            np.clip(
                study.LAMBDA_RATIOS[ratio_index]
                * lambda_max
                / float(derivative[0]),
                1e-6,
                10.0,
            )
        )
        candidate_rows.append(
            np.asarray(
                [
                    math.log(max(raw_lambda, 1e-12)),
                    math.log(max(study.LAMBDA_RATIOS[ratio_index], 1e-12)),
                    float(family_index),
                    *study.penalty_descriptor(
                        derivative,
                        study.canonical_grid(
                            device=torch.device("cpu"), dtype=study.DTYPE
                        ),
                    ),
                ],
                dtype=np.float64,
            )
        )
        x = torch.from_numpy(task.x_train).to(dtype=study.DTYPE)
        y = torch.from_numpy(task.y_train).to(dtype=study.DTYPE)
        gram = x.T @ x / float(task.n)
        eta = (
            0.9
            / torch.linalg.eigvalsh(gram).amax().clamp_min(1e-8).detach()
        )
        _, dynamic = study.solve_pg(
            x,
            y,
            derivative,
            raw_lambda,
            steps=10,
            collect=True,
            gram=gram,
            eta=eta,
        )
        baseline_rows.append(study.v23_task_code(selector, task))
        cheap_rows.append(study.task_summaries(task))
        dynamic_rows.append(dynamic)
        normalized_df = max(raw_target_by_task[task.task_id], 0.0) / float(task.n)
        target.append(math.log1p(normalized_df))
        task_ids.append(task.task_id)

    baseline = finite(np.stack(baseline_rows))
    candidate = finite(np.stack(candidate_rows))
    target_array = np.asarray(target)
    task_id_array = np.asarray(task_ids)
    strata = {
        "dimension": np.asarray(
            ["high" if task.p > task.n else "low" for task in tasks]
        ),
        "error": np.asarray(
            ["gaussian" if task.error_family == "gaussian" else "non_gaussian"
             for task in tasks]
        ),
        "snr": np.asarray(
            [
                "low" if task.snr < 1.0 else "medium" if task.snr < 10.0 else "high"
                for task in tasks
            ]
        ),
    }
    cheap_names = sorted(cheap_rows[0])
    dynamic_names = sorted(dynamic_rows[0])
    cheap_matrix = finite(
        np.asarray([[row[name] for name in cheap_names] for row in cheap_rows])
    )
    dynamic_matrix = finite(
        np.asarray([[row[name] for name in dynamic_names] for row in dynamic_rows])
    )
    # ratio=1 deliberately places at least one coordinate on the zero-KKT
    # threshold. Ordinary derivatives are undefined there, so keep those tasks
    # as boundary diagnostics rather than fitting an a.e. SURE surrogate to them.
    smooth = np.exp(candidate[:, 1]) < 1.0 - 1e-10
    excluded_boundary_tasks = int((~smooth).sum())
    baseline = baseline[smooth]
    candidate = candidate[smooth]
    cheap_matrix = cheap_matrix[smooth]
    dynamic_matrix = dynamic_matrix[smooth]
    target_array = target_array[smooth]
    task_id_array = task_id_array[smooth]
    strata = {name: values[smooth] for name, values in strata.items()}

    group_predicates = {
        "solver_independent_task": lambda name: False,
        "prox_local_sensitivity": lambda name: any(
            token in name for token in ("pprime", "denom", "rho2")
        ),
        "threshold_and_branch_margin": lambda name: any(
            token in name
            for token in ("threshold_margin", "branch_margin", "zero_fraction", "saturated")
        ),
        "support_stability": lambda name: any(
            token in name
            for token in ("active_fraction", "entry_fraction", "exit_fraction", "jaccard")
        ),
        "optimization_state": lambda name: any(
            token in name
            for token in ("beta_abs", "delta_abs", "gradient_abs", "kkt", "_eta", "_gamma")
        ),
        "restricted_hessian": lambda name: name.startswith("active_hessian")
        or name == "restricted_df_proxy"
        or name == "active_support_size",
    }
    rows: list[dict[str, object]] = []
    baseline_metric = evaluate(
        baseline, candidate, None, target_array, task_id_array, strata
    )
    rows.append(
        {
            "group": "v23_baseline",
            "width": baseline.shape[1] + candidate.shape[1],
            **baseline_metric,
            "delta_r2": 0.0,
            "delta_rmse": 0.0,
        }
    )
    for group, predicate in group_predicates.items():
        if group == "solver_independent_task":
            matrix = cheap_matrix
            names = cheap_names
        else:
            selected = [i for i, name in enumerate(dynamic_names) if predicate(name)]
            matrix = dynamic_matrix[:, selected]
            names = [dynamic_names[i] for i in selected]
        metric = evaluate(
            baseline, candidate, matrix, target_array, task_id_array, strata
        )
        rows.append(
            {
                "group": group,
                "width": matrix.shape[1],
                **metric,
                "delta_r2": metric["r2"] - baseline_metric["r2"],
                "delta_rmse": metric["rmse"] - baseline_metric["rmse"],
            }
        )
        (output / f"{group}_features.json").write_text(
            json.dumps(names, indent=2)
        )
    all_metric = evaluate(
        baseline,
        candidate,
        np.concatenate((cheap_matrix, dynamic_matrix), axis=1),
        target_array,
        task_id_array,
        strata,
    )
    rows.append(
        {
            "group": "all_candidate_summaries",
            "width": cheap_matrix.shape[1] + dynamic_matrix.shape[1],
            **all_metric,
            "delta_r2": all_metric["r2"] - baseline_metric["r2"],
            "delta_rmse": all_metric["rmse"] - baseline_metric["rmse"],
        }
    )
    study.write_csv(output / "summary_group_ablation.csv", rows)
    (output / "summary_group_ablation.json").write_text(
        json.dumps(
            {
                "excluded_zero_kkt_boundary_tasks": excluded_boundary_tasks,
                "eligible_smooth_tasks": int(smooth.sum()),
                "target": "log1p(max(conditional_df,0)/N)",
                "rows": rows,
            },
            indent=2,
        )
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
