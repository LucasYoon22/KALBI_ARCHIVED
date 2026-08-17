#!/usr/bin/env python3
"""Create dataset, preprocessing, runtime, command, and caption artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from real_pgtN_common import atomic_json, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "sum": float(array.sum()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
    }


def main() -> None:
    args = parse_args()
    evaluation = args.evaluation_root
    manifest = load_json(args.split_manifest)
    if manifest.get("task_count") != 41:
        raise RuntimeError("expected 41 frozen tasks")

    catalog_rows: list[dict[str, Any]] = []
    preprocessing_rows: list[dict[str, Any]] = []
    kalbi_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    for task in manifest["tasks"]:
        task_id = task["task_id"]
        kalbi = load_json(evaluation / "metrics" / f"{task_id}_kalbi.json")
        baseline = load_json(
            evaluation / "metrics" / f"{task_id}_baselines.json"
        )
        if kalbi.get("status") != "PASS" or baseline.get("status") not in {
            "PASS",
            "PARTIAL_NONCONVEX_TIMEOUT",
        }:
            raise RuntimeError(f"incomplete result for {task_id}")
        kalbi_rows.append(kalbi)
        baseline_rows.append(baseline)
        catalog_rows.append(
            {
                "registry_index": int(task["registry_index"]),
                "task_id": task_id,
                "source": task["source"],
                "target": task["target"],
                "evidence_tier": task["evidence_tier"],
                "N": int(task["N"]),
                "P_raw": int(task["P_raw"]),
                "P_over_N": float(task["P_over_N"]),
                "license": task.get("license"),
                "source_revision": task.get("source_revision"),
                "X_sha256": task["X_sha256"],
                "y_sha256": task["y_sha256"],
                "outer_split_seed": int(task["outer_split_seed"]),
                "outer_split_sha256": task["outer_split_sha256"],
                "outer_split_group_risk": task["outer_split_group_risk"],
            }
        )
        preprocessing_rows.append(
            {
                "registry_index": int(task["registry_index"]),
                "task_id": task_id,
                "P_raw": int(kalbi["P_raw"]),
                "P_transformed": int(kalbi["P_transformed"]),
                "removed_train_constant_features": int(
                    kalbi["P_raw"] - kalbi["P_transformed"]
                ),
                "outer_train_rows": int(kalbi["outer_train_rows"]),
                "outer_test_rows": int(kalbi["outer_test_rows"]),
                "scaler_sha256": kalbi["scaler_sha256"],
                "preprocessing_path": kalbi["preprocessing_path"],
                "preprocessing_sha256": kalbi["preprocessing_sha256"],
                "shared_with_baselines": (
                    kalbi["scaler_sha256"] == baseline["scaler_sha256"]
                ),
            }
        )

    catalog_path = evaluation / "reports" / "dataset_catalog.csv"
    preprocessing_csv = (
        evaluation / "manifests" / "preprocessing_manifest.csv"
    )
    preprocessing_json = (
        evaluation / "manifests" / "preprocessing_manifest.json"
    )
    write_csv(catalog_path, catalog_rows)
    write_csv(preprocessing_csv, preprocessing_rows)
    atomic_json(
        preprocessing_json,
        {
            "status": "PASS",
            "task_count": len(preprocessing_rows),
            "all_shared_with_baselines": all(
                row["shared_with_baselines"] for row in preprocessing_rows
            ),
            "tasks": preprocessing_rows,
        },
    )

    baseline_method_seconds = {}
    for method in ("lasso", "scad", "mcp"):
        values = [
            float(row["methods"][method]["fit_seconds"])
            for row in baseline_rows
            if "fit_seconds" in row["methods"][method]
        ]
        baseline_method_seconds[method] = {
            "evaluated_task_count": len(values),
            **distribution(values),
        }
    complete_baseline_rows = [
        row for row in baseline_rows if row["status"] == "PASS"
    ]
    runtime = {
        "status": "PASS_WITH_NONCONVEX_TIMEOUTS",
        "task_count": 41,
        "complete_nonconvex_task_count": len(complete_baseline_rows),
        "nonconvex_timeout_task_count": (
            len(baseline_rows) - len(complete_baseline_rows)
        ),
        "kalbi_total_seconds": distribution(
            [float(row["runtime"]["total_seconds"]) for row in kalbi_rows]
        ),
        "kalbi_selector_seconds": distribution(
            [float(row["runtime"]["selector_seconds"]) for row in kalbi_rows]
        ),
        "kalbi_beta_path_seconds": distribution(
            [
                float(row["runtime"]["beta_path_and_selected_seconds"])
                for row in kalbi_rows
            ]
        ),
        "coordinate_descent_seconds": distribution(
            [
                float(row["runtime"]["coordinate_descent_seconds"])
                for row in kalbi_rows
            ]
        ),
        "baseline_task_seconds": distribution(
            [float(row["runtime_seconds"]) for row in complete_baseline_rows]
        ),
        "baseline_method_seconds": baseline_method_seconds,
        "maximum_kalbi_peak_gpu_memory_bytes": int(
            max(row["runtime"]["peak_gpu_memory_bytes"] for row in kalbi_rows)
        ),
        "resource_policy": {
            "kalbi_workers": 8,
            "kalbi_cd_workers_per_task": 2,
            "baseline_task_workers": {
                "initial_wave": 12,
                "redistributed_wave": 20,
            },
            "baseline_cv_processes_per_task": 3,
            "tail_cv_processes_per_task": 5,
            "blas_threads_per_process": 1,
        },
    }
    runtime_path = evaluation / "reports" / "runtime_resource_report.json"
    atomic_json(runtime_path, runtime)

    command_paths = [
        Path("/opt/supervisor-scripts/run_real_pgtN_kalbi_worker.sh"),
        Path("/opt/supervisor-scripts/run_real_pgtN_baseline_worker.sh"),
        Path("/opt/supervisor-scripts/run_real_pgtN_cd_retry_worker.sh"),
        Path("/opt/supervisor-scripts/run_real_pgtN_baseline_tail_worker.sh"),
        Path("/opt/supervisor-scripts/run_real_pgtN_lasso_recovery_worker.sh"),
    ]
    command_text: list[str] = []
    for path in command_paths:
        command_text.extend(
            [
                f"### {path}",
                f"SHA256: {sha256(path)}",
                "",
                path.read_text(encoding="utf-8"),
                "",
            ]
        )
    (evaluation / "reports" / "execution_commands.txt").write_text(
        "\n".join(command_text), encoding="utf-8"
    )

    caption = (
        "Evaluation on 41 preregistered real regression tasks with natural P>N. "
        "Each task uses one deterministic row-random 80/20 train/test split; all "
        "preprocessing and inner cross-validation are fit using training rows "
        "only. The coefficient panels compare KALBI with a full-feature bounded "
        "coordinate-descent reference under the same KALBI-selected penalty and "
        "41-point raw-lambda grid. Held-out panels show test MSE normalized by "
        "the training-mean zero predictor, with the KALBI-selected and post-hoc "
        "oracle lambdas marked. Penalty panels display the selected rho and rho' "
        "on |beta| in [0,1]. Prediction panels compare KALBI with CV-LASSO, "
        "CV-SCAD (a=3.7), and CV-MCP (gamma=3.0), tuned by a single fixed inner "
        "CV partition. KALBI and CV-LASSO are available for all 41 tasks; "
        "CV-SCAD and CV-MCP are reported on 37 tasks because four preregistered "
        "tasks exceeded the documented numerical runtime budget. The post-hoc "
        "oracle is diagnostic and is not used as the reported KALBI prediction "
        "result. Because the requested split is row-random, source groups may "
        "overlap between train and test."
    )
    (evaluation / "reports" / "paper_caption_draft.txt").write_text(
        caption + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS_WITH_NONCONVEX_TIMEOUTS",
                "dataset_catalog": str(catalog_path),
                "preprocessing_manifest": str(preprocessing_json),
                "runtime_report": str(runtime_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
