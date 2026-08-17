#!/usr/bin/env python3
"""Write the immutable command, source, environment, and artifact manifest."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy
import sklearn
import skglm
import torch

from real_pgtN_common import atomic_json, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def command_output(command: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.stdout.strip()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": sha256(path) if path.is_file() else None,
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def main() -> None:
    args = parse_args()
    evaluation = args.evaluation_root
    project = args.project_root
    scripts = sorted((evaluation / "scripts").glob("*.py"))
    operational = [
        Path("/opt/supervisor-scripts/run_real_pgtN_kalbi_worker.sh"),
        Path("/opt/supervisor-scripts/run_real_pgtN_baseline_worker.sh"),
        Path("/opt/supervisor-scripts/run_real_pgtN_cd_retry_worker.sh"),
        Path("/opt/supervisor-scripts/run_real_pgtN_baseline_tail_worker.sh"),
        Path("/opt/supervisor-scripts/run_real_pgtN_lasso_recovery_worker.sh"),
        Path("/etc/supervisor/conf.d/kalbi_real_pgtN_eval.conf"),
        Path("/etc/supervisor/conf.d/baseline_real_pgtN_eval.conf"),
        Path("/etc/supervisor/conf.d/kalbi_real_pgtN_cd_retry.conf"),
        Path("/etc/supervisor/conf.d/baseline_real_pgtN_eval_tail.conf"),
        Path("/etc/supervisor/conf.d/baseline_real_pgtN_lasso_recovery.conf"),
    ]
    beta = (
        project
        / "runs/kalbi_v8_adaptive_t10_betaonly_supportmix_v14_target3080_"
        "effbatch32_to60k_20260728/checkpoint_step043000.pt"
    )
    selector = (
        project
        / "runs/kalbi_v24_selectoronly_v14_snr0p2to50_frozenbeta_step42700_"
        "effbatch32_20260728/checkpoint_selector_step009400.pt"
    )
    result_files = sorted((evaluation / "metrics").glob("*.json"))
    path_files = sorted((evaluation / "paths").glob("*.npz"))
    baseline_files = sorted((evaluation / "baselines").glob("*.npz"))
    figure_files = sorted((evaluation / "figures").rglob("*.pdf")) + sorted(
        (evaluation / "figures").rglob("*.png")
    )
    git_status = command_output(
        ["git", "status", "--porcelain"], project
    ).splitlines()
    manifest = {
        "status": "PASS_WITH_NONCONVEX_TIMEOUTS",
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "gpu": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total",
                    "--format=csv,noheader",
                ]
            ).splitlines(),
        },
        "packages": {
            "numpy": numpy.__version__,
            "scikit_learn": sklearn.__version__,
            "skglm": skglm.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "git": {
            "head": command_output(["git", "rev-parse", "HEAD"], project),
            "dirty_entry_count": len(git_status),
            "status_porcelain_first_200": git_status[:200],
        },
        "contracts": {
            "outer_split": "one deterministic row-random 80/20 split",
            "outer_split_master_seed": 20260728,
            "kalbi_path_raw_lambda": "logspace(-4, 0, 41)",
            "kalbi_refinement_sweeps": 10,
            "kalbi_refinement_contract": "gamma_t = eta_t * raw_lambda",
            "coordinate_descent": {
                "features": "all train-nonconstant transformed features",
                "max_sweeps": 10000,
                "retries": 4,
                "solver_tolerance": 2e-4,
                "audit_tolerance": 5e-3,
                "starts": ["zero", "KALBI", "descending continuation"],
            },
            "coordinate_descent_retry": {
                "task_9": {
                    "max_sweeps": 30000,
                    "retries": 6,
                    "solver_tolerance": 1e-4,
                },
                "task_18_first_retry": {
                    "max_sweeps": 30000,
                    "retries": 6,
                    "solver_tolerance": 1e-4,
                },
                "task_18_second_retry": {
                    "max_sweeps": 60000,
                    "retries": 8,
                    "solver_tolerance": 5e-5,
                },
                "audit_tolerance": 5e-3,
                "archived_initial_results": str(
                    evaluation / "reports" / "cd_retry_archive"
                ),
            },
            "baseline_inner_cv": {
                "task_workers": {
                    "initial_wave": 12,
                    "redistributed_wave": 20,
                },
                "folds": "5 if outer-train N >= 100, otherwise 3",
                "parallel_fold_workers_per_task": 3,
                "selection": "minimum mean validation MSE",
                "lambda_ratio_grid": "logspace(0, -4, 100)",
                "lasso_tolerance": 1e-5,
                "lasso_max_iterations": 10000,
                "scad_a": 3.7,
                "mcp_gamma": 3.0,
                "scad_mcp_tolerance": 1e-5,
                "scad_mcp_max_epochs": 10000,
                "completed_task_count": 37,
                "numerical_timeout_task_indices": [13, 14, 15, 18],
                "timeout_policy": (
                    "retain tasks; recover the already frozen CV-LASSO choice; "
                    "report SCAD/MCP as missing rather than substituting a "
                    "different local optimizer"
                ),
                "timeout_archive": str(
                    evaluation / "reports" / "nonconvex_timeout_archive"
                ),
            },
            "preprocessing": (
                "fit on outer-training rows only; exact train-constant removal; "
                "feature and response centering/scaling"
            ),
        },
        "checkpoints": {
            "beta": file_record(beta),
            "selector": file_record(selector),
        },
        "pinned_manifests": {
            name: file_record(evaluation / "manifests" / name)
            for name in (
                "dataset_registry.json",
                "split_manifest.json",
                "checkpoint_compatibility.json",
                "nonconvex_baseline_validation.json",
                "final_validation.json",
            )
        },
        "sources": [file_record(path) for path in scripts],
        "operational_files": [file_record(path) for path in operational],
        "artifacts": {
            "metric_json_count": len(result_files),
            "kalbi_path_count": len(path_files),
            "baseline_path_count": len(baseline_files),
            "figure_count": len(figure_files),
            "figure_manifest": file_record(
                evaluation / "figures" / "figure_manifest.json"
            ),
        },
        "process_status_after_evaluation": command_output(
            ["supervisorctl", "status"]
        ).splitlines(),
    }
    atomic_json(args.output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
