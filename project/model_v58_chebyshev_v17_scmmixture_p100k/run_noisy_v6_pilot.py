#!/usr/bin/env python3
"""Generate and validate a parallel random-shape noisy TabICLv2 pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from model_v58_chebyshev_v17_scmmixture_p100k import (
    rolling_producer_tabicl_noisy as noisy,
)


COMMON_GENERATOR_ARGS = (
    "--batch-size", "1",
    "--threads", "4",
    "--corr-device", "cpu",
    "--n-min", "32",
    "--n-max", "20000",
    "--main-n-min", "200",
    "--small-n-max", "199",
    "--small-n-probability", "0.10",
    "--p-min", "50",
    "--p-max", "100000",
    "--max-cells", "24000000",
    "--p-median", "10000",
    "--p-log-sigma", "0.75",
    "--raw-n-median", "5700",
    "--n-log-sigma", "0.70",
    "--direct-p-cap", "4096",
    "--core-oversample-factor", "1.5",
    "--corr-threshold", "0.95",
    "--hidden-dim", "256",
    "--gaussian-dims", "16",
    "--input-noise-variance-min", "0.001",
    "--input-noise-variance-max", "1.0",
    "--output-noise-variance-min", "0.001",
    "--output-noise-variance-max", "0.25",
    "--clip-value", "10.0",
    "--max-retries", "30",
    "--response-noise-exp-scale", "1.0",
    "--response-noise-sigma-cap", "3.0",
    "--filter-unpredictable",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--tasks-per-worker", type=int, default=2)
    parser.add_argument("--first-core", type=int, default=64)
    parser.add_argument("--cores-per-worker", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=420_807_355)
    parser.add_argument("--seed-stride", type=int, default=50_000_019)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def launch(args: argparse.Namespace) -> None:
    if args.out_root.exists() and any(args.out_root.iterdir()):
        raise RuntimeError(f"pilot output root is not empty: {args.out_root}")
    args.out_root.mkdir(parents=True, exist_ok=True)
    generator = (
        args.project_dir
        / "generate_tabiclv2_graph_scm_mlp512_groupdrop095_noisy_y_v6.py"
    )
    processes: list[tuple[int, subprocess.Popen[Any], Any]] = []
    try:
        for worker_index in range(args.workers):
            seed = args.base_seed + args.seed_stride * worker_index
            first = args.first_core + args.cores_per_worker * worker_index
            cores = range(first, first + args.cores_per_worker)
            worker_root = args.out_root / f"worker_{worker_index:02d}"
            worker_root.mkdir(parents=True, exist_ok=False)
            log_handle = (args.out_root / f"worker_{worker_index:02d}.log").open(
                "w", encoding="utf-8"
            )
            command = [
                "taskset", "-c", ",".join(str(core) for core in cores),
                args.python, "-u", str(generator),
                "--out-dir", str(worker_root),
                "--num-batches", str(args.tasks_per_worker),
                "--start-index", "0",
                "--seed", str(seed),
                *COMMON_GENERATOR_ARGS,
            ]
            env = dict(os.environ)
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": "",
                    "PYTHONHASHSEED": str(seed),
                    "OMP_NUM_THREADS": str(args.cores_per_worker),
                    "MKL_NUM_THREADS": str(args.cores_per_worker),
                    "OPENBLAS_NUM_THREADS": str(args.cores_per_worker),
                    "NUMEXPR_NUM_THREADS": str(args.cores_per_worker),
                }
            )
            process = subprocess.Popen(
                command,
                cwd=args.project_dir,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((worker_index, process, log_handle))

        failures: list[tuple[int, int]] = []
        for worker_index, process, log_handle in processes:
            return_code = process.wait()
            log_handle.close()
            if return_code != 0:
                failures.append((worker_index, return_code))
        if failures:
            raise RuntimeError(f"pilot generator failures: {failures}")
    finally:
        for _, process, log_handle in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
            if not log_handle.closed:
                log_handle.close()


def validate(args: argparse.Namespace) -> dict[str, Any]:
    noisy.configure_noisy_runtime()
    tasks: list[dict[str, Any]] = []
    for worker_root in sorted(args.out_root.glob("worker_*")):
        if not worker_root.is_dir():
            continue
        manifest = json.loads((worker_root / "manifest.json").read_text())
        for entry in manifest["files"]:
            npz_path = worker_root / entry["path"]
            summary = noisy.validate_noisy_generated_npz(str(npz_path), entry)
            single_shard_manifest = dict(manifest)
            single_shard_manifest["num_datasets"] = 1
            single_shard_manifest["files"] = [entry]
            semantic = noisy.validate_noisy_generator_contract(
                "tabicl_v2", single_shard_manifest, entry, summary
            )
            tasks.append(
                {
                    "path": str(npz_path),
                    "n": int(summary["n"]),
                    "p": int(summary["p"]),
                    "bytes": int(npz_path.stat().st_size),
                    "max_abs_feature_correlation": float(summary["max_abs_corr"]),
                    "family": semantic["response_noise_family"],
                    "family_id": int(semantic["response_noise_family_id"]),
                    "sigma_raw": float(semantic["response_noise_sigma_raw"]),
                    "sigma": float(semantic["response_noise_sigma"]),
                    "latent_noisy_correlation": float(
                        semantic["latent_signal_noisy_correlation"]
                    ),
                    "latent_noisy_r2_ceiling": float(
                        semantic["latent_signal_noisy_r2_ceiling"]
                    ),
                    "reconstruction_max_abs_error": float(
                        semantic["noisy_response_reconstruction_max_abs_error"]
                    ),
                }
            )
    expected = args.workers * args.tasks_per_worker
    if len(tasks) != expected:
        raise RuntimeError(f"validated {len(tasks)} tasks; expected {expected}")

    def quantiles(key: str) -> dict[str, float]:
        values = np.asarray([task[key] for task in tasks], dtype=np.float64)
        return {
            "min": float(np.min(values)),
            "p10": float(np.quantile(values, 0.10)),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "p90": float(np.quantile(values, 0.90)),
            "max": float(np.max(values)),
        }

    family_counts = {
        family: sum(task["family"] == family for task in tasks)
        for family, _ in noisy.NOISE_FAMILIES
    }
    audit = {
        "contract": noisy.DATASET_CONTRACT,
        "noise_contract": noisy.RESPONSE_NOISE_CONTRACT,
        "tasks": len(tasks),
        "total_bytes": sum(task["bytes"] for task in tasks),
        "family_counts": family_counts,
        "n": quantiles("n"),
        "p": quantiles("p"),
        "sigma_raw": quantiles("sigma_raw"),
        "sigma": quantiles("sigma"),
        "latent_noisy_correlation": quantiles("latent_noisy_correlation"),
        "latent_noisy_r2_ceiling": quantiles("latent_noisy_r2_ceiling"),
        "max_abs_feature_correlation": quantiles(
            "max_abs_feature_correlation"
        ),
        "reconstruction_max_abs_error": quantiles(
            "reconstruction_max_abs_error"
        ),
        "task_records": tasks,
    }
    audit_path = args.out_root / "pilot_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def main() -> None:
    args = parse_args()
    if not args.validate_only:
        launch(args)
    audit = validate(args)
    print(json.dumps({key: value for key, value in audit.items() if key != "task_records"}, indent=2))


if __name__ == "__main__":
    main()
