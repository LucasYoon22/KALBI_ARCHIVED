#!/usr/bin/env python3
"""Freeze five unbiased unseen high-dimensional TabICLv2 V11 tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable

import numpy as np

from exp122_shape_sampler import SHAPE_SAMPLER_NAME, sample_shape_pair


PROTOCOL_VERSION = "kalbi_paper_unseen_highdim5_freeze_v1"
DEFAULT_ROOT_SEED = 2_026_072_401
SEED_STRIDE = 1_000_003
MODULUS = 2**32 - 1

GENERATOR_ARGUMENTS = {
    "n_min": 32,
    "n_max": 40_000,
    "main_n_min": 200,
    "small_n_max": 199,
    "small_n_probability": 0.10,
    "p_min": 50,
    "p_max": 100_000,
    "max_cells": 24_000_000,
    "p_median": 10_000.0,
    "p_log_sigma": 0.75,
    "raw_n_median": 5_700.0,
    "n_log_sigma": 0.70,
    "scm_core_cap": 2_048,
    "core_oversample_factor": 1.5,
    "corr_threshold": 0.95,
    "bnn_hidden_dim": 64,
    "bnn_output_block": 128,
    "bnn_parent_min": 8,
    "bnn_parent_max": 128,
    "corr_audit_features": 256,
    "clip_value": 10.0,
    "min_y_unique_count": 10,
    "min_y_unique_ratio": 0.02,
    "max_y_dominant_ratio": 0.30,
    "min_y_standardized_iqr": 0.20,
    "y_round_decimals": 6,
    "max_retries": 30,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def iter_json_files(root: Path) -> Iterable[Path]:
    if root.is_dir():
        yield from root.rglob("*.json")


def collect_exclusions(
    data_root: Path, artifacts_root: Path
) -> tuple[set[int], set[str], dict[str, int]]:
    seeds: set[int] = set()
    hashes: set[str] = set()
    counts = {
        "producer_worker_states": 0,
        "producer_seed_schedule_entries": 0,
        "artifact_manifest": 0,
        "parse_failures": 0,
    }

    # This rolling root is generated from deterministic per-worker schedules.
    # Reading the small authoritative worker states proves seed disjointness
    # without reparsing tens of GB of cumulative catalog snapshots.
    worker_states = data_root / "producer" / "workers"
    for path in sorted(worker_states.glob("*/STATE.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            counts["parse_failures"] += 1
            continue
        counts["producer_worker_states"] += 1
        base_seed = int(payload["base_seed"])
        next_batch_id = int(payload["next_batch_id"])
        for batch_id in range(next_batch_id):
            seeds.add(int((base_seed + SEED_STRIDE * batch_id) % MODULUS))
        counts["producer_seed_schedule_entries"] += next_batch_id
        for key in ("last_published", "last_abandoned", "inflight"):
            item = payload.get(key)
            if isinstance(item, dict) and item.get("sha256"):
                hashes.add(str(item["sha256"]))

    for path in iter_json_files(artifacts_root):
        if path.name not in {
            "manifest.json",
            "accepted_tasks.json",
            "reproducibility_manifest.json",
        }:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            counts["parse_failures"] += 1
            continue
        counts["artifact_manifest"] += 1
        candidates: list[dict[str, Any]] = []
        if isinstance(payload, dict):
            for key in ("files", "tasks", "accepted_tasks", "datasets"):
                value = payload.get(key, [])
                if isinstance(value, list):
                    candidates.extend(item for item in value if isinstance(item, dict))
            candidates.append(payload)
        for entry in candidates:
            for key in ("batch_seed", "seed"):
                if entry.get(key) is not None:
                    try:
                        seeds.add(int(entry[key]))
                    except (TypeError, ValueError):
                        pass
            for key in ("sha256", "task_sha256", "data_sha256"):
                if entry.get(key):
                    hashes.add(str(entry[key]))
    return seeds, hashes, counts


def sample_shape(seed: int) -> tuple[int, int]:
    rng = np.random.default_rng(int(seed))
    return sample_shape_pair(rng, **{
        key: GENERATOR_ARGUMENTS[key]
        for key in (
            "n_min",
            "n_max",
            "main_n_min",
            "small_n_max",
            "small_n_probability",
            "p_min",
            "p_max",
            "max_cells",
            "p_median",
            "p_log_sigma",
            "raw_n_median",
            "n_log_sigma",
        )
    })


def generator_command(
    python: Path,
    generator: Path,
    output: Path,
    seed: int,
    threads: int,
) -> list[str]:
    a = GENERATOR_ARGUMENTS
    return [
        str(python),
        str(generator),
        "--out-dir", str(output),
        "--num-batches", "1",
        "--batch-size", "1",
        "--start-index", "0",
        "--end-index-exclusive", "1",
        "--seed", str(seed),
        "--threads", str(threads),
        "--n-min", str(a["n_min"]),
        "--n-max", str(a["n_max"]),
        "--main-n-min", str(a["main_n_min"]),
        "--small-n-max", str(a["small_n_max"]),
        "--small-n-probability", str(a["small_n_probability"]),
        "--p-min", str(a["p_min"]),
        "--p-max", str(a["p_max"]),
        "--max-cells", str(a["max_cells"]),
        "--p-median", str(a["p_median"]),
        "--p-log-sigma", str(a["p_log_sigma"]),
        "--raw-n-median", str(a["raw_n_median"]),
        "--n-log-sigma", str(a["n_log_sigma"]),
        "--scm-core-cap", str(a["scm_core_cap"]),
        "--core-oversample-factor", str(a["core_oversample_factor"]),
        "--corr-threshold", str(a["corr_threshold"]),
        "--corr-device", "cpu",
        "--bnn-hidden-dim", str(a["bnn_hidden_dim"]),
        "--bnn-output-block", str(a["bnn_output_block"]),
        "--bnn-parent-min", str(a["bnn_parent_min"]),
        "--bnn-parent-max", str(a["bnn_parent_max"]),
        "--corr-audit-features", str(a["corr_audit_features"]),
        "--clip-value", str(a["clip_value"]),
        "--min-y-unique-count", str(a["min_y_unique_count"]),
        "--min-y-unique-ratio", str(a["min_y_unique_ratio"]),
        "--max-y-dominant-ratio", str(a["max_y_dominant_ratio"]),
        "--min-y-standardized-iqr", str(a["min_y_standardized_iqr"]),
        "--y-round-decimals", str(a["y_round_decimals"]),
        "--max-retries", str(a["max_retries"]),
        "--filter-unpredictable",
    ]


def read_generated(candidate_dir: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = candidate_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files", [])
    if len(entries) != 1:
        raise RuntimeError(f"expected exactly one generated file, found {len(entries)}")
    entry = dict(entries[0])
    data_path = candidate_dir / str(entry["path"])
    if not data_path.is_file():
        raise RuntimeError(f"generated data file is missing: {data_path}")
    return {"manifest": manifest, "entry": entry}, data_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--root-seed", type=int, default=DEFAULT_ROOT_SEED)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=10_000)
    parser.add_argument("--python", type=Path, default=Path("/venv/main/bin/python"))
    args = parser.parse_args()

    project = args.project.resolve()
    output_root = args.output_root.resolve()
    checkpoint = args.checkpoint.resolve()
    data_root = args.data_root.resolve()
    generator = project / "generate_tabiclv2_scm2048_qfilter_bnn128_expvarnoise_v11.py"
    producer_config = (
        project
        / "model_v58_chebyshev_v17_scmmixture_p100k"
        / "producer_config_tabicl_v11_expvarnoise_rolling100gb.json"
    )
    required = (checkpoint, generator, producer_config)
    if any(not path.is_file() for path in required):
        raise FileNotFoundError(f"required file missing: {required}")
    if output_root.exists():
        raise RuntimeError(f"refusing existing output root: {output_root}")

    for relative in ("protocol", "data", "logs", "reproducibility"):
        (output_root / relative).mkdir(parents=True, exist_ok=False)
    audit_path = output_root / "protocol" / "sampling_audit.jsonl"

    checkpoint_sha = file_sha256(checkpoint)
    frozen_checkpoint = (
        output_root / "reproducibility" / checkpoint.name
    )
    os.link(checkpoint, frozen_checkpoint)
    source_hashes = {
        "generator": file_sha256(generator),
        "producer_config": file_sha256(producer_config),
        "checkpoint": checkpoint_sha,
    }
    shutil.copy2(generator, output_root / "reproducibility" / generator.name)
    shutil.copy2(
        producer_config, output_root / "reproducibility" / producer_config.name
    )

    excluded_seeds, excluded_hashes, exclusion_counts = collect_exclusions(
        data_root, project / "artifacts"
    )
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "root_seed": int(args.root_seed),
        "seed_stride": SEED_STRIDE,
        "seed_modulus": MODULUS,
        "required_conditions": {"n_gt": 300, "p_gt": 1000, "p_gt_n": True},
        "count": int(args.count),
        "shape_sampler": SHAPE_SAMPLER_NAME,
        "generator_arguments": GENERATOR_ARGUMENTS,
        "source_hashes": source_hashes,
        "checkpoint_original": str(checkpoint),
        "checkpoint_frozen": str(frozen_checkpoint),
        "data_root_checked_for_overlap": str(data_root),
        "exclusion_counts": {
            **exclusion_counts,
            "unique_seeds": len(excluded_seeds),
            "unique_hashes": len(excluded_hashes),
        },
        "selection_rule": (
            "first five sequential candidate seeds passing only dimension constraints, "
            "existing generator quality filters, and seed/task-hash overlap checks"
        ),
    }
    atomic_json(output_root / "protocol" / "protocol.json", protocol)

    accepted: list[dict[str, Any]] = []
    for candidate_index in range(int(args.max_candidates)):
        seed = int((int(args.root_seed) + SEED_STRIDE * candidate_index) % MODULUS)
        n, p = sample_shape(seed)
        record: dict[str, Any] = {
            "candidate_index": candidate_index,
            "seed": seed,
            "sampled_n": n,
            "sampled_p": p,
            "p_over_n": float(p / n),
        }
        if not (n > 300 and p > 1000 and p > n):
            record["decision"] = "rejected_dimension"
            append_jsonl(audit_path, record)
            continue
        if seed in excluded_seeds:
            record["decision"] = "rejected_seed_overlap"
            append_jsonl(audit_path, record)
            continue

        candidate_dir = output_root / "data" / f"_candidate_{candidate_index:06d}"
        log_path = output_root / "logs" / f"generator_candidate_{candidate_index:06d}.log"
        command = generator_command(
            args.python.resolve(), generator, candidate_dir, seed, int(args.threads)
        )
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=project,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        record["generator_command"] = command
        record["generator_exit_code"] = int(completed.returncode)
        if completed.returncode != 0:
            record["decision"] = "rejected_existing_generator_quality_failure"
            record["generator_log"] = str(log_path)
            append_jsonl(audit_path, record)
            continue

        generated, data_path = read_generated(candidate_dir)
        entry = generated["entry"]
        data_sha = file_sha256(data_path)
        if (int(entry["n"]), int(entry["p"])) != (n, p):
            raise RuntimeError("generator shape does not reproduce pre-sampled shape")
        if int(entry["batch_seed"]) != seed:
            raise RuntimeError("generator batch seed does not match candidate seed")
        if data_sha in excluded_hashes:
            record["decision"] = "rejected_task_hash_overlap"
            record["task_sha256"] = data_sha
            append_jsonl(audit_path, record)
            continue

        dataset_index = len(accepted) + 1
        destination = output_root / "data" / f"dataset_{dataset_index:02d}"
        candidate_dir.rename(destination)
        data_path = destination / data_path.name
        manifest_path = destination / "manifest.json"
        task_meta = dict(entry.get("task_metadata", [{}])[0])
        accepted_entry = {
            "dataset_index": dataset_index,
            "dataset_id": f"Dataset {dataset_index}",
            "candidate_index": candidate_index,
            "seed": seed,
            "n": n,
            "p": p,
            "p_over_n": float(p / n),
            "relative_data_path": str(data_path.relative_to(output_root)),
            "relative_manifest_path": str(manifest_path.relative_to(output_root)),
            "task_sha256": data_sha,
            "manifest_sha256": file_sha256(manifest_path),
            "generator_sha256": source_hashes["generator"],
            "producer_config_sha256": source_hashes["producer_config"],
            "dataset_contract": entry.get("dataset_contract"),
            "data_policy": entry.get("data_policy"),
            "core_exact_max_abs_corr": entry.get("task_core_exact_max_abs_corr"),
            "sampled_final_max_abs_corr": entry.get("task_sampled_corr_audit_max"),
            "full_pair_exact_certificate": bool(
                entry.get("full_pair_exact_certificate")
            ),
            "official_active_base_features": task_meta.get(
                "official_active_base_features"
            ),
            "generator_attempt": task_meta.get("attempt"),
            "generator_log": str(log_path.relative_to(output_root)),
        }
        accepted.append(accepted_entry)
        excluded_seeds.add(seed)
        excluded_hashes.add(data_sha)
        record.update({
            "decision": "accepted",
            "dataset_index": dataset_index,
            "task_sha256": data_sha,
            "generator_attempt": task_meta.get("attempt"),
        })
        append_jsonl(audit_path, record)
        atomic_json(
            output_root / "protocol" / "accepted_tasks.json",
            {"protocol_version": PROTOCOL_VERSION, "accepted_tasks": accepted},
        )
        if len(accepted) == int(args.count):
            break

    if len(accepted) != int(args.count):
        raise RuntimeError(
            f"only accepted {len(accepted)} tasks after {args.max_candidates} candidates"
        )
    result = {
        **protocol,
        "accepted_tasks": accepted,
        "sampling_audit_sha256": file_sha256(audit_path),
    }
    atomic_json(
        output_root / "reproducibility" / "reproducibility_manifest.json", result
    )
    print(json.dumps({"accepted_tasks": accepted}, indent=2), flush=True)


if __name__ == "__main__":
    main()
