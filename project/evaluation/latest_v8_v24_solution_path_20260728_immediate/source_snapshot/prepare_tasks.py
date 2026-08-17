#!/usr/bin/env python3
"""Freeze four unseen V14 tasks without model-performance filtering."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from exp122_shape_sampler import SHAPE_SAMPLER_NAME, sample_shape_pair


SOURCE_IDS = {"linear": 1, "tabicl": 0}
SHAPE_SEEDS = {"linear": 2_026_072_801, "tabicl": 2_026_072_802}
GENERATOR_SEED_BASE = {"linear": 2_147_100_001, "tabicl": 2_147_200_000}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def task_hash(x: np.ndarray, y: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(x).tobytes())
    digest.update(np.ascontiguousarray(y).tobytes())
    return digest.hexdigest()


def recursively_collect_provenance(
    value: Any, seeds: set[int], hashes: set[str]
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lower = key.lower()
            if "seed" in lower and isinstance(child, (int, float)):
                seeds.add(int(child))
            if (
                ("task" in lower or "dataset" in lower)
                and ("sha" in lower or "hash" in lower)
                and isinstance(child, str)
                and len(child) == 64
            ):
                hashes.add(child.lower())
            recursively_collect_provenance(child, seeds, hashes)
    elif isinstance(value, list):
        for child in value:
            recursively_collect_provenance(child, seeds, hashes)


def scan_prior_provenance(
    roots: list[Path], excluded_root: Path
) -> tuple[set[int], set[str], list[str]]:
    seeds: set[int] = set()
    hashes: set[str] = set()
    scanned: list[str] = []
    excluded = excluded_root.resolve()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            resolved = path.resolve()
            if excluded == resolved or excluded in resolved.parents:
                continue
            if any(part.startswith(".venv") for part in resolved.parts):
                continue
            try:
                if path.stat().st_size > 25_000_000:
                    continue
                document = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            recursively_collect_provenance(document, seeds, hashes)
            scanned.append(str(resolved))
    return seeds, hashes, scanned


def split_indices(n: int, seed: int) -> dict[str, np.ndarray]:
    permutation = np.random.default_rng(seed).permutation(n).astype(np.int64)
    context_count = int(math.floor(0.60 * n))
    validation_count = int(math.floor(0.20 * n))
    test_count = n - context_count - validation_count
    if min(context_count, validation_count, test_count) < 32:
        raise RuntimeError(f"split too small for N={n}")
    return {
        "context": permutation[:context_count],
        "validation": permutation[
            context_count : context_count + validation_count
        ],
        "test": permutation[context_count + validation_count :],
    }


def fit_preprocessing(
    x: np.ndarray, y: np.ndarray
) -> dict[str, np.ndarray | float]:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    x_mean = x64.mean(axis=0)
    x_scale = x64.std(axis=0)
    active = x_scale > 1e-8
    x_scale = np.where(active, x_scale, 1.0)
    y_mean = float(y64.mean())
    y_scale = float(y64.std())
    if not np.isfinite(x_mean).all() or not np.isfinite(x_scale).all():
        raise RuntimeError("nonfinite context preprocessing")
    if not math.isfinite(y_mean) or not math.isfinite(y_scale) or y_scale <= 1e-8:
        raise RuntimeError("degenerate context target")
    return {
        "x_mean": x_mean,
        "x_scale": x_scale,
        "active": active,
        "y_mean": y_mean,
        "y_scale": y_scale,
    }


def max_abs_correlation(x: np.ndarray) -> float:
    values = np.asarray(x, dtype=np.float64)
    values -= values.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(values, axis=0)
    valid = norm > 1e-12
    normalized = values[:, valid] / norm[valid][None, :]
    correlation = normalized.T @ normalized
    if correlation.shape[0] < 2:
        return 0.0
    return float(
        np.max(np.abs(correlation[np.triu_indices(correlation.shape[0], 1)]))
    )


def max_feature_target_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    x64 -= x64.mean(axis=0, keepdims=True)
    y64 -= y64.mean()
    denominator = np.linalg.norm(x64, axis=0) * np.linalg.norm(y64)
    values = np.divide(
        x64.T @ y64,
        denominator,
        out=np.zeros(x64.shape[1], dtype=np.float64),
        where=denominator > 1e-12,
    )
    return float(np.max(np.abs(values), initial=0.0))


def run_generator(
    *,
    python: Path,
    generator: Path,
    config: dict[str, Any],
    output: Path,
    seed: int,
    n: int,
    p: int,
    log_path: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    command = [
        str(python),
        str(generator),
        "--out-dir",
        str(output),
        "--num-batches",
        "1",
        "--batch-size",
        "1",
        "--start-index",
        "0",
        "--end-index-exclusive",
        "1",
        "--seed",
        str(seed),
        "--threads",
        "4",
        "--corr-device",
        "cpu",
        "--force-n",
        str(n),
        "--force-p",
        str(p),
        *[str(value) for value in config["generator_common_args"]],
        *[
            str(value)
            for value in config["generator_sources"]["tabicl_v2"]["args"]
        ],
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=Path(config["project_dir"]),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"generator exited {completed.returncode}")
    manifest_document = json.loads((output / "manifest.json").read_text())
    entries = (
        manifest_document["files"]
        if isinstance(manifest_document, dict)
        else manifest_document
    )
    if len(entries) != 1:
        raise RuntimeError("generator manifest cardinality drift")
    entry = entries[0]
    entry["generator_wall_seconds"] = time.perf_counter() - started
    return output / entry["path"], entry, manifest_document


def generator_seed(source: str, accepted_index: int) -> int:
    seed = GENERATOR_SEED_BASE[source] + 2 * accepted_index
    if seed % 2 != SOURCE_IDS[source]:
        seed += 1
    return int(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pipeline-config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--rolling-control-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path("/venv/main/bin/python"))
    args = parser.parse_args()

    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError(f"refusing nonempty output root: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for name in ("data", "splits", "metrics", "manifest", "logs", "generator_manifests"):
        (args.output_root / name).mkdir()
    if sha256(args.pipeline_config) != args.expected_config_sha256:
        raise RuntimeError("pipeline-config SHA256 mismatch")
    pipeline_config = json.loads(args.pipeline_config.read_text())
    producer_lock_path = (
        Path(pipeline_config["data"]["data_root"])
        / "producer"
        / "PRODUCER_CONFIG_LOCK.json"
    )
    producer_lock = json.loads(producer_lock_path.read_text())
    config = producer_lock["source_contract"]
    if config["pipeline_config_sha256"] != args.expected_config_sha256:
        raise RuntimeError("producer lock/pipeline-config mismatch")
    generator = Path(config["generator_sources"]["tabicl_v2"]["path"]).resolve()
    if sha256(generator) != config["generator_sources"]["tabicl_v2"]["sha256"]:
        raise RuntimeError("generator SHA256 mismatch")

    provenance_roots = [
        args.project_root / "evaluation",
        args.project_root / "evaluations",
        args.project_root / "validation",
        args.project_root / "isolated_audits",
        args.project_root / "runs",
        args.rolling_control_root,
    ]
    prior_seeds, prior_hashes, scanned = scan_prior_provenance(
        provenance_roots, args.output_root
    )
    provenance_audit = {
        "json_files_scanned": len(scanned),
        "prior_seed_count": len(prior_seeds),
        "prior_task_hash_count": len(prior_hashes),
        "scan_roots": [str(path.resolve()) for path in provenance_roots],
        "catalog_files": [
            path for path in scanned if "/rolling_control/catalogs/" in path
        ],
    }
    write_json(args.output_root / "manifest" / "prior_provenance_audit.json", provenance_audit)

    shape_args = vars(
        SimpleNamespace(
            n_min=32,
            n_max=40_000,
            main_n_min=200,
            small_n_max=199,
            small_n_probability=0.10,
            p_min=50,
            p_max=100_000,
            max_cells=24_000_000,
            p_median=10_000.0,
            p_log_sigma=0.75,
            raw_n_median=5_700.0,
            n_log_sigma=0.70,
        )
    )
    accepted: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for source in ("linear", "tabicl"):
        rng = np.random.default_rng(SHAPE_SEEDS[source])
        accepted_for_source = 0
        eligible_index = 0
        for draw_index in range(50_000):
            n, p = sample_shape_pair(rng, **shape_args)
            base = {
                "source": source,
                "shape_draw_index": draw_index,
                "shape_stream_seed": SHAPE_SEEDS[source],
                "n": int(n),
                "p": int(p),
            }
            if not (n > 200 and p < 1000):
                attempts.append({**base, "disposition": "n_p_contract_rejection"})
                continue
            seed = generator_seed(source, eligible_index)
            eligible_index += 1
            candidate_id = f"{source}_candidate_{eligible_index:03d}"
            if seed in prior_seeds:
                attempts.append(
                    {
                        **base,
                        "candidate_id": candidate_id,
                        "seed": seed,
                        "disposition": "prior_seed_rejection",
                    }
                )
                continue
            candidate_dir = args.output_root / "candidate_work" / candidate_id
            candidate_dir.mkdir(parents=True)
            log_path = args.output_root / "logs" / f"{candidate_id}.log"
            split_seed = int((seed ^ 0x5EED_2026) % (2**32 - 1))
            try:
                data_path, generator_entry, generator_manifest = run_generator(
                    python=args.python,
                    generator=generator,
                    config=config,
                    output=candidate_dir,
                    seed=seed,
                    n=n,
                    p=p,
                    log_path=log_path,
                )
                with np.load(data_path, allow_pickle=False) as archive:
                    x = np.asarray(archive["X"][0], dtype=np.float32)
                    y = np.asarray(archive["y"][0], dtype=np.float32)
                    outcome_id = int(archive["outcome_source_id"][0])
                if x.shape != (n, p) or y.shape != (n,):
                    raise RuntimeError("generated shape mismatch")
                if outcome_id != SOURCE_IDS[source]:
                    raise RuntimeError("outcome parity mismatch")
                if not np.isfinite(x).all() or not np.isfinite(y).all():
                    raise RuntimeError("nonfinite generated task")
                candidate_hash = task_hash(x, y)
                if candidate_hash in prior_hashes:
                    attempts.append(
                        {
                            **base,
                            "candidate_id": candidate_id,
                            "seed": seed,
                            "task_hash": candidate_hash,
                            "disposition": "prior_task_hash_rejection",
                        }
                    )
                    shutil.rmtree(candidate_dir)
                    continue

                split = split_indices(n, split_seed)
                stats = fit_preprocessing(x[split["context"]], y[split["context"]])
                label = f"dataset_{len(accepted) + 1}"
                data_output = args.output_root / "data" / f"{label}.npz"
                split_output = args.output_root / "splits" / f"{label}.npz"
                manifest_output = (
                    args.output_root / "generator_manifests" / f"{label}.json"
                )
                shutil.copy2(data_path, data_output)
                write_json(manifest_output, generator_manifest)
                np.savez_compressed(
                    split_output,
                    context_indices=split["context"],
                    validation_indices=split["validation"],
                    test_indices=split["test"],
                    x_mean=np.asarray(stats["x_mean"]),
                    x_scale=np.asarray(stats["x_scale"]),
                    active=np.asarray(stats["active"]),
                    y_mean=np.asarray(stats["y_mean"]),
                    y_scale=np.asarray(stats["y_scale"]),
                )
                context_y = np.asarray(y[split["context"]], dtype=np.float64)
                validation_y = np.asarray(y[split["validation"]], dtype=np.float64)
                test_y = np.asarray(y[split["test"]], dtype=np.float64)
                context_mean = float(context_y.mean())
                linear_snr = float(generator_entry.get("linear_realized_snr", -1.0))
                linear_support = int(generator_entry.get("linear_support_size", 0))
                record = {
                    **base,
                    "dataset_index": len(accepted) + 1,
                    "task_id": label,
                    "label": label,
                    "outcome_branch": source,
                    "generator_seed": seed,
                    "split_seed": split_seed,
                    "task_hash": candidate_hash,
                    "generator_version": generator_manifest.get("version")
                    if isinstance(generator_manifest, dict)
                    else None,
                    "generator_config_hash": args.expected_config_sha256,
                    "generator_entry": generator_entry,
                    "p_over_n": float(p / n),
                    "row_counts": {key: int(value.size) for key, value in split.items()},
                    "realized_snr": linear_snr if source == "linear" else None,
                    "signal_strength_statistic": (
                        linear_snr
                        if source == "linear"
                        else max_feature_target_correlation(
                            x[split["context"]], y[split["context"]]
                        )
                    ),
                    "signal_strength_definition": (
                        "realized signal/noise variance ratio"
                        if source == "linear"
                        else "maximum absolute context feature-target correlation"
                    ),
                    "target_sparsity": linear_support if source == "linear" else None,
                    "max_abs_feature_correlation": max_abs_correlation(x),
                    "zero_predictor_validation_mse": float(
                        np.mean((validation_y - context_mean) ** 2)
                    ),
                    "zero_predictor_test_mse": float(
                        np.mean((test_y - context_mean) ** 2)
                    ),
                    "accepted_data_path": str(data_output.resolve()),
                    "accepted_data_sha256": sha256(data_output),
                    "split_path": str(split_output.resolve()),
                    "split_sha256": sha256(split_output),
                    "generator_manifest_path": str(manifest_output.resolve()),
                    "generator_manifest_sha256": sha256(manifest_output),
                    "disposition": "accepted",
                    "selection_contract": (
                        "first deterministic unseen seed with N>200, P<1000 and "
                        "generator quality filters passed; no model/performance filter"
                    ),
                }
                attempts.append(record)
                accepted.append(record)
                accepted_for_source += 1
                shutil.rmtree(candidate_dir)
                write_json(args.output_root / "manifest" / "accepted_tasks.json", accepted)
                write_json(args.output_root / "manifest" / "attempt_log.json", attempts)
                print(
                    json.dumps(
                        {
                            "accepted": label,
                            "source": source,
                            "seed": seed,
                            "n": n,
                            "p": p,
                        }
                    ),
                    flush=True,
                )
                if accepted_for_source == 2:
                    break
            except Exception as exc:
                attempts.append(
                    {
                        **base,
                        "candidate_id": candidate_id,
                        "seed": seed,
                        "disposition": "generator_quality_rejection",
                        "error": repr(exc),
                    }
                )
                if candidate_dir.exists():
                    shutil.rmtree(candidate_dir)
        if accepted_for_source != 2:
            raise RuntimeError(f"failed to accept two {source} tasks")

    summary = {
        "status": "PASS",
        "selection_contract": (
            "two controlled-linear and two native tasks; first deterministic unseen "
            "seeds; dimensions from current V14 distribution with only N/P constraints"
        ),
        "shape_sampler": SHAPE_SAMPLER_NAME,
        "shape_seeds": SHAPE_SEEDS,
        "generator_seed_bases": GENERATOR_SEED_BASE,
        "pipeline_config": str(args.pipeline_config.resolve()),
        "pipeline_config_sha256": sha256(args.pipeline_config),
        "producer_config_lock": str(producer_lock_path.resolve()),
        "producer_config_lock_sha256": sha256(producer_lock_path),
        "generator": str(generator),
        "generator_sha256": sha256(generator),
        "accepted_count": len(accepted),
        "accepted": accepted,
        "provenance_audit": provenance_audit,
    }
    write_json(args.output_root / "metrics" / "preparation_summary.json", summary)
    write_json(args.output_root / "manifest" / "attempt_log.json", attempts)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
