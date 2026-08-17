#!/usr/bin/env python3
"""Freeze the first four eligible native-TabICL tasks from the V13 generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from exp122_shape_sampler import SHAPE_SAMPLER_NAME, sample_shape_pair

from .prepare_v24_n200p1000_screened4 import (
    SOURCE_GENERATOR_SEED_BASE,
    SOURCE_SHAPE_SEEDS,
    max_abs_correlation,
    run_generator,
    sha256,
    split_indices,
    standardize_fit,
    write_json,
)


def task_hash(x: np.ndarray, y: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(x).tobytes())
    digest.update(np.ascontiguousarray(y).tobytes())
    return digest.hexdigest()


def generator_seed(index: int) -> int:
    seed = SOURCE_GENERATOR_SEED_BASE["tabicl"] + 2 * index
    return int(seed if seed % 2 == 0 else seed + 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--producer-config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--python", type=Path, default=Path("/venv/main/bin/python"))
    parser.add_argument("--task-count", type=int, default=4)
    parser.add_argument("--max-shape-draws", type=int, default=50_000)
    args = parser.parse_args()

    if args.output_root.exists():
        raise RuntimeError(f"refusing existing output root: {args.output_root}")
    args.output_root.mkdir(parents=True)
    for directory in ("data", "splits", "metrics", "manifest", "logs"):
        (args.output_root / directory).mkdir()
    if sha256(args.producer_config) != args.expected_config_sha256:
        raise RuntimeError("producer config SHA256 mismatch")
    config = json.loads(args.producer_config.read_text())
    generator = Path(
        config["generator_sources"]["tabicl_v2"]["path"]
    ).resolve()
    if sha256(generator) != config["generator_sources"]["tabicl_v2"]["sha256"]:
        raise RuntimeError("active V13 generator SHA256 mismatch")

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
    shape_rng = np.random.default_rng(SOURCE_SHAPE_SEEDS["tabicl"])
    attempts: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    generated_count = 0
    for draw_index in range(args.max_shape_draws):
        n, p = sample_shape_pair(shape_rng, **shape_args)
        base = {
            "source": "tabicl",
            "shape_draw_index": draw_index,
            "shape_stream_seed": SOURCE_SHAPE_SEEDS["tabicl"],
            "n": int(n),
            "p": int(p),
        }
        if not (n > 200 and p < 1000):
            attempts.append({**base, "disposition": "n_p_condition_rejection"})
            continue
        seed = generator_seed(generated_count)
        generated_count += 1
        candidate_id = f"tabicl_candidate_{generated_count:03d}"
        candidate_dir = args.output_root / "candidate_work" / candidate_id
        log_path = args.output_root / "logs" / f"{candidate_id}.log"
        split_seed = int((seed ^ 0x5EED_2026) % (2**32 - 1))
        try:
            data_path, generator_entry = run_generator(
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
                source_id = int(archive["outcome_source_id"][0])
            if source_id != 0:
                raise RuntimeError("native TabICL source parity contract failed")
            split = split_indices(n, split_seed)
            _, _, stats = standardize_fit(
                x[split["context"]], y[split["context"]]
            )
            label = f"tabicl_{len(accepted) + 1}"
            data_output = args.output_root / "data" / f"{label}.npz"
            split_output = args.output_root / "splits" / f"{label}.npz"
            shutil.copy2(data_path, data_output)
            np.savez_compressed(
                split_output,
                context_indices=split["context"],
                screening_indices=split["screening"],
                final_indices=split["final"],
                x_mean=np.asarray(stats["x_mean"]),
                x_scale=np.asarray(stats["x_scale"]),
                active=np.asarray(stats["active"]),
                y_mean=np.asarray(stats["y_mean"]),
                y_scale=np.asarray(stats["y_scale"]),
            )
            context_y = np.asarray(y[split["context"]], dtype=np.float64)
            final_y = np.asarray(y[split["final"]], dtype=np.float64)
            context_x = np.asarray(x[split["context"]], dtype=np.float64)
            centered_x = context_x - context_x.mean(axis=0, keepdims=True)
            centered_y = context_y - context_y.mean()
            denominator = (
                np.linalg.norm(centered_x, axis=0)
                * max(float(np.linalg.norm(centered_y)), 1e-12)
            )
            feature_y_correlation = np.divide(
                centered_x.T @ centered_y,
                denominator,
                out=np.zeros(p, dtype=np.float64),
                where=denominator > 1e-12,
            )
            correlation = max_abs_correlation(x)
            record = {
                **base,
                "candidate_id": candidate_id,
                "generator_seed": seed,
                "split_seed": split_seed,
                "generator_entry": generator_entry,
                "label": label,
                "dataset_index": len(accepted) + 1,
                "disposition": "accepted",
                "selection_contract": (
                    "first eligible native-TabICL task; no performance filter"
                ),
                "task_sha256": task_hash(x, y),
                "accepted_data_path": str(data_output),
                "accepted_data_sha256": sha256(data_output),
                "split_path": str(split_output),
                "split_sha256": sha256(split_output),
                "row_counts": {
                    key: int(value.size) for key, value in split.items()
                },
                "feature_correlation": correlation,
                "context_y_std": float(context_y.std()),
                "context_max_abs_feature_y_correlation": float(
                    np.max(np.abs(feature_y_correlation))
                ),
                "zero_predictor_final_mse": float(
                    np.mean((final_y - context_y.mean()) ** 2)
                ),
            }
            attempts.append(record)
            accepted.append(record)
            shutil.rmtree(candidate_dir)
            write_json(
                args.output_root / "manifest" / "accepted_seeds.json",
                accepted,
            )
            write_json(
                args.output_root / "manifest" / "attempt_log.json",
                attempts,
            )
            print(
                json.dumps(
                    {
                        "accepted": label,
                        "seed": seed,
                        "n": n,
                        "p": p,
                    }
                ),
                flush=True,
            )
            if len(accepted) == args.task_count:
                break
        except Exception as exc:  # noqa: BLE001
            attempts.append(
                {
                    **base,
                    "candidate_id": candidate_id,
                    "generator_seed": seed,
                    "disposition": "generator_quality_rejection",
                    "error": repr(exc),
                    "log_path": str(log_path),
                }
            )
            if candidate_dir.exists():
                shutil.rmtree(candidate_dir)
    if len(accepted) != args.task_count:
        raise RuntimeError(
            f"accepted {len(accepted)} of {args.task_count} requested tasks"
        )
    summary = {
        "status": "PASS",
        "selection_contract": (
            "first four N>200, P<1000 tasks in a fixed native-TabICL stream; "
            "production quality filters only"
        ),
        "shape_sampler": SHAPE_SAMPLER_NAME,
        "shape_stream_seed": SOURCE_SHAPE_SEEDS["tabicl"],
        "producer_config": str(args.producer_config.resolve()),
        "producer_config_sha256": sha256(args.producer_config),
        "generator": str(generator),
        "generator_sha256": sha256(generator),
        "accepted_count": len(accepted),
        "generated_count": generated_count,
        "accepted": accepted,
    }
    write_json(args.output_root / "metrics" / "preparation_summary.json", summary)
    write_json(args.output_root / "manifest" / "attempt_log.json", attempts)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
