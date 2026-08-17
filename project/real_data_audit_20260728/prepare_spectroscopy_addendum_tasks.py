#!/usr/bin/env python3
"""Prepare one scalar regression task per valid spectroscopy target."""

from __future__ import annotations

import gc
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from prepare_real_regression_tasks import SEED, save_task


ROOT = Path(
    "/users/sync_data/real_regression_uci_hf_highdim_signal_screen_20260728"
)
PROCESSED = ROOT / "processed"
MANIFEST = json.loads(
    (ROOT / "manifests" / "spectroscopy_addendum_download_manifest.json").read_text()
)
SCHEMA_AUDIT = json.loads(
    (ROOT / "manifests" / "spectroscopy_addendum_schema_audit.json").read_text()
)
MIN_TASK_ROWS = 80
MIN_UNIQUE_TARGET = 10


def is_spectral_column(name: str) -> bool:
    try:
        float(name)
    except (TypeError, ValueError):
        return False
    return True


def slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return result[:90] or "target"


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("||".join(parts).encode()).digest()
    return SEED + int.from_bytes(digest[:4], "little")


def random_split(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    split = np.full(n, 2, dtype=np.int8)
    context = int(np.floor(0.6 * n))
    screening = int(np.floor(0.2 * n))
    split[order[:context]] = 0
    split[order[context : context + screening]] = 1
    return split


def train_screen_with_locked_test(
    train_count: int, test_count: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(train_count)
    context = int(np.floor(0.75 * train_count))
    split = np.full(train_count + test_count, 2, dtype=np.int8)
    split[order[:context]] = 0
    split[order[context:]] = 1
    return split


def target_quality(y: np.ndarray) -> tuple[bool, str | None]:
    finite = y[np.isfinite(y)]
    if finite.size < MIN_TASK_ROWS:
        return False, f"fewer than {MIN_TASK_ROWS} finite rows"
    if np.std(finite) <= 1e-8:
        return False, "near-zero target variance"
    values, counts = np.unique(finite, return_counts=True)
    if values.size < MIN_UNIQUE_TARGET:
        return False, f"fewer than {MIN_UNIQUE_TARGET} unique target values"
    if counts.max() / counts.sum() > 0.30:
        return False, "one target value occupies more than 30 percent"
    return True, None


def evidence_metadata(n: int, p: int) -> dict[str, Any]:
    small_n = n < 250
    return {
        "screening_min_finite_context_screen": 60 if small_n and p > n else 200,
        "evidence_tier": "exploratory_small_n_highdim" if small_n else "standard",
    }


def should_skip_ramanbench(task_name: str) -> str | None:
    if task_name.startswith("synthetic_"):
        return "explicitly synthetic source"
    if task_name.startswith("bioprocess_analytes_") or task_name == "bioprocess_substrates":
        return "duplicate of standalone source with official train/validation/test split"
    if task_name.startswith("microgel_size_") and task_name != "microgel_size_raw_global":
        return "duplicate transformed view; canonical raw-global view retained"
    return None


def prepare_ramanbench(exclusions: list[dict[str, Any]]) -> list[str]:
    prepared = []
    audit_by_name = {
        item["task_name"]: item for item in SCHEMA_AUDIT["ramanbench"]
    }
    for source in MANIFEST["ramanbench_regression_tasks"]:
        task_name = source["task_name"]
        skip_reason = should_skip_ramanbench(task_name)
        if skip_reason:
            exclusions.append(
                {"source": "RamanBench", "task_name": task_name, "reason": skip_reason}
            )
            continue
        path = Path(source["parquet_file"]["local_path"])
        audit = audit_by_name[task_name]
        schema_columns = pq.ParquetFile(path).schema_arrow.names
        spectral = [name for name in schema_columns if is_spectral_column(name)]
        semantic_targets = [
            str(name) for name in source["target_names"]
            if str(name).lower() not in {"time", "time_h"}
        ]
        if all(target in schema_columns for target in semantic_targets):
            target_specs = [(target, target) for target in semantic_targets]
        elif len(semantic_targets) == 1 and "target" in schema_columns:
            target_specs = [(semantic_targets[0], "target")]
        else:
            exclusions.append(
                {
                    "source": "RamanBench",
                    "task_name": task_name,
                    "reason": (
                        "metadata targets do not map to parquet columns; multi-class "
                        "labels encoded in a single target column are not regression"
                    ),
                    "metadata_targets": semantic_targets,
                }
            )
            continue
        columns = spectral + sorted({actual for _, actual in target_specs})
        frame = pd.read_parquet(path, columns=columns)
        X_all = frame[spectral].to_numpy(np.float32)
        for target, actual_target in target_specs:
            y_all = pd.to_numeric(
                frame[actual_target], errors="coerce"
            ).to_numpy(float)
            valid = np.isfinite(y_all)
            quality_ok, reason = target_quality(y_all[valid])
            if not quality_ok:
                exclusions.append(
                    {
                        "source": "RamanBench",
                        "task_name": task_name,
                        "target": target,
                        "reason": reason,
                    }
                )
                continue
            X = X_all[valid]
            y = y_all[valid]
            task_id = f"hf_ramanbench_{slug(task_name)}__{slug(target)}"
            split = random_split(len(y), stable_seed(task_name, target))
            reality_status = (
                "experimental_or_noise_augmented_mixture_requires_source_review"
                if "mixture" in task_name
                else "empirical_spectroscopy_requires_original_license_review"
            )
            save_task(
                PROCESSED,
                task_id,
                X,
                y,
                split,
                {
                    "source": "HuggingFace HTW-KI-Werkstatt/RamanBench",
                    "source_revision": source["parquet_file"]["revision"],
                    "source_task": task_name,
                    "source_file_sha256": source["parquet_file"]["sha256"],
                    "target": target,
                    "source_target_column": actual_target,
                    "encoding": "published wide-format Raman wavenumber intensities",
                    "split_provenance": "deterministic label-blind random 60_20_20",
                    "reality_status": reality_status,
                    "license_note": (
                        "research mirror; original dataset license and citation must "
                        "be checked before publication"
                    ),
                    **evidence_metadata(len(y), X.shape[1]),
                },
                spectral,
            )
            prepared.append(task_id)
        del frame, X_all
        gc.collect()
    return prepared


def read_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def standalone_splits(source: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    frames = []
    splits = []
    for file_record in source["files"]:
        path = Path(file_record["local_path"])
        if path.name == "README.md":
            continue
        frame = read_frame(path)
        name = path.name.lower()
        if "train" in name:
            code = 0
        elif "val" in name:
            code = 1
        elif "test" in name:
            code = 2
        else:
            raise RuntimeError(f"Cannot identify split for {path}")
        frames.append(frame)
        splits.append(np.full(len(frame), code, dtype=np.int8))
    return pd.concat(frames, ignore_index=True), np.concatenate(splits)


def prepare_standalone(exclusions: list[dict[str, Any]]) -> list[str]:
    prepared = []
    for source in MANIFEST["standalone_sources"]:
        repo = source["repo"]
        if repo == "Avatarr05/GreenHyperSpectra":
            continue
        frame, split = standalone_splits(source)
        spectral = [str(name) for name in frame.columns if is_spectral_column(str(name))]
        targets = [str(name) for name in frame.columns if str(name) not in spectral]
        X_all = frame[spectral].to_numpy(np.float32)
        for target in targets:
            y_all = pd.to_numeric(frame[target], errors="coerce").to_numpy(float)
            valid = np.isfinite(y_all)
            quality_ok, reason = target_quality(y_all[valid])
            if not quality_ok:
                exclusions.append(
                    {
                        "source": repo,
                        "target": target,
                        "reason": reason,
                    }
                )
                continue
            X = X_all[valid]
            y = y_all[valid]
            target_split = split[valid]
            task_id = f"hf_{slug(repo)}__{slug(target)}"
            source_files = [
                item for item in source["files"] if not item["local_path"].endswith("README.md")
            ]
            save_task(
                PROCESSED,
                task_id,
                X,
                y,
                target_split,
                {
                    "source": f"HuggingFace {repo}",
                    "source_revision": source["revision"],
                    "source_file_sha256": [item["sha256"] for item in source_files],
                    "target": target,
                    "encoding": "published wide-format Raman wavenumber intensities",
                    "split_provenance": "official train_validation_test",
                    "reality_status": "empirical_bioprocess_spectroscopy",
                    "license": source["license"],
                    **evidence_metadata(len(y), X.shape[1]),
                },
                spectral,
            )
            prepared.append(task_id)
        del frame, X_all
        gc.collect()
    return prepared


def prepare_green_hyperspectra(exclusions: list[dict[str, Any]]) -> list[str]:
    source = next(
        item for item in MANIFEST["standalone_sources"]
        if item["repo"] == "Avatarr05/GreenHyperSpectra"
    )
    train_record = next(
        item for item in source["files"]
        if item["remote_path"] == "labeled_splits/train.csv"
    )
    test_record = next(
        item for item in source["files"]
        if item["remote_path"] == "labeled_splits/test.csv"
    )
    train = pd.read_csv(train_record["local_path"])
    test = pd.read_csv(test_record["local_path"])
    frame = pd.concat([train, test], ignore_index=True)
    spectral = [str(name) for name in frame.columns if is_spectral_column(str(name))]
    ignored = {"Unnamed: 0", "dataset"}
    targets = [
        str(name) for name in frame.columns
        if str(name) not in spectral and str(name) not in ignored
    ]
    X_all = frame[spectral].to_numpy(np.float32)
    original_train = np.arange(len(train))
    original_test = np.arange(len(train), len(frame))
    prepared = []
    for target in targets:
        y_all = pd.to_numeric(frame[target], errors="coerce").to_numpy(float)
        train_valid = original_train[np.isfinite(y_all[original_train])]
        test_valid = original_test[np.isfinite(y_all[original_test])]
        selected = np.concatenate([train_valid, test_valid])
        y = y_all[selected]
        quality_ok, reason = target_quality(y)
        if not quality_ok or len(test_valid) < 20:
            exclusions.append(
                {
                    "source": source["repo"],
                    "target": target,
                    "reason": reason or "fewer than 20 finite locked-test rows",
                }
            )
            continue
        X = X_all[selected]
        split = train_screen_with_locked_test(
            len(train_valid), len(test_valid), stable_seed(source["repo"], target)
        )
        task_id = f"hf_greenhyperspectra__{slug(target)}"
        save_task(
            PROCESSED,
            task_id,
            X,
            y,
            split,
            {
                "source": "HuggingFace Avatarr05/GreenHyperSpectra",
                "source_revision": source["revision"],
                "source_file_sha256": [
                    train_record["sha256"],
                    test_record["sha256"],
                ],
                "target": target,
                "encoding": "published reflectance bands from 400 to 2450 nm",
                "split_provenance": (
                    "official test locked; published train deterministically split "
                    "75_25 into context_screening"
                ),
                "reality_status": "empirical_multi_source_vegetation_hyperspectroscopy",
                "license": source["license"],
                **evidence_metadata(len(y), X.shape[1]),
            },
            spectral,
        )
        prepared.append(task_id)
    del train, test, frame, X_all
    gc.collect()
    return prepared


def main() -> None:
    exclusions: list[dict[str, Any]] = []
    prepared = []
    prepared.extend(prepare_ramanbench(exclusions))
    prepared.extend(prepare_standalone(exclusions))
    prepared.extend(prepare_green_hyperspectra(exclusions))
    output = {
        "contract": "spectroscopy_scalar_task_preparation_v1",
        "prepared_count": len(prepared),
        "prepared_tasks": prepared,
        "excluded_count": len(exclusions),
        "exclusions": exclusions,
    }
    path = ROOT / "manifests" / "spectroscopy_addendum_preparation_manifest.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"prepared": len(prepared), "excluded": len(exclusions), "manifest": str(path)},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
