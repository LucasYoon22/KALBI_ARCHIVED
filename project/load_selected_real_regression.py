#!/usr/bin/env python3
"""Download the selected OpenML regression datasets and create locked 80/20 splits.

Selection rule
--------------
A screened dataset is excluded when either of the following is true:

* predictors removed by the train-only |corr| >= 0.95 filter > 40%; or
* discovery CV-LASSO outer-test R^2 is not strictly between 0.2 and 0.9.

The 13 retained datasets are pinned by OpenML data ID and server-side MD5.  Raw
X/y are split before any scaling, imputation, constant filtering, or correlation
filtering.  Fit every preprocessing step on X_train only in downstream code.

Examples
--------
List the retained registry without downloading:

    python load_selected_real_regression.py --list

Download and split one dataset:

    python load_selected_real_regression.py \
        --dataset QSAR-TID-10331 --output-dir ./real_regression_data

Download and split all retained datasets (large download):

    python load_selected_real_regression.py \
        --all --output-dir ./real_regression_data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.datasets import fetch_openml


CORR_REMOVAL_LIMIT_PCT = 40.0
TEST_R2_MIN_EXCLUSIVE = 0.2
TEST_R2_MAX_EXCLUSIVE = 0.9
DEFAULT_TEST_FRACTION = 0.20
# Same master seed as the archived candidate CV-LASSO screen.
DEFAULT_SPLIT_SEED = 20260803


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    data_id: int
    openml_name: str
    feature_type: str
    target: str
    openml_md5: str
    reported_n: int
    reported_p_raw: int
    reported_p_after_corr095: int
    corr_removed_pct: float
    reported_p_over_train_n: float
    discovery_test_r2: float


# OpenML metadata was verified on 2026-08-03.  OpenML data IDs pin an exact
# dataset version; the MD5 guard fails loudly if the server payload disagrees.
SELECTED_DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        "Acute_Myeloid_Leukemia", 46134, "Acute_Myeloid_Leukemia",
        "gene expression", "MYC", "2a4154844160a814694d21aaad05047c",
        1875, 4251, 3972, 6.56, 2.648, 0.7289,
    ),
    DatasetSpec(
        "QSAR-TID-10331", 3711, "QSAR-TID-10331",
        "molecular fingerprints", "MEDIAN_PXC50", "a3877259890c73f9a59a9f20048887dd",
        303, 1024, 689, 32.71, 2.847, 0.6701,
    ),
    DatasetSpec(
        "QSAR-TID-17080", 3790, "QSAR-TID-17080",
        "molecular fingerprints", "MEDIAN_PXC50", "6a9815ddcce40e5f7fc0d83584c2382e",
        1135, 1024, 988, 3.52, 1.088, 0.6358,
    ),
    DatasetSpec(
        "QSAR-TID-12666", 3353, "QSAR-TID-12666",
        "molecular fingerprints", "MEDIAN_PXC50", "b6a953d7581ceae1043c4e589d267a19",
        2243, 1024, 1024, 0.00, 0.571, 0.5758,
    ),
    DatasetSpec(
        "QSAR-TID-10980", 3277, "QSAR-TID-10980",
        "molecular fingerprints", "MEDIAN_PXC50", "04ac805b647af27c2020c08350ff0079",
        5766, 1024, 1024, 0.00, 0.222, 0.5565,
    ),
    DatasetSpec(
        "QSAR-TID-216", 3515, "QSAR-TID-216",
        "molecular fingerprints", "MEDIAN_PXC50", "c47b2a889893eaa91d7bdad7f5776fa2",
        520, 1024, 989, 3.42, 2.377, 0.5442,
    ),
    DatasetSpec(
        "QSAR-TID-11575", 4015, "QSAR-TID-11575",
        "molecular fingerprints", "MEDIAN_PXC50", "25e7157c5d7218ba49b1efd2ec5e9833",
        1490, 1024, 1000, 2.34, 0.839, 0.5351,
    ),
    DatasetSpec(
        "CancerDrugResponse_methylation", 46139, "Cancer_Drug_Response_methylation",
        "methylation", "Methotrexate_GDSC_1008", "5c760b5d7c742f758698a611b723d0cc",
        475, 808, 768, 4.95, 2.021, 0.3518,
    ),
    DatasetSpec(
        "QSAR-TID-10929", 3161, "QSAR-TID-10929",
        "molecular fingerprints", "MEDIAN_PXC50", "8510b5b1ab09dc8faafb049c5b2424c0",
        154, 1024, 872, 14.84, 7.089, 0.3482,
    ),
    DatasetSpec(
        "QSAR-TID-100097", 4041, "QSAR-TID-100097",
        "molecular fingerprints", "MEDIAN_PXC50", "fec79a992ebbd4f59ffd670eda3a3a5e",
        894, 1024, 1024, 0.00, 1.432, 0.3427,
    ),
    DatasetSpec(
        "CancerDrugResponse_expression", 46138, "Cancer_Drug_Response_expression",
        "gene expression", "Methotrexate_GDSC_1008", "9a05f918cbc05eb15e482c1c4deff99e",
        475, 697, 697, 0.00, 1.834, 0.3415,
    ),
    DatasetSpec(
        "QSAR-TID-10547", 4057, "QSAR-TID-10547",
        "molecular fingerprints", "MEDIAN_PXC50", "bd3f1368a806be837237a7e5b50c04ee",
        622, 1024, 934, 8.79, 1.879, 0.2594,
    ),
    DatasetSpec(
        "mtp2", 430, "mtp2",
        "molecular descriptors", "oz1143", "96165cf5a9caa1abffbe64c8b7b587f7",
        274, 1142, 779, 31.79, 3.557, 0.2283,
    ),
)


def _validate_registry() -> None:
    labels = [spec.label for spec in SELECTED_DATASETS]
    ids = [spec.data_id for spec in SELECTED_DATASETS]
    if len(labels) != len(set(labels)) or len(ids) != len(set(ids)):
        raise RuntimeError("Dataset labels and OpenML IDs must be unique")
    for spec in SELECTED_DATASETS:
        if spec.corr_removed_pct > CORR_REMOVAL_LIMIT_PCT:
            raise RuntimeError(f"Selection registry contains high-correlation dataset: {spec.label}")
        if not TEST_R2_MIN_EXCLUSIVE < spec.discovery_test_r2 < TEST_R2_MAX_EXCLUSIVE:
            raise RuntimeError(f"Selection registry contains out-of-range R2: {spec.label}")


def deterministic_split(
    n_rows: int,
    seed: int = DEFAULT_SPLIT_SEED,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted train/test positions using an explicitly pinned PCG64 RNG.

    This reproduces the candidate-screen split contract.  The generated indices
    and their SHA-256 hashes are also persisted, so later runs can consume the
    locked indices directly instead of relying on RNG implementation details.
    """

    if n_rows < 2:
        raise ValueError("At least two rows are required")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie strictly between 0 and 1")

    n_train = int(math.floor((1.0 - test_fraction) * n_rows))
    n_train = min(max(n_train, 1), n_rows - 1)
    rng = np.random.Generator(np.random.PCG64(seed))
    permutation = rng.permutation(n_rows)
    train_position = np.sort(permutation[:n_train]).astype(np.int64, copy=False)
    test_position = np.sort(permutation[n_train:]).astype(np.int64, copy=False)
    return train_position, test_position


def _sha256_indices(indices: np.ndarray) -> str:
    canonical = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _safe_name(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")


def _numeric_target(target: Any) -> tuple[np.ndarray, np.ndarray]:
    series = pd.Series(np.asarray(target).reshape(-1))
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(numeric)
    return numeric[valid], valid


def load_dataset(
    spec: DatasetSpec,
    data_home: Path,
) -> tuple[Any, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Fetch one pinned OpenML dataset and drop only rows with invalid targets."""

    bunch = fetch_openml(
        data_id=spec.data_id,
        as_frame=False,
        parser="auto",
        cache=True,
        data_home=data_home,
    )
    details = dict(bunch.details)
    observed_md5 = str(details.get("md5_checksum", ""))
    if observed_md5 != spec.openml_md5:
        raise RuntimeError(
            f"OpenML MD5 mismatch for {spec.label}: "
            f"expected {spec.openml_md5}, observed {observed_md5}"
        )
    if str(details.get("default_target_attribute")) != spec.target:
        raise RuntimeError(
            f"Target mismatch for {spec.label}: expected {spec.target!r}, "
            f"observed {details.get('default_target_attribute')!r}"
        )

    y, valid_target = _numeric_target(bunch.target)
    original_row_index = np.flatnonzero(valid_target).astype(np.int64, copy=False)
    X = bunch.data[valid_target]
    if sparse.issparse(X):
        X = sparse.csr_matrix(X, dtype=np.float64)
    else:
        X = np.asarray(X, dtype=np.float64, order="C")

    feature_names = [str(name) for name in bunch.feature_names]
    if X.shape[1] != len(feature_names):
        raise RuntimeError(f"Feature-name count mismatch for {spec.label}")
    metadata = {
        "openml_details": {
            key: details.get(key)
            for key in (
                "id", "name", "version", "status", "format", "file_id",
                "md5_checksum", "default_target_attribute", "licence", "url",
            )
        },
        "invalid_target_rows_removed": int((~valid_target).sum()),
    }
    return X, y, original_row_index, feature_names, metadata


def _nonfinite_count(X: Any) -> int:
    if sparse.issparse(X):
        return int((~np.isfinite(X.data)).sum())
    return int((~np.isfinite(X)).sum())


def save_locked_split(
    spec: DatasetSpec,
    output_root: Path,
    data_home: Path,
    seed: int,
    test_fraction: float,
    overwrite: bool,
) -> dict[str, Any]:
    """Download one dataset and save raw train/test arrays plus a split manifest."""

    dataset_dir = output_root / _safe_name(spec.label)
    manifest_path = dataset_dir / "split_manifest.json"
    if manifest_path.exists() and not overwrite:
        print(f"[skip] {spec.label}: {manifest_path} already exists")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    print(f"[load] {spec.label} (OpenML {spec.data_id})", flush=True)
    X, y, original_row_index, feature_names, load_metadata = load_dataset(spec, data_home)
    train_position, test_position = deterministic_split(len(y), seed, test_fraction)
    train_original_row = original_row_index[train_position]
    test_original_row = original_row_index[test_position]

    dataset_dir.mkdir(parents=True, exist_ok=True)
    if sparse.issparse(X):
        sparse.save_npz(dataset_dir / "X_train.npz", X[train_position], compressed=True)
        sparse.save_npz(dataset_dir / "X_test.npz", X[test_position], compressed=True)
        x_storage = "scipy_csr_npz"
    else:
        np.save(dataset_dir / "X_train.npy", X[train_position], allow_pickle=False)
        np.save(dataset_dir / "X_test.npy", X[test_position], allow_pickle=False)
        x_storage = "numpy_npy"

    np.save(dataset_dir / "y_train.npy", y[train_position], allow_pickle=False)
    np.save(dataset_dir / "y_test.npy", y[test_position], allow_pickle=False)
    np.save(dataset_dir / "train_original_row_index.npy", train_original_row, allow_pickle=False)
    np.save(dataset_dir / "test_original_row_index.npy", test_original_row, allow_pickle=False)
    (dataset_dir / "feature_names.json").write_text(
        json.dumps(feature_names, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {
        "dataset": asdict(spec),
        "selection_rule": {
            "exclude_if_corr_removed_pct_gt": CORR_REMOVAL_LIMIT_PCT,
            "require_discovery_test_r2_gt": TEST_R2_MIN_EXCLUSIVE,
            "require_discovery_test_r2_lt": TEST_R2_MAX_EXCLUSIVE,
        },
        "split": {
            "algorithm": "numpy.random.Generator(PCG64).permutation",
            "seed": int(seed),
            "test_fraction": float(test_fraction),
            "n_total_after_invalid_target_removal": int(len(y)),
            "n_train": int(len(train_position)),
            "n_test": int(len(test_position)),
            "train_original_row_index_sha256": _sha256_indices(train_original_row),
            "test_original_row_index_sha256": _sha256_indices(test_original_row),
        },
        "matrix": {
            "n_features_raw": int(X.shape[1]),
            "x_storage": x_storage,
            "x_dtype": str(X.dtype),
            "nonfinite_predictor_cells": _nonfinite_count(X),
            "preprocessing_applied_before_split": False,
        },
        **load_metadata,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[done] {spec.label}: train={len(train_position)}, test={len(test_position)}, "
        f"p={X.shape[1]}, storage={x_storage}",
        flush=True,
    )
    return manifest


def print_registry() -> None:
    header = (
        f"{'dataset':36} {'did':>6} {'N':>6} {'P':>6} "
        f"{'corr_rm%':>9} {'P_after':>8} {'P/n':>6} {'R2':>8}"
    )
    print(header)
    print("-" * len(header))
    for spec in SELECTED_DATASETS:
        print(
            f"{spec.label:36} {spec.data_id:6d} {spec.reported_n:6d} "
            f"{spec.reported_p_raw:6d} {spec.corr_removed_pct:9.2f} "
            f"{spec.reported_p_after_corr095:8d} "
            f"{spec.reported_p_over_train_n:6.3f} "
            f"{spec.discovery_test_r2:8.4f}"
        )


def run_self_test() -> None:
    _validate_registry()
    train_a, test_a = deterministic_split(137, DEFAULT_SPLIT_SEED, 0.20)
    train_b, test_b = deterministic_split(137, DEFAULT_SPLIT_SEED, 0.20)
    assert np.array_equal(train_a, train_b)
    assert np.array_equal(test_a, test_b)
    assert len(np.intersect1d(train_a, test_a)) == 0
    assert np.array_equal(np.sort(np.concatenate([train_a, test_a])), np.arange(137))
    print("self-test passed")
    print("train index SHA-256:", _sha256_indices(train_a))
    print("test index SHA-256: ", _sha256_indices(test_a))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print the retained registry")
    parser.add_argument("--self-test", action="store_true", help="test registry and split invariants")
    parser.add_argument("--all", action="store_true", help="download all 13 retained datasets")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="LABEL",
        help="download one retained label; may be repeated",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("real_regression_data"),
        help="directory for saved train/test arrays and manifests",
    )
    parser.add_argument(
        "--data-home",
        type=Path,
        default=None,
        help="scikit-learn OpenML cache (default: OUTPUT_DIR/.openml_cache)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    _validate_registry()

    if args.list:
        print_registry()
    if args.self_test:
        run_self_test()

    by_label = {spec.label: spec for spec in SELECTED_DATASETS}
    unknown = sorted(set(args.dataset) - set(by_label))
    if unknown:
        raise SystemExit(
            "Unknown dataset label(s): " + ", ".join(unknown) +
            ". Run with --list for valid labels."
        )

    selected = list(SELECTED_DATASETS) if args.all else [by_label[x] for x in args.dataset]
    if not selected:
        if not args.list and not args.self_test:
            print_registry()
            print("\nNo download requested. Use --dataset LABEL or --all.")
        return

    output_root = args.output_dir.resolve()
    data_home = (args.data_home or (output_root / ".openml_cache")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    data_home.mkdir(parents=True, exist_ok=True)

    manifests = []
    failures = []
    for spec in selected:
        try:
            manifests.append(
                save_locked_split(
                    spec,
                    output_root=output_root,
                    data_home=data_home,
                    seed=args.seed,
                    test_fraction=args.test_fraction,
                    overwrite=args.overwrite,
                )
            )
        except Exception as exc:  # continue a large batch after one failure
            failures.append({
                "label": spec.label,
                "data_id": spec.data_id,
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"[fail] {spec.label}: {type(exc).__name__}: {exc}", flush=True)

    batch_manifest = {
        "selection_rule": {
            "exclude_if_corr_removed_pct_gt": CORR_REMOVAL_LIMIT_PCT,
            "require_discovery_test_r2_gt": TEST_R2_MIN_EXCLUSIVE,
            "require_discovery_test_r2_lt": TEST_R2_MAX_EXCLUSIVE,
        },
        "requested": [spec.label for spec in selected],
        "successful": [m["dataset"]["label"] for m in manifests],
        "failures": failures,
        "seed": int(args.seed),
        "test_fraction": float(args.test_fraction),
    }
    (output_root / "batch_manifest.json").write_text(
        json.dumps(batch_manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if failures:
        raise SystemExit(f"{len(failures)} dataset(s) failed; see batch_manifest.json")


if __name__ == "__main__":
    main()
