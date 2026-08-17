#!/usr/bin/env python3
"""Prepare leakage-auditable real regression tasks from downloaded raw files."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.lib.format import open_memmap
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse
from scipy.io import loadmat


SEED = 20260728
RNG = np.random.default_rng(SEED)
KMER_K = 6
KMER_DIM = 4**KMER_K
MORGAN_DIM = 2048
SDSS_DIM = 4096


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path, chunk_size: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_indices(n: int, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or n <= limit:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=limit, replace=False))


def random_split(n: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    split = np.full(n, 2, dtype=np.int8)
    n_context = int(math.floor(0.6 * n))
    n_screen = int(math.floor(0.2 * n))
    split[order[:n_context]] = 0
    split[order[n_context : n_context + n_screen]] = 1
    return split


def save_task(
    output_root: Path,
    task_id: str,
    X: np.ndarray | sparse.spmatrix | Path,
    y: np.ndarray,
    split: np.ndarray,
    metadata: dict[str, Any],
    feature_names: list[str] | None = None,
) -> None:
    task_root = output_root / task_id
    task_root.mkdir(parents=True, exist_ok=True)
    if isinstance(X, Path):
        x_path = X
        storage = "dense_npy"
        shape = list(np.load(x_path, mmap_mode="r").shape)
    elif sparse.issparse(X):
        x_path = task_root / "X.npz"
        sparse.save_npz(x_path, X.tocsr().astype(np.float32))
        storage = "sparse_csr_npz"
        shape = list(X.shape)
    else:
        x_path = task_root / "X.npy"
        np.save(x_path, np.asarray(X, dtype=np.float32))
        storage = "dense_npy"
        shape = list(X.shape)
    np.save(task_root / "y.npy", np.asarray(y, dtype=np.float64))
    np.save(task_root / "split.npy", np.asarray(split, dtype=np.int8))
    if feature_names is not None:
        (task_root / "feature_names.json").write_text(
            json.dumps(feature_names) + "\n"
        )
    unique, counts = np.unique(y[np.isfinite(y)], return_counts=True)
    metadata = {
        **metadata,
        "task_id": task_id,
        "created_unix": time.time(),
        "seed": SEED,
        "X_storage": storage,
        "X_path": str(x_path),
        "N": int(shape[0]),
        "P": int(shape[1]),
        "P_over_N": float(shape[1] / shape[0]),
        "y_finite_fraction": float(np.isfinite(y).mean()),
        "y_unique_count": int(unique.size),
        "y_max_value_fraction": float(counts.max() / counts.sum()) if counts.size else 1.0,
        "split_counts": {
            "context": int((split == 0).sum()),
            "screening": int((split == 1).sum()),
            "locked_test": int((split == 2).sum()),
        },
    }
    json_dump(task_root / "metadata.json", metadata)
    print(
        f"prepared {task_id}: N={shape[0]:,} P={shape[1]:,} "
        f"splits={metadata['split_counts']}",
        flush=True,
    )


def read_parquet_parts(paths: Iterable[Path], columns: list[str]) -> pd.DataFrame:
    frames = [pd.read_parquet(path, columns=columns) for path in sorted(paths)]
    return pd.concat(frames, ignore_index=True)


def base4_kmer_csr(sequences: list[str], k: int = KMER_K) -> sparse.csr_matrix:
    mapping = np.full(256, -1, dtype=np.int16)
    for char, value in zip(b"ACGT", range(4)):
        mapping[char] = value
        mapping[char + 32] = value
    dim = 4**k
    mask = dim - 1
    indptr = [0]
    all_indices: list[np.ndarray] = []
    all_data: list[np.ndarray] = []
    for sequence in sequences:
        cleaned = re.sub(r"[^ACGTacgt]", "", sequence).encode()
        codes = mapping[np.frombuffer(cleaned, dtype=np.uint8)]
        counts: dict[int, int] = {}
        code = 0
        valid = 0
        for value in codes:
            if value < 0:
                code = 0
                valid = 0
                continue
            code = ((code << 2) | int(value)) & mask
            valid += 1
            if valid >= k:
                counts[code] = counts.get(code, 0) + 1
        if counts:
            indices = np.fromiter(sorted(counts), dtype=np.int32)
            values = np.fromiter((counts[int(index)] for index in indices), dtype=np.float32)
            values /= max(float(values.sum()), 1.0)
        else:
            indices = np.empty(0, dtype=np.int32)
            values = np.empty(0, dtype=np.float32)
        all_indices.append(indices)
        all_data.append(values)
        indptr.append(indptr[-1] + indices.size)
    return sparse.csr_matrix(
        (
            np.concatenate(all_data) if all_data else np.empty(0, dtype=np.float32),
            np.concatenate(all_indices) if all_indices else np.empty(0, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(sequences), dim),
        dtype=np.float32,
    )


def morgan_csr(smiles: list[str]) -> tuple[sparse.csr_matrix, np.ndarray]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=MORGAN_DIM)
    indptr = [0]
    indices_parts: list[np.ndarray] = []
    valid = np.zeros(len(smiles), dtype=bool)
    for row, text in enumerate(smiles):
        molecule = Chem.MolFromSmiles(str(text))
        if molecule is None:
            indices = np.empty(0, dtype=np.int32)
        else:
            fingerprint = generator.GetFingerprint(molecule)
            on_bits = list(fingerprint.GetOnBits())
            indices = np.asarray(on_bits, dtype=np.int32)
            valid[row] = True
        indices_parts.append(indices)
        indptr.append(indptr[-1] + indices.size)
    indices = np.concatenate(indices_parts) if indices_parts else np.empty(0, dtype=np.int32)
    data = np.ones(indices.size, dtype=np.float32)
    matrix = sparse.csr_matrix(
        (data, indices, np.asarray(indptr, dtype=np.int64)),
        shape=(len(smiles), MORGAN_DIM),
        dtype=np.float32,
    )
    return matrix, valid


def prepare_sequence_task(
    output_root: Path,
    task_id: str,
    split_paths: dict[str, list[Path]],
    sequence_column: str,
    target_column: str,
    limits: dict[str, int | None],
    source: str,
) -> None:
    sequence_parts: list[str] = []
    y_parts: list[np.ndarray] = []
    split_parts: list[np.ndarray] = []
    split_code = {"train": 0, "validation": 1, "test": 2}
    for split_name, paths in split_paths.items():
        frame = read_parquet_parts(paths, [sequence_column, target_column])
        indices = deterministic_indices(
            len(frame), limits.get(split_name), SEED + split_code[split_name] * 101
        )
        frame = frame.iloc[indices]
        y = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(float)
        valid = np.isfinite(y) & frame[sequence_column].notna().to_numpy()
        sequence_parts.extend(frame.loc[valid, sequence_column].astype(str).tolist())
        y_parts.append(y[valid])
        split_parts.append(np.full(valid.sum(), split_code[split_name], dtype=np.int8))
    X = base4_kmer_csr(sequence_parts)
    y_all = np.concatenate(y_parts)
    split_all = np.concatenate(split_parts)
    save_task(
        output_root,
        task_id,
        X,
        y_all,
        split_all,
        {
            "source": source,
            "target": target_column,
            "encoding": "normalized_6mer_count_ACGT_4096",
            "split_provenance": "official_or_train-derived_fixed_before_target",
            "reality_status": "empirical",
            "test_y_access_policy": "indices fixed before target values; test used only post-acceptance",
        },
    )


def prepare_molecule_task(
    output_root: Path,
    task_id: str,
    split_paths: dict[str, list[Path]],
    smiles_column: str,
    target_column: str,
    source: str,
    max_total: int | None = None,
    random_split_if_train_only: bool = False,
) -> None:
    smiles_all: list[str] = []
    y_parts: list[np.ndarray] = []
    split_parts: list[np.ndarray] = []
    split_code = {"train": 0, "validation": 1, "test": 2}
    for split_name, paths in split_paths.items():
        frame = read_parquet_parts(paths, [smiles_column, target_column])
        limit = max_total if len(split_paths) == 1 else None
        indices = deterministic_indices(
            len(frame), limit, SEED + split_code.get(split_name, 0) * 101
        )
        frame = frame.iloc[indices]
        smiles_all.extend(frame[smiles_column].astype(str).tolist())
        y_parts.append(pd.to_numeric(frame[target_column], errors="coerce").to_numpy(float))
        split_parts.append(
            np.full(len(frame), split_code.get(split_name, 0), dtype=np.int8)
        )
    X, valid_smiles = morgan_csr(smiles_all)
    y = np.concatenate(y_parts)
    split = np.concatenate(split_parts)
    valid = valid_smiles & np.isfinite(y)
    X = X[valid]
    y = y[valid]
    split = split[valid]
    if random_split_if_train_only:
        split = random_split(len(y), SEED)
    save_task(
        output_root,
        task_id,
        X,
        y,
        split,
        {
            "source": source,
            "target": target_column,
            "encoding": "RDKit_Morgan_radius2_2048_binary",
            "split_provenance": (
                "deterministic_random_60_20_20"
                if random_split_if_train_only
                else "official"
            ),
            "reality_status": "empirical_measurement",
        },
    )


def parse_spectrum(text: str, target_dim: int = SDSS_DIM) -> np.ndarray:
    values = np.fromstring(str(text).strip()[1:-1], sep=",", dtype=np.float32)
    if values.size == 0:
        return np.full(target_dim, np.nan, dtype=np.float32)
    if values.size == target_dim:
        return values
    source_x = np.linspace(0.0, 1.0, values.size, dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, target_dim, dtype=np.float32)
    return np.interp(target_x, source_x, values).astype(np.float32)


def prepare_sdss(output_root: Path, hf_root: Path) -> None:
    base = hf_root / "BrunoBarreto__sdss_dr12_stars_regression" / "default"
    limits = {"train": 20000, "validation": 4000, "test": 5000}
    split_code = {"train": 0, "validation": 1, "test": 2}
    rows: list[tuple[str, float, int]] = []
    for split_name in ("train", "validation", "test"):
        paths = sorted((base / split_name).glob("*.parquet"))
        table = pa.concat_tables(
            [pq.read_table(path, columns=["processed_flux", "catalog_teff"]) for path in paths]
        )
        indices = deterministic_indices(
            len(table), limits[split_name], SEED + split_code[split_name] * 101
        )
        selected = table.take(pa.array(indices))
        spectra = selected.column("processed_flux").to_pylist()
        targets = np.asarray(selected.column("catalog_teff").to_numpy(), dtype=float)
        rows.extend(
            (str(spectrum), float(target), split_code[split_name])
            for spectrum, target in zip(spectra, targets)
        )
    task_root = output_root / "hf_sdss_dr12_teff_flux4096"
    task_root.mkdir(parents=True, exist_ok=True)
    x_path = task_root / "X.npy"
    X = open_memmap(x_path, mode="w+", dtype=np.float32, shape=(len(rows), SDSS_DIM))
    y = np.empty(len(rows), dtype=np.float64)
    split = np.empty(len(rows), dtype=np.int8)
    for index, (spectrum, target, split_value) in enumerate(rows):
        X[index] = parse_spectrum(spectrum)
        y[index] = target
        split[index] = split_value
    X.flush()
    save_task(
        output_root,
        "hf_sdss_dr12_teff_flux4096",
        x_path,
        y,
        split,
        {
            "source": "HuggingFace BrunoBarreto/sdss_dr12_stars_regression",
            "target": "catalog_teff",
            "encoding": "processed_flux_linear_interpolation_to_4096",
            "split_provenance": "official; deterministic row subsampling before target",
            "reality_status": "empirical_stellar_spectra",
        },
    )


def prepare_uci_numeric(
    output_root: Path,
    task_id: str,
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    metadata: dict[str, Any],
    split: np.ndarray | None = None,
) -> None:
    X = frame[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    y = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(float)
    valid = np.isfinite(y)
    X = X[valid]
    y = y[valid]
    actual_split = random_split(len(y), SEED) if split is None else split[valid]
    save_task(
        output_root,
        task_id,
        X,
        y,
        actual_split,
        metadata,
        feature_columns,
    )


def prepare_uci_standard(output_root: Path, uci_root: Path) -> None:
    # CT slice localization: patient-grouped split prevents slice leakage.
    ct_zip = next(uci_root.glob("206_*.zip"))
    with zipfile.ZipFile(ct_zip) as archive:
        frame = pd.read_csv(archive.open("slice_localization_data.csv"))
    patients = frame["patientId"].to_numpy()
    unique_patients = np.unique(patients)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(unique_patients)
    patient_split = {
        patient: 0 if i < 0.6 * len(order) else 1 if i < 0.8 * len(order) else 2
        for i, patient in enumerate(order)
    }
    split = np.asarray([patient_split[patient] for patient in patients], dtype=np.int8)
    prepare_uci_numeric(
        output_root,
        "uci_ct_slice_location",
        frame,
        [column for column in frame if column.startswith("value")],
        "reference",
        {
            "source": "UCI 206",
            "target": "reference",
            "encoding": "published_numeric_features",
            "split_provenance": "patient-grouped deterministic 60_20_20",
            "reality_status": "empirical_medical_image_features",
        },
        split,
    )

    # UJIIndoorLoc: official validation is locked test; training split creates context/screening.
    uji_zip = next(uci_root.glob("310_*.zip"))
    with zipfile.ZipFile(uji_zip) as archive:
        train = pd.read_csv(archive.open("UJIndoorLoc/trainingData.csv"))
        test = pd.read_csv(archive.open("UJIndoorLoc/validationData.csv"))
    wap_columns = [column for column in train if column.startswith("WAP")]
    train_split = random_split(len(train), SEED)
    train_split[train_split == 2] = 1
    combined = pd.concat([train, test], ignore_index=True)
    split = np.concatenate([train_split, np.full(len(test), 2, dtype=np.int8)])
    prepare_uci_numeric(
        output_root,
        "uci_ujiindoor_longitude",
        combined,
        wap_columns,
        "LONGITUDE",
        {
            "source": "UCI 310",
            "target": "LONGITUDE",
            "encoding": "520_wifi_RSSI_features_only",
            "split_provenance": "train deterministic context/screening; official validation locked test",
            "reality_status": "empirical_indoor_wifi",
        },
        split,
    )

    # Communities and Crime normalized.
    crime_zip = next(uci_root.glob("183_*.zip"))
    with zipfile.ZipFile(crime_zip) as archive:
        data = pd.read_csv(archive.open("communities.data"), header=None, na_values="?")
    X = data.iloc[:, 5:-1].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    y = pd.to_numeric(data.iloc[:, -1], errors="coerce").to_numpy(float)
    save_task(
        output_root,
        "uci_communities_violent_crime",
        X,
        y,
        random_split(len(y), SEED),
        {
            "source": "UCI 183",
            "target": "ViolentCrimesPerPop",
            "encoding": "published_numeric_features; identifiers removed",
            "split_provenance": "deterministic random 60_20_20",
            "reality_status": "empirical_community_statistics",
        },
    )

    # Geographical origin of music: last two columns are latitude/longitude.
    music_zip = next(uci_root.glob("315_*.zip"))
    member = "Geographical Original of Music/default_features_1059_tracks.txt"
    with zipfile.ZipFile(music_zip) as archive:
        data = pd.read_csv(archive.open(member), header=None)
    save_task(
        output_root,
        "uci_music_geographic_latitude",
        data.iloc[:, :-2].to_numpy(np.float32),
        data.iloc[:, -2].to_numpy(float),
        random_split(len(data), SEED),
        {
            "source": "UCI 315",
            "target": "latitude",
            "encoding": "published_audio_features",
            "split_provenance": "deterministic random 60_20_20",
            "reality_status": "empirical_music",
        },
    )

    # Online News Popularity.
    news_zip = next(uci_root.glob("332_*.zip"))
    with zipfile.ZipFile(news_zip) as archive:
        frame = pd.read_csv(
            archive.open("OnlineNewsPopularity/OnlineNewsPopularity.csv"),
            skipinitialspace=True,
        )
    frame.columns = [column.strip() for column in frame]
    feature_columns = [
        column for column in frame if column not in {"url", "shares", "timedelta"}
    ]
    prepare_uci_numeric(
        output_root,
        "uci_online_news_shares",
        frame,
        feature_columns,
        "shares",
        {
            "source": "UCI 332",
            "target": "shares",
            "encoding": "published_numeric_features; URL/timedelta removed",
            "split_provenance": "deterministic random 60_20_20",
            "reality_status": "empirical_online_news",
        },
    )

    # Superconductivity.
    superconduct_zip = next(uci_root.glob("464_*.zip"))
    with zipfile.ZipFile(superconduct_zip) as archive:
        frame = pd.read_csv(archive.open("train.csv"))
    prepare_uci_numeric(
        output_root,
        "uci_superconductivity_critical_temp",
        frame,
        [column for column in frame if column != "critical_temp"],
        "critical_temp",
        {
            "source": "UCI 464",
            "target": "critical_temp",
            "encoding": "published_material_composition_statistics",
            "split_provenance": "deterministic random 60_20_20",
            "reality_status": "empirical_material_database",
        },
    )

    # BlogFeedback: official temporal test files are locked test.
    blog_zip = next(uci_root.glob("304_*.zip"))
    with zipfile.ZipFile(blog_zip) as archive:
        train = pd.read_csv(archive.open("blogData_train.csv"), header=None)
        test_members = sorted(
            name for name in archive.namelist() if "blogData_test" in name
        )
        test = pd.concat(
            [pd.read_csv(archive.open(name), header=None) for name in test_members],
            ignore_index=True,
        )
    train_split = random_split(len(train), SEED)
    train_split[train_split == 2] = 1
    frame = pd.concat([train, test], ignore_index=True)
    split = np.concatenate([train_split, np.full(len(test), 2, dtype=np.int8)])
    X = frame.iloc[:, :-1].to_numpy(np.float32)
    y = frame.iloc[:, -1].to_numpy(float)
    save_task(
        output_root,
        "uci_blogfeedback_comments",
        X,
        y,
        split,
        {
            "source": "UCI 304",
            "target": "next_24h_comment_count",
            "encoding": "published_numeric_features",
            "split_provenance": "train context/screening; official chronological files locked test",
            "reality_status": "empirical_blog",
        },
    )


def prepare_buzz(output_root: Path, uci_root: Path) -> None:
    buzz_zip = next(uci_root.glob("248_*.zip"))
    with zipfile.ZipFile(buzz_zip) as archive:
        nested = archive.read("regression.tar.gz")
    with tarfile.open(fileobj=io.BytesIO(nested), mode="r:gz") as tar:
        for name, task_id in (
            ("./regression/TomsHardware/TomsHardware.data", "uci_buzz_tomshardware"),
            ("./regression/Twitter/Twitter.data", "uci_buzz_twitter"),
        ):
            extracted = tar.extractfile(name)
            if extracted is None:
                continue
            frame = pd.read_csv(extracted, header=None)
            save_task(
                output_root,
                task_id,
                frame.iloc[:, :-1].to_numpy(np.float32),
                frame.iloc[:, -1].to_numpy(float),
                random_split(len(frame), SEED),
                {
                    "source": "UCI 248",
                    "target": "buzz_count",
                    "encoding": "published_numeric_features",
                    "split_provenance": "deterministic random 60_20_20",
                    "reality_status": "empirical_social_media",
                },
            )


def prepare_twin_gas(output_root: Path, uci_root: Path) -> None:
    gas_zip = next(uci_root.glob("361_*.zip"))
    task_root = output_root / "uci_twin_gas_concentration_raw"
    task_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(gas_zip) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if name.startswith("data1/") and name.endswith(".txt")
        )
        first = np.loadtxt(io.BytesIO(archive.read(members[0])), dtype=np.float32)
        target_length = 60000
        p = target_length * 8
        x_path = task_root / "X.npy"
        X = open_memmap(x_path, mode="w+", dtype=np.float32, shape=(len(members), p))
        y = np.empty(len(members), dtype=np.float64)
        split = np.empty(len(members), dtype=np.int8)
        for index, name in enumerate(members):
            values = np.loadtxt(io.BytesIO(archive.read(name)), dtype=np.float32)
            if values.ndim != 2 or values.shape[1] != 9 or values.shape[0] < 2:
                raise RuntimeError(f"Invalid twin-gas shape {values.shape}: {name}")
            target_time = np.linspace(
                float(values[0, 0]),
                float(values[-1, 0]),
                target_length,
                dtype=np.float64,
            )
            resampled = np.empty((8, target_length), dtype=np.float32)
            for channel in range(8):
                resampled[channel] = np.interp(
                    target_time,
                    values[:, 0].astype(np.float64, copy=False),
                    values[:, channel + 1].astype(np.float64, copy=False),
                )
            X[index] = resampled.reshape(-1)
            match = re.search(r"/B(\d+)_G(Ea|Ey|CO|Me)_F(\d{3})_", name)
            if not match:
                raise RuntimeError(f"Cannot parse twin-gas filename: {name}")
            board = int(match.group(1))
            gas = match.group(2)
            level = int(match.group(3))
            y[index] = level * (2.5 if gas in {"CO", "Me"} else 1.25)
            split[index] = 0 if board <= 3 else 1 if board == 4 else 2
            if (index + 1) % 80 == 0:
                print(f"twin gas parsed {index + 1}/{len(members)}", flush=True)
        X.flush()
    save_task(
        output_root,
        "uci_twin_gas_concentration_raw",
        x_path,
        y,
        split,
        {
            "source": "UCI 361",
            "target": "physical_gas_concentration_ppm",
            "encoding": (
                "8 sensor resistance traces interpolated on each experiment's "
                "60000-point uniform time grid; time column then removed"
            ),
            "split_provenance": "board-grouped B1-B3 context B4 screening B5 locked test",
            "reality_status": "empirical_sensor_experiments",
        },
    )


def prepare_hf_molecules(output_root: Path, hf_root: Path) -> None:
    specs = [
        (
            "hf_moleculenet_esol_morgan2048",
            "scikit-fingerprints__MoleculeNet_ESOL",
            "SMILES",
            "label",
            True,
        ),
        (
            "hf_moleculenet_lipophilicity_morgan2048",
            "scikit-fingerprints__MoleculeNet_Lipophilicity",
            "SMILES",
            "label",
            True,
        ),
        (
            "hf_moleculenet_freesolv_morgan2048",
            "scikit-fingerprints__MoleculeNet_FreeSolv",
            "SMILES",
            "label",
            True,
        ),
        ("hf_bace_regression_morgan2048", "zpn__bace_regression", "smiles", "target", False),
        ("hf_clearance_morgan2048", "zpn__clearance", "smiles", "target", False),
    ]
    for task_id, repo_dir, smiles_column, target_column, random_only in specs:
        base = hf_root / repo_dir / "default"
        paths = {
            split_name: sorted((base / split_name).glob("*.parquet"))
            for split_name in ("train", "validation", "test")
            if (base / split_name).exists()
        }
        prepare_molecule_task(
            output_root,
            task_id,
            paths,
            smiles_column,
            target_column,
            f"HuggingFace {repo_dir.replace('__', '/')}",
            random_split_if_train_only=random_only,
        )

    qm9 = hf_root / "yairschiff__qm9" / "default" / "train"
    prepare_molecule_task(
        output_root,
        "hf_qm9_gap_morgan2048",
        {"train": sorted(qm9.glob("*.parquet"))},
        "canonical_smiles",
        "gap",
        "HuggingFace yairschiff/qm9",
        max_total=50000,
        random_split_if_train_only=True,
    )


def prepare_hf_sequences(output_root: Path, hf_root: Path) -> None:
    malinois = hf_root / "HuggingFaceBio__malinois-mpra-regression" / "default"
    prepare_sequence_task(
        output_root,
        "hf_malinois_k562_log2fc_kmer6",
        {
            split_name: sorted((malinois / split_name).glob("*.parquet"))
            for split_name in ("train", "validation", "test")
        },
        "sequence",
        "K562_log2FC",
        {"train": 50000, "validation": 10000, "test": 10000},
        "HuggingFaceBio/malinois-mpra-regression",
    )
    hu = hf_root / "hu-lab__Gene_Expression_Prediction" / "default"
    # No labeled test split: split the official validation by deterministic index.
    train = read_parquet_parts(sorted((hu / "train").glob("*.parquet")), ["sequence", "label"])
    validation = read_parquet_parts(
        sorted((hu / "validation").glob("*.parquet")), ["sequence", "label"]
    )
    train = train.iloc[deterministic_indices(len(train), 50000, SEED)]
    valid_indices = deterministic_indices(len(validation), 20000, SEED + 101)
    validation = validation.iloc[valid_indices].reset_index(drop=True)
    midpoint = len(validation) // 2
    temp_root = output_root / "_temporary_hu"
    temp_root.mkdir(exist_ok=True)
    # Reuse the sequence encoder directly without writing temporary parquet.
    frames = [
        (train, 0),
        (validation.iloc[:midpoint], 1),
        (validation.iloc[midpoint:], 2),
    ]
    sequences: list[str] = []
    y_parts: list[np.ndarray] = []
    split_parts: list[np.ndarray] = []
    for frame, code in frames:
        y = pd.to_numeric(frame["label"], errors="coerce").to_numpy(float)
        valid = np.isfinite(y) & frame["sequence"].notna().to_numpy()
        sequences.extend(frame.loc[valid, "sequence"].astype(str).tolist())
        y_parts.append(y[valid])
        split_parts.append(np.full(valid.sum(), code, dtype=np.int8))
    save_task(
        output_root,
        "hf_hulab_gene_expression_kmer6",
        base4_kmer_csr(sequences),
        np.concatenate(y_parts),
        np.concatenate(split_parts),
        {
            "source": "HuggingFace hu-lab/Gene_Expression_Prediction",
            "target": "label",
            "encoding": "normalized_6mer_count_ACGT_4096",
            "split_provenance": "official train; validation deterministically halved screen/test",
            "reality_status": "requires_card_provenance_review",
        },
    )
    zrimec = hf_root / "MarcAmil__Zrimec_gene_expression_transcripts" / "default"
    train_paths = sorted((zrimec / "train").glob("*.parquet"))
    test_paths = sorted((zrimec / "test").glob("*.parquet"))
    train = read_parquet_parts(train_paths, ["TU_sequence", "residual_boxcox_TPM"])
    train_indices = random_split(len(train), SEED)
    train_indices[train_indices == 2] = 1
    test = read_parquet_parts(test_paths, ["TU_sequence", "residual_boxcox_TPM"])
    frame = pd.concat([train, test], ignore_index=True)
    y = pd.to_numeric(frame["residual_boxcox_TPM"], errors="coerce").to_numpy(float)
    valid = np.isfinite(y) & frame["TU_sequence"].notna().to_numpy()
    split = np.concatenate(
        [train_indices, np.full(len(test), 2, dtype=np.int8)]
    )[valid]
    save_task(
        output_root,
        "hf_zrimec_gene_expression_kmer6",
        base4_kmer_csr(frame.loc[valid, "TU_sequence"].astype(str).tolist()),
        y[valid],
        split,
        {
            "source": "HuggingFace MarcAmil/Zrimec_gene_expression_transcripts",
            "target": "residual_boxcox_TPM",
            "encoding": "normalized_6mer_count_ACGT_4096",
            "split_provenance": "train context/screening; official test locked test",
            "reality_status": "empirical_RNAseq_expression",
        },
    )


def prepare_hf_tabular(output_root: Path, hf_root: Path) -> None:
    base = hf_root / "inria-soda__tabular-benchmark"
    for config_root in sorted(base.glob("reg_*")):
        paths = sorted(config_root.glob("train/*.parquet"))
        if not paths:
            continue
        frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        indices = deterministic_indices(len(frame), 50000, SEED)
        frame = frame.iloc[indices].reset_index(drop=True)
        target_column = frame.columns[-1]
        feature_columns = list(frame.columns[:-1])
        y = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(float)
        valid = np.isfinite(y)
        frame = frame.loc[valid, feature_columns]
        y = y[valid]
        categorical_indices = [
            index
            for index, column in enumerate(feature_columns)
            if (
                config_root.name.startswith("reg_cat_")
                and pd.api.types.is_integer_dtype(frame[column])
                and frame[column].nunique(dropna=True) <= 256
            )
        ]
        X = frame.apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        synthetic_name = any(
            token in config_root.name.lower()
            for token in ("topo_", "yprop_", "friedman", "synthetic")
        )
        save_task(
            output_root,
            f"hf_tabbench_{config_root.name}",
            X,
            y,
            random_split(len(y), SEED),
            {
                "source": "HuggingFace inria-soda/tabular-benchmark",
                "source_config": config_root.name,
                "target": str(target_column),
                "encoding": (
                    "published_numeric_or_integer_category_codes; categorical indices "
                    "recorded for train-only one-hot"
                ),
                "categorical_feature_indices": categorical_indices,
                "split_provenance": "deterministic random 60_20_20 after label-blind row cap 50000",
                "reality_status": "synthetic_benchmark"
                if synthetic_name
                else "real_benchmark_requires_license_review",
            },
            [str(column) for column in feature_columns],
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--families",
        nargs="+",
        default=[
            "uci_standard",
            "buzz",
            "molecules",
            "sequences",
            "sdss",
            "twin_gas",
            "tabular_hf",
        ],
    )
    args = parser.parse_args()
    root = args.root.resolve()
    uci_root = root / "raw" / "uci"
    hf_root = root / "raw" / "huggingface"
    output_root = root / "processed"
    output_root.mkdir(exist_ok=True)
    rejected = [
        {
            "source_id": "UCI 318",
            "name": "NoisyOffice",
            "reason": "image/OCR bundle without a continuous scalar target contract",
        },
        {
            "source_id": "UCI 321",
            "name": "ElectricityLoadDiagrams20112014",
            "reason": "forecasting time series without a repository-defined scalar target",
        },
        {
            "source_id": "UCI 328",
            "name": "Greenhouse Gas Observing Network",
            "reason": "multi-output inverse problem; no unambiguous scalar row target",
        },
        {
            "source_id": "UCI 378",
            "name": "DrivFace",
            "reason": "head-angle target has fewer than 10 unique values",
        },
        {
            "source_id": "UCI 447",
            "name": "Hydraulic condition monitoring",
            "reason": "targets are 3-4-level ordinal conditions, not continuous regression",
        },
        {
            "source_id": "UCI 554",
            "name": "CNNpred",
            "reason": "published outcome is directional classification/time leakage risk",
        },
        {
            "source_id": "UCI 606",
            "name": "Wikipedia Math Essentials",
            "reason": "graph/time-series bundle without an unambiguous scalar target",
        },
        {
            "source_id": "UCI 882",
            "name": "Large-scale Wave Energy Farm",
            "reason": "simulated; Total_Power is directly derived from included Power columns",
        },
        {
            "source_id": "HuggingFace wanglab/promoter_rbs_gene_expression",
            "name": "promoter_rbs_gene_expression",
            "reason": "HTTP 401 gated/private",
        },
    ]
    json_dump(root / "manifests" / "pre_screen_rejections.json", rejected)

    for family in args.families:
        print(f"== preparing family {family}", flush=True)
        if family == "uci_standard":
            prepare_uci_standard(output_root, uci_root)
        elif family == "buzz":
            prepare_buzz(output_root, uci_root)
        elif family == "molecules":
            prepare_hf_molecules(output_root, hf_root)
        elif family == "sequences":
            prepare_hf_sequences(output_root, hf_root)
        elif family == "sdss":
            prepare_sdss(output_root, hf_root)
        elif family == "twin_gas":
            prepare_twin_gas(output_root, uci_root)
        elif family == "tabular_hf":
            prepare_hf_tabular(output_root, hf_root)
        else:
            raise ValueError(f"Unknown family: {family}")

    prepared = sorted(path.parent.name for path in output_root.glob("*/metadata.json"))
    json_dump(
        root / "manifests" / "preparation_manifest.json",
        {
            "created_unix": time.time(),
            "families": args.families,
            "prepared_tasks": prepared,
            "prepared_count": len(prepared),
            "seed": SEED,
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
    )
    print(f"Prepared {len(prepared)} tasks", flush=True)


if __name__ == "__main__":
    main()
