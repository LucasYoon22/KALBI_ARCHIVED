#!/usr/bin/env python3
"""Download pinned UCI/HF regression candidates with checksums and disk guard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


MIN_FREE_BYTES = 300 * 1024**3
HF_MIRROR = "https://hf-mirror.com"

UCI_IDS = {
    183,  # Communities and Crime
    206,  # Relative location of CT slices
    211,  # Communities and Crime Unnormalized
    248,  # Buzz in social media
    304,  # BlogFeedback
    310,  # UJIIndoorLoc
    315,  # Geographical Origin of Music
    318,  # NoisyOffice
    321,  # Electricity load time series
    328,  # Greenhouse Gas Observing Network
    332,  # Online News Popularity
    361,  # Twin gas sensor arrays
    378,  # DrivFace
    447,  # Hydraulic condition monitoring
    464,  # Superconductivity
    554,  # CNNpred
    606,  # Wikipedia Math Essentials
    882,  # Large-scale Wave Energy Farm
}

HF_POLICIES = {
    "inria-soda/tabular-benchmark": {"configs_prefix": "reg_", "splits": None},
    "HuggingFaceBio/malinois-mpra-regression": {"splits": None},
    "scikit-fingerprints/MoleculeNet_ESOL": {"splits": None},
    "scikit-fingerprints/MoleculeNet_Lipophilicity": {"splits": None},
    "scikit-fingerprints/MoleculeNet_FreeSolv": {"splits": None},
    "zpn/bace_regression": {"splits": None},
    "zpn/clearance": {"splits": None},
    "yairschiff/qm9": {"splits": None},
    "MarcAmil/Zrimec_gene_expression_transcripts": {"splits": None},
    "hu-lab/Gene_Expression_Prediction": {"splits": {"train", "validation"}},
    "BrunoBarreto/sdss_dr12_stars_regression": {"splits": None},
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def disk_guard(root: Path) -> int:
    free = shutil.disk_usage(root).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(
            f"Disk guard triggered: {free / 1024**3:.1f} GiB free < 300 GiB"
        )
    return free


def curl_download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    command = [
        "curl",
        "--fail",
        "--location",
        "--retry",
        "10",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        url,
    ]
    subprocess.run(command, check=True)
    os.replace(partial, destination)


def select_uci(catalog: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for item in catalog:
        dataset_id = int(item["source_id"])
        if dataset_id not in UCI_IDS:
            continue
        raw = json.loads(item["raw_metadata_json"])
        slug = raw["slug"]
        url = f"https://archive.ics.uci.edu/static/public/{dataset_id}/{slug}.zip"
        jobs.append(
            {
                "source": "UCI",
                "source_id": str(dataset_id),
                "dataset": raw["Name"],
                "config": "",
                "split": "",
                "revision": raw.get("DOI") or "",
                "license": "see_source_metadata",
                "url": url,
                "destination": str(root / "raw" / "uci" / f"{dataset_id}_{slug}.zip"),
            }
        )
    missing = UCI_IDS.difference(int(job["source_id"]) for job in jobs)
    if missing:
        raise RuntimeError(f"UCI IDs missing from catalog: {sorted(missing)}")
    return jobs


def select_hf(records: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for record in records:
        dataset = record.get("dataset")
        if dataset not in HF_POLICIES or record.get("fatal_error"):
            continue
        policy = HF_POLICIES[dataset]
        config = record["config"]
        split = record["split"]
        prefix = policy.get("configs_prefix")
        if prefix and not config.startswith(prefix):
            continue
        allowed_splits = policy.get("splits")
        if allowed_splits is not None and split not in allowed_splits:
            continue
        parquet_sha = record.get("parquet_sha")
        if not parquet_sha:
            raise RuntimeError(f"No parquet revision SHA for {dataset}")
        for index, parquet_file in enumerate(record.get("parquet_files") or []):
            original_url = parquet_file["url"]
            path = original_url.split("/resolve/", 1)[1].split("/", 1)[1]
            pinned_url = (
                f"{HF_MIRROR}/datasets/{dataset}/resolve/{parquet_sha}/{path}"
            )
            filename = Path(path).name
            destination = (
                root
                / "raw"
                / "huggingface"
                / dataset.replace("/", "__")
                / config
                / split
                / filename
            )
            jobs.append(
                {
                    "source": "HuggingFace",
                    "source_id": dataset,
                    "dataset": dataset,
                    "config": config,
                    "split": split,
                    "revision": parquet_sha,
                    "main_revision": record.get("main_sha") or "",
                    "license": record.get("license") or "unknown",
                    "url": pinned_url,
                    "destination": str(destination),
                    "declared_bytes": int(parquet_file.get("size") or 0),
                    "file_index": index,
                }
            )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--uci-catalog", type=Path, required=True)
    parser.add_argument("--hf-records", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    free_before = disk_guard(root)
    manifests = root / "manifests"
    logs = root / "logs"
    manifests.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)

    uci_catalog = json.loads(args.uci_catalog.read_text())
    hf_records = json.loads(args.hf_records.read_text())
    jobs = select_uci(uci_catalog, root) + select_hf(hf_records, root)
    manifest_path = manifests / "download_manifest.json"
    manifest: dict[str, Any] = {
        "created_unix": time.time(),
        "free_bytes_before": free_before,
        "free_space_guard_bytes": MIN_FREE_BYTES,
        "jobs_total": len(jobs),
        "jobs": {},
    }
    lock = threading.Lock()
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        manifest["jobs"].update(previous.get("jobs") or {})

    def run(job: dict[str, Any]) -> dict[str, Any]:
        key = hashlib.sha256(
            (job["url"] + "\n" + job["destination"]).encode()
        ).hexdigest()[:20]
        destination = Path(job["destination"])
        existing = manifest["jobs"].get(key)
        if (
            existing
            and existing.get("status") == "complete"
            and destination.exists()
            and destination.stat().st_size == existing.get("bytes")
            and sha256_file(destination) == existing.get("sha256")
        ):
            return existing
        disk_guard(root)
        started = time.time()
        result = {**job, "key": key, "status": "running", "started_unix": started}
        with lock:
            manifest["jobs"][key] = result
            atomic_json(manifest_path, manifest)
        try:
            curl_download(job["url"], destination)
            result.update(
                {
                    "status": "complete",
                    "finished_unix": time.time(),
                    "elapsed_seconds": time.time() - started,
                    "bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                    "free_bytes_after": disk_guard(root),
                }
            )
        except Exception as error:
            result.update(
                {
                    "status": "failed",
                    "finished_unix": time.time(),
                    "error": repr(error),
                }
            )
        with lock:
            manifest["jobs"][key] = result
            atomic_json(manifest_path, manifest)
        print(
            f"{result['status']:>8} {job['source']:>11} "
            f"{job['dataset']} {job['config']} {job['split']} "
            f"{result.get('bytes', 0) / 1024**2:.1f} MiB",
            flush=True,
        )
        return result

    print(f"Prepared {len(jobs)} downloads with {args.workers} workers", flush=True)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())

    complete = [result for result in results if result["status"] == "complete"]
    failed = [result for result in results if result["status"] == "failed"]
    manifest.update(
        {
            "finished_unix": time.time(),
            "complete": len(complete),
            "failed": len(failed),
            "downloaded_bytes": sum(int(result.get("bytes") or 0) for result in complete),
            "free_bytes_after": disk_guard(root),
        }
    )
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "jobs": len(jobs),
                "complete": len(complete),
                "failed": len(failed),
                "downloaded_gib": manifest["downloaded_bytes"] / 1024**3,
                "free_gib": manifest["free_bytes_after"] / 1024**3,
            },
            indent=2,
        ),
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
