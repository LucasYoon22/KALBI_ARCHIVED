#!/usr/bin/env python3
"""Download pinned high-dimensional spectroscopy regression sources."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


MIRROR = "https://hf-mirror.com"
MIN_FREE_BYTES = 300 * 1024**3
RAMAN_BENCH = "HTW-KI-Werkstatt/RamanBench"
STANDALONE_REPOS = [
    "chlange/SubstrateMixRaman",
    "chlange/RamanSpectraBioprocessAnalystesAnton785",
    "chlange/RamanSpectraBioprocessAnalystesKaiser",
    "chlange/RamanSpectraBioprocessAnalystesMetrohm",
    "chlange/RamanSpectraBioprocessAnalystesMettlerToledo",
    "chlange/RamanSpectraBioprocessAnalystesTec5",
    "chlange/RamanSpectraBioprocessAnalystesTimegate",
    "chlange/RamanSpectraBioprocessAnalystesTornado",
    "Avatarr05/GreenHyperSpectra",
]


def get_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "kalbi-real-data-audit/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def repo_metadata(repo: str) -> dict[str, Any]:
    return get_json(f"{MIRROR}/api/datasets/{repo}")


def resolve_url(repo: str, revision: str, path: str) -> str:
    encoded_path = "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))
    return f"{MIRROR}/datasets/{repo}/resolve/{revision}/{encoded_path}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download(repo: str, revision: str, remote_path: str, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < MIN_FREE_BYTES:
        raise RuntimeError("Free storage is below the 300 GiB audit guard")
    partial = destination.with_suffix(destination.suffix + ".partial")
    url = resolve_url(repo, revision, remote_path)
    if not destination.exists():
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "6",
                "--retry-all-errors",
                "--continue-at",
                "-",
                "--silent",
                "--show-error",
                "--output",
                str(partial),
                url,
            ],
            check=True,
        )
        os.replace(partial, destination)
    return {
        "repo": repo,
        "revision": revision,
        "remote_path": remote_path,
        "local_path": str(destination),
        "url": url,
        "size_bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def download_raman_bench(raw_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    info = repo_metadata(RAMAN_BENCH)
    revision = info["sha"]
    sibling_paths = {item["rfilename"] for item in info["siblings"]}
    repo_root = raw_root / RAMAN_BENCH.replace("/", "__")
    files = [
        download(RAMAN_BENCH, revision, "README.md", repo_root / "README.md")
    ]
    regression_tasks = []
    metadata_paths = sorted(
        path for path in sibling_paths if path.endswith("/metadata.json")
    )
    for index, metadata_path in enumerate(metadata_paths, start=1):
        metadata_record = download(
            RAMAN_BENCH,
            revision,
            metadata_path,
            repo_root / metadata_path,
        )
        files.append(metadata_record)
        metadata = json.loads(Path(metadata_record["local_path"]).read_text())
        if int(metadata.get("task_type", -1)) != 2:
            continue
        task_name = metadata_path.split("/", 1)[0]
        parquet_candidates = [
            f"{task_name}/train-00000-of-00001.parquet",
            f"{task_name}/data/train-00000-of-00001.parquet",
        ]
        parquet_path = next(
            (path for path in parquet_candidates if path in sibling_paths),
            None,
        )
        if parquet_path is None:
            raise RuntimeError(
                f"Missing RamanBench regression parquet for {task_name}: "
                f"{parquet_candidates}"
            )
        parquet_record = download(
            RAMAN_BENCH,
            revision,
            parquet_path,
            repo_root / parquet_path,
        )
        files.append(parquet_record)
        regression_tasks.append(
            {
                "task_name": task_name,
                "target_names": metadata.get("target_names"),
                "task_type": metadata.get("task_type"),
                "metadata_file": metadata_record,
                "parquet_file": parquet_record,
            }
        )
        print(
            f"RamanBench regression {len(regression_tasks)}: {task_name}",
            flush=True,
        )
    return files, regression_tasks


def configured_paths(info: dict[str, Any], repo: str) -> list[str]:
    if repo == "Avatarr05/GreenHyperSpectra":
        return ["README.md", "labeled_splits/train.csv", "labeled_splits/test.csv"]
    paths = ["README.md"]
    for config in (info.get("cardData") or {}).get("configs") or []:
        for item in config.get("data_files") or []:
            path = item["path"] if isinstance(item, dict) else item
            paths.append(path)
    return sorted(set(paths))


def download_standalone(raw_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    files = []
    sources = []
    for repo in STANDALONE_REPOS:
        info = repo_metadata(repo)
        revision = info["sha"]
        repo_root = raw_root / repo.replace("/", "__")
        records = []
        for remote_path in configured_paths(info, repo):
            records.append(
                download(repo, revision, remote_path, repo_root / remote_path)
            )
        files.extend(records)
        sources.append(
            {
                "repo": repo,
                "revision": revision,
                "license": (info.get("cardData") or {}).get("license"),
                "tags": info.get("tags"),
                "files": records,
            }
        )
        print(f"Standalone source complete: {repo}", flush=True)
    return files, sources


def main() -> None:
    root = Path(
        "/users/sync_data/real_regression_uci_hf_highdim_signal_screen_20260728"
    )
    raw_root = root / "raw" / "huggingface_spectroscopy_addendum"
    manifest_root = root / "manifests"
    raman_files, raman_tasks = download_raman_bench(raw_root)
    standalone_files, standalone_sources = download_standalone(raw_root)
    all_files = raman_files + standalone_files
    manifest = {
        "contract": "pinned_hf_spectroscopy_regression_addendum_v1",
        "ramanbench_regression_task_count": len(raman_tasks),
        "standalone_source_count": len(standalone_sources),
        "file_count": len(all_files),
        "total_size_bytes": sum(item["size_bytes"] for item in all_files),
        "free_bytes_after": shutil.disk_usage(root).free,
        "files": all_files,
        "ramanbench_regression_tasks": raman_tasks,
        "standalone_sources": standalone_sources,
    }
    save_json(manifest_root / "spectroscopy_addendum_download_manifest.json", manifest)
    print(
        json.dumps(
            {
                "ramanbench_regression_tasks": len(raman_tasks),
                "standalone_sources": len(standalone_sources),
                "files": len(all_files),
                "total_size_gib": manifest["total_size_bytes"] / 1024**3,
                "free_gib": manifest["free_bytes_after"] / 1024**3,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
