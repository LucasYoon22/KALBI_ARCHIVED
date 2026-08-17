#!/usr/bin/env python3
"""Inspect downloaded spectroscopy schemas without loading feature matrices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(
    "/users/sync_data/real_regression_uci_hf_highdim_signal_screen_20260728"
)
RAW = ROOT / "raw" / "huggingface_spectroscopy_addendum"
MANIFEST = json.loads(
    (ROOT / "manifests" / "spectroscopy_addendum_download_manifest.json").read_text()
)


def is_spectral_column(name: str) -> bool:
    try:
        float(name)
    except (TypeError, ValueError):
        return False
    return True


def target_stats(path: Path, target: str) -> dict[str, Any]:
    values = pd.read_parquet(path, columns=[target])[target]
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[numeric.notna()]
    return {
        "rows": len(values),
        "finite": len(finite),
        "unique": int(finite.nunique()),
        "minimum": float(finite.min()) if len(finite) else None,
        "maximum": float(finite.max()) if len(finite) else None,
        "std": float(finite.std()) if len(finite) > 1 else None,
    }


def inspect_ramanbench() -> list[dict[str, Any]]:
    records = []
    for task in MANIFEST["ramanbench_regression_tasks"]:
        path = Path(task["parquet_file"]["local_path"])
        parquet = pq.ParquetFile(path)
        columns = parquet.schema_arrow.names
        spectral = [name for name in columns if is_spectral_column(name)]
        targets = [str(name) for name in task["target_names"]]
        records.append(
            {
                "source": "RamanBench",
                "task_name": task["task_name"],
                "path": str(path),
                "rows": parquet.metadata.num_rows,
                "columns": parquet.metadata.num_columns,
                "spectral_columns": len(spectral),
                "first_spectral": spectral[0] if spectral else None,
                "last_spectral": spectral[-1] if spectral else None,
                "target_names": targets,
                "other_columns": [
                    name for name in columns if name not in spectral and name not in targets
                ],
                "target_stats": {
                    target: target_stats(path, target)
                    for target in targets
                    if target in columns
                },
            }
        )
    return records


def inspect_standalone() -> list[dict[str, Any]]:
    records = []
    for source in MANIFEST["standalone_sources"]:
        repo = source["repo"]
        split_records = []
        all_non_spectral = set()
        for file_record in source["files"]:
            path = Path(file_record["local_path"])
            if path.name == "README.md":
                continue
            if path.suffix == ".parquet":
                parquet = pq.ParquetFile(path)
                columns = parquet.schema_arrow.names
                rows = parquet.metadata.num_rows
            else:
                frame = pd.read_csv(path, nrows=3)
                columns = list(frame.columns)
                with path.open(errors="replace") as handle:
                    rows = max(sum(1 for _ in handle) - 1, 0)
            spectral = [str(name) for name in columns if is_spectral_column(str(name))]
            non_spectral = [str(name) for name in columns if str(name) not in spectral]
            all_non_spectral.update(non_spectral)
            split_records.append(
                {
                    "path": str(path),
                    "rows": rows,
                    "columns": len(columns),
                    "spectral_columns": len(spectral),
                    "first_spectral": spectral[0] if spectral else None,
                    "last_spectral": spectral[-1] if spectral else None,
                    "non_spectral_columns": non_spectral,
                }
            )
        records.append(
            {
                "source": "standalone",
                "repo": repo,
                "revision": source["revision"],
                "license": source["license"],
                "splits": split_records,
                "non_spectral_columns_union": sorted(all_non_spectral),
            }
        )
    return records


def main() -> None:
    result = {
        "ramanbench": inspect_ramanbench(),
        "standalone": inspect_standalone(),
    }
    output = ROOT / "manifests" / "spectroscopy_addendum_schema_audit.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for item in result["ramanbench"]:
        print(
            f"RB {item['task_name']}: N={item['rows']} P={item['spectral_columns']} "
            f"targets={item['target_names']} other={item['other_columns']}",
            flush=True,
        )
    for item in result["standalone"]:
        split = item["splits"][0]
        print(
            f"ST {item['repo']}: P={split['spectral_columns']} "
            f"non_spectral={item['non_spectral_columns_union']}",
            flush=True,
        )
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
