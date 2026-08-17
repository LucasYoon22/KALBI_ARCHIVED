#!/usr/bin/env python3
"""Inspect schemas, splits, sizes, revisions, and parquet files for selected HF data."""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


HF = "https://huggingface.co"
VIEWER = "https://datasets-server.huggingface.co"

DATASETS = (
    "inria-soda/tabular-benchmark",
    "HuggingFaceBio/malinois-mpra-regression",
    "scikit-fingerprints/MoleculeNet_ESOL",
    "scikit-fingerprints/MoleculeNet_Lipophilicity",
    "scikit-fingerprints/MoleculeNet_FreeSolv",
    "zpn/bace_regression",
    "zpn/clearance",
    "yairschiff/qm9",
    "wanglab/promoter_rbs_gene_expression",
    "MarcAmil/Zrimec_gene_expression_transcripts",
    "hu-lab/Gene_Expression_Prediction",
    "BrunoBarreto/sdss_dr12_stars_regression",
    "michaelmallari/house-prices-advanced-regression-techniques",
)


def get_json(url: str, timeout: int = 60) -> Any:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "KALBI-real-regression-audit/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def viewer(path: str, **params: str) -> Any:
    query = urllib.parse.urlencode(params)
    return get_json(f"{VIEWER}/{path}?{query}")


def first_rows(dataset: str, config: str, split: str) -> dict[str, Any]:
    try:
        return viewer("first-rows", dataset=dataset, config=config, split=split)
    except Exception as error:
        return {"error": repr(error)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for dataset in DATASETS:
        print(f"Inspecting {dataset}", flush=True)
        try:
            info = get_json(f"{HF}/api/datasets/{dataset}")
            refs = get_json(f"{HF}/api/datasets/{dataset}/refs")
            validity = viewer("is-valid", dataset=dataset)
            splits_response = viewer("splits", dataset=dataset)
            size = viewer("size", dataset=dataset)
            parquet = viewer("parquet", dataset=dataset)
        except Exception as error:
            records.append({"dataset": dataset, "fatal_error": repr(error)})
            continue

        branches = refs.get("branches") or []
        converts = refs.get("converts") or []
        main_sha = next(
            (
                item.get("targetCommit")
                for item in branches
                if item.get("name") == "main"
            ),
            info.get("sha"),
        )
        parquet_sha = next(
            (
                item.get("targetCommit")
                for item in converts
                if item.get("name") == "parquet"
            ),
            None,
        )
        card = info.get("cardData") or {}
        license_value = card.get("license")
        if isinstance(license_value, list):
            license_value = "|".join(map(str, license_value))

        split_items = splits_response.get("splits") or []
        parquet_files = parquet.get("parquet_files") or []
        for split_item in split_items:
            config = split_item["config"]
            split = split_item["split"]
            if dataset == "inria-soda/tabular-benchmark" and not config.startswith("reg_"):
                continue
            sample = first_rows(dataset, config, split)
            features = sample.get("features") or []
            rows = sample.get("rows") or []
            sample_row = rows[0].get("row", {}) if rows else {}
            files = [
                item
                for item in parquet_files
                if item.get("config") == config and item.get("split") == split
            ]
            record = {
                "dataset": dataset,
                "config": config,
                "split": split,
                "main_sha": main_sha,
                "parquet_sha": parquet_sha,
                "license": license_value or "",
                "gated": info.get("gated") or False,
                "private": info.get("private") or False,
                "viewer_valid": bool(validity.get("viewer")),
                "parquet_valid": bool(validity.get("parquet")),
                "num_columns": len(features),
                "columns": [item.get("name") for item in features],
                "feature_schema": features,
                "sample_row": sample_row,
                "parquet_files": files,
                "parquet_bytes": sum(int(item.get("size") or 0) for item in files),
                "first_rows_error": sample.get("error") or "",
            }
            records.append(record)
            summaries.append(
                {
                    "dataset": dataset,
                    "config": config,
                    "split": split,
                    "main_sha": main_sha,
                    "parquet_sha": parquet_sha or "",
                    "license": license_value or "",
                    "num_columns": len(features),
                    "columns": "|".join(str(item.get("name")) for item in features),
                    "parquet_files": len(files),
                    "parquet_bytes": record["parquet_bytes"],
                    "viewer_valid": record["viewer_valid"],
                    "parquet_valid": record["parquet_valid"],
                    "error": record["first_rows_error"],
                }
            )

        dataset_record = {
            "dataset": dataset,
            "main_sha": main_sha,
            "parquet_sha": parquet_sha,
            "license": license_value,
            "info": info,
            "refs": refs,
            "validity": validity,
            "size": size,
        }
        (args.output_dir / f"{dataset.replace('/', '__')}.json").write_text(
            json.dumps(dataset_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    (args.output_dir / "selected_hf_schema_records.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = sorted({key for row in summaries for key in row})
    with (args.output_dir / "selected_hf_schema_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    manifest = {
        "created_unix": time.time(),
        "datasets_requested": DATASETS,
        "records": len(records),
        "summary_rows": len(summaries),
    }
    (args.output_dir / "inspection_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
