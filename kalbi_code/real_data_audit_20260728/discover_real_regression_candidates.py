#!/usr/bin/env python3
"""Discover real regression datasets from official UCI and Hugging Face APIs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


UCI_ENDPOINT = "https://archive.ics.uci.edu/api/trpc/donated_datasets.search"
HF_API = "https://huggingface.co/api/datasets"
HF_DATASET_SERVER = "https://datasets-server.huggingface.co"
MIN_FREE_BYTES = 300 * 1024**3

HF_SEARCH_TERMS = (
    "regression",
    "tabular regression",
    "high dimensional regression",
    "gene expression regression",
    "transcriptomics regression",
    "drug response regression",
    "GDSC",
    "CCLE",
    "DepMap",
    "E2006",
    "tfidf regression",
    "spectroscopy regression",
    "NIR regression",
    "chemometrics regression",
    "molecular property regression",
    "solubility regression",
    "lipophilicity regression",
    "binding affinity regression",
    "materials property regression",
    "toxicity regression",
    "continuous target",
)

CONTINUOUS_HINTS = (
    "regression",
    "continuous",
    "affinity",
    "solubility",
    "lipophilicity",
    "expression",
    "response",
    "toxicity",
    "property",
    "spectroscopy",
    "chemometrics",
    "tfidf",
)

EXCLUSION_HINTS = (
    "image classification",
    "text classification",
    "token classification",
    "object detection",
    "question answering",
    "reinforcement",
    "synthetic",
)


def http_json(url: str, retries: int = 5, timeout: int = 60) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "KALBI-real-regression-audit/1.0",
            "Accept": "application/json",
        },
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"GET failed after {retries} attempts: {url}") from last_error


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def uci_search(skip: int, take: int = 100) -> dict[str, Any]:
    payload = {
        "0": {
            "json": {
                "Area": [],
                "Keywords": [],
                "Variables": [],
                "Types": [],
                "Task": "Regression",
                "NumInstances": [],
                "NumFeatures": [],
                "FeatureTypes": None,
                "Python": None,
                "skip": skip,
                "take": take,
                "sort": "desc",
                "orderBy": "NumFeatures",
                "search": "",
            },
            "meta": {
                "values": {
                    "FeatureTypes": ["undefined"],
                    "Python": ["undefined"],
                }
            },
        }
    }
    query = urllib.parse.urlencode(
        {"batch": "1", "input": compact_json(payload)}, quote_via=urllib.parse.quote
    )
    result = http_json(f"{UCI_ENDPOINT}?{query}")
    return result[0]["result"]["data"]["json"]


def discover_uci() -> list[dict[str, Any]]:
    first = uci_search(0)
    count = int(first["count"])
    datasets = list(first["datasets"])
    for skip in range(100, count, 100):
        datasets.extend(uci_search(skip)["datasets"])

    seen: set[int] = set()
    rows: list[dict[str, Any]] = []
    for item in datasets:
        dataset_id = int(item["ID"])
        if dataset_id in seen:
            continue
        seen.add(dataset_id)
        n = item.get("NumInstances")
        p = item.get("NumFeatures")
        tasks = str(item.get("Task") or "")
        variables = item.get("variables") or []
        target_variables = [
            variable.get("name")
            for variable in variables
            if str(variable.get("role") or "").lower() == "target"
        ]
        tier = ""
        if isinstance(n, int) and isinstance(p, int):
            if n >= 200 and p > n:
                tier = "A_metadata_highdim"
            elif n >= 200 and p >= 1000:
                tier = "B_metadata_largeP"
            elif n >= 200 and p >= 50:
                tier = "C_metadata_moderateP"
        rows.append(
            {
                "source": "UCI",
                "source_id": str(dataset_id),
                "name": item.get("Name"),
                "url": f"https://archive.ics.uci.edu/dataset/{dataset_id}/"
                f"{item.get('slug', '')}",
                "revision": item.get("DOI") or "",
                "license": "",
                "task_metadata": tasks,
                "n_metadata": n,
                "p_metadata": p,
                "p_over_n_metadata": (float(p) / float(n)) if n and p else "",
                "target_metadata": "|".join(filter(None, target_variables)),
                "feature_types": item.get("FeatureTypes") or "",
                "tier_metadata": tier,
                "search_terms": "official_regression_catalog",
                "downloads": "",
                "likes": "",
                "gated": "",
                "private": "",
                "card_excerpt": str(item.get("Abstract") or "")[:500],
                "raw_metadata_json": compact_json(item),
            }
        )
    if len(rows) != count:
        raise RuntimeError(f"UCI count mismatch: expected {count}, got {len(rows)}")
    return rows


def hf_search(term: str, limit: int = 100) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "search": term,
            "limit": str(limit),
            "full": "true",
            "sort": "downloads",
            "direction": "-1",
        }
    )
    result = http_json(f"{HF_API}?{query}")
    if not isinstance(result, list):
        raise RuntimeError(f"Unexpected Hugging Face response for {term!r}")
    return result


def safe_dataset_server(path: str, dataset_id: str) -> Any:
    query = urllib.parse.urlencode({"dataset": dataset_id})
    try:
        return http_json(f"{HF_DATASET_SERVER}/{path}?{query}", retries=2, timeout=15)
    except RuntimeError:
        return None


def card_text(item: dict[str, Any]) -> str:
    card = item.get("cardData") or {}
    fields = [
        item.get("id"),
        item.get("description"),
        card.get("pretty_name"),
        card.get("task_categories"),
        card.get("task_ids"),
        card.get("tags"),
        card.get("dataset_info"),
    ]
    return " ".join(str(field) for field in fields if field is not None).lower()


def parse_hf_size(size_info: Any) -> tuple[int | None, int | None]:
    if not isinstance(size_info, dict):
        return None, None
    size = size_info.get("size") or size_info
    rows = None
    bytes_value = None
    if isinstance(size, dict):
        dataset = size.get("dataset") or {}
        if isinstance(dataset, dict):
            rows = dataset.get("num_rows")
            bytes_value = dataset.get("num_bytes_original_files")
        if rows is None:
            rows = size.get("num_rows")
        if bytes_value is None:
            bytes_value = size.get("num_bytes_original_files")
    return rows if isinstance(rows, int) else None, (
        bytes_value if isinstance(bytes_value, int) else None
    )


def inspect_hf_viewer(dataset_id: str) -> tuple[Any, Any, Any]:
    validity = safe_dataset_server("is-valid", dataset_id)
    size_info = None
    splits_info = None
    if validity and validity.get("viewer"):
        size_info = safe_dataset_server("size", dataset_id)
        splits_info = safe_dataset_server("splits", dataset_id)
    return validity, size_info, splits_info


def discover_hf() -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    terms_by_id: dict[str, set[str]] = {}
    for term in HF_SEARCH_TERMS:
        found = hf_search(term)
        print(f"HF search {term!r}: {len(found)} results", flush=True)
        for item in found:
            dataset_id = item.get("id")
            if not dataset_id:
                continue
            terms_by_id.setdefault(dataset_id, set()).add(term)
            existing = merged.get(dataset_id)
            if existing is None or len(compact_json(item)) > len(compact_json(existing)):
                merged[dataset_id] = item

    inspect_ids: list[str] = []
    for dataset_id, item in merged.items():
        text = card_text(item)
        positive = sum(hint in text for hint in CONTINUOUS_HINTS)
        negative = sum(hint in text for hint in EXCLUSION_HINTS)
        task_categories = (item.get("cardData") or {}).get("task_categories") or []
        if ("tabular-regression" in task_categories or positive >= 1) and negative == 0:
            inspect_ids.append(dataset_id)
    print(
        f"HF unique candidates: {len(merged)}; viewer inspection: {len(inspect_ids)}",
        flush=True,
    )
    viewer_results: dict[str, tuple[Any, Any, Any]] = {}
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(inspect_hf_viewer, dataset_id): dataset_id
            for dataset_id in inspect_ids
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            dataset_id = futures[future]
            try:
                viewer_results[dataset_id] = future.result()
            except Exception as error:
                viewer_results[dataset_id] = ({"error": repr(error)}, None, None)
            if completed % 50 == 0 or completed == len(futures):
                print(
                    f"HF viewer inspection: {completed}/{len(futures)}",
                    flush=True,
                )

    rows: list[dict[str, Any]] = []
    for index, (dataset_id, item) in enumerate(
        sorted(merged.items(), key=lambda pair: int(pair[1].get("downloads") or 0), reverse=True)
    ):
        text = card_text(item)
        positive = sum(hint in text for hint in CONTINUOUS_HINTS)
        negative = sum(hint in text for hint in EXCLUSION_HINTS)
        task_categories = (item.get("cardData") or {}).get("task_categories") or []
        tabular_regression = "tabular-regression" in task_categories
        relevant = tabular_regression or positive >= 1

        # Inspect Dataset Viewer only for plausible candidates to limit API load.
        validity, size_info, splits_info = viewer_results.get(
            dataset_id, (None, None, None)
        )
        n_metadata, byte_size = parse_hf_size(size_info)
        card = item.get("cardData") or {}
        license_value = card.get("license") or ""
        if isinstance(license_value, list):
            license_value = "|".join(str(value) for value in license_value)
        rows.append(
            {
                "source": "HuggingFace",
                "source_id": dataset_id,
                "name": card.get("pretty_name") or dataset_id.rsplit("/", 1)[-1],
                "url": f"https://huggingface.co/datasets/{dataset_id}",
                "revision": item.get("sha") or "",
                "license": license_value,
                "task_metadata": "|".join(map(str, task_categories)),
                "n_metadata": n_metadata or "",
                "p_metadata": "",
                "p_over_n_metadata": "",
                "target_metadata": "",
                "feature_types": "",
                "tier_metadata": "inspect" if relevant and negative == 0 else "metadata_reject",
                "search_terms": "|".join(sorted(terms_by_id[dataset_id])),
                "downloads": item.get("downloads") or 0,
                "likes": item.get("likes") or 0,
                "gated": item.get("gated") or False,
                "private": item.get("private") or False,
                "viewer_valid": bool(validity and validity.get("viewer")),
                "parquet_valid": bool(validity and validity.get("parquet")),
                "byte_size_metadata": byte_size or "",
                "card_excerpt": text[:1000],
                "splits_metadata_json": compact_json(splits_info) if splits_info else "",
                "raw_metadata_json": compact_json(item),
                "discovery_rank": index,
            }
        )
    return rows


def shortlist(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row["source"] == "UCI":
            n = row.get("n_metadata")
            p = row.get("p_metadata")
            if isinstance(n, int) and isinstance(p, int) and n >= 200 and p >= 50:
                selected.append(row)
        elif (
            row.get("tier_metadata") == "inspect"
            and not row.get("gated")
            and not row.get("private")
        ):
            selected.append(row)
    return selected


def require_disk_budget(path: Path) -> int:
    usage = shutil.disk_usage(path.parent if path.parent.exists() else Path("/"))
    if usage.free < MIN_FREE_BYTES:
        raise RuntimeError(
            f"Free space {usage.free / 1024**3:.1f} GiB is below 300 GiB guard"
        )
    return usage.free


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    free_before = require_disk_budget(output_root)
    discovery = output_root / "discovery"
    discovery.mkdir(parents=True, exist_ok=True)

    uci_rows = discover_uci()
    print(f"Discovered {len(uci_rows)} UCI regression catalog entries", flush=True)
    hf_rows = discover_hf()
    print(f"Discovered {len(hf_rows)} unique Hugging Face candidates", flush=True)
    all_rows = uci_rows + hf_rows
    shortlist_rows = shortlist(all_rows)

    fields = sorted({key for row in all_rows for key in row})
    write_csv(discovery / "uci_regression_all.csv", uci_rows, fields)
    write_csv(discovery / "hf_search_all.csv", hf_rows, fields)
    write_csv(discovery / "all_discovered_candidates.csv", all_rows, fields)
    write_csv(discovery / "metadata_shortlist.csv", shortlist_rows, fields)
    write_json(discovery / "uci_regression_all.json", uci_rows)
    write_json(discovery / "hf_search_all.json", hf_rows)
    write_json(discovery / "metadata_shortlist.json", shortlist_rows)

    script_path = Path(__file__).resolve()
    manifest = {
        "created_unix": time.time(),
        "script": str(script_path),
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "output_root": str(output_root),
        "free_bytes_before": free_before,
        "free_space_guard_bytes": MIN_FREE_BYTES,
        "uci_count": len(uci_rows),
        "hf_count": len(hf_rows),
        "shortlist_count": len(shortlist_rows),
        "hf_search_terms": HF_SEARCH_TERMS,
        "selection_note": (
            "Metadata shortlist only; final acceptance requires target/schema/license/"
            "leakage audit and held-out linear-signal screening without KALBI."
        ),
    }
    write_json(discovery / "discovery_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
