#!/usr/bin/env python3
"""Quantify post-generation feature scale drift in a frozen V13 sample."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(args.manifest).open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    task_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        with np.load(row["absolute_path"], allow_pickle=False) as archive:
            x = archive["X"][0].astype(np.float32)
            source = archive["feature_source"].astype(np.uint8).reshape(-1)
        std = np.std(x.astype(np.float64), axis=0)
        mean = np.mean(x.astype(np.float64), axis=0)
        for source_id, source_name in ((0, "core"), (1, "bnn")):
            mask = source == source_id
            values = std[mask]
            feature_rows.extend(
                {
                    "task_hash": row["task_hash"],
                    "audit_stratum": row["audit_stratum"],
                    "source": source_name,
                    "feature_std": float(value),
                }
                for value in values[values < 0.9]
            )
            task_rows.append(
                {
                    "task_hash": row["task_hash"],
                    "audit_stratum": row["audit_stratum"],
                    "source": source_name,
                    "n": int(row["n"]),
                    "p": int(row["p"]),
                    "feature_count": int(mask.sum()),
                    "std_min": float(np.min(values)) if values.size else None,
                    "std_median": float(np.median(values)) if values.size else None,
                    "std_max": float(np.max(values)) if values.size else None,
                    "count_std_lt_0p9": int(np.count_nonzero(values < 0.9)),
                    "count_std_lt_0p5": int(np.count_nonzero(values < 0.5)),
                    "count_std_lt_0p2": int(np.count_nonzero(values < 0.2)),
                    "max_abs_mean": float(np.max(np.abs(mean[mask]))) if values.size else None,
                }
            )
        if (index + 1) % 25 == 0:
            print(f"audited {index + 1}/{len(rows)}", flush=True)

    with (output_dir / "feature_scaling_task_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_rows[0]))
        writer.writeheader()
        writer.writerows(task_rows)
    if feature_rows:
        with (output_dir / "feature_scaling_affected_features.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0]))
            writer.writeheader()
            writer.writerows(feature_rows)

    summary: dict[str, object] = {"task_count": len(rows), "sources": {}}
    for source_name in ("core", "bnn"):
        selected = [r for r in task_rows if r["source"] == source_name]
        feature_count = sum(int(r["feature_count"]) for r in selected)
        for threshold_name in ("count_std_lt_0p9", "count_std_lt_0p5", "count_std_lt_0p2"):
            count = sum(int(r[threshold_name]) for r in selected)
            summary["sources"].setdefault(source_name, {})[threshold_name] = {
                "count": count,
                "fraction": count / max(feature_count, 1),
                "tasks_affected": sum(int(r[threshold_name]) > 0 for r in selected),
            }
        summary["sources"][source_name]["feature_count"] = feature_count
        summary["sources"][source_name]["global_min_std"] = min(
            float(r["std_min"]) for r in selected if r["std_min"] is not None
        )
    (output_dir / "feature_scaling_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
