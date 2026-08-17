#!/usr/bin/env python3
"""Verify deterministic replay and contracts for isolated V13 tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def array_digest(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_a = Path(args.run_a)
    run_b = Path(args.run_b)
    files_a = sorted(run_a.glob("*.npz"))
    files_b = sorted(run_b.glob("*.npz"))

    comparison = []
    for path_b in files_b:
        path_a = run_a / path_b.name
        with np.load(path_a, allow_pickle=False) as a, np.load(
            path_b, allow_pickle=False
        ) as b:
            same_keys = a.files == b.files
            mismatches = [
                key
                for key in sorted(set(a.files).intersection(b.files))
                if array_digest(a[key]) != array_digest(b[key])
            ]
        comparison.append(
            {
                "file": path_b.name,
                "same_key_order": same_keys,
                "mismatched_arrays": mismatches,
                "exact_match": same_keys and not mismatches,
            }
        )

    branches: Counter[str] = Counter()
    strict_support_failures: list[str] = []
    all_task_digests: list[str] = []
    core_std_lt_0p9 = 0
    core_feature_count = 0
    bnn_std_lt_0p9 = 0
    bnn_feature_count = 0
    for path in files_a:
        with np.load(path, allow_pickle=False) as archive:
            outcome = int(archive["outcome_source_id"].reshape(-1)[0])
            reference = int(archive["linear_support_reference_id"].reshape(-1)[0])
            n = int(archive["X"].shape[1])
            k = int(archive["linear_support_size"].reshape(-1)[0])
            x = archive["X"][0].astype(np.float64)
            source = archive["feature_source"].reshape(-1)
            payload = hashlib.sha256()
            for key in sorted(archive.files):
                payload.update(key.encode())
                payload.update(array_digest(archive[key]).encode())
            all_task_digests.append(payload.hexdigest())
        if outcome == 0:
            branches["tabicl"] += 1
        elif reference == 0:
            branches["linear_p"] += 1
        else:
            branches["linear_n"] += 1
        if outcome == 1 and not (3 * k < n):
            strict_support_failures.append(path.name)
        std = np.std(x, axis=0)
        core = source == 0
        bnn = source == 1
        core_std_lt_0p9 += int(np.count_nonzero(std[core] < 0.9))
        core_feature_count += int(core.sum())
        bnn_std_lt_0p9 += int(np.count_nonzero(std[bnn] < 0.9))
        bnn_feature_count += int(bnn.sum())

    result = {
        "run_a_task_count": len(files_a),
        "run_b_task_count": len(files_b),
        "run_a_branch_counts": dict(branches),
        "assignment_note": (
            "The live deterministic 50/25/25 rule yields 30/15/15 over 60 "
            "consecutive batch seeds; forcing 20/20/20 would not replay the live generator."
        ),
        "strict_support_failures": strict_support_failures,
        "duplicate_task_payloads_run_a": len(all_task_digests)
        - len(set(all_task_digests)),
        "exact_replay_comparison": comparison,
        "all_compared_tasks_exact": bool(comparison)
        and all(item["exact_match"] for item in comparison),
        "feature_scaling": {
            "core_std_lt_0p9_count": core_std_lt_0p9,
            "core_feature_count": core_feature_count,
            "core_std_lt_0p9_fraction": core_std_lt_0p9
            / max(core_feature_count, 1),
            "bnn_std_lt_0p9_count": bnn_std_lt_0p9,
            "bnn_feature_count": bnn_feature_count,
            "bnn_std_lt_0p9_fraction": bnn_std_lt_0p9
            / max(bnn_feature_count, 1),
        },
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
