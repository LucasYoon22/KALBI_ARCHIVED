#!/usr/bin/env python3
"""Runtime test-y permutation sentinel for corr095 KALBI inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluate_corr095_kalbi_bundle import (
    build_components,
    decode_queries,
    run_selector,
)
from real_pgtN_common import (
    atomic_json,
    fit_preprocessing,
    load_split_manifest,
    load_x,
    select_rows,
    sha256,
)


def freeze_output(
    task: dict[str, Any],
    model: torch.nn.Module,
    selector: torch.nn.Module,
    bundle: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    x_store = load_x(Path(task["X_path"]))
    y_store = np.load(task["y_path"], mmap_mode="r")
    with np.load(task["outer_split_path"], allow_pickle=False) as split:
        train = np.asarray(split["train_indices"], dtype=np.int64)
    x_train_raw = select_rows(x_store, train)
    y_train_raw = np.asarray(y_store[train], dtype=np.float64)
    x_train, y_train, scaler = fit_preprocessing(
        x_train_raw, y_train_raw
    )
    x = torch.from_numpy(x_train)[None].to(device)
    y = torch.from_numpy(y_train)[None].to(device)
    with torch.inference_mode():
        encoded = model.encode_dataset(model.precompute(x, y))
        selected = run_selector(bundle, selector, x, y, encoded)
        raw_lambda = selected["raw_lambda"].reshape(1, 1)
        decoded = decode_queries(
            model, encoded, selected, raw_lambda.float()
        )
    arrays = [
        np.asarray(selected["raw_lambda"].float().cpu()),
        np.asarray(selected["penalty_rho"].float().cpu()),
        np.asarray(selected["penalty_drho"].float().cpu()),
        np.asarray(decoded["beta_q"].float().cpu()),
        np.asarray(scaler["retained_original_indices"], dtype=np.int64),
    ]
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return {
        "fingerprint": digest.hexdigest(),
        "raw_lambda": float(selected["raw_lambda"].reshape(-1)[0]),
        "scaler_sha256": scaler["scaler_sha256"],
        "retained_count": int(x_train.shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-indices", default="1,16,40")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    manifest = load_split_manifest(args.split_manifest)
    selected_indices = {int(value) for value in args.task_indices.split(",")}
    tasks = [
        task
        for task in manifest["tasks"]
        if int(task["registry_index"]) in selected_indices
    ]
    bundles = {
        value["bundle_id"]: value
        for value in json.loads(
            args.bundle_manifest.read_text(encoding="utf-8")
        )["bundles"]
    }
    bundle = bundles[args.bundle_id]
    device = torch.device(args.device)
    _, _, model, selector = build_components(bundle, args.config, device)
    cache = args.evaluation_root / "cache" / "leakage_sentinel"
    cache.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for task in tasks:
        original = freeze_output(task, model, selector, bundle, device)
        y = np.asarray(np.load(task["y_path"]), dtype=np.float64).copy()
        with np.load(task["outer_split_path"], allow_pickle=False) as split:
            train = np.asarray(split["train_indices"], dtype=np.int64)
            test = np.asarray(split["test_indices"], dtype=np.int64)
        before_train = y[train].copy()
        permuted = y.copy()
        rng = np.random.default_rng(20260728 + int(task["registry_index"]))
        permuted[test] = permuted[test][rng.permutation(test.size)]
        temporary = cache / f"{task['task_id']}_test_y_permuted.npy"
        np.save(temporary, permuted)
        altered_task = dict(task)
        altered_task["y_path"] = str(temporary)
        altered_task["y_sha256"] = sha256(temporary)
        if not np.array_equal(before_train, permuted[train]):
            raise RuntimeError("sentinel changed training y")
        changed_test = not np.array_equal(y[test], permuted[test])
        altered = freeze_output(
            altered_task, model, selector, bundle, device
        )
        passed = (
            changed_test
            and original["fingerprint"] == altered["fingerprint"]
            and original["scaler_sha256"] == altered["scaler_sha256"]
        )
        results.append(
            {
                "registry_index": int(task["registry_index"]),
                "task_id": task["task_id"],
                "test_y_changed": changed_test,
                "output_identical": (
                    original["fingerprint"] == altered["fingerprint"]
                ),
                "scaler_identical": (
                    original["scaler_sha256"] == altered["scaler_sha256"]
                ),
                "status": "PASS" if passed else "FAIL",
            }
        )
        temporary.unlink()

    baseline_source = (
        args.evaluation_root
        / "scripts"
        / "evaluate_real_pgtN_cv_baselines_gramcd.py"
    ).read_text(encoding="utf-8")
    freeze_position = baseline_source.index("method_freeze_time = time.time()")
    test_position = baseline_source.index(
        "y_test_raw = np.asarray(y_store[test_index]"
    )
    static_baseline_pass = freeze_position < test_position
    status = (
        "PASS"
        if static_baseline_pass
        and all(value["status"] == "PASS" for value in results)
        else "FAIL"
    )
    report = {
        "status": status,
        "bundle_id": args.bundle_id,
        "runtime_test_y_permutation_sentinels": results,
        "baseline_static_test_y_after_method_freeze": static_baseline_pass,
    }
    atomic_json(
        args.evaluation_root / "audits" / "leakage_audit.json", report
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
