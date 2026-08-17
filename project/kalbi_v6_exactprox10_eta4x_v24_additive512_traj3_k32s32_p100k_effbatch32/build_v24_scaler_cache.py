#!/usr/bin/env python3
"""Build the immutable robust V24 source scaler from rolling tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .model import build_model
from .selector_v24_additive512_trajectory import (
    DirectPenaltyMLPQuerySelectorV24Additive512,
    FixedCoordinateScalerV24,
    V24_SCALER_TRANSFORM_CONTRACT,
    V24_SCALER_VERSION,
    load_v23_state_into_v24,
    make_identity_v24_scaler_payload,
)
from train_fpk import _load_yaml


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_catalog(control_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    pointer = json.loads(
        (control_root / "CURRENT.json").read_text(encoding="utf-8")
    )
    path = control_root / "catalogs" / pointer["catalog_file"]
    if file_sha256(path) != pointer["catalog_sha256"]:
        raise RuntimeError("rolling CURRENT catalog hash mismatch")
    return pointer, json.loads(path.read_text(encoding="utf-8"))


def _entry_stratum(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    metadata = entry.get("metadata", {})
    manifest = metadata.get("generator_manifest_entry", {})
    task = (
        manifest.get("task_metadata", [{}])[0]
        if manifest.get("task_metadata")
        else {}
    )
    n = int(entry["n"])
    p = int(entry["p"])
    return (
        min(4, int(np.log10(max(n, 1)) * 2)),
        min(6, int(np.log10(max(p, 1)) * 2)),
        min(5, int(np.log2(max(p / max(n, 1), 1e-6))) + 3),
        str(task.get("outcome_source", manifest.get("outcome_source", "unknown"))),
        int(task.get("linear_snr_regime_id", 255)),
        int(task.get("linear_support_reference_id", 255)),
    )


def select_entries(
    catalog: Mapping[str, Any],
    *,
    task_count: int,
    seed: int,
    max_entry_bytes: int,
) -> list[dict[str, Any]]:
    entries = [
        dict(entry)
        for entry in catalog["files"]
        if entry.get("prior") == "tabicl_v2"
        and int(entry["bytes"]) <= int(max_entry_bytes)
    ]
    if len(entries) < task_count:
        raise RuntimeError("too few rolling tasks satisfy scaler memory bound")
    rng = np.random.default_rng(seed)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for entry in entries:
        groups.setdefault(_entry_stratum(entry), []).append(entry)
    for group in groups.values():
        group.sort(key=lambda item: int(item["shard_id"]))
        rng.shuffle(group)
    keys = sorted(groups, key=repr)
    rng.shuffle(keys)
    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < task_count:
        key = keys[cursor % len(keys)]
        if groups[key]:
            selected.append(groups[key].pop())
        cursor += 1
        if cursor > task_count * max(10, len(keys)) and len(selected) < task_count:
            raise RuntimeError("stratified task selection stalled")
    return selected


def load_context(
    data_root: Path,
    entry: Mapping[str, Any],
    *,
    split_seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    path = data_root / str(entry["relative_path"])
    with np.load(path, allow_pickle=False) as archive:
        x = torch.from_numpy(np.asarray(archive["X"][0], dtype=np.float32))
        y = torch.from_numpy(np.asarray(archive["y"][0], dtype=np.float32))
    if tuple(x.shape) != (int(entry["n"]), int(entry["p"])):
        raise RuntimeError("rolling scaler task shape drift")
    generator = torch.Generator().manual_seed(int(split_seed))
    index = torch.randperm(x.shape[0], generator=generator)[: x.shape[0] // 2]
    x = x.index_select(0, index).to(device)[None]
    y = y.index_select(0, index).to(device)[None]
    x_mean = x.mean(1, keepdim=True)
    x_std = x.std(1, keepdim=True, correction=0)
    active = x_std > 1e-6
    x = torch.where(active, (x - x_mean) / x_std.clamp_min(1e-8), 0.0)
    y = (y - y.mean(1, keepdim=True)) / y.std(
        1, keepdim=True, correction=0
    ).clamp_min(1e-8)
    if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
        raise RuntimeError("V24 scaler standardization produced nonfinite data")
    return x, y


def robust_fit(values: torch.Tensor) -> dict[str, Any]:
    data = values.double()
    if not bool(torch.isfinite(data).all()):
        raise RuntimeError("V24 scaler source contains nonfinite values")
    center = data.median(dim=0).values
    mad = (data - center).abs().median(dim=0).values
    scale = 1.4826 * mad
    q25 = torch.quantile(data, 0.25, dim=0)
    q75 = torch.quantile(data, 0.75, dim=0)
    iqr_scale = (q75 - q25) / 1.349
    std_scale = data.std(dim=0, unbiased=False)
    mad_zero = scale < 1e-6
    scale = torch.where(mad_zero, iqr_scale, scale)
    iqr_zero = scale < 1e-6
    scale = torch.where(iqr_zero, std_scale, scale)
    constant = scale < 1e-6
    # Explicit constant-coordinate policy: center the coordinate and use unit
    # scale. It remains exactly zero and cannot dominate another source.
    scale = torch.where(constant, torch.ones_like(scale), scale)
    scale = scale.clamp_min(1e-4)
    standardized = (data - center) / scale
    clipping = (standardized.abs() > 8.0).double().mean(dim=0)
    return {
        "center": center.float(),
        "scale": scale.float(),
        "mad_fallback_count": int(mad_zero.sum()),
        "iqr_fallback_count": int(iqr_zero.sum()),
        "constant_coordinate_count": int(constant.sum()),
        "clipping_fraction": clipping.float(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--task-count", type=int, default=256)
    parser.add_argument("--selection-seed", type=int, default=202607270924)
    parser.add_argument("--split-seed", type=int, default=202607270925)
    parser.add_argument("--max-entry-bytes", type=int, default=100_000_000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.task_count < 256:
        raise RuntimeError("production V24 scaler requires at least 256 tasks")
    if file_sha256(args.source_checkpoint) != args.source_checkpoint_sha256:
        raise RuntimeError("V24 source checkpoint SHA256 mismatch")
    pointer, catalog = load_catalog(args.control_root)
    entries = select_entries(
        catalog,
        task_count=args.task_count,
        seed=args.selection_seed,
        max_entry_bytes=args.max_entry_bytes,
    )
    checkpoint = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=False
    )
    device = torch.device(args.device)
    beta = build_model(_load_yaml(str(args.config)), device)
    beta.load_state_dict(checkpoint["model_state_dict"], strict=True)
    beta.eval()
    for parameter in beta.parameters():
        parameter.requires_grad_(False)
    torch.manual_seed(202607270926)
    selector = DirectPenaltyMLPQuerySelectorV24Additive512(
        hidden=512,
        derivative_floor=1e-6,
        squareplus_scale=0.02,
        direct_head_init_std=1e-3,
    )
    import_report = load_v23_state_into_v24(
        selector, checkpoint["selector_state_dict"]
    )
    selector.load_v24_scaler_cache(make_identity_v24_scaler_payload())
    with torch.no_grad():
        selector.global_gate.zero_()
        selector.trajectory_gate.zero_()
        selector.proposal_gate.zero_()
        selector.initial_norm_gate.zero_()
    selector.to(device).eval()

    collected: dict[str, list[torch.Tensor]] = {
        "global": [],
        "trajectory": [],
        "proposal": [],
    }
    manifest: list[dict[str, Any]] = []
    with torch.no_grad():
        for task_index, entry in enumerate(entries):
            x, y = load_context(
                args.data_root,
                entry,
                split_seed=args.split_seed + task_index,
                device=device,
            )
            encoded = beta.encode_dataset(beta.precompute(x, y))
            result = selector(
                x,
                y,
                encoded["lambda_max"].float(),
                solver_context=encoded,
                return_representation=True,
            )
            raw = {
                "global": result["v24_global_summary"],
                "trajectory": result["v24_solver_trajectory"],
                "proposal": result["v24_proposal_summary"],
            }
            for name, value in raw.items():
                transformed = FixedCoordinateScalerV24.transform(name, value)
                collected[name].append(transformed.detach().cpu())
            manifest.append(
                {
                    "index": task_index,
                    "shard_id": int(entry["shard_id"]),
                    "seed": int(entry["seed"]),
                    "sha256": str(entry["sha256"]),
                    "n": int(entry["n"]),
                    "p": int(entry["p"]),
                    "bytes": int(entry["bytes"]),
                    "split_seed": args.split_seed + task_index,
                    "stratum": list(_entry_stratum(entry)),
                }
            )
            del x, y, encoded, result, raw
            torch.cuda.empty_cache()
            print(
                json.dumps(
                    {
                        "kind": "v24_scaler_progress",
                        "task": task_index + 1,
                        "total": len(entries),
                        "n": int(entry["n"]),
                        "p": int(entry["p"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    fitted = {
        name: robust_fit(torch.cat(values, dim=0))
        for name, values in collected.items()
    }
    task_manifest = {
        "version": "v24_scaler_task_manifest_v1",
        "selection_seed": args.selection_seed,
        "split_seed_base": args.split_seed,
        "task_count": len(manifest),
        "catalog_pointer": pointer,
        "tasks": manifest,
    }
    manifest_sha = canonical_sha256(task_manifest)
    semantic = {
        "version": V24_SCALER_VERSION,
        "transform_contract": V24_SCALER_TRANSFORM_CONTRACT,
        "cache_size": len(entries),
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": args.source_checkpoint_sha256,
        "task_manifest_sha256": manifest_sha,
        "global_center": fitted["global"]["center"].tolist(),
        "global_scale": fitted["global"]["scale"].tolist(),
        "trajectory_center": fitted["trajectory"]["center"].tolist(),
        "trajectory_scale": fitted["trajectory"]["scale"].tolist(),
        "proposal_center": fitted["proposal"]["center"].tolist(),
        "proposal_scale": fitted["proposal"]["scale"].tolist(),
    }
    cache_sha = canonical_sha256(semantic)
    payload = {
        **semantic,
        "cache_sha256": cache_sha,
        "global_center": fitted["global"]["center"],
        "global_scale": fitted["global"]["scale"],
        "trajectory_center": fitted["trajectory"]["center"],
        "trajectory_scale": fitted["trajectory"]["scale"],
        "proposal_center": fitted["proposal"]["center"],
        "proposal_scale": fitted["proposal"]["scale"],
        "metadata": {
            "estimator": "coordinatewise_median_1p4826MAD_v1",
            "clip": [-8.0, 8.0],
            "constant_coordinate_policy": "center_and_unit_scale",
            "selection": "deterministic_round_robin_stratified_v1",
            "catalog_pointer": pointer,
            "source_state_import": import_report,
            "diagnostics": {
                name: {
                    key: (
                        value.tolist()
                        if isinstance(value, torch.Tensor)
                        else value
                    )
                    for key, value in report.items()
                    if key not in ("center", "scale")
                }
                for name, report in fitted.items()
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.manifest_output.write_text(
        json.dumps(task_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    reloaded = torch.load(args.output, map_location="cpu", weights_only=False)
    check = DirectPenaltyMLPQuerySelectorV24Additive512(
        hidden=512,
        derivative_floor=1e-6,
        squareplus_scale=0.02,
        direct_head_init_std=1e-3,
    )
    check.load_v24_scaler_cache(reloaded)
    state = check.state_dict()
    clone = DirectPenaltyMLPQuerySelectorV24Additive512(
        hidden=512,
        derivative_floor=1e-6,
        squareplus_scale=0.02,
        direct_head_init_std=1e-3,
    )
    clone.load_state_dict(state, strict=True)
    if not bool(clone.v24_scaler.cache_loaded.item()):
        raise RuntimeError("V24 scaler failed strict state_dict reload")
    print(
        json.dumps(
            {
                "kind": "v24_scaler_complete",
                "output": str(args.output),
                "file_sha256": file_sha256(args.output),
                "cache_sha256": cache_sha,
                "manifest": str(args.manifest_output),
                "manifest_sha256": manifest_sha,
                "task_count": len(entries),
                "diagnostics": payload["metadata"]["diagnostics"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
