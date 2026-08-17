#!/usr/bin/env python3
"""Build the fixed V22 input-scaling cache from immutable rolling tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .selector_v22_v20geometry_scaled import (
    DirectPenaltyMLPQuerySelectorV22V20GeometryScaled,
    V22_SCALER_STD_FLOOR,
    V22_SCALER_VERSION,
    load_v20_state_dict,
)


class CoordinateWelford:
    """Masked parallel-Welford moments for a fixed trailing tensor shape."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = tuple(shape)
        self.count = torch.zeros(self.shape, dtype=torch.float64)
        self.mean = torch.zeros(self.shape, dtype=torch.float64)
        self.m2 = torch.zeros(self.shape, dtype=torch.float64)

    @torch.no_grad()
    def update(self, values: torch.Tensor, mask: torch.Tensor) -> None:
        if tuple(values.shape[-len(self.shape) :]) != self.shape:
            raise ValueError("Welford value shape drift")
        if values.shape != mask.shape:
            raise ValueError("Welford mask shape drift")
        reduce_dims = tuple(range(values.ndim - len(self.shape)))
        values64 = values.detach().double().cpu()
        mask64 = mask.detach().double().cpu()
        batch_count = mask64.sum(dim=reduce_dims)
        batch_sum = (values64 * mask64).sum(dim=reduce_dims)
        batch_mean = batch_sum / batch_count.clamp_min(1.0)
        centered = (values64 - batch_mean) * mask64
        batch_m2 = centered.square().sum(dim=reduce_dims)

        combined_count = self.count + batch_count
        delta = batch_mean - self.mean
        merge_weight = (
            batch_count / combined_count.clamp_min(1.0)
        )
        self.mean = self.mean + delta * merge_weight
        self.m2 = (
            self.m2
            + batch_m2
            + delta.square()
            * self.count
            * batch_count
            / combined_count.clamp_min(1.0)
        )
        self.count = combined_count

    def payload(self) -> dict[str, torch.Tensor]:
        variance = self.m2 / self.count.clamp_min(1.0)
        std = variance.clamp_min(0.0).sqrt()
        structural_zero = self.count <= 0
        return {
            "mean": torch.where(
                structural_zero, torch.zeros_like(self.mean), self.mean
            ).float(),
            "std": torch.where(
                structural_zero,
                torch.ones_like(std),
                std.clamp_min(V22_SCALER_STD_FLOOR),
            ).float(),
            "count": self.count.long(),
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_catalog(control_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    pointer_path = control_root / "CURRENT.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    catalog_path = control_root / "catalogs" / pointer["catalog_file"]
    if sha256(catalog_path) != pointer["catalog_sha256"]:
        raise RuntimeError("CURRENT catalog SHA256 mismatch")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    return pointer, catalog


def select_entries(
    catalog: Mapping[str, Any],
    *,
    task_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    entries = [
        dict(entry)
        for entry in catalog["files"]
        if entry.get("prior") == "tabicl_v2"
    ]
    if len(entries) < task_count:
        raise RuntimeError("catalog has too few TabICLv2 tasks")
    entries.sort(key=lambda item: int(item["shard_id"]))
    rng = np.random.default_rng(seed)
    selected_indices = np.sort(
        rng.choice(len(entries), size=task_count, replace=False)
    )
    return [entries[int(index)] for index in selected_indices]


def load_standardized_train(
    data_root: Path,
    entry: Mapping[str, Any],
    *,
    split_seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    path = data_root / str(entry["relative_path"])
    with np.load(path, allow_pickle=False) as archive:
        x_raw = np.asarray(archive["X"][0], dtype=np.float32)
        y_raw = np.asarray(archive["y"][0], dtype=np.float32)
    rows, features = x_raw.shape
    if (rows, features) != (int(entry["n"]), int(entry["p"])):
        raise RuntimeError("calibration task shape drift")
    generator = torch.Generator(device="cpu").manual_seed(int(split_seed))
    train_rows = rows // 2
    train_index = torch.randperm(rows, generator=generator)[:train_rows]
    x = torch.from_numpy(x_raw)[None].to(device)
    y = torch.from_numpy(y_raw)[None].to(device)
    x_train = x.index_select(1, train_index.to(device))
    y_train = y.index_select(1, train_index.to(device))
    x_mean = x_train.mean(1, keepdim=True)
    x_std_raw = x_train.std(1, keepdim=True, correction=0)
    x_active = x_std_raw > 1e-6
    x_train = torch.where(
        x_active,
        (x_train - x_mean) / x_std_raw.clamp_min(1e-8),
        0.0,
    )
    y_train = (
        y_train - y_train.mean(1, keepdim=True)
    ) / y_train.std(1, keepdim=True, correction=0).clamp_min(1e-8)
    if not bool(torch.isfinite(x_train).all()) or not bool(
        torch.isfinite(y_train).all()
    ):
        raise RuntimeError("calibration standardization produced nonfinite data")
    return x_train, y_train


def lambda_maximum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bnp,bn->bp", x.float(), y.float()).abs().amax(
        dim=-1
    ) / float(x.shape[1])


@torch.no_grad()
def collect_pass(
    selector: DirectPenaltyMLPQuerySelectorV22V20GeometryScaled,
    entries: list[dict[str, Any]],
    *,
    data_root: Path,
    split_seed_base: int,
    accumulators: Mapping[str, CoordinateWelford],
    interaction_only: bool,
    device: torch.device,
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for task_index, entry in enumerate(entries):
        x_train, y_train = load_standardized_train(
            data_root,
            entry,
            split_seed=split_seed_base + task_index,
            device=device,
        )
        trajectory = selector.encode_context(x_train, y_train)
        source, mask = selector.collect_scaler_sources(
            trajectory,
            lambda_maximum(x_train, y_train),
            scale_residual_for_interaction=interaction_only,
        )
        names = ("interaction",) if interaction_only else tuple(
            name for name in source if name != "interaction"
        )
        for name in names:
            accumulators[name].update(source[name], mask[name])
        if not interaction_only:
            manifest.append(
                {
                    "task_index": task_index,
                    "shard_id": int(entry["shard_id"]),
                    "seed": int(entry["seed"]),
                    "sha256": str(entry["sha256"]),
                    "n": int(entry["n"]),
                    "p": int(entry["p"]),
                    "split_seed": split_seed_base + task_index,
                    "active_stages": int(
                        trajectory["stage_active"].sum().item()
                    ),
                    "selected_correlation_pairs": int(
                        trajectory[
                            "support_correlation_valid_pair_count"
                        ].item()
                    ),
                }
            )
        del x_train, y_train, trajectory, source, mask
        torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "pass": 2 if interaction_only else 1,
                    "task": task_index + 1,
                    "task_count": len(entries),
                    "n": int(entry["n"]),
                    "p": int(entry["p"]),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--v20-checkpoint", type=Path, required=True)
    parser.add_argument("--v20-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-count", type=int, default=64)
    parser.add_argument("--selection-seed", type=int, default=202607250922)
    parser.add_argument("--split-seed", type=int, default=202607250923)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if sha256(args.v20_checkpoint) != args.v20_checkpoint_sha256:
        raise RuntimeError("V20 checkpoint SHA256 mismatch")
    pointer, catalog = load_catalog(args.control_root)
    entries = select_entries(
        catalog, task_count=args.task_count, seed=args.selection_seed
    )
    checkpoint = torch.load(
        args.v20_checkpoint, map_location="cpu", weights_only=False
    )
    torch.manual_seed(202607250924)
    selector = DirectPenaltyMLPQuerySelectorV22V20GeometryScaled(
        hidden=512,
        lambda_min_ratio=1e-6,
        lambda_max_ratio=1.0,
        initial_lambda_ratio=0.1,
        derivative_floor=1e-6,
        squareplus_scale=0.02,
        direct_head_init_std=1e-3,
    )
    import_report = load_v20_state_dict(
        selector, checkpoint["selector_state_dict"]
    )
    device = torch.device(args.device)
    selector.to(device).eval()
    accumulators = {
        name: CoordinateWelford(shape)
        for name, shape in selector.input_scaler.shapes.items()
    }
    manifest = collect_pass(
        selector,
        entries,
        data_root=args.data_root,
        split_seed_base=args.split_seed,
        accumulators=accumulators,
        interaction_only=False,
        device=device,
    )
    interim_payload = {
        "version": V22_SCALER_VERSION,
        "sources": {
            name: accumulator.payload()
            for name, accumulator in accumulators.items()
        },
        "metadata": {"interim_for_interaction_pass": True},
    }
    # Interaction is defined after residual scaling, so load pass-one residual
    # moments before computing the row-query/key interaction statistics.
    interim_payload["sources"]["interaction"] = {
        "mean": torch.zeros(selector.input_scaler.shapes["interaction"]),
        "std": torch.ones(selector.input_scaler.shapes["interaction"]),
        "count": torch.ones(
            selector.input_scaler.shapes["interaction"], dtype=torch.long
        ),
    }
    selector.load_input_scaler_cache(interim_payload)
    collect_pass(
        selector,
        entries,
        data_root=args.data_root,
        split_seed_base=args.split_seed,
        accumulators=accumulators,
        interaction_only=True,
        device=device,
    )
    payload = {
        "version": V22_SCALER_VERSION,
        "sources": {
            name: accumulator.payload()
            for name, accumulator in accumulators.items()
        },
        "metadata": {
            "estimator": "masked_parallel_welford_population_std_v1",
            "two_pass_interaction": True,
            "task_count": len(entries),
            "selection_seed": args.selection_seed,
            "split_seed_base": args.split_seed,
            "current_pointer": pointer,
            "catalog_sha256": pointer["catalog_sha256"],
            "v20_checkpoint": str(args.v20_checkpoint.resolve()),
            "v20_checkpoint_sha256": args.v20_checkpoint_sha256,
            "v20_state_import": import_report,
            "tasks": manifest,
        },
    }
    selector.load_input_scaler_cache(payload)
    report = selector.input_scaler.architecture_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    reloaded = torch.load(args.output, map_location="cpu", weights_only=False)
    if reloaded["version"] != V22_SCALER_VERSION:
        raise RuntimeError("saved V22 scaler cache failed strict reload")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256(args.output),
                "task_count": len(entries),
                "report": report,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
