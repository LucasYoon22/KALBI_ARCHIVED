#!/usr/bin/env python3
"""Paired V23/V24 held-out-loss comparison on fixed manifest tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .model import build_model
from .selector_v23_independent_rawlambda import (
    DirectPenaltyMLPQuerySelectorV23IndependentRawLambda,
)
from .selector_v24_additive512_trajectory import (
    DirectPenaltyMLPQuerySelectorV24Additive512,
)
from .train_selector_only_v24_frozen_kalbi_v6_cachedtraj_8gpu import (
    split_and_standardize_variable,
)
from train_fpk import _load_yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_v23() -> DirectPenaltyMLPQuerySelectorV23IndependentRawLambda:
    return DirectPenaltyMLPQuerySelectorV23IndependentRawLambda(
        hidden=512,
        derivative_floor=1e-6,
        squareplus_scale=0.02,
        direct_head_init_std=1e-3,
    )


def make_v24() -> DirectPenaltyMLPQuerySelectorV24Additive512:
    return DirectPenaltyMLPQuerySelectorV24Additive512(
        hidden=512,
        derivative_floor=1e-6,
        squareplus_scale=0.02,
        direct_head_init_std=1e-3,
    )


def heldout_metrics(
    beta: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    query: dict[str, torch.Tensor],
    x_test: torch.Tensor,
    y_test: torch.Tensor,
) -> tuple[float, float, float]:
    decoded = beta.decode_queries(
        encoded,
        lambdas=query["lambdas"],
        lambda_max=encoded["lambda_max"].float()[:, None],
        penalty_shape_features=query["penalty_shape_features"],
        penalty_d0=query["penalty_d0"],
        return_aux=True,
    )
    coefficient = decoded["beta_q"]
    prediction = torch.einsum("bnp,bqp->bqn", x_test, coefficient)
    target = y_test[:, None, :]
    f_hat = 0.5 * (prediction - target).square().mean()
    f_zero = 0.5 * target.square().mean()
    ratio = f_hat / f_zero.clamp_min(1e-12)
    return float(torch.log1p(ratio)), float(f_hat), float(ratio)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v23-checkpoint", type=Path, required=True)
    parser.add_argument("--v23-sha256", required=True)
    parser.add_argument("--v24-checkpoint", type=Path, required=True)
    parser.add_argument("--v24-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rolling-control-root", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if sha256(args.v23_checkpoint) != args.v23_sha256:
        raise RuntimeError("V23 checkpoint SHA256 mismatch")
    if sha256(args.v24_checkpoint) != args.v24_sha256:
        raise RuntimeError("V24 checkpoint SHA256 mismatch")
    v23_checkpoint = torch.load(
        args.v23_checkpoint, map_location="cpu", weights_only=False
    )
    v24_checkpoint = torch.load(
        args.v24_checkpoint, map_location="cpu", weights_only=False
    )
    device = torch.device(args.device)
    beta = build_model(_load_yaml(str(args.config)), device)
    beta.load_state_dict(v23_checkpoint["model_state_dict"], strict=True)
    beta.eval()
    for parameter in beta.parameters():
        parameter.requires_grad_(False)

    selectors = {"v23": make_v23(), "v24": make_v24()}
    selectors["v23"].load_state_dict(
        v23_checkpoint["selector_state_dict"], strict=True
    )
    selectors["v24"].load_state_dict(
        v24_checkpoint["selector_state_dict"], strict=True
    )
    for selector in selectors.values():
        selector.to(device).eval()

    manifest = json.loads(args.manifest.read_text())
    selected = manifest["tasks"][: args.tasks]
    pointer = manifest["catalog_pointer"]
    catalog_path = (
        args.rolling_control_root / "catalogs" / pointer["catalog_file"]
    )
    catalog = json.loads(catalog_path.read_text())
    entries = {
        int(entry["shard_id"]): entry for entry in catalog["files"]
    }
    records: list[dict[str, float | int | str]] = []

    with torch.no_grad():
        for position, task in enumerate(selected):
            entry = entries[int(task["shard_id"])]
            path = Path(catalog["data_root"]) / entry["relative_path"]
            if sha256(path) != task["sha256"]:
                raise RuntimeError(f"task hash mismatch: {path}")
            with np.load(path, allow_pickle=False) as archive:
                x_np = np.asarray(archive["X"][0], dtype=np.float32)
                y_np = np.asarray(archive["y"][0], dtype=np.float32)
            generator = torch.Generator(device="cpu").manual_seed(
                int(task["split_seed"])
            )
            x_train, y_train, x_test, y_test = split_and_standardize_variable(
                [x_np], [y_np], generator=generator, device=device
            )
            encoded = beta.encode_dataset(beta.precompute(x_train, y_train))
            row: dict[str, float | int | str] = {
                "position": position,
                "shard_id": int(task["shard_id"]),
                "n": int(task["n"]),
                "p": int(task["p"]),
                "stratum": str(task["stratum"]),
            }
            for name, selector in selectors.items():
                selector_kwargs = {"return_representation": True}
                if name == "v24":
                    selector_kwargs["solver_context"] = encoded
                query = selector(
                    x_train,
                    y_train,
                    encoded["lambda_max"].float(),
                    **selector_kwargs,
                )
                loss, f_hat, ratio = heldout_metrics(
                    beta, encoded, query, x_test, y_test
                )
                row[f"{name}_loss"] = loss
                row[f"{name}_f_hat"] = f_hat
                row[f"{name}_ratio"] = ratio
                row[f"{name}_raw_lambda"] = float(query["raw_lambda"][0])
            row["loss_delta_v24_minus_v23"] = (
                float(row["v24_loss"]) - float(row["v23_loss"])
            )
            records.append(row)

    v23_losses = [float(row["v23_loss"]) for row in records]
    v24_losses = [float(row["v24_loss"]) for row in records]
    deltas = [float(row["loss_delta_v24_minus_v23"]) for row in records]
    report = {
        "status": "PASS",
        "selection_rule": "first_tasks_of_precommitted_v24_scaler_manifest",
        "diagnostic_scope": (
            "paired fixed-task diagnostic; tasks contributed scaler statistics "
            "and are not an unseen generalization set"
        ),
        "task_count": len(records),
        "v23_checkpoint": str(args.v23_checkpoint),
        "v23_checkpoint_sha256": args.v23_sha256,
        "v24_checkpoint": str(args.v24_checkpoint),
        "v24_checkpoint_sha256": args.v24_sha256,
        "v23_loss": summarize(v23_losses),
        "v24_loss": summarize(v24_losses),
        "paired_delta_v24_minus_v23": summarize(deltas),
        "v24_better_task_count": sum(delta < 0 for delta in deltas),
        "v23_better_task_count": sum(delta > 0 for delta in deltas),
        "equal_task_count": sum(delta == 0 for delta in deltas),
        "records": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
