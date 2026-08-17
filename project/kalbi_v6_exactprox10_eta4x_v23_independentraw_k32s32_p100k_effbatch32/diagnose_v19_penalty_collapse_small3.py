#!/usr/bin/env python3
"""Measure where V19 penalty diversity disappears on generated small tasks."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from .canonical_penalty import build_penalty
from .eval_kalbi_v2_common import (
    encode_dataset,
    split_standardize,
    tensors_from_arrays,
    vector_correlation,
)
from .report_kalbi_v3_v19_small_training_task import (
    load_components_v19,
    load_generated_task,
)


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-npz", type=Path, action="append", required=True)
    parser.add_argument("--split-seed", type=int, action="append", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.data_npz) != len(args.split_seed):
        raise ValueError("data and split-seed counts differ")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint, model, selector = load_components_v19(
        args.checkpoint, args.config, device
    )
    records: list[dict[str, object]] = []
    arrays: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "task_code",
            "penalty_code",
            "latent",
            "proposal",
            "rho",
            "drho",
        )
    }

    for path, split_seed in zip(args.data_npz, args.split_seed, strict=True):
        x_raw, y_raw, max_corr, provenance = load_generated_task(path)
        x_train, y_train, _, _ = split_standardize(x_raw, y_raw, split_seed)
        x_device, y_device = tensors_from_arrays(x_train, y_train, device)
        encoded = encode_dataset(model, x_device, y_device)
        with torch.inference_mode():
            result = selector(
                x_device,
                y_device,
                encoded["lambda_max"].float().reshape(-1),
                return_representation=True,
            )
        trajectory = result["v19_trajectory"]
        task_arrays = {
            "task_code": result["v19_task_code"],
            "penalty_code": result["v19_penalty_code"],
            "latent": result["penalty_latent_raw"],
            "proposal": selector.penalty_parameter_head._last_parameter_proposal,
            "rho": result["penalty_rho"],
            "drho": result["penalty_drho"],
        }
        for name, value in task_arrays.items():
            if value is None or not torch.is_tensor(value):
                raise RuntimeError(f"missing tensor {name}")
            arrays[name].append(value.reshape(-1).detach().float().cpu().numpy())
        stage_active = trajectory["stage_active"].reshape(-1).detach().cpu().numpy()
        selected_indices = (
            trajectory["selected_indices"].reshape(-1).detach().cpu().numpy()
        )
        residual_energy = (
            trajectory["residual_energy_ratio"].reshape(-1).detach().cpu().numpy()
        )
        records.append(
            {
                "task": path.parent.name,
                "n": provenance["n"],
                "p": provenance["p"],
                "split_seed": int(split_seed),
                "max_abs_corr": max_corr,
                "active_stages": int(np.count_nonzero(stage_active)),
                "selected_valid_features": int(np.count_nonzero(selected_indices >= 0)),
                "final_active_residual_energy": float(
                    residual_energy[np.flatnonzero(stage_active)[-1]]
                ),
                "effective_lambda_ratio": float(
                    result["effective_lambda_ratio"].reshape(-1)[0]
                ),
                "d0": float(result["penalty_d0"].reshape(-1)[0]),
                "reference_distance": float(
                    result["penalty_reference_distance"].reshape(-1)[0]
                ),
            }
        )

    stacked = {name: np.stack(values) for name, values in arrays.items()}
    reference = build_penalty(
        selector.penalty_parameter_head.parameter_mean.detach().float()[None],
        derivative_floor=selector.penalty_parameter_head.derivative_floor,
        squareplus_scale=selector.penalty_parameter_head.squareplus_scale,
        knots=selector.penalty_parameter_head.penalty_grid.detach().float(),
    )
    reference_rho = reference["rho"].reshape(-1).cpu().numpy()
    reference_drho = reference["drho"].reshape(-1).cpu().numpy()
    pairwise: list[dict[str, object]] = []
    for left, right in itertools.combinations(range(len(records)), 2):
        row: dict[str, object] = {
            "tasks": [records[left]["task"], records[right]["task"]]
        }
        for name in stacked:
            delta = stacked[name][left] - stacked[name][right]
            row[f"{name}_rmse"] = rms(delta)
            row[f"{name}_correlation"] = vector_correlation(
                stacked[name][left], stacked[name][right]
            )
        pairwise.append(row)

    stage_summary = {}
    for name, values in stacked.items():
        stage_summary[name] = {
            "across_task_std_mean": float(values.std(axis=0).mean()),
            "across_task_std_rms": rms(values.std(axis=0)),
            "mean_vector_rms": rms(values.mean(axis=0)),
            "relative_std_rms": rms(values.std(axis=0))
            / max(rms(values.mean(axis=0)), 1e-12),
        }
    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": int(checkpoint["completed_steps"]),
        "tasks": records,
        "stage_diversity": stage_summary,
        "pairwise": pairwise,
        "reference_penalty": {
            "d0": float(reference["d0"].reshape(-1)[0]),
            "selected_rho_rmse_to_reference": [
                rms(rho - reference_rho) for rho in stacked["rho"]
            ],
            "selected_drho_rmse_to_reference": [
                rms(drho - reference_drho) for drho in stacked["drho"]
            ],
        },
        "parameter_cache": {
            "cache_size": int(selector.penalty_parameter_head.cache_size),
            "parameter_std_min": float(
                selector.penalty_parameter_head.parameter_std.min()
            ),
            "parameter_std_median": float(
                selector.penalty_parameter_head.parameter_std.median()
            ),
            "parameter_std_max": float(
                selector.penalty_parameter_head.parameter_std.max()
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
