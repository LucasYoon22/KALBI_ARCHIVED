#!/usr/bin/env python3
"""Select and gate 20 representative random penalties from the 300K cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .canonical_penalty import CONTRACT_VERSION, build_penalty, canonical_grid
from .canonical_penalty_cache import CACHE_VERSION, file_sha256


def metrics(parameters: torch.Tensor, chunk: int = 8192) -> dict[str, torch.Tensor]:
    pieces = {key: [] for key in ("cv", "tv", "drop", "negative_fraction", "nonconvex_score")}
    for start in range(0, parameters.shape[0], int(chunk)):
        encoded = build_penalty(parameters[start : start + int(chunk)])
        drho = encoded["drho"]
        delta = drho[:, 1:] - drho[:, :-1]
        mean = drho.mean(1).clamp_min(1e-12)
        cv = drho.std(1, unbiased=False) / mean
        tv = delta.abs().sum(1) / mean
        running_future_min = torch.flip(
            torch.cummin(torch.flip(drho, dims=(1,)), dim=1).values,
            dims=(1,),
        )
        drop = ((drho - running_future_min).amax(1) / drho.amax(1).clamp_min(1e-12))
        negative_fraction = (delta < 0.0).float().mean(1)
        negative_tv = (-delta).clamp_min(0.0).sum(1) / mean
        score = drop * negative_tv * (0.25 + negative_fraction)
        for key, value in (
            ("cv", cv),
            ("tv", tv),
            ("drop", drop),
            ("negative_fraction", negative_fraction),
            ("nonconvex_score", score),
        ):
            pieces[key].append(value.cpu())
    return {key: torch.cat(value) for key, value in pieces.items()}


def select_twenty(parameters: torch.Tensor, all_metrics: dict[str, torch.Tensor], seed: int) -> list[int]:
    ridge = int(torch.argmin(all_metrics["cv"]))
    nonconvex = int(torch.argmax(all_metrics["nonconvex_score"]))
    rng = np.random.default_rng(int(seed))
    candidates = rng.choice(parameters.shape[0], size=min(20_000, parameters.shape[0]), replace=False)
    candidates = np.unique(np.concatenate([candidates, [ridge, nonconvex]])).astype(np.int64)
    rho = build_penalty(parameters[candidates])["rho"].cpu()
    index_of = {int(global_index): local for local, global_index in enumerate(candidates)}
    selected = [ridge, nonconvex]
    selected_local = [index_of[ridge], index_of[nonconvex]]
    while len(selected) < 20:
        distance = torch.cdist(rho, rho[selected_local]).amin(1)
        distance[selected_local] = -1.0
        local = int(torch.argmax(distance))
        selected_local.append(local)
        selected.append(int(candidates[local]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202607170148)
    args = parser.parse_args()
    payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    if payload.get("cache_version") != CACHE_VERSION:
        raise RuntimeError("diversity audit received the wrong cache version")
    if payload.get("canonical_contract_version") != CONTRACT_VERSION:
        raise RuntimeError("diversity audit received the wrong canonical contract")
    parameters = payload["direct_parameters"].float()
    all_metrics = metrics(parameters)
    selected = select_twenty(parameters, all_metrics, args.seed)
    encoded = build_penalty(parameters[selected])
    rho = encoded["rho"].cpu()
    drho = encoded["drho"].cpu()
    pairwise = torch.cdist(rho, rho)
    off_diagonal = pairwise[~torch.eye(20, dtype=torch.bool)]
    ridge_index = int(torch.argmin(all_metrics["cv"]))
    nonconvex_index = int(torch.argmax(all_metrics["nonconvex_score"]))
    report = {
        "status": "PASS",
        "cache": str(args.cache.resolve()),
        "cache_sha256": file_sha256(args.cache),
        "cache_count": int(parameters.shape[0]),
        "canonical_contract_version": CONTRACT_VERSION,
        "selection": "seeded_20k_candidate_maximin_with_global_ridge_and_nonconvex_anchors",
        "seed": int(args.seed),
        "selected_indices": selected,
        "ridge_anchor": {
            "index": ridge_index,
            "derivative_cv": float(all_metrics["cv"][ridge_index]),
            "derivative_tv_over_mean": float(all_metrics["tv"][ridge_index]),
        },
        "highly_nonconvex_anchor": {
            "index": nonconvex_index,
            "future_derivative_drop_fraction": float(all_metrics["drop"][nonconvex_index]),
            "negative_curvature_interval_fraction": float(all_metrics["negative_fraction"][nonconvex_index]),
            "nonconvex_score": float(all_metrics["nonconvex_score"][nonconvex_index]),
        },
        "selected_min_pairwise_rho_l2": float(off_diagonal.amin()),
        "selected_median_pairwise_rho_l2": float(off_diagonal.median()),
        "selected_drho_tv_min": float(encoded["metadata4"][:, 3].amin()),
        "selected_drho_tv_max": float(encoded["metadata4"][:, 3].amax()),
        "min_drho": float(drho.amin()),
        "rho6_error_max": float((rho[:, -1] - 1.0).abs().amax()),
        "dead_last_weight_max_abs": float(encoded["relu_weights"][:, -1].abs().amax()),
        "gates": {
            "ridge_derivative_cv_lt_0p10": float(all_metrics["cv"][ridge_index]) < 0.10,
            "nonconvex_drop_gt_0p50": float(all_metrics["drop"][nonconvex_index]) > 0.50,
            "nonconvex_negative_fraction_gt_0p15": float(all_metrics["negative_fraction"][nonconvex_index]) > 0.15,
            "selected_min_pairwise_rho_l2_gt_0p01": float(off_diagonal.amin()) > 0.01,
            "all_monotone": float(drho.amin()) > 0.0,
            "rho6_exact_2e6": float((rho[:, -1] - 1.0).abs().amax()) <= 2e-6,
            "dead_weight_exact_zero": float(encoded["relu_weights"][:, -1].abs().amax()) == 0.0,
        },
    }
    if not all(report["gates"].values()):
        report["status"] = "FAIL"
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "diversity_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(
        args.out_dir / "selected_penalties.npz",
        indices=np.asarray(selected, dtype=np.int64),
        grid=canonical_grid().numpy(),
        rho=rho.numpy(),
        drho=drho.numpy(),
        direct_parameters=parameters[selected].numpy(),
    )
    grid = canonical_grid().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for row in range(20):
        width = 3.0 if selected[row] in {ridge_index, nonconvex_index} else 1.1
        label = None
        if selected[row] == ridge_index:
            label = "ridge-like anchor"
        elif selected[row] == nonconvex_index:
            label = "highly non-convex anchor"
        axes[0].plot(grid, rho[row], linewidth=width, alpha=0.9, label=label)
        axes[1].plot(grid, drho[row], linewidth=width, alpha=0.9, label=label)
    axes[0].set(title="20 canonical random penalties", xlabel="|beta|", ylabel="rho")
    axes[1].set(title="Derivative shapes", xlabel="|beta|", ylabel="rho'")
    for axis in axes:
        axis.grid(alpha=0.25)
        if axis.get_legend_handles_labels()[0]:
            axis.legend()
    fig.savefig(args.out_dir / "penalty_diversity_20.png", dpi=220)
    plt.close(fig)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

