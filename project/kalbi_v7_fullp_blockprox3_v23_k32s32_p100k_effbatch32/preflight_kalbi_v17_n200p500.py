#!/usr/bin/env python3
"""Validate Kalbi V1 shared support and V17 decode on a real P=500 shard."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from train_fpk import _load_yaml

from .canonical_penalty_cache import CanonicalPenaltyCache
from .model import MODEL_VERSION, SHARED_SUPPORT_CONTRACT, build_model
from .selector_v17_stagegain import (
    DirectPenaltyMLPQuerySelectorV17StageGain,
    V17_SELECTOR_VERSION,
)


DATA_ROOT = Path(
    "/users/sync_data/"
    "kalbi_tabpfnv1_tabiclv2_scm_xy_50_50_groupdrop095_"
    "N200_P500_100GB_seed88073500"
)
CONTROL_ROOT = DATA_ROOT / "smoke_control_128"
CONFIG_PATH = Path("/users/minsuk/fm-lm-lambda/fpk_v33_kkt0.yaml")
PENALTY_CACHE_PATH = Path(
    "/users/minsuk/fm-lm-lambda/penalty_cache_300k_v12_canonical_final.pt"
)


def load_first_shard() -> tuple[np.ndarray, np.ndarray, str]:
    pointer = json.loads((CONTROL_ROOT / "CURRENT.json").read_text())
    catalog = json.loads(
        (CONTROL_ROOT / "catalogs" / pointer["catalog_file"]).read_text()
    )
    relative_path = catalog["files"][0]["relative_path"]
    path = DATA_ROOT / relative_path
    with np.load(path, allow_pickle=False) as payload:
        x = np.asarray(payload["X"][0], dtype=np.float32)
        y = np.asarray(payload["y"][0], dtype=np.float32)
    return x, y, str(path)


def standardize(x: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    xt = torch.from_numpy(x).unsqueeze(0)
    yt = torch.from_numpy(y).reshape(1, -1)
    xt = xt - xt.mean(dim=1, keepdim=True)
    xt = xt / xt.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
    yt = yt - yt.mean(dim=1, keepdim=True)
    yt = yt / yt.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
    return xt, yt


def main() -> None:
    torch.manual_seed(202607211700)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_raw, y_raw, data_path = load_first_shard()
    x, y = standardize(x_raw, y_raw)
    x = x.to(device)
    y = y.to(device)

    model = build_model(_load_yaml(str(CONFIG_PATH)), device).eval()
    query_count = 8
    lambda_max = (
        torch.einsum("bnp,bn->bp", x.float(), y.float())
        .abs()
        .amax(dim=-1)
        .clamp_min(1e-6)
    )
    ratios = torch.logspace(-4, 0, query_count, device=device).unsqueeze(0)
    lambdas = ratios * lambda_max[:, None]
    shape = torch.zeros(1, query_count, 196, device=device)
    d0 = torch.ones(1, query_count, device=device)

    with torch.no_grad():
        first = model(
            x,
            y,
            lambdas=lambdas,
            lambda_max=lambda_max[:, None],
            penalty_shape_features=shape,
            penalty_d0=d0,
            return_aux=True,
        )
        reversed_query = model(
            x,
            y,
            lambdas=lambdas.flip(1),
            lambda_max=lambda_max[:, None],
            penalty_shape_features=shape.flip(1),
            penalty_d0=d0.flip(1),
            return_aux=True,
        )

        selector = DirectPenaltyMLPQuerySelectorV17StageGain()
        cache = CanonicalPenaltyCache(PENALTY_CACHE_PATH, seed=202607211701)
        cache.load(expected_count=300_000)
        selector.fit_penalty_parameter_cache(cache.direct_parameters)
        selector = selector.to(device).eval()
        encoded = model.encode_dataset(model.precompute(x, y))
        selected_query = selector(
            x, y, encoded["lambda_max"].float(), return_representation=True
        )
        selected_decode = model.decode_queries(
            encoded,
            lambdas=selected_query["lambdas"],
            lambda_max=encoded["lambda_max"].float()[:, None],
            penalty_shape_features=selected_query["penalty_shape_features"],
            penalty_d0=selected_query["penalty_d0"],
            return_aux=True,
        )

    selected = first["v56fast_isis_selected_indices"]
    shared = first["kalbi_shared_isis_selected_indices"]
    active = first["kalbi_shared_isis_stage_active"]
    active_count = int(active.sum())
    active_indices = shared[active].reshape(-1)
    unique_count = int(torch.unique(active_indices).numel())
    reversed_shared = reversed_query["kalbi_shared_isis_selected_indices"]

    checks = {
        "shape_is_n200_p500": tuple(x.shape) == (1, 200, 500),
        "support_has_no_query_axis": tuple(shared.shape) == (1, 64, 16),
        "broadcast_support_is_query_identical": torch.equal(
            selected, selected[:, :1].expand_as(selected)
        ),
        "lambda_reversal_does_not_change_support": torch.equal(
            shared, reversed_shared
        ),
        "active_stages_are_31": active_count == 31,
        "all_496_indices_are_disjoint": unique_count == 496,
        "v17_output_is_finite": bool(
            torch.isfinite(selected_query["lambdas"]).all()
            and torch.isfinite(selected_query["penalty_shape_features"]).all()
        ),
        "v17_to_kalbi_decode_is_finite": bool(
            torch.isfinite(selected_decode["beta_q"]).all()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Kalbi V1 + V17 preflight failed: {checks}")

    print(json.dumps({
        "model_version": MODEL_VERSION,
        "selector_version": V17_SELECTOR_VERSION,
        "shared_support_contract": SHARED_SUPPORT_CONTRACT,
        "device": str(device),
        "data_path": data_path,
        "x_shape": list(x.shape),
        "shared_support_shape": list(shared.shape),
        "broadcast_support_shape": list(selected.shape),
        "active_stages": active_count,
        "selected_features": int(active_indices.numel()),
        "unique_selected_features": unique_count,
        "checks": checks,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
