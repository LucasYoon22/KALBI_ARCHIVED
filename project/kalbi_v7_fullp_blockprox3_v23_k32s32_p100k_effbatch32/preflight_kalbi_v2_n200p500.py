#!/usr/bin/env python3
"""Validate Kalbi V2 on one real N=200, P=500 rolling-data task."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from train_fpk import _load_yaml

from .model import (
    KALBI_RIDGE,
    KALBI_STAGES,
    KALBI_TOPK,
    MODEL_VERSION,
    build_model,
)
from .utils import build_kalbi_ab_trajectory


DATA_ROOT = Path(
    "/users/sync_data/"
    "kalbi_tabpfnv1_tabiclv2_scm_xy_50_50_groupdrop095_"
    "N200_P500_100GB_seed88073500"
)
CONTROL_ROOT = DATA_ROOT / "smoke_control_128"
CONFIG_PATH = Path("/users/minsuk/fm-lm-lambda/fpk_v33_kkt0.yaml")


def load_first_task() -> tuple[np.ndarray, np.ndarray, str]:
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


def standardize(x: np.ndarray, y: np.ndarray) -> tuple[Tensor, Tensor]:
    xt = torch.from_numpy(x).unsqueeze(0)
    yt = torch.from_numpy(y).reshape(1, -1)
    xt = xt - xt.mean(dim=1, keepdim=True)
    xt = xt / xt.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
    yt = yt - yt.mean(dim=1, keepdim=True)
    yt = yt / yt.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
    return xt, yt


Tensor = torch.Tensor


def main() -> None:
    torch.manual_seed(202607212100)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_raw, y_raw, data_path = load_first_task()
    x, y = standardize(x_raw, y_raw)
    x = x.to(device)
    y = y.to(device)

    model = build_model(_load_yaml(str(CONFIG_PATH)), device).train()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    lambda_max = (
        torch.einsum("bnp,bn->bp", x.float(), y.float())
        .abs()
        .amax(dim=-1)
        .clamp_min(1e-6)
    )
    ratios = torch.tensor([[1e-3, 1e-2, 1e-1, 1.0]], device=device)
    lambdas = (ratios * lambda_max[:, None]).requires_grad_(True)
    shape = (0.01 * torch.randn(1, 4, 196, device=device)).requires_grad_(True)
    d0 = torch.ones(1, 4, device=device, requires_grad=True)

    result = model(
        x,
        y,
        lambdas=lambdas,
        lambda_max=lambda_max[:, None],
        penalty_shape_features=shape,
        penalty_d0=d0,
        return_aux=True,
    )
    probe = (
        result["beta_q"].float()
        * torch.linspace(0.5, 1.5, 500, device=device).view(1, 1, 500)
    ).square().mean()
    lambda_grad, shape_grad, d0_grad = torch.autograd.grad(
        probe, (lambdas, shape, d0), allow_unused=False
    )

    indices = result["kalbi_v2_selected_indices"]
    active = result["kalbi_v2_stage_active"]
    active_indices = indices[active].reshape(-1)
    unique_count = int(torch.unique(active_indices).numel())
    A = result["kalbi_v2_A"]
    B = result["kalbi_v2_B"]
    active_count = int(active.sum())

    # Directly verify that explicit centering removes arbitrary offsets from
    # the correlation support decision.
    offset_x = torch.linspace(-4.0, 5.0, 500, device=device).view(1, 1, 500)
    short_a = build_kalbi_ab_trajectory(
        x, y, None, topk=KALBI_TOPK, stages=3, ridge=KALBI_RIDGE
    )
    short_b = build_kalbi_ab_trajectory(
        x + offset_x,
        y + 11.0,
        None,
        topk=KALBI_TOPK,
        stages=3,
        ridge=KALBI_RIDGE,
    )

    expected_active = 500 // KALBI_TOPK
    checks = {
        "shape_is_n200_p500": tuple(x.shape) == (1, 200, 500),
        "configuration_is_k32_s32": (
            model.kalbi_isis_topk == 32 and model.kalbi_isis_stages == 32
        ),
        "no_local_steps_attribute": not hasattr(model, "v46_isis_steps"),
        "no_local_chebyshev_mixer": not hasattr(model, "v46_local_mixer"),
        "A_shape": tuple(A.shape) == (1, 500, 32),
        "B_shape": tuple(B.shape) == (1, 500, 32),
        "finite_A_B_beta": bool(
            torch.isfinite(A).all()
            and torch.isfinite(B).all()
            and torch.isfinite(result["beta_q"]).all()
        ),
        "active_stages_are_floor_P_over_K": active_count == expected_active,
        "all_active_support_is_disjoint": unique_count == expected_active * 32,
        "fixed_ridge_is_1e-6": bool(
            torch.all(result["kalbi_v2_isis_ridge"] == KALBI_RIDGE)
        ),
        "inactive_columns_are_zero": bool(
            (A[..., expected_active:] == 0).all()
            and (B[..., expected_active:] == 0).all()
        ),
        "offset_invariant_centered_support": torch.equal(
            short_a["selected_indices"], short_b["selected_indices"]
        ),
        "lambda_gradient_nonzero": bool(lambda_grad.abs().sum() > 0),
        "shape_gradient_nonzero": bool(shape_grad.abs().sum() > 0),
        "d0_gradient_nonzero": bool(d0_grad.abs().sum() > 0),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Kalbi V2 preflight failed: {checks}")

    print(
        json.dumps(
            {
                "model_version": MODEL_VERSION,
                "device": str(device),
                "data_path": data_path,
                "parameter_count": parameter_count,
                "A_shape": list(A.shape),
                "B_shape": list(B.shape),
                "active_stages": active_count,
                "selected_features": int(active_indices.numel()),
                "unique_selected_features": unique_count,
                "gradient_abs_sum": {
                    "lambda": float(lambda_grad.abs().sum()),
                    "shape": float(shape_grad.abs().sum()),
                    "d0": float(d0_grad.abs().sum()),
                },
                "checks": checks,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
