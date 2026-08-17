#!/usr/bin/env python3
"""Numerical and gradient contracts for the V22 selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .build_v22_input_scaler_cache import (
    load_standardized_train,
    lambda_maximum,
)
from .selector_v19_k32s32 import V22_CORRELATION_QUANTILES
from .selector_v22_v20geometry_scaled import (
    DirectPenaltyMLPQuerySelectorV22V20GeometryScaled,
    load_v20_state_dict,
)


def build_selector() -> DirectPenaltyMLPQuerySelectorV22V20GeometryScaled:
    return DirectPenaltyMLPQuerySelectorV22V20GeometryScaled(
        hidden=512,
        lambda_min_ratio=1e-6,
        lambda_max_ratio=1.0,
        initial_lambda_ratio=0.1,
        derivative_floor=1e-6,
        squareplus_scale=0.02,
        direct_head_init_std=1e-3,
    )


@torch.no_grad()
def brute_force_final_geometry(
    x: torch.Tensor,
    y: torch.Tensor,
    trajectory: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = trajectory["selected_indices"].reshape(1, -1).long()
    probe = trajectory["probe_row_indices"].long()
    x_selected = torch.gather(
        x,
        2,
        selected[:, None, :].expand(1, x.shape[1], selected.shape[1]),
    )
    x_probe = torch.gather(
        x_selected,
        1,
        probe[:, :, None].expand(1, probe.shape[1], selected.shape[1]),
    ).float()
    x_probe = (
        x_probe - x_probe.mean(1, keepdim=True)
    ) / x_probe.std(1, keepdim=True, correction=0).clamp_min(1e-8)
    y_probe = torch.gather(y, 1, probe).float()
    y_probe = y_probe - y_probe.mean(1, keepdim=True)
    y_probe = y_probe / y_probe.square().mean(1, keepdim=True).sqrt().clamp_min(
        1e-8
    )
    rows = x_probe.shape[1]
    gram = torch.einsum("bni,bnj->bij", x_probe, x_probe) / float(rows)
    alpha = trajectory["cumulative_ridge_alpha"].float()
    eye = torch.eye(gram.shape[-1], device=gram.device)[None]
    regularized = gram + (alpha + 1e-8)[:, None, None] * eye
    lower = torch.linalg.cholesky(regularized)
    pivots = torch.diagonal(lower, dim1=-2, dim2=-1)
    prefix = gram.shape[-1] - 32
    cross = gram[:, :prefix, prefix:]
    solved = torch.linalg.solve_triangular(
        lower[:, :prefix, :prefix], cross, upper=False
    )
    schur = (
        regularized[:, prefix:, prefix:]
        - torch.einsum("bik,bil->bkl", solved, solved)
    )
    schur_diag = torch.diagonal(schur, dim1=-2, dim2=-1)
    offdiag = gram[:, torch.triu_indices(
        gram.shape[-1], gram.shape[-1], offset=1, device=gram.device
    )[0], torch.triu_indices(
        gram.shape[-1], gram.shape[-1], offset=1, device=gram.device
    )[1]].abs()
    rhs = torch.einsum("bni,bn->bi", x_probe, y_probe) / float(rows)
    beta = torch.cholesky_solve(rhs.unsqueeze(-1), lower).squeeze(-1)
    residual = y_probe - torch.einsum("bni,bi->bn", x_probe, beta)
    frobenius_sq = gram.square().sum(dim=(-2, -1))
    trace = torch.diagonal(gram, dim1=-2, dim2=-1).sum(-1)
    expected = torch.stack(
        [
            2.0 * pivots.log().sum(-1),
            pivots.amin(-1),
            pivots.amax(-1),
            pivots.amax(-1).log() - pivots.amin(-1).log(),
            schur_diag.amin(-1),
            schur_diag.mean(-1),
            schur_diag.amax(-1),
            offdiag.amax(-1),
            offdiag.mean(-1),
            frobenius_sq.sqrt(),
            (trace.square() / frobenius_sq / gram.shape[-1]).clamp(0.0, 1.0),
            residual.square().mean(-1),
            beta.square().sum(-1).sqrt(),
        ],
        dim=-1,
    )
    quantile_levels = torch.tensor(
        V22_CORRELATION_QUANTILES, device=gram.device
    )
    expected_quantiles = torch.quantile(offdiag[0], quantile_levels)[None]
    return expected, expected_quantiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scaler-cache", type=Path, required=True)
    parser.add_argument("--v20-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cache = torch.load(args.scaler_cache, map_location="cpu", weights_only=False)
    task = cache["metadata"]["tasks"][0]
    entry = {
        "relative_path": (
            f"staging/shards/tabicl_v2/{task['sha256'][:2]}/"
            f"{task['sha256']}.npz"
        ),
        "n": task["n"],
        "p": task["p"],
    }
    device = torch.device(args.device)
    x, y = load_standardized_train(
        args.data_root,
        entry,
        split_seed=task["split_seed"],
        device=device,
    )
    checkpoint = torch.load(
        args.v20_checkpoint, map_location="cpu", weights_only=False
    )
    torch.manual_seed(202607250924)
    selector = build_selector()
    import_report = load_v20_state_dict(
        selector, checkpoint["selector_state_dict"]
    )
    selector.load_input_scaler_cache(cache)
    selector.to(device)
    trajectory = selector.encode_context(x, y)
    expected_geometry, expected_quantiles = brute_force_final_geometry(
        x, y, trajectory
    )
    actual_geometry = trajectory["cumulative_geometry_raw"][:, -1]
    actual_quantiles = trajectory["support_correlation_quantiles"]
    geometry_abs_error = (actual_geometry - expected_geometry).abs()
    quantile_abs_error = (actual_quantiles - expected_quantiles).abs()
    if float(geometry_abs_error.max()) > 2e-3:
        raise RuntimeError(
            f"V22 block geometry mismatch: {geometry_abs_error.tolist()}"
        )
    if float(quantile_abs_error.max()) > 2e-5:
        raise RuntimeError("V22 selected-support quantile mismatch")

    selector.train()
    query = selector(
        x,
        y,
        lambda_maximum(x, y),
        return_representation=True,
    )
    loss = (
        query["lambdas"].float().sum()
        + 1e-3 * query["penalty_shape_features"].float().square().sum()
    )
    loss.backward()
    gradient_norms = {
        "geometry_projection": torch.nn.utils.clip_grad_norm_(
            selector.geometry_projection.parameters(), float("inf")
        ).item(),
        "correlation_quantile_projection": torch.nn.utils.clip_grad_norm_(
            selector.correlation_quantile_projection.parameters(), float("inf")
        ).item(),
        "dimension_context_projection": torch.nn.utils.clip_grad_norm_(
            selector.dimension_context_projection.parameters(), float("inf")
        ).item(),
        "auxiliary_fusion": torch.nn.utils.clip_grad_norm_(
            selector.auxiliary_fusion.parameters(), float("inf")
        ).item(),
        "auxiliary_gate": float(selector.auxiliary_gate.grad.norm()),
        "lambda_head": torch.nn.utils.clip_grad_norm_(
            selector.lambda_head.parameters(), float("inf")
        ).item(),
        "penalty_head": torch.nn.utils.clip_grad_norm_(
            selector.penalty_parameter_head.parameters(), float("inf")
        ).item(),
    }
    if any(
        not torch.isfinite(torch.tensor(value)) or value <= 0.0
        for value in gradient_norms.values()
    ):
        raise RuntimeError(f"V22 gradient contract failed: {gradient_norms}")
    if not bool(torch.isfinite(query["penalty_rho"]).all()):
        raise RuntimeError("V22 penalty rho is nonfinite")
    if float(query["penalty_rho"][..., 0].detach().abs().max()) > 1e-7:
        raise RuntimeError("V22 rho(0) contract failed")

    state = selector.state_dict()
    reloaded = build_selector().to(device)
    reloaded.load_state_dict(state, strict=True)
    reloaded.eval()
    selector.eval()
    with torch.no_grad():
        original_output = selector(x, y, lambda_maximum(x, y))["lambdas"]
        reloaded_output = reloaded(x, y, lambda_maximum(x, y))["lambdas"]
    reload_error = float((original_output - reloaded_output).abs().max())
    if reload_error > 1e-7:
        raise RuntimeError("V22 strict state reload changed output")

    print(
        json.dumps(
            {
                "status": "PASS",
                "parameter_count": sum(
                    parameter.numel() for parameter in selector.parameters()
                ),
                "import_report": import_report,
                "geometry_max_abs_error": float(geometry_abs_error.max()),
                "geometry_abs_error_by_stat": geometry_abs_error[0].tolist(),
                "quantile_max_abs_error": float(quantile_abs_error.max()),
                "gradient_norms": gradient_norms,
                "strict_reload_output_max_abs_error": reload_error,
                "raw_lambda": float(query["raw_lambda"].item()),
                "effective_ratio": float(
                    query["effective_lambda_ratio"].item()
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
