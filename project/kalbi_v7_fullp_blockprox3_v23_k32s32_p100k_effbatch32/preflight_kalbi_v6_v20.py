#!/usr/bin/env python3
"""End-to-end Kalbi V6 exact-prox forward, descent, and gradient preflight."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from train_fpk import _load_yaml

from .canonical_penalty import build_penalty
from .exact_direct96_prox import (
    interpolate_direct96_penalty,
    unpack_direct96_penalty,
)
from .model import (
    KALBI_EXACT_PG_RITZ_SAFETY,
    KALBI_EXACT_PG_STEPS,
    MODEL_VERSION,
    build_model,
)
from .selector_v19_k32s32 import DirectPenaltyMLPQuerySelectorV19StageGain
from .selector_v10 import lambda_from_effective_ratio_v56
from .selector_v20_yconditioned import (
    DirectPenaltyMLPQuerySelectorV20YConditioned,
    V20_NEW_STATE_KEYS,
    load_v19_shared_state_dict,
)
from .train import objective_kalbi_v6_exactprox10


PROJECT = Path("/users/minsuk/fm-lm-lambda")


def gradient_norm(parameters) -> float:
    total = torch.zeros((), device="cuda")
    for parameter in parameters:
        if parameter.grad is not None:
            total = total + parameter.grad.float().square().sum()
    return float(total.sqrt())


def make_task(
    batch: int,
    rows: int,
    features: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    latent_width = min(12, max(4, features // 8))
    latent = torch.randn(batch, rows, latent_width, device=device)
    loading = (
        torch.randn(batch, latent_width, features, device=device)
        / latent_width**0.5
    )
    x = torch.matmul(latent, loading) + 0.4 * torch.randn(
        batch, rows, features, device=device
    )
    beta_true = torch.zeros(batch, features, device=device)
    signal_width = min(10, features)
    beta_true[:, :signal_width] = torch.linspace(
        1.0, -0.6, signal_width, device=device
    )
    y = torch.einsum("bnp,bp->bn", x, beta_true)
    y = y + 0.3 * torch.randn(batch, rows, device=device)
    return x, y, beta_true


def composite_objective(
    beta: torch.Tensor,
    x_centered: torch.Tensor,
    y_centered: torch.Tensor,
    lambdas: torch.Tensor,
    native196: torch.Tensor,
    grid: torch.Tensor,
) -> torch.Tensor:
    prediction = torch.matmul(beta.float(), x_centered.float().transpose(1, 2))
    data_half = 0.5 * (
        prediction - y_centered.float()[:, None, :]
    ).square().mean(dim=-1)
    pieces = unpack_direct96_penalty(native196.float(), grid)
    rho, _, _ = interpolate_direct96_penalty(beta.float().abs(), pieces, grid)
    return data_half + lambdas.float() * rho.sum(dim=-1)


def run_shape_smoke(
    model: torch.nn.Module,
    *,
    rows: int,
    features: int,
    device: torch.device,
) -> dict[str, float]:
    x, y, _ = make_task(1, rows, features, device=device)
    encoded = model.encode_dataset(model.precompute(x, y))
    lambda_max = encoded["lambda_max"].float()
    lambdas = lambda_max[:, None] * torch.tensor(
        [[0.03, 0.15]], device=device
    )
    direct = torch.zeros(2, 96, device=device)
    penalty = build_penalty(direct)
    aux = model.decode_queries(
        encoded,
        lambdas=lambdas,
        lambda_max=lambda_max[:, None],
        penalty_shape_features=penalty["shape_features"].view(1, 2, 196),
        penalty_d0=penalty["d0"].view(1, 2),
        return_aux=True,
    )
    beta = aux["beta_q"].float()
    return {
        "rows": float(rows),
        "features": float(features),
        "beta_abs_max": float(beta.abs().max()),
        "majorization_ratio_max": float(
            aux["kalbi_v6_refinement_majorization_ratio_q"].max()
        ),
        "minimum_curvature_denominator": float(
            aux[
                "kalbi_v6_refinement_min_curvature_denominator_q"
            ].min().detach()
        ),
    }


def run_training_objective_smoke(
    model: torch.nn.Module,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.zero_grad(set_to_none=True)
    x, y, _ = make_task(1, 52, 80, device=device)
    with torch.no_grad():
        encoded = model.encode_dataset(model.precompute(x, y))
        lambda_max = encoded["lambda_max"].float()
    lambdas = lambda_max[:, None] * torch.tensor(
        [[0.035, 0.14]], device=device
    )
    penalty = build_penalty(
        0.1 * torch.randn(2, 96, device=device)
    )
    payload = {
        "X": x,
        "y": y,
        "lambdas": lambdas,
        "lambda_max": lambda_max[:, None].expand(1, 2),
        "shape": penalty["shape_features"].view(1, 2, 196),
        "d0": penalty["d0"].view(1, 2),
        "relu_b": penalty["relu_b"],
        "relu_weights": penalty["relu_weights"],
    }
    result = objective_kalbi_v6_exactprox10(
        model, payload, model._kalbi_v6_penalty_grid
    )
    result["loss"].backward()
    scalar_names = (
        "loss",
        "data_half",
        "penalty",
        "objective",
        "kkt_top32",
        "refinement_step_size",
        "refinement_min_curvature_denominator",
        "refinement_majorization_ratio",
        "refinement_change_rms",
    )
    if any(not bool(torch.isfinite(result[name])) for name in scalar_names):
        raise RuntimeError("Kalbi V6 trainer objective produced nonfinite metrics")
    return {
        "loss": float(result["loss"].detach()),
        "kkt_top32": float(result["kkt_top32"].detach()),
        "majorization_ratio": float(
            result["refinement_majorization_ratio"].detach()
        ),
        "model_gradient": gradient_norm(model.parameters()),
    }


def _standardize_context(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    x_centered = x - x.mean(dim=1, keepdim=True)
    x_scaled = x_centered / x_centered.square().mean(
        dim=1, keepdim=True
    ).sqrt().clamp_min(1e-6)
    y_centered = y - y.mean(dim=1, keepdim=True)
    y_scaled = y_centered / y_centered.square().mean(
        dim=1, keepdim=True
    ).sqrt().clamp_min(1e-6)
    return x_scaled, y_scaled


def run_selector_v20_smoke(
    model: torch.nn.Module,
    *,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(202607240021)
    selector_v19 = DirectPenaltyMLPQuerySelectorV19StageGain()
    selector_v19.fit_penalty_parameter_cache(torch.randn(512, 96))
    torch.manual_seed(202607240022)
    selector = DirectPenaltyMLPQuerySelectorV20YConditioned()
    import_report = load_v19_shared_state_dict(
        selector, selector_v19.state_dict()
    )
    selector_state = selector.state_dict()
    shared_state_exact = all(
        torch.equal(value, selector_state[name])
        for name, value in selector_v19.state_dict().items()
    )
    del selector_v19
    selector = selector.to(device).train()

    ratio_probe = torch.tensor([0.2], device=device, requires_grad=True)
    lmax_probe = torch.tensor([1.5], device=device)
    d0_probe = torch.tensor([0.0], device=device)
    reference_probe = torch.tensor([0.3], device=device, requires_grad=True)
    generalized_scale = lambda_from_effective_ratio_v56(
        ratio_probe,
        lmax_probe,
        d0_probe,
        reference_slope=reference_probe,
    )
    generalized_scale["lambdas"].sum().backward()
    generalized_ratio_error = (
        generalized_scale["effective_ratio"].detach() - ratio_probe.detach()
    ).abs().max()
    if (
        float(generalized_ratio_error) > 1e-7
        or not bool(generalized_scale["ridge_branch"].all())
        or float(ratio_probe.grad.abs().min()) <= 0.0
        or float(reference_probe.grad.abs().min()) <= 0.0
    ):
        raise RuntimeError("V20 generalized lambda-scale contract failed")

    for parameter in model.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)
    x, y, _ = make_task(1, 57, 96, device=device)
    x, y = _standardize_context(x, y)
    encoded = model.encode_dataset(model.precompute(x, y))
    query = selector(
        x,
        y,
        encoded["lambda_max"].float(),
        return_representation=True,
    )
    aux = model.decode_queries(
        encoded,
        lambdas=query["lambdas"],
        lambda_max=encoded["lambda_max"].float()[:, None],
        penalty_shape_features=query["penalty_shape_features"],
        penalty_d0=query["penalty_d0"],
        return_aux=True,
    )
    heldout_x = x + 0.05 * torch.randn_like(x)
    heldout_y = y + 0.05 * torch.randn_like(y)
    prediction = torch.einsum(
        "bnp,bp->bn", heldout_x.float(), aux["beta_q"][:, 0].float()
    )
    heldout_loss = torch.log1p(
        0.5 * (prediction - heldout_y.float()).square().mean(dim=-1)
    ).mean()
    heldout_loss.backward()
    response_token_gradient = gradient_norm(
        selector.response_token_projection.parameters()
    )
    response_pool_gradient = gradient_norm(
        selector.response_pool_projection.parameters()
    )

    selector.eval()
    with torch.no_grad():
        permutation = torch.randperm(x.shape[1], device=device)
        permuted = selector(
            x[:, permutation],
            y[:, permutation],
            encoded["lambda_max"].float(),
            return_representation=True,
        )
        lambda_permutation_error = (
            query["lambda_ratio"].detach() - permuted["lambda_ratio"]
        ).abs().max()
        penalty_permutation_error = (
            query["penalty_shape_features"].detach()
            - permuted["penalty_shape_features"]
        ).abs().max()

        variable_n_shapes: list[list[int]] = []
        for rows in (37, 83):
            x_variable, y_variable, _ = make_task(
                1, rows, 96, device=device
            )
            x_variable, y_variable = _standardize_context(
                x_variable, y_variable
            )
            encoded_variable = model.encode_dataset(
                model.precompute(x_variable, y_variable)
            )
            output = selector(
                x_variable,
                y_variable,
                encoded_variable["lambda_max"].float(),
                return_representation=True,
            )
            variable_n_shapes.append(
                list(output["v20_response_tokens"].shape)
            )

    checks: dict[str, object] = {
        "parameter_count": sum(p.numel() for p in selector.parameters()),
        "new_state_keys": sorted(V20_NEW_STATE_KEYS),
        "state_import": import_report,
        "shared_v19_state_exact": shared_state_exact,
        "heldout_loss": float(heldout_loss.detach()),
        "response_token_gradient": response_token_gradient,
        "response_pool_gradient": response_pool_gradient,
        "lambda_permutation_error": float(lambda_permutation_error),
        "penalty_permutation_error": float(penalty_permutation_error),
        "variable_n_response_token_shapes": variable_n_shapes,
        "heldout_y_used_as_selector_input": False,
        "generalized_lambda_ratio_error": float(generalized_ratio_error),
        "ridge_like_d0_supported": bool(
            generalized_scale["ridge_branch"].all()
        ),
        "lambda_reference_slope_min": float(
            query["penalty_reference_slope"].detach().min()
        ),
        "lambda_reference_matches_max_drho": bool(
            torch.equal(
                query["penalty_reference_slope"].detach(),
                query["penalty_drho"].detach().amax(dim=-1),
            )
        ),
    }
    if checks["parameter_count"] != 21_749_345:
        raise RuntimeError(f"V20 parameter-count drift: {checks}")
    if not shared_state_exact:
        raise RuntimeError(f"V19 shared-state import drift: {checks}")
    if response_token_gradient <= 0.0 or response_pool_gradient <= 0.0:
        raise RuntimeError(f"V20 response gradient is missing: {checks}")
    if float(lambda_permutation_error) > 5e-4:
        raise RuntimeError(f"V20 lambda row-permutation drift: {checks}")
    if float(penalty_permutation_error) > 5e-4:
        raise RuntimeError(f"V20 penalty row-permutation drift: {checks}")
    if variable_n_shapes != [[1, 37, 512], [1, 83, 512]]:
        raise RuntimeError(f"V20 variable-N response shape drift: {checks}")
    if not checks["lambda_reference_matches_max_drho"]:
        raise RuntimeError(f"V20 generalized reference-slope drift: {checks}")
    return checks


def main() -> None:
    torch.manual_seed(202607232355)
    device = torch.device("cuda:0")
    model = build_model(_load_yaml(str(PROJECT / "fpk_v33_kkt0.yaml")), device)
    model.train()

    batch, rows, features, queries = 1, 64, 96, 3
    x, y, beta_true = make_task(batch, rows, features, device=device)
    encoded = model.encode_dataset(model.precompute(x, y))
    lambda_max = encoded["lambda_max"].float()
    lambdas = (
        lambda_max[:, None]
        * torch.tensor([[0.015, 0.06, 0.2]], device=device)
    ).detach().requires_grad_(True)
    direct = (
        0.15 * torch.randn(batch, queries, 96, device=device)
    ).requires_grad_(True)
    penalty = build_penalty(direct.reshape(-1, 96))
    native = penalty["shape_features"].reshape(batch, queries, 196)
    d0 = penalty["d0"].reshape(batch, queries)

    aux = model.decode_queries(
        encoded,
        lambdas=lambdas,
        lambda_max=lambda_max[:, None],
        penalty_shape_features=native,
        penalty_d0=d0,
        return_aux=True,
    )
    beta0 = aux["kalbi_v6_beta0_q"].float()
    beta = aux["beta_q"].float()
    grid = model._kalbi_v6_penalty_grid.float()
    objective0 = composite_objective(
        beta0, encoded["Xc"], encoded["yc"], lambdas, native, grid
    )
    objective1 = composite_objective(
        beta, encoded["Xc"], encoded["yc"], lambdas, native, grid
    )
    objective_path = aux["kalbi_v6_refinement_objective_after_path_q"].float()
    objective_before_path = aux[
        "kalbi_v6_refinement_objective_before_path_q"
    ].float()
    path_continuity_error = (
        objective_before_path[..., 1:] - objective_path[..., :-1]
    ).abs().max()
    per_iteration_delta_max = (
        objective_path - objective_before_path
    ).max()
    target = beta_true[:, None, :].expand_as(beta)
    supervised_loss = (beta - target).square().mean()
    supervised_loss.backward()

    checks = {
        "beta_finite": bool(torch.isfinite(beta).all()),
        "loss_finite": bool(torch.isfinite(supervised_loss)),
        "composite_descent_max": float(
            (objective1 - objective0).max().detach()
        ),
        "per_iteration_objective_delta_max": float(
            per_iteration_delta_max.detach()
        ),
        "objective_path_continuity_error_max": float(
            path_continuity_error.detach()
        ),
        "refinement_path_shape": list(
            aux["kalbi_v6_refinement_step_size_path_q"].shape
        ),
        "refinement_steps": KALBI_EXACT_PG_STEPS,
        "initial_ritz_safety": KALBI_EXACT_PG_RITZ_SAFETY,
        "lambda_gradient": float(lambdas.grad.float().norm()),
        "direct96_gradient": float(direct.grad.float().norm()),
        "v3_ab_head_gradient": gradient_norm(model.kalbi_ab_head.parameters()),
        "base_head_gradient": gradient_norm(model.base_head.parameters()),
        "majorization_ratio_max": float(
            aux["kalbi_v6_refinement_majorization_ratio_q"].max()
        ),
        "minimum_curvature_denominator": float(
            aux["kalbi_v6_refinement_min_curvature_denominator_q"].min()
        ),
        "refinement_change_rms": float(
            aux["kalbi_v6_refinement_change_rms_q"].mean()
        ),
        "trace_fallback_fraction": float(
            aux["kalbi_v6_refinement_trace_fallback_q"].mean()
        ),
        "d0_contract_error_max": float(
            aux["kalbi_v6_refinement_d0_contract_error_q"].max()
        ),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "model_version": MODEL_VERSION,
        "shape_smokes": [
            run_shape_smoke(model, rows=48, features=73, device=device),
            run_shape_smoke(model, rows=91, features=129, device=device),
        ],
        "training_objective_smoke": run_training_objective_smoke(
            model, device=device
        ),
    }
    positive_gradient_keys = (
        "lambda_gradient",
        "direct96_gradient",
        "v3_ab_head_gradient",
        "base_head_gradient",
    )
    if not checks["beta_finite"] or not checks["loss_finite"]:
        raise RuntimeError(f"nonfinite V6 preflight: {checks}")
    if checks["composite_descent_max"] > 2e-5:
        raise RuntimeError(f"V6 composite objective did not descend: {checks}")
    if checks["per_iteration_objective_delta_max"] > 2e-5:
        raise RuntimeError(f"V6 iteration objective did not descend: {checks}")
    if checks["objective_path_continuity_error_max"] > 2e-5:
        raise RuntimeError(f"V6 objective path is discontinuous: {checks}")
    if checks["refinement_path_shape"] != [batch, queries, 10]:
        raise RuntimeError(f"V6 refinement path shape drift: {checks}")
    if checks["refinement_steps"] != 10 or checks["initial_ritz_safety"] != 0.5:
        raise RuntimeError(f"V6 eta4x/T10 constants drifted: {checks}")
    if checks["majorization_ratio_max"] > 0.951:
        raise RuntimeError(f"V6 data majorization contract failed: {checks}")
    if checks["minimum_curvature_denominator"] < 0.099:
        raise RuntimeError(f"V6 curvature contract failed: {checks}")
    if checks["d0_contract_error_max"] > 2e-6:
        raise RuntimeError(f"V6 Direct96 d0 contract failed: {checks}")
    if any(checks[key] <= 0.0 for key in positive_gradient_keys):
        raise RuntimeError(f"missing V6 gradient path: {checks}")
    if checks["training_objective_smoke"]["model_gradient"] <= 0.0:
        raise RuntimeError(f"V6 trainer objective lost model gradients: {checks}")
    checks["selector_v20"] = run_selector_v20_smoke(model, device=device)
    print(json.dumps(checks, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
