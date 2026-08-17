#!/usr/bin/env python3
"""Audit direct96 V11 gradients through an actual frozen V58 decoder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from penalty_relu_mlp_pwl_v33 import make_penalty_features_pwl
from train_fpk import _load_yaml

from .model import build_model
from .selector_v10 import (
    lambda_from_effective_ratio_v56,
    selector_beta_with_inactive_ste,
    summarize_encoded,
)
from .selector_v11_direct import DirectPenaltyMLPQuerySelectorV11
from .train_joint_v11_direct import selector_path_features


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_tasks(
    path: Path | list[Path], task_indices: list[int], seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    paths = [path] if isinstance(path, Path) else list(path)
    X_parts, y_parts = [], []
    for source in paths:
        with np.load(source, allow_pickle=False) as payload:
            X_parts.append(np.asarray(payload["X"][task_indices], dtype=np.float32))
            y_parts.append(np.asarray(payload["y"][task_indices], dtype=np.float32))
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    generator = np.random.default_rng(seed)
    train_x, train_y, test_x, test_y = [], [], [], []
    for task_x, task_y in zip(X, y, strict=True):
        order = generator.permutation(task_x.shape[0])
        train, test = order[:100], order[100:]
        tx, vx = task_x[train].copy(), task_x[test].copy()
        ty, vy = task_y[train].copy(), task_y[test].copy()
        mean = tx.mean(0, dtype=np.float64).astype(np.float32)
        scale = tx.std(0, dtype=np.float64).astype(np.float32)
        constant = scale < 1e-6
        scale[constant] = 1.0
        tx, vx = (tx - mean) / scale, (vx - mean) / scale
        tx[:, constant], vx[:, constant] = 0.0, 0.0
        ymean = np.float32(ty.mean(dtype=np.float64))
        yscale = np.float32(ty.std(dtype=np.float64))
        ty, vy = (ty - ymean) / yscale, (vy - ymean) / yscale
        train_x.append(tx); train_y.append(ty); test_x.append(vx); test_y.append(vy)
    return tuple(
        torch.from_numpy(np.stack(values))
        for values in (train_x, train_y, test_x, test_y)
    )


@torch.no_grad()
def migrate_v10_shared_state(
    selector: DirectPenaltyMLPQuerySelectorV11, v10_state: dict[str, torch.Tensor]
) -> dict[str, int]:
    current = selector.state_dict()
    copied = 0
    for key, value in v10_state.items():
        if key.startswith("penalty_parameter_head."):
            continue
        if key in current and current[key].shape == value.shape:
            current[key] = value
            copied += 1
    selector.load_state_dict(current, strict=True)

    prefix = "penalty_parameter_head."
    reference_b = v10_state[prefix + "reference_b"].float()
    reference_weights = v10_state[prefix + "reference_weights"].float()
    basis = v10_state[prefix + "parameter_basis"].float()
    head = selector.penalty_parameter_head
    mean = torch.cat([reference_b.reshape(1), reference_weights])
    weight_std = basis.square().sum(dim=-1).sqrt().clamp_min(1e-8)
    mass_grid = head.penalty_grid[: reference_weights.numel()]
    mass_coeff = 0.5 * (6.0 - mass_grid).clamp_min(0.0).square()
    b_basis = -(mass_coeff[:, None] * basis).sum(dim=0) / 6.0
    b_std = b_basis.square().sum().sqrt().clamp_min(1e-8)
    std = torch.cat([b_std.reshape(1), weight_std])
    head.parameter_mean.copy_(mean)
    head.parameter_std.copy_(std)
    head.reference_b.copy_(reference_b)
    head.reference_weights.copy_(reference_weights)
    head.projection.bias.copy_(mean)
    head.cache_size.copy_(v10_state[prefix + "cache_size"])
    head.cache_parameter_std_min.copy_(std.amin())
    head.cache_parameter_std_max.copy_(std.amax())
    head.calibrated.fill_(True)
    return {"shared_state_keys_copied": copied, "v11_state_keys": len(current)}


def stats(value: torch.Tensor | None) -> dict[str, float | int | bool]:
    if value is None:
        return {"present": False, "finite": False, "norm": 0.0, "nonzero": 0}
    flat = value.detach().float().reshape(-1)
    return {
        "present": True,
        "finite": bool(torch.isfinite(flat).all()),
        "norm": float(flat.norm()),
        "rms": float(flat.square().mean().sqrt()),
        "nonzero": int((flat != 0.0).sum()),
        "elements": int(flat.numel()),
    }


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.norm() * right.norm()
    if float(denominator) == 0.0:
        return float("nan")
    return float(torch.dot(left, right) / denominator)


def flatten_parameter_gradients(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    hard_gradients: dict[str, torch.Tensor | None],
    prefixes: tuple[str, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    hard_values, ste_values = [], []
    for name, parameter in named_parameters:
        if prefixes is not None and not name.startswith(prefixes):
            continue
        hard = hard_gradients[name]
        ste = parameter.grad
        hard_values.append(
            torch.zeros_like(parameter).reshape(-1)
            if hard is None else hard.detach().reshape(-1)
        )
        ste_values.append(
            torch.zeros_like(parameter).reshape(-1)
            if ste is None else ste.detach().reshape(-1)
        )
    if not hard_values:
        raise RuntimeError(f"empty selector gradient group: {prefixes}")
    return torch.cat(hard_values).float(), torch.cat(ste_values).float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--additional-data", type=Path, nargs="*", default=[])
    parser.add_argument("--penalty-cache", type=Path)
    parser.add_argument("--path-chunk", type=int, default=25)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202607162306)
    parser.add_argument("--allow-partial-direct", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = _load_yaml(args.config)
    model = build_model(config, device)
    load = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    selector = DirectPenaltyMLPQuerySelectorV11(
        ds_feat_dim=9,
        t_feat_dim=32 * 33 // 2,
        krylov_dim=32,
        shape_dim=196,
        path_feat_dim=150,
        hidden=64,
        depth=3,
        lambda_min_ratio=1e-6,
        lambda_max_ratio=1.0,
        initial_lambda_ratio=0.1,
    )
    selector_state = checkpoint["selector_state_dict"]
    if "penalty_parameter_head.parameter_basis" in selector_state:
        migration = migrate_v10_shared_state(selector, selector_state)
    else:
        selector_load = selector.load_state_dict(selector_state, strict=True)
        migration = {
            "source": "native_v11_strict",
            "selector_state_keys": len(selector_state),
            "strict_missing": list(selector_load.missing_keys),
            "strict_unexpected": list(selector_load.unexpected_keys),
        }
    selector.to(device).train()

    data_paths = [args.data, *args.additional_data]
    X_train, y_train, X_test, y_test = split_tasks(
        data_paths, [0, 1, 2, 3], args.seed + 1
    )
    X_train, y_train, X_test, y_test = (
        value.to(device) for value in (X_train, y_train, X_test, y_test)
    )
    with torch.no_grad():
        encoded = model.encode_dataset(model.precompute(X_train, y_train))
        ds_feat, t_feat = summarize_encoded(encoded, 32 * 33 // 2)
    if args.penalty_cache is None:
        path_feat = torch.zeros(4, 150, device=device)
        path_features_source = "zero_audit_control"
    else:
        penalty_payload = torch.load(
            args.penalty_cache, map_location="cpu", weights_only=False
        )
        penalty_segments = penalty_payload["segments"]
        fixed_indices = (0, 49_999, 99_999)
        fixed_segments = [penalty_segments[index] for index in fixed_indices]
        fixed_encoded = make_penalty_features_pwl(
            [[segments] for segments in fixed_segments],
            device="cpu",
            G=96,
            t_max=6.0,
            K=96,
        )
        fixed_shape = fixed_encoded["shape_features"].squeeze(1)[:, :-1].to(device)
        fixed_d0 = fixed_encoded["d0"].squeeze(1).to(device)
        path_ratios = torch.exp(
            torch.linspace(
                float(np.log(1e-6)),
                float(np.log(1.0)),
                50,
                device=device,
            )
        )
        path_feat = selector_path_features(
            model,
            encoded,
            X_train,
            y_train,
            fixed_shape,
            fixed_d0,
            path_ratios,
            args.path_chunk,
        )
        path_features_source = "production_three_penalty_150_path"
    query = selector(
        ds_feat,
        t_feat,
        encoded["lambda_max"].float(),
        path_feat,
        U=encoded.get("U"),
        y_context=y_train,
    )
    native = query["penalty_shape_features"]
    d0 = query["penalty_d0"]
    lambdas = query["lambdas"]
    for value in (native, d0, lambdas):
        value.retain_grad()
    natural_aux = model.decode_queries(
        encoded,
        lambdas=lambdas,
        lambda_max=encoded["lambda_max"].float()[:, None],
        penalty_shape_features=native,
        penalty_d0=d0,
        return_aux=True,
    )
    support_override = {
        "selected_indices": natural_aux[
            "v56fast_isis_selected_indices"
        ].detach(),
        "stage_active": natural_aux["v56fast_isis_stage_active"].detach(),
    }
    aux = model.decode_queries(
        encoded,
        lambdas=lambdas,
        lambda_max=encoded["lambda_max"].float()[:, None],
        penalty_shape_features=native,
        penalty_d0=d0,
        return_aux=True,
        isis_support_override=support_override,
    )
    hard_beta = aux["beta_q"][:, 0].float()
    hard_prediction = torch.einsum("bnp,bp->bn", X_test.float(), hard_beta)
    hard_task_mse = (hard_prediction - y_test.float()).square().mean(dim=-1)
    hard_loss = torch.log1p(0.5 * hard_task_mse).mean()
    raw_internal = selector.penalty_parameter_head._last_latent_raw
    parameter_internal = selector.penalty_parameter_head._last_parameter_proposal
    named_selector_parameters = [
        (name, parameter)
        for name, parameter in selector.named_parameters()
        if parameter.requires_grad
    ]
    hard_gradient_values = torch.autograd.grad(
        hard_loss,
        [raw_internal, *[parameter for _, parameter in named_selector_parameters]],
        retain_graph=True,
        allow_unused=True,
    )
    hard_raw_grad = hard_gradient_values[0]
    hard_parameter_gradients = {
        name: gradient
        for (name, _), gradient in zip(
            named_selector_parameters, hard_gradient_values[1:], strict=True
        )
    }

    beta = selector_beta_with_inactive_ste(aux)[:, 0].float()
    prediction = torch.einsum("bnp,bp->bn", X_test.float(), beta)
    task_mse = (prediction - y_test.float()).square().mean(dim=-1)
    loss = torch.log1p(0.5 * task_mse).mean()
    loss.backward()

    gradient_alignment = {}
    for group, prefixes in (
        ("all_selector", None),
        ("penalty_head", ("penalty_parameter_head.",)),
        ("penalty_trunk", ("penalty_net.",)),
        ("lambda_head", ("lambda_head.",)),
        ("lambda_trunk", ("lambda_net.",)),
    ):
        hard_group, ste_group = flatten_parameter_gradients(
            named_selector_parameters, hard_parameter_gradients, prefixes
        )
        gradient_alignment[group] = {
            "hard_norm": float(hard_group.norm()),
            "ste_norm": float(ste_group.norm()),
            "ste_to_hard_norm_ratio": float(
                ste_group.norm() / hard_group.norm().clamp_min(1e-12)
            ),
            "hard_ste_cosine": cosine(hard_group, ste_group),
            "hard_nonzero_fraction": float((hard_group != 0.0).float().mean()),
            "ste_nonzero_fraction": float((ste_group != 0.0).float().mean()),
        }

    raw_grad = selector.penalty_parameter_head._last_latent_raw.grad
    parameter_grad = selector.penalty_parameter_head._last_parameter_proposal.grad
    proposal_value = (
        selector.penalty_parameter_head._last_parameter_proposal.detach().float()
    )
    proposal_grid = selector.penalty_parameter_head.penalty_grid.to(proposal_value)
    proposal_knots = proposal_grid[:-1]
    proposal_basis = (
        proposal_grid[:, None] - proposal_knots[None, :]
    ).clamp_min(0.0)
    proposal_drho = proposal_value[..., :1] + torch.einsum(
        "...k,tk->...t", proposal_value[..., 1:], proposal_basis
    )
    positivity_floor = float(selector.penalty_parameter_head.derivative_floor)
    positivity_beta = float(selector.penalty_parameter_head.positivity_beta)
    softplus_jacobian = torch.sigmoid(
        positivity_beta * (proposal_drho - positivity_floor)
    )
    raw_coordinate_activity = raw_grad.detach().abs().sum(dim=0) != 0.0
    parameter_coordinate_activity = (
        parameter_grad.detach().abs().sum(dim=0) != 0.0
    )
    raw_base = selector.penalty_parameter_head._last_latent_raw.detach()
    direction_generator = torch.Generator(device=device).manual_seed(args.seed + 99)
    direction = torch.randn(
        raw_base.shape, generator=direction_generator, device=device
    )
    direction = direction / direction.norm().clamp_min(1e-12)

    @torch.no_grad()
    def fixed_support_loss(raw_value: torch.Tensor) -> float:
        penalty = selector.penalty_parameter_head.decode_direct_parameters(raw_value)
        local_native = penalty["shape_features"][:, None, :]
        local_d0 = penalty["drho"][:, :1]
        local_lambda = lambda_from_effective_ratio_v56(
            query["lambda_ratio"].detach(),
            encoded["lambda_max"].float(),
            local_d0[:, 0],
        )["lambdas"][:, None]
        local_aux = model.decode_queries(
            encoded,
            lambdas=local_lambda,
            lambda_max=encoded["lambda_max"].float()[:, None],
            penalty_shape_features=local_native,
            penalty_d0=local_d0,
            return_aux=True,
            isis_support_override=support_override,
        )
        local_beta = selector_beta_with_inactive_ste(local_aux)[:, 0].float()
        local_prediction = torch.einsum(
            "bnp,bp->bn", X_test.float(), local_beta
        )
        local_mse = (local_prediction - y_test.float()).square().mean(dim=-1)
        return float(torch.log1p(0.5 * local_mse).mean())

    predicted_directional = float((hard_raw_grad.detach() * direction).sum())
    finite_differences = []
    for epsilon in (1e-2, 3e-3):
        plus = fixed_support_loss(raw_base + epsilon * direction)
        minus = fixed_support_loss(raw_base - epsilon * direction)
        measured = (plus - minus) / (2.0 * epsilon)
        finite_differences.append(
            {
                "epsilon": epsilon,
                "predicted": predicted_directional,
                "measured": measured,
                "relative_error": abs(predicted_directional - measured)
                / max(abs(measured), 1e-12),
                "same_sign": bool(predicted_directional * measured > 0.0),
            }
        )
    result = {
        "status": "PASS",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "data": [str(path.resolve()) for path in data_paths],
        "task_count": int(X_train.shape[0]),
        "path_features_source": path_features_source,
        "device": str(device),
        "model_strict_missing": load.missing_keys,
        "model_strict_unexpected": load.unexpected_keys,
        "migration": migration,
        "selector_loss": float(loss.detach()),
        "hard_selector_loss": float(hard_loss.detach()),
        "task_mse": task_mse.detach().cpu().tolist(),
        "native196_gradient": stats(native.grad),
        "d0_gradient": stats(d0.grad),
        "lambda_gradient": stats(lambdas.grad),
        "direct96_raw_gradient": stats(raw_grad),
        "direct96_parameter_gradient": stats(parameter_grad),
        "direct96_hard_raw_gradient": stats(hard_raw_grad),
        "direct96_hard_ste_gradient_cosine": float(
            torch.nn.functional.cosine_similarity(
                hard_raw_grad.detach().reshape(1, -1),
                raw_grad.detach().reshape(1, -1),
            )[0]
        ),
        "selector_parameter_hard_ste_alignment": gradient_alignment,
        "direct96_active_coordinates": int(
            raw_coordinate_activity.sum()
        ),
        "direct96_inactive_coordinate_indices": torch.nonzero(
            ~raw_coordinate_activity, as_tuple=False
        ).flatten().cpu().tolist(),
        "direct96_parameter_active_coordinates": int(
            parameter_coordinate_activity.sum()
        ),
        "direct96_parameter_inactive_coordinate_indices": torch.nonzero(
            ~parameter_coordinate_activity, as_tuple=False
        ).flatten().cpu().tolist(),
        "proposal_drho_min": float(proposal_drho.min()),
        "proposal_drho_max": float(proposal_drho.max()),
        "proposal_drho_below_floor_count": int(
            (proposal_drho <= positivity_floor).sum()
        ),
        "softplus_jacobian_zero_count": int(
            (softplus_jacobian == 0.0).sum()
        ),
        "softplus_jacobian_min": float(softplus_jacobian.min()),
        "softplus_jacobian_min_nonzero": float(
            softplus_jacobian[softplus_jacobian > 0.0].min()
        ),
        "direct96_fixed_support_finite_difference": finite_differences,
        "penalty_head_gradient": stats(
            selector.penalty_parameter_head.projection.weight.grad
        ),
        "lambda_head_gradient": stats(selector.lambda_head.weight.grad),
        "rho6_error_max": float(query["penalty_rho6_error"].detach().max()),
        "minimum_drho": float(query["penalty_min_drho"].detach().min()),
        "rho_task_std_mean": float(
            query["penalty_rho"].detach().std(0, unbiased=False).mean()
        ),
        "drho_task_std_mean": float(
            query["penalty_drho"].detach().std(0, unbiased=False).mean()
        ),
        "effective_ratio_min": float(query["lambda_ratio"].detach().min()),
        "effective_ratio_max": float(query["lambda_ratio"].detach().max()),
        "direct_parameter_count": 96,
        "pca_used": False,
    }
    required = (
        result["native196_gradient"],
        result["d0_gradient"],
        result["lambda_gradient"],
        result["direct96_raw_gradient"],
    )
    if not all(item["present"] and item["finite"] and item["norm"] > 0 for item in required):
        raise RuntimeError(f"V11 gradient contract failed: {result}")
    if (
        result["direct96_active_coordinates"] != 96
        and not args.allow_partial_direct
    ):
        raise RuntimeError(f"not all direct coordinates are active: {result}")
    if result["rho6_error_max"] > 2e-6 or result["minimum_drho"] <= 0.0:
        raise RuntimeError(f"V11 validity contract failed: {result}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
