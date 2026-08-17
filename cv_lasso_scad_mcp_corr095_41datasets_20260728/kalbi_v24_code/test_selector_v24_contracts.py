#!/usr/bin/env python3
"""Contract tests for the V24 cached-trajectory selector."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from .model import build_model
from .selector_v23_independent_rawlambda import (
    DirectPenaltyMLPQuerySelectorV23IndependentRawLambda,
)
from .selector_v24_additive512_trajectory import (
    DirectPenaltyMLPQuerySelectorV24Additive512,
    load_v23_state_into_v24,
    make_identity_v24_scaler_payload,
)
from train_fpk import _load_yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def standardized(
    batch: int, rows: int, features: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(batch, rows, features, device=device)
    y = torch.randn(batch, rows, device=device)
    x = (x - x.mean(1, keepdim=True)) / x.std(
        1, keepdim=True, correction=0
    ).clamp_min(1e-6)
    y = (y - y.mean(1, keepdim=True)) / y.std(
        1, keepdim=True, correction=0
    ).clamp_min(1e-6)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scaler-cache", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if sha256(args.checkpoint) != args.checkpoint_sha256:
        raise RuntimeError("test checkpoint hash mismatch")
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    torch.manual_seed(202607270927)
    device = torch.device(args.device)
    beta = build_model(_load_yaml(str(args.config)), device)
    beta.load_state_dict(checkpoint["model_state_dict"], strict=True)
    beta.eval()
    for parameter in beta.parameters():
        parameter.requires_grad_(False)
    v23 = DirectPenaltyMLPQuerySelectorV23IndependentRawLambda(
        hidden=512,
        derivative_floor=1e-6,
        squareplus_scale=0.02,
        direct_head_init_std=1e-3,
    )
    v23.load_state_dict(checkpoint["selector_state_dict"], strict=True)
    v24 = DirectPenaltyMLPQuerySelectorV24Additive512(
        hidden=512,
        derivative_floor=1e-6,
        squareplus_scale=0.02,
        direct_head_init_std=1e-3,
    )
    import_report = load_v23_state_into_v24(
        v24, checkpoint["selector_state_dict"]
    )
    payload = (
        torch.load(args.scaler_cache, map_location="cpu", weights_only=False)
        if args.scaler_cache is not None
        else make_identity_v24_scaler_payload()
    )
    v24.load_v24_scaler_cache(payload)
    with torch.no_grad():
        v24.global_gate.zero_()
        v24.trajectory_gate.zero_()
        v24.proposal_gate.zero_()
        v24.initial_norm_gate.zero_()
        for layer in v24.residual_layers:
            layer.gate.zero_()
    v23.to(device).eval()
    v24.to(device).eval()

    x, y = standardized(2, 96, 160, device)
    encoded = beta.encode_dataset(beta.precompute(x, y))
    with torch.no_grad():
        old = v23(
            x, y, encoded["lambda_max"].float(),
            return_representation=True,
        )
        new = v24(
            x, y, encoded["lambda_max"].float(),
            solver_context=encoded,
            return_representation=True,
        )
    lambda_error = (old["raw_lambda"] - new["raw_lambda"]).abs().max()
    penalty_error = (
        old["penalty_shape_features"] - new["penalty_shape_features"]
    ).abs().max()
    if float(lambda_error) > 2e-6 or float(penalty_error) > 2e-6:
        raise RuntimeError("V23/V24 neutral proposal equivalence failed")
    with torch.no_grad():
        old_beta = beta.decode_queries(
            encoded,
            lambdas=old["lambdas"],
            lambda_max=encoded["lambda_max"].float()[:, None],
            penalty_shape_features=old["penalty_shape_features"],
            penalty_d0=old["penalty_d0"],
            return_aux=True,
        )["beta_q"]
        new_beta = beta.decode_queries(
            encoded,
            lambdas=new["lambdas"],
            lambda_max=encoded["lambda_max"].float()[:, None],
            penalty_shape_features=new["penalty_shape_features"],
            penalty_d0=new["penalty_d0"],
            return_aux=True,
        )["beta_q"]
    beta_error = (old_beta - new_beta).abs().max()
    if float(beta_error) > 2e-5:
        raise RuntimeError("V23/V24 neutral final-beta equivalence failed")
    if new["v24_task_code"].shape != (2, 512):
        raise RuntimeError("V24 task-code shape drift")
    if new["v24_solver_trajectory"].shape != (2, 3, 14):
        raise RuntimeError("V24 trajectory shape drift")
    if new["v24_trajectory_code"].shape != (2, 512):
        raise RuntimeError("V24 trajectory-code shape drift")
    if new["v24_proposal_code"].shape != (2, 512):
        raise RuntimeError("V24 proposal-code shape drift")
    residual_linears = sum(
        isinstance(module, torch.nn.Linear)
        and module.in_features == 512
        and module.out_features == 512
        for module in v24.residual_layers.modules()
    )
    if residual_linears != 4:
        raise RuntimeError("V24 does not contain exactly four residual linears")
    trajectory = new["v24_solver_trajectory"]
    if not bool(torch.isfinite(trajectory).all()):
        raise RuntimeError("V24 trajectory is nonfinite")
    if not bool(
        ((trajectory[..., 5:8] >= 0.0) & (trajectory[..., 5:8] <= 1.0)).all()
    ):
        raise RuntimeError("V24 branch fractions left [0,1]")
    if float(new["v24_probe_gamma_contract_error"].max()) > 1e-7:
        raise RuntimeError("V24 gamma/eta raw-lambda contract failed")
    if float(new["penalty_rho"][:, 0].abs().max()) != 0.0:
        raise RuntimeError("V24 rho(0) is not exact zero")
    if float((new["penalty_rho"][:, -1] - 1.0).abs().max()) > 2e-6:
        raise RuntimeError("V24 rho(6) normalization failed")
    if float(new["penalty_drho"].min()) <= 0.0:
        raise RuntimeError("V24 penalty derivative is not positive")

    # Row and feature permutations may change only floating-point reduction order.
    row_perm = torch.randperm(x.shape[1], device=device)
    feature_perm = torch.randperm(x.shape[2], device=device)
    with torch.no_grad():
        encoded_row = beta.encode_dataset(
            beta.precompute(x[:, row_perm], y[:, row_perm])
        )
        row = v24(
            x[:, row_perm],
            y[:, row_perm],
            encoded_row["lambda_max"].float(),
            solver_context=encoded_row,
            return_representation=True,
        )
        encoded_feature = beta.encode_dataset(
            beta.precompute(x[:, :, feature_perm], y)
        )
        feature = v24(
            x[:, :, feature_perm],
            y,
            encoded_feature["lambda_max"].float(),
            solver_context=encoded_feature,
            return_representation=True,
        )
    row_error = (new["raw_lambda"] - row["raw_lambda"]).abs().max()
    feature_error = (new["raw_lambda"] - feature["raw_lambda"]).abs().max()
    if float(row_error) > 2e-4 or float(feature_error) > 2e-4:
        raise RuntimeError(
            "V24 permutation contract failed: "
            f"row={float(row_error)}, feature={float(feature_error)}"
        )

    # Strict state reload includes the fixed scaler buffers.
    clone = DirectPenaltyMLPQuerySelectorV24Additive512(
        hidden=512,
        derivative_floor=1e-6,
        squareplus_scale=0.02,
        direct_head_init_std=1e-3,
    ).to(device)
    clone.load_state_dict(v24.state_dict(), strict=True)
    with torch.no_grad():
        cloned = clone(
            x, y, encoded["lambda_max"].float(),
            solver_context=encoded,
            return_representation=True,
        )
    reload_error = (new["raw_lambda"] - cloned["raw_lambda"]).abs().max()
    if float(reload_error) > 1e-7:
        raise RuntimeError("V24 strict reload changed output")

    # Exact-zero correction initialization: heads learn on backward one; the
    # newly connected branches receive gradients after that bootstrap update.
    with torch.no_grad():
        v24.global_gate.fill_(1e-3)
        v24.trajectory_gate.fill_(1e-3)
        v24.proposal_gate.fill_(1e-3)
        for layer in v24.residual_layers:
            layer.gate.fill_(1e-2)
    v24.train()
    for parameter in v24.parameters():
        parameter.grad = None
    encoded_one = beta.encode_dataset(beta.precompute(x[:1], y[:1]))
    first = v24(
        x[:1],
        y[:1],
        encoded_one["lambda_max"].float(),
        solver_context=encoded_one,
        return_representation=True,
    )
    first_beta = beta.decode_queries(
        encoded_one,
        lambdas=first["lambdas"],
        lambda_max=encoded_one["lambda_max"].float()[:, None],
        penalty_shape_features=first["penalty_shape_features"],
        penalty_d0=first["penalty_d0"],
        return_aux=True,
    )["beta_q"][:, 0]
    first_loss = (
        torch.einsum("bnp,bp->bn", x[:1], first_beta) - y[:1]
    ).square().mean()
    first_loss.backward()
    first_correction_gradients = {
        "lambda": float(v24.delta_loglambda_head.weight.grad.norm()),
        "penalty": float(v24.delta_penalty_latent_head.weight.grad.norm()),
    }
    if min(first_correction_gradients.values()) <= 0.0:
        raise RuntimeError("V24 first-backward correction gradient is zero")
    with torch.no_grad():
        for module in (
            v24.delta_loglambda_head,
            v24.delta_penalty_latent_head,
        ):
            for parameter in module.parameters():
                parameter.add_(parameter.grad, alpha=-1e-4)
    for parameter in v24.parameters():
        parameter.grad = None
    second = v24(
        x[:1],
        y[:1],
        encoded_one["lambda_max"].float(),
        solver_context=encoded_one,
        return_representation=True,
    )
    second_beta = beta.decode_queries(
        encoded_one,
        lambdas=second["lambdas"],
        lambda_max=encoded_one["lambda_max"].float()[:, None],
        penalty_shape_features=second["penalty_shape_features"],
        penalty_d0=second["penalty_d0"],
        return_aux=True,
    )["beta_q"][:, 0]
    second_loss = (
        torch.einsum("bnp,bp->bn", x[:1], second_beta) - y[:1]
    ).square().mean()
    second_loss.backward()
    second_branch_gradients = {
        "global": float(v24.global_projection[0].weight.grad.norm()),
        "trajectory": float(v24.trajectory_projection[0].weight.grad.norm()),
        "proposal": float(v24.proposal_projection[0].weight.grad.norm()),
        "residual_0": float(v24.residual_layers[0].linear.weight.grad.norm()),
        "residual_1": float(v24.residual_layers[1].linear.weight.grad.norm()),
        "residual_2": float(v24.residual_layers[2].linear.weight.grad.norm()),
        "residual_3": float(v24.residual_layers[3].linear.weight.grad.norm()),
    }
    if min(second_branch_gradients.values()) <= 0.0:
        raise RuntimeError("V24 post-bootstrap branch gradient is zero")

    print(
        {
            "status": "PASS",
            "imported_v23_keys": import_report["imported_count"],
            "new_v24_keys": import_report["new_count"],
            "lambda_equivalence_max_abs": float(lambda_error),
            "penalty_equivalence_max_abs": float(penalty_error),
            "beta_equivalence_max_abs": float(beta_error),
            "row_permutation_lambda_error": float(row_error),
            "feature_permutation_lambda_error": float(feature_error),
            "strict_reload_lambda_error": float(reload_error),
            "gamma_contract_error": float(
                new["v24_probe_gamma_contract_error"].max()
            ),
            "first_correction_gradients": first_correction_gradients,
            "second_branch_gradients": second_branch_gradients,
        }
    )


if __name__ == "__main__":
    main()
