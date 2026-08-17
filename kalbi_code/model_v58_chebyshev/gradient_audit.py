#!/usr/bin/env python3
"""Audit V58 hard-forward compatibility and fixed-support gradients.

The script can run with a V57 checkpoint or with a deterministic fresh state.
It compares V57 and V58 on the same N-by-P task, freezes the natural V58 ISIS
support plan for a second pass, and measures central finite differences of a
held-out MSE with respect to lambda plus all native196 penalty coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cosine(left: list[float], right: list[float]) -> float:
    a = torch.tensor(left, dtype=torch.float64)
    b = torch.tensor(right, dtype=torch.float64)
    denominator = a.norm() * b.norm()
    if float(denominator) == 0.0:
        return float("nan")
    return float(torch.dot(a, b) / denominator)


def relative_error(predicted: list[float], measured: list[float]) -> float:
    a = torch.tensor(predicted, dtype=torch.float64)
    b = torch.tensor(measured, dtype=torch.float64)
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def split_task(
    path: Path, task_index: int, train_rows: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with np.load(path, allow_pickle=False) as payload:
        raw_X = np.asarray(payload["X"][task_index], dtype=np.float32)
        raw_y = np.asarray(payload["y"][task_index], dtype=np.float32)
    if raw_X.ndim != 2 or raw_y.shape != raw_X.shape[:1]:
        raise RuntimeError(f"invalid task shapes X={raw_X.shape} y={raw_y.shape}")
    if not 1 <= train_rows < raw_X.shape[0]:
        raise ValueError("train_rows must leave a nonempty held-out split")
    permutation = np.random.default_rng(seed).permutation(raw_X.shape[0])
    train_index = permutation[:train_rows]
    test_index = permutation[train_rows:]
    x_train = raw_X[train_index]
    x_test = raw_X[test_index]
    y_train = raw_y[train_index]
    y_test = raw_y[test_index]
    x_mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    x_scale = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    constant = x_scale < 1e-6
    x_scale = np.where(constant, 1.0, x_scale)
    x_train = (x_train - x_mean) / x_scale
    x_test = (x_test - x_mean) / x_scale
    x_train[:, constant] = 0.0
    x_test[:, constant] = 0.0
    y_mean = np.float32(y_train.mean(dtype=np.float64))
    y_scale = np.float32(y_train.std(dtype=np.float64))
    if float(y_scale) < 1e-6:
        raise RuntimeError("training response is constant")
    y_train = (y_train - y_mean) / y_scale
    y_test = (y_test - y_mean) / y_scale
    return tuple(
        torch.from_numpy(value[None])
        for value in (x_train, y_train, x_test, y_test)
    )


def make_native196(grid: torch.Tensor) -> torch.Tensor:
    rho = grid
    drho = torch.ones_like(grid)
    pairs = torch.stack([rho, drho], dim=-1).reshape(-1)
    meta4 = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float32)
    return torch.cat([pairs, meta4]).view(1, 1, 196)


def heldout_mse(
    beta: torch.Tensor, X_test: torch.Tensor, y_test: torch.Tensor
) -> torch.Tensor:
    prediction = torch.einsum("bnp,bqp->bqn", X_test.float(), beta.float())
    return (prediction - y_test[:, None, :].float()).square().mean()


def support_plan(aux: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        "selected_indices": aux["v56fast_isis_selected_indices"].detach(),
        "stage_active": aux["v56fast_isis_stage_active"].detach(),
    }


def support_equal(aux: dict[str, Any], plan: dict[str, torch.Tensor]) -> bool:
    return bool(
        torch.equal(aux["v56fast_isis_selected_indices"], plan["selected_indices"])
        and torch.equal(aux["v56fast_isis_stage_active"], plan["stage_active"])
    )


def decode(
    model: torch.nn.Module,
    encoded: dict[str, Any],
    lambdas: torch.Tensor,
    native: torch.Tensor,
    override: dict[str, torch.Tensor] | None,
    return_aux: bool,
):
    return model.decode_queries(
        encoded,
        lambdas=lambdas,
        lambda_max=encoded["lambda_max"].float()[:, None],
        penalty_shape_features=native,
        penalty_d0=native[..., 192],
        return_aux=return_aux,
        **({"isis_support_override": override} if override is not None else {}),
    )


def gradient(
    model: torch.nn.Module,
    encoded: dict[str, Any],
    base_lambda: torch.Tensor,
    base_native: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    override: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    lambdas = base_lambda.detach().clone().requires_grad_(True)
    native = base_native.detach().clone().requires_grad_(True)
    beta = decode(model, encoded, lambdas, native, override, False)
    loss = heldout_mse(beta, X_test, y_test)
    loss.backward()
    if lambdas.grad is None or native.grad is None:
        raise RuntimeError("lambda/native196 gradient is missing")
    return lambdas.grad.detach(), native.grad.detach(), float(loss.detach())


@torch.no_grad()
def loss_and_aux(
    model: torch.nn.Module,
    encoded: dict[str, Any],
    lambdas: torch.Tensor,
    native: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    override: dict[str, torch.Tensor] | None,
) -> tuple[float, dict[str, Any]]:
    aux = decode(model, encoded, lambdas, native, override, True)
    return float(heldout_mse(aux["beta_q"], X_test, y_test)), aux


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--v57-package-root", type=Path, required=True)
    parser.add_argument("--v58-package-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--train-rows", type=int, default=160)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=202607161458)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    # The immutable V57 package must win for its versioned compatibility
    # modules, while the standalone V58 package still resolves from its
    # parent/project root.
    for root in (
        args.project_root.resolve(),
        args.v58_package_root.resolve(),
        args.v57_package_root.resolve(),
    ):
        sys.path.insert(0, str(root))
    if (args.v57_package_root / "v57_spike_support.py").is_file():
        compatibility_modules = {
            name: importlib.import_module(name)
            for name in (
                "model_fpk_v55fast",
                "model_fpk_v56fast",
                "model_fpk_v57fast",
            )
        }
        expected_root = args.v57_package_root.resolve()
        for name, module in compatibility_modules.items():
            module_path = Path(module.__file__).resolve()
            if module_path.parent != expected_root:
                raise RuntimeError(
                    f"{name} resolved outside immutable V57 package: {module_path}"
                )
        from v57_spike_support import build_model as build_v57  # noqa: PLC0415
    else:
        compatibility_modules = {}
        from model_v57.model import build_model as build_v57  # noqa: PLC0415
    from model_v58_chebyshev.model import (  # noqa: PLC0415
        LOCAL_ISIS_GRADIENT_CONTRACT,
        MODEL_VERSION,
        build_model as build_v58,
    )
    from model_v58_chebyshev.utils import make_penalty_grid  # noqa: PLC0415

    config = yaml.safe_load(args.config.read_text()) if args.config else {}
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    v57 = build_v57(config, device)
    v58 = build_v58(config, device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = checkpoint["model_state_dict"]
        checkpoint_sha = sha256(args.checkpoint)
    else:
        state = v57.state_dict()
        checkpoint = None
        checkpoint_sha = None
    load57 = v57.load_state_dict(state, strict=True)
    load58 = v58.load_state_dict(state, strict=True)
    for model in (v57, v58):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    schema57 = {name: (tuple(value.shape), str(value.dtype)) for name, value in v57.state_dict().items()}
    schema58 = {name: (tuple(value.shape), str(value.dtype)) for name, value in v58.state_dict().items()}
    X_train, y_train, X_test, y_test = split_task(
        args.data, args.task_index, args.train_rows, args.seed + 17
    )
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    with torch.no_grad():
        encoded57 = v57.encode_dataset(v57.precompute(X_train, y_train))
        encoded58 = v58.encode_dataset(v58.precompute(X_train, y_train))
    native = make_native196(make_penalty_grid()).to(device)
    lambdas = (encoded57["lambda_max"].float()[:, None] * 0.1).detach()

    loss57, aux57 = loss_and_aux(
        v57, encoded57, lambdas, native, X_test, y_test, None
    )
    loss58, aux58 = loss_and_aux(
        v58, encoded58, lambdas, native, X_test, y_test, None
    )
    plan = support_plan(aux58)
    fixed_loss58, fixed_aux58 = loss_and_aux(
        v58, encoded58, lambdas, native, X_test, y_test, plan
    )
    common_forward_equal = bool(torch.equal(aux57["beta_q"], aux58["beta_q"]))
    natural_fixed_equal = bool(torch.equal(aux58["beta_q"], fixed_aux58["beta_q"]))
    supports_equal = support_equal(aux57, plan) and support_equal(aux58, plan)
    common_tensor_keys = sorted(
        key
        for key in set(aux57) & set(aux58)
        if torch.is_tensor(aux57[key]) and torch.is_tensor(aux58[key])
    )
    common_tensor_mismatches = [
        key
        for key in common_tensor_keys
        if not torch.equal(aux57[key], aux58[key])
    ]
    encoded_tensor_keys = sorted(
        key
        for key in set(encoded57) & set(encoded58)
        if torch.is_tensor(encoded57[key]) and torch.is_tensor(encoded58[key])
    )
    encoded_tensor_mismatches = [
        key
        for key in encoded_tensor_keys
        if not torch.equal(encoded57[key], encoded58[key])
    ]

    grad57_lam, grad57_native, _ = gradient(
        v57, encoded57, lambdas, native, X_test, y_test, None
    )
    grad58_lam, grad58_native, _ = gradient(
        v58, encoded58, lambdas, native, X_test, y_test, None
    )
    grad58_fixed_lam, grad58_fixed_native, _ = gradient(
        v58, encoded58, lambdas, native, X_test, y_test, plan
    )

    lambda_scale = lambdas.detach().abs().clamp_min(1e-3)
    native_scale = native.detach().abs().clamp_min(0.1)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 101)
    directions: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(args.directions):
        d_lambda = torch.randn(lambdas.shape, generator=generator).to(device)
        d_native = torch.randn(native.shape, generator=generator).to(device)
        denominator = torch.cat([d_lambda.reshape(-1), d_native.reshape(-1)]).norm()
        directions.append((d_lambda / denominator, d_native / denominator))

    predicted57: list[float] = []
    predicted58: list[float] = []
    predicted58_fixed: list[float] = []
    fd57: list[float] = []
    fd58: list[float] = []
    fd58_fixed: list[float] = []
    support_change57 = 0
    support_change58 = 0
    epsilon = float(args.epsilon)
    for d_lambda, d_native in directions:
        lambda_step = lambda_scale * d_lambda
        native_step = native_scale * d_native
        predicted57.append(float((grad57_lam * lambda_step).sum() + (grad57_native * native_step).sum()))
        predicted58.append(float((grad58_lam * lambda_step).sum() + (grad58_native * native_step).sum()))
        predicted58_fixed.append(float((grad58_fixed_lam * lambda_step).sum() + (grad58_fixed_native * native_step).sum()))
        plus_lambda = lambdas + epsilon * lambda_step
        minus_lambda = lambdas - epsilon * lambda_step
        plus_native = native + epsilon * native_step
        minus_native = native - epsilon * native_step
        plus57, aux_plus57 = loss_and_aux(v57, encoded57, plus_lambda, plus_native, X_test, y_test, None)
        minus57, aux_minus57 = loss_and_aux(v57, encoded57, minus_lambda, minus_native, X_test, y_test, None)
        plus58, aux_plus58 = loss_and_aux(v58, encoded58, plus_lambda, plus_native, X_test, y_test, None)
        minus58, aux_minus58 = loss_and_aux(v58, encoded58, minus_lambda, minus_native, X_test, y_test, None)
        plus58_fixed, _ = loss_and_aux(v58, encoded58, plus_lambda, plus_native, X_test, y_test, plan)
        minus58_fixed, _ = loss_and_aux(v58, encoded58, minus_lambda, minus_native, X_test, y_test, plan)
        fd57.append((plus57 - minus57) / (2.0 * epsilon))
        fd58.append((plus58 - minus58) / (2.0 * epsilon))
        fd58_fixed.append((plus58_fixed - minus58_fixed) / (2.0 * epsilon))
        support_change57 += int(not support_equal(aux_plus57, plan)) + int(not support_equal(aux_minus57, plan))
        support_change58 += int(not support_equal(aux_plus58, plan)) + int(not support_equal(aux_minus58, plan))

    fixed_cosine = cosine(predicted58_fixed, fd58_fixed)
    natural_cosine = cosine(predicted58, fd58)
    legacy_cosine = cosine(predicted57, fd57)
    report = {
        "status": "PASS" if common_forward_equal and natural_fixed_equal and supports_equal and not common_tensor_mismatches and not encoded_tensor_mismatches and fixed_cosine > 0.9 else "FAIL",
        "model_version": MODEL_VERSION,
        "gradient_contract": LOCAL_ISIS_GRADIENT_CONTRACT,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": checkpoint.get("completed_steps") if checkpoint else None,
        "data": str(args.data.resolve()),
        "data_sha256": sha256(args.data),
        "task_index": args.task_index,
        "train_rows": args.train_rows,
        "test_rows": int(X_test.shape[1]),
        "shape": {"N": int(X_train.shape[1]), "P": int(X_train.shape[2])},
        "state": {
            "v57_keys": len(schema57),
            "v58_keys": len(schema58),
            "schema_identical": schema57 == schema58,
            "parameter_count_v57": sum(p.numel() for p in v57.parameters()),
            "parameter_count_v58": sum(p.numel() for p in v58.parameters()),
            "strict_load_v57": not load57.missing_keys and not load57.unexpected_keys,
            "strict_load_v58": not load58.missing_keys and not load58.unexpected_keys,
            "v57_compatibility_modules": {
                name: str(Path(module.__file__).resolve())
                for name, module in compatibility_modules.items()
            },
        },
        "forward": {
            "v57_v58_beta_bitwise_equal": common_forward_equal,
            "v57_v58_support_equal": supports_equal,
            "v58_natural_fixed_beta_bitwise_equal": natural_fixed_equal,
            "common_aux_tensor_count": len(common_tensor_keys),
            "common_aux_tensor_mismatches": common_tensor_mismatches,
            "encoded_tensor_count": len(encoded_tensor_keys),
            "encoded_tensor_mismatches": encoded_tensor_mismatches,
            "v57_loss": loss57,
            "v58_loss": loss58,
            "v58_fixed_loss": fixed_loss58,
            "active_stages": int(plan["stage_active"].sum()),
        },
        "gradients": {
            "v57_lambda_norm": float(grad57_lam.norm()),
            "v57_native196_norm": float(grad57_native.norm()),
            "v58_lambda_norm": float(grad58_lam.norm()),
            "v58_native196_norm": float(grad58_native.norm()),
            "v58_fixed_lambda_norm": float(grad58_fixed_lam.norm()),
            "v58_fixed_native196_norm": float(grad58_fixed_native.norm()),
            "v58_natural_fixed_lambda_max_abs": float((grad58_lam - grad58_fixed_lam).abs().max()),
            "v58_natural_fixed_native_max_abs": float((grad58_native - grad58_fixed_native).abs().max()),
            "all_v58_finite": bool(torch.isfinite(grad58_lam).all() and torch.isfinite(grad58_native).all()),
        },
        "directional_finite_difference": {
            "directions": args.directions,
            "epsilon": epsilon,
            "legacy_v57_cosine": legacy_cosine,
            "v58_natural_cosine": natural_cosine,
            "v58_fixed_support_cosine": fixed_cosine,
            "legacy_v57_relative_error": relative_error(predicted57, fd57),
            "v58_natural_relative_error": relative_error(predicted58, fd58),
            "v58_fixed_support_relative_error": relative_error(predicted58_fixed, fd58_fixed),
            "v57_perturbed_support_change_count": support_change57,
            "v58_perturbed_support_change_count": support_change58,
            "predicted_v57": predicted57,
            "predicted_v58": predicted58,
            "predicted_v58_fixed": predicted58_fixed,
            "finite_difference_v57": fd57,
            "finite_difference_v58": fd58,
            "finite_difference_v58_fixed": fd58_fixed,
        },
    }
    if report["state"]["schema_identical"] is not True:
        report["status"] = "FAIL"
    if not report["gradients"]["all_v58_finite"]:
        report["status"] = "FAIL"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "forward": report["forward"],
        "gradients": report["gradients"],
        "directional_finite_difference": {
            key: value for key, value in report["directional_finite_difference"].items()
            if key not in {"predicted_v57", "predicted_v58", "predicted_v58_fixed", "finite_difference_v57", "finite_difference_v58", "finite_difference_v58_fixed"}
        },
    }, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
