"""Configurable block partition around the audited KALBI V7 block update."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from kalbi_v7_fullp_blockprox3_v23_k32s32_p100k_effbatch32.block_prox_v7 import (
    _apply_block_update,
    _estimate_block_lipschitz,
)


Tensor = torch.Tensor
SWEEP_REVERSE = (False, True, False)
MIN_BLOCK_SIZE = 256
MAX_BLOCK_SIZE = 4096


def adaptive_block_size(feature_count: int) -> int:
    if feature_count <= 0:
        raise ValueError("feature_count must be positive")
    target = max(1.0, float(feature_count) / 16.0)
    power_of_two = 1 << int(math.ceil(math.log2(target)))
    return min(MAX_BLOCK_SIZE, max(MIN_BLOCK_SIZE, power_of_two))


def feature_blocks(
    feature_count: int, block_size: int
) -> tuple[tuple[int, int], ...]:
    if feature_count <= 0:
        raise ValueError("feature_count must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return tuple(
        (start, min(feature_count, start + int(block_size)))
        for start in range(0, feature_count, int(block_size))
    )


def _get_block_curvature(
    encoded: dict[str, Any],
    blocks: tuple[tuple[int, int], ...],
    block_size: int,
) -> tuple[Tensor, Tensor]:
    cache_key = f"_microbench_block_curvature_{int(block_size)}"
    cached = encoded.get(cache_key)
    if cached is not None:
        return cached
    x_centered = encoded["Xc"].float()
    feature_mask = encoded.get("feature_mask")
    estimates: list[Tensor] = []
    traces: list[Tensor] = []
    with torch.no_grad():
        for start, stop in blocks:
            block_mask = (
                None if feature_mask is None else feature_mask[:, start:stop]
            )
            estimate, trace = _estimate_block_lipschitz(
                x_centered[:, :, start:stop],
                block_mask,
            )
            estimates.append(estimate)
            traces.append(trace)
    cached = (torch.stack(estimates, -1), torch.stack(traces, -1))
    encoded[cache_key] = cached
    return cached


def clear_curvature_cache(encoded: dict[str, Any]) -> None:
    for key in tuple(encoded):
        if key.startswith("_microbench_block_curvature_"):
            del encoded[key]


def _run_sweep(
    model: torch.nn.Module,
    encoded: dict[str, Any],
    beta: Tensor,
    residual: Tensor,
    lambdas: Tensor,
    penalty_d0: Tensor,
    penalty_shape_features: Tensor,
    *,
    reverse: bool,
    block_size: int,
    activation_checkpointing: bool,
) -> dict[str, Tensor]:
    x_centered = encoded["Xc"].float()
    feature_mask = encoded.get("feature_mask")
    feature_count = int(x_centered.shape[-1])
    blocks = feature_blocks(feature_count, block_size)
    estimates, traces = _get_block_curvature(
        encoded, blocks, block_size
    )
    beta_blocks = [beta[..., start:stop] for start, stop in blocks]
    order = (
        range(len(blocks) - 1, -1, -1)
        if reverse
        else range(len(blocks))
    )
    diagnostics: list[tuple[Tensor, ...]] = []
    widths: list[float] = []
    residual_start = residual.detach()

    for block_index in order:
        start, stop = blocks[block_index]
        x_block = x_centered[:, :, start:stop]
        block_mask = (
            None if feature_mask is None else feature_mask[:, start:stop]
        )

        def update(
            current_block: Tensor,
            current_residual: Tensor,
            current_lambdas: Tensor,
            current_d0: Tensor,
            current_shape: Tensor,
            *,
            fixed_x_block: Tensor = x_block,
            fixed_block_mask: Tensor | None = block_mask,
            fixed_lipschitz: Tensor = estimates[:, block_index],
            fixed_trace: Tensor = traces[:, block_index],
        ) -> tuple[Tensor, ...]:
            return _apply_block_update(
                model,
                current_block,
                current_residual,
                fixed_x_block,
                fixed_block_mask,
                current_lambdas,
                current_d0,
                current_shape,
                fixed_lipschitz,
                fixed_trace,
            )

        inputs = (
            beta_blocks[block_index],
            residual,
            lambdas,
            penalty_d0,
            penalty_shape_features,
        )
        use_checkpoint = (
            activation_checkpointing
            and torch.is_grad_enabled()
            and any(tensor.requires_grad for tensor in inputs)
        )
        outputs = (
            checkpoint(
                update,
                *inputs,
                use_reentrant=False,
                preserve_rng_state=False,
            )
            if use_checkpoint
            else update(*inputs)
        )
        beta_blocks[block_index], residual = outputs[0], outputs[1]
        diagnostics.append(outputs[2:])
        widths.append(float(stop - start))

    refined = torch.cat(beta_blocks, dim=-1)
    width_tensor = torch.tensor(
        widths, device=refined.device, dtype=torch.float32
    )
    total_width = float(feature_count)

    def weighted_mean(position: int) -> Tensor:
        values = torch.stack(
            [item[position] for item in diagnostics], -1
        )
        return (values * width_tensor).sum(-1) / total_width

    penalty_before = torch.stack(
        [item[16] for item in diagnostics], -1
    ).sum(-1)
    penalty_after = torch.stack(
        [item[17] for item in diagnostics], -1
    ).sum(-1)
    data_half_before = 0.5 * residual_start.square().mean(-1)
    data_half_after = 0.5 * residual.detach().square().mean(-1)
    objective_before = data_half_before + penalty_before
    objective_after = data_half_after + penalty_after
    tolerance = 1e-6 * torch.maximum(
        torch.ones_like(objective_before), objective_before.abs()
    )
    violations = objective_after > objective_before + tolerance
    if bool(violations.any().item()):
        raise RuntimeError("configurable block sweep increased objective")

    change_energy = torch.stack(
        [item[12] for item in diagnostics], -1
    ).sum(-1)
    active_count = torch.stack(
        [item[13] for item in diagnostics], -1
    ).sum(-1)
    selection_rounds = 0
    for item in diagnostics:
        selection_rounds += 1 + int(item[8].detach().max().item())
        selection_rounds += int(bool(item[9].detach().any().item()))
    prox_calls = len(diagnostics) + selection_rounds
    shape = objective_before.shape

    return {
        "beta": refined,
        "residual": residual,
        "step_size": weighted_mean(0),
        "gamma": weighted_mean(1),
        "kappa": weighted_mean(2),
        "minimum_curvature_denominator": torch.stack(
            [item[3] for item in diagnostics], -1
        ).amin(-1),
        "curvature_step_cap": weighted_mean(4),
        "initial_lipschitz": weighted_mean(5),
        "trace_lipschitz": weighted_mean(6),
        "majorization_ratio": torch.stack(
            [item[7] for item in diagnostics], -1
        ).amax(-1),
        "backtracks": torch.stack(
            [item[8] for item in diagnostics], -1
        ).sum(-1),
        "trace_fallback": weighted_mean(9),
        "data_gradient_rms": (
            (
                torch.stack(
                    [item[10].square() for item in diagnostics], -1
                )
                * width_tensor
            ).sum(-1)
            / total_width
        ).sqrt(),
        "zero_threshold": weighted_mean(11),
        "change_rms": (change_energy / total_width).sqrt(),
        "active_fraction": active_count / total_width,
        "saturated_fraction": weighted_mean(14),
        "d0_contract_error": torch.stack(
            [item[15] for item in diagnostics], -1
        ).amax(-1),
        "data_half_before": data_half_before,
        "data_half_after": data_half_after,
        "penalty_before": penalty_before,
        "penalty_after": penalty_after,
        "objective_before": objective_before,
        "objective_after": objective_after,
        "objective_delta": objective_after - objective_before,
        "rejected_fraction": weighted_mean(18),
        "visited_features": torch.full_like(
            objective_before, float(feature_count)
        ),
        "block_update_count": torch.full(
            shape,
            float(len(blocks)),
            device=refined.device,
        ),
        "exact_prox_call_count_forward": torch.full(
            shape, float(prox_calls), device=refined.device
        ),
        "backtracking_evaluation_count": torch.full(
            shape, float(selection_rounds), device=refined.device
        ),
        "monotonicity_violation_count": violations.float(),
    }


def run_block_refinement(
    model: torch.nn.Module,
    encoded: dict[str, Any],
    beta_initial: Tensor,
    lambdas: Tensor,
    penalty_d0: Tensor,
    penalty_shape_features: Tensor,
    *,
    block_size: int,
    activation_checkpointing: bool = True,
) -> tuple[Tensor, list[dict[str, Tensor]]]:
    if "Xc" not in encoded or "yc" not in encoded:
        raise RuntimeError("block refinement requires retained centered X/y")
    x_centered = encoded["Xc"].float()
    y_centered = encoded["yc"].float()
    beta = beta_initial.float()
    residual = (
        torch.matmul(beta, x_centered.transpose(1, 2))
        - y_centered[:, None, :]
    )
    history: list[dict[str, Tensor]] = []
    for reverse in SWEEP_REVERSE:
        sweep = _run_sweep(
            model,
            encoded,
            beta,
            residual,
            lambdas,
            penalty_d0,
            penalty_shape_features,
            reverse=reverse,
            block_size=int(block_size),
            activation_checkpointing=activation_checkpointing,
        )
        beta = sweep["beta"]
        residual = sweep["residual"]
        history.append(sweep)
    if not bool(torch.isfinite(beta).all().item()):
        raise RuntimeError("block refinement produced nonfinite beta")
    return beta, history


__all__ = [
    "MAX_BLOCK_SIZE",
    "MIN_BLOCK_SIZE",
    "SWEEP_REVERSE",
    "adaptive_block_size",
    "clear_curvature_cache",
    "feature_blocks",
    "run_block_refinement",
]

