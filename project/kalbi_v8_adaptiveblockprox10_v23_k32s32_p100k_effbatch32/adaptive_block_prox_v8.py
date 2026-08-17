"""Adaptive full-P block-coordinate Direct96 refinement for KALBI V8."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from .block_prox_v7 import _apply_block_update, _estimate_block_lipschitz


Tensor = torch.Tensor


@dataclass(frozen=True)
class AdaptiveBlockProxConfig:
    sweeps: int = 10
    target_blocks: int = 16
    min_block_size: int = 256
    max_block_size: int = 4096
    activation_checkpointing: bool = True

    def validate(self) -> None:
        if self.sweeps <= 0:
            raise ValueError("sweeps must be positive")
        if self.target_blocks <= 0:
            raise ValueError("target_blocks must be positive")
        if self.min_block_size <= 0:
            raise ValueError("min_block_size must be positive")
        if self.max_block_size < self.min_block_size:
            raise ValueError("max_block_size must be at least min_block_size")
        for value, label in (
            (self.min_block_size, "min_block_size"),
            (self.max_block_size, "max_block_size"),
        ):
            if value & (value - 1):
                raise ValueError(f"{label} must be a power of two")


DEFAULT_CONFIG = AdaptiveBlockProxConfig()
ADAPTIVE_BLOCK_FORMULA_VERSION = (
    "pow2_ceil_P_over_16_clamp_256_4096_v1"
)
SWEEP_ORDER_VERSION = "forward_reverse_alternating_v1"


def adaptive_block_size(
    feature_count: int,
    config: AdaptiveBlockProxConfig = DEFAULT_CONFIG,
) -> int:
    config.validate()
    if feature_count <= 0:
        raise ValueError("feature_count must be positive")
    target = max(
        1, (int(feature_count) + config.target_blocks - 1)
        // config.target_blocks
    )
    power_of_two = 1 << (target - 1).bit_length()
    return min(
        config.max_block_size,
        max(config.min_block_size, power_of_two),
    )


def sweep_reverse_flags(
    config: AdaptiveBlockProxConfig = DEFAULT_CONFIG,
) -> tuple[bool, ...]:
    config.validate()
    return tuple(index % 2 == 1 for index in range(config.sweeps))


def feature_blocks(
    feature_count: int,
    block_size: int,
) -> tuple[tuple[int, int], ...]:
    if feature_count <= 0:
        raise ValueError("feature_count must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return tuple(
        (start, min(feature_count, start + int(block_size)))
        for start in range(0, feature_count, int(block_size))
    )


def valid_feature_count(encoded: dict[str, Any]) -> int:
    x_centered = encoded["Xc"]
    full_width = int(x_centered.shape[-1])
    feature_mask = encoded.get("feature_mask")
    if feature_mask is None:
        return full_width
    mask = feature_mask.bool()
    if mask.ndim != 2 or int(mask.shape[-1]) != full_width:
        raise RuntimeError("feature mask shape does not match centered X")
    counts = mask.sum(dim=-1)
    if not bool((counts == counts[0]).all().item()):
        raise RuntimeError(
            "adaptive block refinement requires a common valid P per batch"
        )
    count = int(counts[0].item())
    if count <= 0:
        raise RuntimeError("feature mask contains no valid feature")
    expected = (
        torch.arange(full_width, device=mask.device)[None, :] < count
    )
    if not bool((mask == expected).all().item()):
        raise RuntimeError(
            "adaptive block refinement requires a contiguous prefix mask"
        )
    return count


def _curvature_cache_key(
    x_centered: Tensor,
    feature_count: int,
    block_size: int,
) -> str:
    device = x_centered.device
    return (
        "_kalbi_v8_block_curvature_"
        f"p{feature_count}_b{block_size}_"
        f"{device.type}{device.index}_{str(x_centered.dtype)}"
    )


def _get_block_curvature(
    encoded: dict[str, Any],
    blocks: tuple[tuple[int, int], ...],
    feature_count: int,
    block_size: int,
) -> tuple[Tensor, Tensor]:
    x_centered = encoded["Xc"].float()
    cache_key = _curvature_cache_key(
        x_centered, feature_count, block_size
    )
    cached = encoded.get(cache_key)
    if cached is not None:
        return cached
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
        if key.startswith("_kalbi_v8_block_curvature_"):
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
    feature_count: int,
    block_size: int,
    activation_checkpointing: bool,
) -> dict[str, Tensor]:
    x_centered = encoded["Xc"].float()
    feature_mask = encoded.get("feature_mask")
    full_width = int(x_centered.shape[-1])
    blocks = feature_blocks(feature_count, block_size)
    estimates, traces = _get_block_curvature(
        encoded, blocks, feature_count, block_size
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

    refined_valid = torch.cat(beta_blocks, dim=-1)
    if feature_count < full_width:
        refined = torch.cat(
            [refined_valid, beta[..., feature_count:] * 0.0], dim=-1
        )
    else:
        refined = refined_valid
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
        raise RuntimeError("adaptive block sweep increased objective")

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
        "adaptive_block_size": torch.full(
            shape, float(block_size), device=refined.device
        ),
        "blocks_per_sweep": torch.full(
            shape, float(len(blocks)), device=refined.device
        ),
        "block_update_count": torch.full(
            shape, float(len(blocks)), device=refined.device
        ),
        "exact_prox_call_count_forward": torch.full(
            shape, float(prox_calls), device=refined.device
        ),
        "backtracking_evaluation_count": torch.full(
            shape, float(selection_rounds), device=refined.device
        ),
        "monotonicity_violation_count": violations.float(),
    }


def run_full_p_adaptive_block_refinement(
    model: torch.nn.Module,
    encoded: dict[str, Any],
    beta_initial: Tensor,
    lambdas: Tensor,
    penalty_d0: Tensor,
    penalty_shape_features: Tensor,
    *,
    config: AdaptiveBlockProxConfig = DEFAULT_CONFIG,
) -> tuple[Tensor, list[dict[str, Tensor]]]:
    config.validate()
    if "Xc" not in encoded or "yc" not in encoded:
        raise RuntimeError("KALBI V8 requires retained centered X and y")
    x_centered = encoded["Xc"].float()
    y_centered = encoded["yc"].float()
    feature_count = valid_feature_count(encoded)
    block_size = adaptive_block_size(feature_count, config)
    beta = beta_initial.float()
    feature_mask = encoded.get("feature_mask")
    if feature_mask is not None:
        beta = beta * feature_mask[:, None, :].float()
    residual = (
        torch.matmul(beta, x_centered.transpose(1, 2))
        - y_centered[:, None, :]
    )
    history: list[dict[str, Tensor]] = []
    for reverse in sweep_reverse_flags(config):
        sweep = _run_sweep(
            model,
            encoded,
            beta,
            residual,
            lambdas,
            penalty_d0,
            penalty_shape_features,
            reverse=reverse,
            feature_count=feature_count,
            block_size=block_size,
            activation_checkpointing=config.activation_checkpointing,
        )
        beta = sweep["beta"]
        residual = sweep["residual"]
        history.append(sweep)
    if not bool(torch.isfinite(beta).all().item()):
        raise RuntimeError("KALBI V8 produced nonfinite coefficients")
    return beta, history


__all__ = [
    "ADAPTIVE_BLOCK_FORMULA_VERSION",
    "AdaptiveBlockProxConfig",
    "DEFAULT_CONFIG",
    "SWEEP_ORDER_VERSION",
    "adaptive_block_size",
    "clear_curvature_cache",
    "feature_blocks",
    "run_full_p_adaptive_block_refinement",
    "sweep_reverse_flags",
    "valid_feature_count",
]
