"""Full-feature block-coordinate Direct96 proximal refinement for KALBI V7."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from .exact_direct96_prox import (
    DEFAULT_CURVATURE_MARGIN,
    curvature_safe_step_cap,
    exact_direct96_prox,
    interpolate_direct96_penalty,
    unpack_direct96_penalty,
)
from .utils import strip_convexity_flag


Tensor = torch.Tensor

SWEEPS = 3
BLOCK_SIZE = 4096
POWER_ITERATIONS = 2
SPECTRAL_SAFETY = 1.25
DATA_MARGIN = 0.95
CURVATURE_MARGIN = DEFAULT_CURVATURE_MARGIN
BACKTRACK_FACTOR = 0.5
MAX_BACKTRACKS = 8
FEATURE_CHUNK = 4096
SWEEP_REVERSE = (False, True, False)


def feature_blocks(feature_count: int) -> tuple[tuple[int, int], ...]:
    if feature_count <= 0:
        raise ValueError("feature_count must be positive")
    return tuple(
        (start, min(feature_count, start + BLOCK_SIZE))
        for start in range(0, feature_count, BLOCK_SIZE)
    )


@torch.no_grad()
def _estimate_block_lipschitz(
    x_block: Tensor,
    block_mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Estimate local curvature; exact directional backtracking is authoritative."""

    batch_size, sample_count, width = x_block.shape
    index = torch.arange(width, device=x_block.device, dtype=torch.float32)
    seed = (
        torch.cos(index * 0.61803398875)
        + torch.sin(index * 0.41421356237)
    )
    vector = seed[None].expand(batch_size, -1)
    if block_mask is not None:
        vector = vector * block_mask.float()
    vector = vector / vector.square().sum(
        -1, keepdim=True
    ).sqrt().clamp_min(1e-12)
    for _ in range(POWER_ITERATIONS):
        row_projection = torch.matmul(
            x_block, vector[..., None]
        ).squeeze(-1)
        vector = torch.matmul(
            x_block.transpose(1, 2), row_projection[..., None]
        ).squeeze(-1) / float(max(1, sample_count))
        if block_mask is not None:
            vector = vector * block_mask.float()
        vector = vector / vector.square().sum(
            -1, keepdim=True
        ).sqrt().clamp_min(1e-12)
    row_projection = torch.matmul(
        x_block, vector[..., None]
    ).squeeze(-1)
    rayleigh = row_projection.square().sum(-1) / float(max(1, sample_count))
    diagonal = x_block.square().mean(1)
    if block_mask is not None:
        diagonal = diagonal * block_mask.float()
    diagonal_max = diagonal.amax(-1).clamp_min(1e-8)
    estimate = torch.maximum(
        SPECTRAL_SAFETY * rayleigh, diagonal_max
    ).clamp_min(1e-8)
    trace = diagonal.sum(-1).clamp_min(1e-8)
    return estimate, trace


def _get_block_curvature(
    encoded: dict[str, Any],
    blocks: tuple[tuple[int, int], ...],
) -> tuple[Tensor, Tensor]:
    cached = encoded.get("_kalbi_v7_block_curvature")
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
    encoded["_kalbi_v7_block_curvature"] = cached
    return cached


@torch.no_grad()
def _select_block_step(
    model: torch.nn.Module,
    beta_block: Tensor,
    residual: Tensor,
    gradient: Tensor,
    x_block: Tensor,
    block_mask: Tensor | None,
    lambdas: Tensor,
    penalty_shape_features: Tensor,
    lipschitz_estimate: Tensor,
    trace_lipschitz: Tensor,
) -> dict[str, Tensor]:
    grid = model._kalbi_v7_penalty_grid.float()
    native = strip_convexity_flag(penalty_shape_features).detach().float()
    lam = lambdas.detach().float().clamp_min(0.0)
    pieces = unpack_direct96_penalty(native, grid)
    curvature_cap = curvature_safe_step_cap(
        lam,
        pieces.kappa,
        margin=CURVATURE_MARGIN,
    )
    initial_lipschitz = lipschitz_estimate[:, None].expand_as(lam)
    trace_expanded = trace_lipschitz[:, None].expand_as(lam)
    step_size = torch.minimum(
        DATA_MARGIN / initial_lipschitz,
        curvature_cap,
    )
    accepted = torch.zeros_like(step_size, dtype=torch.bool)
    accepted_ratio = torch.zeros_like(step_size)
    accepted_penalty_before = torch.zeros_like(step_size)
    accepted_penalty_after = torch.zeros_like(step_size)
    backtracks = torch.zeros_like(step_size, dtype=torch.int64)
    sample_count = max(1, int(x_block.shape[1]))

    def evaluate(candidate_step: Tensor) -> tuple[Tensor, ...]:
        gamma = candidate_step * lam
        proposal = beta_block.detach() - candidate_step[..., None] * (
            gradient.detach()
        )
        candidate, _ = exact_direct96_prox(
            proposal,
            gamma,
            native,
            grid,
            feature_chunk=FEATURE_CHUNK,
            curvature_margin=CURVATURE_MARGIN,
        )
        if block_mask is not None:
            candidate = candidate * block_mask[:, None, :].float()
        delta = candidate - beta_block.detach()
        x_delta = torch.matmul(delta, x_block.transpose(1, 2))
        delta_energy = delta.square().sum(-1)
        ratio = torch.where(
            delta_energy <= 1e-20,
            torch.zeros_like(candidate_step),
            candidate_step
            * (x_delta.square().sum(-1) / float(sample_count))
            / delta_energy.clamp_min(1e-20),
        )
        rho_before, _, _ = interpolate_direct96_penalty(
            beta_block.detach().abs(), pieces, grid
        )
        rho_after, _, _ = interpolate_direct96_penalty(
            candidate.abs(), pieces, grid
        )
        penalty_before = lam * rho_before.sum(-1)
        penalty_after = lam * rho_after.sum(-1)
        data_before = 0.5 * residual.detach().square().mean(-1)
        data_after = 0.5 * (
            residual.detach() + x_delta
        ).square().mean(-1)
        objective_before = data_before + penalty_before
        objective_after = data_after + penalty_after
        tolerance = 1e-6 * torch.maximum(
            torch.ones_like(objective_before), objective_before.abs()
        )
        passed = (
            (delta_energy <= 1e-20)
            | (
                torch.isfinite(ratio)
                & (ratio <= DATA_MARGIN + 1e-6)
            )
        ) & torch.isfinite(objective_after) & (
            objective_after <= objective_before + tolerance
        )
        return passed, ratio, penalty_before, penalty_after

    for attempt in range(MAX_BACKTRACKS + 1):
        passed, ratio, penalty_before, penalty_after = evaluate(step_size)
        newly_accepted = (~accepted) & passed
        accepted_ratio = torch.where(
            newly_accepted, ratio, accepted_ratio
        )
        accepted_penalty_before = torch.where(
            newly_accepted, penalty_before, accepted_penalty_before
        )
        accepted_penalty_after = torch.where(
            newly_accepted, penalty_after, accepted_penalty_after
        )
        accepted = accepted | passed
        unresolved = ~accepted
        if not bool(unresolved.any().item()):
            break
        if attempt < MAX_BACKTRACKS:
            step_size = torch.where(
                unresolved, step_size * BACKTRACK_FACTOR, step_size
            )
            backtracks = backtracks + unresolved.to(backtracks.dtype)

    used_trace_fallback = ~accepted
    if bool(used_trace_fallback.any().item()):
        fallback_step = torch.minimum(
            DATA_MARGIN / trace_expanded, curvature_cap
        )
        step_size = torch.where(
            used_trace_fallback, fallback_step, step_size
        )
        passed, ratio, penalty_before, penalty_after = evaluate(step_size)
        use_fallback = used_trace_fallback & passed
        accepted_ratio = torch.where(
            use_fallback, ratio, accepted_ratio
        )
        accepted_penalty_before = torch.where(
            use_fallback, penalty_before, accepted_penalty_before
        )
        accepted_penalty_after = torch.where(
            use_fallback, penalty_after, accepted_penalty_after
        )
        accepted = accepted | use_fallback

    rejected = ~accepted
    step_size = torch.where(rejected, torch.zeros_like(step_size), step_size)
    if bool(rejected.any().item()):
        rho_current, _, _ = interpolate_direct96_penalty(
            beta_block.detach().abs(), pieces, grid
        )
        current_penalty = lam * rho_current.sum(-1)
        accepted_penalty_before = torch.where(
            rejected, current_penalty, accepted_penalty_before
        )
        accepted_penalty_after = torch.where(
            rejected, current_penalty, accepted_penalty_after
        )
    return {
        "step_size": step_size.detach(),
        "kappa": pieces.kappa.detach(),
        "curvature_step_cap": curvature_cap.detach(),
        "initial_lipschitz": initial_lipschitz.detach(),
        "trace_lipschitz": trace_expanded.detach(),
        "majorization_ratio": accepted_ratio.detach(),
        "backtracks": backtracks.detach(),
        "trace_fallback": used_trace_fallback.float().detach(),
        "rejected": rejected.float().detach(),
        "penalty_before": accepted_penalty_before.detach(),
        "penalty_after": accepted_penalty_after.detach(),
    }


def _apply_block_update(
    model: torch.nn.Module,
    beta_block: Tensor,
    residual: Tensor,
    x_block: Tensor,
    block_mask: Tensor | None,
    lambdas: Tensor,
    penalty_d0: Tensor,
    penalty_shape_features: Tensor,
    lipschitz_estimate: Tensor,
    trace_lipschitz: Tensor,
) -> tuple[Tensor, ...]:
    sample_count = max(1, int(x_block.shape[1]))
    gradient = torch.matmul(residual, x_block) / float(sample_count)
    with torch.no_grad():
        selected = _select_block_step(
            model,
            beta_block,
            residual,
            gradient,
            x_block,
            block_mask,
            lambdas,
            penalty_shape_features,
            lipschitz_estimate,
            trace_lipschitz,
        )
    step_size = selected["step_size"]
    gamma = step_size * lambdas.float().clamp_min(0.0)
    proposal = beta_block - step_size[..., None] * gradient
    updated, prox = exact_direct96_prox(
        proposal,
        gamma,
        strip_convexity_flag(penalty_shape_features).float(),
        model._kalbi_v7_penalty_grid.float(),
        feature_chunk=FEATURE_CHUNK,
        curvature_margin=CURVATURE_MARGIN,
    )
    if block_mask is not None:
        updated = updated * block_mask[:, None, :].float()
    change = updated - beta_block
    updated_residual = residual + torch.matmul(
        change, x_block.transpose(1, 2)
    )
    native_d0 = strip_convexity_flag(
        penalty_shape_features
    ).float()[..., 1]
    return (
        updated,
        updated_residual,
        step_size,
        gamma.detach(),
        prox["kappa"].detach(),
        prox["minimum_denominator"].detach(),
        selected["curvature_step_cap"],
        selected["initial_lipschitz"],
        selected["trace_lipschitz"],
        selected["majorization_ratio"],
        selected["backtracks"].float(),
        selected["trace_fallback"],
        gradient.detach().square().mean(-1).sqrt(),
        prox["zero_threshold"].detach(),
        change.detach().square().sum(-1),
        (updated.detach() != 0.0).float().sum(-1),
        prox["saturated_fraction"].detach(),
        (native_d0.detach() - penalty_d0.detach().float()).abs(),
        selected["penalty_before"],
        selected["penalty_after"],
        selected["rejected"],
    )


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
) -> dict[str, Tensor]:
    x_centered = encoded["Xc"].float()
    feature_mask = encoded.get("feature_mask")
    feature_count = int(x_centered.shape[-1])
    blocks = feature_blocks(feature_count)
    estimates, traces = _get_block_curvature(encoded, blocks)
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
        use_checkpoint = torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in inputs
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
    if bool((objective_after > objective_before + tolerance).any().item()):
        raise RuntimeError("KALBI V7 block sweep increased the objective")
    change_energy = torch.stack(
        [item[12] for item in diagnostics], -1
    ).sum(-1)
    active_count = torch.stack(
        [item[13] for item in diagnostics], -1
    ).sum(-1)
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
    }


def run_full_p_block_refinement(
    model: torch.nn.Module,
    encoded: dict[str, Any],
    beta_initial: Tensor,
    lambdas: Tensor,
    penalty_d0: Tensor,
    penalty_shape_features: Tensor,
) -> tuple[Tensor, list[dict[str, Tensor]]]:
    if "Xc" not in encoded or "yc" not in encoded:
        raise RuntimeError("KALBI V7 requires retained centered X and y")
    x_centered = encoded["Xc"].float()
    y_centered = encoded["yc"].float()
    beta = beta_initial.float()
    residual = torch.matmul(
        beta, x_centered.transpose(1, 2)
    ) - y_centered[:, None, :]
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
        )
        beta = sweep["beta"]
        residual = sweep["residual"]
        history.append(sweep)
    if not bool(torch.isfinite(beta).all().item()):
        raise RuntimeError("KALBI V7 produced nonfinite coefficients")
    return beta, history


__all__ = [
    "BACKTRACK_FACTOR",
    "BLOCK_SIZE",
    "CURVATURE_MARGIN",
    "DATA_MARGIN",
    "FEATURE_CHUNK",
    "MAX_BACKTRACKS",
    "POWER_ITERATIONS",
    "SPECTRAL_SAFETY",
    "SWEEPS",
    "SWEEP_REVERSE",
    "feature_blocks",
    "run_full_p_block_refinement",
]
