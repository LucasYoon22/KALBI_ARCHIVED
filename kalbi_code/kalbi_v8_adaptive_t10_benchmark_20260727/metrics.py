"""Objective, KKT, tensor-contract, and timing metrics."""

from __future__ import annotations

import hashlib
import math
import statistics
from typing import Any

import torch

from kalbi_v7_fullp_blockprox3_v23_k32s32_p100k_effbatch32.exact_direct96_prox import (
    interpolate_direct96_penalty,
    unpack_direct96_penalty,
)
from kalbi_v7_fullp_blockprox3_v23_k32s32_p100k_effbatch32.utils import (
    strip_convexity_flag,
)


Tensor = torch.Tensor


def tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(str(tensor.dtype).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def objective_components(
    *,
    x: Tensor,
    y: Tensor,
    beta: Tensor,
    lambdas: Tensor,
    penalty_shape_features: Tensor,
    grid: Tensor,
) -> dict[str, Tensor]:
    prediction = torch.einsum("bnp,bqp->bqn", x.float(), beta.float())
    residual = y.float().unsqueeze(1) - prediction
    data_half = 0.5 * residual.square().mean(-1)
    pieces = unpack_direct96_penalty(
        strip_convexity_flag(penalty_shape_features).float(),
        grid.float(),
    )
    rho, _, _ = interpolate_direct96_penalty(
        beta.float().abs(), pieces, grid.float()
    )
    penalty = lambdas.float() * rho.sum(-1)
    objective = data_half + penalty
    f_zero = 0.5 * y.float().square().mean(-1)
    return {
        "prediction": prediction,
        "residual": residual,
        "data_half": data_half,
        "penalty": penalty,
        "objective": objective,
        "f_zero": f_zero,
    }


@torch.no_grad()
def kkt_metrics(
    *,
    x: Tensor,
    beta: Tensor,
    residual: Tensor,
    lambdas: Tensor,
    penalty_shape_features: Tensor,
    grid: Tensor,
    feature_chunk: int = 4096,
) -> dict[str, Tensor]:
    sample_count = max(1, int(x.shape[1]))
    gradient = -torch.einsum(
        "bnp,bqn->bqp", x.float(), residual.float()
    ) / float(sample_count)
    pieces = unpack_direct96_penalty(
        strip_convexity_flag(penalty_shape_features).float(),
        grid.float(),
    )
    d0 = pieces.drho[..., 0]
    residual_chunks: list[Tensor] = []
    nonzero_chunks: list[Tensor] = []
    zero_chunks: list[Tensor] = []
    feature_count = int(beta.shape[-1])
    for start in range(0, feature_count, int(feature_chunk)):
        stop = min(feature_count, start + int(feature_chunk))
        beta_chunk = beta[..., start:stop].float()
        magnitude = beta_chunk.abs()
        _, derivative, _ = interpolate_direct96_penalty(
            magnitude, pieces, grid.float()
        )
        nonzero = magnitude > 1e-12
        nonzero_residual = (
            gradient[..., start:stop]
            + lambdas.float()[..., None]
            * derivative
            * beta_chunk.sign()
        )
        zero_residual = torch.relu(
            gradient[..., start:stop].abs()
            - lambdas.float()[..., None] * d0[..., None]
        )
        combined = torch.where(
            nonzero, nonzero_residual, zero_residual
        )
        residual_chunks.append(combined)
        nonzero_chunks.append(
            torch.where(nonzero, nonzero_residual, torch.zeros_like(combined))
        )
        zero_chunks.append(
            torch.where(~nonzero, zero_residual, torch.zeros_like(combined))
        )
    full = torch.cat(residual_chunks, -1)
    nonzero_full = torch.cat(nonzero_chunks, -1)
    zero_full = torch.cat(zero_chunks, -1)
    nonzero_mask = beta.abs() > 1e-12
    zero_mask = ~nonzero_mask
    full_rms = full.square().mean(-1).sqrt()
    nonzero_rms = (
        nonzero_full.square().sum(-1)
        / nonzero_mask.sum(-1).clamp_min(1)
    ).sqrt()
    zero_rms = (
        zero_full.square().sum(-1)
        / zero_mask.sum(-1).clamp_min(1)
    ).sqrt()
    diagonal = x.float().square().mean(1).clamp_min(1e-6)
    score = gradient.abs() / diagonal[:, None, :].sqrt()
    beta_scale = beta.abs().amax(-1, keepdim=True).clamp_min(1e-6)
    gradient_scale = score.amax(-1, keepdim=True).clamp_min(1e-6)
    score = score + 0.5 * gradient_scale * (beta.abs() / beta_scale)
    selected = torch.topk(
        score, k=min(32, feature_count), dim=-1
    ).indices
    selected_residual = torch.gather(full, -1, selected)
    top32 = (
        selected_residual.square().sum(-1) + 1e-30
    ).sqrt()
    return {
        "kkt_top32": top32,
        "kkt_full_rms": full_rms,
        "kkt_nonzero_rms": nonzero_rms,
        "kkt_zero_rms": zero_rms,
        "kkt_max_abs": full.abs().amax(-1),
    }


def summarize_timing(values: list[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("timing values are empty")

    def percentile(probability: float) -> float:
        position = probability * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    mean = statistics.mean(ordered)
    standard_deviation = (
        statistics.stdev(ordered) if len(ordered) > 1 else 0.0
    )
    return {
        "mean_seconds": mean,
        "median_seconds": statistics.median(ordered),
        "p90_seconds": percentile(0.9),
        "standard_deviation_seconds": standard_deviation,
        "coefficient_of_variation": (
            standard_deviation / mean if mean > 0.0 else 0.0
        ),
        "minimum_seconds": ordered[0],
        "maximum_seconds": ordered[-1],
    }


def tensors_to_lists(payload: dict[str, Tensor]) -> dict[str, Any]:
    return {
        key: value.detach().float().cpu().flatten().tolist()
        for key, value in payload.items()
    }


__all__ = [
    "kkt_metrics",
    "objective_components",
    "summarize_timing",
    "tensor_sha256",
    "tensors_to_lists",
]
