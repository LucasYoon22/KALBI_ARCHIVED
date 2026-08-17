"""Compute the shared pre-refinement neural estimate without running a solver."""

from __future__ import annotations

from typing import Any

import torch

from kalbi_v8_adaptiveblockprox10_v23_k32s32_p100k_effbatch32 import model as v8


Tensor = torch.Tensor


def compute_beta_initial(
    model: torch.nn.Module,
    encoded: dict[str, Any],
    *,
    lambdas: Tensor,
    lambda_max: Tensor,
    penalty_shape_features: Tensor,
    penalty_d0: Tensor,
) -> dict[str, Tensor]:
    lam, _, d0, native_shape, prox_condition, condition = (
        v8._resolve_query_context(
            model,
            encoded,
            lambdas,
            lambda_max,
            None,
            None,
            penalty_shape_features,
            penalty_d0,
        )
    )
    base, alpha = v8._global_krylov_ridge(
        model, encoded, lam, condition
    )
    trajectory = v8._get_ab_trajectory(model, encoded)
    model_dtype = condition.dtype
    head_output = model.kalbi_ab_head(
        A=trajectory["A"].to(model_dtype),
        B=trajectory["B"].to(model_dtype),
        Xty=encoded["Xty"].to(model_dtype),
        gram_diag=encoded["gram_diag"].to(model_dtype),
        log_abs_Xty=encoded["log_abs_Xty"].to(model_dtype),
        base=base,
        support=trajectory["support_mask"],
        film_cond=prox_condition[..., : v8.FEATURE_FILM_DIM].to(model_dtype),
    )
    correction = head_output["correction"].to(base.dtype)
    log_threshold_multiplier = head_output[
        "log_threshold_multiplier"
    ].float().clamp(
        v8.KALBI_THRESHOLD_LOG_MULTIPLIER_MIN,
        v8.KALBI_THRESHOLD_LOG_MULTIPLIER_MAX,
    )
    gram_diagonal = encoded["gram_diag"].float().clamp_min(1e-4)
    threshold_base = (
        lam.float()[..., None]
        * d0.float()[..., None]
        / gram_diagonal[:, None, :]
    )
    threshold = torch.nan_to_num(
        threshold_base * torch.exp(log_threshold_multiplier),
        nan=0.0,
        posinf=1e6,
        neginf=0.0,
    ).clamp_min(0.0)
    prethreshold = torch.nan_to_num(
        base.float() + correction.float(),
        nan=0.0,
        posinf=20.0,
        neginf=-20.0,
    )
    beta = torch.sign(prethreshold) * torch.relu(
        prethreshold.abs() - threshold
    )
    beta = beta.clamp(-v8.KALBI_BETA_BOUND, v8.KALBI_BETA_BOUND).to(
        base.dtype
    )
    feature_mask = encoded.get("feature_mask")
    if feature_mask is not None:
        beta = beta * feature_mask[:, None, :].to(beta.dtype)
    beta = torch.nan_to_num(
        beta,
        nan=0.0,
        posinf=v8.KALBI_BETA_BOUND,
        neginf=-v8.KALBI_BETA_BOUND,
    )
    return {
        "beta_initial": beta,
        "lambdas": lam,
        "penalty_d0": d0,
        "penalty_shape_features": native_shape,
        "base": base,
        "alpha": alpha,
        "prethreshold": prethreshold,
        "log_threshold_multiplier": log_threshold_multiplier,
        "gram_diagonal": gram_diagonal,
        "threshold": threshold,
    }


__all__ = ["compute_beta_initial"]
