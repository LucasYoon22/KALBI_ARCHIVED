#!/usr/bin/env python3
"""Selector V20: V19 full-P ISIS conditioned directly on context-train y.

The response branch is variable-N safe and permutation symmetric.  It projects
each standardized context response to hidden width, pools mean, population
standard deviation, and maximum over rows, and maps the pooled 3H vector back
to H.  Held-out responses are not accepted by this module's API.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from .selector_v17_k32s32 import _batch_gather_rows
from .selector_v19_k32s32 import (
    DirectPenaltyMLPQuerySelectorV19StageGain,
    V19_FEATURE_CHUNK_SIZE,
    V19_PENALTY_RESIDUAL_LAYERS,
    V17_HIDDEN_DIM,
    V17_MAX_PROBE_ROWS,
    V17_STAGES,
    V17_STAGE_GAIN_RATE,
    V17_TOPK,
)


Tensor = torch.Tensor

V20_SELECTOR_VERSION = (
    "v20_k32s32_direct96_residualtrajectory_fullp_isis32x32_"
    "stagegain0p5pct_hidden512_probe2048_chunk4096_noreplacement_"
    "localres_localmarg_cumulativejointbeta_ylinear_meanstdmax_"
    "penalty_preln_reslinear3_nopostnorm_generalizedmaxslope_v2"
)
V20_RESPONSE_POOL_INIT_STD = 1e-3
V20_NEW_STATE_KEYS = frozenset(
    {
        "response_token_projection.0.weight",
        "response_token_projection.0.bias",
        "response_pool_projection.weight",
        "response_pool_projection.bias",
    }
)


class DirectPenaltyMLPQuerySelectorV20YConditioned(
    DirectPenaltyMLPQuerySelectorV19StageGain
):
    """V19 selector plus a symmetric context-response representation."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.response_token_projection = nn.Sequential(
            nn.Linear(1, self.hidden),
            nn.GELU(),
        )
        self.response_pool_projection = nn.Linear(3 * self.hidden, self.hidden)
        nn.init.normal_(
            self.response_pool_projection.weight,
            mean=0.0,
            std=V20_RESPONSE_POOL_INIT_STD,
        )
        nn.init.zeros_(self.response_pool_projection.bias)
        self.selector_version = V20_SELECTOR_VERSION

    def encode_context(
        self,
        x_context: Tensor,
        y_context: Tensor,
        *,
        feature_mask: Tensor | None = None,
    ) -> dict[str, Tensor | int | float | str | bool]:
        trajectory = dict(
            super().encode_context(
                x_context,
                y_context,
                feature_mask=feature_mask,
            )
        )
        if y_context.ndim == 3 and y_context.shape[-1] == 1:
            y_context = y_context[..., 0]
        if y_context.ndim != 2:
            raise ValueError("V20 context y must have shape [B,N] or [B,N,1]")
        probe_row_indices = trajectory.get("probe_row_indices")
        if not torch.is_tensor(probe_row_indices):
            raise TypeError("V20 trajectory is missing probe-row indices")
        response_probe = _batch_gather_rows(
            y_context.float(), probe_row_indices.long()
        )
        if response_probe.ndim != 2:
            raise RuntimeError("V20 response probe must have shape [B,N_probe]")
        if not bool(torch.isfinite(response_probe).all().item()):
            raise RuntimeError("V20 response probe contains nonfinite values")
        trajectory.update(
            {
                "response_probe": response_probe,
                "response_source": "standardized_context_train_y_only",
                "response_pooling": "mean_population_std_max",
                "heldout_response_used": False,
            }
        )
        return trajectory

    def _build_fusion_input(
        self,
        *,
        interaction_code: Tensor,
        coefficient_code: Tensor,
        metadata_code: Tensor,
        trajectory: dict[str, Tensor | int | float | str | bool],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        response_probe = trajectory.get("response_probe")
        if not torch.is_tensor(response_probe):
            raise TypeError("V20 trajectory is missing the context-response probe")
        response_probe = response_probe.float()
        if response_probe.ndim != 2:
            raise ValueError("V20 response probe must have shape [B,N_probe]")
        if response_probe.shape[0] != interaction_code.shape[0]:
            raise ValueError("V20 response batch does not match trajectory batch")

        response_tokens = self.response_token_projection(
            response_probe.unsqueeze(-1)
        )
        response_mean = response_tokens.mean(dim=1)
        response_std = response_tokens.std(dim=1, unbiased=False)
        response_max = response_tokens.amax(dim=1)
        response_pooled = torch.cat(
            [response_mean, response_std, response_max], dim=-1
        )
        response_code = self.response_pool_projection(response_pooled)
        fusion_input = (
            interaction_code + coefficient_code + metadata_code + response_code
        )
        return fusion_input, {
            "v20_response_tokens": response_tokens,
            "v20_response_pooled": response_pooled,
            "v20_response_code": response_code,
            "v20_fusion_input": fusion_input,
        }

    def _lambda_reference_slope(
        self, encoded: dict[str, Tensor]
    ) -> Tensor:
        """Use the maximum penalty force over the bounded Direct96 domain.

        Direct96 fixes rho(6)=1, so max rho' is strictly positive and has a
        stable scale. It equals d0 for the usual L1/SCAD/MCP-like decreasing
        derivative paths, while remaining valid for ridge-like paths with
        d0 approximately zero.
        """
        derivative = encoded["drho"].float()
        if derivative.ndim < 2:
            raise ValueError("V20 penalty derivative must include the grid axis")
        return derivative.amax(dim=-1)

    def forward_from_trajectory(
        self,
        trajectory: dict[str, Tensor | int | float | str | bool],
        lambda_max: Tensor,
        *,
        return_representation: bool = False,
    ) -> dict[str, Any]:
        result = super().forward_from_trajectory(
            trajectory,
            lambda_max,
            return_representation=return_representation,
        )
        if return_representation:
            result.update(
                {
                    "v20_task_code": result["v19_task_code"],
                    "v20_penalty_code": result["v19_penalty_code"],
                    "v20_trajectory": trajectory,
                }
            )
        return result

    def architecture_report(self) -> dict[str, Any]:
        report = super().architecture_report()
        report.update(
            {
                "selector_version": self.selector_version,
                "architecture_generation": 20,
                "v20_lineage": "v19_plus_context_train_y_symmetric_pool",
                "context_response_input": "standardized_context_train_y_only",
                "heldout_response_used": False,
                "response_token_projection": [1, self.hidden],
                "response_token_activation": "GELU",
                "response_pooling": ["mean", "population_std", "max"],
                "response_pool_width": 3 * self.hidden,
                "response_pool_projection": [3 * self.hidden, self.hidden],
                "response_pool_projection_init_std": (
                    V20_RESPONSE_POOL_INIT_STD
                ),
                "lambda_ratio_definition": (
                    "lambda_times_max_drho_over_0_to_6_divided_by_lambda_max"
                ),
                "general_penalty_lambda_scaling": True,
                "ridge_like_d0_supported": True,
                "variable_n_safe": True,
                "row_permutation_symmetric_response_branch": True,
                "attention_used": False,
                "new_state_keys": sorted(V20_NEW_STATE_KEYS),
                "total_parameters": sum(p.numel() for p in self.parameters()),
            }
        )
        return report


def load_v19_shared_state_dict(
    selector: DirectPenaltyMLPQuerySelectorV20YConditioned,
    state_dict: Mapping[str, Tensor],
) -> dict[str, list[str]]:
    """Import V19 and reject every mismatch except the new response branch."""

    incompatible = selector.load_state_dict(state_dict, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing != set(V20_NEW_STATE_KEYS) or unexpected:
        raise RuntimeError(
            "V19-to-V20 state mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return {
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
    }


__all__ = [
    "DirectPenaltyMLPQuerySelectorV20YConditioned",
    "V20_NEW_STATE_KEYS",
    "V20_RESPONSE_POOL_INIT_STD",
    "V20_SELECTOR_VERSION",
    "V17_HIDDEN_DIM",
    "V17_MAX_PROBE_ROWS",
    "V17_STAGES",
    "V17_STAGE_GAIN_RATE",
    "V17_TOPK",
    "V19_FEATURE_CHUNK_SIZE",
    "V19_PENALTY_RESIDUAL_LAYERS",
    "load_v19_shared_state_dict",
]
