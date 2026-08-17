#!/usr/bin/env python3
"""V23: V22 representation with an independent public raw-lambda head."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from .selector_v22_v20geometry_scaled import (
    DirectPenaltyMLPQuerySelectorV22V20GeometryScaled,
    V17_HIDDEN_DIM,
    V17_MAX_PROBE_ROWS,
    V17_STAGES,
    V17_STAGE_GAIN_RATE,
    V17_TOPK,
    V19_FEATURE_CHUNK_SIZE,
    V19_PENALTY_RESIDUAL_LAYERS,
    V22_SCALER_VERSION,
)


Tensor = torch.Tensor

V23_SELECTOR_VERSION = (
    "v23_v22geometry_k32s32_independent_rawlambda1e-6to10_"
    "d0_zeroKKT_telemetry_only_v1"
)
V23_RAW_LAMBDA_MIN = 1e-6
V23_RAW_LAMBDA_MAX = 10.0
V23_INITIAL_RAW_LAMBDA = 0.04
V23_LAMBDA_OUTPUT_CONTRACT = (
    "independent_raw_lambda_head_no_d0_or_maxdrho_inverse_conversion_v1"
)
V23_LAMBDA_HEAD_STATE_KEYS = frozenset(
    {"lambda_head.weight", "lambda_head.bias"}
)


class DirectPenaltyMLPQuerySelectorV23IndependentRawLambda(
    DirectPenaltyMLPQuerySelectorV22V20GeometryScaled
):
    """Keep V22 task/penalty features while decoupling lambda from penalty scale."""

    def __init__(
        self,
        *,
        raw_lambda_min: float = V23_RAW_LAMBDA_MIN,
        raw_lambda_max: float = V23_RAW_LAMBDA_MAX,
        initial_raw_lambda: float = V23_INITIAL_RAW_LAMBDA,
        **kwargs: Any,
    ) -> None:
        if not 0.0 < raw_lambda_min < raw_lambda_max:
            raise ValueError("V23 raw-lambda support must be positive and ordered")
        if not raw_lambda_min < initial_raw_lambda < raw_lambda_max:
            raise ValueError("V23 initial raw lambda must lie inside its support")

        # The inherited V22 forward is used only to build the common
        # representation and canonical penalty. Its ratio result is overwritten.
        kwargs.setdefault("lambda_min_ratio", 1e-6)
        kwargs.setdefault("lambda_max_ratio", 1.0)
        kwargs.setdefault("initial_lambda_ratio", 0.1)
        super().__init__(**kwargs)
        self.raw_lambda_min = float(raw_lambda_min)
        self.raw_lambda_max = float(raw_lambda_max)
        self.initial_raw_lambda = float(initial_raw_lambda)
        self.selector_version = V23_SELECTOR_VERSION
        self.reset_raw_lambda_head()

    @torch.no_grad()
    def reset_raw_lambda_head(
        self, initial_raw_lambda: float | None = None
    ) -> None:
        initial = (
            self.initial_raw_lambda
            if initial_raw_lambda is None
            else float(initial_raw_lambda)
        )
        if not self.raw_lambda_min < initial < self.raw_lambda_max:
            raise ValueError("raw-lambda reset value lies outside V23 support")
        unit = (
            (math.log(initial) - math.log(self.raw_lambda_min))
            / (math.log(self.raw_lambda_max) - math.log(self.raw_lambda_min))
        )
        unit = min(max(unit, 1e-6), 1.0 - 1e-6)
        bias = math.log(unit / (1.0 - unit))
        torch.nn.init.zeros_(self.lambda_head.weight)
        torch.nn.init.constant_(self.lambda_head.bias, bias)

    def forward_from_trajectory(
        self,
        trajectory: dict[str, Tensor | int | float | str | bool],
        lambda_max: Tensor,
        *,
        return_representation: bool = False,
    ) -> dict[str, Any]:
        # V22 computes the shared representation and canonical penalty. The
        # inherited slope-normalized lambda is discarded before returning.
        result = super().forward_from_trajectory(
            trajectory,
            lambda_max,
            return_representation=True,
        )
        task_code = result["v22_task_code"]
        lambda_logit = self.lambda_head(task_code).squeeze(-1)
        lambda_unit = torch.sigmoid(lambda_logit)
        raw_lambda = torch.exp(
            math.log(self.raw_lambda_min)
            + (
                math.log(self.raw_lambda_max)
                - math.log(self.raw_lambda_min)
            )
            * lambda_unit
        ).clamp(self.raw_lambda_min, self.raw_lambda_max)

        lambda_max_safe = lambda_max.float().clamp_min(1e-12)
        d0 = result["penalty_d0"].reshape(-1).float()
        max_drho = result["penalty_drho"].float().amax(dim=-1)
        raw_lambda_over_lambda_max = raw_lambda / lambda_max_safe
        zero_kkt_ratio = raw_lambda * d0 / lambda_max_safe
        force_ratio = raw_lambda * max_drho / lambda_max_safe

        result.update(
            {
                "lambdas": raw_lambda[:, None],
                "raw_lambda": raw_lambda,
                # Kept for trainer compatibility; V23 defines this key exactly.
                "lambda_ratio": raw_lambda_over_lambda_max,
                "raw_lambda_over_lambda_max": raw_lambda_over_lambda_max,
                "effective_lambda_ratio": zero_kkt_ratio,
                "force_ratio": force_ratio,
                "zero_kkt_ratio": zero_kkt_ratio,
                "raw_lambda_logit": lambda_logit,
                "raw_lambda_unit": lambda_unit,
                "penalty_reference_slope": max_drho,
                "penalty_ridge_branch": d0 <= 1e-7,
                "penalty_encoding_version": self.selector_version,
                "lambda_output_contract": V23_LAMBDA_OUTPUT_CONTRACT,
            }
        )

        if not return_representation:
            for key in list(result):
                if key.startswith(("v17_", "v19_", "v20_", "v21_", "v22_")):
                    result.pop(key)
        return result

    def architecture_report(self) -> dict[str, Any]:
        report = super().architecture_report()
        report.update(
            {
                "selector_version": self.selector_version,
                "architecture_generation": 23,
                "lineage": "v22_geometry_plus_independent_raw_lambda_head",
                "v20_lambda_contract_preserved": False,
                "raw_lambda_support": [
                    self.raw_lambda_min,
                    self.raw_lambda_max,
                ],
                "initial_raw_lambda": self.initial_raw_lambda,
                "lambda_output_contract": V23_LAMBDA_OUTPUT_CONTRACT,
                "lambda_max_role": "representation_and_telemetry_only",
                "d0_role": "zero_kkt_telemetry_and_beta_solver_penalty_only",
                "max_drho_role": "force_ratio_telemetry_only",
                "raw_lambda_depends_on_penalty_shape": False,
                "total_parameters": sum(p.numel() for p in self.parameters()),
            }
        )
        return report


@torch.no_grad()
def load_v22_state_without_lambda_head(
    selector: DirectPenaltyMLPQuerySelectorV23IndependentRawLambda,
    state_dict: Mapping[str, Tensor],
    *,
    initial_raw_lambda: float = V23_INITIAL_RAW_LAMBDA,
) -> dict[str, Any]:
    """Strictly import every V22 tensor except the semantically changed head."""

    expected_keys = set(selector.state_dict())
    observed_keys = set(state_dict)
    if expected_keys != observed_keys:
        raise RuntimeError(
            "V22-to-V23 state inventory drift: "
            f"missing={sorted(expected_keys - observed_keys)}, "
            f"unexpected={sorted(observed_keys - expected_keys)}"
        )
    shared_state = {
        key: value
        for key, value in state_dict.items()
        if key not in V23_LAMBDA_HEAD_STATE_KEYS
    }
    incompatible = selector.load_state_dict(shared_state, strict=False)
    if (
        set(incompatible.missing_keys) != set(V23_LAMBDA_HEAD_STATE_KEYS)
        or incompatible.unexpected_keys
    ):
        raise RuntimeError(
            "V22-to-V23 shared-state mismatch: "
            f"missing={sorted(incompatible.missing_keys)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    selector.reset_raw_lambda_head(initial_raw_lambda)
    loaded = selector.state_dict()
    mismatched = [
        key
        for key, value in state_dict.items()
        if key not in V23_LAMBDA_HEAD_STATE_KEYS
        and not torch.equal(loaded[key].cpu(), value.cpu())
    ]
    if mismatched:
        raise RuntimeError(
            f"V22-to-V23 shared tensors changed during import: {mismatched}"
        )
    return {
        "source_tensor_count": len(state_dict),
        "shared_tensor_count": len(shared_state),
        "reinitialized_keys": sorted(V23_LAMBDA_HEAD_STATE_KEYS),
        "initial_raw_lambda": float(initial_raw_lambda),
        "strict_shared_tensor_audit": True,
    }


__all__ = [
    "DirectPenaltyMLPQuerySelectorV23IndependentRawLambda",
    "V17_HIDDEN_DIM",
    "V17_MAX_PROBE_ROWS",
    "V17_STAGES",
    "V17_STAGE_GAIN_RATE",
    "V17_TOPK",
    "V19_FEATURE_CHUNK_SIZE",
    "V19_PENALTY_RESIDUAL_LAYERS",
    "V22_SCALER_VERSION",
    "V23_INITIAL_RAW_LAMBDA",
    "V23_LAMBDA_HEAD_STATE_KEYS",
    "V23_LAMBDA_OUTPUT_CONTRACT",
    "V23_RAW_LAMBDA_MAX",
    "V23_RAW_LAMBDA_MIN",
    "V23_SELECTOR_VERSION",
    "load_v22_state_without_lambda_head",
]
