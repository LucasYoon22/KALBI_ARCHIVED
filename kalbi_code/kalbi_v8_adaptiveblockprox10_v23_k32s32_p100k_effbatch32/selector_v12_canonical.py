#!/usr/bin/env python3
"""V12 canonical selector: 512-wide, four-depth MLP trunks.

V12 preserves V11-canonical's exact shared Direct96 penalty builder and changes
only selector capacity.  Every actual MLP trunk (context-token, lambda, and
penalty) uses the same ResidualMLP depth contract of four.  The dataset/path
adapters and scalar/direct96 output heads remain projections, not MLP trunks.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .selector_v11_canonical import DirectPenaltyMLPQuerySelectorV11Canonical
from .utils import ResidualMLP


V12_SELECTOR_VERSION = (
    "v12_v58cheb_canonical_direct96_hidden512_all_mlp_depth4_sharedbuilder_v1"
)
V12_HIDDEN_DIM = 512
V12_MLP_DEPTH = 4


class DirectPenaltyMLPQuerySelectorV12Canonical(
    DirectPenaltyMLPQuerySelectorV11Canonical
):
    """Capacity-expanded selector with an explicit immutable architecture."""

    def __init__(
        self,
        *,
        hidden: int = V12_HIDDEN_DIM,
        depth: int = V12_MLP_DEPTH,
        **kwargs: Any,
    ) -> None:
        if int(hidden) != V12_HIDDEN_DIM:
            raise ValueError(
                f"V12 hidden dimension is fixed at {V12_HIDDEN_DIM}, got {hidden}"
            )
        if int(depth) != V12_MLP_DEPTH:
            raise ValueError(
                f"V12 MLP depth is fixed at {V12_MLP_DEPTH}, got {depth}"
            )
        super().__init__(hidden=int(hidden), depth=int(depth), **kwargs)
        # V10/V11 fixed this auxiliary token MLP at depth 2 independently of
        # the lambda/penalty trunks.  V12 makes the user's "all MLPs depth 4"
        # requirement literal by rebuilding it at the same width/depth.
        self.y_token_mlp = ResidualMLP(
            int(hidden),
            int(hidden),
            int(hidden),
            depth=int(depth),
            zero_init_last=False,
            act=nn.GELU,
        )
        self.selector_version = V12_SELECTOR_VERSION
        self.selector_hidden_dim = int(hidden)
        self.selector_mlp_depth = int(depth)

    def architecture_report(self) -> dict[str, Any]:
        def report(module: nn.Module) -> dict[str, int]:
            return {
                "residual_mlp_depth_contract": V12_MLP_DEPTH,
                "linear_modules": sum(
                    isinstance(item, nn.Linear) for item in module.modules()
                ),
                "parameters": sum(parameter.numel() for parameter in module.parameters()),
            }

        return {
            "selector_version": V12_SELECTOR_VERSION,
            "hidden_dim": V12_HIDDEN_DIM,
            "all_mlp_depth": V12_MLP_DEPTH,
            "y_token_mlp": report(self.y_token_mlp),
            "lambda_mlp": report(self.lambda_net),
            "penalty_mlp": report(self.penalty_net),
            "total_parameters": sum(parameter.numel() for parameter in self.parameters()),
        }

