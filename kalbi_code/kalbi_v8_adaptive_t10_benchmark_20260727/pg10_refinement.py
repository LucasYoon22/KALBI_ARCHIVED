"""Invoke the unchanged V6 eta4x exact-PG contract on shared beta_initial."""

from __future__ import annotations

import importlib
from typing import Any

import torch


Tensor = torch.Tensor
PG_STEPS = 10
V6_MODULE = (
    "kalbi_v6_exactprox10_eta4x_v23_independentraw_"
    "k32s32_p100k_effbatch32.model"
)


def _v6_module():
    return importlib.import_module(V6_MODULE)


def prepare_pg10_model(model: torch.nn.Module) -> None:
    if not hasattr(model, "_kalbi_v6_penalty_grid"):
        model.register_buffer(
            "_kalbi_v6_penalty_grid",
            model._kalbi_v7_penalty_grid.detach().clone(),
            persistent=False,
        )


def run_pg10_refinement(
    model: torch.nn.Module,
    encoded: dict[str, Any],
    beta_initial: Tensor,
    lambdas: Tensor,
    penalty_d0: Tensor,
    penalty_shape_features: Tensor,
) -> tuple[Tensor, list[dict[str, Tensor]]]:
    prepare_pg10_model(model)
    module = _v6_module()
    beta = beta_initial
    history: list[dict[str, Tensor]] = []
    for _ in range(PG_STEPS):
        result = module._refine_beta_kalbi_v6(
            model,
            encoded,
            beta,
            lambdas,
            penalty_d0,
            penalty_shape_features,
        )
        beta = result["beta"]
        history.append(result)
    return beta, history


def pg10_call_counts(
    history: list[dict[str, Tensor]],
) -> tuple[int, int]:
    selection_evaluations = 0
    for step in history:
        selection_evaluations += 1 + int(
            step["backtracks"].detach().max().item()
        )
        selection_evaluations += int(
            bool(step["trace_fallback"].detach().any().item())
        )
    exact_prox_calls = selection_evaluations + len(history)
    return exact_prox_calls, selection_evaluations


__all__ = [
    "PG_STEPS",
    "pg10_call_counts",
    "prepare_pg10_model",
    "run_pg10_refinement",
]

