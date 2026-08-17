#!/usr/bin/env python3
"""Exact bounded coordinate descent for canonical Direct96 penalties."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np


class CanonicalCDKernel:
    """ctypes adapter for the residual-based canonical CD kernel."""

    def __init__(self, library_path: Path) -> None:
        self.path = library_path.resolve()
        library = ctypes.CDLL(str(self.path))
        pointer = ctypes.POINTER(ctypes.c_double)
        function = library.canonical_cd_solve
        function.argtypes = [
            pointer,
            pointer,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            pointer,
            ctypes.c_double,
            pointer,
            ctypes.c_int,
            pointer,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            pointer,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        function.restype = ctypes.c_int
        self._library = library
        self._function = function

    @staticmethod
    def _pointer(value: np.ndarray) -> ctypes.POINTER(ctypes.c_double):
        return value.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def solve(
        self,
        *,
        x_columns: np.ndarray,
        y: np.ndarray,
        raw_lambda: float,
        knots: np.ndarray,
        relu_b: float,
        relu_weights: np.ndarray,
        beta_init: np.ndarray,
        max_sweeps: int,
        tolerance: float,
        zero_tolerance: float,
    ) -> dict[str, Any]:
        x_columns = np.ascontiguousarray(x_columns, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        knots = np.ascontiguousarray(knots, dtype=np.float64)
        relu_weights = np.ascontiguousarray(relu_weights, dtype=np.float64)
        beta_init = np.ascontiguousarray(beta_init, dtype=np.float64)
        if x_columns.ndim != 2:
            raise ValueError("x_columns must have shape [P,N]")
        features, rows = x_columns.shape
        if y.shape != (rows,) or beta_init.shape != (features,):
            raise ValueError("CD input shape mismatch")
        if knots.shape != relu_weights.shape:
            raise ValueError("canonical knot/weight shape mismatch")

        beta = np.empty(features, dtype=np.float64)
        sweeps = ctypes.c_int()
        last_update = ctypes.c_double()
        fixed_point = ctypes.c_double()
        status = self._function(
            self._pointer(x_columns),
            self._pointer(y),
            int(rows),
            int(features),
            float(raw_lambda),
            self._pointer(knots),
            float(relu_b),
            self._pointer(relu_weights),
            int(knots.size),
            self._pointer(beta_init),
            int(max_sweeps),
            float(tolerance),
            float(zero_tolerance),
            self._pointer(beta),
            ctypes.byref(sweeps),
            ctypes.byref(last_update),
            ctypes.byref(fixed_point),
        )
        if status < 0:
            raise RuntimeError(f"CD kernel rejected inputs with status {status}")
        return {
            "beta": beta,
            "converged": status == 0,
            "status": int(status),
            "sweeps": int(sweeps.value),
            "last_max_update": float(last_update.value),
            "fixed_point_max": float(fixed_point.value),
        }


def canonical_penalty_values(
    beta: np.ndarray,
    knots: np.ndarray,
    relu_b: float,
    relu_weights: np.ndarray,
) -> np.ndarray:
    """Evaluate rho(|beta_j|) using the exact ReLU2 representation."""
    values = np.abs(np.asarray(beta, dtype=np.float64))
    result = float(relu_b) * values
    for knot, weight in zip(knots, relu_weights, strict=True):
        result += 0.5 * float(weight) * np.maximum(values - float(knot), 0.0) ** 2
    return result


def objective_components(
    x: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    raw_lambda: float,
    knots: np.ndarray,
    relu_b: float,
    relu_weights: np.ndarray,
) -> tuple[float, float, float]:
    residual = np.asarray(y, dtype=np.float64) - np.asarray(
        x, dtype=np.float64
    ) @ np.asarray(beta, dtype=np.float64)
    data_half = 0.5 * float(np.mean(residual * residual))
    penalty = float(raw_lambda) * float(
        canonical_penalty_values(beta, knots, relu_b, relu_weights).sum()
    )
    return data_half, penalty, data_half + penalty


def projected_kkt_max(
    x: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    raw_lambda: float,
    knots: np.ndarray,
    relu_b: float,
    relu_weights: np.ndarray,
    bound: float,
) -> float:
    residual = np.asarray(y, dtype=np.float64) - np.asarray(
        x, dtype=np.float64
    ) @ beta
    gradient = -(np.asarray(x, dtype=np.float64).T @ residual) / float(x.shape[0])
    absolute = np.abs(beta)
    derivative = np.full(beta.shape, float(relu_b), dtype=np.float64)
    for knot, weight in zip(knots, relu_weights, strict=True):
        derivative += float(weight) * np.maximum(absolute - float(knot), 0.0)
    stationarity = gradient + float(raw_lambda) * derivative * np.sign(beta)
    residual_kkt = np.empty_like(stationarity)
    zero = absolute <= 1e-9
    upper = beta >= float(bound) - 1e-8
    lower = beta <= -float(bound) + 1e-8
    interior = ~(zero | upper | lower)
    residual_kkt[zero] = np.maximum(
        np.abs(gradient[zero]) - float(raw_lambda) * float(relu_b), 0.0
    )
    residual_kkt[upper] = np.maximum(stationarity[upper], 0.0)
    residual_kkt[lower] = np.maximum(-stationarity[lower], 0.0)
    residual_kkt[interior] = np.abs(stationarity[interior])
    return float(np.max(residual_kkt, initial=0.0))


def _run_to_convergence(
    *,
    kernel: CanonicalCDKernel,
    x_columns: np.ndarray,
    y: np.ndarray,
    raw_lambda: float,
    knots: np.ndarray,
    relu_b: float,
    relu_weights: np.ndarray,
    beta_init: np.ndarray,
    max_sweeps: int,
    retries: int,
    tolerance: float,
    zero_tolerance: float,
) -> dict[str, Any]:
    current = np.asarray(beta_init, dtype=np.float64).copy()
    total_sweeps = 0
    result: dict[str, Any] | None = None
    for _ in range(int(retries) + 1):
        result = kernel.solve(
            x_columns=x_columns,
            y=y,
            raw_lambda=raw_lambda,
            knots=knots,
            relu_b=relu_b,
            relu_weights=relu_weights,
            beta_init=current,
            max_sweeps=max_sweeps,
            tolerance=tolerance,
            zero_tolerance=zero_tolerance,
        )
        total_sweeps += int(result["sweeps"])
        current = result["beta"]
        if result["converged"]:
            break
    if result is None or not result["converged"]:
        fixed_point = None if result is None else result["fixed_point_max"]
        raise RuntimeError(
            f"CD did not converge after {total_sweeps} sweeps; "
            f"fixed_point={fixed_point}"
        )
    result["sweeps"] = total_sweeps
    return result


def solve_multistart_path(
    *,
    kernel: CanonicalCDKernel,
    x: np.ndarray,
    y: np.ndarray,
    raw_lambdas: np.ndarray,
    knots: np.ndarray,
    relu_b: float,
    relu_weights: np.ndarray,
    model_beta: np.ndarray,
    max_sweeps: int,
    retries: int,
    tolerance: float,
    zero_tolerance: float,
) -> dict[str, Any]:
    """Solve high-to-low lambda paths with warm, model, and zero starts."""
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    lambdas = np.asarray(raw_lambdas, dtype=np.float64)
    model = np.asarray(model_beta, dtype=np.float64)
    bound = float(knots[-1])
    clipped_model = np.clip(model, -bound, bound)
    reference = np.empty_like(model)
    reference_objective = np.empty(lambdas.size, dtype=np.float64)
    reference_kkt = np.empty(lambdas.size, dtype=np.float64)
    fixed_point = np.empty(lambdas.size, dtype=np.float64)
    sweeps = np.empty(lambdas.size, dtype=np.int64)
    selected_start: list[str] = [""] * int(lambdas.size)
    x_columns = np.ascontiguousarray(x64.T)
    previous: np.ndarray | None = None

    for index in np.argsort(lambdas)[::-1]:
        starts: list[tuple[str, np.ndarray]] = []
        if previous is not None:
            starts.append(("warm", previous))
        starts.extend(
            [
                ("kalbi_clipped", clipped_model[index]),
                ("zero", np.zeros(model.shape[1], dtype=np.float64)),
            ]
        )
        unique: list[tuple[str, np.ndarray]] = []
        for name, candidate in starts:
            if any(
                np.max(np.abs(candidate - existing)) <= 1e-12
                for _, existing in unique
            ):
                continue
            unique.append((name, candidate))

        candidates: list[tuple[float, str, dict[str, Any]]] = []
        failures: list[str] = []
        for name, initial in unique:
            try:
                result = _run_to_convergence(
                    kernel=kernel,
                    x_columns=x_columns,
                    y=y64,
                    raw_lambda=float(lambdas[index]),
                    knots=knots,
                    relu_b=relu_b,
                    relu_weights=relu_weights,
                    beta_init=initial,
                    max_sweeps=max_sweeps,
                    retries=retries,
                    tolerance=tolerance,
                    zero_tolerance=zero_tolerance,
                )
            except RuntimeError as error:
                failures.append(f"{name}: {error}")
                continue
            objective = objective_components(
                x64,
                y64,
                result["beta"],
                float(lambdas[index]),
                knots,
                relu_b,
                relu_weights,
            )[2]
            candidates.append((objective, name, result))
        if not candidates:
            raise RuntimeError(
                f"all CD starts failed at path index {index}: {failures}"
            )
        objective, name, best = min(candidates, key=lambda value: value[0])
        reference[index] = best["beta"]
        reference_objective[index] = objective
        reference_kkt[index] = projected_kkt_max(
            x64,
            y64,
            best["beta"],
            float(lambdas[index]),
            knots,
            relu_b,
            relu_weights,
            bound,
        )
        fixed_point[index] = float(best["fixed_point_max"])
        sweeps[index] = int(best["sweeps"])
        selected_start[index] = name
        previous = best["beta"].copy()

    return {
        "reference_beta": reference,
        "reference_objective": reference_objective,
        "reference_projected_kkt_max": reference_kkt,
        "reference_fixed_point_max": fixed_point,
        "reference_sweeps": sweeps,
        "reference_selected_start": np.asarray(selected_start),
        "clipped_model_beta": clipped_model,
    }


__all__ = [
    "CanonicalCDKernel",
    "canonical_penalty_values",
    "objective_components",
    "projected_kkt_max",
    "solve_multistart_path",
]
