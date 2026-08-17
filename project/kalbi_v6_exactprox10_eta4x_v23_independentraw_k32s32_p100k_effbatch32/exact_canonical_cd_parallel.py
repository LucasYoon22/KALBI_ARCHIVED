#!/usr/bin/env python3
"""Parallel multistart wrapper for the exact canonical CD kernel."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
from typing import Any

import numpy as np

from .exact_canonical_cd import (
    CanonicalCDKernel,
    _run_to_convergence,
    objective_components,
    projected_kkt_max,
)


def solve_multistart_path_parallel(
    *,
    library_path: Path,
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
    workers: int,
    include_warm: bool = True,
) -> dict[str, Any]:
    """Match the serial warm/model/zero search while parallelizing fixed starts."""
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    lambdas = np.asarray(raw_lambdas, dtype=np.float64)
    model = np.asarray(model_beta, dtype=np.float64)
    bound = float(knots[-1])
    clipped_model = np.clip(model, -bound, bound)
    zero = np.zeros(model.shape[1], dtype=np.float64)
    x_columns = np.ascontiguousarray(x64.T)
    thread_state = threading.local()

    def kernel_for_thread() -> CanonicalCDKernel:
        kernel = getattr(thread_state, "kernel", None)
        if kernel is None:
            kernel = CanonicalCDKernel(library_path)
            thread_state.kernel = kernel
        return kernel

    def run_start(
        index: int,
        name: str,
        initial: np.ndarray,
    ) -> tuple[int, str, np.ndarray, dict[str, Any] | None, str | None]:
        try:
            result = _run_to_convergence(
                kernel=kernel_for_thread(),
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
            return index, name, initial, result, None
        except RuntimeError as error:
            return index, name, initial, None, str(error)

    fixed: dict[tuple[int, str], tuple[np.ndarray, dict[str, Any] | None, str | None]] = {}
    jobs: list[tuple[int, str, np.ndarray]] = []
    for index in range(lambdas.size):
        jobs.append((index, "kalbi_clipped", clipped_model[index]))
        if np.max(np.abs(clipped_model[index])) > 1e-12:
            jobs.append((index, "zero", zero))

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = [
            executor.submit(run_start, index, name, initial)
            for index, name, initial in jobs
        ]
        for future in as_completed(futures):
            index, name, initial, result, error = future.result()
            fixed[(index, name)] = (initial, result, error)

    reference = np.empty_like(model)
    reference_objective = np.empty(lambdas.size, dtype=np.float64)
    reference_kkt = np.empty(lambdas.size, dtype=np.float64)
    fixed_point = np.empty(lambdas.size, dtype=np.float64)
    sweeps = np.empty(lambdas.size, dtype=np.int64)
    selected_start: list[str] = [""] * int(lambdas.size)
    previous: np.ndarray | None = None

    for index in np.argsort(lambdas)[::-1]:
        ordered: list[
            tuple[str, np.ndarray, dict[str, Any] | None, str | None]
        ] = []
        if include_warm and previous is not None:
            _, _, warm_initial, warm_result, warm_error = run_start(
                int(index), "warm", previous
            )
            ordered.append(
                ("warm", warm_initial, warm_result, warm_error)
            )

        model_initial, model_result, model_error = fixed[
            (int(index), "kalbi_clipped")
        ]
        ordered.append(
            ("kalbi_clipped", model_initial, model_result, model_error)
        )
        zero_entry = fixed.get((int(index), "zero"))
        if zero_entry is not None:
            zero_initial, zero_result, zero_error = zero_entry
            ordered.append(("zero", zero_initial, zero_result, zero_error))

        unique: list[
            tuple[str, np.ndarray, dict[str, Any] | None, str | None]
        ] = []
        for candidate in ordered:
            if any(
                np.max(np.abs(candidate[1] - existing[1])) <= 1e-12
                for existing in unique
            ):
                continue
            unique.append(candidate)

        candidates: list[tuple[float, str, dict[str, Any]]] = []
        failures: list[str] = []
        for name, _, result, error in unique:
            if result is None:
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


__all__ = ["solve_multistart_path_parallel"]
