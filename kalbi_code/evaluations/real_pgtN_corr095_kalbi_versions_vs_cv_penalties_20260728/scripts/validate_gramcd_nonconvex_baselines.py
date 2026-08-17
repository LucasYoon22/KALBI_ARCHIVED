#!/usr/bin/env python3
"""Validate the exact objective and KKT contract used by GramCD baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from evaluate_real_pgtN_cv_baselines_gramcd import (
    fit_path,
    objective_and_kkt,
    penalty_value,
)
from kalbi_v8_adaptiveblockprox10_v23_k32s32_p100k_effbatch32.benchmark_kalbi_v3_cv_lasso_scad import (
    _lambda_max,
)
from real_pgtN_common import atomic_json


def brute_scalar(
    method: str,
    z: float,
    raw_lambda: float,
    args: SimpleNamespace,
) -> tuple[float, float]:
    grid = np.linspace(-4.0, 4.0, 400_001, dtype=np.float64)
    objective = (
        0.5 * (grid - float(z)) ** 2
        + penalty_value(method, np.abs(grid), raw_lambda, args)
    )
    index = int(np.argmin(objective))
    return float(grid[index]), float(objective[index])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args_cli = parser.parse_args()
    args = SimpleNamespace(
        lasso_tolerance=1e-10,
        lasso_max_iter=100000,
        scad_gamma=3.7,
        mcp_gamma=3.0,
        nonconvex_tolerance=1e-10,
        nonconvex_max_iter=1000,
        nonconvex_max_epochs=300000,
    )

    z = np.asarray(
        [-2.0, -1.2, -0.7, -0.3, 0.0, 0.25, 0.65, 1.15, 1.9],
        dtype=np.float64,
    )
    n = z.size
    x = np.sqrt(n) * np.eye(n, dtype=np.float64)
    y = np.sqrt(n) * z
    lambda_max = _lambda_max(x, y)
    ratios = np.asarray([1.0, 0.5, 0.25], dtype=np.float64)
    raw_lambda = float(lambda_max * ratios[-1])
    results: dict[str, Any] = {}

    for method in ("lasso", "scad", "mcp"):
        beta_path, diagnostics = fit_path(method, x, y, ratios, args)
        beta = beta_path[-1]
        brute = np.asarray(
            [
                brute_scalar(method, value, raw_lambda, args)[0]
                for value in z
            ],
            dtype=np.float64,
        )
        max_beta_error = float(np.max(np.abs(beta - brute)))
        objective, kkt = objective_and_kkt(
            method, x, y, beta, raw_lambda, args
        )
        zero_objective, _ = objective_and_kkt(
            method, x, y, np.zeros_like(beta), raw_lambda, args
        )
        endpoint_nnz = int(np.count_nonzero(np.abs(beta_path[0]) > 1e-8))
        tolerance = 2e-4 if method == "lasso" else 3e-3
        passed = (
            max_beta_error <= tolerance
            and kkt <= 5e-5
            and objective <= zero_objective + 1e-9
            and endpoint_nnz == 0
        )
        results[method] = {
            "status": "PASS" if passed else "FAIL",
            "lambda_max": float(lambda_max),
            "raw_lambda": raw_lambda,
            "max_beta_error_vs_dense_global_scalar_search": max_beta_error,
            "kkt_max": float(kkt),
            "objective": float(objective),
            "zero_objective": float(zero_objective),
            "lambda_max_zero_nnz": endpoint_nnz,
            "solver_diagnostics": diagnostics,
        }

    overall = all(value["status"] == "PASS" for value in results.values())
    manifest = {
        "status": "PASS" if overall else "FAIL",
        "solver": {
            "lasso": "sklearn coordinate descent",
            "scad": "skglm GramCD",
            "mcp": "skglm GramCD",
        },
        "design": "orthogonal X=sqrt(N)I; dense scalar global search",
        "objective_contract": (
            "0.5/N ||y-X beta||^2 + sum_j p_lambda(|beta_j|)"
        ),
        "lambda_ratios": ratios.tolist(),
        "methods": results,
    }
    atomic_json(args_cli.output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
