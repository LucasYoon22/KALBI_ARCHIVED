#!/usr/bin/env python3
"""Check cached-Gram path parity with skglm GramCD.solve()."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from skglm.penalties import MCPenalty, SCAD
from skglm.solvers import GramCD

from evaluate_real_pgtN_cv_baselines_gramcd import (
    gram_nonconvex_path,
    objective_and_kkt,
)
from kalbi_v8_adaptiveblockprox10_v23_k32s32_p100k_effbatch32.benchmark_kalbi_v3_cv_lasso_scad import (
    _lambda_max,
    _lasso_path,
)
from real_pgtN_common import atomic_json


def original_path(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    ratios: np.ndarray,
    *,
    gamma: float,
    tolerance: float,
    max_iter: int,
) -> np.ndarray:
    if method == "lasso":
        path, _, _, _ = _lasso_path(
            x,
            y,
            ratios,
            tolerance=tolerance,
            max_iter=max_iter,
        )
        return np.asarray(path, dtype=np.float64)
    lambda_max = _lambda_max(x, y)
    solver = GramCD(
        max_iter=max_iter,
        use_acc=False,
        greedy_cd=True,
        tol=tolerance,
        fit_intercept=False,
        warm_start=True,
        verbose=0,
    )
    coefficient = np.zeros(x.shape[1], dtype=np.float64)
    prediction = np.zeros(x.shape[0], dtype=np.float64)
    path = []
    for ratio in ratios:
        raw_lambda = lambda_max * float(ratio)
        penalty = (
            SCAD(raw_lambda, gamma)
            if method == "scad"
            else MCPenalty(raw_lambda, gamma)
        )
        coefficient, _, _ = solver.solve(
            x, y, None, penalty, coefficient, prediction
        )
        coefficient = np.asarray(coefficient, dtype=np.float64)
        prediction = x @ coefficient
        path.append(coefficient.copy())
    return np.asarray(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args_cli = parser.parse_args()
    rng = np.random.default_rng(20260728)
    x = rng.normal(size=(80, 24))
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    true = np.zeros(24)
    true[[0, 3, 8, 15]] = [1.1, -0.7, 0.45, -0.3]
    y = x @ true + 0.35 * rng.normal(size=80)
    y = (y - y.mean()) / y.std()
    ratios = np.asarray([1.0, 0.7, 0.4, 0.2, 0.1, 0.03])
    namespace = SimpleNamespace(scad_gamma=3.7, mcp_gamma=3.0)
    results = {}

    for method, gamma in (("lasso", 0.0), ("scad", 3.7), ("mcp", 3.0)):
        started = time.perf_counter()
        original = original_path(
            method,
            x,
            y,
            ratios,
            gamma=gamma,
            tolerance=1e-10,
            max_iter=1000,
        )
        original_seconds = time.perf_counter() - started
        started = time.perf_counter()
        cached, stops, iterations = gram_nonconvex_path(
            method,
            x,
            y,
            ratios,
            gamma=gamma,
            tolerance=1e-10,
            max_iter=1000,
        )
        cached_seconds = time.perf_counter() - started
        coefficient_error = float(np.max(np.abs(original - cached)))
        objective_error = 0.0
        kkt_error = 0.0
        lambda_max = _lambda_max(x, y)
        for index, ratio in enumerate(ratios):
            raw_lambda = float(lambda_max * ratio)
            original_objective, original_kkt = objective_and_kkt(
                method,
                x,
                y,
                original[index],
                raw_lambda,
                namespace,
            )
            cached_objective, cached_kkt = objective_and_kkt(
                method,
                x,
                y,
                cached[index],
                raw_lambda,
                namespace,
            )
            objective_error = max(
                objective_error, abs(original_objective - cached_objective)
            )
            kkt_error = max(kkt_error, abs(original_kkt - cached_kkt))
        coefficient_tolerance = 1e-8 if method == "lasso" else 1e-10
        passed = (
            coefficient_error <= coefficient_tolerance
            and objective_error <= 1e-12
            and kkt_error <= coefficient_tolerance
            and float(np.max(stops)) <= 1e-8
        )
        results[method] = {
            "status": "PASS" if passed else "FAIL",
            "max_coefficient_error": coefficient_error,
            "max_objective_error": objective_error,
            "max_kkt_error": kkt_error,
            "max_cached_stop_criterion": float(np.max(stops)),
            "max_cached_iterations": int(np.max(iterations)),
            "original_seconds": original_seconds,
            "cached_seconds": cached_seconds,
        }

    status = (
        "PASS"
        if all(value["status"] == "PASS" for value in results.values())
        else "FAIL"
    )
    report = {
        "status": status,
        "contract": "reuse X.T@X and X.T@y; preserve GramCD epochs and warm start",
        "methods": results,
    }
    atomic_json(args_cli.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
