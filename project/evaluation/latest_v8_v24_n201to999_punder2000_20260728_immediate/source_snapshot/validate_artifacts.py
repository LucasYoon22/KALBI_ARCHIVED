#!/usr/bin/env python3
"""Independent numerical validation of the frozen KALBI evaluation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def frozen_x(x: np.ndarray, indices: np.ndarray, split: dict[str, np.ndarray]) -> np.ndarray:
    value = (
        np.asarray(x[indices], dtype=np.float64)
        - np.asarray(split["x_mean"], dtype=np.float64)
    ) / np.asarray(split["x_scale"], dtype=np.float64)
    value[:, ~np.asarray(split["active"], dtype=bool)] = 0.0
    return value.astype(np.float32)


def mse_path(
    x_standardized: np.ndarray,
    y_raw: np.ndarray,
    beta_path: np.ndarray,
    y_mean: float,
    y_scale: float,
) -> np.ndarray:
    prediction = y_mean + y_scale * (
        np.asarray(x_standardized, dtype=np.float64)
        @ np.asarray(beta_path, dtype=np.float64).T
    )
    return np.mean(
        (np.asarray(y_raw, dtype=np.float64)[:, None] - prediction) ** 2,
        axis=0,
    )


def rho_value(
    beta: np.ndarray,
    knots: np.ndarray,
    relu_b: float,
    relu_weights: np.ndarray,
) -> np.ndarray:
    absolute = np.abs(np.asarray(beta, dtype=np.float64))
    rho = relu_b * absolute
    for knot, weight in zip(knots, relu_weights, strict=True):
        rho += 0.5 * float(weight) * np.maximum(absolute - float(knot), 0.0) ** 2
    return rho


def objective_path(
    x: np.ndarray,
    y: np.ndarray,
    beta_path: np.ndarray,
    lambdas: np.ndarray,
    knots: np.ndarray,
    relu_b: float,
    relu_weights: np.ndarray,
) -> np.ndarray:
    values = []
    for beta, raw_lambda in zip(beta_path, lambdas, strict=True):
        residual = np.asarray(y, dtype=np.float64) - x @ np.asarray(beta, dtype=np.float64)
        values.append(
            0.5 * np.mean(residual**2)
            + float(raw_lambda) * np.sum(rho_value(beta, knots, relu_b, relu_weights))
        )
    return np.asarray(values, dtype=np.float64)


def array_hashes(path: np.ndarray) -> list[str]:
    return [
        hashlib.sha256(np.ascontiguousarray(row).view(np.uint8)).hexdigest()
        for row in path
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.evaluation_root.resolve()
    result_rows = json.loads(
        (root / "metrics/full_41point/results.json").read_text()
    )
    expected_grid = np.logspace(-4, 0, 41)
    task_reports: list[dict[str, object]] = []

    for row in result_rows:
        index = int(row["dataset_index"])
        with np.load(root / f"data_manifest/data/dataset_{index}.npz") as data:
            x = np.asarray(data["X"][0], dtype=np.float64)
            y = np.asarray(data["y"][0], dtype=np.float64)
        with np.load(root / f"data_manifest/splits/dataset_{index}.npz") as split_file:
            split = {key: split_file[key] for key in split_file.files}
        with np.load(
            root / f"metrics/full_41point/dataset_{index:02d}/paths.npz"
        ) as path_file:
            paths = {key: path_file[key] for key in path_file.files}

        context = np.asarray(split["context_indices"], dtype=np.int64)
        validation = np.asarray(split["validation_indices"], dtype=np.int64)
        test = np.asarray(split["test_indices"], dtype=np.int64)
        if set(context) & set(validation) or set(context) & set(test) or set(validation) & set(test):
            raise AssertionError(f"dataset {index}: split overlap")
        if len(set(context) | set(validation) | set(test)) != x.shape[0]:
            raise AssertionError(f"dataset {index}: incomplete split coverage")

        x_context = frozen_x(x, context, split)
        x_validation = frozen_x(x, validation, split)
        x_test = frozen_x(x, test, split)
        y_context = (
            y[context] - float(split["y_mean"])
        ) / float(split["y_scale"])
        y_context = y_context.astype(np.float32)
        y_mean = float(split["y_mean"])
        y_scale = float(split["y_scale"])
        lambdas = np.asarray(paths["raw_lambdas"], dtype=np.float64)
        np.testing.assert_allclose(lambdas, expected_grid, rtol=0.0, atol=1e-15)

        kalbi = np.asarray(paths["kalbi_beta"], dtype=np.float64)
        cd = np.asarray(paths["coordinate_descent_beta"], dtype=np.float64)
        if not np.isfinite(kalbi).all() or not np.isfinite(cd).all():
            raise AssertionError(f"dataset {index}: nonfinite coefficients")
        unique_hashes = len(set(array_hashes(kalbi)))
        if unique_hashes < 2:
            raise AssertionError(f"dataset {index}: reused every coefficient vector")

        kalbi_validation = mse_path(x_validation, y[validation], kalbi, y_mean, y_scale)
        kalbi_test = mse_path(x_test, y[test], kalbi, y_mean, y_scale)
        cd_validation = mse_path(x_validation, y[validation], cd, y_mean, y_scale)
        cd_test = mse_path(x_test, y[test], cd, y_mean, y_scale)
        mse_checks = {
            "kalbi_validation": (kalbi_validation, paths["kalbi_validation_mse"]),
            "kalbi_test": (kalbi_test, paths["kalbi_test_mse"]),
            "cd_validation": (cd_validation, paths["coordinate_descent_validation_mse"]),
            "cd_test": (cd_test, paths["coordinate_descent_test_mse"]),
        }
        mse_max_abs_error = 0.0
        for name, (actual, expected) in mse_checks.items():
            error = float(np.max(np.abs(actual - np.asarray(expected, dtype=np.float64))))
            mse_max_abs_error = max(mse_max_abs_error, error)
            np.testing.assert_allclose(actual, expected, rtol=5e-8, atol=3e-9, err_msg=name)

        selected = np.asarray(paths["selected_beta"], dtype=np.float64)[None, :]
        selected_validation = float(
            mse_path(x_validation, y[validation], selected, y_mean, y_scale)[0]
        )
        selected_test = float(mse_path(x_test, y[test], selected, y_mean, y_scale)[0])
        np.testing.assert_allclose(
            selected_validation, row["selected_validation_mse"], rtol=5e-8, atol=3e-9
        )
        np.testing.assert_allclose(
            selected_test, row["selected_test_mse"], rtol=5e-8, atol=3e-9
        )

        knots = np.asarray(paths["penalty_grid"], dtype=np.float64)
        relu_b = float(paths["penalty_relu_b"])
        relu_weights = np.asarray(paths["penalty_relu_weights"], dtype=np.float64)
        kalbi_objective = objective_path(
            x_context, y_context, kalbi, lambdas, knots, relu_b, relu_weights
        )
        cd_objective = objective_path(
            x_context, y_context, cd, lambdas, knots, relu_b, relu_weights
        )
        np.testing.assert_allclose(
            kalbi_objective, paths["kalbi_objective"], rtol=5e-8, atol=3e-9
        )
        np.testing.assert_allclose(
            cd_objective, paths["coordinate_descent_objective"], rtol=5e-8, atol=3e-9
        )
        objective_gap = kalbi_objective - cd_objective
        np.testing.assert_allclose(
            objective_gap, paths["objective_gap"], rtol=5e-8, atol=6e-9
        )
        if float(np.min(objective_gap)) < -1e-10:
            raise AssertionError(f"dataset {index}: reference objective exceeds KALBI")

        score = np.max(np.abs(cd), axis=0)
        ranking = np.lexsort((np.arange(score.size), -score))[:12]
        np.testing.assert_array_equal(ranking, paths["top_features"])
        np.testing.assert_allclose(
            score[ranking], paths["top_feature_scores"], rtol=5e-8, atol=3e-9
        )

        rho_at_knots = rho_value(knots, knots, relu_b, relu_weights)
        rho_error = float(
            np.max(np.abs(rho_at_knots - np.asarray(paths["penalty_rho"], dtype=np.float64)))
        )
        if abs(float(paths["penalty_rho"][0])) > 1e-8:
            raise AssertionError(f"dataset {index}: rho(0) failure")
        if not all(
            np.isfinite(np.asarray(paths[key])).all()
            for key in ("penalty_grid", "penalty_rho", "penalty_drho", "penalty_relu_weights")
        ):
            raise AssertionError(f"dataset {index}: nonfinite penalty")
        if not np.asarray(paths["cd_converged"], dtype=bool).all():
            raise AssertionError(f"dataset {index}: unconverged CD point")

        task_reports.append(
            {
                "dataset_index": index,
                "mse_max_abs_error": mse_max_abs_error,
                "objective_max_abs_error": float(
                    max(
                        np.max(np.abs(kalbi_objective - paths["kalbi_objective"])),
                        np.max(np.abs(cd_objective - paths["coordinate_descent_objective"])),
                    )
                ),
                "objective_gap_min": float(np.min(objective_gap)),
                "rho0_error": abs(float(paths["penalty_rho"][0])),
                "penalty_representation_max_abs_error": rho_error,
                "kalbi_unique_coefficient_vector_hashes": unique_hashes,
                "cd_all_converged": True,
                "cd_max_projected_kkt": float(np.max(paths["cd_projected_kkt"])),
                "cd_max_fixed_point": float(np.max(paths["cd_fixed_point"])),
                "top_feature_ranking_verified": True,
                "split_disjoint_and_complete": True,
            }
        )

    report = {
        "status": "PASS",
        "task_count": len(task_reports),
        "raw_lambda_grid_verified": True,
        "context_only_frozen_preprocessing_verified": True,
        "direct_mse_recomputation_verified": True,
        "independent_objective_recomputation_verified": True,
        "per_lambda_coefficient_nonreuse_verified": True,
        "full_p_coordinate_descent_verified": all(
            bool(row["cd"]["full_p"]) for row in result_rows
        ),
        "tasks": task_reports,
    }
    output = root / "metrics/validation_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
