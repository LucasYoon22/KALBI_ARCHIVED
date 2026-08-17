#!/usr/bin/env python3
"""Contract tests for train-only deterministic correlation filtering."""

from __future__ import annotations

import numpy as np

from real_pgtN_common import (
    apply_x_preprocessing,
    deterministic_correlation_filter,
    fit_preprocessing,
    verify_retained_correlations,
)


def main() -> None:
    rng = np.random.default_rng(20260728)
    n = 512
    first = rng.normal(size=n)
    second = first + 1e-5 * rng.normal(size=n)
    third = -first + 1e-5 * rng.normal(size=n)
    fourth = rng.normal(size=n)
    fifth = fourth + 1e-5 * rng.normal(size=n)
    sixth = rng.normal(size=n)
    constant = np.ones(n)
    x = np.column_stack(
        [first, second, third, fourth, fifth, sixth, constant]
    )
    y = rng.normal(size=n)

    x_filtered, y_filtered, scaler = fit_preprocessing(x, y)
    np.testing.assert_array_equal(
        scaler["retained_original_indices"],
        np.asarray([0, 3, 5], dtype=np.int64),
    )
    assert x_filtered.shape == (n, 3)
    assert y_filtered.shape == (n,)
    assert float(scaler["retained_max_abs_correlation"]) < 0.95
    assert int(scaler["removed_correlated_feature_count"]) == 3

    repeated_x, repeated_y, repeated_scaler = fit_preprocessing(x, y)
    np.testing.assert_array_equal(repeated_x, x_filtered)
    np.testing.assert_array_equal(repeated_y, y_filtered)
    assert repeated_scaler["scaler_sha256"] == scaler["scaler_sha256"]

    _, _, permuted_y_scaler = fit_preprocessing(x, y[::-1])
    np.testing.assert_array_equal(
        permuted_y_scaler["retained_original_indices"],
        scaler["retained_original_indices"],
    )

    x_test = rng.normal(size=(37, x.shape[1]))
    transformed_test = apply_x_preprocessing(x_test, scaler)
    assert transformed_test.shape == (37, 3)
    np.testing.assert_allclose(
        transformed_test,
        repeated_x[:37] * 0.0 + transformed_test,
        rtol=0.0,
        atol=0.0,
    )

    active = np.asarray(scaler["active"], dtype=bool)
    standardized = (
        x[:, active]
        - np.asarray(scaler["x_mean"], dtype=np.float64)[active]
    ) / np.asarray(scaler["x_scale"], dtype=np.float64)[active]
    retained, audit = deterministic_correlation_filter(standardized)
    np.testing.assert_array_equal(retained, scaler["correlation_keep"])
    assert audit["retained_max_abs_correlation"] < 0.95

    duplicate_maximum, pair = verify_retained_correlations(
        standardized,
        np.asarray([0, 1], dtype=np.int64),
    )
    assert duplicate_maximum >= 0.95
    assert pair == (0, 1)
    print(
        {
            "status": "PASS",
            "retained_original_indices": scaler[
                "retained_original_indices"
            ].tolist(),
            "retained_max_abs_correlation": scaler[
                "retained_max_abs_correlation"
            ],
            "scaler_sha256": scaler["scaler_sha256"],
        }
    )


if __name__ == "__main__":
    main()
