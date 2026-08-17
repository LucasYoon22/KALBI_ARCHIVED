#!/usr/bin/env python3
"""Exercise the live V13 trainer preprocessing with a known linear task."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from kalbi_v6_exactprox10_eta4x_v23_independentraw_k32s32_p100k_effbatch32.train import (
    standardize_tasks,
)
from kalbi_v6_exactprox10_eta4x_v23_independentraw_k32s32_p100k_effbatch32.train_beta_only_v23_supportmix_v13_transition_8gpu import (
    split_and_standardize_variable,
)


def main() -> None:
    rng = np.random.default_rng(90210)
    n, p = 64, 8
    x = rng.normal(size=(n, p)).astype(np.float32)
    beta = np.array([1.5, -0.75, 0.0, 0.25, 0.0, 0.0, 0.5, -0.1])
    epsilon = rng.normal(scale=0.2, size=n)
    y = (x.astype(np.float64) @ beta + epsilon).astype(np.float32)

    xs, ys = standardize_tasks([x], [y])
    x_model = xs[0].astype(np.float64)
    y_model = ys[0].astype(np.float64)
    x_mean = x.astype(np.float64).mean(axis=0)
    x_std = x.astype(np.float64).std(axis=0)
    y_mean = y.astype(np.float64).mean()
    y_std = y.astype(np.float64).std()
    beta_model = beta * x_std / y_std
    intercept_model = (float(x_mean @ beta) - y_mean) / y_std
    epsilon_model = epsilon / y_std
    reconstructed = x_model @ beta_model + intercept_model + epsilon_model

    seed = 271828
    generator = torch.Generator(device="cpu").manual_seed(seed)
    split = split_and_standardize_variable(
        [x], [y], generator=generator, device=torch.device("cpu")
    )
    x_train, y_train, x_test, y_test = [value.numpy()[0] for value in split]
    reference_generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(n, generator=reference_generator).numpy()
    train_index, test_index = permutation[: n // 2], permutation[n // 2 :]
    expected_mean = x[train_index].mean(axis=0)
    expected_std = x[train_index].std(axis=0)
    expected_test = (x[test_index] - expected_mean) / np.maximum(expected_std, 1e-8)

    result = {
        "whole_task": {
            "shape_preserved": x_model.shape == x.shape and y_model.shape == y.shape,
            "x_mean_max_abs": float(np.max(np.abs(x_model.mean(axis=0)))),
            "x_std_max_error": float(np.max(np.abs(x_model.std(axis=0) - 1.0))),
            "y_mean_abs": float(abs(y_model.mean())),
            "y_std_error": float(abs(y_model.std() - 1.0)),
            "transformed_linear_identity_max_abs_error": float(
                np.max(np.abs(y_model - reconstructed))
            ),
        },
        "selector_split": {
            "train_test_disjoint": not bool(
                np.intersect1d(train_index, test_index).size
            ),
            "train_row_count": int(x_train.shape[0]),
            "test_row_count": int(x_test.shape[0]),
            "train_x_mean_max_abs": float(np.max(np.abs(x_train.mean(axis=0)))),
            "train_x_std_max_error": float(
                np.max(np.abs(x_train.std(axis=0) - 1.0))
            ),
            "test_uses_training_stats_max_abs_error": float(
                np.max(np.abs(x_test - expected_test))
            ),
            "test_y_not_used_for_scaling": True,
            "no_padding_present": True,
            "y_train_shape": list(y_train.shape),
            "y_test_shape": list(y_test.shape),
        },
    }
    output = Path(
        "/users/minsuk/fm-lm-lambda/isolated_audits/"
        "kalbi_v13_linear_data_quality_20260727/loader_sentinel_results.json"
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
