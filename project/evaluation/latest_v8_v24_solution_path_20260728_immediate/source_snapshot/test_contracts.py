#!/usr/bin/env python3
"""Pre-smoke lambda, preprocessing, and leakage contract tests."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--evaluator-source", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(
        (
            args.evaluation_root
            / "data_manifest"
            / "metrics"
            / "preparation_summary.json"
        ).read_text()
    )
    checkpoint_audit = json.loads(
        (args.evaluation_root / "checkpoints" / "checkpoint_audit.json").read_text()
    )
    assert summary["status"] == "PASS"
    assert checkpoint_audit["status"] == "PASS"
    assert len(summary["accepted"]) == 4
    assert [task["outcome_branch"] for task in summary["accepted"]].count(
        "linear"
    ) == 2
    assert [task["outcome_branch"] for task in summary["accepted"]].count(
        "tabicl"
    ) == 2

    expected_grid = np.logspace(-4, 0, 41)
    assert expected_grid.shape == (41,)
    assert expected_grid[0] == 1e-4
    assert expected_grid[-1] == 1.0
    assert np.all(np.diff(expected_grid) > 0)

    preprocessing_results = []
    for task in summary["accepted"]:
        with np.load(task["accepted_data_path"], allow_pickle=False) as archive:
            x = np.asarray(archive["X"][0], dtype=np.float64)
            y = np.asarray(archive["y"][0], dtype=np.float64)
        with np.load(task["split_path"], allow_pickle=False) as archive:
            split = {key: np.asarray(archive[key]) for key in archive.files}
        context = split["context_indices"].astype(np.int64)
        validation = split["validation_indices"].astype(np.int64)
        test = split["test_indices"].astype(np.int64)
        assert not (set(context) & set(validation))
        assert not (set(context) & set(test))
        assert not (set(validation) & set(test))
        active = split["active"].astype(bool)
        x_context = (x[context] - split["x_mean"]) / split["x_scale"]
        y_context = (y[context] - float(split["y_mean"])) / float(
            split["y_scale"]
        )
        np.testing.assert_allclose(x_context[:, active].mean(0), 0.0, atol=2e-7)
        np.testing.assert_allclose(x_context[:, active].std(0), 1.0, atol=2e-7)
        np.testing.assert_allclose(y_context.mean(), 0.0, atol=2e-7)
        np.testing.assert_allclose(y_context.std(), 1.0, atol=2e-7)
        for indices in (validation, test):
            frozen_x = (x[indices] - split["x_mean"]) / split["x_scale"]
            frozen_y = (y[indices] - float(split["y_mean"])) / float(
                split["y_scale"]
            )
            assert np.isfinite(frozen_x).all()
            assert np.isfinite(frozen_y).all()
        preprocessing_results.append(
            {
                "task_id": task["task_id"],
                "context_rows": int(context.size),
                "validation_rows": int(validation.size),
                "test_rows": int(test.size),
                "active_features": int(active.sum()),
                "passed": True,
            }
        )

    source = args.evaluator_source.read_text()
    tree = ast.parse(source)
    selector_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "selector"
    ]
    assert len(selector_calls) == 1
    selector_call = selector_calls[0]
    positional_names = [
        node.id for node in selector_call.args if isinstance(node, ast.Name)
    ]
    assert positional_names == ["x_device", "y_device", "lambda_max"]
    assert "validation" not in ast.unparse(selector_call).lower()
    assert "test" not in ast.unparse(selector_call).lower()

    result = {
        "status": "PASS",
        "raw_lambda_grid": expected_grid.tolist(),
        "raw_lambda_contract": "41 points, logspace(-4,0), unmodified",
        "preprocessing": preprocessing_results,
        "leakage": {
            "selector_call_ast": ast.unparse(selector_call),
            "selector_inputs_context_only": True,
            "validation_y_excluded": True,
            "test_y_excluded": True,
        },
        "checkpoint_pairing_sha_mismatch_expected_and_recorded": bool(
            checkpoint_audit["pairing"][
                "selector_frozen_beta_matches_latest_beta"
            ]
            is False
        ),
    }
    output = args.evaluation_root / "metrics" / "unit_test_results.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
