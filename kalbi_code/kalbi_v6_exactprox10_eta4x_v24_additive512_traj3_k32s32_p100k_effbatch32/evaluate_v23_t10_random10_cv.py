#!/usr/bin/env python3
"""Evaluate frozen KALBI T=10 and CV baselines on ten unseen random tasks."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import math
from pathlib import Path
from typing import Any
import warnings

import matplotlib
import numpy as np
import torch
from sklearn.model_selection import KFold

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .benchmark_kalbi_v3_cv_lasso_scad import _cv_method, _fit_selected
from .eval_kalbi_v2_common import (
    encode_dataset,
    file_sha256,
    tensors_from_arrays,
    vector_correlation,
)
from .exact_canonical_cd import CanonicalCDKernel
from .exact_canonical_cd_parallel import solve_multistart_path_parallel
from .paper_eval_unseen_dimension5 import (
    load_frozen_task,
    objective_paths,
    prediction_mse,
    split_standardize_three_way,
)
from .report_latest_v23_selectoronly_solution_selector_mixed6 import (
    SEED_STRIDE,
    load_components_v23,
    top_features,
)


BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
RED = "#D55E00"
GREY = "#666666"
LIGHT_GREY = "#D0D0D0"
FEATURE_COLORS = (
    BLUE,
    ORANGE,
    GREEN,
    RED,
    PURPLE,
    "#56B4E9",
    "#8A6D3B",
    "#4C566A",
)


def _predict_path(
    model: torch.nn.Module,
    encoded: dict[str, Any],
    raw_lambdas: torch.Tensor,
    penalty_shape: torch.Tensor,
    penalty_d0: torch.Tensor,
    *,
    query_chunk: int,
) -> np.ndarray:
    raw_lambdas = raw_lambdas.reshape(-1).float()
    penalty_shape = penalty_shape.reshape(raw_lambdas.numel(), -1).float()
    penalty_d0 = penalty_d0.reshape(-1).float()
    lambda_max = encoded["lambda_max"].float().reshape(-1)[0]
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, raw_lambdas.numel(), int(query_chunk)):
            stop = min(start + int(query_chunk), raw_lambdas.numel())
            beta = model.decode_queries(
                encoded,
                lambdas=raw_lambdas[start:stop][None],
                lambda_max=lambda_max.expand(1, stop - start),
                penalty_shape_features=penalty_shape[start:stop][None],
                penalty_d0=penalty_d0[start:stop][None],
            )
            outputs.append(beta[0].float().cpu())
    return torch.cat(outputs).numpy().astype(np.float64)


def _oracle_index(values: np.ndarray, raw_lambdas: np.ndarray) -> int:
    minimum = float(np.min(values))
    ties = np.flatnonzero(
        np.isclose(values, minimum, rtol=1e-12, atol=1e-14)
    )
    return int(ties[np.argmax(raw_lambdas[ties])])


def _cv_method_parallel(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    ratios_descending: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run independent CV folds in parallel without changing each warm path."""

    def evaluate_fold(
        fold_index: int,
        fold: tuple[np.ndarray, np.ndarray],
    ) -> tuple[int, dict[str, Any]]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Degrees of freedom <= 0 for slice",
                category=RuntimeWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in divide",
                category=RuntimeWarning,
            )
            result = _cv_method(
                method,
                x,
                y,
                ratios_descending,
                [fold],
                args,
            )
        result["fold_diagnostics"][0]["fold"] = fold_index
        return fold_index, result

    completed: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(
        max_workers=min(int(args.cv_workers), len(folds))
    ) as executor:
        futures = [
            executor.submit(evaluate_fold, index, fold)
            for index, fold in enumerate(folds)
        ]
        for future in as_completed(futures):
            index, result = future.result()
            completed[index] = result

    ordered = [completed[index] for index in range(len(folds))]
    scores = np.concatenate(
        [np.asarray(result["fold_mse"]) for result in ordered],
        axis=0,
    )
    mean_mse = scores.mean(axis=0)
    standard_error = scores.std(axis=0, ddof=1) / math.sqrt(
        scores.shape[0]
    )
    selected_index = int(np.argmin(mean_mse))
    return {
        "fold_mse": scores,
        "mean_mse": mean_mse,
        "standard_error": standard_error,
        "selected_index": selected_index,
        "selected_ratio": float(ratios_descending[selected_index]),
        "fold_diagnostics": [
            result["fold_diagnostics"][0] for result in ordered
        ],
    }


def _plot_solution_group(
    tasks: list[dict[str, Any]], output: Path
) -> None:
    figure, axes = plt.subplots(
        len(tasks),
        4,
        figsize=(13.0, 2.25 * len(tasks)),
        squeeze=False,
    )
    for row, task in enumerate(tasks):
        x_axis = np.log10(task["raw_lambdas"])
        for color_index, feature in enumerate(task["top_features"]):
            color = FEATURE_COLORS[color_index % len(FEATURE_COLORS)]
            axes[row, 0].plot(
                x_axis,
                task["kalbi_beta"][:, feature],
                color=color,
                lw=1.05,
            )
            axes[row, 0].plot(
                x_axis,
                task["cd_beta"][:, feature],
                color=color,
                lw=0.85,
                ls=":",
            )
        axes[row, 0].axhline(0.0, color=LIGHT_GREY, lw=0.6)
        axes[row, 1].plot(
            x_axis,
            task["kalbi_objective"] / task["zero_objective"],
            color=BLUE,
            lw=1.2,
        )
        axes[row, 1].plot(
            x_axis,
            task["cd_objective"] / task["zero_objective"],
            color=GREY,
            lw=1.0,
            ls=":",
        )
        axes[row, 2].plot(
            x_axis,
            task["kalbi_test_mse"] / task["zero_test_mse"],
            color=BLUE,
            lw=1.2,
        )
        axes[row, 2].plot(
            x_axis,
            task["cd_test_mse"] / task["zero_test_mse"],
            color=GREY,
            lw=1.0,
            ls=":",
        )
        axes[row, 2].axhline(1.0, color=LIGHT_GREY, lw=0.8, ls="--")
        axes[row, 3].plot(
            x_axis,
            np.count_nonzero(np.abs(task["kalbi_beta"]) > 1e-4, axis=1),
            color=BLUE,
            lw=1.2,
        )
        axes[row, 3].plot(
            x_axis,
            np.count_nonzero(np.abs(task["cd_beta"]) > 1e-4, axis=1),
            color=GREY,
            lw=1.0,
            ls=":",
        )
        selected_x = math.log10(task["selected_raw_lambda"])
        for column in range(4):
            axes[row, column].axvline(
                selected_x, color=ORANGE, lw=0.8, ls="--"
            )
            axes[row, column].grid(axis="y", color="#E8E8E8", lw=0.5)
        axes[row, 0].set_ylabel(
            f"D{task['dataset_index']}: "
            f"N={task['n']:,}, P={task['p']:,}"
        )
        if row == 0:
            for column, title in enumerate(
                (
                    "Coefficient solution path",
                    r"Training objective $F/F_0$",
                    "Held-out MSE / zero MSE",
                    r"Nonzero coefficients ($10^{-4}$)",
                )
            ):
                axes[row, column].set_title(title)
        if row == len(tasks) - 1:
            for column in range(4):
                axes[row, column].set_xlabel(
                    r"$\log_{10}(\lambda_{\rm raw})$"
                )
    handles = [
        Line2D([0], [0], color=BLUE, lw=1.2, label="KALBI"),
        Line2D(
            [0],
            [0],
            color=GREY,
            lw=1.0,
            ls=":",
            label="Coordinate descent reference",
        ),
        Line2D(
            [0],
            [0],
            color=ORANGE,
            lw=0.8,
            ls="--",
            label="KALBI-selected lambda",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.003),
    )
    figure.subplots_adjust(
        left=0.095,
        right=0.99,
        top=0.96,
        bottom=0.085,
        hspace=0.42,
        wspace=0.30,
    )
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_selector_group(
    tasks: list[dict[str, Any]], output: Path
) -> None:
    figure, axes = plt.subplots(
        len(tasks),
        2,
        figsize=(10.6, 2.3 * len(tasks)),
        squeeze=False,
    )
    for row, task in enumerate(tasks):
        x_axis = np.log10(task["raw_lambdas"])
        selected_x = math.log10(task["selected_raw_lambda"])
        oracle = task["oracle_index"]
        axes[row, 0].plot(
            x_axis,
            task["kalbi_test_mse"] / task["zero_test_mse"],
            color=BLUE,
            lw=1.3,
        )
        axes[row, 0].plot(
            x_axis,
            task["cd_test_mse"] / task["zero_test_mse"],
            color=GREY,
            lw=1.0,
            ls=":",
        )
        axes[row, 0].axhline(1.0, color=LIGHT_GREY, lw=0.8, ls="--")
        axes[row, 0].axvline(
            selected_x, color=ORANGE, lw=0.9, ls="--"
        )
        axes[row, 0].scatter(
            [selected_x],
            [task["selected_test_mse"] / task["zero_test_mse"]],
            color=ORANGE,
            edgecolor="#202124",
            linewidth=0.35,
            marker="D",
            s=26,
            zorder=4,
        )
        axes[row, 0].scatter(
            [x_axis[oracle]],
            [task["kalbi_test_mse"][oracle] / task["zero_test_mse"]],
            color="white",
            edgecolor=BLUE,
            linewidth=0.7,
            marker="*",
            s=55,
            zorder=4,
        )
        axes[row, 0].set_ylabel(
            f"D{task['dataset_index']}\nTest MSE / zero MSE"
        )
        axes[row, 0].set_title(
            f"Lambda selection (N={task['n']:,}, P={task['p']:,})"
        )
        axes[row, 0].grid(axis="y", color="#E8E8E8", lw=0.5)
        if row == len(tasks) - 1:
            axes[row, 0].set_xlabel(r"$\log_{10}(\lambda_{\rm raw})$")

        derivative = axes[row, 1].twinx()
        axes[row, 1].plot(
            task["penalty_grid"],
            task["penalty_rho"],
            color=GREEN,
            lw=1.3,
        )
        derivative.plot(
            task["penalty_grid"],
            task["penalty_drho"],
            color=ORANGE,
            lw=1.0,
            ls="--",
        )
        axes[row, 1].set_ylim(bottom=0.0)
        derivative.set_ylim(bottom=0.0)
        axes[row, 1].set_ylabel(r"$\rho$", color=GREEN)
        derivative.set_ylabel(r"$\rho'$", color=ORANGE)
        axes[row, 1].set_title("KALBI-selected penalty")
        axes[row, 1].grid(axis="y", color="#E8E8E8", lw=0.5)
        if row == len(tasks) - 1:
            axes[row, 1].set_xlabel(r"$|\beta|$")
    handles = [
        Line2D([0], [0], color=BLUE, lw=1.3, label="KALBI"),
        Line2D(
            [0],
            [0],
            color=GREY,
            lw=1.0,
            ls=":",
            label="Coordinate descent reference",
        ),
        Line2D(
            [0],
            [0],
            color=ORANGE,
            lw=0.9,
            ls="--",
            marker="D",
            label="KALBI-selected lambda",
        ),
        Line2D(
            [0],
            [0],
            color=BLUE,
            marker="*",
            markerfacecolor="white",
            lw=0,
            label="KALBI path oracle",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.003),
    )
    figure.subplots_adjust(
        left=0.105,
        right=0.90,
        top=0.97,
        bottom=0.09,
        hspace=0.50,
        wspace=0.38,
    )
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_cv_comparison(
    rows: list[dict[str, Any]], output: Path
) -> None:
    values = np.asarray(
        [
            [
                row["kalbi_normalized_mse"],
                row["cv_lasso_normalized_mse"],
                row["cv_scad_normalized_mse"],
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    methods = ("KALBI", "CV LASSO", "CV SCAD")
    colors = (BLUE, GREEN, PURPLE)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.4, 4.2),
        gridspec_kw={"width_ratios": [3.6, 1.15]},
    )
    positions = np.arange(len(rows), dtype=np.float64)
    width = 0.25
    for index, (method, color) in enumerate(zip(methods, colors)):
        axes[0].bar(
            positions + (index - 1) * width,
            values[:, index],
            width=width,
            color=color,
            label=method,
        )
    axes[0].axhline(
        1.0,
        color=GREY,
        lw=0.9,
        ls="--",
        label="Zero-predictor baseline",
    )
    axes[0].set_xticks(
        positions,
        [
            f"D{row['dataset_index']}\n"
            f"{row['n']:,}x{row['p']:,}"
            for row in rows
        ],
        fontsize=7,
    )
    axes[0].set_ylabel("Held-out MSE / zero-predictor MSE")
    axes[0].set_title("Per-dataset held-out prediction error")
    axes[0].grid(axis="y", color="#E8E8E8", lw=0.5)

    geometric = np.exp(
        np.mean(np.log(np.maximum(values, 1e-12)), axis=0)
    )
    aggregate_positions = np.arange(len(methods))
    axes[1].bar(
        aggregate_positions,
        geometric,
        color=colors,
        width=0.65,
    )
    axes[1].axhline(1.0, color=GREY, lw=0.9, ls="--")
    axes[1].set_xticks(
        aggregate_positions,
        methods,
        rotation=25,
        ha="right",
    )
    axes[1].set_ylabel("Geometric mean normalized MSE")
    axes[1].set_title("Aggregate")
    axes[1].grid(axis="y", color="#E8E8E8", lw=0.5)
    for index, value in enumerate(geometric):
        axes[1].text(
            index,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.003),
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.99,
        top=0.90,
        bottom=0.23,
        wspace=0.28,
    )
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cd-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--query-chunk", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=2_026_072_607)
    parser.add_argument("--max-sweeps", type=int, default=2000)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--cd-workers", type=int, default=8)
    parser.add_argument("--cd-include-warm", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--zero-tolerance", type=float, default=1e-10)
    parser.add_argument("--cv-seed", type=int, default=2_026_072_610)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-grid-size", type=int, default=25)
    parser.add_argument("--cv-min-ratio", type=float, default=1e-4)
    parser.add_argument("--cv-workers", type=int, default=5)
    parser.add_argument("--lasso-tolerance", type=float, default=1e-8)
    parser.add_argument("--lasso-max-iter", type=int, default=30_000)
    parser.add_argument("--scad-gamma", type=float, default=3.7)
    parser.add_argument("--scad-tolerance", type=float, default=1e-7)
    parser.add_argument("--scad-max-iter", type=int, default=100)
    parser.add_argument("--scad-max-epochs", type=int, default=100_000)
    parser.add_argument("--dataset-indices", type=int, nargs="*")
    parser.add_argument("--skip-finalize", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    prepared_root = args.prepared_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.resume:
        raise RuntimeError(f"refusing existing output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    frozen = json.loads(
        (
            prepared_root
            / "reproducibility"
            / "reproducibility_manifest.json"
        ).read_text(encoding="utf-8")
    )
    if frozen.get("dimension_regime") != "random":
        raise RuntimeError("expected unconditioned random-dimension manifest")
    entries = list(frozen["accepted_tasks"])
    if len(entries) != 10:
        raise RuntimeError("expected exactly ten frozen tasks")
    if args.dataset_indices:
        requested = set(args.dataset_indices)
        invalid = requested.difference(
            int(entry["dataset_index"]) for entry in entries
        )
        if invalid:
            raise RuntimeError(
                f"unknown dataset indices: {sorted(invalid)}"
            )
        entries = [
            entry
            for entry in entries
            if int(entry["dataset_index"]) in requested
        ]
    checkpoint_path = Path(frozen["checkpoint_frozen"]).resolve()
    if file_sha256(checkpoint_path) != frozen["source_hashes"]["checkpoint"]:
        raise RuntimeError("frozen checkpoint SHA mismatch")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint, model, selector = load_components_v23(
        checkpoint_path,
        args.config.resolve(),
        device,
    )
    kernel = CanonicalCDKernel(args.cd_library.resolve())
    raw_lambdas = np.logspace(
        -4.0, 0.0, int(args.grid_size), dtype=np.float64
    )
    cv_ratios = np.logspace(
        0.0,
        math.log10(float(args.cv_min_ratio)),
        int(args.cv_grid_size),
        dtype=np.float64,
    )
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    plots: list[dict[str, Any]] = []

    for entry in entries:
        dataset_index = int(entry["dataset_index"])
        task_dir = output_dir / f"dataset_{dataset_index:02d}"
        metrics_file = task_dir / "metrics.json"
        paths_file = task_dir / "paths.npz"
        if args.resume and metrics_file.is_file() and paths_file.is_file():
            metric = json.loads(metrics_file.read_text(encoding="utf-8"))
            with np.load(paths_file, allow_pickle=False) as payload:
                path = {key: np.asarray(payload[key]) for key in payload.files}
            rows.append(metric["summary"])
            details.append(metric)
            plots.append(
                {
                    **metric["plot_metadata"],
                    **path,
                }
            )
            print(
                json.dumps(
                    {"dataset": dataset_index, "status": "resume_existing"}
                ),
                flush=True,
            )
            continue

        print(
            json.dumps(
                {
                    "dataset": dataset_index,
                    "status": "starting",
                    "n": int(entry["n"]),
                    "p": int(entry["p"]),
                }
            ),
            flush=True,
        )
        x_raw, y_raw, provenance = load_frozen_task(prepared_root, entry)
        split_seed = int(args.split_seed) + SEED_STRIDE * (
            dataset_index - 1
        )
        split = split_standardize_three_way(x_raw, y_raw, split_seed)
        x_train = np.asarray(split["x_train"], dtype=np.float32)
        y_train = np.asarray(split["y_train"], dtype=np.float32)
        x_test = np.asarray(split["x_test"], dtype=np.float32)
        y_test = np.asarray(split["y_test"], dtype=np.float32)
        x_device, y_device = tensors_from_arrays(x_train, y_train, device)
        encoded = encode_dataset(model, x_device, y_device)
        lambda_max = encoded["lambda_max"].float().reshape(-1)
        with torch.inference_mode():
            selected = selector(
                x_device,
                y_device,
                lambda_max,
                return_representation=True,
            )
        selected_raw_lambda = float(selected["raw_lambda"].reshape(-1)[0])
        penalty_shape = selected["penalty_shape_features"].reshape(1, -1)
        penalty_d0 = selected["penalty_d0"].reshape(-1)
        count = raw_lambdas.size
        kalbi_beta = _predict_path(
            model,
            encoded,
            torch.as_tensor(
                raw_lambdas, device=device, dtype=torch.float32
            ),
            penalty_shape.expand(count, -1),
            penalty_d0.expand(count),
            query_chunk=int(args.query_chunk),
        )
        selected_beta = _predict_path(
            model,
            encoded,
            torch.as_tensor(
                [selected_raw_lambda],
                device=device,
                dtype=torch.float32,
            ),
            penalty_shape,
            penalty_d0,
            query_chunk=1,
        )
        print(
            json.dumps(
                {"dataset": dataset_index, "status": "kalbi_path_complete"}
            ),
            flush=True,
        )
        knots = (
            selector.penalty_parameter_head.penalty_grid.detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        relu_b = float(selected["penalty_relu_b"].reshape(-1)[0])
        relu_weights = (
            selected["penalty_relu_weights"]
            .reshape(-1, knots.size)[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        penalty_rho = (
            selected["penalty_rho"]
            .reshape(-1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        penalty_drho = (
            selected["penalty_drho"]
            .reshape(-1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        cd = solve_multistart_path_parallel(
            library_path=kernel.path,
            x=x_train,
            y=y_train,
            raw_lambdas=raw_lambdas,
            knots=knots,
            relu_b=relu_b,
            relu_weights=relu_weights,
            model_beta=kalbi_beta,
            max_sweeps=int(args.max_sweeps),
            retries=int(args.retries),
            tolerance=float(args.tolerance),
            zero_tolerance=float(args.zero_tolerance),
            workers=int(args.cd_workers),
            include_warm=bool(args.cd_include_warm),
        )
        cd_beta = np.asarray(cd["reference_beta"], dtype=np.float64)
        if float(np.max(cd["reference_fixed_point_max"])) > max(
            2e-3, 5.0 * float(args.tolerance)
        ):
            raise RuntimeError("CD fixed-point audit failed")
        if float(np.max(cd["reference_projected_kkt_max"])) > max(
            2e-3, 5.0 * float(args.tolerance)
        ):
            raise RuntimeError("CD KKT audit failed")
        print(
            json.dumps(
                {
                    "dataset": dataset_index,
                    "status": "cd_complete",
                    "max_fixed_point": float(
                        np.max(cd["reference_fixed_point_max"])
                    ),
                    "max_projected_kkt": float(
                        np.max(cd["reference_projected_kkt_max"])
                    ),
                }
            ),
            flush=True,
        )

        _, _, kalbi_objective = objective_paths(
            x_train,
            y_train,
            kalbi_beta,
            raw_lambdas,
            knots,
            relu_b,
            relu_weights,
        )
        cd_objective = np.asarray(
            cd["reference_objective"], dtype=np.float64
        )
        kalbi_test_mse = prediction_mse(x_test, y_test, kalbi_beta)
        cd_test_mse = prediction_mse(x_test, y_test, cd_beta)
        selected_test_mse = float(
            prediction_mse(x_test, y_test, selected_beta)[0]
        )
        oracle_index = _oracle_index(kalbi_test_mse, raw_lambdas)
        zero_test_mse = float(
            np.mean(np.asarray(y_test, dtype=np.float64) ** 2)
        )
        zero_objective = 0.5 * float(
            np.mean(np.asarray(y_train, dtype=np.float64) ** 2)
        )

        folds = list(
            KFold(
                n_splits=int(args.cv_folds),
                shuffle=True,
                random_state=int(args.cv_seed) + dataset_index,
            ).split(x_train)
        )
        print(
            json.dumps(
                {"dataset": dataset_index, "status": "lasso_cv_start"}
            ),
            flush=True,
        )
        lasso_cv = _cv_method_parallel(
            "lasso", x_train, y_train, cv_ratios, folds, args
        )
        lasso_fit = _fit_selected(
            "lasso",
            x_train,
            y_train,
            x_test,
            y_test,
            cv_ratios,
            int(lasso_cv["selected_index"]),
            args,
        )
        print(
            json.dumps(
                {"dataset": dataset_index, "status": "lasso_cv_complete"}
            ),
            flush=True,
        )
        print(
            json.dumps(
                {"dataset": dataset_index, "status": "scad_cv_start"}
            ),
            flush=True,
        )
        scad_cv = _cv_method_parallel(
            "scad", x_train, y_train, cv_ratios, folds, args
        )
        scad_fit = _fit_selected(
            "scad",
            x_train,
            y_train,
            x_test,
            y_test,
            cv_ratios,
            int(scad_cv["selected_index"]),
            args,
        )
        print(
            json.dumps(
                {"dataset": dataset_index, "status": "scad_cv_complete"}
            ),
            flush=True,
        )
        summary = {
            "dataset_index": dataset_index,
            "seed": int(entry["seed"]),
            "n": int(entry["n"]),
            "p": int(entry["p"]),
            "p_over_n": float(entry["p_over_n"]),
            "train_rows": int(x_train.shape[0]),
            "test_rows": int(x_test.shape[0]),
            "task_sha256": str(entry["task_sha256"]),
            "split_seed": split_seed,
            "lambda_max": float(lambda_max[0]),
            "selected_raw_lambda": selected_raw_lambda,
            "penalty_d0": float(penalty_d0[0]),
            "zero_predictor_test_mse": zero_test_mse,
            "kalbi_test_mse": selected_test_mse,
            "kalbi_normalized_mse": selected_test_mse / zero_test_mse,
            "kalbi_nnz_1e4": int(
                np.count_nonzero(np.abs(selected_beta[0]) > 1e-4)
            ),
            "kalbi_path_oracle_mse": float(
                kalbi_test_mse[oracle_index]
            ),
            "kalbi_path_oracle_raw_lambda": float(
                raw_lambdas[oracle_index]
            ),
            "kalbi_selected_over_oracle": selected_test_mse
            / max(float(kalbi_test_mse[oracle_index]), 1e-12),
            "cv_lasso_selected_ratio": float(lasso_cv["selected_ratio"]),
            "cv_lasso_test_mse": float(lasso_fit["test_mse"]),
            "cv_lasso_normalized_mse": float(lasso_fit["test_mse"])
            / zero_test_mse,
            "cv_lasso_nnz_1e4": int(lasso_fit["nnz_1e4"]),
            "cv_scad_selected_ratio": float(scad_cv["selected_ratio"]),
            "cv_scad_test_mse": float(scad_fit["test_mse"]),
            "cv_scad_normalized_mse": float(scad_fit["test_mse"])
            / zero_test_mse,
            "cv_scad_nnz_1e4": int(scad_fit["nnz_1e4"]),
            "path_correlation_with_cd": float(
                vector_correlation(kalbi_beta, cd_beta)
            ),
            "coefficient_rmse_with_cd": float(
                np.sqrt(np.mean((kalbi_beta - cd_beta) ** 2))
            ),
            "objective_gap_mean": float(
                np.mean(kalbi_objective - cd_objective)
            ),
            "cd_max_fixed_point": float(
                np.max(cd["reference_fixed_point_max"])
            ),
            "cd_max_projected_kkt": float(
                np.max(cd["reference_projected_kkt_max"])
            ),
        }
        plot_metadata = {
            "dataset_index": dataset_index,
            "n": int(entry["n"]),
            "p": int(entry["p"]),
            "selected_raw_lambda": selected_raw_lambda,
            "selected_test_mse": selected_test_mse,
            "oracle_index": oracle_index,
            "zero_test_mse": zero_test_mse,
            "zero_objective": zero_objective,
        }
        metric = {
            "summary": summary,
            "plot_metadata": plot_metadata,
            "provenance": {
                "data_path": provenance["data_path"],
                "manifest_path": provenance["manifest_path"],
                "core_exact_max_abs_corr": provenance[
                    "core_exact_max_abs_corr"
                ],
                "sampled_final_max_abs_corr": provenance[
                    "sampled_final_max_abs_corr"
                ],
            },
            "lasso_cv_mean_mse": lasso_cv["mean_mse"].tolist(),
            "lasso_cv_standard_error": lasso_cv[
                "standard_error"
            ].tolist(),
            "lasso_cv_fold_diagnostics": lasso_cv["fold_diagnostics"],
            "lasso_final_diagnostics": lasso_fit["diagnostics"],
            "scad_cv_mean_mse": scad_cv["mean_mse"].tolist(),
            "scad_cv_standard_error": scad_cv[
                "standard_error"
            ].tolist(),
            "scad_cv_fold_diagnostics": scad_cv["fold_diagnostics"],
            "scad_final_diagnostics": scad_fit["diagnostics"],
        }
        task_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            paths_file,
            raw_lambdas=raw_lambdas,
            kalbi_beta=kalbi_beta.astype(np.float32),
            cd_beta=cd_beta.astype(np.float32),
            kalbi_objective=kalbi_objective,
            cd_objective=cd_objective,
            kalbi_test_mse=kalbi_test_mse,
            cd_test_mse=cd_test_mse,
            penalty_grid=knots,
            penalty_rho=penalty_rho,
            penalty_drho=penalty_drho,
            top_features=top_features(cd_beta),
        )
        metrics_file.write_text(
            json.dumps(metric, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        path = {
            "raw_lambdas": raw_lambdas,
            "kalbi_beta": kalbi_beta,
            "cd_beta": cd_beta,
            "kalbi_objective": kalbi_objective,
            "cd_objective": cd_objective,
            "kalbi_test_mse": kalbi_test_mse,
            "cd_test_mse": cd_test_mse,
            "penalty_grid": knots,
            "penalty_rho": penalty_rho,
            "penalty_drho": penalty_drho,
            "top_features": top_features(cd_beta),
        }
        rows.append(summary)
        details.append(metric)
        plots.append({**plot_metadata, **path})
        print(json.dumps(summary, sort_keys=True), flush=True)

    if args.skip_finalize:
        print(
            json.dumps(
                {
                    "status": "worker_complete",
                    "datasets": [
                        int(entry["dataset_index"]) for entry in entries
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    _write_csv(output_dir / "comparison_metrics.csv", rows)
    values = np.asarray(
        [
            [
                row["kalbi_normalized_mse"],
                row["cv_lasso_normalized_mse"],
                row["cv_scad_normalized_mse"],
            ]
            for row in rows
        ]
    )
    methods = ("KALBI", "CV LASSO", "CV SCAD")
    aggregate = {
        method: {
            "mean_normalized_mse": float(np.mean(values[:, index])),
            "geometric_mean_normalized_mse": float(
                np.exp(
                    np.mean(
                        np.log(np.maximum(values[:, index], 1e-12))
                    )
                )
            ),
            "median_normalized_mse": float(
                np.median(values[:, index])
            ),
            "wins": int(
                np.count_nonzero(np.argmin(values, axis=1) == index)
            ),
        }
        for index, method in enumerate(methods)
    }
    report = {
        "evaluation": "KALBI T=10 random-dimension unseen10 with CV baselines",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_step": int(checkpoint["completed_steps"]),
        "contract": {
            "refinement_steps": 10,
            "eta_multiplier": 1.0,
            "raw_lambda_grid": [
                float(raw_lambdas[0]),
                float(raw_lambdas[-1]),
                int(raw_lambdas.size),
            ],
            "cv_folds": int(args.cv_folds),
            "cv_workers": int(args.cv_workers),
            "lasso_tolerance": float(args.lasso_tolerance),
            "lasso_max_iter": int(args.lasso_max_iter),
            "scad_tolerance": float(args.scad_tolerance),
            "scad_max_iter": int(args.scad_max_iter),
            "scad_max_epochs": int(args.scad_max_epochs),
            "cv_ratio_grid": [
                float(cv_ratios[0]),
                float(cv_ratios[-1]),
                int(cv_ratios.size),
            ],
            "selector_inputs": "outer training X and y only",
            "outer_test_used_for_selection": False,
            "cd_full_feature": True,
            "cd_starts": ["kalbi_clipped", "zero"],
        },
        "aggregate": aggregate,
        "tasks": details,
    }
    (output_dir / "comparison_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_solution_group(
        plots[:5], output_dir / "solution_paths_random10_part1"
    )
    _plot_solution_group(
        plots[5:], output_dir / "solution_paths_random10_part2"
    )
    _plot_selector_group(
        plots[:5], output_dir / "lambda_selector_random10_part1"
    )
    _plot_selector_group(
        plots[5:], output_dir / "lambda_selector_random10_part2"
    )
    _plot_cv_comparison(
        rows, output_dir / "kalbi_vs_cv_lasso_scad_random10"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(output_dir),
                "aggregate": aggregate,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
