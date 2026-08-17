#!/usr/bin/env python3
"""Evaluate Kalbi V5 exact-prox refinement at multiple iteration depths."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .eval_kalbi_v2_common import (
    encode_dataset,
    file_sha256,
    split_standardize,
    tensors_from_arrays,
    vector_correlation,
)
from .model import (
    MODEL_VERSION,
    V3_SOURCE_MODEL_VERSION,
    _refine_beta_kalbi_v5,
)
from .report_kalbi_v3_v19_small_training_task import load_generated_task
from .report_kalbi_v5_v19_training_task import (
    load_components_v19,
    objective_paths,
    safe_loss_curve,
    style_axis,
    test_mse,
)
from .report_v3_v4_v5_context_lowdim4 import penalty_contract


INK = "#20242A"
BLUE = "#2367A6"
ORANGE = "#CF6A32"
OLIVE = "#768B36"
MAGENTA = "#B64F7C"
GOLD = "#D39B22"
TEAL = "#188B87"
ITERATION_COLORS = (BLUE, ORANGE, OLIVE, MAGENTA, TEAL, GOLD)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def predict_iteration_sweep(
    model: torch.nn.Module,
    encoded: dict[str, Any],
    lambdas: torch.Tensor,
    shape: torch.Tensor,
    d0: torch.Tensor,
    iterations: tuple[int, ...],
    *,
    chunk_size: int = 8,
) -> tuple[
    dict[int, np.ndarray],
    np.ndarray,
    dict[int, float],
    dict[int, np.ndarray],
]:
    lambdas = lambdas.reshape(-1).float()
    shape = shape.reshape(lambdas.numel(), -1).float()
    d0 = d0.reshape(-1).float()
    lambda_max = encoded["lambda_max"].float().reshape(-1)[0]
    targets = set(iterations)
    max_iterations = max(iterations)
    beta_parts: dict[int, list[torch.Tensor]] = {
        iteration: [] for iteration in iterations
    }
    beta0_parts: list[torch.Tensor] = []
    per_iteration_seconds = {
        iteration: 0.0 for iteration in range(1, max_iterations + 1)
    }
    per_iteration_change_parts: dict[int, list[torch.Tensor]] = {
        iteration: [] for iteration in iterations
    }

    with torch.inference_mode(), torch.autocast(
        device_type=lambdas.device.type,
        dtype=torch.bfloat16,
        enabled=lambdas.device.type == "cuda",
    ):
        for start in range(0, lambdas.numel(), int(chunk_size)):
            stop = min(start + int(chunk_size), lambdas.numel())
            chunk_lambdas = lambdas[start:stop][None]
            chunk_shape = shape[start:stop][None]
            chunk_d0 = d0[start:stop][None]

            synchronize(lambdas.device)
            started = time.perf_counter()
            output = model.decode_queries(
                encoded,
                lambdas=chunk_lambdas,
                lambda_max=lambda_max.expand(1, stop - start),
                penalty_shape_features=chunk_shape,
                penalty_d0=chunk_d0,
                return_aux=True,
            )
            synchronize(lambdas.device)
            per_iteration_seconds[1] += time.perf_counter() - started

            beta0 = output["kalbi_v5_beta0_q"]
            beta_current = output["beta_q"]
            beta0_parts.append(beta0[0].float().cpu())
            if 1 in targets:
                beta_parts[1].append(beta_current[0].float().cpu())
                per_iteration_change_parts[1].append(
                    (beta_current - beta0)
                    .float()
                    .square()
                    .mean(-1)
                    .sqrt()[0]
                    .cpu()
                )

            for iteration in range(2, max_iterations + 1):
                beta_previous = beta_current
                synchronize(lambdas.device)
                started = time.perf_counter()
                refinement = _refine_beta_kalbi_v5(
                    model,
                    encoded,
                    beta_previous,
                    chunk_lambdas,
                    chunk_d0,
                    chunk_shape,
                )
                beta_current = refinement["beta"]
                synchronize(lambdas.device)
                per_iteration_seconds[iteration] += (
                    time.perf_counter() - started
                )
                if iteration in targets:
                    beta_parts[iteration].append(
                        beta_current[0].float().cpu()
                    )
                    per_iteration_change_parts[iteration].append(
                        (beta_current - beta0)
                        .float()
                        .square()
                        .mean(-1)
                        .sqrt()[0]
                        .cpu()
                    )

    beta_by_iteration = {
        iteration: torch.cat(beta_parts[iteration])
        .numpy()
        .astype(np.float64)
        for iteration in iterations
    }
    beta0 = torch.cat(beta0_parts).numpy().astype(np.float64)
    cumulative_seconds: dict[int, float] = {}
    running_seconds = 0.0
    for iteration in range(1, max_iterations + 1):
        running_seconds += per_iteration_seconds[iteration]
        if iteration in targets:
            cumulative_seconds[iteration] = running_seconds
    cumulative_change = {
        iteration: torch.cat(per_iteration_change_parts[iteration])
        .numpy()
        .astype(np.float64)
        for iteration in iterations
    }
    return (
        beta_by_iteration,
        beta0,
        cumulative_seconds,
        cumulative_change,
    )


def plot_task(
    output_path: Path,
    *,
    task_name: str,
    n: int,
    p: int,
    ratios: np.ndarray,
    iterations: tuple[int, ...],
    loss_by_iteration: dict[int, np.ndarray],
    mse_by_iteration: dict[int, np.ndarray],
    exact_loss: np.ndarray,
    exact_mse: np.ndarray,
    selected_index: int,
    metrics_by_iteration: dict[int, dict[str, float]],
) -> None:
    x_axis = np.log10(ratios)
    figure, axes = plt.subplots(1, 3, figsize=(20.0, 5.9))

    axis = axes[0]
    for color, iteration in zip(
        ITERATION_COLORS, iterations, strict=False
    ):
        axis.plot(
            x_axis,
            loss_by_iteration[iteration],
            color=color,
            lw=1.7,
            label=f"T={iteration}",
        )
    axis.plot(
        x_axis,
        exact_loss,
        color=INK,
        lw=1.8,
        ls=":",
        label="exact CD",
    )
    axis.axvline(x_axis[selected_index], color=GOLD, lw=1.2, ls="--")
    axis.set_xlabel("log10(effective lambda ratio)", fontsize=12)
    axis.set_ylabel("log1p(F/F0)", fontsize=12)
    axis.set_title("Training objective path", fontsize=13)
    axis.legend(frameon=False, fontsize=9, ncol=2)
    style_axis(axis)

    axis = axes[1]
    for color, iteration in zip(
        ITERATION_COLORS, iterations, strict=False
    ):
        axis.plot(
            x_axis,
            np.log10(mse_by_iteration[iteration].clip(1e-12)),
            color=color,
            lw=1.7,
            label=f"T={iteration}",
        )
        axis.scatter(
            [x_axis[selected_index]],
            [
                np.log10(
                    max(
                        mse_by_iteration[iteration][selected_index],
                        1e-12,
                    )
                )
            ],
            color=color,
            s=36,
            edgecolor=INK,
            linewidth=0.4,
            zorder=5,
        )
    axis.plot(
        x_axis,
        np.log10(exact_mse.clip(1e-12)),
        color=INK,
        lw=1.8,
        ls=":",
        label="exact CD",
    )
    axis.axvline(x_axis[selected_index], color=GOLD, lw=1.2, ls="--")
    axis.set_xlabel("log10(effective lambda ratio)", fontsize=12)
    axis.set_ylabel("log10(held-out MSE)", fontsize=12)
    axis.set_title("Held-out error path", fontsize=13)
    axis.legend(frameon=False, fontsize=9, ncol=2)
    style_axis(axis)

    axis = axes[2]
    iteration_axis = np.asarray(iterations, dtype=np.float64)
    loss_rmse = np.asarray(
        [
            metrics_by_iteration[iteration]["loss_rmse"]
            for iteration in iterations
        ]
    )
    path_correlation = np.asarray(
        [
            metrics_by_iteration[iteration]["path_correlation"]
            for iteration in iterations
        ]
    )
    axis.plot(
        iteration_axis,
        loss_rmse,
        color=BLUE,
        marker="o",
        lw=2.0,
        label="loss RMSE",
    )
    axis.set_xlabel("exact-prox iterations", fontsize=12)
    axis.set_ylabel("loss RMSE vs exact CD", color=BLUE, fontsize=12)
    axis.tick_params(axis="y", colors=BLUE)
    axis.set_xscale("log")
    axis.set_xticks(iterations, [str(value) for value in iterations])
    style_axis(axis)
    twin = axis.twinx()
    twin.plot(
        iteration_axis,
        path_correlation,
        color=ORANGE,
        marker="s",
        lw=2.0,
        label="path correlation",
    )
    twin.set_ylabel("coefficient correlation", color=ORANGE, fontsize=12)
    twin.tick_params(axis="y", colors=ORANGE)
    twin.spines["top"].set_visible(False)
    handles_left, labels_left = axis.get_legend_handles_labels()
    handles_right, labels_right = twin.get_legend_handles_labels()
    axis.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        frameon=False,
        fontsize=9,
    )
    axis.set_title("Approximation by iteration depth", fontsize=13)

    figure.suptitle(
        f"Kalbi V5 iteration sweep | {task_name} | N={n}, P={p}",
        fontsize=15,
        color=INK,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.0)
    figure.savefig(output_path, dpi=220, facecolor="white")
    plt.close(figure)


def plot_summary(
    output_path: Path,
    *,
    iterations: tuple[int, ...],
    tasks: list[dict[str, Any]],
    aggregate: dict[int, dict[str, float]],
) -> None:
    iteration_axis = np.asarray(iterations, dtype=np.float64)
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))
    panels = (
        ("loss_rmse", "Loss RMSE vs exact CD", "RMSE"),
        (
            "path_correlation",
            "Coefficient path correlation vs exact CD",
            "correlation",
        ),
        (
            "selected_mse",
            "Held-out MSE at fixed selector lambda",
            "MSE",
        ),
        (
            "cumulative_decode_seconds",
            "Cumulative beta decode time",
            "seconds per four-task mean",
        ),
    )

    for axis, (key, title, ylabel) in zip(
        axes.flat, panels, strict=True
    ):
        for task in tasks:
            values = [
                task["iterations"][str(iteration)][key]
                for iteration in iterations
            ]
            axis.plot(
                iteration_axis,
                values,
                color="#B1B7BE",
                lw=1.0,
                alpha=0.65,
            )
        mean_values = [
            aggregate[iteration][key] for iteration in iterations
        ]
        axis.plot(
            iteration_axis,
            mean_values,
            color=BLUE,
            marker="o",
            lw=2.4,
            label="mean",
        )
        axis.set_xscale("log")
        axis.set_xticks(iterations, [str(value) for value in iterations])
        axis.set_xlabel("exact-prox iterations", fontsize=11)
        axis.set_ylabel(ylabel, fontsize=11)
        axis.set_title(title, fontsize=12)
        style_axis(axis)
        axis.legend(frameon=False, fontsize=9)

    axes[0, 0].axhline(
        np.mean(
            [task["baselines"]["v4_loss_rmse"] for task in tasks]
        ),
        color=ORANGE,
        lw=1.3,
        ls="--",
        label="V4 mean",
    )
    axes[0, 1].axhline(
        np.mean(
            [
                task["baselines"]["v4_path_correlation"]
                for task in tasks
            ]
        ),
        color=ORANGE,
        lw=1.3,
        ls="--",
        label="V4 mean",
    )
    axes[1, 0].axhline(
        np.mean(
            [task["baselines"]["v4_selected_mse"] for task in tasks]
        ),
        color=ORANGE,
        lw=1.3,
        ls="--",
        label="V4 mean",
    )
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.legend(frameon=False, fontsize=9)

    figure.suptitle(
        "Kalbi V5 exact-prox iteration-depth sweep",
        fontsize=16,
        color=INK,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0, w_pad=2.0)
    figure.savefig(output_path, dpi=220, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-eval", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=(1, 3, 5, 10, 20),
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    iterations = tuple(sorted(set(args.iterations)))
    if not iterations or iterations[0] < 1:
        parser.error("--iterations must contain positive integers")
    if args.out_dir.exists():
        raise RuntimeError(f"refusing existing output directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    source_metrics = json.loads((args.source_eval / "metrics.json").read_text())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint, model, selector = load_components_v19(
        args.v3_checkpoint,
        args.config,
        device,
    )
    if checkpoint.get("metadata", {}).get("model_version") != V3_SOURCE_MODEL_VERSION:
        raise RuntimeError("V5 source checkpoint is not strict Kalbi V3")
    checkpoint_summary = {
        "path": str(args.v3_checkpoint.resolve()),
        "sha256": file_sha256(args.v3_checkpoint),
        "step": int(checkpoint["completed_steps"]),
        "source_model_version": V3_SOURCE_MODEL_VERSION,
        "evaluated_model_version": MODEL_VERSION,
    }
    del checkpoint
    gc.collect()

    task_results: list[dict[str, Any]] = []
    for task_metadata in source_metrics["tasks"]:
        task_name = str(task_metadata["task"])
        print(f"evaluating iteration sweep on {task_name}", flush=True)
        x_raw, y_raw, max_corr, provenance = load_generated_task(
            Path(task_metadata["relative_path"])
        )
        x_train, y_train, x_test, y_test = split_standardize(
            x_raw,
            y_raw,
            int(task_metadata["split_seed"]),
        )
        x_device, y_device = tensors_from_arrays(x_train, y_train, device)

        synchronize(device)
        encode_started = time.perf_counter()
        encoded = encode_dataset(model, x_device, y_device)
        synchronize(device)
        encode_seconds = time.perf_counter() - encode_started

        lambda_max_tensor = encoded["lambda_max"].float().reshape(-1)
        lambda_max = float(lambda_max_tensor[0])
        if not math.isclose(
            lambda_max,
            float(task_metadata["lambda_max"]),
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            raise RuntimeError(f"{task_name} lambda_max drift")
        with torch.inference_mode():
            selected = selector(
                x_device,
                y_device,
                lambda_max_tensor,
                return_representation=True,
            )
        contract = penalty_contract(selector, selected)
        if not math.isclose(
            contract["selected_ratio"],
            float(task_metadata["v3_selector"]["selected_ratio"]),
            rel_tol=1e-6,
            abs_tol=1e-8,
        ):
            raise RuntimeError(f"{task_name} selector output drift")

        source_task = args.source_eval / task_name
        with np.load(source_task / "paths.npz", allow_pickle=False) as arrays:
            ratios = np.asarray(
                arrays["effective_lambda_ratios"], dtype=np.float64
            )
            source = {
                key: np.asarray(arrays[key], dtype=np.float64)
                for key in (
                    "v3_beta",
                    "v4_shared_penalty_beta",
                    "exact_cd_beta",
                    "v3_train_loss",
                    "v4_shared_penalty_train_loss",
                    "exact_cd_train_loss",
                    "v3_test_mse",
                    "v4_shared_penalty_test_mse",
                    "exact_cd_test_mse",
                )
            }
        raw_lambdas = (
            ratios * lambda_max / max(float(contract["d0"]), 1e-12)
        )
        shape = contract["shape"].expand(ratios.size, -1)
        d0 = contract["d0_tensor"].expand(ratios.size)
        (
            beta_by_iteration,
            beta0,
            cumulative_seconds,
            cumulative_change,
        ) = predict_iteration_sweep(
            model,
            encoded,
            torch.as_tensor(
                raw_lambdas,
                device=device,
                dtype=torch.float32,
            ),
            shape,
            d0,
            iterations,
        )
        if not np.allclose(
            beta0,
            source["v3_beta"],
            rtol=1e-5,
            atol=1e-6,
        ):
            raise RuntimeError(f"{task_name} V3 beta initialization drift")

        f_zero = 0.5 * float(
            np.mean(np.asarray(y_train, dtype=np.float64) ** 2)
        )
        selected_index = int(
            np.argmin(np.abs(ratios - contract["selected_ratio"]))
        )
        loss_by_iteration: dict[int, np.ndarray] = {}
        mse_by_iteration: dict[int, np.ndarray] = {}
        metrics_by_iteration: dict[int, dict[str, float]] = {}
        previous_objective: np.ndarray | None = None
        for iteration in iterations:
            beta = beta_by_iteration[iteration]
            _, _, objective = objective_paths(
                x_train,
                y_train,
                beta,
                raw_lambdas,
                contract["knots"],
                contract["relu_b"],
                contract["relu_weights"],
            )
            loss = safe_loss_curve(objective, f_zero)
            mse = test_mse(x_test, y_test, beta)
            loss_by_iteration[iteration] = loss
            mse_by_iteration[iteration] = mse
            metrics_by_iteration[iteration] = {
                "loss_rmse": float(
                    np.sqrt(
                        np.nanmean(
                            (loss - source["exact_cd_train_loss"]) ** 2
                        )
                    )
                ),
                "path_correlation": float(
                    np.mean(
                        [
                            vector_correlation(
                                beta[index],
                                source["exact_cd_beta"][index],
                            )
                            for index in range(ratios.size)
                        ]
                    )
                ),
                "selected_mse": float(mse[selected_index]),
                "oracle_mse": float(np.min(mse)),
                "median_cumulative_change_rms": float(
                    np.median(cumulative_change[iteration])
                ),
                "cumulative_decode_seconds": float(
                    cumulative_seconds[iteration]
                ),
                "objective_nonincrease_fraction_from_previous_target": (
                    1.0
                    if previous_objective is None
                    else float(
                        np.mean(
                            objective
                            <= previous_objective
                            + 1e-8
                            * np.maximum(1.0, np.abs(previous_objective))
                        )
                    )
                ),
            }
            previous_objective = objective

        task_output = args.out_dir / task_name
        task_output.mkdir(parents=True)
        np.savez_compressed(
            task_output / "iteration_sweep_paths.npz",
            effective_lambda_ratios=ratios,
            exact_cd_train_loss=source["exact_cd_train_loss"],
            exact_cd_test_mse=source["exact_cd_test_mse"],
            **{
                f"v5_t{iteration}_beta": beta_by_iteration[iteration].astype(
                    np.float32
                )
                for iteration in iterations
            },
            **{
                f"v5_t{iteration}_train_loss": loss_by_iteration[iteration]
                for iteration in iterations
            },
            **{
                f"v5_t{iteration}_test_mse": mse_by_iteration[iteration]
                for iteration in iterations
            },
        )
        figure_path = task_output / "v5_iteration_sweep.png"
        plot_task(
            figure_path,
            task_name=task_name,
            n=int(provenance["n"]),
            p=int(provenance["p"]),
            ratios=ratios,
            iterations=iterations,
            loss_by_iteration=loss_by_iteration,
            mse_by_iteration=mse_by_iteration,
            exact_loss=source["exact_cd_train_loss"],
            exact_mse=source["exact_cd_test_mse"],
            selected_index=selected_index,
            metrics_by_iteration=metrics_by_iteration,
        )
        task_result = {
            "task": task_name,
            "n": int(provenance["n"]),
            "p": int(provenance["p"]),
            "max_abs_feature_corr": float(max_corr),
            "selected_ratio": float(contract["selected_ratio"]),
            "encode_seconds": encode_seconds,
            "iterations": {
                str(iteration): metrics_by_iteration[iteration]
                for iteration in iterations
            },
            "baselines": {
                "v3_loss_rmse": float(
                    np.sqrt(
                        np.nanmean(
                            (
                                source["v3_train_loss"]
                                - source["exact_cd_train_loss"]
                            )
                            ** 2
                        )
                    )
                ),
                "v4_loss_rmse": float(
                    np.sqrt(
                        np.nanmean(
                            (
                                source["v4_shared_penalty_train_loss"]
                                - source["exact_cd_train_loss"]
                            )
                            ** 2
                        )
                    )
                ),
                "v3_path_correlation": float(
                    np.mean(
                        [
                            vector_correlation(
                                source["v3_beta"][index],
                                source["exact_cd_beta"][index],
                            )
                            for index in range(ratios.size)
                        ]
                    )
                ),
                "v4_path_correlation": float(
                    np.mean(
                        [
                            vector_correlation(
                                source["v4_shared_penalty_beta"][index],
                                source["exact_cd_beta"][index],
                            )
                            for index in range(ratios.size)
                        ]
                    )
                ),
                "v3_selected_mse": float(
                    source["v3_test_mse"][selected_index]
                ),
                "v4_selected_mse": float(
                    source["v4_shared_penalty_test_mse"][selected_index]
                ),
            },
            "figure": str(figure_path.resolve()),
        }
        (task_output / "metrics.json").write_text(
            json.dumps(task_result, indent=2, sort_keys=True) + "\n"
        )
        task_results.append(task_result)
        print(json.dumps(task_result, sort_keys=True), flush=True)

    aggregate = {
        iteration: {
            key: float(
                np.mean(
                    [
                        task["iterations"][str(iteration)][key]
                        for task in task_results
                    ]
                )
            )
            for key in (
                "loss_rmse",
                "path_correlation",
                "selected_mse",
                "oracle_mse",
                "median_cumulative_change_rms",
                "cumulative_decode_seconds",
                "objective_nonincrease_fraction_from_previous_target",
            )
        }
        for iteration in iterations
    }
    summary_path = args.out_dir / "v5_iteration_sweep_summary.png"
    plot_summary(
        summary_path,
        iterations=iterations,
        tasks=task_results,
        aggregate=aggregate,
    )
    report = {
        "report_version": "kalbi_v5_exactprox_iteration_sweep_lowdim4_v1",
        "iterations": list(iterations),
        "checkpoint": checkpoint_summary,
        "source_evaluation": str(args.source_eval.resolve()),
        "comparison_contract": (
            "same tasks, splits, standardization, V3 initialization, "
            "V3 selector penalty, effective-lambda grid, and exact-CD reference"
        ),
        "aggregate": {
            str(iteration): aggregate[iteration] for iteration in iterations
        },
        "tasks": task_results,
        "figure": str(summary_path.resolve()),
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
