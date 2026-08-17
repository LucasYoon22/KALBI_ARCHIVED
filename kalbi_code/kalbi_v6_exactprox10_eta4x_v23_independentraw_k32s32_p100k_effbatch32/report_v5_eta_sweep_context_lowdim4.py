#!/usr/bin/env python3
"""Compare aggressive Kalbi V5 exact-prox step-size initializations."""

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

from . import model as model_module
from .eval_kalbi_v2_common import (
    encode_dataset,
    file_sha256,
    split_standardize,
    tensors_from_arrays,
    vector_correlation,
)
from .model import MODEL_VERSION, V3_SOURCE_MODEL_VERSION
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
SAFETY_COLORS = (BLUE, ORANGE, OLIVE, MAGENTA, GOLD)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def safety_key(safety: float) -> str:
    return f"{float(safety):g}"


def eta_factor(safety: float) -> float:
    return 2.0 / float(safety)


def predict_eta_trajectory(
    model: torch.nn.Module,
    encoded: dict[str, Any],
    beta0: np.ndarray,
    lambdas: torch.Tensor,
    shape: torch.Tensor,
    d0: torch.Tensor,
    iterations: tuple[int, ...],
    *,
    ritz_safety: float,
    chunk_size: int = 8,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, float],
    dict[int, dict[str, float]],
]:
    if ritz_safety <= 0.0:
        raise ValueError("ritz_safety must be positive")
    model_module.KALBI_EXACT_PG_RITZ_SAFETY = float(ritz_safety)
    lambdas = lambdas.reshape(-1).float()
    shape = shape.reshape(lambdas.numel(), -1).float()
    d0 = d0.reshape(-1).float()
    max_iterations = max(iterations)
    targets = set(iterations)
    beta0_tensor = torch.as_tensor(
        beta0,
        device=lambdas.device,
        dtype=torch.float32,
    )
    beta_parts: dict[int, list[torch.Tensor]] = {
        iteration: [] for iteration in iterations
    }
    per_iteration_seconds = {
        iteration: 0.0 for iteration in range(1, max_iterations + 1)
    }
    diagnostics: dict[int, dict[str, list[torch.Tensor]]] = {
        iteration: {
            "step_size": [],
            "backtracks": [],
            "majorization_ratio": [],
            "curvature_step_cap": [],
            "change_rms": [],
        }
        for iteration in range(1, max_iterations + 1)
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
            beta_current = beta0_tensor[start:stop][None]

            for iteration in range(1, max_iterations + 1):
                synchronize(lambdas.device)
                started = time.perf_counter()
                refinement = model_module._refine_beta_kalbi_v5(
                    model,
                    encoded,
                    beta_current,
                    chunk_lambdas,
                    chunk_d0,
                    chunk_shape,
                )
                beta_current = refinement["beta"]
                synchronize(lambdas.device)
                per_iteration_seconds[iteration] += (
                    time.perf_counter() - started
                )
                for key in diagnostics[iteration]:
                    diagnostics[iteration][key].append(
                        refinement[key][0].float().reshape(-1).cpu()
                    )
                if iteration in targets:
                    beta_parts[iteration].append(
                        beta_current[0].float().cpu()
                    )

    beta_by_iteration = {
        iteration: torch.cat(beta_parts[iteration])
        .numpy()
        .astype(np.float64)
        for iteration in iterations
    }
    cumulative_seconds: dict[int, float] = {}
    running_seconds = 0.0
    for iteration in range(1, max_iterations + 1):
        running_seconds += per_iteration_seconds[iteration]
        if iteration in targets:
            cumulative_seconds[iteration] = running_seconds

    summarized: dict[int, dict[str, float]] = {}
    for target in iterations:
        stacked = {
            key: torch.cat(
                [
                    torch.cat(diagnostics[iteration][key])
                    for iteration in range(1, target + 1)
                ]
            )
            .numpy()
            .astype(np.float64)
            for key in diagnostics[1]
        }
        summarized[target] = {
            "accepted_eta_median": float(np.median(stacked["step_size"])),
            "accepted_eta_min": float(np.min(stacked["step_size"])),
            "accepted_eta_max": float(np.max(stacked["step_size"])),
            "mean_backtracks": float(np.mean(stacked["backtracks"])),
            "backtrack_fraction": float(np.mean(stacked["backtracks"] > 0)),
            "majorization_ratio_median": float(
                np.median(stacked["majorization_ratio"])
            ),
            "majorization_ratio_max": float(
                np.max(stacked["majorization_ratio"])
            ),
            "curvature_cap_binding_fraction": float(
                np.mean(
                    np.isclose(
                        stacked["step_size"],
                        stacked["curvature_step_cap"],
                        rtol=1e-5,
                        atol=1e-10,
                    )
                )
            ),
            "last_step_change_rms_median": float(
                np.median(
                    torch.cat(diagnostics[target]["change_rms"])
                    .numpy()
                    .astype(np.float64)
                )
            ),
        }
    return beta_by_iteration, cumulative_seconds, summarized


def plot_task(
    output_path: Path,
    *,
    task_name: str,
    n: int,
    p: int,
    ratios: np.ndarray,
    iterations: tuple[int, ...],
    safeties: tuple[float, ...],
    loss_curves: dict[str, dict[int, np.ndarray]],
    exact_loss: np.ndarray,
    selected_iteration: int,
    metrics: dict[str, dict[int, dict[str, float]]],
) -> None:
    x_axis = np.log10(ratios)
    factors = [eta_factor(safety) for safety in safeties]
    figure, axes = plt.subplots(1, 3, figsize=(20.0, 5.9))

    axis = axes[0]
    for color, safety in zip(SAFETY_COLORS, safeties, strict=False):
        key = safety_key(safety)
        factor = eta_factor(safety)
        axis.plot(
            x_axis,
            loss_curves[key][selected_iteration],
            color=color,
            lw=1.8,
            label=f"eta {factor:g}x",
        )
    axis.plot(
        x_axis,
        exact_loss,
        color=INK,
        lw=1.8,
        ls=":",
        label="exact CD",
    )
    axis.set_xlabel("log10(effective lambda ratio)", fontsize=12)
    axis.set_ylabel("log1p(F/F0)", fontsize=12)
    axis.set_title(
        f"Training objective at T={selected_iteration}",
        fontsize=13,
    )
    axis.legend(frameon=False, fontsize=9)
    style_axis(axis)

    axis = axes[1]
    iteration_axis = np.asarray(iterations, dtype=np.float64)
    for color, safety in zip(SAFETY_COLORS, safeties, strict=False):
        key = safety_key(safety)
        factor = eta_factor(safety)
        values = [
            metrics[key][iteration]["loss_rmse"]
            for iteration in iterations
        ]
        axis.plot(
            iteration_axis,
            values,
            color=color,
            marker="o",
            lw=2.0,
            label=f"eta {factor:g}x",
        )
    axis.set_xscale("log")
    axis.set_xticks(iterations, [str(value) for value in iterations])
    axis.set_xlabel("exact-prox iterations", fontsize=12)
    axis.set_ylabel("loss RMSE vs exact CD", fontsize=12)
    axis.set_title("Approximation rate", fontsize=13)
    axis.legend(frameon=False, fontsize=9)
    style_axis(axis)

    axis = axes[2]
    eta_values = [
        metrics[safety_key(safety)][selected_iteration][
            "accepted_eta_median"
        ]
        for safety in safeties
    ]
    backtrack_values = [
        metrics[safety_key(safety)][selected_iteration][
            "backtrack_fraction"
        ]
        for safety in safeties
    ]
    axis.plot(
        factors,
        eta_values,
        color=BLUE,
        marker="o",
        lw=2.0,
        label="accepted eta",
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks(factors, [f"{factor:g}x" for factor in factors])
    axis.set_xlabel("initial eta multiplier", fontsize=12)
    axis.set_ylabel("median accepted eta", color=BLUE, fontsize=12)
    axis.tick_params(axis="y", colors=BLUE)
    style_axis(axis)
    twin = axis.twinx()
    twin.plot(
        factors,
        backtrack_values,
        color=ORANGE,
        marker="s",
        lw=2.0,
        label="backtrack fraction",
    )
    twin.set_ylabel("backtrack fraction", color=ORANGE, fontsize=12)
    twin.tick_params(axis="y", colors=ORANGE)
    twin.set_ylim(-0.03, max(0.1, 1.1 * max(backtrack_values)))
    twin.spines["top"].set_visible(False)
    handles_left, labels_left = axis.get_legend_handles_labels()
    handles_right, labels_right = twin.get_legend_handles_labels()
    axis.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        frameon=False,
        fontsize=9,
    )
    axis.set_title(
        f"Line-search behavior through T={selected_iteration}",
        fontsize=13,
    )

    figure.suptitle(
        f"Kalbi V5 eta sweep | {task_name} | N={n}, P={p}",
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
    safeties: tuple[float, ...],
    aggregate: dict[str, dict[int, dict[str, float]]],
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
            "cumulative_refinement_seconds",
            "Cumulative refinement time",
            "seconds per task",
        ),
    )

    for axis, (metric, title, ylabel) in zip(
        axes.flat, panels, strict=True
    ):
        for color, safety in zip(
            SAFETY_COLORS, safeties, strict=False
        ):
            key = safety_key(safety)
            factor = eta_factor(safety)
            values = [
                aggregate[key][iteration][metric]
                for iteration in iterations
            ]
            axis.plot(
                iteration_axis,
                values,
                color=color,
                marker="o",
                lw=2.1,
                label=f"eta {factor:g}x",
            )
        axis.set_xscale("log")
        axis.set_xticks(iterations, [str(value) for value in iterations])
        axis.set_xlabel("exact-prox iterations", fontsize=11)
        axis.set_ylabel(ylabel, fontsize=11)
        axis.set_title(title, fontsize=12)
        axis.legend(frameon=False, fontsize=9)
        style_axis(axis)

    figure.suptitle(
        "Kalbi V5 proximal-gradient eta sweep",
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
        "--ritz-safeties",
        type=float,
        nargs="+",
        default=(2.0, 1.0, 0.5, 0.25),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=(1, 3, 5, 10),
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    safeties = tuple(args.ritz_safeties)
    iterations = tuple(sorted(set(args.iterations)))
    if not safeties or min(safeties) <= 0.0:
        parser.error("--ritz-safeties must contain positive values")
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
        print(f"evaluating eta sweep on {task_name}", flush=True)
        x_raw, y_raw, max_corr, provenance = load_generated_task(
            Path(task_metadata["relative_path"])
        )
        x_train, y_train, x_test, y_test = split_standardize(
            x_raw,
            y_raw,
            int(task_metadata["split_seed"]),
        )
        x_device, y_device = tensors_from_arrays(x_train, y_train, device)
        encoded = encode_dataset(model, x_device, y_device)
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

        source_task = args.source_eval / task_name
        with np.load(source_task / "paths.npz", allow_pickle=False) as arrays:
            ratios = np.asarray(
                arrays["effective_lambda_ratios"], dtype=np.float64
            )
            source = {
                key: np.asarray(arrays[key], dtype=np.float64)
                for key in (
                    "v3_beta",
                    "exact_cd_beta",
                    "exact_cd_train_loss",
                )
            }
        raw_lambdas = (
            ratios * lambda_max / max(float(contract["d0"]), 1e-12)
        )
        lambdas_tensor = torch.as_tensor(
            raw_lambdas,
            device=device,
            dtype=torch.float32,
        )
        shape = contract["shape"].expand(ratios.size, -1)
        d0 = contract["d0_tensor"].expand(ratios.size)
        f_zero = 0.5 * float(
            np.mean(np.asarray(y_train, dtype=np.float64) ** 2)
        )
        selected_index = int(
            np.argmin(np.abs(ratios - contract["selected_ratio"]))
        )

        loss_curves: dict[str, dict[int, np.ndarray]] = {}
        metrics: dict[str, dict[int, dict[str, float]]] = {}
        beta_to_save: dict[str, np.ndarray] = {}
        for safety in safeties:
            key = safety_key(safety)
            factor = eta_factor(safety)
            print(
                f"  safety={safety:g}, initial_eta_factor={factor:g}x",
                flush=True,
            )
            (
                beta_by_iteration,
                cumulative_seconds,
                diagnostics,
            ) = predict_eta_trajectory(
                model,
                encoded,
                source["v3_beta"],
                lambdas_tensor,
                shape,
                d0,
                iterations,
                ritz_safety=safety,
            )
            loss_curves[key] = {}
            metrics[key] = {}
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
                loss_curves[key][iteration] = loss
                metric = {
                    "loss_rmse": float(
                        np.sqrt(
                            np.nanmean(
                                (
                                    loss - source["exact_cd_train_loss"]
                                )
                                ** 2
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
                    "cumulative_refinement_seconds": float(
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
                                * np.maximum(
                                    1.0, np.abs(previous_objective)
                                )
                            )
                        )
                    ),
                    **diagnostics[iteration],
                }
                metrics[key][iteration] = metric
                previous_objective = objective
                beta_to_save[
                    f"eta{factor:g}x_t{iteration}_beta"
                ] = beta.astype(np.float32)

        task_output = args.out_dir / task_name
        task_output.mkdir(parents=True)
        np.savez_compressed(
            task_output / "eta_sweep_paths.npz",
            effective_lambda_ratios=ratios,
            exact_cd_train_loss=source["exact_cd_train_loss"],
            **beta_to_save,
            **{
                f"eta{eta_factor(safety):g}x_t{iteration}_train_loss": (
                    loss_curves[safety_key(safety)][iteration]
                )
                for safety in safeties
                for iteration in iterations
            },
        )
        figure_path = task_output / "v5_eta_sweep.png"
        plot_task(
            figure_path,
            task_name=task_name,
            n=int(provenance["n"]),
            p=int(provenance["p"]),
            ratios=ratios,
            iterations=iterations,
            safeties=safeties,
            loss_curves=loss_curves,
            exact_loss=source["exact_cd_train_loss"],
            selected_iteration=max(iterations),
            metrics=metrics,
        )
        task_result = {
            "task": task_name,
            "n": int(provenance["n"]),
            "p": int(provenance["p"]),
            "max_abs_feature_corr": float(max_corr),
            "selected_ratio": float(contract["selected_ratio"]),
            "settings": {
                key: {
                    "ritz_safety": float(safety),
                    "initial_eta_factor": eta_factor(safety),
                    "iterations": {
                        str(iteration): metrics[key][iteration]
                        for iteration in iterations
                    },
                }
                for safety in safeties
                for key in (safety_key(safety),)
            },
            "figure": str(figure_path.resolve()),
        }
        (task_output / "metrics.json").write_text(
            json.dumps(task_result, indent=2, sort_keys=True) + "\n"
        )
        task_results.append(task_result)
        print(json.dumps(task_result, sort_keys=True), flush=True)

    aggregate: dict[str, dict[int, dict[str, float]]] = {}
    aggregate_keys = (
        "loss_rmse",
        "path_correlation",
        "selected_mse",
        "oracle_mse",
        "cumulative_refinement_seconds",
        "accepted_eta_median",
        "accepted_eta_min",
        "accepted_eta_max",
        "mean_backtracks",
        "backtrack_fraction",
        "majorization_ratio_median",
        "majorization_ratio_max",
        "curvature_cap_binding_fraction",
        "last_step_change_rms_median",
        "objective_nonincrease_fraction_from_previous_target",
    )
    for safety in safeties:
        key = safety_key(safety)
        aggregate[key] = {}
        for iteration in iterations:
            aggregate[key][iteration] = {
                metric: float(
                    np.mean(
                        [
                            task["settings"][key]["iterations"][
                                str(iteration)
                            ][metric]
                            for task in task_results
                        ]
                    )
                )
                for metric in aggregate_keys
            }

    summary_path = args.out_dir / "v5_eta_sweep_summary.png"
    plot_summary(
        summary_path,
        iterations=iterations,
        safeties=safeties,
        aggregate=aggregate,
    )
    report = {
        "report_version": "kalbi_v5_exactprox_eta_sweep_lowdim4_v1",
        "iterations": list(iterations),
        "ritz_safeties": list(safeties),
        "eta_factors": [eta_factor(safety) for safety in safeties],
        "checkpoint": checkpoint_summary,
        "source_evaluation": str(args.source_eval.resolve()),
        "comparison_contract": (
            "same tasks, splits, standardization, V3 beta initialization, "
            "V3 selector penalty, lambda grid, and exact-CD reference; "
            "only the initial Ritz safety changes"
        ),
        "aggregate": {
            key: {
                str(iteration): aggregate[key][iteration]
                for iteration in iterations
            }
            for key in aggregate
        },
        "tasks": task_results,
        "figure": str(summary_path.resolve()),
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    model_module.KALBI_EXACT_PG_RITZ_SAFETY = 2.0


if __name__ == "__main__":
    main()
