#!/usr/bin/env python3
"""Add Kalbi V5 exact-prox to the fixed V3/V4 low-dimensional comparison."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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


INK = "#20242A"
LIGHT = "#B1B7BE"
BLUE = "#2367A6"
ORANGE = "#CF6A32"
OLIVE = "#768B36"
GOLD = "#D39B22"
FEATURE_COLORS = (
    "#2367A6",
    "#CF6A32",
    "#768B36",
    "#B64F7C",
    "#D39B22",
    "#535B65",
    "#6F8FB3",
    "#9D704B",
)

AUX_KEYS = (
    "kalbi_v5_refinement_step_size_q",
    "kalbi_v5_refinement_gamma_q",
    "kalbi_v5_refinement_kappa_q",
    "kalbi_v5_refinement_min_curvature_denominator_q",
    "kalbi_v5_refinement_curvature_step_cap_q",
    "kalbi_v5_refinement_majorization_ratio_q",
    "kalbi_v5_refinement_backtracks_q",
    "kalbi_v5_refinement_change_rms_q",
    "kalbi_v5_refined_active_fraction_q",
    "kalbi_v5_beta0_active_fraction_q",
)


def predict_with_aux(
    model: torch.nn.Module,
    encoded: dict[str, Any],
    lambdas: torch.Tensor,
    shape: torch.Tensor,
    d0: torch.Tensor,
    *,
    chunk_size: int = 8,
    refinement_steps: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if refinement_steps < 1:
        raise ValueError("refinement_steps must be at least 1")
    lambdas = lambdas.reshape(-1).float()
    shape = shape.reshape(lambdas.numel(), -1).float()
    d0 = d0.reshape(-1).float()
    lambda_max = encoded["lambda_max"].float().reshape(-1)[0]
    beta_parts: list[torch.Tensor] = []
    beta0_parts: list[torch.Tensor] = []
    aux_parts: dict[str, list[torch.Tensor]] = {key: [] for key in AUX_KEYS}
    with torch.inference_mode(), torch.autocast(
        device_type=lambdas.device.type,
        dtype=torch.bfloat16,
        enabled=lambdas.device.type == "cuda",
    ):
        for start in range(0, lambdas.numel(), int(chunk_size)):
            stop = min(start + int(chunk_size), lambdas.numel())
            output = model.decode_queries(
                encoded,
                lambdas=lambdas[start:stop][None],
                lambda_max=lambda_max.expand(1, stop - start),
                penalty_shape_features=shape[start:stop][None],
                penalty_d0=d0[start:stop][None],
                return_aux=True,
            )
            beta_current = output["beta_q"]
            final_refinement = None
            for _ in range(1, refinement_steps):
                final_refinement = _refine_beta_kalbi_v5(
                    model,
                    encoded,
                    beta_current,
                    lambdas[start:stop][None],
                    d0[start:stop][None],
                    shape[start:stop][None],
                )
                beta_current = final_refinement["beta"]
            beta_parts.append(beta_current[0].float().cpu())
            beta0_parts.append(output["kalbi_v5_beta0_q"][0].float().cpu())
            for key in AUX_KEYS:
                if (
                    final_refinement is None
                    or key == "kalbi_v5_beta0_active_fraction_q"
                ):
                    value = output[key]
                else:
                    refinement_key = key.removeprefix(
                        "kalbi_v5_refinement_"
                    ).removesuffix("_q")
                    if key == "kalbi_v5_refined_active_fraction_q":
                        refinement_key = "active_fraction"
                    elif refinement_key == "min_curvature_denominator":
                        refinement_key = "minimum_curvature_denominator"
                    value = final_refinement[refinement_key]
                aux_parts[key].append(value[0].float().reshape(-1).cpu())
    beta = torch.cat(beta_parts).numpy().astype(np.float64)
    beta0 = torch.cat(beta0_parts).numpy().astype(np.float64)
    aux = {
        key: torch.cat(parts).numpy().astype(np.float64)
        for key, parts in aux_parts.items()
    }
    return beta, beta0, aux


def penalty_contract(
    selector: torch.nn.Module,
    selected: dict[str, torch.Tensor],
) -> dict[str, Any]:
    return {
        "knots": (
            selector.penalty_parameter_head.penalty_grid.detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        ),
        "relu_b": float(selected["penalty_relu_b"].reshape(-1)[0]),
        "relu_weights": (
            selected["penalty_relu_weights"]
            .reshape(-1, selected["penalty_relu_weights"].shape[-1])[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        ),
        "shape": selected["penalty_shape_features"].reshape(1, -1),
        "d0_tensor": selected["penalty_d0"].reshape(-1),
        "d0": float(selected["penalty_d0"].reshape(-1)[0]),
        "selected_ratio": float(
            selected["effective_lambda_ratio"].reshape(-1)[0]
        ),
    }


def plot_task(
    output_path: Path,
    *,
    task_name: str,
    n: int,
    p: int,
    ratios: np.ndarray,
    v3_beta: np.ndarray,
    v4_beta: np.ndarray,
    v5_beta: np.ndarray,
    cd_beta: np.ndarray,
    v3_loss: np.ndarray,
    v4_loss: np.ndarray,
    v5_loss: np.ndarray,
    cd_loss: np.ndarray,
    v3_mse: np.ndarray,
    v4_mse: np.ndarray,
    v5_mse: np.ndarray,
    cd_mse: np.ndarray,
    selected_index: int,
    metrics: dict[str, Any],
    refinement_steps: int,
) -> None:
    x_axis = np.log10(ratios)
    score = np.maximum.reduce(
        (
            np.max(np.abs(v3_beta), axis=0),
            np.max(np.abs(v4_beta), axis=0),
            np.max(np.abs(v5_beta), axis=0),
            np.max(np.abs(cd_beta), axis=0),
        )
    )
    top = np.argsort(-score)[:8]
    figure, axes = plt.subplots(1, 3, figsize=(20.0, 5.9))

    axis = axes[0]
    for color_index, feature in enumerate(top):
        color = FEATURE_COLORS[color_index % len(FEATURE_COLORS)]
        axis.plot(x_axis, v3_beta[:, feature], color=color, lw=1.35)
        axis.plot(
            x_axis,
            v4_beta[:, feature],
            color=color,
            lw=1.15,
            ls="--",
        )
        axis.plot(
            x_axis,
            v5_beta[:, feature],
            color=color,
            lw=1.15,
            ls="-.",
        )
        axis.plot(
            x_axis,
            cd_beta[:, feature],
            color=color,
            lw=0.95,
            ls=":",
            alpha=0.75,
        )
    axis.axhline(0.0, color=INK, lw=0.6)
    axis.axvline(x_axis[selected_index], color=GOLD, lw=1.3, ls="--")
    axis.set_yscale("symlog", linthresh=0.02)
    axis.set_xlabel("log10(effective lambda ratio)", fontsize=12)
    axis.set_ylabel("coefficient", fontsize=12)
    axis.set_title("Shared-penalty solution path", fontsize=13)
    axis.legend(
        handles=[
            Line2D([0], [0], color=INK, lw=1.5, label="V3"),
            Line2D([0], [0], color=INK, lw=1.5, ls="--", label="V4 LLA"),
            Line2D(
                [0],
                [0],
                color=INK,
                lw=1.5,
                ls="-.",
                label=f"V5 exact prox x{refinement_steps}",
            ),
            Line2D([0], [0], color=INK, lw=1.5, ls=":", label="exact CD"),
        ],
        frameon=False,
        fontsize=9,
    )
    style_axis(axis)

    axis = axes[1]
    axis.plot(x_axis, v3_loss, color=BLUE, lw=1.9, label="V3")
    axis.plot(x_axis, v4_loss, color=ORANGE, lw=1.9, label="V4 LLA")
    axis.plot(
        x_axis,
        v5_loss,
        color=OLIVE,
        lw=2.1,
        label=f"V5 exact prox x{refinement_steps}",
    )
    axis.plot(x_axis, cd_loss, color=INK, lw=1.7, ls=":", label="exact CD")
    axis.axvline(
        x_axis[selected_index],
        color=GOLD,
        lw=1.3,
        ls="--",
        label="fixed V3 selector lambda",
    )
    axis.set_xlabel("log10(effective lambda ratio)", fontsize=12)
    axis.set_ylabel("log1p(F/F0)", fontsize=12)
    axis.set_title(
        "Training objective | RMSE "
        f"V3={metrics['v3_loss_rmse']:.3g}, "
        f"V4={metrics['v4_loss_rmse']:.3g}, "
        f"V5={metrics['v5_loss_rmse']:.3g}",
        fontsize=12,
    )
    axis.legend(frameon=False, fontsize=9)
    style_axis(axis)

    axis = axes[2]
    axis.plot(
        x_axis,
        np.log10(v3_mse.clip(1e-12)),
        color=BLUE,
        lw=1.9,
        label="V3",
    )
    axis.plot(
        x_axis,
        np.log10(v4_mse.clip(1e-12)),
        color=ORANGE,
        lw=1.9,
        label="V4 LLA",
    )
    axis.plot(
        x_axis,
        np.log10(v5_mse.clip(1e-12)),
        color=OLIVE,
        lw=2.1,
        label=f"V5 exact prox x{refinement_steps}",
    )
    axis.plot(
        x_axis,
        np.log10(cd_mse.clip(1e-12)),
        color=INK,
        lw=1.6,
        ls=":",
        label="exact CD",
    )
    for values, color, marker in (
        (v3_mse, BLUE, "o"),
        (v4_mse, ORANGE, "s"),
        (v5_mse, OLIVE, "D"),
    ):
        axis.scatter(
            [x_axis[selected_index]],
            [np.log10(max(values[selected_index], 1e-12))],
            color=color,
            marker=marker,
            s=55,
            edgecolor=INK,
            linewidth=0.5,
            zorder=5,
        )
    axis.set_xlabel("log10(effective lambda ratio)", fontsize=12)
    axis.set_ylabel("log10(held-out MSE)", fontsize=12)
    axis.set_title("Held-out error under the same selector", fontsize=13)
    axis.legend(frameon=False, fontsize=9)
    style_axis(axis)

    figure.suptitle(
        f"Kalbi V3 vs V4 vs V5 x{refinement_steps} | "
        f"{task_name} | N={n}, P={p}",
        fontsize=15,
        color=INK,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.8)
    figure.savefig(output_path, dpi=220, facecolor="white")
    plt.close(figure)


def plot_summary(
    output_path: Path,
    tasks: list[dict[str, Any]],
    refinement_steps: int,
) -> None:
    labels = [task["task"].split("_", 1)[0] for task in tasks]
    positions = np.arange(len(tasks))
    width = 0.25
    panels = (
        (
            "loss_rmse",
            "Training loss RMSE vs exact CD",
            "path RMSE",
        ),
        (
            "path_correlation",
            "Solution path correlation vs exact CD",
            "correlation",
        ),
        (
            "selected_mse",
            "Held-out MSE at fixed selected lambda",
            "MSE",
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(18.0, 5.6))
    for axis, (suffix, title, ylabel) in zip(axes, panels, strict=True):
        for offset, version, color in (
            (-width, "v3", BLUE),
            (0.0, "v4", ORANGE),
            (width, "v5", OLIVE),
        ):
            values = np.asarray(
                [task[f"{version}_{suffix}"] for task in tasks],
                dtype=np.float64,
            )
            axis.bar(
                positions + offset,
                values,
                width,
                color=color,
                label=version.upper(),
            )
        axis.set_xticks(positions, labels)
        axis.set_ylabel(ylabel, fontsize=11)
        axis.set_title(title, fontsize=12)
        style_axis(axis)
    axes[0].legend(frameon=False, fontsize=10)
    figure.suptitle(
        f"One-step Kalbi V4 LLA vs V5 exact-prox x{refinement_steps}",
        fontsize=15,
        color=INK,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.8)
    figure.savefig(output_path, dpi=220, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-eval", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--refinement-steps", type=int, default=1)
    args = parser.parse_args()
    if args.refinement_steps < 1:
        parser.error("--refinement-steps must be at least 1")

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
        print(f"evaluating V5 on {task_name}", flush=True)
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
                    "v3_raw_lambdas",
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
        if not math.isclose(
            contract["selected_ratio"],
            float(task_metadata["v3_selector"]["selected_ratio"]),
            rel_tol=1e-6,
            abs_tol=1e-8,
        ):
            raise RuntimeError(f"{task_name} selector output drift")
        raw_lambdas = (
            ratios * lambda_max / max(float(contract["d0"]), 1e-12)
        )
        if not np.allclose(
            raw_lambdas,
            source["v3_raw_lambdas"],
            rtol=2e-5,
            atol=1e-8,
        ):
            raise RuntimeError(f"{task_name} raw lambda contract drift")
        shape = contract["shape"].expand(ratios.size, -1)
        d0 = contract["d0_tensor"].expand(ratios.size)
        v5_beta, v5_beta0, aux = predict_with_aux(
            model,
            encoded,
            torch.as_tensor(
                raw_lambdas,
                device=device,
                dtype=torch.float32,
            ),
            shape,
            d0,
            refinement_steps=args.refinement_steps,
        )
        f_zero = 0.5 * float(np.mean(np.asarray(y_train, dtype=np.float64) ** 2))
        _, _, v5_objective = objective_paths(
            x_train,
            y_train,
            v5_beta,
            raw_lambdas,
            contract["knots"],
            contract["relu_b"],
            contract["relu_weights"],
        )
        v5_loss = safe_loss_curve(v5_objective, f_zero)
        v5_mse = test_mse(x_test, y_test, v5_beta)
        selected_index = int(
            np.argmin(np.abs(ratios - contract["selected_ratio"]))
        )
        correlations = {
            version: float(
                np.mean(
                    [
                        vector_correlation(beta[index], source["exact_cd_beta"][index])
                        for index in range(ratios.size)
                    ]
                )
            )
            for version, beta in (
                ("v3", source["v3_beta"]),
                ("v4", source["v4_shared_penalty_beta"]),
                ("v5", v5_beta),
            )
        }
        metrics = {
            "task": task_name,
            "n": int(provenance["n"]),
            "p": int(provenance["p"]),
            "max_abs_feature_corr": float(max_corr),
            "selected_ratio": float(contract["selected_ratio"]),
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
            "v5_loss_rmse": float(
                np.sqrt(
                    np.nanmean(
                        (v5_loss - source["exact_cd_train_loss"]) ** 2
                    )
                )
            ),
            "v3_path_correlation": correlations["v3"],
            "v4_path_correlation": correlations["v4"],
            "v5_path_correlation": correlations["v5"],
            "v3_selected_mse": float(
                source["v3_test_mse"][selected_index]
            ),
            "v4_selected_mse": float(
                source["v4_shared_penalty_test_mse"][selected_index]
            ),
            "v5_selected_mse": float(v5_mse[selected_index]),
            "v3_oracle_mse": float(np.min(source["v3_test_mse"])),
            "v4_oracle_mse": float(
                np.min(source["v4_shared_penalty_test_mse"])
            ),
            "v5_oracle_mse": float(np.min(v5_mse)),
            "v5_better_loss_fraction_vs_v3": float(
                np.mean(v5_loss < source["v3_train_loss"])
            ),
            "v5_better_loss_fraction_vs_v4": float(
                np.mean(v5_loss < source["v4_shared_penalty_train_loss"])
            ),
            "v5_beta0_max_abs_difference_vs_v3": float(
                np.max(np.abs(v5_beta0 - source["v3_beta"]))
            ),
            "v5_median_step_size": float(
                np.median(aux["kalbi_v5_refinement_step_size_q"])
            ),
            "v5_selected_step_size": float(
                aux["kalbi_v5_refinement_step_size_q"][selected_index]
            ),
            "v5_median_change_rms": float(
                np.median(aux["kalbi_v5_refinement_change_rms_q"])
            ),
            "v5_median_cumulative_change_rms": float(
                np.median(
                    np.sqrt(np.mean((v5_beta - v5_beta0) ** 2, axis=-1))
                )
            ),
            "v5_selected_change_rms": float(
                aux["kalbi_v5_refinement_change_rms_q"][selected_index]
            ),
            "v5_selected_kappa": float(
                aux["kalbi_v5_refinement_kappa_q"][selected_index]
            ),
            "v5_selected_backtracks": float(
                aux["kalbi_v5_refinement_backtracks_q"][selected_index]
            ),
        }
        task_output = args.out_dir / task_name
        task_output.mkdir(parents=True)
        np.savez_compressed(
            task_output / "v5_paths.npz",
            effective_lambda_ratios=ratios,
            v5_beta=v5_beta.astype(np.float32),
            v5_beta0=v5_beta0.astype(np.float32),
            v5_train_loss=v5_loss,
            v5_test_mse=v5_mse,
            **aux,
        )
        figure_path = task_output / "v3_v4_v5_solution_loss_lambda.png"
        plot_task(
            figure_path,
            task_name=task_name,
            n=int(provenance["n"]),
            p=int(provenance["p"]),
            ratios=ratios,
            v3_beta=source["v3_beta"],
            v4_beta=source["v4_shared_penalty_beta"],
            v5_beta=v5_beta,
            cd_beta=source["exact_cd_beta"],
            v3_loss=source["v3_train_loss"],
            v4_loss=source["v4_shared_penalty_train_loss"],
            v5_loss=v5_loss,
            cd_loss=source["exact_cd_train_loss"],
            v3_mse=source["v3_test_mse"],
            v4_mse=source["v4_shared_penalty_test_mse"],
            v5_mse=v5_mse,
            cd_mse=source["exact_cd_test_mse"],
            selected_index=selected_index,
            metrics=metrics,
            refinement_steps=args.refinement_steps,
        )
        metrics["figure"] = str(figure_path.resolve())
        (task_output / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n"
        )
        task_results.append(metrics)
        print(json.dumps(metrics, sort_keys=True), flush=True)

    summary_path = args.out_dir / "v3_v4_v5_summary.png"
    plot_summary(summary_path, task_results, args.refinement_steps)
    aggregate = {
        "report_version": (
            "kalbi_v3_v4_v5_exactprox"
            f"{args.refinement_steps}_context_lowdim4_v1"
        ),
        "v5_refinement_steps": args.refinement_steps,
        "checkpoint": checkpoint_summary,
        "source_evaluation": str(args.source_eval.resolve()),
        "tasks": task_results,
        "mean": {
            key: float(np.mean([task[key] for task in task_results]))
            for key in (
                "v3_loss_rmse",
                "v4_loss_rmse",
                "v5_loss_rmse",
                "v3_path_correlation",
                "v4_path_correlation",
                "v5_path_correlation",
                "v3_selected_mse",
                "v4_selected_mse",
                "v5_selected_mse",
                "v5_better_loss_fraction_vs_v3",
                "v5_better_loss_fraction_vs_v4",
                "v5_median_step_size",
                "v5_median_change_rms",
                "v5_median_cumulative_change_rms",
            )
        },
        "figure": str(summary_path.resolve()),
        "comparison_contract": (
            "same tasks, splits, standardization, V3 selector penalty, "
            "effective-lambda grid, and converged exact-CD reference"
        ),
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(aggregate, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
