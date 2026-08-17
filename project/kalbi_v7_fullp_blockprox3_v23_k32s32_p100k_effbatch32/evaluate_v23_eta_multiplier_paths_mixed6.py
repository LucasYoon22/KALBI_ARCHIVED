#!/usr/bin/env python3
"""Evaluation-only KALBI refinement eta multiplier sweep."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

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

from . import model as model_module
from .eval_kalbi_v2_common import (
    encode_dataset,
    file_sha256,
    tensors_from_arrays,
    vector_correlation,
)
from .exact_direct96_prox import exact_direct96_prox
from .model import (
    KALBI_BETA_BOUND,
    KALBI_EXACT_PG_DATA_MARGIN,
    KALBI_EXACT_PG_FEATURE_CHUNK,
)
from .paper_eval_unseen_dimension5 import (
    load_frozen_task,
    objective_paths,
    prediction_mse,
    split_standardize_three_way,
)
from .report_latest_v23_selectoronly_solution_selector_mixed6 import (
    SEED_STRIDE,
    load_components_v23,
)
from .utils import strip_convexity_flag


FACTORS = (1.0, 2.0, 5.0, 10.0)
COLORS = {
    1.0: "#0072B2",
    2.0: "#009E73",
    5.0: "#E69F00",
    10.0: "#D55E00",
}
GREY = "#666666"
LIGHT_GREY = "#D0D0D0"


def _refine_eta_scaled_once(
    model: torch.nn.Module,
    encoded: dict[str, Any],
    beta: torch.Tensor,
    raw_lambda: torch.Tensor,
    penalty_shape: torch.Tensor,
    eta_multiplier: float,
) -> dict[str, torch.Tensor]:
    beta0 = beta.float()
    x_centered = encoded["Xc"].float()
    y_centered = encoded["yc"].float()
    sample_count = max(1, int(x_centered.shape[1]))
    residual0 = (
        torch.matmul(beta0, x_centered.transpose(1, 2))
        - y_centered[:, None, :]
    )
    data_gradient = torch.matmul(residual0, x_centered) / float(sample_count)
    with torch.no_grad():
        base_step = model_module._select_exact_pg_step(
            model,
            encoded,
            beta0,
            residual0,
            data_gradient,
            raw_lambda,
            penalty_shape,
        )
    base_eta = base_step["step_size"]
    requested_eta = base_eta * float(eta_multiplier)
    eta = torch.minimum(requested_eta, base_step["curvature_step_cap"])
    gamma = eta * raw_lambda.float()
    proposal = beta0 - eta[..., None] * data_gradient
    refined, _ = exact_direct96_prox(
        proposal,
        gamma,
        strip_convexity_flag(penalty_shape).float(),
        model._kalbi_v6_penalty_grid.float(),
        feature_chunk=KALBI_EXACT_PG_FEATURE_CHUNK,
        curvature_margin=model_module.KALBI_EXACT_PG_CURVATURE_MARGIN,
    )
    feature_mask = encoded.get("feature_mask")
    if feature_mask is not None:
        refined = refined * feature_mask[:, None, :].float()
    refined = refined.clamp(-KALBI_BETA_BOUND, KALBI_BETA_BOUND)
    if not bool(torch.isfinite(refined).all().item()):
        raise RuntimeError("eta multiplier refinement produced nonfinite beta")

    delta = refined - beta0
    delta_energy = delta.square().sum(-1)
    x_delta = torch.matmul(delta, x_centered.transpose(1, 2))
    directional_curvature = (
        x_delta.square().sum(-1) / float(sample_count)
    ) / delta_energy.clamp_min(1e-20)
    majorization_ratio = eta * directional_curvature
    majorization_ratio = torch.where(
        delta_energy <= 1e-20,
        torch.zeros_like(majorization_ratio),
        majorization_ratio,
    )
    return {
        "beta": refined,
        "base_eta": base_eta,
        "requested_eta": requested_eta,
        "eta": eta,
        "gamma": gamma,
        "curvature_capped": eta < requested_eta * (1.0 - 1e-7),
        "majorization_ratio": majorization_ratio,
        "change_rms": delta.square().mean(-1).sqrt(),
    }


def _predict_eta_multiplier(
    model: torch.nn.Module,
    encoded: dict[str, Any],
    raw_lambdas: torch.Tensor,
    penalty_shape: torch.Tensor,
    penalty_d0: torch.Tensor,
    *,
    eta_multiplier: float,
    iterations: int,
    query_chunk: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    raw_lambdas = raw_lambdas.reshape(-1).float()
    penalty_shape = penalty_shape.reshape(raw_lambdas.numel(), -1).float()
    penalty_d0 = penalty_d0.reshape(-1).float()
    lambda_max = encoded["lambda_max"].float().reshape(-1)[0]
    betas: list[torch.Tensor] = []
    eta_ratios: list[torch.Tensor] = []
    cap_flags: list[torch.Tensor] = []
    majorization: list[torch.Tensor] = []
    final_changes: list[torch.Tensor] = []

    with torch.inference_mode():
        for start in range(0, raw_lambdas.numel(), int(query_chunk)):
            stop = min(start + int(query_chunk), raw_lambdas.numel())
            chunk_lambda = raw_lambdas[start:stop][None]
            chunk_shape = penalty_shape[start:stop][None]
            chunk_d0 = penalty_d0[start:stop][None]
            decoded = model.decode_queries(
                encoded,
                lambdas=chunk_lambda,
                lambda_max=lambda_max.expand(1, stop - start),
                penalty_shape_features=chunk_shape,
                penalty_d0=chunk_d0,
                return_aux=True,
            )
            beta = decoded["kalbi_v6_beta0_q"].float()
            ratio_history: list[torch.Tensor] = []
            cap_history: list[torch.Tensor] = []
            majorization_history: list[torch.Tensor] = []
            change_history: list[torch.Tensor] = []
            for _ in range(int(iterations)):
                update = _refine_eta_scaled_once(
                    model,
                    encoded,
                    beta,
                    chunk_lambda,
                    chunk_shape,
                    float(eta_multiplier),
                )
                beta = update["beta"]
                ratio_history.append(
                    update["eta"]
                    / update["base_eta"].clamp_min(1e-20)
                )
                cap_history.append(update["curvature_capped"].float())
                majorization_history.append(update["majorization_ratio"])
                change_history.append(update["change_rms"])
            betas.append(beta[0].cpu())
            eta_ratios.append(torch.stack(ratio_history, dim=-1)[0].cpu())
            cap_flags.append(torch.stack(cap_history, dim=-1)[0].cpu())
            majorization.append(
                torch.stack(majorization_history, dim=-1)[0].cpu()
            )
            final_changes.append(torch.stack(change_history, dim=-1)[0].cpu())

    return (
        torch.cat(betas).numpy().astype(np.float64),
        {
            "eta_ratio": torch.cat(eta_ratios).numpy(),
            "curvature_capped": torch.cat(cap_flags).numpy(),
            "majorization_ratio": torch.cat(majorization).numpy(),
            "change_rms": torch.cat(final_changes).numpy(),
        },
    )


def _oracle_index(values: np.ndarray, raw_lambdas: np.ndarray) -> int:
    minimum = float(np.min(values))
    ties = np.flatnonzero(
        np.isclose(values, minimum, rtol=1e-12, atol=1e-14)
    )
    return int(ties[np.argmax(raw_lambdas[ties])])


def _plot_solution_paths(
    tasks: list[dict[str, Any]], output: Path
) -> None:
    figure, axes = plt.subplots(
        len(tasks),
        4,
        figsize=(13.0, 2.1 * len(tasks)),
        squeeze=False,
    )
    for row, task in enumerate(tasks):
        x_axis = np.log10(task["raw_lambdas"])
        for factor in FACTORS:
            beta = task["betas"][factor]
            axes[row, 0].plot(
                x_axis,
                np.sqrt(np.mean(beta**2, axis=1)),
                color=COLORS[factor],
                lw=1.1,
            )
            axes[row, 1].plot(
                x_axis,
                task["objectives"][factor] / task["zero_objective"],
                color=COLORS[factor],
                lw=1.1,
            )
            axes[row, 2].plot(
                x_axis,
                task["test_mse"][factor] / task["zero_test_mse"],
                color=COLORS[factor],
                lw=1.1,
            )
            axes[row, 3].plot(
                x_axis,
                np.count_nonzero(np.abs(beta) > 1e-4, axis=1),
                color=COLORS[factor],
                lw=1.1,
            )
        reference_beta = task["reference_beta"]
        axes[row, 0].plot(
            x_axis,
            np.sqrt(np.mean(reference_beta**2, axis=1)),
            color=GREY,
            lw=1.0,
            ls="--",
        )
        axes[row, 1].plot(
            x_axis,
            task["reference_objective"] / task["zero_objective"],
            color=GREY,
            lw=1.0,
            ls="--",
        )
        axes[row, 2].plot(
            x_axis,
            task["reference_test_mse"] / task["zero_test_mse"],
            color=GREY,
            lw=1.0,
            ls="--",
        )
        axes[row, 3].plot(
            x_axis,
            np.count_nonzero(np.abs(reference_beta) > 1e-4, axis=1),
            color=GREY,
            lw=1.0,
            ls="--",
        )
        axes[row, 2].axhline(1.0, color=LIGHT_GREY, lw=0.8, ls=":")
        axes[row, 1].set_yscale("log")
        axes[row, 2].set_yscale("log")
        for column in range(4):
            axes[row, column].grid(axis="y", color="#E8E8E8", lw=0.5)
        axes[row, 0].set_ylabel(
            f"Dataset {task['dataset_index']}\nN={task['n']}, P={task['p']}"
        )
        if row == 0:
            for column, title in enumerate(
                (
                    "Coefficient RMS path",
                    r"Training objective $F/F_0$",
                    "Held-out MSE / zero MSE",
                    r"Sparsity ($|\beta|>10^{-4}$)",
                )
            ):
                axes[row, column].set_title(title)
        if row == len(tasks) - 1:
            for column in range(4):
                axes[row, column].set_xlabel(
                    r"$\log_{10}(\lambda_{\rm raw})$"
                )
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[factor],
            lw=1.3,
            label=f"KALBI eta x{factor:g}, T=100",
        )
        for factor in FACTORS
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color=GREY,
            lw=1.0,
            ls="--",
            label="Coordinate descent reference",
        )
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.002),
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.99,
        top=0.965,
        bottom=0.075,
        hspace=0.40,
        wspace=0.28,
    )
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_selector(
    tasks: list[dict[str, Any]], output: Path
) -> None:
    figure, axes = plt.subplots(
        len(tasks),
        2,
        figsize=(10.6, 2.25 * len(tasks)),
        squeeze=False,
    )
    for row, task in enumerate(tasks):
        x_axis = np.log10(task["raw_lambdas"])
        selected_x = math.log10(task["selected_raw_lambda"])
        for factor in FACTORS:
            axes[row, 0].plot(
                x_axis,
                task["test_mse"][factor] / task["zero_test_mse"],
                color=COLORS[factor],
                lw=1.2,
            )
            axes[row, 0].scatter(
                [selected_x],
                [
                    task["selected_test_mse"][factor]
                    / task["zero_test_mse"]
                ],
                color=COLORS[factor],
                s=24,
                zorder=4,
            )
        axes[row, 0].plot(
            x_axis,
            task["reference_test_mse"] / task["zero_test_mse"],
            color=GREY,
            lw=1.0,
            ls="--",
        )
        axes[row, 0].axvline(
            selected_x, color="#CC3311", lw=0.9, ls=":"
        )
        axes[row, 0].axhline(1.0, color=LIGHT_GREY, lw=0.8, ls=":")
        axes[row, 0].set_yscale("log")
        axes[row, 0].set_ylabel(
            f"Dataset {task['dataset_index']}\nTest MSE / zero MSE"
        )
        axes[row, 0].set_title(
            f"Lambda selection (N={task['n']}, P={task['p']})"
        )
        axes[row, 0].grid(axis="y", color="#E8E8E8", lw=0.5)
        if row == len(tasks) - 1:
            axes[row, 0].set_xlabel(r"$\log_{10}(\lambda_{\rm raw})$")

        derivative_axis = axes[row, 1].twinx()
        axes[row, 1].plot(
            task["penalty_grid"],
            task["penalty_rho"],
            color="#0072B2",
            lw=1.4,
        )
        derivative_axis.plot(
            task["penalty_grid"],
            task["penalty_drho"],
            color="#E69F00",
            lw=1.0,
            ls="--",
        )
        axes[row, 1].set_ylim(bottom=0.0)
        derivative_axis.set_ylim(bottom=0.0)
        axes[row, 1].set_ylabel(r"$\rho$", color="#0072B2")
        derivative_axis.set_ylabel(r"$\rho'$", color="#E69F00")
        axes[row, 1].set_title("KALBI-selected penalty")
        axes[row, 1].grid(axis="y", color="#E8E8E8", lw=0.5)
        if row == len(tasks) - 1:
            axes[row, 1].set_xlabel(r"$|\beta|$")

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[factor],
            lw=1.3,
            marker="o",
            label=f"eta x{factor:g}, T=100",
        )
        for factor in FACTORS
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color=GREY,
                lw=1.0,
                ls="--",
                label="Coordinate descent reference",
            ),
            Line2D(
                [0],
                [0],
                color="#CC3311",
                lw=0.9,
                ls=":",
                label="KALBI-selected lambda",
            ),
        ]
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.002),
    )
    figure.subplots_adjust(
        left=0.11,
        right=0.90,
        top=0.975,
        bottom=0.085,
        hspace=0.52,
        wspace=0.38,
    )
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--paths-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--split-seed", type=int, default=2_026_072_407)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--query-chunk", type=int, default=8)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    prepared_root = args.prepared_root.resolve()
    paths_root = args.paths_root.resolve()
    frozen = json.loads(
        (
            prepared_root
            / "reproducibility"
            / "reproducibility_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = list(frozen["accepted_tasks"])
    if len(entries) != 6:
        raise RuntimeError("expected exactly six frozen tasks")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint, model, selector = load_components_v23(
        args.checkpoint.resolve(), args.config.resolve(), device
    )
    task_plots: list[dict[str, Any]] = []
    task_metrics: list[dict[str, Any]] = []

    for entry in entries:
        dataset_index = int(entry["dataset_index"])
        print(
            json.dumps(
                {
                    "status": "starting",
                    "dataset": dataset_index,
                    "n": int(entry["n"]),
                    "p": int(entry["p"]),
                }
            ),
            flush=True,
        )
        x_raw, y_raw, _ = load_frozen_task(prepared_root, entry)
        split = split_standardize_three_way(
            x_raw,
            y_raw,
            int(args.split_seed) + SEED_STRIDE * (dataset_index - 1),
        )
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

        path_file = paths_root / f"dataset_{dataset_index:02d}" / "paths.npz"
        with np.load(path_file, allow_pickle=False) as payload:
            saved = {key: np.asarray(payload[key]) for key in payload.files}
        raw_lambdas = np.asarray(saved["raw_lambdas"], dtype=np.float64)
        selected_raw_lambda = float(selected["raw_lambda"].reshape(-1)[0])
        if not math.isclose(
            selected_raw_lambda,
            float(saved["selected_raw_lambda"]),
            rel_tol=2e-6,
            abs_tol=1e-8,
        ):
            raise RuntimeError("selector/path raw lambda mismatch")

        evaluation_lambdas = np.concatenate(
            [raw_lambdas, np.asarray([selected_raw_lambda])]
        )
        count = evaluation_lambdas.size
        shape = selected["penalty_shape_features"].reshape(1, -1).expand(
            count, -1
        )
        d0 = selected["penalty_d0"].reshape(-1).expand(count)
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

        betas: dict[float, np.ndarray] = {}
        objectives: dict[float, np.ndarray] = {}
        test_mse: dict[float, np.ndarray] = {}
        selected_test_mse: dict[float, float] = {}
        factor_metrics: dict[str, Any] = {}
        for factor in FACTORS:
            beta_all, diagnostics = _predict_eta_multiplier(
                model,
                encoded,
                torch.as_tensor(
                    evaluation_lambdas,
                    device=device,
                    dtype=torch.float32,
                ),
                shape,
                d0,
                eta_multiplier=factor,
                iterations=int(args.iterations),
                query_chunk=int(args.query_chunk),
            )
            beta_path = beta_all[:-1]
            selected_beta = beta_all[-1:]
            if factor == 1.0:
                reproduction_max_abs = float(
                    np.max(
                        np.abs(
                            beta_path
                            - np.asarray(
                                saved["gamma_eta_beta"],
                                dtype=np.float64,
                            )
                        ),
                        initial=0.0,
                    )
                )
                if reproduction_max_abs > 2e-5:
                    raise RuntimeError(
                        "eta 1x path reproduction failed: "
                        f"max_abs={reproduction_max_abs:.6g}"
                    )
            _, _, objective = objective_paths(
                x_train,
                y_train,
                beta_path,
                raw_lambdas,
                knots,
                relu_b,
                relu_weights,
            )
            mse = prediction_mse(x_test, y_test, beta_path)
            selected_mse = float(
                prediction_mse(x_test, y_test, selected_beta)[0]
            )
            betas[factor] = beta_path
            objectives[factor] = objective
            test_mse[factor] = mse
            selected_test_mse[factor] = selected_mse
            oracle_index = _oracle_index(mse, raw_lambdas)
            factor_metrics[f"{factor:g}x"] = {
                "eta_ratio_median": float(
                    np.median(diagnostics["eta_ratio"])
                ),
                "eta_ratio_min": float(np.min(diagnostics["eta_ratio"])),
                "eta_ratio_max": float(np.max(diagnostics["eta_ratio"])),
                "curvature_cap_fraction": float(
                    np.mean(diagnostics["curvature_capped"] > 0.5)
                ),
                "data_majorization_violation_fraction": float(
                    np.mean(
                        diagnostics["majorization_ratio"]
                        > KALBI_EXACT_PG_DATA_MARGIN + 1e-6
                    )
                ),
                "max_majorization_ratio": float(
                    np.max(diagnostics["majorization_ratio"])
                ),
                "last_change_rms_median": float(
                    np.median(diagnostics["change_rms"][:, -1])
                ),
                "path_correlation_with_cd": float(
                    vector_correlation(
                        beta_path,
                        np.asarray(
                            saved["coordinate_descent_beta"],
                            dtype=np.float64,
                        ),
                    )
                ),
                "objective_gap_mean": float(
                    np.mean(
                        objective
                        - np.asarray(
                            saved["coordinate_descent_objective"],
                            dtype=np.float64,
                        )
                    )
                ),
                "selector_test_mse": selected_mse,
                "oracle_test_mse": float(mse[oracle_index]),
                "oracle_raw_lambda": float(raw_lambdas[oracle_index]),
                "selector_over_oracle_mse": selected_mse
                / max(float(mse[oracle_index]), 1e-12),
            }

        zero_objective = 0.5 * float(
            np.mean(np.asarray(y_train, dtype=np.float64) ** 2)
        )
        zero_test_mse = float(
            np.mean(np.asarray(y_test, dtype=np.float64) ** 2)
        )
        task = {
            "dataset_index": dataset_index,
            "n": int(entry["n"]),
            "p": int(entry["p"]),
            "raw_lambdas": raw_lambdas,
            "selected_raw_lambda": selected_raw_lambda,
            "zero_objective": zero_objective,
            "zero_test_mse": zero_test_mse,
            "betas": betas,
            "objectives": objectives,
            "test_mse": test_mse,
            "selected_test_mse": selected_test_mse,
            "reference_beta": np.asarray(
                saved["coordinate_descent_beta"], dtype=np.float64
            ),
            "reference_objective": np.asarray(
                saved["coordinate_descent_objective"], dtype=np.float64
            ),
            "reference_test_mse": np.asarray(
                saved["coordinate_descent_test_mse"], dtype=np.float64
            ),
            "penalty_grid": np.asarray(saved["penalty_grid"]),
            "penalty_rho": np.asarray(saved["penalty_rho"]),
            "penalty_drho": np.asarray(saved["penalty_drho"]),
        }
        task_plots.append(task)
        metric = {
            "dataset_index": dataset_index,
            "n": int(entry["n"]),
            "p": int(entry["p"]),
            "task_sha256": str(entry["task_sha256"]),
            "selected_raw_lambda": selected_raw_lambda,
            "factors": factor_metrics,
        }
        task_metrics.append(metric)
        task_dir = output_dir / f"dataset_{dataset_index:02d}"
        task_dir.mkdir()
        np.savez_compressed(
            task_dir / "eta_multiplier_paths.npz",
            raw_lambdas=raw_lambdas,
            selected_raw_lambda=np.asarray(selected_raw_lambda),
            eta1_beta=betas[1.0].astype(np.float32),
            eta2_beta=betas[2.0].astype(np.float32),
            eta5_beta=betas[5.0].astype(np.float32),
            eta10_beta=betas[10.0].astype(np.float32),
            eta1_test_mse=test_mse[1.0],
            eta2_test_mse=test_mse[2.0],
            eta5_test_mse=test_mse[5.0],
            eta10_test_mse=test_mse[10.0],
        )
        (task_dir / "metrics.json").write_text(
            json.dumps(metric, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(metric, sort_keys=True), flush=True)

    aggregate: dict[str, Any] = {}
    for factor in FACTORS:
        key = f"{factor:g}x"
        aggregate[key] = {
            field: float(
                np.mean(
                    [task["factors"][key][field] for task in task_metrics]
                )
            )
            for field in (
                "eta_ratio_median",
                "curvature_cap_fraction",
                "data_majorization_violation_fraction",
                "path_correlation_with_cd",
                "objective_gap_mean",
                "selector_test_mse",
                "oracle_test_mse",
                "selector_over_oracle_mse",
            )
        }
    report = {
        "evaluation": "KALBI T=100 refinement eta multiplier sweep",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint.resolve()),
        "checkpoint_step": int(checkpoint["completed_steps"]),
        "contract": {
            "beta_solver_retrained": False,
            "selector_retrained": False,
            "factors": list(FACTORS),
            "iterations": int(args.iterations),
            "raw_lambda_and_penalty_fixed_across_factors": True,
            "gamma": "scaled_eta * raw_lambda",
            "curvature_cap_applied": True,
            "data_majorization_backtracking_after_scaling": False,
        },
        "aggregate": aggregate,
        "tasks": task_metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_solution_paths(
        task_plots, output_dir / "eta_multiplier_solution_paths_mixed6"
    )
    _plot_selector(
        task_plots, output_dir / "eta_multiplier_lambda_selector_mixed6"
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
