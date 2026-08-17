#!/usr/bin/env python3
"""Isolated SURE/divergence summary-stat identification study.

This script does not modify model checkpoints or training state. It studies the
conditional divergence of a ten-step Direct96 proximal-gradient estimator with
fixed penalty and lambda, and compares inexpensive trajectory summaries with
exact/JVP divergence targets. The current V23 task code is used as the baseline
context representation when a checkpoint is supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kalbi_v6_exactprox10_eta4x_v23_independentraw_k32s32_p100k_effbatch32.canonical_penalty import (
    canonical_grid,
    native196_from_derivative_knots,
)
from kalbi_v6_exactprox10_eta4x_v23_independentraw_k32s32_p100k_effbatch32.selector_v23_independent_rawlambda import (
    DirectPenaltyMLPQuerySelectorV23IndependentRawLambda,
)


DTYPE = torch.float64
QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
LAMBDA_RATIOS = np.logspace(-4, 0, 7)
PENALTY_FAMILIES = ("l1", "ridge", "log", "scad_like", "bridge")
PRIMARY_SHAPES = (
    (24, 16),
    (24, 48),
    (32, 24),
    (32, 64),
    (48, 32),
    (48, 96),
    (64, 48),
    (64, 128),
    (96, 64),
    (96, 192),
    (128, 96),
    (128, 256),
)


@dataclass(frozen=True)
class Task:
    task_id: int
    seed: int
    n: int
    p: int
    correlation: float
    sparsity: int
    snr: float
    error_family: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    noise_variance: float
    max_abs_corr: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return json_ready(value.detach().cpu().numpy())
    if isinstance(value, Path):
        return str(value)
    return value


def make_design(rng: np.random.Generator, n: int, p: int, corr: float) -> np.ndarray:
    innovation = rng.normal(size=(n, p))
    if corr > 0.0:
        scale = math.sqrt(max(1e-10, 1.0 - corr * corr))
        for j in range(1, p):
            innovation[:, j] = corr * innovation[:, j - 1] + scale * innovation[:, j]
    x = innovation
    x -= x.mean(axis=0, keepdims=True)
    x /= np.maximum(x.std(axis=0, keepdims=True), 1e-8)
    return x.astype(np.float64)


def make_noise(rng: np.random.Generator, n: int, family: str) -> np.ndarray:
    if family == "gaussian":
        value = rng.normal(size=n)
    elif family == "student_t5":
        value = rng.standard_t(df=5, size=n) / math.sqrt(5.0 / 3.0)
    elif family == "laplace":
        value = rng.laplace(size=n) / math.sqrt(2.0)
    elif family == "contaminated":
        contamination = rng.uniform(size=n) < 0.08
        value = rng.normal(size=n)
        value[contamination] = rng.normal(scale=6.0, size=int(contamination.sum()))
        value /= math.sqrt(0.92 + 0.08 * 36.0)
    else:
        raise ValueError(f"unknown error family {family}")
    value -= value.mean()
    return value / max(value.std(), 1e-8)


def max_abs_corr(x: np.ndarray) -> float:
    if x.shape[1] < 2:
        return 0.0
    correlation = np.corrcoef(x, rowvar=False)
    np.fill_diagonal(correlation, 0.0)
    return float(np.nanmax(np.abs(correlation)))


def make_task(task_id: int, seed: int, *, non_gaussian: bool) -> Task:
    rng = np.random.default_rng(seed)
    n, p = PRIMARY_SHAPES[task_id % len(PRIMARY_SHAPES)]
    corr = (0.0, 0.3, 0.6, 0.85)[(task_id // len(PRIMARY_SHAPES)) % 4]
    x_train = make_design(rng, n, p, corr)
    x_test = make_design(rng, max(256, 3 * n), p, corr)
    max_k = max(1, min(p, max(1, n // 3 - 1)))
    sparsity_levels = sorted(set((1, min(3, max_k), max(1, int(math.sqrt(max_k))), max_k)))
    sparsity = sparsity_levels[(task_id // 4) % len(sparsity_levels)]
    support = rng.choice(p, size=sparsity, replace=False)
    beta = np.zeros(p)
    coefficient_family = task_id % 3
    if coefficient_family == 0:
        beta[support] = rng.normal(size=sparsity)
    elif coefficient_family == 1:
        beta[support] = rng.choice((-1.0, 1.0), size=sparsity)
    else:
        beta[support] = rng.laplace(size=sparsity)
    signal = x_train @ beta
    target_snr = float(np.exp(rng.uniform(math.log(0.05), math.log(200.0))))
    signal_scale = math.sqrt(target_snr / (1.0 + target_snr))
    signal_std = max(signal.std(), 1e-8)
    beta *= signal_scale / signal_std
    signal = x_train @ beta
    noise_variance = 1.0 / (1.0 + target_snr)
    family = (
        ("student_t5", "laplace", "contaminated")[task_id % 3]
        if non_gaussian
        else "gaussian"
    )
    y_train = signal + math.sqrt(noise_variance) * make_noise(rng, n, family)
    y_test = x_test @ beta + math.sqrt(noise_variance) * make_noise(
        rng, x_test.shape[0], family
    )
    y_train -= y_train.mean()
    y_test -= y_train.mean()
    realized_signal = float(np.var(signal))
    realized_snr = realized_signal / max(noise_variance, 1e-12)
    return Task(
        task_id=task_id,
        seed=seed,
        n=n,
        p=p,
        correlation=corr,
        sparsity=sparsity,
        snr=realized_snr,
        error_family=family,
        x_train=x_train,
        y_train=y_train.astype(np.float64),
        x_test=x_test,
        y_test=y_test.astype(np.float64),
        noise_variance=noise_variance,
        max_abs_corr=max_abs_corr(x_train),
    )


def penalty_derivatives(device: torch.device) -> dict[str, torch.Tensor]:
    grid = canonical_grid(device=device, dtype=DTYPE)
    eps = torch.tensor(0.05, dtype=DTYPE, device=device)
    raw = {
        "l1": torch.ones_like(grid),
        "ridge": grid + eps,
        "log": 1.0 / (grid + 0.30),
        "scad_like": (1.0 - grid / 3.0).clamp_min(0.03),
        "bridge": 1.0 / torch.sqrt(grid + eps),
    }
    result: dict[str, torch.Tensor] = {}
    for name, derivative in raw.items():
        mass = torch.trapezoid(derivative, grid)
        result[name] = derivative / mass
    return result


def penalty_descriptor(derivative: torch.Tensor, grid: torch.Tensor) -> np.ndarray:
    slope = torch.diff(derivative) / torch.diff(grid)
    values = torch.cat(
        (
            derivative[[0, -1]],
            derivative.min().reshape(1),
            derivative.max().reshape(1),
            torch.diff(derivative).abs().sum().reshape(1),
            slope.min().reshape(1),
            slope.max().reshape(1),
            torch.quantile(
                slope,
                torch.tensor(
                    (0.1, 0.5, 0.9),
                    device=grid.device,
                    dtype=slope.dtype,
                ),
            ),
        )
    )
    return values.detach().cpu().numpy().astype(np.float64)


def local_prox(
    values: torch.Tensor,
    gamma: torch.Tensor,
    derivative: torch.Tensor,
    grid: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    slopes = torch.diff(derivative) / torch.diff(grid)
    magnitude = values.abs()
    h = grid + gamma * derivative
    upper = torch.searchsorted(h.contiguous(), magnitude.contiguous(), right=True)
    zero = upper == 0
    saturated = upper >= grid.numel()
    lower = (upper - 1).clamp(0, grid.numel() - 2)
    upper_interval = lower + 1
    lower_grid = grid[lower]
    upper_grid = grid[upper_interval]
    lower_drho = derivative[lower]
    slope = slopes[lower]
    denominator = 1.0 + gamma * slope
    stationary = (
        magnitude - gamma * (lower_drho - slope * lower_grid)
    ) / denominator
    stationary = torch.maximum(lower_grid, torch.minimum(stationary, upper_grid))
    positive = torch.where(zero, torch.zeros_like(stationary), stationary)
    positive = torch.where(saturated, torch.full_like(positive, grid[-1]), positive)
    prox = torch.sign(values) * positive
    prox_derivative = torch.where(
        zero | saturated,
        torch.zeros_like(denominator),
        denominator.reciprocal(),
    )
    nearest_h_distance = (magnitude[:, None] - h[None, :]).abs().amin(dim=1)
    return prox, {
        "prox_derivative": prox_derivative,
        "denominator": denominator,
        "slope": slope,
        "zero": zero,
        "saturated": saturated,
        "threshold_margin": magnitude - gamma * derivative[0],
        "branch_margin": nearest_h_distance,
    }


def interpolate_derivative(
    magnitude: torch.Tensor, derivative: torch.Tensor, grid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    slopes = torch.diff(derivative) / torch.diff(grid)
    value = magnitude.clamp(0.0, float(grid[-1]))
    upper = torch.searchsorted(grid, value.contiguous(), right=True).clamp(
        1, grid.numel() - 1
    )
    lower = upper - 1
    slope = slopes[lower]
    drho = derivative[lower] + slope * (value - grid[lower])
    return drho, slope


def array_stats(prefix: str, values: torch.Tensor) -> dict[str, float]:
    value = values.detach().float().reshape(-1)
    quantile = torch.quantile(
        value, torch.tensor(QUANTILES, device=value.device, dtype=value.dtype)
    )
    result = {
        f"{prefix}_mean": float(value.mean()),
        f"{prefix}_std": float(value.std(unbiased=False)),
    }
    result.update(
        {f"{prefix}_q{int(100*q):02d}": float(v) for q, v in zip(QUANTILES, quantile)}
    )
    return result


def solve_pg(
    x: torch.Tensor,
    y: torch.Tensor,
    derivative: torch.Tensor,
    raw_lambda: float,
    *,
    steps: int = 10,
    collect: bool = True,
    compact: bool = False,
    gram: torch.Tensor | None = None,
    eta: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    n, p = x.shape
    grid = canonical_grid(device=x.device, dtype=x.dtype)
    if eta is None:
        if gram is not None:
            lipschitz = torch.linalg.eigvalsh(gram).amax().clamp_min(1e-8).detach()
        else:
            vector = torch.ones(p, dtype=x.dtype, device=x.device)
            vector = vector / vector.norm().clamp_min(1e-12)
            for _ in range(12):
                vector = x.T @ (x @ vector) / float(n)
                vector = vector / vector.norm().clamp_min(1e-12)
            lipschitz = torch.dot(vector, x.T @ (x @ vector) / float(n)).clamp_min(
                1e-8
            ).detach()
        eta = 0.9 / lipschitz
    gamma = eta * float(raw_lambda)
    beta = torch.zeros(p, dtype=x.dtype, device=x.device)
    previous_active = torch.zeros(p, dtype=torch.bool, device=x.device)
    summary: dict[str, float] = {}
    for iteration in range(steps):
        residual = x @ beta - y
        gradient = x.T @ residual / float(n)
        proposal = beta - eta * gradient
        next_beta, prox = local_prox(proposal, gamma, derivative, grid)
        active = next_beta != 0.0
        drho, rho2 = interpolate_derivative(next_beta.abs(), derivative, grid)
        signed_penalty = float(raw_lambda) * drho * torch.sign(next_beta)
        active_kkt = (gradient + signed_penalty).abs()
        zero_kkt = (gradient.abs() - float(raw_lambda) * derivative[0]).clamp_min(0.0)
        kkt = torch.where(active, active_kkt, zero_kkt)
        if collect:
            tag = f"t{iteration:02d}"
            if compact:
                delta = next_beta - beta
                summary.update(
                    {
                        f"{tag}_pprime_mean": float(
                            prox["prox_derivative"].mean()
                        ),
                        f"{tag}_pprime_max": float(
                            prox["prox_derivative"].amax()
                        ),
                        f"{tag}_denom_min": float(prox["denominator"].amin()),
                        f"{tag}_denom_mean": float(prox["denominator"].mean()),
                        f"{tag}_threshold_abs_min": float(
                            prox["threshold_margin"].abs().amin()
                        ),
                        f"{tag}_active_fraction": float(active.float().mean()),
                        f"{tag}_zero_fraction": float(prox["zero"].float().mean()),
                        f"{tag}_saturated_fraction": float(
                            prox["saturated"].float().mean()
                        ),
                        f"{tag}_beta_rms": float(next_beta.square().mean().sqrt()),
                        f"{tag}_delta_rms": float(delta.square().mean().sqrt()),
                        f"{tag}_gradient_rms": float(
                            gradient.square().mean().sqrt()
                        ),
                        f"{tag}_kkt_rms": float(kkt.square().mean().sqrt()),
                        f"{tag}_eta": float(eta),
                        f"{tag}_gamma": float(gamma),
                    }
                )
            else:
                for name, value in (
                    ("pprime", prox["prox_derivative"]),
                    ("denom", prox["denominator"]),
                    ("threshold_margin", prox["threshold_margin"]),
                    ("branch_margin", prox["branch_margin"]),
                    ("beta_abs", next_beta.abs()),
                    ("delta_abs", (next_beta - beta).abs()),
                    ("gradient_abs", gradient.abs()),
                    ("kkt", kkt),
                    ("rho2", rho2),
                ):
                    summary.update(array_stats(f"{tag}_{name}", value))
            union = active | previous_active
            if not compact:
                summary[f"{tag}_active_fraction"] = float(active.float().mean())
                summary[f"{tag}_entry_fraction"] = float(
                    (active & ~previous_active).float().mean()
                )
                summary[f"{tag}_exit_fraction"] = float(
                    (~active & previous_active).float().mean()
                )
                summary[f"{tag}_jaccard_previous"] = float(
                    (active & previous_active).sum().float()
                    / union.sum().clamp_min(1).float()
                )
                summary[f"{tag}_eta"] = float(eta)
                summary[f"{tag}_gamma"] = float(gamma)
                summary[f"{tag}_zero_fraction"] = float(
                    prox["zero"].float().mean()
                )
                summary[f"{tag}_saturated_fraction"] = float(
                    prox["saturated"].float().mean()
                )
        previous_active = active
        beta = next_beta
    if collect and not compact:
        support = torch.nonzero(beta != 0.0).reshape(-1)
        if support.numel() > 64:
            support = torch.topk(beta.abs(), k=64).indices
        if support.numel() > 0:
            active_gram = (
                gram[support][:, support]
                if gram is not None
                else x[:, support].T @ x[:, support] / float(n)
            )
            active_drho, active_rho2 = interpolate_derivative(
                beta[support].abs(), derivative, grid
            )
            del active_drho
            hessian = active_gram + float(raw_lambda) * torch.diag(active_rho2)
            eig = torch.linalg.eigvalsh(0.5 * (hessian + hessian.T))
            stable = eig.abs().clamp_min(1e-8)
            summary.update(array_stats("active_hessian_eig", eig))
            summary["active_hessian_logabsdet"] = float(stable.log().sum())
            summary["active_hessian_condition_abs"] = float(stable.max() / stable.min())
            summary["active_support_size"] = float(support.numel())
            try:
                inverse = torch.linalg.pinv(hessian)
                leverage = torch.trace(active_gram @ inverse)
                summary["restricted_df_proxy"] = float(leverage)
            except RuntimeError:
                summary["restricted_df_proxy"] = float("nan")
        else:
            for key in (
                "active_hessian_logabsdet",
                "active_hessian_condition_abs",
                "active_support_size",
                "restricted_df_proxy",
            ):
                summary[key] = 0.0
            summary.update(array_stats("active_hessian_eig", torch.zeros(1, dtype=x.dtype)))
    return beta, summary


def task_summaries(task: Task) -> dict[str, float]:
    x = task.x_train
    y = task.y_train
    rhs = x.T @ y / float(task.n)
    singular = np.linalg.svd(x, compute_uv=False)
    gram_eig = singular * singular / float(task.n)
    result = {
        "log_n": math.log(task.n),
        "log_p": math.log(task.p),
        "log_p_over_n": math.log(task.p / task.n),
        "lambda_max": float(np.max(np.abs(rhs))),
        "y_skew": float(np.mean(((y - y.mean()) / max(y.std(), 1e-8)) ** 3)),
        "y_kurtosis": float(np.mean(((y - y.mean()) / max(y.std(), 1e-8)) ** 4)),
        "y_iqr": float(np.quantile(y, 0.75) - np.quantile(y, 0.25)),
        "marginal_l1": float(np.mean(np.abs(rhs))),
        "marginal_l2": float(np.sqrt(np.mean(rhs * rhs))),
        "marginal_max": float(np.max(np.abs(rhs))),
        "gram_trace": float(np.sum(gram_eig)),
        "gram_frobenius": float(np.sqrt(np.sum(gram_eig * gram_eig))),
        "gram_effective_rank": float(
            np.sum(gram_eig) ** 2 / max(np.sum(gram_eig * gram_eig), 1e-12)
        ),
        "max_abs_corr": task.max_abs_corr,
    }
    for q in QUANTILES:
        result[f"abs_rhs_q{int(100*q):02d}"] = float(np.quantile(np.abs(rhs), q))
        result[f"gram_eig_q{int(100*q):02d}"] = float(np.quantile(gram_eig, q))
        result[f"y_q{int(100*q):02d}"] = float(np.quantile(y, q))
    return result


def load_v23_selector(checkpoint_path: Path) -> DirectPenaltyMLPQuerySelectorV23IndependentRawLambda:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    selector = DirectPenaltyMLPQuerySelectorV23IndependentRawLambda()
    selector.load_state_dict(checkpoint["selector_state_dict"], strict=True)
    selector.eval()
    return selector


@torch.no_grad()
def v23_task_code(
    selector: DirectPenaltyMLPQuerySelectorV23IndependentRawLambda,
    task: Task,
) -> np.ndarray:
    x = torch.from_numpy(task.x_train).float()[None]
    y = torch.from_numpy(task.y_train).float()[None]
    lambda_max = (x.transpose(1, 2) @ y[..., None]).abs().amax(dim=1).reshape(-1)
    lambda_max = lambda_max / float(task.n)
    result = selector(x, y, lambda_max, return_representation=True)
    return result["v22_task_code"][0].float().cpu().numpy().astype(np.float64)


def estimator_factory(
    x: torch.Tensor,
    derivative: torch.Tensor,
    raw_lambda: float,
    steps: int,
) -> Callable[[torch.Tensor], torch.Tensor]:
    gram = x.T @ x / float(x.shape[0])
    eta = 0.9 / torch.linalg.eigvalsh(gram).amax().clamp_min(1e-8).detach()

    def estimator(y: torch.Tensor) -> torch.Tensor:
        beta, _ = solve_pg(
            x,
            y,
            derivative,
            raw_lambda,
            steps=steps,
            collect=False,
            gram=gram,
            eta=eta,
        )
        return x @ beta

    return estimator


def hutchinson(
    estimator: Callable[[torch.Tensor], torch.Tensor],
    y: torch.Tensor,
    probes: int,
    seed: int,
) -> list[float]:
    generator = torch.Generator(device=y.device).manual_seed(seed)
    values: list[float] = []
    for _ in range(probes):
        z = torch.empty_like(y).bernoulli_(0.5, generator=generator).mul_(2.0).sub_(1.0)
        _, jvp = torch.autograd.functional.jvp(
            estimator, y, z, create_graph=False, strict=False
        )
        values.append(float(torch.dot(z, jvp)))
    return values


def exact_trace(
    estimator: Callable[[torch.Tensor], torch.Tensor], y: torch.Tensor
) -> float:
    jacobian = torch.autograd.functional.jacobian(
        estimator, y, create_graph=False, strict=False, vectorize=True
    )
    return float(torch.trace(jacobian))


def finite_difference_trace(
    estimator: Callable[[torch.Tensor], torch.Tensor],
    y: torch.Tensor,
    epsilon: float = 1e-4,
) -> float:
    trace = 0.0
    with torch.no_grad():
        for index in range(y.numel()):
            direction = torch.zeros_like(y)
            direction[index] = epsilon
            plus = estimator(y + direction)
            minus = estimator(y - direction)
            trace += float((plus[index] - minus[index]) / (2.0 * epsilon))
    return trace


def split_indices(task_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(task_ids)
    train_tasks = unique[np.mod(unique, 5) != 0]
    test_tasks = unique[np.mod(unique, 5) == 0]
    return np.isin(task_ids, train_tasks), np.isin(task_ids, test_tasks)


def finite_matrix(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    result[~np.isfinite(result)] = 0.0
    return result


def fit_divergence_models(
    baseline: np.ndarray,
    cheap: np.ndarray,
    dynamic: np.ndarray,
    target: np.ndarray,
    task_ids: np.ndarray,
) -> dict[str, Any]:
    train, test = split_indices(task_ids)
    outputs: dict[str, Any] = {}
    variants = {
        "v23": baseline,
        "v23_plus_task": np.concatenate((baseline, cheap), axis=1),
        "v23_plus_task_dynamic": np.concatenate((baseline, cheap, dynamic), axis=1),
    }
    for name, features in variants.items():
        components = min(32, features.shape[1], max(2, int(train.sum()) - 2))
        model = make_pipeline(
            StandardScaler(),
            PCA(n_components=components, random_state=0),
            RidgeCV(alphas=np.logspace(-4, 4, 17)),
        )
        model.fit(features[train], target[train])
        prediction = model.predict(features[test])
        outputs[name] = {
            "r2": float(r2_score(target[test], prediction)),
            "rmse": float(math.sqrt(mean_squared_error(target[test], prediction))),
            "test_count": int(test.sum()),
        }
    return outputs


def fit_risk_models(
    baseline: np.ndarray,
    cheap: np.ndarray,
    candidate: np.ndarray,
    dynamic: np.ndarray,
    target: np.ndarray,
    task_ids: np.ndarray,
) -> dict[str, Any]:
    train, test = split_indices(task_ids)
    baseline_pca = PCA(n_components=min(32, baseline.shape[1]), random_state=0)
    baseline_scaled = StandardScaler().fit_transform(baseline[train])
    baseline_pca.fit(baseline_scaled)
    baseline_all = baseline_pca.transform(
        StandardScaler().fit(baseline[train]).transform(baseline)
    )
    variants = {
        "v23_candidate": np.concatenate((baseline_all, candidate), axis=1),
        "v23_task_candidate": np.concatenate((baseline_all, cheap, candidate), axis=1),
        "v23_task_candidate_dynamic": np.concatenate(
            (baseline_all, cheap, candidate, dynamic), axis=1
        ),
    }
    outputs: dict[str, Any] = {}
    for name, features in variants.items():
        model = ExtraTreesRegressor(
            n_estimators=160,
            min_samples_leaf=3,
            max_features=0.7,
            n_jobs=8,
            random_state=19,
        )
        model.fit(features[train], target[train])
        prediction = model.predict(features[test])
        task_regrets = []
        ratios = []
        for task_id in np.unique(task_ids[test]):
            positions = np.where(test & (task_ids == task_id))[0]
            selected = positions[int(np.argmin(prediction[np.isin(np.where(test)[0], positions)]))]
            oracle = positions[int(np.argmin(target[positions]))]
            task_regrets.append(float(target[selected] - target[oracle]))
            ratios.append(float(target[selected] / max(target[oracle], 1e-12)))
        outputs[name] = {
            "r2": float(r2_score(target[test], prediction)),
            "rmse": float(math.sqrt(mean_squared_error(target[test], prediction))),
            "mean_oracle_regret": float(np.mean(task_regrets)),
            "median_selected_oracle_ratio": float(np.median(ratios)),
            "p90_selected_oracle_ratio": float(np.quantile(ratios, 0.9)),
        }
    return outputs


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_study(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    selector = load_v23_selector(checkpoint)
    device = torch.device("cpu")
    grid = canonical_grid(device=device, dtype=DTYPE)
    derivatives = penalty_derivatives(device)
    tasks = [
        make_task(
            task_id,
            args.seed + task_id * 1009,
            non_gaussian=task_id >= int(round(args.tasks * args.gaussian_fraction)),
        )
        for task_id in range(args.tasks)
    ]
    manifest_rows = []
    for task in tasks:
        manifest_rows.append(
            {
                "task_id": task.task_id,
                "seed": task.seed,
                "n": task.n,
                "p": task.p,
                "p_over_n": task.p / task.n,
                "correlation": task.correlation,
                "max_abs_corr": task.max_abs_corr,
                "sparsity": task.sparsity,
                "snr": task.snr,
                "error_family": task.error_family,
                "noise_variance": task.noise_variance,
            }
        )
    write_csv(out / "frozen_task_manifest.csv", manifest_rows)
    (out / "frozen_task_manifest.json").write_text(
        json.dumps(manifest_rows, indent=2, sort_keys=True)
    )

    task_codes: dict[int, np.ndarray] = {}
    cheap_features: dict[int, dict[str, float]] = {}
    task_start = time.perf_counter()
    for task in tasks:
        task_codes[task.task_id] = v23_task_code(selector, task)
        cheap_features[task.task_id] = task_summaries(task)
    task_encoding_seconds = time.perf_counter() - task_start

    divergence_rows: list[dict[str, Any]] = []
    divergence_dynamic: list[dict[str, float]] = []
    exact_count = 0
    jvp_start = time.perf_counter()
    for task in tasks:
        family_index = task.task_id % len(PENALTY_FAMILIES)
        ratio_index = (task.task_id // len(PENALTY_FAMILIES)) % len(LAMBDA_RATIOS)
        family = PENALTY_FAMILIES[family_index]
        derivative = derivatives[family]
        lambda_max = float(np.max(np.abs(task.x_train.T @ task.y_train / task.n)))
        raw_lambda = float(
            np.clip(LAMBDA_RATIOS[ratio_index] * lambda_max / float(derivative[0]), 1e-6, 10.0)
        )
        x = torch.from_numpy(task.x_train).to(dtype=DTYPE)
        y = torch.from_numpy(task.y_train).to(dtype=DTYPE)
        gram = x.T @ x / float(task.n)
        eta = 0.9 / torch.linalg.eigvalsh(gram).amax().clamp_min(1e-8).detach()
        beta, dynamic = solve_pg(
            x,
            y,
            derivative,
            raw_lambda,
            collect=True,
            gram=gram,
            eta=eta,
        )
        estimator = estimator_factory(x, derivative, raw_lambda, steps=args.steps)
        probe_values = hutchinson(estimator, y, probes=8, seed=task.seed + 700_001)
        row: dict[str, Any] = {
            "task_id": task.task_id,
            "penalty_family": family,
            "raw_lambda": raw_lambda,
            "zero_kkt_ratio": raw_lambda * float(derivative[0]) / max(lambda_max, 1e-12),
            "hutch_r1": probe_values[0],
            "hutch_r2": float(np.mean(probe_values[:2])),
            "hutch_r4": float(np.mean(probe_values[:4])),
            "hutch_r8": float(np.mean(probe_values)),
            "hutch_probe_std": float(np.std(probe_values, ddof=0)),
            "conditional_divergence_target": float(np.mean(probe_values)),
            "exact_trace": float("nan"),
            "finite_difference_trace": float("nan"),
        }
        if task.n <= args.exact_max_n and exact_count < args.exact_tasks:
            row["exact_trace"] = exact_trace(estimator, y)
            row["finite_difference_trace"] = finite_difference_trace(estimator, y)
            row["conditional_divergence_target"] = row["exact_trace"]
            exact_count += 1
        row["train_mse"] = float(torch.mean((x @ beta - y) ** 2))
        divergence_rows.append(row)
        divergence_dynamic.append(dynamic)
    jvp_seconds = time.perf_counter() - jvp_start
    write_csv(out / "divergence_targets.csv", divergence_rows)

    risk_rows: list[dict[str, Any]] = []
    risk_dynamic: list[dict[str, float]] = []
    risk_start = time.perf_counter()
    for task in tasks:
        x = torch.from_numpy(task.x_train).to(dtype=DTYPE)
        y = torch.from_numpy(task.y_train).to(dtype=DTYPE)
        x_test = torch.from_numpy(task.x_test).to(dtype=DTYPE)
        y_test = torch.from_numpy(task.y_test).to(dtype=DTYPE)
        gram = x.T @ x / float(task.n)
        eta = 0.9 / torch.linalg.eigvalsh(gram).amax().clamp_min(1e-8).detach()
        lambda_max = float(np.max(np.abs(task.x_train.T @ task.y_train / task.n)))
        for family in PENALTY_FAMILIES:
            derivative = derivatives[family]
            descriptor = penalty_descriptor(derivative, grid)
            for ratio_index, ratio in enumerate(LAMBDA_RATIOS):
                raw_unclipped = float(ratio * lambda_max / float(derivative[0]))
                raw_lambda = float(np.clip(raw_unclipped, 1e-6, 10.0))
                beta, dynamic = solve_pg(
                    x,
                    y,
                    derivative,
                    raw_lambda,
                    collect=True,
                    gram=gram,
                    eta=eta,
                )
                prediction = x_test @ beta
                test_mse = float(torch.mean((prediction - y_test) ** 2))
                risk_rows.append(
                    {
                        "task_id": task.task_id,
                        "penalty_family": family,
                        "penalty_index": PENALTY_FAMILIES.index(family),
                        "lambda_ratio_index": ratio_index,
                        "zero_kkt_ratio": float(ratio),
                        "raw_lambda": raw_lambda,
                        "raw_lambda_was_clipped": int(raw_lambda != raw_unclipped),
                        "test_mse": test_mse,
                        **{f"penalty_descriptor_{i:02d}": float(v) for i, v in enumerate(descriptor)},
                    }
                )
                risk_dynamic.append(dynamic)
    risk_seconds = time.perf_counter() - risk_start
    write_csv(out / "candidate_risk.csv", risk_rows)

    cheap_names = sorted(next(iter(cheap_features.values())))
    dynamic_names = sorted(divergence_dynamic[0])
    risk_dynamic_names = sorted(risk_dynamic[0])
    baseline_div = finite_matrix(np.stack([task_codes[row["task_id"]] for row in divergence_rows]))
    cheap_div = finite_matrix(
        np.asarray(
            [[cheap_features[row["task_id"]][name] for name in cheap_names] for row in divergence_rows]
        )
    )
    dynamic_div = finite_matrix(
        np.asarray([[values[name] for name in dynamic_names] for values in divergence_dynamic])
    )
    target_div = np.asarray([row["conditional_divergence_target"] for row in divergence_rows])
    task_ids_div = np.asarray([row["task_id"] for row in divergence_rows])
    divergence_models = fit_divergence_models(
        baseline_div, cheap_div, dynamic_div, target_div, task_ids_div
    )

    baseline_risk = finite_matrix(np.stack([task_codes[row["task_id"]] for row in risk_rows]))
    cheap_risk = finite_matrix(
        np.asarray(
            [[cheap_features[row["task_id"]][name] for name in cheap_names] for row in risk_rows]
        )
    )
    candidate_names = sorted(
        key
        for key in risk_rows[0]
        if key.startswith("penalty_descriptor_")
    )
    candidate_risk = finite_matrix(
        np.asarray(
            [
                [
                    math.log(max(row["raw_lambda"], 1e-12)),
                    math.log(max(row["zero_kkt_ratio"], 1e-12)),
                    float(row["penalty_index"]),
                    *[row[name] for name in candidate_names],
                ]
                for row in risk_rows
            ]
        )
    )
    dynamic_risk = finite_matrix(
        np.asarray([[values[name] for name in risk_dynamic_names] for values in risk_dynamic])
    )
    target_risk = np.asarray([row["test_mse"] for row in risk_rows])
    task_ids_risk = np.asarray([row["task_id"] for row in risk_rows])
    risk_models = fit_risk_models(
        baseline_risk,
        cheap_risk,
        candidate_risk,
        dynamic_risk,
        target_risk,
        task_ids_risk,
    )

    exact_rows = [row for row in divergence_rows if np.isfinite(row["exact_trace"])]
    hutchinson_accuracy: dict[str, Any] = {}
    if exact_rows:
        exact = np.asarray([row["exact_trace"] for row in exact_rows])
        finite_difference = np.asarray([row["finite_difference_trace"] for row in exact_rows])
        hutchinson_accuracy["finite_difference"] = {
            "rmse": float(math.sqrt(mean_squared_error(exact, finite_difference))),
            "max_abs_error": float(np.max(np.abs(exact - finite_difference))),
        }
        for count in (1, 2, 4, 8):
            estimate = np.asarray([row[f"hutch_r{count}"] for row in exact_rows])
            hutchinson_accuracy[f"r{count}"] = {
                "rmse": float(math.sqrt(mean_squared_error(exact, estimate))),
                "bias": float(np.mean(estimate - exact)),
                "correlation": float(np.corrcoef(exact, estimate)[0, 1]),
            }

    summary = {
        "version": "kalbi_sure_divergence_identification_v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "tasks": args.tasks,
        "gaussian_tasks": sum(task.error_family == "gaussian" for task in tasks),
        "non_gaussian_tasks": sum(task.error_family != "gaussian" for task in tasks),
        "steps": args.steps,
        "exact_tasks": exact_count,
        "v23_code_width": int(next(iter(task_codes.values())).size),
        "cheap_summary_width": len(cheap_names),
        "dynamic_summary_width": len(dynamic_names),
        "candidate_count": len(risk_rows),
        "timing_seconds": {
            "v23_task_encoding": task_encoding_seconds,
            "divergence_targets": jvp_seconds,
            "candidate_risk_paths": risk_seconds,
        },
        "hutchinson_accuracy": hutchinson_accuracy,
        "divergence_models": divergence_models,
        "risk_models": risk_models,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "limitations": [
            "conditional divergence holds penalty and raw lambda fixed",
            "preprocessing statistics are treated as fixed",
            "current V23 selector trajectory is no_grad, so adaptive selector divergence is excluded",
            "non-Gaussian strata are stress tests and do not provide unbiased SURE",
        ],
        "feature_names": {
            "cheap": cheap_names,
            "dynamic": dynamic_names,
            "risk_dynamic": risk_dynamic_names,
            "candidate": candidate_names,
        },
    }
    (out / "study_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, sort_keys=True)
    )
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))


def runtime_case(n: int, p: int, threads: int, seed: int) -> dict[str, Any]:
    torch.set_num_threads(threads)
    rng = np.random.default_rng(seed)
    x_np = make_design(rng, n, p, 0.3)
    y_np = rng.normal(size=n)
    x = torch.from_numpy(x_np).float()
    y = torch.from_numpy(y_np).float()
    derivative = penalty_derivatives(torch.device("cpu"))["log"].float()
    lambda_max = float((x.T @ y / float(n)).abs().max())
    raw_lambda = 0.1 * lambda_max / float(derivative[0])
    vector = torch.ones(p, dtype=x.dtype)
    vector /= vector.norm().clamp_min(1e-12)
    for _ in range(12):
        vector = x.T @ (x @ vector) / float(n)
        vector /= vector.norm().clamp_min(1e-12)
    lipschitz = torch.dot(
        vector, x.T @ (x @ vector) / float(n)
    ).clamp_min(1e-8)
    eta = (0.9 / lipschitz).detach()
    solve_pg(x, y, derivative, raw_lambda, collect=False, eta=eta)
    start = time.perf_counter()
    solve_pg(x, y, derivative, raw_lambda, collect=False, eta=eta)
    base = time.perf_counter() - start
    start = time.perf_counter()
    _, summary = solve_pg(
        x, y, derivative, raw_lambda, collect=True, eta=eta
    )
    enhanced = time.perf_counter() - start
    solve_pg(
        x, y, derivative, raw_lambda, collect=True, compact=True, eta=eta
    )
    start = time.perf_counter()
    _, compact_summary = solve_pg(
        x, y, derivative, raw_lambda, collect=True, compact=True, eta=eta
    )
    compact = time.perf_counter() - start
    def estimator(response: torch.Tensor) -> torch.Tensor:
        beta, _ = solve_pg(
            x,
            response,
            derivative,
            raw_lambda,
            steps=10,
            collect=False,
            eta=eta,
        )
        return x @ beta
    hutchinson(estimator, y, probes=1, seed=seed + 123)
    start = time.perf_counter()
    hutchinson(estimator, y, probes=1, seed=seed + 456)
    hutch_r1 = time.perf_counter() - start
    return {
        "n": n,
        "p": p,
        "threads": threads,
        "base_seconds": base,
        "summary_seconds": enhanced,
        "incremental_fraction": enhanced / max(base, 1e-12) - 1.0,
        "compact_summary_seconds": compact,
        "compact_incremental_fraction": compact / max(base, 1e-12) - 1.0,
        "compact_summary_width": len(compact_summary),
        "hutch_r1_seconds": hutch_r1,
        "hutch_r1_incremental_fraction": hutch_r1 / max(base, 1e-12),
        "summary_width": len(summary),
        "dtype": "float32",
    }


def run_runtime(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = ((200, 500), (500, 1000), (1000, 5000), (500, 20000))
    rows = [
        runtime_case(n, p, threads, args.seed + n + p)
        for threads in (1, args.threads)
        for n, p in cases
    ]
    write_csv(out / "runtime_cpu.csv", rows)
    (out / "runtime_cpu.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("study", "runtime"), default="study")
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--tasks", type=int, default=256)
    parser.add_argument("--gaussian-fraction", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=2026072701)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--exact-tasks", type=int, default=16)
    parser.add_argument("--exact-max-n", type=int, default=24)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    if args.mode == "study" and not args.checkpoint:
        parser.error("--checkpoint is required in study mode")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.mode == "study":
        run_study(arguments)
    else:
        run_runtime(arguments)
