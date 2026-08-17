#!/usr/bin/env python3
"""Generate and freeze four leakage-safe V13 evaluation tasks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold

from exp122_shape_sampler import SHAPE_SAMPLER_NAME, sample_shape_pair


SOURCE_IDS = {"tabicl": 0, "linear": 1}
SOURCE_SHAPE_SEEDS = {
    "linear": 2_026_072_701,
    "tabicl": 2_026_072_702,
}
SOURCE_GENERATOR_SEED_BASE = {
    "linear": 2_126_070_001,
    "tabicl": 2_126_080_000,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def standardize_fit(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | float]]:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64).reshape(-1)
    x_mean = x64.mean(axis=0)
    x_scale = x64.std(axis=0)
    active = x_scale > 1e-8
    x_scale = np.where(active, x_scale, 1.0)
    y_mean = float(y64.mean())
    y_scale = max(float(y64.std()), 1e-8)
    return (
        np.asfortranarray((x64 - x_mean) / x_scale),
        np.ascontiguousarray((y64 - y_mean) / y_scale),
        {
            "x_mean": x_mean,
            "x_scale": x_scale,
            "active": active,
            "y_mean": y_mean,
            "y_scale": y_scale,
        },
    )


def standardize_x(
    x: np.ndarray, stats: dict[str, np.ndarray | float]
) -> np.ndarray:
    return np.asfortranarray(
        (np.asarray(x, dtype=np.float64) - stats["x_mean"])
        / stats["x_scale"]
    )


def lambda_maximum(x: np.ndarray, y: np.ndarray) -> float:
    return max(float(np.max(np.abs(x.T @ y)) / x.shape[0]), 1e-12)


def lasso_path(
    x: np.ndarray,
    y: np.ndarray,
    ratios_descending: np.ndarray,
    *,
    tolerance: float,
    max_iter: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    lambda_max = lambda_maximum(x, y)
    model = Lasso(
        alpha=lambda_max,
        fit_intercept=False,
        max_iter=max_iter,
        tol=tolerance,
        warm_start=True,
        selection="cyclic",
    )
    coefficients: list[np.ndarray] = []
    iterations: list[int] = []
    dual_gaps: list[float] = []
    warning_count = 0
    for ratio in ratios_descending:
        model.set_params(alpha=lambda_max * float(ratio))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x, y)
        warning_count += sum(
            issubclass(item.category, ConvergenceWarning) for item in caught
        )
        beta = np.asarray(model.coef_, dtype=np.float64)
        if not np.isfinite(beta).all() or not math.isfinite(
            float(model.dual_gap_)
        ):
            raise RuntimeError("CV-LASSO produced a nonfinite result")
        coefficients.append(beta.copy())
        iterations.append(int(model.n_iter_))
        dual_gaps.append(float(model.dual_gap_))
    return np.asarray(coefficients), {
        "max_iterations": max(iterations),
        "max_dual_gap": max(dual_gaps),
        "convergence_warning_count": warning_count,
    }


def predict_original_scale(
    x: np.ndarray,
    beta_path: np.ndarray,
    stats: dict[str, np.ndarray | float],
) -> np.ndarray:
    x_scaled = standardize_x(x, stats)
    return float(stats["y_mean"]) + float(stats["y_scale"]) * (
        np.asarray(beta_path, dtype=np.float64) @ x_scaled.T
    )


def r2_score(y: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    estimate = np.asarray(prediction, dtype=np.float64).reshape(-1)
    denominator = float(np.sum((target - target.mean()) ** 2))
    if denominator <= 1e-12:
        raise RuntimeError("R2 target is degenerate")
    return float(1.0 - np.sum((target - estimate) ** 2) / denominator)


def cv_lasso_screen(
    x_context: np.ndarray,
    y_context: np.ndarray,
    x_screen: np.ndarray,
    y_screen: np.ndarray,
    *,
    cv_seed: int,
    folds: int,
    grid_size: int,
    min_ratio: float,
    tolerance: float,
    max_iter: int,
) -> dict[str, Any]:
    ratios = np.logspace(
        0.0, math.log10(min_ratio), grid_size, dtype=np.float64
    )
    splitter = KFold(n_splits=folds, shuffle=True, random_state=cv_seed)
    fold_splits = list(splitter.split(x_context))

    def evaluate_fold(
        indexed_split: tuple[int, tuple[np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        fold_index, (fit_index, validation_index) = indexed_split
        x_fit, y_fit, stats = standardize_fit(
            x_context[fit_index], y_context[fit_index]
        )
        beta, diagnostic = lasso_path(
            x_fit,
            y_fit,
            ratios,
            tolerance=tolerance,
            max_iter=max_iter,
        )
        prediction = predict_original_scale(
            x_context[validation_index], beta, stats
        )
        mse = np.mean(
            (
                prediction
                - np.asarray(y_context[validation_index], dtype=np.float64)[
                    None, :
                ]
            )
            ** 2,
            axis=1,
        )
        return mse, {"fold": fold_index, **diagnostic}

    with ThreadPoolExecutor(max_workers=min(folds, 4)) as executor:
        fold_results = list(
            executor.map(evaluate_fold, enumerate(fold_splits))
        )
    score = np.asarray([result[0] for result in fold_results])
    fold_diagnostics = [result[1] for result in fold_results]
    mean_mse = score.mean(axis=0)
    selected_index = int(np.argmin(mean_mse))
    x_fit, y_fit, stats = standardize_fit(x_context, y_context)
    beta_prefix, final_diagnostic = lasso_path(
        x_fit,
        y_fit,
        ratios[: selected_index + 1],
        tolerance=tolerance,
        max_iter=max_iter,
    )
    beta = beta_prefix[-1]
    screen_prediction = predict_original_scale(
        x_screen, beta[None, :], stats
    )[0]
    screen_mse = float(
        np.mean(
            (
                np.asarray(y_screen, dtype=np.float64)
                - screen_prediction
            )
            ** 2
        )
    )
    return {
        "ratios_descending": ratios,
        "fold_mse": score,
        "mean_mse": mean_mse,
        "selected_index": selected_index,
        "selected_ratio": float(ratios[selected_index]),
        "selected_raw_lambda_standardized": float(
            lambda_maximum(x_fit, y_fit) * ratios[selected_index]
        ),
        "beta_standardized": beta,
        "screen_prediction": screen_prediction,
        "screen_mse": screen_mse,
        "screen_r2": r2_score(y_screen, screen_prediction),
        "fold_diagnostics": fold_diagnostics,
        "final_diagnostic": final_diagnostic,
        "full_context_stats": stats,
    }


def split_indices(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n).astype(np.int64)
    context_count = int(math.floor(0.60 * n))
    screening_count = int(math.floor(0.20 * n))
    final_count = n - context_count - screening_count
    if min(context_count, screening_count, final_count) < 32:
        raise RuntimeError(f"split has fewer than 32 rows: N={n}")
    return {
        "context": permutation[:context_count],
        "screening": permutation[
            context_count : context_count + screening_count
        ],
        "final": permutation[context_count + screening_count :],
    }


def source_generator_seed(source: str, eligible_index: int) -> int:
    parity = SOURCE_IDS[source]
    candidate = SOURCE_GENERATOR_SEED_BASE[source] + 2 * eligible_index
    if candidate % 2 != parity:
        candidate += 1
    return int(candidate)


def run_generator(
    *,
    python: Path,
    generator: Path,
    config: dict[str, Any],
    output: Path,
    seed: int,
    n: int,
    p: int,
    log_path: Path,
) -> tuple[Path, dict[str, Any]]:
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        manifest_document = json.loads(manifest_path.read_text())
        manifest = (
            manifest_document.get("files")
            if isinstance(manifest_document, dict)
            else manifest_document
        )
        if isinstance(manifest, list) and len(manifest) == 1:
            data_path = output / manifest[0]["path"]
            if data_path.is_file():
                entry = dict(manifest[0])
                entry["generator_wall_seconds"] = 0.0
                entry["reused_completed_candidate"] = True
                return data_path, entry
    command = [
        str(python),
        str(generator),
        "--out-dir",
        str(output),
        "--num-batches",
        "1",
        "--batch-size",
        "1",
        "--start-index",
        "0",
        "--end-index-exclusive",
        "1",
        "--seed",
        str(seed),
        "--threads",
        "4",
        "--corr-device",
        "cpu",
        "--force-n",
        str(n),
        "--force-p",
        str(p),
        *[str(value) for value in config["generator_common_args"]],
        *[
            str(value)
            for value in config["generator_sources"]["tabicl_v2"]["args"]
        ],
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=Path(config["project_dir"]),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"generator exited {completed.returncode}; see {log_path}"
        )
    manifest_document = json.loads((output / "manifest.json").read_text())
    if isinstance(manifest_document, dict):
        manifest = manifest_document.get("files")
    else:
        manifest = manifest_document
    if not isinstance(manifest, list) or len(manifest) != 1:
        raise RuntimeError("candidate generator manifest cardinality drift")
    entry = manifest[0]
    entry["generator_wall_seconds"] = time.perf_counter() - started
    return output / entry["path"], entry


def max_abs_correlation(x: np.ndarray) -> dict[str, float]:
    values = np.asarray(x, dtype=np.float64)
    values -= values.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(values, axis=0)
    valid = norms > 1e-12
    normalized = values[:, valid] / norms[valid][None, :]
    corr = np.clip(normalized.T @ normalized, -1.0, 1.0)
    upper = np.abs(corr[np.triu_indices(corr.shape[0], 1)])
    if upper.size == 0:
        upper = np.asarray([0.0])
    return {
        "max": float(upper.max()),
        "mean": float(upper.mean()),
        "median": float(np.median(upper)),
        "q90": float(np.quantile(upper, 0.90)),
        "q99": float(np.quantile(upper, 0.99)),
        "valid_features": int(valid.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--producer-config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--python", type=Path, default=Path("/venv/main/bin/python"))
    parser.add_argument("--accepted-per-source", type=int, default=2)
    parser.add_argument("--max-shape-draws", type=int, default=50_000)
    parser.add_argument("--max-generated-per-source", type=int, default=100)
    parser.add_argument("--screening-r2-threshold", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-grid-size", type=int, default=25)
    parser.add_argument("--cv-min-ratio", type=float, default=1e-4)
    parser.add_argument("--lasso-tolerance", type=float, default=1e-8)
    parser.add_argument("--lasso-max-iter", type=int, default=30_000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.output_root.exists():
        if not args.resume:
            raise RuntimeError(
                f"refusing existing output root: {args.output_root}"
            )
        if (args.output_root / "manifest" / "accepted_seeds.json").exists():
            raise RuntimeError(
                "resume is only allowed before the first accepted task"
            )
    else:
        args.output_root.mkdir(parents=True)
    for directory in ("data", "splits", "metrics", "manifest", "logs"):
        (args.output_root / directory).mkdir(exist_ok=True)
    if sha256(args.producer_config) != args.expected_config_sha256:
        raise RuntimeError("producer config SHA256 mismatch")
    config = json.loads(args.producer_config.read_text())
    generator = Path(
        config["generator_sources"]["tabicl_v2"]["path"]
    ).resolve()
    expected_generator_sha = config["generator_sources"]["tabicl_v2"][
        "sha256"
    ]
    if sha256(generator) != expected_generator_sha:
        raise RuntimeError("active V13 generator SHA256 mismatch")

    common = SimpleNamespace(
        n_min=32,
        n_max=40_000,
        main_n_min=200,
        small_n_max=199,
        small_n_probability=0.10,
        p_min=50,
        p_max=100_000,
        max_cells=24_000_000,
        p_median=10_000.0,
        p_log_sigma=0.75,
        raw_n_median=5_700.0,
        n_log_sigma=0.70,
    )
    shape_kwargs = vars(common)
    acceptance_log: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []

    for source in ("linear", "tabicl"):
        shape_rng = np.random.default_rng(SOURCE_SHAPE_SEEDS[source])
        source_accepted = 0
        generated_count = 0
        for draw_index in range(args.max_shape_draws):
            n, p = sample_shape_pair(shape_rng, **shape_kwargs)
            base_record: dict[str, Any] = {
                "source": source,
                "shape_draw_index": draw_index,
                "shape_stream_seed": SOURCE_SHAPE_SEEDS[source],
                "n": int(n),
                "p": int(p),
            }
            if not (n > 200 and p < 1000):
                acceptance_log.append(
                    {**base_record, "disposition": "n_p_condition_rejection"}
                )
                continue
            if generated_count >= args.max_generated_per_source:
                break
            generator_seed = source_generator_seed(source, generated_count)
            generated_count += 1
            candidate_id = f"{source}_candidate_{generated_count:03d}"
            candidate_dir = args.output_root / "candidate_work" / candidate_id
            log_path = args.output_root / "logs" / f"{candidate_id}.log"
            split_seed = int(
                (generator_seed ^ 0x5EED_2026) % (2**32 - 1)
            )
            cv_seed = int((split_seed ^ 0xC0FF_EE11) % (2**32 - 1))
            if (
                args.resume
                and log_path.is_file()
                and not candidate_dir.exists()
            ):
                acceptance_log.append(
                    {
                        **base_record,
                        "candidate_id": candidate_id,
                        "generator_seed": generator_seed,
                        "split_seed": split_seed,
                        "cv_seed": cv_seed,
                        "disposition": "screening_lasso_rejection",
                        "screening_metrics_unavailable_reason": (
                            "completed before interrupted progress persistence"
                        ),
                        "log_path": str(log_path),
                    }
                )
                continue
            try:
                data_path, generator_entry = run_generator(
                    python=args.python,
                    generator=generator,
                    config=config,
                    output=candidate_dir,
                    seed=generator_seed,
                    n=n,
                    p=p,
                    log_path=log_path,
                )
                with np.load(data_path, allow_pickle=False) as archive:
                    x = np.asarray(archive["X"][0], dtype=np.float32)
                    y = np.asarray(archive["y"][0], dtype=np.float32)
                    outcome_source_id = int(archive["outcome_source_id"][0])
                if outcome_source_id != SOURCE_IDS[source]:
                    raise RuntimeError("forced source parity contract failed")
                split = split_indices(n, split_seed)
                screening = cv_lasso_screen(
                    x[split["context"]],
                    y[split["context"]],
                    x[split["screening"]],
                    y[split["screening"]],
                    cv_seed=cv_seed,
                    folds=args.cv_folds,
                    grid_size=args.cv_grid_size,
                    min_ratio=args.cv_min_ratio,
                    tolerance=args.lasso_tolerance,
                    max_iter=args.lasso_max_iter,
                )
                record = {
                    **base_record,
                    "candidate_id": candidate_id,
                    "generator_seed": generator_seed,
                    "split_seed": split_seed,
                    "cv_seed": cv_seed,
                    "generator_entry": generator_entry,
                    "task_sha256": sha256(data_path),
                    "screening_lasso_r2": screening["screen_r2"],
                    "screening_lasso_mse": screening["screen_mse"],
                    "screening_lasso_selected_ratio": screening[
                        "selected_ratio"
                    ],
                    "screening_lasso_selected_raw_lambda_standardized": screening[
                        "selected_raw_lambda_standardized"
                    ],
                }
                if screening["screen_r2"] <= args.screening_r2_threshold:
                    acceptance_log.append(
                        {**record, "disposition": "screening_lasso_rejection"}
                    )
                    shutil.rmtree(candidate_dir)
                    continue

                label = f"{source}_{source_accepted + 1}"
                accepted_data_path = args.output_root / "data" / f"{label}.npz"
                shutil.copy2(data_path, accepted_data_path)
                final_prediction = predict_original_scale(
                    x[split["final"]],
                    np.asarray(screening["beta_standardized"])[None, :],
                    screening["full_context_stats"],
                )[0]
                final_mse = float(
                    np.mean(
                        (
                            np.asarray(y[split["final"]], dtype=np.float64)
                            - final_prediction
                        )
                        ** 2
                    )
                )
                final_r2 = r2_score(y[split["final"]], final_prediction)
                stats = screening["full_context_stats"]
                split_path = args.output_root / "splits" / f"{label}.npz"
                np.savez_compressed(
                    split_path,
                    context_indices=split["context"],
                    screening_indices=split["screening"],
                    final_indices=split["final"],
                    x_mean=np.asarray(stats["x_mean"]),
                    x_scale=np.asarray(stats["x_scale"]),
                    active=np.asarray(stats["active"]),
                    y_mean=np.asarray(stats["y_mean"]),
                    y_scale=np.asarray(stats["y_scale"]),
                    screening_lasso_beta=np.asarray(
                        screening["beta_standardized"]
                    ),
                    cv_ratios=screening["ratios_descending"],
                    cv_fold_mse=screening["fold_mse"],
                    cv_mean_mse=screening["mean_mse"],
                )
                correlation = max_abs_correlation(x)
                accepted_record = {
                    **record,
                    "label": label,
                    "dataset_index": len(accepted) + 1,
                    "disposition": "accepted",
                    "accepted_data_path": str(accepted_data_path),
                    "accepted_data_sha256": sha256(accepted_data_path),
                    "split_path": str(split_path),
                    "split_sha256": sha256(split_path),
                    "row_counts": {
                        key: int(value.size) for key, value in split.items()
                    },
                    "feature_correlation": correlation,
                    "zero_predictor_screening_mse": float(
                        np.mean(
                            (
                                y[split["screening"]]
                                - y[split["context"]].mean()
                            )
                            ** 2
                        )
                    ),
                    "zero_predictor_final_mse": float(
                        np.mean(
                            (
                                y[split["final"]]
                                - y[split["context"]].mean()
                            )
                            ** 2
                        )
                    ),
                    "final_lasso_mse": final_mse,
                    "final_lasso_r2": final_r2,
                    "cv_fold_diagnostics": screening["fold_diagnostics"],
                    "cv_final_diagnostic": screening["final_diagnostic"],
                }
                acceptance_log.append(accepted_record)
                accepted.append(accepted_record)
                source_accepted += 1
                shutil.rmtree(candidate_dir)
                write_json(
                    args.output_root / "manifest" / "accepted_seeds.json",
                    accepted,
                )
                write_json(
                    args.output_root / "manifest" / "acceptance_log.json",
                    acceptance_log,
                )
                if source_accepted >= args.accepted_per_source:
                    break
            except Exception as exc:  # noqa: BLE001
                acceptance_log.append(
                    {
                        **base_record,
                        "candidate_id": candidate_id,
                        "generator_seed": generator_seed,
                        "disposition": "generator_quality_rejection",
                        "error": repr(exc),
                        "log_path": str(log_path),
                    }
                )
                if candidate_dir.exists():
                    shutil.rmtree(candidate_dir)
        if source_accepted < args.accepted_per_source:
            write_json(
                args.output_root / "manifest" / "acceptance_log.json",
                acceptance_log,
            )
            raise RuntimeError(
                f"accepted only {source_accepted} {source} tasks after "
                f"{args.max_shape_draws} shape draws and {generated_count} "
                "generated candidates"
            )

    write_json(
        args.output_root / "manifest" / "accepted_seeds.json", accepted
    )
    write_json(
        args.output_root / "manifest" / "acceptance_log.json",
        acceptance_log,
    )
    rejected = [
        item for item in acceptance_log if item["disposition"] != "accepted"
    ]
    write_json(
        args.output_root / "manifest" / "rejected_seeds.json", rejected
    )
    with (args.output_root / "metrics" / "acceptance_log.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "source",
            "shape_draw_index",
            "shape_stream_seed",
            "n",
            "p",
            "candidate_id",
            "generator_seed",
            "disposition",
            "screening_lasso_r2",
            "screening_lasso_mse",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(acceptance_log)
    summary = {
        "status": "PASS",
        "shape_sampler": SHAPE_SAMPLER_NAME,
        "producer_config": str(args.producer_config.resolve()),
        "producer_config_sha256": sha256(args.producer_config),
        "generator": str(generator),
        "generator_sha256": sha256(generator),
        "screening_threshold": args.screening_r2_threshold,
        "screening_contract": (
            "context-only inner CV; screening holdout acceptance; final test "
            "unused until acceptance"
        ),
        "accepted_count": len(accepted),
        "accepted": accepted,
        "attempt_counts": {
            source: sum(item["source"] == source for item in acceptance_log)
            for source in SOURCE_IDS
        },
        "generated_counts": {
            source: sum(
                item["source"] == source and "candidate_id" in item
                for item in acceptance_log
            )
            for source in SOURCE_IDS
        },
        "screening_rejection_counts": {
            source: sum(
                item["source"] == source
                and item["disposition"] == "screening_lasso_rejection"
                for item in acceptance_log
            )
            for source in SOURCE_IDS
        },
    }
    write_json(args.output_root / "metrics" / "preparation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
