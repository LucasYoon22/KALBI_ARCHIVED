#!/usr/bin/env python3
"""Audit SCM/group-drop pilot tasks against correlation and signal nulls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, action="append", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=202607141100)
    p.add_argument("--max-corr-features", type=int, default=400)
    p.add_argument("--ridge-features", type=int, default=128)
    p.add_argument("--max-tasks-per-mode", type=int, default=0, help="0 audits every manifest-backed task")
    p.add_argument("--report-label", default="pilot")
    p.add_argument("--snr-min", type=float, default=3.0)
    p.add_argument("--snr-max", type=float, default=30.0)
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def standardize_fit_apply(x_fit: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mu = x_fit.mean(axis=0, keepdims=True)
    sd = x_fit.std(axis=0, keepdims=True)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return tuple(((a - mu) / sd).astype(np.float64, copy=False) for a in (x_fit, *others))


def ridge_from_svd(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    return vt.T @ ((s / (s * s + alpha)) * (u.T @ y))


def leakage_free_ridge(
    x: np.ndarray,
    y: np.ndarray,
    *,
    split_seed: int,
    max_features: int,
) -> dict[str, float]:
    rng = np.random.default_rng(split_seed)
    n, p = x.shape
    order = rng.permutation(n)
    n_train = n // 2
    train_idx, test_idx = order[:n_train], order[n_train:]
    train_order = rng.permutation(n_train)
    n_fit = max(2, int(round(0.75 * n_train)))
    fit_local, val_local = train_order[:n_fit], train_order[n_fit:]
    if val_local.size < 2:
        raise ValueError("pilot task too small for nested train/validation split")

    x_train_raw = x[train_idx].astype(np.float64, copy=False)
    x_test_raw = x[test_idx].astype(np.float64, copy=False)
    y_train_raw = y[train_idx].astype(np.float64, copy=False)
    y_test_raw = y[test_idx].astype(np.float64, copy=False)

    x_fit_raw = x_train_raw[fit_local]
    y_fit_raw = y_train_raw[fit_local]
    x_fit_all, x_val_all = standardize_fit_apply(x_fit_raw, x_train_raw[val_local])
    y_fit, y_val = standardize_fit_apply(y_fit_raw.reshape(-1, 1), y_train_raw[val_local].reshape(-1, 1))
    y_fit = y_fit.reshape(-1)
    y_val = y_val.reshape(-1)
    marginal = np.abs(x_fit_all.T @ y_fit) / max(1, y_fit.size)
    q = min(max_features, p, max(1, n_fit - 2))
    selected = np.argpartition(marginal, -q)[-q:]
    selected = selected[np.argsort(marginal[selected])[::-1]]

    xf = x_fit_all[:, selected]
    xv = x_val_all[:, selected]
    u, s, vt = np.linalg.svd(xf, full_matrices=False)
    uty = u.T @ y_fit
    alpha_grid = n_fit * np.logspace(-5.0, 3.0, 17)
    val_mse: list[float] = []
    for alpha in alpha_grid:
        beta = vt.T @ ((s / (s * s + alpha)) * uty)
        val_mse.append(float(np.mean((y_val - xv @ beta) ** 2)))
    best_i = int(np.argmin(val_mse))
    best_alpha_ratio = float(alpha_grid[best_i] / n_fit)

    x_train, x_test = standardize_fit_apply(x_train_raw[:, selected], x_test_raw[:, selected])
    y_train, y_test = standardize_fit_apply(y_train_raw.reshape(-1, 1), y_test_raw.reshape(-1, 1))
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    beta = ridge_from_svd(x_train, y_train, best_alpha_ratio * n_train)
    pred = x_test @ beta
    mse = float(np.mean((y_test - pred) ** 2))
    denom = float(np.sum((y_test - y_test.mean()) ** 2))
    r2 = float(1.0 - np.sum((y_test - pred) ** 2) / max(denom, 1e-12))
    return {
        "test_r2": r2,
        "test_mse": mse,
        "selected_features": int(q),
        "best_alpha_over_n": best_alpha_ratio,
        "max_fit_marginal_abs_corr": float(marginal.max()),
    }


def corr_view(x: np.ndarray, feature_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = x[:, feature_idx].astype(np.float64, copy=False)
    a -= a.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(a, axis=0)
    keep = np.isfinite(norm) & (norm > 1e-10)
    if not np.all(keep):
        raise ValueError("stored pilot unexpectedly contains a constant sampled column")
    z = a / norm[None, :]
    c = np.clip(z.T @ z, -1.0, 1.0)
    off = np.abs(c[np.triu_indices(c.shape[0], 1)])
    return c, off


def greedy_order(corr: np.ndarray) -> np.ndarray:
    a = np.abs(corr)
    q = a.shape[0]
    start = int(np.argmax(a.sum(axis=1)))
    order = [start]
    used = np.zeros(q, dtype=bool)
    used[start] = True
    current = start
    for _ in range(1, q):
        scores = a[current].copy()
        scores[used] = -np.inf
        current = int(np.argmax(scores))
        order.append(current)
        used[current] = True
    return np.asarray(order, dtype=np.int64)


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def plot_heatmaps(rows: list[dict[str, Any]], mode: str, out: Path, report_label: str) -> None:
    subset = [r for r in rows if r["mode"] == mode]
    subset = sorted(subset, key=lambda r: r["sample_corr_max"])
    choose = np.linspace(0, len(subset) - 1, min(6, len(subset)), dtype=int)
    picked = [subset[i] for i in choose]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.0), constrained_layout=True)
    im = None
    for ax, row in zip(axes.ravel(), picked):
        c = row.pop("_corr") if False else row["_corr"]
        order = greedy_order(c)
        im = ax.imshow(c[np.ix_(order, order)], vmin=-0.95, vmax=0.95, cmap="coolwarm", interpolation="nearest")
        ax.set_title(
            f"task {row['task_index_global']}  N={row['n']:,}, P={row['p']:,}\n"
            f"sample max={row['sample_corr_max']:.3f}, exact max={row['exact_max_abs_corr']:.3f}",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(picked):]:
        ax.axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="Pearson correlation")
    label = "pure TabICLv2 graph_scm after group-drop" if mode == "direct" else "TabICLv2 core + random MLP512"
    fig.suptitle(f"TabICLv2/group-drop {report_label} feature correlations — {label}\nGreedy |r| ordering; deterministic columns/task", fontsize=15, fontweight="bold")
    fig.savefig(out, dpi=220, facecolor="white")
    plt.close(fig)


def plot_summary(rows: list[dict[str, Any]], metrics: dict[str, Any], out: Path, report_label: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)
    colors = {"direct": "#0072B2", "expanded": "#D55E00"}
    for mode in ("direct", "expanded"):
        sub = [r for r in rows if r["mode"] == mode]
        axes[0, 0].scatter(
            [r["empirical_snr"] for r in sub],
            [r["actual_test_r2"] for r in sub],
            label=mode,
            color=colors[mode],
            s=48,
            alpha=0.85,
        )
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("empirical signal/noise variance ratio")
    axes[0, 0].set_ylabel("held-out ridge R²")
    axes[0, 0].axhline(0.25, color="black", ls=":", lw=1)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    positions = [1, 2, 4, 5]
    values = []
    labels = []
    for mode in ("direct", "expanded"):
        sub = [r for r in rows if r["mode"] == mode]
        values.extend([[r["actual_test_r2"] for r in sub], [r["null_test_r2"] for r in sub]])
        labels.extend([f"{mode}\nactual", f"{mode}\nperm-y"])
    bp = axes[0, 1].boxplot(
        values,
        positions=positions,
        widths=0.7,
        patch_artist=True,
        tick_labels=labels,
        showfliers=False,
    )
    for patch, color in zip(bp["boxes"], [colors["direct"], "#999999", colors["expanded"], "#999999"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    axes[0, 1].axhline(0.0, color="black", lw=1)
    axes[0, 1].set_ylabel("held-out R²")
    axes[0, 1].grid(axis="y", alpha=0.25)
    flat_box = np.concatenate([np.asarray(v, dtype=float) for v in values])
    finite_box = flat_box[np.isfinite(flat_box)]
    lower_view = min(-0.25, max(-1.0, float(np.quantile(finite_box, 0.02)) - 0.10))
    axes[0, 1].set_ylim(lower_view, 1.05)
    hidden_low = int(np.sum(finite_box < lower_view))
    if hidden_low:
        axes[0, 1].text(
            0.02,
            0.03,
            f"{hidden_low} extreme low value(s) below view; full min={finite_box.min():.2f}",
            transform=axes[0, 1].transAxes,
            fontsize=8,
            color="#555555",
        )

    modes = ["direct", "expanded"]
    actual95 = [metrics["modes"][m]["correlation"]["actual"]["p95"] for m in modes]
    null95 = [metrics["modes"][m]["correlation"]["gaussian_null"]["p95"] for m in modes]
    x = np.arange(2)
    axes[1, 0].bar(x - 0.18, actual95, 0.36, label="actual TabICLv2-derived", color=[colors[m] for m in modes])
    axes[1, 0].bar(x + 0.18, null95, 0.36, label="matched Gaussian null", color="#999999")
    axes[1, 0].set_xticks(x, modes)
    axes[1, 0].set_ylabel("pooled p95 |feature correlation|")
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.25)

    for mode in modes:
        sub = [r for r in rows if r["mode"] == mode]
        axes[1, 1].hist(
            [r["exact_max_abs_corr"] for r in sub],
            bins=np.linspace(0.80, 0.95, 16),
            alpha=0.6,
            color=colors[mode],
            label=mode,
        )
    axes[1, 1].axvline(0.95, color="black", ls="--", label="exclusive threshold")
    axes[1, 1].set_xlabel("exact stored max |correlation|")
    axes[1, 1].set_ylabel("tasks")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)

    status = "PASS" if metrics["all_gates_pass"] else "FAIL"
    fig.suptitle(f"TabICLv2/group-drop {report_label}: correlation and held-out linear-signal gates — {status}", fontsize=15, fontweight="bold")
    fig.savefig(out, dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    global_task = 0
    mode_counts = {"direct": 0, "expanded": 0}

    for root in args.root:
        root_manifest = root / "manifest.json"
        manifest_paths = [root_manifest] if root_manifest.is_file() else sorted(root.glob("part_*/manifest.json"))
        records: list[tuple[Path, dict[str, Any], Path]] = []
        if manifest_paths:
            for manifest_path in manifest_paths:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for entry in manifest.get("files", []):
                    records.append((manifest_path, entry, manifest_path.parent / entry["path"]))
        else:
            # V12 rolling production publishes one immutable metadata JSON per
            # shard instead of rewriting an ever-growing generator manifest.
            inbox_paths = sorted((root / "staging" / "inbox").rglob("*.json"))
            for inbox_path in inbox_paths:
                shard = json.loads(inbox_path.read_text(encoding="utf-8"))
                metadata = shard.get("metadata", {})
                entry = metadata.get("generator_manifest_entry")
                if not isinstance(entry, dict):
                    raise ValueError(f"rolling shard lacks generator manifest entry: {inbox_path}")
                records.append((inbox_path, entry, root / shard["relative_path"]))
        if not records:
            raise FileNotFoundError(f"no manifest-backed or rolling-inbox tasks found under {root}")
        for manifest_path, entry, path in records:
                entry_mode = "direct" if int(entry.get("requested_p", entry["p"])) <= 4096 else "expanded"
                if args.max_tasks_per_mode > 0 and mode_counts[entry_mode] >= args.max_tasks_per_mode:
                    continue
                z = np.load(path)
                x_all = z["X"]
                y_all = z["y"]
                exact_all = z["exact_max_abs_corr"]
                snr_all = z["target_snr"]
                provenance.append({
                    "root": str(root),
                    "manifest": str(manifest_path),
                    "file": str(path),
                    "file_sha256": sha256(path),
                    "n": int(entry["n"]),
                    "p": int(entry["p"]),
                    "requested_p": int(entry.get("requested_p", entry["p"])),
                    "num_tasks": int(entry["num_tasks"]),
                })
                for task_idx in range(x_all.shape[0]):
                    x = x_all[task_idx]
                    y = y_all[task_idx]
                    n, p = x.shape
                    mode = entry_mode
                    if args.max_tasks_per_mode > 0 and mode_counts[mode] >= args.max_tasks_per_mode:
                        continue
                    q = min(args.max_corr_features, p)
                    feat = np.sort(rng.choice(p, size=q, replace=False))
                    corr, off = corr_view(x, feat)
                    g = rng.normal(size=(n, q)).astype(np.float16)
                    null_corr, null_off = corr_view(g, np.arange(q))
                    actual = leakage_free_ridge(
                        x, y, split_seed=args.seed + 10_003 * global_task, max_features=args.ridge_features
                    )
                    perm_rng = np.random.default_rng(args.seed + 20_003 * global_task + 17)
                    y_perm = perm_rng.permutation(y)
                    null = leakage_free_ridge(
                        x, y_perm, split_seed=args.seed + 10_003 * global_task, max_features=args.ridge_features
                    )
                    meta = entry.get("task_metadata", [{}] * x_all.shape[0])[task_idx]
                    chunks = list(meta.get("mlp_chunks", []))
                    rows.append({
                        "task_index_global": int(global_task),
                        "root": str(root),
                        "file": str(path),
                        "task_index_file": int(task_idx),
                        "mode": mode,
                        "n": int(n),
                        "p": int(p),
                        "requested_p": int(meta.get("requested_p", entry.get("requested_p", p))),
                        "size_regime": "small_n" if n < 200 else "regular_n",
                        "corr_feature_count": int(q),
                        "feature_indices_sha256": hashlib.sha256(feat.astype(np.int64).tobytes()).hexdigest(),
                        "exact_max_abs_corr": float(exact_all[task_idx]),
                        "sample_corr_mean": float(off.mean()),
                        "sample_corr_p95": float(np.quantile(off, 0.95)),
                        "sample_corr_p99": float(np.quantile(off, 0.99)),
                        "sample_corr_max": float(off.max()),
                        "sample_effective_rank_pr": float(q * q / np.sum(corr * corr)),
                        "null_corr_mean": float(null_off.mean()),
                        "null_corr_p95": float(np.quantile(null_off, 0.95)),
                        "null_corr_p99": float(np.quantile(null_off, 0.99)),
                        "null_corr_max": float(null_off.max()),
                        "null_effective_rank_pr": float(q * q / np.sum(null_corr * null_corr)),
                        "target_snr": float(snr_all[task_idx]),
                        "empirical_snr": float(
                            meta.get("empirical_signal_to_noise_variance_ratio", snr_all[task_idx])
                        ),
                        "oracle_linear_r2": float(
                            meta.get("oracle_linear_r2", snr_all[task_idx] / (1 + snr_all[task_idx]))
                        ),
                        "support_size": int(meta.get("support_size", -1)),
                        "mlp_chunk_count": len(chunks),
                        "mlp_input_variance_median": float(
                            np.median([c["gaussian_input_variance"] for c in chunks])
                        ) if chunks else float("nan"),
                        "mlp_input_realized_ratio_median": float(
                            np.median([
                                c["gaussian_input_realized_variance"] / c["gaussian_input_variance"]
                                for c in chunks
                            ])
                        ) if chunks else float("nan"),
                        "mlp_output_variance_median": float(
                            np.median([c["output_noise_variance"] for c in chunks])
                        ) if chunks else float("nan"),
                        "mlp_output_realized_ratio_median": float(
                            np.median([
                                c["output_noise_realized_variance"] / c["output_noise_variance"]
                                for c in chunks
                            ])
                        ) if chunks else float("nan"),
                        "actual_test_r2": actual["test_r2"],
                        "actual_test_mse": actual["test_mse"],
                        "actual_max_fit_marginal": actual["max_fit_marginal_abs_corr"],
                        "actual_alpha_over_n": actual["best_alpha_over_n"],
                        "null_test_r2": null["test_r2"],
                        "null_test_mse": null["test_mse"],
                        "null_max_fit_marginal": null["max_fit_marginal_abs_corr"],
                        "null_alpha_over_n": null["best_alpha_over_n"],
                        "_corr": corr,
                        "_off": off,
                        "_null_off": null_off,
                    })
                    global_task += 1
                    mode_counts[mode] += 1
                z.close()

    metrics: dict[str, Any] = {
        "seed": int(args.seed),
        "num_tasks": len(rows),
        "gate_contract": {
            "exact_stored_max_abs_corr_exclusive": 0.95,
            "bulk_p99_abs_corr_max": 0.90,
            "actual_to_gaussian_p95_min_ratio": 1.25,
            "actual_to_gaussian_mean_min_ratio": 1.10,
            "empirical_snr_range_inclusive": [float(args.snr_min), float(args.snr_max)],
            "median_heldout_ridge_r2_min": 0.25,
            "median_r2_advantage_over_permutation_min": 0.25,
        },
        "modes": {},
        "size_regimes": {},
    }
    all_pass = True
    for mode in ("direct", "expanded"):
        sub = [r for r in rows if r["mode"] == mode]
        actual_off = np.concatenate([r["_off"] for r in sub])
        null_off = np.concatenate([r["_null_off"] for r in sub])
        actual_corr = summarize(actual_off)
        null_corr = summarize(null_off)
        r2 = np.asarray([r["actual_test_r2"] for r in sub])
        null_r2 = np.asarray([r["null_test_r2"] for r in sub])
        snr = np.asarray([r["empirical_snr"] for r in sub])
        exact_max = np.asarray([r["exact_max_abs_corr"] for r in sub])
        gates = {
            "exact_max_below_0p95": bool(np.all(exact_max < 0.95)),
            "bulk_p99_below_0p90": bool(actual_corr["p99"] < 0.90),
            "p95_not_gaussian_null_like": bool(actual_corr["p95"] / null_corr["p95"] >= 1.25),
            "mean_not_gaussian_null_like": bool(actual_corr["mean"] / null_corr["mean"] >= 1.10),
            "controlled_snr_in_range": bool(np.all((snr >= args.snr_min - 1e-5) & (snr <= args.snr_max + 1e-4))),
            "median_heldout_ridge_r2_at_least_0p25": bool(np.median(r2) >= 0.25),
            "median_r2_advantage_over_permutation_at_least_0p25": bool(np.median(r2) - np.median(null_r2) >= 0.25),
        }
        mode_pass = all(gates.values())
        all_pass = all_pass and mode_pass
        metrics["modes"][mode] = {
            "num_tasks": len(sub),
            "correlation": {
                "actual": actual_corr,
                "gaussian_null": null_corr,
                "actual_to_null_mean_ratio": actual_corr["mean"] / null_corr["mean"],
                "actual_to_null_p95_ratio": actual_corr["p95"] / null_corr["p95"],
                "median_participation_ratio_effective_rank": float(np.median([r["sample_effective_rank_pr"] for r in sub])),
                "null_median_participation_ratio_effective_rank": float(np.median([r["null_effective_rank_pr"] for r in sub])),
                "exact_task_max_min_median_max": [float(exact_max.min()), float(np.median(exact_max)), float(exact_max.max())],
            },
            "signal": {
                "empirical_snr_min_median_max": [float(snr.min()), float(np.median(snr)), float(snr.max())],
                "heldout_ridge_r2_min_median_max": [float(r2.min()), float(np.median(r2)), float(r2.max())],
                "permutation_y_r2_min_median_max": [float(null_r2.min()), float(np.median(null_r2)), float(null_r2.max())],
                "median_r2_advantage": float(np.median(r2) - np.median(null_r2)),
                "median_actual_max_fit_marginal": float(np.median([r["actual_max_fit_marginal"] for r in sub])),
                "median_null_max_fit_marginal": float(np.median([r["null_max_fit_marginal"] for r in sub])),
            },
            "gates": gates,
            "all_gates_pass": mode_pass,
        }
    for size_regime in ("small_n", "regular_n"):
        sub = [r for r in rows if r["size_regime"] == size_regime]
        if not sub:
            metrics["size_regimes"][size_regime] = {"num_tasks": 0, "all_gates_pass": False}
            all_pass = False
            continue
        r2 = np.asarray([r["actual_test_r2"] for r in sub])
        null_r2 = np.asarray([r["null_test_r2"] for r in sub])
        snr = np.asarray([r["empirical_snr"] for r in sub])
        exact_max = np.asarray([r["exact_max_abs_corr"] for r in sub])
        min_r2 = 0.10 if size_regime == "small_n" else 0.25
        min_adv = 0.10 if size_regime == "small_n" else 0.25
        gates = {
            "exact_max_below_0p95": bool(np.all(exact_max < 0.95)),
            "controlled_snr_in_range": bool(np.all((snr >= args.snr_min - 1e-5) & (snr <= args.snr_max + 1e-4))),
            "median_heldout_ridge_r2": bool(np.median(r2) >= min_r2),
            "median_advantage_over_permutation": bool(np.median(r2) - np.median(null_r2) >= min_adv),
        }
        regime_pass = all(gates.values())
        all_pass = all_pass and regime_pass
        metrics["size_regimes"][size_regime] = {
            "num_tasks": len(sub),
            "n_min_median_max": [int(min(r["n"] for r in sub)), float(np.median([r["n"] for r in sub])), int(max(r["n"] for r in sub))],
            "empirical_snr_min_median_max": [float(snr.min()), float(np.median(snr)), float(snr.max())],
            "heldout_ridge_r2_min_median_max": [float(r2.min()), float(np.median(r2)), float(r2.max())],
            "permutation_y_r2_min_median_max": [float(null_r2.min()), float(np.median(null_r2)), float(null_r2.max())],
            "gates": gates,
            "all_gates_pass": regime_pass,
        }
    expanded_rows = [r for r in rows if r["mode"] == "expanded"]
    input_ratios = np.asarray([r["mlp_input_realized_ratio_median"] for r in expanded_rows], dtype=float)
    output_ratios = np.asarray([r["mlp_output_realized_ratio_median"] for r in expanded_rows], dtype=float)
    noise_scale_pass = bool(
        np.all(np.isfinite(input_ratios))
        and np.all(np.isfinite(output_ratios))
        and np.all((input_ratios >= 0.95) & (input_ratios <= 1.05))
        and np.all((output_ratios >= 0.95) & (output_ratios <= 1.05))
    )
    metrics["mlp_noise_scale_audit"] = {
        "input_realized_over_sampled_min_median_max": [float(input_ratios.min()), float(np.median(input_ratios)), float(input_ratios.max())],
        "output_realized_over_sampled_min_median_max": [float(output_ratios.min()), float(np.median(output_ratios)), float(output_ratios.max())],
        "all_gates_pass": noise_scale_pass,
    }
    all_pass = all_pass and noise_scale_pass
    metrics["all_gates_pass"] = bool(all_pass)

    plot_heatmaps(rows, "direct", args.output_dir / "feature_correlation_direct_greedy.png", args.report_label)
    plot_heatmaps(rows, "expanded", args.output_dir / "feature_correlation_expanded_greedy.png", args.report_label)
    plot_summary(rows, metrics, args.output_dir / "snr_correlation_gate_summary.png", args.report_label)

    clean_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    with (args.output_dir / "per_task_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(clean_rows[0]))
        writer.writeheader()
        writer.writerows(clean_rows)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "# TabICLv2/group-drop pilot gates\n\n"
        f"Overall gate status: **{'PASS' if all_pass else 'FAIL'}**.\n\n"
        "The audit uses deterministic 400-column correlation views and matched Gaussian matrices, plus a random half-row test split. "
        "Feature selection is performed only inside training rows, ridge strength is selected on a nested validation split, and a paired permutation-y pipeline provides the signal null.\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
