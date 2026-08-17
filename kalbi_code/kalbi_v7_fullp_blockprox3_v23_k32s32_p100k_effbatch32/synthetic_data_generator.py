#!/usr/bin/env python3
"""Generate and correlation-repair the fixed N=200/P=400 V58 dataset.

Subcommands:
  generate  official TabICLv2 graph_scm batches (one output part directory)
  repair    deterministic constant/high-correlation Gaussian replacement

The generate defaults reproduce the small-experiment shape contract. Production
multi-part jobs call one process per part with a distinct stable seed, then run
the repair subcommand over the common root.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch

GENERATOR_VERSION = "v58_tabiclv2_graph_scm_n200_p400_v1"
REPAIR_VERSION = "v58_corrle095_gaussianfill_v1"


def sample_task_shape(
    n_min: int,
    n_max: int,
    p_min: int,
    p_max: int,
    max_cells: int,
    p_mode: int,
    p_sigma: float,
    p_floor_weight: float,
) -> tuple[int, int]:
    lo, hi = math.log10(p_min), math.log10(p_max)
    if p_min == p_max:
        p = int(p_min)
    elif np.random.random() < p_floor_weight:
        logp = np.random.uniform(lo, hi)
        p = int(np.clip(int(10**logp), p_min, p_max))
    else:
        while True:
            logp = np.random.normal(math.log10(p_mode), p_sigma)
            if lo <= logp <= hi:
                break
        p = int(np.clip(int(10**logp), p_min, p_max))

    n_hi = min(n_max, max_cells / p)
    if n_hi < n_min:
        raise ValueError(f"max_cells={max_cells} cannot fit n_min={n_min} at p={p}")
    n = (
        int(n_min)
        if n_min == n_max
        else int(math.exp(np.random.uniform(math.log(n_min), math.log(n_hi))))
    )
    return int(np.clip(n, n_min, n_max)), p

def standard_scale_np(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, keepdims=True)
    sd = np.where(sd > eps, sd, 1.0)
    y = (x - mu) / sd
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

def expand_highdim(
    x_base: np.ndarray,
    p: int,
    rng: np.random.Generator,
    noise_min: float,
    noise_max: float,
    pure_noise_frac: float,
) -> np.ndarray:
    n, q = x_base.shape
    x_base = standard_scale_np(x_base)
    if p <= q:
        return x_base[:, :p].astype(np.float32, copy=False)

    out = np.empty((n, p), dtype=np.float32)
    out[:, :q] = x_base
    j = q
    chunk = 256
    while j < p:
        c = min(chunk, p - j)
        pure = rng.random(c) < pure_noise_frac
        idx1 = rng.integers(0, q, size=c)
        idx2 = rng.integers(0, q, size=c)
        w1 = rng.normal(0.0, 1.0, size=c).astype(np.float32)
        w2 = rng.normal(0.0, 0.35, size=c).astype(np.float32)
        z = x_base[:, idx1] * w1[None, :] + x_base[:, idx2] * w2[None, :]
        # A few nonlinear warped copies preserve TabICLv2-like heterogeneous marginals.
        warp = rng.random(c) < 0.2
        if np.any(warp):
            z[:, warp] = np.tanh(z[:, warp])
        sigma = np.exp(rng.uniform(math.log(noise_min), math.log(noise_max), size=c)).astype(np.float32)
        z = z + rng.normal(0.0, 1.0, size=(n, c)).astype(np.float32) * sigma[None, :]
        if np.any(pure):
            z[:, pure] = rng.normal(0.0, 1.0, size=(n, int(pure.sum()))).astype(np.float32)
        out[:, j : j + c] = standard_scale_np(z)
        j += c
    perm = rng.permutation(p)
    return out[:, perm].astype(np.float32, copy=False)

class TabICLv2GraphGenerator:
    def __init__(
        self,
        tabicl_src: Path,
        base_max_features: int,
        filter_unpredictable: bool,
        seed: int,
    ) -> None:
        sys.path.insert(0, str(tabicl_src))
        from tabicl.prior._dataset import PriorDataset
        from tabicl.prior.graph_lib._config import PriorConfig

        self.PriorDataset = PriorDataset
        self.PriorConfig = PriorConfig
        self.base_max_features = int(base_max_features)
        self.filter_unpredictable = bool(filter_unpredictable)
        self.seed = int(seed)

    def generate_base(self, n: int, q: int) -> tuple[np.ndarray, np.ndarray, int]:
        config = self.PriorConfig(
            filter_unpredictable_datasets=self.filter_unpredictable,
            filter_unpredictable_graphs=self.filter_unpredictable,
        )
        ds = self.PriorDataset(
            regression=True,
            batch_size=1,
            batch_size_per_gp=1,
            min_features=q,
            max_features=q,
            min_seq_len=n,
            max_seq_len=n + 1,
            prior_type="graph_scm",
            config=config,
            n_jobs=1,
            device="cpu",
            seq_len_per_gp=False,
            log_n_features=False,
            log_seq_len=False,
            min_train_size=0.8,
            max_train_size=0.81,
        )
        x, y, d, _, _ = ds.get_batch(1)
        return (
            x[0].detach().cpu().numpy().astype(np.float32, copy=False),
            y[0].detach().cpu().numpy().astype(np.float32, copy=False),
            int(d[0].item()),
        )

def write_manifest(out_dir: Path, entries: list[dict[str, Any]]) -> None:
    entries = sorted(entries, key=lambda item: item["path"])
    manifest = {
        "num_datasets": int(sum(entry["num_tasks"] for entry in entries)),
        "files": entries,
    }
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

def load_manifest_entries(manifest_path: Path, resume_from: int) -> list[dict[str, Any]]:
    if resume_from <= 0 or not manifest_path.exists():
        return []
    with manifest_path.open("r") as f:
        old = json.load(f)
    return [
        item
        for item in old.get("files", [])
        if int(Path(item["path"]).stem.split("_")[-1]) < resume_from
    ]

def summarize(path: Path) -> None:
    z = np.load(path)
    print(f"Summary for {path}")
    for key in z.files:
        a = z[key].astype(np.float32)
        print(
            f"  {key}: shape={a.shape} dtype={z[key].dtype} "
            f"mean={float(a.mean()):.5g} std={float(a.std()):.5g} "
            f"min={float(a.min()):.5g} max={float(a.max()):.5g}"
        )

def parse_generate_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tabicl-src", type=Path, default=Path("/users/minsuk/fm-lm-lambda/external/tabicl/src"))
    p.add_argument("--num-batches", type=int, default=125)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-min", type=int, default=100)
    p.add_argument("--n-max", type=int, default=100)
    p.add_argument("--n-multiplier", type=float, default=2.0)
    p.add_argument("--p-min", type=int, default=400)
    p.add_argument("--p-max", type=int, default=400)
    p.add_argument("--max-cells", type=int, default=20_000_000)
    p.add_argument("--p-mode", type=int, default=400)
    p.add_argument("--p-sigma", type=float, default=0.45)
    p.add_argument("--p-floor-weight", type=float, default=0.02)
    p.add_argument("--target-batch-mb", type=float, default=256.0)
    p.add_argument("--base-max-features", type=int, default=512)
    p.add_argument("--expansion-noise-min", type=float, default=0.05)
    p.add_argument("--expansion-noise-max", type=float, default=0.5)
    p.add_argument("--pure-noise-frac", type=float, default=0.05)
    p.add_argument(
        "--filter-unpredictable",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--resume-from", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clip-value", type=float, default=10.0)
    p.add_argument("--summarize-first", action="store_true")
    return p.parse_args()

def generate_main() -> None:
    args = parse_generate_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gen = TabICLv2GraphGenerator(args.tabicl_src, args.base_max_features, args.filter_unpredictable, args.seed)
    entries = load_manifest_entries(args.out_dir / "manifest.json", args.resume_from)

    print("TabICLv2 graph_scm high-dimensional NPZ generation")
    print(f"  out_dir: {args.out_dir}")
    print(f"  tabicl_src: {args.tabicl_src}")
    print(f"  prior: official TabICLv2 graph_scm core")
    if args.p_max <= args.base_max_features:
        print("  feature path: official direct graph_scm block; no high-P expansion")
    else:
        print(
            "  feature path: official graph_scm core plus noisy high-P expansion, "
            f"base_max_features={args.base_max_features}"
        )
    print(f"  filter_unpredictable: {args.filter_unpredictable}")
    print(f"  num_batches: {args.num_batches}, batch_size cap: {args.batch_size}")
    print(f"  p distribution: mode={args.p_mode}, sigma={args.p_sigma}, floor={args.p_floor_weight}")
    print(f"  seed: {args.seed}")
    start = time.time()

    for batch_idx in range(args.resume_from, args.num_batches):
        t0 = time.time()
        # Stable batch identity makes resumed generation independent of the
        # number of earlier batches produced in the current process.
        batch_seed = int(
            np.random.SeedSequence([args.seed, batch_idx]).generate_state(1)[0]
        )
        np.random.seed(batch_seed)
        random.seed(batch_seed)
        torch.manual_seed(batch_seed)
        sampled_n, p = sample_task_shape(
            args.n_min,
            args.n_max,
            args.p_min,
            args.p_max,
            args.max_cells,
            args.p_mode,
            args.p_sigma,
            args.p_floor_weight,
        )
        n = max(1, int(round(sampled_n * args.n_multiplier)))
        target_cells = args.target_batch_mb * 1e6 / 2.0
        bs = int(np.clip(round(target_cells / (n * p)), 1, args.batch_size))
        q = int(min(p, args.base_max_features))
        direct_official_block = p <= q
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        ds: list[int] = []
        for task_index in range(bs):
            task_seed = int(
                np.random.SeedSequence(
                    [args.seed, batch_idx, task_index]
                ).generate_state(1)[0]
            )
            np.random.seed(task_seed)
            random.seed(task_seed)
            torch.manual_seed(task_seed)
            task_rng = np.random.default_rng(task_seed)
            xb, y, d_base = gen.generate_base(n, q)
            x = expand_highdim(
                xb,
                p,
                task_rng,
                args.expansion_noise_min,
                args.expansion_noise_max,
                args.pure_noise_frac,
            )
            y = standard_scale_np(y.reshape(-1, 1)).reshape(-1)
            if args.clip_value > 0:
                x = np.clip(x, -args.clip_value, args.clip_value)
                y = np.clip(y, -args.clip_value, args.clip_value)
            xs.append(x.astype(np.float16))
            ys.append(y.astype(np.float32))
            ds.append(min(int(d_base), p))

        X = np.stack(xs, axis=0)
        Y = np.stack(ys, axis=0)
        if X.shape != (32, 200, 400) or Y.shape != (32, 200):
            raise RuntimeError(
                f"V58 fixed-shape contract failed: X={X.shape}, y={Y.shape}"
            )
        path = args.out_dir / f"batch_{batch_idx:06d}.npz"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing batch: {path}")
        atomic_savez(path, x=X, y=Y)
        entries.append(
            {
                "path": path.name,
                "num_tasks": int(bs),
                "n": int(n),
                "sampled_n": int(sampled_n),
                "n_multiplier": float(args.n_multiplier),
                "p": int(p),
                "base_features": int(q),
                "prior": "tabiclv2_graph_scm",
                "official_core_prior": "graph_scm",
                "feature_path": (
                    "official_direct_graph_scm_block"
                    if direct_official_block
                    else "official_core_plus_correlated_noisy_mixtures"
                ),
                "highdim_expansion": (
                    "none" if direct_official_block else "correlated_noisy_mixtures"
                ),
                "filter_unpredictable": bool(args.filter_unpredictable),
                "generator_version": GENERATOR_VERSION,
                "batch_seed": batch_seed,
                "active_base_features_min": int(min(ds)),
                "active_base_features_max": int(max(ds)),
            }
        )
        write_manifest(args.out_dir, entries)
        elapsed = time.time() - t0
        done = batch_idx + 1
        remain = max(0, args.num_batches - done)
        eta_h = remain * elapsed / 3600.0
        print(
            f"[{done}/{args.num_batches}] saved {bs} tasks "
            f"(sampled_n={sampled_n}, n={n}, p={p}, base={q}), "
            f"time={elapsed:.1f}s, ETA={eta_h:.1f}h",
            flush=True,
        )

    print(f"Done. Total time: {(time.time() - start) / 3600.0:.2f}h")
    print(f"Saved NPZ data to {args.out_dir}")
    if args.summarize_first and (args.out_dir / "batch_000000.npz").exists():
        summarize(args.out_dir / "batch_000000.npz")

def parse_repair_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=88071102)
    parser.add_argument("--constant-eps", type=float, default=1e-6)
    parser.add_argument("--boundary-recheck-margin", type=float, default=5e-4)
    parser.add_argument("--max-repair-attempts", type=int, default=20)
    parser.add_argument("--parts", default="", help="Comma-separated part indices; empty means all")
    parser.add_argument("--max-files-per-part", type=int, default=0)
    parser.add_argument("--expected-n", type=int, default=0)
    parser.add_argument("--expected-p", type=int, default=0)
    return parser.parse_args()

def atomic_savez(path: Path, *, x: np.ndarray, y: np.ndarray) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        np.savez(handle, X=x, y=y)
    os.replace(temp, path)

def atomic_write_text(path: Path, text: str) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text)
    os.replace(temp, path)

def corr_batch(
    x: np.ndarray,
    device: torch.device,
    constant_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return absolute correlation matrices, valid masks, and off-diagonal maxima."""
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    centered = xt - xt.mean(dim=1, keepdim=True)
    norms = torch.linalg.vector_norm(centered, dim=1)
    valid = norms > constant_eps
    normalized = centered / norms.clamp_min(constant_eps).unsqueeze(1)
    normalized = normalized * valid.unsqueeze(1)
    corr = torch.bmm(normalized.transpose(1, 2), normalized).abs()
    corr.diagonal(dim1=1, dim2=2).zero_()
    maxima = corr.amax(dim=(1, 2))
    return (
        corr.cpu().numpy(),
        valid.cpu().numpy(),
        maxima.cpu().numpy().astype(np.float64),
    )

def abs_corr_f64(x: np.ndarray, constant_eps: float) -> tuple[np.ndarray, np.ndarray, float]:
    a = np.asarray(x, dtype=np.float64)
    centered = a - a.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    valid = norms > constant_eps
    normalized = np.zeros_like(centered)
    normalized[:, valid] = centered[:, valid] / norms[valid]
    corr = np.abs(normalized.T @ normalized)
    np.fill_diagonal(corr, 0.0)
    return corr, valid, float(corr.max(initial=0.0))

def greedy_replacement_mask(
    corr: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    initial_replace: np.ndarray | None = None,
) -> np.ndarray:
    """Greedily remove high-degree vertices until no forbidden edge remains."""
    replace = (~valid).copy() if initial_replace is None else initial_replace.copy()
    keep = valid & ~replace
    adjacency = (corr > threshold) & keep[:, None] & keep[None, :]
    np.fill_diagonal(adjacency, False)

    while True:
        degree = adjacency.sum(axis=1)
        max_degree = int(degree.max(initial=0))
        if max_degree == 0:
            break
        candidates = np.flatnonzero(degree == max_degree)
        weighted_degree = (corr * adjacency).sum(axis=1)
        best_weight = float(weighted_degree[candidates].max())
        tied = candidates[np.isclose(weighted_degree[candidates], best_weight, rtol=0.0, atol=1e-12)]
        # Stable tie-break: replace the largest original column index.
        chosen = int(tied[-1])
        replace[chosen] = True
        adjacency[chosen, :] = False
        adjacency[:, chosen] = False
    return replace

def gaussian_fill(
    x: np.ndarray,
    replace: np.ndarray,
    *,
    base_seed: int,
    part_index: int,
    file_index: int,
    task_index: int,
    attempt: int,
) -> np.ndarray:
    out = np.asarray(x, dtype=np.float16).copy()
    columns = np.flatnonzero(replace)
    if columns.size == 0:
        return out
    seed = np.random.SeedSequence(
        [base_seed, part_index, file_index, task_index, attempt]
    )
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((out.shape[0], columns.size), dtype=np.float32)
    out[:, columns] = g.astype(np.float16)
    return out

def repair_task_after_cast(
    source_x: np.ndarray,
    replace: np.ndarray,
    *,
    threshold: float,
    constant_eps: float,
    base_seed: int,
    part_index: int,
    file_index: int,
    task_index: int,
    max_attempts: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Resolve float64 boundary conflicts and the negligible Gaussian collision case."""
    current_replace = replace.copy()
    for attempt in range(max_attempts):
        out = gaussian_fill(
            source_x,
            current_replace,
            base_seed=base_seed,
            part_index=part_index,
            file_index=file_index,
            task_index=task_index,
            attempt=attempt,
        )
        corr, valid, maximum = abs_corr_f64(out, constant_eps)
        if maximum <= threshold:
            if not bool(valid.all()):
                raise RuntimeError("Gaussian fill unexpectedly produced a constant column")
            return out, current_replace, maximum, attempt + 1

        original = ~current_replace
        original_conflict = (
            (corr > threshold)
            & original[:, None]
            & original[None, :]
        )
        np.fill_diagonal(original_conflict, False)
        if bool(original_conflict.any()):
            # Incorporate any boundary edges missed by float32 screening.
            current_replace = greedy_replacement_mask(
                corr,
                valid=np.ones(corr.shape[0], dtype=bool),
                threshold=threshold,
                initial_replace=current_replace,
            )
        # Otherwise resampling all Gaussian columns on the next attempt is enough.
    raise RuntimeError(
        f"failed to obtain max |r| <= {threshold} after {max_attempts} attempts "
        f"for part={part_index} file={file_index} task={task_index}"
    )

def repair_main() -> None:
    args = parse_repair_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output root: {output_root}")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")
    if args.boundary_recheck_margin <= 0:
        raise ValueError("boundary recheck margin must be positive")
    if (args.expected_n > 0) != (args.expected_p > 0):
        raise ValueError("--expected-n and --expected-p must be provided together")
    expected_shape: tuple[int, int] | None = None
    if args.expected_n > 0:
        expected_shape = (args.expected_n, args.expected_p)

    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    selected_parts = None
    if args.parts.strip():
        selected_parts = {int(item) for item in args.parts.split(",")}
    part_dirs = sorted(
        (
            path
            for path in source_root.glob("part_*")
            if path.is_dir()
            and (selected_parts is None or int(path.name.split("_")[-1]) in selected_parts)
        ),
        key=lambda path: int(path.name.split("_")[-1]),
    )
    if not part_dirs:
        raise FileNotFoundError("no selected part_* directories")

    started = time.time()
    audit_path = output_root / "task_transform_audit.jsonl"
    audit_handle = audit_path.open("a")
    total_tasks = 0
    total_replaced = 0
    total_constants = 0
    total_boundary_rechecks = 0
    all_replacement_counts: list[int] = []
    all_before_maxima: list[float] = []
    all_after_maxima: list[float] = []
    part_summaries: list[dict] = []

    try:
        for part_dir in part_dirs:
            part_index = int(part_dir.name.split("_")[-1])
            manifest = json.loads((part_dir / "manifest.json").read_text())
            entries = manifest["files"]
            if args.max_files_per_part > 0:
                entries = entries[: args.max_files_per_part]
            out_part = output_root / part_dir.name
            out_part.mkdir(parents=True, exist_ok=True)
            new_entries: list[dict] = []
            part_tasks = 0
            part_replaced = 0
            part_constants = 0
            part_after_max = 0.0

            for file_index, entry in enumerate(entries):
                source_file = part_dir / entry["path"]
                with np.load(source_file) as data:
                    x = np.asarray(data["X"])
                    y = np.asarray(data["y"])
                if x.ndim != 3:
                    raise RuntimeError(f"unexpected X shape {x.shape} in {source_file}")
                current_shape = (int(x.shape[1]), int(x.shape[2]))
                if expected_shape is None:
                    expected_shape = current_shape
                if current_shape != expected_shape:
                    raise RuntimeError(
                        f"expected X task shape {expected_shape}, got {current_shape} in {source_file}"
                    )
                if x.dtype != np.float16:
                    raise RuntimeError(f"expected float16 X, got {x.dtype} in {source_file}")
                if y.shape != (x.shape[0], x.shape[1]) or y.dtype != np.float32:
                    raise RuntimeError(f"unexpected y {y.shape}/{y.dtype} in {source_file}")

                corr, valid, before_maxima = corr_batch(x, device, args.constant_eps)
                output_x = np.empty_like(x)
                replacement_counts: list[int] = []
                constant_counts: list[int] = []
                masks: list[np.ndarray] = []

                for task_index in range(x.shape[0]):
                    replace = greedy_replacement_mask(
                        corr[task_index],
                        valid[task_index],
                        args.threshold,
                    )
                    masks.append(replace)
                    replacement_counts.append(int(replace.sum()))
                    constant_counts.append(int((~valid[task_index]).sum()))
                    output_x[task_index] = gaussian_fill(
                        x[task_index],
                        replace,
                        base_seed=args.seed,
                        part_index=part_index,
                        file_index=file_index,
                        task_index=task_index,
                        attempt=0,
                    )

                _, _, after_maxima = corr_batch(output_x, device, args.constant_eps)
                boundary = np.flatnonzero(
                    after_maxima >= args.threshold - args.boundary_recheck_margin
                )
                for task_index in boundary:
                    repaired, new_mask, exact_max, attempts = repair_task_after_cast(
                        x[task_index],
                        masks[task_index],
                        threshold=args.threshold,
                        constant_eps=args.constant_eps,
                        base_seed=args.seed,
                        part_index=part_index,
                        file_index=file_index,
                        task_index=int(task_index),
                        max_attempts=args.max_repair_attempts,
                    )
                    output_x[task_index] = repaired
                    masks[task_index] = new_mask
                    replacement_counts[task_index] = int(new_mask.sum())
                    after_maxima[task_index] = exact_max
                    total_boundary_rechecks += 1

                # Final batched guard, followed by float64 checks near the boundary.
                _, valid_after, final_maxima = corr_batch(output_x, device, args.constant_eps)
                if not bool(valid_after.all()):
                    raise RuntimeError(f"constant output feature found in {source_file}")
                final_boundary = np.flatnonzero(
                    final_maxima >= args.threshold - args.boundary_recheck_margin
                )
                for task_index in final_boundary:
                    _, valid_exact, exact_max = abs_corr_f64(
                        output_x[task_index], args.constant_eps
                    )
                    if not bool(valid_exact.all()) or exact_max > args.threshold:
                        raise RuntimeError(
                            f"final threshold invariant failed: {source_file} task={task_index} "
                            f"max={exact_max}"
                        )
                    final_maxima[task_index] = exact_max

                output_file = out_part / entry["path"]
                atomic_savez(output_file, x=output_x, y=y)
                with np.load(output_file) as saved:
                    if saved["X"].shape != x.shape or saved["X"].dtype != np.float16:
                        raise RuntimeError(f"saved X invariant failed in {output_file}")
                    if not np.array_equal(saved["y"], y):
                        raise RuntimeError(f"y changed in {output_file}")

                new_entry = dict(entry)
                new_entry["correlation_repair"] = {
                    "metric": "max_abs_offdiagonal_pearson_feature_correlation",
                    "threshold": float(args.threshold),
                    "strategy": "greedy_high_degree_feature_replacement",
                    "replacement_distribution": "independent_standard_gaussian",
                    "seed": int(args.seed),
                    "replaced_features_total": int(sum(replacement_counts)),
                    "constant_features_replaced": int(sum(constant_counts)),
                    "max_before": float(np.max(before_maxima)),
                    "max_after": float(np.max(final_maxima)),
                }
                new_entries.append(new_entry)

                audit_handle.write(
                    json.dumps(
                        {
                            "part": part_dir.name,
                            "source_path": str(source_file),
                            "output_path": str(output_file),
                            "num_tasks": int(x.shape[0]),
                            "replacement_counts": replacement_counts,
                            "constant_feature_counts": constant_counts,
                            "max_before": before_maxima.tolist(),
                            "max_after": final_maxima.tolist(),
                        }
                    )
                    + "\n"
                )
                audit_handle.flush()

                file_replaced = int(sum(replacement_counts))
                file_constants = int(sum(constant_counts))
                part_tasks += int(x.shape[0])
                part_replaced += file_replaced
                part_constants += file_constants
                part_after_max = max(part_after_max, float(np.max(final_maxima)))
                total_tasks += int(x.shape[0])
                total_replaced += file_replaced
                total_constants += file_constants
                all_replacement_counts.extend(replacement_counts)
                all_before_maxima.extend(before_maxima.tolist())
                all_after_maxima.extend(final_maxima.tolist())

                if (file_index + 1) % 25 == 0 or file_index + 1 == len(entries):
                    print(
                        json.dumps(
                            {
                                "part": part_dir.name,
                                "files_done": file_index + 1,
                                "files_total": len(entries),
                                "tasks": part_tasks,
                                "mean_replaced_features": part_replaced / max(1, part_tasks),
                                "max_after": part_after_max,
                                "elapsed_sec": round(time.time() - started, 2),
                            }
                        ),
                        flush=True,
                    )

            transformed_manifest = dict(manifest)
            transformed_manifest["files"] = new_entries
            transformed_manifest["num_datasets"] = int(sum(x["num_tasks"] for x in new_entries))
            transformed_manifest["feature_correlation_repair"] = {
                "source_root": str(source_root),
                "metric": "max_abs_offdiagonal_pearson_feature_correlation",
                "threshold": float(args.threshold),
                "strategy": "greedy_high_degree_feature_replacement",
                "replacement_distribution": "independent_standard_gaussian",
                "seed": int(args.seed),
                "constant_features": "replaced_with_gaussian",
                "response": "copied_verbatim",
            }
            atomic_write_text(
                out_part / "manifest.json",
                json.dumps(transformed_manifest, indent=2) + "\n",
            )
            part_summary = {
                "part": part_dir.name,
                "files": len(new_entries),
                "tasks": part_tasks,
                "replaced_features_total": part_replaced,
                "constant_features_replaced": part_constants,
                "mean_replaced_features_per_task": part_replaced / max(1, part_tasks),
                "max_after": part_after_max,
            }
            part_summaries.append(part_summary)
            print(json.dumps(part_summary), flush=True)
    finally:
        audit_handle.close()

    replacement_array = np.asarray(all_replacement_counts, dtype=np.int64)
    before_array = np.asarray(all_before_maxima, dtype=np.float64)
    after_array = np.asarray(all_after_maxima, dtype=np.float64)
    if expected_shape is None:
        raise RuntimeError("no dataset files were processed")
    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "source_preserved": True,
        "tasks": total_tasks,
        "expected_shape_per_task": list(expected_shape),
        "x_dtype": "float16",
        "y_dtype": "float32",
        "threshold": float(args.threshold),
        "seed": int(args.seed),
        "strategy": "greedy_high_degree_feature_replacement",
        "replacement_distribution": "independent_standard_gaussian",
        "replaced_features_total": total_replaced,
        "constant_features_replaced": total_constants,
        "replacement_count_quantiles": {
            str(q): float(np.quantile(replacement_array, q))
            for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
        },
        "max_before_quantiles": {
            str(q): float(np.quantile(before_array, q))
            for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
        },
        "max_after_quantiles": {
            str(q): float(np.quantile(after_array, q))
            for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
        },
        "boundary_rechecks_float64": total_boundary_rechecks,
        "parts": part_summaries,
        "elapsed_sec": time.time() - started,
    }
    if float(after_array.max(initial=0.0)) > args.threshold:
        raise RuntimeError("global threshold invariant failed")
    atomic_write_text(
        output_root / "transform_summary.json",
        json.dumps(summary, indent=2) + "\n",
    )
    atomic_write_text(output_root / "TRANSFORM_COMPLETE.marker", "complete\n")
    print(json.dumps(summary, indent=2), flush=True)

def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"generate", "repair"}:
        raise SystemExit("usage: synthetic_data_generator.py {generate|repair} [arguments]")
    command = sys.argv.pop(1)
    generate_main() if command == "generate" else repair_main()

if __name__ == "__main__":
    main()
