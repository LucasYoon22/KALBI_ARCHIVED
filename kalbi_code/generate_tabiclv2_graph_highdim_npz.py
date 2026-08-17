#!/usr/bin/env python3
"""Generate high-dimensional NPZ regression batches from the TabICLv2 graph prior.

This uses the official TabICL repository's graph_scm prior for a base feature
block and expands large-p tasks with correlated noisy mixtures of that block.
The expansion keeps the existing high-dimensional Lasso training format
practical while avoiding the old hand-written MLP-SCM prior.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


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
    if np.random.random() < p_floor_weight:
        logp = np.random.uniform(lo, hi)
    else:
        while True:
            logp = np.random.normal(math.log10(p_mode), p_sigma)
            if lo <= logp <= hi:
                break
    p = int(np.clip(int(10**logp), p_min, p_max))

    n_hi = min(n_max, max_cells / p)
    if n_hi < n_min:
        raise ValueError(f"max_cells={max_cells} cannot fit n_min={n_min} at p={p}")
    n = int(math.exp(np.random.uniform(math.log(n_min), math.log(n_hi))))
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tabicl-src", type=Path, default=Path("/users/minsuk/fm-lm-lambda/external/tabicl/src"))
    p.add_argument("--num-batches", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-min", type=int, default=100)
    p.add_argument("--n-max", type=int, default=10_000)
    p.add_argument("--n-multiplier", type=float, default=2.0)
    p.add_argument("--p-min", type=int, default=3)
    p.add_argument("--p-max", type=int, default=100_000)
    p.add_argument("--max-cells", type=int, default=20_000_000)
    p.add_argument("--p-mode", type=int, default=16_000)
    p.add_argument("--p-sigma", type=float, default=0.45)
    p.add_argument("--p-floor-weight", type=float, default=0.02)
    p.add_argument("--target-batch-mb", type=float, default=256.0)
    p.add_argument("--base-max-features", type=int, default=512)
    p.add_argument("--expansion-noise-min", type=float, default=0.05)
    p.add_argument("--expansion-noise-max", type=float, default=0.5)
    p.add_argument("--pure-noise-frac", type=float, default=0.05)
    p.add_argument("--filter-unpredictable", action="store_true")
    p.add_argument("--resume-from", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clip-value", type=float, default=10.0)
    p.add_argument("--summarize-first", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gen = TabICLv2GraphGenerator(args.tabicl_src, args.base_max_features, args.filter_unpredictable, args.seed)
    entries = load_manifest_entries(args.out_dir / "manifest.json", args.resume_from)

    print("TabICLv2 graph_scm high-dimensional NPZ generation")
    print(f"  out_dir: {args.out_dir}")
    print(f"  tabicl_src: {args.tabicl_src}")
    print(f"  prior: official TabICLv2 graph_scm core")
    print(f"  highdim expansion: noisy correlated mixtures, base_max_features={args.base_max_features}")
    print(f"  filter_unpredictable: {args.filter_unpredictable}")
    print(f"  num_batches: {args.num_batches}, batch_size cap: {args.batch_size}")
    print(f"  p distribution: mode={args.p_mode}, sigma={args.p_sigma}, floor={args.p_floor_weight}")
    print(f"  seed: {args.seed}")
    start = time.time()

    for batch_idx in range(args.resume_from, args.num_batches):
        t0 = time.time()
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
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        ds: list[int] = []
        for _ in range(bs):
            xb, y, d_base = gen.generate_base(n, q)
            x = expand_highdim(
                xb,
                p,
                rng,
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
        path = args.out_dir / f"batch_{batch_idx:06d}.npz"
        np.savez(path, X=X, y=Y)
        entries.append(
            {
                "path": path.name,
                "num_tasks": int(bs),
                "n": int(n),
                "sampled_n": int(sampled_n),
                "n_multiplier": float(args.n_multiplier),
                "p": int(p),
                "base_features": int(q),
                "prior": "tabiclv2_graph_scm_highdim",
                "official_core_prior": "graph_scm",
                "highdim_expansion": "correlated_noisy_mixtures",
                "filter_unpredictable": bool(args.filter_unpredictable),
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


if __name__ == "__main__":
    main()
