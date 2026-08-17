#!/usr/bin/env python3
"""Standalone V58-Chebyshev beta-solver trainer for N=200/P=400 data.

This is the cleaned production trainer for the adjacent model.py and utils.py.
It preserves the validated V57 topology, data, and objective contracts while
importing no versioned model module. The public penalty input is native196:
96 interleaved rho/drho pairs plus meta4. The serialized penalty cache stores
continuous segments and never stores a V58 convexity coordinate.

V58 keeps only the discrete support decision detached.  All continuous local
ISIS computations are differentiable, so selector lambda/native196 gradients
track the fixed-support hard forward.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


SOURCE_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = SOURCE_DIR.parent
PROJECT = Path(os.environ.get("FM_PROJECT", str(BUNDLE_ROOT))).resolve()
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(BUNDLE_ROOT))

import model_fpk
from penalty_sampler_v4 import patch_model_fpk_sampler

patch_model_fpk_sampler()

from model_fpk import PenaltySampler, PenaltySamplerConfig
from penalty_relu_mlp_pwl_v33 import (
    batch_segments_to_relu_params,
    compute_penalty_relu_cached,
    make_penalty_features_pwl,
)
from train_fpk import _build_X_y_tensors, _compute_lambda_max, _load_yaml

model_fpk.make_penalty_features_from_segments = make_penalty_features_pwl

from model_v58_chebyshev.model import (
    LOCAL_ISIS_GRADIENT_CONTRACT,
    V58FAST_EXPANDED_DIM,
    V58FAST_FEATURE_FILM_DIM,
    V58FAST_META_DIM,
    V58FAST_MODEL_VERSION,
    V58FAST_NATIVE_DIM,
    V58FAST_PENALTY_CONTRACT_VERSION,
    V58FAST_PROX_COND_DIM,
    build_model,
    native196_to_expanded388,
    strip_convexity_flag,
)


NATIVE_GRID_SIZE = 96
SOURCE_NATIVE197_DIM = 2 * NATIVE_GRID_SIZE + 5
TRAINER_SEMANTIC_VERSION = (
    "v58cheb_fixed_support_selector_gradient_lambda_floor_1e-6_v1"
)
CHECKPOINT_VERSION = "v58cheb_standalone_exact_resume_v1"
EXPECTED_WORLD_SIZE = 8
TASKS_PER_RANK_UPDATE = 4
DEFAULT_BATCH_PER_RANK = 4
DEFAULT_GRAD_ACCUM_STEPS = 1
GLOBAL_EFFECTIVE_BATCH = EXPECTED_WORLD_SIZE * TASKS_PER_RANK_UPDATE
LAMBDA_LOW_MIN = 1e-6
OBJECTIVE_CONTRACT = "mean_log1p_Fhat_over_Fzero_v1"
KKT_CONTRACT = "detached_top32_kkt_relu_every_update_v1"
EXECUTION_CONTRACT = "batched_tasks_per_rank4_v1"

TRAIN_PATH = SOURCE_DIR / "train.py"
MODEL_PATH = SOURCE_DIR / "model.py"
UTILS_PATH = SOURCE_DIR / "utils.py"
PENALTY_GENERATOR_PATH = SOURCE_DIR / "penalty_generator.py"
SYNTHETIC_DATA_GENERATOR_PATH = SOURCE_DIR / "synthetic_data_generator.py"
ROOT_DEPENDENCY_PATHS = {
    "model_fpk_sha256": PROJECT / "model_fpk.py",
    "penalty_sampler_v4_sha256": PROJECT / "penalty_sampler_v4.py",
    "penalty_sampler_v9_sha256": PROJECT / "penalty_sampler_v9.py",
    "penalty_relu_mlp_pwl_v33_sha256": PROJECT / "penalty_relu_mlp_pwl_v33.py",
    "train_fpk_sha256": PROJECT / "train_fpk.py",
    "shape_feature_normalization_sha256": PROJECT / "shape_feature_normalization.py",
}
EXPECTED_MODULE_PATHS = {
    "model_fpk": PROJECT / "model_fpk.py",
    "penalty_sampler_v4": PROJECT / "penalty_sampler_v4.py",
    "penalty_relu_mlp_pwl_v33": PROJECT / "penalty_relu_mlp_pwl_v33.py",
    "train_fpk": PROJECT / "train_fpk.py",
    "shape_feature_normalization": PROJECT / "shape_feature_normalization.py",
    "model_v58_chebyshev.model": MODEL_PATH,
    "model_v58_chebyshev.utils": UTILS_PATH,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def loaded_project_module_closure() -> dict[str, dict[str, str]]:
    """Hash every already-imported Python source below the FM project root."""
    closure: dict[str, dict[str, str]] = {}
    for module_name, module in sorted(sys.modules.items()):
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            continue
        path = Path(raw_path).resolve()
        try:
            relative = path.relative_to(PROJECT)
        except ValueError:
            continue
        if path.suffix != ".py" or not path.is_file():
            continue
        closure[module_name] = {
            "path": str(path),
            "relative_to_project": str(relative),
            "sha256": file_sha256(path),
        }
    if not closure:
        raise TrainerError("loaded project-module closure is unexpectedly empty")
    return closure


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().contiguous().cpu()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_part(data_root: Path, part_name: str, threshold: float) -> dict[str, Any]:
    """Validate an explicit manifest-backed part and return an immutable inventory."""
    root = data_root.resolve()
    part = (root / part_name).resolve()
    accepted_markers = [
        root / "TRANSFORM_COMPLETE.marker",
        root / "TRANSFER_ABORTED_FOR_IMMEDIATE_4GPU_LAUNCH.marker",
    ]
    marker = next((candidate for candidate in accepted_markers if candidate.exists()), None)
    summary_path = root / "transform_summary.json"
    manifest_path = part / "manifest.json"
    if marker is None:
        raise RuntimeError(
            "dataset is missing both the complete-transform and legacy partial-transfer markers"
        )
    if not summary_path.exists() or not manifest_path.exists():
        raise RuntimeError(f"missing transform summary or manifest for explicit part: {part}")

    summary = json.loads(summary_path.read_text())
    if list(summary.get("expected_shape_per_task", [])) != [200, 400]:
        raise RuntimeError(f"summary shape contract mismatch: {summary_path}")
    if summary.get("replacement_distribution") != "independent_standard_gaussian":
        raise RuntimeError(f"summary replacement distribution mismatch: {summary_path}")
    if float(summary.get("max_after_global", float("inf"))) >= float(threshold):
        raise RuntimeError(f"summary correlation threshold violation: {summary_path}")

    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    entries = manifest.get("files", [])
    if not entries:
        raise RuntimeError(f"empty manifest: {manifest_path}")
    listed_names = [str(entry["path"]) for entry in entries]
    if len(listed_names) != len(set(listed_names)):
        raise RuntimeError(f"duplicate paths in manifest: {manifest_path}")
    actual_names = sorted(path.name for path in part.glob("*.npz"))
    if sorted(listed_names) != actual_names:
        missing = sorted(set(listed_names) - set(actual_names))
        extra = sorted(set(actual_names) - set(listed_names))
        raise RuntimeError(
            f"part inventory mismatch: missing={missing[:8]} extra={extra[:8]} "
            f"listed={len(listed_names)} actual={len(actual_names)}"
        )

    inventory_digest = hashlib.sha256()
    inventory_digest.update(manifest_bytes)
    files: list[str] = []
    total_bytes = 0
    total_tasks = 0
    for entry in entries:
        path = part / str(entry["path"])
        if int(entry.get("n", -1)) != 200 or int(entry.get("p", -1)) != 400:
            raise RuntimeError(f"manifest shape mismatch: {path}")
        if int(entry.get("num_tasks", -1)) != 32:
            raise RuntimeError(f"manifest task-count mismatch: {path}")
        repair = entry.get("correlation_repair", {})
        if repair.get("replacement_distribution") != "independent_standard_gaussian":
            raise RuntimeError(f"uncertified replacement distribution: {path}")
        if float(repair.get("threshold", float("nan"))) != float(threshold):
            raise RuntimeError(f"correlation threshold metadata mismatch: {path}")
        if float(repair.get("max_after", float("inf"))) >= float(threshold):
            raise RuntimeError(f"correlation threshold exceeded: {path}")
        content_sha = file_sha256(path)
        try:
            with np.load(path, allow_pickle=False) as data:
                if sorted(data.files) != ["X", "y"]:
                    raise RuntimeError(f"unexpected keys {data.files}")
                X = data["X"]
                y = data["y"]
                if X.shape != (32, 200, 400) or y.shape != (32, 200):
                    raise RuntimeError(f"unexpected shapes X={X.shape}, y={y.shape}")
                if X.dtype != np.float16 or y.dtype != np.float32:
                    raise RuntimeError(f"unexpected dtypes X={X.dtype}, y={y.dtype}")
                if not np.isfinite(X).all() or not np.isfinite(y).all():
                    raise RuntimeError("non-finite payload")
        except Exception as exc:
            raise RuntimeError(f"invalid NPZ {path}: {exc!r}") from exc
        size = path.stat().st_size
        total_bytes += size
        total_tasks += 32
        files.append(str(path))
        inventory_digest.update(path.name.encode("utf-8"))
        inventory_digest.update(str(size).encode("ascii"))
        inventory_digest.update(content_sha.encode("ascii"))

    return {
        "data_root": str(root),
        "part": part_name,
        "files": files,
        "file_count": len(files),
        "task_count": total_tasks,
        "bytes": total_bytes,
        "inventory_sha256": inventory_digest.hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "payload_sha256_bound_to_manifest": True,
        "correlation_certificate": "manifest_max_after_strictly_lt_threshold_payload_sha256_pinned",
        "dataset_marker": marker.name,
        "partial_transfer_explicit": marker.name.startswith("TRANSFER_ABORTED"),
    }


class PartStream:
    """Exact-resumable task stream over one immutable manifest-backed part."""

    VERSION = "v58_small_part_stream_batched_exact_v1"

    def __init__(
        self,
        files: list[str],
        *,
        seed: int,
        rank: int,
        world: int,
        cache_files: int = 2,
        state: Mapping[str, Any] | None = None,
    ) -> None:
        rank_files = list(files[rank::world])
        if not rank_files:
            raise RuntimeError(f"no training files for rank {rank}/{world}")
        self.files = [str(Path(path).resolve()) for path in rank_files]
        self.rank = int(rank)
        self.world = int(world)
        self.seed = int(seed)
        self.cache_files = max(1, int(cache_files))
        self.references = [
            (file_index, task_index)
            for file_index in range(len(self.files))
            for task_index in range(32)
        ]
        self.rng = np.random.default_rng(self.seed + 1_000_003 * self.rank)
        self.order = np.arange(len(self.references), dtype=np.int64)
        self.rng.shuffle(self.order)
        self.cursor = 0
        self.cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.cache_order: list[int] = []
        if state is not None:
            self.load_state_dict(state)

    def _load_file(self, file_index: int) -> tuple[np.ndarray, np.ndarray]:
        if file_index in self.cache:
            self.cache_order.remove(file_index)
            self.cache_order.append(file_index)
            return self.cache[file_index]
        with np.load(self.files[file_index], allow_pickle=False) as data:
            X = np.asarray(data["X"], dtype=np.float32)
            y = np.asarray(data["y"], dtype=np.float32)
        if X.shape != (32, 200, 400) or y.shape != (32, 200):
            raise RuntimeError(
                f"runtime task file changed: {self.files[file_index]} X={X.shape} y={y.shape}"
            )
        value = (X, y)
        self.cache[file_index] = value
        self.cache_order.append(file_index)
        while len(self.cache_order) > self.cache_files:
            victim = self.cache_order.pop(0)
            self.cache.pop(victim, None)
        return value

    def next_task(self) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if self.cursor >= len(self.order):
            self.rng.shuffle(self.order)
            self.cursor = 0
        order_index = int(self.order[self.cursor])
        self.cursor += 1
        file_index, task_index = self.references[order_index]
        X, y = self._load_file(file_index)
        return (
            np.asarray(X[task_index], dtype=np.float32),
            np.asarray(y[task_index], dtype=np.float32),
            {
                "file": self.files[file_index],
                "file_index": file_index,
                "task_index": task_index,
                "n": 200,
                "p": 400,
            },
        )

    def next_batch(
        self,
        batch_size: int = 1,
        *,
        return_info: bool = False,
    ) -> tuple[list[np.ndarray], list[np.ndarray]] | tuple[
        list[np.ndarray], list[np.ndarray], list[dict[str, Any]]
    ]:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        tasks = [self.next_task() for _ in range(int(batch_size))]
        Xs = [task[0] for task in tasks]
        ys = [task[1] for task in tasks]
        infos = [task[2] for task in tasks]
        return (Xs, ys, infos) if return_info else (Xs, ys)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "files": list(self.files),
            "rank": self.rank,
            "world": self.world,
            "seed": self.seed,
            "order": self.order.tolist(),
            "cursor": self.cursor,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "version": self.VERSION,
            "files": self.files,
            "rank": self.rank,
            "world": self.world,
            "seed": self.seed,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise RuntimeError(f"stream resume mismatch for {key}")
        order = np.asarray(state["order"], dtype=np.int64)
        if order.shape != (len(self.references),) or not np.array_equal(
            np.sort(order), np.arange(len(self.references), dtype=np.int64)
        ):
            raise RuntimeError("stream resume order is not a permutation")
        cursor = int(state["cursor"])
        if cursor < 0 or cursor > len(order):
            raise RuntimeError("stream resume cursor is invalid")
        self.order = order
        self.cursor = cursor
        self.rng.bit_generator.state = state["rng_state"]


class PenaltyCache:
    """Load continuous segments and expose only V58 native196/meta4 features."""

    def __init__(self, path: Path, seed: int):
        self.path = path
        self.rng = np.random.default_rng(seed)

    def load(self) -> None:
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        self.segments = payload["segments"]
        self.pool_size = len(self.segments)
        encoded = model_fpk.make_penalty_features_from_segments(
            [[segments] for segments in self.segments]
        )
        native197 = encoded["shape_features"].squeeze(1).contiguous()
        self.d0 = encoded["d0"].squeeze(1).contiguous()
        self.relu_knots, self.relu_b, self.relu_weights = batch_segments_to_relu_params(
            self.segments, K=96, t_max=6.0, device="cpu"
        )
        if native197.shape != (self.pool_size, SOURCE_NATIVE197_DIM):
            raise RuntimeError(f"source PWL197 cache shape mismatch: {native197.shape}")
        self.removed_convexity_flag = native197[:, -1].clone()
        valid_flags = (self.removed_convexity_flag == 1) | (
            self.removed_convexity_flag == -1
        )
        if not bool(valid_flags.all()):
            raise RuntimeError("source convexity metadata is not encoded as +/-1")
        self.shape_features = strip_convexity_flag(native197).contiguous()
        if self.shape_features.shape != (self.pool_size, V58FAST_NATIVE_DIM):
            raise RuntimeError(f"V58 native196 cache shape mismatch: {self.shape_features.shape}")
        if not torch.equal(self.shape_features, native197[:, :-1]):
            raise RuntimeError("native196 conversion changed more than the final flag")
        if not torch.isfinite(self.shape_features).all() or not torch.isfinite(self.d0).all():
            raise RuntimeError("non-finite native penalty cache features")
        drho0_error = (self.shape_features[:, 1] - self.d0).abs().max().item()
        metadata_d0_error = (
            self.shape_features[:, 2 * NATIVE_GRID_SIZE] - self.d0
        ).abs().max().item()
        if drho0_error > 1e-6 or metadata_d0_error > 1e-6:
            raise RuntimeError(
                "native interleaved penalty layout check failed: "
                f"drho0_error={drho0_error}, metadata_d0_error={metadata_d0_error}"
            )
        self.native_layout_max_error = max(drho0_error, metadata_d0_error)

    def state_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path.resolve()),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("path") != str(self.path.resolve()):
            raise RuntimeError("penalty-cache path changed on resume")
        self.rng.bit_generator.state = state["rng_state"]


def standardize_tasks(
    Xs: list[np.ndarray], ys: list[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    X = np.stack(Xs).astype(np.float32, copy=False)
    y = np.stack(ys).astype(np.float32, copy=False)
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    y = (y - y.mean(axis=1, keepdims=True)) / (y.std(axis=1, keepdims=True) + 1e-8)
    return list(X), list(y)


def refit_normalizers(
    model: torch.nn.Module,
    stream: PartStream,
    sampler,
    cache: PenaltyCache,
    device: torch.device,
) -> None:
    """Refit all native196/expanded388 buffers using the V55 sampling policy."""
    with torch.no_grad():
        eps = 1e-12
        lambdas = sampler._sample_lambdas(1024, 8).reshape(-1).to(device).float()
        lambda_max_values = []
        # Match Exp125's calibration sample count.  This static small-data
        # adapter has one prior, so the 16 slots all draw from that immutable
        # stream instead of alternating two rolling priors.
        for _ in range(16):
            Xs, ys = standardize_tasks(*stream.next_batch())
            lambda_max_values.append(
                _compute_lambda_max(Xs, ys).reshape(-1).to(device).float()
            )
        lambda_max_pool = torch.cat(lambda_max_values).clamp_min(eps)
        d0_pool = cache.d0.to(device).float()
        count = lambdas.shape[0]
        lambda_max = lambda_max_pool[
            torch.randint(0, lambda_max_pool.shape[0], (count,), device=device)
        ]
        d0 = d0_pool[torch.randint(0, d0_pool.shape[0], (count,), device=device)]
        ratio = (lambdas / lambda_max.clamp_min(eps)).clamp_min(eps)
        threshold_ratio = (
            lambdas * d0.clamp_min(eps) / lambda_max.clamp_min(eps)
        ).clamp_min(eps)
        lambda_features = torch.stack(
            [
                ratio.log().clamp(-8.0, 1.0) / 4.0,
                ratio.clamp(0.0, 10.0).sqrt() / math.sqrt(10.0),
                torch.tanh(lambda_max.clamp_min(eps).log() / 5.0),
                threshold_ratio.log().clamp(-8.0, 1.0) / 4.0,
                torch.tanh(d0 / 2.0),
                (lambdas + 1e-7).log(),
            ],
            dim=-1,
        )
        model.cond_embed._lambda_mean.copy_(
            lambda_features.mean(0).to(model.cond_embed._lambda_mean.dtype)
        )
        model.cond_embed._lambda_scale.copy_(
            lambda_features.std(0).clamp_min(1e-6).to(
                model.cond_embed._lambda_scale.dtype
            )
        )
        shape_pool = cache.shape_features.to(device).float()
        model.cond_embed._shape_mean.copy_(
            shape_pool.mean(0).to(model.cond_embed._shape_mean.dtype)
        )
        model.cond_embed._shape_scale.copy_(
            shape_pool.std(0).clamp_min(1e-6).to(model.cond_embed._shape_scale.dtype)
        )
        # Exp125 deterministically spans the full penalty pool for the shape
        # normalizer; random subset selection would change the model contract.
        sample_count = min(4096, cache.pool_size)
        indices = torch.linspace(
            0, cache.pool_size - 1, sample_count, device=device
        ).round().long()
        shape = shape_pool[indices]
        lambda_raw = sampler._sample_lambdas(sample_count, 1).view(
            sample_count
        ).to(device).float()
        expanded = native196_to_expanded388(shape, lambda_raw)
        model._shape_mean_388.copy_(
            expanded.mean(0).to(model._shape_mean_388.dtype)
        )
        model._shape_scale_388.copy_(
            expanded.std(0).clamp_min(1e-6).to(model._shape_scale_388.dtype)
        )


def make_query_payload(
    Xs: list[np.ndarray],
    ys: list[np.ndarray],
    cache: PenaltyCache,
    sampler,
    device: torch.device,
    fixed_penalty_indices: np.ndarray | None = None,
    fixed_lambdas: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """V55 query sampler with the public penalty tensor locked to 196-D."""
    Xs, ys = standardize_tasks(Xs, ys)
    batch_size = len(Xs)
    n_penalty, n_lambda = 4, 8
    query_count = n_penalty * n_lambda
    if fixed_penalty_indices is None:
        penalty_indices = cache.rng.integers(
            0, cache.pool_size, size=batch_size * n_penalty
        )
    else:
        penalty_indices = np.asarray(fixed_penalty_indices, dtype=np.int64)
        if penalty_indices.shape != (batch_size * n_penalty,):
            raise RuntimeError(f"fixed penalty index shape mismatch: {penalty_indices.shape}")
    full_indices = np.repeat(
        penalty_indices.reshape(batch_size, n_penalty), n_lambda, axis=1
    ).reshape(-1)
    shape = cache.shape_features[full_indices].view(
        batch_size, query_count, V58FAST_NATIVE_DIM
    ).to(device)
    d0 = cache.d0[full_indices].view(batch_size, query_count).to(device)
    relu_b = cache.relu_b[full_indices].to(device)
    relu_weights = cache.relu_weights[full_indices].to(device)
    if fixed_lambdas is None:
        lambda8 = sampler._sample_lambdas(batch_size, n_lambda)
    else:
        lambda8 = fixed_lambdas.clone()
        if lambda8.shape != (batch_size, n_lambda):
            raise RuntimeError(f"fixed lambda shape mismatch: {lambda8.shape}")
    lambdas = (
        lambda8.unsqueeze(1)
        .expand(batch_size, n_penalty, n_lambda)
        .reshape(batch_size, query_count)
        .to(device)
        .float()
    )
    lambda_max = _compute_lambda_max(Xs, ys).to(device).float().view(
        batch_size
    ).clamp_min(1e-12)
    lambda_max = lambda_max.unsqueeze(1).expand(batch_size, query_count)
    X, y = _build_X_y_tensors(
        Xs, ys, device=device, dtype=torch.float32
    )
    if shape.shape[-1] != V58FAST_NATIVE_DIM:
        raise RuntimeError("V58 beta solver did not receive native196")
    return {
        "X": X,
        "y": y,
        "lambdas": lambdas,
        "lambda_max": lambda_max,
        "shape": shape,
        "d0": d0,
        "relu_b": relu_b,
        "relu_weights": relu_weights,
    }


def module_grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().item())
    return math.sqrt(total)


class TrainerError(RuntimeError):
    """Fail-closed error for the Exp125-derived V58 training contract."""


def make_sampler(config: Mapping[str, Any], seed: int) -> PenaltySampler:
    raw = dict(config.get("sampler", {}))
    raw["lambda_low_min"] = LAMBDA_LOW_MIN
    fields = {field.name for field in dataclasses.fields(PenaltySamplerConfig)}
    sampler = PenaltySampler(
        PenaltySamplerConfig(**{key: value for key, value in raw.items() if key in fields}),
        seed=int(seed),
    )
    if float(sampler.cfg.lambda_low_min) != LAMBDA_LOW_MIN:
        raise TrainerError("V58 lambda floor override was not applied")
    return sampler


def capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
    }


def restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"])


def nested_state_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().contiguous().cpu()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii") + b"\0")
            digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(
                item, key=lambda candidate: (type(candidate).__name__, repr(candidate))
            ):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode("ascii") + b"\0")
            digest.update(str(len(item)).encode("ascii") + b"\0")
            for child in item:
                visit(child)
        elif item is None or isinstance(item, (bool, int, float, str)):
            digest.update(type(item).__name__.encode("ascii") + b"\0")
            digest.update(repr(item).encode("utf-8") + b"\0")
        else:
            raise TrainerError(f"unsupported state type in canonical hash: {type(item)!r}")

    visit(value)
    return digest.hexdigest()


def nested_tensor_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            nested_tensor_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            nested_tensor_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def reduce_mean(values: Sequence[float], device: torch.device, world: int) -> list[float]:
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    if world > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= float(world)
    return [float(item) for item in tensor.tolist()]


def atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def compute_topk_kkt_relu_metric(
    *,
    X: torch.Tensor,
    y: torch.Tensor,
    beta: torch.Tensor,
    residual: torch.Tensor,
    lambdas: torch.Tensor,
    relu_knots: torch.Tensor,
    relu_b: torch.Tensor,
    relu_weights: torch.Tensor,
    topk: int = 32,
) -> torch.Tensor:
    """V53fast detached TopK KKT telemetry without a P-by-P Gram matrix."""
    batch_size, query_count, feature_count = beta.shape
    n_rows = int(X.shape[1])
    selected_count = max(1, min(int(topk), int(feature_count)))
    with torch.no_grad():
        beta_float = beta.detach().float()
        residual_float = residual.detach().float()
        lambda_float = lambdas.detach().float()
        gradient = -torch.einsum(
            "bnp,bqn->bqp", X.float(), residual_float
        ) / float(max(1, n_rows))
        diagonal = X.float().square().mean(dim=1).clamp_min(1e-6)
        gradient_score = gradient.abs() / diagonal[:, None, :].sqrt()
        beta_abs = beta_float.abs()
        beta_scale = beta_abs.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        gradient_scale = gradient_score.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        score = gradient_score + 0.5 * gradient_scale * (beta_abs / beta_scale)
        indices = torch.topk(score, k=selected_count, dim=-1).indices
        gradient_selected = torch.gather(gradient, -1, indices)
        beta_selected = torch.gather(beta_float, -1, indices)
        abs_selected = beta_selected.abs()
        flat_abs = abs_selected.reshape(batch_size * query_count, selected_count)
        flat_b = relu_b.detach().float().reshape(batch_size * query_count)
        flat_weights = relu_weights.detach().float().reshape(
            batch_size * query_count, -1
        )
        activation = torch.relu(
            flat_abs.unsqueeze(-1)
            - relu_knots.detach().float().view(1, 1, -1)
        )
        derivative = flat_b[:, None] + (
            activation * flat_weights[:, None, :]
        ).sum(dim=-1)
        derivative = derivative.reshape(batch_size, query_count, selected_count)
        d0 = flat_b.reshape(batch_size, query_count)
        sign_beta = beta_selected.sign()
        nonzero = (abs_selected > 1e-12).float()
        residual_nonzero = (
            gradient_selected
            + lambda_float.unsqueeze(-1) * derivative * sign_beta
        )
        residual_zero = torch.relu(
            gradient_selected.abs()
            - lambda_float.unsqueeze(-1) * d0.unsqueeze(-1)
        )
        kkt_residual = nonzero * residual_nonzero + (1.0 - nonzero) * residual_zero
        return torch.sqrt(
            (kkt_residual * kkt_residual).sum(dim=-1) + 1e-30
        ).mean()


def objective_exp125_v58(
    model: torch.nn.Module,
    payload: Mapping[str, torch.Tensor],
    knots: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Validated V58 beta objective with detached KKT diagnostics."""

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        beta = model(
            payload["X"],
            payload["y"],
            lambdas=payload["lambdas"],
            lambda_max=payload["lambda_max"],
            penalty_shape_features=payload["shape"],
            penalty_d0=payload["d0"],
        )
    beta = beta.float()
    X = payload["X"].float()
    y = payload["y"].float()
    n_train = int(X.shape[1])
    prediction = torch.einsum("bnp,bqp->bqn", X, beta)
    residual = y.unsqueeze(1) - prediction
    data_half = 0.5 / float(n_train) * residual.square().sum(dim=-1)
    penalty = compute_penalty_relu_cached(
        beta,
        payload["lambdas"].float(),
        knots,
        payload["relu_b"],
        payload["relu_weights"],
    )
    f_hat = data_half + penalty
    f_zero = 0.5 / float(n_train) * y.square().sum(dim=-1)
    loss = torch.log1p(
        f_hat / f_zero.unsqueeze(1).clamp_min(1e-12)
    ).mean()
    kkt = compute_topk_kkt_relu_metric(
        X=X,
        y=y,
        beta=beta,
        residual=residual,
        lambdas=payload["lambdas"],
        relu_knots=knots,
        relu_b=payload["relu_b"],
        relu_weights=payload["relu_weights"],
        topk=32,
    )
    return {
        "loss": loss,
        "data_half": data_half.mean(),
        "mse": (2.0 * data_half).mean(),
        "penalty": penalty.mean(),
        "objective": f_hat.mean(),
        "f_zero": f_zero.mean(),
        "kkt_top32": kkt,
        "nnz": (beta.abs() > 1e-6).float().sum(dim=-1).mean(),
        "beta": beta,
    }


def objective_kalbi_v5_exactprox(
    model: torch.nn.Module,
    payload: Mapping[str, torch.Tensor],
    knots: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """V5 objective with exact Direct96 proximal-gradient telemetry."""

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        aux = model(
            payload["X"],
            payload["y"],
            lambdas=payload["lambdas"],
            lambda_max=payload["lambda_max"],
            penalty_shape_features=payload["shape"],
            penalty_d0=payload["d0"],
            return_aux=True,
        )
    if not isinstance(aux, Mapping) or "beta_q" not in aux:
        raise RuntimeError("Kalbi V5 objective requires decode auxiliary output")
    beta = aux["beta_q"].float()
    X = payload["X"].float()
    y = payload["y"].float()
    n_train = int(X.shape[1])
    prediction = torch.einsum("bnp,bqp->bqn", X, beta)
    residual = y.unsqueeze(1) - prediction
    data_half = 0.5 / float(n_train) * residual.square().sum(dim=-1)
    penalty = compute_penalty_relu_cached(
        beta,
        payload["lambdas"].float(),
        knots,
        payload["relu_b"],
        payload["relu_weights"],
    )
    f_hat = data_half + penalty
    f_zero = 0.5 / float(n_train) * y.square().sum(dim=-1)
    loss = torch.log1p(
        f_hat / f_zero.unsqueeze(1).clamp_min(1e-12)
    ).mean()
    kkt = compute_topk_kkt_relu_metric(
        X=X,
        y=y,
        beta=beta,
        residual=residual,
        lambdas=payload["lambdas"],
        relu_knots=knots,
        relu_b=payload["relu_b"],
        relu_weights=payload["relu_weights"],
        topk=32,
    )
    telemetry_names = {
        "refinement_step_size": "kalbi_v5_refinement_step_size_q",
        "refinement_gamma": "kalbi_v5_refinement_gamma_q",
        "refinement_kappa": "kalbi_v5_refinement_kappa_q",
        "refinement_min_curvature_denominator": (
            "kalbi_v5_refinement_min_curvature_denominator_q"
        ),
        "refinement_initial_lipschitz": (
            "kalbi_v5_refinement_initial_lipschitz_q"
        ),
        "refinement_majorization_ratio": (
            "kalbi_v5_refinement_majorization_ratio_q"
        ),
        "refinement_backtracks": "kalbi_v5_refinement_backtracks_q",
        "refinement_trace_fallback": (
            "kalbi_v5_refinement_trace_fallback_q"
        ),
        "refinement_data_gradient_rms": (
            "kalbi_v5_refinement_data_gradient_rms_q"
        ),
        "refinement_zero_threshold": (
            "kalbi_v5_refinement_zero_threshold_q"
        ),
        "refinement_change_rms": "kalbi_v5_refinement_change_rms_q",
        "refinement_active_fraction": (
            "kalbi_v5_refined_active_fraction_q"
        ),
        "refinement_saturated_fraction": (
            "kalbi_v5_refinement_saturated_fraction_q"
        ),
        "refinement_d0_contract_error": (
            "kalbi_v5_refinement_d0_contract_error_q"
        ),
    }
    telemetry: dict[str, torch.Tensor] = {}
    for output_name, aux_name in telemetry_names.items():
        if aux_name not in aux:
            raise RuntimeError(f"Kalbi V5 telemetry missing: {aux_name}")
        telemetry[output_name] = aux[aux_name].float().mean()
    return {
        "loss": loss,
        "data_half": data_half.mean(),
        "mse": (2.0 * data_half).mean(),
        "penalty": penalty.mean(),
        "objective": f_hat.mean(),
        "f_zero": f_zero.mean(),
        "kkt_top32": kkt,
        "nnz": (beta.abs() > 1e-6).float().sum(dim=-1).mean(),
        "beta": beta,
        **telemetry,
    }


def objective_kalbi_v7_blockprox3(
    model: torch.nn.Module,
    payload: Mapping[str, torch.Tensor],
    knots: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """V7 objective with three full-P block-proximal sweep telemetry."""

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        aux = model(
            payload["X"],
            payload["y"],
            lambdas=payload["lambdas"],
            lambda_max=payload["lambda_max"],
            penalty_shape_features=payload["shape"],
            penalty_d0=payload["d0"],
            return_aux=True,
        )
    if not isinstance(aux, Mapping) or "beta_q" not in aux:
        raise RuntimeError("KALBI V7 objective requires decode auxiliary output")
    if int(aux.get("kalbi_v7_refinement_sweeps", -1)) != 3:
        raise RuntimeError(
            "KALBI V7 objective requires exactly three full-P block sweeps"
        )
    visited_features = aux.get(
        "kalbi_v7_refinement_visited_features_path_q"
    )
    expected_features = int(payload["X"].shape[-1])
    if (
        visited_features is None
        or visited_features.ndim != 3
        or int(visited_features.shape[-1]) != 3
        or not bool((visited_features == expected_features).all())
    ):
        raise RuntimeError(
            "KALBI V7 objective requires every feature to be visited exactly "
            "once in each of three sweeps"
        )

    beta = aux["beta_q"].float()
    X = payload["X"].float()
    y = payload["y"].float()
    n_train = int(X.shape[1])
    prediction = torch.einsum("bnp,bqp->bqn", X, beta)
    residual = y.unsqueeze(1) - prediction
    data_half = 0.5 / float(n_train) * residual.square().sum(dim=-1)
    penalty = compute_penalty_relu_cached(
        beta,
        payload["lambdas"].float(),
        knots,
        payload["relu_b"],
        payload["relu_weights"],
    )
    f_hat = data_half + penalty
    f_zero = 0.5 / float(n_train) * y.square().sum(dim=-1)
    loss = torch.log1p(
        f_hat / f_zero.unsqueeze(1).clamp_min(1e-12)
    ).mean()
    kkt = compute_topk_kkt_relu_metric(
        X=X,
        y=y,
        beta=beta,
        residual=residual,
        lambdas=payload["lambdas"],
        relu_knots=knots,
        relu_b=payload["relu_b"],
        relu_weights=payload["relu_weights"],
        topk=32,
    )

    path_names = (
        "step_size",
        "gamma",
        "kappa",
        "min_curvature_denominator",
        "curvature_step_cap",
        "initial_lipschitz",
        "trace_lipschitz",
        "majorization_ratio",
        "backtracks",
        "trace_fallback",
        "data_gradient_rms",
        "zero_threshold",
        "change_rms",
        "active_fraction",
        "saturated_fraction",
        "d0_contract_error",
        "data_half_before",
        "data_half_after",
        "penalty_before",
        "penalty_after",
        "objective_before",
        "objective_after",
        "objective_delta",
    )
    telemetry: dict[str, torch.Tensor] = {}
    for name in path_names:
        aux_name = f"kalbi_v7_refinement_{name}_path_q"
        if aux_name not in aux:
            raise RuntimeError(f"KALBI V7 telemetry missing: {aux_name}")
        path = aux[aux_name].float()
        if path.ndim != 3 or path.shape[-1] != 3:
            raise RuntimeError(
                f"KALBI V7 telemetry shape drift for {aux_name}: "
                f"{tuple(path.shape)}"
            )
        telemetry[f"refinement_{name}"] = path.mean()
        telemetry[f"refinement_{name}_final"] = path[..., -1].mean()
        telemetry[f"refinement_{name}_by_iteration"] = path.mean(dim=(0, 1))

    telemetry.update(
        {
            "refinement_majorization_ratio_max": aux[
                "kalbi_v7_refinement_majorization_ratio_max_q"
            ].float().max(),
            "refinement_min_curvature_denominator_min": aux[
                "kalbi_v7_refinement_min_curvature_denominator_min_q"
            ].float().min(),
            "refinement_backtracks_total": aux[
                "kalbi_v7_refinement_backtracks_total_q"
            ].float().mean(),
            "refinement_trace_fallback_fraction": aux[
                "kalbi_v7_refinement_trace_fallback_fraction_q"
            ].float().mean(),
            "refinement_objective_decrease": aux[
                "kalbi_v7_refinement_objective_decrease_q"
            ].float().mean(),
        }
    )
    return {
        "loss": loss,
        "data_half": data_half.mean(),
        "mse": (2.0 * data_half).mean(),
        "penalty": penalty.mean(),
        "objective": f_hat.mean(),
        "f_zero": f_zero.mean(),
        "kkt_top32": kkt,
        "nnz": (beta.abs() > 1e-6).float().sum(dim=-1).mean(),
        "beta": beta,
        **telemetry,
    }


# Compatibility import for copied launchers. In this package it resolves to
# the V7 three-sweep objective and cannot silently run the V6 PG10 contract.
objective_kalbi_v6_exactprox10 = objective_kalbi_v7_blockprox3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--part", default="part_0")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT / "fpk_v33_kkt0.yaml")
    parser.add_argument(
        "--penalty-cache-source",
        type=Path,
        default=PROJECT / "penalty_cache_100k_v9_small_experiment.pt",
    )
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--holdout-files", type=int, default=8)
    parser.add_argument("--cache-files", type=int, default=2)
    parser.add_argument("--batch-per-rank", type=int, default=DEFAULT_BATCH_PER_RANK)
    parser.add_argument(
        "--grad-accum-steps", type=int, default=DEFAULT_GRAD_ACCUM_STEPS
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--max-abs-feature-corr", type=float, default=0.95)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sha256")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.batch_per_rank <= 0 or args.grad_accum_steps <= 0:
        parser.error("batch-per-rank and grad-accum-steps must be positive")
    if args.batch_per_rank * args.grad_accum_steps != TASKS_PER_RANK_UPDATE:
        parser.error(
            "V58 requires batch-per-rank * grad-accum-steps == "
            f"{TASKS_PER_RANK_UPDATE}"
        )

    if args.self_test:
        print(
            json.dumps(
                {
                    "trainer_semantic_version": TRAINER_SEMANTIC_VERSION,
                    "checkpoint_version": CHECKPOINT_VERSION,
                    "world_size": EXPECTED_WORLD_SIZE,
                    "batch_per_rank": args.batch_per_rank,
                    "grad_accum_steps": args.grad_accum_steps,
                    "tasks_per_rank_update": TASKS_PER_RANK_UPDATE,
                    "global_effective_batch": GLOBAL_EFFECTIVE_BATCH,
                    "penalty_pool_contract": "full_100k_no_subset",
                    "objective_contract": OBJECTIVE_CONTRACT,
                    "lambda_low_min": LAMBDA_LOW_MIN,
                    "kkt_contract": KKT_CONTRACT,
                    "execution_contract": EXECUTION_CONTRACT,
                    "exact_resume": True,
                    "imports_versioned_model_modules": False,
                    "standalone_bundle": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.data_root is None:
        parser.error("--data-root is required unless --self-test is used")
    if args.preflight_only:
        print(
            json.dumps(
                validate_part(args.data_root, args.part, args.max_abs_feature_corr),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.out_dir is None:
        parser.error("--out-dir is required for training")
    if (
        args.steps <= 0
        or args.checkpoint_every <= 0
        or args.eval_every <= 0
        or args.cache_files <= 0
    ):
        parser.error("steps/checkpoint-every/eval-every/cache-files must be positive")
    if args.stop_after is not None and not (0 < args.stop_after <= args.steps):
        parser.error("--stop-after must lie in [1, steps]")
    if args.resume is not None and args.resume_sha256 is None:
        parser.error("--resume requires --resume-sha256")

    rank = int(os.environ.get("RANK", -1))
    world = int(os.environ.get("WORLD_SIZE", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if rank < 0 or local_rank < 0 or world != EXPECTED_WORLD_SIZE:
        raise TrainerError("V58 Exp125-contract training requires exactly eight torchrun ranks")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}")

    inventory_holder: list[Any] = [None]
    if rank == 0:
        inventory_holder[0] = validate_part(
            args.data_root, args.part, args.max_abs_feature_corr
        )
    dist.broadcast_object_list(inventory_holder, src=0)
    inventory = inventory_holder[0]
    files = list(inventory["files"])
    holdout_count = int(args.holdout_files)
    if holdout_count < world or holdout_count >= len(files):
        raise TrainerError(
            f"holdout-files must be >= {world} and < {len(files)}, got {holdout_count}"
        )
    train_files = files[:-holdout_count]
    validation_files = files[-holdout_count:]

    penalty_path = args.penalty_cache_source.resolve()
    if not penalty_path.is_file():
        raise TrainerError(f"full penalty cache is missing: {penalty_path}")
    source_holder: list[Any] = [None]
    if rank == 0:
        missing_dependencies = [
            str(path) for path in ROOT_DEPENDENCY_PATHS.values() if not path.is_file()
        ]
        if missing_dependencies:
            raise TrainerError(
                f"required root dependencies are missing: {missing_dependencies}"
            )
        module_resolution = {}
        for module_name, expected_path in sorted(EXPECTED_MODULE_PATHS.items()):
            module = sys.modules.get(module_name)
            actual_path = Path(getattr(module, "__file__", "")).resolve()
            expected_path = expected_path.resolve()
            if actual_path != expected_path:
                raise TrainerError(
                    f"module resolution mismatch for {module_name}: "
                    f"{actual_path} != {expected_path}"
                )
            module_resolution[module_name] = str(actual_path)
        source_holder[0] = {
            "trainer_sha256": file_sha256(Path(__file__)),
            "model_sha256": file_sha256(MODEL_PATH),
            "utils_sha256": file_sha256(UTILS_PATH),
            "penalty_generator_sha256": file_sha256(PENALTY_GENERATOR_PATH),
            "synthetic_data_generator_sha256": file_sha256(SYNTHETIC_DATA_GENERATOR_PATH),
            "config_sha256": file_sha256(args.config),
            "penalty_cache_sha256": file_sha256(penalty_path),
            "root_dependency_hashes": {
                name: file_sha256(path)
                for name, path in sorted(ROOT_DEPENDENCY_PATHS.items())
            },
            "environment": {
                "python": sys.version,
                "numpy": np.__version__,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_capability": list(torch.cuda.get_device_capability(device)),
            },
            "module_resolution": module_resolution,
            "loaded_project_module_closure": loaded_project_module_closure(),
        }
    dist.broadcast_object_list(source_holder, src=0)
    source_hashes = source_holder[0]
    exact_training_arguments = {
        "steps": int(args.steps),
        "seed": int(args.seed),
        "lr": float(args.lr),
        "warmup_steps": int(args.warmup_steps),
        "eval_every": int(args.eval_every),
        "checkpoint_every": int(args.checkpoint_every),
        "holdout_files": holdout_count,
        "cache_files": int(args.cache_files),
        "batch_per_rank": int(args.batch_per_rank),
        "grad_accum_steps": int(args.grad_accum_steps),
        "data_root": str(args.data_root.resolve()),
        "part": str(args.part),
        "data_inventory_sha256": inventory["inventory_sha256"],
        "config": str(args.config.resolve()),
        "penalty_cache": str(penalty_path),
        "penalty_cache_sha256": source_hashes["penalty_cache_sha256"],
        "optimizer_contract": "AdamW_wd1e-4_betas0p9_0p95_v1",
        "scheduler_contract": "linear_warmup_then_cosine_to_total_steps_v1",
        "objective_contract": OBJECTIVE_CONTRACT,
        "lambda_low_min": LAMBDA_LOW_MIN,
        "kkt_contract": KKT_CONTRACT,
        "execution_contract": EXECUTION_CONTRACT,
        "tasks_per_rank_update": TASKS_PER_RANK_UPDATE,
        "global_effective_batch": GLOBAL_EFFECTIVE_BATCH,
    }

    start_checkpoint: dict[str, Any] | None
    if args.resume is None:
        if rank == 0 and args.out_dir.exists() and any(args.out_dir.iterdir()):
            raise TrainerError(f"refusing nonempty fresh output directory: {args.out_dir}")
        start_checkpoint = None
    else:
        if file_sha256(args.resume) != args.resume_sha256:
            raise TrainerError("resume checkpoint byte hash mismatch")
        start_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        metadata = start_checkpoint.get("metadata", {})
        required_resume = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "trainer_semantic_version": TRAINER_SEMANTIC_VERSION,
            "model_version": V58FAST_MODEL_VERSION,
            "penalty_contract_version": V58FAST_PENALTY_CONTRACT_VERSION,
            "world_size": world,
            "exact_training_arguments": exact_training_arguments,
            "out_dir": str(args.out_dir.resolve()),
            "source_hashes": source_hashes,
        }
        for key, expected in required_resume.items():
            if metadata.get(key) != expected:
                raise TrainerError(f"resume contract mismatch for {key}")
        if args.resume.resolve().parent != args.out_dir.resolve():
            raise TrainerError("resume checkpoint must remain in its pinned output directory")
        if not bool(metadata.get("resume_exact", False)):
            raise TrainerError("resume checkpoint is not exact-resumable")

    if rank == 0:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        inventory_path = args.out_dir / "immutable_inventory.json"
        if start_checkpoint is None:
            atomic_json(inventory, inventory_path)
        elif json.loads(inventory_path.read_text())["inventory_sha256"] != inventory[
            "inventory_sha256"
        ]:
            raise TrainerError("resume inventory file changed")
    dist.barrier()

    events_path = args.out_dir / "events.jsonl"

    def emit(event: Mapping[str, Any]) -> None:
        if rank != 0:
            return
        record = {"timestamp": time.time(), **dict(event)}
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        with events_path.open("a") as handle:
            handle.write(line + "\n")

    # All ranks construct identical initial weights.  Calibration is isolated
    # on rank 0 and broadcast by DDP; it never consumes a training stream.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config = _load_yaml(str(args.config))
    sampler = make_sampler(config, args.seed + 1_000_003 * rank + 211)
    penalty_cache = PenaltyCache(penalty_path, args.seed + 101 * rank)
    penalty_cache.load()
    if penalty_cache.pool_size != 100_000:
        raise TrainerError(
            f"full penalty cache must contain exactly 100000 entries, got {penalty_cache.pool_size}"
        )
    model = build_model(config, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    state_element_count = sum(tensor.numel() for tensor in model.state_dict().values())
    if parameter_count != 4_909_208 or state_element_count != 4_910_640:
        raise TrainerError(
            "V58 architecture drift: "
            f"parameters={parameter_count}, state_elements={state_element_count}"
        )

    if start_checkpoint is None:
        if rank == 0:
            calibration_stream = PartStream(
                train_files,
                seed=args.seed + 71,
                rank=0,
                world=world,
                cache_files=args.cache_files,
            )
            calibration_sampler = make_sampler(config, args.seed + 90_000_211)
            calibration_cache = PenaltyCache(penalty_path, args.seed + 90_000_313)
            calibration_cache.load()
            with torch.random.fork_rng(devices=[local_rank]):
                torch.manual_seed(args.seed + 90_000_419)
                torch.cuda.manual_seed(args.seed + 90_000_419)
                refit_normalizers(
                    model,
                    calibration_stream,
                    calibration_sampler,
                    calibration_cache,
                    device,
                )
            normalization_audit: dict[str, Any] | None = {
                "isolated_rank0_calibration": True,
                "training_stream_consumed": False,
                "training_sampler_consumed": False,
                "full_penalty_pool_size": calibration_cache.pool_size,
                "calibration_task_count": 16,
                "shape_index_policy": "4096_evenly_spaced_across_full_100k_pool",
            }
            del calibration_cache, calibration_sampler, calibration_stream
        else:
            normalization_audit = None
    else:
        normalization_audit = dict(start_checkpoint["metadata"]["normalization_audit"])

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=True,
        find_unused_parameters=True,
        init_sync=True,
    )
    core = model.module
    normalization_holder: list[Any] = [normalization_audit if rank == 0 else None]
    dist.broadcast_object_list(normalization_holder, src=0)
    normalization_audit = normalization_holder[0]

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.95)
    )

    def lr_multiplier(step: int) -> float:
        if step < args.warmup_steps:
            return float(step + 1) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(
            max(1, args.steps - args.warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    completed_steps = 0
    stream_state = None
    if start_checkpoint is not None:
        core.load_state_dict(start_checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(start_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(start_checkpoint["scheduler_state_dict"])
        completed_steps = int(start_checkpoint["completed_steps"])
        if completed_steps >= args.steps:
            raise TrainerError("resume checkpoint is already at/after the requested final step")
        rank_states = start_checkpoint["rank_states"]
        if sorted(int(item["rank"]) for item in rank_states) != list(range(world)):
            raise TrainerError("resume checkpoint does not contain exactly one state per rank")
        rank_state = next(item for item in rank_states if int(item["rank"]) == rank)
        stream_state = rank_state["stream"]
        penalty_cache.load_state_dict(rank_state["penalty_cache"])
        sampler.load_state_dict(rank_state["sampler"])
        restore_rng(rank_state["rng"])
    else:
        rank_seed = args.seed + 2_000_003 * rank + 503
        random.seed(rank_seed)
        np.random.seed(rank_seed % (2**32 - 1))
        torch.manual_seed(rank_seed)
        torch.cuda.manual_seed(rank_seed)

    stream = PartStream(
        train_files,
        seed=args.seed + 10_000,
        rank=rank,
        world=world,
        cache_files=args.cache_files,
        state=stream_state,
    )
    knots = penalty_cache.relu_knots.to(device)
    model_hash = state_dict_sha256(core.state_dict())
    model_hashes = [None for _ in range(world)]
    dist.all_gather_object(model_hashes, model_hash)
    if len(set(model_hashes)) != 1:
        raise TrainerError(f"initial DDP model mismatch: {model_hashes}")

    # Fixed validation never consumes the stochastic training cache or sampler.
    validation_path = validation_files[rank % len(validation_files)]
    with np.load(validation_path, allow_pickle=False) as validation_data:
        validation_task_index = rank % 32
        validation_X = [
            np.asarray(validation_data["X"][validation_task_index], dtype=np.float32)
        ]
        validation_y = [
            np.asarray(validation_data["y"][validation_task_index], dtype=np.float32)
        ]
    fixed_penalty_indices = np.asarray(
        [((rank + 1) * 137 + index * 977) % penalty_cache.pool_size for index in range(4)],
        dtype=np.int64,
    )
    validation_sampler = make_sampler(config, args.seed + 70_000_003 * rank + 919)
    fixed_lambdas = validation_sampler._sample_lambdas(1, 8)
    validation_payload = make_query_payload(
        validation_X,
        validation_y,
        penalty_cache,
        validation_sampler,
        device,
        fixed_penalty_indices=fixed_penalty_indices,
        fixed_lambdas=fixed_lambdas,
    )
    del validation_sampler

    emit(
        {
            "kind": "start" if start_checkpoint is None else "resume",
            "trainer_semantic_version": TRAINER_SEMANTIC_VERSION,
            "checkpoint_version": CHECKPOINT_VERSION,
            "model_version": V58FAST_MODEL_VERSION,
            "penalty_contract_version": V58FAST_PENALTY_CONTRACT_VERSION,
            "penalty_input_dim": V58FAST_NATIVE_DIM,
            "penalty_meta_dim": V58FAST_META_DIM,
            "expanded_penalty_dim": V58FAST_EXPANDED_DIM,
            "feature_film_dim": V58FAST_FEATURE_FILM_DIM,
            "prox_condition_dim": V58FAST_PROX_COND_DIM,
            "convexity_flag_present": False,
            "v55_checkpoint_compatible": False,
            "selector_enabled": False,
            "beta_uses_full_N": True,
            "train_test_split_for_beta": False,
            "objective_contract": OBJECTIVE_CONTRACT,
            "objective_formula": (
                "mean(log(1 + F_hat/F_zero)); "
                "F_hat=RSS/(2*N)+lambda*sum_j rho(abs(beta_j)); "
                "F_zero=sum(y^2)/(2*N)"
            ),
            "lambda_low_min": LAMBDA_LOW_MIN,
            "kkt_contract": KKT_CONTRACT,
            "kkt_in_training_loss": False,
            "execution_contract": EXECUTION_CONTRACT,
            "world_size": world,
            "batch_per_rank": args.batch_per_rank,
            "grad_accum_steps": args.grad_accum_steps,
            "tasks_per_rank_update": TASKS_PER_RANK_UPDATE,
            "global_effective_batch": GLOBAL_EFFECTIVE_BATCH,
            "queries_per_task": 32,
            "global_query_evaluations_per_update": GLOBAL_EFFECTIVE_BATCH * 32,
            "penalty_pool_size": penalty_cache.pool_size,
            "penalty_pool_sampling": "full_100k_rank_specific_rng",
            "normalization_audit": normalization_audit,
            "exact_resume": True,
            "completed_steps": completed_steps,
            "fixed_validation_task_count": world,
            "fixed_validation_scope": "one immutable holdout task per rank; eight total",
            "data_mode": "static_single_prior_tabiclv2_part0_small_data_test",
            "train_file_count": len(train_files),
            "holdout_file_count": len(validation_files),
            "inventory": {key: value for key, value in inventory.items() if key != "files"},
            "source_hashes": source_hashes,
            "parameter_count": parameter_count,
            "state_element_count": state_element_count,
            "topk": 32,
            "local_steps": 32,
            "requested_stages": 28,
            "effective_active_stages_for_P400": 12,
            "local_isis_gradient_contract": LOCAL_ISIS_GRADIENT_CONTRACT,
            "discrete_support_gradient": "detached",
            "continuous_local_path_gradient": "enabled",
        }
    )

    def evaluate(step: int) -> None:
        model.eval()
        with torch.no_grad():
            result = objective_exp125_v58(core, validation_payload, knots)
        values = reduce_mean(
            [
                float(result["loss"].item()),
                float(result["data_half"].item()),
                float(result["mse"].item()),
                float(result["penalty"].item()),
                float(result["objective"].item()),
                float(result["f_zero"].item()),
                float(result["kkt_top32"].item()),
                float(result["nnz"].item()),
            ],
            device,
            world,
        )
        if not all(math.isfinite(value) for value in values):
            raise TrainerError(f"nonfinite fixed validation at step {step}: {values}")
        emit(
            {
                "kind": "fixed_validation",
                "step": step,
                "validation_task_count": world,
                "objective_log1p_Fhat_over_Fzero": values[0],
                "L_data_half_mse_train_N": values[1],
                "train_mse_equals_2_L_data": values[2],
                "L_pen": values[3],
                "F_hat_mean": values[4],
                "F_zero_mean": values[5],
                "KKT_top32_detached": values[6],
                "KKT_in_training_loss": False,
                "nnz_mean": values[7],
            }
        )
        model.train()

    def save_checkpoint(reason: str) -> dict[str, Any]:
        local_state = {
            "rank": rank,
            "rng": capture_rng(),
            "stream": stream.state_dict(),
            "penalty_cache": penalty_cache.state_dict(),
            "sampler": sampler.state_dict(),
        }
        rank_states = [None for _ in range(world)]
        dist.all_gather_object(rank_states, local_state)
        model_fingerprint = state_dict_sha256(core.state_dict())
        optimizer_fingerprint = nested_state_sha256(optimizer.state_dict())
        scheduler_fingerprint = nested_state_sha256(scheduler.state_dict())
        fingerprints = [None for _ in range(world)]
        optimizer_fingerprints = [None for _ in range(world)]
        scheduler_fingerprints = [None for _ in range(world)]
        dist.all_gather_object(fingerprints, model_fingerprint)
        dist.all_gather_object(optimizer_fingerprints, optimizer_fingerprint)
        dist.all_gather_object(scheduler_fingerprints, scheduler_fingerprint)
        if len(set(fingerprints)) != 1:
            raise TrainerError(f"DDP model mismatch at step {completed_steps}")
        if len(set(optimizer_fingerprints)) != 1:
            raise TrainerError(f"DDP optimizer mismatch at step {completed_steps}")
        if len(set(scheduler_fingerprints)) != 1:
            raise TrainerError(f"DDP scheduler mismatch at step {completed_steps}")

        result_holder: list[Any] = [None]
        if rank == 0:
            checkpoint_path = args.out_dir / f"checkpoint_step{completed_steps:06d}.pt"
            payload = {
                "step": completed_steps,
                "completed_steps": completed_steps,
                "model_state_dict": core.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "rank_states": rank_states,
                "metadata": {
                    "checkpoint_version": CHECKPOINT_VERSION,
                    "trainer_semantic_version": TRAINER_SEMANTIC_VERSION,
                    "model_version": V58FAST_MODEL_VERSION,
                    "penalty_contract_version": V58FAST_PENALTY_CONTRACT_VERSION,
                    "penalty_input_dim": V58FAST_NATIVE_DIM,
                    "penalty_meta_dim": V58FAST_META_DIM,
                    "expanded_penalty_dim": V58FAST_EXPANDED_DIM,
                    "feature_film_dim": V58FAST_FEATURE_FILM_DIM,
                    "prox_condition_dim": V58FAST_PROX_COND_DIM,
                    "convexity_flag_present": False,
                    "v55_checkpoint_compatible": False,
                    "selector_enabled": False,
                    "resume_exact": True,
                    "world_size": world,
                    "batch_per_rank": args.batch_per_rank,
                    "grad_accum_steps": args.grad_accum_steps,
                    "tasks_per_rank_update": TASKS_PER_RANK_UPDATE,
                    "global_effective_batch": GLOBAL_EFFECTIVE_BATCH,
                    "penalty_pool_size": penalty_cache.pool_size,
                    "topk": 32,
                    "local_steps": 32,
                    "requested_stages": 28,
                    "effective_active_stages_for_P400": 12,
                    "data_inventory_sha256": inventory["inventory_sha256"],
                    "objective_contract": OBJECTIVE_CONTRACT,
                    "local_isis_gradient_contract": LOCAL_ISIS_GRADIENT_CONTRACT,
                    "lambda_low_min": LAMBDA_LOW_MIN,
                    "kkt_contract": KKT_CONTRACT,
                    "kkt_in_training_loss": False,
                    "execution_contract": EXECUTION_CONTRACT,
                    "source_hashes": source_hashes,
                    "normalization_audit": normalization_audit,
                    "exact_training_arguments": exact_training_arguments,
                    "out_dir": str(args.out_dir.resolve()),
                    "model_state_sha256": model_fingerprint,
                    "optimizer_state_sha256": optimizer_fingerprint,
                    "scheduler_state_sha256": scheduler_fingerprint,
                    "reason": reason,
                },
            }
            atomic_torch_save(payload, checkpoint_path)
            reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if sorted(int(item["rank"]) for item in reloaded["rank_states"]) != list(
                range(world)
            ):
                raise TrainerError("checkpoint rank-state inventory failed reload")
            with torch.random.fork_rng(devices=[local_rank]):
                fresh = build_model(config, device)
                fresh.load_state_dict(reloaded["model_state_dict"], strict=True)
                model_ok = state_dict_sha256(fresh.state_dict()) == model_fingerprint
                fresh_optimizer = torch.optim.AdamW(
                    fresh.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.95)
                )
                fresh_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    fresh_optimizer, lr_multiplier
                )
                fresh_optimizer.load_state_dict(reloaded["optimizer_state_dict"])
                fresh_scheduler.load_state_dict(reloaded["scheduler_state_dict"])
                optimizer_ok = nested_tensor_equal(
                    fresh_optimizer.state_dict(), optimizer.state_dict()
                )
                scheduler_ok = nested_tensor_equal(
                    fresh_scheduler.state_dict(), scheduler.state_dict()
                )
                del fresh_scheduler, fresh_optimizer, fresh
            if not model_ok or not optimizer_ok or not scheduler_ok:
                raise TrainerError(
                    "checkpoint reload verification failed "
                    f"model={model_ok} optimizer={optimizer_ok} scheduler={scheduler_ok}"
                )
            checkpoint_sha = file_sha256(checkpoint_path)
            sidecar = checkpoint_path.with_suffix(".sha256")
            sidecar_tmp = sidecar.with_suffix(".sha256.tmp")
            sidecar_tmp.write_text(f"{checkpoint_sha}  {checkpoint_path.name}\n")
            os.replace(sidecar_tmp, sidecar)
            result_holder[0] = {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha,
                "model_state_sha256": model_fingerprint,
                "optimizer_state_sha256": optimizer_fingerprint,
                "scheduler_state_sha256": scheduler_fingerprint,
                "reload_verified": True,
                "resume_exact": True,
            }
        dist.broadcast_object_list(result_holder, src=0)
        dist.barrier()
        emit(
            {
                "kind": "checkpoint",
                "step": completed_steps,
                "reason": reason,
                **result_holder[0],
            }
        )
        return dict(result_holder[0])

    evaluate(completed_steps)
    target_step = args.steps if args.stop_after is None else min(args.steps, args.stop_after)
    last_checkpoint: dict[str, Any] | None = None
    while completed_steps < target_step:
        started = time.time()
        optimizer.zero_grad(set_to_none=True)
        metric_sums = {
            "loss": 0.0,
            "data_half": 0.0,
            "mse": 0.0,
            "penalty": 0.0,
            "objective": 0.0,
            "f_zero": 0.0,
            "kkt_top32": 0.0,
            "nnz": 0.0,
        }
        task_infos: list[dict[str, Any]] = []
        sampled_lambda_min = math.inf
        sampled_lambda_max = -math.inf
        for accumulation_index in range(args.grad_accum_steps):
            Xs, ys, data_infos = stream.next_batch(
                args.batch_per_rank, return_info=True
            )
            task_infos.extend(data_infos)
            payload = make_query_payload(
                Xs, ys, penalty_cache, sampler, device
            )
            current_lambda_min = float(payload["lambdas"].detach().min().item())
            current_lambda_max = float(payload["lambdas"].detach().max().item())
            if current_lambda_min < LAMBDA_LOW_MIN - 1e-8:
                raise TrainerError(
                    f"sampled lambda {current_lambda_min} below V58 floor {LAMBDA_LOW_MIN}"
                )
            sampled_lambda_min = min(sampled_lambda_min, current_lambda_min)
            sampled_lambda_max = max(sampled_lambda_max, current_lambda_max)
            sync_context = (
                model.no_sync()
                if accumulation_index < args.grad_accum_steps - 1
                else contextlib.nullcontext()
            )
            with sync_context:
                result = objective_exp125_v58(model, payload, knots)
                if not torch.isfinite(result["loss"]):
                    raise TrainerError("nonfinite beta objective")
                (result["loss"] / float(args.grad_accum_steps)).backward()
            for key in metric_sums:
                metric_sums[key] += float(result[key].detach().item()) / float(
                    args.grad_accum_steps
                )

        head_grad_norms = {
            "shape": module_grad_norm(core.cond_embed.penalty_shape_encoder),
            "query": module_grad_norm(core.query_mixer),
            "feature": module_grad_norm(core.feature_head),
            "local": module_grad_norm(core.v46_local_mixer),
        }
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0).item()
        )
        if not math.isfinite(gradient_norm) or not all(
            math.isfinite(value) for value in head_grad_norms.values()
        ):
            raise TrainerError("nonfinite beta gradient norm")
        optimizer.step()
        scheduler.step()
        completed_steps += 1
        reduced_values = reduce_mean(
            [metric_sums[key] for key in metric_sums], device, world
        )
        reduced = dict(zip(metric_sums, reduced_values))
        if not all(math.isfinite(value) for value in reduced.values()):
            raise TrainerError("nonfinite globally reduced training metric")
        emit(
            {
                "kind": "train_update",
                "step": completed_steps,
                "objective_log1p_Fhat_over_Fzero": reduced["loss"],
                "L_data_half_mse_train_N": reduced["data_half"],
                "train_mse_equals_2_L_data": reduced["mse"],
                "L_pen": reduced["penalty"],
                "F_hat_mean": reduced["objective"],
                "F_zero_mean": reduced["f_zero"],
                "KKT_top32_detached": reduced["kkt_top32"],
                "KKT_in_training_loss": False,
                "nnz_mean": reduced["nnz"],
                "loss_denominator_contract": (
                    "F_zero=sum(y^2)/(2*N_train), computed per task"
                ),
                "N_train_min": 200,
                "N_train_max": 200,
                "P_min": 400,
                "P_max": 400,
                "global_task_batch": GLOBAL_EFFECTIVE_BATCH,
                "batch_per_rank": args.batch_per_rank,
                "grad_accum_steps": args.grad_accum_steps,
                "tasks_per_rank_update": TASKS_PER_RANK_UPDATE,
                "global_query_evaluations": GLOBAL_EFFECTIVE_BATCH * 32,
                "sampled_lambda_min": sampled_lambda_min,
                "sampled_lambda_max": sampled_lambda_max,
                "lambda_low_min_contract": LAMBDA_LOW_MIN,
                "penalty_pool_size": penalty_cache.pool_size,
                "feature_correlation_contract": (
                    "manifest_max_after_strictly_lt_0p95_and_payload_sha256_pinned"
                ),
                "rank0_task_refs": [
                    {
                        "file": Path(info["file"]).name,
                        "task_index": int(info["task_index"]),
                    }
                    for info in task_infos
                ],
                "gradient_norm_preclip": gradient_norm,
                "head_gradient_norms_preclip": head_grad_norms,
                "lr": scheduler.get_last_lr()[0],
                "elapsed_seconds": time.time() - started,
                "KKT_status": "computed_detached_every_update_not_in_loss",
            }
        )
        if completed_steps % args.eval_every == 0 or completed_steps == target_step:
            evaluate(completed_steps)
        if completed_steps % args.checkpoint_every == 0:
            last_checkpoint = save_checkpoint("scheduled")

    paused = completed_steps < args.steps
    if last_checkpoint is None or completed_steps % args.checkpoint_every != 0:
        last_checkpoint = save_checkpoint("paused" if paused else "final")
    if rank == 0:
        terminal_payload = {
            "status": "PAUSED" if paused else "COMPLETE",
            "completed_steps": completed_steps,
            "requested_steps": args.steps,
            "trainer_semantic_version": TRAINER_SEMANTIC_VERSION,
            "checkpoint_version": CHECKPOINT_VERSION,
            "model_version": V58FAST_MODEL_VERSION,
            "objective_contract": OBJECTIVE_CONTRACT,
            "lambda_low_min": LAMBDA_LOW_MIN,
            "kkt_contract": KKT_CONTRACT,
            "kkt_in_training_loss": False,
            "execution_contract": EXECUTION_CONTRACT,
            "batch_per_rank": args.batch_per_rank,
            "grad_accum_steps": args.grad_accum_steps,
            "tasks_per_rank_update": TASKS_PER_RANK_UPDATE,
            "global_effective_batch": GLOBAL_EFFECTIVE_BATCH,
            "penalty_pool_size": penalty_cache.pool_size,
            "exact_resume": True,
            "source_hashes": source_hashes,
            "final_checkpoint": last_checkpoint,
        }
        atomic_json(
            terminal_payload,
            args.out_dir / ("PAUSED.json" if paused else "COMPLETE.json"),
        )
        stale_marker = args.out_dir / ("COMPLETE.json" if paused else "PAUSED.json")
        stale_marker.unlink(missing_ok=True)
        emit({"kind": "paused" if paused else "complete", **terminal_payload})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
