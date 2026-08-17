#!/usr/bin/env python3
"""V58-Chebyshev + V12 Canonical on a crash-safe rolling 100GB corpus.

The random beta-query path and learned selector both call the same
``canonical_penalty.build_penalty`` function.  The trainer adopts one immutable
catalog on every 500-update boundary, clears every task-file cache even when
CURRENT is unchanged, checkpoints only after that reload, and pins the exact
catalog needed for resume before an older lease can be released.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
import random
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .canonical_penalty import (
    CONTRACT_VERSION,
    DIRECT_PARAMETER_DIM,
    build_penalty,
    canonical_grid,
)
from .canonical_penalty_cache import (
    CACHE_VERSION,
    CanonicalPenaltyCache,
)
from .model import (
    BASIS_VERSION,
    LOCAL_ISIS_GRADIENT_CONTRACT,
    MODEL_VERSION,
    PENALTY_CONTRACT_VERSION,
    build_model,
)
from .rolling_catalog import CatalogStore, CapacityPolicy
from .selector_v10 import summarize_encoded
from .selector_v11_direct import V11_DIRECT_PENALTY_ENCODING_VERSION
from .selector_v12_canonical import (
    DirectPenaltyMLPQuerySelectorV12Canonical,
    V12_HIDDEN_DIM,
    V12_MLP_DEPTH,
    V12_SELECTOR_VERSION,
)
from .train import (
    ROOT_DEPENDENCY_PATHS,
    TrainerError,
    atomic_torch_save,
    capture_rng,
    make_sampler,
    module_grad_norm,
    objective_exp125_v58,
    restore_rng,
    state_dict_sha256,
    standardize_tasks,
)
from .train_joint_v11_direct import (
    assert_ddp_synced,
    file_sha256,
    grad_stats,
    module_sha256,
    selector_path_features,
    tensor_gradient_stats,
)
from .utils import native196_to_expanded388
from train_fpk import _build_X_y_tensors, _compute_lambda_max, _load_yaml


TRAINER_VERSION = "v58cheb_v12_canonical_rolling100gb_reload500_v1"
CHECKPOINT_VERSION = "v58cheb_v12_canonical_rolling_checkpoint_v1"
DATASET_CONTRACT = "tabiclv2_graph_scm_featureprior_groupdrop095_sparse_linear_v4"
EXPECTED_WORLD_SIZE = 8
RELOAD_INTERVAL_UPDATES = 500
LAMBDA_LOW_MIN_CONTRACT = 1e-6
SELECTOR_RATIO_MIN = 1e-6
SELECTOR_RATIO_MAX = 1.0
SELECTOR_OBJECTIVE = "mean(log1p(heldout_mse/2))_actual_hard_beta_q"
SELECTOR_PATH_FEATURES = 150
AUDITED_REFERENCE_INDICES = (81_846, 1_713, 81_840)


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def optimizer_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def validate_pipeline_contract(
    catalog: Mapping[str, Any], expected_sha256: str
) -> None:
    contract = catalog.get("contract", {})
    if contract.get("pipeline_config_sha256") != expected_sha256:
        raise TrainerError("catalog top-level pipeline hash drift")
    observed = {
        entry.get("pipeline_config_sha256") for entry in catalog.get("files", [])
    }
    if observed != {expected_sha256}:
        raise TrainerError(f"catalog entry pipeline hash drift: {observed}")
    for entry in catalog.get("files", []):
        if entry.get("prior") != "tabicl_v2":
            raise TrainerError("rolling catalog contains a non-TabICLv2 shard")
        if int(entry.get("num_tasks", -1)) != 1:
            raise TrainerError("rolling catalog shard does not contain one task")
        n = int(entry.get("n", -1))
        p = int(entry.get("p", -1))
        if not (32 <= n <= 20_000 and 50 <= p <= 100_000 and n * p <= 24_000_000):
            raise TrainerError(f"catalog task shape outside contract: N={n} P={p}")
        metadata = entry.get("metadata", {})
        semantic = metadata.get("semantic_contract", {}) if isinstance(metadata, Mapping) else {}
        if semantic.get("contract") != DATASET_CONTRACT:
            raise TrainerError("catalog shard lacks the canonical V4 semantic certificate")
        target_snr = float(semantic.get("target_snr", float("nan")))
        if not math.isfinite(target_snr) or not 1.0 <= target_snr <= 10.0:
            raise TrainerError("catalog shard SNR certificate is outside [1,10]")


def adopt_catalog_package(
    store: CatalogStore,
    package: Mapping[str, Any],
    *,
    rank: int,
    world: int,
    expected_pipeline_sha256: str,
    rank0_hashes_verified: bool,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    pointer = package.get("pointer")
    catalog = package.get("catalog")
    if not isinstance(pointer, Mapping) or not isinstance(catalog, Mapping):
        raise TrainerError("catalog package lacks pointer/catalog")
    pointer = dict(pointer)
    catalog = dict(catalog)
    if int(pointer.get("epoch", -1)) != int(catalog.get("epoch", -2)):
        raise TrainerError("CURRENT pointer/catalog epoch mismatch")
    validate_pipeline_contract(catalog, expected_pipeline_sha256)
    entries = store.entries_for_rank(catalog, rank)
    if not entries:
        raise TrainerError(f"catalog assigns no shard to rank {rank}")
    if rank != 0 or not rank0_hashes_verified:
        store.verify_catalog_files(catalog, hashes=False)
    digest = str(pointer.get("catalog_sha256", ""))
    gathered: list[Any] = [None for _ in range(world)]
    dist.all_gather_object(gathered, digest)
    if len(set(gathered)) != 1:
        raise TrainerError(f"ranks adopted different catalog hashes: {gathered}")
    return pointer, catalog, entries


def load_current_catalog(
    store: CatalogStore,
    *,
    rank: int,
    world: int,
    expected_pipeline_sha256: str,
    active_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], bool]:
    shared: list[Any] = [None]
    if rank == 0:
        pointer, catalog = store.load_current(verify_files=False)
        changed = str(pointer["catalog_sha256"]) != active_sha256
        # Hash all bytes at first adoption and after each immutable transition.
        # An unchanged 500-step reload still verifies paths and sizes on all ranks.
        store.verify_catalog_files(catalog, hashes=bool(changed))
        validate_pipeline_contract(catalog, expected_pipeline_sha256)
        shared[0] = {
            "pointer": pointer,
            "catalog": catalog,
            "changed": bool(changed),
            "rank0_hashes_verified": bool(changed),
        }
    dist.broadcast_object_list(shared, src=0)
    package = shared[0]
    pointer, catalog, entries = adopt_catalog_package(
        store,
        package,
        rank=rank,
        world=world,
        expected_pipeline_sha256=expected_pipeline_sha256,
        rank0_hashes_verified=bool(package["rank0_hashes_verified"]),
    )
    return pointer, catalog, entries, bool(package["changed"])


class RollingSingleTaskStream:
    """Exact-resumable rank-local stream with an explicit reloadable LRU."""

    VERSION = "v12_canonical_catalog_single_task_stream_v1"

    def __init__(
        self,
        entries: Sequence[Mapping[str, Any]],
        *,
        catalog_sha256: str,
        seed: int,
        rank: int,
        cache_files: int,
        state: Mapping[str, Any] | None = None,
    ) -> None:
        self.seed = int(seed)
        self.rank = int(rank)
        self.cache_files = max(0, int(cache_files))
        self.rng = np.random.default_rng(self.seed + 1_000_003 * self.rank)
        self.cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.reload_count = 0
        self.refresh(entries, catalog_sha256=catalog_sha256)
        if state is not None:
            self.load_state_dict(state)

    @staticmethod
    def _identity(entry: Mapping[str, Any]) -> tuple[int, str, str, int, int]:
        return (
            int(entry["shard_id"]),
            str(Path(entry["absolute_path"]).resolve()),
            str(entry["sha256"]),
            int(entry["n"]),
            int(entry["p"]),
        )

    def refresh(
        self, entries: Sequence[Mapping[str, Any]], *, catalog_sha256: str
    ) -> None:
        values = [dict(entry) for entry in entries]
        values.sort(key=lambda item: int(item["shard_id"]))
        if not values:
            raise TrainerError(f"rank {self.rank} received an empty catalog view")
        identities = [self._identity(entry) for entry in values]
        if len({item[0] for item in identities}) != len(identities):
            raise TrainerError("rank catalog view has duplicate shard IDs")
        for entry, identity in zip(values, identities):
            _shard, path_text, digest, n, p = identity
            path = Path(path_text)
            if not path.is_file() or path.stat().st_size != int(entry["bytes"]):
                raise TrainerError(f"catalog shard missing/changed at reload: {path}")
            if len(digest) != 64 or int(entry.get("num_tasks", -1)) != 1:
                raise TrainerError("rank catalog entry identity is malformed")
            if not (32 <= n <= 20_000 and 50 <= p <= 100_000 and n * p <= 24_000_000):
                raise TrainerError("rank catalog entry shape is outside contract")
        self.entries = values
        self.identities = identities
        self.catalog_sha256 = str(catalog_sha256)
        self.order = np.arange(len(values), dtype=np.int64)
        self.rng.shuffle(self.order)
        self.cursor = 0
        self.cache.clear()
        self.reload_count += 1

    def _load(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        entry = self.entries[index]
        path = str(Path(entry["absolute_path"]).resolve())
        if path in self.cache:
            value = self.cache.pop(path)
            self.cache[path] = value
            return value
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "X", "y", "exact_max_abs_corr", "corr_threshold",
                "target_snr", "target_seed",
            }
            if not required.issubset(archive.files):
                raise TrainerError(f"runtime shard lacks certified arrays: {path}")
            x_raw = archive["X"]
            y_raw = archive["y"]
            corr = np.asarray(archive["exact_max_abs_corr"], dtype=np.float32)
            threshold = np.asarray(archive["corr_threshold"], dtype=np.float32)
            snr = np.asarray(archive["target_snr"], dtype=np.float32)
            if x_raw.dtype != np.float16 or y_raw.dtype != np.float32:
                raise TrainerError(f"runtime shard dtype drift: {path}")
            expected_x = (1, int(entry["n"]), int(entry["p"]))
            expected_y = (1, int(entry["n"]))
            if x_raw.shape != expected_x or y_raw.shape != expected_y:
                raise TrainerError(
                    f"runtime shard shape drift: {path} X={x_raw.shape} y={y_raw.shape}"
                )
            if corr.shape != (1,) or threshold.shape != (1,) or snr.shape != (1,):
                raise TrainerError("runtime shard certificate shape drift")
            if not (
                np.isfinite(x_raw).all()
                and np.isfinite(y_raw).all()
                and np.isfinite(corr).all()
                and float(corr[0]) < float(threshold[0])
                and abs(float(threshold[0]) - 0.95) <= 1e-6
                and 1.0 <= float(snr[0]) <= 10.0
            ):
                raise TrainerError(f"runtime shard semantic certificate failed: {path}")
            x = np.asarray(x_raw[0], dtype=np.float32)
            y = np.asarray(y_raw[0], dtype=np.float32)
        value = (x, y)
        if self.cache_files > 0:
            self.cache[path] = value
            while len(self.cache) > self.cache_files:
                self.cache.popitem(last=False)
        return value

    def next_batch(
        self, batch_size: int = 1
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
        if int(batch_size) != 1:
            raise TrainerError("rolling variable-shape stream requires batch_size=1")
        if self.cursor >= len(self.order):
            self.rng.shuffle(self.order)
            self.cursor = 0
        index = int(self.order[self.cursor])
        self.cursor += 1
        x, y = self._load(index)
        entry = self.entries[index]
        return [x], [y], [{
            "shard_id": int(entry["shard_id"]),
            "path": str(entry["absolute_path"]),
            "n": int(entry["n"]),
            "p": int(entry["p"]),
            "catalog_sha256": self.catalog_sha256,
        }]

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "seed": self.seed,
            "rank": self.rank,
            "catalog_sha256": self.catalog_sha256,
            "identities": [list(value) for value in self.identities],
            "order": self.order.tolist(),
            "cursor": self.cursor,
            "rng_state": self.rng.bit_generator.state,
            "reload_count": self.reload_count,
            "cache_entries_serialized": 0,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "version": self.VERSION,
            "seed": self.seed,
            "rank": self.rank,
            "catalog_sha256": self.catalog_sha256,
            "identities": [list(value) for value in self.identities],
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise TrainerError(f"rolling stream resume mismatch for {key}")
        order = np.asarray(state["order"], dtype=np.int64)
        if order.shape != (len(self.entries),) or not np.array_equal(
            np.sort(order), np.arange(len(self.entries), dtype=np.int64)
        ):
            raise TrainerError("rolling stream resume order is invalid")
        cursor = int(state["cursor"])
        if not 0 <= cursor <= len(order):
            raise TrainerError("rolling stream resume cursor is invalid")
        self.order = order
        self.cursor = cursor
        self.rng.bit_generator.state = state["rng_state"]
        self.reload_count = int(state["reload_count"])
        self.cache.clear()


def split_and_standardize_variable(
    xs: list[np.ndarray],
    ys: list[np.ndarray],
    *,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(xs) != 1 or len(ys) != 1:
        raise TrainerError("variable-shape selector split requires one task")
    x = torch.from_numpy(np.asarray(xs[0], dtype=np.float32))[None].to(device)
    y = torch.from_numpy(np.asarray(ys[0], dtype=np.float32))[None].to(device)
    rows, features = int(x.shape[1]), int(x.shape[2])
    train_rows = rows // 2
    if train_rows < 16 or rows - train_rows < 16:
        raise TrainerError(f"task has too few rows for selector split: N={rows}")
    permutation = torch.randperm(rows, generator=generator)
    train_index = permutation[:train_rows].to(device)
    test_index = permutation[train_rows:].to(device)
    x_train_raw = x.index_select(1, train_index)
    x_test_raw = x.index_select(1, test_index)
    y_train_raw = y.index_select(1, train_index)
    y_test_raw = y.index_select(1, test_index)
    x_mean = x_train_raw.mean(1, keepdim=True)
    x_std_raw = x_train_raw.std(1, keepdim=True, correction=0)
    active = x_std_raw > 1e-6
    x_std = x_std_raw.clamp_min(1e-8)
    y_mean = y_train_raw.mean(1, keepdim=True)
    y_std = y_train_raw.std(1, keepdim=True, correction=0).clamp_min(1e-8)
    values = (
        torch.where(active, (x_train_raw - x_mean) / x_std, 0.0),
        (y_train_raw - y_mean) / y_std,
        torch.where(active, (x_test_raw - x_mean) / x_std, 0.0),
        (y_test_raw - y_mean) / y_std,
    )
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise TrainerError("selector split produced nonfinite standardized data")
    if values[0].shape != (1, train_rows, features):
        raise TrainerError("selector split shape drift")
    return values


def make_canonical_query_payload(
    xs: list[np.ndarray],
    ys: list[np.ndarray],
    cache: CanonicalPenaltyCache,
    sampler: Any,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    xs, ys = standardize_tasks(xs, ys)
    if len(xs) != 1:
        raise TrainerError("canonical rolling beta payload requires one task")
    _indices, encoded4 = cache.sample_encoded(4, device)
    repeat = torch.arange(4, device=device).repeat_interleave(8)
    shape = encoded4["shape_features"].index_select(0, repeat)[None]
    d0 = encoded4["d0"].index_select(0, repeat)[None]
    relu_b = encoded4["relu_b"].index_select(0, repeat)
    relu_weights = encoded4["relu_weights"].index_select(0, repeat)
    lambdas = sampler._sample_lambdas(1, 8).to(device).float()
    lambdas = lambdas[:, None, :].expand(1, 4, 8).reshape(1, 32)
    lambda_max = _compute_lambda_max(xs, ys).to(device).float().view(1, 1)
    lambda_max = lambda_max.expand(1, 32).clamp_min(1e-12)
    x, y = _build_X_y_tensors(xs, ys, device=device, dtype=torch.float32)
    if shape.shape != (1, 32, 196) or d0.shape != (1, 32):
        raise TrainerError("canonical beta payload native196 layout drift")
    if not torch.equal(relu_weights[:, -1], torch.zeros_like(relu_weights[:, -1])):
        raise TrainerError("canonical beta payload dead ReLU2 weight is not exact zero")
    # The beta query consumes the exact tensors produced by the one shared builder.
    if not torch.equal(shape[0, ::8], encoded4["shape_features"]):
        raise TrainerError("beta-query path copied/changed canonical native196")
    return {
        "X": x,
        "y": y,
        "lambdas": lambdas,
        "lambda_max": lambda_max,
        "shape": shape,
        "d0": d0,
        "relu_b": relu_b,
        "relu_weights": relu_weights,
    }


@torch.no_grad()
def refit_canonical_shape_normalizers(
    model: torch.nn.Module,
    cache: CanonicalPenaltyCache,
    sampler: Any,
    *,
    chunk: int = 4096,
) -> dict[str, Any]:
    total = 0
    sum_value = torch.zeros(196, dtype=torch.float64)
    sum_square = torch.zeros(196, dtype=torch.float64)
    for start in range(0, cache.pool_size, int(chunk)):
        direct = cache.direct_parameters[start : start + int(chunk)]
        shape = build_penalty(direct)["shape_features"].double()
        total += int(shape.shape[0])
        sum_value += shape.sum(0)
        sum_square += shape.square().sum(0)
    mean = sum_value / float(total)
    variance = (
        sum_square - sum_value.square() / float(total)
    ) / float(max(1, total - 1))
    scale = variance.clamp_min(0.0).sqrt().clamp_min(1e-6)
    model.cond_embed._shape_mean.copy_(mean.to(model.cond_embed._shape_mean))
    model.cond_embed._shape_scale.copy_(scale.to(model.cond_embed._shape_scale))

    indices = torch.linspace(0, cache.pool_size - 1, 4096).round().long()
    shape = build_penalty(cache.direct_parameters.index_select(0, indices))[
        "shape_features"
    ]
    lambda_raw = sampler._sample_lambdas(4096, 1).reshape(-1).float().cpu()
    expanded = native196_to_expanded388(shape, lambda_raw)
    expanded_mean = expanded.mean(0)
    expanded_scale = expanded.std(0).clamp_min(1e-6)
    model._shape_mean_388.copy_(expanded_mean.to(model._shape_mean_388))
    model._shape_scale_388.copy_(expanded_scale.to(model._shape_scale_388))
    return {
        "cache_count": total,
        "native_shape_scale_min": float(scale.min()),
        "native_shape_scale_max": float(scale.max()),
        "expanded_shape_scale_min": float(expanded_scale.min()),
        "expanded_shape_scale_max": float(expanded_scale.max()),
        "lambda_normalizer_preserved_from_warm_checkpoint": True,
    }


def gather_rank_objects(value: Any, world: int) -> list[Any]:
    output: list[Any] = [None for _ in range(world)]
    dist.all_gather_object(output, value)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rolling-data-root", type=Path, required=True)
    parser.add_argument("--rolling-control-root", type=Path, required=True)
    parser.add_argument("--expected-pipeline-sha256", required=True)
    parser.add_argument("--trainer-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--penalty-cache", type=Path, required=True)
    parser.add_argument("--penalty-cache-sha256", required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--warm-start-sha256", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sha256")
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--beta-lr", type=float, default=2e-4)
    parser.add_argument("--selector-lr", type=float, default=3e-4)
    parser.add_argument("--direct-head-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--batch-per-rank", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--selector-grad-accum-steps", type=int, default=4)
    parser.add_argument("--selector-hidden", type=int, default=V12_HIDDEN_DIM)
    parser.add_argument("--path-chunk", type=int, default=25)
    parser.add_argument("--cache-files", type=int, default=2)
    parser.add_argument("--reload-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=202607170230)
    args = parser.parse_args()

    if args.resume is not None and not args.resume_sha256:
        parser.error("--resume requires --resume-sha256")
    if args.batch_per_rank != 1:
        parser.error("variable-shape rolling training requires batch-per-rank=1")
    if args.selector_hidden != V12_HIDDEN_DIM:
        parser.error(f"V12 selector hidden dimension is fixed at {V12_HIDDEN_DIM}")
    if args.grad_accum_steps != 4 or args.selector_grad_accum_steps != 4:
        parser.error("global beta and selector task batches are fixed at 32")
    if args.reload_every != RELOAD_INTERVAL_UPDATES:
        parser.error("catalog/dataloader reload interval is fixed at 500 updates")
    if args.checkpoint_every != args.reload_every:
        parser.error("checkpoints must align exactly with catalog reload boundaries")
    if args.steps <= 0 or args.steps % args.reload_every != 0:
        parser.error("steps must be a positive multiple of 500")

    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world = int(os.environ.get("WORLD_SIZE", -1))
    if rank < 0 or local_rank < 0 or world != EXPECTED_WORLD_SIZE:
        raise TrainerError("canonical rolling trainer requires exactly eight ranks")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    for path, expected, label in (
        (args.warm_start, args.warm_start_sha256, "warm checkpoint"),
        (args.penalty_cache, args.penalty_cache_sha256, "canonical cache"),
    ):
        if not path.is_file() or file_sha256(path) != expected:
            raise TrainerError(f"{label} SHA256 mismatch")

    random.seed(args.seed + rank)
    np.random.seed((args.seed + rank) % (2**32 - 1))
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    policy = CapacityPolicy(world_size=world)
    store = CatalogStore(
        args.rolling_control_root,
        args.rolling_data_root,
        policy=policy,
        priors=("tabicl_v2",),
    )
    resume_checkpoint = None
    if args.resume is not None:
        if file_sha256(args.resume) != args.resume_sha256:
            raise TrainerError("resume checkpoint SHA256 mismatch")
        resume_checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        if resume_checkpoint.get("metadata", {}).get("checkpoint_version") != CHECKPOINT_VERSION:
            raise TrainerError("resume checkpoint version mismatch")

    if resume_checkpoint is None:
        active_pointer, active_catalog, rank_entries, _changed = load_current_catalog(
            store,
            rank=rank,
            world=world,
            expected_pipeline_sha256=args.expected_pipeline_sha256,
            active_sha256=None,
        )
    else:
        package = {
            "pointer": resume_checkpoint["catalog_pointer"],
            "catalog": resume_checkpoint["catalog_payload"],
        }
        active_pointer, active_catalog, rank_entries = adopt_catalog_package(
            store,
            package,
            rank=rank,
            world=world,
            expected_pipeline_sha256=args.expected_pipeline_sha256,
            rank0_hashes_verified=False,
        )

    active_catalog_sha = str(active_pointer["catalog_sha256"])
    if is_main:
        store.acquire_lease(args.trainer_id, active_catalog_sha)
    dist.barrier()
    store.write_ack(args.trainer_id, rank, world, active_catalog_sha)
    dist.barrier()

    cache = CanonicalPenaltyCache(
        args.penalty_cache, args.seed + 101 * rank
    )
    cache_report = cache.load(expected_count=300_000)
    sampler = make_sampler(_load_yaml(str(args.config)), args.seed + 1_000_003 * rank + 211)
    if not math.isclose(
        float(sampler.cfg.lambda_low_min), LAMBDA_LOW_MIN_CONTRACT,
        rel_tol=0.0, abs_tol=1e-15,
    ):
        raise TrainerError("beta lambda floor contract drift")

    rank_state = None
    if resume_checkpoint is not None:
        rank_states = resume_checkpoint["rank_states"]
        if sorted(int(item["rank"]) for item in rank_states) != list(range(world)):
            raise TrainerError("resume rank-state inventory drift")
        rank_state = next(item for item in rank_states if int(item["rank"]) == rank)
        cache.load_state_dict(rank_state["penalty_cache"])
        sampler.load_state_dict(rank_state["sampler"])
        restore_rng(rank_state["rng"])

    stream = RollingSingleTaskStream(
        rank_entries,
        catalog_sha256=active_catalog_sha,
        seed=args.seed + 10_000,
        rank=rank,
        cache_files=args.cache_files,
        state=None if rank_state is None else rank_state["stream"],
    )
    split_generator = torch.Generator(device="cpu").manual_seed(
        args.seed + 3_000_017 * rank
    )
    if rank_state is not None:
        split_generator.set_state(rank_state["split_generator_state"])

    config = _load_yaml(str(args.config))
    warm = torch.load(args.warm_start, map_location="cpu", weights_only=False)
    warm_metadata = warm.get("metadata", {})
    required_warm_metadata = {
        "checkpoint_version": "v58cheb_v11_direct96_joint_checkpoint_v1",
        "model_version": MODEL_VERSION,
        "basis_version": BASIS_VERSION,
        "penalty_contract_version": PENALTY_CONTRACT_VERSION,
        "v11_direct_encoding_version": V11_DIRECT_PENALTY_ENCODING_VERSION,
        "world_size": EXPECTED_WORLD_SIZE,
    }
    for key, expected in required_warm_metadata.items():
        if warm_metadata.get(key) != expected:
            raise TrainerError(
                f"warm V58/V11 checkpoint contract mismatch for {key}: "
                f"{warm_metadata.get(key)!r} != {expected!r}"
            )
    if not isinstance(warm.get("model_state_dict"), Mapping) or not isinstance(
        warm.get("selector_state_dict"), Mapping
    ):
        raise TrainerError("warm checkpoint lacks V58 model or V11 selector state")
    model = build_model(config, device)
    model.load_state_dict(warm["model_state_dict"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != 4_909_208:
        raise TrainerError("V58-Chebyshev model parameter-count drift")
    shape_refit_holder: list[Any] = [None]
    if resume_checkpoint is None:
        if is_main:
            shape_refit_holder[0] = refit_canonical_shape_normalizers(
                model, cache, sampler
            )
        dist.broadcast_object_list(shape_refit_holder, src=0)
        for tensor in model.state_dict().values():
            dist.broadcast(tensor, src=0)

    selector_state_holder: list[Any] = [None]
    selector_initialization_holder: list[Any] = [None]
    if resume_checkpoint is None and is_main:
        torch.manual_seed(args.seed + 17)
        selector_cpu = DirectPenaltyMLPQuerySelectorV12Canonical(
            ds_feat_dim=9,
            t_feat_dim=32 * 33 // 2,
            krylov_dim=32,
            shape_dim=196,
            path_feat_dim=SELECTOR_PATH_FEATURES,
            hidden=args.selector_hidden,
            depth=V12_MLP_DEPTH,
            lambda_min_ratio=SELECTOR_RATIO_MIN,
            lambda_max_ratio=SELECTOR_RATIO_MAX,
            initial_lambda_ratio=0.1,
            derivative_floor=1e-6,
            squareplus_scale=0.02,
            direct_head_init_std=1e-3,
        )
        calibration = selector_cpu.fit_penalty_parameter_cache(
            cache.direct_parameters
        )
        selector_state_holder[0] = selector_cpu.state_dict()
        selector_initialization_holder[0] = {
            "calibration": calibration,
            "architecture": selector_cpu.architecture_report(),
            "initialization": "new_v12_selector_from_seed",
            "warm_v11_selector_loaded": False,
            "reason": "hidden64/depth3 and hidden512/depth4 schemas are incompatible",
        }
    dist.broadcast_object_list(selector_state_holder, src=0)
    dist.broadcast_object_list(selector_initialization_holder, src=0)
    torch.manual_seed(args.seed + 17)
    selector = DirectPenaltyMLPQuerySelectorV12Canonical(
        ds_feat_dim=9,
        t_feat_dim=32 * 33 // 2,
        krylov_dim=32,
        shape_dim=196,
        path_feat_dim=SELECTOR_PATH_FEATURES,
        hidden=args.selector_hidden,
        depth=V12_MLP_DEPTH,
        lambda_min_ratio=SELECTOR_RATIO_MIN,
        lambda_max_ratio=SELECTOR_RATIO_MAX,
        initial_lambda_ratio=0.1,
        derivative_floor=1e-6,
        squareplus_scale=0.02,
        direct_head_init_std=1e-3,
    )
    if resume_checkpoint is None:
        selector.load_state_dict(selector_state_holder[0], strict=True)
    else:
        selector.load_state_dict(resume_checkpoint["selector_state_dict"], strict=True)
    selector.to(device)
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)

    model = DDP(
        model, device_ids=[local_rank], output_device=local_rank,
        broadcast_buffers=True, find_unused_parameters=True, init_sync=True,
    )
    selector_train = DDP(
        selector, device_ids=[local_rank], output_device=local_rank,
        broadcast_buffers=True, find_unused_parameters=True, init_sync=True,
    )
    core = model.module
    selector_core = selector_train.module
    beta_optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.beta_lr, weight_decay=1e-4, betas=(0.9, 0.95)
    )
    direct_head_parameters = list(selector_core.penalty_parameter_head.parameters())
    direct_ids = {id(parameter) for parameter in direct_head_parameters}
    selector_other_parameters = [
        parameter for parameter in selector_train.parameters()
        if id(parameter) not in direct_ids
    ]
    selector_optimizer = torch.optim.AdamW(
        [
            {"params": selector_other_parameters, "lr": args.selector_lr,
             "name": "selector_context_lambda_trunks"},
            {"params": direct_head_parameters, "lr": args.direct_head_lr,
             "name": "direct96_penalty_head"},
        ],
        lr=args.selector_lr,
        weight_decay=0.0,
    )

    def multiplier(step: int) -> float:
        if step < args.warmup_steps:
            return float(step + 1) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(
            max(1, args.steps - args.warmup_steps)
        )
        return 0.5 * (
            1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0))
        )

    beta_scheduler = torch.optim.lr_scheduler.LambdaLR(beta_optimizer, multiplier)
    completed_steps = 0
    previous_checkpoint_id: str | None = None
    warm_beta_optimizer_resumed = False
    if resume_checkpoint is None:
        if "beta_optimizer_state_dict" in warm:
            beta_optimizer.load_state_dict(warm["beta_optimizer_state_dict"])
            optimizer_to_device(beta_optimizer, device)
            warm_beta_optimizer_resumed = True
        if "beta_scheduler_state_dict" in warm:
            beta_scheduler.load_state_dict(warm["beta_scheduler_state_dict"])
    else:
        beta_optimizer.load_state_dict(resume_checkpoint["beta_optimizer_state_dict"])
        selector_optimizer.load_state_dict(resume_checkpoint["selector_optimizer_state_dict"])
        beta_scheduler.load_state_dict(resume_checkpoint["beta_scheduler_state_dict"])
        optimizer_to_device(beta_optimizer, device)
        optimizer_to_device(selector_optimizer, device)
        completed_steps = int(resume_checkpoint["completed_steps"])
        previous_checkpoint_id = resume_checkpoint["metadata"].get("checkpoint_id")

    knots = canonical_grid(device=device)
    reference_encoded = cache.encoded_indices(
        torch.tensor(AUDITED_REFERENCE_INDICES), device
    )
    fixed_shape = reference_encoded["shape_features"]
    fixed_d0 = reference_encoded["d0"]
    path_ratios = torch.exp(torch.linspace(
        math.log(SELECTOR_RATIO_MIN), math.log(SELECTOR_RATIO_MAX), 50,
        device=device,
    ))

    if is_main:
        if resume_checkpoint is None:
            args.out_dir.mkdir(parents=True, exist_ok=False)
        elif args.resume.resolve().parent != args.out_dir.resolve():
            raise TrainerError("resume checkpoint must remain in its output directory")
    dist.barrier()
    events_path = args.out_dir / "events.jsonl"

    def emit(event: Mapping[str, Any]) -> None:
        if not is_main:
            return
        record = {"timestamp": time.time(), **dict(event)}
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    source_hashes = {
        "trainer": file_sha256(Path(__file__)),
        "model": file_sha256(Path(__file__).with_name("model.py")),
        "utils": file_sha256(Path(__file__).with_name("utils.py")),
        "beta_trainer_helpers": file_sha256(Path(__file__).with_name("train.py")),
        "v11_joint_helper_import": file_sha256(
            Path(__file__).with_name("train_joint_v11_direct.py")
        ),
        "selector_v10": file_sha256(Path(__file__).with_name("selector_v10.py")),
        "selector_v11_direct_encoding": file_sha256(
            Path(__file__).with_name("selector_v11_direct.py")
        ),
        "selector_v11_canonical_base": file_sha256(
            Path(__file__).with_name("selector_v11_canonical.py")
        ),
        "selector_v12_canonical": file_sha256(
            Path(__file__).with_name("selector_v12_canonical.py")
        ),
        "canonical_penalty": file_sha256(
            Path(__file__).with_name("canonical_penalty.py")
        ),
        "canonical_cache_adapter": file_sha256(
            Path(__file__).with_name("canonical_penalty_cache.py")
        ),
        "rolling_catalog": file_sha256(
            Path(__file__).with_name("rolling_catalog.py")
        ),
        "config": file_sha256(args.config),
        "penalty_cache": args.penalty_cache_sha256,
        "warm_start": args.warm_start_sha256,
        "root_dependencies": {
            name: file_sha256(path)
            for name, path in sorted(ROOT_DEPENDENCY_PATHS.items())
        },
    }
    exact_arguments = {
        "rolling_data_root": str(args.rolling_data_root.resolve()),
        "rolling_control_root": str(args.rolling_control_root.resolve()),
        "expected_pipeline_sha256": args.expected_pipeline_sha256,
        "trainer_id": args.trainer_id,
        "steps": args.steps,
        "beta_lr": args.beta_lr,
        "selector_lr": args.selector_lr,
        "direct_head_lr": args.direct_head_lr,
        "selector_hidden": args.selector_hidden,
        "selector_mlp_depth": V12_MLP_DEPTH,
        "batch_per_rank": args.batch_per_rank,
        "grad_accum_steps": args.grad_accum_steps,
        "selector_grad_accum_steps": args.selector_grad_accum_steps,
        "reload_every": args.reload_every,
        "checkpoint_every": args.checkpoint_every,
        "log_every": args.log_every,
        "seed": args.seed,
    }
    if resume_checkpoint is not None:
        metadata = resume_checkpoint["metadata"]
        required_resume = {
            "trainer_version": TRAINER_VERSION,
            "checkpoint_version": CHECKPOINT_VERSION,
            "canonical_penalty_contract": CONTRACT_VERSION,
            "canonical_cache_version": CACHE_VERSION,
            "selector_version": V12_SELECTOR_VERSION,
            "source_hashes": source_hashes,
            "exact_training_arguments": exact_arguments,
            "world_size": world,
        }
        for key, expected in required_resume.items():
            if metadata.get(key) != expected:
                raise TrainerError(f"resume contract mismatch for {key}")

    emit({
        "kind": "start" if resume_checkpoint is None else "resume",
        "trainer_version": TRAINER_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_version": MODEL_VERSION,
        "basis_version": BASIS_VERSION,
        "penalty_contract_version": PENALTY_CONTRACT_VERSION,
        "canonical_penalty_contract": CONTRACT_VERSION,
        "selector_version": V12_SELECTOR_VERSION,
        "canonical_cache": cache_report,
        "dataset_contract": DATASET_CONTRACT,
        "selector_objective": SELECTOR_OBJECTIVE,
        "selector_uses_actual_hard_beta_q": True,
        "selector_inactive_ste_used": False,
        "world_size": world,
        "global_beta_task_batch": world * args.grad_accum_steps,
        "global_selector_task_batch": world * args.selector_grad_accum_steps,
        "reload_interval_updates": args.reload_every,
        "active_catalog_epoch": int(active_pointer["epoch"]),
        "active_catalog_sha256": active_catalog_sha,
        "active_catalog_files": len(active_catalog["files"]),
        "active_catalog_bytes": int(active_catalog["accounting"]["total_bytes"]),
        "shape_normalizer_refit": shape_refit_holder[0],
        "selector_initialization": selector_initialization_holder[0],
        "warm_beta_optimizer_resumed": warm_beta_optimizer_resumed,
        "selector_optimizer_reset_for_new_contract": resume_checkpoint is None,
        "source_hashes": source_hashes,
        "exact_training_arguments": exact_arguments,
        "completed_steps": completed_steps,
    })

    beta_hash_before = assert_ddp_synced(core, world, "initial_beta")
    selector_hash_before = assert_ddp_synced(selector_core, world, "initial_selector")
    compute_seconds = 0.0
    compute_steps = 0

    while completed_steps < args.steps:
        started = time.perf_counter()
        beta_optimizer.zero_grad(set_to_none=True)
        beta_metrics = {key: 0.0 for key in (
            "loss", "data_half", "mse", "penalty", "objective", "f_zero",
            "kkt_top32", "nnz",
        )}
        beta_shapes: list[tuple[int, int]] = []
        sampled_lambda_min = math.inf
        for accumulation in range(args.grad_accum_steps):
            xs, ys, info = stream.next_batch()
            beta_shapes.append((int(info[0]["n"]), int(info[0]["p"])))
            payload = make_canonical_query_payload(xs, ys, cache, sampler, device)
            sampled_lambda_min = min(sampled_lambda_min, float(payload["lambdas"].min()))
            if sampled_lambda_min < LAMBDA_LOW_MIN_CONTRACT - 1e-12:
                raise TrainerError("sampled beta lambda violated 1e-6 floor")
            sync = model.no_sync() if accumulation < 3 else contextlib.nullcontext()
            with sync:
                result = objective_exp125_v58(model, payload, knots)
                if not bool(torch.isfinite(result["loss"])):
                    raise TrainerError("nonfinite beta loss")
                (result["loss"] / 4.0).backward()
            for key in beta_metrics:
                beta_metrics[key] += float(result[key].detach()) / 4.0
        beta_group_gradients = {
            "shape": module_grad_norm(core.cond_embed.penalty_shape_encoder),
            "query": module_grad_norm(core.query_mixer),
            "feature": module_grad_norm(core.feature_head),
            "local": module_grad_norm(core.v46_local_mixer),
        }
        beta_total_gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0))
        if not math.isfinite(beta_total_gradient) or beta_total_gradient <= 0.0:
            raise TrainerError("beta gradient is nonfinite or zero")
        beta_optimizer.step()
        beta_scheduler.step()

        selector_optimizer.zero_grad(set_to_none=True)
        was_training = core.training
        core.eval()
        for parameter in core.parameters():
            parameter.grad = None
            parameter.requires_grad_(False)
        selector_losses: list[torch.Tensor] = []
        selector_mses: list[torch.Tensor] = []
        direct_gradients: list[torch.Tensor] = []
        native_gradients: list[torch.Tensor] = []
        d0_gradients: list[torch.Tensor] = []
        lambda_gradients: list[torch.Tensor] = []
        selector_ratios: list[torch.Tensor] = []
        selector_rhos: list[torch.Tensor] = []
        selector_drhos: list[torch.Tensor] = []
        positivity_minimum = math.inf
        selector_shapes: list[tuple[int, int]] = []
        try:
            for accumulation in range(args.selector_grad_accum_steps):
                xs, ys, info = stream.next_batch()
                selector_shapes.append((int(info[0]["n"]), int(info[0]["p"])))
                x_train, y_train, x_test, y_test = split_and_standardize_variable(
                    xs, ys, generator=split_generator, device=device
                )
                with torch.no_grad():
                    encoded = core.encode_dataset(core.precompute(x_train, y_train))
                    dataset_features, krylov_features = summarize_encoded(
                        encoded, 32 * 33 // 2
                    )
                    path_features = selector_path_features(
                        core, encoded, x_train, y_train,
                        fixed_shape, fixed_d0, path_ratios, args.path_chunk,
                    )
                sync = (
                    selector_train.no_sync()
                    if accumulation < args.selector_grad_accum_steps - 1
                    else contextlib.nullcontext()
                )
                with sync:
                    query = selector_train(
                        dataset_features,
                        krylov_features,
                        encoded["lambda_max"].float(),
                        path_features,
                        U=encoded.get("U"),
                        y_context=y_train,
                    )
                    for name in ("penalty_shape_features", "penalty_d0", "lambdas"):
                        query[name].retain_grad()
                    positivity_minimum = min(
                        positivity_minimum,
                        float(query["penalty_positivity_jacobian_min"].min().detach()),
                    )
                    aux = core.decode_queries(
                        encoded,
                        lambdas=query["lambdas"],
                        lambda_max=encoded["lambda_max"].float()[:, None],
                        penalty_shape_features=query["penalty_shape_features"],
                        penalty_d0=query["penalty_d0"],
                        return_aux=True,
                    )
                    if "beta_q" not in aux:
                        raise TrainerError("V58 decode aux lacks actual hard beta_q")
                    beta = aux["beta_q"][:, 0].float()
                    prediction = torch.einsum("bnp,bp->bn", x_test.float(), beta)
                    task_mse = (prediction - y_test.float()).square().mean(-1)
                    selector_loss = torch.log1p(0.5 * task_mse).mean()
                    if not bool(torch.isfinite(selector_loss)):
                        raise TrainerError("nonfinite selector loss")
                    proposal = selector_core.penalty_parameter_head._last_parameter_proposal
                    if proposal is None:
                        raise TrainerError("selector lost its Direct96 proposal tensor")
                    (selector_loss / float(args.selector_grad_accum_steps)).backward()
                if proposal.grad is None:
                    raise TrainerError("Direct96 proposal gradient is missing")
                direct_gradients.append(proposal.grad.detach().float())
                for name, destination in (
                    ("penalty_shape_features", native_gradients),
                    ("penalty_d0", d0_gradients),
                    ("lambdas", lambda_gradients),
                ):
                    if query[name].grad is None:
                        raise TrainerError(f"selector output gradient missing for {name}")
                    destination.append(query[name].grad.detach().float())
                selector_losses.append(selector_loss.detach())
                selector_mses.append(task_mse.detach())
                selector_ratios.append(query["lambda_ratio"].detach())
                selector_rhos.append(query["penalty_rho"].detach())
                selector_drhos.append(query["penalty_drho"].detach())

            direct_matrix = torch.cat(
                [value.reshape(-1, DIRECT_PARAMETER_DIM) for value in direct_gradients], 0
            )
            coordinate_mask = direct_matrix.abs().sum(0) != 0.0
            if int(coordinate_mask.sum()) != DIRECT_PARAMETER_DIM:
                inactive = torch.nonzero(~coordinate_mask).flatten().cpu().tolist()
                raise TrainerError(f"Direct96 gradient coverage failed: inactive={inactive}")
            if not math.isfinite(positivity_minimum) or positivity_minimum <= 0.0:
                raise TrainerError("canonical positivity map Jacobian reached zero/nonfinite")
            direct_stats = {
                "norm": float(direct_matrix.norm()),
                "rms": float(direct_matrix.square().mean().sqrt()),
                "nonzero_fraction": float((direct_matrix != 0.0).float().mean()),
                "active_coordinates": int(coordinate_mask.sum()),
                "total_coordinates": DIRECT_PARAMETER_DIM,
            }
            native_stats = tensor_gradient_stats(native_gradients)
            d0_stats = tensor_gradient_stats(d0_gradients)
            lambda_stats = tensor_gradient_stats(lambda_gradients)
            for label, stats in (
                ("Direct96", direct_stats), ("native196", native_stats),
                ("d0", d0_stats), ("lambda", lambda_stats),
            ):
                if not math.isfinite(float(stats["norm"])) or float(stats["norm"]) <= 0.0:
                    raise TrainerError(f"selector {label} gradient is nonfinite or zero")
            group_stats = {
                "penalty_head": grad_stats(selector_core.penalty_parameter_head.parameters()),
                "lambda_head": grad_stats(selector_core.lambda_head.parameters()),
                "penalty_trunk": grad_stats(selector_core.penalty_net.parameters()),
                "lambda_trunk": grad_stats(selector_core.lambda_net.parameters()),
            }
            if completed_steps + 1 >= 2 and any(
                float(group_stats[name]["norm"]) <= 0.0
                for name in ("penalty_head", "lambda_head", "penalty_trunk", "lambda_trunk")
            ):
                raise TrainerError("one selector head/trunk has zero gradient")
            if any(parameter.grad is not None for parameter in core.parameters()):
                raise TrainerError("frozen beta model received selector parameter gradients")
            selector_total_gradient = float(
                torch.nn.utils.clip_grad_norm_(selector_train.parameters(), 5.0)
            )
            if not math.isfinite(selector_total_gradient):
                raise TrainerError("selector total gradient is nonfinite")
            selector_optimizer.step()
        finally:
            for parameter in core.parameters():
                parameter.requires_grad_(True)
            core.train(was_training)

        completed_steps += 1
        compute_seconds += time.perf_counter() - started
        compute_steps += 1
        all_rho = torch.cat(selector_rhos)
        all_drho = torch.cat(selector_drhos)
        selector_metrics = {
            "selector_objective": float(torch.stack(selector_losses).mean()),
            "selector_heldout_mse": float(torch.cat(selector_mses).mean()),
            "selector_effective_ratio_min": float(torch.cat(selector_ratios).min()),
            "selector_effective_ratio_max": float(torch.cat(selector_ratios).max()),
            "penalty_rho_batch_std_mean": float(all_rho.std(0, unbiased=False).mean()),
            "penalty_drho_batch_std_mean": float(all_drho.std(0, unbiased=False).mean()),
            "penalty_min_drho": float(all_drho.min()),
            "penalty_rho6_error_max": float((all_rho[:, -1] - 1.0).abs().max()),
            "positivity_jacobian_min": positivity_minimum,
            "direct96_gradient": direct_stats,
            "native196_gradient": native_stats,
            "d0_gradient": d0_stats,
            "lambda_gradient": lambda_stats,
            "selector_group_gradients": group_stats,
            "selector_total_gradient_preclip": selector_total_gradient,
        }
        if completed_steps == 1:
            if assert_ddp_synced(core, world, "post_step1_beta") == beta_hash_before:
                raise TrainerError("beta model did not change on first update")
            if assert_ddp_synced(selector_core, world, "post_step1_selector") == selector_hash_before:
                raise TrainerError("selector did not change on first update")

        if completed_steps % args.log_every == 0:
            local_monitor = {
                "rank": rank,
                "catalog_epoch": int(active_pointer["epoch"]),
                "catalog_sha256": active_catalog_sha,
                "stream_reload_count": stream.reload_count,
                "beta_metrics": beta_metrics,
                "beta_total_gradient_preclip": beta_total_gradient,
                "beta_group_gradients": beta_group_gradients,
                "sampled_lambda_min": sampled_lambda_min,
                "beta_shapes": beta_shapes,
                "selector_shapes": selector_shapes,
                **selector_metrics,
            }
            monitors = gather_rank_objects(local_monitor, world)
            emit({
                "kind": "train_update",
                "step": completed_steps,
                "average_compute_seconds_per_step": compute_seconds / compute_steps,
                "rank_monitors": monitors,
            })
            compute_seconds = 0.0
            compute_steps = 0

        if completed_steps % args.reload_every == 0:
            old_catalog_sha = active_catalog_sha
            old_checkpoint_id = previous_checkpoint_id
            active_pointer, active_catalog, rank_entries, catalog_changed = load_current_catalog(
                store,
                rank=rank,
                world=world,
                expected_pipeline_sha256=args.expected_pipeline_sha256,
                active_sha256=active_catalog_sha,
            )
            active_catalog_sha = str(active_pointer["catalog_sha256"])
            if is_main and active_catalog_sha != old_catalog_sha:
                store.acquire_lease(args.trainer_id, active_catalog_sha)
            dist.barrier()
            stream.refresh(rank_entries, catalog_sha256=active_catalog_sha)
            store.write_ack(args.trainer_id, rank, world, active_catalog_sha)
            dist.barrier()

            local_state = {
                "rank": rank,
                "rng": capture_rng(),
                "stream": stream.state_dict(),
                "penalty_cache": cache.state_dict(),
                "sampler": sampler.state_dict(),
                "split_generator_state": split_generator.get_state(),
            }
            rank_states = gather_rank_objects(local_state, world)
            checkpoint_id = f"{args.trainer_id}_step{completed_steps:06d}"
            checkpoint_path = args.out_dir / f"checkpoint_step{completed_steps:06d}.pt"
            if is_main:
                payload = {
                    "completed_steps": completed_steps,
                    "model_state_dict": core.state_dict(),
                    "selector_state_dict": selector_core.state_dict(),
                    "beta_optimizer_state_dict": beta_optimizer.state_dict(),
                    "selector_optimizer_state_dict": selector_optimizer.state_dict(),
                    "beta_scheduler_state_dict": beta_scheduler.state_dict(),
                    "rank_states": rank_states,
                    "catalog_pointer": active_pointer,
                    "catalog_payload": active_catalog,
                    "metadata": {
                        "checkpoint_id": checkpoint_id,
                        "checkpoint_version": CHECKPOINT_VERSION,
                        "trainer_version": TRAINER_VERSION,
                        "model_version": MODEL_VERSION,
                        "basis_version": BASIS_VERSION,
                        "penalty_contract_version": PENALTY_CONTRACT_VERSION,
                        "canonical_penalty_contract": CONTRACT_VERSION,
                        "selector_version": V12_SELECTOR_VERSION,
                        "canonical_cache_version": CACHE_VERSION,
                        "catalog_epoch": int(active_pointer["epoch"]),
                        "catalog_sha256": active_catalog_sha,
                        "catalog_reloaded_before_checkpoint": True,
                        "dataloader_cache_cleared_before_checkpoint": True,
                        "reload_interval_updates": RELOAD_INTERVAL_UPDATES,
                        "world_size": world,
                        "source_hashes": source_hashes,
                        "exact_training_arguments": exact_arguments,
                        "warm_start_sha256": args.warm_start_sha256,
                        "penalty_cache_sha256": args.penalty_cache_sha256,
                    },
                }
                atomic_torch_save(payload, checkpoint_path)
                observed_checkpoint_sha = file_sha256(checkpoint_path)
                reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if (
                    int(reloaded["completed_steps"]) != completed_steps
                    or reloaded["metadata"]["catalog_sha256"] != active_catalog_sha
                    or state_dict_sha256(reloaded["model_state_dict"]) != module_sha256(core)
                    or state_dict_sha256(reloaded["selector_state_dict"]) != module_sha256(selector_core)
                ):
                    raise TrainerError("written checkpoint failed strict reload verification")
                store.register_checkpoint_pin(
                    checkpoint_id,
                    active_catalog_sha,
                    checkpoint_sha256=observed_checkpoint_sha,
                    resumable=True,
                    reload_verified=True,
                )
                if old_checkpoint_id and old_checkpoint_id != checkpoint_id:
                    store.mark_checkpoint_nonresumable(old_checkpoint_id)
                if active_catalog_sha != old_catalog_sha:
                    store.release_lease(
                        args.trainer_id,
                        expected_catalog_sha256=old_catalog_sha,
                    )
                emit({
                    "kind": "catalog_reload_checkpoint",
                    "step": completed_steps,
                    "catalog_changed": catalog_changed,
                    "old_catalog_sha256": old_catalog_sha,
                    "catalog_epoch": int(active_pointer["epoch"]),
                    "catalog_sha256": active_catalog_sha,
                    "catalog_files": len(active_catalog["files"]),
                    "catalog_bytes": int(active_catalog["accounting"]["total_bytes"]),
                    "stream_cache_cleared_all_ranks": True,
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": observed_checkpoint_sha,
                    "checkpoint_id": checkpoint_id,
                    "reload_verified": True,
                })
            previous_checkpoint_id = checkpoint_id
            dist.barrier()

    emit({
        "kind": "complete",
        "step": completed_steps,
        "catalog_epoch": int(active_pointer["epoch"]),
        "catalog_sha256": active_catalog_sha,
    })
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
