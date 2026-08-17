#!/usr/bin/env python3
"""Train Kalbi V3 with a fresh V17-K32S32 selector on P<=100K data.

The random beta-query path and learned selector both call the same
``canonical_penalty.build_penalty`` function.  The trainer adopts one immutable
catalog on every 100-update boundary, clears every task-file cache even when
CURRENT is unchanged, checkpoints only after that reload, and pins the exact
catalog needed for resume before an older lease can be released.  The beta
solver uses lambda-independent K32/S32 A/B ISIS ridge trajectories and
evaluates all 32 random queries in one call.  V17 consumes only standardized X/y through its fixed
32-stage, K=32 residual trajectory and emits one lambda plus one Direct96 penalty.
To preserve beta-solver throughput, beta is updated every step while V17 is updated on
a configurable sparse schedule.  The selector objective is held-out prediction
loss only; no lambda teacher, ranking loss, or auxiliary target is added.
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
    KALBI_BETA_BOUND,
    KALBI_RIDGE,
    KALBI_STAGES,
    KALBI_TOPK,
    LOCAL_ISIS_GRADIENT_CONTRACT,
    MODEL_VERSION,
    PENALTY_CONTRACT_VERSION,
    SHARED_SUPPORT_CONTRACT,
    build_model,
)
from .rolling_catalog import CatalogStore, CapacityPolicy
from .selector_v17_k32s32 import (
    DirectPenaltyMLPQuerySelectorV17StageGain,
    V17_CANDIDATE_POOL,
    V17_HIDDEN_DIM,
    V17_MAX_PROBE_ROWS,
    V17_SELECTOR_VERSION,
    V17_STAGES,
    V17_STAGE_GAIN_RATE,
    V17_TOPK,
)
from .train import (
    ROOT_DEPENDENCY_PATHS,
    TrainerError,
    atomic_torch_save,
    capture_rng,
    make_sampler,
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
    tensor_gradient_stats,
)
from .utils import fixed_ridge_numerical_diagnostics, native196_to_expanded388
from train_fpk import _build_X_y_tensors, _compute_lambda_max, _load_yaml


TRAINER_VERSION = (
    "kalbi_v3_v17k32s32_penaltymlp_tabiclv2_noisy_y_v6_p100k_world8_"
    "effbatch32_abfilm_learnedthreshold_k32s32_rolling100gb_reload100_v1"
)
CHECKPOINT_VERSION = (
    "kalbi_v3_v17k32s32_penaltymlp_tabiclv2_noisy_y_v6_p100k_"
    "effbatch32_checkpoint_v1"
)
JOINT_SOURCE_CHECKPOINT_VERSIONS = (
    "kalbi_v3_v17k32s32_penaltymlp_p100k_effbatch32_checkpoint_v2",
    "kalbi_v3_v17k32s32_penaltymlp_tabiclv2only_p100k_effbatch32_checkpoint_v1",
)
BETA_SOURCE_CHECKPOINT_VERSIONS = (
    "kalbi_v2_v17_stagegain_p100k_effbatch32_checkpoint_v3",
    "kalbi_v3_v17k32s32_p100k_effbatch32_checkpoint_v1",
)
SCM_CONTRACTS = {
    "tabicl_v2": (
        "tabiclv2_graph_scm_xy_groupdrop095_sampled_noise_v6",
        "official_tabiclv2_graph_scm_y_plus_sampled_error_clip10_rescale_v1",
    ),
}
RESPONSE_NOISE_CONTRACT = (
    "capped_exp1_scale_mixture_centered_unit_error_additive_v1"
)
RESPONSE_NOISE_FAMILIES = (
    "gaussian",
    "student_t",
    "laplace",
    "contaminated_gaussian",
    "centered_lognormal",
    "heteroskedastic_gaussian",
)
RESPONSE_NOISE_SIGMA_CAP = 3.0
EXPECTED_WORLD_SIZE = 8
EXPECTED_GRAD_ACCUM_STEPS = 4
EXPECTED_SELECTOR_GRAD_ACCUM_STEPS = 4
RELOAD_INTERVAL_UPDATES = 100
LAMBDA_LOW_MIN_CONTRACT = 1e-6
SELECTOR_RATIO_MIN = 1e-6
SELECTOR_RATIO_MAX = 1.0
SELECTOR_OBJECTIVE = "mean(log1p(heldout_mse/2))_v17_prediction_only_v1"
MAX_CONSECUTIVE_GLOBAL_PENALTY_ZERO_UPDATES = 25
EXPECTED_MODEL_PARAMETER_COUNT = 3_588_547
EXPECTED_SELECTOR_PARAMETER_COUNT = 3_327_073


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
    observed_priors: set[str] = set()
    for entry in catalog.get("files", []):
        prior = str(entry.get("prior"))
        if prior not in SCM_CONTRACTS:
            raise TrainerError(f"rolling catalog contains an unsupported prior: {prior}")
        observed_priors.add(prior)
        if int(entry.get("num_tasks", -1)) != 1:
            raise TrainerError("rolling catalog shard does not contain one task")
        n = int(entry.get("n", -1))
        p = int(entry.get("p", -1))
        if not (32 <= n <= 20_000 and 50 <= p <= 100_000 and n * p <= 24_000_000):
            raise TrainerError(f"catalog task shape outside contract: N={n} P={p}")
        metadata = entry.get("metadata", {})
        semantic = metadata.get("semantic_contract", {}) if isinstance(metadata, Mapping) else {}
        dataset_contract, response_contract = SCM_CONTRACTS[prior]
        if semantic.get("contract") != dataset_contract:
            raise TrainerError("catalog shard lacks the SCM-response semantic certificate")
        if semantic.get("target_contract") != response_contract:
            raise TrainerError("catalog shard response contract drift")
        if semantic.get("response_noise_contract") != RESPONSE_NOISE_CONTRACT:
            raise TrainerError("catalog shard response-noise contract drift")
        family_id = int(semantic.get("response_noise_family_id", -1))
        family = str(semantic.get("response_noise_family", ""))
        if not 0 <= family_id < len(RESPONSE_NOISE_FAMILIES):
            raise TrainerError("catalog shard response-noise family id is invalid")
        if family != RESPONSE_NOISE_FAMILIES[family_id]:
            raise TrainerError("catalog shard response-noise family is inconsistent")
        sigma_raw = float(semantic.get("response_noise_sigma_raw", float("nan")))
        sigma = float(semantic.get("response_noise_sigma", float("nan")))
        if not math.isfinite(sigma_raw) or sigma_raw < 0.0:
            raise TrainerError("catalog shard raw response-noise sigma is invalid")
        if not math.isfinite(sigma) or not 0.0 <= sigma <= RESPONSE_NOISE_SIGMA_CAP:
            raise TrainerError("catalog shard response-noise sigma is invalid")
        if not math.isclose(
            sigma,
            min(sigma_raw, RESPONSE_NOISE_SIGMA_CAP),
            rel_tol=0.0,
            abs_tol=2e-6,
        ):
            raise TrainerError("catalog shard response-noise sigma cap is inconsistent")
        latent_correlation = float(
            semantic.get("latent_signal_noisy_correlation", float("nan"))
        )
        latent_r2 = float(
            semantic.get("latent_signal_noisy_r2_ceiling", float("nan"))
        )
        if not math.isfinite(latent_correlation) or abs(latent_correlation) > 1.0 + 3e-6:
            raise TrainerError("catalog shard latent/noisy correlation is invalid")
        if not math.isclose(
            latent_r2, latent_correlation * latent_correlation,
            rel_tol=0.0, abs_tol=5e-6,
        ):
            raise TrainerError("catalog shard latent/noisy R2 certificate is invalid")
        reconstruction_error = float(
            semantic.get("noisy_response_reconstruction_max_abs_error", float("nan"))
        )
        if not math.isfinite(reconstruction_error) or reconstruction_error > 3e-6:
            raise TrainerError("catalog shard noisy-response reconstruction failed")
        response_mean = float(semantic.get("scm_response_mean_max_abs", float("nan")))
        response_std_min = float(semantic.get("scm_response_std_min", float("nan")))
        response_std_max = float(semantic.get("scm_response_std_max", float("nan")))
        if not math.isfinite(response_mean) or response_mean > 2e-6:
            raise TrainerError("catalog shard SCM response is not centered")
        if not (
            math.isfinite(response_std_min)
            and math.isfinite(response_std_max)
            and abs(response_std_min - 1.0) <= 2e-6
            and abs(response_std_max - 1.0) <= 2e-6
        ):
            raise TrainerError("catalog shard SCM response is not unit scale")
    if observed_priors != set(SCM_CONTRACTS):
        raise TrainerError(f"rolling catalog prior set drift: {observed_priors}")


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
    # Rank 0 authenticates CURRENT and hashes every shard after a transition.
    # Each rank checks only its assigned paths before ACKing the catalog.
    store.verify_catalog_files(
        {"data_root": catalog["data_root"], "files": entries}, hashes=False
    )
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
        # For an unchanged reload, assigned-path checks are distributed by rank.
        if changed:
            store.verify_catalog_files(catalog, hashes=True)
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

    VERSION = "v14_scmmixture_50_50_catalog_single_task_stream_v1"

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
        self.prior_orders: dict[str, np.ndarray] = {}
        self.prior_cursors: dict[str, int] = {}
        for prior in SCM_CONTRACTS:
            order = np.asarray(
                [index for index, entry in enumerate(values) if entry.get("prior") == prior],
                dtype=np.int64,
            )
            if order.size == 0:
                raise TrainerError(f"rank {self.rank} received no {prior} shards")
            self.rng.shuffle(order)
            self.prior_orders[prior] = order
            self.prior_cursors[prior] = 0
        self.next_prior_index = 0
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
                "scm_response_mean", "scm_response_std", "batch_seed",
            }
            if not required.issubset(archive.files):
                raise TrainerError(f"runtime shard lacks certified arrays: {path}")
            x_raw = archive["X"]
            y_raw = archive["y"]
            corr = np.asarray(archive["exact_max_abs_corr"], dtype=np.float32)
            threshold = np.asarray(archive["corr_threshold"], dtype=np.float32)
            response_mean = np.asarray(archive["scm_response_mean"], dtype=np.float32)
            response_std = np.asarray(archive["scm_response_std"], dtype=np.float32)
            batch_seed = archive["batch_seed"]
            if x_raw.dtype != np.float16 or y_raw.dtype != np.float32:
                raise TrainerError(f"runtime shard dtype drift: {path}")
            expected_x = (1, int(entry["n"]), int(entry["p"]))
            expected_y = (1, int(entry["n"]))
            if x_raw.shape != expected_x or y_raw.shape != expected_y:
                raise TrainerError(
                    f"runtime shard shape drift: {path} X={x_raw.shape} y={y_raw.shape}"
                )
            if (
                corr.shape != (1,)
                or threshold.shape != (1,)
                or response_mean.shape != (1,)
                or response_std.shape != (1,)
                or batch_seed.shape != (1,)
                or batch_seed.dtype != np.uint64
            ):
                raise TrainerError("runtime shard certificate shape drift")
            realized_mean = float(np.mean(y_raw[0].astype(np.float64)))
            realized_std = float(np.std(y_raw[0].astype(np.float64)))
            if not (
                np.isfinite(x_raw).all()
                and np.isfinite(y_raw).all()
                and np.isfinite(corr).all()
                and float(corr[0]) < float(threshold[0])
                and abs(float(threshold[0]) - 0.95) <= 1e-6
                and abs(float(response_mean[0]) - realized_mean) <= 2e-6
                and abs(float(response_std[0]) - realized_std) <= 2e-6
                and abs(realized_mean) <= 2e-6
                and abs(realized_std - 1.0) <= 2e-6
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
        priors = tuple(SCM_CONTRACTS)
        prior = priors[self.next_prior_index % len(priors)]
        self.next_prior_index += 1
        order = self.prior_orders[prior]
        cursor = self.prior_cursors[prior]
        if cursor >= len(order):
            self.rng.shuffle(order)
            cursor = 0
        index = int(order[cursor])
        self.prior_cursors[prior] = cursor + 1
        x, y = self._load(index)
        entry = self.entries[index]
        return [x], [y], [{
            "shard_id": int(entry["shard_id"]),
            "path": str(entry["absolute_path"]),
            "n": int(entry["n"]),
            "p": int(entry["p"]),
            "catalog_sha256": self.catalog_sha256,
            "prior": str(entry["prior"]),
        }]

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "seed": self.seed,
            "rank": self.rank,
            "catalog_sha256": self.catalog_sha256,
            "identities": [list(value) for value in self.identities],
            "prior_orders": {
                prior: order.tolist() for prior, order in self.prior_orders.items()
            },
            "prior_cursors": dict(self.prior_cursors),
            "next_prior_index": self.next_prior_index,
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
        prior_orders = state.get("prior_orders")
        prior_cursors = state.get("prior_cursors")
        if not isinstance(prior_orders, Mapping) or not isinstance(prior_cursors, Mapping):
            raise TrainerError("rolling stream resume mixture state is missing")
        restored_orders: dict[str, np.ndarray] = {}
        restored_cursors: dict[str, int] = {}
        for prior in SCM_CONTRACTS:
            expected_indices = np.asarray(
                [index for index, entry in enumerate(self.entries) if entry.get("prior") == prior],
                dtype=np.int64,
            )
            order = np.asarray(prior_orders.get(prior), dtype=np.int64)
            if order.shape != expected_indices.shape or not np.array_equal(
                np.sort(order), np.sort(expected_indices)
            ):
                raise TrainerError(f"rolling stream resume order is invalid for {prior}")
            cursor = int(prior_cursors.get(prior, -1))
            if not 0 <= cursor <= len(order):
                raise TrainerError(f"rolling stream resume cursor is invalid for {prior}")
            restored_orders[prior] = order
            restored_cursors[prior] = cursor
        next_prior_index = int(state.get("next_prior_index", -1))
        if next_prior_index < 0:
            raise TrainerError("rolling stream resume next-prior index is invalid")
        self.prior_orders = restored_orders
        self.prior_cursors = restored_cursors
        self.next_prior_index = next_prior_index
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
        "lambda_normalizer_uses_fresh_v58_initialization": True,
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
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sha256")
    parser.add_argument("--initialize-beta-from", type=Path)
    parser.add_argument("--initialize-beta-sha256")
    parser.add_argument("--initialize-joint-from", type=Path)
    parser.add_argument("--initialize-joint-sha256")
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--beta-lr", type=float, default=1e-4)
    parser.add_argument("--feature-lr-scale", type=float, default=1.0)
    parser.add_argument("--query-lr-scale", type=float, default=0.5)
    parser.add_argument("--shape-lr-scale", type=float, default=0.5)
    parser.add_argument("--local-lr-scale", type=float, default=0.25)
    parser.add_argument("--other-lr-scale", type=float, default=0.5)
    parser.add_argument("--clip-feature", type=float, default=5.0)
    parser.add_argument("--clip-query", type=float, default=1.0)
    parser.add_argument("--clip-shape", type=float, default=1.0)
    parser.add_argument("--clip-local", type=float, default=1.0)
    parser.add_argument("--clip-other", type=float, default=2.0)
    parser.add_argument("--selector-lr", type=float, default=3e-4)
    parser.add_argument("--direct-head-lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--batch-per-rank", type=int, default=1)
    parser.add_argument(
        "--grad-accum-steps", type=int, default=EXPECTED_GRAD_ACCUM_STEPS
    )
    parser.add_argument(
        "--selector-grad-accum-steps",
        type=int,
        default=EXPECTED_SELECTOR_GRAD_ACCUM_STEPS,
    )
    parser.add_argument("--selector-update-every", type=int, default=4)
    parser.add_argument("--selector-hidden", type=int, default=V17_HIDDEN_DIM)
    parser.add_argument("--cache-files", type=int, default=2)
    parser.add_argument("--reload-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=202607170230)
    parser.add_argument(
        "--smoke-small-catalog",
        action="store_true",
        help="use the isolated small-catalog capacity policy for DDP preflight",
    )
    parser.add_argument(
        "--beta-only",
        action="store_true",
        help=(
            "update only Kalbi V3 while carrying V17 state unchanged"
        ),
    )
    args = parser.parse_args()

    initialization_modes = sum((
        bool(args.scratch),
        args.resume is not None,
        args.initialize_beta_from is not None,
        args.initialize_joint_from is not None,
    ))
    if initialization_modes != 1:
        parser.error(
            "choose exactly one of --scratch, --resume, --initialize-beta-from, "
            "or --initialize-joint-from"
        )
    if args.resume is not None and not args.resume_sha256:
        parser.error("--resume requires --resume-sha256")
    if args.initialize_beta_from is not None and not args.initialize_beta_sha256:
        parser.error("--initialize-beta-from requires --initialize-beta-sha256")
    if args.initialize_joint_from is not None and not args.initialize_joint_sha256:
        parser.error("--initialize-joint-from requires --initialize-joint-sha256")
    if args.batch_per_rank != 1:
        parser.error("variable-shape rolling training requires batch-per-rank=1")
    if args.selector_hidden != V17_HIDDEN_DIM:
        parser.error(f"V17 selector hidden dimension is fixed at {V17_HIDDEN_DIM}")
    if args.grad_accum_steps != EXPECTED_GRAD_ACCUM_STEPS:
        parser.error("8-GPU global beta task batch is fixed at 32")
    if args.selector_grad_accum_steps != EXPECTED_SELECTOR_GRAD_ACCUM_STEPS:
        parser.error("8-GPU global V17 selector task batch is fixed at 32")
    if args.selector_update_every <= 0:
        parser.error("selector update interval must be positive")
    if args.reload_every != RELOAD_INTERVAL_UPDATES:
        parser.error(
            f"catalog/dataloader reload interval is fixed at "
            f"{RELOAD_INTERVAL_UPDATES} updates"
        )
    if args.reload_every % args.selector_update_every != 0:
        parser.error("selector interval must divide the reload interval")
    if args.checkpoint_every != args.reload_every:
        parser.error("checkpoints must align exactly with catalog reload boundaries")
    if args.steps <= 0 or args.steps % args.reload_every != 0:
        parser.error(
            f"steps must be a positive multiple of {RELOAD_INTERVAL_UPDATES}"
        )
    for value, label in (
        (args.feature_lr_scale, "feature LR scale"),
        (args.query_lr_scale, "query LR scale"),
        (args.shape_lr_scale, "shape LR scale"),
        (args.local_lr_scale, "local LR scale"),
        (args.other_lr_scale, "other LR scale"),
        (args.clip_feature, "feature clip"),
        (args.clip_query, "query clip"),
        (args.clip_shape, "shape clip"),
        (args.clip_local, "local clip"),
        (args.clip_other, "other clip"),
    ):
        if value <= 0.0:
            parser.error(f"{label} must be positive")
    if args.stop_after is not None and not 1 <= args.stop_after <= args.steps:
        parser.error("stop-after must lie in [1, steps]")

    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world = int(os.environ.get("WORLD_SIZE", -1))
    if rank < 0 or local_rank < 0 or world != EXPECTED_WORLD_SIZE:
        raise TrainerError("P100K rolling trainer requires exactly eight ranks")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    def init_trace(stage: str) -> None:
        if os.environ.get("FM_INIT_TRACE") == "1":
            print(
                json.dumps(
                    {"kind": "init_trace", "rank": rank, "stage": stage},
                    sort_keys=True,
                ),
                flush=True,
            )

    init_trace("distributed_initialized")

    if not args.penalty_cache.is_file() or file_sha256(
        args.penalty_cache
    ) != args.penalty_cache_sha256:
        raise TrainerError("canonical cache SHA256 mismatch")
    init_trace("penalty_cache_hash_verified")

    random.seed(args.seed + rank)
    np.random.seed((args.seed + rank) % (2**32 - 1))
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    policy = (
        CapacityPolicy(
            active_min_bytes=20_000_000,
            active_max_bytes=40_000_000,
            prior_min_bytes=10_000_000,
            prior_max_bytes=20_000_000,
            hard_cap_bytes=100_000_000,
            min_free_bytes=0,
            world_size=world,
        )
        if args.smoke_small_catalog
        else CapacityPolicy(
            active_min_bytes=49_000_000_000,
            active_max_bytes=102_000_000_000,
            prior_min_bytes=49_000_000_000,
            prior_max_bytes=102_000_000_000,
            hard_cap_bytes=130_000_000_000,
            min_free_bytes=300_000_000_000,
            world_size=world,
        )
    )
    store = CatalogStore(
        args.rolling_control_root,
        args.rolling_data_root,
        policy=policy,
        priors=("tabicl_v2",),
    )
    resume_checkpoint = None
    beta_source_checkpoint = None
    joint_source_checkpoint = None
    if args.resume is not None:
        if file_sha256(args.resume) != args.resume_sha256:
            raise TrainerError("resume checkpoint SHA256 mismatch")
        resume_checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        if resume_checkpoint.get("metadata", {}).get("checkpoint_version") != CHECKPOINT_VERSION:
            raise TrainerError("resume checkpoint version mismatch")
    elif args.initialize_beta_from is not None:
        if file_sha256(args.initialize_beta_from) != args.initialize_beta_sha256:
            raise TrainerError("beta-source checkpoint SHA256 mismatch")
        beta_source_checkpoint = torch.load(
            args.initialize_beta_from, map_location="cpu", weights_only=False
        )
        source_metadata = beta_source_checkpoint.get("metadata", {})
        if source_metadata.get("checkpoint_version") not in (
            BETA_SOURCE_CHECKPOINT_VERSIONS
        ):
            raise TrainerError("beta-source checkpoint version mismatch")
        source_contract = {
            "model_version": MODEL_VERSION,
            "basis_version": BASIS_VERSION,
            "shared_support_contract": SHARED_SUPPORT_CONTRACT,
            "canonical_penalty_contract": CONTRACT_VERSION,
            "canonical_cache_version": CACHE_VERSION,
            "world_size": world,
        }
        for key, expected in source_contract.items():
            if source_metadata.get(key) != expected:
                raise TrainerError(f"beta-source contract mismatch for {key}")
    elif args.initialize_joint_from is not None:
        if file_sha256(args.initialize_joint_from) != args.initialize_joint_sha256:
            raise TrainerError("joint-source checkpoint SHA256 mismatch")
        joint_source_checkpoint = torch.load(
            args.initialize_joint_from, map_location="cpu", weights_only=False
        )
        source_metadata = joint_source_checkpoint.get("metadata", {})
        if source_metadata.get("checkpoint_version") not in (
            JOINT_SOURCE_CHECKPOINT_VERSIONS
        ):
            raise TrainerError("joint-source checkpoint version mismatch")
        source_contract = {
            "model_version": MODEL_VERSION,
            "basis_version": BASIS_VERSION,
            "shared_support_contract": SHARED_SUPPORT_CONTRACT,
            "canonical_penalty_contract": CONTRACT_VERSION,
            "canonical_cache_version": CACHE_VERSION,
            "selector_version": V17_SELECTOR_VERSION,
            "world_size": world,
        }
        for key, expected in source_contract.items():
            if source_metadata.get(key) != expected:
                raise TrainerError(f"joint-source contract mismatch for {key}")

    initial_checkpoint = (
        resume_checkpoint or beta_source_checkpoint or joint_source_checkpoint
    )
    importing_beta_checkpoint = beta_source_checkpoint is not None
    importing_joint_checkpoint = joint_source_checkpoint is not None

    if resume_checkpoint is not None:
        initialization_provenance = dict(
            resume_checkpoint["metadata"]["initialization_provenance"]
        )
    elif beta_source_checkpoint is not None:
        initialization_provenance = {
            "mode": "beta_checkpoint_import_selector_scratch",
            "beta_source_checkpoint": str(args.initialize_beta_from.resolve()),
            "beta_source_checkpoint_sha256": args.initialize_beta_sha256,
            "beta_source_completed_steps": int(
                beta_source_checkpoint["completed_steps"]
            ),
            "beta_model_state_imported": True,
            "beta_optimizer_state_imported": True,
            "beta_scheduler_state_imported": True,
            "rank_stream_state_imported": True,
            "selector_state_imported": False,
            "selector_optimizer_state_imported": False,
            "selector_update_counter_reset": True,
        }
    elif joint_source_checkpoint is not None:
        initialization_provenance = {
            "mode": "joint_checkpoint_import_fresh_tabiclv2_noisy_y_v6_catalog",
            "joint_source_checkpoint": str(args.initialize_joint_from.resolve()),
            "joint_source_checkpoint_sha256": args.initialize_joint_sha256,
            "joint_source_completed_steps": int(
                joint_source_checkpoint["completed_steps"]
            ),
            "beta_model_state_imported": True,
            "beta_optimizer_state_imported": True,
            "beta_scheduler_state_imported": True,
            "selector_state_imported": True,
            "selector_optimizer_state_imported": True,
            "rank_stream_state_imported": False,
            "catalog_state_imported": False,
        }
    else:
        initialization_provenance = {
            "mode": "full_scratch",
            "beta_model_state_imported": False,
            "selector_state_imported": False,
        }
    run_initialization_mode = str(initialization_provenance["mode"])
    run_was_full_scratch = run_initialization_mode == "full_scratch"
    run_beta_imported = run_initialization_mode in {
        "beta_checkpoint_import_selector_scratch",
        "joint_checkpoint_import_fresh_tabiclv2_noisy_y_v6_catalog",
    }
    run_selector_imported = (
        run_initialization_mode
        == "joint_checkpoint_import_fresh_tabiclv2_noisy_y_v6_catalog"
    )
    run_selector_started_from_scratch = run_initialization_mode in {
        "full_scratch",
        "beta_checkpoint_import_selector_scratch",
    }

    if initial_checkpoint is None or importing_joint_checkpoint:
        active_pointer, active_catalog, rank_entries, _changed = load_current_catalog(
            store,
            rank=rank,
            world=world,
            expected_pipeline_sha256=args.expected_pipeline_sha256,
            active_sha256=None,
        )
    else:
        package = {
            "pointer": initial_checkpoint["catalog_pointer"],
            "catalog": initial_checkpoint["catalog_payload"],
        }
        active_pointer, active_catalog, rank_entries = adopt_catalog_package(
            store,
            package,
            rank=rank,
            world=world,
            expected_pipeline_sha256=args.expected_pipeline_sha256,
            rank0_hashes_verified=False,
        )
    init_trace("catalog_adopted")

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
    init_trace("penalty_cache_loaded")
    sampler = make_sampler(_load_yaml(str(args.config)), args.seed + 1_000_003 * rank + 211)
    if not math.isclose(
        float(sampler.cfg.lambda_low_min), LAMBDA_LOW_MIN_CONTRACT,
        rel_tol=0.0, abs_tol=1e-15,
    ):
        raise TrainerError("beta lambda floor contract drift")

    rank_state = None
    if resume_checkpoint is not None or beta_source_checkpoint is not None:
        rank_states = initial_checkpoint["rank_states"]
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
        state=(
            None if rank_state is None else rank_state["stream"]
        ),
    )
    split_generator = torch.Generator(device="cpu").manual_seed(
        args.seed + 3_000_017 * rank
    )
    if rank_state is not None:
        split_generator.set_state(rank_state["split_generator_state"])

    config = _load_yaml(str(args.config))
    init_trace("model_build_started")
    model = build_model(config, device)
    init_trace("model_build_finished")
    if (
        int(model.kalbi_isis_topk) != KALBI_TOPK
        or int(model.kalbi_isis_stages) != KALBI_STAGES
        or float(model.kalbi_isis_ridge) != KALBI_RIDGE
    ):
        raise TrainerError("Kalbi V3 K/S/ridge architecture contract drift")
    model_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    if model_parameter_count != EXPECTED_MODEL_PARAMETER_COUNT:
        raise TrainerError(
            "Kalbi V3 model parameter-count drift: "
            f"expected {EXPECTED_MODEL_PARAMETER_COUNT}, got {model_parameter_count}"
        )
    shape_refit_holder: list[Any] = [None]
    if initial_checkpoint is None:
        if is_main:
            shape_refit_holder[0] = refit_canonical_shape_normalizers(
                model, cache, sampler
            )
        dist.broadcast_object_list(shape_refit_holder, src=0)
        for tensor in model.state_dict().values():
            dist.broadcast(tensor, src=0)
    init_trace("shape_normalizers_ready")

    selector_state_holder: list[Any] = [None]
    selector_initialization_holder: list[Any] = [None]
    initialize_selector_from_scratch = (
        resume_checkpoint is None and joint_source_checkpoint is None
    )
    if initialize_selector_from_scratch and is_main:
        init_trace("selector_cpu_initialization_started")
        torch.manual_seed(args.seed + 17)
        selector_cpu = DirectPenaltyMLPQuerySelectorV17StageGain(
            hidden=args.selector_hidden,
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
            "initialization": "new_v17_k32s32_penaltymlp_selector_from_seed",
            "reason": "new penalty-specific MLP selector branch",
        }
        init_trace("selector_cpu_initialization_finished")
    dist.broadcast_object_list(selector_state_holder, src=0)
    dist.broadcast_object_list(selector_initialization_holder, src=0)
    torch.manual_seed(args.seed + 17)
    selector = DirectPenaltyMLPQuerySelectorV17StageGain(
        hidden=args.selector_hidden,
        lambda_min_ratio=SELECTOR_RATIO_MIN,
        lambda_max_ratio=SELECTOR_RATIO_MAX,
        initial_lambda_ratio=0.1,
        derivative_floor=1e-6,
        squareplus_scale=0.02,
        direct_head_init_std=1e-3,
    )
    if initialize_selector_from_scratch:
        selector.load_state_dict(selector_state_holder[0], strict=True)
    else:
        selector.load_state_dict(
            initial_checkpoint["selector_state_dict"], strict=True
        )
    selector.to(device)
    selector_parameter_count = sum(
        parameter.numel() for parameter in selector.parameters()
    )
    if selector_parameter_count != EXPECTED_SELECTOR_PARAMETER_COUNT:
        raise TrainerError(
            "V17-K32S32 parameter-count drift: "
            f"expected {EXPECTED_SELECTOR_PARAMETER_COUNT}, "
            f"got {selector_parameter_count}"
        )
    init_trace("selector_on_device")
    if initial_checkpoint is not None:
        model.load_state_dict(initial_checkpoint["model_state_dict"], strict=True)

    model = DDP(
        model, device_ids=[local_rank], output_device=local_rank,
        broadcast_buffers=True, find_unused_parameters=True, init_sync=True,
    )
    selector_train = DDP(
        selector, device_ids=[local_rank], output_device=local_rank,
        broadcast_buffers=True, find_unused_parameters=True, init_sync=True,
    )
    init_trace("ddp_wrappers_ready")
    core = model.module
    selector_core = selector_train.module
    beta_parameter_groups: dict[str, list[torch.nn.Parameter]] = {
        "trajectory": [],
        "condition": [],
        "shape": [],
        "base": [],
        "other": [],
    }
    for name, parameter in core.named_parameters():
        if name.startswith("kalbi_ab_head."):
            group = "trajectory"
        elif name.startswith("cond_embed.penalty_shape_encoder."):
            group = "shape"
        elif name.startswith("cond_embed."):
            group = "condition"
        elif name.startswith("base_head."):
            group = "base"
        else:
            group = "other"
        beta_parameter_groups[group].append(parameter)
    grouped_parameter_ids = {
        id(parameter)
        for parameters in beta_parameter_groups.values()
        for parameter in parameters
    }
    if grouped_parameter_ids != {id(parameter) for parameter in core.parameters()}:
        raise TrainerError("Kalbi V3 beta optimizer parameter partition is incomplete")
    beta_lr_scales = {
        "trajectory": args.feature_lr_scale,
        "condition": args.query_lr_scale,
        "shape": args.shape_lr_scale,
        "base": args.other_lr_scale,
        "other": args.other_lr_scale,
    }
    beta_clip_limits = {
        "trajectory": args.clip_feature,
        "condition": args.clip_query,
        "shape": args.clip_shape,
        "base": args.clip_other,
        "other": args.clip_other,
    }
    active_beta_group_names = tuple(
        name for name, parameters in beta_parameter_groups.items() if parameters
    )
    if not active_beta_group_names:
        raise TrainerError("Kalbi V3 optimizer has no nonempty parameter group")
    beta_optimizer = torch.optim.AdamW(
        [
            {
                "params": beta_parameter_groups[name],
                "lr": args.beta_lr * beta_lr_scales[name],
                "name": name,
            }
            for name in active_beta_group_names
        ],
        lr=args.beta_lr,
        weight_decay=1e-4,
        betas=(0.9, 0.95),
    )
    penalty_branch_parameters = [
        *selector_core.penalty_mlp.parameters(),
        *selector_core.penalty_parameter_head.parameters(),
    ]
    penalty_branch_ids = {id(parameter) for parameter in penalty_branch_parameters}
    selector_other_parameters = [
        parameter for parameter in selector_train.parameters()
        if id(parameter) not in penalty_branch_ids
    ]
    selector_optimizer = torch.optim.AdamW(
        [
            {"params": selector_other_parameters, "lr": args.selector_lr,
             "name": "selector_context_lambda_trunks"},
            {"params": penalty_branch_parameters, "lr": args.direct_head_lr,
             "name": "penalty_mlp_direct96_head"},
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
    if initial_checkpoint is not None:
        beta_optimizer.load_state_dict(initial_checkpoint["beta_optimizer_state_dict"])
        beta_scheduler.load_state_dict(initial_checkpoint["beta_scheduler_state_dict"])
        optimizer_to_device(beta_optimizer, device)
        if resume_checkpoint is not None or joint_source_checkpoint is not None:
            selector_optimizer.load_state_dict(
                initial_checkpoint["selector_optimizer_state_dict"]
            )
            optimizer_to_device(selector_optimizer, device)
        completed_steps = int(initial_checkpoint["completed_steps"])
        if resume_checkpoint is not None or joint_source_checkpoint is not None:
            previous_checkpoint_id = initial_checkpoint["metadata"].get(
                "checkpoint_id"
            )

    knots = canonical_grid(device=device)

    if is_main:
        if resume_checkpoint is None:
            if args.out_dir.exists():
                if not importing_beta_checkpoint and not importing_joint_checkpoint:
                    raise TrainerError(
                        f"refusing existing non-resume output directory: {args.out_dir}"
                    )
                if any(args.out_dir.glob("checkpoint_step*.pt")):
                    raise TrainerError(
                        "beta import cannot target a directory containing checkpoints"
                    )
            else:
                args.out_dir.mkdir(parents=True, exist_ok=False)
        elif args.resume.resolve().parent != args.out_dir.resolve():
            raise TrainerError("resume checkpoint must remain in its output directory")
    dist.barrier()
    init_trace("output_directory_ready")
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
        "selector_v12_canonical_base": file_sha256(
            Path(__file__).with_name("selector_v12_canonical.py")
        ),
        "selector_v17_k32s32": file_sha256(
            Path(__file__).with_name("selector_v17_k32s32.py")
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
        "beta_lr_scales": beta_lr_scales,
        "beta_clip_limits": beta_clip_limits,
        "warmup_steps": args.warmup_steps,
        "selector_lr": args.selector_lr,
        "direct_head_lr": args.direct_head_lr,
        "selector_hidden": args.selector_hidden,
        "selector_stages": V17_STAGES,
        "selector_topk": V17_TOPK,
        "selector_candidate_pool": V17_CANDIDATE_POOL,
        "selector_max_probe_rows": V17_MAX_PROBE_ROWS,
        "selector_stage_gain_rate": V17_STAGE_GAIN_RATE,
        "selector_prediction_loss_only": True,
        "lambda_teacher_used": False,
        "beta_only": bool(args.beta_only),
        "beta_query_chunk": None,
        "model_parameter_count": model_parameter_count,
        "selector_parameter_count": selector_parameter_count,
        "kalbi_shared_support_contract": SHARED_SUPPORT_CONTRACT,
        "kalbi_isis_topk": int(core.kalbi_isis_topk),
        "kalbi_isis_stages": int(core.kalbi_isis_stages),
        "kalbi_isis_ridge": float(core.kalbi_isis_ridge),
        "kalbi_local_basis": None,
        "kalbi_terminal_threshold": {
            "type": "featurewise_penalty_aware_learned_multiplier",
            "base": "lambda_times_d0_divided_by_gram_diagonal",
            "head_inputs": "Krylov_base_plus_ISIS_A_B_plus_local_stats_plus_FiLM",
            "applies_outside_isis_support": True,
            "canonical_beta_bound": KALBI_BETA_BOUND,
        },
        "batch_per_rank": args.batch_per_rank,
        "grad_accum_steps": args.grad_accum_steps,
        "selector_grad_accum_steps": args.selector_grad_accum_steps,
        "selector_update_every": args.selector_update_every,
        "reload_every": args.reload_every,
        "checkpoint_every": args.checkpoint_every,
        "log_every": args.log_every,
        "seed": args.seed,
        "smoke_small_catalog": bool(args.smoke_small_catalog),
        "initialization_provenance": initialization_provenance,
        "run_initialized_from_scratch": run_was_full_scratch,
        "beta_initialized_from_checkpoint": run_beta_imported,
        "selector_initialized_from_scratch": run_selector_started_from_scratch,
        "selector_initialized_from_checkpoint": run_selector_imported,
    }
    if resume_checkpoint is not None:
        metadata = resume_checkpoint["metadata"]
        required_resume = {
            "trainer_version": TRAINER_VERSION,
            "checkpoint_version": CHECKPOINT_VERSION,
            "canonical_penalty_contract": CONTRACT_VERSION,
            "canonical_cache_version": CACHE_VERSION,
            "selector_version": V17_SELECTOR_VERSION,
            "source_hashes": source_hashes,
            "exact_training_arguments": exact_arguments,
            "world_size": world,
        }
        for key, expected in required_resume.items():
            if metadata.get(key) != expected:
                raise TrainerError(f"resume contract mismatch for {key}")

    event_kind = (
        "resume"
        if resume_checkpoint is not None
        else "joint_checkpoint_import_start"
        if joint_source_checkpoint is not None
        else "beta_checkpoint_import_start"
        if importing_beta_checkpoint
        else "start"
    )
    emit({
        "kind": event_kind,
        "trainer_version": TRAINER_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_version": MODEL_VERSION,
        "basis_version": BASIS_VERSION,
        "shared_support_contract": SHARED_SUPPORT_CONTRACT,
        "penalty_contract_version": PENALTY_CONTRACT_VERSION,
        "canonical_penalty_contract": CONTRACT_VERSION,
        "selector_version": V17_SELECTOR_VERSION,
        "canonical_cache": cache_report,
        "scm_prior_contracts": {
            prior: {"dataset_contract": values[0], "response_contract": values[1]}
            for prior, values in SCM_CONTRACTS.items()
        },
        "mixture_contract": (
            "TabICLv2-only variable-shape P100K rolling catalog with sampled response noise"
        ),
        "selector_objective": SELECTOR_OBJECTIVE,
        "selector_input_contract": "standardized_X_y_only",
        "selector_uses_beta_solver_input": False,
        "selector_stages": V17_STAGES,
        "selector_topk": V17_TOPK,
        "selector_candidate_pool": V17_CANDIDATE_POOL,
        "selector_max_probe_rows": V17_MAX_PROBE_ROWS,
        "selector_stage_gain_rate": V17_STAGE_GAIN_RATE,
        "selector_parameter_count": selector_parameter_count,
        "selector_prediction_loss_only": True,
        "lambda_teacher_used": False,
        "lambda_teacher_weight": 0.0,
        "training_mode": "beta_only" if args.beta_only else "joint_beta_selector",
        "selector_update_enabled": not args.beta_only,
        "selector_update_every": args.selector_update_every,
        "world_size": world,
        "global_beta_task_batch": world * args.grad_accum_steps,
        "global_selector_task_batch": world * args.selector_grad_accum_steps,
        "smoke_small_catalog": bool(args.smoke_small_catalog),
        "reload_interval_updates": args.reload_every,
        "active_catalog_epoch": int(active_pointer["epoch"]),
        "active_catalog_sha256": active_catalog_sha,
        "active_catalog_files": len(active_catalog["files"]),
        "active_catalog_bytes": int(active_catalog["accounting"]["total_bytes"]),
        "shape_normalizer_refit": shape_refit_holder[0],
        "selector_initialization": selector_initialization_holder[0],
        "initialization_provenance": initialization_provenance,
        "run_initialized_from_scratch": run_was_full_scratch,
        "beta_initialized_from_checkpoint": run_beta_imported,
        "selector_initialized_from_scratch": run_selector_started_from_scratch,
        "selector_initialized_from_checkpoint": run_selector_imported,
        "resuming": resume_checkpoint is not None,
        "fixed_ridge_numerics": fixed_ridge_numerical_diagnostics(),
        "source_hashes": source_hashes,
        "exact_training_arguments": exact_arguments,
        "completed_steps": completed_steps,
    })

    beta_hash_before = assert_ddp_synced(core, world, "initial_beta")
    selector_hash_before = assert_ddp_synced(selector_core, world, "initial_selector")
    compute_seconds = 0.0
    compute_steps = 0
    trainer_state_source = resume_checkpoint or joint_source_checkpoint
    consecutive_global_penalty_zero_updates = (
        0
        if trainer_state_source is None
        else int(trainer_state_source.get("trainer_state", {}).get(
            "consecutive_global_penalty_zero_updates", 0
        ))
    )
    selector_updates = (
        0
        if trainer_state_source is None
        else int(trainer_state_source.get("trainer_state", {}).get(
            "selector_updates", 0
        ))
    )
    selector_zero_gradient_skips = (
        0
        if trainer_state_source is None
        else int(trainer_state_source.get("trainer_state", {}).get(
            "selector_zero_gradient_skips", 0
        ))
    )

    target_steps = args.steps if args.stop_after is None else args.stop_after
    while completed_steps < target_steps:
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
            sync = (
                model.no_sync()
                if accumulation < args.grad_accum_steps - 1
                else contextlib.nullcontext()
            )
            with sync:
                result = objective_exp125_v58(model, payload, knots)
                if not bool(torch.isfinite(result["loss"])):
                    raise TrainerError("nonfinite beta loss")
                (result["loss"] / float(args.grad_accum_steps)).backward()
            for key in beta_metrics:
                beta_metrics[key] += (
                    float(result[key].detach()) / float(args.grad_accum_steps)
                )
        beta_group_gradient_tensors = {
            name: torch.nn.utils.clip_grad_norm_(
                beta_parameter_groups[name],
                beta_clip_limits[name],
                error_if_nonfinite=False,
            ).detach()
            for name in active_beta_group_names
        }
        beta_total_gradient_preclip = float(torch.stack([
            gradient.float().square()
            for gradient in beta_group_gradient_tensors.values()
        ]).sum().sqrt())
        if (
            not math.isfinite(beta_total_gradient_preclip)
            or beta_total_gradient_preclip <= 0.0
        ):
            raise TrainerError("beta gradient is nonfinite or zero")
        beta_group_gradients_preclip: dict[str, float] = {}
        beta_group_gradients_postclip: dict[str, float] = {}
        if (completed_steps + 1) % args.log_every == 0:
            beta_group_gradients_preclip = {
                name: float(gradient)
                for name, gradient in beta_group_gradient_tensors.items()
            }
            beta_group_gradients_postclip = {
                name: min(beta_group_gradients_preclip[name], beta_clip_limits[name])
                for name in beta_group_gradient_tensors
            }
        beta_optimizer.step()
        beta_scheduler.step()

        selector_due = (
            not args.beta_only
            and (completed_steps + 1) % args.selector_update_every == 0
        )
        if not selector_due:
            completed_steps += 1
            if completed_steps == 1:
                if assert_ddp_synced(
                    core, world, "post_step1_beta"
                ) == beta_hash_before:
                    raise TrainerError("beta model did not change on first update")
            compute_seconds += time.perf_counter() - started
            compute_steps += 1
            training_mode = (
                "beta_only"
                if args.beta_only
                else "joint_beta_v17_sparse_schedule"
            )

            if completed_steps % args.log_every == 0:
                local_monitor = {
                    "rank": rank,
                    "catalog_epoch": int(active_pointer["epoch"]),
                    "catalog_sha256": active_catalog_sha,
                    "stream_reload_count": stream.reload_count,
                    "beta_metrics": beta_metrics,
                    "beta_total_gradient_preclip": beta_total_gradient_preclip,
                    "beta_group_gradients_preclip": beta_group_gradients_preclip,
                    "beta_group_gradients_postclip": beta_group_gradients_postclip,
                    "sampled_lambda_min": sampled_lambda_min,
                    "kalbi_fixed_ridge_numerics": (
                        fixed_ridge_numerical_diagnostics()
                    ),
                    "beta_shapes": beta_shapes,
                    "training_mode": training_mode,
                    "selector_update_performed": False,
                    "selector_updates": selector_updates,
                    "lambda_teacher_used": False,
                }
                monitors = gather_rank_objects(local_monitor, world)
                emit({
                    "kind": "train_update",
                    "step": completed_steps,
                    "training_mode": training_mode,
                    "selector_update_performed": False,
                    "selector_updates": selector_updates,
                    "lambda_teacher_used": False,
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
                        "trainer_state": {
                            "consecutive_global_penalty_zero_updates": (
                                consecutive_global_penalty_zero_updates
                            ),
                            "selector_frozen_beta_only": True,
                            "selector_updates": selector_updates,
                            "selector_zero_gradient_skips": (
                                selector_zero_gradient_skips
                            ),
                        },
                        "catalog_pointer": active_pointer,
                        "catalog_payload": active_catalog,
                        "metadata": {
                            "checkpoint_id": checkpoint_id,
                            "checkpoint_version": CHECKPOINT_VERSION,
                            "trainer_version": TRAINER_VERSION,
                            "model_version": MODEL_VERSION,
                            "basis_version": BASIS_VERSION,
                            "shared_support_contract": SHARED_SUPPORT_CONTRACT,
                            "penalty_contract_version": PENALTY_CONTRACT_VERSION,
                            "canonical_penalty_contract": CONTRACT_VERSION,
                            "selector_version": V17_SELECTOR_VERSION,
                            "training_mode": "beta_only",
                            "selector_update_enabled": False,
                            "selector_state_carried_forward_unchanged": True,
                            "selector_prediction_loss_only": True,
                            "lambda_teacher_used": False,
                            "lambda_teacher_weight": 0.0,
                            "canonical_cache_version": CACHE_VERSION,
                            "catalog_epoch": int(active_pointer["epoch"]),
                            "catalog_sha256": active_catalog_sha,
                            "catalog_reloaded_before_checkpoint": True,
                            "dataloader_cache_cleared_before_checkpoint": True,
                            "reload_interval_updates": RELOAD_INTERVAL_UPDATES,
                            "world_size": world,
                            "source_hashes": source_hashes,
                            "exact_training_arguments": exact_arguments,
                            "penalty_cache_sha256": args.penalty_cache_sha256,
                            "initialization_provenance": initialization_provenance,
                            "scratch_initialization": run_was_full_scratch,
                            "beta_initialized_from_checkpoint": run_beta_imported,
                            "selector_initialized_from_scratch": (
                                run_selector_started_from_scratch
                            ),
                        },
                    }
                    atomic_torch_save(payload, checkpoint_path)
                    observed_checkpoint_sha = file_sha256(checkpoint_path)
                    reloaded = torch.load(
                        checkpoint_path, map_location="cpu", weights_only=False
                    )
                    if (
                        int(reloaded["completed_steps"]) != completed_steps
                        or reloaded["metadata"]["catalog_sha256"] != active_catalog_sha
                        or state_dict_sha256(reloaded["model_state_dict"])
                        != module_sha256(core)
                        or state_dict_sha256(reloaded["selector_state_dict"])
                        != module_sha256(selector_core)
                    ):
                        raise TrainerError(
                            "written beta-only checkpoint failed strict reload verification"
                        )
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
                        "training_mode": "beta_only",
                        "selector_update_performed": False,
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
            continue

        selector_optimizer.zero_grad(set_to_none=True)
        was_training = core.training
        core.eval()
        for parameter in core.parameters():
            parameter.grad = None
            parameter.requires_grad_(False)
        selector_losses: list[torch.Tensor] = []
        selector_hard_losses: list[torch.Tensor] = []
        selector_mses: list[torch.Tensor] = []
        direct_gradients: list[torch.Tensor] = []
        native_gradients: list[torch.Tensor] = []
        d0_gradients: list[torch.Tensor] = []
        lambda_gradients: list[torch.Tensor] = []
        selector_ratios: list[torch.Tensor] = []
        selector_beta_nonzero_fractions: list[torch.Tensor] = []
        selector_threshold_means: list[torch.Tensor] = []
        selector_threshold_mins: list[torch.Tensor] = []
        selector_threshold_maxes: list[torch.Tensor] = []
        selector_log_multiplier_means: list[torch.Tensor] = []
        selector_log_multiplier_mins: list[torch.Tensor] = []
        selector_log_multiplier_maxes: list[torch.Tensor] = []
        selector_prethreshold_max_abs: list[torch.Tensor] = []
        selector_beta_max_abs: list[torch.Tensor] = []
        trajectory_active_stage_counts: list[torch.Tensor] = []
        trajectory_residual_energy_final: list[torch.Tensor] = []
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
                sync = (
                    selector_train.no_sync()
                    if accumulation < args.selector_grad_accum_steps - 1
                    else contextlib.nullcontext()
                )
                with sync:
                    query = selector_train(
                        x_train,
                        y_train,
                        encoded["lambda_max"].float(),
                        return_representation=True,
                    )
                    trajectory = query["v17_trajectory"]
                    active_count = trajectory["stage_active"].sum(-1).float()
                    if not bool((active_count >= 1.0).all()):
                        raise TrainerError("V17 trajectory produced no active stage")
                    trajectory_active_stage_counts.append(active_count.detach())
                    trajectory_residual_energy_final.append(
                        trajectory["residual_energy_ratio"][:, -1].detach().float()
                    )
                    for name in (
                        "penalty_shape_features", "penalty_d0", "lambdas",
                    ):
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
                        raise TrainerError("Kalbi V3 decode aux lacks beta_q")
                    required_threshold_aux = (
                        "kalbi_v3_threshold_q",
                        "kalbi_v3_threshold_log_multiplier_q",
                        "kalbi_v3_prethreshold_beta_q",
                    )
                    if any(name not in aux for name in required_threshold_aux):
                        raise TrainerError("Kalbi V3 threshold telemetry is missing")
                    beta = aux["beta_q"][:, 0].float()
                    threshold = aux["kalbi_v3_threshold_q"][:, 0].float()
                    log_multiplier = aux[
                        "kalbi_v3_threshold_log_multiplier_q"
                    ][:, 0].float()
                    prethreshold_beta = aux[
                        "kalbi_v3_prethreshold_beta_q"
                    ][:, 0].float()
                    if not all(
                        bool(torch.isfinite(value).all())
                        for value in (beta, threshold, log_multiplier, prethreshold_beta)
                    ):
                        raise TrainerError("Kalbi V3 threshold path is nonfinite")
                    selector_beta_nonzero_fractions.append(
                        (beta != 0.0).float().mean().detach().reshape(1)
                    )
                    selector_threshold_means.append(
                        threshold.mean().detach().reshape(1)
                    )
                    selector_threshold_mins.append(
                        threshold.min().detach().reshape(1)
                    )
                    selector_threshold_maxes.append(
                        threshold.max().detach().reshape(1)
                    )
                    selector_log_multiplier_means.append(
                        log_multiplier.mean().detach().reshape(1)
                    )
                    selector_log_multiplier_mins.append(
                        log_multiplier.min().detach().reshape(1)
                    )
                    selector_log_multiplier_maxes.append(
                        log_multiplier.max().detach().reshape(1)
                    )
                    selector_prethreshold_max_abs.append(
                        prethreshold_beta.abs().max().detach().reshape(1)
                    )
                    selector_beta_max_abs.append(
                        beta.abs().max().detach().reshape(1)
                    )
                    prediction = torch.einsum("bnp,bp->bn", x_test.float(), beta)
                    task_mse = (prediction - y_test.float()).square().mean(-1)
                    hard_loss = torch.log1p(0.5 * task_mse).mean()
                    selector_loss = hard_loss
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
                selector_hard_losses.append(hard_loss.detach())
                selector_mses.append(task_mse.detach())
                selector_ratios.append(query["lambda_ratio"].detach())
                selector_rhos.append(query["penalty_rho"].detach())
                selector_drhos.append(query["penalty_drho"].detach())

            direct_matrix = torch.cat(
                [value.reshape(-1, DIRECT_PARAMETER_DIM) for value in direct_gradients], 0
            )
            if not bool(torch.isfinite(direct_matrix).all()):
                raise TrainerError("Direct96 output gradient is nonfinite")
            global_coordinate_abs = direct_matrix.abs().sum(0)
            dist.all_reduce(global_coordinate_abs, op=dist.ReduceOp.SUM)
            global_coordinate_mask = global_coordinate_abs != 0.0
            global_active_coordinates = int(global_coordinate_mask.sum())
            local_hard_beta_nonzero_tasks = int(sum(
                bool(float(value.item()) > 0.0)
                for value in selector_beta_nonzero_fractions
            ))
            global_hard_beta_nonzero_tasks_tensor = torch.tensor(
                local_hard_beta_nonzero_tasks,
                dtype=torch.int64,
                device=device,
            )
            dist.all_reduce(
                global_hard_beta_nonzero_tasks_tensor,
                op=dist.ReduceOp.SUM,
            )
            global_hard_beta_nonzero_tasks = int(
                global_hard_beta_nonzero_tasks_tensor.item()
            )
            # A task may legitimately emit an all-zero hard beta, in which case
            # its penalty output gradient is exactly zero.  Partial global
            # coordinate death is not legitimate under squareplus and remains
            # fail-closed. A globally zero penalty gradient can be legitimate
            # for a hard-zero beta task and is monitored across updates.
            if global_active_coordinates not in (0, DIRECT_PARAMETER_DIM):
                inactive = torch.nonzero(
                    ~global_coordinate_mask
                ).flatten().cpu().tolist()
                raise TrainerError(
                    f"global Direct96 partial coordinate death: inactive={inactive}"
                )
            if not math.isfinite(positivity_minimum) or positivity_minimum <= 0.0:
                raise TrainerError("canonical positivity map Jacobian reached zero/nonfinite")
            direct_stats = {
                "norm": float(direct_matrix.norm()),
                "rms": float(direct_matrix.square().mean().sqrt()),
                "nonzero_fraction": float((direct_matrix != 0.0).float().mean()),
                "local_active_coordinates": int(
                    (direct_matrix.abs().sum(0) != 0.0).sum()
                ),
                "global_active_coordinates": global_active_coordinates,
                "total_coordinates": DIRECT_PARAMETER_DIM,
            }
            native_stats = tensor_gradient_stats(native_gradients)
            d0_stats = tensor_gradient_stats(d0_gradients)
            lambda_stats = tensor_gradient_stats(lambda_gradients)
            for label, stats in (
                ("Direct96", direct_stats), ("native196", native_stats),
                ("d0", d0_stats), ("lambda", lambda_stats),
            ):
                if not math.isfinite(float(stats["norm"])):
                    raise TrainerError(f"selector {label} gradient is nonfinite")
            selector_diagnostic_due = (
                selector_updates == 0
                or (completed_steps + 1) % args.log_every == 0
            )
            group_stats = {
                "penalty_mlp": grad_stats(
                    selector_core.penalty_mlp.parameters()
                ),
                "penalty_head": grad_stats(
                    selector_core.penalty_parameter_head.parameters()
                ),
            }
            if selector_diagnostic_due:
                group_stats.update({
                    "lambda_head": grad_stats(selector_core.lambda_head.parameters()),
                    "row_query": grad_stats(selector_core.row_query.parameters()),
                    "row_key": grad_stats(selector_core.row_key.parameters()),
                    "interaction": grad_stats(selector_core.interaction_proj.parameters()),
                    "residual_coefficients": grad_stats(
                        selector_core.residual_coefficient_proj.parameters()
                    ),
                    "marginal_coefficients": grad_stats(
                        selector_core.marginal_coefficient_proj.parameters()
                    ),
                    "stage_metadata": grad_stats(
                        selector_core.stage_metadata_proj.parameters()
                    ),
                    "fusion": grad_stats(selector_core.fusion_mlp.parameters()),
                })
            for name, stats in group_stats.items():
                if not math.isfinite(float(stats["norm"])):
                    raise TrainerError(f"selector parameter group {name} is nonfinite")
            if (
                selector_diagnostic_due
                and global_hard_beta_nonzero_tasks > 0
                and any(
                    float(group_stats[name]["norm"]) <= 0.0
                    for name in (
                        "lambda_head", "row_query", "row_key", "interaction",
                        "residual_coefficients", "marginal_coefficients",
                        "stage_metadata", "fusion"
                    )
                )
            ):
                raise TrainerError(
                    "V17 context/lambda selector gradient is zero"
                )
            global_penalty_zero = any(
                float(group_stats[name]["norm"]) <= 0.0
                for name in ("penalty_mlp", "penalty_head")
            )
            if global_penalty_zero:
                consecutive_global_penalty_zero_updates += 1
            else:
                consecutive_global_penalty_zero_updates = 0
            if (
                consecutive_global_penalty_zero_updates
                > MAX_CONSECUTIVE_GLOBAL_PENALTY_ZERO_UPDATES
            ):
                raise TrainerError(
                    "V17 penalty branch stayed globally zero under prediction-only loss"
                )
            if any(parameter.grad is not None for parameter in core.parameters()):
                raise TrainerError("frozen beta model received selector parameter gradients")
            selector_total_gradient = float(
                torch.nn.utils.clip_grad_norm_(selector_train.parameters(), 5.0)
            )
            if not math.isfinite(selector_total_gradient):
                raise TrainerError("selector total gradient is nonfinite")
            selector_update_applied = selector_total_gradient > 0.0
            if not selector_update_applied:
                if (
                    global_hard_beta_nonzero_tasks != 0
                    or global_active_coordinates != 0
                ):
                    raise TrainerError(
                        "selector gradient is zero despite a nonzero hard-beta path"
                    )
                selector_zero_gradient_skips += 1
                emit({
                    "kind": "selector_update_skipped",
                    "step": completed_steps + 1,
                    "reason": "global_hard_beta_is_exact_zero",
                    "global_hard_beta_nonzero_tasks": 0,
                    "global_hard_beta_tasks": (
                        world * args.selector_grad_accum_steps
                    ),
                    "selector_updates": selector_updates,
                    "selector_zero_gradient_skips": (
                        selector_zero_gradient_skips
                    ),
                })
            else:
                selector_optimizer.step()
                selector_updates += 1
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
            "selector_hard_prediction_loss": float(
                torch.stack(selector_hard_losses).mean()
            ),
            "selector_heldout_mse": float(torch.cat(selector_mses).mean()),
            "selector_effective_ratio_min": float(torch.cat(selector_ratios).min()),
            "selector_effective_ratio_max": float(torch.cat(selector_ratios).max()),
            "selected_hard_beta_nonzero_fraction": float(
                torch.cat(selector_beta_nonzero_fractions).mean()
            ),
            "kalbi_v3_threshold_mean": float(
                torch.cat(selector_threshold_means).mean()
            ),
            "kalbi_v3_threshold_min": float(
                torch.cat(selector_threshold_mins).min()
            ),
            "kalbi_v3_threshold_max": float(
                torch.cat(selector_threshold_maxes).max()
            ),
            "kalbi_v3_log_threshold_multiplier_mean": float(
                torch.cat(selector_log_multiplier_means).mean()
            ),
            "kalbi_v3_log_threshold_multiplier_min": float(
                torch.cat(selector_log_multiplier_mins).min()
            ),
            "kalbi_v3_log_threshold_multiplier_max": float(
                torch.cat(selector_log_multiplier_maxes).max()
            ),
            "kalbi_v3_prethreshold_max_abs": float(
                torch.cat(selector_prethreshold_max_abs).max()
            ),
            "kalbi_v3_beta_max_abs": float(
                torch.cat(selector_beta_max_abs).max()
            ),
            "kalbi_v3_beta_bound": KALBI_BETA_BOUND,
            "v17_active_stages_min": float(
                torch.cat(trajectory_active_stage_counts).min()
            ),
            "v17_active_stages_max": float(
                torch.cat(trajectory_active_stage_counts).max()
            ),
            "v17_final_residual_energy_mean": float(
                torch.cat(trajectory_residual_energy_final).mean()
            ),
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
            "consecutive_global_penalty_zero_updates": (
                consecutive_global_penalty_zero_updates
            ),
            "selector_total_gradient_preclip": selector_total_gradient,
            "selector_update_performed": selector_update_applied,
            "selector_update_skipped_global_hard_zero": (
                not selector_update_applied
            ),
            "selector_updates": selector_updates,
            "selector_zero_gradient_skips": selector_zero_gradient_skips,
            "global_hard_beta_nonzero_tasks": global_hard_beta_nonzero_tasks,
            "global_hard_beta_tasks": world * args.selector_grad_accum_steps,
        }
        if selector_update_applied and selector_updates == 1:
            if assert_ddp_synced(
                selector_core, world, "post_first_selector_update"
            ) == selector_hash_before:
                raise TrainerError("selector did not change on first scheduled update")

        if completed_steps % args.log_every == 0:
            local_monitor = {
                "rank": rank,
                "catalog_epoch": int(active_pointer["epoch"]),
                "catalog_sha256": active_catalog_sha,
                "stream_reload_count": stream.reload_count,
                "beta_metrics": beta_metrics,
                "beta_total_gradient_preclip": beta_total_gradient_preclip,
                "beta_group_gradients_preclip": beta_group_gradients_preclip,
                "beta_group_gradients_postclip": beta_group_gradients_postclip,
                "sampled_lambda_min": sampled_lambda_min,
                "kalbi_fixed_ridge_numerics": (
                    fixed_ridge_numerical_diagnostics()
                ),
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
                    "trainer_state": {
                        "consecutive_global_penalty_zero_updates": (
                            consecutive_global_penalty_zero_updates
                        ),
                        "selector_updates": selector_updates,
                        "selector_zero_gradient_skips": (
                            selector_zero_gradient_skips
                        ),
                    },
                    "catalog_pointer": active_pointer,
                    "catalog_payload": active_catalog,
                    "metadata": {
                        "checkpoint_id": checkpoint_id,
                        "checkpoint_version": CHECKPOINT_VERSION,
                        "trainer_version": TRAINER_VERSION,
                        "model_version": MODEL_VERSION,
                        "basis_version": BASIS_VERSION,
                        "shared_support_contract": SHARED_SUPPORT_CONTRACT,
                        "penalty_contract_version": PENALTY_CONTRACT_VERSION,
                        "canonical_penalty_contract": CONTRACT_VERSION,
                        "selector_version": V17_SELECTOR_VERSION,
                        "selector_input_contract": "standardized_X_y_only",
                        "selector_stages": V17_STAGES,
                        "selector_topk": V17_TOPK,
                        "selector_candidate_pool": V17_CANDIDATE_POOL,
                        "selector_max_probe_rows": V17_MAX_PROBE_ROWS,
                        "selector_stage_gain_rate": V17_STAGE_GAIN_RATE,
                        "selector_update_every": args.selector_update_every,
                        "selector_prediction_loss_only": True,
                        "lambda_teacher_used": False,
                        "lambda_teacher_weight": 0.0,
                        "canonical_cache_version": CACHE_VERSION,
                        "catalog_epoch": int(active_pointer["epoch"]),
                        "catalog_sha256": active_catalog_sha,
                        "catalog_reloaded_before_checkpoint": True,
                        "dataloader_cache_cleared_before_checkpoint": True,
                        "reload_interval_updates": RELOAD_INTERVAL_UPDATES,
                        "world_size": world,
                        "source_hashes": source_hashes,
                        "exact_training_arguments": exact_arguments,
                        "penalty_cache_sha256": args.penalty_cache_sha256,
                        "initialization_provenance": initialization_provenance,
                        "scratch_initialization": run_was_full_scratch,
                        "beta_initialized_from_checkpoint": run_beta_imported,
                        "selector_initialized_from_scratch": (
                            run_selector_started_from_scratch
                        ),
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
