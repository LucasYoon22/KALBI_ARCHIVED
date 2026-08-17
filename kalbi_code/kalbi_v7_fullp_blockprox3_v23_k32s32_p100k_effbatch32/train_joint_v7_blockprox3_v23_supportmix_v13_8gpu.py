#!/usr/bin/env python3
"""Jointly train KALBI V7 and independent-raw-lambda V23 on rolling data.

The random beta-query path and learned selector both call the same
``canonical_penalty.build_penalty`` function.  The trainer adopts one immutable
catalog on every 100-update boundary, clears every task-file cache even when
CURRENT is unchanged, checkpoints only after that reload, and pins the exact
catalog needed for resume before an older lease can be released.  The beta
solver preserves the V3 K32/S32 A/B path and adds three matrix-free exact
full-P block-coordinate Direct96 proximal sweeps. It evaluates all 32 random
queries in one call. V23 consumes only standardized context-train X/y, rescoring every
valid feature at each of its 32 residual-ISIS stages. It retains V20's
canonical Direct96 penalty and V22's fixed-calibrated representation,
cumulative-support conditioning geometry, selected-support correlation
quantiles, and log(N)/log(P)/log(lambda_max). Its public lambda is emitted
directly on the beta solver's raw support; d0 and max(rho') are telemetry only.
The selector objective is held-out prediction loss only.
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
    KALBI_BLOCK_PROX_BLOCK_SIZE,
    KALBI_BLOCK_PROX_SWEEPS,
    KALBI_EXACT_PG_BACKTRACK_FACTOR,
    KALBI_EXACT_PG_CURVATURE_MARGIN,
    KALBI_EXACT_PG_DATA_MARGIN,
    KALBI_EXACT_PG_FEATURE_CHUNK,
    KALBI_EXACT_PG_MAX_BACKTRACKS,
    KALBI_EXACT_PG_RITZ_SAFETY,
    KALBI_EXACT_PG_STEPS,
    KALBI_RIDGE,
    KALBI_STAGES,
    KALBI_TOPK,
    LOCAL_ISIS_GRADIENT_CONTRACT,
    MODEL_VERSION,
    PENALTY_CONTRACT_VERSION,
    SHARED_SUPPORT_CONTRACT,
    V3_SOURCE_MODEL_VERSION,
    build_model,
)
from .rolling_catalog import CatalogStore, CapacityPolicy
from .selector_v20_yconditioned import V20_SELECTOR_VERSION
from .selector_v22_v20geometry_scaled import (
    load_v20_state_dict,
    V19_FEATURE_CHUNK_SIZE,
    V22_SCALER_VERSION,
    V22_SELECTOR_VERSION,
)
from .selector_v23_independent_rawlambda import (
    DirectPenaltyMLPQuerySelectorV23IndependentRawLambda,
    load_v22_state_without_lambda_head,
    V17_HIDDEN_DIM,
    V17_MAX_PROBE_ROWS,
    V17_STAGES,
    V17_STAGE_GAIN_RATE,
    V17_TOPK,
    V23_INITIAL_RAW_LAMBDA,
    V23_LAMBDA_OUTPUT_CONTRACT,
    V23_RAW_LAMBDA_MAX,
    V23_RAW_LAMBDA_MIN,
    V23_SELECTOR_VERSION,
)
from .train import (
    ROOT_DEPENDENCY_PATHS,
    TrainerError,
    atomic_torch_save,
    capture_rng,
    make_sampler,
    objective_kalbi_v7_blockprox3,
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
    "kalbi_v7_v23_supportmix_v13_p100k_world8_effbatch32_"
    "fullP_blockprox3_block4096_rolling100gb_reload100_rawLambdaDirect_v1"
)
CHECKPOINT_VERSION = (
    "kalbi_v7_v23_supportmix_v13_fullP_blockprox3_block4096_"
    "p100k_effbatch32_checkpoint_v1"
)
V7_BETA_SOURCE_CHECKPOINT_VERSION = (
    "kalbi_v6_v23_v22geometry_independentraw_k32s32_exactDirect96proxPG10_"
    "eta4x_tabiclv2_expvarnoise_v11_p100k_effbatch32_checkpoint_v1"
)
V7_SELECTOR_SOURCE_CHECKPOINT_VERSION = (
    "selectoronly_v23_frozen_kalbi_v6_k32s32_exactDirect96proxPG10_eta4x_"
    "tabiclv2_expvarnoise_v11_p100k_effbatch32_heldoutFhatFzero_checkpoint_v2"
)
V7_SELECTOR_SOURCE_TRAINING_MODE = "selector_only_frozen_beta"
V7_SOURCE_MODEL_VERSION = (
    "kalbi_v6_krylov_abfilm_threshold_exactproxpg10_eta4x_k32_stages32_v1"
)
V7_SOURCE_BASIS_VERSION = (
    "global_krylov_ridge_plus_ab_trajectory_threshold_plus_exactproxpg10_v6"
)
V7_SOURCE_SHARED_SUPPORT_CONTRACT = (
    "centered_corr_A_B_ridge1e-6_supportlocal_correction_global_threshold_"
    "fullP_exactgrad_exactDirect96proxPG10_eta4x_refinement_v6"
)
V22_SOURCE_TRAINER_VERSION = (
    "kalbi_v6_v22_v20geometry_fixedscaled_k32s32_tabiclv2_v11_p100k_"
    "world8_effbatch32_exactDirect96proxPG10_eta4x_rolling100gb_reload100_"
    "generalizedMaxSlopeLambda_v1"
)
V22_SOURCE_CHECKPOINT_VERSION = (
    "kalbi_v6_v22_v20geometry_fixedscaled_k32s32_exactDirect96proxPG10_"
    "eta4x_tabiclv2_expvarnoise_v11_p100k_effbatch32_checkpoint_v1"
)
V21_SOURCE_TRAINER_VERSION = (
    "kalbi_v6_v21_k32s32_dimensionconcat_derivative96_tabiclv2_v11_p100k_"
    "world8_effbatch32_exactDirect96proxPG10_eta4x_rolling100gb_reload100_"
    "rawLambdaDirect_v1"
)
V21_SOURCE_CHECKPOINT_VERSION = (
    "kalbi_v6_v21_k32s32_exactDirect96proxPG10_eta4x_"
    "tabiclv2_expvarnoise_v11_p100k_effbatch32_rawLambdaDirect_checkpoint_v1"
)
V21_SOURCE_SELECTOR_VERSION = (
    "v21_k32s32_fullp_isis_dimensionconcat_yconditioned_"
    "rawlambda1e-6to10_derivative96_softplus_trapezoid_rho6_v1"
)
V23_FUSION_BRANCHES = (
    "residual_gram",
    "beta_trajectory",
    "stage_metadata",
    "context_y",
    "support_geometry",
    "support_correlation_quantiles",
    "dimension_scale",
)
V20_SOURCE_TRAINER_VERSION = (
    "kalbi_v6_v20y_k32s32_cumulativebeta_tabiclv2_expvarnoise_v11_p100k_"
    "world8_effbatch32_exactDirect96proxPG10_eta4x_rolling100gb_reload100_"
    "generalizedMaxSlopeLambda_v2"
)
V20_SOURCE_CHECKPOINT_VERSION = (
    "kalbi_v6_v20y_k32s32_exactDirect96proxPG10_eta4x_"
    "tabiclv2_expvarnoise_v11_p100k_effbatch32_generalizedMaxSlope_"
    "checkpoint_v2"
)
LEGACY_V1_TRAINER_VERSION = (
    "kalbi_v6_v20y_k32s32_cumulativebeta_tabiclv2_expvarnoise_v11_p100k_"
    "world8_effbatch32_exactDirect96proxPG10_eta4x_rolling100gb_reload100_v1"
)
LEGACY_V1_CHECKPOINT_VERSION = (
    "kalbi_v6_v20y_k32s32_exactDirect96proxPG10_eta4x_"
    "tabiclv2_expvarnoise_v11_p100k_effbatch32_checkpoint_v1"
)
LEGACY_V1_SELECTOR_VERSION = (
    "v20_k32s32_direct96_residualtrajectory_fullp_isis32x32_"
    "stagegain0p5pct_hidden512_probe2048_chunk4096_noreplacement_"
    "localres_localmarg_cumulativejointbeta_ylinear_meanstdmax_"
    "penalty_preln_reslinear3_nopostnorm_v1"
)
LEGACY_V1_CHANGED_SOURCE_HASHES = {
    "trainer": "ee132145caf304453c531ff2a3d35c812eb192bd51b5027bc13de3fd3c60d6eb",
    "selector_v10": "8f5e49d03bb413345ba47937774f41346fa682591b734e2493d5ebfb9f13eb2a",
    "selector_v19_k32s32": (
        "af5f268a366babc0f5b550df797565beaab6b050ddf054cb25448ada3371be58"
    ),
    "selector_v20_yconditioned": (
        "ff4a3f2c8bf63a8c46c63b8f5bc292b9b1790b034c5a1706ce5f7e82be2f72de"
    ),
}
SELECTOR_LAMBDA_SCALING_CONTRACT = V23_LAMBDA_OUTPUT_CONTRACT
JOINT_SOURCE_CHECKPOINT_VERSIONS = (
    "kalbi_v5_v19k32s32_exactDirect96proxPG1_tabiclv2_expvarnoise_v11_"
    "p100k_effbatch32_checkpoint_v1",
    "kalbi_v3_v19k32s32_cumulativebeta_tabiclv2_expvarnoise_v11_p100k_"
    "effbatch32_checkpoint_v1",
)
BETA_SOURCE_CHECKPOINT_VERSIONS = (
    "kalbi_v2_v17_stagegain_p100k_effbatch32_checkpoint_v3",
    "kalbi_v3_v17k32s32_p100k_effbatch32_checkpoint_v1",
    "kalbi_v3_v17k32s32_penaltymlp_tabiclv2only_p100k_"
    "effbatch32_checkpoint_v1",
    "kalbi_v3_v18k32s32_fullp_penaltyresmlp_tabiclv2only_p100k_"
    "effbatch32_checkpoint_v1",
)
SCM_CONTRACTS = {
    "tabicl_v2": (
        "tabiclv2_graph_scm2048_qfilter_bnn128_expvarnoise_"
        "mixed50_tabicl_linear_supportmix_pn_strictn3_v13",
        "mixed50_official_tabicl_or_controlled_linear_supportmix_pn50_"
        "strictn3_y_qualityfilter_center_scale_clip10_rescale_v2",
    ),
}
TABICL_TASK_TARGET_CONTRACT = (
    "official_tabiclv2_graph_scm_y_qualityfilter_center_scale_clip10_rescale_v2"
)
LINEAR_TASK_TARGET_CONTRACT = (
    "controlled_linear_y_supportmix_pn50_strictn3_snr_errorfamily_"
    "qualityfilter_center_scale_clip10_rescale_v2"
)
OUTCOME_SOURCE_TABICL = 0
OUTCOME_SOURCE_LINEAR = 1
OUTCOME_SOURCE_NAMES = {
    OUTCOME_SOURCE_TABICL: "tabicl",
    OUTCOME_SOURCE_LINEAR: "linear",
}
SUPPORT_REGIME_NAMES = (
    "ultra_sparse",
    "sparse",
    "moderate",
    "large",
    "very_dense",
)
COEFFICIENT_FAMILY_NAMES = (
    "gaussian",
    "laplace",
    "rademacher_loguniform",
    "student_t",
    "two_scale_gaussian",
)
SNR_REGIME_NAMES = (
    "very_low",
    "low",
    "medium",
    "high",
    "near_noiseless",
)
ERROR_FAMILY_NAMES = (
    "gaussian",
    "student_t",
    "laplace",
    "logistic",
    "centered_lognormal",
    "contaminated_gaussian",
    "heteroskedastic_gaussian",
)
V12_SOURCE_PIPELINE_SHA256 = (
    "d7304642abb7707bb24443c205a351ae2f990ce18b15662784ba55c78593baf4"
)
EXPECTED_WORLD_SIZE = 8
EXPECTED_GRAD_ACCUM_STEPS = 4
EXPECTED_SELECTOR_GRAD_ACCUM_STEPS = 4
RELOAD_INTERVAL_UPDATES = 100
LAMBDA_LOW_MIN_CONTRACT = 1e-6
SELECTOR_OBJECTIVE = (
    "mean(log1p(F_hat/F_zero))_"
    "F_hat=heldout_mse/2_F_zero=zero_predictor_heldout_mse/2_v2"
)
MAX_CONSECUTIVE_GLOBAL_PENALTY_ZERO_UPDATES = 25
EXPECTED_SELECTOR_PARAMETER_COUNT = 24_339_899
V5_SOURCE_MODEL_VERSION = (
    "kalbi_v5_krylov_abfilm_threshold_exactproxpg1_k32_stages32_v1"
)
V5_SOURCE_BASIS_VERSION = (
    "global_krylov_ridge_plus_ab_trajectory_threshold_plus_exactproxpg1_v5"
)
V5_SOURCE_SHARED_SUPPORT_CONTRACT = (
    "centered_corr_A_B_ridge1e-6_supportlocal_correction_global_threshold_"
    "fullP_exactgrad_exactDirect96proxPG_refinement_v5"
)
V19_SOURCE_SELECTOR_VERSION = (
    "v19_k32s32_direct96_residualtrajectory_fullp_isis32x32_"
    "stagegain0p5pct_hidden512_probe2048_chunk4096_noreplacement_"
    "localres_localmarg_cumulativejointbeta_concat_"
    "penalty_preln_reslinear3_nopostnorm_v2"
)
V3_SOURCE_BASIS_VERSION = (
    "global_krylov_ridge_plus_ab_trajectory_learned_terminal_threshold_v3"
)
V3_SOURCE_SHARED_SUPPORT_CONTRACT = (
    "centered_corr_A_B_ridge1e-6_supportlocal_correction_global_threshold_v3"
)
EXPECTED_MODEL_PARAMETER_COUNT = 3_588_547
REFINEMENT_PATH_METRIC_NAMES = (
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
REFINEMENT_AGGREGATE_METRIC_NAMES = (
    "refinement_majorization_ratio_max",
    "refinement_min_curvature_denominator_min",
    "refinement_backtracks_total",
    "refinement_trace_fallback_fraction",
    "refinement_objective_decrease",
)


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
        if not (32 <= n <= 40_000 and 50 <= p <= 100_000 and n * p <= 24_000_000):
            raise TrainerError(f"catalog task shape outside contract: N={n} P={p}")
        metadata = entry.get("metadata", {})
        semantic = metadata.get("semantic_contract", {}) if isinstance(metadata, Mapping) else {}
        dataset_contract, response_contract = SCM_CONTRACTS[prior]
        if semantic.get("contract") != dataset_contract:
            raise TrainerError("catalog shard lacks the SCM-response semantic certificate")
        if semantic.get("target_contract") != response_contract:
            raise TrainerError("catalog shard response contract drift")
        try:
            outcome_source_id = int(semantic["outcome_source_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TrainerError("catalog shard lacks outcome-source certificate") from exc
        if outcome_source_id not in OUTCOME_SOURCE_NAMES:
            raise TrainerError("catalog shard outcome-source id is invalid")
        if semantic.get("outcome_source") != OUTCOME_SOURCE_NAMES[outcome_source_id]:
            raise TrainerError("catalog shard outcome-source name drift")
        expected_task_target = (
            TABICL_TASK_TARGET_CONTRACT
            if outcome_source_id == OUTCOME_SOURCE_TABICL
            else LINEAR_TASK_TARGET_CONTRACT
        )
        if semantic.get("task_target_contract") != expected_task_target:
            raise TrainerError("catalog shard task target contract drift")
        if outcome_source_id == OUTCOME_SOURCE_LINEAR:
            support_size = int(semantic.get("linear_support_size", -1))
            support_fraction = float(
                semantic.get("linear_support_fraction", float("nan"))
            )
            support_fraction_of_n = float(
                semantic.get("linear_support_fraction_of_n", float("nan"))
            )
            support_reference_id = int(
                semantic.get("linear_support_reference_id", -1)
            )
            target_snr = float(semantic.get("linear_target_snr", float("nan")))
            realized_snr = float(
                semantic.get("linear_realized_snr", float("nan"))
            )
            if (
                not 1 <= support_size <= p
                or not 3 * support_size < n
                or not math.isfinite(support_fraction)
                or not math.isclose(
                    support_fraction,
                    support_size / float(p),
                    rel_tol=2e-6,
                    abs_tol=2e-8,
                )
                or not math.isfinite(support_fraction_of_n)
                or not math.isclose(
                    support_fraction_of_n,
                    support_size / float(n),
                    rel_tol=2e-6,
                    abs_tol=2e-8,
                )
                or support_reference_id not in (0, 1)
                or semantic.get("linear_support_reference")
                not in ("p_based", "n_based")
                or semantic.get("linear_support_strict_n_third_verified")
                is not True
                or not math.isfinite(target_snr)
                or not math.isfinite(realized_snr)
                or target_snr <= 0.0
                or realized_snr <= 0.0
                or abs(math.log(realized_snr / target_snr)) > 2e-5
            ):
                raise TrainerError("catalog shard linear-response certificate failed")
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
        raise TrainerError(
            f"rolling catalog prior inventory drift: {observed_priors}"
        )


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


class TrainerCatalogStore(CatalogStore):
    """Resolve epochs from already verified pointers before scanning history."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._adopted_epochs: dict[str, int] = {}

    def remember_adopted_pointer(self, pointer: Mapping[str, Any]) -> None:
        digest = pointer.get("catalog_sha256")
        epoch = pointer.get("epoch")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(epoch, int)
            or epoch < 0
        ):
            raise TrainerError("cannot cache an invalid adopted catalog pointer")
        existing = self._adopted_epochs.get(digest)
        if existing is not None and existing != epoch:
            raise TrainerError("adopted catalog digest changed epoch")
        self._adopted_epochs[digest] = epoch

    def _catalog_epoch_from_current_or_chain(
        self, catalog_sha256: str
    ) -> int | None:
        known = self._adopted_epochs.get(catalog_sha256)
        if known is not None:
            return known
        return super()._catalog_epoch_from_current_or_chain(catalog_sha256)


class RollingSingleTaskStream:
    """Exact-resumable rank-local stream with an explicit reloadable LRU."""

    VERSION = (
        "v13_mixed50_tabicl_linear_pn50_strictn3_"
        "catalog_single_task_stream_v1"
    )

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
            if not (32 <= n <= 40_000 and 50 <= p <= 100_000 and n * p <= 24_000_000):
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
                "X",
                "y",
                "feature_noise_variance",
                "feature_noise_sigma",
                "feature_noise_family",
                "feature_source",
                "sampled_corr_audit_max",
                "core_exact_max_abs_corr",
                "scm_response_mean",
                "scm_response_std",
                "batch_seed",
                "outcome_source_id",
                "linear_support_size",
                "linear_support_fraction",
                "linear_support_fraction_of_n",
                "linear_support_regime_id",
                "linear_support_reference_id",
                "linear_coefficient_family_id",
                "linear_snr_regime_id",
                "linear_error_family_id",
                "linear_target_snr",
                "linear_realized_snr",
            }
            if not required.issubset(archive.files):
                raise TrainerError(f"runtime shard lacks certified arrays: {path}")
            x_raw = archive["X"]
            y_raw = archive["y"]
            variance = archive["feature_noise_variance"]
            sigma = archive["feature_noise_sigma"]
            family = archive["feature_noise_family"]
            source = archive["feature_source"]
            sampled_corr = np.asarray(
                archive["sampled_corr_audit_max"], dtype=np.float32
            )
            core_corr = np.asarray(
                archive["core_exact_max_abs_corr"], dtype=np.float32
            )
            response_mean = np.asarray(archive["scm_response_mean"], dtype=np.float32)
            response_std = np.asarray(archive["scm_response_std"], dtype=np.float32)
            batch_seed = archive["batch_seed"]
            outcome_source_id = archive["outcome_source_id"]
            linear_support_size = archive["linear_support_size"]
            linear_support_fraction = archive["linear_support_fraction"]
            linear_support_fraction_of_n = archive[
                "linear_support_fraction_of_n"
            ]
            linear_support_regime_id = archive["linear_support_regime_id"]
            linear_support_reference_id = archive[
                "linear_support_reference_id"
            ]
            linear_coefficient_family_id = archive[
                "linear_coefficient_family_id"
            ]
            linear_snr_regime_id = archive["linear_snr_regime_id"]
            linear_error_family_id = archive["linear_error_family_id"]
            linear_target_snr = archive["linear_target_snr"]
            linear_realized_snr = archive["linear_realized_snr"]
            if x_raw.dtype != np.float16 or y_raw.dtype != np.float32:
                raise TrainerError(f"runtime shard dtype drift: {path}")
            expected_x = (1, int(entry["n"]), int(entry["p"]))
            expected_y = (1, int(entry["n"]))
            if x_raw.shape != expected_x or y_raw.shape != expected_y:
                raise TrainerError(
                    f"runtime shard shape drift: {path} X={x_raw.shape} y={y_raw.shape}"
                )
            feature_shape = (1, int(entry["p"]))
            if (
                variance.shape != feature_shape
                or variance.dtype != np.float32
                or sigma.shape != feature_shape
                or sigma.dtype != np.float32
                or family.shape != feature_shape
                or family.dtype != np.uint8
                or source.shape != feature_shape
                or source.dtype != np.uint8
                or sampled_corr.shape != (1,)
                or core_corr.shape != (1,)
                or response_mean.shape != (1,)
                or response_std.shape != (1,)
                or batch_seed.shape != (1,)
                or batch_seed.dtype != np.uint64
                or outcome_source_id.shape != (1,)
                or outcome_source_id.dtype != np.uint8
                or linear_support_size.shape != (1,)
                or linear_support_size.dtype != np.int64
                or linear_support_fraction.shape != (1,)
                or linear_support_fraction.dtype != np.float32
                or linear_support_fraction_of_n.shape != (1,)
                or linear_support_fraction_of_n.dtype != np.float32
                or linear_support_regime_id.shape != (1,)
                or linear_support_regime_id.dtype != np.uint8
                or linear_support_reference_id.shape != (1,)
                or linear_support_reference_id.dtype != np.uint8
                or linear_coefficient_family_id.shape != (1,)
                or linear_coefficient_family_id.dtype != np.uint8
                or linear_snr_regime_id.shape != (1,)
                or linear_snr_regime_id.dtype != np.uint8
                or linear_error_family_id.shape != (1,)
                or linear_error_family_id.dtype != np.uint8
                or linear_target_snr.shape != (1,)
                or linear_target_snr.dtype != np.float32
                or linear_realized_snr.shape != (1,)
                or linear_realized_snr.dtype != np.float32
            ):
                raise TrainerError("runtime shard certificate shape drift")
            realized_mean = float(np.mean(y_raw[0].astype(np.float64)))
            realized_std = float(np.std(y_raw[0].astype(np.float64)))
            core_features = min(int(entry["p"]), 2_048)
            if not (
                np.isfinite(x_raw).all()
                and np.isfinite(y_raw).all()
                and np.isfinite(variance).all()
                and np.isfinite(sigma).all()
                and np.isfinite(sampled_corr).all()
                and np.isfinite(core_corr).all()
                and np.all(variance >= 0.0)
                and np.allclose(
                    sigma.astype(np.float64) ** 2,
                    variance.astype(np.float64),
                    rtol=3e-6,
                    atol=3e-7,
                )
                and np.all(family == 0)
                and np.all(np.isin(source, (0, 1)))
                and int(np.count_nonzero(source == 0)) == core_features
                and int(np.count_nonzero(source == 1))
                == int(entry["p"]) - core_features
                and 0.0 <= float(sampled_corr[0]) < 0.95
                and 0.0 <= float(core_corr[0]) < 0.95
                and abs(float(response_mean[0]) - realized_mean) <= 2e-6
                and abs(float(response_std[0]) - realized_std) <= 2e-6
                and abs(realized_mean) <= 2e-6
                and abs(realized_std - 1.0) <= 2e-6
            ):
                raise TrainerError(f"runtime shard semantic certificate failed: {path}")
            source_id = int(outcome_source_id[0])
            if source_id not in OUTCOME_SOURCE_NAMES:
                raise TrainerError(f"runtime outcome-source id is invalid: {path}")
            if source_id == OUTCOME_SOURCE_TABICL:
                if not (
                    int(linear_support_size[0]) == 0
                    and float(linear_support_fraction[0]) == 0.0
                    and float(linear_support_fraction_of_n[0]) == 0.0
                    and int(linear_support_regime_id[0]) == 255
                    and int(linear_support_reference_id[0]) == 255
                    and int(linear_coefficient_family_id[0]) == 255
                    and int(linear_snr_regime_id[0]) == 255
                    and int(linear_error_family_id[0]) == 255
                    and float(linear_target_snr[0]) == -1.0
                    and float(linear_realized_snr[0]) == -1.0
                ):
                    raise TrainerError(
                        f"runtime TabICL sentinel certificate failed: {path}"
                    )
            else:
                support_size = int(linear_support_size[0])
                target_snr = float(linear_target_snr[0])
                realized_snr = float(linear_realized_snr[0])
                if not (
                    1 <= support_size <= int(entry["p"])
                    and 3 * support_size < int(entry["n"])
                    and math.isclose(
                        float(linear_support_fraction[0]),
                        support_size / float(entry["p"]),
                        rel_tol=2e-6,
                        abs_tol=2e-8,
                    )
                    and math.isclose(
                        float(linear_support_fraction_of_n[0]),
                        support_size / float(entry["n"]),
                        rel_tol=2e-6,
                        abs_tol=2e-8,
                    )
                    and int(linear_support_regime_id[0])
                    < len(SUPPORT_REGIME_NAMES)
                    and int(linear_support_reference_id[0]) in (0, 1)
                    and int(linear_coefficient_family_id[0])
                    < len(COEFFICIENT_FAMILY_NAMES)
                    and int(linear_snr_regime_id[0]) < len(SNR_REGIME_NAMES)
                    and int(linear_error_family_id[0]) < len(ERROR_FAMILY_NAMES)
                    and math.isfinite(target_snr)
                    and math.isfinite(realized_snr)
                    and target_snr > 0.0
                    and realized_snr > 0.0
                    and abs(math.log(realized_snr / target_snr)) <= 2e-5
                ):
                    raise TrainerError(
                        f"runtime linear-response certificate failed: {path}"
                    )
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
        metadata = entry.get("metadata", {})
        semantic = (
            metadata.get("semantic_contract", {})
            if isinstance(metadata, Mapping)
            else {}
        )
        return [x], [y], [{
            "shard_id": int(entry["shard_id"]),
            "path": str(entry["absolute_path"]),
            "n": int(entry["n"]),
            "p": int(entry["p"]),
            "catalog_sha256": self.catalog_sha256,
            "prior": str(entry["prior"]),
            "outcome_source": str(semantic["outcome_source"]),
            "outcome_source_id": int(semantic["outcome_source_id"]),
            "linear_support_size": int(semantic["linear_support_size"]),
            "linear_support_fraction": float(
                semantic["linear_support_fraction"]
            ),
            "linear_support_regime_id": int(
                semantic["linear_support_regime_id"]
            ),
            "linear_coefficient_family_id": int(
                semantic["linear_coefficient_family_id"]
            ),
            "linear_snr_regime_id": int(semantic["linear_snr_regime_id"]),
            "linear_error_family_id": int(
                semantic["linear_error_family_id"]
            ),
            "linear_target_snr": float(semantic["linear_target_snr"]),
            "linear_realized_snr": float(semantic["linear_realized_snr"]),
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


def audit_optimizer_state_by_parameter_name(
    *,
    label: str,
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    source_state_dict: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind source optimizer slots to target parameters through stable names."""

    name_by_parameter_id = {
        id(parameter): name for name, parameter in module.named_parameters()
    }
    source_groups = list(source_state_dict.get("param_groups", ()))
    target_groups = list(optimizer.param_groups)
    if len(source_groups) != len(target_groups):
        raise TrainerError(
            f"{label} optimizer group-count mismatch: "
            f"source={len(source_groups)}, target={len(target_groups)}"
        )

    named_groups: list[dict[str, Any]] = []
    checked_state_tensors = 0
    source_states = source_state_dict.get("state", {})
    for group_index, (source_group, target_group) in enumerate(
        zip(source_groups, target_groups)
    ):
        source_name = source_group.get("name")
        target_name = target_group.get("name")
        if source_name != target_name:
            raise TrainerError(
                f"{label} optimizer group-name mismatch at {group_index}: "
                f"source={source_name!r}, target={target_name!r}"
            )
        source_ids = list(source_group.get("params", ()))
        target_parameters = list(target_group.get("params", ()))
        if len(source_ids) != len(target_parameters):
            raise TrainerError(
                f"{label} optimizer parameter-count mismatch in "
                f"group {target_name!r}"
            )
        parameter_names: list[str] = []
        for source_id, target_parameter in zip(
            source_ids, target_parameters
        ):
            parameter_name = name_by_parameter_id.get(id(target_parameter))
            if parameter_name is None:
                raise TrainerError(
                    f"{label} optimizer contains a parameter outside its module"
                )
            parameter_names.append(parameter_name)
            source_parameter_state = source_states.get(source_id, {})
            for state_name, state_value in source_parameter_state.items():
                if not torch.is_tensor(state_value) or state_value.ndim == 0:
                    continue
                if tuple(state_value.shape) != tuple(target_parameter.shape):
                    raise TrainerError(
                        f"{label} optimizer state shape mismatch for "
                        f"{parameter_name}.{state_name}: "
                        f"source={tuple(state_value.shape)}, "
                        f"target={tuple(target_parameter.shape)}"
                    )
                checked_state_tensors += 1
        named_groups.append(
            {
                "name": target_name,
                "parameter_names": parameter_names,
            }
        )

    flat_names = [
        name
        for group in named_groups
        for name in group["parameter_names"]
    ]
    expected_names = [name for name, _ in module.named_parameters()]
    if sorted(flat_names) != sorted(expected_names):
        missing = sorted(set(expected_names) - set(flat_names))
        duplicate_or_extra = sorted(set(flat_names) - set(expected_names))
        raise TrainerError(
            f"{label} optimizer named-parameter partition mismatch: "
            f"missing={missing}, extra={duplicate_or_extra}"
        )
    signature_json = json.dumps(named_groups, sort_keys=True)
    return {
        "groups": named_groups,
        "parameter_count": len(flat_names),
        "checked_state_tensors": checked_state_tensors,
        "named_signature_sha256": hashlib.sha256(
            signature_json.encode("utf-8")
        ).hexdigest(),
        "numeric_parameter_ids_manually_merged": False,
    }


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
    parser.add_argument("--input-scaler-cache", type=Path, required=True)
    parser.add_argument("--input-scaler-cache-sha256", required=True)
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sha256")
    parser.add_argument("--transition-v7-beta-from", type=Path)
    parser.add_argument("--transition-v7-beta-sha256")
    parser.add_argument("--transition-v7-selector-from", type=Path)
    parser.add_argument("--transition-v7-selector-sha256")
    parser.add_argument(
        "--resume-beta-only-from-joint",
        action="store_true",
        help=(
            "perform the audited one-time transition from a strict joint V23 "
            "checkpoint to beta-only training"
        ),
    )
    parser.add_argument(
        "--resume-beta-only-data-transition",
        action="store_true",
        help=(
            "perform the audited one-time transition of an existing beta-only "
            "V23 run to the V12 mixed TabICL/controlled-linear rolling catalog"
        ),
    )
    parser.add_argument("--transition-v22-from", type=Path)
    parser.add_argument("--transition-v22-sha256")
    parser.add_argument("--transition-v21-from", type=Path)
    parser.add_argument("--transition-v21-sha256")
    parser.add_argument("--selector-v20-from", type=Path)
    parser.add_argument("--selector-v20-sha256")
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
            "update only Kalbi V6 while carrying V23 state unchanged"
        ),
    )
    args = parser.parse_args()

    initialization_modes = sum((
        bool(args.scratch),
        args.resume is not None,
        args.transition_v7_beta_from is not None,
        args.transition_v22_from is not None,
        args.transition_v21_from is not None,
        args.initialize_beta_from is not None,
        args.initialize_joint_from is not None,
    ))
    if initialization_modes != 1:
        parser.error(
            "choose exactly one of --scratch, --resume, --transition-v22-from, "
            "--transition-v7-beta-from, --transition-v21-from, "
            "--initialize-beta-from, or "
            "--initialize-joint-from"
        )
    if args.resume is not None and not args.resume_sha256:
        parser.error("--resume requires --resume-sha256")
    transition_v7_values = (
        args.transition_v7_beta_from,
        args.transition_v7_beta_sha256,
        args.transition_v7_selector_from,
        args.transition_v7_selector_sha256,
    )
    if any(value is not None for value in transition_v7_values) and not all(
        value is not None for value in transition_v7_values
    ):
        parser.error(
            "V7 transition requires beta/selector paths and both SHA256 values"
        )
    if args.transition_v7_beta_from is not None and args.beta_only:
        parser.error("V7 transition is a joint run and forbids --beta-only")
    if args.resume_beta_only_from_joint and not (
        args.resume is not None and args.beta_only
    ):
        parser.error(
            "--resume-beta-only-from-joint requires both --resume and --beta-only"
        )
    if args.resume_beta_only_data_transition and not (
        args.resume is not None and args.beta_only
    ):
        parser.error(
            "--resume-beta-only-data-transition requires both --resume and "
            "--beta-only"
        )
    if (
        args.resume_beta_only_from_joint
        and args.resume_beta_only_data_transition
    ):
        parser.error("the two one-time beta-only resume transitions are exclusive")
    if (
        args.transition_v22_from is not None
        and not args.transition_v22_sha256
    ):
        parser.error("--transition-v22-from requires --transition-v22-sha256")
    if (
        args.transition_v21_from is not None
        and not args.transition_v21_sha256
    ):
        parser.error("--transition-v21-from requires --transition-v21-sha256")
    if (args.selector_v20_from is None) != (args.selector_v20_sha256 is None):
        parser.error(
            "--selector-v20-from and --selector-v20-sha256 must be provided together"
        )
    if args.transition_v21_from is not None and args.selector_v20_from is None:
        parser.error("V21-to-V22 transition requires --selector-v20-from")
    if args.initialize_beta_from is not None and not args.initialize_beta_sha256:
        parser.error("--initialize-beta-from requires --initialize-beta-sha256")
    if args.initialize_joint_from is not None and not args.initialize_joint_sha256:
        parser.error("--initialize-joint-from requires --initialize-joint-sha256")
    if args.batch_per_rank != 1:
        parser.error("variable-shape rolling training requires batch-per-rank=1")
    if args.selector_hidden != V17_HIDDEN_DIM:
        parser.error(f"V23 selector hidden dimension is fixed at {V17_HIDDEN_DIM}")
    if args.grad_accum_steps != EXPECTED_GRAD_ACCUM_STEPS:
        parser.error("8-GPU global beta task batch is fixed at 32")
    if args.selector_grad_accum_steps != EXPECTED_SELECTOR_GRAD_ACCUM_STEPS:
        parser.error("8-GPU global V23 selector task batch is fixed at 32")
    if args.selector_update_every != 4:
        parser.error("KALBI V7 selector update interval is fixed at 4")
    if not math.isclose(args.beta_lr, 1e-4, rel_tol=0.0, abs_tol=1e-15):
        parser.error("KALBI V7 beta LR is fixed at 1e-4")
    if not math.isclose(args.selector_lr, 3e-4, rel_tol=0.0, abs_tol=1e-15):
        parser.error("KALBI V7 selector LR is fixed at 3e-4")
    if not math.isclose(
        args.direct_head_lr, 3e-4, rel_tol=0.0, abs_tol=1e-15
    ):
        parser.error("KALBI V7 penalty-head LR is fixed at 3e-4")
    if args.steps != 60_000:
        parser.error("KALBI V7 beta total step is fixed at 60000")
    if args.reload_every != RELOAD_INTERVAL_UPDATES:
        parser.error(
            f"catalog/dataloader reload interval is fixed at "
            f"{RELOAD_INTERVAL_UPDATES} updates"
        )
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
    if not args.input_scaler_cache.is_file() or file_sha256(
        args.input_scaler_cache
    ) != args.input_scaler_cache_sha256:
        raise TrainerError("V22 input-scaler cache SHA256 mismatch")
    input_scaler_payload = torch.load(
        args.input_scaler_cache, map_location="cpu", weights_only=False
    )
    if input_scaler_payload.get("version") != V22_SCALER_VERSION:
        raise TrainerError("V22 input-scaler cache version mismatch")
    init_trace("input_scaler_cache_verified")

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
            active_min_bytes=500_000_000,
            active_max_bytes=102_000_000_000,
            prior_min_bytes=500_000_000,
            prior_max_bytes=102_000_000_000,
            hard_cap_bytes=130_000_000_000,
            min_free_bytes=300_000_000_000,
            world_size=world,
        )
    )
    store = TrainerCatalogStore(
        args.rolling_control_root,
        args.rolling_data_root,
        policy=policy,
        priors=("tabicl_v2",),
    )
    resume_checkpoint = None
    v7_beta_transition_checkpoint = None
    v7_selector_transition_checkpoint = None
    v22_transition_checkpoint = None
    v21_transition_checkpoint = None
    v20_selector_checkpoint = None
    beta_source_checkpoint = None
    joint_source_checkpoint = None
    if args.transition_v7_beta_from is not None:
        if (
            file_sha256(args.transition_v7_beta_from)
            != args.transition_v7_beta_sha256
        ):
            raise TrainerError("V7 beta-source checkpoint SHA256 mismatch")
        if (
            file_sha256(args.transition_v7_selector_from)
            != args.transition_v7_selector_sha256
        ):
            raise TrainerError("V7 selector-source checkpoint SHA256 mismatch")
        v7_beta_transition_checkpoint = torch.load(
            args.transition_v7_beta_from,
            map_location="cpu",
            weights_only=False,
        )
        v7_selector_transition_checkpoint = torch.load(
            args.transition_v7_selector_from,
            map_location="cpu",
            weights_only=False,
        )
        beta_metadata = v7_beta_transition_checkpoint.get("metadata", {})
        selector_metadata = v7_selector_transition_checkpoint.get(
            "metadata", {}
        )
        beta_contract = {
            "checkpoint_version": V7_BETA_SOURCE_CHECKPOINT_VERSION,
            "model_version": V7_SOURCE_MODEL_VERSION,
            "basis_version": V7_SOURCE_BASIS_VERSION,
            "shared_support_contract": V7_SOURCE_SHARED_SUPPORT_CONTRACT,
            "selector_version": V23_SELECTOR_VERSION,
            "world_size": world,
            "training_mode": "beta_only",
        }
        selector_contract = {
            "checkpoint_version": V7_SELECTOR_SOURCE_CHECKPOINT_VERSION,
            "model_version": V7_SOURCE_MODEL_VERSION,
            "selector_version": V23_SELECTOR_VERSION,
            "world_size": world,
            "training_mode": V7_SELECTOR_SOURCE_TRAINING_MODE,
            "selector_objective": SELECTOR_OBJECTIVE,
        }
        for key, expected in beta_contract.items():
            if beta_metadata.get(key) != expected:
                raise TrainerError(
                    f"V7 beta-source contract mismatch for {key}: "
                    f"expected={expected!r}, observed={beta_metadata.get(key)!r}"
                )
        for key, expected in selector_contract.items():
            if selector_metadata.get(key) != expected:
                raise TrainerError(
                    f"V7 selector-source contract mismatch for {key}: "
                    f"expected={expected!r}, "
                    f"observed={selector_metadata.get(key)!r}"
                )
        if "beta_optimizer_state_dict" not in v7_beta_transition_checkpoint:
            raise TrainerError("V7 beta source lacks beta optimizer state")
        if "beta_scheduler_state_dict" not in v7_beta_transition_checkpoint:
            raise TrainerError("V7 beta source lacks beta scheduler state")
        if "selector_optimizer_state_dict" not in v7_selector_transition_checkpoint:
            raise TrainerError("V7 selector source lacks selector optimizer state")
        if int(v7_selector_transition_checkpoint["trainer_state"].get(
            "selector_updates", -1
        )) != int(v7_selector_transition_checkpoint["completed_steps"]):
            raise TrainerError(
                "V7 selector source update counter does not match completed steps"
            )
    elif args.resume is not None:
        if file_sha256(args.resume) != args.resume_sha256:
            raise TrainerError("resume checkpoint SHA256 mismatch")
        resume_checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        resume_version = resume_checkpoint.get("metadata", {}).get(
            "checkpoint_version"
        )
        if resume_version != CHECKPOINT_VERSION:
            raise TrainerError("resume checkpoint version mismatch")
    elif args.transition_v22_from is not None:
        if (
            file_sha256(args.transition_v22_from)
            != args.transition_v22_sha256
        ):
            raise TrainerError("V22 transition checkpoint SHA256 mismatch")
        v22_transition_checkpoint = torch.load(
            args.transition_v22_from,
            map_location="cpu",
            weights_only=False,
        )
        source_metadata = v22_transition_checkpoint.get("metadata", {})
        source_contract = {
            "trainer_version": V22_SOURCE_TRAINER_VERSION,
            "checkpoint_version": V22_SOURCE_CHECKPOINT_VERSION,
            "model_version": MODEL_VERSION,
            "basis_version": BASIS_VERSION,
            "shared_support_contract": SHARED_SUPPORT_CONTRACT,
            "canonical_penalty_contract": CONTRACT_VERSION,
            "canonical_cache_version": CACHE_VERSION,
            "selector_version": V22_SELECTOR_VERSION,
            "world_size": world,
        }
        for key, expected in source_contract.items():
            if source_metadata.get(key) != expected:
                raise TrainerError(
                    f"V22 transition contract mismatch for {key}: "
                    f"expected={expected!r}, "
                    f"observed={source_metadata.get(key)!r}"
                )
    elif args.transition_v21_from is not None:
        if (
            file_sha256(args.transition_v21_from)
            != args.transition_v21_sha256
        ):
            raise TrainerError("V21 transition checkpoint SHA256 mismatch")
        v21_transition_checkpoint = torch.load(
            args.transition_v21_from,
            map_location="cpu",
            weights_only=False,
        )
        source_metadata = v21_transition_checkpoint.get("metadata", {})
        source_contract = {
            "trainer_version": V21_SOURCE_TRAINER_VERSION,
            "checkpoint_version": V21_SOURCE_CHECKPOINT_VERSION,
            "model_version": MODEL_VERSION,
            "basis_version": BASIS_VERSION,
            "shared_support_contract": SHARED_SUPPORT_CONTRACT,
            "canonical_penalty_contract": CONTRACT_VERSION,
            "canonical_cache_version": CACHE_VERSION,
            "selector_version": V21_SOURCE_SELECTOR_VERSION,
            "world_size": world,
        }
        for key, expected in source_contract.items():
            if source_metadata.get(key) != expected:
                raise TrainerError(
                    f"V21 transition contract mismatch for {key}: "
                    f"expected={expected!r}, observed={source_metadata.get(key)!r}"
                )
        if file_sha256(args.selector_v20_from) != args.selector_v20_sha256:
            raise TrainerError("V20 selector-source checkpoint SHA256 mismatch")
        v20_selector_checkpoint = torch.load(
            args.selector_v20_from,
            map_location="cpu",
            weights_only=False,
        )
        selector_source_metadata = v20_selector_checkpoint.get("metadata", {})
        selector_source_contract = {
            "trainer_version": V20_SOURCE_TRAINER_VERSION,
            "checkpoint_version": V20_SOURCE_CHECKPOINT_VERSION,
            "selector_version": V20_SELECTOR_VERSION,
            "canonical_penalty_contract": CONTRACT_VERSION,
            "canonical_cache_version": CACHE_VERSION,
            "world_size": world,
        }
        for key, expected in selector_source_contract.items():
            if selector_source_metadata.get(key) != expected:
                raise TrainerError(
                    f"V20 selector-source contract mismatch for {key}: "
                    f"expected={expected!r}, "
                    f"observed={selector_source_metadata.get(key)!r}"
                )
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
            "model_version": V3_SOURCE_MODEL_VERSION,
            "basis_version": V3_SOURCE_BASIS_VERSION,
            "shared_support_contract": V3_SOURCE_SHARED_SUPPORT_CONTRACT,
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
        source_checkpoint_version = source_metadata.get("checkpoint_version")
        if source_checkpoint_version == JOINT_SOURCE_CHECKPOINT_VERSIONS[0]:
            source_beta_contract = {
                "model_version": V5_SOURCE_MODEL_VERSION,
                "basis_version": V5_SOURCE_BASIS_VERSION,
                "shared_support_contract": V5_SOURCE_SHARED_SUPPORT_CONTRACT,
            }
        else:
            source_beta_contract = {
                "model_version": V3_SOURCE_MODEL_VERSION,
                "basis_version": V3_SOURCE_BASIS_VERSION,
                "shared_support_contract": V3_SOURCE_SHARED_SUPPORT_CONTRACT,
            }
        source_contract = {
            **source_beta_contract,
            "canonical_penalty_contract": CONTRACT_VERSION,
            "canonical_cache_version": CACHE_VERSION,
            "selector_version": V19_SOURCE_SELECTOR_VERSION,
            "world_size": world,
        }
        for key, expected in source_contract.items():
            if source_metadata.get(key) != expected:
                raise TrainerError(f"joint-source contract mismatch for {key}")

    initial_checkpoint = (
        v7_beta_transition_checkpoint
        or resume_checkpoint
        or v22_transition_checkpoint
        or v21_transition_checkpoint
        or beta_source_checkpoint
        or joint_source_checkpoint
    )
    transitioning_v7 = v7_beta_transition_checkpoint is not None
    transitioning_v22 = v22_transition_checkpoint is not None
    transitioning_v21 = v21_transition_checkpoint is not None
    importing_beta_checkpoint = beta_source_checkpoint is not None
    importing_joint_checkpoint = joint_source_checkpoint is not None

    if transitioning_v7:
        initialization_provenance = {
            "mode": "v6_beta_only_plus_v23_selector_only_to_v7_joint",
            "beta_source_checkpoint": str(
                args.transition_v7_beta_from.resolve()
            ),
            "beta_source_checkpoint_sha256": args.transition_v7_beta_sha256,
            "beta_source_completed_steps": int(
                v7_beta_transition_checkpoint["completed_steps"]
            ),
            "selector_source_checkpoint": str(
                args.transition_v7_selector_from.resolve()
            ),
            "selector_source_checkpoint_sha256": (
                args.transition_v7_selector_sha256
            ),
            "selector_source_completed_steps": int(
                v7_selector_transition_checkpoint["completed_steps"]
            ),
            "beta_model_state_imported_strict": True,
            "beta_optimizer_state_imported": True,
            "beta_scheduler_state_imported": True,
            "selector_state_imported_strict": True,
            "selector_optimizer_state_imported": True,
            "rank_rng_state_source": "beta_checkpoint",
            "rolling_catalog_state_source": "beta_checkpoint",
            "rolling_stream_state_source": "beta_checkpoint",
            "v6_pg10_removed": True,
            "v7_full_p_block_prox_sweeps": KALBI_BLOCK_PROX_SWEEPS,
            "v7_block_size": KALBI_BLOCK_PROX_BLOCK_SIZE,
            "input_scaler_cache": str(args.input_scaler_cache.resolve()),
            "input_scaler_cache_sha256": args.input_scaler_cache_sha256,
        }
    elif resume_checkpoint is not None:
        initialization_provenance = dict(
            resume_checkpoint["metadata"]["initialization_provenance"]
        )
        if args.resume_beta_only_from_joint:
            initialization_provenance["target_host_migration"] = {
                "source_checkpoint": str(args.resume.resolve()),
                "source_checkpoint_sha256": args.resume_sha256,
                "source_completed_steps": int(
                    resume_checkpoint["completed_steps"]
                ),
                "beta_model_state_imported": True,
                "beta_optimizer_state_imported": True,
                "beta_scheduler_state_imported": True,
                "rank_rng_state_imported": True,
                "penalty_sampler_state_imported": True,
                "selector_state_carried_unchanged": True,
                "selector_optimizer_state_carried_without_instantiation": True,
                "catalog_state_imported": False,
                "rolling_stream_state_imported": False,
                "reason": "new target rolling root and seed namespace",
            }
        elif args.resume_beta_only_data_transition:
            initialization_provenance["data_contract_transition"] = {
                "source_checkpoint": str(args.resume.resolve()),
                "source_checkpoint_sha256": args.resume_sha256,
                "source_completed_steps": int(
                    resume_checkpoint["completed_steps"]
                ),
                "source_dataset_contract": (
                    "tabiclv2_graph_scm2048_qfilter_bnn128_expvarnoise_"
                    "mixed50_tabicl_linear_v12"
                ),
                "target_dataset_contract": SCM_CONTRACTS["tabicl_v2"][0],
                "source_target_contract": (
                    "mixed50_official_tabicl_or_controlled_linear_y_"
                    "qualityfilter_center_scale_clip10_rescale_v1"
                ),
                "target_target_contract": SCM_CONTRACTS["tabicl_v2"][1],
                "beta_model_state_imported": True,
                "beta_optimizer_state_imported": True,
                "beta_scheduler_state_imported": True,
                "rank_rng_state_imported": True,
                "penalty_sampler_state_imported": True,
                "selector_state_carried_unchanged": True,
                "selector_optimizer_state_carried_without_instantiation": True,
                "catalog_state_imported": False,
                "rolling_stream_state_imported": False,
                "reason": (
                    "replace P-only linear support sampling with deterministic "
                    "50/50 P/N-referenced support under strict k<N/3"
                ),
            }
    elif v22_transition_checkpoint is not None:
        initialization_provenance = {
            "mode": (
                "v22_beta_and_shared_selector_optimizer_continuation_"
                "v23_independent_raw_lambda_head"
            ),
            "v22_source_checkpoint": str(
                args.transition_v22_from.resolve()
            ),
            "v22_source_checkpoint_sha256": args.transition_v22_sha256,
            "v22_source_completed_steps": int(
                v22_transition_checkpoint["completed_steps"]
            ),
            "v22_source_checkpoint_version": (
                v22_transition_checkpoint["metadata"]["checkpoint_version"]
            ),
            "beta_model_state_imported": True,
            "beta_optimizer_state_imported": True,
            "beta_scheduler_state_imported": False,
            "beta_scheduler_restarted_for_v23_phase": True,
            "rank_stream_state_imported": True,
            "catalog_state_imported": True,
            "selector_state_imported": (
                "all_v22_tensors_except_lambda_head"
            ),
            "selector_penalty_and_geometry_state_imported": True,
            "selector_lambda_head_reinitialized": True,
            "selector_lambda_head_initial_raw_lambda": (
                V23_INITIAL_RAW_LAMBDA
            ),
            "selector_optimizer_state_imported": (
                "all_v22_parameter_states_except_lambda_head"
            ),
            "selector_update_counter_reset": True,
            "input_scaler_cache": str(args.input_scaler_cache.resolve()),
            "input_scaler_cache_sha256": args.input_scaler_cache_sha256,
        }
    elif v21_transition_checkpoint is not None:
        initialization_provenance = {
            "mode": (
                "v21_beta_optimizer_stream_continuation_"
                "v20_selector_to_v22_geometry"
            ),
            "v21_beta_source_checkpoint": str(
                args.transition_v21_from.resolve()
            ),
            "v21_beta_source_checkpoint_sha256": args.transition_v21_sha256,
            "v21_source_completed_steps": int(
                v21_transition_checkpoint["completed_steps"]
            ),
            "v21_source_checkpoint_version": (
                v21_transition_checkpoint["metadata"]["checkpoint_version"]
            ),
            "v20_selector_source_checkpoint": str(
                args.selector_v20_from.resolve()
            ),
            "v20_selector_source_checkpoint_sha256": args.selector_v20_sha256,
            "beta_model_state_imported": True,
            "beta_optimizer_state_imported": True,
            "beta_scheduler_state_imported": True,
            "rank_stream_state_imported": True,
            "catalog_state_imported": True,
            "selector_state_imported": "all_v20_selector_tensors",
            "selector_v20_canonical_head_imported": True,
            "selector_v22_geometry_and_scaler_modules_fresh": True,
            "input_scaler_cache": str(args.input_scaler_cache.resolve()),
            "input_scaler_cache_sha256": args.input_scaler_cache_sha256,
            "selector_optimizer_state_imported": False,
            "selector_update_counter_reset": True,
        }
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
            "rank_stream_state_imported": False,
            "catalog_state_imported": False,
            "selector_state_imported": False,
            "selector_optimizer_state_imported": False,
            "selector_update_counter_reset": True,
        }
    elif joint_source_checkpoint is not None:
        initialization_provenance = {
            "mode": "v3_or_v5_beta_import_v23_selector_scratch",
            "joint_source_checkpoint": str(args.initialize_joint_from.resolve()),
            "joint_source_checkpoint_sha256": args.initialize_joint_sha256,
            "joint_source_completed_steps": int(
                joint_source_checkpoint["completed_steps"]
            ),
            "joint_source_checkpoint_version": joint_source_checkpoint[
                "metadata"
            ]["checkpoint_version"],
            "joint_source_model_version": joint_source_checkpoint["metadata"][
                "model_version"
            ],
            "beta_model_state_imported": True,
            "v3_or_v5_shared_beta_state_imported_strict": True,
            "v6_exact_prox_has_no_trainable_parameters": True,
            "v6_refinement_steps": KALBI_EXACT_PG_STEPS,
            "v6_initial_ritz_safety": KALBI_EXACT_PG_RITZ_SAFETY,
            "beta_optimizer_state_imported": False,
            "beta_scheduler_state_imported": False,
            "selector_state_imported": False,
            "selector_v23_scratch": True,
            "selector_optimizer_state_imported": False,
            "selector_update_counter_reset": True,
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
        "v6_beta_only_plus_v23_selector_only_to_v7_joint",
        "beta_checkpoint_import_selector_scratch",
        "v3_or_v5_beta_import_v23_selector_scratch",
        "v21_beta_optimizer_stream_continuation_v20_selector_to_v22_geometry",
        (
            "v22_beta_and_shared_selector_optimizer_continuation_"
            "v23_independent_raw_lambda_head"
        ),
    }
    run_selector_imported = (
        transitioning_v7 or transitioning_v22 or transitioning_v21
    )
    run_selector_started_from_scratch = run_initialization_mode in {
        "full_scratch",
        "beta_checkpoint_import_selector_scratch",
        "v3_or_v5_beta_import_v23_selector_scratch",
    }

    if (
        initial_checkpoint is None
        or importing_beta_checkpoint
        or importing_joint_checkpoint
        or args.resume_beta_only_from_joint
        or args.resume_beta_only_data_transition
    ):
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

    store.remember_adopted_pointer(active_pointer)
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
    if (
        transitioning_v7
        or resume_checkpoint is not None
        or v22_transition_checkpoint is not None
        or v21_transition_checkpoint is not None
    ):
        rank_states = initial_checkpoint["rank_states"]
        if sorted(int(item["rank"]) for item in rank_states) != list(range(world)):
            raise TrainerError("continuation rank-state inventory drift")
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
            None
            if (
                rank_state is None
                or args.resume_beta_only_from_joint
                or args.resume_beta_only_data_transition
            )
            else rank_state["stream"]
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
        raise TrainerError("KALBI V7 K/S/ridge architecture contract drift")
    if (
        str(model.kalbi_refinement_mode) != "full_p_block_prox"
        or int(model.kalbi_refinement_sweeps) != 3
        or int(model.kalbi_refinement_block_size) != 4096
    ):
        raise TrainerError(
            "KALBI V7 full-P block-prox refinement contract drift"
        )
    model_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    if model_parameter_count != EXPECTED_MODEL_PARAMETER_COUNT:
        raise TrainerError(
            "KALBI V7 model parameter-count drift: "
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
        resume_checkpoint is None and not transitioning_v7
    )
    if initialize_selector_from_scratch and is_main:
        init_trace("selector_cpu_initialization_started")
        torch.manual_seed(args.seed + 17)
        selector_cpu = DirectPenaltyMLPQuerySelectorV23IndependentRawLambda(
            hidden=args.selector_hidden,
            raw_lambda_min=V23_RAW_LAMBDA_MIN,
            raw_lambda_max=V23_RAW_LAMBDA_MAX,
            initial_raw_lambda=V23_INITIAL_RAW_LAMBDA,
            derivative_floor=1e-6,
            squareplus_scale=0.02,
            direct_head_init_std=1e-3,
        )
        calibration = None
        v22_import_report = None
        v20_import_report = None
        if v22_transition_checkpoint is not None:
            v22_import_report = load_v22_state_without_lambda_head(
                selector_cpu,
                v22_transition_checkpoint["selector_state_dict"],
                initial_raw_lambda=V23_INITIAL_RAW_LAMBDA,
            )
        elif v20_selector_checkpoint is not None:
            v20_import_report = load_v20_state_dict(
                selector_cpu,
                v20_selector_checkpoint["selector_state_dict"],
            )
        else:
            calibration = selector_cpu.fit_penalty_parameter_cache(
                cache.direct_parameters
            )
        selector_cpu.load_input_scaler_cache(input_scaler_payload)
        selector_state_holder[0] = selector_cpu.state_dict()
        selector_initialization_holder[0] = {
            "calibration": calibration,
            "architecture": selector_cpu.architecture_report(),
            "initialization": (
                "all_v22_state_except_reinitialized_raw_lambda_head"
                if v22_import_report is not None
                else
                "all_v20_state_plus_fresh_v22_geometry"
                if v20_import_report is not None
                else "new_v23_k32s32_selector_from_seed"
            ),
            "v22_state_import": v22_import_report,
            "v20_state_import": v20_import_report,
            "reason": (
                "V22 representation and canonical penalty with an independent "
                "raw-lambda head; d0 and max-drho are telemetry only"
            ),
        }
        init_trace("selector_cpu_initialization_finished")
    dist.broadcast_object_list(selector_state_holder, src=0)
    dist.broadcast_object_list(selector_initialization_holder, src=0)
    torch.manual_seed(args.seed + 17)
    selector = DirectPenaltyMLPQuerySelectorV23IndependentRawLambda(
        hidden=args.selector_hidden,
        raw_lambda_min=V23_RAW_LAMBDA_MIN,
        raw_lambda_max=V23_RAW_LAMBDA_MAX,
        initial_raw_lambda=V23_INITIAL_RAW_LAMBDA,
        derivative_floor=1e-6,
        squareplus_scale=0.02,
        direct_head_init_std=1e-3,
    )
    if initialize_selector_from_scratch:
        selector.load_state_dict(selector_state_holder[0], strict=True)
    elif transitioning_v7:
        selector.load_state_dict(
            v7_selector_transition_checkpoint["selector_state_dict"],
            strict=True,
        )
    elif resume_checkpoint is not None:
        selector.load_state_dict(
            initial_checkpoint["selector_state_dict"], strict=True
        )
    else:
        raise TrainerError("selector initialization mode is inconsistent")
    selector.load_input_scaler_cache(input_scaler_payload)
    selector_parameter_count = sum(
        parameter.numel() for parameter in selector.parameters()
    )
    if selector_parameter_count != EXPECTED_SELECTOR_PARAMETER_COUNT:
        raise TrainerError(
            "V23-K32S32 parameter-count drift: "
            f"expected {EXPECTED_SELECTOR_PARAMETER_COUNT}, "
            f"got {selector_parameter_count}"
        )
    if args.beta_only:
        selector.eval()
        for parameter in selector.parameters():
            parameter.requires_grad_(False)
        init_trace("selector_frozen_on_cpu")
    else:
        selector.to(device)
        init_trace("selector_on_device")
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
    elif initial_checkpoint is not None:
        model.load_state_dict(
            initial_checkpoint["model_state_dict"], strict=True
        )

    model = DDP(
        model, device_ids=[local_rank], output_device=local_rank,
        broadcast_buffers=True, find_unused_parameters=False, init_sync=True,
    )
    if args.beta_only:
        selector_train = selector
        selector_core = selector
    else:
        selector_train = DDP(
            selector, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=True, find_unused_parameters=False, init_sync=True,
        )
        selector_core = selector_train.module
    init_trace("ddp_wrappers_ready")
    core = model.module
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
        raise TrainerError("Kalbi V6 beta optimizer parameter partition is incomplete")
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
        raise TrainerError("Kalbi V6 optimizer has no nonempty parameter group")
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
    carried_selector_optimizer_state_dict = (
        resume_checkpoint.get("selector_optimizer_state_dict")
        if args.beta_only and resume_checkpoint is not None
        else None
    )
    selector_optimizer: torch.optim.Optimizer | None = None
    if not args.beta_only:
        penalty_branch_parameters = [
            *selector_core.penalty_mlp.parameters(),
            *selector_core.penalty_parameter_head.parameters(),
        ]
        penalty_branch_ids = {
            id(parameter) for parameter in penalty_branch_parameters
        }
        selector_other_parameters = [
            parameter for parameter in selector_train.parameters()
            if id(parameter) not in penalty_branch_ids
        ]
        selector_optimizer = torch.optim.AdamW(
            [
                {
                    "params": selector_other_parameters,
                    "lr": args.selector_lr,
                    "name": "selector_context_lambda_trunks",
                },
                {
                    "params": penalty_branch_parameters,
                    "lr": args.direct_head_lr,
                    "name": "penalty_mlp_direct96_head",
                },
            ],
            lr=args.selector_lr,
            weight_decay=0.0,
        )

    phase_target_steps = (
        args.steps
        - int(initialization_provenance.get("v22_source_completed_steps", 0))
    )
    if phase_target_steps <= 0:
        raise TrainerError("V23 continuation phase has no remaining updates")

    def multiplier(step: int) -> float:
        if step < args.warmup_steps:
            return float(step + 1) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(
            max(1, phase_target_steps - args.warmup_steps)
        )
        return 0.5 * (
            1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0))
        )

    beta_scheduler = torch.optim.lr_scheduler.LambdaLR(beta_optimizer, multiplier)
    completed_steps = 0
    previous_checkpoint_id: str | None = None
    beta_optimizer_transition_report: dict[str, Any] | None = None
    beta_continuation_checkpoint = (
        v7_beta_transition_checkpoint
        or resume_checkpoint
        or v22_transition_checkpoint
        or v21_transition_checkpoint
    )
    if beta_continuation_checkpoint is not None:
        if transitioning_v7:
            beta_optimizer_transition_report = (
                audit_optimizer_state_by_parameter_name(
                    label="V7 beta",
                    module=core,
                    optimizer=beta_optimizer,
                    source_state_dict=beta_continuation_checkpoint[
                        "beta_optimizer_state_dict"
                    ],
                )
            )
        beta_optimizer.load_state_dict(
            beta_continuation_checkpoint["beta_optimizer_state_dict"]
        )
        if v22_transition_checkpoint is None:
            beta_scheduler.load_state_dict(
                beta_continuation_checkpoint["beta_scheduler_state_dict"]
            )
        else:
            # V22 ended at the floor of its cosine schedule. Keep its Adam
            # moments but start a warm V23 continuation phase.
            new_lrs = [
                base_lr * multiplier(beta_scheduler.last_epoch)
                for base_lr in beta_scheduler.base_lrs
            ]
            for group, learning_rate in zip(
                beta_optimizer.param_groups, new_lrs
            ):
                group["lr"] = learning_rate
            beta_scheduler._last_lr = new_lrs
        optimizer_to_device(beta_optimizer, device)
        completed_steps = int(beta_continuation_checkpoint["completed_steps"])
        previous_checkpoint_id = beta_continuation_checkpoint["metadata"].get(
            "checkpoint_id"
        )
        if args.resume_beta_only_from_joint:
            previous_checkpoint_id = None
        if args.resume_beta_only_data_transition:
            previous_checkpoint_id = None
        if transitioning_v7:
            previous_checkpoint_id = None
    selector_optimizer_transition_report: dict[str, Any] | None = None
    if args.beta_only:
        if carried_selector_optimizer_state_dict is None:
            raise TrainerError(
                "beta-only resume requires carried selector optimizer state"
            )
        selector_optimizer_transition_report = {
            "optimizer_instantiated": False,
            "state_carried_without_instantiation": True,
            "state_entry_count": len(
                carried_selector_optimizer_state_dict.get("state", {})
            ),
        }
    elif transitioning_v7:
        assert selector_optimizer is not None
        selector_optimizer_transition_report = (
            audit_optimizer_state_by_parameter_name(
                label="V7 selector",
                module=selector_core,
                optimizer=selector_optimizer,
                source_state_dict=v7_selector_transition_checkpoint[
                    "selector_optimizer_state_dict"
                ],
            )
        )
        selector_optimizer.load_state_dict(
            v7_selector_transition_checkpoint[
                "selector_optimizer_state_dict"
            ]
        )
        optimizer_to_device(selector_optimizer, device)
        selector_optimizer_transition_report.update({
            "source": "selector_only_step_20300",
            "state_entry_count": len(selector_optimizer.state),
            "whole_optimizer_state_loaded_after_named_parameter_audit": True,
        })
    elif resume_checkpoint is not None:
        assert selector_optimizer is not None
        selector_optimizer.load_state_dict(
            resume_checkpoint["selector_optimizer_state_dict"]
        )
        optimizer_to_device(selector_optimizer, device)
    elif v22_transition_checkpoint is not None:
        assert selector_optimizer is not None
        selector_optimizer.load_state_dict(
            v22_transition_checkpoint["selector_optimizer_state_dict"]
        )
        imported_state_count = len(selector_optimizer.state)
        reset_state_count = 0
        for parameter in selector_core.lambda_head.parameters():
            if parameter in selector_optimizer.state:
                selector_optimizer.state.pop(parameter)
                reset_state_count += 1
        selector_optimizer_transition_report = {
            "imported_parameter_state_count": imported_state_count,
            "reset_lambda_parameter_state_count": reset_state_count,
            "remaining_parameter_state_count": len(selector_optimizer.state),
            "shared_optimizer_state_preserved": True,
        }
        optimizer_to_device(selector_optimizer, device)

    knots = canonical_grid(device=device)

    if is_main:
        if resume_checkpoint is None:
            if args.out_dir.exists():
                if not (
                    importing_beta_checkpoint
                    or importing_joint_checkpoint
                    or transitioning_v7
                    or transitioning_v22
                    or transitioning_v21
                ):
                    raise TrainerError(
                        f"refusing existing non-resume output directory: {args.out_dir}"
                    )
                if any(args.out_dir.glob("checkpoint_step*.pt")):
                    raise TrainerError(
                        "initialization import cannot target a directory "
                        "containing checkpoints"
                    )
            else:
                args.out_dir.mkdir(parents=True, exist_ok=False)
        elif (
            not args.resume_beta_only_data_transition
            and args.resume.resolve().parent != args.out_dir.resolve()
        ):
            raise TrainerError("resume checkpoint must remain in its output directory")
        elif args.resume_beta_only_data_transition:
            if args.out_dir.exists() and any(
                args.out_dir.glob("checkpoint_step*.pt")
            ):
                raise TrainerError(
                    "beta data-transition output directory already has checkpoints"
                )
            args.out_dir.mkdir(parents=True, exist_ok=True)
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
        "block_prox_v7": file_sha256(
            Path(__file__).with_name("block_prox_v7.py")
        ),
        "exact_direct96_prox": file_sha256(
            Path(__file__).with_name("exact_direct96_prox.py")
        ),
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
        "selector_v17_k32s32_base": file_sha256(
            Path(__file__).with_name("selector_v17_k32s32.py")
        ),
        "selector_v18_k32s32_base": file_sha256(
            Path(__file__).with_name("selector_v18_k32s32.py")
        ),
        "selector_v19_k32s32": file_sha256(
            Path(__file__).with_name("selector_v19_k32s32.py")
        ),
        "selector_v20_yconditioned": file_sha256(
            Path(__file__).with_name("selector_v20_yconditioned.py")
        ),
        "selector_v22_v20geometry_scaled": file_sha256(
            Path(__file__).with_name(
                "selector_v22_v20geometry_scaled.py"
            )
        ),
        "selector_v23_independent_rawlambda": file_sha256(
            Path(__file__).with_name(
                "selector_v23_independent_rawlambda.py"
            )
        ),
        "input_scaler_cache": args.input_scaler_cache_sha256,
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
        "beta_schedule_phase_steps": phase_target_steps,
        "beta_schedule_restarted_from_v22": transitioning_v22,
        "selector_lr": args.selector_lr,
        "direct_head_lr": args.direct_head_lr,
        "selector_hidden": args.selector_hidden,
        "selector_stages": V17_STAGES,
        "selector_topk": V17_TOPK,
        "selector_candidate_pool": None,
        "selector_feature_chunk_size": V19_FEATURE_CHUNK_SIZE,
        "selector_all_features_rescored_each_stage": True,
        "selector_selection_without_replacement": True,
        "selector_max_probe_rows": V17_MAX_PROBE_ROWS,
        "selector_stage_gain_rate": V17_STAGE_GAIN_RATE,
        "selector_cumulative_joint_ridge_path": True,
        "selector_cumulative_refit_changes_screening_residual": False,
        "selector_cumulative_path_shape": [V17_STAGES * V17_TOPK, V17_STAGES],
        "selector_prediction_loss_only": True,
        "selector_fusion_branches": list(V23_FUSION_BRANCHES),
        "selector_fusion_mode": (
            "scaled_v20_sum_fusion_plus_gated_geometry_quantile_dimension"
        ),
        "selector_dimension_context": [
            "log_actual_N",
            "log_actual_P",
            "log_context_lambda_max",
        ],
        "selector_public_lambda": (
            "independent_raw_lambda_directly_on_beta_solver_support"
        ),
        "selector_raw_lambda_support": [
            V23_RAW_LAMBDA_MIN,
            V23_RAW_LAMBDA_MAX,
        ],
        "selector_initial_raw_lambda": V23_INITIAL_RAW_LAMBDA,
        "selector_penalty_output": (
            "V20_canonical_Direct96_rho_and_drho_consistent_builder"
        ),
        "selector_d0_role": (
            "beta_solver_penalty_mechanics_and_zero_kkt_telemetry"
        ),
        "selector_max_drho_role": "force_ratio_telemetry_only",
        "selector_penalty_shape_controls_raw_lambda_conversion": False,
        "input_scaler_cache": str(args.input_scaler_cache.resolve()),
        "input_scaler_cache_sha256": args.input_scaler_cache_sha256,
        "input_scaler_version": V22_SCALER_VERSION,
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
        "kalbi_v7_full_p_block_proximal_descent": {
            "mode": "full_p_block_prox",
            "sweeps": KALBI_BLOCK_PROX_SWEEPS,
            "block_size": KALBI_BLOCK_PROX_BLOCK_SIZE,
            "sweep_order": ["forward", "reverse", "forward"],
            "data_gradient": "current_residual_block_gradient_without_Gram",
            "penalty": "Direct96_piecewise_quadratic_integral_of_linear_drho",
            "prox": "exact_scalar_interval_search_plus_closed_form",
            "curvature_margin": KALBI_EXACT_PG_CURVATURE_MARGIN,
            "data_majorization_margin": KALBI_EXACT_PG_DATA_MARGIN,
            "block_spectral_safety": KALBI_EXACT_PG_RITZ_SAFETY,
            "backtrack_factor": KALBI_EXACT_PG_BACKTRACK_FACTOR,
            "max_backtracks": KALBI_EXACT_PG_MAX_BACKTRACKS,
            "feature_chunk": KALBI_EXACT_PG_FEATURE_CHUNK,
            "trace_fallback_is_block_lipschitz_upper_bound": True,
            "all_valid_features_visited_per_sweep": True,
            "activation_checkpointing": True,
            "p_by_p_matrix": False,
            "straight_through_estimator": False,
            "learned_refinement_parameters": False,
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
            "selector_version": V23_SELECTOR_VERSION,
            "selector_lambda_scaling_contract": (
                SELECTOR_LAMBDA_SCALING_CONTRACT
            ),
            "world_size": world,
        }
        for key, expected in required_resume.items():
            if metadata.get(key) != expected:
                raise TrainerError(f"resume contract mismatch for {key}")
        if args.resume_beta_only_data_transition:
            expected_source_contracts = {
                "tabicl_v2": {
                    "dataset_contract": (
                        "tabiclv2_graph_scm2048_qfilter_bnn128_"
                        "expvarnoise_mixed50_tabicl_linear_v12"
                    ),
                    "response_contract": (
                        "mixed50_official_tabicl_or_controlled_linear_y_"
                        "qualityfilter_center_scale_clip10_rescale_v1"
                    ),
                }
            }
            source_contracts = metadata.get("scm_prior_contracts")
            if source_contracts is not None and (
                source_contracts != expected_source_contracts
            ):
                raise TrainerError(
                    "data-contract transition source SCM contract is not V12"
                )
            source_pipeline_sha256 = (
                metadata.get("exact_training_arguments", {}).get(
                    "expected_pipeline_sha256"
                )
            )
            if source_pipeline_sha256 != V12_SOURCE_PIPELINE_SHA256:
                raise TrainerError(
                    "data-contract transition source pipeline is not certified V12"
                )
        if (
            args.resume_beta_only_from_joint
            or args.resume_beta_only_data_transition
        ):
            prior_hashes = dict(metadata.get("source_hashes", {}))
            current_hashes = dict(source_hashes)
            prior_trainer_hash = prior_hashes.pop("trainer", None)
            current_trainer_hash = current_hashes.pop("trainer", None)
            if (
                prior_hashes != current_hashes
                or not prior_trainer_hash
                or not current_trainer_hash
                or prior_trainer_hash == current_trainer_hash
            ):
                transition_name = (
                    "joint-to-beta-only"
                    if args.resume_beta_only_from_joint
                    else "beta-only data-contract"
                )
                raise TrainerError(
                    f"{transition_name} source audit permits only the trainer "
                    "hash to change"
                )
            prior_arguments = dict(
                metadata.get("exact_training_arguments", {})
            )
            if args.resume_beta_only_from_joint:
                if prior_arguments.get("beta_only") is not False:
                    raise TrainerError(
                        "joint-to-beta-only transition source is not a joint run"
                    )
                prior_arguments["beta_only"] = True
            elif prior_arguments.get("beta_only") is not True:
                raise TrainerError(
                    "data-contract transition source is not a beta-only run"
                )
            prior_arguments["trainer_id"] = exact_arguments["trainer_id"]
            prior_arguments["rolling_data_root"] = exact_arguments[
                "rolling_data_root"
            ]
            prior_arguments["rolling_control_root"] = exact_arguments[
                "rolling_control_root"
            ]
            prior_arguments["expected_pipeline_sha256"] = exact_arguments[
                "expected_pipeline_sha256"
            ]
            prior_arguments["initialization_provenance"] = exact_arguments[
                "initialization_provenance"
            ]
            # These fields describe how the source run originally entered V23,
            # not optimizer or model semantics of the resumed target process.
            for provenance_key in (
                "beta_schedule_restarted_from_v22",
                "selector_initialized_from_checkpoint",
            ):
                prior_arguments[provenance_key] = exact_arguments[
                    provenance_key
                ]
            if prior_arguments != exact_arguments:
                argument_diff = {
                    key: {
                        "checkpoint": prior_arguments.get(key),
                        "target": exact_arguments.get(key),
                    }
                    for key in sorted(
                        set(prior_arguments) | set(exact_arguments)
                    )
                    if prior_arguments.get(key) != exact_arguments.get(key)
                }
                transition_name = (
                    "joint-to-beta-only"
                    if args.resume_beta_only_from_joint
                    else "beta-only data-contract"
                )
                raise TrainerError(
                    f"{transition_name} exact argument contract mismatch: "
                    + json.dumps(argument_diff, sort_keys=True, default=str)
                )
        else:
            if metadata.get("source_hashes") != source_hashes:
                raise TrainerError("resume contract mismatch for source_hashes")
            if metadata.get("exact_training_arguments") != exact_arguments:
                raise TrainerError(
                    "resume contract mismatch for exact_training_arguments"
                )

    event_kind = (
        "v7_joint_transition_from_beta_and_selector_checkpoints"
        if transitioning_v7
        else "joint_to_beta_only_strict_resume"
        if args.resume_beta_only_from_joint
        else "beta_only_mixed_response_data_contract_transition"
        if args.resume_beta_only_data_transition
        else "resume"
        if resume_checkpoint is not None
        else "v22_continuation_v23_independent_raw_lambda"
        if transitioning_v22
        else "v21_beta_continuation_v20_selector_to_v22_geometry"
        if transitioning_v21
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
        "selector_version": V23_SELECTOR_VERSION,
        "selector_lambda_scaling_contract": SELECTOR_LAMBDA_SCALING_CONTRACT,
        "canonical_cache": cache_report,
        "scm_prior_contracts": {
            prior: {"dataset_contract": values[0], "response_contract": values[1]}
            for prior, values in SCM_CONTRACTS.items()
        },
        "mixture_contract": (
            "TabICLv2 V13 SCM2048-qfilter plus BNN128 with independent "
            "featurewise Exp(1) noise; 50pct official TabICL outcome, "
            "25pct P-referenced linear outcome conditioned on k<N/3, and "
            "25pct N-referenced linear outcome with k<N/3"
        ),
        "selector_objective": SELECTOR_OBJECTIVE,
        "selector_input_contract": (
            "standardized_context_train_X_y_only_no_heldout_y"
        ),
        "selector_y_branch": (
            "Linear1x512_GELU_symmetric_mean_populationstd_max_Linear1536x512"
        ),
        "selector_dimension_branch": (
            "cached_scaled_logN_logP_logLambdaMax_Linear3x512_GELU_"
            "Linear512x512"
        ),
        "selector_fusion": (
            "scaled_V20_task_code_plus_learned_gate_times_"
            "geometry_quantile_dimension_auxiliary_code"
        ),
        "selector_lambda_output": (
            "independent_raw_lambda_directly_used_by_KALBI_V7"
        ),
        "selector_raw_lambda_support": [
            V23_RAW_LAMBDA_MIN,
            V23_RAW_LAMBDA_MAX,
        ],
        "selector_ratio_outputs": (
            "raw_over_lambdaMax_zeroKKT_and_force_are_telemetry_only"
        ),
        "selector_penalty_head": (
            "V20_canonical_Direct96_head"
        ),
        "selector_input_scaler_cache_sha256": args.input_scaler_cache_sha256,
        "selector_heldout_y_used_as_input": False,
        "selector_uses_beta_solver_input": False,
        "selector_stages": V17_STAGES,
        "selector_topk": V17_TOPK,
        "selector_candidate_pool": None,
        "selector_feature_chunk_size": V19_FEATURE_CHUNK_SIZE,
        "selector_all_features_rescored_each_stage": True,
        "selector_selection_without_replacement": True,
        "selector_max_probe_rows": V17_MAX_PROBE_ROWS,
        "selector_stage_gain_rate": V17_STAGE_GAIN_RATE,
        "selector_cumulative_joint_ridge_path": True,
        "selector_cumulative_refit_changes_screening_residual": False,
        "selector_cumulative_path_shape": [V17_STAGES * V17_TOPK, V17_STAGES],
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
        "beta_optimizer_transition": beta_optimizer_transition_report,
        "selector_optimizer_transition": selector_optimizer_transition_report,
        "beta_schedule_phase_steps": phase_target_steps,
        "beta_schedule_restarted_from_v22": transitioning_v22,
        "resuming": resume_checkpoint is not None,
        "fixed_ridge_numerics": fixed_ridge_numerical_diagnostics(),
        "source_hashes": source_hashes,
        "exact_training_arguments": exact_arguments,
        "completed_steps": completed_steps,
    })

    beta_hash_before = assert_ddp_synced(core, world, "initial_beta")
    selector_hash_before = assert_ddp_synced(selector_core, world, "initial_selector")
    if args.beta_only:
        if selector_optimizer is not None:
            raise TrainerError("selector optimizer exists in beta-only mode")
        if any(parameter.requires_grad for parameter in selector_core.parameters()):
            raise TrainerError("selector has trainable parameters in beta-only mode")
    compute_seconds = 0.0
    compute_steps = 0
    selector_trainer_state_source = (
        v7_selector_transition_checkpoint
        if transitioning_v7
        else resume_checkpoint
    )
    consecutive_global_penalty_zero_updates = (
        0
        if selector_trainer_state_source is None
        else int(selector_trainer_state_source.get("trainer_state", {}).get(
            "consecutive_global_penalty_zero_updates", 0
        ))
    )
    selector_updates = (
        0
        if selector_trainer_state_source is None
        else int(selector_trainer_state_source.get("trainer_state", {}).get(
            "selector_updates", 0
        ))
    )
    selector_zero_gradient_skips = (
        0
        if selector_trainer_state_source is None
        else int(selector_trainer_state_source.get("trainer_state", {}).get(
            "selector_zero_gradient_skips", 0
        ))
    )
    starting_completed_steps = completed_steps
    starting_selector_updates = selector_updates

    target_steps = args.steps if args.stop_after is None else args.stop_after
    while completed_steps < target_steps:
        started = time.perf_counter()
        beta_optimizer.zero_grad(set_to_none=True)
        beta_metric_names = (
            "loss",
            "data_half",
            "mse",
            "penalty",
            "objective",
            "f_zero",
            "kkt_top32",
            "nnz",
            *tuple(
                f"refinement_{name}" for name in REFINEMENT_PATH_METRIC_NAMES
            ),
            *tuple(
                f"refinement_{name}_final"
                for name in REFINEMENT_PATH_METRIC_NAMES
            ),
            *REFINEMENT_AGGREGATE_METRIC_NAMES,
        )
        beta_metrics = {key: 0.0 for key in beta_metric_names}
        beta_refinement_by_iteration = {
            name: torch.zeros(
                KALBI_EXACT_PG_STEPS, dtype=torch.float32, device=device
            )
            for name in REFINEMENT_PATH_METRIC_NAMES
        }
        beta_shapes: list[tuple[int, int]] = []
        beta_task_strata: list[dict[str, Any]] = []
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
                result = objective_kalbi_v7_blockprox3(model, payload, knots)
                if not bool(torch.isfinite(result["loss"])):
                    raise TrainerError("nonfinite beta loss")
                (result["loss"] / float(args.grad_accum_steps)).backward()
            for key in beta_metrics:
                beta_metrics[key] += (
                    float(result[key].detach()) / float(args.grad_accum_steps)
                )
            task_info = info[0]
            source_id = int(task_info["outcome_source_id"])
            task_stratum = {
                "outcome_source": task_info["outcome_source"],
                "outcome_source_id": source_id,
                "n": int(task_info["n"]),
                "p": int(task_info["p"]),
                "loss": float(result["loss"].detach()),
                "mse": float(result["mse"].detach()),
                "objective": float(result["objective"].detach()),
                "kkt_top32": float(result["kkt_top32"].detach()),
            }
            if source_id == OUTCOME_SOURCE_LINEAR:
                task_stratum.update(
                    {
                        "support_size": int(
                            task_info["linear_support_size"]
                        ),
                        "support_fraction": float(
                            task_info["linear_support_fraction"]
                        ),
                        "support_regime": SUPPORT_REGIME_NAMES[
                            int(task_info["linear_support_regime_id"])
                        ],
                        "coefficient_family": COEFFICIENT_FAMILY_NAMES[
                            int(task_info["linear_coefficient_family_id"])
                        ],
                        "snr_regime": SNR_REGIME_NAMES[
                            int(task_info["linear_snr_regime_id"])
                        ],
                        "error_family": ERROR_FAMILY_NAMES[
                            int(task_info["linear_error_family_id"])
                        ],
                        "target_snr": float(task_info["linear_target_snr"]),
                        "realized_snr": float(
                            task_info["linear_realized_snr"]
                        ),
                    }
                )
            beta_task_strata.append(task_stratum)
            for name in REFINEMENT_PATH_METRIC_NAMES:
                path = result[f"refinement_{name}_by_iteration"].detach().float()
                if path.shape != (KALBI_EXACT_PG_STEPS,):
                    raise TrainerError(
                        f"beta refinement path shape drift for {name}: "
                        f"{tuple(path.shape)}"
                    )
                beta_refinement_by_iteration[name].add_(
                    path / float(args.grad_accum_steps)
                )
        beta_refinement_by_iteration_json = {
            name: values.cpu().tolist()
            for name, values in beta_refinement_by_iteration.items()
        }
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
            if completed_steps == starting_completed_steps + 1:
                if assert_ddp_synced(
                    core, world, "post_first_continuation_beta_update"
                ) == beta_hash_before:
                    raise TrainerError(
                        "beta model did not change on first continuation update"
                    )
            compute_seconds += time.perf_counter() - started
            compute_steps += 1
            training_mode = (
                "beta_only"
                if args.beta_only
                else "joint_beta_v23_sparse_schedule"
            )

            if completed_steps % args.log_every == 0:
                local_monitor = {
                    "rank": rank,
                    "catalog_epoch": int(active_pointer["epoch"]),
                    "catalog_sha256": active_catalog_sha,
                    "stream_reload_count": stream.reload_count,
                    "beta_metrics": beta_metrics,
                    "beta_refinement_by_iteration": (
                        beta_refinement_by_iteration_json
                    ),
                    "beta_total_gradient_preclip": beta_total_gradient_preclip,
                    "beta_group_gradients_preclip": beta_group_gradients_preclip,
                    "beta_group_gradients_postclip": beta_group_gradients_postclip,
                    "sampled_lambda_min": sampled_lambda_min,
                    "kalbi_fixed_ridge_numerics": (
                        fixed_ridge_numerical_diagnostics()
                    ),
                    "beta_shapes": beta_shapes,
                    "beta_task_strata": beta_task_strata,
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
                store.remember_adopted_pointer(active_pointer)
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
                        "selector_optimizer_state_dict": (
                            carried_selector_optimizer_state_dict
                        ),
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
                            "selector_version": V23_SELECTOR_VERSION,
                            "selector_lambda_scaling_contract": (
                                SELECTOR_LAMBDA_SCALING_CONTRACT
                            ),
                            "training_mode": "beta_only",
                            "selector_update_enabled": False,
                            "selector_state_carried_forward_unchanged": True,
                            "selector_trainable_parameter_count": 0,
                            "selector_optimizer_instantiated": False,
                            "selector_optimizer_state_carried_unchanged": True,
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
                            "beta_optimizer_transition": (
                                beta_optimizer_transition_report
                            ),
                            "selector_optimizer_transition": (
                                selector_optimizer_transition_report
                            ),
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
                        or state_dict_sha256(reloaded["selector_state_dict"])
                        != selector_hash_before
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

        assert selector_optimizer is not None
        selector_optimizer.zero_grad(set_to_none=True)
        was_training = core.training
        core.eval()
        for parameter in core.parameters():
            parameter.grad = None
            parameter.requires_grad_(False)
        selector_losses: list[torch.Tensor] = []
        selector_hard_losses: list[torch.Tensor] = []
        selector_mses: list[torch.Tensor] = []
        selector_f_hats: list[torch.Tensor] = []
        selector_f_zeros: list[torch.Tensor] = []
        selector_fhat_fzero_ratios: list[torch.Tensor] = []
        direct_gradients: list[torch.Tensor] = []
        native_gradients: list[torch.Tensor] = []
        d0_gradients: list[torch.Tensor] = []
        lambda_gradients: list[torch.Tensor] = []
        selector_ratios: list[torch.Tensor] = []
        selector_raw_lambdas: list[torch.Tensor] = []
        selector_lambda_maxes: list[torch.Tensor] = []
        selector_force_ratios: list[torch.Tensor] = []
        selector_zero_kkt_ratios: list[torch.Tensor] = []
        selector_concavity_violations: list[torch.Tensor] = []
        selector_derivative_tvs: list[torch.Tensor] = []
        selector_branch_codes: dict[str, list[torch.Tensor]] = {
            "residual_gram": [],
            "beta_trajectory": [],
            "stage_metadata": [],
            "context_y": [],
            "support_geometry": [],
            "support_correlation_quantiles": [],
            "dimension_scale": [],
        }
        selector_beta_nonzero_fractions: list[torch.Tensor] = []
        selector_threshold_means: list[torch.Tensor] = []
        selector_threshold_mins: list[torch.Tensor] = []
        selector_threshold_maxes: list[torch.Tensor] = []
        selector_log_multiplier_means: list[torch.Tensor] = []
        selector_log_multiplier_mins: list[torch.Tensor] = []
        selector_log_multiplier_maxes: list[torch.Tensor] = []
        selector_prethreshold_max_abs: list[torch.Tensor] = []
        selector_beta_max_abs: list[torch.Tensor] = []
        selector_refinement_step_sizes: list[torch.Tensor] = []
        selector_refinement_gammas: list[torch.Tensor] = []
        selector_refinement_kappas: list[torch.Tensor] = []
        selector_refinement_min_denominators: list[torch.Tensor] = []
        selector_refinement_majorization_ratios: list[torch.Tensor] = []
        selector_refinement_backtracks: list[torch.Tensor] = []
        selector_refinement_trace_fallbacks: list[torch.Tensor] = []
        selector_refinement_gradient_rms: list[torch.Tensor] = []
        selector_refinement_zero_thresholds: list[torch.Tensor] = []
        selector_refinement_change_rms: list[torch.Tensor] = []
        selector_refinement_active_fractions: list[torch.Tensor] = []
        selector_refinement_saturated_fractions: list[torch.Tensor] = []
        trajectory_active_stage_counts: list[torch.Tensor] = []
        trajectory_residual_energy_final: list[torch.Tensor] = []
        selector_rhos: list[torch.Tensor] = []
        selector_drhos: list[torch.Tensor] = []
        selector_d0_values: list[torch.Tensor] = []
        selector_reference_slopes: list[torch.Tensor] = []
        selector_ridge_branches: list[torch.Tensor] = []
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
                    trajectory = query["v22_trajectory"]
                    active_count = trajectory["stage_active"].sum(-1).float()
                    if not bool((active_count >= 1.0).all()):
                        raise TrainerError("V23 trajectory produced no active stage")
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
                    selector_d0_values.append(
                        query["penalty_d0"].detach().float().reshape(-1)
                    )
                    selector_reference_slopes.append(
                        query["penalty_reference_slope"].detach().float().reshape(-1)
                    )
                    selector_ridge_branches.append(
                        query["penalty_ridge_branch"].detach().float().reshape(-1)
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
                        raise TrainerError("KALBI V7 decode aux lacks beta_q")
                    required_threshold_aux = (
                        "kalbi_v3_threshold_q",
                        "kalbi_v3_threshold_log_multiplier_q",
                        "kalbi_v3_prethreshold_beta_q",
                    )
                    if any(name not in aux for name in required_threshold_aux):
                        raise TrainerError("Kalbi V3 precursor telemetry is missing")
                    required_refinement_aux = (
                        "kalbi_v6_refinement_step_size_path_q",
                        "kalbi_v6_refinement_gamma_path_q",
                        "kalbi_v6_refinement_kappa_path_q",
                        "kalbi_v6_refinement_min_curvature_denominator_path_q",
                        "kalbi_v6_refinement_majorization_ratio_path_q",
                        "kalbi_v6_refinement_backtracks_path_q",
                        "kalbi_v6_refinement_trace_fallback_path_q",
                        "kalbi_v6_refinement_data_gradient_rms_path_q",
                        "kalbi_v6_refinement_zero_threshold_path_q",
                        "kalbi_v6_refinement_change_rms_path_q",
                        "kalbi_v6_refinement_active_fraction_path_q",
                        "kalbi_v6_refinement_saturated_fraction_path_q",
                    )
                    if any(name not in aux for name in required_refinement_aux):
                        raise TrainerError("KALBI V7 refinement telemetry is missing")
                    visited_features = aux.get(
                        "kalbi_v7_refinement_visited_features_path_q"
                    )
                    if (
                        visited_features is None
                        or visited_features.ndim != 3
                        or int(visited_features.shape[-1]) != 3
                        or not bool(
                            (
                                visited_features
                                == int(x_train.shape[-1])
                            ).all()
                        )
                    ):
                        raise TrainerError(
                            "KALBI V7 did not visit every valid feature in "
                            "each of its three block-prox sweeps"
                        )
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
                        raise TrainerError("KALBI V7 beta path is nonfinite")
                    refinement_values = tuple(
                        aux[name].float() for name in required_refinement_aux
                    )
                    if not all(
                        bool(torch.isfinite(value).all())
                        for value in refinement_values
                    ):
                        raise TrainerError("KALBI V7 refinement path is nonfinite")
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
                    selector_refinement_step_sizes.append(
                        refinement_values[0].mean().detach().reshape(1)
                    )
                    selector_refinement_gammas.append(
                        refinement_values[1].mean().detach().reshape(1)
                    )
                    selector_refinement_kappas.append(
                        refinement_values[2].mean().detach().reshape(1)
                    )
                    selector_refinement_min_denominators.append(
                        refinement_values[3].min().detach().reshape(1)
                    )
                    selector_refinement_majorization_ratios.append(
                        refinement_values[4].max().detach().reshape(1)
                    )
                    selector_refinement_backtracks.append(
                        refinement_values[5].mean().detach().reshape(1)
                    )
                    selector_refinement_trace_fallbacks.append(
                        refinement_values[6].mean().detach().reshape(1)
                    )
                    selector_refinement_gradient_rms.append(
                        refinement_values[7].mean().detach().reshape(1)
                    )
                    selector_refinement_zero_thresholds.append(
                        refinement_values[8].mean().detach().reshape(1)
                    )
                    selector_refinement_change_rms.append(
                        refinement_values[9].mean().detach().reshape(1)
                    )
                    selector_refinement_active_fractions.append(
                        refinement_values[10].mean().detach().reshape(1)
                    )
                    selector_refinement_saturated_fractions.append(
                        refinement_values[11].mean().detach().reshape(1)
                    )
                    prediction = torch.einsum("bnp,bp->bn", x_test.float(), beta)
                    task_mse = (prediction - y_test.float()).square().mean(-1)
                    task_f_hat = 0.5 * task_mse
                    task_f_zero = 0.5 * y_test.float().square().mean(-1)
                    if not bool(
                        torch.isfinite(task_f_zero).all()
                        and (task_f_zero > 1e-12).all()
                    ):
                        raise TrainerError(
                            "selector zero-predictor F_zero is degenerate"
                        )
                    task_fhat_fzero_ratio = task_f_hat / task_f_zero
                    hard_loss = torch.log1p(task_fhat_fzero_ratio).mean()
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
                selector_f_hats.append(task_f_hat.detach())
                selector_f_zeros.append(task_f_zero.detach())
                selector_fhat_fzero_ratios.append(
                    task_fhat_fzero_ratio.detach()
                )
                selector_ratios.append(query["lambda_ratio"].detach())
                selector_raw_lambdas.append(query["raw_lambda"].detach())
                selector_lambda_maxes.append(
                    encoded["lambda_max"].float().detach().reshape(-1)
                )
                selector_force_ratios.append(query["force_ratio"].detach())
                selector_zero_kkt_ratios.append(
                    query["zero_kkt_ratio"].detach()
                )
                selector_concavity_violations.append(
                    query[
                        "penalty_concavity_violation_fraction"
                    ].detach()
                )
                selector_derivative_tvs.append(
                    query["penalty_derivative_tv"].detach()
                )
                for branch_name, query_name in (
                    ("residual_gram", "v21_interaction_code"),
                    ("beta_trajectory", "v21_coefficient_code"),
                    ("stage_metadata", "v21_metadata_code"),
                    ("context_y", "v21_response_code"),
                    ("support_geometry", "v22_geometry_code"),
                    (
                        "support_correlation_quantiles",
                        "v22_correlation_quantile_code",
                    ),
                    ("dimension_scale", "v21_dimension_context_code"),
                ):
                    selector_branch_codes[branch_name].append(
                        query[query_name].detach().float()
                    )
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
                    "cumulative_coefficients": grad_stats(
                        selector_core.cumulative_coefficient_proj.parameters()
                    ),
                    "coefficient_concat": grad_stats(
                        selector_core.coefficient_concat_proj.parameters()
                    ),
                    "stage_metadata": grad_stats(
                        selector_core.stage_metadata_proj.parameters()
                    ),
                    "response_token_projection": grad_stats(
                        selector_core.response_token_projection.parameters()
                    ),
                    "response_pool_projection": grad_stats(
                        selector_core.response_pool_projection.parameters()
                    ),
                    "dimension_context_projection": grad_stats(
                        selector_core.dimension_context_projection.parameters()
                    ),
                    "geometry_projection": grad_stats(
                        selector_core.geometry_projection.parameters()
                    ),
                    "correlation_quantile_projection": grad_stats(
                        selector_core.correlation_quantile_projection.parameters()
                    ),
                    "auxiliary_fusion": grad_stats(
                        selector_core.auxiliary_fusion.parameters()
                    ),
                    "auxiliary_gate": grad_stats(
                        [selector_core.auxiliary_gate]
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
                        "cumulative_coefficients", "coefficient_concat",
                        "stage_metadata", "response_token_projection",
                        "response_pool_projection",
                        "dimension_context_projection", "geometry_projection",
                        "correlation_quantile_projection", "auxiliary_fusion",
                        "auxiliary_gate", "fusion"
                    )
                )
            ):
                raise TrainerError(
                    "V23 context/geometry/dimension/lambda selector gradient is zero"
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
                    "V23 penalty branch stayed globally zero under "
                    "prediction-only loss"
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
        if completed_steps == starting_completed_steps + 1:
            if assert_ddp_synced(
                core, world, "post_first_continuation_beta_update"
            ) == beta_hash_before:
                raise TrainerError(
                    "beta model did not change on first continuation update"
                )
        compute_seconds += time.perf_counter() - started
        compute_steps += 1
        all_rho = torch.cat(selector_rhos)
        all_drho = torch.cat(selector_drhos)
        branch_metrics: dict[str, float] = {}
        for branch_name, values in selector_branch_codes.items():
            branch_matrix = torch.cat(values, dim=0)
            branch_norms = branch_matrix.norm(dim=-1)
            branch_metrics[
                f"selector_branch_{branch_name}_norm_mean"
            ] = float(branch_norms.mean())
            branch_metrics[
                f"selector_branch_{branch_name}_norm_min"
            ] = float(branch_norms.min())
            branch_metrics[
                f"selector_branch_{branch_name}_norm_max"
            ] = float(branch_norms.max())
            branch_metrics[
                f"selector_branch_{branch_name}_task_std_mean"
            ] = float(branch_matrix.std(dim=0, unbiased=False).mean())
        selector_metrics = {
            "selector_objective": float(torch.stack(selector_losses).mean()),
            "selector_hard_prediction_loss": float(
                torch.stack(selector_hard_losses).mean()
            ),
            "selector_heldout_mse": float(torch.cat(selector_mses).mean()),
            "selector_heldout_f_hat": float(
                torch.cat(selector_f_hats).mean()
            ),
            "selector_heldout_f_zero": float(
                torch.cat(selector_f_zeros).mean()
            ),
            "selector_heldout_fhat_fzero_ratio": float(
                torch.cat(selector_fhat_fzero_ratios).mean()
            ),
            "selector_raw_lambda_min": float(
                torch.cat(selector_raw_lambdas).min()
            ),
            "selector_raw_lambda_max": float(
                torch.cat(selector_raw_lambdas).max()
            ),
            "selector_context_lambda_max_min": float(
                torch.cat(selector_lambda_maxes).min()
            ),
            "selector_context_lambda_max_max": float(
                torch.cat(selector_lambda_maxes).max()
            ),
            "selector_raw_lambda_over_lambda_max_min": float(
                torch.cat(selector_ratios).min()
            ),
            "selector_raw_lambda_over_lambda_max_max": float(
                torch.cat(selector_ratios).max()
            ),
            "selector_force_ratio_min": float(
                torch.cat(selector_force_ratios).min()
            ),
            "selector_force_ratio_max": float(
                torch.cat(selector_force_ratios).max()
            ),
            "selector_zero_kkt_ratio_min": float(
                torch.cat(selector_zero_kkt_ratios).min()
            ),
            "selector_zero_kkt_ratio_max": float(
                torch.cat(selector_zero_kkt_ratios).max()
            ),
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
            "kalbi_v6_refinement_step_size_mean": float(
                torch.cat(selector_refinement_step_sizes).mean()
            ),
            "kalbi_v6_refinement_gamma_mean": float(
                torch.cat(selector_refinement_gammas).mean()
            ),
            "kalbi_v6_refinement_kappa_mean": float(
                torch.cat(selector_refinement_kappas).mean()
            ),
            "kalbi_v6_refinement_min_curvature_denominator": float(
                torch.cat(selector_refinement_min_denominators).min()
            ),
            "kalbi_v6_refinement_majorization_ratio_max": float(
                torch.cat(selector_refinement_majorization_ratios).max()
            ),
            "kalbi_v6_refinement_backtracks_mean": float(
                torch.cat(selector_refinement_backtracks).mean()
            ),
            "kalbi_v6_refinement_trace_fallback_fraction": float(
                torch.cat(selector_refinement_trace_fallbacks).mean()
            ),
            "kalbi_v6_refinement_data_gradient_rms_mean": float(
                torch.cat(selector_refinement_gradient_rms).mean()
            ),
            "kalbi_v6_refinement_zero_threshold_mean": float(
                torch.cat(selector_refinement_zero_thresholds).mean()
            ),
            "kalbi_v6_refinement_change_rms_mean": float(
                torch.cat(selector_refinement_change_rms).mean()
            ),
            "kalbi_v6_refined_active_fraction_mean": float(
                torch.cat(selector_refinement_active_fractions).mean()
            ),
            "kalbi_v6_refinement_saturated_fraction_mean": float(
                torch.cat(selector_refinement_saturated_fractions).mean()
            ),
            "v22_active_stages_min": float(
                torch.cat(trajectory_active_stage_counts).min()
            ),
            "v22_active_stages_max": float(
                torch.cat(trajectory_active_stage_counts).max()
            ),
            "v22_final_residual_energy_mean": float(
                torch.cat(trajectory_residual_energy_final).mean()
            ),
            "penalty_rho_batch_std_mean": float(all_rho.std(0, unbiased=False).mean()),
            "penalty_drho_batch_std_mean": float(all_drho.std(0, unbiased=False).mean()),
            "penalty_min_drho": float(all_drho.min()),
            "penalty_d0_min": float(torch.cat(selector_d0_values).min()),
            "penalty_d0_max": float(torch.cat(selector_d0_values).max()),
            "penalty_reference_slope_min": float(
                torch.cat(selector_reference_slopes).min()
            ),
            "penalty_reference_slope_max": float(
                torch.cat(selector_reference_slopes).max()
            ),
            "penalty_ridge_like_fraction": float(
                torch.cat(selector_ridge_branches).mean()
            ),
            "penalty_rho6_error_max": float((all_rho[:, -1] - 1.0).abs().max()),
            "penalty_rho0_error_max": float(all_rho[:, 0].abs().max()),
            "penalty_derivative_tv_mean": float(
                torch.cat(selector_derivative_tvs).mean()
            ),
            "penalty_concavity_violation_fraction_mean": float(
                torch.cat(selector_concavity_violations).mean()
            ),
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
            **branch_metrics,
        }
        if (
            selector_update_applied
            and selector_updates == starting_selector_updates + 1
        ):
            if assert_ddp_synced(
                selector_core, world, "post_first_continuation_selector_update"
            ) == selector_hash_before:
                raise TrainerError(
                    "selector did not change on first continuation update"
                )

        if completed_steps % args.log_every == 0:
            local_monitor = {
                "rank": rank,
                "catalog_epoch": int(active_pointer["epoch"]),
                "catalog_sha256": active_catalog_sha,
                "stream_reload_count": stream.reload_count,
                "beta_metrics": beta_metrics,
                "beta_refinement_by_iteration": (
                    beta_refinement_by_iteration_json
                ),
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
            store.remember_adopted_pointer(active_pointer)
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
                        "selector_version": V23_SELECTOR_VERSION,
                        "selector_lambda_scaling_contract": (
                            SELECTOR_LAMBDA_SCALING_CONTRACT
                        ),
                        "selector_input_contract": "standardized_X_y_only",
                        "selector_stages": V17_STAGES,
                        "selector_topk": V17_TOPK,
                        "selector_candidate_pool": None,
                        "selector_feature_chunk_size": V19_FEATURE_CHUNK_SIZE,
                        "selector_all_features_rescored_each_stage": True,
                        "selector_selection_without_replacement": True,
                        "selector_max_probe_rows": V17_MAX_PROBE_ROWS,
                        "selector_stage_gain_rate": V17_STAGE_GAIN_RATE,
                        "selector_cumulative_joint_ridge_path": True,
                        "selector_cumulative_refit_changes_screening_residual": False,
                        "selector_cumulative_path_shape": [
                            V17_STAGES * V17_TOPK,
                            V17_STAGES,
                        ],
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
                        "beta_optimizer_transition": (
                            beta_optimizer_transition_report
                        ),
                        "selector_optimizer_transition": (
                            selector_optimizer_transition_report
                        ),
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
