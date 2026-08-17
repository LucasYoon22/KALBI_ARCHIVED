#!/usr/bin/env python3
"""Train only V24 while keeping the Kalbi V6 beta solver bitwise frozen.

The source checkpoint contributes the strict Kalbi V6 and V23 state plus the
V23 AdamW state. No beta optimizer or beta scheduler is constructed or saved.
Kalbi V6 remains in eval mode with every parameter ``requires_grad=False``;
autograd is intentionally retained through its operations so the normalized
held-out objective log(1 + F_hat/F_zero) reaches V23 lambda and penalty outputs.
The trainer adopts one immutable rolling catalog at every 100 selector updates
and checkpoints only after the catalog reload.
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
from .selector_v24_additive512_trajectory import (
    DirectPenaltyMLPQuerySelectorV24Additive512,
    V24_LAMBDA_OUTPUT_CONTRACT,
    V24_SCALER_TRANSFORM_CONTRACT,
    V24_SCALER_VERSION,
    V24_SELECTOR_VERSION,
    load_v23_state_into_v24,
)
from .train import (
    ROOT_DEPENDENCY_PATHS,
    TrainerError,
    atomic_torch_save,
    capture_rng,
    make_sampler,
    objective_kalbi_v6_exactprox10,
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
from kalbi_v6_exactprox10_eta4x_v23_independentraw_k32s32_p100k_effbatch32.train_beta_only_v23_supportmix_v13_transition_8gpu import (
    RollingSingleTaskStream as V13RollingSingleTaskStream,
)


JOINT_V23_TRAINER_VERSION = (
    "kalbi_v6_v23_v22geometry_independentraw_k32s32_tabiclv2_v11_p100k_"
    "world8_effbatch32_exactDirect96proxPG10_eta4x_rolling100gb_reload100_"
    "rawLambdaDirect_v1"
)
JOINT_V23_CHECKPOINT_VERSION = (
    "kalbi_v6_v23_v22geometry_independentraw_k32s32_exactDirect96proxPG10_"
    "eta4x_tabiclv2_expvarnoise_v11_p100k_effbatch32_checkpoint_v1"
)
V13_TRAINER_VERSION = (
    "selectoronly_v24_additive512_traj3_cachedscaled_frozen_kalbi_v6_"
    "k32s32_tabiclv2_v13_p100k_"
    "world8_effbatch32_exactDirect96proxPG10_eta4x_rolling100gb_reload100_"
    "heldoutFhatFzero_v1"
)
V13_CHECKPOINT_VERSION = (
    "selectoronly_v24_additive512_traj3_cachedscaled_frozen_kalbi_v6_"
    "k32s32_exactDirect96proxPG10_eta4x_tabiclv2_v13_p100k_effbatch32_"
    "heldoutFhatFzero_checkpoint_v1"
)
TRAINER_VERSION = (
    "selectoronly_v24_additive512_traj3_cachedscaled_frozen_kalbi_v6_"
    "k32s32_tabiclv2_v14_snr0p2to50_p100k_"
    "world8_effbatch32_exactDirect96proxPG10_eta4x_rolling100gb_reload100_"
    "heldoutFhatFzero_v1"
)
CHECKPOINT_VERSION = (
    "selectoronly_v24_additive512_traj3_cachedscaled_frozen_kalbi_v6_"
    "k32s32_exactDirect96proxPG10_eta4x_tabiclv2_v14_snr0p2to50_"
    "p100k_effbatch32_heldoutFhatFzero_checkpoint_v1"
)
V13_PIPELINE_SHA256 = (
    "99f81c6a5bf5083a56f3268a56a0700cd64865b1ff0bba9b23386088d11a9559"
)
V14_PIPELINE_SHA256 = (
    "4fd706b7b254eddb4a832b8107fa3379d128f152716c7ae9f142539064d7e707"
)
V13_DATA_ROOT = Path(
    "/users/sync_data/"
    "kalbi_tabiclv2_v13_mixed50_linear_pn50_strictn3_"
    "p100k_100GB_target3080_seed160076430"
)
V14_DATA_ROOT = Path(
    "/users/sync_data/"
    "kalbi_tabiclv2_v14_mixed50_linear_pn50_strictn3_"
    "snr0p2to50_p100k_100GB_target3080_seed160076430"
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
SELECTOR_LAMBDA_SCALING_CONTRACT = V24_LAMBDA_OUTPUT_CONTRACT
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
        "mixed50_tabicl_linear_supportmix_pn_strictn3_snr0p2to50_v14",
        "mixed50_official_tabicl_or_controlled_linear_supportmix_pn50_"
        "strictn3_snr_loguniform_0p2_50_y_qualityfilter_center_scale_"
        "clip10_rescale_v3",
    ),
}
TABICL_TASK_TARGET_CONTRACT = (
    "official_tabiclv2_graph_scm_y_qualityfilter_center_scale_clip10_rescale_v2"
)
LINEAR_TASK_TARGET_CONTRACT = (
    "controlled_linear_y_supportmix_pn50_strictn3_snr_loguniform_0p2_50_"
    "errorfamily_qualityfilter_center_scale_clip10_rescale_v3"
)
V14_GENERATOR_SHA256 = (
    "d2548b21c63f220ec2d2c5f34bcf1a341e17b4440eb3db80f1ad3ad1029e812f"
)
V14_LINEAR_SNR_CONTRACT = "linear_target_snr_loguniform_0p2_50_v1"
V14_LINEAR_SNR_DISTRIBUTION = "log_uniform"
V14_LINEAR_SNR_MIN = 0.2
V14_LINEAR_SNR_MAX = 50.0
V14_SNR_REGIME_NAMES = (
    "log_quantile_1",
    "log_quantile_2",
    "log_quantile_3",
    "log_quantile_4",
    "log_quantile_5",
)
OUTCOME_SOURCE_TABICL = 0
OUTCOME_SOURCE_LINEAR = 1
OUTCOME_SOURCE_NAMES = {
    OUTCOME_SOURCE_TABICL: "tabicl",
    OUTCOME_SOURCE_LINEAR: "linear",
}
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
EXPECTED_SELECTOR_PARAMETER_COUNT = 25_501_220
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


def migrate_v23_optimizer_state_by_name(
    target_optimizer: torch.optim.Optimizer,
    target_selector: torch.nn.Module,
    source_checkpoint: Mapping[str, Any],
    *,
    selector_lr: float,
    direct_head_lr: float,
    device: torch.device,
) -> dict[str, Any]:
    """Import V23 AdamW moments by fully qualified parameter name."""
    source_selector = DirectPenaltyMLPQuerySelectorV23IndependentRawLambda(
        hidden=V17_HIDDEN_DIM,
        raw_lambda_min=V23_RAW_LAMBDA_MIN,
        raw_lambda_max=V23_RAW_LAMBDA_MAX,
        initial_raw_lambda=V23_INITIAL_RAW_LAMBDA,
        derivative_floor=1e-6,
        squareplus_scale=0.02,
        direct_head_init_std=1e-3,
    )
    source_selector.load_state_dict(
        source_checkpoint["selector_state_dict"], strict=True
    )
    source_penalty = [
        *source_selector.penalty_mlp.parameters(),
        *source_selector.penalty_parameter_head.parameters(),
    ]
    source_penalty_ids = {id(parameter) for parameter in source_penalty}
    source_other = [
        parameter
        for parameter in source_selector.parameters()
        if id(parameter) not in source_penalty_ids
    ]
    source_optimizer = torch.optim.AdamW(
        [
            {
                "params": source_other,
                "lr": selector_lr,
                "name": "selector_context_lambda_trunks",
            },
            {
                "params": source_penalty,
                "lr": direct_head_lr,
                "name": "penalty_mlp_direct96_head",
            },
        ],
        lr=selector_lr,
        weight_decay=0.0,
    )
    source_optimizer.load_state_dict(
        source_checkpoint["selector_optimizer_state_dict"]
    )
    source_named = dict(source_selector.named_parameters())
    target_named = dict(target_selector.named_parameters())
    imported: list[str] = []
    fresh: list[str] = []
    missing_source_state: list[str] = []
    for name, target_parameter in target_named.items():
        source_parameter = source_named.get(name)
        if source_parameter is None:
            fresh.append(name)
            continue
        source_state = source_optimizer.state.get(source_parameter)
        if not source_state:
            missing_source_state.append(name)
            continue
        copied: dict[str, Any] = {}
        for key, value in source_state.items():
            if isinstance(value, torch.Tensor):
                if key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if value.shape != target_parameter.shape:
                        raise TrainerError(
                            f"optimizer moment shape mismatch for {name}.{key}"
                        )
                copied[key] = value.detach().clone().to(device)
            else:
                copied[key] = value
        target_optimizer.state[target_parameter] = copied
        imported.append(name)
    unexpected = sorted(set(source_named) - set(target_named))
    if unexpected:
        raise TrainerError(
            f"V23 optimizer migration has unexpected source names: {unexpected}"
        )
    return {
        "mode": "fully_qualified_parameter_name",
        "imported_parameter_state_count": len(imported),
        "fresh_v24_parameter_count": len(fresh),
        "source_parameters_without_moments": len(missing_source_state),
        "imported_names": sorted(imported),
        "fresh_names": sorted(fresh),
        "source_names_without_moments": sorted(missing_source_state),
        "shared_optimizer_state_preserved": True,
    }


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
    snr_edges = np.geomspace(
        V14_LINEAR_SNR_MIN,
        V14_LINEAR_SNR_MAX,
        len(V14_SNR_REGIME_NAMES) + 1,
    )
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
        source_hashes = entry.get("source_hashes", {})
        if source_hashes.get("generator_tabicl_v2") != V14_GENERATOR_SHA256:
            raise TrainerError("V14 catalog generator SHA mismatch")
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
                or target_snr < V14_LINEAR_SNR_MIN
                or target_snr > V14_LINEAR_SNR_MAX
                or realized_snr < V14_LINEAR_SNR_MIN
                or realized_snr > V14_LINEAR_SNR_MAX
                or abs(math.log(realized_snr / target_snr)) > 2e-5
                or semantic.get("linear_snr_contract")
                != V14_LINEAR_SNR_CONTRACT
                or semantic.get("linear_snr_distribution")
                != V14_LINEAR_SNR_DISTRIBUTION
            ):
                raise TrainerError("catalog shard linear-response certificate failed")
            expected_regime = int(
                np.searchsorted(snr_edges[1:-1], target_snr, side="right")
            )
            if int(semantic.get("linear_snr_regime_id", -1)) != expected_regime:
                raise TrainerError("catalog shard linear SNR regime mismatch")
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


# Use the exact V13 stream implementation that produced the pinned rank states.
# This preserves its NPZ semantic-certificate checks and exact resume contract.
RollingSingleTaskStream = V13RollingSingleTaskStream


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


def selector_heldout_fhat_fzero_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mean log(1 + F_hat/F_zero) and its per-task components."""
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise TrainerError(
            "selector held-out prediction and target must have matching [B,N] shape"
        )
    f_hat = 0.5 * (prediction - target).square().mean(dim=-1)
    f_zero = 0.5 * target.square().mean(dim=-1)
    if not bool(torch.isfinite(f_hat).all()) or not bool(
        torch.isfinite(f_zero).all()
    ):
        raise TrainerError("selector held-out F_hat/F_zero is nonfinite")
    if not bool((f_zero > 1e-12).all()):
        raise TrainerError("selector held-out zero-predictor F_zero is degenerate")
    ratio = f_hat / f_zero
    loss = torch.log1p(ratio).mean()
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(ratio).all()):
        raise TrainerError("selector normalized held-out objective is nonfinite")
    return loss, f_hat, f_zero, ratio


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
    parser.add_argument("--input-scaler-cache", type=Path, required=True)
    parser.add_argument("--input-scaler-cache-sha256", required=True)
    parser.add_argument("--v24-scaler-cache", type=Path, required=True)
    parser.add_argument("--v24-scaler-cache-sha256", required=True)
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sha256")
    parser.add_argument(
        "--resume-data-transition-from-v13",
        action="store_true",
        help=(
            "preserve the frozen beta model, V24 selector, optimizer, RNG, "
            "penalty-cache and sampler state while resetting only the V13 "
            "catalog/stream state onto the certified V14 rolling pipeline"
        ),
    )
    parser.add_argument("--initialize-selector-only-from", type=Path)
    parser.add_argument("--initialize-selector-only-sha256")
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
    parser.add_argument("--selector-update-every", type=int, default=1)
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

    unsupported_initialization = any((
        bool(args.scratch),
        args.transition_v22_from is not None,
        args.transition_v21_from is not None,
        args.initialize_beta_from is not None,
        args.initialize_joint_from is not None,
        args.selector_v20_from is not None,
    ))
    if unsupported_initialization or args.beta_only:
        parser.error(
            "selector-only trainer accepts only --resume or "
            "--initialize-selector-only-from"
        )
    initialization_modes = sum((
        args.resume is not None,
        args.initialize_selector_only_from is not None,
    ))
    if initialization_modes != 1:
        parser.error(
            "choose exactly one of --resume or "
            "--initialize-selector-only-from"
        )
    if args.resume is not None and not args.resume_sha256:
        parser.error("--resume requires --resume-sha256")
    if args.resume_data_transition_from_v13 and args.resume is None:
        parser.error("V13-to-V14 data transition requires --resume")
    if args.resume_data_transition_from_v13:
        if args.expected_pipeline_sha256 != V14_PIPELINE_SHA256:
            parser.error("data transition target is not the certified V14 pipeline")
        if args.rolling_data_root.resolve() != V14_DATA_ROOT.resolve():
            parser.error("data transition target root is not the certified V14 root")
    if (
        args.initialize_selector_only_from is not None
        and not args.initialize_selector_only_sha256
    ):
        parser.error(
            "--initialize-selector-only-from requires "
            "--initialize-selector-only-sha256"
        )
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
        parser.error(f"V24 selector hidden dimension is fixed at {V17_HIDDEN_DIM}")
    if args.grad_accum_steps != EXPECTED_GRAD_ACCUM_STEPS:
        parser.error("8-GPU global beta task batch is fixed at 32")
    if args.selector_grad_accum_steps != EXPECTED_SELECTOR_GRAD_ACCUM_STEPS:
        parser.error("8-GPU global V24 selector task batch is fixed at 32")
    if args.selector_update_every <= 0:
        parser.error("selector update interval must be positive")
    if args.selector_update_every != 1:
        parser.error("selector-only mode performs one selector update per step")
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
    if not args.input_scaler_cache.is_file() or file_sha256(
        args.input_scaler_cache
    ) != args.input_scaler_cache_sha256:
        raise TrainerError("V22 input-scaler cache SHA256 mismatch")
    input_scaler_payload = torch.load(
        args.input_scaler_cache, map_location="cpu", weights_only=False
    )
    if input_scaler_payload.get("version") != V22_SCALER_VERSION:
        raise TrainerError("V22 input-scaler cache version mismatch")
    if not args.v24_scaler_cache.is_file() or file_sha256(
        args.v24_scaler_cache
    ) != args.v24_scaler_cache_sha256:
        raise TrainerError("V24 input-scaler cache SHA256 mismatch")
    v24_scaler_payload = torch.load(
        args.v24_scaler_cache, map_location="cpu", weights_only=False
    )
    if v24_scaler_payload.get("version") != V24_SCALER_VERSION:
        raise TrainerError("V24 input-scaler cache version mismatch")
    if (
        v24_scaler_payload.get("transform_contract")
        != V24_SCALER_TRANSFORM_CONTRACT
    ):
        raise TrainerError("V24 scaler transform contract mismatch")
    if int(v24_scaler_payload.get("cache_size", 0)) < 256:
        raise TrainerError("V24 production scaler cache has fewer than 256 tasks")
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
    data_transition_from_v13 = bool(args.resume_data_transition_from_v13)
    v22_transition_checkpoint = None
    v21_transition_checkpoint = None
    v20_selector_checkpoint = None
    beta_source_checkpoint = None
    joint_source_checkpoint = None
    selector_only_source_checkpoint = None
    if args.resume is not None:
        if file_sha256(args.resume) != args.resume_sha256:
            raise TrainerError("resume checkpoint SHA256 mismatch")
        resume_checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        resume_version = resume_checkpoint.get("metadata", {}).get(
            "checkpoint_version"
        )
        expected_resume_version = (
            V13_CHECKPOINT_VERSION
            if data_transition_from_v13
            else CHECKPOINT_VERSION
        )
        if resume_version != expected_resume_version:
            raise TrainerError("resume checkpoint version mismatch")
        if data_transition_from_v13:
            source_metadata = resume_checkpoint.get("metadata", {})
            source_contract = {
                "trainer_version": V13_TRAINER_VERSION,
                "checkpoint_version": V13_CHECKPOINT_VERSION,
                "model_version": MODEL_VERSION,
                "basis_version": BASIS_VERSION,
                "shared_support_contract": SHARED_SUPPORT_CONTRACT,
                "canonical_penalty_contract": CONTRACT_VERSION,
                "canonical_cache_version": CACHE_VERSION,
                "selector_version": V24_SELECTOR_VERSION,
                "selector_lambda_scaling_contract": (
                    SELECTOR_LAMBDA_SCALING_CONTRACT
                ),
                "training_mode": "selector_only_frozen_beta",
                "world_size": world,
            }
            for key, expected in source_contract.items():
                if source_metadata.get(key) != expected:
                    raise TrainerError(
                        f"V13 data-transition source mismatch for {key}"
                    )
            source_arguments = source_metadata.get(
                "exact_training_arguments", {}
            )
            if (
                source_arguments.get("expected_pipeline_sha256")
                != V13_PIPELINE_SHA256
            ):
                raise TrainerError(
                    "data-transition source is not the certified V13 pipeline"
                )
            if Path(source_arguments.get("rolling_data_root", "")).resolve() != (
                V13_DATA_ROOT.resolve()
            ):
                raise TrainerError(
                    "data-transition source is not the certified V13 root"
                )
    elif args.initialize_selector_only_from is not None:
        if (
            file_sha256(args.initialize_selector_only_from)
            != args.initialize_selector_only_sha256
        ):
            raise TrainerError("selector-only source checkpoint SHA256 mismatch")
        selector_only_source_checkpoint = torch.load(
            args.initialize_selector_only_from,
            map_location="cpu",
            weights_only=False,
        )
        source_metadata = selector_only_source_checkpoint.get("metadata", {})
        source_contract = {
            "trainer_version": JOINT_V23_TRAINER_VERSION,
            "checkpoint_version": JOINT_V23_CHECKPOINT_VERSION,
            "model_version": MODEL_VERSION,
            "basis_version": BASIS_VERSION,
            "shared_support_contract": SHARED_SUPPORT_CONTRACT,
            "canonical_penalty_contract": CONTRACT_VERSION,
            "canonical_cache_version": CACHE_VERSION,
            "selector_version": V23_SELECTOR_VERSION,
            "selector_lambda_scaling_contract": (
                V23_LAMBDA_OUTPUT_CONTRACT
            ),
            "world_size": world,
        }
        for key, expected in source_contract.items():
            if source_metadata.get(key) != expected:
                raise TrainerError(
                    f"selector-only source contract mismatch for {key}: "
                    f"expected={expected!r}, "
                    f"observed={source_metadata.get(key)!r}"
                )
        for required_key in (
            "model_state_dict",
            "selector_state_dict",
            "selector_optimizer_state_dict",
            "rank_states",
            "catalog_pointer",
            "catalog_payload",
        ):
            if required_key not in selector_only_source_checkpoint:
                raise TrainerError(
                    f"selector-only source missing {required_key}"
                )
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
        resume_checkpoint
        or selector_only_source_checkpoint
        or v22_transition_checkpoint
        or v21_transition_checkpoint
        or beta_source_checkpoint
        or joint_source_checkpoint
    )
    transitioning_v22 = v22_transition_checkpoint is not None
    transitioning_v21 = v21_transition_checkpoint is not None
    importing_beta_checkpoint = beta_source_checkpoint is not None
    importing_joint_checkpoint = joint_source_checkpoint is not None
    importing_selector_only_checkpoint = (
        selector_only_source_checkpoint is not None
    )

    if resume_checkpoint is not None:
        initialization_provenance = dict(
            resume_checkpoint["metadata"]["initialization_provenance"]
        )
        if data_transition_from_v13:
            transitions = list(
                initialization_provenance.get("data_transitions", [])
            )
            transitions.append(
                {
                    "mode": "v24_selector_v13_to_v14_snr0p2to50",
                    "source_checkpoint": str(args.resume.resolve()),
                    "source_checkpoint_sha256": args.resume_sha256,
                    "source_completed_selector_steps": int(
                        resume_checkpoint["completed_steps"]
                    ),
                    "source_pipeline_sha256": V13_PIPELINE_SHA256,
                    "target_pipeline_sha256": V14_PIPELINE_SHA256,
                    "model_state_imported_strict": True,
                    "selector_state_imported_strict": True,
                    "selector_optimizer_state_imported": True,
                    "rank_rng_state_imported": True,
                    "penalty_cache_state_imported": True,
                    "penalty_sampler_state_imported": True,
                    "split_generator_state_imported": True,
                    "rank_stream_state_imported": False,
                    "catalog_state_imported": False,
                }
            )
            initialization_provenance["data_transitions"] = transitions
            initialization_provenance[
                "active_data_contract"
            ] = "tabiclv2_v14_supportmix_snr0p2to50"
    elif selector_only_source_checkpoint is not None:
        initialization_provenance = {
            "mode": "selector_only_v24_from_joint_v23_frozen_kalbi_v6",
            "source_checkpoint": str(
                args.initialize_selector_only_from.resolve()
            ),
            "source_checkpoint_sha256": (
                args.initialize_selector_only_sha256
            ),
            "source_completed_joint_steps": int(
                selector_only_source_checkpoint["completed_steps"]
            ),
            "source_checkpoint_id": selector_only_source_checkpoint[
                "metadata"
            ].get("checkpoint_id"),
            "beta_model_state_imported_strict": True,
            "beta_model_frozen_for_entire_phase": True,
            "beta_optimizer_state_imported": False,
            "beta_optimizer_constructed": False,
            "beta_scheduler_state_imported": False,
            "selector_state_imported": (
                "all_v23_tensors_strict_plus_explicit_v24_new_keys"
            ),
            "selector_optimizer_state_imported": (
                "v23_moments_by_fully_qualified_parameter_name"
            ),
            "v24_scaler_cache": str(args.v24_scaler_cache.resolve()),
            "v24_scaler_cache_sha256": args.v24_scaler_cache_sha256,
            "selector_update_counter_reset": True,
            "rank_stream_state_imported": True,
            "catalog_state_imported": True,
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
        "selector_only_v24_from_joint_v23_frozen_kalbi_v6",
        "beta_checkpoint_import_selector_scratch",
        "v3_or_v5_beta_import_v23_selector_scratch",
        "v21_beta_optimizer_stream_continuation_v20_selector_to_v22_geometry",
        (
            "v22_beta_and_shared_selector_optimizer_continuation_"
            "v23_independent_raw_lambda_head"
        ),
    }
    run_selector_imported = (
        resume_checkpoint is not None
        or transitioning_v22
        or transitioning_v21
        or importing_selector_only_checkpoint
    )
    run_selector_started_from_scratch = run_initialization_mode in {
        "full_scratch",
        "beta_checkpoint_import_selector_scratch",
        "v3_or_v5_beta_import_v23_selector_scratch",
    }

    if (
        initial_checkpoint is None
        or data_transition_from_v13
        or importing_beta_checkpoint
        or importing_joint_checkpoint
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
        resume_checkpoint is not None
        or selector_only_source_checkpoint is not None
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
            if rank_state is None or data_transition_from_v13
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
        raise TrainerError("Kalbi V6 K/S/ridge architecture contract drift")
    model_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    if model_parameter_count != EXPECTED_MODEL_PARAMETER_COUNT:
        raise TrainerError(
            "Kalbi V6 model parameter-count drift: "
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
    if is_main:
        init_trace("selector_cpu_initialization_started")
        torch.manual_seed(args.seed + 17)
        selector_cpu = DirectPenaltyMLPQuerySelectorV24Additive512(
            hidden=args.selector_hidden,
            derivative_floor=1e-6,
            squareplus_scale=0.02,
            direct_head_init_std=1e-3,
        )
        if resume_checkpoint is not None:
            selector_cpu.load_state_dict(
                resume_checkpoint["selector_state_dict"], strict=True
            )
            import_report = {
                "mode": "strict_v24_resume",
                "imported_count": len(resume_checkpoint["selector_state_dict"]),
                "new_count": 0,
            }
        elif selector_only_source_checkpoint is not None:
            import_report = load_v23_state_into_v24(
                selector_cpu,
                selector_only_source_checkpoint["selector_state_dict"],
            )
        else:
            raise TrainerError("V24 selector initialization mode is inconsistent")
        selector_cpu.load_input_scaler_cache(input_scaler_payload)
        selector_cpu.load_v24_scaler_cache(v24_scaler_payload)
        selector_state_holder[0] = selector_cpu.state_dict()
        selector_initialization_holder[0] = {
            "architecture": selector_cpu.architecture_report(),
            "initialization": (
                "strict_v24_resume"
                if resume_checkpoint is not None
                else "all_v23_state_plus_fresh_v24_correction"
            ),
            "v23_state_import": import_report,
            "reason": (
                "Reuse the full V23 initial proposal and add fixed-scaled "
                "global, T=3 trajectory, and proposal residual corrections"
            ),
            "v24_scaler_cache": str(args.v24_scaler_cache.resolve()),
            "v24_scaler_cache_sha256": args.v24_scaler_cache_sha256,
        }
        init_trace("selector_cpu_initialization_finished")
    dist.broadcast_object_list(selector_state_holder, src=0)
    dist.broadcast_object_list(selector_initialization_holder, src=0)
    torch.manual_seed(args.seed + 17)
    selector = DirectPenaltyMLPQuerySelectorV24Additive512(
        hidden=args.selector_hidden,
        derivative_floor=1e-6,
        squareplus_scale=0.02,
        direct_head_init_std=1e-3,
    )
    selector.load_state_dict(selector_state_holder[0], strict=True)
    selector.to(device)
    selector_parameter_count = sum(
        parameter.numel() for parameter in selector.parameters()
    )
    if selector_parameter_count != EXPECTED_SELECTOR_PARAMETER_COUNT:
        raise TrainerError(
            "V24-K32S32 parameter-count drift: "
            f"expected {EXPECTED_SELECTOR_PARAMETER_COUNT}, "
            f"got {selector_parameter_count}"
        )
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
    selector_train = DDP(
        selector, device_ids=[local_rank], output_device=local_rank,
        broadcast_buffers=True, find_unused_parameters=False, init_sync=True,
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
    core.eval()
    for parameter in core.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    beta_optimizer = None
    penalty_branch_parameters = [
        *selector_core.penalty_mlp.parameters(),
        *selector_core.penalty_parameter_head.parameters(),
        *selector_core.delta_penalty_latent_head.parameters(),
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

    phase_target_steps = args.steps
    if phase_target_steps <= 0:
        raise TrainerError("V24 continuation phase has no remaining updates")

    def multiplier(step: int) -> float:
        if step < args.warmup_steps:
            return float(step + 1) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(
            max(1, phase_target_steps - args.warmup_steps)
        )
        return 0.5 * (
            1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0))
        )

    beta_scheduler = None
    completed_steps = 0
    previous_checkpoint_id: str | None = None
    if resume_checkpoint is not None:
        completed_steps = int(resume_checkpoint["completed_steps"])
        if not data_transition_from_v13:
            previous_checkpoint_id = resume_checkpoint["metadata"].get(
                "checkpoint_id"
            )
    selector_optimizer_transition_report: dict[str, Any] | None = None
    if resume_checkpoint is not None:
        selector_optimizer.load_state_dict(
            resume_checkpoint["selector_optimizer_state_dict"]
        )
        optimizer_to_device(selector_optimizer, device)
    elif selector_only_source_checkpoint is not None:
        selector_optimizer_transition_report = (
            migrate_v23_optimizer_state_by_name(
                selector_optimizer,
                selector_core,
                selector_only_source_checkpoint,
                selector_lr=args.selector_lr,
                direct_head_lr=args.direct_head_lr,
                device=device,
            )
        )
    elif v22_transition_checkpoint is not None:
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
                    or importing_selector_only_checkpoint
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
        elif data_transition_from_v13:
            if args.out_dir.exists() and any(
                args.out_dir.glob("checkpoint_selector_step*.pt")
            ):
                raise TrainerError(
                    "V13-to-V14 transition output already contains checkpoints"
                )
            args.out_dir.mkdir(parents=True, exist_ok=True)
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
        "v24_scaler_cache": args.v24_scaler_cache_sha256,
        "selector_v24": file_sha256(
            Path(__file__).with_name(
                "selector_v24_additive512_trajectory.py"
            )
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
        "training_mode": "selector_only_frozen_beta",
        "beta_optimizer_constructed": False,
        "beta_scheduler_constructed": False,
        "beta_parameters_require_grad": False,
        "source_joint_step": int(
            initialization_provenance.get(
                "source_completed_joint_steps", 0
            )
        ),
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
            "v23_task_plus_global512_plus_trajectory512_plus_proposal512_additive"
        ),
        "selector_probe_steps": 3,
        "selector_probe_trajectory_shape": [3, 14],
        "selector_probe_initial_beta": "zero",
        "selector_probe_gamma": "eta_times_initial_raw_lambda",
        "selector_residual_linear_layers": 4,
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
        "v24_scaler_cache": str(args.v24_scaler_cache.resolve()),
        "v24_scaler_cache_sha256": args.v24_scaler_cache_sha256,
        "v24_scaler_version": V24_SCALER_VERSION,
        "v24_scaler_transform_contract": V24_SCALER_TRANSFORM_CONTRACT,
        "v24_scaler_semantic_sha256": v24_scaler_payload["cache_sha256"],
        "v24_scaler_task_manifest_sha256": v24_scaler_payload[
            "task_manifest_sha256"
        ],
        "lambda_teacher_used": False,
        "beta_only": False,
        "selector_only": True,
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
        "kalbi_v6_exact_proximal_gradient": {
            "steps": KALBI_EXACT_PG_STEPS,
            "data_gradient": "exact_Xt_Xbeta_minus_y_over_N_without_Gram",
            "penalty": "Direct96_piecewise_quadratic_integral_of_linear_drho",
            "prox": "exact_scalar_interval_search_plus_closed_form",
            "curvature_margin": KALBI_EXACT_PG_CURVATURE_MARGIN,
            "data_majorization_margin": KALBI_EXACT_PG_DATA_MARGIN,
            "ritz_safety": KALBI_EXACT_PG_RITZ_SAFETY,
            "backtrack_factor": KALBI_EXACT_PG_BACKTRACK_FACTOR,
            "max_backtracks": KALBI_EXACT_PG_MAX_BACKTRACKS,
            "feature_chunk": KALBI_EXACT_PG_FEATURE_CHUNK,
            "trace_fallback_is_global_lipschitz_upper_bound": True,
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
    if resume_checkpoint is not None and not data_transition_from_v13:
        metadata = resume_checkpoint["metadata"]
        required_resume = {
            "trainer_version": TRAINER_VERSION,
            "checkpoint_version": CHECKPOINT_VERSION,
            "canonical_penalty_contract": CONTRACT_VERSION,
            "canonical_cache_version": CACHE_VERSION,
            "selector_version": V24_SELECTOR_VERSION,
            "selector_lambda_scaling_contract": (
                SELECTOR_LAMBDA_SCALING_CONTRACT
            ),
            "source_hashes": source_hashes,
            "exact_training_arguments": exact_arguments,
            "world_size": world,
        }
        for key, expected in required_resume.items():
            if metadata.get(key) != expected:
                raise TrainerError(f"resume contract mismatch for {key}")

    event_kind = (
        "v13_to_v14_data_transition"
        if data_transition_from_v13
        else "resume"
        if resume_checkpoint is not None
        else "selector_only_from_joint_v23"
        if importing_selector_only_checkpoint
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
        "selector_version": V24_SELECTOR_VERSION,
        "selector_lambda_scaling_contract": SELECTOR_LAMBDA_SCALING_CONTRACT,
        "canonical_cache": cache_report,
        "scm_prior_contracts": {
            prior: {"dataset_contract": values[0], "response_contract": values[1]}
            for prior, values in SCM_CONTRACTS.items()
        },
        "mixture_contract": (
            "TabICLv2 V11 SCM2048-qfilter plus BNN128 with independent "
            "featurewise Exp(1) noise, variable-shape P100K rolling catalog"
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
            "V23_task_plus_global512_plus_trajectory512_plus_proposal512"
        ),
        "selector_lambda_output": (
            "independent_raw_lambda_directly_used_by_Kalbi_V6"
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
        "selector_v24_scaler_cache_sha256": args.v24_scaler_cache_sha256,
        "selector_v24_scaler_semantic_sha256": v24_scaler_payload[
            "cache_sha256"
        ],
        "selector_v24_scaler_task_count": int(
            v24_scaler_payload["cache_size"]
        ),
        "selector_v24_scaler_transform_contract": (
            V24_SCALER_TRANSFORM_CONTRACT
        ),
        "selector_heldout_y_used_as_input": False,
        "selector_uses_beta_solver_input": (
            "encoded_context_only_for_T3_matrix_free_probe"
        ),
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
        "training_mode": "selector_only_frozen_beta",
        "selector_update_enabled": True,
        "beta_update_enabled": False,
        "beta_optimizer_constructed": False,
        "beta_scheduler_constructed": False,
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
        "selector_optimizer_transition": selector_optimizer_transition_report,
        "source_joint_step": int(
            initialization_provenance.get(
                "source_completed_joint_steps", 0
            )
        ),
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
    trainer_state_source = resume_checkpoint
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

    def run_selector_only_phase() -> None:
        nonlocal active_pointer
        nonlocal active_catalog
        nonlocal rank_entries
        nonlocal active_catalog_sha
        nonlocal previous_checkpoint_id
        nonlocal completed_steps
        nonlocal selector_updates
        nonlocal selector_zero_gradient_skips
        nonlocal consecutive_global_penalty_zero_updates

        core.eval()
        selector_train.train()
        for parameter in core.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        frozen_beta_hash = assert_ddp_synced(
            core, world, "selector_only_frozen_beta_start"
        )
        source_joint_step = int(
            initialization_provenance.get(
                "source_completed_joint_steps", 0
            )
        )
        target_selector_steps = (
            args.steps if args.stop_after is None else args.stop_after
        )
        phase_compute_seconds = 0.0
        phase_compute_steps = 0

        while completed_steps < target_selector_steps:
            started = time.perf_counter()
            selector_optimizer.zero_grad(set_to_none=True)
            heldout_losses: list[torch.Tensor] = []
            heldout_mses: list[torch.Tensor] = []
            heldout_f_hats: list[torch.Tensor] = []
            heldout_f_zeros: list[torch.Tensor] = []
            heldout_fhat_fzero_ratios: list[torch.Tensor] = []
            raw_lambdas: list[torch.Tensor] = []
            lambda_maxes: list[torch.Tensor] = []
            d0_values: list[torch.Tensor] = []
            max_drho_values: list[torch.Tensor] = []
            rho_values: list[torch.Tensor] = []
            drho_values: list[torch.Tensor] = []
            beta_active_fractions: list[torch.Tensor] = []
            v24_initial_lambdas: list[torch.Tensor] = []
            v24_delta_loglambdas: list[torch.Tensor] = []
            v24_code_norms: dict[str, list[torch.Tensor]] = {
                "global": [],
                "trajectory": [],
                "proposal": [],
            }
            v24_residual_update_rms: list[torch.Tensor] = []
            v24_probe_trajectories: list[torch.Tensor] = []
            v24_probe_gamma_errors: list[torch.Tensor] = []
            v24_scaler_clip_fractions: dict[str, list[torch.Tensor]] = {
                "global": [],
                "trajectory": [],
                "proposal": [],
            }
            output_gradient_norms: dict[str, list[float]] = {
                "penalty_shape_features": [],
                "penalty_d0": [],
                "lambdas": [],
            }
            selector_shapes: list[tuple[int, int]] = []

            for accumulation in range(args.selector_grad_accum_steps):
                xs, ys, info = stream.next_batch()
                selector_shapes.append(
                    (int(info[0]["n"]), int(info[0]["p"]))
                )
                x_train, y_train, x_test, y_test = (
                    split_and_standardize_variable(
                        xs,
                        ys,
                        generator=split_generator,
                        device=device,
                    )
                )
                with torch.no_grad():
                    encoded = core.encode_dataset(
                        core.precompute(x_train, y_train)
                    )
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
                        solver_context=encoded,
                        return_representation=True,
                    )
                    v24_initial_lambdas.append(
                        query["v24_initial_raw_lambda"].detach().reshape(-1)
                    )
                    v24_delta_loglambdas.append(
                        query["v24_delta_loglambda"].detach().reshape(-1)
                    )
                    v24_code_norms["global"].append(
                        query["v24_global_code_norm"].detach().reshape(-1)
                    )
                    v24_code_norms["trajectory"].append(
                        query["v24_trajectory_code_norm"].detach().reshape(-1)
                    )
                    v24_code_norms["proposal"].append(
                        query["v24_proposal_code_norm"].detach().reshape(-1)
                    )
                    v24_residual_update_rms.append(
                        query["v24_residual_update_rms"].detach()
                    )
                    v24_probe_trajectories.append(
                        query["v24_solver_trajectory"].detach()
                    )
                    v24_probe_gamma_errors.append(
                        query["v24_probe_gamma_contract_error"].detach()
                    )
                    for source_name in v24_scaler_clip_fractions:
                        scaled_value = query[
                            f"v24_{source_name}_scaled"
                        ].detach()
                        v24_scaler_clip_fractions[source_name].append(
                            (scaled_value.abs() >= 8.0).float().mean().reshape(1)
                        )
                    for output_name in (
                        "penalty_shape_features",
                        "penalty_d0",
                        "lambdas",
                    ):
                        query[output_name].retain_grad()
                    aux = core.decode_queries(
                        encoded,
                        lambdas=query["lambdas"],
                        lambda_max=encoded["lambda_max"].float()[:, None],
                        penalty_shape_features=query[
                            "penalty_shape_features"
                        ],
                        penalty_d0=query["penalty_d0"],
                        return_aux=True,
                    )
                    beta = aux["beta_q"][:, 0].float()
                    prediction = torch.einsum(
                        "bnp,bp->bn", x_test.float(), beta
                    )
                    task_mse = (
                        prediction - y_test.float()
                    ).square().mean(-1)
                    (
                        selector_loss,
                        task_f_hat,
                        task_f_zero,
                        task_fhat_fzero_ratio,
                    ) = selector_heldout_fhat_fzero_loss(
                        prediction,
                        y_test.float(),
                    )
                    (
                        selector_loss
                        / float(args.selector_grad_accum_steps)
                    ).backward()

                for output_name in output_gradient_norms:
                    gradient = query[output_name].grad
                    if gradient is None:
                        raise TrainerError(
                            f"selector output gradient missing for "
                            f"{output_name}"
                        )
                    if not bool(torch.isfinite(gradient).all()):
                        raise TrainerError(
                            f"selector output gradient nonfinite for "
                            f"{output_name}"
                        )
                    output_gradient_norms[output_name].append(
                        float(gradient.float().norm())
                    )

                heldout_losses.append(selector_loss.detach())
                heldout_mses.append(task_mse.detach())
                heldout_f_hats.append(task_f_hat.detach())
                heldout_f_zeros.append(task_f_zero.detach())
                heldout_fhat_fzero_ratios.append(
                    task_fhat_fzero_ratio.detach()
                )
                raw_lambdas.append(query["raw_lambda"].detach())
                lambda_maxes.append(
                    encoded["lambda_max"].float().detach().reshape(-1)
                )
                d0_values.append(
                    query["penalty_d0"].detach().float().reshape(-1)
                )
                rho_values.append(query["penalty_rho"].detach().float())
                drho = query["penalty_drho"].detach().float()
                drho_values.append(drho)
                max_drho_values.append(drho.max(-1).values.reshape(-1))
                beta_active_fractions.append(
                    (beta.detach() != 0.0).float().mean().reshape(1)
                )

            if any(parameter.grad is not None for parameter in core.parameters()):
                raise TrainerError(
                    "frozen beta solver received parameter gradients"
                )
            if any(
                parameter.requires_grad for parameter in core.parameters()
            ):
                raise TrainerError(
                    "frozen beta solver parameter was re-enabled"
                )

            group_stats = {
                "lambda_head": grad_stats(
                    selector_core.lambda_head.parameters()
                ),
                "penalty_mlp": grad_stats(
                    selector_core.penalty_mlp.parameters()
                ),
                "penalty_head": grad_stats(
                    selector_core.penalty_parameter_head.parameters()
                ),
                "fusion": grad_stats(
                    selector_core.fusion_mlp.parameters()
                ),
                "v24_global_projection": grad_stats(
                    selector_core.global_projection.parameters()
                ),
                "v24_trajectory_projection": grad_stats(
                    selector_core.trajectory_projection.parameters()
                ),
                "v24_proposal_projection": grad_stats(
                    selector_core.proposal_projection.parameters()
                ),
                "v24_residual_layers": grad_stats(
                    selector_core.residual_layers.parameters()
                ),
                "v24_delta_loglambda": grad_stats(
                    selector_core.delta_loglambda_head.parameters()
                ),
                "v24_delta_penalty": grad_stats(
                    selector_core.delta_penalty_latent_head.parameters()
                ),
            }
            selector_total_gradient = float(
                torch.nn.utils.clip_grad_norm_(
                    selector_train.parameters(), 5.0
                )
            )
            if not math.isfinite(selector_total_gradient):
                raise TrainerError(
                    "selector total gradient is nonfinite"
                )
            if completed_steps == 0:
                for name in (
                    "v24_delta_loglambda",
                    "v24_delta_penalty",
                ):
                    if float(group_stats[name]["norm"]) <= 0.0:
                        raise TrainerError(
                            f"V24 first-backward correction gradient is zero: {name}"
                        )
            if completed_steps >= 1:
                for name in (
                    "v24_global_projection",
                    "v24_trajectory_projection",
                    "v24_proposal_projection",
                    "v24_residual_layers",
                    "v24_delta_loglambda",
                    "v24_delta_penalty",
                ):
                    if float(group_stats[name]["norm"]) <= 0.0:
                        raise TrainerError(
                            f"V24 post-bootstrap branch gradient is zero: {name}"
                        )
            penalty_gradient_zero = any(
                float(group_stats[name]["norm"]) <= 0.0
                for name in ("penalty_mlp", "penalty_head")
            )
            if penalty_gradient_zero:
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

            selector_update_applied = selector_total_gradient > 0.0
            if selector_update_applied:
                selector_optimizer.step()
                selector_updates += 1
            else:
                selector_zero_gradient_skips += 1
                emit({
                    "kind": "selector_update_skipped",
                    "selector_step": completed_steps + 1,
                    "reason": "global_selector_gradient_is_zero",
                    "selector_zero_gradient_skips": (
                        selector_zero_gradient_skips
                    ),
                })

            completed_steps += 1
            phase_compute_seconds += time.perf_counter() - started
            phase_compute_steps += 1
            all_rho = torch.cat(rho_values)
            all_drho = torch.cat(drho_values)
            local_metrics = {
                "rank": rank,
                "selector_step": completed_steps,
                "source_joint_step": source_joint_step,
                "catalog_epoch": int(active_pointer["epoch"]),
                "catalog_sha256": active_catalog_sha,
                "selector_heldout_loss": float(
                    torch.stack(heldout_losses).mean()
                ),
                "selector_heldout_mse": float(
                    torch.cat(heldout_mses).mean()
                ),
                "selector_heldout_f_hat": float(
                    torch.cat(heldout_f_hats).mean()
                ),
                "selector_heldout_f_zero": float(
                    torch.cat(heldout_f_zeros).mean()
                ),
                "selector_heldout_fhat_over_fzero": float(
                    torch.cat(heldout_fhat_fzero_ratios).mean()
                ),
                "selector_raw_lambda_min": float(
                    torch.cat(raw_lambdas).min()
                ),
                "selector_raw_lambda_max": float(
                    torch.cat(raw_lambdas).max()
                ),
                "context_lambda_max_min": float(
                    torch.cat(lambda_maxes).min()
                ),
                "context_lambda_max_max": float(
                    torch.cat(lambda_maxes).max()
                ),
                "penalty_d0_min": float(torch.cat(d0_values).min()),
                "penalty_d0_max": float(torch.cat(d0_values).max()),
                "penalty_max_drho_min": float(
                    torch.cat(max_drho_values).min()
                ),
                "penalty_max_drho_max": float(
                    torch.cat(max_drho_values).max()
                ),
                "penalty_rho0_error_max": float(
                    all_rho[:, 0].abs().max()
                ),
                "penalty_rho6_error_max": float(
                    (all_rho[:, -1] - 1.0).abs().max()
                ),
                "penalty_min_drho": float(all_drho.min()),
                "penalty_rho_task_std_mean": float(
                    all_rho.std(0, unbiased=False).mean()
                ),
                "penalty_drho_task_std_mean": float(
                    all_drho.std(0, unbiased=False).mean()
                ),
                "selected_beta_active_fraction": float(
                    torch.cat(beta_active_fractions).mean()
                ),
                "v24_initial_raw_lambda_mean": float(
                    torch.cat(v24_initial_lambdas).mean()
                ),
                "v24_corrected_raw_lambda_mean": float(
                    torch.cat(raw_lambdas).mean()
                ),
                "v24_delta_loglambda_mean": float(
                    torch.cat(v24_delta_loglambdas).mean()
                ),
                "v24_code_norms": {
                    name: float(torch.cat(values).mean())
                    for name, values in v24_code_norms.items()
                },
                "v24_post_gate_code_norms": {
                    "global": float(
                        torch.cat(v24_code_norms["global"]).mean()
                        * selector_core.global_gate.detach().abs()
                    ),
                    "trajectory": float(
                        torch.cat(v24_code_norms["trajectory"]).mean()
                        * selector_core.trajectory_gate.detach().abs()
                    ),
                    "proposal": float(
                        torch.cat(v24_code_norms["proposal"]).mean()
                        * selector_core.proposal_gate.detach().abs()
                    ),
                },
                "v24_fusion_gates": {
                    "global": float(selector_core.global_gate.detach()),
                    "initial_norm": float(
                        selector_core.initial_norm_gate.detach()
                    ),
                    "trajectory": float(
                        selector_core.trajectory_gate.detach()
                    ),
                    "proposal": float(selector_core.proposal_gate.detach()),
                    "residual": [
                        float(layer.gate.detach())
                        for layer in selector_core.residual_layers
                    ],
                },
                "v24_residual_update_rms": (
                    torch.cat(v24_residual_update_rms).mean(0).tolist()
                ),
                "v24_probe_stat_mean": (
                    torch.cat(v24_probe_trajectories).mean(0).tolist()
                ),
                "v24_probe_stat_std": (
                    torch.cat(v24_probe_trajectories)
                    .std(0, unbiased=False)
                    .tolist()
                ),
                "v24_probe_gamma_contract_error_max": float(
                    torch.cat(v24_probe_gamma_errors).max()
                ),
                "v24_scaler_clipping_fraction": {
                    name: float(torch.cat(values).mean())
                    for name, values in v24_scaler_clip_fractions.items()
                },
                "selector_total_gradient_preclip": (
                    selector_total_gradient
                ),
                "selector_group_gradients": group_stats,
                "selector_output_gradient_norms": {
                    name: {
                        "min": min(values),
                        "max": max(values),
                        "mean": sum(values) / len(values),
                    }
                    for name, values in output_gradient_norms.items()
                },
                "selector_update_performed": selector_update_applied,
                "selector_updates": selector_updates,
                "selector_zero_gradient_skips": (
                    selector_zero_gradient_skips
                ),
                "consecutive_global_penalty_zero_updates": (
                    consecutive_global_penalty_zero_updates
                ),
                "selector_shapes": selector_shapes,
                "beta_parameters_trainable": 0,
                "beta_parameters_with_grad": 0,
            }

            if completed_steps == 1:
                if selector_update_applied and assert_ddp_synced(
                    selector_core,
                    world,
                    "selector_only_post_first_update",
                ) == selector_hash_before:
                    raise TrainerError(
                        "selector state did not change on first update"
                    )

            if completed_steps % args.log_every == 0:
                if module_sha256(core) != frozen_beta_hash:
                    raise TrainerError(
                        "frozen beta solver hash changed"
                    )
                monitors = gather_rank_objects(local_metrics, world)
                emit({
                    "kind": "selector_train_update",
                    "selector_step": completed_steps,
                    "source_joint_step": source_joint_step,
                    "average_compute_seconds_per_selector_step": (
                        phase_compute_seconds / phase_compute_steps
                    ),
                    "frozen_beta_state_sha256": frozen_beta_hash,
                    "rank_monitors": monitors,
                })
                phase_compute_seconds = 0.0
                phase_compute_steps = 0

            if completed_steps % args.reload_every == 0:
                old_catalog_sha = active_catalog_sha
                old_checkpoint_id = previous_checkpoint_id
                (
                    active_pointer,
                    active_catalog,
                    rank_entries,
                    catalog_changed,
                ) = load_current_catalog(
                    store,
                    rank=rank,
                    world=world,
                    expected_pipeline_sha256=(
                        args.expected_pipeline_sha256
                    ),
                    active_sha256=active_catalog_sha,
                )
                store.remember_adopted_pointer(active_pointer)
                active_catalog_sha = str(
                    active_pointer["catalog_sha256"]
                )
                if is_main and active_catalog_sha != old_catalog_sha:
                    store.acquire_lease(
                        args.trainer_id, active_catalog_sha
                    )
                dist.barrier()
                stream.refresh(
                    rank_entries,
                    catalog_sha256=active_catalog_sha,
                )
                store.write_ack(
                    args.trainer_id,
                    rank,
                    world,
                    active_catalog_sha,
                )
                dist.barrier()

                local_state = {
                    "rank": rank,
                    "rng": capture_rng(),
                    "stream": stream.state_dict(),
                    "penalty_cache": cache.state_dict(),
                    "sampler": sampler.state_dict(),
                    "split_generator_state": (
                        split_generator.get_state()
                    ),
                }
                rank_states = gather_rank_objects(local_state, world)
                checkpoint_id = (
                    f"{args.trainer_id}_selector"
                    f"{completed_steps:06d}"
                )
                checkpoint_path = (
                    args.out_dir
                    / f"checkpoint_selector_step"
                    f"{completed_steps:06d}.pt"
                )
                if is_main:
                    if module_sha256(core) != frozen_beta_hash:
                        raise TrainerError(
                            "frozen beta hash changed before checkpoint"
                        )
                    payload = {
                        "completed_steps": completed_steps,
                        "model_state_dict": core.state_dict(),
                        "selector_state_dict": (
                            selector_core.state_dict()
                        ),
                        "selector_optimizer_state_dict": (
                            selector_optimizer.state_dict()
                        ),
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
                            "shared_support_contract": (
                                SHARED_SUPPORT_CONTRACT
                            ),
                            "penalty_contract_version": (
                                PENALTY_CONTRACT_VERSION
                            ),
                            "canonical_penalty_contract": (
                                CONTRACT_VERSION
                            ),
                            "canonical_cache_version": CACHE_VERSION,
                            "selector_version": V24_SELECTOR_VERSION,
                            "selector_lambda_scaling_contract": (
                                SELECTOR_LAMBDA_SCALING_CONTRACT
                            ),
                            "selector_objective": (
                                SELECTOR_OBJECTIVE
                            ),
                            "training_mode": (
                                "selector_only_frozen_beta"
                            ),
                            "source_joint_step": source_joint_step,
                            "frozen_beta_state_sha256": (
                                frozen_beta_hash
                            ),
                            "beta_optimizer_present": False,
                            "beta_scheduler_present": False,
                            "selector_update_every": 1,
                            "selector_prediction_loss_only": True,
                            "heldout_y_used_as_selector_input": False,
                            "catalog_epoch": int(
                                active_pointer["epoch"]
                            ),
                            "catalog_sha256": active_catalog_sha,
                            "catalog_reloaded_before_checkpoint": True,
                            "reload_interval_updates": (
                                RELOAD_INTERVAL_UPDATES
                            ),
                            "world_size": world,
                            "source_hashes": source_hashes,
                            "exact_training_arguments": (
                                exact_arguments
                            ),
                            "penalty_cache_sha256": (
                                args.penalty_cache_sha256
                            ),
                            "initialization_provenance": (
                                initialization_provenance
                            ),
                        },
                    }
                    atomic_torch_save(payload, checkpoint_path)
                    observed_checkpoint_sha = file_sha256(
                        checkpoint_path
                    )
                    reloaded = torch.load(
                        checkpoint_path,
                        map_location="cpu",
                        weights_only=False,
                    )
                    if (
                        int(reloaded["completed_steps"])
                        != completed_steps
                        or reloaded["metadata"]["catalog_sha256"]
                        != active_catalog_sha
                        or state_dict_sha256(
                            reloaded["model_state_dict"]
                        )
                        != frozen_beta_hash
                        or state_dict_sha256(
                            reloaded["selector_state_dict"]
                        )
                        != module_sha256(selector_core)
                        or "beta_optimizer_state_dict" in reloaded
                        or "beta_scheduler_state_dict" in reloaded
                    ):
                        raise TrainerError(
                            "selector-only checkpoint failed strict "
                            "reload verification"
                        )
                    store.register_checkpoint_pin(
                        checkpoint_id,
                        active_catalog_sha,
                        checkpoint_sha256=observed_checkpoint_sha,
                        resumable=True,
                        reload_verified=True,
                    )
                    if (
                        old_checkpoint_id
                        and old_checkpoint_id != checkpoint_id
                    ):
                        store.mark_checkpoint_nonresumable(
                            old_checkpoint_id
                        )
                    if active_catalog_sha != old_catalog_sha:
                        store.release_lease(
                            args.trainer_id,
                            expected_catalog_sha256=old_catalog_sha,
                        )
                    emit({
                        "kind": "selector_catalog_reload_checkpoint",
                        "selector_step": completed_steps,
                        "source_joint_step": source_joint_step,
                        "catalog_changed": catalog_changed,
                        "old_catalog_sha256": old_catalog_sha,
                        "catalog_epoch": int(
                            active_pointer["epoch"]
                        ),
                        "catalog_sha256": active_catalog_sha,
                        "checkpoint": str(checkpoint_path),
                        "checkpoint_sha256": (
                            observed_checkpoint_sha
                        ),
                        "checkpoint_id": checkpoint_id,
                        "frozen_beta_state_sha256": (
                            frozen_beta_hash
                        ),
                        "reload_verified": True,
                    })
                previous_checkpoint_id = checkpoint_id
                dist.barrier()

        emit({
            "kind": "selector_only_complete",
            "selector_step": completed_steps,
            "source_joint_step": source_joint_step,
            "catalog_epoch": int(active_pointer["epoch"]),
            "catalog_sha256": active_catalog_sha,
            "frozen_beta_state_sha256": frozen_beta_hash,
        })
        dist.barrier()
        dist.destroy_process_group()

    run_selector_only_phase()
    return

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
                result = objective_kalbi_v6_exactprox10(model, payload, knots)
                if not bool(torch.isfinite(result["loss"])):
                    raise TrainerError("nonfinite beta loss")
                (result["loss"] / float(args.grad_accum_steps)).backward()
            for key in beta_metrics:
                beta_metrics[key] += (
                    float(result[key].detach()) / float(args.grad_accum_steps)
                )
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
                            "selector_version": V24_SELECTOR_VERSION,
                            "selector_lambda_scaling_contract": (
                                SELECTOR_LAMBDA_SCALING_CONTRACT
                            ),
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
                        solver_context=encoded,
                        return_representation=True,
                    )
                    trajectory = query["v22_trajectory"]
                    active_count = trajectory["stage_active"].sum(-1).float()
                    if not bool((active_count >= 1.0).all()):
                        raise TrainerError("V24 trajectory produced no active stage")
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
                        raise TrainerError("Kalbi V6 decode aux lacks beta_q")
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
                        raise TrainerError("Kalbi V6 refinement telemetry is missing")
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
                        raise TrainerError("Kalbi V6 beta path is nonfinite")
                    refinement_values = tuple(
                        aux[name].float() for name in required_refinement_aux
                    )
                    if not all(
                        bool(torch.isfinite(value).all())
                        for value in refinement_values
                    ):
                        raise TrainerError("Kalbi V6 refinement path is nonfinite")
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
                    hard_loss, _, _, _ = selector_heldout_fhat_fzero_loss(
                        prediction,
                        y_test.float(),
                    )
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
                        "selector_version": V24_SELECTOR_VERSION,
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
