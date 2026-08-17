#!/usr/bin/env python3
"""KALBI V8 adaptive-block T10 beta-only trainer for V14 rolling data.

This entrypoint has no selector model, selector forward pass, selector DDP
wrapper, or selector optimizer. The certified V23 selector payload is carried
as opaque CPU checkpoint state solely for backward-compatible resumption. The
one-time V6/V13-to-V8/V14 transition imports model, optimizer, scheduler, RNG,
penalty-cache, and sampler state while resetting catalog and stream state.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .adaptive_block_prox_v8 import (
    ADAPTIVE_BLOCK_FORMULA_VERSION,
    SWEEP_ORDER_VERSION,
)
from .canonical_penalty import CONTRACT_VERSION, canonical_grid
from .canonical_penalty_cache import CACHE_VERSION, CanonicalPenaltyCache
from .model import (
    BASIS_VERSION,
    KALBI_BETA_BOUND,
    KALBI_BLOCK_PROX_BACKTRACK_FACTOR,
    KALBI_BLOCK_PROX_CURVATURE_MARGIN,
    KALBI_BLOCK_PROX_DATA_MARGIN,
    KALBI_BLOCK_PROX_FEATURE_CHUNK,
    KALBI_BLOCK_PROX_MAX_BACKTRACKS,
    KALBI_BLOCK_PROX_MAX_BLOCK_SIZE,
    KALBI_BLOCK_PROX_MIN_BLOCK_SIZE,
    KALBI_BLOCK_PROX_POWER_ITERATIONS,
    KALBI_BLOCK_PROX_SPECTRAL_SAFETY,
    KALBI_REFINEMENT_MODE,
    KALBI_REFINEMENT_SWEEPS,
    KALBI_RIDGE,
    KALBI_STAGES,
    KALBI_TOPK,
    MODEL_VERSION,
    PENALTY_CONTRACT_VERSION,
    SHARED_SUPPORT_CONTRACT,
    build_model,
)
from .rolling_catalog import CapacityPolicy
from .train import (
    ROOT_DEPENDENCY_PATHS,
    TrainerError,
    atomic_torch_save,
    capture_rng,
    make_sampler,
    objective_kalbi_v8_adaptiveblockprox10,
    restore_rng,
    state_dict_sha256,
)
from . import (
    train_beta_only_v23_supportmix_v13_transition_8gpu as supportmix_runtime,
)
from .train_beta_only_v23_supportmix_v13_transition_8gpu import (
    COEFFICIENT_FAMILY_NAMES,
    ERROR_FAMILY_NAMES,
    LAMBDA_LOW_MIN_CONTRACT,
    LINEAR_TASK_TARGET_CONTRACT,
    OUTCOME_SOURCE_LINEAR,
    OUTCOME_SOURCE_NAMES,
    RollingSingleTaskStream,
    SCM_CONTRACTS,
    SNR_REGIME_NAMES,
    SUPPORT_REGIME_NAMES,
    TABICL_TASK_TARGET_CONTRACT,
    TrainerCatalogStore,
    adopt_catalog_package,
    gather_rank_objects,
    load_current_catalog,
    make_canonical_query_payload,
    optimizer_to_device,
)
from .train_joint_v11_direct import file_sha256, module_sha256
from .utils import fixed_ridge_numerical_diagnostics
from train_fpk import _load_yaml


TRAINER_VERSION = (
    "kalbi_v8_adaptiveblockprox10_v23_betaonly_supportmix_v14_snr0p2to50_"
    "p100k_world8_effbatch32_rolling100gb_reload100_rawlambda_v1"
)
CHECKPOINT_VERSION = (
    "kalbi_v8_adaptiveblockprox10_v23_betaonly_supportmix_v14_snr0p2to50_"
    "p100k_effbatch32_checkpoint_v1"
)
SOURCE_CHECKPOINT_PATH = Path(
    "/users/minsuk/fm-lm-lambda/runs/"
    "kalbi_v23_betaonly_supportmix_v13_target3080_effbatch32_to60k_20260726/"
    "checkpoint_step042900.pt"
)
SOURCE_CHECKPOINT_SHA256 = (
    "f6c43ffb941c508df040327fdbe47da2a5889e4b3100bdb895e3c2ca7c9d300f"
)
SOURCE_CHECKPOINT_VERSION = (
    "kalbi_v6_v23_v22geometry_independentraw_k32s32_exactDirect96proxPG10_"
    "eta4x_tabiclv2_expvarnoise_v11_p100k_effbatch32_checkpoint_v1"
)
SOURCE_TRAINER_VERSION = (
    "kalbi_v6_v23_v22geometry_independentraw_k32s32_tabiclv2_v11_p100k_"
    "world8_effbatch32_exactDirect96proxPG10_eta4x_rolling100gb_reload100_"
    "rawLambdaDirect_v1"
)
SOURCE_MODEL_VERSION = (
    "kalbi_v6_krylov_abfilm_threshold_exactproxpg10_eta4x_k32_stages32_v1"
)
SOURCE_BASIS_VERSION = (
    "global_krylov_ridge_plus_ab_trajectory_threshold_plus_exactproxpg10_v6"
)
SOURCE_SHARED_SUPPORT_CONTRACT = (
    "centered_corr_A_B_ridge1e-6_supportlocal_correction_global_threshold_"
    "fullP_exactgrad_exactDirect96proxPG10_eta4x_refinement_v6"
)
SOURCE_SELECTOR_VERSION = (
    "v23_v22geometry_k32s32_independent_rawlambda1e-6to10_"
    "d0_zeroKKT_telemetry_only_v1"
)
SOURCE_PIPELINE_SHA256 = (
    "99f81c6a5bf5083a56f3268a56a0700cd64865b1ff0bba9b23386088d11a9559"
)
CERTIFIED_PIPELINE_SHA256 = (
    "4fd706b7b254eddb4a832b8107fa3379d128f152716c7ae9f142539064d7e707"
)
CERTIFIED_ROLLING_CONTROL_ROOT = Path(
    "/users/sync_data/"
    "kalbi_tabiclv2_v14_mixed50_linear_pn50_strictn3_snr0p2to50_"
    "p100k_100GB_target3080_seed160076430/rolling_control"
)
V14_GENERATOR_SHA256 = (
    "d2548b21c63f220ec2d2c5f34bcf1a341e17b4440eb3db80f1ad3ad1029e812f"
)
V14_DATASET_CONTRACT = (
    "tabiclv2_graph_scm2048_qfilter_bnn128_expvarnoise_mixed50_tabicl_"
    "linear_supportmix_pn_strictn3_snr0p2to50_v14"
)
V14_TARGET_CONTRACT = (
    "mixed50_official_tabicl_or_controlled_linear_supportmix_pn50_"
    "strictn3_snr_loguniform_0p2_50_y_qualityfilter_center_scale_"
    "clip10_rescale_v3"
)
V14_LINEAR_TASK_TARGET_CONTRACT = (
    "controlled_linear_y_supportmix_pn50_strictn3_snr_loguniform_0p2_50_"
    "errorfamily_qualityfilter_center_scale_clip10_rescale_v3"
)
V14_LINEAR_SNR_CONTRACT = "linear_target_snr_loguniform_0p2_50_v1"
V14_LINEAR_SNR_DISTRIBUTION = "log_uniform"
V14_LINEAR_SNR_MIN = 0.2
V14_LINEAR_SNR_MAX = 50.0
SNR_REGIME_NAMES = (
    "log_quantile_1",
    "log_quantile_2",
    "log_quantile_3",
    "log_quantile_4",
    "log_quantile_5",
)
EXPECTED_WORLD_SIZE = 8
EXPECTED_GRAD_ACCUM_STEPS = 4
EXPECTED_MODEL_PARAMETER_COUNT = 3_588_547
SOURCE_COMPLETED_STEPS = 42_900
TARGET_STEPS = 60_000
SCHEDULE_PHASE_STEPS = 30_000
RELOAD_INTERVAL_UPDATES = 100
SELECTOR_STATE_SHA256 = (
    "8921997a893f57ef35834925d1e3e58c31cbdb416f35eface7d7717628773de2"
)


def configure_v14_supportmix_runtime() -> None:
    """Retarget the shared immutable-shard reader to the V14 data contract."""

    supportmix_runtime.SCM_CONTRACTS = {
        "tabicl_v2": (V14_DATASET_CONTRACT, V14_TARGET_CONTRACT),
    }
    supportmix_runtime.LINEAR_TASK_TARGET_CONTRACT = (
        V14_LINEAR_TASK_TARGET_CONTRACT
    )
    supportmix_runtime.SNR_REGIME_NAMES = SNR_REGIME_NAMES
    RollingSingleTaskStream.VERSION = (
        "v14_mixed50_tabicl_linear_pn50_strictn3_snr0p2to50_"
        "catalog_single_task_stream_v1"
    )


def validate_v14_catalog_extensions(
    catalog: Mapping[str, Any],
    expected_pipeline_sha256: str,
) -> None:
    """Check the V14 SNR fields not present in the shared V13 validator."""

    if (
        catalog.get("contract", {}).get("pipeline_config_sha256")
        != expected_pipeline_sha256
    ):
        raise TrainerError("V14 catalog pipeline SHA mismatch")
    files = catalog.get("files", ())
    if len(files) < EXPECTED_WORLD_SIZE:
        raise TrainerError("V14 catalog has fewer shards than DDP ranks")
    snr_edges = np.geomspace(
        V14_LINEAR_SNR_MIN,
        V14_LINEAR_SNR_MAX,
        len(SNR_REGIME_NAMES) + 1,
    )
    for entry in files:
        source_hashes = entry.get("source_hashes", {})
        if source_hashes.get("generator_tabicl_v2") != V14_GENERATOR_SHA256:
            raise TrainerError("V14 catalog generator SHA mismatch")
        metadata = entry.get("metadata", {})
        semantic = (
            metadata.get("semantic_contract", {})
            if isinstance(metadata, Mapping)
            else {}
        )
        if int(semantic.get("outcome_source_id", -1)) != OUTCOME_SOURCE_LINEAR:
            continue
        target_snr = float(semantic.get("linear_target_snr", float("nan")))
        realized_snr = float(
            semantic.get("linear_realized_snr", float("nan"))
        )
        if (
            semantic.get("linear_snr_contract") != V14_LINEAR_SNR_CONTRACT
            or semantic.get("linear_snr_distribution")
            != V14_LINEAR_SNR_DISTRIBUTION
            or not math.isfinite(target_snr)
            or not math.isfinite(realized_snr)
            or target_snr < V14_LINEAR_SNR_MIN
            or target_snr > V14_LINEAR_SNR_MAX
            or realized_snr < V14_LINEAR_SNR_MIN
            or realized_snr > V14_LINEAR_SNR_MAX
            or abs(math.log(realized_snr / target_snr)) > 2e-5
        ):
            raise TrainerError("V14 catalog linear SNR contract failed")
        expected_regime = int(
            np.searchsorted(snr_edges[1:-1], target_snr, side="right")
        )
        if int(semantic.get("linear_snr_regime_id", -1)) != expected_regime:
            raise TrainerError("V14 catalog linear SNR regime mismatch")

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
    "refinement_adaptive_block_size",
    "refinement_blocks_per_sweep",
    "refinement_total_block_updates",
    "refinement_monotonicity_violations",
    "refinement_rejected_fraction",
    "refinement_exact_prox_calls",
    "refinement_backtracking_evaluations",
)


def opaque_payload_sha256(value: Any) -> str:
    """Hash nested CPU checkpoint payloads without instantiating their owner."""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(
                tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            )
        elif isinstance(item, Mapping):
            digest.update(b"mapping")
            for key in sorted(item, key=lambda candidate: str(candidate)):
                update(str(key))
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for nested in item:
                update(nested)
        else:
            digest.update(type(item).__name__.encode())
            digest.update(repr(item).encode())

    update(value)
    return digest.hexdigest()


def audit_state_dict_compatibility(
    target: torch.nn.Module,
    source_state_dict: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    target_state = target.state_dict()
    source_keys = set(source_state_dict)
    target_keys = set(target_state)
    if source_keys != target_keys:
        raise TrainerError(
            "V6-to-V8 model-state key drift: "
            f"missing={sorted(target_keys - source_keys)}, "
            f"extra={sorted(source_keys - target_keys)}"
        )
    shape_mismatches = {
        key: {
            "source": list(source_state_dict[key].shape),
            "target": list(target_state[key].shape),
        }
        for key in sorted(target_keys)
        if tuple(source_state_dict[key].shape) != tuple(target_state[key].shape)
    }
    if shape_mismatches:
        raise TrainerError(
            "V6-to-V8 model-state shape drift: "
            + json.dumps(shape_mismatches, sort_keys=True)
        )
    target.load_state_dict(source_state_dict, strict=True)
    return {
        "strict_key_match": True,
        "strict_shape_match": True,
        "state_tensor_count": len(target_state),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in target.parameters()
        ),
    }


def audit_optimizer_state_by_parameter_name(
    *,
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    source_state_dict: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind numeric optimizer slots to the target through stable names."""

    name_by_parameter_id = {
        id(parameter): name for name, parameter in module.named_parameters()
    }
    source_groups = list(source_state_dict.get("param_groups", ()))
    target_groups = list(optimizer.param_groups)
    if len(source_groups) != len(target_groups):
        raise TrainerError("beta optimizer group-count mismatch")
    named_groups: list[dict[str, Any]] = []
    checked_state_tensors = 0
    source_states = source_state_dict.get("state", {})
    for group_index, (source_group, target_group) in enumerate(
        zip(source_groups, target_groups)
    ):
        if source_group.get("name") != target_group.get("name"):
            raise TrainerError(
                f"beta optimizer group-name mismatch at {group_index}"
            )
        source_ids = list(source_group.get("params", ()))
        target_parameters = list(target_group.get("params", ()))
        if len(source_ids) != len(target_parameters):
            raise TrainerError("beta optimizer group parameter-count mismatch")
        parameter_names: list[str] = []
        for source_id, target_parameter in zip(source_ids, target_parameters):
            parameter_name = name_by_parameter_id.get(id(target_parameter))
            if parameter_name is None:
                raise TrainerError("optimizer parameter is outside beta model")
            parameter_names.append(parameter_name)
            for state_name, state_value in source_states.get(
                source_id, {}
            ).items():
                if not torch.is_tensor(state_value) or state_value.ndim == 0:
                    continue
                if tuple(state_value.shape) != tuple(target_parameter.shape):
                    raise TrainerError(
                        "optimizer state shape mismatch for "
                        f"{parameter_name}.{state_name}"
                    )
                checked_state_tensors += 1
        named_groups.append(
            {
                "name": target_group.get("name"),
                "parameter_names": parameter_names,
            }
        )
    observed_names = [
        name for group in named_groups for name in group["parameter_names"]
    ]
    expected_names = [name for name, _ in module.named_parameters()]
    if sorted(observed_names) != sorted(expected_names):
        raise TrainerError("optimizer named-parameter partition mismatch")
    signature = json.dumps(named_groups, sort_keys=True).encode()
    return {
        "groups": named_groups,
        "parameter_count": len(observed_names),
        "checked_state_tensors": checked_state_tensors,
        "named_signature_sha256": hashlib.sha256(signature).hexdigest(),
        "numeric_parameter_ids_manually_merged": False,
    }


def build_beta_parameter_groups(
    model: torch.nn.Module,
) -> dict[str, list[torch.nn.Parameter]]:
    groups: dict[str, list[torch.nn.Parameter]] = {
        "trajectory": [],
        "condition": [],
        "shape": [],
        "base": [],
        "other": [],
    }
    for name, parameter in model.named_parameters():
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
        groups[group].append(parameter)
    grouped = {
        id(parameter)
        for parameters in groups.values()
        for parameter in parameters
    }
    if grouped != {id(parameter) for parameter in model.parameters()}:
        raise TrainerError("beta optimizer parameter partition is incomplete")
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rolling-data-root", type=Path, required=True)
    parser.add_argument("--rolling-control-root", type=Path, required=True)
    parser.add_argument("--expected-pipeline-sha256", required=True)
    parser.add_argument("--trainer-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--penalty-cache", type=Path, required=True)
    parser.add_argument("--penalty-cache-sha256", required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--resume-sha256", required=True)
    parser.add_argument(
        "--resume-refinement-and-data-transition-from-v6-pg10-v13",
        action="store_true",
    )
    parser.add_argument("--steps", type=int, default=TARGET_STEPS)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--checkpoint-at-stop", action="store_true")
    parser.add_argument("--isolated-smoke", action="store_true")
    parser.add_argument("--beta-lr", type=float, default=1e-4)
    parser.add_argument("--feature-lr-scale", type=float, default=1.0)
    parser.add_argument("--query-lr-scale", type=float, default=1.0)
    parser.add_argument("--shape-lr-scale", type=float, default=1.0)
    parser.add_argument("--other-lr-scale", type=float, default=1.0)
    parser.add_argument("--clip-feature", type=float, default=20.0)
    parser.add_argument("--clip-query", type=float, default=20.0)
    parser.add_argument("--clip-shape", type=float, default=20.0)
    parser.add_argument("--clip-other", type=float, default=20.0)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--batch-per-rank", type=int, default=1)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=EXPECTED_GRAD_ACCUM_STEPS,
    )
    parser.add_argument("--cache-files", type=int, default=2)
    parser.add_argument("--reload-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=202607232355)
    args = parser.parse_args()
    if args.steps != TARGET_STEPS:
        parser.error(f"absolute beta target is fixed at {TARGET_STEPS}")
    if args.expected_pipeline_sha256 != CERTIFIED_PIPELINE_SHA256:
        parser.error("expected pipeline SHA is not certified V14")
    if args.batch_per_rank != 1:
        parser.error("variable-shape rolling training requires batch-per-rank=1")
    if args.grad_accum_steps != EXPECTED_GRAD_ACCUM_STEPS:
        parser.error("8-GPU effective beta task batch is fixed at 32")
    if args.reload_every != RELOAD_INTERVAL_UPDATES:
        parser.error("rolling reload interval is fixed at 100 beta steps")
    if args.checkpoint_every != args.reload_every:
        parser.error("checkpoint and reload intervals must match")
    if args.stop_after is not None and not (
        SOURCE_COMPLETED_STEPS < args.stop_after <= TARGET_STEPS
    ):
        parser.error("stop-after must be after source step and at most 60000")
    if args.isolated_smoke:
        if args.stop_after is None or args.stop_after > SOURCE_COMPLETED_STEPS + 10:
            parser.error("isolated smoke must stop within ten beta updates")
        if args.rolling_control_root.resolve() == (
            CERTIFIED_ROLLING_CONTROL_ROOT.resolve()
        ):
            parser.error("isolated smoke cannot use production control root")
    elif args.checkpoint_at_stop:
        parser.error("checkpoint-at-stop is reserved for isolated smoke")
    return args


def main() -> None:
    args = parse_args()
    configure_v14_supportmix_runtime()
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world = int(os.environ.get("WORLD_SIZE", -1))
    if rank < 0 or local_rank < 0 or world != EXPECTED_WORLD_SIZE:
        raise TrainerError("KALBI V8 trainer requires exactly eight ranks")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    if file_sha256(args.resume) != args.resume_sha256:
        raise TrainerError("resume checkpoint SHA256 mismatch")
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    transition = bool(
        args.resume_refinement_and_data_transition_from_v6_pg10_v13
    )
    if transition:
        if args.resume.resolve() != SOURCE_CHECKPOINT_PATH.resolve():
            raise TrainerError("V6-to-V8 transition source path mismatch")
        if args.resume_sha256 != SOURCE_CHECKPOINT_SHA256:
            raise TrainerError("V6-to-V8 transition source SHA mismatch")
        expected_source = {
            "checkpoint_version": SOURCE_CHECKPOINT_VERSION,
            "trainer_version": SOURCE_TRAINER_VERSION,
            "model_version": SOURCE_MODEL_VERSION,
            "basis_version": SOURCE_BASIS_VERSION,
            "shared_support_contract": SOURCE_SHARED_SUPPORT_CONTRACT,
            "training_mode": "beta_only",
            "world_size": world,
            "selector_version": SOURCE_SELECTOR_VERSION,
        }
        for key, expected in expected_source.items():
            if metadata.get(key) != expected:
                raise TrainerError(
                    f"transition source contract mismatch for {key}"
                )
        if int(checkpoint.get("completed_steps", -1)) != SOURCE_COMPLETED_STEPS:
            raise TrainerError("transition source completed-step mismatch")
        source_pipeline = metadata.get("exact_training_arguments", {}).get(
            "expected_pipeline_sha256"
        )
        if source_pipeline != SOURCE_PIPELINE_SHA256:
            raise TrainerError("transition source is not the certified V13 pipeline")
    else:
        expected_resume = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "trainer_version": TRAINER_VERSION,
            "model_version": MODEL_VERSION,
            "basis_version": BASIS_VERSION,
            "shared_support_contract": SHARED_SUPPORT_CONTRACT,
            "training_mode": "beta_only",
            "world_size": world,
        }
        for key, expected in expected_resume.items():
            if metadata.get(key) != expected:
                raise TrainerError(f"V8 resume contract mismatch for {key}")
        if args.resume.resolve().parent != args.out_dir.resolve():
            raise TrainerError("normal V8 resume must stay in its run directory")

    if not args.penalty_cache.is_file() or (
        file_sha256(args.penalty_cache) != args.penalty_cache_sha256
    ):
        raise TrainerError("canonical penalty cache SHA256 mismatch")

    selector_state = checkpoint.get("selector_state_dict")
    selector_optimizer_state = checkpoint.get("selector_optimizer_state_dict")
    if not isinstance(selector_state, Mapping):
        raise TrainerError("opaque selector state is missing")
    if not isinstance(selector_optimizer_state, Mapping):
        raise TrainerError("opaque selector optimizer state is missing")
    if any(
        tensor.device.type != "cpu"
        for tensor in selector_state.values()
        if torch.is_tensor(tensor)
    ):
        raise TrainerError("opaque selector payload is not CPU-only")
    selector_state_hash = state_dict_sha256(selector_state)
    if selector_state_hash != SELECTOR_STATE_SHA256:
        raise TrainerError("opaque selector state SHA mismatch")
    selector_optimizer_hash = opaque_payload_sha256(selector_optimizer_state)
    selector_hashes = gather_rank_objects(
        {
            "state": selector_state_hash,
            "optimizer": selector_optimizer_hash,
        },
        world,
    )
    if len({json.dumps(item, sort_keys=True) for item in selector_hashes}) != 1:
        raise TrainerError("opaque selector payload differs across ranks")
    opaque_selector_state = selector_state if is_main else None
    opaque_selector_optimizer_state = (
        selector_optimizer_state if is_main else None
    )
    carried_trainer_state = dict(checkpoint.get("trainer_state", {}))
    selector_update_count = int(carried_trainer_state.get("selector_updates", 0))

    policy = CapacityPolicy(
        active_min_bytes=500_000_000,
        active_max_bytes=102_000_000_000,
        prior_min_bytes=500_000_000,
        prior_max_bytes=102_000_000_000,
        hard_cap_bytes=130_000_000_000,
        min_free_bytes=300_000_000_000,
        world_size=world,
    )
    store = TrainerCatalogStore(
        args.rolling_control_root,
        args.rolling_data_root,
        policy=policy,
        priors=("tabicl_v2",),
    )
    if transition:
        (
            active_pointer,
            active_catalog,
            rank_entries,
            _catalog_changed,
        ) = load_current_catalog(
            store,
            rank=rank,
            world=world,
            expected_pipeline_sha256=args.expected_pipeline_sha256,
            active_sha256=None,
        )
    else:
        package = {
            "pointer": checkpoint["catalog_pointer"],
            "catalog": checkpoint["catalog_payload"],
        }
        active_pointer, active_catalog, rank_entries = adopt_catalog_package(
            store,
            package,
            rank=rank,
            world=world,
            expected_pipeline_sha256=args.expected_pipeline_sha256,
            rank0_hashes_verified=False,
        )
    validate_v14_catalog_extensions(
        active_catalog,
        args.expected_pipeline_sha256,
    )
    store.remember_adopted_pointer(active_pointer)
    active_catalog_sha = str(active_pointer["catalog_sha256"])
    if is_main:
        store.acquire_lease(args.trainer_id, active_catalog_sha)
    dist.barrier()
    store.write_ack(args.trainer_id, rank, world, active_catalog_sha)
    dist.barrier()

    rank_states = checkpoint.get("rank_states", ())
    if sorted(int(item["rank"]) for item in rank_states) != list(range(world)):
        raise TrainerError("resume rank-state inventory mismatch")
    rank_state = next(
        item for item in rank_states if int(item["rank"]) == rank
    )
    cache = CanonicalPenaltyCache(
        args.penalty_cache,
        args.seed + 101 * rank,
    )
    cache_report = cache.load(expected_count=300_000)
    cache.load_state_dict(rank_state["penalty_cache"])
    sampler = make_sampler(
        _load_yaml(str(args.config)),
        args.seed + 1_000_003 * rank + 211,
    )
    sampler.load_state_dict(rank_state["sampler"])
    if not math.isclose(
        float(sampler.cfg.lambda_low_min),
        LAMBDA_LOW_MIN_CONTRACT,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise TrainerError("raw-lambda sampling floor drift")
    stream = RollingSingleTaskStream(
        rank_entries,
        catalog_sha256=active_catalog_sha,
        seed=args.seed + 10_000,
        rank=rank,
        cache_files=args.cache_files,
        state=None if transition else rank_state["stream"],
    )
    carried_split_generator_state = rank_state["split_generator_state"]

    config = _load_yaml(str(args.config))
    model = build_model(config, device)
    if (
        int(model.kalbi_isis_topk) != KALBI_TOPK
        or int(model.kalbi_isis_stages) != KALBI_STAGES
        or float(model.kalbi_isis_ridge) != KALBI_RIDGE
    ):
        raise TrainerError("KALBI V8 K/S/ridge architecture drift")
    if sum(parameter.numel() for parameter in model.parameters()) != (
        EXPECTED_MODEL_PARAMETER_COUNT
    ):
        raise TrainerError("KALBI V8 trainable parameter-count drift")
    model_state_audit = audit_state_dict_compatibility(
        model,
        checkpoint["model_state_dict"],
    )
    source_model_hash = state_dict_sha256(checkpoint["model_state_dict"])
    if module_sha256(model) != source_model_hash:
        raise TrainerError("strict model load changed source weights")
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=True,
        find_unused_parameters=False,
        init_sync=True,
    )
    core = model.module

    beta_groups = build_beta_parameter_groups(core)
    lr_scales = {
        "trajectory": args.feature_lr_scale,
        "condition": args.query_lr_scale,
        "shape": args.shape_lr_scale,
        "base": args.other_lr_scale,
        "other": args.other_lr_scale,
    }
    clip_limits = {
        "trajectory": args.clip_feature,
        "condition": args.clip_query,
        "shape": args.clip_shape,
        "base": args.clip_other,
        "other": args.clip_other,
    }
    active_group_names = tuple(
        name for name, parameters in beta_groups.items() if parameters
    )
    beta_optimizer = torch.optim.AdamW(
        [
            {
                "params": beta_groups[name],
                "lr": args.beta_lr * lr_scales[name],
                "name": name,
            }
            for name in active_group_names
        ],
        lr=args.beta_lr,
        weight_decay=1e-4,
        betas=(0.9, 0.95),
    )
    optimizer_audit = audit_optimizer_state_by_parameter_name(
        module=core,
        optimizer=beta_optimizer,
        source_state_dict=checkpoint["beta_optimizer_state_dict"],
    )
    beta_optimizer.load_state_dict(checkpoint["beta_optimizer_state_dict"])
    optimizer_to_device(beta_optimizer, device)

    def multiplier(step: int) -> float:
        if step < args.warmup_steps:
            return float(step + 1) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(
            max(1, SCHEDULE_PHASE_STEPS - args.warmup_steps)
        )
        return 0.5 * (
            1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0))
        )

    beta_scheduler = torch.optim.lr_scheduler.LambdaLR(
        beta_optimizer,
        multiplier,
    )
    beta_scheduler.load_state_dict(checkpoint["beta_scheduler_state_dict"])
    completed_steps = int(checkpoint["completed_steps"])
    previous_checkpoint_id = (
        None if transition else metadata.get("checkpoint_id")
    )
    knots = canonical_grid(device=device)

    if is_main:
        if transition:
            if args.out_dir.exists() and any(
                args.out_dir.glob("checkpoint_step*.pt")
            ):
                raise TrainerError(
                    "V8 transition output already contains checkpoints"
                )
            args.out_dir.mkdir(parents=True, exist_ok=True)
        elif not args.out_dir.is_dir():
            raise TrainerError("V8 resume output directory is missing")
    dist.barrier()
    events_path = args.out_dir / "events.jsonl"

    def emit(event: Mapping[str, Any]) -> None:
        if not is_main:
            return
        record = {"timestamp": time.time(), **dict(event)}
        line = json.dumps(record, sort_keys=True, default=str)
        print(line, flush=True)
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    source_hashes = {
        "trainer": file_sha256(Path(__file__)),
        "model": file_sha256(Path(__file__).with_name("model.py")),
        "adaptive_block_prox_v8": file_sha256(
            Path(__file__).with_name("adaptive_block_prox_v8.py")
        ),
        "block_prox_v7_kernel": file_sha256(
            Path(__file__).with_name("block_prox_v7.py")
        ),
        "exact_direct96_prox": file_sha256(
            Path(__file__).with_name("exact_direct96_prox.py")
        ),
        "beta_objective": file_sha256(Path(__file__).with_name("train.py")),
        "rolling_beta_helpers": file_sha256(
            Path(__file__).with_name(
                "train_beta_only_v23_supportmix_v13_transition_8gpu.py"
            )
        ),
        "rolling_catalog": file_sha256(
            Path(__file__).with_name("rolling_catalog.py")
        ),
        "canonical_penalty": file_sha256(
            Path(__file__).with_name("canonical_penalty.py")
        ),
        "canonical_penalty_cache": file_sha256(
            Path(__file__).with_name("canonical_penalty_cache.py")
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
        "beta_lr_scales": lr_scales,
        "beta_clip_limits": clip_limits,
        "warmup_steps": args.warmup_steps,
        "beta_schedule_phase_steps": SCHEDULE_PHASE_STEPS,
        "batch_per_rank": args.batch_per_rank,
        "grad_accum_steps": args.grad_accum_steps,
        "cache_files": args.cache_files,
        "reload_every": args.reload_every,
        "checkpoint_every": args.checkpoint_every,
        "log_every": args.log_every,
        "seed": args.seed,
        "training_mode": "beta_only",
        "selector_update_enabled": False,
        "selector_payload_carried_opaque": True,
        "model_parameter_count": EXPECTED_MODEL_PARAMETER_COUNT,
        "refinement_mode": KALBI_REFINEMENT_MODE,
        "refinement_sweeps": KALBI_REFINEMENT_SWEEPS,
        "refinement_sweep_order": SWEEP_ORDER_VERSION,
        "adaptive_block_formula": ADAPTIVE_BLOCK_FORMULA_VERSION,
        "adaptive_block_min": KALBI_BLOCK_PROX_MIN_BLOCK_SIZE,
        "adaptive_block_max": KALBI_BLOCK_PROX_MAX_BLOCK_SIZE,
        "data_contract": V14_DATASET_CONTRACT,
        "target_contract": V14_TARGET_CONTRACT,
        "linear_snr_contract": V14_LINEAR_SNR_CONTRACT,
        "linear_snr_bounds": [V14_LINEAR_SNR_MIN, V14_LINEAR_SNR_MAX],
        "rolling_stream_version": RollingSingleTaskStream.VERSION,
        "isolated_smoke": bool(args.isolated_smoke),
    }
    if not transition:
        if metadata.get("source_hashes") != source_hashes:
            raise TrainerError("normal V8 resume source-hash mismatch")
        if metadata.get("exact_training_arguments") != exact_arguments:
            raise TrainerError("normal V8 resume argument mismatch")
        if metadata.get("selector_state_sha256") != selector_state_hash:
            raise TrainerError("normal V8 resume selector hash mismatch")

    transition_audit = {
        "mode": (
            "v6_pg10_v13_to_v8_adaptive_blockprox10_v14"
            if transition
            else "v8_strict_resume"
        ),
        "source_checkpoint": str(args.resume.resolve()),
        "source_checkpoint_sha256": args.resume_sha256,
        "source_completed_steps": completed_steps,
        "model_state": model_state_audit,
        "optimizer": optimizer_audit,
        "beta_scheduler_state_imported": True,
        "rank_rng_state_imported": True,
        "penalty_cache_state_imported": True,
        "penalty_sampler_state_imported": True,
        "stream_state_imported": not transition,
        "catalog_state_imported": not transition,
        "source_pipeline_sha256": (
            SOURCE_PIPELINE_SHA256 if transition else CERTIFIED_PIPELINE_SHA256
        ),
        "target_pipeline_sha256": CERTIFIED_PIPELINE_SHA256,
        "selector_payload_carried_opaque": True,
        "selector_state_sha256": selector_state_hash,
        "selector_optimizer_payload_sha256": selector_optimizer_hash,
        "selector_updates_carried": selector_update_count,
    }
    emit(
        {
            "kind": "start",
            "trainer_version": TRAINER_VERSION,
            "checkpoint_version": CHECKPOINT_VERSION,
            "model_version": MODEL_VERSION,
            "training_mode": "beta_only",
            "selector_model_instantiated": False,
            "selector_ddp_instantiated": False,
            "selector_optimizer_instantiated": False,
            "selector_update_enabled": False,
            "completed_steps": completed_steps,
            "catalog_epoch": int(active_pointer["epoch"]),
            "catalog_sha256": active_catalog_sha,
            "cache_report": cache_report,
            "transition_audit": transition_audit,
            "source_hashes": source_hashes,
            "exact_training_arguments": exact_arguments,
        }
    )

    # Restore global RNG only after model/DDP/optimizer construction so setup
    # cannot consume the checkpointed continuation stream.
    restore_rng(rank_state["rng"])
    initial_beta_hash = module_sha256(core)
    initial_hashes = gather_rank_objects(initial_beta_hash, world)
    if len(set(initial_hashes)) != 1:
        raise TrainerError("initial beta model hash differs across ranks")
    if initial_beta_hash != source_model_hash:
        raise TrainerError("DDP initialization changed resumed beta weights")

    def checkpoint_payload(
        *,
        checkpoint_id: str,
        rank_states_payload: list[dict[str, Any]],
        catalog_reloaded: bool,
    ) -> dict[str, Any]:
        if opaque_selector_state is None:
            raise TrainerError("rank 0 lost opaque selector state")
        if opaque_selector_optimizer_state is None:
            raise TrainerError("rank 0 lost opaque selector optimizer state")
        trainer_state = dict(carried_trainer_state)
        trainer_state.update(
            {
                "selector_frozen_beta_only": True,
                "selector_updates": selector_update_count,
                "selector_payload_carried_opaque": True,
            }
        )
        return {
            "completed_steps": completed_steps,
            "model_state_dict": core.state_dict(),
            "selector_state_dict": opaque_selector_state,
            "beta_optimizer_state_dict": beta_optimizer.state_dict(),
            "selector_optimizer_state_dict": opaque_selector_optimizer_state,
            "beta_scheduler_state_dict": beta_scheduler.state_dict(),
            "rank_states": rank_states_payload,
            "trainer_state": trainer_state,
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
                "canonical_cache_version": CACHE_VERSION,
                "training_mode": "beta_only",
                "world_size": world,
                "selector_version": SOURCE_SELECTOR_VERSION,
                "selector_update_enabled": False,
                "selector_payload_carried_opaque": True,
                "selector_state_unchanged": True,
                "selector_state_sha256": selector_state_hash,
                "selector_optimizer_payload_sha256": (
                    selector_optimizer_hash
                ),
                "selector_trainable_parameter_count": 0,
                "selector_optimizer_instantiated": False,
                "selector_updates": selector_update_count,
                "refinement_mode": KALBI_REFINEMENT_MODE,
                "refinement_sweeps": KALBI_REFINEMENT_SWEEPS,
                "refinement_sweep_order": SWEEP_ORDER_VERSION,
                "adaptive_block_formula": ADAPTIVE_BLOCK_FORMULA_VERSION,
                "catalog_epoch": int(active_pointer["epoch"]),
                "catalog_sha256": active_catalog_sha,
                "catalog_reloaded_before_checkpoint": catalog_reloaded,
                "reload_interval_updates": RELOAD_INTERVAL_UPDATES,
                "source_hashes": source_hashes,
                "exact_training_arguments": exact_arguments,
                "transition_audit": transition_audit,
                "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            },
        }

    def save_checkpoint(*, reload_catalog: bool) -> dict[str, Any]:
        nonlocal active_pointer, active_catalog, active_catalog_sha
        nonlocal previous_checkpoint_id, rank_entries
        old_catalog_sha = active_catalog_sha
        old_checkpoint_id = previous_checkpoint_id
        catalog_changed = False
        if reload_catalog:
            (
                active_pointer,
                active_catalog,
                rank_entries,
                catalog_changed,
            ) = load_current_catalog(
                store,
                rank=rank,
                world=world,
                expected_pipeline_sha256=args.expected_pipeline_sha256,
                active_sha256=active_catalog_sha,
            )
            validate_v14_catalog_extensions(
                active_catalog,
                args.expected_pipeline_sha256,
            )
            store.remember_adopted_pointer(active_pointer)
            active_catalog_sha = str(active_pointer["catalog_sha256"])
            if is_main and active_catalog_sha != old_catalog_sha:
                store.acquire_lease(args.trainer_id, active_catalog_sha)
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
            "split_generator_state": carried_split_generator_state,
        }
        gathered_states = gather_rank_objects(local_state, world)
        checkpoint_id = f"{args.trainer_id}_step{completed_steps:06d}"
        checkpoint_path = (
            args.out_dir / f"checkpoint_step{completed_steps:06d}.pt"
        )
        result: dict[str, Any] = {}
        if is_main:
            payload = checkpoint_payload(
                checkpoint_id=checkpoint_id,
                rank_states_payload=gathered_states,
                catalog_reloaded=reload_catalog,
            )
            atomic_torch_save(payload, checkpoint_path)
            checkpoint_sha = file_sha256(checkpoint_path)
            reloaded = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            if (
                int(reloaded["completed_steps"]) != completed_steps
                or state_dict_sha256(reloaded["model_state_dict"])
                != module_sha256(core)
                or state_dict_sha256(reloaded["selector_state_dict"])
                != selector_state_hash
                or opaque_payload_sha256(
                    reloaded["selector_optimizer_state_dict"]
                )
                != selector_optimizer_hash
                or reloaded["metadata"]["catalog_sha256"]
                != active_catalog_sha
            ):
                raise TrainerError("V8 checkpoint strict reload failed")
            store.register_checkpoint_pin(
                checkpoint_id,
                active_catalog_sha,
                checkpoint_sha256=checkpoint_sha,
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
            result = {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_id": checkpoint_id,
                "catalog_changed": catalog_changed,
                "catalog_epoch": int(active_pointer["epoch"]),
                "catalog_sha256": active_catalog_sha,
                "strict_reload": True,
                "selector_state_sha256": selector_state_hash,
            }
            emit({"kind": "checkpoint", "step": completed_steps, **result})
        previous_checkpoint_id = checkpoint_id
        dist.barrier()
        return result

    resume_completed_steps = completed_steps
    target_steps = (
        args.steps if args.stop_after is None else args.stop_after
    )
    compute_seconds = 0.0
    compute_steps = 0
    while completed_steps < target_steps:
        torch.cuda.reset_peak_memory_stats(device)
        step_started = time.perf_counter()
        beta_optimizer.zero_grad(set_to_none=True)
        metric_names = (
            "loss",
            "data_half",
            "mse",
            "penalty",
            "objective",
            "f_zero",
            "objective_over_f_zero",
            "kkt_top32",
            "nnz",
            "nnz_1e4",
            *tuple(
                f"refinement_{name}"
                for name in REFINEMENT_PATH_METRIC_NAMES
            ),
            *tuple(
                f"refinement_{name}_final"
                for name in REFINEMENT_PATH_METRIC_NAMES
            ),
            *REFINEMENT_AGGREGATE_METRIC_NAMES,
        )
        metrics = {name: 0.0 for name in metric_names}
        paths = {
            name: torch.zeros(
                KALBI_REFINEMENT_SWEEPS,
                dtype=torch.float32,
                device=device,
            )
            for name in REFINEMENT_PATH_METRIC_NAMES
        }
        shapes: list[tuple[int, int]] = []
        strata: list[dict[str, Any]] = []
        sampled_lambdas: list[torch.Tensor] = []
        forward_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        backward_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

        for accumulation in range(args.grad_accum_steps):
            xs, ys, info = stream.next_batch()
            task_info = info[0]
            shapes.append((int(task_info["n"]), int(task_info["p"])))
            payload = make_canonical_query_payload(
                xs,
                ys,
                cache,
                sampler,
                device,
            )
            sampled_lambdas.append(
                payload["lambdas"].detach().float().cpu().reshape(-1)
            )
            if float(payload["lambdas"].min()) < (
                LAMBDA_LOW_MIN_CONTRACT - 1e-12
            ):
                raise TrainerError("sampled raw lambda violated support")
            sync = (
                model.no_sync()
                if accumulation < args.grad_accum_steps - 1
                else contextlib.nullcontext()
            )
            forward_start = torch.cuda.Event(enable_timing=True)
            forward_end = torch.cuda.Event(enable_timing=True)
            backward_start = torch.cuda.Event(enable_timing=True)
            backward_end = torch.cuda.Event(enable_timing=True)
            with sync:
                forward_start.record()
                result = objective_kalbi_v8_adaptiveblockprox10(
                    model,
                    payload,
                    knots,
                )
                forward_end.record()
                if not bool(torch.isfinite(result["loss"])):
                    raise TrainerError("nonfinite beta loss")
                backward_start.record()
                (
                    result["loss"] / float(args.grad_accum_steps)
                ).backward()
                backward_end.record()
            forward_events.append((forward_start, forward_end))
            backward_events.append((backward_start, backward_end))

            for name in metrics:
                metrics[name] += (
                    float(result[name].detach())
                    / float(args.grad_accum_steps)
                )
            for name in REFINEMENT_PATH_METRIC_NAMES:
                path = result[
                    f"refinement_{name}_by_iteration"
                ].detach().float()
                if path.shape != (KALBI_REFINEMENT_SWEEPS,):
                    raise TrainerError(
                        f"ten-sweep telemetry shape drift for {name}"
                    )
                paths[name].add_(
                    path / float(args.grad_accum_steps)
                )
            source_id = int(task_info["outcome_source_id"])
            stratum = {
                "outcome_source": OUTCOME_SOURCE_NAMES[source_id],
                "n": int(task_info["n"]),
                "p": int(task_info["p"]),
                "loss": float(result["loss"].detach()),
                "objective": float(result["objective"].detach()),
                "kkt_top32": float(result["kkt_top32"].detach()),
                "adaptive_block_size": float(
                    result["refinement_adaptive_block_size"].detach()
                ),
                "blocks_per_sweep": float(
                    result["refinement_blocks_per_sweep"].detach()
                ),
            }
            if source_id == OUTCOME_SOURCE_LINEAR:
                stratum.update(
                    {
                        "support_size": int(
                            task_info["linear_support_size"]
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
                        "realized_snr": float(
                            task_info["linear_realized_snr"]
                        ),
                    }
                )
            strata.append(stratum)

        group_gradient_tensors = {
            name: torch.nn.utils.clip_grad_norm_(
                beta_groups[name],
                clip_limits[name],
                error_if_nonfinite=False,
            ).detach()
            for name in active_group_names
        }
        total_gradient = float(
            torch.stack(
                [
                    value.float().square()
                    for value in group_gradient_tensors.values()
                ]
            ).sum().sqrt()
        )
        if not math.isfinite(total_gradient) or total_gradient <= 0.0:
            raise TrainerError("beta gradient is nonfinite or zero")
        beta_optimizer.step()
        beta_scheduler.step()
        completed_steps += 1
        torch.cuda.synchronize(device)
        forward_seconds = sum(
            start.elapsed_time(end) for start, end in forward_events
        ) / 1000.0
        backward_seconds = sum(
            start.elapsed_time(end) for start, end in backward_events
        ) / 1000.0
        total_seconds = time.perf_counter() - step_started
        compute_seconds += total_seconds
        compute_steps += 1

        if completed_steps == SOURCE_COMPLETED_STEPS + 1:
            post_update_hash = module_sha256(core)
            hashes = gather_rank_objects(post_update_hash, world)
            if len(set(hashes)) != 1:
                raise TrainerError("post-update beta hash differs across ranks")
            if post_update_hash == initial_beta_hash:
                raise TrainerError("first V8 beta update changed no parameter")

        if completed_steps % args.log_every == 0 or args.isolated_smoke:
            lambda_values = torch.cat(sampled_lambdas)
            path_json = {
                name: value.cpu().tolist()
                for name, value in paths.items()
            }
            local_monitor = {
                "rank": rank,
                "catalog_epoch": int(active_pointer["epoch"]),
                "catalog_sha256": active_catalog_sha,
                "stream_reload_count": stream.reload_count,
                "beta_metrics": metrics,
                "beta_refinement_by_sweep": path_json,
                "beta_total_gradient_preclip": total_gradient,
                "beta_group_gradients_preclip": {
                    name: float(value)
                    for name, value in group_gradient_tensors.items()
                },
                "raw_lambda_min": float(lambda_values.min()),
                "raw_lambda_median": float(lambda_values.median()),
                "raw_lambda_max": float(lambda_values.max()),
                "beta_shapes": shapes,
                "beta_task_strata": strata,
                "forward_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
                "total_step_seconds": total_seconds,
                "peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                ),
                "selector_update_performed": False,
                "selector_updates": selector_update_count,
                "selector_state_sha256": selector_state_hash,
                "fixed_ridge_numerics": (
                    fixed_ridge_numerical_diagnostics()
                ),
            }
            monitors = gather_rank_objects(local_monitor, world)
            emit(
                {
                    "kind": "train_update",
                    "step": completed_steps,
                    "training_mode": "beta_only",
                    "average_compute_seconds_per_step": (
                        compute_seconds / compute_steps
                    ),
                    "rank_monitors": monitors,
                }
            )
            compute_seconds = 0.0
            compute_steps = 0

        if completed_steps % args.reload_every == 0:
            save_checkpoint(reload_catalog=True)

    if (
        args.checkpoint_at_stop
        and completed_steps > resume_completed_steps
        and completed_steps % args.reload_every != 0
    ):
        save_checkpoint(reload_catalog=False)

    final_beta_hash = module_sha256(core)
    final_hashes = gather_rank_objects(final_beta_hash, world)
    if len(set(final_hashes)) != 1:
        raise TrainerError("final beta model hash differs across ranks")
    if (
        is_main
        and state_dict_sha256(opaque_selector_state) != selector_state_hash
    ):
        raise TrainerError("opaque selector state changed")
    if transition and file_sha256(args.resume) != SOURCE_CHECKPOINT_SHA256:
        raise TrainerError("source checkpoint changed during V8 run")
    emit(
        {
            "kind": "complete",
            "step": completed_steps,
            "model_sha256": final_beta_hash,
            "selector_state_sha256": selector_state_hash,
            "selector_state_unchanged": True,
            "selector_updates": selector_update_count,
            "catalog_epoch": int(active_pointer["epoch"]),
            "catalog_sha256": active_catalog_sha,
        }
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
