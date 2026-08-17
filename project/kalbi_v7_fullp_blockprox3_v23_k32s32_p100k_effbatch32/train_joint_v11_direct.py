#!/usr/bin/env python3
"""Joint V58-Chebyshev beta-solver and native196 direct96 V11 training.

Each optimizer update has two deliberately separate phases:

1. update V58-Chebyshev on the original full-row penalized objective;
2. freeze V58-Chebyshev weights and update V11 through its fixed-support
   differentiable decoder path using a leakage-free 100/100 row split and
   held-out prediction loss.

The public penalty payload is always native196/meta4.  The beta-query sampler
and V11 effective-ratio lower bounds are pinned to 1e-6 and serialized in every
checkpoint/event.  No lambda-times-penalty term is added to selector loss.
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
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .model import (
    BASE_MODEL_VERSION,
    BASIS_VERSION,
    LOCAL_ISIS_GRADIENT_CONTRACT,
    MODEL_VERSION,
    PENALTY_CONTRACT_VERSION,
    build_model,
)
from .selector_v10 import (
    selector_beta_with_inactive_ste,
    summarize_encoded,
)
from .selector_v11_direct import (
    DirectPenaltyMLPQuerySelectorV11,
    V11_DIRECT_PENALTY_ENCODING_VERSION,
)
from .train import (
    ROOT_DEPENDENCY_PATHS,
    PartStream,
    PenaltyCache,
    TrainerError,
    atomic_torch_save,
    capture_rng,
    make_query_payload,
    make_sampler,
    module_grad_norm,
    objective_exp125_v58,
    restore_rng,
    state_dict_sha256,
    validate_part,
)
from train_fpk import _build_X_y_tensors, _load_yaml


TRAINER_VERSION = "v58cheb_v11_direct96_joint_small_n200_p400_accum4_v1"
CHECKPOINT_VERSION = "v58cheb_v11_direct96_joint_checkpoint_v1"
EXPECTED_WORLD_SIZE = 8
LAMBDA_LOW_MIN_CONTRACT = 1e-6
SELECTOR_RATIO_MIN = 1e-6
SELECTOR_RATIO_MAX = 1.0
SELECTOR_OBJECTIVE = "mean(log1p(heldout_mse/2))"
SELECTOR_PATH_FEATURES = 150


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def module_sha256(module: torch.nn.Module) -> str:
    return state_dict_sha256(module.state_dict())


def assert_ddp_synced(
    module: torch.nn.Module, world: int, label: str
) -> str:
    fingerprint = module_sha256(module)
    fingerprints = [None for _ in range(world)]
    dist.all_gather_object(fingerprints, fingerprint)
    if len(set(fingerprints)) != 1:
        raise TrainerError(f"DDP state mismatch at {label}: {fingerprints}")
    return fingerprint


def grad_stats(parameters: Iterable[torch.nn.Parameter]) -> dict[str, float | int]:
    values = [
        parameter.grad.detach().float().reshape(-1)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not values:
        return {"norm": 0.0, "rms": 0.0, "nonzero_fraction": 0.0, "elements": 0}
    flat = torch.cat(values)
    return {
        "norm": float(flat.norm()),
        "rms": float(flat.square().mean().sqrt()),
        "nonzero_fraction": float((flat != 0.0).float().mean()),
        "elements": int(flat.numel()),
    }


def tensor_gradient_stats(values: list[torch.Tensor]) -> dict[str, float | int]:
    if not values:
        return {
            "norm": 0.0,
            "rms": 0.0,
            "nonzero_fraction": 0.0,
            "elements": 0,
        }
    flat = torch.cat([value.detach().float().reshape(-1) for value in values])
    return {
        "norm": float(flat.norm()),
        "rms": float(flat.square().mean().sqrt()),
        "nonzero_fraction": float((flat != 0.0).float().mean()),
        "elements": int(flat.numel()),
    }


@torch.no_grad()
def migrate_shared_v10_selector_state(
    selector: DirectPenaltyMLPQuerySelectorV11,
    source: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Copy task encoders, split trunks, and lambda head but reset direct96."""
    current = selector.state_dict()
    copied: list[str] = []
    skipped: list[str] = []
    for key, value in source.items():
        if key.startswith("penalty_parameter_head."):
            skipped.append(key)
            continue
        if key in current and current[key].shape == value.shape:
            current[key] = value
            copied.append(key)
        else:
            skipped.append(key)
    selector.load_state_dict(current, strict=True)
    return {
        "source": "v10_shared_context_lambda_penalty_trunk",
        "copied_key_count": len(copied),
        "skipped_key_count": len(skipped),
        "direct96_head_reset_from_cache_mean": True,
    }


def split_and_standardize(
    Xs: list[np.ndarray],
    ys: list[np.ndarray],
    *,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X = torch.from_numpy(np.stack(Xs).astype(np.float32, copy=False)).to(device)
    y = torch.from_numpy(np.stack(ys).astype(np.float32, copy=False)).to(device)
    batch, rows, features = X.shape
    train_rows = rows // 2
    if rows != 200 or features != 400 or train_rows != 100:
        raise TrainerError(f"joint split requires stored N=200/P=400, got {X.shape}")
    train_indices = []
    test_indices = []
    for _ in range(batch):
        permutation = torch.randperm(rows, generator=generator)
        train_indices.append(permutation[:train_rows])
        test_indices.append(permutation[train_rows:])
    train_index = torch.stack(train_indices).to(device)
    test_index = torch.stack(test_indices).to(device)
    X_train_raw = torch.gather(
        X, 1, train_index[:, :, None].expand(-1, -1, features)
    )
    X_test_raw = torch.gather(
        X, 1, test_index[:, :, None].expand(-1, -1, features)
    )
    y_train_raw = torch.gather(y, 1, train_index)
    y_test_raw = torch.gather(y, 1, test_index)
    X_mean = X_train_raw.mean(dim=1, keepdim=True)
    X_std_raw = X_train_raw.std(dim=1, keepdim=True, correction=0)
    active = X_std_raw > 1e-6
    X_std = X_std_raw.clamp_min(1e-8)
    y_mean = y_train_raw.mean(dim=1, keepdim=True)
    y_std = y_train_raw.std(dim=1, keepdim=True, correction=0).clamp_min(1e-8)
    X_train = torch.where(active, (X_train_raw - X_mean) / X_std, 0.0)
    X_test = torch.where(active, (X_test_raw - X_mean) / X_std, 0.0)
    y_train = (y_train_raw - y_mean) / y_std
    y_test = (y_test_raw - y_mean) / y_std
    for name, value in (
        ("X_train", X_train),
        ("y_train", y_train),
        ("X_test", X_test),
        ("y_test", y_test),
    ):
        if not bool(torch.isfinite(value).all()):
            raise TrainerError(f"nonfinite standardized selector tensor {name}")
    return X_train, y_train, X_test, y_test


@torch.no_grad()
def selector_path_features(
    model: torch.nn.Module,
    encoded: dict[str, Any],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    fixed_shape: torch.Tensor,
    fixed_d0: torch.Tensor,
    effective_ratios: torch.Tensor,
    chunk: int,
) -> torch.Tensor:
    """Three scale-aligned reference-penalty paths, 50 ratios each."""
    batch = X_train.shape[0]
    penalty_count = fixed_shape.shape[0]
    ratio_count = effective_ratios.numel()
    if penalty_count * ratio_count != SELECTOR_PATH_FEATURES:
        raise TrainerError("selector path feature width drift")
    lambda_max = encoded["lambda_max"].float().clamp_min(1e-12)
    base = y_train.square().mean(dim=1, keepdim=True).clamp_min(1e-8)
    shape = fixed_shape[:, None, :].expand(
        penalty_count, ratio_count, fixed_shape.shape[-1]
    ).reshape(1, -1, fixed_shape.shape[-1]).expand(batch, -1, -1)
    d0 = fixed_d0[:, None].expand(penalty_count, ratio_count).reshape(1, -1)
    d0 = d0.expand(batch, -1)
    ratio = effective_ratios[None, None, :].expand(
        batch, penalty_count, ratio_count
    ).reshape(batch, -1)
    # Match origin threshold strength: lambda * d0 / lambda_max = ratio.
    lambdas = lambda_max[:, None] * ratio / d0.clamp_min(1e-8)
    values = []
    for start in range(0, lambdas.shape[1], int(chunk)):
        end = min(start + int(chunk), lambdas.shape[1])
        beta = model.decode_queries(
            encoded,
            lambdas=lambdas[:, start:end],
            lambda_max=lambda_max[:, None],
            penalty_shape_features=shape[:, start:end],
            penalty_d0=d0[:, start:end],
            return_aux=False,
        ).float()
        prediction = torch.einsum("bnp,bqp->bqn", X_train.float(), beta)
        mse = (prediction - y_train[:, None, :].float()).square().mean(dim=-1)
        values.append(torch.log1p(mse / base))
    result = torch.cat(values, dim=1).clamp(0.0, 20.0)
    if result.shape != (batch, SELECTOR_PATH_FEATURES):
        raise TrainerError(f"selector path shape drift: {result.shape}")
    return result


@torch.no_grad()
def selector_snapshot(
    selector: DirectPenaltyMLPQuerySelectorV11,
    model: torch.nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    fixed_shape: torch.Tensor,
    fixed_d0: torch.Tensor,
    path_ratios: torch.Tensor,
    path_chunk: int,
) -> dict[str, Any]:
    encoded = model.encode_dataset(model.precompute(X_train, y_train))
    dataset_features, krylov_features = summarize_encoded(
        encoded, 32 * 33 // 2
    )
    path = selector_path_features(
        model, encoded, X_train, y_train,
        fixed_shape, fixed_d0, path_ratios, path_chunk,
    )
    query = selector(
        dataset_features,
        krylov_features,
        encoded["lambda_max"].float(),
        path,
        U=encoded.get("U"),
        y_context=y_train,
    )
    rho = query["penalty_rho"].detach().cpu()
    drho = query["penalty_drho"].detach().cpu()
    return {
        "effective_ratio": query["lambda_ratio"].detach().cpu().tolist(),
        "raw_ratio": (
            query["lambdas"][:, 0] / encoded["lambda_max"].float()
        ).detach().cpu().tolist(),
        "rho": rho.tolist(),
        "drho": drho.tolist(),
        "rho_between_task_std_mean": float(rho.std(0, unbiased=False).mean()),
        "drho_between_task_std_mean": float(drho.std(0, unbiased=False).mean()),
        "min_drho": float(query["penalty_min_drho"].min().cpu()),
        "rho6_error_max": float(query["penalty_rho6_error"].max().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--part", default="part_0")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--penalty-cache", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--warm-start-sha256", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sha256")
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--beta-lr", type=float, default=2e-4)
    parser.add_argument("--selector-lr", type=float, default=3e-4)
    parser.add_argument("--direct-head-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--batch-per-rank", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--selector-grad-accum-steps", type=int, default=4)
    parser.add_argument("--selector-every", type=int, default=1)
    parser.add_argument("--selector-hidden", type=int, default=64)
    parser.add_argument("--direct-positivity-beta", type=float, default=100.0)
    parser.add_argument("--direct-head-init-std", type=float, default=1e-3)
    parser.add_argument("--path-chunk", type=int, default=25)
    parser.add_argument("--holdout-files", type=int, default=8)
    parser.add_argument("--cache-files", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=202607161405)
    parser.add_argument("--max-abs-feature-corr", type=float, default=0.95)
    args = parser.parse_args()
    if args.batch_per_rank * args.grad_accum_steps != 4:
        parser.error("batch-per-rank * grad-accum-steps must equal 4")
    if args.batch_per_rank * args.selector_grad_accum_steps != 4:
        parser.error(
            "batch-per-rank * selector-grad-accum-steps must equal 4"
        )
    if args.selector_every <= 0 or args.path_chunk <= 0:
        parser.error("selector-every and path-chunk must be positive")
    if args.selector_lr <= 0.0 or args.direct_head_lr <= 0.0:
        parser.error("selector-lr and direct-head-lr must be positive")
    if args.resume is not None and not args.resume_sha256:
        parser.error("--resume requires --resume-sha256")
    target_step = args.steps if args.stop_after is None else min(
        args.steps, args.stop_after
    )
    if target_step <= 0:
        parser.error("target step must be positive")

    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world = int(os.environ.get("WORLD_SIZE", -1))
    if rank < 0 or local_rank < 0 or world != EXPECTED_WORLD_SIZE:
        raise TrainerError("joint V58-Chebyshev/V11 requires exactly eight ranks")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    inventory_holder: list[Any] = [None]
    if is_main:
        inventory_holder[0] = validate_part(
            args.data_root, args.part, args.max_abs_feature_corr
        )
    dist.broadcast_object_list(inventory_holder, src=0)
    inventory = inventory_holder[0]
    files = list(inventory["files"])
    if args.holdout_files < world or args.holdout_files >= len(files):
        raise TrainerError("invalid holdout file count")
    train_files = files[:-args.holdout_files]

    if file_sha256(args.warm_start) != args.warm_start_sha256:
        raise TrainerError("warm-start checkpoint SHA256 mismatch")
    if not args.penalty_cache.is_file():
        raise TrainerError("penalty cache is missing")
    config = _load_yaml(str(args.config))
    warm_checkpoint = torch.load(
        args.warm_start, map_location="cpu", weights_only=False
    )
    warm_metadata = warm_checkpoint.get("metadata", {})
    warm_model_version = warm_metadata.get("model_version")
    if warm_model_version not in {BASE_MODEL_VERSION, MODEL_VERSION}:
        raise TrainerError(
            f"unsupported warm-start model version: {warm_model_version!r}"
        )
    if warm_metadata.get("penalty_contract_version") != PENALTY_CONTRACT_VERSION:
        raise TrainerError("warm-start penalty contract mismatch")
    if warm_model_version == BASE_MODEL_VERSION:
        legacy_warm_contract = {
            "penalty_input_dim": 196,
            "penalty_meta_dim": 4,
            "expanded_penalty_dim": 388,
            "convexity_flag_present": False,
        }
        for key, expected in legacy_warm_contract.items():
            if warm_metadata.get(key) != expected:
                raise TrainerError(
                    f"warm-start contract mismatch {key}: "
                    f"{warm_metadata.get(key)!r} != {expected!r}"
                )

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = build_model(config, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    state_element_count = sum(
        tensor.numel() for tensor in model.state_dict().values()
    )
    if parameter_count != 4_909_208 or state_element_count != 4_910_640:
        raise TrainerError(
            "V58-Chebyshev architecture drift: "
            f"parameters={parameter_count}, state_elements={state_element_count}"
        )
    load_result = model.load_state_dict(
        warm_checkpoint["model_state_dict"], strict=True
    )
    if load_result.missing_keys or load_result.unexpected_keys:
        raise TrainerError("strict warm-start unexpectedly reported schema drift")
    warm_model_hash = module_sha256(model)

    cache = PenaltyCache(args.penalty_cache.resolve(), args.seed + 101 * rank)
    cache.load()
    if cache.pool_size != 100_000:
        raise TrainerError("V11 calibration requires the full 100k penalty cache")
    sampler = make_sampler(config, args.seed + 1_000_003 * rank + 211)
    if not math.isclose(
        float(sampler.cfg.lambda_low_min),
        LAMBDA_LOW_MIN_CONTRACT,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise TrainerError(
            "lambda_low_min_contract drift: "
            f"{sampler.cfg.lambda_low_min} != {LAMBDA_LOW_MIN_CONTRACT}"
        )

    selector_holder: list[Any] = [None]
    calibration_holder: list[Any] = [None]
    if is_main:
        torch.manual_seed(args.seed + 17)
        selector_cpu = DirectPenaltyMLPQuerySelectorV11(
            ds_feat_dim=9,
            t_feat_dim=32 * 33 // 2,
            krylov_dim=32,
            shape_dim=196,
            path_feat_dim=SELECTOR_PATH_FEATURES,
            hidden=args.selector_hidden,
            depth=3,
            lambda_min_ratio=SELECTOR_RATIO_MIN,
            lambda_max_ratio=SELECTOR_RATIO_MAX,
            initial_lambda_ratio=0.1,
            positivity_beta=args.direct_positivity_beta,
            direct_head_init_std=args.direct_head_init_std,
        )
        calibration_holder[0] = selector_cpu.fit_penalty_parameter_cache(
            cache.relu_b, cache.relu_weights
        )
        if "selector_state_dict" in warm_checkpoint:
            calibration_holder[0]["shared_v10_migration"] = (
                migrate_shared_v10_selector_state(
                    selector_cpu, warm_checkpoint["selector_state_dict"]
                )
            )
        selector_holder[0] = selector_cpu.state_dict()
    dist.broadcast_object_list(selector_holder, src=0)
    dist.broadcast_object_list(calibration_holder, src=0)
    torch.manual_seed(args.seed + 17)
    selector = DirectPenaltyMLPQuerySelectorV11(
        ds_feat_dim=9,
        t_feat_dim=32 * 33 // 2,
        krylov_dim=32,
        shape_dim=196,
        path_feat_dim=SELECTOR_PATH_FEATURES,
        hidden=args.selector_hidden,
        depth=3,
        lambda_min_ratio=SELECTOR_RATIO_MIN,
        lambda_max_ratio=SELECTOR_RATIO_MAX,
        initial_lambda_ratio=0.1,
        positivity_beta=args.direct_positivity_beta,
        direct_head_init_std=args.direct_head_init_std,
    )
    selector.load_state_dict(selector_holder[0], strict=True)
    selector.to(device)

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=True,
        find_unused_parameters=True,
        init_sync=True,
    )
    selector_train = DDP(
        selector,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=True,
        find_unused_parameters=True,
        init_sync=True,
    )
    core = model.module
    selector_core = selector_train.module
    beta_optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.beta_lr, weight_decay=1e-4, betas=(0.9, 0.95)
    )
    direct_head_parameters = list(
        selector_core.penalty_parameter_head.parameters()
    )
    direct_head_parameter_ids = {id(parameter) for parameter in direct_head_parameters}
    selector_other_parameters = [
        parameter
        for parameter in selector_train.parameters()
        if id(parameter) not in direct_head_parameter_ids
    ]
    if not direct_head_parameters or not selector_other_parameters:
        raise TrainerError("V11 selector optimizer parameter split is empty")
    selector_optimizer = torch.optim.AdamW(
        [
            {
                "params": selector_other_parameters,
                "lr": args.selector_lr,
                "name": "selector_context_lambda_trunks",
            },
            {
                "params": direct_head_parameters,
                "lr": args.direct_head_lr,
                "name": "direct96_penalty_head",
            },
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
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    beta_scheduler = torch.optim.lr_scheduler.LambdaLR(
        beta_optimizer, multiplier
    )
    exact_training_arguments = {
        "data_root": str(args.data_root.resolve()),
        "part": args.part,
        "out_dir": str(args.out_dir.resolve()),
        "steps": int(args.steps),
        "beta_lr": float(args.beta_lr),
        "selector_lr": float(args.selector_lr),
        "direct_head_lr": float(args.direct_head_lr),
        "warmup_steps": int(args.warmup_steps),
        "batch_per_rank": int(args.batch_per_rank),
        "grad_accum_steps": int(args.grad_accum_steps),
        "selector_grad_accum_steps": int(args.selector_grad_accum_steps),
        "selector_every": int(args.selector_every),
        "selector_hidden": int(args.selector_hidden),
        "direct_positivity_beta": float(args.direct_positivity_beta),
        "direct_head_init_std": float(args.direct_head_init_std),
        "path_chunk": int(args.path_chunk),
        "holdout_files": int(args.holdout_files),
        "cache_files": int(args.cache_files),
        "checkpoint_every": int(args.checkpoint_every),
        "seed": int(args.seed),
        "max_abs_feature_corr": float(args.max_abs_feature_corr),
    }
    resume_checkpoint = None
    rank_state = None
    completed_steps = 0
    if args.resume is not None:
        if file_sha256(args.resume) != args.resume_sha256:
            raise TrainerError("resume checkpoint SHA256 mismatch")
        resume_checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        resume_metadata = resume_checkpoint.get("metadata", {})
        required_resume = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "trainer_version": TRAINER_VERSION,
            "model_version": MODEL_VERSION,
            "basis_version": BASIS_VERSION,
            "penalty_contract_version": PENALTY_CONTRACT_VERSION,
            "v11_direct_encoding_version": V11_DIRECT_PENALTY_ENCODING_VERSION,
            "lambda_low_min_contract": LAMBDA_LOW_MIN_CONTRACT,
            "selector_effective_ratio_min": SELECTOR_RATIO_MIN,
            "selector_effective_ratio_max": SELECTOR_RATIO_MAX,
            "selector_objective": SELECTOR_OBJECTIVE,
            "selector_objective_has_lambda_penalty_term": False,
            "warm_start_sha256": args.warm_start_sha256,
            "data_inventory_sha256": inventory["inventory_sha256"],
            "world_size": world,
            "exact_training_arguments": exact_training_arguments,
        }
        for key, expected in required_resume.items():
            if resume_metadata.get(key) != expected:
                raise TrainerError(f"resume contract mismatch for {key}")
        core.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
        selector_core.load_state_dict(
            resume_checkpoint["selector_state_dict"], strict=True
        )
        beta_optimizer.load_state_dict(
            resume_checkpoint["beta_optimizer_state_dict"]
        )
        selector_optimizer.load_state_dict(
            resume_checkpoint["selector_optimizer_state_dict"]
        )
        beta_scheduler.load_state_dict(
            resume_checkpoint["beta_scheduler_state_dict"]
        )
        completed_steps = int(resume_checkpoint["completed_steps"])
        rank_states = resume_checkpoint["rank_states"]
        if sorted(int(item["rank"]) for item in rank_states) != list(range(world)):
            raise TrainerError("resume checkpoint rank-state inventory drift")
        rank_state = next(item for item in rank_states if int(item["rank"]) == rank)
        cache.load_state_dict(rank_state["penalty_cache"])
        sampler.load_state_dict(rank_state["sampler"])
        restore_rng(rank_state["rng"])
    stream = PartStream(
        train_files,
        seed=args.seed + 10_000,
        rank=rank,
        world=world,
        cache_files=args.cache_files,
        state=None if rank_state is None else rank_state["stream"],
    )
    split_generator = torch.Generator(device="cpu").manual_seed(
        args.seed + 3_000_017 * rank
    )
    if rank_state is not None:
        split_generator.set_state(rank_state["split_generator_state"])
    knots = cache.relu_knots.to(device)
    fixed_indices = torch.tensor([0, 49_999, 99_999], dtype=torch.long)
    fixed_shape = cache.shape_features.index_select(0, fixed_indices).to(device)
    fixed_d0 = cache.d0.index_select(0, fixed_indices).to(device)
    path_ratios = torch.exp(
        torch.linspace(
            math.log(SELECTOR_RATIO_MIN),
            math.log(SELECTOR_RATIO_MAX),
            50,
            device=device,
        )
    )

    if is_main:
        if resume_checkpoint is None:
            args.out_dir.mkdir(parents=True, exist_ok=False)
        elif not args.out_dir.is_dir() or args.resume.resolve().parent != args.out_dir.resolve():
            raise TrainerError(
                "resume checkpoint must remain inside its existing output directory"
            )
    dist.barrier()
    events_path = args.out_dir / "events.jsonl"

    def emit(event: Mapping[str, Any]) -> None:
        if not is_main:
            return
        record = {"timestamp": time.time(), **dict(event)}
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        with events_path.open("a") as handle:
            handle.write(line + "\n")

    source_hashes = {
        "trainer": file_sha256(Path(__file__)),
        "model": file_sha256(Path(__file__).with_name("model.py")),
        "utils": file_sha256(Path(__file__).with_name("utils.py")),
        "selector_v10_base": file_sha256(Path(__file__).with_name("selector_v10.py")),
        "selector_v11_direct": file_sha256(
            Path(__file__).with_name("selector_v11_direct.py")
        ),
        "config": file_sha256(args.config),
        "penalty_cache": file_sha256(args.penalty_cache),
        "warm_start": args.warm_start_sha256,
        "root_dependencies": {
            name: file_sha256(path)
            for name, path in sorted(ROOT_DEPENDENCY_PATHS.items())
        },
    }
    if resume_checkpoint is not None:
        if resume_checkpoint["metadata"].get("source_hashes") != source_hashes:
            raise TrainerError("resume source hash closure changed")
    emit(
        {
            "kind": "start" if resume_checkpoint is None else "resume",
            "trainer_version": TRAINER_VERSION,
            "checkpoint_version": CHECKPOINT_VERSION,
            "model_version": MODEL_VERSION,
            "basis_version": BASIS_VERSION,
            "penalty_contract_version": PENALTY_CONTRACT_VERSION,
            "local_isis_gradient_contract": LOCAL_ISIS_GRADIENT_CONTRACT,
            "v11_direct_encoding_version": V11_DIRECT_PENALTY_ENCODING_VERSION,
            "selector_objective": SELECTOR_OBJECTIVE,
            "selector_objective_has_lambda_penalty_term": False,
            "lambda_low_min_contract": LAMBDA_LOW_MIN_CONTRACT,
            "selector_effective_ratio_min": SELECTOR_RATIO_MIN,
            "selector_effective_ratio_max": SELECTOR_RATIO_MAX,
            "warm_start_model_hash": warm_model_hash,
            "warm_start_checkpoint": str(args.warm_start.resolve()),
            "world_size": world,
            "global_beta_task_batch": world * args.batch_per_rank * args.grad_accum_steps,
            "selector_microbatch_per_rank": args.batch_per_rank,
            "selector_gradient_accumulation_steps": (
                args.selector_grad_accum_steps
            ),
            "global_selector_task_batch": (
                world * args.batch_per_rank * args.selector_grad_accum_steps
            ),
            "calibration": calibration_holder[0],
            "source_hashes": source_hashes,
            "exact_training_arguments": exact_training_arguments,
            "completed_steps": completed_steps,
            "resume_checkpoint": None if args.resume is None else str(args.resume.resolve()),
        }
    )

    if resume_checkpoint is None:
        initial_snapshot_inputs = stream.next_batch(args.batch_per_rank)
        initial_split = split_and_standardize(
            *initial_snapshot_inputs, generator=split_generator, device=device
        )
        reference_X_train = initial_split[0]
        reference_y_train = initial_split[1]
        if is_main:
            initial_snapshot = selector_snapshot(
                selector_core,
                core,
                reference_X_train,
                reference_y_train,
                fixed_shape,
                fixed_d0,
                path_ratios,
                args.path_chunk,
            )
        else:
            initial_snapshot = None
    else:
        reference_X_train = resume_checkpoint[
            "selector_reference_X_train"
        ].to(device)
        reference_y_train = resume_checkpoint[
            "selector_reference_y_train"
        ].to(device)
        initial_snapshot = (
            resume_checkpoint["selector_reference_snapshot"] if is_main else None
        )
    dist.barrier()
    beta_hash_before = assert_ddp_synced(core, world, "post_init_beta")
    selector_hash_before = assert_ddp_synced(
        selector_core, world, "post_init_selector"
    )
    compute_seconds_accumulator = 0.0
    compute_timed_steps = 0
    while completed_steps < target_step:
        step_compute_started = time.perf_counter()
        beta_optimizer.zero_grad(set_to_none=True)
        beta_metrics = {key: 0.0 for key in (
            "loss", "data_half", "mse", "penalty", "objective", "f_zero",
            "kkt_top32", "nnz",
        )}
        sampled_lambda_min = math.inf
        for accumulation in range(args.grad_accum_steps):
            Xs, ys = stream.next_batch(args.batch_per_rank)
            payload = make_query_payload(Xs, ys, cache, sampler, device)
            sampled_lambda_min = min(
                sampled_lambda_min, float(payload["lambdas"].min().detach())
            )
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
                beta_metrics[key] += float(result[key].detach()) / float(
                    args.grad_accum_steps
                )
        beta_group_gradients = {
            "shape": module_grad_norm(core.cond_embed.penalty_shape_encoder),
            "query": module_grad_norm(core.query_mixer),
            "feature": module_grad_norm(core.feature_head),
            "local": module_grad_norm(core.v46_local_mixer),
        }
        beta_total_gradient = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
        )
        if not math.isfinite(beta_total_gradient) or beta_total_gradient <= 0.0:
            raise TrainerError("beta gradient is nonfinite or zero")
        if not all(math.isfinite(value) for value in beta_group_gradients.values()):
            raise TrainerError("nonfinite beta group gradient")
        beta_optimizer.step()
        beta_scheduler.step()

        selector_metrics: dict[str, Any] = {}
        if (completed_steps + 1) % args.selector_every == 0:
            selector_optimizer.zero_grad(set_to_none=True)
            was_training = core.training
            core.eval()
            for parameter in core.parameters():
                parameter.grad = None
                parameter.requires_grad_(False)
            selector_losses: list[torch.Tensor] = []
            selector_task_mses: list[torch.Tensor] = []
            selector_effective_ratios: list[torch.Tensor] = []
            selector_raw_ratios: list[torch.Tensor] = []
            selector_effective_errors: list[torch.Tensor] = []
            selector_betas: list[torch.Tensor] = []
            selector_penalty_rhos: list[torch.Tensor] = []
            selector_penalty_drhos: list[torch.Tensor] = []
            selector_penalty_min_drhos: list[torch.Tensor] = []
            selector_penalty_rho6_errors: list[torch.Tensor] = []
            selector_latent_gradients: list[torch.Tensor] = []
            selector_parameter_gradients: list[torch.Tensor] = []
            selector_native196_gradients: list[torch.Tensor] = []
            selector_d0_gradients: list[torch.Tensor] = []
            selector_lambda_output_gradients: list[torch.Tensor] = []
            try:
                for selector_accumulation in range(
                    args.selector_grad_accum_steps
                ):
                    Xs, ys = stream.next_batch(args.batch_per_rank)
                    X_train, y_train, X_test, y_test = split_and_standardize(
                        Xs, ys, generator=split_generator, device=device
                    )
                    with torch.no_grad():
                        encoded = core.encode_dataset(
                            core.precompute(X_train, y_train)
                        )
                        dataset_features, krylov_features = summarize_encoded(
                            encoded, 32 * 33 // 2
                        )
                        path = selector_path_features(
                            core,
                            encoded,
                            X_train,
                            y_train,
                            fixed_shape,
                            fixed_d0,
                            path_ratios,
                            args.path_chunk,
                        )
                    selector_sync = (
                        selector_train.no_sync()
                        if selector_accumulation
                        < args.selector_grad_accum_steps - 1
                        else contextlib.nullcontext()
                    )
                    with selector_sync:
                        query = selector_train(
                            dataset_features,
                            krylov_features,
                            encoded["lambda_max"].float(),
                            path,
                            U=encoded.get("U"),
                            y_context=y_train,
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
                        beta = selector_beta_with_inactive_ste(aux)[:, 0].float()
                        prediction = torch.einsum(
                            "bnp,bp->bn", X_test.float(), beta
                        )
                        task_mse = (
                            prediction - y_test.float()
                        ).square().mean(dim=-1)
                        selector_loss = torch.log1p(0.5 * task_mse).mean()
                        if not bool(torch.isfinite(selector_loss)):
                            raise TrainerError("nonfinite selector loss")
                        latent_raw_internal = getattr(
                            selector_core.penalty_parameter_head,
                            "_last_latent_raw",
                            None,
                        )
                        parameter_proposal_internal = getattr(
                            selector_core.penalty_parameter_head,
                            "_last_parameter_proposal",
                            None,
                        )
                        (
                            selector_loss
                            / float(args.selector_grad_accum_steps)
                        ).backward()
                    latent_gradient = (
                        latent_raw_internal.grad
                        if latent_raw_internal is not None
                        else None
                    )
                    if latent_gradient is None:
                        raise TrainerError(
                            "V11 internal direct-parameter gradient is missing"
                        )
                    parameter_gradient = (
                        parameter_proposal_internal.grad
                        if parameter_proposal_internal is not None
                        else None
                    )
                    if parameter_gradient is None:
                        raise TrainerError(
                            "V11 actual MLP-parameter proposal gradient is missing"
                        )
                    for output_name, destination in (
                        (
                            "penalty_shape_features",
                            selector_native196_gradients,
                        ),
                        ("penalty_d0", selector_d0_gradients),
                        ("lambdas", selector_lambda_output_gradients),
                    ):
                        output_gradient = query[output_name].grad
                        if output_gradient is None:
                            raise TrainerError(
                                f"V11 {output_name} output gradient is missing"
                            )
                        destination.append(output_gradient.detach())
                    selector_losses.append(selector_loss.detach())
                    selector_task_mses.append(task_mse.detach())
                    selector_effective_ratios.append(
                        query["lambda_ratio"].detach().reshape(-1)
                    )
                    raw_ratio = (
                        query["lambdas"][:, 0]
                        / encoded["lambda_max"].float()
                    )
                    selector_raw_ratios.append(raw_ratio.detach().reshape(-1))
                    effective_error = (
                        query["lambdas"][:, 0]
                        * query["penalty_d0"][:, 0]
                        / encoded["lambda_max"].float()
                        - query["lambda_ratio"]
                    ).abs()
                    selector_effective_errors.append(
                        effective_error.detach().reshape(-1)
                    )
                    selector_betas.append(beta.detach())
                    selector_penalty_rhos.append(query["penalty_rho"].detach())
                    selector_penalty_drhos.append(query["penalty_drho"].detach())
                    selector_penalty_min_drhos.append(
                        query["penalty_min_drho"].detach().reshape(-1)
                    )
                    selector_penalty_rho6_errors.append(
                        query["penalty_rho6_error"].detach().reshape(-1)
                    )
                    selector_latent_gradients.append(
                        latent_gradient.detach().float()
                    )
                    selector_parameter_gradients.append(
                        parameter_gradient.detach().float()
                    )

                latent_gradient_matrix = torch.cat(
                    [
                        gradient.reshape(-1, gradient.shape[-1])
                        for gradient in selector_latent_gradients
                    ],
                    dim=0,
                )
                if not bool(torch.isfinite(latent_gradient_matrix).all()):
                    raise TrainerError("nonfinite V11 direct-parameter gradient")
                latent_gradient_flat = latent_gradient_matrix.reshape(-1)
                latent_gradient_stats = {
                    "norm": float(latent_gradient_flat.norm()),
                    "rms": float(
                        latent_gradient_flat.square().mean().sqrt()
                    ),
                    "nonzero_fraction": float(
                        (latent_gradient_flat != 0.0).float().mean()
                    ),
                    "coordinate_nonzero_fraction": float(
                        (latent_gradient_matrix.abs().sum(dim=0) != 0.0)
                        .float()
                        .mean()
                    ),
                    "elements": int(latent_gradient_flat.numel()),
                }
                native196_gradient_stats = tensor_gradient_stats(
                    selector_native196_gradients
                )
                parameter_gradient_matrix = torch.cat(
                    [
                        gradient.reshape(-1, gradient.shape[-1])
                        for gradient in selector_parameter_gradients
                    ],
                    dim=0,
                )
                parameter_gradient_flat = parameter_gradient_matrix.reshape(-1)
                parameter_gradient_stats = {
                    "norm": float(parameter_gradient_flat.norm()),
                    "rms": float(
                        parameter_gradient_flat.square().mean().sqrt()
                    ),
                    "nonzero_fraction": float(
                        (parameter_gradient_flat != 0.0).float().mean()
                    ),
                    "coordinate_nonzero_fraction": float(
                        (parameter_gradient_matrix.abs().sum(dim=0) != 0.0)
                        .float()
                        .mean()
                    ),
                    "elements": int(parameter_gradient_flat.numel()),
                }
                d0_gradient_stats = tensor_gradient_stats(selector_d0_gradients)
                lambda_output_gradient_stats = tensor_gradient_stats(
                    selector_lambda_output_gradients
                )
                if (
                    not math.isfinite(float(latent_gradient_stats["norm"]))
                    or float(latent_gradient_stats["norm"]) <= 0.0
                ):
                    raise TrainerError(
                        "V11 direct-parameter gradient is nonfinite or zero"
                    )
                for name, output_stats in (
                    ("direct96_actual_parameter", parameter_gradient_stats),
                    ("native196", native196_gradient_stats),
                    ("d0", d0_gradient_stats),
                    ("lambda", lambda_output_gradient_stats),
                ):
                    if (
                        not math.isfinite(float(output_stats["norm"]))
                        or float(output_stats["norm"]) <= 0.0
                    ):
                        raise TrainerError(
                            f"V11 {name} output gradient is nonfinite or zero"
                        )
                groups = {
                    "penalty_head": grad_stats(
                        selector_core.penalty_parameter_head.parameters()
                    ),
                    "lambda_head": grad_stats(selector_core.lambda_head.parameters()),
                    "penalty_trunk": grad_stats(selector_core.penalty_net.parameters()),
                    "lambda_trunk": grad_stats(selector_core.lambda_net.parameters()),
                }
                for name, stats in groups.items():
                    if not math.isfinite(float(stats["norm"])):
                        raise TrainerError(f"nonfinite selector gradient in {name}")
                if float(groups["penalty_head"]["norm"]) <= 0.0:
                    raise TrainerError("V11 penalty-head gradient is zero")
                if float(groups["lambda_head"]["norm"]) <= 0.0:
                    raise TrainerError("V11 lambda-head gradient is zero")
                # The shared selector intentionally zero-initializes lambda_head.weight to
                # begin at the common effective ratio.  Therefore the first
                # update cannot reach lambda_net, but the changed head weight
                # must open that trunk from the second selector update onward.
                if completed_steps + 1 >= 2:
                    if float(groups["lambda_trunk"]["norm"]) <= 0.0:
                        raise TrainerError(
                            "V11 lambda trunk did not open after its expected "
                            "one-update zero-initialization delay"
                        )
                    if float(groups["penalty_trunk"]["norm"]) <= 0.0:
                        raise TrainerError("V11 penalty trunk gradient is zero")
                if any(parameter.grad is not None for parameter in core.parameters()):
                    raise TrainerError("frozen beta model received selector gradient")
                selector_total_gradient = float(
                    torch.nn.utils.clip_grad_norm_(selector_train.parameters(), 5.0)
                )
                if not math.isfinite(selector_total_gradient):
                    raise TrainerError("nonfinite selector total gradient")
                selector_optimizer.step()
                all_task_mse = torch.cat(selector_task_mses)
                all_effective_ratios = torch.cat(selector_effective_ratios)
                all_raw_ratios = torch.cat(selector_raw_ratios)
                all_effective_errors = torch.cat(selector_effective_errors)
                all_beta = torch.cat(selector_betas, dim=0)
                all_penalty_rho = torch.cat(selector_penalty_rhos, dim=0)
                all_penalty_drho = torch.cat(selector_penalty_drhos, dim=0)
                all_penalty_min_drho = torch.cat(selector_penalty_min_drhos)
                all_penalty_rho6_error = torch.cat(
                    selector_penalty_rho6_errors
                )
                selector_metrics = {
                    "selector_gradient_accumulation_steps": int(
                        args.selector_grad_accum_steps
                    ),
                    "selector_local_task_batch": int(
                        args.batch_per_rank * args.selector_grad_accum_steps
                    ),
                    "selector_objective": float(
                        torch.stack(selector_losses).mean()
                    ),
                    "selector_heldout_mse": float(all_task_mse.mean()),
                    "selector_effective_ratio_min": float(
                        all_effective_ratios.min()
                    ),
                    "selector_effective_ratio_max": float(
                        all_effective_ratios.max()
                    ),
                    "selector_raw_ratio_min": float(all_raw_ratios.min()),
                    "selector_raw_ratio_max": float(all_raw_ratios.max()),
                    "effective_ratio_max_abs_error": float(
                        all_effective_errors.max()
                    ),
                    "zero_beta_fraction": float(
                        (all_beta == 0.0).float().mean()
                    ),
                    "penalty_rho_batch_std_mean": float(
                        all_penalty_rho.std(0, unbiased=False).mean()
                    ),
                    "penalty_drho_batch_std_mean": float(
                        all_penalty_drho.std(0, unbiased=False).mean()
                    ),
                    "penalty_min_drho": float(all_penalty_min_drho.min()),
                    "penalty_rho6_error_max": float(
                        all_penalty_rho6_error.max()
                    ),
                    "penalty_latent_gradient": latent_gradient_stats,
                    "direct96_raw_gradient": latent_gradient_stats,
                    "direct96_parameter_gradient": parameter_gradient_stats,
                    "native196_output_gradient": native196_gradient_stats,
                    "d0_output_gradient": d0_gradient_stats,
                    "lambda_output_gradient": lambda_output_gradient_stats,
                    "selector_group_gradients": groups,
                    "selector_total_gradient_preclip": selector_total_gradient,
                }
            finally:
                for parameter in core.parameters():
                    parameter.requires_grad_(True)
                core.train(was_training)

        completed_steps += 1
        compute_seconds_accumulator += time.perf_counter() - step_compute_started
        compute_timed_steps += 1
        if completed_steps == 1:
            beta_hash_after = assert_ddp_synced(
                core, world, "post_first_beta_update"
            )
            selector_hash_after = assert_ddp_synced(
                selector_core, world, "post_first_selector_update"
            )
            if beta_hash_after == beta_hash_before:
                raise TrainerError("beta model hash did not change after first update")
            if selector_hash_after == selector_hash_before:
                raise TrainerError("V11 hash did not change after first update")
        if completed_steps % args.log_every == 0:
            local_monitor = {
                "rank": rank,
                "beta_metrics": beta_metrics,
                "beta_total_gradient_preclip": beta_total_gradient,
                "beta_group_gradients": beta_group_gradients,
                "sampled_lambda_min": sampled_lambda_min,
                "compute_seconds_interval": compute_seconds_accumulator,
                "compute_timed_steps": compute_timed_steps,
                "average_compute_seconds_per_step": (
                    compute_seconds_accumulator / compute_timed_steps
                ),
                **selector_metrics,
            }
            rank_monitors = [None for _ in range(world)]
            dist.all_gather_object(rank_monitors, local_monitor)
            rank_average_compute_seconds = [
                float(item["average_compute_seconds_per_step"])
                for item in rank_monitors
            ]
            rank_timed_steps = [
                int(item["compute_timed_steps"]) for item in rank_monitors
            ]
            if rank_timed_steps != [compute_timed_steps] * world:
                raise TrainerError("rank compute-timing window drift")
            if not all(
                math.isfinite(value) and value > 0.0
                for value in rank_average_compute_seconds
            ):
                raise TrainerError("nonfinite or nonpositive compute timing")
            emit(
                {
                    "kind": "train_update",
                    "step": completed_steps,
                    "lambda_low_min_contract": LAMBDA_LOW_MIN_CONTRACT,
                    "compute_timed_steps": compute_timed_steps,
                    "average_compute_seconds_per_step": (
                        sum(rank_average_compute_seconds) / world
                    ),
                    "max_rank_average_compute_seconds_per_step": max(
                        rank_average_compute_seconds
                    ),
                    "rank_monitors": rank_monitors,
                }
            )
            compute_seconds_accumulator = 0.0
            compute_timed_steps = 0

        should_save = (
            completed_steps % args.checkpoint_every == 0
            or completed_steps == target_step
        )
        if should_save:
            local_state = {
                "rank": rank,
                "rng": capture_rng(),
                "stream": stream.state_dict(),
                "penalty_cache": cache.state_dict(),
                "sampler": sampler.state_dict(),
                "split_generator_state": split_generator.get_state(),
            }
            rank_states = [None for _ in range(world)]
            dist.all_gather_object(rank_states, local_state)
            if is_main:
                checkpoint_path = args.out_dir / (
                    f"checkpoint_step{completed_steps:06d}.pt"
                )
                payload = {
                    "completed_steps": completed_steps,
                    "model_state_dict": core.state_dict(),
                    "selector_state_dict": selector_core.state_dict(),
                    "beta_optimizer_state_dict": beta_optimizer.state_dict(),
                    "selector_optimizer_state_dict": selector_optimizer.state_dict(),
                    "beta_scheduler_state_dict": beta_scheduler.state_dict(),
                    "rank_states": rank_states,
                    "selector_reference_X_train": reference_X_train.detach().cpu(),
                    "selector_reference_y_train": reference_y_train.detach().cpu(),
                    "selector_reference_snapshot": initial_snapshot,
                    "metadata": {
                        "checkpoint_version": CHECKPOINT_VERSION,
                        "trainer_version": TRAINER_VERSION,
                        "model_version": MODEL_VERSION,
                        "basis_version": BASIS_VERSION,
                        "penalty_contract_version": PENALTY_CONTRACT_VERSION,
                        "v11_direct_encoding_version": (
                            V11_DIRECT_PENALTY_ENCODING_VERSION
                        ),
                        "lambda_low_min_contract": LAMBDA_LOW_MIN_CONTRACT,
                        "selector_effective_ratio_min": SELECTOR_RATIO_MIN,
                        "selector_effective_ratio_max": SELECTOR_RATIO_MAX,
                        "selector_objective": SELECTOR_OBJECTIVE,
                        "selector_objective_has_lambda_penalty_term": False,
                        "warm_start_sha256": args.warm_start_sha256,
                        "source_hashes": source_hashes,
                        "data_inventory_sha256": inventory["inventory_sha256"],
                        "world_size": world,
                        "exact_training_arguments": exact_training_arguments,
                    },
                }
                atomic_torch_save(payload, checkpoint_path)
                emit(
                    {
                        "kind": "checkpoint",
                        "step": completed_steps,
                        "path": str(checkpoint_path),
                        "sha256": file_sha256(checkpoint_path),
                        "model_state_sha256": module_sha256(core),
                        "selector_state_sha256": module_sha256(selector_core),
                    }
                )
            dist.barrier()

    if is_main:
        final_snapshot = selector_snapshot(
            selector_core,
            core,
            reference_X_train,
            reference_y_train,
            fixed_shape,
            fixed_d0,
            path_ratios,
            args.path_chunk,
        )
        initial_rho = torch.tensor(initial_snapshot["rho"])
        final_rho = torch.tensor(final_snapshot["rho"])
        initial_drho = torch.tensor(initial_snapshot["drho"])
        final_drho = torch.tensor(final_snapshot["drho"])
        emit(
            {
                "kind": "selector_output_change",
                "step": completed_steps,
                "rho_l2_mean": float((final_rho - initial_rho).norm(dim=-1).mean()),
                "rho_max_abs": float((final_rho - initial_rho).abs().max()),
                "drho_l2_mean": float(
                    (final_drho - initial_drho).norm(dim=-1).mean()
                ),
                "drho_max_abs": float((final_drho - initial_drho).abs().max()),
                "initial": initial_snapshot,
                "final": final_snapshot,
            }
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
