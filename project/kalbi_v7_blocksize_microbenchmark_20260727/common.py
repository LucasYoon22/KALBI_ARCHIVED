"""Shared immutable inputs and model-loading helpers for the benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kalbi_v7_fullp_blockprox3_v23_k32s32_p100k_effbatch32.canonical_penalty_cache import (
    CanonicalPenaltyCache,
)
from kalbi_v7_fullp_blockprox3_v23_k32s32_p100k_effbatch32.model import (
    build_model,
)
from train_fpk import _load_yaml

from .metrics import tensor_sha256


PROJECT = Path("/users/minsuk/fm-lm-lambda")
PACKAGE = PROJECT / "kalbi_v7_blocksize_microbenchmark_20260727"
OUTPUT = (
    PROJECT
    / "evaluations"
    / "kalbi_v7_blocksize_microbenchmark_20260727"
)
CONFIG = PROJECT / "fpk_v33_kkt0.yaml"
CHECKPOINT = (
    PROJECT
    / "runs"
    / "kalbi_v23_betaonly_supportmix_v13_target3080_effbatch32_to60k_20260726"
    / "checkpoint_step042500.pt"
)
CHECKPOINT_SHA256 = (
    "ecdd65c5d36e67fd45848870b550d72f885bf117cfc6294fbcd633fa356e04d5"
)
METHODS = ("pg10", "block4096", "block256", "adaptive")
METHOD_LABELS = {
    "pg10": "PG10",
    "block4096": "Block-3 / B=4096",
    "block256": "Block-3 / B=256",
    "adaptive": "Block-3 / adaptive",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_model_and_checkpoint(
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if sha256_file(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("pinned beta checkpoint SHA256 changed")
    checkpoint = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=False
    )
    model = build_model(_load_yaml(str(CONFIG)), device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def load_query_payload(device: torch.device) -> dict[str, torch.Tensor]:
    bank = json.loads((OUTPUT / "query_bank.json").read_text())
    cache_path = Path(bank["penalty_cache_path"])
    if sha256_file(cache_path) != bank["penalty_cache_sha256"]:
        raise RuntimeError("canonical penalty cache SHA256 changed")
    cache = CanonicalPenaltyCache(cache_path, seed=int(bank["seed"]))
    metadata = cache.load(expected_count=int(bank["penalty_cache_count"]))
    if metadata["sha256"] != bank["penalty_cache_sha256"]:
        raise RuntimeError("canonical cache runtime audit failed")
    indices = np.asarray(
        [item["penalty_cache_index"] for item in bank["queries"]],
        dtype=np.int64,
    )
    encoded = cache.encoded_indices(indices, device)
    encoded["lambdas"] = torch.tensor(
        [item["raw_lambda"] for item in bank["queries"]],
        dtype=torch.float32,
        device=device,
    )
    encoded["indices"] = torch.from_numpy(indices).to(device)
    return {
        key: (
            value.unsqueeze(0)
            if torch.is_tensor(value)
            and key not in {"knots"}
            and value.ndim >= 1
            else value
        )
        for key, value in encoded.items()
    }


def standardized_task(
    task_entry: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    path = Path(task_entry["frozen_path"])
    if sha256_file(path) != task_entry["task_sha256"]:
        raise RuntimeError(f"frozen task SHA256 mismatch: {path}")
    with np.load(path, allow_pickle=False) as payload:
        x = np.asarray(payload["X"][0], dtype=np.float32)
        y = np.asarray(payload["y"][0], dtype=np.float32)
    expected = (int(task_entry["n"]), int(task_entry["p"]))
    if x.shape != expected or y.shape != (expected[0],):
        raise RuntimeError(
            f"task shape mismatch: X={x.shape}, y={y.shape}, expected={expected}"
        )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError("frozen task contains nonfinite values")
    x_mean = x.mean(0, keepdims=True)
    x_scale = x.std(0, keepdims=True) + 1e-8
    y_mean = float(y.mean())
    y_scale = float(y.std()) + 1e-8
    x = np.ascontiguousarray((x - x_mean) / x_scale, dtype=np.float32)
    y = np.ascontiguousarray((y - y_mean) / y_scale, dtype=np.float32)
    x_tensor = torch.from_numpy(x).unsqueeze(0)
    y_tensor = torch.from_numpy(y).unsqueeze(0)
    preprocessing = {
        "contract": "whole frozen training task standardized independently",
        "x_mean_sha256": tensor_sha256(torch.from_numpy(x_mean)),
        "x_scale_sha256": tensor_sha256(torch.from_numpy(x_scale)),
        "y_mean": y_mean,
        "y_scale": y_scale,
        "x_standardized_sha256": tensor_sha256(x_tensor),
        "y_standardized_sha256": tensor_sha256(y_tensor),
    }
    return x_tensor, y_tensor, preprocessing


def lambda_maximum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (
        torch.einsum("bnp,bn->bp", x.float(), y.float())
        .abs()
        .amax(-1)
        .div(float(x.shape[1]))
        .unsqueeze(1)
    )


def load_precomputed(
    task_index: int, device: torch.device
) -> dict[str, Any]:
    path = OUTPUT / "precomputed" / f"task_{task_index:02d}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in (
        "x",
        "y",
        "lambda_max",
        "lambdas",
        "penalty_shape_features",
        "penalty_d0",
        "rho",
        "drho",
        "feature_mask",
        "beta_initial",
        "prethreshold",
        "log_threshold_multiplier",
        "gram_diagonal",
    ):
        payload[key] = payload[key].to(device)
    for name, expected in payload["tensor_hashes"].items():
        observed = tensor_sha256(payload[name])
        if observed != expected:
            raise RuntimeError(
                f"precomputed tensor hash mismatch for {name}: "
                f"{observed} != {expected}"
            )
    return payload


__all__ = [
    "CHECKPOINT",
    "CHECKPOINT_SHA256",
    "CONFIG",
    "METHODS",
    "METHOD_LABELS",
    "OUTPUT",
    "PACKAGE",
    "PROJECT",
    "atomic_json_dump",
    "atomic_torch_save",
    "lambda_maximum",
    "load_model_and_checkpoint",
    "load_precomputed",
    "load_query_payload",
    "sha256_file",
    "standardized_task",
]
