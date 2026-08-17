#!/usr/bin/env python3
"""Compact 300K Direct96 cache and runtime beta-query adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .canonical_penalty import (
    CONTRACT_VERSION,
    DIRECT_PARAMETER_DIM,
    build_penalty,
)


CACHE_VERSION = "v12_canonical_direct96_cache_300k_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CanonicalPenaltyCache:
    def __init__(self, path: Path, seed: int) -> None:
        self.path = path.resolve()
        self.rng = np.random.default_rng(int(seed))

    def load(self, *, expected_count: int = 300_000) -> dict[str, Any]:
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        if payload.get("cache_version") != CACHE_VERSION:
            raise RuntimeError("canonical penalty-cache version mismatch")
        if payload.get("canonical_contract_version") != CONTRACT_VERSION:
            raise RuntimeError("canonical penalty contract mismatch")
        parameters = payload.get("direct_parameters")
        if not torch.is_tensor(parameters):
            raise RuntimeError("canonical cache lacks Direct96 tensor")
        parameters = parameters.contiguous().float()
        if parameters.shape != (int(expected_count), DIRECT_PARAMETER_DIM):
            raise RuntimeError(
                f"canonical cache shape {tuple(parameters.shape)} != "
                f"({expected_count},{DIRECT_PARAMETER_DIM})"
            )
        if not bool(torch.isfinite(parameters).all().item()):
            raise RuntimeError("canonical cache contains nonfinite Direct96 values")
        source_hashes = payload.get("source_hashes")
        if not isinstance(source_hashes, Mapping):
            raise RuntimeError("canonical cache lacks its exact source-hash closure")
        project = Path(__file__).resolve().parent.parent
        expected_sources = {
            "generator": Path(__file__).with_name(
                "generate_canonical_penalty_cache.py"
            ),
            "canonical_builder": Path(__file__).with_name("canonical_penalty.py"),
            "penalty_sampler_v9": project / "penalty_sampler_v9.py",
            "penalty_relu_converter": project / "penalty_relu_mlp_pwl_v33.py",
        }
        for label, source_path in expected_sources.items():
            if not source_path.is_file():
                raise RuntimeError(f"canonical cache source is missing: {source_path}")
            actual = file_sha256(source_path)
            if source_hashes.get(label) != actual:
                raise RuntimeError(
                    f"canonical cache source hash mismatch for {label}: "
                    f"{source_hashes.get(label)} != {actual}"
                )
        self.direct_parameters = parameters
        self.pool_size = int(parameters.shape[0])
        self.metadata = {
            key: value for key, value in payload.items() if key != "direct_parameters"
        }
        return {
            "cache_version": CACHE_VERSION,
            "canonical_contract_version": CONTRACT_VERSION,
            "count": self.pool_size,
            "sha256": file_sha256(self.path),
            "source_hashes": dict(source_hashes),
        }

    def encoded_indices(
        self, indices: np.ndarray | torch.Tensor, device: torch.device
    ) -> dict[str, torch.Tensor]:
        index = torch.as_tensor(indices, dtype=torch.long)
        direct = self.direct_parameters.index_select(0, index).to(device)
        return build_penalty(direct)

    def sample_encoded(
        self, count: int, device: torch.device
    ) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
        indices = self.rng.integers(0, self.pool_size, size=int(count), dtype=np.int64)
        return indices, self.encoded_indices(indices, device)

    def state_dict(self) -> dict[str, Any]:
        return {
            "cache_version": CACHE_VERSION,
            "path": str(self.path),
            "sha256": file_sha256(self.path),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("cache_version") != CACHE_VERSION:
            raise RuntimeError("canonical penalty-cache resume version mismatch")
        if state.get("path") != str(self.path):
            raise RuntimeError("canonical penalty-cache path changed on resume")
        if state.get("sha256") != file_sha256(self.path):
            raise RuntimeError("canonical penalty-cache bytes changed on resume")
        self.rng.bit_generator.state = state["rng_state"]
