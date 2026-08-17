from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

# `model_fpk` must be importable from CWD or PYTHONPATH.
from model_fpk import (
    FpkConfig,
    make_fpk_model,
    PenaltySampler,
    PenaltySamplerConfig,
    SampledBatch,
    QueryLossInputs,
    compute_batch_loss,
    BatchLossOutputs,
    LASSO_FAMILY, RIDGE_FAMILY, ENET_FAMILY, MCP_FAMILY, SCAD_FAMILY,
    PID_LASSO, PID_RIDGE, PID_ENET, PID_MCP, PID_SCAD,
    W_OBJ, W_KKT, W_PROX, W_DELTA,
)

# Optional helpers from model_fpk. Different model_fpk builds expose
# slightly different surface areas; we degrade gracefully if a helper
# is missing. The diagnostic / eval code below has a fallback path for
# every name imported here.
try:
    from model_fpk import center_xy as _fpk_center_xy  # type: ignore
except ImportError:
    _fpk_center_xy = None

try:
    from model_fpk import d0_from_segments as _fpk_d0_from_segments  # type: ignore
except ImportError:
    _fpk_d0_from_segments = None

try:
    from model_fpk import count_parameters as _fpk_count_parameters  # type: ignore
except ImportError:
    _fpk_count_parameters = None


log = logging.getLogger("train_fpk")


# =============================================================================
# Config loading
# =============================================================================

def _load_yaml(path: str) -> Dict[str, Any]:
    """Tiny YAML loader. Prefer PyYAML; fall back to a minimal hand parser
    so the smoke path doesn't require pyyaml on a fresh container."""
    try:
        import yaml  # type: ignore
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except ImportError:
        return _minimal_yaml_load(path)


def _minimal_yaml_load(path: str) -> Dict[str, Any]:
    """Bare-bones YAML parser for the fpk_v2.yaml shape.

    Supports: ``key: value`` (scalar | list ``[a, b, c]``), nested mapping
    via indent, integers / floats / booleans / quoted-strings, and ``#``
    comments. No anchors / multi-line / blocks. Enough for our config.
    """
    def _parse_scalar(s: str) -> Any:
        s = s.strip()
        if not s:
            return None
        if s.lower() in ("true", "yes"):
            return True
        if s.lower() in ("false", "no"):
            return False
        if s.lower() in ("null", "none", "~"):
            return None
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            if not inner:
                return []
            return [_parse_scalar(x) for x in _split_top_commas(inner)]
        if s.startswith('"') and s.endswith('"'):
            return s[1:-1]
        if s.startswith("'") and s.endswith("'"):
            return s[1:-1]
        # number?
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            pass
        return s  # bare string

    def _split_top_commas(s: str) -> List[str]:
        depth = 0
        out: List[str] = []
        cur: List[str] = []
        for ch in s:
            if ch in "[(":
                depth += 1
            elif ch in "])":
                depth -= 1
            if ch == "," and depth == 0:
                out.append("".join(cur))
                cur = []
            else:
                cur.append(ch)
        if cur:
            out.append("".join(cur))
        return [x.strip() for x in out if x.strip()]

    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]
    with open(path, "r") as f:
        for raw_line in f:
            # Strip comments, but only `#` not preceded by an open quote.
            line = raw_line.rstrip("\n")
            if "#" in line:
                in_quote = False
                for i, ch in enumerate(line):
                    if ch in ('"', "'"):
                        in_quote = not in_quote
                    elif ch == "#" and not in_quote:
                        line = line[:i].rstrip()
                        break
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            key_part, _, val_part = line.lstrip(" ").partition(":")
            key = key_part.strip()
            val_str = val_part.strip()
            # Unwind stack
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if val_str == "":
                # Begin a nested mapping
                new_map: Dict[str, Any] = {}
                parent[key] = new_map
                stack.append((indent, new_map))
            else:
                parent[key] = _parse_scalar(val_str)
    return root


# =============================================================================
# DDP helpers (no-op on single-process)
# =============================================================================

def get_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def get_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def is_main_rank() -> bool:
    return get_rank() == 0


def setup_ddp_if_needed() -> None:
    if "LOCAL_RANK" in os.environ and not torch.distributed.is_initialized():
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters. Use model_fpk.count_parameters if exposed."""
    if _fpk_count_parameters is not None:
        try:
            return int(_fpk_count_parameters(model))
        except Exception:
            pass
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# =============================================================================
# Data loaders (ported from train_unified_krylov)
# =============================================================================
# The Krylov data loader produces (B, N, P) GPU tensors directly. FPK's
# train_step still works on per-task CPU numpy arrays (list[np.ndarray]),
# because PenaltySampler / _compute_lambda_max / _build_X_y_tensors assume
# that shape contract. The classes below therefore expose a numpy-side
# adapter `next_numpy_batch(step)` that yields the FPK contract:
#     Xcs : list[B] of (N, P) float64 CENTERED matrices
#     ycs : list[B] of (N,)   float64 CENTERED responses
#     task_ids : list[B] of stable string ids
#     info : dict with shard / loader metadata for JSONL logging
# Centering happens here so `_compute_lambda_max` and the model both see
# a centered problem regardless of the raw .npz contents.


def _center_np(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Subtract column / response mean. The FPK loss is centered, so we
    center here once and the rest of the pipeline can assume centered."""
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    return Xc, yc


def _load_npz_panel(
    path: Path,
    *,
    max_data_tasks: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single .npz shard into raw (B, N, P) / (B, N) numpy arrays.

    Accepts the same key conventions as the Krylov loader:
        ("X", "y"), ("x", "y"), ("X_train", "y_train").
    Returns float64 numpy arrays. Centering is NOT applied here; that is
    done per-batch in `_to_centered_panel` so that shard-cycling caches
    can keep the raw arrays.
    """
    if path.suffix.lower() != ".npz":
        raise ValueError(f"data path currently supports .npz only, got {path}")
    data = np.load(path)
    keys: Optional[Tuple[str, str]] = None
    for xk, yk in (("X", "y"), ("x", "y"), ("X_train", "y_train")):
        if xk in data and yk in data:
            keys = (xk, yk)
            break
    if keys is None:
        raise ValueError(
            f"npz at {path} must contain X/y, x/y, or X_train/y_train arrays"
        )

    X_np = np.asarray(data[keys[0]], dtype=np.float64)
    y_np = np.asarray(data[keys[1]], dtype=np.float64)
    if X_np.ndim == 2:
        X_np = X_np[None, ...]
    if y_np.ndim == 1:
        y_np = y_np[None, :]
    if y_np.ndim == 3 and y_np.shape[-1] == 1:
        y_np = y_np[..., 0]
    if X_np.ndim != 3 or y_np.ndim != 2:
        raise ValueError(
            f"npz {path}: expected X=(B,N,P) or (N,P), y=(B,N) or (N,); "
            f"got X={X_np.shape}, y={y_np.shape}"
        )
    if X_np.shape[:2] != y_np.shape:
        raise ValueError(
            f"npz {path}: X/y shape mismatch X={X_np.shape}, y={y_np.shape}"
        )
    if max_data_tasks is not None:
        X_np = X_np[: int(max_data_tasks)]
        y_np = y_np[: int(max_data_tasks)]
    if X_np.shape[0] == 0:
        raise ValueError(f"npz {path}: zero tasks after max_data_tasks slicing")
    return X_np, y_np


def _to_centered_panel(
    X_np: np.ndarray,
    y_np: np.ndarray,
    task_id_prefix: str,
    task_indices: Sequence[int],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """Slice raw panel by `task_indices`, center each (X, y) pair, and
    return the FPK loader contract."""
    Xcs: List[np.ndarray] = []
    ycs: List[np.ndarray] = []
    task_ids: List[str] = []
    for i in task_indices:
        Xc, yc = _center_np(X_np[i], y_np[i])
        Xcs.append(Xc)
        ycs.append(yc)
        task_ids.append(f"{task_id_prefix}_t{int(i)}")
    return Xcs, ycs, task_ids


@dataclass(frozen=True)
class ShardInfo:
    path: Path
    num_tasks: int
    N: int
    P: int
    has_lambda_grid: bool = False


def _inspect_npz_shard(path: Path) -> Optional[ShardInfo]:
    try:
        with np.load(path, allow_pickle=False) as data:
            keys: Optional[Tuple[str, str]] = None
            for xk, yk in (("X", "y"), ("x", "y"), ("X_train", "y_train")):
                if xk in data and yk in data:
                    keys = (xk, yk)
                    break
            if keys is None:
                return None
            X_shape = tuple(data[keys[0]].shape)
            y_shape = tuple(data[keys[1]].shape)
            has_lambda_grid = "lambda_grid" in data
    except Exception:
        return None

    if len(X_shape) == 2:
        num_tasks, N, P = 1, int(X_shape[0]), int(X_shape[1])
    elif len(X_shape) == 3:
        num_tasks, N, P = int(X_shape[0]), int(X_shape[1]), int(X_shape[2])
    else:
        return None

    valid_y = (
        (len(y_shape) == 1 and num_tasks == 1 and int(y_shape[0]) == N)
        or (len(y_shape) == 2 and int(y_shape[0]) == num_tasks and int(y_shape[1]) == N)
        or (
            len(y_shape) == 3
            and int(y_shape[0]) == num_tasks
            and int(y_shape[1]) == N
            and int(y_shape[2]) == 1
        )
    )
    if not valid_y or num_tasks <= 0 or N <= 0 or P <= 0:
        return None
    return ShardInfo(
        path=Path(path), num_tasks=num_tasks, N=N, P=P,
        has_lambda_grid=bool(has_lambda_grid),
    )


def _manifest_candidate_paths(root: Path) -> List[Path]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        if is_main_rank():
            log.warning("failed to parse manifest %s: %s", manifest_path, exc)
        return []
    entries = manifest.get("files", [])
    if not isinstance(entries, list):
        return []
    paths: List[Path] = []
    for entry in entries:
        rel: Optional[str] = None
        if isinstance(entry, str):
            rel = entry
        elif isinstance(entry, dict):
            for key in ("path", "file", "filename", "name"):
                val = entry.get(key)
                if isinstance(val, str):
                    rel = val
                    break
        if not rel:
            continue
        p = Path(rel)
        if not p.is_absolute():
            p = root / p
        if p.exists() and p.suffix.lower() == ".npz":
            paths.append(p)
    return paths


def _is_obvious_non_data_path(path: Path) -> bool:
    bad_parts = {"_logs", "_qc", "logs", "qc", "__pycache__"}
    return any(part in bad_parts for part in path.parts)


def discover_npz_shards(
    path: Path,
    *,
    shard_glob: str = "**/*.npz",
    max_shards: Optional[int] = None,
    max_data_tasks: Optional[int] = None,
) -> List[ShardInfo]:
    """Discover sync21-style `.npz` shards and inspect X/y shapes.

    `max_data_tasks` is a global cap over all discovered shards. It
    truncates the effective task count in the final shard rather than
    being applied to every shard independently.
    """
    path = Path(path)
    manifest_paths: List[Path] = []
    if path.is_file():
        candidates: List[Path] = [path]
    elif path.is_dir():
        manifest_paths = _manifest_candidate_paths(path)
        if manifest_paths:
            candidates = manifest_paths
        else:
            candidates = sorted(
                path.glob(shard_glob) if shard_glob else path.rglob("*.npz")
            )
    else:
        raise FileNotFoundError(f"data path does not exist: {path}")

    def _collect_infos(candidate_paths: Sequence[Path]) -> List[ShardInfo]:
        deduped: List[Path] = []
        seen = set()
        for p in candidate_paths:
            p = Path(p)
            if p in seen or p.suffix.lower() != ".npz" or _is_obvious_non_data_path(p):
                continue
            seen.add(p)
            deduped.append(p)
        if max_shards is not None:
            deduped = deduped[: int(max_shards)]

        out: List[ShardInfo] = []
        remaining = None if max_data_tasks is None else int(max_data_tasks)
        for candidate in deduped:
            info = _inspect_npz_shard(candidate)
            if info is None:
                continue
            if remaining is not None:
                if remaining <= 0:
                    break
                if info.num_tasks > remaining:
                    info = ShardInfo(
                        path=info.path,
                        num_tasks=remaining,
                        N=info.N,
                        P=info.P,
                        has_lambda_grid=info.has_lambda_grid,
                    )
                remaining -= info.num_tasks
            out.append(info)
        return out

    shards = _collect_infos(candidates)
    if not shards and path.is_dir() and manifest_paths:
        fallback_candidates = sorted(
            path.glob(shard_glob) if shard_glob else path.rglob("*.npz")
        )
        shards = _collect_infos(fallback_candidates)

    if not shards:
        raise ValueError(f"no valid X/y .npz shards found under {path}")

    if is_main_rank():
        total_tasks = sum(s.num_tasks for s in shards)
        unique_shapes = sorted({(s.N, s.P) for s in shards})
        log.info(
            "data_shards_discovered root=%s num_shards=%d total_tasks=%d "
            "unique_shapes=%s lambda_grid_shards=%d",
            path, len(shards), total_tasks, unique_shapes,
            sum(1 for s in shards if s.has_lambda_grid),
        )
    return shards


class StaticBatchSource:
    """Single .npz file pinned as one batch panel."""

    def __init__(self, path: Path, *, max_data_tasks: Optional[int] = None):
        self.path = Path(path)
        X_np, y_np = _load_npz_panel(self.path, max_data_tasks=max_data_tasks)
        self.X = X_np
        self.y = y_np
        self._task_id_prefix = self.path.stem

    def summary(self) -> Dict[str, Any]:
        return {
            "data_loader": "single",
            "num_shards": 1,
            "total_tasks": int(self.X.shape[0]),
            "unique_shapes": [[int(self.X.shape[1]), int(self.X.shape[2])]],
            "steps_per_data_epoch": 1,
            "batch_size": int(self.X.shape[0]),
            "shuffle_shards": False,
            "shuffle_within_shard": False,
            "path": str(self.path),
        }

    def next_numpy_batch(
        self, step: int, *, batch_size: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], Dict[str, Any]]:
        del step
        B_total = int(self.X.shape[0])
        if batch_size is None or batch_size >= B_total:
            indices = list(range(B_total))
        else:
            # deterministic: just take the first B tasks
            indices = list(range(int(batch_size)))
        Xcs, ycs, task_ids = _to_centered_panel(
            self.X, self.y, self._task_id_prefix, indices,
        )
        info = {
            "data_loader": "single",
            "shard_path": str(self.path),
            "B": len(Xcs),
            "N": int(self.X.shape[1]),
            "P": int(self.X.shape[2]),
        }
        return Xcs, ycs, task_ids, info


class ShardCyclingBatchSource:
    """Round-robin over .npz shards with task shuffling, cache, and
    optional wrap-last-batch. Ported from train_unified_krylov."""

    def __init__(
        self,
        shards: Sequence[ShardInfo],
        *,
        batch_size: int,
        seed: int = 0,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True,
        drop_last: bool = False,
        wrap_last_batch: bool = True,
        allow_mixed_shapes: bool = True,
        cache_size: int = 1,
    ):
        if not shards:
            raise ValueError("ShardCyclingBatchSource requires at least one shard")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.shards = list(shards)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_within_shard = bool(shuffle_within_shard)
        self.drop_last = bool(drop_last)
        self.wrap_last_batch = bool(wrap_last_batch)
        self.cache_size = max(1, int(cache_size))
        self._cache: "OrderedDict[Path, Tuple[np.ndarray, np.ndarray]]" = OrderedDict()
        self._units_cache: Dict[int, List[Tuple[int, int]]] = {}
        self._perm_cache: Dict[Tuple[int, int], List[int]] = {}

        unique_shapes = {(s.N, s.P) for s in self.shards}
        if len(unique_shapes) > 1 and not allow_mixed_shapes:
            raise ValueError(
                f"mixed shard shapes disabled, got {sorted(unique_shapes)}"
            )
        self.steps_per_data_epoch = sum(
            self._num_chunks(s.num_tasks) for s in self.shards
        )
        if self.steps_per_data_epoch <= 0:
            raise ValueError(
                "no trainable shard chunks; reduce batch_size or disable drop_last"
            )

    def _num_chunks(self, num_tasks: int) -> int:
        if self.drop_last:
            return int(num_tasks) // self.batch_size
        return int(math.ceil(float(num_tasks) / float(self.batch_size)))

    def _epoch_units(self, epoch: int) -> List[Tuple[int, int]]:
        if epoch in self._units_cache:
            return self._units_cache[epoch]
        shard_order = list(range(len(self.shards)))
        rng = random.Random(self.seed + int(epoch))
        if self.shuffle_shards:
            rng.shuffle(shard_order)
        units: List[Tuple[int, int]] = []
        for shard_idx in shard_order:
            chunk_order = list(range(self._num_chunks(self.shards[shard_idx].num_tasks)))
            if self.shuffle_within_shard:
                crng = random.Random(
                    self.seed + 1000003 * int(epoch) + 9176 * shard_idx
                )
                crng.shuffle(chunk_order)
            units.extend((shard_idx, chunk_idx) for chunk_idx in chunk_order)
        self._units_cache = {epoch: units}
        self._perm_cache = {
            key: val for key, val in self._perm_cache.items() if key[0] == epoch
        }
        return units

    def _task_perm(self, epoch: int, shard_idx: int) -> List[int]:
        key = (int(epoch), int(shard_idx))
        if key in self._perm_cache:
            return self._perm_cache[key]
        n = self.shards[shard_idx].num_tasks
        perm = list(range(n))
        if self.shuffle_within_shard:
            rng = random.Random(self.seed + 2000003 * int(epoch) + 7919 * shard_idx)
            rng.shuffle(perm)
        self._perm_cache[key] = perm
        return perm

    def _load_shard(self, shard_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        info = self.shards[shard_idx]
        path = info.path
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        X_np, y_np = _load_npz_panel(path, max_data_tasks=info.num_tasks)
        if not bool(np.isfinite(X_np).all()) or not bool(np.isfinite(y_np).all()):
            raise ValueError(f"non-finite X/y values in shard {path}")
        self._cache[path] = (X_np, y_np)
        self._cache.move_to_end(path)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return X_np, y_np

    def summary(self) -> Dict[str, Any]:
        unique_shapes = sorted({(s.N, s.P) for s in self.shards})
        return {
            "data_loader": "shard_cycle",
            "num_shards": len(self.shards),
            "total_tasks": int(sum(s.num_tasks for s in self.shards)),
            "unique_shapes": [[int(n), int(p)] for n, p in unique_shapes],
            "lambda_grid_shards": int(sum(1 for s in self.shards if s.has_lambda_grid)),
            "steps_per_data_epoch": int(self.steps_per_data_epoch),
            "batch_size": int(self.batch_size),
            "shuffle_shards": self.shuffle_shards,
            "shuffle_within_shard": self.shuffle_within_shard,
            "drop_last": self.drop_last,
            "wrap_last_batch": self.wrap_last_batch,
            "cache_size": int(self.cache_size),
        }

    def next_numpy_batch(
        self, step: int, *, batch_size: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], Dict[str, Any]]:
        # `batch_size` arg is ignored here -- the source is constructed
        # with a fixed batch_size, and changing it mid-epoch would break
        # the chunk arithmetic.
        del batch_size
        epoch = int(step) // self.steps_per_data_epoch
        pos = int(step) % self.steps_per_data_epoch
        units = self._epoch_units(epoch)
        shard_idx, chunk_idx = units[pos]
        info = self.shards[shard_idx]
        perm = self._task_perm(epoch, shard_idx)
        start = chunk_idx * self.batch_size
        if self.wrap_last_batch:
            task_indices = [
                perm[(start + j) % info.num_tasks]
                for j in range(self.batch_size)
            ]
        else:
            task_indices = perm[start: min(start + self.batch_size, info.num_tasks)]
        if not task_indices:
            raise RuntimeError("empty shard-cycle batch")
        X_np, y_np = self._load_shard(shard_idx)
        Xcs, ycs, task_ids = _to_centered_panel(
            X_np, y_np, f"{info.path.stem}_e{epoch}", task_indices,
        )
        return Xcs, ycs, task_ids, {
            "data_loader": "shard_cycle",
            "shard_idx": int(shard_idx),
            "shard_path": str(info.path),
            "chunk_idx": int(chunk_idx),
            "data_epoch": int(epoch),
            "batch_task_indices": [int(i) for i in task_indices],
            "B": len(Xcs),
            "N": int(X_np.shape[1]),
            "P": int(X_np.shape[2]),
        }


def build_training_data_source(
    *,
    data_path: Optional[str],
    data_loader: str,
    shard_glob: str,
    max_shards: Optional[int],
    max_data_tasks: Optional[int],
    batch_size: int,
    seed: int,
    shuffle_shards: bool,
    shuffle_within_shard: bool,
    drop_last: bool,
    wrap_last_batch: bool,
    shard_cache_size: int,
):
    """Returns a data source object exposing `next_numpy_batch(step)`
    and `summary()`, or `None` if synthetic-stub mode is requested.

    The expected contract for the rest of the trainer:
        Xcs, ycs, task_ids, info = source.next_numpy_batch(step)
    where Xcs / ycs are already centered.
    """
    if not data_path:
        return None
    path = Path(data_path)
    mode = data_loader
    if mode == "auto":
        mode = "shard_cycle" if path.is_dir() else "single"
    if mode == "single":
        if path.is_dir():
            raise ValueError(
                "data_loader=single requires data_path to be a .npz file"
            )
        return StaticBatchSource(path, max_data_tasks=max_data_tasks)
    if mode == "shard_cycle":
        shards = discover_npz_shards(
            path,
            shard_glob=shard_glob,
            max_shards=max_shards,
            max_data_tasks=max_data_tasks,
        )
        if len(shards) == 1 and is_main_rank():
            log.warning(
                "shard_cycle has only one discovered shard: %s",
                shards[0].path,
            )
        return ShardCyclingBatchSource(
            shards,
            batch_size=batch_size,
            seed=seed,
            shuffle_shards=shuffle_shards,
            shuffle_within_shard=shuffle_within_shard,
            drop_last=drop_last,
            wrap_last_batch=wrap_last_batch,
            allow_mixed_shapes=True,
            cache_size=shard_cache_size,
        )
    raise ValueError(f"unknown data_loader mode: {mode!r}")


# =============================================================================
# Synthetic stub (kept for smoke / unit testing without sync21)
# =============================================================================

def _stub_load_task_batch(
    rng: np.random.Generator,
    *,
    B: int,
    P: int,
    N: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], Dict[str, Any]]:
    """Synthetic (N, P) sparse regression batch.

    Same shape contract as the .npz loaders:
        Xcs : list[B] of (N, P) float64 CENTERED matrices
        ycs : list[B] of (N,) float64 CENTERED responses
        task_ids: list[B] of stable string ids
        info : dict with B/N/P and data_loader='synthetic'
    """
    Xcs, ycs, task_ids = [], [], []
    for b in range(B):
        X = rng.standard_normal((N, P))
        X = X - X.mean(axis=0, keepdims=True)
        beta_true = np.zeros(P)
        beta_true[: max(1, P // 4)] = rng.standard_normal(max(1, P // 4)) * 2.0
        y = X @ beta_true + 0.1 * rng.standard_normal(N)
        y = y - y.mean()
        Xcs.append(X)
        ycs.append(y)
        task_ids.append(f"stub_b{b}_seed{int(rng.integers(0, 2**31 - 1))}")
    info = {"data_loader": "synthetic", "B": B, "N": N, "P": P}
    return Xcs, ycs, task_ids, info


# =============================================================================
# Per-step utilities
# =============================================================================

def _compute_lambda_max(
    Xcs: List[np.ndarray], ycs: List[np.ndarray],
) -> torch.Tensor:
    """Per-task lambda_max for centered (X, y).

    lambda_max_b = max_j |Xc_b[:, j]^T yc_b| / N

    Even though Xcs / ycs should already be centered (the loaders do
    that before returning), we recenter here as a paranoid safety net so
    this stays consistent whether the data path is the synthetic stub,
    a .npz loader, or the eventual sync21 loader. This matches the
    centered-problem assumption used in the Krylov objective and CD
    reference.
    """
    out: List[float] = []
    for X, y in zip(Xcs, ycs):
        # Recenter -- cheap, and guarantees the FPK loss view.
        Xc = X - X.mean(axis=0, keepdims=True)
        yc = y - float(y.mean())
        N = max(1, int(Xc.shape[0]))
        Xty = (Xc.T @ yc) / float(N)
        val = float(np.max(np.abs(Xty)))
        if not math.isfinite(val) or val <= 0.0:
            val = 1.0  # avoid 0 / nan blowing up the sampler
        out.append(val)
    return torch.tensor(out, dtype=torch.float32)


def _build_X_y_tensors(
    Xcs: Sequence[np.ndarray],
    ycs: Sequence[np.ndarray],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack (B, N, P) and (B, N) tensors. `dtype` is the data dtype
    (typically fp32; the autocast wrapper handles the model side)."""
    Xs = np.stack(Xcs, axis=0)
    ys = np.stack(ycs, axis=0)
    return (
        torch.tensor(Xs, dtype=dtype, device=device),
        torch.tensor(ys, dtype=dtype, device=device),
    )


def _flatten_queries(
    Xc_batch: torch.Tensor,        # (B, N, P) fp32
    yc_batch: torch.Tensor,        # (B, N)   fp32
    beta_q: torch.Tensor,          # (B, Q, P) any dtype
    sampled: SampledBatch,
) -> List[QueryLossInputs]:
    """Build the per-query loss inputs list."""
    B, Q, P = beta_q.shape
    out: List[QueryLossInputs] = []
    for b in range(B):
        Xc_b = Xc_batch[b]
        yc_b = yc_batch[b]
        for q in range(Q):
            out.append(QueryLossInputs(
                Xc=Xc_b,
                yc=yc_b,
                beta_hat=beta_q[b, q].to(Xc_b.dtype),
                lam=float(sampled.lambdas[b, q].item()),
                segments=sampled.segments_grid[b][q],
            ))
    return out


# =============================================================================
# Optimizer + LR schedule
# =============================================================================

def _make_optimizer(model: nn.Module, train_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        betas=tuple(train_cfg.get("betas", [0.9, 0.95])),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )


def _lr_lambda(step: int, train_cfg: Dict[str, Any]) -> float:
    """Linear warmup then either constant or cosine to 0."""
    warmup = int(train_cfg.get("lr_warmup_steps", 1000))
    total = int(train_cfg.get("total_steps", 50_000))
    schedule = train_cfg.get("lr_schedule", "warmup_then_constant")
    if step < warmup:
        return float(step + 1) / float(max(1, warmup))
    if schedule == "warmup_then_constant":
        return 1.0
    if schedule == "warmup_then_cosine":
        t = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, t))))
    raise ValueError(f"unknown lr_schedule {schedule!r}")


# =============================================================================
# Checkpointing (richer payload ported from train_unified_krylov)
# =============================================================================

def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    sampler: PenaltySampler,
    lr_sched: Optional[torch.optim.lr_scheduler._LRScheduler],
    step: int,
    cfg: Dict[str, Any],
    applied_model_cfg: Dict[str, Any],
    latest_metrics: Optional[Dict[str, Any]] = None,
    parameter_count: Optional[int] = None,
    data_source_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """Atomic checkpoint write with a richer payload.

    Stores:
      - model / optimizer / sampler state (FPK contract)
      - lr scheduler state (so resume doesn't have to fast-forward)
      - RNG states (python random, numpy, torch cpu, torch cuda all-ranks)
      - latest training-side metrics + parameter count (Krylov-style)
      - config + applied_model_cfg + data_source_summary (provenance)
    """
    payload: Dict[str, Any] = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "sampler_state_dict": sampler.state_dict(),
        "lr_scheduler_state_dict": (
            lr_sched.state_dict() if lr_sched is not None else None
        ),
        "config": cfg,
        "applied_model_cfg": applied_model_cfg,
        "latest_metrics": dict(latest_metrics or {}),
        "parameter_count": int(parameter_count) if parameter_count is not None else None,
        "data_source_summary": dict(data_source_summary or {}),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
        "torch_version": torch.__version__,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _maybe_resume(
    resume_path: Optional[str],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    sampler: PenaltySampler,
    lr_sched: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    restore_rng: bool = True,
) -> int:
    """Returns the step to resume from (0 if no resume).

    Backwards-compatible with older FPK checkpoints that did not store
    the lr scheduler / latest metrics / RNG payload.
    """
    if not resume_path:
        return 0
    # weights_only=False: our own checkpoints contain the run config
    # (TorchVersion etc); safe because the trainer is the producer.
    ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    sampler.load_state_dict(ckpt["sampler_state_dict"])
    if lr_sched is not None and ckpt.get("lr_scheduler_state_dict") is not None:
        try:
            lr_sched.load_state_dict(ckpt["lr_scheduler_state_dict"])
        except Exception as exc:
            if is_main_rank():
                log.warning("failed to load lr scheduler state: %s", exc)
    if restore_rng and "rng_state" in ckpt:
        rng_state = ckpt["rng_state"] or {}
        try:
            if "python" in rng_state:
                random.setstate(rng_state["python"])
            if "numpy" in rng_state:
                np.random.set_state(rng_state["numpy"])
            if "torch_cpu" in rng_state:
                torch.set_rng_state(rng_state["torch_cpu"])
            if rng_state.get("torch_cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
        except Exception as exc:
            if is_main_rank():
                log.warning("partial RNG restore failed: %s", exc)
    return int(ckpt["step"]) + 1


# =============================================================================
# Logging utilities
# =============================================================================

class JsonlLogger:
    """Notion-syncable JSON-lines training log."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Open in line-buffered append mode.
        self._fh = open(self.path, "a", buffering=1)

    def log(self, record: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, default=str) + "\n")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass


# =============================================================================
# Autocast: model forward in BF16, loss in fp32
# =============================================================================

def _autocast_ctx(precision_cfg: Dict[str, Any], device: torch.device):
    """Per A2 mixed: encoder/model in bf16, feature builder stays fp32.

    NOTE: caller is responsible for only wrapping the MODEL FORWARD in
    this context, not the loss / KKT / prox computation. The Krylov
    driver did the same split; FPK had a bug where the loss landed
    inside the autocast block. See `train_step` below for the fix.
    """
    if not precision_cfg.get("boundary_cast", True):
        return nullcontext()
    if precision_cfg.get("model", "bf16") != "bf16":
        return nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device.type == "cpu":
        try:
            return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        except (TypeError, RuntimeError):
            return nullcontext()
    return nullcontext()


# =============================================================================
# Evaluation helpers (ported from train_unified_krylov)
# =============================================================================
# These work on the SAME segment objects produced by the FPK sampler.
# We assume each segment exposes the standard fields:
#     seg.lo, seg.hi, seg.c, seg.d, seg.h
# and the methods `seg.eval_rho(t)`, `seg.eval_drho(t)`. If the model_fpk
# build does not provide the eval_* methods, we fall back to the closed
# form  rho(t) = c + d*t + 0.5*h*t^2,  rho'(t) = d + h*t.


def _seg_rho_scalar_fallback(seg: Any, t: float) -> float:
    return float(seg.c) + float(seg.d) * t + 0.5 * float(seg.h) * t * t


def _seg_drho_scalar_fallback(seg: Any, t: float) -> float:
    return float(seg.d) + float(seg.h) * t


def _segment_for_t(segments: Sequence[Any], t: float) -> Any:
    for seg in segments:
        lo = float(seg.lo)
        hi = float(seg.hi)
        if lo <= t and (not math.isfinite(hi) or t <= hi):
            return seg
    return segments[-1]


def _segment_rho_scalar(t: float, segments: Sequence[Any]) -> float:
    t = max(0.0, float(t))
    seg = _segment_for_t(segments, t)
    if hasattr(seg, "eval_rho"):
        try:
            return float(seg.eval_rho(t))
        except Exception:
            pass
    return _seg_rho_scalar_fallback(seg, t)


def _segment_drho_scalar(t: float, segments: Sequence[Any]) -> float:
    t = max(0.0, float(t))
    seg = _segment_for_t(segments, t)
    if hasattr(seg, "eval_drho"):
        try:
            return float(seg.eval_drho(t))
        except Exception:
            pass
    return _seg_drho_scalar_fallback(seg, t)


def _d0_from_segments(segments: Sequence[Any]) -> float:
    """rho'(0+). Prefer the helper from model_fpk if exposed."""
    if _fpk_d0_from_segments is not None:
        try:
            return float(_fpk_d0_from_segments(segments))
        except Exception:
            pass
    # rho'(0) of the segment containing t=0:  d + h * 0 = d.
    return _segment_drho_scalar(0.0, segments)


def _segment_summary(segments: Sequence[Any]) -> List[Dict[str, Optional[float]]]:
    out: List[Dict[str, Optional[float]]] = []
    for seg in segments:
        hi = float(seg.hi)
        out.append({
            "lo": float(seg.lo),
            "hi": None if math.isinf(hi) else hi,
            "c": float(seg.c),
            "d": float(seg.d),
            "h": float(seg.h),
        })
    return out


def scalar_segment_prox(
    a: float, z: float, lam: float, segments: Sequence[Any],
) -> float:
    """Evaluation-only scalar SegmentList prox; no sklearn dependency.

    Minimizes  0.5 a (b - z)^2 + lam * rho(|b|).
    """
    a = float(a)
    z = float(z)
    lam = float(lam)
    if not (math.isfinite(a) and a > 0.0):
        raise ValueError(f"scalar_segment_prox requires a > 0, got {a}")
    if not math.isfinite(z) or not math.isfinite(lam):
        raise ValueError("scalar_segment_prox requires finite z and lam")

    eps = 1e-12
    candidates: List[float] = [0.0]
    for seg in segments:
        lo = max(0.0, float(seg.lo))
        hi = float(seg.hi)
        candidates.extend([lo, -lo])
        if math.isfinite(hi):
            candidates.extend([hi, -hi])
        for s in (1.0, -1.0):
            denom = a + lam * float(seg.h)
            if denom <= eps:
                continue
            t_star = (s * a * z - lam * float(seg.d)) / denom
            if t_star < lo - 1e-10:
                continue
            if math.isfinite(hi) and t_star > hi + 1e-10:
                continue
            if t_star >= 0.0 and math.isfinite(t_star):
                candidates.append(s * max(0.0, t_star))

    def obj(b: float) -> float:
        return 0.5 * a * (b - z) * (b - z) + lam * _segment_rho_scalar(abs(b), segments)

    best_b = 0.0
    best_obj = float("inf")
    for b in candidates:
        if not math.isfinite(b):
            continue
        val = obj(float(b))
        if val < best_obj:
            best_obj = val
            best_b = float(b)
    return best_b


def penalized_objective(
    Xc: torch.Tensor, yc: torch.Tensor, beta: torch.Tensor,
    segments: Sequence[Any], lam: float,
) -> float:
    """0.5 / N * ||yc - Xc beta||^2 + lam * sum_j rho(|beta_j|).

    Computed in fp64 on CPU for numerical stability."""
    Xd = Xc.detach().cpu().double() if isinstance(Xc, torch.Tensor) else torch.as_tensor(Xc).double()
    yd = yc.detach().cpu().double().view(-1) if isinstance(yc, torch.Tensor) else torch.as_tensor(yc).double().view(-1)
    bd = beta.detach().cpu().double().view(-1) if isinstance(beta, torch.Tensor) else torch.as_tensor(beta).double().view(-1)
    N = max(1, int(Xd.shape[0]))
    residual = yd - Xd.mv(bd)
    data = 0.5 * float(residual.dot(residual).item()) / float(N)
    penalty = float(lam) * sum(
        _segment_rho_scalar(abs(float(v)), segments) for v in bd
    )
    return float(data + penalty)


def approx_kkt_residual(
    Xc: torch.Tensor, yc: torch.Tensor, beta: torch.Tensor,
    segments: Sequence[Any], lam: float,
) -> float:
    """Approximate coordinate KKT residual for evaluation diagnostics."""
    Xd = Xc.detach().cpu().double() if isinstance(Xc, torch.Tensor) else torch.as_tensor(Xc).double()
    yd = yc.detach().cpu().double().view(-1) if isinstance(yc, torch.Tensor) else torch.as_tensor(yc).double().view(-1)
    bd = beta.detach().cpu().double().view(-1) if isinstance(beta, torch.Tensor) else torch.as_tensor(beta).double().view(-1)
    N = max(1, int(Xd.shape[0]))
    grad = Xd.t().mv(Xd.mv(bd) - yd) / float(N)
    d0 = max(0.0, _d0_from_segments(segments))
    vals: List[float] = []
    for j, bj_t in enumerate(bd):
        bj = float(bj_t)
        gj = float(grad[j])
        if abs(bj) <= 1e-4:
            vals.append(max(0.0, abs(gj) - float(lam) * d0))
        else:
            drho = _segment_drho_scalar(abs(bj), segments)
            vals.append(abs(gj + float(lam) * drho * (1.0 if bj > 0.0 else -1.0)))
    return float(max(vals) if vals else 0.0)


def coordinate_descent_reference(
    Xc: torch.Tensor,
    yc: torch.Tensor,
    segments: Sequence[Any],
    lam: float,
    max_sweeps: int = 50,
    tol: float = 1e-6,
    warm_start: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Custom SegmentList-based coordinate-descent reference, not sklearn.

    The input must already be centered. For nonconvex penalties this is
    a lightweight reference point, not a global-optimality certificate.
    """
    Xd = Xc.detach().cpu().double() if isinstance(Xc, torch.Tensor) else torch.as_tensor(Xc).double()
    yd = yc.detach().cpu().double().view(-1) if isinstance(yc, torch.Tensor) else torch.as_tensor(yc).double().view(-1)
    if Xd.dim() != 2 or yd.dim() != 1 or Xd.shape[0] != yd.shape[0]:
        raise ValueError("coordinate_descent_reference expects Xc=(N,P), yc=(N,)")
    N, P = Xd.shape
    if warm_start is None:
        beta = torch.zeros(P, dtype=torch.float64)
    else:
        beta = warm_start.detach().cpu().double().view(P).clone()
    residual = yd - Xd.mv(beta)
    diag = Xd.square().mean(dim=0)

    converged = False
    max_delta = 0.0
    num_sweeps = 0
    for sweep in range(int(max_sweeps)):
        max_delta = 0.0
        for j in range(P):
            a = float(diag[j])
            old = float(beta[j])
            if a <= 1e-12 or not math.isfinite(a):
                new = 0.0
            else:
                xj = Xd[:, j]
                z = old + float(xj.dot(residual).item()) / (float(N) * a)
                new = scalar_segment_prox(a, z, float(lam), segments)
            if new != old:
                delta = new - old
                beta[j] = new
                residual -= Xd[:, j] * delta
                max_delta = max(max_delta, abs(delta))
        num_sweeps = sweep + 1
        if max_delta <= float(tol) * (1.0 + float(beta.abs().max().item())):
            converged = True
            break

    objective = penalized_objective(Xd, yd, beta, segments, float(lam))
    return {
        "beta": beta.float(),
        "objective": float(objective),
        "converged": bool(converged),
        "num_sweeps": int(num_sweeps),
        "max_delta": float(max_delta),
        "note": "custom SegmentList-based coordinate-descent reference, not sklearn",
    }


def nnz(beta: torch.Tensor, threshold: float = 1e-4) -> int:
    return int((beta.detach().abs() > float(threshold)).sum().item())


def beta_corr(
    beta_hat: torch.Tensor, beta_ref: torch.Tensor,
    active_only: bool = False, threshold: float = 1e-4,
) -> float:
    bh = beta_hat.detach().cpu().float().view(-1)
    br = beta_ref.detach().cpu().float().view(-1)
    if active_only:
        mask = (bh.abs() > threshold) | (br.abs() > threshold)
        if int(mask.sum()) == 0:
            return 1.0
        bh = bh[mask]
        br = br[mask]
    bh = bh - bh.mean()
    br = br - br.mean()
    nh = float(bh.norm().item())
    nr = float(br.norm().item())
    if nh <= 1e-12 and nr <= 1e-12:
        return 1.0
    if nh <= 1e-12 or nr <= 1e-12:
        return 0.0
    return float(torch.dot(bh, br).item() / (nh * nr))


def support_metrics(
    beta_hat: torch.Tensor, beta_ref: torch.Tensor, threshold: float = 1e-4,
) -> Dict[str, float]:
    pred = beta_hat.detach().abs() > float(threshold)
    ref = beta_ref.detach().abs() > float(threshold)
    tp = int((pred & ref).sum().item())
    fp = int((pred & ~ref).sum().item())
    fn = int((~pred & ref).sum().item())
    if tp == 0 and fp == 0 and fn == 0:
        precision = recall = f1 = 1.0
    else:
        precision = tp / float(tp + fp) if (tp + fp) else 0.0
        recall = tp / float(tp + fn) if (tp + fn) else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) else 0.0
        )
    return {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "tp": tp, "fp": fp, "fn": fn,
    }


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def ridge_guard_metrics(
    aux_out: Dict[str, Any], sampled: SampledBatch,
) -> Dict[str, float]:
    """Check d0 / tau behavior on Ridge queries (penalty_type_id ==
    RIDGE_FAMILY)."""
    pid = sampled.penalty_type_id
    ridge_mask = (pid == PID_RIDGE)
    if not bool(ridge_mask.any()):
        return {"ridge_n": 0.0, "ridge_d0_max": 0.0, "ridge_tau_max": float("nan")}
    d0 = sampled.d0[ridge_mask].detach().abs().max() if hasattr(sampled, "d0") else torch.tensor(float("nan"))
    tau = aux_out.get("tau_hat_q") if isinstance(aux_out, dict) else None
    if isinstance(tau, torch.Tensor):
        tau_max = float(tau.detach()[ridge_mask].abs().max().cpu())
    else:
        tau_max = float("nan")
    return {
        "ridge_n": float(ridge_mask.sum().detach().cpu()),
        "ridge_d0_max": float(d0.detach().cpu()) if hasattr(d0, "detach") else float(d0),
        "ridge_tau_max": tau_max,
    }


def _kind_counts(sampled: SampledBatch) -> Dict[str, int]:
    """Per-family counts over (B, Q) for diagnostic logging."""
    pid = sampled.penalty_type_id.detach().cpu().view(-1).tolist()
    name_for = {
        PID_LASSO: "lasso",
        PID_RIDGE: "ridge",
        PID_ENET:  "enet",
        PID_MCP:   "mcp",
        PID_SCAD:  "scad",
    }
    out: Dict[str, int] = {}
    for v in pid:
        key = name_for.get(int(v), f"pid_{int(v)}")
        out[key] = out.get(key, 0) + 1
    # `random` queries are not a separate pid; pull from sampler metadata if available.
    meta = getattr(sampled, "metadata", None) or {}
    if isinstance(meta, dict):
        if "n_random" in meta:
            out["random_theta"] = int(meta["n_random"])
        if "n_known" in meta:
            out["known_family"] = int(meta["n_known"])
    return out


# =============================================================================
# One training step (loss now lives OUTSIDE the autocast block)
# =============================================================================

@dataclass
class StepResult:
    total_loss: float
    L_obj: float
    L_kkt: float
    L_prox: float
    L_delta: float
    grad_norm: float
    lr: float
    wall_time_sec: float
    stage: str
    finite_loss: bool
    # diagnostic / Krylov-style extras
    beta_abs_mean: float
    nnz_1em4: float
    sampler_meta: Dict[str, Any] = field(default_factory=dict)
    kind_counts: Dict[str, int] = field(default_factory=dict)
    ridge: Dict[str, float] = field(default_factory=dict)
    data_info: Dict[str, Any] = field(default_factory=dict)
    output_shapes: Dict[str, List[int]] = field(default_factory=dict)


def train_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    sampler: PenaltySampler,
    step: int,
    cfg: Dict[str, Any],
    device: torch.device,
    rng: np.random.Generator,
    data_source: Any,
) -> StepResult:
    """One training step.

    Important contracts versus the previous FPK driver:
      - Loss / KKT / prox / delta are computed in fp32 OUTSIDE the
        autocast context. Only `model.forward` runs in BF16. This matches
        the Krylov driver and avoids BF16 noise in the prox / KKT
        residuals.
      - Per-task data comes from `data_source` (a Static or Shard-cycling
        .npz source) when configured, falling back to the synthetic stub.
      - Returned StepResult carries enough metadata for the Krylov-style
        JSONL log (sampler metadata, kind counts, ridge guard, data
        info, output shapes, finite-loss flag, beta_abs_mean, nnz).
    """
    t0 = time.time()
    training_cfg = cfg["training"]
    loss_cfg = cfg.get("loss", {})

    B = int(training_cfg["batch_size"])
    P = int(training_cfg["P"])
    N_fallback = int(training_cfg.get("N", 350))

    # 1. Load task batch.
    if data_source is not None:
        Xcs, ycs, task_ids, data_info = data_source.next_numpy_batch(step, batch_size=B)
    else:
        Xcs, ycs, task_ids, data_info = _stub_load_task_batch(
            rng, B=B, P=P, N=N_fallback,
        )

    # 2. Compute lambda_max per task (CPU numpy; safe re centering).
    lambda_max = _compute_lambda_max(Xcs, ycs)

    # 3. Sample one batch of (penalty, lambda) queries via the FPK
    #    sampler (random θ + known families, stage curriculum).
    sampled = sampler.sample(
        batch_size=B, lambda_max=lambda_max, device=device, step=step,
    )

    # 4. Move (X, y) to device. fp32 — feature builder stays fp32 per A2.
    Xc_dev, yc_dev = _build_X_y_tensors(Xcs, ycs, device=device, dtype=torch.float32)

    # 5. MODEL FORWARD in autocast (BF16). Loss / KKT / prox live below
    #    in fp32 outside the autocast block.
    optimizer.zero_grad(set_to_none=True)
    precision_cfg = training_cfg.get("precision", {})
    with _autocast_ctx(precision_cfg, device):
        dec = model.forward(
            X=Xc_dev,
            y=yc_dev,
            lambdas=sampled.lambdas,
            lambda_max=sampled.lambda_max,
            penalty_type_id=sampled.penalty_type_id,
            penalty_params=sampled.penalty_params,
            penalty_shape_features=sampled.shape_features,
            penalty_d0=sampled.d0,
            return_aux=True,
        )
    beta_q_raw = dec["beta_q"]            # (B, Q, P), possibly bf16
    beta_q = beta_q_raw.float()           # fp32 for loss / metrics

    # 6. Loss in fp32 (NOT inside the autocast block).
    q_inputs = _flatten_queries(Xc_dev.float(), yc_dev.float(), beta_q, sampled)
    out: BatchLossOutputs = compute_batch_loss(
        q_inputs,
        step=step,
        w_obj=float(loss_cfg.get("w_obj", W_OBJ)),
        w_kkt=float(loss_cfg.get("w_kkt", W_KKT)),
        w_prox=float(loss_cfg.get("w_prox", W_PROX)),
        w_delta=float(loss_cfg.get("w_delta", W_DELTA)),
        beta_hat_all=beta_q,
    )
    finite_loss = bool(torch.isfinite(out.total.detach()).item())

    out.total.backward()
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    if grad_clip > 0:
        grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
    else:
        grad_norm = 0.0
    optimizer.step()

    # 7. Krylov-style diagnostics
    beta_abs_mean = float(beta_q.detach().abs().mean().cpu())
    nnz_1em4 = float((beta_q.detach().abs() > 1e-4).float().sum(dim=-1).mean().cpu())
    sampler_meta = dict(getattr(sampled, "metadata", {}) or {})
    kc = _kind_counts(sampled)
    rg = ridge_guard_metrics(dec, sampled)

    current_lr = optimizer.param_groups[0]["lr"]
    return StepResult(
        total_loss=float(out.total.item()),
        L_obj=float(out.L_obj.item()),
        L_kkt=float(out.L_kkt.item()),
        L_prox=float(out.L_prox.item()),
        L_delta=float(out.L_delta.item()),
        grad_norm=grad_norm,
        lr=current_lr,
        wall_time_sec=time.time() - t0,
        stage=getattr(sampled, "stage", sampler_meta.get("stage", "?")),
        finite_loss=finite_loss,
        beta_abs_mean=beta_abs_mean,
        nnz_1em4=nnz_1em4,
        sampler_meta=sampler_meta,
        kind_counts=kc,
        ridge=rg,
        data_info=data_info,
        output_shapes={
            "beta_q": list(beta_q.shape),
            "lambda": list(sampled.lambdas.shape),
        },
    )


# =============================================================================
# Penalty diagnostics dump (Krylov-style, using the FPK sampler)
# =============================================================================

def dump_penalty_diagnostics(
    *,
    out_path: Path,
    sampler_cfg: PenaltySamplerConfig,
    seed: int,
    num_samples: int = 32,
    grid_size: int = 257,
    grid_max: float = 5.0,
    diagnostic_step: int = 0,
    plot: bool = False,
) -> Dict[str, Any]:
    """Dump rho / rho' / d0 / knot statistics for `num_samples` random
    penalties drawn from the FPK sampler. Mirrors the Krylov
    `dump_penalty_diagnostics` output (JSON + optional PNGs) but uses
    the FPK sampler, NOT the legacy `sample_random_segments` smoke
    sampler.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Force random-θ-only samples by overriding the ratios.
    random_only_cfg = PenaltySamplerConfig(
        **{
            **{f: getattr(sampler_cfg, f) for f in sampler_cfg.__dataclass_fields__},
            "queries_per_task": 1,
            "known_family_ratio": 0.0,
            "random_theta_ratio": 1.0,
        }
    )
    known_only_cfg = PenaltySamplerConfig(
        **{
            **{f: getattr(sampler_cfg, f) for f in sampler_cfg.__dataclass_fields__},
            "queries_per_task": 1,
            "known_family_ratio": 1.0,
            "random_theta_ratio": 0.0,
        }
    )

    cpu = torch.device("cpu")
    lambda_max_dummy = torch.ones(1, dtype=torch.float32)
    grid_vals = np.linspace(0.0, float(grid_max), int(grid_size)).tolist()

    # ---- Random-θ samples
    rand_sampler = PenaltySampler(random_only_cfg, seed=seed)
    samples: List[Dict[str, Any]] = []
    d0_values: List[float] = []
    knot_counts: List[int] = []
    rho_rows: List[List[float]] = []
    drho_rows: List[List[float]] = []
    for i in range(int(num_samples)):
        sampled = rand_sampler.sample(
            batch_size=1, lambda_max=lambda_max_dummy, device=cpu,
            step=diagnostic_step,
        )
        segs = sampled.segments_grid[0][0]
        rho = [_segment_rho_scalar(t, segs) for t in grid_vals]
        drho = [_segment_drho_scalar(t, segs) for t in grid_vals]
        if not all(math.isfinite(v) for v in rho + drho):
            raise RuntimeError(
                f"sampled non-finite random penalty diagnostics at i={i}"
            )
        d0 = _d0_from_segments(segs)
        knot_count = max(0, len(segs) - 1)
        d0_values.append(float(d0))
        knot_counts.append(int(knot_count))
        rho_rows.append(rho)
        drho_rows.append(drho)
        samples.append({
            "index": i,
            "num_segments": len(segs),
            "knot_count": knot_count,
            "d0": float(d0),
            "segments": _segment_summary(segs),
            "rho": rho,
            "rho_prime": drho,
        })

    # ---- Known-family overlay (lasso / ridge / enet / mcp / scad).
    known_overlay: Dict[str, Dict[str, float]] = {}
    try:
        known_sampler = PenaltySampler(known_only_cfg, seed=seed + 1)
        # Sample a wide batch and bin by penalty_type_id.
        sampled_known = known_sampler.sample(
            batch_size=64, lambda_max=torch.ones(64, dtype=torch.float32),
            device=cpu, step=diagnostic_step,
        )
        pid_to_name = {
            PID_LASSO: "lasso",
            PID_RIDGE: "ridge",
            PID_ENET:  "enet",
            PID_MCP:   "mcp",
            PID_SCAD:  "scad",
        }
        seen: Dict[str, Sequence[Any]] = {}
        pid_flat = sampled_known.penalty_type_id.detach().cpu().view(-1).tolist()
        for idx_flat, pid in enumerate(pid_flat):
            b = idx_flat // sampled_known.lambdas.shape[1]
            q = idx_flat % sampled_known.lambdas.shape[1]
            name = pid_to_name.get(int(pid))
            if name is None or name in seen:
                continue
            seen[name] = sampled_known.segments_grid[b][q]
        for name, segs in seen.items():
            known_overlay[name] = {
                "d0": float(_d0_from_segments(segs)),
                "num_segments": len(segs),
                "segments": _segment_summary(segs),  # type: ignore[arg-type]
            }
    except Exception as exc:  # noqa: BLE001
        if is_main_rank():
            log.warning("known-family overlay sampling failed: %s", exc)

    payload: Dict[str, Any] = {
        "seed": seed,
        "diagnostic_step": int(diagnostic_step),
        "num_samples": int(num_samples),
        "grid_t": grid_vals,
        "d0_values": d0_values,
        "knot_counts": knot_counts,
        "samples": samples,
        "known_family_overlay": known_overlay,
        "sampler_cfg": {
            f: getattr(sampler_cfg, f) for f in sampler_cfg.__dataclass_fields__
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if is_main_rank():
        log.info("saved_penalty_diagnostics=%s", out_path)

    if plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            if is_main_rank():
                log.warning(
                    "matplotlib unavailable; saved diagnostics JSON without plots"
                )
            return payload
        xs = grid_vals
        for rows, filename, ylabel in [
            (rho_rows, "sampled_rho.png", "rho(t)"),
            (drho_rows, "sampled_drho.png", "rho'(t)"),
        ]:
            plt.figure(figsize=(7, 4))
            for row in rows:
                plt.plot(xs, row, alpha=0.35)
            try:
                plt.xscale("symlog", linthresh=1e-4)
            except Exception:
                pass
            plt.xlabel("t")
            plt.ylabel(ylabel)
            plt.tight_layout()
            plt.savefig(out_path.parent / filename, dpi=150)
            plt.close()
        plt.figure(figsize=(6, 4))
        plt.hist(d0_values, bins=min(20, max(5, len(d0_values) // 2)))
        plt.xlabel("d0")
        plt.tight_layout()
        plt.savefig(out_path.parent / "d0_hist.png", dpi=150)
        plt.close()
        plt.figure(figsize=(6, 4))
        plt.hist(knot_counts, bins=min(20, max(5, len(knot_counts) // 2)))
        plt.xlabel("knot count")
        plt.tight_layout()
        plt.savefig(out_path.parent / "knot_count_hist.png", dpi=150)
        plt.close()
    return payload


# =============================================================================
# Eval-smoke: CD reference, objective gap, beta corr, support F1, KKT
# =============================================================================

def eval_smoke(
    *,
    model: nn.Module,
    sampler: PenaltySampler,
    cfg: Dict[str, Any],
    device: torch.device,
    rng: np.random.Generator,
    data_source: Any,
    eval_reference_queries: int = 4,
    reference_max_sweeps: int = 50,
    reference_tol: float = 1e-6,
    eval_step: int = 0,
    metrics_out: Optional[Path] = None,
) -> Dict[str, Any]:
    """Smoke evaluation: one batch + custom CD reference per selected
    query. Returns the aggregate metrics dict and (optionally) writes it
    to `metrics_out` as JSON. Mirrors the Krylov `eval_smoke` output but
    uses the FPK model + FPK sampler.
    """
    training_cfg = cfg["training"]
    B = int(training_cfg["batch_size"])
    P = int(training_cfg["P"])
    N_fallback = int(training_cfg.get("N", 350))

    # 1. Load one task batch.
    if data_source is not None:
        Xcs, ycs, task_ids, _data_info = data_source.next_numpy_batch(eval_step, batch_size=B)
    else:
        Xcs, ycs, task_ids, _data_info = _stub_load_task_batch(
            rng, B=B, P=P, N=N_fallback,
        )
    lambda_max = _compute_lambda_max(Xcs, ycs)

    # 2. Sample queries.
    sampled = sampler.sample(
        batch_size=B, lambda_max=lambda_max, device=device, step=eval_step,
    )
    Xc_dev, yc_dev = _build_X_y_tensors(Xcs, ycs, device=device, dtype=torch.float32)

    # 3. Forward (no grad, autocast for model only).
    precision_cfg = training_cfg.get("precision", {})
    model.eval()
    with torch.no_grad():
        with _autocast_ctx(precision_cfg, device):
            dec = model.forward(
                X=Xc_dev,
                y=yc_dev,
                lambdas=sampled.lambdas,
                lambda_max=sampled.lambda_max,
                penalty_type_id=sampled.penalty_type_id,
                penalty_params=sampled.penalty_params,
                penalty_shape_features=sampled.shape_features,
                penalty_d0=sampled.d0,
                return_aux=True,
            )
        beta_q = dec["beta_q"].float()  # (B, Q, P) fp32

        # FPK loss for training-comparable scalar.
        q_inputs = _flatten_queries(Xc_dev.float(), yc_dev.float(), beta_q, sampled)
        out: BatchLossOutputs = compute_batch_loss(
            q_inputs,
            step=eval_step,
            w_obj=1.0, w_kkt=0.0, w_prox=0.0, w_delta=0.0,  # objective-only eval scalar
            beta_hat_all=beta_q,
        )

    # 4. Per-query CD reference (on task b=0 for cheap eval).
    selected = list(range(min(int(sampled.lambdas.shape[1]),
                              max(1, int(eval_reference_queries)))))
    per_query: List[Dict[str, Any]] = []
    Xc_b0 = Xc_dev[0]
    yc_b0 = yc_dev[0]
    for q in selected:
        segments = sampled.segments_grid[0][q]
        lam = float(sampled.lambdas[0, q].detach().cpu())
        beta_model = beta_q[0, q].detach().cpu()
        ref = coordinate_descent_reference(
            Xc_b0, yc_b0, segments, lam,
            max_sweeps=reference_max_sweeps, tol=reference_tol,
            warm_start=None,
        )
        beta_ref = ref["beta"]
        obj_model = penalized_objective(Xc_b0, yc_b0, beta_model, segments, lam)
        obj_ref = float(ref["objective"])
        obj_gap = (obj_model / obj_ref - 1.0) if abs(obj_ref) > 1e-12 else float("nan")
        supp = support_metrics(beta_model, beta_ref)
        per_query.append({
            "query_index": int(q),
            "penalty_type_id": int(sampled.penalty_type_id[0, q].item()),
            "lambda": lam,
            "d0": float(sampled.d0[0, q].item()) if hasattr(sampled, "d0") else None,
            "objective_model": float(obj_model),
            "objective_ref": float(obj_ref),
            "objective_gap": float(obj_gap),
            "beta_corr_all": beta_corr(beta_model, beta_ref, active_only=False),
            "beta_corr_active": beta_corr(beta_model, beta_ref, active_only=True),
            "nnz_model": nnz(beta_model),
            "nnz_ref": nnz(beta_ref),
            "precision": supp["precision"],
            "recall": supp["recall"],
            "f1": supp["f1"],
            "kkt_residual_approx": approx_kkt_residual(
                Xc_b0, yc_b0, beta_model, segments, lam,
            ),
            "reference_converged": bool(ref["converged"]),
            "reference_num_sweeps": int(ref["num_sweeps"]),
            "reference_note": ref["note"],
        })

    metrics: Dict[str, Any] = {
        "eval_step": int(eval_step),
        "B": int(Xc_dev.shape[0]),
        "Q": int(sampled.lambdas.shape[1]),
        "N": int(Xc_dev.shape[1]),
        "P": int(Xc_dev.shape[2]),
        "selected_query_indices": selected,
        "loss_fpk_total": float(out.total.detach().cpu()),
        "loss_fpk_L_obj": float(out.L_obj.detach().cpu()),
        "loss_fpk_L_kkt": float(out.L_kkt.detach().cpu()),
        "loss_fpk_L_prox": float(out.L_prox.detach().cpu()),
        "loss_fpk_L_delta": float(out.L_delta.detach().cpu()),
        "objective_model": _mean([m["objective_model"] for m in per_query]),
        "objective_ref": _mean([m["objective_ref"] for m in per_query]),
        "objective_gap": _mean([m["objective_gap"] for m in per_query]),
        "beta_corr_all": _mean([m["beta_corr_all"] for m in per_query]),
        "beta_corr_active": _mean([m["beta_corr_active"] for m in per_query]),
        "nnz_model": _mean([m["nnz_model"] for m in per_query]),
        "nnz_ref": _mean([m["nnz_ref"] for m in per_query]),
        "precision": _mean([m["precision"] for m in per_query]),
        "recall": _mean([m["recall"] for m in per_query]),
        "f1": _mean([m["f1"] for m in per_query]),
        "kkt_residual_approx": _mean([m["kkt_residual_approx"] for m in per_query]),
        "reference_converged_all": all(bool(m["reference_converged"]) for m in per_query),
        "reference_num_sweeps_max": max(
            (int(m["reference_num_sweeps"]) for m in per_query), default=0,
        ),
        "reference_note": "custom SegmentList-based coordinate-descent reference, not sklearn",
        "per_query": per_query,
    }
    finite_scalars = [
        metrics["loss_fpk_total"], metrics["objective_model"],
        metrics["objective_ref"], metrics["kkt_residual_approx"],
    ]
    metrics["finite_score_diagnostics"] = all(
        math.isfinite(float(v)) for v in finite_scalars
    )
    text = json.dumps(metrics, indent=2, ensure_ascii=False, default=str)
    if is_main_rank():
        print(text, flush=True)
    if metrics_out is not None:
        metrics_out = Path(metrics_out)
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_text(text + "\n", encoding="utf-8")
        if is_main_rank():
            log.info("saved_eval_metrics=%s", metrics_out)
    return metrics


# =============================================================================
# Capacity sweep hint
# =============================================================================

def print_capacity_sweep_hint() -> None:
    """Recipe for the action-item sweep: M = 8, 16, 32 with hidden=512,
    batch=64, mid-heavy lambda sampler, random fraction 0.8, 50k steps.
    """
    hint_lines = []
    for krylov_dim in (8, 16, 32):
        hint_lines.append(
            f"python train_fpk.py --config configs/fpk_v2.yaml "
            f"--mode main --run-name fpk_sweep_M{krylov_dim} "
            f"--override training.batch_size=64 training.queries_per_task=8 "
            f"training.N=500 training.P=50 training.total_steps=50000 "
            f"model.krylov_dim={krylov_dim} model.feature_head_hidden=512 "
            f"sampler.random_theta_ratio=0.8 sampler.known_family_ratio=0.2 "
            f"sampler.lambda_mid_weight=0.5"
        )
    print("\n".join(hint_lines), flush=True)


# =============================================================================
# Config overrides from CLI (--override key.path=value)
# =============================================================================

def _apply_cli_overrides(cfg: Dict[str, Any], overrides: Sequence[str]) -> None:
    """Apply `--override foo.bar=3 baz.q=0.5` style overrides in place."""
    if not overrides:
        return
    for entry in overrides:
        if "=" not in entry:
            raise ValueError(f"--override expects key.path=value, got {entry!r}")
        key_path, _, raw_val = entry.partition("=")
        # Best-effort literal parsing.
        val: Any
        s = raw_val.strip()
        if s.lower() in ("true", "yes"):
            val = True
        elif s.lower() in ("false", "no"):
            val = False
        elif s.lower() in ("null", "none", "~"):
            val = None
        else:
            try:
                val = int(s)
            except ValueError:
                try:
                    val = float(s)
                except ValueError:
                    val = s
        parts = key_path.strip().split(".")
        node = cfg
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = val


# =============================================================================
# Main
# =============================================================================

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False,
                        help="Path to YAML config (e.g. configs/fpk_v2.yaml). "
                             "Required for main / smoke / eval-smoke / diagnostics.")
    parser.add_argument(
        "--mode",
        choices=["smoke", "main", "eval-smoke", "diagnostics", "capacity-sweep-hint"],
        default="main",
        help=(
            "smoke = B=4, total_steps=50; main = config value; "
            "eval-smoke = forward + CD reference on one batch; "
            "diagnostics = dump random-θ rho/d0/knot to JSON; "
            "capacity-sweep-hint = print recommended sweep commands."
        ),
    )
    parser.add_argument("--resume", type=str, default=None,
                        help="path/to/ckpt.pt to resume from")
    parser.add_argument("--out-dir", type=str, default="runs",
                        help="output root; run-name subdir is created inside")
    parser.add_argument("--run-name", type=str, default=None,
                        help="override config's experiment_name")
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sync21-root", type=str, default=None,
                        help="override training.sync21_root in the config")
    # Krylov-style data loader CLI (override config.training.data.*)
    parser.add_argument("--data-path", type=str, default=None,
                        help=".npz file or directory of .npz shards")
    parser.add_argument("--data-loader", choices=["auto", "single", "shard_cycle"],
                        default="auto")
    parser.add_argument("--shard-glob", type=str, default="**/*.npz")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-data-tasks", type=int, default=None)
    parser.add_argument("--shard-cache-size", type=int, default=1)
    parser.add_argument("--shuffle-shards", action="store_true", default=True)
    parser.add_argument("--no-shuffle-shards", dest="shuffle_shards",
                        action="store_false")
    parser.add_argument("--shuffle-within-shard", action="store_true", default=True)
    parser.add_argument("--no-shuffle-within-shard", dest="shuffle_within_shard",
                        action="store_false")
    parser.add_argument("--drop-last-data-batch", action="store_true", default=False)
    parser.add_argument("--wrap-last-data-batch", action="store_true", default=True)
    parser.add_argument("--no-wrap-last-data-batch", dest="wrap_last_data_batch",
                        action="store_false")
    # Eval-smoke specific
    parser.add_argument("--metrics-out", type=str, default=None,
                        help="(eval-smoke) write the metrics JSON here")
    parser.add_argument("--eval-reference-queries", type=int, default=4)
    parser.add_argument("--reference-max-sweeps", type=int, default=50)
    parser.add_argument("--reference-tol", type=float, default=1e-6)
    # Diagnostics-specific
    parser.add_argument("--diagnostics-out", type=str, default=None,
                        help="(diagnostics) where to write penalty_diagnostics.json")
    parser.add_argument("--num-diagnostic-penalties", type=int, default=32)
    parser.add_argument("--diagnostic-step", type=int, default=0)
    parser.add_argument("--plot-penalty-diagnostics", action="store_true")
    # Generic config overrides
    parser.add_argument("--override", nargs="*", default=None,
                        help="key.path=value pairs to override the YAML config")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # ---- capacity sweep hint exits early; no DDP, no config required.
    if args.mode == "capacity-sweep-hint":
        print_capacity_sweep_hint()
        return

    if not args.config:
        raise SystemExit(
            "--config is required for modes "
            "{smoke, main, eval-smoke, diagnostics}"
        )

    setup_ddp_if_needed()
    rank = get_rank()
    world = get_world_size()

    cfg = _load_yaml(args.config)
    if args.override:
        _apply_cli_overrides(cfg, args.override)

    run_name = args.run_name or cfg.get("experiment_name", "fpk_run")
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    if args.sync21_root is not None:
        cfg.setdefault("training", {})["sync21_root"] = args.sync21_root

    # Data CLI: pull into cfg.training.data so resume sees the same dict.
    data_cfg = cfg.setdefault("training", {}).setdefault("data", {})
    if args.data_path is not None:
        data_cfg["path"] = args.data_path
    data_cfg.setdefault("loader", args.data_loader)
    data_cfg.setdefault("shard_glob", args.shard_glob)
    data_cfg.setdefault("max_shards", args.max_shards)
    data_cfg.setdefault("max_data_tasks", args.max_data_tasks)
    data_cfg.setdefault("shard_cache_size", args.shard_cache_size)
    data_cfg.setdefault("shuffle_shards", args.shuffle_shards)
    data_cfg.setdefault("shuffle_within_shard", args.shuffle_within_shard)
    data_cfg.setdefault("drop_last_data_batch", args.drop_last_data_batch)
    data_cfg.setdefault("wrap_last_data_batch", args.wrap_last_data_batch)

    # Smoke mode overrides
    if args.mode == "smoke":
        cfg.setdefault("training", {}).update({
            "batch_size": 4,
            "queries_per_task": 4,
            "P": 10,
            "N": 50,
            "total_steps": 50,
            "log_every": 5,
            "save_every": 50,
            "eval_every": 50,
            "checkpoint_trajectory_steps": [50],
        })
        cfg.setdefault("sampler", {})["queries_per_task"] = 4

    if args.use_cuda and torch.cuda.is_available():
        # Under DDP, each rank must bind to its own local GPU. torchrun sets
        # LOCAL_RANK per worker; setup_ddp_if_needed() already called
        # torch.cuda.set_device(local_rank), so we mirror that choice here.
        # Without this, every rank would land on cuda:{args.device} (default 0)
        # and NCCL would refuse: "Duplicate GPU detected".
        if "LOCAL_RANK" in os.environ:
            device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        else:
            device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

    if is_main_rank():
        log.info("rank=%d world=%d mode=%s device=%s run=%s",
                 rank, world, args.mode, device, run_name)
        log.info("config: %s", args.config)

    # ---- MODEL HOOK (filled per fpk_decisions.md §2.2) ---------------------
    fpk_cfg_dict = dict(cfg.get("model", {}))
    penalty_shape_cfg = dict(cfg.get("penalty_shape", {}))
    if penalty_shape_cfg:
        # Rename yaml keys to FpkConfig field names
        fpk_cfg_dict.update({
            "penalty_grid_size": penalty_shape_cfg.get("grid_size", 24),
            "penalty_grid_channels": penalty_shape_cfg.get("grid_channels", 4),
            "meta_dim": penalty_shape_cfg.get("meta_dim", 5),
            "shape_feature_dim": penalty_shape_cfg.get("shape_feature_dim", 101),
            "penalty_shape_hidden": penalty_shape_cfg.get("encoder_hidden", 128),
            "penalty_shape_depth": penalty_shape_cfg.get("encoder_depth", 2),
            "penalty_shape_code_dim": penalty_shape_cfg.get("shape_code_dim", 24),
        })
    fpk_cfg = FpkConfig()
    valid_keys = set(fpk_cfg.__dataclass_fields__)
    valid_overrides = {k: v for k, v in fpk_cfg_dict.items() if k in valid_keys}
    model, applied_model_cfg = make_fpk_model(
        cfg=fpk_cfg, device=device, **valid_overrides
    )
    parameter_count = count_parameters(model)

    # Wrap DDP if launched via torchrun.
    raw_model = model
    if world > 1 and torch.distributed.is_initialized():
        if device.type == "cuda":
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[device.index], output_device=device.index,
            )
        else:
            model = torch.nn.parallel.DistributedDataParallel(model)

    # Optimizer + LR schedule
    train_cfg = cfg.get("training", {})
    optimizer = _make_optimizer(model, train_cfg)
    lr_sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: _lr_lambda(s, train_cfg),
    )

    # Sampler
    #
    # Default `lambda_high_max` = 0.3 (was 1.0).
    #
    # Rationale: with the project's typical data normalization,
    # `lambda_max = ||X^T y||_inf / N` clusters around 0.2 (p10 ~ 0.13, p90 ~ 0.38
    # observed; see diag.json from diagnose_lambda_ratio.py). The high band
    # `[lambda_mid_high, lambda_high_max] = [0.1, 1.0]` therefore overshoots
    # lambda_max on most tasks: ~13% of queries land in the dead zone
    # (lambda > lambda_max -> beta_star = 0). Setting `lambda_high_max = 0.3`
    # keeps the high band entirely below the p90 of lambda_max, which drops
    # dead-zone mass to ~7% and bumps transition mass from 45.75% to ~48% --
    # verdict goes MARGINAL -> GOOD. Override via YAML or
    # `--override sampler.lambda_high_max=...` if your data normalization is
    # different and `lambda_max` distribution is centered elsewhere.
    sampler_cfg_yaml = cfg.get("sampler", {})
    sampler_cfg = PenaltySamplerConfig(
        queries_per_task=int(train_cfg.get("queries_per_task", 8)),
        known_family_ratio=float(sampler_cfg_yaml.get("known_family_ratio", 0.20)),
        random_theta_ratio=float(sampler_cfg_yaml.get("random_theta_ratio", 0.80)),
        lambda_low_min=float(sampler_cfg_yaml.get("lambda_low_min", 1e-3)),
        lambda_mid_low=float(sampler_cfg_yaml.get("lambda_mid_low", 1e-2)),
        lambda_mid_high=float(sampler_cfg_yaml.get("lambda_mid_high", 1e-1)),
        lambda_high_max=float(sampler_cfg_yaml.get("lambda_high_max", 0.3)),
        lambda_low_weight=float(sampler_cfg_yaml.get("lambda_low_weight", 0.30)),
        lambda_mid_weight=float(sampler_cfg_yaml.get("lambda_mid_weight", 0.50)),
        lambda_high_weight=float(sampler_cfg_yaml.get("lambda_high_weight", 0.20)),
        K_min=int(sampler_cfg_yaml.get("K_min", 3)),
        K_max=int(sampler_cfg_yaml.get("K_max", 64)),
        monotone=bool(sampler_cfg_yaml.get("monotone", False)),
        knot_strata=tuple(sampler_cfg_yaml.get("knot_strata", [0.60, 0.25, 0.15])),
        d0_bucket_weights=tuple(sampler_cfg_yaml.get(
            "d0_bucket_weights", [0.10, 0.25, 0.45, 0.20])),
        tail_floor=float(sampler_cfg_yaml.get("tail_floor", 0.02)),
        stage_step_boundaries=tuple(sampler_cfg_yaml.get(
            "stage_step_boundaries", [10_000, 30_000])),
    )
    sampler = PenaltySampler(sampler_cfg, seed=seed)

    # Build data source (Krylov-ported). None means synthetic stub.
    try:
        data_source = build_training_data_source(
            data_path=data_cfg.get("path"),
            data_loader=data_cfg.get("loader", "auto"),
            shard_glob=data_cfg.get("shard_glob", "**/*.npz"),
            max_shards=data_cfg.get("max_shards"),
            max_data_tasks=data_cfg.get("max_data_tasks"),
            batch_size=int(train_cfg.get("batch_size", 4)),
            seed=seed,
            shuffle_shards=bool(data_cfg.get("shuffle_shards", True)),
            shuffle_within_shard=bool(data_cfg.get("shuffle_within_shard", True)),
            drop_last=bool(data_cfg.get("drop_last_data_batch", False)),
            wrap_last_batch=bool(data_cfg.get("wrap_last_data_batch", True)),
            shard_cache_size=int(data_cfg.get("shard_cache_size", 1)),
        )
    except Exception as exc:
        if is_main_rank():
            log.warning(
                "data source construction failed (%s); falling back to synthetic stub",
                exc,
            )
        data_source = None
    data_source_summary = (
        data_source.summary() if data_source is not None
        else {
            "data_loader": "synthetic",
            "num_shards": 0,
            "total_tasks": None,
            "unique_shapes": [[int(train_cfg.get("N", 350)), int(train_cfg.get("P", 50))]],
            "steps_per_data_epoch": None,
            "batch_size": int(train_cfg.get("batch_size", 4)),
        }
    )

    # ---- Special modes that don't run the training loop -------------------
    if args.mode == "diagnostics":
        if is_main_rank():
            diag_path = Path(
                args.diagnostics_out or (out_dir / "penalty_diagnostics.json")
            )
            dump_penalty_diagnostics(
                out_path=diag_path,
                sampler_cfg=sampler_cfg,
                seed=seed,
                num_samples=int(args.num_diagnostic_penalties),
                diagnostic_step=int(args.diagnostic_step),
                plot=bool(args.plot_penalty_diagnostics),
            )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        return

    if args.mode == "eval-smoke":
        rng_eval = np.random.default_rng(seed + rank * 1000003)
        if args.resume:
            _ = _maybe_resume(
                args.resume, model=raw_model, optimizer=optimizer,
                sampler=sampler, lr_sched=lr_sched,
            )
        metrics_path = (
            Path(args.metrics_out) if args.metrics_out
            else out_dir / "eval_smoke_metrics.json"
        )
        eval_smoke(
            model=raw_model, sampler=sampler, cfg=cfg, device=device,
            rng=rng_eval, data_source=data_source,
            eval_reference_queries=int(args.eval_reference_queries),
            reference_max_sweeps=int(args.reference_max_sweeps),
            reference_tol=float(args.reference_tol),
            eval_step=0, metrics_out=metrics_path,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        return

    # ---- Standard main / smoke training loop ------------------------------
    start_step = _maybe_resume(
        args.resume, model=raw_model, optimizer=optimizer,
        sampler=sampler, lr_sched=lr_sched,
    )

    # Logging
    jsonl = JsonlLogger(out_dir / "log.jsonl") if is_main_rank() else None

    rng = np.random.default_rng(seed + rank * 1000003)
    total_steps = int(train_cfg.get("total_steps", 50_000))
    log_every = int(train_cfg.get("log_every", 10))
    save_every = int(train_cfg.get("save_every", 1000))
    eval_every = int(train_cfg.get("eval_every", 2500))
    trajectory = list(train_cfg.get(
        "checkpoint_trajectory_steps",
        [1000, 5000, 8500, 10000, 25000, 50000],
    ))
    trajectory_set = set(trajectory)

    # Run-config preamble (Krylov-style structured emit)
    if is_main_rank() and jsonl is not None:
        jsonl.log({
            "event": "run_config",
            "seed": seed,
            "mode": args.mode,
            "rank": rank,
            "world": world,
            "device": str(device),
            "run_name": run_name,
            "config_path": args.config,
            "B": int(train_cfg.get("batch_size", 4)),
            "Q": int(train_cfg.get("queries_per_task", 8)),
            "N": int(train_cfg.get("N", 350)),
            "P": int(train_cfg.get("P", 50)),
            "total_steps": total_steps,
            "lr": float(train_cfg.get("lr", 1e-3)),
            "lr_schedule": train_cfg.get("lr_schedule", "warmup_then_constant"),
            "loss_weights": {
                "w_obj": float(cfg.get("loss", {}).get("w_obj", W_OBJ)),
                "w_kkt": float(cfg.get("loss", {}).get("w_kkt", W_KKT)),
                "w_prox": float(cfg.get("loss", {}).get("w_prox", W_PROX)),
                "w_delta": float(cfg.get("loss", {}).get("w_delta", W_DELTA)),
            },
            "sampler": {
                "known_family_ratio": sampler_cfg.known_family_ratio,
                "random_theta_ratio": sampler_cfg.random_theta_ratio,
                "lambda_low_weight": sampler_cfg.lambda_low_weight,
                "lambda_mid_weight": sampler_cfg.lambda_mid_weight,
                "lambda_high_weight": sampler_cfg.lambda_high_weight,
                "K_min": sampler_cfg.K_min,
                "K_max": sampler_cfg.K_max,
                "stage_step_boundaries": list(sampler_cfg.stage_step_boundaries),
            },
            "model_cfg": applied_model_cfg,
            "parameter_count": parameter_count,
            "data_source": data_source_summary,
        })

    latest_metrics: Dict[str, Any] = {}
    t_start = time.perf_counter()

    try:
        for step in range(start_step, total_steps):
            try:
                result = train_step(
                    model=model, optimizer=optimizer, sampler=sampler,
                    step=step, cfg=cfg, device=device, rng=rng,
                    data_source=data_source,
                )
                lr_sched.step()

                # Track latest metrics for checkpoint payload.
                latest_metrics = {
                    "step": step,
                    "stage": result.stage,
                    "total_loss": result.total_loss,
                    "L_obj": result.L_obj,
                    "L_kkt": result.L_kkt,
                    "L_prox": result.L_prox,
                    "L_delta": result.L_delta,
                    "grad_norm": result.grad_norm,
                    "lr": result.lr,
                    "beta_abs_mean": result.beta_abs_mean,
                    "nnz_1e-4": result.nnz_1em4,
                    "finite_loss": result.finite_loss,
                    "kind_counts": result.kind_counts,
                    "ridge": result.ridge,
                    "parameter_count": parameter_count,
                }

                if (step % log_every == 0) and is_main_rank():
                    log.info(
                        "step=%d stage=%s loss=%.4f L_obj=%.4f L_kkt=%.4f "
                        "L_prox=%.4f L_delta=%.4f gn=%.3f lr=%.2e "
                        "beta_abs=%.4f nnz=%.1f finite=%s t=%.3fs",
                        step, result.stage, result.total_loss,
                        result.L_obj, result.L_kkt, result.L_prox,
                        result.L_delta, result.grad_norm, result.lr,
                        result.beta_abs_mean, result.nnz_1em4,
                        result.finite_loss, result.wall_time_sec,
                    )
                    if jsonl is not None:
                        elapsed = time.perf_counter() - t_start
                        di = result.data_info or {}
                        shard_path = di.get("shard_path")
                        jsonl.log({
                            "step": step,
                            "elapsed_sec": round(elapsed, 3),
                            "step_elapsed_sec": round(result.wall_time_sec, 3),
                            "stage": result.stage,
                            # FPK loss decomposition
                            "total_loss": round(result.total_loss, 6),
                            "L_obj": round(result.L_obj, 6),
                            "L_kkt": round(result.L_kkt, 6),
                            "L_prox": round(result.L_prox, 6),
                            "L_delta": round(result.L_delta, 6),
                            "grad_norm": round(result.grad_norm, 4),
                            "lr": result.lr,
                            # Krylov-style scalar diagnostics
                            "beta_abs_mean": round(result.beta_abs_mean, 6),
                            "nnz_1e-4": round(result.nnz_1em4, 3),
                            "finite_loss": result.finite_loss,
                            # sampler / penalty mix
                            "sampler_meta": result.sampler_meta,
                            "kind_counts": result.kind_counts,
                            "ridge_n": result.ridge.get("ridge_n"),
                            "ridge_d0_max": result.ridge.get("ridge_d0_max"),
                            "ridge_tau_max": result.ridge.get("ridge_tau_max"),
                            # data loader
                            "data_loader": di.get("data_loader"),
                            "data_epoch": di.get("data_epoch"),
                            "shard_idx": di.get("shard_idx"),
                            "shard_path": Path(shard_path).name if shard_path else None,
                            "batch_B": di.get("B"),
                            "batch_N": di.get("N"),
                            "batch_P": di.get("P"),
                            # tensor shapes
                            "output_shapes": result.output_shapes,
                            "parameter_count": parameter_count,
                        })

                # Save snapshots
                if is_main_rank():
                    if step > 0 and (
                        step % save_every == 0 or (step + 1) in trajectory_set
                    ):
                        path = ckpt_dir / f"ckpt_step{step:07d}.pt"
                        _save_checkpoint(
                            path,
                            model=raw_model, optimizer=optimizer,
                            sampler=sampler, lr_sched=lr_sched,
                            step=step, cfg=cfg,
                            applied_model_cfg=applied_model_cfg,
                            latest_metrics=latest_metrics,
                            parameter_count=parameter_count,
                            data_source_summary=data_source_summary,
                        )
                        log.info("saved checkpoint -> %s", path)

                if step > 0 and step % eval_every == 0 and is_main_rank():
                    # Inline eval-smoke for live diagnostics.
                    try:
                        eval_smoke(
                            model=raw_model, sampler=sampler, cfg=cfg,
                            device=device, rng=rng, data_source=data_source,
                            eval_reference_queries=int(args.eval_reference_queries),
                            reference_max_sweeps=int(args.reference_max_sweeps),
                            reference_tol=float(args.reference_tol),
                            eval_step=step,
                            metrics_out=out_dir / f"eval_smoke_step{step:07d}.json",
                        )
                    except Exception as exc:  # noqa: BLE001
                        log.warning("inline eval-smoke failed at step %d: %s", step, exc)
                    # Switch back to train mode (eval_smoke flipped to .eval()).
                    model.train()

            except KeyboardInterrupt:
                log.info("interrupted at step %d; saving and exiting", step)
                if is_main_rank():
                    _save_checkpoint(
                        ckpt_dir / f"ckpt_interrupted_step{step:07d}.pt",
                        model=raw_model, optimizer=optimizer,
                        sampler=sampler, lr_sched=lr_sched,
                        step=step, cfg=cfg,
                        applied_model_cfg=applied_model_cfg,
                        latest_metrics=latest_metrics,
                        parameter_count=parameter_count,
                        data_source_summary=data_source_summary,
                    )
                break

        # Final checkpoint
        if is_main_rank():
            path = ckpt_dir / f"ckpt_final_step{total_steps:07d}.pt"
            _save_checkpoint(
                path,
                model=raw_model, optimizer=optimizer, sampler=sampler,
                lr_sched=lr_sched, step=total_steps - 1, cfg=cfg,
                applied_model_cfg=applied_model_cfg,
                latest_metrics=latest_metrics,
                parameter_count=parameter_count,
                data_source_summary=data_source_summary,
            )
            log.info("final checkpoint -> %s", path)

    finally:
        if jsonl is not None:
            jsonl.close()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
