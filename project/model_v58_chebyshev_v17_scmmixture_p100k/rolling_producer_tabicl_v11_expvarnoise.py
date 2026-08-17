#!/usr/bin/env python3
"""Rolling producer for TabICLv2 V11 feature-noise tasks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from model_v58_chebyshev_v14_scmmixture import rolling_producer as producer
from model_v58_chebyshev_v14_scmmixture.rolling_catalog import (
    CapacityPolicy,
    CatalogStore,
)
from model_v58_chebyshev_v17_scmmixture_p100k import (
    rolling_producer_backpressure_fixed as fixed,
)


PRIORS = ("tabicl_v2",)
PIPELINE_VERSION = (
    "v19_kalbi_tabiclv2_scm2048_bnn128_expvarnoise_v11_rolling100gb_v1"
)
SOURCE_VERSION = "v19_tabiclv2_v11_expvarnoise_cpu_rolling_producer_v1"
CURATOR_VERSION = "v19_tabiclv2_v11_expvarnoise_rolling100gb_curator_v1"

GENERATOR = "generate_tabiclv2_scm2048_qfilter_bnn128_expvarnoise_v11.py"
PRIOR_NAME = (
    "official_tabiclv2_graph_scm_xy_expvarnoise_qfiltered_core2048_"
    "then_random_bnn128_expvarnoise_v1"
)
DATASET_CONTRACT = "tabiclv2_graph_scm2048_qfilter_bnn128_expvarnoise_v11"
DATA_POLICY = (
    "v7_allfuncs_qfilter_noisy_core2048_bnn128_expvarnoise_"
    "sampled_final_audit_v1"
)
TARGET_CONTRACT = (
    "official_tabiclv2_graph_scm_y_qualityfilter_center_scale_clip10_rescale_v2"
)
FEATURE_NOISE_CONTRACT = (
    "all_features_variance_iid_exponential_rate1_error_gaussian_v1"
)
BNN_CONTRACT = (
    "fixed_noisy_qfiltered_scm2048_core_parents_random_gaussian_weight_"
    "prior_block128_depth1or2_v1"
)
CORRELATION_AUDIT_CONTRACT = (
    "deterministic_uniform_feature_sample_float16_pearson_"
    "not_full_pair_certificate_v1"
)
CORE_CORRELATION_CERTIFICATE = (
    "exact incremental all-pair Pearson after featurewise noise and "
    "float16 roundtrip"
)
SHAPE_SAMPLER = (
    "smallN10pct_coupled_lognormal_median_N2000_P10000_"
    "maxN40K_maxP200K_v3"
)
TABICL_COMMIT = "46b91961db4f8873dd049ec09990698a435e1e29"
CORRELATION_THRESHOLD = 0.95
NOISE_VARIANCE_RATE = 1.0
SCM_CORE_CAP = 2_048
BNN_OUTPUT_BLOCK = 128
BNN_HIDDEN_DIM = 64
BNN_PARENT_MIN = 8
BNN_PARENT_MAX = 128
CORR_AUDIT_FEATURES = 256
N_MIN = 32
N_MAX = 40_000
P_MIN = 50
P_MAX = 100_000
MAX_CELLS = 24_000_000

V11_PRIOR_CONTRACT = {
    "generator": GENERATOR,
    "prior_name": PRIOR_NAME,
    "dataset_contract": DATASET_CONTRACT,
    "data_policy": DATA_POLICY,
    "target_contract": TARGET_CONTRACT,
    "official_core_prior": "TabICLv2 PriorDataset prior_type=graph_scm",
    "target_source": "official TabICLv2 PriorDataset graph_scm response",
    "direct_mode": "direct_qfiltered_scm",
    "highp_mode": "noisy_qfiltered_scm2048_plus_noisy_bnn128",
    "commit": TABICL_COMMIT,
}

_FIXED_UNADOPTED_STAGING_NPZ_BYTES = fixed._unadopted_staging_npz_bytes


def tabicl_v11_policy(*, world_size: int = 8) -> CapacityPolicy:
    # A small epoch-0 catalog lets training start while rolling generation
    # grows the active set to the 100 GB target.
    return CapacityPolicy(
        active_min_bytes=500_000_000,
        active_max_bytes=102_000_000_000,
        prior_min_bytes=500_000_000,
        prior_max_bytes=102_000_000_000,
        hard_cap_bytes=130_000_000_000,
        min_free_bytes=300_000_000_000,
        world_size=world_size,
    )


class TabICLV11CatalogStore(CatalogStore):
    def __init__(
        self,
        control_root: Any,
        data_root: Any,
        policy: CapacityPolicy | None = None,
        *,
        priors: Any = PRIORS,
    ) -> None:
        if tuple(priors) != PRIORS:
            raise ValueError(f"V11 TabICL store received priors={tuple(priors)!r}")
        super().__init__(
            control_root,
            data_root,
            policy=policy or tabicl_v11_policy(),
            priors=PRIORS,
        )


def _bootstrap_aware_staging_npz_bytes(
    root: str | Path,
    *,
    adopted_shard_ids: set[int] | None = None,
    prior: str | None = None,
) -> int:
    root_path = Path(root).expanduser().resolve()
    if not (root_path / "rolling_control" / "CURRENT.json").is_file():
        return fixed._ORIGINAL_STAGING_NPZ_BYTES(
            root_path,
            adopted_shard_ids=adopted_shard_ids,
            prior=prior,
        )
    return _FIXED_UNADOPTED_STAGING_NPZ_BYTES(
        root_path,
        adopted_shard_ids=adopted_shard_ids,
        prior=prior,
    )


def _require_exact(
    mapping: Mapping[str, Any], key: str, expected: Any, *, where: str
) -> Any:
    actual = mapping.get(key)
    if actual != expected:
        raise producer.ProducerError(
            f"{where}.{key} contract mismatch: {actual!r} != {expected!r}"
        )
    return actual


def _require_close(
    mapping: Mapping[str, Any],
    key: str,
    expected: float,
    *,
    where: str,
    atol: float = 1e-8,
) -> float:
    try:
        actual = float(mapping.get(key))
    except (TypeError, ValueError) as exc:
        raise producer.ProducerError(f"{where}.{key} is not numeric") from exc
    if not math.isfinite(actual) or not math.isclose(
        actual, expected, rel_tol=0.0, abs_tol=atol
    ):
        raise producer.ProducerError(
            f"{where}.{key} contract mismatch: {actual!r} != {expected!r}"
        )
    return actual


def validate_v11_generated_npz(
    npz_path: str | Path,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as archive:
        required = {
            "X",
            "y",
            "feature_noise_variance",
            "feature_noise_sigma",
            "feature_noise_family",
            "feature_source",
            "bnn_block_id",
            "corr_audit_feature_indices",
            "sampled_corr_audit_max",
            "core_exact_max_abs_corr",
            "scm_response_mean",
            "scm_response_std",
            "batch_seed",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise producer.ProducerError(
                f"V11 generated NPZ lacks required arrays: {missing}"
            )

        x = archive["X"]
        y = archive["y"]
        variance = archive["feature_noise_variance"]
        sigma = archive["feature_noise_sigma"]
        family = archive["feature_noise_family"]
        source = archive["feature_source"]
        block_id = archive["bnn_block_id"]
        audit_indices = archive["corr_audit_feature_indices"]
        sampled_corr = archive["sampled_corr_audit_max"]
        core_corr = archive["core_exact_max_abs_corr"]
        response_mean = archive["scm_response_mean"]
        response_std = archive["scm_response_std"]
        batch_seed = archive["batch_seed"]

        if x.ndim != 3 or y.ndim != 2 or x.shape[:2] != y.shape:
            raise producer.ProducerError(f"invalid V11 X/y shapes: {x.shape}, {y.shape}")
        if x.shape[0] != 1 or x.dtype != np.float16 or y.dtype != np.float32:
            raise producer.ProducerError(
                f"invalid V11 task batch/dtypes: X={x.shape}/{x.dtype}, y={y.shape}/{y.dtype}"
            )
        batch, n, p = x.shape
        feature_shape = (batch, p)
        expected_dtypes = {
            "feature_noise_variance": (variance, np.float32),
            "feature_noise_sigma": (sigma, np.float32),
            "feature_noise_family": (family, np.uint8),
            "feature_source": (source, np.uint8),
            "bnn_block_id": (block_id, np.int32),
        }
        for name, (value, dtype) in expected_dtypes.items():
            if value.shape != feature_shape or value.dtype != dtype:
                raise producer.ProducerError(
                    f"invalid {name}: shape={value.shape} dtype={value.dtype}"
                )
        audit_width = min(CORR_AUDIT_FEATURES, p)
        if audit_indices.shape != (batch, audit_width) or audit_indices.dtype != np.int64:
            raise producer.ProducerError("invalid V11 correlation-audit indices")
        for name, value in (
            ("sampled_corr_audit_max", sampled_corr),
            ("core_exact_max_abs_corr", core_corr),
            ("scm_response_mean", response_mean),
            ("scm_response_std", response_std),
        ):
            if value.shape != (batch,) or value.dtype != np.float32:
                raise producer.ProducerError(
                    f"invalid {name}: shape={value.shape} dtype={value.dtype}"
                )
        if batch_seed.shape != (batch,) or batch_seed.dtype != np.uint64:
            raise producer.ProducerError("invalid V11 batch seed certificate")
        if not all(
            np.isfinite(value).all()
            for value in (x, y, variance, sigma, sampled_corr, core_corr)
        ):
            raise producer.ProducerError("V11 shard contains nonfinite values")
        if np.any(variance < 0.0) or not np.allclose(
            sigma.astype(np.float64) ** 2,
            variance.astype(np.float64),
            rtol=3e-6,
            atol=3e-7,
        ):
            raise producer.ProducerError("V11 feature variance/sigma certificate failed")
        if np.any(family != 0):
            raise producer.ProducerError("V11 feature-noise family is not Gaussian")
        if not np.all(np.isin(source, (0, 1))):
            raise producer.ProducerError("V11 feature source id is invalid")
        core_features = min(p, SCM_CORE_CAP)
        if int(np.count_nonzero(source == 0)) != core_features:
            raise producer.ProducerError("V11 SCM-core feature count is invalid")
        if int(np.count_nonzero(source == 1)) != p - core_features:
            raise producer.ProducerError("V11 BNN-extension feature count is invalid")
        if np.any(block_id[source == 0] != -1) or np.any(block_id[source == 1] < 0):
            raise producer.ProducerError("V11 BNN block-id certificate is invalid")
        if np.any(sampled_corr < 0.0) or np.any(sampled_corr >= CORRELATION_THRESHOLD):
            raise producer.ProducerError("V11 sampled correlation audit failed")
        if np.any(core_corr < 0.0) or np.any(core_corr >= CORRELATION_THRESHOLD):
            raise producer.ProducerError("V11 exact SCM-core correlation certificate failed")

        realized_mean = y.astype(np.float64).mean(axis=1)
        realized_std = y.astype(np.float64).std(axis=1)
        if not np.allclose(response_mean, realized_mean, rtol=0.0, atol=2e-6):
            raise producer.ProducerError("V11 response mean certificate failed")
        if not np.allclose(response_std, realized_std, rtol=0.0, atol=2e-6):
            raise producer.ProducerError("V11 response std certificate failed")
        if np.any(np.abs(realized_mean) > 2e-6) or np.any(
            np.abs(realized_std - 1.0) > 2e-6
        ):
            raise producer.ProducerError("V11 response is not centered and unit scale")

        observed_max_corr = float(max(float(sampled_corr.max()), float(core_corr.max())))
        if int(entry.get("num_tasks", -1)) != batch:
            raise producer.ProducerError("V11 manifest task count disagrees with NPZ")
        if (int(entry.get("n", -1)), int(entry.get("p", -1))) != (n, p):
            raise producer.ProducerError("V11 manifest shape disagrees with NPZ")
        _require_close(
            entry,
            "task_sampled_corr_audit_max",
            float(sampled_corr.max()),
            where="manifest.entry",
            atol=1e-6,
        )
        _require_close(
            entry,
            "task_core_exact_max_abs_corr",
            float(core_corr.max()),
            where="manifest.entry",
            atol=1e-6,
        )

        return {
            "num_tasks": batch,
            "n": n,
            "p": p,
            "x_dtype": str(x.dtype),
            "y_dtype": str(y.dtype),
            "max_abs_corr": observed_max_corr,
            "corr_threshold_min": CORRELATION_THRESHOLD,
            "scm_response_mean_max_abs": float(np.max(np.abs(realized_mean))),
            "scm_response_std_min": float(np.min(realized_std)),
            "scm_response_std_max": float(np.max(realized_std)),
            "batch_seed_min": int(batch_seed.min()),
            "batch_seed_max": int(batch_seed.max()),
            "sampled_corr_audit_max": float(sampled_corr.max()),
            "core_exact_max_abs_corr": float(core_corr.max()),
            "feature_noise_variance_min": float(variance.min()),
            "feature_noise_variance_median": float(np.median(variance)),
            "feature_noise_variance_max": float(variance.max()),
            "feature_noise_variance_mean": float(variance.mean(dtype=np.float64)),
            "scm_core_features": core_features,
            "bnn_extension_features": p - core_features,
        }


def validate_v11_generator_contract(
    prior: str,
    manifest: Mapping[str, Any],
    entry: Mapping[str, Any],
    npz_summary: Mapping[str, Any],
) -> dict[str, Any]:
    if prior != "tabicl_v2":
        raise producer.ProducerError(f"unsupported V11 prior: {prior!r}")
    for key, expected in (
        ("generator", GENERATOR),
        ("prior", PRIOR_NAME),
        ("dataset_contract", DATASET_CONTRACT),
        ("num_datasets", 1),
    ):
        _require_exact(manifest, key, expected, where="manifest")

    settings = manifest.get("settings")
    if not isinstance(settings, Mapping):
        raise producer.ProducerError("V11 manifest.settings is not an object")
    settings_exact = {
        "shape_sampler": SHAPE_SAMPLER,
        "batch_size": 1,
        "n_min": N_MIN,
        "n_max": N_MAX,
        "main_n_min": 200,
        "small_n_max": 199,
        "small_n_probability": 0.10,
        "p_min": P_MIN,
        "p_max": P_MAX,
        "max_cells": MAX_CELLS,
        "p_median": 10_000.0,
        "p_log_sigma": 0.75,
        "raw_n_median": 5_700.0,
        "n_log_sigma": 0.70,
        "scm_core_cap": SCM_CORE_CAP,
        "core_oversample_factor": 1.5,
        "core_candidate_cap": 3_072,
        "core_correlation_threshold_exclusive": CORRELATION_THRESHOLD,
        "core_correlation_certification": CORE_CORRELATION_CERTIFICATE,
        "core_corr_device": "cpu",
        "bnn_contract": BNN_CONTRACT,
        "bnn_hidden_dim": BNN_HIDDEN_DIM,
        "bnn_output_block": BNN_OUTPUT_BLOCK,
        "bnn_parent_min": BNN_PARENT_MIN,
        "bnn_parent_max": BNN_PARENT_MAX,
        "feature_noise_contract": FEATURE_NOISE_CONTRACT,
        "noise_variance_rate": NOISE_VARIANCE_RATE,
        "correlation_audit_contract": CORRELATION_AUDIT_CONTRACT,
        "corr_audit_features": CORR_AUDIT_FEATURES,
        "target_contract": TARGET_CONTRACT,
        "official_tabicl_commit": TABICL_COMMIT,
        "tabicl_function_types": "default",
        "data_policy": DATA_POLICY,
    }
    for key, expected in settings_exact.items():
        actual = settings.get(key)
        if isinstance(expected, float):
            _require_close(settings, key, expected, where="manifest.settings")
        elif actual != expected:
            raise producer.ProducerError(
                f"manifest.settings.{key} contract mismatch: {actual!r} != {expected!r}"
            )

    entry_exact = {
        "num_tasks": 1,
        "shape_sampler": SHAPE_SAMPLER,
        "prior": PRIOR_NAME,
        "dataset_contract": DATASET_CONTRACT,
        "data_policy": DATA_POLICY,
        "target_contract": TARGET_CONTRACT,
        "correlation_audit_contract": CORRELATION_AUDIT_CONTRACT,
        "core_exact_correlation_certificate": True,
    }
    for key, expected in entry_exact.items():
        _require_exact(entry, key, expected, where="manifest.entry")
    n = int(entry.get("n", -1))
    p = int(entry.get("p", -1))
    if not (N_MIN <= n <= N_MAX and P_MIN <= p <= P_MAX and n * p <= MAX_CELLS):
        raise producer.ProducerError(f"V11 shape outside contract: N={n}, P={p}")
    if (n, p) != (int(npz_summary["n"]), int(npz_summary["p"])):
        raise producer.ProducerError("V11 semantic shape disagrees with NPZ")
    _require_exact(
        entry,
        "full_pair_exact_certificate",
        p <= SCM_CORE_CAP,
        where="manifest.entry",
    )

    metadata = entry.get("task_metadata")
    if not isinstance(metadata, list) or len(metadata) != 1 or not isinstance(
        metadata[0], Mapping
    ):
        raise producer.ProducerError("V11 entry needs exactly one task metadata object")
    task = metadata[0]
    expected_mode = (
        "direct_qfiltered_scm"
        if p <= SCM_CORE_CAP
        else "noisy_qfiltered_scm2048_plus_noisy_bnn128"
    )
    for key, expected in (
        ("task_index", 0),
        ("mode", expected_mode),
        ("n", n),
        ("p", p),
        ("scm_core_features", min(p, SCM_CORE_CAP)),
        ("bnn_extension_features", max(0, p - SCM_CORE_CAP)),
        ("core_correlation_certification", CORE_CORRELATION_CERTIFICATE),
        ("core_added_noise", True),
        ("bnn_contract", BNN_CONTRACT),
        ("feature_noise_contract", FEATURE_NOISE_CONTRACT),
        ("full_pair_exact_certificate", p <= SCM_CORE_CAP),
        ("final_sampled_corr_audit_passed", True),
        ("correlation_audit_contract", CORRELATION_AUDIT_CONTRACT),
    ):
        _require_exact(task, key, expected, where="task")
    _require_close(
        task,
        "core_correlation_threshold_exclusive",
        CORRELATION_THRESHOLD,
        where="task",
    )
    _require_close(
        task,
        "core_exact_max_abs_corr",
        float(npz_summary["core_exact_max_abs_corr"]),
        where="task",
        atol=1e-6,
    )
    _require_close(task, "target_final_mean", 0.0, where="task", atol=2e-6)
    _require_close(task, "target_final_std", 1.0, where="task", atol=2e-6)

    source_counts = task.get("feature_source_counts")
    if source_counts != {
        "scm_core": int(npz_summary["scm_core_features"]),
        "bnn_extension": int(npz_summary["bnn_extension_features"]),
    }:
        raise producer.ProducerError("V11 task feature-source counts are inconsistent")
    variance_summary = task.get("feature_noise_variance_summary")
    if not isinstance(variance_summary, Mapping):
        raise producer.ProducerError("V11 task lacks feature-noise summary")
    for key, expected in (
        ("min", npz_summary["feature_noise_variance_min"]),
        ("median", npz_summary["feature_noise_variance_median"]),
        ("max", npz_summary["feature_noise_variance_max"]),
        ("mean", npz_summary["feature_noise_variance_mean"]),
    ):
        _require_close(
            variance_summary,
            key,
            float(expected),
            where="task.feature_noise_variance_summary",
            atol=max(2e-6, abs(float(expected)) * 2e-6),
        )

    bnn_blocks = task.get("bnn_blocks")
    if not isinstance(bnn_blocks, list):
        raise producer.ProducerError("V11 task bnn_blocks is not a list")
    expected_blocks = math.ceil(max(0, p - SCM_CORE_CAP) / BNN_OUTPUT_BLOCK)
    if len(bnn_blocks) != expected_blocks:
        raise producer.ProducerError("V11 BNN block count is inconsistent")
    allowed_activations = {"gelu", "tanh", "relu", "sigmoid", "sin_pi"}
    for index, block in enumerate(bnn_blocks):
        if not isinstance(block, Mapping):
            raise producer.ProducerError(f"V11 BNN block {index} is not an object")
        output_features = int(block.get("output_features", -1))
        parent_count = int(block.get("parent_count", -1))
        if not 1 <= output_features <= BNN_OUTPUT_BLOCK:
            raise producer.ProducerError("V11 BNN output block width is invalid")
        if not BNN_PARENT_MIN <= parent_count <= BNN_PARENT_MAX:
            raise producer.ProducerError("V11 BNN parent count is invalid")
        if int(block.get("hidden_dim", -1)) != BNN_HIDDEN_DIM:
            raise producer.ProducerError("V11 BNN hidden dimension is invalid")
        if int(block.get("depth", -1)) not in (1, 2):
            raise producer.ProducerError("V11 BNN depth is invalid")
        if block.get("activation") not in allowed_activations:
            raise producer.ProducerError("V11 BNN activation is invalid")
        noise = block.get("noise")
        if not isinstance(noise, Mapping):
            raise producer.ProducerError("V11 BNN block lacks noise metadata")
        _require_exact(
            noise,
            "noise_variance_distribution",
            "Exponential(rate=1)",
            where=f"task.bnn_blocks[{index}].noise",
        )
        _require_exact(
            noise,
            "error_distribution",
            "Normal(0, feature_noise_variance)",
            where=f"task.bnn_blocks[{index}].noise",
        )

    return {
        "contract": DATASET_CONTRACT,
        "prior": prior,
        "n": n,
        "p": p,
        "requested_p": p,
        "mode": expected_mode,
        "target_contract": TARGET_CONTRACT,
        "feature_noise_contract": FEATURE_NOISE_CONTRACT,
        "noise_variance_distribution": "Exponential(rate=1)",
        "noise_error_distribution": "Normal(0, feature_noise_variance)",
        "scm_core_features": int(npz_summary["scm_core_features"]),
        "bnn_extension_features": int(npz_summary["bnn_extension_features"]),
        "bnn_blocks": expected_blocks,
        "full_pair_exact_certificate": p <= SCM_CORE_CAP,
        "core_exact_max_abs_corr": float(npz_summary["core_exact_max_abs_corr"]),
        "sampled_corr_audit_max": float(npz_summary["sampled_corr_audit_max"]),
        "scm_response_mean_max_abs": float(
            npz_summary["scm_response_mean_max_abs"]
        ),
        "scm_response_std_min": float(npz_summary["scm_response_std_min"]),
        "scm_response_std_max": float(npz_summary["scm_response_std_max"]),
    }


def configure_v11_runtime() -> None:
    producer.PRIOR_KEYS = PRIORS
    producer.PIPELINE_VERSION = PIPELINE_VERSION
    producer.SOURCE_VERSION = SOURCE_VERSION
    producer.PRIOR_CONTRACTS["tabicl_v2"] = dict(V11_PRIOR_CONTRACT)
    producer.validate_generated_npz = validate_v11_generated_npz
    producer.validate_generator_contract = validate_v11_generator_contract
    producer.CatalogStore = TabICLV11CatalogStore
    fixed.CatalogStore = TabICLV11CatalogStore
    fixed._unadopted_staging_npz_bytes = _bootstrap_aware_staging_npz_bytes


def main() -> int:
    configure_v11_runtime()
    return fixed.main()


if __name__ == "__main__":
    raise SystemExit(main())
