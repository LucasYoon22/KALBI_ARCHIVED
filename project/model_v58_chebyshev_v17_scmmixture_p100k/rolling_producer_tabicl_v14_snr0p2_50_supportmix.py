#!/usr/bin/env python3
"""Rolling producer for V14 50/50 TabICL/linear-outcome tasks."""

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
    "v24_kalbi_tabiclv2_x_mixed50_tabicl_linear_pn50_strictn3_"
    "snr0p2to50_y_v14_rolling100gb_v1"
)
SOURCE_VERSION = "v24_tabiclv2_v14_supportmix_snr0p2_50_cpu_rolling_producer_v1"
CURATOR_VERSION = "v24_tabiclv2_v14_supportmix_snr0p2_50_rolling100gb_curator_v1"

GENERATOR = (
    "generate_tabiclv2_scm2048_qfilter_bnn128_"
    "mixedlinear_supportmix_v14.py"
)
PRIOR_NAME = (
    "tabiclv2_graph_scm_x_expvarnoise_qfiltered_core2048_then_bnn128_"
    "mixed50_official_or_controlled_linear_supportmix_pn_strictn3_"
    "snr_loguniform_0p2_50_y_v3"
)
DATASET_CONTRACT = (
    "tabiclv2_graph_scm2048_qfilter_bnn128_expvarnoise_"
    "mixed50_tabicl_linear_supportmix_pn_strictn3_snr0p2to50_v14"
)
DATA_POLICY = (
    "v14_tabicl_x_qfilter_noisy_core2048_bnn128_expvarnoise_"
    "mixed50_outcome_linear_pn50_strictn3_snr0p2to50_"
    "sampled_final_audit_v1"
)
TARGET_CONTRACT = (
    "mixed50_official_tabicl_or_controlled_linear_supportmix_pn50_"
    "strictn3_snr_loguniform_0p2_50_y_"
    "qualityfilter_center_scale_clip10_rescale_v3"
)
TABICL_TARGET_CONTRACT = (
    "official_tabiclv2_graph_scm_y_qualityfilter_center_scale_clip10_rescale_v2"
)
LINEAR_TARGET_CONTRACT = (
    "controlled_linear_y_supportmix_pn50_strictn3_"
    "snr_loguniform_0p2_50_errorfamily_"
    "qualityfilter_center_scale_clip10_rescale_v3"
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
OUTCOME_SOURCE_TABICL = 0
OUTCOME_SOURCE_LINEAR = 1
OUTCOME_SOURCE_NAMES = {
    OUTCOME_SOURCE_TABICL: "tabicl",
    OUTCOME_SOURCE_LINEAR: "linear",
}
SUPPORT_REFERENCE_P = 0
SUPPORT_REFERENCE_N = 1
SUPPORT_REFERENCE_NAMES = {
    SUPPORT_REFERENCE_P: "p_based",
    SUPPORT_REFERENCE_N: "n_based",
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
    "log_quantile_1",
    "log_quantile_2",
    "log_quantile_3",
    "log_quantile_4",
    "log_quantile_5",
)
LINEAR_SNR_MIN = 0.2
LINEAR_SNR_MAX = 50.0
LINEAR_SNR_DISTRIBUTION = "log_uniform"
LINEAR_SNR_CONTRACT = "linear_target_snr_loguniform_0p2_50_v1"
SNR_TELEMETRY_BIN_EDGES = tuple(
    float(value)
    for value in np.geomspace(
        LINEAR_SNR_MIN,
        LINEAR_SNR_MAX,
        len(SNR_REGIME_NAMES) + 1,
    )
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

V14_PRIOR_CONTRACT = {
    "generator": GENERATOR,
    "prior_name": PRIOR_NAME,
    "dataset_contract": DATASET_CONTRACT,
    "data_policy": DATA_POLICY,
    "target_contract": TARGET_CONTRACT,
    "official_core_prior": "TabICLv2 PriorDataset prior_type=graph_scm",
    "target_source": (
        "50pct official TabICLv2 response, 50pct controlled linear response "
        "with target SNR log-uniform on [0.2, 50]"
    ),
    "direct_mode": "direct_qfiltered_scm",
    "highp_mode": "noisy_qfiltered_scm2048_plus_noisy_bnn128",
    "commit": TABICL_COMMIT,
}

_FIXED_UNADOPTED_STAGING_NPZ_BYTES = fixed._unadopted_staging_npz_bytes


def tabicl_v14_policy(*, world_size: int = 8) -> CapacityPolicy:
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


class TabICLV14CatalogStore(CatalogStore):
    def __init__(
        self,
        control_root: Any,
        data_root: Any,
        policy: CapacityPolicy | None = None,
        *,
        priors: Any = PRIORS,
    ) -> None:
        if tuple(priors) != PRIORS:
            raise ValueError(f"V14 TabICL store received priors={tuple(priors)!r}")
        super().__init__(
            control_root,
            data_root,
            policy=policy or tabicl_v14_policy(),
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


def validate_v14_generated_npz(
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
        missing = sorted(required - set(archive.files))
        if missing:
            raise producer.ProducerError(
                f"V14 generated NPZ lacks required arrays: {missing}"
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
        linear_coefficient_family_id = archive["linear_coefficient_family_id"]
        linear_snr_regime_id = archive["linear_snr_regime_id"]
        linear_error_family_id = archive["linear_error_family_id"]
        linear_target_snr = archive["linear_target_snr"]
        linear_realized_snr = archive["linear_realized_snr"]

        if x.ndim != 3 or y.ndim != 2 or x.shape[:2] != y.shape:
            raise producer.ProducerError(f"invalid V14 X/y shapes: {x.shape}, {y.shape}")
        if x.shape[0] != 1 or x.dtype != np.float16 or y.dtype != np.float32:
            raise producer.ProducerError(
                f"invalid V14 task batch/dtypes: X={x.shape}/{x.dtype}, y={y.shape}/{y.dtype}"
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
            raise producer.ProducerError("invalid V14 correlation-audit indices")
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
            raise producer.ProducerError("invalid V14 batch seed certificate")
        outcome_dtypes = {
            "outcome_source_id": (outcome_source_id, np.uint8),
            "linear_support_size": (linear_support_size, np.int64),
            "linear_support_fraction": (linear_support_fraction, np.float32),
            "linear_support_fraction_of_n": (
                linear_support_fraction_of_n,
                np.float32,
            ),
            "linear_support_regime_id": (linear_support_regime_id, np.uint8),
            "linear_support_reference_id": (
                linear_support_reference_id,
                np.uint8,
            ),
            "linear_coefficient_family_id": (
                linear_coefficient_family_id,
                np.uint8,
            ),
            "linear_snr_regime_id": (linear_snr_regime_id, np.uint8),
            "linear_error_family_id": (linear_error_family_id, np.uint8),
            "linear_target_snr": (linear_target_snr, np.float32),
            "linear_realized_snr": (linear_realized_snr, np.float32),
        }
        for name, (value, dtype) in outcome_dtypes.items():
            if value.shape != (batch,) or value.dtype != dtype:
                raise producer.ProducerError(
                    f"invalid {name}: shape={value.shape} dtype={value.dtype}"
                )
        if not all(
            np.isfinite(value).all()
            for value in (
                x,
                y,
                variance,
                sigma,
                sampled_corr,
                core_corr,
                linear_support_fraction,
                linear_support_fraction_of_n,
                linear_target_snr,
                linear_realized_snr,
            )
        ):
            raise producer.ProducerError("V14 shard contains nonfinite values")
        if np.any(variance < 0.0) or not np.allclose(
            sigma.astype(np.float64) ** 2,
            variance.astype(np.float64),
            rtol=3e-6,
            atol=3e-7,
        ):
            raise producer.ProducerError("V14 feature variance/sigma certificate failed")
        if np.any(family != 0):
            raise producer.ProducerError("V14 feature-noise family is not Gaussian")
        if not np.all(np.isin(source, (0, 1))):
            raise producer.ProducerError("V14 feature source id is invalid")
        core_features = min(p, SCM_CORE_CAP)
        if int(np.count_nonzero(source == 0)) != core_features:
            raise producer.ProducerError("V14 SCM-core feature count is invalid")
        if int(np.count_nonzero(source == 1)) != p - core_features:
            raise producer.ProducerError("V14 BNN-extension feature count is invalid")
        if np.any(block_id[source == 0] != -1) or np.any(block_id[source == 1] < 0):
            raise producer.ProducerError("V14 BNN block-id certificate is invalid")
        if np.any(sampled_corr < 0.0) or np.any(sampled_corr >= CORRELATION_THRESHOLD):
            raise producer.ProducerError("V14 sampled correlation audit failed")
        if np.any(core_corr < 0.0) or np.any(core_corr >= CORRELATION_THRESHOLD):
            raise producer.ProducerError("V14 exact SCM-core correlation certificate failed")
        if not np.all(
            np.isin(
                outcome_source_id,
                (OUTCOME_SOURCE_TABICL, OUTCOME_SOURCE_LINEAR),
            )
        ):
            raise producer.ProducerError("V14 outcome-source id is invalid")
        tabicl_mask = outcome_source_id == OUTCOME_SOURCE_TABICL
        linear_mask = outcome_source_id == OUTCOME_SOURCE_LINEAR
        if np.any(linear_support_size[tabicl_mask] != 0) or np.any(
            linear_support_fraction[tabicl_mask] != 0.0
        ) or np.any(
            linear_support_fraction_of_n[tabicl_mask] != 0.0
        ):
            raise producer.ProducerError("V14 TabICL outcome has linear support metadata")
        for name, value in (
            ("support regime", linear_support_regime_id),
            ("support reference", linear_support_reference_id),
            ("coefficient family", linear_coefficient_family_id),
            ("SNR regime", linear_snr_regime_id),
            ("error family", linear_error_family_id),
        ):
            if np.any(value[tabicl_mask] != 255):
                raise producer.ProducerError(
                    f"V14 TabICL outcome has non-sentinel {name}"
                )
        if np.any(linear_target_snr[tabicl_mask] != -1.0) or np.any(
            linear_realized_snr[tabicl_mask] != -1.0
        ):
            raise producer.ProducerError("V14 TabICL outcome has linear SNR metadata")
        if np.any(linear_support_size[linear_mask] < 1) or np.any(
            linear_support_size[linear_mask] > p
        ):
            raise producer.ProducerError("V14 linear support size is invalid")
        if np.any(3 * linear_support_size[linear_mask] >= n):
            raise producer.ProducerError(
                "V14 linear support violates strict k < N/3"
            )
        expected_support_fraction = (
            linear_support_size[linear_mask].astype(np.float64) / float(p)
        )
        if not np.allclose(
            linear_support_fraction[linear_mask],
            expected_support_fraction,
            rtol=2e-6,
            atol=2e-8,
        ):
            raise producer.ProducerError("V14 linear support fraction is inconsistent")
        expected_support_fraction_of_n = (
            linear_support_size[linear_mask].astype(np.float64) / float(n)
        )
        if not np.allclose(
            linear_support_fraction_of_n[linear_mask],
            expected_support_fraction_of_n,
            rtol=2e-6,
            atol=2e-8,
        ):
            raise producer.ProducerError(
                "V14 linear N-support fraction is inconsistent"
            )
        if np.any(
            ~np.isin(
                linear_support_reference_id[linear_mask],
                (SUPPORT_REFERENCE_P, SUPPORT_REFERENCE_N),
            )
        ):
            raise producer.ProducerError(
                "V14 linear support reference id is invalid"
            )
        linear_id_bounds = (
            ("support regime", linear_support_regime_id, len(SUPPORT_REGIME_NAMES)),
            (
                "coefficient family",
                linear_coefficient_family_id,
                len(COEFFICIENT_FAMILY_NAMES),
            ),
            ("SNR regime", linear_snr_regime_id, len(SNR_REGIME_NAMES)),
            ("error family", linear_error_family_id, len(ERROR_FAMILY_NAMES)),
        )
        for name, value, upper in linear_id_bounds:
            if np.any(value[linear_mask] >= upper):
                raise producer.ProducerError(f"V14 linear {name} id is invalid")
        if np.any(linear_target_snr[linear_mask] <= 0.0) or np.any(
            linear_realized_snr[linear_mask] <= 0.0
        ):
            raise producer.ProducerError("V14 linear SNR is not positive")
        if linear_mask.any():
            target_snr = linear_target_snr[linear_mask].astype(np.float64)
            realized_snr = linear_realized_snr[linear_mask].astype(np.float64)
            bound_tolerance = 2e-6
            if np.any(target_snr < LINEAR_SNR_MIN - bound_tolerance) or np.any(
                target_snr > LINEAR_SNR_MAX + bound_tolerance
            ):
                raise producer.ProducerError(
                    "V14 target linear SNR escaped [0.2, 50]"
                )
            if np.any(realized_snr < LINEAR_SNR_MIN - bound_tolerance) or np.any(
                realized_snr > LINEAR_SNR_MAX + bound_tolerance
            ):
                raise producer.ProducerError(
                    "V14 realized linear SNR escaped [0.2, 50]"
                )
            expected_regime = np.searchsorted(
                np.asarray(SNR_TELEMETRY_BIN_EDGES[1:-1]),
                target_snr,
                side="right",
            ).astype(np.uint8)
            if not np.array_equal(
                linear_snr_regime_id[linear_mask], expected_regime
            ):
                raise producer.ProducerError(
                    "V14 linear SNR telemetry bin is inconsistent"
                )
            log_snr_error = np.abs(
                np.log(realized_snr / target_snr)
            )
            if np.any(log_snr_error > 2e-5):
                raise producer.ProducerError(
                    "V14 realized linear SNR disagrees with target"
                )

        realized_mean = y.astype(np.float64).mean(axis=1)
        realized_std = y.astype(np.float64).std(axis=1)
        if not np.allclose(response_mean, realized_mean, rtol=0.0, atol=2e-6):
            raise producer.ProducerError("V14 response mean certificate failed")
        if not np.allclose(response_std, realized_std, rtol=0.0, atol=2e-6):
            raise producer.ProducerError("V14 response std certificate failed")
        if np.any(np.abs(realized_mean) > 2e-6) or np.any(
            np.abs(realized_std - 1.0) > 2e-6
        ):
            raise producer.ProducerError("V14 response is not centered and unit scale")

        observed_max_corr = float(max(float(sampled_corr.max()), float(core_corr.max())))
        if int(entry.get("num_tasks", -1)) != batch:
            raise producer.ProducerError("V14 manifest task count disagrees with NPZ")
        if (int(entry.get("n", -1)), int(entry.get("p", -1))) != (n, p):
            raise producer.ProducerError("V14 manifest shape disagrees with NPZ")
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
        entry_outcome_id = int(entry.get("outcome_source_id", -1))
        npz_outcome_id = int(outcome_source_id[0])
        if entry_outcome_id != npz_outcome_id:
            raise producer.ProducerError(
                "V14 manifest outcome source disagrees with NPZ"
            )
        _require_exact(
            entry,
            "outcome_source",
            OUTCOME_SOURCE_NAMES[npz_outcome_id],
            where="manifest.entry",
        )
        if int(entry.get("linear_support_size", -1)) != int(
            linear_support_size[0]
        ):
            raise producer.ProducerError(
                "V14 manifest linear support size disagrees with NPZ"
            )
        for key, expected in (
            ("linear_support_fraction", linear_support_fraction[0]),
            (
                "linear_support_fraction_of_n",
                linear_support_fraction_of_n[0],
            ),
            ("linear_target_snr", linear_target_snr[0]),
            ("linear_realized_snr", linear_realized_snr[0]),
        ):
            _require_close(
                entry,
                key,
                float(expected),
                where="manifest.entry",
                atol=max(2e-6, abs(float(expected)) * 2e-6),
            )
        if int(entry.get("linear_support_reference_id", -1)) != int(
            linear_support_reference_id[0]
        ):
            raise producer.ProducerError(
                "V14 manifest support reference disagrees with NPZ"
            )
        expected_reference_name = (
            SUPPORT_REFERENCE_NAMES[int(linear_support_reference_id[0])]
            if npz_outcome_id == OUTCOME_SOURCE_LINEAR
            else "not_applicable"
        )
        _require_exact(
            entry,
            "linear_support_reference",
            expected_reference_name,
            where="manifest.entry",
        )
        _require_exact(
            entry,
            "linear_support_strict_n_third_verified",
            True,
            where="manifest.entry",
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
            "outcome_source_id": npz_outcome_id,
            "outcome_source": OUTCOME_SOURCE_NAMES[npz_outcome_id],
            "linear_support_size": int(linear_support_size[0]),
            "linear_support_fraction": float(linear_support_fraction[0]),
            "linear_support_fraction_of_n": float(
                linear_support_fraction_of_n[0]
            ),
            "linear_support_regime_id": int(linear_support_regime_id[0]),
            "linear_support_reference_id": int(
                linear_support_reference_id[0]
            ),
            "linear_coefficient_family_id": int(
                linear_coefficient_family_id[0]
            ),
            "linear_snr_regime_id": int(linear_snr_regime_id[0]),
            "linear_error_family_id": int(linear_error_family_id[0]),
            "linear_target_snr": float(linear_target_snr[0]),
            "linear_realized_snr": float(linear_realized_snr[0]),
        }


def validate_v14_generator_contract(
    prior: str,
    manifest: Mapping[str, Any],
    entry: Mapping[str, Any],
    npz_summary: Mapping[str, Any],
) -> dict[str, Any]:
    if prior != "tabicl_v2":
        raise producer.ProducerError(f"unsupported V14 prior: {prior!r}")
    for key, expected in (
        ("generator", GENERATOR),
        ("prior", PRIOR_NAME),
        ("dataset_contract", DATASET_CONTRACT),
        ("num_datasets", 1),
    ):
        _require_exact(manifest, key, expected, where="manifest")

    settings = manifest.get("settings")
    if not isinstance(settings, Mapping):
        raise producer.ProducerError("V14 manifest.settings is not an object")
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
        "outcome_mixture_contract": (
            "batch_seed_parity_exact_alternation_50pct_tabicl_"
            "50pct_controlled_linear_v2"
        ),
        "outcome_source_map": {"0": "tabicl", "1": "linear"},
        "tabicl_target_contract": TABICL_TARGET_CONTRACT,
        "linear_target_contract": LINEAR_TARGET_CONTRACT,
        "linear_support_reference_assignment": (
            "bit1(batch_seed + task_index), exact alternation among "
            "linear tasks: 50pct P-referenced conditional on k<N/3 and "
            "50pct N-referenced"
        ),
        "linear_support_reference_probabilities": {
            "p_based": 0.5,
            "n_based": 0.5,
        },
        "linear_support_strict_constraint": "k < N/3",
        "linear_support_integer_cap": "min(P, floor((N-1)/3))",
        "p_based_support_policy": (
            "original V12 P-scale mixture conditioned by rejection on "
            "k<N/3; no clipping or cap pile-up"
        ),
        "n_based_support_policy": (
            "same regime probabilities on N scale; very_dense is "
            "integer-uniform from ceil(0.2N) through floor((N-1)/3)"
        ),
        "linear_support_regimes": {
            "ultra_sparse": 0.15,
            "sparse": 0.20,
            "moderate": 0.25,
            "large": 0.25,
            "very_dense": 0.15,
        },
        "linear_coefficient_families": {
            "gaussian": 0.30,
            "laplace": 0.20,
            "rademacher_loguniform": 0.20,
            "student_t": 0.15,
            "two_scale_gaussian": 0.15,
        },
        "linear_snr_contract": LINEAR_SNR_CONTRACT,
        "linear_snr_distribution": LINEAR_SNR_DISTRIBUTION,
        "linear_snr_min": LINEAR_SNR_MIN,
        "linear_snr_max": LINEAR_SNR_MAX,
        "linear_snr_telemetry_bins": {
            name: [
                SNR_TELEMETRY_BIN_EDGES[index],
                SNR_TELEMETRY_BIN_EDGES[index + 1],
            ]
            for index, name in enumerate(SNR_REGIME_NAMES)
        },
        "linear_error_families": {
            "gaussian": 0.25,
            "student_t": 0.20,
            "laplace": 0.15,
            "logistic": 0.10,
            "centered_lognormal": 0.10,
            "contaminated_gaussian": 0.10,
            "heteroskedastic_gaussian": 0.10,
        },
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
        raise producer.ProducerError(f"V14 shape outside contract: N={n}, P={p}")
    if (n, p) != (int(npz_summary["n"]), int(npz_summary["p"])):
        raise producer.ProducerError("V14 semantic shape disagrees with NPZ")
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
        raise producer.ProducerError("V14 entry needs exactly one task metadata object")
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
        ("mixed_target_contract", TARGET_CONTRACT),
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
    outcome_source_id = int(npz_summary["outcome_source_id"])
    outcome_source = OUTCOME_SOURCE_NAMES[outcome_source_id]
    for key, expected in (
        ("outcome_source", outcome_source),
        ("outcome_source_id", outcome_source_id),
        ("outcome_source_assignment", "parity(batch_seed + task_index)"),
        (
            "linear_support_reference_assignment",
            (
                "bit1(batch_seed + task_index)"
                if outcome_source_id == OUTCOME_SOURCE_LINEAR
                else "not_applicable"
            ),
        ),
    ):
        _require_exact(task, key, expected, where="task")
    expected_target_contract = (
        TABICL_TARGET_CONTRACT
        if outcome_source_id == OUTCOME_SOURCE_TABICL
        else LINEAR_TARGET_CONTRACT
    )
    _require_exact(
        task,
        "target_contract",
        expected_target_contract,
        where="task",
    )
    if int(task.get("linear_support_size", -1)) != int(
        npz_summary["linear_support_size"]
    ):
        raise producer.ProducerError("V14 task linear support size is inconsistent")
    for key, expected in (
        ("linear_support_fraction", npz_summary["linear_support_fraction"]),
        (
            "linear_support_fraction_of_n",
            npz_summary["linear_support_fraction_of_n"],
        ),
        ("linear_target_snr", npz_summary["linear_target_snr"]),
        ("linear_realized_snr", npz_summary["linear_realized_snr"]),
    ):
        _require_close(
            task,
            key,
            float(expected),
            where="task",
            atol=max(2e-6, abs(float(expected)) * 2e-6),
        )
    for key, summary_key in (
        ("linear_support_regime_id", "linear_support_regime_id"),
        ("linear_support_reference_id", "linear_support_reference_id"),
        ("linear_coefficient_family_id", "linear_coefficient_family_id"),
        ("linear_snr_regime_id", "linear_snr_regime_id"),
        ("linear_error_family_id", "linear_error_family_id"),
    ):
        _require_exact(task, key, int(npz_summary[summary_key]), where="task")
    if outcome_source_id == OUTCOME_SOURCE_LINEAR:
        support_reference_id = int(
            npz_summary["linear_support_reference_id"]
        )
        expected_reference_id = (
            int(npz_summary["batch_seed_min"]) >> 1
        ) & 1
        if support_reference_id != expected_reference_id:
            raise producer.ProducerError(
                "V14 support-reference assignment is not deterministic"
            )
        _require_exact(
            task,
            "linear_support_reference",
            SUPPORT_REFERENCE_NAMES[support_reference_id],
            where="task",
        )
        _require_exact(
            task,
            "linear_support_strict_n_third_verified",
            True,
            where="task",
        )
        _require_exact(
            task,
            "linear_support_max_allowed",
            min(p, (n - 1) // 3),
            where="task",
        )
        if 3 * int(npz_summary["linear_support_size"]) >= n:
            raise producer.ProducerError(
                "V14 task violates strict k < N/3"
            )
        _require_exact(
            task,
            "linear_snr_contract",
            LINEAR_SNR_CONTRACT,
            where="task",
        )
        _require_exact(
            task,
            "linear_snr_distribution",
            LINEAR_SNR_DISTRIBUTION,
            where="task",
        )
        _require_close(
            task,
            "linear_snr_min",
            LINEAR_SNR_MIN,
            where="task",
        )
        _require_close(
            task,
            "linear_snr_max",
            LINEAR_SNR_MAX,
            where="task",
        )
        for key, names, summary_key in (
            (
                "linear_support_regime",
                SUPPORT_REGIME_NAMES,
                "linear_support_regime_id",
            ),
            (
                "linear_coefficient_family",
                COEFFICIENT_FAMILY_NAMES,
                "linear_coefficient_family_id",
            ),
            ("linear_snr_regime", SNR_REGIME_NAMES, "linear_snr_regime_id"),
            (
                "linear_error_family",
                ERROR_FAMILY_NAMES,
                "linear_error_family_id",
            ),
        ):
            _require_exact(
                task,
                key,
                names[int(npz_summary[summary_key])],
                where="task",
            )

    source_counts = task.get("feature_source_counts")
    if source_counts != {
        "scm_core": int(npz_summary["scm_core_features"]),
        "bnn_extension": int(npz_summary["bnn_extension_features"]),
    }:
        raise producer.ProducerError("V14 task feature-source counts are inconsistent")
    variance_summary = task.get("feature_noise_variance_summary")
    if not isinstance(variance_summary, Mapping):
        raise producer.ProducerError("V14 task lacks feature-noise summary")
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
        raise producer.ProducerError("V14 task bnn_blocks is not a list")
    expected_blocks = math.ceil(max(0, p - SCM_CORE_CAP) / BNN_OUTPUT_BLOCK)
    if len(bnn_blocks) != expected_blocks:
        raise producer.ProducerError("V14 BNN block count is inconsistent")
    allowed_activations = {"gelu", "tanh", "relu", "sigmoid", "sin_pi"}
    for index, block in enumerate(bnn_blocks):
        if not isinstance(block, Mapping):
            raise producer.ProducerError(f"V14 BNN block {index} is not an object")
        output_features = int(block.get("output_features", -1))
        parent_count = int(block.get("parent_count", -1))
        if not 1 <= output_features <= BNN_OUTPUT_BLOCK:
            raise producer.ProducerError("V14 BNN output block width is invalid")
        if not BNN_PARENT_MIN <= parent_count <= BNN_PARENT_MAX:
            raise producer.ProducerError("V14 BNN parent count is invalid")
        if int(block.get("hidden_dim", -1)) != BNN_HIDDEN_DIM:
            raise producer.ProducerError("V14 BNN hidden dimension is invalid")
        if int(block.get("depth", -1)) not in (1, 2):
            raise producer.ProducerError("V14 BNN depth is invalid")
        if block.get("activation") not in allowed_activations:
            raise producer.ProducerError("V14 BNN activation is invalid")
        noise = block.get("noise")
        if not isinstance(noise, Mapping):
            raise producer.ProducerError("V14 BNN block lacks noise metadata")
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
        "task_target_contract": expected_target_contract,
        "outcome_source": outcome_source,
        "outcome_source_id": outcome_source_id,
        "linear_support_size": int(npz_summary["linear_support_size"]),
        "linear_support_fraction": float(
            npz_summary["linear_support_fraction"]
        ),
        "linear_support_fraction_of_n": float(
            npz_summary["linear_support_fraction_of_n"]
        ),
        "linear_support_regime_id": int(
            npz_summary["linear_support_regime_id"]
        ),
        "linear_support_reference_id": int(
            npz_summary["linear_support_reference_id"]
        ),
        "linear_support_reference": (
            SUPPORT_REFERENCE_NAMES[
                int(npz_summary["linear_support_reference_id"])
            ]
            if outcome_source_id == OUTCOME_SOURCE_LINEAR
            else "not_applicable"
        ),
        "linear_support_strict_n_third_verified": True,
        "linear_coefficient_family_id": int(
            npz_summary["linear_coefficient_family_id"]
        ),
        "linear_snr_regime_id": int(npz_summary["linear_snr_regime_id"]),
        "linear_error_family_id": int(npz_summary["linear_error_family_id"]),
        "linear_target_snr": float(npz_summary["linear_target_snr"]),
        "linear_realized_snr": float(npz_summary["linear_realized_snr"]),
        "linear_snr_contract": (
            LINEAR_SNR_CONTRACT
            if outcome_source_id == OUTCOME_SOURCE_LINEAR
            else "not_applicable"
        ),
        "linear_snr_distribution": (
            LINEAR_SNR_DISTRIBUTION
            if outcome_source_id == OUTCOME_SOURCE_LINEAR
            else "not_applicable"
        ),
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


def configure_v14_runtime() -> None:
    producer.PRIOR_KEYS = PRIORS
    producer.PIPELINE_VERSION = PIPELINE_VERSION
    producer.SOURCE_VERSION = SOURCE_VERSION
    producer.PRIOR_CONTRACTS["tabicl_v2"] = dict(V14_PRIOR_CONTRACT)
    producer.validate_generated_npz = validate_v14_generated_npz
    producer.validate_generator_contract = validate_v14_generator_contract
    producer.CatalogStore = TabICLV14CatalogStore
    fixed.CatalogStore = TabICLV14CatalogStore
    fixed._unadopted_staging_npz_bytes = _bootstrap_aware_staging_npz_bytes


def main() -> int:
    configure_v14_runtime()
    return fixed.main()


if __name__ == "__main__":
    raise SystemExit(main())
