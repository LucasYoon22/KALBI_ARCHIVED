#!/usr/bin/env python3
"""Run the P<=100K rolling producer for noisy-response TabICLv2 tasks."""

from __future__ import annotations

import copy
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
PIPELINE_VERSION = "v14_kalbi_tabiclv2_sampled_response_noise_rolling100gb_v1"
SOURCE_VERSION = "v14_tabiclv2_noisy_y_v6_cpu_rolling_producer_v1"
CURATOR_VERSION = "v14_tabiclv2_noisy_y_v6_rolling100gb_curator_v1"

GENERATOR = "generate_tabiclv2_graph_scm_mlp512_groupdrop095_noisy_y_v6.py"
PRIOR_NAME = (
    "official_tabiclv2_graph_scm_xy_groupdrop095_then_random_mlp512_"
    "sampled_response_noise_v1"
)
DATASET_CONTRACT = "tabiclv2_graph_scm_xy_groupdrop095_sampled_noise_v6"
DATA_POLICY = "tabiclv2_graph_scm_xy_exact_corr_lt_0p95_v1"
TARGET_CONTRACT = (
    "official_tabiclv2_graph_scm_y_plus_sampled_error_clip10_rescale_v1"
)
LATENT_TARGET_CONTRACT = (
    "official_tabiclv2_graph_scm_y_center_scale_clip10_rescale_v1"
)
RESPONSE_NOISE_CONTRACT = (
    "capped_exp1_scale_mixture_centered_unit_error_additive_v1"
)
TARGET_SOURCE = (
    "official TabICLv2 PriorDataset graph_scm response plus sampled error"
)
TARGET_TRANSFORMATION = (
    "standardize and clip official signal; add sigma times centered unit-scale "
    "sampled error; clip; center and scale again"
)
OFFICIAL_TARGET_USAGE = (
    "retained as the latent signal from the same official TabICLv2 graph_scm "
    "draw, then augmented with certified sampled response error"
)
NOISE_SIGMA_DISTRIBUTION = "min(Exponential(scale), sigma_cap)"
NOISE_NORMALIZATION = (
    "each sampled error vector is empirically centered and unit-scaled "
    "before multiplication by sigma"
)
NOISE_FAMILIES = (
    ("gaussian", 0.35),
    ("student_t", 0.20),
    ("laplace", 0.15),
    ("contaminated_gaussian", 0.10),
    ("centered_lognormal", 0.10),
    ("heteroskedastic_gaussian", 0.10),
)
NOISE_EXP_SCALE = 1.0
NOISE_SIGMA_CAP = 3.0
TARGET_CLIP_VALUE = 10.0

NOISY_PRIOR_CONTRACT = {
    "generator": GENERATOR,
    "prior_name": PRIOR_NAME,
    "dataset_contract": DATASET_CONTRACT,
    "data_policy": DATA_POLICY,
    "target_contract": TARGET_CONTRACT,
    "official_core_prior": "TabICLv2 PriorDataset prior_type=graph_scm",
    "target_source": TARGET_SOURCE,
    "direct_mode": "direct_tabiclv2",
    "highp_mode": "tabiclv2_plus_random_mlp512",
    "commit": "46b91961db4f8873dd049ec09990698a435e1e29",
}

_BASE_VALIDATE_GENERATED_NPZ = producer.validate_generated_npz
_BASE_VALIDATE_GENERATOR_CONTRACT = producer.validate_generator_contract
_FIXED_UNADOPTED_STAGING_NPZ_BYTES = fixed._unadopted_staging_npz_bytes


def tabicl_noisy_policy(*, world_size: int = 8) -> CapacityPolicy:
    return CapacityPolicy(
        active_min_bytes=49_000_000_000,
        active_max_bytes=102_000_000_000,
        prior_min_bytes=49_000_000_000,
        prior_max_bytes=102_000_000_000,
        hard_cap_bytes=130_000_000_000,
        min_free_bytes=300_000_000_000,
        world_size=world_size,
    )


class TabICLNoisyCatalogStore(CatalogStore):
    def __init__(
        self,
        control_root: Any,
        data_root: Any,
        policy: CapacityPolicy | None = None,
        *,
        priors: Any = PRIORS,
    ) -> None:
        if tuple(priors) != PRIORS:
            raise ValueError(f"noisy TabICLv2 store received priors={tuple(priors)!r}")
        super().__init__(
            control_root,
            data_root,
            policy=policy or tabicl_noisy_policy(),
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


def _close(actual: Any, expected: float, name: str, atol: float = 2e-6) -> float:
    try:
        value = float(actual)
    except (TypeError, ValueError) as exc:
        raise producer.ProducerError(f"{name} is not numeric") from exc
    if not math.isfinite(value) or not math.isclose(
        value, expected, rel_tol=0.0, abs_tol=atol
    ):
        raise producer.ProducerError(
            f"{name} mismatch: {value!r} != {expected!r} (atol={atol})"
        )
    return value


def validate_noisy_generated_npz(
    npz_path: str,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    summary = dict(_BASE_VALIDATE_GENERATED_NPZ(npz_path, entry))
    with np.load(npz_path, allow_pickle=False) as archive:
        required = {
            "latent_scm_signal",
            "response_noise_unit",
            "response_noise_sigma_raw",
            "response_noise_sigma",
            "response_noise_family_id",
            "latent_signal_noisy_correlation",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise producer.ProducerError(
                f"noisy-response NPZ lacks required arrays: {missing}"
            )
        y = archive["y"]
        signal = archive["latent_scm_signal"]
        unit_error = archive["response_noise_unit"]
        sigma_raw = archive["response_noise_sigma_raw"]
        sigma = archive["response_noise_sigma"]
        family_id = archive["response_noise_family_id"]
        signal_y_corr = archive["latent_signal_noisy_correlation"]

        task_shape = y.shape
        vector_shape = (y.shape[0],)
        if signal.shape != task_shape or signal.dtype != np.float32:
            raise producer.ProducerError(
                f"invalid latent_scm_signal: shape={signal.shape} dtype={signal.dtype}"
            )
        if unit_error.shape != task_shape or unit_error.dtype != np.float32:
            raise producer.ProducerError(
                f"invalid response_noise_unit: shape={unit_error.shape} "
                f"dtype={unit_error.dtype}"
            )
        for name, value in (
            ("response_noise_sigma_raw", sigma_raw),
            ("response_noise_sigma", sigma),
            ("latent_signal_noisy_correlation", signal_y_corr),
        ):
            if value.shape != vector_shape or value.dtype != np.float32:
                raise producer.ProducerError(
                    f"invalid {name}: shape={value.shape} dtype={value.dtype}"
                )
            if not np.isfinite(value).all():
                raise producer.ProducerError(f"nonfinite {name}")
        if family_id.shape != vector_shape or family_id.dtype != np.uint8:
            raise producer.ProducerError(
                f"invalid response_noise_family_id: shape={family_id.shape} "
                f"dtype={family_id.dtype}"
            )

        signal64 = signal.astype(np.float64)
        error64 = unit_error.astype(np.float64)
        y64 = y.astype(np.float64)
        for name, value in (("latent signal", signal64), ("unit error", error64)):
            means = value.mean(axis=1)
            stds = value.std(axis=1)
            if not np.all(np.abs(means) <= 2e-6):
                raise producer.ProducerError(f"{name} is not centered")
            if not np.all(np.abs(stds - 1.0) <= 2e-6):
                raise producer.ProducerError(f"{name} is not unit scale")

        if not np.all(sigma_raw >= 0.0):
            raise producer.ProducerError("negative raw response-noise sigma")
        expected_sigma = np.minimum(sigma_raw, NOISE_SIGMA_CAP)
        if not np.allclose(sigma, expected_sigma, rtol=0.0, atol=2e-7):
            raise producer.ProducerError("capped response-noise sigma is inconsistent")
        if not np.all((family_id >= 0) & (family_id < len(NOISE_FAMILIES))):
            raise producer.ProducerError("response-noise family id is out of range")

        reconstructed = signal64 + sigma.astype(np.float64)[:, None] * error64
        reconstructed = np.clip(
            reconstructed, -TARGET_CLIP_VALUE, TARGET_CLIP_VALUE
        )
        reconstructed -= reconstructed.mean(axis=1, keepdims=True)
        reconstructed_std = reconstructed.std(axis=1, keepdims=True)
        if not np.all(reconstructed_std > 1e-7):
            raise producer.ProducerError("reconstructed noisy response is degenerate")
        reconstructed /= reconstructed_std
        if not np.allclose(y64, reconstructed, rtol=0.0, atol=3e-6):
            max_delta = float(np.max(np.abs(y64 - reconstructed)))
            raise producer.ProducerError(
                f"stored y does not reconstruct from signal+noise; max_delta={max_delta}"
            )

        realized_corr = np.mean(signal64 * y64, axis=1)
        if not np.allclose(signal_y_corr, realized_corr, rtol=0.0, atol=3e-6):
            raise producer.ProducerError(
                "latent/noisy-response correlation certificate is inconsistent"
            )
        if not np.all(np.abs(realized_corr) <= 1.0 + 3e-6):
            raise producer.ProducerError("invalid latent/noisy-response correlation")

        summary.update(
            {
                "response_noise_sigma_raw_min": float(np.min(sigma_raw)),
                "response_noise_sigma_raw_max": float(np.max(sigma_raw)),
                "response_noise_sigma_min": float(np.min(sigma)),
                "response_noise_sigma_max": float(np.max(sigma)),
                "response_noise_family_ids": sorted(
                    int(value) for value in np.unique(family_id)
                ),
                "latent_signal_noisy_correlation_min": float(
                    np.min(realized_corr)
                ),
                "latent_signal_noisy_correlation_max": float(
                    np.max(realized_corr)
                ),
                "noisy_response_reconstruction_max_abs_error": float(
                    np.max(np.abs(y64 - reconstructed))
                ),
            }
        )
    return summary


def _validate_family_parameters(
    family: str,
    parameters: Any,
    *,
    n: int,
    p: int,
) -> None:
    if not isinstance(parameters, Mapping):
        raise producer.ProducerError("response_noise_family_parameters is not an object")
    if family in {"gaussian", "laplace"}:
        if parameters:
            raise producer.ProducerError(f"{family} noise has unexpected parameters")
    elif family == "student_t":
        df = int(parameters.get("degrees_of_freedom", -1))
        if not 3 <= df <= 10:
            raise producer.ProducerError("Student-t degrees of freedom is invalid")
    elif family == "contaminated_gaussian":
        probability = float(parameters.get("contamination_probability", math.nan))
        outlier_scale = float(parameters.get("outlier_scale", math.nan))
        count = int(parameters.get("contaminated_count", -1))
        if not (0.02 <= probability <= 0.10 and 3.0 <= outlier_scale <= 8.0):
            raise producer.ProducerError("contaminated-Gaussian parameters are invalid")
        if not 0 <= count <= n:
            raise producer.ProducerError("contaminated-Gaussian count is invalid")
    elif family == "centered_lognormal":
        log_scale = float(parameters.get("log_scale", math.nan))
        if not 0.35 <= log_scale <= 1.0:
            raise producer.ProducerError("centered-lognormal scale is invalid")
    elif family == "heteroskedastic_gaussian":
        feature_index = int(parameters.get("feature_index_after_permutation", -1))
        power = float(parameters.get("heteroskedastic_power", math.nan))
        scale_min = float(parameters.get("conditional_scale_min", math.nan))
        scale_max = float(parameters.get("conditional_scale_max", math.nan))
        if not 0 <= feature_index < p or not 0.5 <= power <= 1.5:
            raise producer.ProducerError("heteroskedastic-noise parameters are invalid")
        if not (math.isfinite(scale_min) and scale_min >= 0.35):
            raise producer.ProducerError("heteroskedastic minimum scale is invalid")
        if not (math.isfinite(scale_max) and scale_max >= scale_min):
            raise producer.ProducerError("heteroskedastic maximum scale is invalid")
    else:
        raise producer.ProducerError(f"unknown response-noise family: {family!r}")


def validate_noisy_generator_contract(
    prior: str,
    manifest: Mapping[str, Any],
    entry: Mapping[str, Any],
    npz_summary: Mapping[str, Any],
) -> dict[str, Any]:
    if prior != "tabicl_v2":
        raise producer.ProducerError(f"unsupported noisy-response prior: {prior!r}")

    adapted_manifest = copy.deepcopy(dict(manifest))
    adapted_settings = adapted_manifest.get("settings")
    if not isinstance(adapted_settings, dict):
        raise producer.ProducerError("manifest.settings must be an object")
    adapted_settings["target_transformation"] = (
        "center, divide by population standard deviation, clip to +/-clip_value, "
        "center and scale again"
    )
    adapted_settings["official_target_usage"] = (
        "retained from the same official TabICLv2 graph_scm draw, then standardized"
    )
    semantic = dict(
        _BASE_VALIDATE_GENERATOR_CONTRACT(
            prior, adapted_manifest, entry, npz_summary
        )
    )

    settings = manifest["settings"]
    expected_families = [
        {"name": name, "probability": probability}
        for name, probability in NOISE_FAMILIES
    ]
    settings_exact = {
        "target_transformation": TARGET_TRANSFORMATION,
        "official_target_usage": OFFICIAL_TARGET_USAGE,
        "response_noise_contract": RESPONSE_NOISE_CONTRACT,
        "response_noise_sigma_distribution": NOISE_SIGMA_DISTRIBUTION,
        "response_noise_families": expected_families,
        "response_noise_normalization": NOISE_NORMALIZATION,
    }
    for key, expected in settings_exact.items():
        actual = settings.get(key)
        if actual != expected:
            raise producer.ProducerError(
                f"manifest.settings.{key} contract mismatch: {actual!r} != {expected!r}"
            )
    _close(
        settings.get("response_noise_exp_scale"),
        NOISE_EXP_SCALE,
        "manifest.settings.response_noise_exp_scale",
        atol=1e-12,
    )
    _close(
        settings.get("response_noise_sigma_cap"),
        NOISE_SIGMA_CAP,
        "manifest.settings.response_noise_sigma_cap",
        atol=1e-12,
    )

    if entry.get("response_noise_contract") != RESPONSE_NOISE_CONTRACT:
        raise producer.ProducerError("manifest.entry response-noise contract mismatch")
    sigma = float(npz_summary["response_noise_sigma_min"])
    if not math.isclose(
        sigma,
        float(npz_summary["response_noise_sigma_max"]),
        rel_tol=0.0,
        abs_tol=2e-7,
    ):
        raise producer.ProducerError("one-task shard has inconsistent response-noise sigma")
    corr = float(npz_summary["latent_signal_noisy_correlation_min"])
    if not math.isclose(
        corr,
        float(npz_summary["latent_signal_noisy_correlation_max"]),
        rel_tol=0.0,
        abs_tol=3e-6,
    ):
        raise producer.ProducerError("one-task shard has inconsistent signal correlation")
    for key in (
        "task_response_noise_sigma_min",
        "task_response_noise_sigma_median",
        "task_response_noise_sigma_max",
    ):
        _close(entry.get(key), sigma, f"manifest.entry.{key}")
    for key in (
        "task_latent_signal_noisy_correlation_min",
        "task_latent_signal_noisy_correlation_median",
        "task_latent_signal_noisy_correlation_max",
    ):
        _close(entry.get(key), corr, f"manifest.entry.{key}", atol=3e-6)

    task = entry["task_metadata"][0]
    if task.get("latent_target_contract") != LATENT_TARGET_CONTRACT:
        raise producer.ProducerError("task latent-target contract mismatch")
    if task.get("response_noise_contract") != RESPONSE_NOISE_CONTRACT:
        raise producer.ProducerError("task response-noise contract mismatch")
    _close(task.get("latent_signal_final_mean"), 0.0, "task.latent_signal_final_mean")
    _close(task.get("latent_signal_final_std"), 1.0, "task.latent_signal_final_std")
    _close(task.get("response_noise_unit_mean"), 0.0, "task.response_noise_unit_mean")
    _close(task.get("response_noise_unit_std"), 1.0, "task.response_noise_unit_std")

    sigma_raw = float(npz_summary["response_noise_sigma_raw_min"])
    _close(task.get("response_noise_sigma_raw"), sigma_raw, "task.response_noise_sigma_raw")
    _close(task.get("response_noise_sigma"), sigma, "task.response_noise_sigma")
    if bool(task.get("response_noise_sigma_was_capped")) != bool(
        sigma_raw > NOISE_SIGMA_CAP
    ):
        raise producer.ProducerError("task sigma-capping certificate is inconsistent")
    _close(task.get("response_noise_scaled_std"), sigma, "task.response_noise_scaled_std")
    expected_snr = 1.0 / max(sigma * sigma, 1e-16)
    _close(
        task.get("response_noise_realized_snr"),
        expected_snr,
        "task.response_noise_realized_snr",
        atol=max(2e-6, 2e-6 * expected_snr),
    )
    _close(
        task.get("response_noise_theoretical_signal_fraction"),
        1.0 / (1.0 + sigma * sigma),
        "task.response_noise_theoretical_signal_fraction",
    )
    _close(task.get("latent_signal_noisy_correlation"), corr, "task.signal_y_corr", atol=3e-6)
    _close(task.get("latent_signal_noisy_r2_ceiling"), corr * corr, "task.signal_y_r2", atol=5e-6)

    family_id = int(task.get("response_noise_family_id", -1))
    ids = npz_summary["response_noise_family_ids"]
    if ids != [family_id] or not 0 <= family_id < len(NOISE_FAMILIES):
        raise producer.ProducerError("task response-noise family id is inconsistent")
    family = str(task.get("response_noise_family", ""))
    if family != NOISE_FAMILIES[family_id][0]:
        raise producer.ProducerError("task response-noise family name is inconsistent")
    _validate_family_parameters(
        family,
        task.get("response_noise_family_parameters"),
        n=int(npz_summary["n"]),
        p=int(npz_summary["p"]),
    )

    semantic.update(
        {
            "response_noise_contract": RESPONSE_NOISE_CONTRACT,
            "response_noise_family": family,
            "response_noise_family_id": family_id,
            "response_noise_sigma_raw": sigma_raw,
            "response_noise_sigma": sigma,
            "latent_signal_noisy_correlation": corr,
            "latent_signal_noisy_r2_ceiling": corr * corr,
            "noisy_response_reconstruction_max_abs_error": float(
                npz_summary["noisy_response_reconstruction_max_abs_error"]
            ),
        }
    )
    return semantic


def configure_noisy_runtime() -> None:
    producer.PRIOR_KEYS = PRIORS
    producer.PIPELINE_VERSION = PIPELINE_VERSION
    producer.SOURCE_VERSION = SOURCE_VERSION
    producer.PRIOR_CONTRACTS["tabicl_v2"] = dict(NOISY_PRIOR_CONTRACT)
    producer.validate_generated_npz = validate_noisy_generated_npz
    producer.validate_generator_contract = validate_noisy_generator_contract
    producer.CatalogStore = TabICLNoisyCatalogStore
    fixed.CatalogStore = TabICLNoisyCatalogStore
    fixed._unadopted_staging_npz_bytes = _bootstrap_aware_staging_npz_bytes


def main() -> int:
    configure_noisy_runtime()
    return fixed.main()


if __name__ == "__main__":
    raise SystemExit(main())
