#!/usr/bin/env python3
"""Run the proven P<=100K rolling producer for a TabICLv2-only corpus."""

from __future__ import annotations

from typing import Any

from model_v58_chebyshev_v14_scmmixture import rolling_producer as producer
from model_v58_chebyshev_v14_scmmixture.rolling_catalog import (
    CapacityPolicy,
    CatalogStore,
)
from model_v58_chebyshev_v17_scmmixture_p100k import (
    rolling_producer_backpressure_fixed as fixed,
)


PRIORS = ("tabicl_v2",)
PIPELINE_VERSION = "v14_kalbi_tabiclv2_scm_xy_only_rolling100gb_v1"
SOURCE_VERSION = "v14_tabiclv2_only_cpu_rolling_producer_v1"


def tabicl_only_policy(*, world_size: int = 8) -> CapacityPolicy:
    return CapacityPolicy(
        active_min_bytes=49_000_000_000,
        active_max_bytes=102_000_000_000,
        prior_min_bytes=49_000_000_000,
        prior_max_bytes=102_000_000_000,
        hard_cap_bytes=130_000_000_000,
        min_free_bytes=300_000_000_000,
        world_size=world_size,
    )


class TabICLOnlyCatalogStore(CatalogStore):
    def __init__(
        self,
        control_root: Any,
        data_root: Any,
        policy: CapacityPolicy | None = None,
        *,
        priors: Any = PRIORS,
    ) -> None:
        if tuple(priors) != PRIORS:
            raise ValueError(f"TabICLv2-only store received priors={tuple(priors)!r}")
        super().__init__(
            control_root,
            data_root,
            policy=policy or tabicl_only_policy(),
            priors=PRIORS,
        )


def configure_single_prior_runtime() -> None:
    producer.PRIOR_KEYS = PRIORS
    producer.PIPELINE_VERSION = PIPELINE_VERSION
    producer.SOURCE_VERSION = SOURCE_VERSION
    producer.CatalogStore = TabICLOnlyCatalogStore
    fixed.CatalogStore = TabICLOnlyCatalogStore


def main() -> int:
    configure_single_prior_runtime()
    return fixed.main()


if __name__ == "__main__":
    raise SystemExit(main())
