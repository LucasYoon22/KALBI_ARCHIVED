#!/usr/bin/env python3
"""Publish and rotate a single-prior TabICLv2 rolling catalog."""

from __future__ import annotations

from model_v58_chebyshev_v14_scmmixture import rolling_curator as curator
from model_v58_chebyshev_v14_scmmixture import rolling_producer as producer

from .rolling_producer_tabicl_only import (
    PRIORS,
    TabICLOnlyCatalogStore,
    configure_single_prior_runtime,
)


def main() -> None:
    configure_single_prior_runtime()
    producer.PRIOR_KEYS = PRIORS
    curator.PRIOR_KEYS = PRIORS
    curator.CatalogStore = TabICLOnlyCatalogStore
    curator.CURATOR_VERSION = "v14_tabiclv2_only_rolling100gb_curator_v1"
    curator.main()


if __name__ == "__main__":
    main()
