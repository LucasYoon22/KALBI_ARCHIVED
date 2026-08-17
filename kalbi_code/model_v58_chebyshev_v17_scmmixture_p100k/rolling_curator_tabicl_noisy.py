#!/usr/bin/env python3
"""Publish and rotate the noisy-response TabICLv2 rolling catalog."""

from __future__ import annotations

from model_v58_chebyshev_v14_scmmixture import rolling_curator as curator
from model_v58_chebyshev_v14_scmmixture import rolling_producer as producer

from .rolling_producer_tabicl_noisy import (
    CURATOR_VERSION,
    PRIORS,
    TabICLNoisyCatalogStore,
    configure_noisy_runtime,
)


def main() -> None:
    configure_noisy_runtime()
    producer.PRIOR_KEYS = PRIORS
    curator.PRIOR_KEYS = PRIORS
    curator.CatalogStore = TabICLNoisyCatalogStore
    curator.CURATOR_VERSION = CURATOR_VERSION
    curator.main()


if __name__ == "__main__":
    main()
