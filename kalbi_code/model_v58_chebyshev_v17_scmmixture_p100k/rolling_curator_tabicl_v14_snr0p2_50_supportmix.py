#!/usr/bin/env python3
"""Publish a small V14 epoch 0, then grow and rotate it toward 100 GB."""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

from model_v58_chebyshev_v14_scmmixture import rolling_curator as curator
from model_v58_chebyshev_v14_scmmixture import rolling_producer as producer

from .rolling_producer_tabicl_v14_snr0p2_50_supportmix import (
    CURATOR_VERSION,
    PRIORS,
    TabICLV14CatalogStore,
    configure_v14_runtime,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trainer-id", required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--bootstrap-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--target-bytes", type=int, default=100_000_000_000)
    parser.add_argument("--refresh-bytes", type=int, default=2_000_000_000)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    args = parser.parse_args()
    if not 500_000_000 <= args.bootstrap_bytes <= args.target_bytes:
        parser.error("bootstrap bytes must lie in [500 MB, target bytes]")
    if args.target_bytes != 100_000_000_000:
        parser.error("V14 rolling target is fixed at 100 GB")

    configure_v14_runtime()
    producer.PRIOR_KEYS = PRIORS
    curator.PRIOR_KEYS = PRIORS
    curator.CatalogStore = TabICLV14CatalogStore
    curator.CURATOR_VERSION = CURATOR_VERSION

    config, contract_sha = producer.load_and_validate_config(args.config)
    root = args.root.resolve()
    root_lock = producer.ensure_immutable_root_lock(root, config, contract_sha)
    store = TabICLV14CatalogStore(
        config["control_root"], root, priors=PRIORS
    )
    stopped = False

    def request_stop(_signal: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    while not stopped:
        try:
            if not store.current_path.exists():
                curator.bootstrap_if_ready(
                    store,
                    root,
                    root_lock,
                    target_bytes=args.bootstrap_bytes,
                )
            else:
                curator.gc_ready_retirements(
                    store, args.trainer_id, args.world_size
                )
                curator.publish_if_ready(
                    store,
                    root,
                    trainer_id=args.trainer_id,
                    world_size=args.world_size,
                    refresh_bytes=args.refresh_bytes,
                    target_bytes=args.target_bytes,
                )
        except curator.PublishDeferred as exc:
            print(
                json.dumps({"kind": "publish_deferred", "reason": str(exc)}),
                flush=True,
            )
        except (
            curator.CatalogError,
            OSError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            print(
                json.dumps({"kind": "curator_error", "error": repr(exc)}),
                flush=True,
            )
            raise
        time.sleep(max(1.0, float(args.poll_seconds)))


if __name__ == "__main__":
    main()
