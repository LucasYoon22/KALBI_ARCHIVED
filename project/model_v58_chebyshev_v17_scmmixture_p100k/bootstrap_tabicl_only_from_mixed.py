#!/usr/bin/env python3
"""Bootstrap the TabICLv2-only root from validated shards in a mixed catalog."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from model_v58_chebyshev_v14_scmmixture import rolling_producer as producer
from model_v58_chebyshev_v14_scmmixture.rolling_catalog import CatalogStore

from .rolling_producer_tabicl_only import (
    PRIORS,
    TabICLOnlyCatalogStore,
    configure_single_prior_runtime,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    args = parser.parse_args()

    configure_single_prior_runtime()
    config, contract_sha = producer.load_and_validate_config(args.config)
    target_root = args.target_root.resolve()
    root_lock = producer.ensure_immutable_root_lock(
        target_root, config, contract_sha
    )
    target_store = TabICLOnlyCatalogStore(
        config["control_root"], target_root, priors=PRIORS
    )
    if target_store.current_path.exists():
        pointer, catalog = target_store.load_current(verify_files=True)
        print(json.dumps({
            "kind": "already_bootstrapped",
            "epoch": pointer["epoch"],
            "catalog_sha256": pointer["catalog_sha256"],
            "files": len(catalog["files"]),
            "bytes": catalog["accounting"]["total_bytes"],
        }, sort_keys=True))
        return

    source_root = args.source_root.resolve()
    source_store = CatalogStore(
        source_root / "rolling_control",
        source_root,
        priors=("tabpfn_v1", "tabicl_v2"),
    )
    _pointer, source_catalog = source_store.load_current(verify_files=False)
    source_entries = [
        entry for entry in source_catalog["files"]
        if entry["prior"] == "tabicl_v2"
    ]
    if not source_entries:
        raise RuntimeError("source catalog contains no TabICLv2 entries")

    migrated = []
    for source_entry in source_entries:
        entry = dict(source_entry)
        source_path = source_store.resolve_entry_path(entry)
        target_path = target_root / entry["relative_path"]
        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source_path, target_path)
        except FileExistsError:
            if producer.sha256_file(target_path) != entry["sha256"]:
                raise RuntimeError(f"target hardlink collision: {target_path}")
        entry["pipeline_config_sha256"] = root_lock["pipeline_config_sha256"]
        entry["source_hashes"] = root_lock["source_hashes"]
        entry["origin"] = "validated_tabicl_v2_hardlink_migration"
        migrated.append(entry)

    pointer, catalog = target_store.bootstrap_from_staged_entries(
        migrated,
        pipeline_config_sha256=root_lock["pipeline_config_sha256"],
        source_hashes=root_lock["source_hashes"],
        verify_files=True,
        check_free_space=True,
    )
    print(json.dumps({
        "kind": "tabicl_only_bootstrap_complete",
        "epoch": pointer["epoch"],
        "catalog_sha256": pointer["catalog_sha256"],
        "files": len(catalog["files"]),
        "bytes": catalog["accounting"]["total_bytes"],
        "priors": sorted(catalog["accounting"]["per_prior"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
