#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65536

PROJECT="/users/minsuk/fm-lm-lambda"
PACKAGE="model_v58_chebyshev_v14_scmmixture"
DATA_ROOT="/users/sync_data/kalbi_tabpfnv1_tabiclv2_scm_xy_50_50_groupdrop095_100GB_seed88073400"
CONTROL_ROOT="${DATA_ROOT}/rolling_control"
PIPELINE_SHA256="7196a1a59ebe2f4a79efa2bb4569b141f3f7e0d3efd0ded08498620dc5df4f8a"
TRAINER_PROGRAM="v58chebyshev_v14_scmmixture_bigdata_30k_20260717"
LOG_DIR="${PROJECT}/logs/v14_scmmixture_rolling100gb"

mkdir -p "${LOG_DIR}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"

while [[ ! -f "${CONTROL_ROOT}/CURRENT.json" ]]; do
  sleep 30
done

/venv/main/bin/python3 - <<'PY'
import json
from pathlib import Path

from model_v58_chebyshev_v14_scmmixture.rolling_catalog import CatalogStore
from model_v58_chebyshev_v14_scmmixture.train_joint_v14_isisconditioned_rolling import validate_pipeline_contract

data_root = Path('/users/sync_data/kalbi_tabpfnv1_tabiclv2_scm_xy_50_50_groupdrop095_100GB_seed88073400')
control_root = data_root / 'rolling_control'
pipeline_sha256 = '7196a1a59ebe2f4a79efa2bb4569b141f3f7e0d3efd0ded08498620dc5df4f8a'
priors = ('tabpfn_v1', 'tabicl_v2')
store = CatalogStore(control_root, data_root, priors=priors)
pointer, catalog = store.load_current(verify_files=True)
validate_pipeline_contract(catalog, pipeline_sha256)
active_bytes = sum(int(entry['bytes']) for entry in catalog['files'])
if not 98_000_000_000 <= active_bytes <= 102_000_000_000:
    raise RuntimeError(f'bootstrap catalog bytes outside contract: {active_bytes}')
per_prior = catalog['accounting']['per_prior']
for prior in priors:
    prior_bytes = int(per_prior[prior]['total_bytes'])
    if not 49_000_000_000 <= prior_bytes <= 51_000_000_000:
        raise RuntimeError(f'{prior} bootstrap bytes outside 50:50 band: {prior_bytes}')
    if int(per_prior[prior]['num_files']) < 8:
        raise RuntimeError(f'{prior} cannot assign at least one shard to every rank')
print(json.dumps({
    'kind': 'scmmixture_50_50_bootstrap_catalog_verified',
    'epoch': int(pointer['epoch']),
    'catalog_sha256': pointer['catalog_sha256'],
    'files': len(catalog['files']),
    'bytes': active_bytes,
    'per_prior': per_prior,
    'pipeline_sha256': pipeline_sha256,
}, sort_keys=True), flush=True)
PY

/usr/local/bin/supervisorctl start "${TRAINER_PROGRAM}"
