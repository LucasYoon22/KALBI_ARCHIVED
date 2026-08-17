#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65536

PROJECT="/users/minsuk/fm-lm-lambda"
PACKAGE="model_v58_chebyshev_v14_scmresponse"
DATA_ROOT="/users/sync_data/tabiclv2_graph_scm_y_groupdrop095_medianN2000_medianP10000_100GB_tailP100K_mlp512_gaussian16_seed88073200"
CONTROL_ROOT="${DATA_ROOT}/rolling_control"
PIPELINE_SHA256="e65dfad2aa7c78e9a3e4a63b6ad2fba11cf953b612fc598c4966ce39c8793266"
TRAINER_PROGRAM="v58chebyshev_v14_scmresponse_bigdata_30k_20260717"
LOG_DIR="${PROJECT}/logs/v14_scmresponse_rolling100gb"

mkdir -p "${LOG_DIR}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"

while [[ ! -f "${CONTROL_ROOT}/CURRENT.json" ]]; do
  sleep 30
done

/venv/main/bin/python3 - <<'PY'
import json
from pathlib import Path

from model_v58_chebyshev_v14_scmresponse.rolling_catalog import CatalogStore
from model_v58_chebyshev_v14_scmresponse.train_joint_v14_isisconditioned_rolling import validate_pipeline_contract

data_root = Path('/users/sync_data/tabiclv2_graph_scm_y_groupdrop095_medianN2000_medianP10000_100GB_tailP100K_mlp512_gaussian16_seed88073200')
control_root = data_root / 'rolling_control'
pipeline_sha256 = 'e65dfad2aa7c78e9a3e4a63b6ad2fba11cf953b612fc598c4966ce39c8793266'
store = CatalogStore(control_root, data_root, priors=('tabicl_v2',))
pointer, catalog = store.load_current(verify_files=True)
validate_pipeline_contract(catalog, pipeline_sha256)
active_bytes = sum(int(entry['bytes']) for entry in catalog['files'])
if not 98_000_000_000 <= active_bytes <= 102_000_000_000:
    raise RuntimeError(f'bootstrap catalog bytes outside contract: {active_bytes}')
if len(catalog['files']) < 8:
    raise RuntimeError('bootstrap catalog cannot assign at least one shard to every rank')
print(json.dumps({
    'kind': 'scmresponse_bootstrap_catalog_verified',
    'epoch': int(pointer['epoch']),
    'catalog_sha256': pointer['catalog_sha256'],
    'files': len(catalog['files']),
    'bytes': active_bytes,
    'pipeline_sha256': pipeline_sha256,
}, sort_keys=True), flush=True)
PY

/usr/local/bin/supervisorctl start "${TRAINER_PROGRAM}"
