from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from datetime import datetime, timezone


BACKUP = Path(
    "/Users/minsuk.shin/Library/Mobile Documents/com~apple~CloudDocs/Projects/"
    "FM-LM/backups/kalbi_v2_v17_core_20260722_054519_KST"
)
PRODUCTION = ("19508", "root@175.155.64.224")
REFERENCE = ("40801", "root@38.29.145.27")
EXCLUDED_PARTS = {"__pycache__", ".ipynb_checkpoints"}
EXCLUDED_SUFFIXES = {".pyc", ".md", ".log", ".npz", ".npy"}
EXCLUDED_NAMES = {".DS_Store"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ssh(port: str, host: str, command: str) -> str:
    result = subprocess.run(
        ["ssh", "-p", port, "-o", "BatchMode=yes", host, command],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout


def remote_manifest(
    endpoint: tuple[str, str],
    roots: dict[str, str],
    files: dict[str, str],
) -> dict[str, dict[str, int | str]]:
    payload = json.dumps({"roots": roots, "files": files})
    script = r'''
import hashlib, json, sys
from pathlib import Path
spec = json.loads(sys.argv[1])
excluded_parts = {"__pycache__", ".ipynb_checkpoints"}
excluded_suffixes = {".pyc", ".md", ".log", ".npz", ".npy"}
excluded_names = {".DS_Store"}
def included(path):
    return (not any(part in excluded_parts for part in path.parts)
            and path.suffix not in excluded_suffixes
            and path.name not in excluded_names)
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
out = {}
for label, raw_root in spec["roots"].items():
    root = Path(raw_root)
    for path in sorted(root.rglob("*")):
        if path.is_file() and included(path.relative_to(root)):
            key = label + "/" + path.relative_to(root).as_posix()
            out[key] = {"bytes": path.stat().st_size, "sha256": digest(path)}
for key, raw_path in spec["files"].items():
    path = Path(raw_path)
    out[key] = {"bytes": path.stat().st_size, "sha256": digest(path)}
print(json.dumps(out, sort_keys=True))
'''
    command = "/usr/bin/python3 -c {} {}".format(
        shlex.quote(script), shlex.quote(payload)
    )
    return json.loads(ssh(*endpoint, command))


def local_source_manifest() -> dict[str, dict[str, int | str]]:
    ignored_top_level = {
        "BACKUP_README.txt",
        "MANIFEST.json",
        "SHA256SUMS.txt",
        "VERIFICATION.json",
        "RUN_STATE.json",
    }
    result: dict[str, dict[str, int | str]] = {}
    for path in sorted(BACKUP.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(BACKUP)
        if relative.parts[0] == "environment" or relative.name in ignored_top_level:
            continue
        result[relative.as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    return result


production_roots = {
    "production_8gpu/kalbi_v2_v17_p100k_effbatch32": (
        "/users/minsuk/fm-lm-lambda/kalbi_v2_v17_p100k_effbatch32"
    ),
    "production_8gpu/model_v58_chebyshev_v17_scmmixture_p100k": (
        "/users/minsuk/fm-lm-lambda/model_v58_chebyshev_v17_scmmixture_p100k"
    ),
    "data_generation/model_v58_chebyshev_v14_scmmixture": (
        "/users/minsuk/fm-lm-lambda/model_v58_chebyshev_v14_scmmixture"
    ),
    "base_runtime/model_v58_chebyshev": (
        "/users/minsuk/fm-lm-lambda/model_v58_chebyshev"
    ),
    "data_generation/tabicl": "/users/minsuk/fm-lm-lambda/external/tabicl",
    "data_generation/TabPFN-v1.0.0": (
        "/users/minsuk/fm-lm-lambda/external/TabPFN-v1.0.0"
    ),
}

base_names = [
    "fpk_v4_common.py",
    "model_fpk.py",
    "penalty_relu_mlp_pwl_v33.py",
    "penalty_sampler_v3.py",
    "penalty_sampler_v4.py",
    "penalty_sampler_v4_v5base.py",
    "penalty_sampler_v8.py",
    "penalty_sampler_v9.py",
    "shape_feature_normalization.py",
    "train_fpk.py",
    "fpk_v33_kkt0.yaml",
]
generator_names = [
    "exp122_shape_sampler.py",
    "exp122_sparse_linear_target.py",
    "generate_tabiclv2_graph_highdim_npz.py",
    "generate_tabiclv2_graph_scm_mlp512_groupdrop095_scm_y_v5.py",
    "generate_tabpfn_v1_causal_scm_mlp512_groupdrop095_scm_y_v2.py",
    "generate_tabpfn_v1_causal_scm_mlp512_groupdrop095_sparse_linear_v1.py",
]
supervisor_names = [
    "kalbi_v2_v17_p100k_effbatch32.conf",
    "kalbi_v2_v17_p100k_curator.conf",
    "kalbi_v2_v17_p100k_effbatch32_preflight.conf",
    "v58_v17_p100k.conf",
]
production_files = {
    **{
        f"base_runtime/root_files/{name}": f"/users/minsuk/fm-lm-lambda/{name}"
        for name in base_names
    },
    **{
        f"data_generation/root_files/{name}": f"/users/minsuk/fm-lm-lambda/{name}"
        for name in generator_names
    },
    **{
        f"supervisor/{name}": f"/etc/supervisor/conf.d/{name}"
        for name in supervisor_names
    },
    "required_artifacts/penalty_cache_300k_v12_canonical_final.pt": (
        "/users/minsuk/fm-lm-lambda/penalty_cache_300k_v12_canonical_final.pt"
    ),
}
reference_roots = {
    "reference_n200p500/kalbi_v2_v17_n200p500_joint": (
        "/users/minsuk/fm-lm-lambda/kalbi_v2_v17_n200p500_joint"
    )
}

production_remote = remote_manifest(PRODUCTION, production_roots, production_files)
reference_remote = remote_manifest(REFERENCE, reference_roots, {})
local_manifest = local_source_manifest()
remote_combined = {**production_remote, **reference_remote}

missing_local = sorted(set(remote_combined) - set(local_manifest))
remote_mismatch = sorted(
    key
    for key in set(remote_combined) & set(local_manifest)
    if remote_combined[key] != local_manifest[key]
)
unverified_local = sorted(
    key
    for key in set(local_manifest) - set(remote_combined)
    if not key.startswith("local_working_copy/")
)
if missing_local or remote_mismatch or unverified_local:
    raise SystemExit(
        json.dumps(
            {
                "missing_local": missing_local,
                "remote_mismatch": remote_mismatch,
                "unverified_local": unverified_local,
            },
            indent=2,
        )
    )

env_script = r'''
import json, platform, socket, subprocess, sys
import numpy, torch
def run(args):
    return subprocess.run(args, text=True, capture_output=True).stdout.strip()
print(json.dumps({
    "hostname": socket.gethostname(),
    "platform": platform.platform(),
    "python": sys.version,
    "numpy": numpy.__version__,
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "gpu": run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader,nounits"]),
}, sort_keys=True))
'''

environment_dir = BACKUP / "environment"
environment_dir.mkdir(parents=True, exist_ok=True)
for label, endpoint in (("production", PRODUCTION), ("reference", REFERENCE)):
    command = "/venv/main/bin/python3 -c {}".format(shlex.quote(env_script))
    data = json.loads(ssh(*endpoint, command))
    (environment_dir / f"{label}_environment.json").write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n"
    )
    freeze = ssh(*endpoint, "/venv/main/bin/pip freeze")
    (environment_dir / f"{label}_pip_freeze.txt").write_text(freeze)

(environment_dir / "production_remote_manifest.json").write_text(
    json.dumps(production_remote, indent=2, sort_keys=True) + "\n"
)
(environment_dir / "reference_remote_manifest.json").write_text(
    json.dumps(reference_remote, indent=2, sort_keys=True) + "\n"
)

run_script = r'''
import glob, hashlib, json, os, subprocess
run = "/users/minsuk/fm-lm-lambda/runs/kalbi_v2_v17_p100k_effbatch32_scratch_30k_20260721"
events_path = run + "/events.jsonl"
with open(events_path, "rb") as f:
    size = os.path.getsize(events_path)
    f.seek(max(0, size - 16 * 1024 * 1024))
    events = [json.loads(line) for line in f.read().splitlines() if line.strip()]
last_train = next(event for event in reversed(events) if event.get("kind") == "train_update")
rank0 = (last_train.get("rank_monitors") or [{}])[0]
checkpoints = sorted(glob.glob(run + "/checkpoint_step*.pt"))
latest = checkpoints[-1]
h = hashlib.sha256()
with open(latest, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
status = subprocess.run(["supervisorctl", "status"], text=True, capture_output=True).stdout.splitlines()
active = [line for line in status if any(token in line for token in ["kalbi_v2_v17_p100k", "v58v17_p100k_tab"])]
current_path = "/users/sync_data/kalbi_tabpfnv1_tabiclv2_scm_xy_50_50_groupdrop095_100GB_seed88073400/rolling_control/CURRENT.json"
with open(current_path) as f:
    current = json.load(f)
print(json.dumps({
    "run": run,
    "latest_train_step": last_train.get("step"),
    "catalog_epoch": rank0.get("catalog_epoch"),
    "catalog_sha256": rank0.get("catalog_sha256"),
    "latest_checkpoint": latest,
    "latest_checkpoint_bytes": os.path.getsize(latest),
    "latest_checkpoint_sha256": h.hexdigest(),
    "rolling_current": current,
    "supervisor_status": active,
}, sort_keys=True))
'''
run_command = "/usr/bin/python3 -c {}".format(shlex.quote(run_script))
run_state = json.loads(ssh(*PRODUCTION, run_command))
(BACKUP / "RUN_STATE.json").write_text(
    json.dumps(run_state, indent=2, sort_keys=True) + "\n"
)

verification = {
    "version": "kalbi_v2_v17_core_backup_verification_v1",
    "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    "remote_verified_files": len(remote_combined),
    "local_working_copy_files": sum(
        key.startswith("local_working_copy/") for key in local_manifest
    ),
    "missing_local": missing_local,
    "remote_mismatch": remote_mismatch,
    "unverified_local_excluding_local_working_copy": unverified_local,
    "bitwise_remote_verification_passed": True,
}
(BACKUP / "VERIFICATION.json").write_text(
    json.dumps(verification, indent=2, sort_keys=True) + "\n"
)

readme = f"""Kalbi V2 + Lambda Selector V17 core backup
Created: 2026-07-22 05:45:19 KST
Finalized UTC: {verification['verified_at_utc']}

Authoritative production server:
  ssh -p 19508 root@175.155.64.224
Authoritative N=200, P=500 reference server:
  ssh -p 40801 root@38.29.145.27

Contents:
  production_8gpu/       Active Kalbi V2 + V17 production package and producer wrapper
  reference_n200p500/    Small-data trainer, model, reports, and evaluation code
  data_generation/       Rolling generation package, exact generators, TabICL and TabPFN sources
  base_runtime/          FPK base model and recursive project-local Python dependencies
  supervisor/            Exact Supervisor configurations for trainer, curator, and 16 producers
  required_artifacts/    Exact canonical Direct96 penalty cache required by the trainer
  local_working_copy/    Local Codex working copy at backup time
  environment/           Python/PyTorch/CUDA snapshots, pip freezes, and remote source manifests
  RUN_STATE.json         Active run, catalog, Supervisor, and latest-checkpoint identity
  MANIFEST.json          Local file size and SHA256 inventory
  SHA256SUMS.txt         Portable SHA256 list
  VERIFICATION.json      Remote-to-local bitwise comparison result

Intentionally excluded:
  - 100GB rolling data payload
  - training checkpoints and optimizer states
  - logs, pycache, generated NPZ/NPY files, and Jupyter checkpoint folders
  - Markdown files; research documentation remains in the Obsidian vault

Required binary artifacts included:
  - penalty_cache_300k_v12_canonical_final.pt
  - official TabPFN-v1 prior checkpoint inside data_generation/TabPFN-v1.0.0

Latest training checkpoint is not copied. Its exact remote path and SHA256 are in RUN_STATE.json.
Research architecture note:
  /Users/minsuk.shin/Documents/Aa/Research/FM for Linear Model/Kalbi V2 A-B Ridge Trajectory Architecture K32 S32 2026-07-21.md
"""
(BACKUP / "BACKUP_README.txt").write_text(readme)

final_entries: dict[str, dict[str, int | str]] = {}
for path in sorted(BACKUP.rglob("*")):
    if not path.is_file():
        continue
    relative = path.relative_to(BACKUP).as_posix()
    if relative in {"MANIFEST.json", "SHA256SUMS.txt"}:
        continue
    final_entries[relative] = {
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }

manifest = {
    "version": "kalbi_v2_v17_core_backup_manifest_v1",
    "created_at_kst": "2026-07-22T05:45:19+09:00",
    "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
    "backup_root": str(BACKUP),
    "file_count": len(final_entries),
    "total_bytes": sum(int(value["bytes"]) for value in final_entries.values()),
    "entries": final_entries,
}
(BACKUP / "MANIFEST.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
(BACKUP / "SHA256SUMS.txt").write_text(
    "".join(
        f"{value['sha256']}  {key}\n"
        for key, value in sorted(final_entries.items())
    )
)

print(
    json.dumps(
        {
            "backup_root": str(BACKUP),
            "file_count": manifest["file_count"],
            "total_bytes": manifest["total_bytes"],
            "remote_verified_files": verification["remote_verified_files"],
            "latest_train_step": run_state["latest_train_step"],
            "latest_checkpoint": run_state["latest_checkpoint"],
            "latest_checkpoint_sha256": run_state["latest_checkpoint_sha256"],
        },
        indent=2,
        sort_keys=True,
    )
)
