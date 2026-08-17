#!/usr/bin/env python3
"""V15 raw npz -> rolling 카탈로그, semantic_contract 인증서까지 정직하게 채운다.

manifest.json 은 안 쓴다 (508개 중 42개만 있어서 조인 불가능함을 확인함).
필요한 인증서 값은 npz 배열 자체에 전부 들어있다
(outcome_source_id, linear_support_*, linear_target_snr, linear_realized_snr,
linear_snr_regime_id, scm_response_mean, scm_response_std).  트레이너
(train_selector_only_..._v14_transition_8gpu.py, apply_v15_contract_patch.py
적용본)가 카탈로그 항목마다 요구하는 semantic_contract 를 이 npz 값 그대로
채운다 -- 지어내는 값이 없다.

    linear_support_strict_n_third_verified 는 "3*support_size < n" 을
    이 스크립트가 직접 계산한다 (npz 에 이미 존재하는 support_size, n 으로부터
    나오는 참/거짓이라 계산이지 창작이 아니다).  False 인 조밀-서포트 태스크는
    트레이너 검증을 못 통과하므로 여기서 제외한다.

    python3 build_rolling_catalog_v15.py \
        --src /workspace/rolling_v15_8gpu/data/tabicl_v2 \
        --root /workspace/rolling_v15_8gpu \
        --generator generate_tabiclv2_scm2048_qfilter_bnn128_mixedlinear_rayleigh_v15.py \
        --world-size 8
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import shutil
import sys
from pathlib import Path

import numpy as np

TABICL_TASK_TARGET_CONTRACT = (
    "official_tabiclv2_graph_scm_y_qualityfilter_center_scale_clip10_rescale_v2"
)
LINEAR_TASK_TARGET_CONTRACT_V15 = (
    "controlled_linear_y_supportmix_pn50_maxsupp0p6n_"
    "snr_loguniform_0p2_50_errorfamily_"
    "qualityfilter_center_scale_clip10_rescale_rq_v4"
)
V15_DATASET_CONTRACT = (
    "tabiclv2_graph_scm2048_qfilter_bnn128_expvarnoise_"
    "mixed50_tabicl_linear_supportmix_pn_maxsupp0p6n_snr0p2to50_rq_v15"
)
V15_MIXED_RESPONSE_CONTRACT = (
    "mixed50_official_tabicl_or_controlled_linear_supportmix_pn50_"
    "maxsupp0p6n_snr_loguniform_0p2_50_y_"
    "qualityfilter_center_scale_clip10_rescale_rq_v4"
)
LINEAR_SNR_CONTRACT = "linear_target_snr_loguniform_0p2_50_v1"
LINEAR_SNR_DISTRIBUTION = "log_uniform"
OUTCOME_SOURCE_NAMES = {0: "tabicl", 1: "linear"}
SUPPORT_REFERENCE_NAMES = {0: "p_based", 1: "n_based"}
OUTCOME_SOURCE_TABICL = 0
OUTCOME_SOURCE_LINEAR = 1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def npz_facts(path: Path) -> dict | None:
    try:
        with np.load(path, allow_pickle=False) as z:
            x = z["X"]
            if x.ndim != 3 or x.shape[0] != 1:
                return None
            n, p = int(x.shape[1]), int(x.shape[2])
            if n <= 0 or p <= 0:
                return None

            corr = 0.0
            for key in ("sampled_corr_audit_max", "core_exact_max_abs_corr"):
                if key in z.files:
                    v = np.asarray(z[key], dtype=np.float64)
                    v = v[np.isfinite(v)]
                    if v.size:
                        corr = max(corr, float(v.max()))
            if not (0.0 <= corr < 0.95):
                return None

            seed = int(np.asarray(z["batch_seed"]).reshape(-1)[0]) \
                if "batch_seed" in z.files else 0

            outcome_source_id = int(np.asarray(z["outcome_source_id"]).reshape(-1)[0])
            if outcome_source_id not in (OUTCOME_SOURCE_TABICL, OUTCOME_SOURCE_LINEAR):
                return None

            # next_batch() (train_beta_only_v23..._transition_8gpu.py) 는
            # outcome 종류와 상관없이 이 8개 필드를 무조건 int()/float() 로
            # 캐스팅해서 읽는다.  tabicl-outcome 태스크도 생성기가 이미
            # 센티널 값(255, -1.0, 0)을 넣어뒀으므로 조건 없이 그대로 옮긴다.
            support_size = int(np.asarray(z["linear_support_size"]).reshape(-1)[0])
            support_fraction = float(
                np.asarray(z["linear_support_fraction"]).reshape(-1)[0]
            )
            support_regime_id = int(
                np.asarray(z["linear_support_regime_id"]).reshape(-1)[0]
            )
            coefficient_family_id = int(
                np.asarray(z["linear_coefficient_family_id"]).reshape(-1)[0]
            )
            error_family_id = int(
                np.asarray(z["linear_error_family_id"]).reshape(-1)[0]
            )
            snr_regime_id = int(
                np.asarray(z["linear_snr_regime_id"]).reshape(-1)[0]
            )
            target_snr = float(np.asarray(z["linear_target_snr"]).reshape(-1)[0])
            realized_snr = float(np.asarray(z["linear_realized_snr"]).reshape(-1)[0])

            semantic = {
                "contract": V15_DATASET_CONTRACT,
                "target_contract": V15_MIXED_RESPONSE_CONTRACT,
                "outcome_source_id": outcome_source_id,
                "outcome_source": OUTCOME_SOURCE_NAMES.get(outcome_source_id),
                "linear_support_size": support_size,
                "linear_support_fraction": support_fraction,
                "linear_support_regime_id": support_regime_id,
                "linear_coefficient_family_id": coefficient_family_id,
                "linear_error_family_id": error_family_id,
                "linear_snr_regime_id": snr_regime_id,
                "linear_target_snr": target_snr,
                "linear_realized_snr": realized_snr,
            }

            if outcome_source_id == OUTCOME_SOURCE_LINEAR:
                if not (3 * support_size < n):
                    return None                    # 조밀 서포트, 정당하게 제외
                support_ref_id = int(
                    np.asarray(z["linear_support_reference_id"]).reshape(-1)[0]
                )
                semantic.update(
                    task_target_contract=LINEAR_TASK_TARGET_CONTRACT_V15,
                    linear_support_fraction_of_n=float(
                        np.asarray(z["linear_support_fraction_of_n"]).reshape(-1)[0]
                    ),
                    linear_support_reference_id=support_ref_id,
                    linear_support_reference=SUPPORT_REFERENCE_NAMES.get(support_ref_id),
                    linear_support_strict_n_third_verified=True,
                    linear_snr_contract=LINEAR_SNR_CONTRACT,
                    linear_snr_distribution=LINEAR_SNR_DISTRIBUTION,
                )
            else:
                semantic["task_target_contract"] = TABICL_TASK_TARGET_CONTRACT

            mean = abs(float(np.asarray(z["scm_response_mean"]).reshape(-1)[0]))
            std = float(np.asarray(z["scm_response_std"]).reshape(-1)[0])
            semantic["scm_response_mean_max_abs"] = mean
            semantic["scm_response_std_min"] = std
            semantic["scm_response_std_max"] = std
    except Exception:                                              # noqa: BLE001
        return None

    return dict(n=n, p=p, num_tasks=1, task_max_abs_corr_max=corr,
                seed=abs(seed), semantic=semantic)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--generator", required=True)
    ap.add_argument("--pkg", default=os.environ.get(
        "KALBI_PKG",
        "kalbi_v6_exactprox10_eta4x_v24_additive512_traj3_k32s32_p100k_effbatch32"))
    ap.add_argument("--world-size", type=int, default=8)
    ap.add_argument("--prior", default="tabicl_v2")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--move", action="store_true")
    args = ap.parse_args()

    rc = importlib.import_module(f"{args.pkg}.rolling_catalog")

    root = Path(args.root).resolve()
    data_root = root / "data"
    control_root = root / "rolling_control"
    prior_dir = data_root / args.prior
    prior_dir.mkdir(parents=True, exist_ok=True)
    control_root.mkdir(parents=True, exist_ok=True)

    policy = rc.CapacityPolicy(
        active_min_bytes=20_000_000, active_max_bytes=40_000_000,
        prior_min_bytes=10_000_000, prior_max_bytes=20_000_000,
        hard_cap_bytes=100_000_000, min_free_bytes=0,
        world_size=args.world_size,
    ) if args.smoke else rc.CapacityPolicy(
        active_min_bytes=500_000_000, active_max_bytes=102_000_000_000,
        prior_min_bytes=500_000_000, prior_max_bytes=102_000_000_000,
        hard_cap_bytes=130_000_000_000, min_free_bytes=300_000_000_000,
        world_size=args.world_size,
    )
    store = rc.CatalogStore(control_root, data_root,
                            policy=policy, priors=(args.prior,))

    pipeline_hash = sha256_file(Path(args.generator))
    print(f"  pipeline sha256  {pipeline_hash}")
    print(f"  world_size {args.world_size}   smoke {args.smoke}\n")

    files = sorted(Path(args.src).glob("batch_*.npz"))
    if args.limit:
        files = files[: args.limit]
    print(f"  후보 {len(files)}개")

    raws, skipped_dense, skipped_other, total = [], 0, 0, 0
    for f in files:
        facts = npz_facts(f)
        if facts is None:
            try:
                with np.load(f, allow_pickle=False) as z:
                    osid = int(np.asarray(z["outcome_source_id"]).reshape(-1)[0])
                    ssz = int(np.asarray(z["linear_support_size"]).reshape(-1)[0])
                    n_ = int(z["X"].shape[1])
                    dense = osid == OUTCOME_SOURCE_LINEAR and not (3 * ssz < n_)
            except Exception:                                       # noqa: BLE001
                dense = False
            if dense:
                skipped_dense += 1
            else:
                skipped_other += 1
            continue
        dst = prior_dir / f.name
        if not dst.exists():
            (shutil.move if args.move else shutil.copy2)(str(f), str(dst))
        size = dst.stat().st_size
        total += size
        raws.append(dict(
            relative_path=f"{args.prior}/{f.name}",
            prior=args.prior, bytes=size, sha256=sha256_file(dst),
            seed=facts["seed"], n=facts["n"], p=facts["p"],
            num_tasks=facts["num_tasks"],
            task_max_abs_corr_max=facts["task_max_abs_corr_max"],
            pipeline_config_sha256=pipeline_hash,
            source_hashes={"generator_tabicl_v2": pipeline_hash},
            metadata={"semantic_contract": facts["semantic"]},
        ))
        if len(raws) % 50 == 0:
            print(f"    {len(raws)}개 처리")

    if not raws:
        print("  등록할 항목이 없다")
        return 1
    print(f"  등록 {len(raws)}개   조밀서포트제외 {skipped_dense}개   "
          f"기타건너뜀 {skipped_other}개   {total/1e9:.2f} GB\n")

    entries = [store._normalise_entry(r, assigned_id=i)
               for i, r in enumerate(raws)]
    entries.sort(key=store._entry_sort_key)
    entries = [store._normalise_entry(e, assigned_id=i)
               for i, e in enumerate(entries)]
    entries.sort(key=store._entry_sort_key)

    catalog = {
        "version": rc.CATALOG_VERSION,
        "epoch": 0,
        "parent": None,
        "data_root": str(data_root),
        "contract": {
            "migration": "fresh_canonical_staging_bootstrap_v1",
            "pipeline_config_sha256": pipeline_hash,
            "source_hashes": {"generator_tabicl_v2": pipeline_hash},
        },
        "files": entries,
        "additions": [int(e["shard_id"]) for e in entries],
        "retirements": [],
        "accounting": store._accounting(entries),
        "assignments": store._assignments(entries),
        "capacity_policy": policy.to_dict(),
    }
    store._validate_catalog_payload(catalog)
    print("  카탈로그 검증 통과")

    payload = rc._json_bytes(catalog)
    digest = hashlib.sha256(payload).hexdigest()
    name = f"CATALOG_EPOCH_{0:08d}_{digest[:16]}.json"
    store.catalog_dir.mkdir(parents=True, exist_ok=True)
    (store.catalog_dir / name).write_bytes(payload)
    pointer = {
        "version": rc.POINTER_VERSION,
        "epoch": 0,
        "catalog_file": name,
        "catalog_sha256": digest,
    }
    rc.atomic_write_json(control_root / "CURRENT.json", pointer)

    ptr, cat = store.load_current()
    print(f"  CURRENT 로드 확인   epoch {cat['epoch']}   files {len(cat['files'])}")
    for r in range(args.world_size):
        k = len(store.entries_for_rank(cat, r))
        print(f"    rank {r}  shard {k}")

    print()
    print("  학습 인자로 넘길 값")
    print(f"    --rolling-control-root {control_root}")
    print(f"    --rolling-data-root    {data_root}")
    print(f"    --expected-pipeline-sha256 {pipeline_hash}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
