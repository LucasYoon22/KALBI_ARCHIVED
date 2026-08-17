#!/usr/bin/env python3
"""raw V15 npz 를 학습이 읽을 수 있는 rolling 카탈로그로 등록한다.

학습 스크립트는 npz 폴더를 직접 못 읽는다.  TrainerCatalogStore 가
control_root/CURRENT.json 을 읽고, 거기서 가리키는 카탈로그 파일을 열어
항목마다 sha256 / 차원 / 상관 인증서 / pipeline 해시를 검증한다.

여기서는 카탈로그를 손으로 조립하지 않는다.  store 의 내부 정규화 함수를
그대로 호출해서 만든다.  그래야 _validate_catalog_payload 를 반드시 통과한다.

    entries  = [store._normalise_entry(raw, assigned_id=i) for ...]
    entries.sort(key=store._entry_sort_key)
    catalog  = {version, epoch, data_root, files, accounting, assignments,
                capacity_policy}

항목 필드는 전부 npz 안에서 나온다.

    n, p           X.shape
    num_tasks      X.shape[0] (배치 차원)
    corr           sampled_corr_audit_max / core_exact_max_abs_corr 의 최대
    seed           batch_seed
    sha256         파일 해시
    pipeline       생성기 파일의 sha256.  --expected-pipeline-sha256 과 같은 값

디렉터리 배치

    <root>/rolling_control/CURRENT.json      포인터
    <root>/rolling_control/EPOCH_00000000.json
    <root>/data/tabicl_v2/batch_xxxxxx.npz   prior 이름이 첫 경로 조각

    python3 build_rolling_catalog.py \
        --src /workspace/v15_pool \
        --root /workspace/rolling_v15 \
        --generator <PKG>/generate_..._v15.py \
        --world-size 4 --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def npz_facts(path: Path) -> dict | None:
    """카탈로그 항목에 필요한 값을 npz 에서 뽑는다."""
    try:
        with np.load(path, allow_pickle=False) as z:
            x = z["X"]
            num_tasks, n, p = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2])) \
                if x.ndim == 3 else (1, int(x.shape[0]), int(x.shape[1]))
            corr = 0.0
            for key in ("sampled_corr_audit_max", "core_exact_max_abs_corr"):
                if key in z.files:
                    v = np.asarray(z[key], dtype=np.float64)
                    v = v[np.isfinite(v)]
                    if v.size:
                        corr = max(corr, float(v.max()))
            seed = 0
            if "batch_seed" in z.files:
                seed = int(np.asarray(z["batch_seed"]).reshape(-1)[0])
    except Exception:                                              # noqa: BLE001
        return None
    if n <= 0 or p <= 0 or num_tasks <= 0:
        return None
    if not (0.0 <= corr < 0.95):
        return None                                # 상관 인증서 조건 미달
    return dict(n=n, p=p, num_tasks=num_tasks,
                task_max_abs_corr_max=corr, seed=abs(seed))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="npz 가 있는 폴더")
    ap.add_argument("--root", required=True, help="만들 rolling root")
    ap.add_argument("--generator", required=True,
                    help="V15 생성기 경로.  pipeline sha256 의 근거")
    ap.add_argument("--pkg", default=os.environ.get(
        "KALBI_PKG",
        "kalbi_v6_exactprox10_eta4x_v24_additive512_traj3_k32s32_p100k_effbatch32"))
    ap.add_argument("--world-size", type=int, default=4)
    ap.add_argument("--prior", default="tabicl_v2")
    ap.add_argument("--smoke", action="store_true",
                    help="학습의 --smoke-small-catalog 와 같은 용량 정책")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--move", action="store_true",
                    help="복사 대신 이동.  디스크가 빠듯할 때")
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
    # TrainerCatalogStore 는 학습 스크립트가 CatalogStore 를 상속해 만든
    # 하위 클래스다.  epoch 해석만 덧붙였을 뿐 검증 로직은 기반 클래스에 있다.
    store = rc.CatalogStore(control_root, data_root,
                            policy=policy, priors=(args.prior,))

    pipeline_hash = sha256_file(Path(args.generator))
    print(f"  pipeline sha256  {pipeline_hash}")
    print(f"  world_size {args.world_size}   smoke {args.smoke}\n")

    files = sorted(Path(args.src).glob("batch_*.npz"))
    if args.limit:
        files = files[: args.limit]
    print(f"  후보 {len(files)}개")

    raws, skipped, total = [], 0, 0
    for f in files:
        facts = npz_facts(f)
        if facts is None:
            skipped += 1
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
            source_hashes={"generator": pipeline_hash},
        ))
        if len(raws) % 50 == 0:
            print(f"    {len(raws)}개 처리")

    if not raws:
        print("  등록할 항목이 없다")
        return 1
    print(f"  등록 {len(raws)}개   건너뜀 {skipped}개   "
          f"{total/1e9:.2f} GB\n")

    entries = [store._normalise_entry(r, assigned_id=i)
               for i, r in enumerate(raws)]
    entries.sort(key=store._entry_sort_key)
    entries = [store._normalise_entry(e, assigned_id=i)
               for i, e in enumerate(entries)]
    entries.sort(key=store._entry_sort_key)

    catalog = {
        "version": rc.CATALOG_VERSION,
        "epoch": 0,
        "data_root": str(data_root),
        "files": entries,
        "accounting": store._accounting(entries),
        "assignments": store._assignments(entries),
        "capacity_policy": policy.to_dict(),
    }
    store._validate_catalog_payload(catalog)          # 실패하면 여기서 죽는다
    print("  카탈로그 검증 통과")

    # 카탈로그 본문은 control_root/catalogs/ 아래, 이름은
    #     CATALOG_EPOCH_<8자리>_<sha256 앞 16자리>.json
    # 규약이다.  해시가 이름에 들어가므로 먼저 바이트를 확정한 뒤 이름을 짓는다.
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
    print(f"  CURRENT 로드 확인   epoch {cat['epoch']}   "
          f"files {len(cat['files'])}")
    for r in range(args.world_size):
        k = len(store.entries_for_rank(cat, r))
        print(f"    rank {r}  shard {k}")

    print()
    print("  학습 인자로 넘길 값")
    print(f"    --rolling-control-root {control_root}")
    print(f"    --rolling-data-root    {data_root}")
    print(f"    --expected-pipeline-sha256 {pipeline_hash}")
    if args.smoke:
        print("    --smoke-small-catalog")
    return 0


if __name__ == "__main__":
    sys.exit(main())
