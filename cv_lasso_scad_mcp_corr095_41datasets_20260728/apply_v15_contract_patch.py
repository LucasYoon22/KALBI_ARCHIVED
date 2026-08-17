#!/usr/bin/env python3
"""트레이너의 V14 계약 상수 3개를 V15 실측값으로 바꾼다.  로직은 안 건드린다.

바뀌는 상수와 근거
    SCM_CONTRACTS["tabicl_v2"]     V15 manifest.json 의 dataset_contract /
                                    files[0].target_contract 그대로
    LINEAR_TASK_TARGET_CONTRACT    V15 생성기의 LINEAR_RESPONSE_CONTRACT 그대로
    V14_GENERATOR_SHA256           V15 생성기 파일의 sha256
                                    (826bf53977ec5d90c300544e5340e2e2b1f227f0ad0e66690ae661e70eba5065)

안 바뀌는 것 (이미 V15 와 문자 그대로 일치해서 건드릴 필요가 없었음)
    TABICL_TASK_TARGET_CONTRACT
    V14_LINEAR_SNR_CONTRACT / DISTRIBUTION / MIN / MAX
    V14_SNR_REGIME_NAMES (5개, snr_edges 도 그대로 재사용)

linear_support_strict_n_third_verified 같은 검증 *로직*은 그대로 둔다.
그건 "3*k < n" 이라는 진짜 통계적 조건이라 V15 조밀-서포트 태스크는
정당하게 걸러지는 게 맞다 (build_rolling_catalog.py 쪽에서 사전에 제외).

    python3 apply_v15_contract_patch.py <경로>
    python3 apply_v15_contract_patch.py <경로> --revert
    python3 apply_v15_contract_patch.py <경로> --check
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

V15_DATASET_CONTRACT = (
    "tabiclv2_graph_scm2048_qfilter_bnn128_expvarnoise_"
    "mixed50_tabicl_linear_supportmix_pn_maxsupp0p6n_snr0p2to50_rq_v15"
)
V15_MIXED_RESPONSE_CONTRACT = (
    "mixed50_official_tabicl_or_controlled_linear_supportmix_pn50_"
    "maxsupp0p6n_snr_loguniform_0p2_50_y_"
    "qualityfilter_center_scale_clip10_rescale_rq_v4"
)
V15_LINEAR_RESPONSE_CONTRACT = (
    "controlled_linear_y_supportmix_pn50_maxsupp0p6n_"
    "snr_loguniform_0p2_50_errorfamily_"
    "qualityfilter_center_scale_clip10_rescale_rq_v4"
)
V15_GENERATOR_SHA256 = (
    "826bf53977ec5d90c300544e5340e2e2b1f227f0ad0e66690ae661e70eba5065"
)

PAIRS = [
    (
        '    "tabicl_v2": (\n'
        '        "tabiclv2_graph_scm2048_qfilter_bnn128_expvarnoise_"\n'
        '        "mixed50_tabicl_linear_supportmix_pn_strictn3_snr0p2to50_v14",\n'
        '        "mixed50_official_tabicl_or_controlled_linear_supportmix_pn50_"\n'
        '        "strictn3_snr_loguniform_0p2_50_y_qualityfilter_center_scale_"\n'
        '        "clip10_rescale_v3",\n'
        '    ),',
        '    "tabicl_v2": (\n'
        f'        {V15_DATASET_CONTRACT!r},\n'
        f'        {V15_MIXED_RESPONSE_CONTRACT!r},\n'
        '    ),',
    ),
    (
        'LINEAR_TASK_TARGET_CONTRACT = (\n'
        '    "controlled_linear_y_supportmix_pn50_strictn3_snr_loguniform_0p2_50_"\n'
        '    "errorfamily_qualityfilter_center_scale_clip10_rescale_v3"\n'
        ')',
        f'LINEAR_TASK_TARGET_CONTRACT = {V15_LINEAR_RESPONSE_CONTRACT!r}',
    ),
    (
        'V14_GENERATOR_SHA256 = (\n'
        '    "d2548b21c63f220ec2d2c5f34bcf1a341e17b4440eb3db80f1ad3ad1029e812f"\n'
        ')',
        f'V14_GENERATOR_SHA256 = {V15_GENERATOR_SHA256!r}',
    ),
]
MARK = V15_GENERATOR_SHA256


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=Path)
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    t: Path = args.target
    if not t.is_file():
        print(f"  없음: {t}")
        return 1
    text = t.read_text(encoding="utf-8")
    applied = MARK in text
    backup = t.with_suffix(".py.v14contract")

    if args.check:
        print(f"  {t.name}\n  sha256 {sha(t)}\n  "
              f"V15 계약 패치 {'적용됨' if applied else '미적용'}")
        return 0

    if args.revert:
        if not backup.is_file():
            print(f"  백업 없음: {backup}")
            return 1
        shutil.copy2(backup, t)
        print(f"  되돌림 (V14 계약).  sha256 {sha(t)}")
        return 0

    if applied:
        print(f"  이미 적용됨.  sha256 {sha(t)}")
        return 0

    for old, _ in PAIRS:
        n = text.count(old)
        if n != 1:
            print(f"  앵커를 {n}번 찾음 (1이어야 함):\n  {old[:80]!r} ...")
            return 1

    if not backup.is_file():
        shutil.copy2(t, backup)
        print(f"  백업 생성  {backup.name}")

    for old, new in PAIRS:
        text = text.replace(old, new, 1)
    t.write_text(text, encoding="utf-8")

    import py_compile
    try:
        py_compile.compile(str(t), doraise=True)
    except Exception as exc:                                       # noqa: BLE001
        shutil.copy2(backup, t)
        print(f"  컴파일 실패, 되돌림: {exc}")
        return 1

    print("  적용 완료  (SCM_CONTRACTS, LINEAR_TASK_TARGET_CONTRACT, "
          "V14_GENERATOR_SHA256 -> V15 실측값)")
    print(f"  sha256 {sha(t)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
