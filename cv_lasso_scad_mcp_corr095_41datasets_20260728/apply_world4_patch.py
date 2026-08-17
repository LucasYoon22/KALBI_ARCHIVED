#!/usr/bin/env python3
"""학습 스크립트를 4-GPU 로 돌릴 수 있게 한다.  유효 배치는 그대로 32.

    8 ranks x batch 1 x accum 4 = 32
    4 ranks x batch 1 x accum 8 = 32      <- 동일

랭크당 작업량이 같으므로 VRAM 도 같고, 최적화 궤적도 같다.
바뀌는 건 벽시계 시간뿐이다 (스텝당 순차 accum 이 2배).

왜 필요한가
    A4000(미국, 4 GPU) 에는 코드·캐시·체크포인트·torch 가 이미 다 있다.
    8 GPU 는 3080(중국) 뿐인데 전송이 분당 1 MB 로 막혔다.
    시험 학습은 A4000 에서 바로 돌리는 편이 빠르다.

예상 소요 (A4000 4장, 3080 8장 대비 약 2.5배)
    100 스텝   7.3분 -> 18분
    2k  스텝   2.5시간 -> 6시간
    30k 스텝   37시간 -> 3.7일     <- 전체 학습은 8 GPU 를 구하는 편이 낫다

    python3.11 apply_world4_patch.py <PKG>/train_selector_only_..._8gpu.py
    python3.11 apply_world4_patch.py <경로> --revert
    python3.11 apply_world4_patch.py <경로> --check
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

PAIRS = [
    ("EXPECTED_WORLD_SIZE = 8", "EXPECTED_WORLD_SIZE = 4"),
    ("EXPECTED_GRAD_ACCUM_STEPS = 4", "EXPECTED_GRAD_ACCUM_STEPS = 8"),
    ("EXPECTED_SELECTOR_GRAD_ACCUM_STEPS = 4",
     "EXPECTED_SELECTOR_GRAD_ACCUM_STEPS = 8"),
    ('raise TrainerError("P100K rolling trainer requires exactly eight ranks")',
     'raise TrainerError(\n            f"P100K rolling trainer requires exactly "\n            f"{EXPECTED_WORLD_SIZE} ranks, got {world}"\n        )'),
]
MARK = "EXPECTED_WORLD_SIZE = 4"


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
    backup = t.with_suffix(".py.world8")

    if args.check:
        print(f"  {t.name}\n  sha256 {sha(t)}\n  "
              f"4-GPU 패치 {'적용됨' if applied else '미적용'}")
        return 0

    if args.revert:
        if not backup.is_file():
            print(f"  백업 없음: {backup}")
            return 1
        shutil.copy2(backup, t)
        print(f"  되돌림 (8 GPU).  sha256 {sha(t)}")
        return 0

    if applied:
        print(f"  이미 적용됨.  sha256 {sha(t)}")
        return 0

    for old, _ in PAIRS:
        if text.count(old) != 1:
            print(f"  앵커를 {text.count(old)}번 찾음 (1이어야 함):\n  {old!r}")
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

    print("  적용 완료  (world 8->4, accum 4->8, 유효 배치 32 유지)")
    print(f"  sha256 {sha(t)}")
    print()
    print("  실행 시 인자도 같이 바꿔야 한다:")
    print("    --grad-accum-steps 8 --selector-grad-accum-steps 8")
    print("    torchrun --nproc_per_node=4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
