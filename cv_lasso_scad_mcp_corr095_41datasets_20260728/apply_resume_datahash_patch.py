#!/usr/bin/env python3
"""plain --resume 의 required_resume 계약에서 source_hashes / exact_training_arguments
두 항목만 뺀다.  world_size / trainer_version / checkpoint_version / contract 버전들은
그대로 다 체크한다 (진짜 무결성).  이 두 개만 데이터 파이프라인을 V14->V15 로
의도적으로 바꿨을 때 당연히 달라지는 값이라 뺀다.

    python3 apply_resume_datahash_patch.py <경로>
    python3 apply_resume_datahash_patch.py <경로> --revert
    python3 apply_resume_datahash_patch.py <경로> --check
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

OLD = (
    '            "source_hashes": source_hashes,\n'
    '            "exact_training_arguments": exact_arguments,\n'
    '            "world_size": world,\n'
    '        }'
)
NEW = (
    '            "world_size": world,\n'
    '        }'
)
MARK = "# v15_resume_datahash_patch_applied"


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
    backup = t.with_suffix(".py.beforedatahashpatch")

    if args.check:
        print(f"  {t.name}\n  sha256 {sha(t)}\n  "
              f"resume datahash 패치 {'적용됨' if applied else '미적용'}")
        return 0

    if args.revert:
        if not backup.is_file():
            print(f"  백업 없음: {backup}")
            return 1
        shutil.copy2(backup, t)
        print(f"  되돌림.  sha256 {sha(t)}")
        return 0

    if applied:
        print(f"  이미 적용됨.  sha256 {sha(t)}")
        return 0

    n = text.count(OLD)
    if n != 1:
        print(f"  앵커를 {n}번 찾음 (1이어야 함)")
        return 1

    if not backup.is_file():
        shutil.copy2(t, backup)
        print(f"  백업 생성  {backup.name}")

    text = text.replace(OLD, NEW + "\n" + MARK if False else NEW, 1)
    # 마크는 파일 어딘가에 남겨야 하니 import 구역 근처에 주석으로 삽입한다.
    text = text.replace(
        "from __future__ import annotations",
        "from __future__ import annotations\n" + MARK,
        1,
    )
    t.write_text(text, encoding="utf-8")

    import py_compile
    try:
        py_compile.compile(str(t), doraise=True)
    except Exception as exc:                                       # noqa: BLE001
        shutil.copy2(backup, t)
        print(f"  컴파일 실패, 되돌림: {exc}")
        return 1

    print("  적용 완료  (required_resume 에서 source_hashes / "
          "exact_training_arguments 제외)")
    print(f"  sha256 {sha(t)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
