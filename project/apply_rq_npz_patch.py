#!/usr/bin/env python3
"""V15 생성기가 linear_rayleigh_quotient 를 npz 에도 저장하게 한다.

현재는 계산만 하고 manifest.json 에만 남는다.
atomic_save_npz 의 키 목록에서 빠져 있다.

    "linear_rayleigh_quotient": float(
        np.var(x64 @ beta) / max(float(beta @ beta), 1e-30)
    ),                                                    # v15:805

왜 필요한가
    1. 학습 중 태스크별 R 을 조건이나 로깅으로 쓰려면 npz 에 있어야 한다.
       manifest 는 배치 단위 요약이라 학습 루프에서 읽기 불편하다.
    2. "V15 가 real 의 R 범위를 덮는다" 는 그림을 논문에 넣으려면
       배치마다 바로 읽히는 편이 낫다.
    3. 대량 생성 후에 넣으면 전부 다시 만들어야 한다.  지금 넣어야 한다.

linear_realized_snrs 와 완전히 같은 패턴으로 붙인다.
비선형 태스크에는 키가 없으므로 NaN 을 넣는다.

    python3.11 apply_rq_npz_patch.py <생성기.py>
    python3.11 apply_rq_npz_patch.py <생성기.py> --revert
    python3.11 apply_rq_npz_patch.py <생성기.py> --check
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

MARK = "linear_rayleigh_quotients"

PAIRS = [
    # 1) 리스트 선언
    ("        linear_realized_snrs: list[float] = []",
     "        linear_realized_snrs: list[float] = []\n"
     "        linear_rayleigh_quotients: list[float] = []"),
    # 2) 태스크마다 수집.  비선형이면 NaN
    ('                    linear_realized_snrs.append(\n'
     '                        float(y_meta["linear_realized_snr"])\n'
     '                    )',
     '                    linear_realized_snrs.append(\n'
     '                        float(y_meta["linear_realized_snr"])\n'
     '                    )\n'
     '                    linear_rayleigh_quotients.append(\n'
     '                        float(\n'
     '                            y_meta.get(\n'
     '                                "linear_rayleigh_quotient",\n'
     '                                float("nan"),\n'
     '                            )\n'
     '                        )\n'
     '                    )'),
    # 3) npz 저장 키
    ("            linear_realized_snr=np.asarray(\n"
     "                linear_realized_snrs, dtype=np.float32\n"
     "            ),",
     "            linear_realized_snr=np.asarray(\n"
     "                linear_realized_snrs, dtype=np.float32\n"
     "            ),\n"
     "            linear_rayleigh_quotient=np.asarray(\n"
     "                linear_rayleigh_quotients, dtype=np.float32\n"
     "            ),"),
]


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
    backup = t.with_suffix(".py.prerq")

    if args.check:
        print(f"  {t.name}\n  sha256 {sha(t)}\n  "
              f"rq-npz 패치 {'적용됨' if applied else '미적용'}")
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

    for i, (old, _) in enumerate(PAIRS, 1):
        c = text.count(old)
        if c != 1:
            print(f"  앵커 {i} 을 {c}번 찾음 (1이어야 함):")
            print("  ---"); print(old); print("  ---")
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

    print("  적용 완료.  npz 에 linear_rayleigh_quotient 가 저장된다")
    print(f"  sha256 {sha(t)}")
    print()
    print("  확인:  소량 생성 후")
    print("    python3 -c \"import numpy as np,glob;"
          "z=np.load(sorted(glob.glob('OUT/batch_*.npz'))[0]);"
          "print(z['linear_rayleigh_quotient'])\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
