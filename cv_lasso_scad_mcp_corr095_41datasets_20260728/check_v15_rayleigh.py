#!/usr/bin/env python3
"""V15 대량 생성 전 점검.  요청한 R 이 실제로 나오는지 본다.

생성기는 목표를 support Gram 의 고유값 범위로 잘라낸다.

    target = np.clip(target, low + 1e-9, high - 1e-9)      # v15:41

도달 못 하면 조용히 잘린다.  잘린 사실은 target 과 실현값을 비교해야 보인다.

featurewise Exp(1) 노이즈가 그 범위를 좁힌다.

    G' = D G D + diag(v_j / (1 + v_j)),   v_j ~ Exp(1)
    E[v/(1+v)] ~ 0.4   ->   w' ~ 0.6 w + 0.4

고유값이 1 쪽으로 압축되어 R 의 하한이 0.4 근처에서 막힌다.
real 13 의 R 범위는 [0.110, 6.06] 이고, 실패하는 데이터셋이 저-R 쪽이다.

주의: linear_rayleigh_quotient 는 npz 가 아니라 manifest.json 에 있다.
atomic_save_npz 의 키 목록에서 빠져 있다.

    python3.11 check_v15_rayleigh.py --dir out_v14_rq3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REAL_LO, REAL_HI = 0.110, 6.06          # real 13 실측 R 범위
NOISE_FLOOR = 0.4                        # Exp(1) 노이즈가 만드는 예상 하한


def collect(node, key, out):
    """중첩 JSON 어디에 있든 key 값을 모은다."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key and isinstance(v, (int, float)):
                out.append(float(v))
            else:
                collect(v, key, out)
    elif isinstance(node, list):
        for v in node:
            collect(v, key, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--out", default="v15_rayleigh_check.json")
    args = ap.parse_args()

    path = Path(args.dir) / args.manifest
    if not path.is_file():
        print(f"  없음: {path}")
        return
    man = json.loads(path.read_text(encoding="utf-8"))

    contract = str(man.get("dataset_contract", ""))
    print(f"  계약  {contract}")
    print(f"  V15 여부  {'예' if 'rq_v15' in contract else '아니오'}\n")

    R, T, S = [], [], []
    collect(man, "linear_rayleigh_quotient", R)
    collect(man, "linear_rayleigh_target", T)
    collect(man, "linear_support_size", S)
    R = np.asarray(R, dtype=np.float64)
    R = R[np.isfinite(R) & (R > 0)]
    T = np.asarray(T, dtype=np.float64)
    T = T[np.isfinite(T) & (T > 0)]

    if R.size == 0:
        print("  linear_rayleigh_quotient 를 못 찾았다")
        return
    print(f"  태스크 {R.size}개   (타겟 기록된 것 {T.size}개)\n")

    qs = (1, 5, 10, 25, 50, 75, 90, 95, 99)
    print("  실현된 R 분포")
    print("    " + "   ".join(f"{q}% {np.percentile(R, q):.3f}" for q in qs))
    print(f"    최소 {R.min():.4f}   최대 {R.max():.3f}")

    print()
    print(f"  real 13 범위  [{REAL_LO}, {REAL_HI}]")
    below = float((R < NOISE_FLOOR).mean())
    print(f"    R < {NOISE_FLOOR}  {below:6.2%}   <- Exp(1) 노이즈 하한 예측선")
    print(f"    R < 0.25 {float((R < 0.25).mean()):6.2%}")
    print(f"    R < {REAL_LO*1.5:.3f} {float((R < REAL_LO*1.5).mean()):6.2%}"
          "   <- 실패 데이터셋이 있는 구간")
    print(f"    R > 2.0  {float((R > 2.0).mean()):6.2%}")

    if T.size > 100:
        n = min(T.size, R.size)
        d = np.abs(np.log(R[:n]) - np.log(T[:n]))
        print()
        print("  요청 target 대비 실현 R  (순서 정렬이 같다는 가정)")
        print(f"    |log R - log target| > 0.05 인 비율  {float((d > 0.05).mean()):6.2%}")
        print(f"    요청 target  최소 {T.min():.4f}  최대 {T.max():.3f}")
        low = T[:n] < NOISE_FLOOR
        if low.sum() > 0:
            print(f"    target < {NOISE_FLOOR} 요청 {int(low.sum())}건의 실현 R "
                  f"중앙 {np.median(R[:n][low]):.3f}")
            print("      이 값이 0.4 근처면 노이즈 하한에 막힌 것이다")

    Path(args.out).write_text(json.dumps({
        "contract": contract, "n": int(R.size),
        "R_percentiles": {str(q): float(np.percentile(R, q)) for q in qs},
        "R_min": float(R.min()), "R_max": float(R.max()),
        "frac_below_noise_floor": below,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    if below < 0.05:
        print("  판정: 저-R 을 못 만들고 있다.  대량 생성 전에 노이즈를 고쳐라.")
        print("        add_featurewise_exp_variance_gaussian_noise 에 rate 를 뚫고")
        print("        태스크마다 흔들어라.  rate 를 올리면 v_j 가 작아져")
        print("        고유값 압축이 줄고 R 하한이 내려간다.")
    else:
        print("  판정: 저-R 이 나오고 있다.  대량 생성 진행 가능.")
    print(f"\n  저장: {args.out}")


if __name__ == "__main__":
    main()
