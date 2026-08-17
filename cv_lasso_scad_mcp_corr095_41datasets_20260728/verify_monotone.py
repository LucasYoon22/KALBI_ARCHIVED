#!/usr/bin/env python3
"""셀렉터 헤드의 단조 패치를 검증한다.

통과 조건
    1. canonical_penalty.py 가 원본 그대로       (캐시 무결성)
    2. 300k 페널티 캐시가 정상 로드됨
    3. rho' 비증가 / kappa = 1 / rho(6) = 1
    4. 캐시 통계로 디코딩한 d0 가 유용 구간에 걸침

4번이 핵심이다.  학습은 N(0,1) 이 아니라 fit_parameter_cache 가 맞춘
parameter_mean/std 로 보정된 분포를 본다.

    python3.11 verify_monotone.py --cache penalty_cache_300k_v12_canonical_final.pt
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import torch

DEFAULT_PKG = (
    "kalbi_v6_exactprox10_eta4x_v24_additive512_traj3_k32s32_p100k_effbatch32"
)
# real 13 family sweep 승자들이 실제로 쓴 d0 구간
USEFUL_LO, USEFUL_HI = 0.167, 8.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default=os.environ.get("KALBI_PKG", DEFAULT_PKG))
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260811)
    args = ap.parse_args()

    sel = importlib.import_module(f"{args.pkg}.selector_v11_canonical")
    cp = importlib.import_module(f"{args.pkg}.canonical_penalty")

    flag = getattr(sel, "MONOTONE_HINGE_WEIGHTS", None)
    scale = getattr(sel, "MONOTONE_HINGE_SCALE", None)
    print(f"  MONOTONE_HINGE_WEIGHTS = {flag}")
    print(f"  MONOTONE_HINGE_SCALE   = {scale}")
    if flag is not True:
        print("  패치 미적용.  apply_monotone_patch.py 를 먼저 실행하라")
        return 1
    if hasattr(cp, "MONOTONE_HINGE_WEIGHTS"):
        print("  canonical_penalty.py 에 패치가 남아 있다.")
        print("  --revert 로 되돌려라.  캐시 소스 해시가 깨진다")
        return 1
    print()

    g = torch.Generator().manual_seed(args.seed)
    knots = cp.canonical_grid()
    build = lambda p: sel.build_penalty_monotone(
        p, derivative_floor=cp.DEFAULT_DERIVATIVE_FLOOR,
        squareplus_scale=cp.DEFAULT_SQUAREPLUS_SCALE, knots=knots)

    def structural(params: torch.Tensor, label: str) -> tuple[bool, np.ndarray]:
        d0s = []
        rise = -1e30
        kmin, kmax = 1e30, -1e30
        r6min, r6max = 1e30, -1e30
        for i in range(0, params.shape[0], 2048):
            enc = build(params[i:i + 2048])
            d, r = enc["drho"], enc["rho"]
            rise = max(rise, float((d[:, 1:] - d[:, :-1]).amax()))
            k = d[:, 0] / d.amax(-1).clamp_min(1e-12)
            kmin, kmax = min(kmin, float(k.min())), max(kmax, float(k.max()))
            r6min = min(r6min, float(r[:, -1].min()))
            r6max = max(r6max, float(r[:, -1].max()))
            d0s.append(d[:, 0].numpy())
        c1, c2 = rise <= 1e-6, abs(kmin - 1) < 1e-5 and abs(kmax - 1) < 1e-5
        c3 = abs(r6min - 1) < 1e-4 and abs(r6max - 1) < 1e-4
        print(f"  {label:22s} 최대상승 {rise:+.1e} {'OK' if c1 else '실패'}   "
              f"kappa [{kmin:.6f},{kmax:.6f}] {'OK' if c2 else '실패'}   "
              f"rho(6) [{r6min:.5f},{r6max:.5f}] {'OK' if c3 else '실패'}")
        return c1 and c2 and c3, np.concatenate(d0s)

    ok = True
    for s in (0.1, 1.0, 10.0):
        raw = torch.randn(4096, cp.DIRECT_PARAMETER_DIM, generator=g) * s
        good, _ = structural(raw, f"무작위 scale {s}")
        ok &= good

    if args.cache is None:
        print()
        print("  --cache 없이는 대표성 있는 판정을 못 한다.")
        print("  학습이 보는 분포는 캐시 통계로 보정된 것이다")
        return 0 if ok else 1

    print()
    cache_mod = importlib.import_module(f"{args.pkg}.canonical_penalty_cache")
    cache = cache_mod.CanonicalPenaltyCache(args.cache, 0)
    try:
        meta = cache.load()
    except Exception as exc:                                       # noqa: BLE001
        print(f"  캐시 로드 실패: {type(exc).__name__}: {exc}")
        print("  canonical_penalty.py 가 원본인지 확인하라")
        return 1
    # load() 는 메타데이터만 반환하고 파라미터는 인스턴스에 담는다
    params = cache.direct_parameters
    print(f"  캐시 로드 OK   {tuple(params.shape)}   "
          f"contract {meta['canonical_contract_version'][:28]}...")

    idx = torch.randperm(params.shape[0], generator=g)[:args.samples]
    good, d0 = structural(params.index_select(0, idx), "캐시 파라미터")
    ok &= good

    print()
    print("  캐시 파라미터 d0 분포   (학습이 실제로 보는 것)")
    line = "   ".join(f"{q}% {np.percentile(d0, q):.3f}"
                      for q in (1, 5, 25, 50, 75, 95, 99))
    print(f"    {line}")

    inside = float(((d0 >= USEFUL_LO) & (d0 <= USEFUL_HI)).mean())
    med = float(np.median(d0))
    print()
    print(f"  유용 구간 [{USEFUL_LO}, {USEFUL_HI}] 안에 든 비율   {inside:.1%}")
    print("    lasso 0.167  scad6 0.286  scad3.7 0.426  mcp4 0.5")
    print("    mcp2 1.0     mcp0.5 4.0    mcp0.25 8.0")

    if inside < 0.5:
        ok = False
        want = float(scale) * med / 1.0
        print()
        print("  질량이 유용 구간 밖이다.")
        print(f"    d0 는 MONOTONE_HINGE_SCALE 에 반비례한다.")
        print(f"    중앙 {med:.2f} -> 1.0 로 옮기려면 "
              f"MONOTONE_HINGE_SCALE = {want:.1f}")
        print(f"    selector_v11_canonical.py 에서 그 값만 고치고 재실행")

    print()
    print("  전체 " + ("통과.  학습 시작해도 된다" if ok else "실패.  학습 걸지 마라"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
