#!/usr/bin/env python3
"""v14_gen.py 의 레일리 타깃팅 패치가 실제로 R 커버리지를 만드는지 검증한다.

TabICL 없이 돌아간다.  상관 구조가 있는 X 를 직접 만들고
generate_controlled_linear_response 를 그대로 호출해서
메타데이터의 linear_rayleigh_quotient 분포를 본다.

사용법:
    cd ~/Desktop/KALBI/cv_lasso_scad_mcp_corr095_41datasets_20260728
    python3 check_rayleigh_patch.py
    python3 check_rayleigh_patch.py --tasks 400 --n 400 --p 300
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# real 13개 벤치마크에서 측정된 KALBI 계수의 레일리 몫
REAL_R = np.array([
    0.0564, 0.150, 0.346, 0.794, 0.894,
    1.79, 2.45, 2.63, 2.68, 3.46, 5.35, 5.45, 6.08,
])


def _stub_missing_imports() -> None:
    """torch / tabicl 이 없어도 선형 경로만 쓰면 되므로 최소 스텁을 꽂는다."""
    import types

    for name in ("torch", "tabicl"):
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
        except ImportError:
            module = types.ModuleType(name)
            module.__getattr__ = lambda _attr: None                # type: ignore[attr-defined]
            sys.modules[name] = module

    import re

    source = Path(__file__).with_name("v14_gen.py").read_text(encoding="utf-8")
    for name in re.findall(r"^from ([A-Za-z_][\w]*) import", source, re.M):
        if name in sys.modules or name in ("__future__", "pathlib", "typing"):
            continue
        try:
            importlib.import_module(name)
        except ImportError:
            module = types.ModuleType(name)
            module.__getattr__ = lambda _attr: object              # type: ignore[attr-defined]
            sys.modules[name] = module


def load_generator(path: Path):
    _stub_missing_imports()
    spec = importlib.util.spec_from_file_location("v14_gen", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["v14_gen"] = module
    spec.loader.exec_module(module)                                # noqa: S102
    return module


def make_correlated_x(n: int, p: int, rng: np.random.Generator) -> np.ndarray:
    """블록 상관 구조를 가진 표준화 설계행렬. corr 0.95 필터는 통과하도록."""
    n_blocks = max(1, p // 12)
    factors = rng.standard_normal((n, n_blocks))
    assign = rng.integers(0, n_blocks, size=p)
    loading = rng.uniform(0.55, 0.92, size=p)
    x = (
        factors[:, assign] * loading
        + rng.standard_normal((n, p)) * np.sqrt(1.0 - loading ** 2)
    )
    x -= x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True)
    return x / np.maximum(sd, 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default=str(HERE / "v14_gen.py"))
    ap.add_argument("--tasks", type=int, default=300)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--p", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20260809)
    args = ap.parse_args()

    gen = load_generator(Path(args.gen))
    print(f"generator : {args.gen}")
    print(f"contract  : {gen.LINEAR_RESPONSE_CONTRACT}")
    print(f"max k     : {gen.MAX_SUPPORT_FRACTION_OF_N} * (N-1) "
          f"= {int(gen.MAX_SUPPORT_FRACTION_OF_N * (args.n - 1))}")
    print()

    rng = np.random.default_rng(args.seed)
    rows = []
    for index in range(args.tasks):
        x = make_correlated_x(args.n, args.p, rng)
        reference = (
            gen.SUPPORT_REFERENCE_N if index % 2 else gen.SUPPORT_REFERENCE_P
        )
        try:
            _, meta = gen.generate_controlled_linear_response(
                x,
                rng,
                support_reference_id=reference,
                clip_value=10.0,
                min_unique_count=1,
                min_unique_ratio=0.0,
                max_dominant_ratio=1.0,
                min_standardized_iqr=0.0,
                round_decimals=6,
            )
        except Exception as exc:                                   # noqa: BLE001
            print(f"  [실패] task {index}: {type(exc).__name__}: {exc}")
            continue
        rows.append(meta)

    if not rows:
        print("생성된 태스크가 없습니다."); return

    R = np.array([r["linear_rayleigh_quotient"] for r in rows])
    nb = np.array([r["linear_beta_l2_norm"] for r in rows])
    mx = np.array([r["linear_beta_max_abs"] for r in rows])
    k = np.array([r["linear_support_size"] for r in rows], dtype=float)
    kn = np.array([r["linear_beta_nnz_over_n"] for r in rows])
    on = np.array([bool(r.get("linear_rayleigh_targeted")) for r in rows])

    def q(v, *levels):
        return "  ".join(f"{p:>4.0%} {np.quantile(v, p):8.3f}" for p in levels)

    print(f"태스크 {len(rows)}개   (타깃팅 적용 {on.sum()} / {len(rows)})")
    print()
    print("  R(beta)   " + q(R, .00, .05, .25, .50, .75, .95, 1.0))
    print("  ||beta||  " + q(nb, .00, .05, .25, .50, .75, .95, 1.0))
    print("  max|b_j|  " + q(mx, .00, .05, .25, .50, .75, .95, 1.0))
    print("  k         " + q(k, .00, .25, .50, .75, 1.0))
    print("  k / N     " + q(kn, .00, .25, .50, .75, 1.0))
    print()

    print("  === 커버리지 판정 ===")
    lo, hi = REAL_R.min(), REAL_R.max()
    inside = int(((R >= lo) & (R <= hi)).sum())
    print(f"  real R 범위 [{lo:.3f}, {hi:.3f}] 안에 드는 태스크 : "
          f"{inside} / {len(rows)}  ({inside / len(rows):.1%})")
    print(f"  R < 1  인 태스크 : {int((R < 1).sum()):4d} / {len(rows)}"
          f"  ({(R < 1).mean():.1%})   <-- 실패 구간, 패치 전에는 거의 0")
    print(f"  R < 0.3 인 태스크 : {int((R < 0.3).sum()):4d} / {len(rows)}"
          f"  ({(R < 0.3).mean():.1%})   <-- 붕괴 구간")
    print(f"  R > 3  인 태스크 : {int((R > 3).sum()):4d} / {len(rows)}"
          f"  ({(R > 3).mean():.1%})")
    print(f"  ||beta|| > 2 : {int((nb > 2).sum())} / {len(rows)}"
          f"   최대 {nb.max():.2f}")
    print(f"  3k < N 위반  : {int((kn > 1/3).sum())} / {len(rows)}"
          f"   <-- 가드 완화가 먹혔는지")
    print()

    edges = np.array([0, .1, .3, 1, 3, 10, np.inf])
    counts = np.histogram(R, bins=edges)[0]
    labels = ["<0.1", "0.1-0.3", "0.3-1", "1-3", "3-10", ">10"]
    width = max(counts.max(), 1)
    print("  R(beta) 히스토그램")
    for label, count in zip(labels, counts):
        bar = "#" * int(40 * count / width)
        print(f"    {label:>8s} {count:5d}  {bar}")
    print()
    print("  [비교] real 13개 R : " +
          "  ".join(f"{v:.3f}" for v in np.sort(REAL_R)))
    print("  [패치 전] V14 R    : 전부 1 근처 (무작위 support + 등방 beta)")


if __name__ == "__main__":
    main()
