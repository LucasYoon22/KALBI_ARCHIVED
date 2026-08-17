#!/usr/bin/env python3
"""히트맵용 |corr| 부분행렬을 뽑아서 작은 npz 하나로 저장한다.

원격에는 matplotlib 이 없으므로 그리기는 분리한다.  여기서는 계산만 하고,
K x K 행렬 몇 개만 담긴 수 MB 짜리 파일을 만들어 내려받게 한다.
(K=220 이면 패널당 48,400 개 float32 = 190 KB.)

열 선택
    real 과 synthetic 을 같은 잣대로 보려면 열을 고르는 규칙이 같아야 한다.
    기본은 참조 그림과 같이 "앞에서 K 개" 다.  생성기가 블록 구조를 열
    순서에 심어두면 그게 보인다.  --column-mode random 은 무작위 K 개로,
    순서에 의존하지 않는 전형적인 모습을 본다.

표준화
    corr_box_plot.py 와 동일하게 열별 표준화 후 Z^T Z / n 을 쓴다.
    real 은 infer_kalbi_13.preprocess 가 낸 Ztr (train 기준 표준화 +
    0.95 그리디 필터) 을, synthetic 은 npz 의 X 를 그대로 쓴다.

    python3 corr_heatmap_extract.py \
        --real real_regression_data \
        --synth V15=/workspace/rolling_v15_8gpu/data/tabicl_v2 \
        --k 220 --per-source 6 --out corr_heatmap_data.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def abs_corr_block(X: np.ndarray, k: int, mode: str,
                   rng: np.random.Generator) -> np.ndarray | None:
    """열 k 개를 골라 |corr| (k x k) 를 낸다."""
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    if p < 2 or n < 3:
        return None

    if mode == "random" and p > k:
        idx = np.sort(rng.choice(p, size=k, replace=False))
    else:
        idx = np.arange(min(k, p))
    X = X[:, idx]

    mu = X.mean(0)
    sd = X.std(0)
    keep = np.isfinite(sd) & (sd > 1e-12)
    if keep.sum() < 2:
        return None
    Z = (X[:, keep] - mu[keep]) / sd[keep]
    C = np.clip((Z.T @ Z) / Z.shape[0], -1.0, 1.0)
    return np.abs(C).astype(np.float32)


def load_design(z) -> np.ndarray | None:
    for key in ("X", "x", "design", "features"):
        if key in z:
            a = np.asarray(z[key])
            break
    else:
        return None
    while a.ndim > 2:
        a = a[0]
    return np.asarray(a, dtype=np.float64) if a.ndim == 2 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None)
    ap.add_argument("--synth", action="append", default=[],
                    help="NAME=경로")
    ap.add_argument("--pattern", default="*.npz")
    ap.add_argument("--k", type=int, default=220,
                    help="패널마다 보여줄 열 개수")
    ap.add_argument("--per-source", type=int, default=6,
                    help="소스마다 뽑을 패널 개수.  0 이면 전부.")
    ap.add_argument("--column-mode", choices=["first", "random"],
                    default="first")
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--out", default="corr_heatmap_data.npz")
    args = ap.parse_args()

    blocks: dict[str, np.ndarray] = {}
    meta: list[dict] = []

    # ---------------------------------------------------------------- real
    if args.real:
        print("real")
        try:
            from infer_kalbi_13 import load_split, preprocess
        except Exception as exc:                                   # noqa: BLE001
            print(f"  import 실패 ({type(exc).__name__}: {exc}) — 건너뛴다")
            load_split = None
        if load_split is not None:
            dirs = sorted(d for d in Path(args.real).iterdir()
                          if d.is_dir() and not d.name.startswith("."))
            # 앞에서 순서대로 per_source 개 (0 이면 전부).  결정적이라 재현된다.
            want = len(dirs) if args.per_source <= 0 else args.per_source
            picked = 0
            for i, d in enumerate(dirs):
                if picked >= want:
                    break
                rng = np.random.default_rng(args.seed + i)
                try:
                    Ztr, _, _, _ = preprocess(*load_split(d), cache_dir=d)
                    C = abs_corr_block(Ztr, args.k, args.column_mode, rng)
                except Exception as exc:                           # noqa: BLE001
                    print(f"  {d.name[:32]:32s} 실패 {type(exc).__name__}")
                    continue
                if C is None:
                    continue
                key = f"Real|{d.name}"
                blocks[key] = C
                off = C[np.triu_indices(C.shape[0], 1)]
                meta.append(dict(source="Real", label=d.name,
                                 n=int(Ztr.shape[0]), p=int(Ztr.shape[1]),
                                 shown=int(C.shape[0]),
                                 q99=float(np.percentile(off, 99)),
                                 median=float(np.median(off))))
                print(f"  {d.name[:32]:32s} n={Ztr.shape[0]:5d} "
                      f"p={Ztr.shape[1]:5d}  q99={meta[-1]['q99']:.3f}")
                picked += 1

    # ------------------------------------------------------------- synth
    for spec in args.synth:
        if "=" not in spec:
            print(f"  형식 오류: {spec}")
            continue
        name, path = spec.split("=", 1)
        print(name)
        files = sorted(Path(path).glob(args.pattern))
        if not files:
            print(f"  파일 없음: {path}/{args.pattern}")
            continue
        # per_source <= 0 이면 전부, 아니면 전체에서 고르게 퍼뜨려 뽑는다
        # (앞쪽만 보면 생성 순서에 편향된다).
        if args.per_source <= 0:
            order = list(range(len(files)))
        else:
            order = list(dict.fromkeys(
                np.linspace(0, len(files) - 1, args.per_source)
                .astype(int).tolist()))
        for j, fi in enumerate(order):
            f = files[fi]
            if args.per_source <= 0 and j % 50 == 0:
                print(f"  ... {j}/{len(order)}")
            rng = np.random.default_rng(args.seed + 1000 + j)
            try:
                with np.load(f, allow_pickle=False) as z:
                    X = load_design(z)
                if X is None:
                    continue
                C = abs_corr_block(X, args.k, args.column_mode, rng)
            except Exception as exc:                               # noqa: BLE001
                print(f"  {f.name} 실패 {type(exc).__name__}")
                continue
            if C is None:
                continue
            key = f"{name}|{f.stem}"
            blocks[key] = C
            off = C[np.triu_indices(C.shape[0], 1)]
            meta.append(dict(source=name, label=f.stem,
                             n=int(X.shape[0]), p=int(X.shape[1]),
                             shown=int(C.shape[0]),
                             q99=float(np.percentile(off, 99)),
                             median=float(np.median(off))))
            if args.per_source > 0:
                print(f"  {f.stem:32s} n={X.shape[0]:5d} p={X.shape[1]:5d}  "
                      f"q99={meta[-1]['q99']:.3f}")

    if not blocks:
        raise SystemExit("뽑은 게 없다.  경로를 확인해라.")

    np.savez_compressed(args.out, meta=json.dumps(meta, ensure_ascii=False),
                        **blocks)
    size = Path(args.out).stat().st_size / 1e6
    print(f"\n  저장: {args.out}  ({len(blocks)}개 패널, {size:.1f} MB)")


if __name__ == "__main__":
    main()
