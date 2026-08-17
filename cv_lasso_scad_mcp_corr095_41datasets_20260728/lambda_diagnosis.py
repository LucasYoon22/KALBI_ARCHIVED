#!/usr/bin/env python3
"""real 13개에서 'lambda 선택 문제'인지 '경로 자체 문제'인지 가른다.

셀렉터의 페널티 rho 는 고정한 채 lambda 격자를 훑고, 세 지점을 비교한다.

    selector   셀렉터가 실제로 고른 lambda
    oracle     경로에서 test R2 가 최대인 lambda   (상한, 실전 불가)
    nnz<=aN    nnz 가 a*N 이하가 되는 가장 작은 lambda  (실전 가능한 규칙)

판정
    oracle 이 크게 좋으면   -> lambda 선택 문제.  재학습 없이 개선 여지
    oracle 도 나쁘면       -> 경로 자체가 나쁨.  솔버/페널티를 고쳐야 함
    nnz<=aN 이 oracle 에 가까우면 -> 아주 단순한 규칙만으로 회복 가능

    python3.11 lambda_diagnosis.py --data-dir real_regression_data
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from infer_kalbi_13 import load_kalbi, load_split, preprocess
from solution_path import kalbi_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="lambda_diagnosis.json")
    ap.add_argument("--grid", type=int, default=96)
    ap.add_argument("--ratio-min", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="nnz <= alpha*N 규칙")
    ap.add_argument("--monotone", action="store_true",
                    help="Direct96 에 단조 비증가 제약을 걸고 경로를 뽑는다")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_num_threads(4)
    model, selector = load_kalbi(device)
    if args.monotone:
        import canonical_penalty_monotone as mono
        mono.patch()
    print()

    ratios = np.geomspace(1.0, args.ratio_min, args.grid)
    root = Path(args.data_dir)
    dirs = sorted(d for d in root.iterdir()
                  if d.is_dir() and not d.name.startswith("."))

    print(f"{'dataset':30s}{'N':>6}{'P':>6} | "
          f"{'selector':>22} | {'oracle':>22} | {'nnz<=aN':>22}")
    print(f"{'':30s}{'':6}{'':6} | {'r':>8}{'nnz':>7}{'R2':>7} | "
          f"{'r':>8}{'nnz':>7}{'R2':>7} | {'r':>8}{'nnz':>7}{'R2':>7}")
    print("-" * 118)

    rows = []
    for d in dirs:
        try:
            Ztr, ytr, Zte, yte = preprocess(*load_split(d), cache_dir=d)
        except Exception as exc:                                   # noqa: BLE001
            print(f"{d.name[:30]:30s} 로드 실패 {type(exc).__name__}")
            continue

        t0 = time.perf_counter()
        try:
            B, lam_max, lam_sel = kalbi_path(model, selector, Ztr, ytr,
                                             device, ratios)
        except Exception as exc:                                   # noqa: BLE001
            print(f"{d.name[:30]:30s} 경로 실패 {type(exc).__name__}: {exc}")
            continue

        ss = max(float(((yte - yte.mean()) ** 2).sum()), 1e-12)
        r2 = 1.0 - ((yte[None] - B @ Zte.T) ** 2).sum(1) / ss
        nnz = (np.abs(B) > 1e-8).sum(1)
        N = Ztr.shape[0]

        i_sel = int(np.argmin(np.abs(ratios - lam_sel / max(lam_max, 1e-12))))
        i_orc = int(np.argmax(r2))
        cap = np.flatnonzero(nnz <= args.alpha * N)
        i_cap = int(cap[-1]) if cap.size else 0        # 조건 만족하는 가장 작은 lambda

        row = dict(label=d.name, n_tr=int(N), p=int(Ztr.shape[1]),
                   lambda_max=lam_max, alpha=args.alpha,
                   sel_ratio=float(ratios[i_sel]), sel_nnz=int(nnz[i_sel]),
                   sel_r2=float(r2[i_sel]),
                   orc_ratio=float(ratios[i_orc]), orc_nnz=int(nnz[i_orc]),
                   orc_r2=float(r2[i_orc]),
                   cap_ratio=float(ratios[i_cap]), cap_nnz=int(nnz[i_cap]),
                   cap_r2=float(r2[i_cap]),
                   nnz_at_lambda_max=int(nnz[0]),
                   secs=time.perf_counter() - t0)
        rows.append(row)
        print(f"{d.name[:30]:30s}{N:6d}{Ztr.shape[1]:6d} | "
              f"{row['sel_ratio']:8.1e}{row['sel_nnz']:7d}{row['sel_r2']:7.3f} | "
              f"{row['orc_ratio']:8.1e}{row['orc_nnz']:7d}{row['orc_r2']:7.3f} | "
              f"{row['cap_ratio']:8.1e}{row['cap_nnz']:7d}{row['cap_r2']:7.3f}",
              flush=True)

    if not rows:
        return
    Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                              encoding="utf-8")

    g = lambda k: np.array([r[k] for r in rows], dtype=float)
    s, o, c = g("sel_r2"), g("orc_r2"), g("cap_r2")
    print("-" * 118)
    print(f"  test R2 중앙     selector {np.median(s):+.3f}   "
          f"oracle {np.median(o):+.3f}   nnz<=aN {np.median(c):+.3f}")
    print(f"  R2 < 0 개수      selector {int((s<0).sum())}/{len(s)}   "
          f"oracle {int((o<0).sum())}/{len(o)}   nnz<=aN {int((c<0).sum())}/{len(c)}")
    print(f"  놓친 폭 (oracle - selector)  중앙 {np.median(o-s):+.3f}   "
          f"최대 {np.max(o-s):+.3f}")
    print(f"  단순 규칙이 회수한 몫 (cap - selector) 중앙 {np.median(c-s):+.3f}")
    print()
    print(f"  lambda=lambda_max 에서의 nnz : "
          f"{[int(v) for v in g('nnz_at_lambda_max')]}")
    print("     정의상 0 이어야 한다.  0 이 아니면 rho'(0) 이 너무 작다는 뜻")
    print()
    gap = float(np.median(o - s))
    if gap > 0.2:
        print("  판정: lambda 선택 문제가 크다.  경로에 훨씬 나은 지점이 있다")
    elif gap > 0.05:
        print("  판정: lambda 선택이 일부 기여.  경로 문제도 함께 있다")
    else:
        print("  판정: 경로 자체가 나쁘다.  lambda 를 바꿔도 회복이 안 된다")
    print(f"\n  저장: {args.out}")


if __name__ == "__main__":
    main()
