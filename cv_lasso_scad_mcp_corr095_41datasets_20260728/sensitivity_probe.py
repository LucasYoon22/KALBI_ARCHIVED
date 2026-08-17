#!/usr/bin/env python3
"""KALBI 해가 입력의 미세 섭동에 얼마나 민감한지 측정한다.

동기:  같은 코드·같은 체크포인트인데 CPU 와 CUDA 가
       nnz 380 대 665,  R2 +0.67 대 -2.08 을 냈다.
       lambda 는 1% 밖에 다르지 않았다.

방법:  입력을 상대 eps 만큼 흔들고 해가 얼마나 움직이는지 본다.

           증폭계수 = (||d beta|| / ||beta||) / eps

       조건이 좋으면 O(1),  병적이면 1/eps 규모까지 커진다.

    python3.11 sensitivity_probe.py --data-dir real_regression_data
    python3.11 sensitivity_probe.py --data-dir real_regression_data --eps 1e-4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from infer_kalbi_13 import load_kalbi, load_split, preprocess, kalbi_beta


def summarise(beta, Ztr, Zte, yte, ss):
    return dict(nnz=int((np.abs(beta) > 1e-8).sum()),
                norm=float(np.linalg.norm(beta)),
                r2=1.0 - float(((yte - Zte @ beta) ** 2).sum()) / ss)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="sensitivity_probe.json")
    ap.add_argument("--eps", type=float, default=1e-5, help="상대 섭동 크기")
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--max-cells", type=int, default=3_000_000)
    ap.add_argument("--only", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_num_threads(4)
    model, selector = load_kalbi(device)
    print()

    root = Path(args.data_dir)
    dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if args.only:
        dirs = [d for d in dirs if args.only in d.name]

    rows = []
    print(f"{'dataset':30s}{'N':>6}{'P':>6}{'nnz0':>7}{'R2_0':>8}"
          f"{'nnz 범위':>12}{'R2 범위':>16}{'증폭':>10}{'lam_min':>10}")
    print("-" * 106)

    for d in dirs:
        try:
            Ztr, ytr, Zte, yte = preprocess(*load_split(d), cache_dir=d)
        except Exception as exc:                                   # noqa: BLE001
            print(f"{d.name[:30]:30s} 로드 실패 {type(exc).__name__}")
            continue
        if Ztr.shape[0] * Ztr.shape[1] > args.max_cells:
            print(f"{d.name[:30]:30s} 건너뜀 (셀 수 초과)")
            continue

        t0 = time.perf_counter()
        ss = max(float(((yte - yte.mean()) ** 2).sum()), 1e-12)
        base, lam0, _, _ = kalbi_beta(model, selector, Ztr, ytr, device)
        b0 = summarise(base, Ztr, Zte, yte, ss)
        nb0 = max(b0["norm"], 1e-12)

        nnzs, r2s, amps = [b0["nnz"]], [b0["r2"]], []
        rng = np.random.default_rng(0)
        for rep in range(args.repeats):
            noise = rng.standard_normal(Ztr.shape).astype(np.float32)
            noise /= max(float(np.linalg.norm(noise)), 1e-12)
            Zp = (Ztr + args.eps * float(np.linalg.norm(Ztr)) * noise
                  ).astype(np.float32)
            try:
                bp, _, _, _ = kalbi_beta(model, selector, Zp, ytr, device)
            except Exception:                                      # noqa: BLE001
                continue
            st = summarise(bp, Ztr, Zte, yte, ss)
            nnzs.append(st["nnz"]); r2s.append(st["r2"])
            amps.append(float(np.linalg.norm(bp - base)) / nb0 / args.eps)

        # 부분 Gram 최소 고유값 (선택된 좌표 기준)
        S = np.flatnonzero(np.abs(base) > 1e-8)
        lam_min = np.nan
        if 1 <= S.size <= 1500:
            G = Ztr[:, S].T @ Ztr[:, S] / Ztr.shape[0]
            lam_min = float(np.linalg.eigvalsh(G.astype(np.float64))[0])

        row = dict(label=d.name, n_tr=int(Ztr.shape[0]), p=int(Ztr.shape[1]),
                   nnz0=b0["nnz"], r2_0=b0["r2"], norm0=b0["norm"],
                   nnz_min=int(min(nnzs)), nnz_max=int(max(nnzs)),
                   r2_min=float(min(r2s)), r2_max=float(max(r2s)),
                   amplification=float(np.median(amps)) if amps else np.nan,
                   subgram_lambda_min=lam_min, secs=time.perf_counter() - t0)
        rows.append(row)
        print(f"{d.name[:30]:30s}{row['n_tr']:6d}{row['p']:6d}"
              f"{row['nnz0']:7d}{row['r2_0']:8.3f}"
              f"{row['nnz_min']:6d}-{row['nnz_max']:<5d}"
              f"{row['r2_min']:8.2f}~{row['r2_max']:<7.2f}"
              f"{row['amplification']:10.1f}{lam_min:10.2e}", flush=True)

    if not rows:
        return
    Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                              encoding="utf-8")

    amp = np.array([r["amplification"] for r in rows], dtype=float)
    lm = np.array([r["subgram_lambda_min"] for r in rows], dtype=float)
    spread = np.array([r["r2_max"] - r["r2_min"] for r in rows], dtype=float)
    print("-" * 106)
    print(f"  섭동 크기 eps = {args.eps:.0e},  반복 {args.repeats}회")
    print(f"  증폭계수    중앙 {np.nanmedian(amp):.1f}   최대 {np.nanmax(amp):.1f}")
    print(f"     1 근처면 안정,  1/eps = {1/args.eps:.0e} 에 가까우면 완전 병적")
    print(f"  R2 흔들림   중앙 {np.nanmedian(spread):.3f}   최대 {np.nanmax(spread):.3f}")
    ok = np.isfinite(amp) & np.isfinite(lm) & (lm > 0)
    if ok.sum() > 3:
        rho = np.corrcoef(np.log(lm[ok]), np.log(np.clip(amp[ok], 1e-9, None)))[0, 1]
        print(f"  corr( log lambda_min(G_S) , log 증폭 ) = {rho:+.3f}")
        print("     음수면 '부분 Gram 이 특이할수록 불안정' 확인")
    print(f"\n  저장: {args.out}")


if __name__ == "__main__":
    main()
