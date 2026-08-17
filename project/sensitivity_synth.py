#!/usr/bin/env python3
"""합성 태스크에서 참 R(beta) 별로 KALBI 해의 민감도를 잰다.

real 13개에서 확인한 것:
    입력을 1e-5 흔들면 해가 1e5 배로 움직인다.
    corr( log lambda_min(G_S), log 증폭 ) = -0.740

여기서 물을 것:
    R(beta) < 0.6 에서 효율이 0 인 게 '성능이 나쁘다' 인가,
    아니면 '해가 애초에 유일하지 않다' 인가?

    저역에서 증폭이 1e5 이면 후자다.  그러면 "KALBI 가 못 푼다" 가 아니라
    "그 영역에서는 KALBI 의 출력이 정의되지 않는다" 가 맞는 서술이 된다.

    cd ~/Desktop/KALBI/project
    python3.11 sensitivity_synth.py --data-dir out_v14_probe --limit 40
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from infer_kalbi_13 import load_kalbi, kalbi_beta
from test_kalbi_on_synth import read_task, split_standardize, load_true_rayleigh

BINS = [(0, .3, "< 0.3"), (.3, .6, "0.3-0.6"), (.6, 1, "0.6-1"),
        (1, 2, "1-2"), (2, 4, "2-4"), (4, np.inf, "> 4")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="sensitivity_synth.json")
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260809)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_num_threads(4)
    model, selector = load_kalbi(device)
    print()

    root = Path(args.data_dir)
    truth = load_true_rayleigh(root)
    ordered = truth.get("__ordered__") if isinstance(truth, dict) else None

    rows, seen = [], 0
    for path in sorted(root.glob("batch_*.npz")):
        if len(rows) >= args.limit:
            break
        task = read_task(path)
        if task is None:
            continue
        entry = (ordered[seen] if ordered and seen < len(ordered)
                 else truth.get(path.name))
        seen += 1
        x, y, _ = task

        t0 = time.perf_counter()
        try:
            Ztr, ytr, Zte, yte = split_standardize(x, y, args.seed)
            base, lam, _, _ = kalbi_beta(model, selector, Ztr, ytr, device)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  [건너뜀] {path.name}: {type(exc).__name__}")
            continue

        ss = max(float(((yte - yte.mean()) ** 2).sum()), 1e-12)
        r2 = lambda b: 1.0 - float(((yte - Zte @ b) ** 2).sum()) / ss
        nb0 = max(float(np.linalg.norm(base)), 1e-12)

        r2s, nnzs, amps = [r2(base)], [int((np.abs(base) > 1e-8).sum())], []
        rng = np.random.default_rng(0)
        for _ in range(args.repeats):
            noise = rng.standard_normal(Ztr.shape).astype(np.float32)
            noise /= max(float(np.linalg.norm(noise)), 1e-12)
            Zp = (Ztr + args.eps * float(np.linalg.norm(Ztr)) * noise
                  ).astype(np.float32)
            try:
                bp, _, _, _ = kalbi_beta(model, selector, Zp, ytr, device)
            except Exception:                                      # noqa: BLE001
                continue
            r2s.append(r2(bp))
            nnzs.append(int((np.abs(bp) > 1e-8).sum()))
            amps.append(float(np.linalg.norm(bp - base)) / nb0 / args.eps)

        S = np.flatnonzero(np.abs(base) > 1e-8)
        lam_min = np.nan
        if 1 <= S.size <= 1200:
            G = Ztr[:, S].T @ Ztr[:, S] / Ztr.shape[0]
            lam_min = float(np.linalg.eigvalsh(G.astype(np.float64))[0])

        rows.append(dict(
            batch=path.name, n_tr=int(Ztr.shape[0]), p=int(Ztr.shape[1]),
            true_R=float(entry["linear_rayleigh_quotient"]) if entry else np.nan,
            true_k=float(entry["linear_support_size"]) if entry else np.nan,
            snr=float(entry.get("linear_realized_snr", np.nan)) if entry else np.nan,
            nnz0=nnzs[0], r2_0=r2s[0],
            nnz_min=int(min(nnzs)), nnz_max=int(max(nnzs)),
            r2_min=float(min(r2s)), r2_max=float(max(r2s)),
            amplification=float(np.median(amps)) if amps else np.nan,
            subgram_lambda_min=lam_min, secs=time.perf_counter() - t0))
        if len(rows) % 10 == 0:
            print(f"  {len(rows)}개 완료 ...", flush=True)

    if not rows:
        print("결과 없음"); return
    Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                              encoding="utf-8")

    g = lambda k: np.array([r[k] for r in rows], dtype=float)
    R, amp, lm = g("true_R"), g("amplification"), g("subgram_lambda_min")
    spread = g("r2_max") - g("r2_min")
    nnz, N = g("nnz0"), g("n_tr")

    print()
    print("=" * 92)
    print(f"합성 태스크 {len(rows)}개   eps={args.eps:.0e}, 반복 {args.repeats}회")
    print("=" * 92)
    print(f"{'참 R':>10}{'n':>5}{'증폭 중앙':>12}{'증폭 최대':>12}"
          f"{'R2 흔들림':>12}{'nnz/N':>8}{'lam_min<=0':>12}")
    print("-" * 92)
    for lo, hi, label in BINS:
        m = (R >= lo) & (R < hi)
        if m.sum() < 2:
            continue
        neg = int((lm[m] <= 0).sum())
        print(f"{label:>10}{int(m.sum()):>5}{np.nanmedian(amp[m]):>12.0f}"
              f"{np.nanmax(amp[m]):>12.0f}{np.nanmedian(spread[m]):>12.3f}"
              f"{np.nanmedian(nnz[m] / N[m]):>8.2f}{neg:>7}/{int(m.sum()):<4}")
    print("-" * 92)
    lo_m, hi_m = R < 0.6, R >= 0.6
    for m, name in ((lo_m, "R <  0.6"), (hi_m, "R >= 0.6")):
        if m.sum():
            print(f"  {name} : n={int(m.sum()):3d}  증폭 중앙 {np.nanmedian(amp[m]):9.0f}"
                  f"   R2 흔들림 중앙 {np.nanmedian(spread[m]):.3f}")
    ok = np.isfinite(amp) & np.isfinite(R) & (R > 0)
    if ok.sum() > 3:
        rho = np.corrcoef(np.log(R[ok]), np.log(np.clip(amp[ok], 1e-9, None)))[0, 1]
        print(f"  corr( log 참R , log 증폭 ) = {rho:+.3f}")
        print("     음수면 'R 낮을수록 해가 불안정' — real 13개와 같은 패턴")
    print()
    print(f"  [real 13개] 증폭 중앙 27889,  corr(log lam_min, log 증폭) = -0.740")
    print(f"\n  저장: {args.out}")


if __name__ == "__main__":
    main()
