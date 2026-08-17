#!/usr/bin/env python3
"""probe_*.json 을 (1) shape/개수 (2) R 의존성 두 축으로 뜯어본다.

    python3.11 describe_probe.py
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np

BINS = [(0, .3, "< 0.3"), (.3, .6, "0.3 - 0.6"), (.6, 1, "0.6 - 1"),
        (1, 2, "1 - 2"), (2, 4, "2 - 4"), (4, np.inf, "> 4")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="probe_*.json")
    args = ap.parse_args()

    rows: list[dict] = []
    per_file: Counter = Counter()
    for path in sorted(glob.glob(args.pattern)):
        part = json.load(open(path, encoding="utf-8"))
        for r in part:
            r["src"] = Path(path).stem.replace("probe_", "")
        per_file[Path(path).stem.replace("probe_", "")] = len(part)
        rows.extend(part)
    if not rows:
        print("probe_*.json 없음"); return

    g = lambda k: np.array([r.get(k, np.nan) for r in rows], dtype=float)
    R, r2, snr = g("true_R"), g("r2"), g("snr")
    kR, knnz, knorm, kmax = g("kalbi_R"), g("kalbi_nnz"), g("kalbi_norm"), g("kalbi_maxabs")
    lam, tk, N, P = g("lam_ratio"), g("true_k"), g("n_tr"), g("p")
    eff = r2 / np.clip(snr / (1 + snr), 1e-9, None)
    ok = np.isfinite(R) & np.isfinite(r2)

    # ---------------------------------------------------------- (1) 규모
    print("=" * 94)
    print("1.  무엇을 몇 개 돌렸나")
    print("=" * 94)
    print(f"{'실행 (shape)':>26}{'N_train':>9}{'P':>7}{'선형 태스크':>12}")
    print("-" * 94)
    shapes: dict[str, list] = {}
    for r in rows:
        shapes.setdefault(r["src"], []).append(r)
    for name in sorted(shapes, key=lambda s: -len(shapes[s])):
        part = shapes[name]
        print(f"{name[:26]:>26}{int(np.median([x['n_tr'] for x in part])):>9}"
              f"{int(np.median([x['p'] for x in part])):>7}{len(part):>12}")
    print("-" * 94)
    print(f"{'합계':>26}{'':>9}{'':>7}{len(rows):>12}")
    print()
    print(f"  N_train 범위 : {int(np.nanmin(N))} ~ {int(np.nanmax(N))}"
          f"   (생성 N 의 절반, 50/50 행 분할)")
    print(f"  P 범위       : {int(np.nanmin(P))} ~ {int(np.nanmax(P))}")
    print(f"  KALBI 내부   : ISIS 스크리닝 후 exact proximal gradient 10 스텝 (고정)")
    print(f"  태스크당 추론: 1회 (셀렉터가 lambda/penalty 를 한 번 고름)")
    print()

    # ---------------------------------------------------------- (2) R 의존성
    print("=" * 94)
    print("2.  참 R(beta) 에 따라 KALBI 의 출력이 어떻게 변하나")
    print("=" * 94)
    print(f"{'참 R':>11}{'n':>5}{'참k/N':>8}{'참R중앙':>9}"
          f"{'KALBI R':>9}{'nnz/N':>8}{'||bhat||':>10}{'max|b|':>8}"
          f"{'lam비':>9}{'효율':>8}")
    print("-" * 94)
    for lo, hi, label in BINS:
        m = ok & (R >= lo) & (R < hi)
        if m.sum() < 2:
            continue
        med = lambda v: np.nanmedian(v[m])
        print(f"{label:>11}{int(m.sum()):>5}{med(tk / N):>8.3f}{med(R):>9.3f}"
              f"{med(kR):>9.3f}{med(knnz / N):>8.2f}{med(knorm):>10.3f}"
              f"{med(kmax):>8.3f}{med(lam):>9.1e}{med(eff):>8.3f}")
    print("-" * 94)
    print()

    # ---------------------------------------------------------- (3) 핵심 대조
    print("=" * 94)
    print("3.  KALBI 는 참 R 을 따라가는가")
    print("=" * 94)
    v = ok & np.isfinite(kR) & (kR > 0)
    lr, lk = np.log(np.clip(R[v], 1e-6, None)), np.log(kR[v])
    slope = np.polyfit(lr, lk, 1)[0]
    print(f"  참 R  범위 {R[v].min():.3f} ~ {R[v].max():.2f}"
          f"   중앙 {np.median(R[v]):.3f}   로그표준편차 {lr.std():.3f}")
    print(f"  KALBI R 범위 {kR[v].min():.3f} ~ {kR[v].max():.2f}"
          f"   중앙 {np.median(kR[v]):.3f}   로그표준편차 {lk.std():.3f}")
    print()
    print(f"  log-log 기울기  d log R_hat / d log R_true = {slope:+.3f}")
    print(f"  corr(log R_true, log R_hat)                = {np.corrcoef(lr, lk)[0, 1]:+.3f}")
    print("     기울기 1 이면 완전히 추종,  0 이면 참 R 을 전혀 반영 못 함")
    print()
    lo_m, hi_m = v & (R < 0.6), v & (R >= 0.6)
    print(f"  참 R < 0.6  (n={int(lo_m.sum())}) : "
          f"참 R 중앙 {np.median(R[lo_m]):.3f}  ->  KALBI R 중앙 {np.median(kR[lo_m]):.3f}")
    print(f"  참 R >= 0.6 (n={int(hi_m.sum())}) : "
          f"참 R 중앙 {np.median(R[hi_m]):.3f}  ->  KALBI R 중앙 {np.median(kR[hi_m]):.3f}")
    print()
    print("=" * 94)
    print("4.  희소성")
    print("=" * 94)
    print(f"  참 k/N        중앙 {np.nanmedian(tk / N):.4f}   "
          f"5% {np.nanquantile(tk / N, .05):.4f}   95% {np.nanquantile(tk / N, .95):.3f}")
    print(f"  KALBI nnz/N   중앙 {np.nanmedian(knnz / N):.2f}   "
          f"5% {np.nanquantile(knnz / N, .05):.2f}   95% {np.nanquantile(knnz / N, .95):.2f}")
    print(f"  nnz > N 인 태스크 : {int((knnz > N).sum())} / {len(rows)}"
          f"  ({(knnz > N).mean():.0%})")
    print(f"  과선택 배율 중앙  : {np.nanmedian(knnz / np.maximum(tk, 1)):.0f} 배")
    print(f"  clip6 총합        : {sum(r.get('kalbi_clip6', 0) for r in rows)}"
          f"   max|b_j| 최대 {np.nanmax(kmax):.3f}   (Direct96 상한 6)")

    # ---------------------------------------------------------- (5) CV LASSO
    cv = [r for r in rows if np.isfinite(r.get("cv_r2", np.nan))]
    if not cv:
        return
    print()
    print("=" * 94)
    print(f"5.  CV LASSO 대조 — 문제가 어려운 건가, KALBI 가 못 하는 건가  (n={len(cv)})")
    print("=" * 94)
    cR = np.array([r["true_R"] for r in cv])
    kk = np.array([r["r2"] for r in cv])
    cc = np.array([r["cv_r2"] for r in cv])
    kz = np.array([r["kalbi_nnz"] for r in cv], float)
    cz = np.array([r.get("cv_nnz", np.nan) for r in cv], float)
    nn = np.array([r["n_tr"] for r in cv], float)
    print(f"{'참 R':>11}{'n':>5}{'KALBI R2':>10}{'LASSO R2':>10}{'차이':>9}"
          f"{'KALBI<0':>9}{'LASSO<0':>9}{'K nnz/N':>9}{'L nnz/N':>9}")
    print("-" * 94)
    for lo, hi, label in BINS:
        m = (cR >= lo) & (cR < hi)
        if m.sum() < 2:
            continue
        print(f"{label:>11}{int(m.sum()):>5}{np.median(kk[m]):>10.3f}"
              f"{np.median(cc[m]):>10.3f}{np.median(cc[m]) - np.median(kk[m]):>+9.3f}"
              f"{(kk[m] < 0).mean():>9.0%}{(cc[m] < 0).mean():>9.0%}"
              f"{np.median(kz[m] / nn[m]):>9.2f}{np.median(cz[m] / nn[m]):>9.3f}")
    print("-" * 94)
    lo_m = cR < 0.6
    if lo_m.any():
        print(f"  참 R < 0.6 : KALBI R2 중앙 {np.median(kk[lo_m]):+.3f}   "
              f"LASSO R2 중앙 {np.median(cc[lo_m]):+.3f}")
        print(f"     LASSO 가 이긴 비율 {np.mean(cc[lo_m] > kk[lo_m]):.0%}")
    hi_m = cR >= 0.6
    if hi_m.any():
        print(f"  참 R >= 0.6: KALBI R2 중앙 {np.median(kk[hi_m]):+.3f}   "
              f"LASSO R2 중앙 {np.median(cc[hi_m]):+.3f}")
        print(f"     LASSO 가 이긴 비율 {np.mean(cc[hi_m] > kk[hi_m]):.0%}")
    print()
    print("  저역에서 LASSO 는 되는데 KALBI 만 안 되면 -> 문제 난이도가 아니라 모델 한계")


if __name__ == "__main__":
    main()
