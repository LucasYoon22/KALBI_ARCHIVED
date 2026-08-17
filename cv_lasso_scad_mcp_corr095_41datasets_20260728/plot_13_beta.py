#!/usr/bin/env python3
"""measure_13_beta.py 결과를 6패널 그림으로 그린다.

사용법:
    python3 plot_13_beta.py --json measure_13_beta.json --data-dir ./real_regression_data

패널 1~4 는 JSON 만으로 그리고, 패널 5~6 (계수 stem) 은 대표 두 데이터셋만
다시 계산한다.  --no-stem 을 주면 재계산 없이 4패널만 그린다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

warnings.filterwarnings("ignore")

ACC, BLUE, MUT = "#8C3A2E", "#3A5A80", "#6E7B88"
NRM = "||beta||"


def setup_font():
    """한글 폰트를 찾아 설정. 없으면 영문 라벨로 폴백."""
    for name in ["AppleGothic", "Apple SD Gothic Neo", "Malgun Gothic",
                 "NanumGothic", "Noto Sans CJK KR"]:
        try:
            fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return True
        except Exception:
            continue
    print("  [경고] 한글 폰트를 못 찾아 영문 라벨로 그립니다.")
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="measure_13_beta.json")
    ap.add_argument("--data-dir", default="./real_regression_data")
    ap.add_argument("--out", default="beta_real13.png")
    ap.add_argument("--no-stem", action="store_true", help="계수 stem 패널 생략")
    ap.add_argument("--grid", type=int, default=100)
    args = ap.parse_args()

    if not os.path.exists(args.json):
        print(f"{args.json} 없음. measure_13_beta.py 를 먼저 실행하십시오."); sys.exit(1)
    rows = json.load(open(args.json, encoding="utf-8"))
    ko = setup_font()
    L = (lambda k, e: k if ko else e)

    nb = np.array([r["norm"] for r in rows])
    mx = np.array([r["maxabs"] for r in rows])
    R  = np.array([r["R"] for r in rows], dtype=float)
    K  = np.array([r["nnz"] for r in rows], dtype=float)

    nrow = 2 if args.no_stem else 3
    fig = plt.figure(figsize=(13, 3.3 * nrow + 0.9))
    gs = fig.add_gridspec(nrow, 2, hspace=.52, wspace=.24,
                          left=.075, right=.975, top=.90, bottom=.08)
    plt.rcParams.update({"axes.grid": True, "grid.alpha": .25,
                         "axes.spines.top": False, "axes.spines.right": False})

    def lospace(v, lo=None, hi=None, n=24):
        v = v[v > 0]
        return np.logspace(np.log10(lo or v.min() * .7), np.log10(hi or v.max() * 1.4), n)

    # (1) ||beta||
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(nb, bins=lospace(nb, 0.01, 300), color=BLUE, alpha=.85, edgecolor="w")
    ax.axvline(1, color=ACC, ls="--", lw=1.6)
    ax.text(1.4, ax.get_ylim()[1] * .8, "= 1\nV14", color=ACC, fontsize=8, va="center")
    ax.axvline(10, color=MUT, ls=":", lw=1.3)
    ax.set_xscale("log"); ax.set_xlabel(NRM); ax.set_ylabel(L("데이터셋 수", "count"))
    ax.set_title(f"(1)  {NRM} " + L("분포", "distribution") + f"  ·  {len(rows)}" + L("개", ""),
                 loc="left", fontweight="bold")

    # (2) max|beta_j|
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(mx, bins=lospace(mx, 0.01, 200), color=BLUE, alpha=.85, edgecolor="w")
    ax.axvline(6, color=ACC, ls="--", lw=1.6)
    ax.text(7.5, ax.get_ylim()[1] * .8, "Direct96\n" + L("상한 6", "cap 6"),
            color=ACC, fontsize=8, va="center")
    ax.axvline(1, color=MUT, ls=":", lw=1.3)
    ax.set_xscale("log"); ax.set_xlabel("max |beta_j|"); ax.set_ylabel(L("데이터셋 수", "count"))
    ax.set_title("(2)  max |beta_j| " + L("분포", "distribution"), loc="left", fontweight="bold")

    # (3) R vs ||beta||
    ax = fig.add_subplot(gs[1, 0])
    rr = np.logspace(np.log10(max(np.nanmin(R) * .3, 1e-7)), np.log10(np.nanmax(R) * 3), 100)
    for c2, st, lb in [(1.0, "-", "Var(Xb)=1"), (0.5, "--", "0.5"), (0.25, ":", "0.25")]:
        ax.plot(rr, np.sqrt(c2 / rr), color=MUT, ls=st, lw=1, zorder=1, label=lb)
    ax.scatter(R, nb, s=42, c=[ACC if v > 10 else BLUE for v in nb], alpha=.9, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(L("레일리 몫  R(beta)", "Rayleigh quotient  R(beta)")); ax.set_ylabel(NRM)
    ax.legend(fontsize=7, title=L("이론선", "theory"), title_fontsize=7, loc="upper right")
    ax.set_title(f"(3)  {NRM} = sqrt( Var(Xb) / R(b) )", loc="left", fontweight="bold")

    # (4) nnz vs max
    ax = fig.add_subplot(gs[1, 1])
    kk = np.logspace(0, np.log10(max(K.max(), 10)) + .3, 50)
    ax.plot(kk, 1 / np.sqrt(kk), color=MUT, ls="--", lw=1, label="1/sqrt(k)")
    ax.scatter(K, mx, s=42, c=[ACC if v > 6 else BLUE for v in mx], alpha=.9, zorder=3)
    ax.axhline(6, color=ACC, ls="--", lw=1.4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("nnz"); ax.set_ylabel("max |beta_j|")
    ax.legend(fontsize=7)
    ax.set_title("(4)  nnz " + L("대 최대 계수", "vs max coefficient"), loc="left", fontweight="bold")

    # (5)(6) 계수 stem — 대표 두 개만 재계산
    if not args.no_stem:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from measure_13_beta import load_split, fit_preprocessing, corr_filter, cv_lasso
        srt = sorted(rows, key=lambda z: -z["norm"])
        picks = [srt[0], srt[len(srt) // 2]]
        for i, r in enumerate(picks):
            d = os.path.join(args.data_dir, r["label"])
            print(f"  stem 패널 계산 중: {r['label']} ...", flush=True)
            Xtr, ytr, _, _ = load_split(d)
            act, m, s = fit_preprocessing(Xtr)
            Z = ((Xtr - m) / s)[:, act]
            Z = Z[:, corr_filter(Z)]
            ys = ytr.std() or 1.0
            beta, *_ = cv_lasso(Z, (ytr - ytr.mean()) / ys, args.grid)
            v = beta[np.abs(beta) > 1e-8]
            ax = fig.add_subplot(gs[2, i])
            c = ACC if r["norm"] > 10 else BLUE
            ml, sl, _ = ax.stem(np.arange(len(v)), v, basefmt="#AAAAAA")
            plt.setp(sl, color=c, lw=1.2); plt.setp(ml, color=c, ms=3.5)
            ax.axhline(6, color=ACC, ls=":", lw=1); ax.axhline(-6, color=ACC, ls=":", lw=1)
            ax.set_xlabel(L("계수 인덱스 (0 이 아닌 것만)", "nonzero coefficient index"))
            ax.set_ylabel("beta_j")
            ax.set_title("(%d)  %s\n%s=%.2f   max=%.2f   R=%.2e   nnz=%d" %
                         (5 + i, r["label"][:38], NRM, r["norm"], r["maxabs"], r["R"], r["nnz"]),
                         loc="left", fontweight="bold", fontsize=8.5)

    fig.suptitle(L("새 13개 벤치마크의 CV LASSO 계수  (표준화 공간, corr 0.95 필터 후)",
                   "CV LASSO coefficients, 13 benchmark datasets (standardized, after corr-0.95 filter)"),
                 fontsize=13, fontweight="bold", x=.075, ha="left", y=.96)
    fig.savefig(args.out, dpi=155)
    print(f"\n  저장: {args.out}")

    print()
    print(f"  {NRM}     최소 {nb.min():.3f}  중앙 {np.median(nb):.3f}  최대 {nb.max():.3f}")
    print(f"  max|b_j|      최소 {mx.min():.3f}  중앙 {np.median(mx):.3f}  최대 {mx.max():.3f}")
    print(f"  R(beta)       최소 {np.nanmin(R):.2e}  중앙 {np.nanmedian(R):.2e}")
    print()
    print(f"  ||beta|| > 10 :  {int((nb > 10).sum())} / {len(rows)}   <-- 폭발 케이스")
    print(f"  max|b_j| > 6  :  {int((mx > 6).sum())} / {len(rows)}   <-- Direct96 정의역 밖")
    print()
    print("  [비교] V14 32 태스크   : ||beta|| 0.854~1.144,  max|b_j| 최대 1.0000")
    print("  [비교] 이전 41개       : ||beta|| 중앙 0.716,  12%가 >10,  최대 174.59")


if __name__ == "__main__":
    main()
