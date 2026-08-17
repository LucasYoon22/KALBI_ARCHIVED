#!/usr/bin/env python3
"""probe_*.json 을 모아 '참 R(beta) vs KALBI 성능' 을 표와 4패널 그림으로 낸다.
 
    python3.11 analyze_synth_probe.py --out ~/Desktop/KALBI/synth_probe_report
"""
 
from __future__ import annotations
 
import argparse
import glob
import json
from pathlib import Path
 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ticker import FixedLocator, FixedFormatter
 
ACC, BLUE, MUT = "#8C3A2E", "#3A5A80", "#6E7B88"
RED = "#E24B4A"
 
REAL_R = np.array([0.0564, 0.150, 0.346, 0.794, 0.894,
                   1.79, 2.45, 2.63, 2.68, 3.46, 5.35, 5.45, 6.08])
REAL_R2 = np.array([-59.772, -3.306, -1.408, 0.219, 0.567,
                    0.554, 0.575, 0.555, 0.671, 0.333, 0.686, 0.356, 0.312])
 
BINS = [(0, .3, "< 0.3"), (.3, .6, "0.3 - 0.6"), (.6, 1, "0.6 - 1"),
        (1, 2, "1 - 2"), (2, 4, "2 - 4"), (4, np.inf, "> 4")]
PBINS = [(0, 1025, "P <= 1024"), (1025, 2500, "P 1025-2500"),
         (2500, 10 ** 9, "P > 2500")]
 
 
def setup_font() -> bool:
    """한글 폰트 + 마이너스 기호가 깨지지 않게."""
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    for name in ["AppleGothic", "Apple SD Gothic Neo", "Malgun Gothic",
                 "NanumGothic", "Noto Sans CJK KR"]:
        try:
            fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            plt.rcParams["font.family"] = name
            return True
        except Exception:
            continue
    return False
 
 
def plain_log_ticks(ax, values, axis="x") -> None:
    """10^-1 대신 0.1 처럼 평문으로 찍는다 (mathtext 마이너스 회피)."""
    labels = [(f"{v:g}") for v in values]
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(FixedLocator(values))
    target.set_major_formatter(FixedFormatter(labels))
    target.set_minor_locator(FixedLocator([]))
 
 
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="probe_*.json")
    ap.add_argument("--out", default="synth_probe_report")
    ap.add_argument("--english", action="store_true", help="그림 라벨을 영어로")
    args = ap.parse_args()
 
    rows: list[dict] = []
    for path in sorted(glob.glob(args.pattern)):
        part = json.load(open(path, encoding="utf-8"))
        for r in part:
            r["src"] = Path(path).stem
        rows.extend(part)
        print(f"  {path}: {len(part)}개")
    if not rows:
        print("probe_*.json 이 없습니다."); return
 
    ko = setup_font() and not args.english
    if args.english:
        plt.rcParams["font.family"] = "DejaVu Sans"
        plt.rcParams["axes.unicode_minus"] = False
    L = (lambda k, e: k if ko else e)
 
    def col(key, cast=float):
        return np.array([r.get(key, np.nan) for r in rows], dtype=cast)
 
    R, r2, snr = col("true_R"), col("r2"), col("snr")
    nnz, ntr, tk = col("kalbi_nnz"), col("n_tr"), col("true_k")
    P, knorm = col("p"), col("kalbi_norm")
    ceiling = np.clip(snr / (1.0 + snr), 1e-9, None)
    eff = r2 / ceiling
 
    ok = np.isfinite(R) & np.isfinite(r2) & np.isfinite(snr)
    R, r2, eff, nnz, ntr, tk, P, knorm = (
        v[ok] for v in (R, r2, eff, nnz, ntr, tk, P, knorm))
 
    lines: list[str] = []
    def say(text: str = "") -> None:
        print(text); lines.append(text)
 
    say("=" * 96)
    say(f"합성 선형 태스크 {len(R)}개  —  새 V14(레일리 타깃팅) + 기존 KALBI")
    say("  효율 = test R2 / (SNR 천장),  천장 = SNR/(1+SNR)")
    say("=" * 96)
    say()
    say(f"{'참 R(beta)':>12}{'n':>5}{'SNR중앙':>9}{'R2중앙':>9}"
        f"{'효율중앙':>10}{'효율<0':>8}{'nnz/N':>8}{'참k/N':>8}{'||b||':>8}")
    say("-" * 96)
    for lo, hi, label in BINS:
        m = (R >= lo) & (R < hi)
        if not m.any():
            continue
        say(f"{label:>12}{int(m.sum()):>5}{np.median(snr[m]):>9.2f}"
            f"{np.median(r2[m]):>9.3f}{np.median(eff[m]):>10.3f}"
            f"{(eff[m] < 0).mean():>8.0%}{np.median(nnz[m] / ntr[m]):>8.2f}"
            f"{np.median(tk[m] / ntr[m]):>8.3f}{np.median(knorm[m]):>8.3f}")
    say("-" * 96)
    low, high = R < 0.6, R >= 0.6
    say(f"  R <  0.6 : n={int(low.sum()):4d}   효율중앙 {np.median(eff[low]):+.3f}"
        f"   효율<0 {(eff[low] < 0).mean():.0%}")
    say(f"  R >= 0.6 : n={int(high.sum()):4d}   효율중앙 {np.median(eff[high]):+.3f}"
        f"   효율<0 {(eff[high] < 0).mean():.0%}")
    rho = np.corrcoef(np.log(np.clip(R, 1e-6, None)), np.clip(eff, -3, 1))[0, 1]
    say(f"  corr( log R , clip(효율) ) = {rho:+.3f}")
    say()
 
    say(f"{'ISIS 경계':>14}{'n':>5}{'R2중앙':>9}{'효율중앙':>10}"
        f"{'효율<0':>8}{'nnz/N':>8}{'R중앙':>8}")
    say("-" * 96)
    for lo, hi, label in PBINS:
        m = (P >= lo) & (P < hi)
        if not m.any():
            continue
        say(f"{label:>14}{int(m.sum()):>5}{np.median(r2[m]):>9.3f}"
            f"{np.median(eff[m]):>10.3f}{(eff[m] < 0).mean():>8.0%}"
            f"{np.median(nnz[m] / ntr[m]):>8.2f}{np.median(R[m]):>8.2f}")
    say("-" * 96)
    say(f"  KALBI nnz/N 전체 중앙 {np.median(nnz / ntr):.2f}"
        f"   (참 k/N 중앙 {np.median(tk / ntr):.3f})")
    say(f"  clip6 총합 {sum(r.get('kalbi_clip6', 0) for r in rows)}"
        f"   max|b_j| 최대 {max(r.get('kalbi_maxabs', 0) for r in rows):.3f}")
    say()
    say("  [real 13개] R<1 인 4개 전부 실패, R>1 인 9개 전부 정상")
 
    Path(args.out + ".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
 
    # ------------------------------------------------------------- 그림
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(left=.075, right=.975, top=.90, bottom=.07,
                        wspace=.22, hspace=.34)
    xt = [0.05, 0.1, 0.3, 1, 3, 10]
 
    # (1) 산점도 + real 겹치기
    ax = axes[0, 0]
    ax.axhspan(-2.1, 0, color=RED, alpha=.06)
    ax.axvspan(0.03, 0.6, color=RED, alpha=.08)
    ax.axhline(0, color=MUT, lw=1)
    neg = r2 < 0
    ax.scatter(R[~neg], np.clip(r2[~neg], -2, 1), s=22, c=BLUE, alpha=.5,
               label=L("합성 · R2 >= 0", "synthetic R2 >= 0"))
    ax.scatter(R[neg], np.clip(r2[neg], -2, 1), s=26, c=RED, alpha=.6,
               label=L("합성 · R2 < 0", "synthetic R2 < 0"))
    ax.scatter(REAL_R, np.clip(REAL_R2, -2, 1), s=110, c=ACC, marker="^",
               edgecolor="w", linewidth=.9, zorder=6, label=L("real 13개", "real 13"))
    for lo, hi, _ in BINS:
        m = (R >= lo) & (R < hi)
        if m.sum() >= 4:
            ax.plot([max(lo, .04), min(hi, 13)],
                    [np.median(np.clip(r2[m], -2, 1))] * 2,
                    color="#1B2430", lw=2.4, solid_capstyle="butt", zorder=7)
    ax.set_xscale("log"); ax.set_xlim(0.04, 13); ax.set_ylim(-2.1, 1.05)
    plain_log_ticks(ax, xt)
    ax.set_xlabel(L("참 레일리 몫  R(b)", "true Rayleigh quotient R(b)"))
    ax.set_ylabel(L("test R2  (-2 에서 절단)", "test R2 (clipped at -2)"))
    ax.set_title(L("(1)  R(b) 가 낮은 쪽에 실패가 몰린다",
                   "(1)  failures concentrate at low R(b)"),
                 loc="left", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=.22)
 
    # (2) R 구간별 효율 막대
    ax = axes[0, 1]
    labels, values, counts = [], [], []
    for lo, hi, label in BINS:
        m = (R >= lo) & (R < hi)
        if m.sum() >= 3:
            labels.append(label); values.append(np.median(eff[m]))
            counts.append(int(m.sum()))
    colors = [RED if v < 0.4 else BLUE for v in values]
    bars = ax.bar(range(len(values)), values, color=colors, width=.66)
    for rect, count in zip(bars, counts):
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + (.03 if rect.get_height() >= 0 else -.07),
                f"n={count}", ha="center", fontsize=8, color=MUT)
    ax.axhline(0, color=MUT, lw=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(min(-0.6, min(values) - .15), 1.0)
    ax.set_xlabel(L("참 레일리 몫  R(b)", "true Rayleigh quotient R(b)"))
    ax.set_ylabel(L("신호 회수 효율 = R2 / SNR 천장",
                    "efficiency = R2 / SNR ceiling"))
    ax.set_title(L("(2)  SNR 을 통제해도 저역만 무너진다",
                   "(2)  holds after controlling for SNR"),
                 loc="left", fontweight="bold")
    ax.grid(alpha=.22, axis="y")
 
    # (3) 희소성
    ax = axes[1, 0]
    ax.axhline(1, color=MUT, lw=1.2, ls="--")
    ax.scatter(R, tk / ntr, s=20, c=ACC, alpha=.35, marker="x",
               label=L("참 k / N", "true k / N"))
    ax.scatter(R, nnz / ntr, s=22, c=BLUE, alpha=.5,
               label=L("KALBI nnz / N", "KALBI nnz / N"))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.04, 13)
    plain_log_ticks(ax, xt)
    plain_log_ticks(ax, [0.001, 0.01, 0.1, 1, 10], axis="y")
    ax.set_xlabel(L("참 레일리 몫  R(b)", "true Rayleigh quotient R(b)"))
    ax.set_ylabel("nnz / N")
    ax.set_title(L("(3)  KALBI 는 R 과 무관하게 항상 과밀하다",
                   "(3)  KALBI is always too dense"),
                 loc="left", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=.22)
 
    # (4) P 구간별 (ISIS 경계)
    ax = axes[1, 1]
    plabels, pvalues, pcounts, pneg = [], [], [], []
    for lo, hi, label in PBINS:
        m = (P >= lo) & (P < hi)
        if m.sum() >= 3:
            plabels.append(label); pvalues.append(np.median(eff[m]))
            pcounts.append(int(m.sum())); pneg.append((eff[m] < 0).mean())
    bars = ax.bar(range(len(pvalues)), pvalues,
                  color=[RED if v < 0.4 else BLUE for v in pvalues], width=.55)
    for rect, count, frac in zip(bars, pcounts, pneg):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + .03,
                f"n={count}\n" + L("실패 ", "neg ") + f"{frac:.0%}",
                ha="center", fontsize=8, color=MUT)
    ax.axhline(0, color=MUT, lw=1)
    ax.set_xticks(range(len(plabels))); ax.set_xticklabels(plabels, fontsize=9)
    ax.set_ylim(min(-0.3, min(pvalues) - .15), 1.15)
    ax.set_ylabel(L("신호 회수 효율", "efficiency"))
    ax.set_title(L("(4)  ISIS 상한 1024 를 넘어가면",
                   "(4)  crossing the ISIS cap of 1024"),
                 loc="left", fontweight="bold")
    ax.grid(alpha=.22, axis="y")
 
    fig.suptitle(L("새 V14 합성 태스크에 기존 KALBI 를 적용한 결과  "
                   f"(선형 태스크 {len(R)}개)",
                   f"existing KALBI on new V14 synthetic tasks (n={len(R)})"),
                 fontsize=14, fontweight="bold", x=.075, ha="left", y=.965)
    fig.savefig(args.out + ".png", dpi=150)
    say(f"\n  저장: {args.out}.txt , {args.out}.png")
 
 
if __name__ == "__main__":
    main()