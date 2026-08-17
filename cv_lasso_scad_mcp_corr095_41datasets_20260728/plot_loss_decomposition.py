#!/usr/bin/env python3
"""R 구간별 test 손실 분해 + 증폭 법칙 검증.

분해 (합 = 1):
    1/(1+s)                      잡음 바닥      (누구도 못 줄임)
    s/(1+s) - R2_LASSO           추정 손실      (희소 추정량이면 다 겪음)
    R2_LASSO - R2_KALBI          KALBI 고유 손실
    R2_KALBI                     설명된 분산

증폭 법칙:
    beta 가 train Gram 의 거의-영공간에 놓이면 test 에서 Gram 오차가
    ||beta||^2 = 1/R 배로 증폭된다.  초과 손실이 ||beta||^2 에 비례하는지 본다.

사용법:
    cd ~/Desktop/KALBI/project
    python3.11 plot_loss_decomposition.py --out ~/Desktop/KALBI/loss_decomposition
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

BINS = [(0, .3, "< 0.3"), (.3, .6, "0.3-0.6"), (.6, 1, "0.6-1"),
        (1, 2, "1-2"), (2, 4, "2-4"), (4, np.inf, "> 4")]

NOISE, ESTIM, KALBI, EXPL = "#B4B2A9", "#85B7EB", "#E24B4A", "#1D9E75"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="probe_*.json")
    ap.add_argument("--out", default="loss_decomposition")
    ap.add_argument("--clip", type=float, default=-2.0, help="R2 하한 절단")
    args = ap.parse_args()

    rows: list[dict] = []
    for path in sorted(glob.glob(args.pattern)):
        rows.extend(json.load(open(path, encoding="utf-8")))
    rows = [r for r in rows if np.isfinite(r.get("cv_r2", np.nan))
            and np.isfinite(r.get("true_R", np.nan))
            and np.isfinite(r.get("snr", np.nan))]
    if not rows:
        print("cv_r2 가 있는 태스크가 없습니다. --baseline 으로 다시 돌리세요.")
        return

    g = lambda k: np.array([r[k] for r in rows], dtype=float)
    R, s = g("true_R"), g("snr")
    rk = np.clip(g("r2"), args.clip, 1.0)
    rl = np.clip(g("cv_r2"), args.clip, 1.0)
    nb, N, P = g("kalbi_norm"), g("n_tr"), g("p")

    ceiling = s / (1.0 + s)
    noise = 1.0 - ceiling
    estim = ceiling - rl
    extra = rl - rk
    expl = rk

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))
    fig.subplots_adjust(left=.07, right=.98, top=.86, bottom=.15, wspace=.24)

    # ---------------------------------------------------------- (1) 분해
    labels, stacks, counts = [], [], []
    for lo, hi, label in BINS:
        m = (R >= lo) & (R < hi)
        if m.sum() < 3:
            continue
        labels.append(label); counts.append(int(m.sum()))
        stacks.append([np.median(noise[m]), np.median(estim[m]),
                       np.median(extra[m]), np.median(expl[m])])
    stacks = np.asarray(stacks)

    ax = axes[0]
    bottom_pos = np.zeros(len(labels))
    for col, color, name in zip(
            range(4), (NOISE, ESTIM, KALBI, EXPL),
            ("irreducible noise", "estimation loss (LASSO)",
             "KALBI-specific loss", "explained by KALBI")):
        v = stacks[:, col]
        ax.bar(range(len(labels)), v, bottom=np.where(v >= 0, bottom_pos, 0),
               color=color, width=.68, label=name)
        bottom_pos = bottom_pos + np.clip(v, 0, None)
    ax.axhline(0, color="#5F5E5A", lw=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{l}\nn={c}" for l, c in zip(labels, counts)],
                       fontsize=9)
    ax.set_xlabel("true Rayleigh quotient R(b)")
    ax.set_ylabel("share of test variance")
    ax.set_title("(1)  where the variance goes", loc="left", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", framealpha=.95)
    ax.grid(alpha=.22, axis="y")

    # ---------------------------------------------------------- (2) 증폭 법칙
    ax = axes[1]
    amp = nb ** 2 * np.sqrt(P / np.maximum(N, 1))
    excess = np.clip(ceiling - rk, 1e-4, None)
    ok = np.isfinite(amp) & (amp > 0)
    ax.scatter(amp[ok], excess[ok], s=16, c="#3A5A80", alpha=.35)

    lg = np.log(amp[ok]); le = np.log(excess[ok])
    slope, intercept = np.polyfit(lg, le, 1)
    xs = np.logspace(np.log10(amp[ok].min()), np.log10(amp[ok].max()), 50)
    ax.plot(xs, np.exp(intercept) * xs ** slope, color="#E24B4A", lw=2,
            label=f"fit: slope {slope:+.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"amplifier   $\|\hat\beta\|^2 \sqrt{P/N}$")
    ax.set_ylabel("excess loss   (SNR ceiling - test $R^2$)")
    ax.set_title("(2)  excess loss scales with $\\|\\hat\\beta\\|^2 = 1/R$",
                 loc="left", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=.22)

    fig.suptitle(f"test-loss decomposition by Rayleigh quotient  (n={len(rows)})",
                 fontsize=13, fontweight="bold", x=.07, ha="left", y=.965)
    fig.savefig(args.out + ".png", dpi=150)

    # ---------------------------------------------------------- 표
    out = []
    def say(t=""):
        print(t); out.append(t)

    say("=" * 86)
    say(f"R 구간별 test 분산 분해  (n={len(rows)}, 각 성분 중앙값)")
    say("=" * 86)
    say(f"{'참 R':>10}{'n':>5}{'잡음':>9}{'추정손실':>10}"
        f"{'KALBI손실':>11}{'설명됨':>9}{'KALBI/추정':>11}")
    say("-" * 86)
    for label, count, row in zip(labels, counts, stacks):
        ratio = row[2] / row[1] if abs(row[1]) > 1e-9 else np.nan
        say(f"{label:>10}{count:>5}{row[0]:>9.3f}{row[1]:>10.3f}"
            f"{row[2]:>11.3f}{row[3]:>9.3f}{ratio:>11.2f}")
    say("-" * 86)
    say()
    say(f"  증폭 법칙  log(초과손실) ~ {slope:+.3f} * log(||b||^2 sqrt(P/N))")
    say("     기울기 1 에 가까우면 '큰 계수가 Gram 오차를 증폭' 가설과 일치")
    say(f"     corr = {np.corrcoef(lg, le)[0, 1]:+.3f}")
    say()
    lo_m, hi_m = R < 0.6, R >= 0.6
    for m, name in ((lo_m, "R <  0.6"), (hi_m, "R >= 0.6")):
        if m.sum():
            say(f"  {name} : n={int(m.sum()):4d}   "
                f"KALBI 고유손실 중앙 {np.median(extra[m]):+.3f}   "
                f"추정손실 중앙 {np.median(estim[m]):+.3f}   "
                f"||b|| 중앙 {np.median(nb[m]):.2f}")
    say()
    say(f"  R2 는 {args.clip} 에서 절단 (극단 이상치가 중앙값을 왜곡하지 않게)")
    say(f"  저장: {args.out}.png , {args.out}.txt")
    Path(args.out + ".txt").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
