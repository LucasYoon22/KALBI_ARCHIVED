#!/usr/bin/env python3
"""새 V14 생성기가 실제로 무엇을 뽑았는지 프로파일링한다.

manifest.json 만 읽으므로 npz 를 지운 런도 포함된다.

핵심 질문 네 가지
  1. R 분포가 real 13개를 덮는가
  2. 의도한 목표 R* 가 얼마나 클리핑되어 살아남았는가
  3. ||beta|| = 1/sqrt(R) 항등식이 실제로 성립하는가
  4. support 선택 모드가 R 에 어떤 기여를 하는가

사용법:
    cd ~/Desktop/KALBI/project
    python3.11 profile_generator.py --out ~/Desktop/KALBI/generator_profile
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

REAL_R = np.array([0.0564, 0.150, 0.346, 0.794, 0.894,
                   1.79, 2.45, 2.63, 2.68, 3.46, 5.35, 5.45, 6.08])

BLUE, ACC, MUT, GREEN = "#3A5A80", "#8C3A2E", "#6E7B88", "#1D9E75"


def collect(patterns: list[str]) -> list[dict]:
    seen, rows = set(), []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            try:
                payload = json.loads(Path(path).read_text(encoding="utf-8"))
            except Exception:                                      # noqa: BLE001
                continue
            found: list[dict] = []

            def walk(node):
                if isinstance(node, dict):
                    if "linear_rayleigh_quotient" in node:
                        found.append(node)
                    for v in node.values():
                        walk(v)
                elif isinstance(node, list):
                    for v in node:
                        walk(v)

            walk(payload)
            for entry in found:
                entry["_run"] = Path(path).parent.name
            rows.extend(found)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", nargs="*", default=[
        "out_v14_*/manifest.json", "big_*/manifest.json", "on_*/manifest.json"])
    ap.add_argument("--out", default="generator_profile")
    args = ap.parse_args()

    rows = collect(args.patterns)
    if not rows:
        print("manifest 를 찾지 못했습니다."); return

    g = lambda k: np.array([r.get(k, np.nan) for r in rows], dtype=float)
    s = lambda k, d="?": np.array([r.get(k, d) for r in rows], dtype=object)

    R = g("linear_rayleigh_quotient")
    tgt = g("linear_rayleigh_target")
    nb, mx = g("linear_beta_l2_norm"), g("linear_beta_max_abs")
    k = g("linear_support_size")
    kn, kp = g("linear_support_fraction_of_n"), g("linear_support_fraction")
    emin, emax = g("linear_subgram_eig_min"), g("linear_subgram_eig_max")
    snr = g("linear_realized_snr")
    mode = s("linear_support_selection", "random")
    on = np.array([bool(r.get("linear_rayleigh_targeted")) for r in rows])
    run = s("_run")
    ok = np.isfinite(R) & (R > 0)

    out: list[str] = []
    def say(t: str = "") -> None:
        print(t); out.append(t)

    q = lambda v, *p: "  ".join(
        f"{int(x * 100):>3}% {np.nanquantile(v, x):8.3f}" for x in p)

    say("=" * 96)
    say(f"새 V14 생성기 프로파일  —  선형 태스크 {int(ok.sum())}개"
        f"  ({len(set(run))}개 런)")
    say("=" * 96)
    say()
    say("  R(beta)   " + q(R[ok], 0, .05, .25, .5, .75, .95, 1))
    say("  ||beta||  " + q(nb[ok], 0, .05, .25, .5, .75, .95, 1))
    say("  max|b_j|  " + q(mx[ok], 0, .05, .25, .5, .75, .95, 1))
    say("  k         " + q(k[ok], 0, .25, .5, .75, .95, 1))
    say("  k / N     " + q(kn[ok], 0, .25, .5, .75, .95, 1))
    say("  k / P     " + q(kp[ok], 0, .25, .5, .75, .95, 1))
    say("  SNR       " + q(snr[ok], 0, .25, .5, .75, 1))
    say()

    say("  === real 13개 커버리지 ===")
    lo, hi = REAL_R.min(), REAL_R.max()
    inside = ((R[ok] >= lo) & (R[ok] <= hi)).sum()
    say(f"  real 범위 [{lo:.3f}, {hi:.2f}] 안 : {inside} / {int(ok.sum())}"
        f"  ({inside / ok.sum():.0%})")
    for lo2, hi2, label in ((0, .3, "R < 0.3"), (.3, .6, "0.3-0.6"),
                            (.6, 1, "0.6-1"), (1, 3, "1-3"),
                            (3, np.inf, "R > 3")):
        m = ok & (R >= lo2) & (R < hi2)
        rm = ((REAL_R >= lo2) & (REAL_R < hi2)).sum()
        say(f"    {label:>9} : 생성 {m.sum():5d} ({m.mean():5.1%})"
            f"   real {rm}/13 ({rm / 13:5.1%})")
    say()

    say("  === 타깃팅 적용 여부 ===")
    for flag, label in ((on & ok, "적용"), (~on & ok, "미적용")):
        if flag.sum():
            say(f"    {label:>6} n={int(flag.sum()):5d}  "
                f"R 중앙 {np.nanmedian(R[flag]):6.3f}  "
                f"5% {np.nanquantile(R[flag], .05):6.3f}  "
                f"95% {np.nanquantile(R[flag], .95):6.3f}  "
                f"k 중앙 {np.nanmedian(k[flag]):5.0f}")
    say()

    say("  === support 선택 모드별 ===")
    say(f"    {'모드':>18}{'n':>7}{'R 중앙':>9}{'R 5%':>8}{'R 95%':>8}"
        f"{'eig_max 중앙':>13}{'eig_min 중앙':>13}")
    for name in sorted(set(mode[ok])):
        m = ok & (mode == name)
        if m.sum() < 3:
            continue
        say(f"    {str(name):>18}{int(m.sum()):>7}{np.nanmedian(R[m]):>9.3f}"
            f"{np.nanquantile(R[m], .05):>8.3f}{np.nanquantile(R[m], .95):>8.3f}"
            f"{np.nanmedian(emax[m]):>13.2f}{np.nanmedian(emin[m]):>13.4f}")
    say()

    say("  === 목표 R* 가 얼마나 살아남았나 ===")
    have = ok & np.isfinite(tgt) & (tgt > 0)
    if have.sum():
        ratio = R[have] / tgt[have]
        say(f"    타깃 기록된 태스크 {int(have.sum())}개")
        say(f"    실현/목표 비율   중앙 {np.nanmedian(ratio):.3f}"
            f"   5% {np.nanquantile(ratio, .05):.3f}"
            f"   95% {np.nanquantile(ratio, .95):.3f}")
        clipped = np.abs(np.log(ratio)) > 0.1
        say(f"    10% 이상 어긋난 태스크 : {int(clipped.sum())}"
            f" ({clipped.mean():.0%})   <- 부분 Gram 범위로 클리핑된 몫")
        say(f"    목표 R*    " + q(tgt[have], 0, .25, .5, .75, 1))
    say()

    say("  === ||beta|| = 1/sqrt(R) 항등식 검증 ===")
    pred = 1.0 / np.sqrt(R[ok])
    rel = np.abs(nb[ok] - pred) / np.maximum(pred, 1e-12)
    say(f"    상대오차   중앙 {np.nanmedian(rel):.2e}   최대 {np.nanmax(rel):.2e}")
    say("    (생성기가 Var(X beta)=1 로 맞추므로 정확히 성립해야 함)")
    say()
    say(f"    max|b_j| 최대 {np.nanmax(mx[ok]):.3f}   "
        f"Direct96 상한 6 초과 : {int((mx[ok] > 6).sum())}개")

    Path(args.out + ".txt").write_text("\n".join(out) + "\n", encoding="utf-8")

    # ------------------------------------------------------------- 그림
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6))
    fig.subplots_adjust(left=.07, right=.98, top=.90, bottom=.08,
                        wspace=.22, hspace=.33)
    edges = np.logspace(np.log10(max(np.nanmin(R[ok]), 1e-3)),
                        np.log10(np.nanmax(R[ok])), 42)

    ax = axes[0, 0]
    ax.hist([R[ok & on], R[ok & ~on]], bins=edges, stacked=True,
            color=[BLUE, MUT], edgecolor="white", linewidth=.3,
            label=["eigen-targeted", "original sampler"])
    for v in REAL_R:
        ax.axvline(v, color=ACC, lw=1.4, alpha=.85,
                   ymax=.09, solid_capstyle="butt")
    ax.set_xscale("log")
    ax.set_xlabel("R(beta)"); ax.set_ylabel("tasks")
    ax.set_title("(1)  generated R, with the 13 real values as ticks",
                 loc="left", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=.2)

    ax = axes[0, 1]
    if have.sum():
        ax.scatter(tgt[have], R[have], s=12, c=BLUE, alpha=.3)
        line = np.logspace(np.log10(np.nanmin(tgt[have])),
                           np.log10(np.nanmax(tgt[have])), 40)
        ax.plot(line, line, color=MUT, ls="--", lw=1.3, label="no clipping")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.legend(fontsize=8)
    ax.set_xlabel("target R*"); ax.set_ylabel("realised R(beta)")
    ax.set_title("(2)  how much of the target survives clipping",
                 loc="left", fontweight="bold")
    ax.grid(alpha=.2)

    ax = axes[1, 0]
    ax.scatter(R[ok], nb[ok], s=10, c=BLUE, alpha=.3)
    line = np.logspace(np.log10(np.nanmin(R[ok])), np.log10(np.nanmax(R[ok])), 60)
    ax.plot(line, 1 / np.sqrt(line), color=ACC, lw=1.6,
            label=r"$\|\beta\| = 1/\sqrt{R}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("R(beta)"); ax.set_ylabel(r"$\|\beta\|$")
    ax.set_title("(3)  identity check", loc="left", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=.2)

    ax = axes[1, 1]
    for name, color in zip(("random", "correlated_block", "mixed"),
                           (MUT, BLUE, GREEN)):
        m = ok & (mode == name)
        if m.sum() >= 5:
            ax.hist(R[m], bins=edges, histtype="step", lw=1.8,
                    color=color, label=f"{name} (n={int(m.sum())})")
    ax.set_xscale("log")
    ax.set_xlabel("R(beta)"); ax.set_ylabel("tasks")
    ax.set_title("(4)  contribution of each support mode",
                 loc="left", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=.2)

    fig.suptitle(f"new V14 generator output profile  (n={int(ok.sum())})",
                 fontsize=13, fontweight="bold", x=.07, ha="left", y=.965)
    fig.savefig(args.out + ".png", dpi=150)
    print(f"\n  저장: {args.out}.txt , {args.out}.png")


if __name__ == "__main__":
    main()
