#!/usr/bin/env python3
"""corr_box_plot.py 와 생성기 감사값이 낸 JSON 을 박스플롯으로 그린다.

두 입력을 쓴다.

    corr_v15.json        corr_box_plot.py 의 출력.
                         real 13 과 V15 각각에 대해 태스크별
                         max / p95 / median |corr| (열 300개 부분표본).

    v15_corr_audit.json  V15 npz 에 생성기가 직접 저장해 둔 값 508개.
                         exact   = core_exact_max_abs_corr  (정확한 max)
                         sampled = sampled_corr_audit_max   (감사 256열의 max)

왜 나눠 그리는가

    max 는 부분표본에 편향된다.  열을 300개로 줄이면 쌍은 제곱으로 줄어든다.
    real 은 p 중앙 988 이라 쌍의 9.2% 를 보지만, V15 는 p 가 4000 대라
    0.5% 만 본다.  그래서 왼쪽 패널의 V15 max 는 진짜 max 가 아니다.
    분위수(median, p95)는 균등 부분표본에서 불편추정이므로 그대로 비교해도 된다.

    오른쪽 패널은 생성기가 전수로 잰 exact max 를 따로 놓아,
    "V15 가 필터 문턱까지 실제로 도달하는가" 를 편향 없이 보여준다.

    python3 plot_corr_box.py
    python3 plot_corr_box.py --out corr_box --dpi 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#1F4E79"      # real
ORANGE = "#C55A11"    # V15
MUT = "#6E7B88"
THRESHOLD = 0.95


def box_panel(ax, data, labels, colors, title, tag, show_threshold=False,
              seed=0):
    """박스 + 개별 점.  n 이 작을 때 분포를 숨기지 않으려고 점을 같이 찍는다."""
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=.5,
                    showfliers=False,
                    medianprops=dict(color="black", lw=1.6),
                    whiskerprops=dict(color=MUT),
                    capprops=dict(color=MUT))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(.30)
        patch.set_edgecolor(col)
        patch.set_linewidth(1.3)

    rng = np.random.default_rng(seed)
    for i, (col, d) in enumerate(zip(colors, data)):
        d = np.asarray(d)
        # 점이 많으면 투명도를 낮춰 뭉침을 읽을 수 있게 한다.
        alpha = .55 if d.size <= 80 else .18
        size = 15 if d.size <= 80 else 6
        jitter = rng.normal(0, .055, size=d.size)
        ax.scatter(np.full(d.size, i + 1) + jitter, d, s=size, color=col,
                   alpha=alpha, edgecolor="none", zorder=3)

    if show_threshold:
        ax.axhline(THRESHOLD, color=MUT, ls="--", lw=1.2, zorder=1)
        ax.text(.02, THRESHOLD + .012, "0.95 filter", color=MUT, fontsize=8,
                transform=ax.get_yaxis_transform(), va="bottom")

    ax.set_title(f"{tag}.  {title}", loc="left", fontweight="bold", fontsize=10.5)
    ax.set_ylabel("|corr|")
    ax.grid(alpha=.18, axis="y")
    ax.set_ylim(-.03, 1.03)
    ax.set_axisbelow(True)


def annotate_medians(ax, data, fmt="{:.3f}"):
    for i, d in enumerate(data):
        m = float(np.median(np.asarray(d)))
        ax.text(i + 1, 1.005, fmt.format(m), ha="center", va="bottom",
                fontsize=8.5, color="black", fontweight="bold")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--box-json", default="corr_v15.json")
    ap.add_argument("--audit-json", default="v15_corr_audit.json")
    ap.add_argument("--out", default="corr_box")
    ap.add_argument("--dpi", type=int, default=170)
    args = ap.parse_args()

    box = json.loads(Path(args.box_json).read_text(encoding="utf-8"))
    audit = json.loads(Path(args.audit_json).read_text(encoding="utf-8"))

    rows = {r["source"]: r for r in box["rows"]}
    if "Real" not in rows or "V15" not in rows:
        raise SystemExit(f"Real/V15 둘 다 필요하다.  있는 것: {list(rows)}")

    real, v15 = rows["Real"], rows["V15"]
    n_real, n_v15 = real["n"], v15["n"]
    n_audit = len(audit["exact"])

    fig = plt.figure(figsize=(13.5, 5.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.05],
                          left=.055, right=.985, top=.80, bottom=.13,
                          wspace=.34)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

    labels = [f"Real\nn={n_real}", f"V15\nn={n_v15}"]
    colors = [BLUE, ORANGE]

    # --- A, B: 분위수.  부분표본에 불편이므로 직접 비교해도 된다.
    for ax, key, title, tag in (
            (axes[0], "median_all", "median |corr|  (typical pair)", "A"),
            (axes[1], "p95_all", "95th pct |corr|  (upper tail)", "B")):
        data = [np.array(real[key]), np.array(v15[key])]
        box_panel(ax, data, labels, colors, title, tag)
        annotate_medians(ax, data)
        ax.set_ylim(-.02, .62)

    # --- C: max, 같은 부분표본 절차.  V15 쪽이 편향돼 있다는 걸 표시한다.
    data_c = [np.array(real["max_all"]), np.array(v15["max_all"])]
    box_panel(axes[2], data_c, labels, colors,
              "max |corr|  (subsampled)", "C",
              show_threshold=True)
    annotate_medians(axes[2], data_c)
    axes[2].text(.5, .045,
                 "V15 biased low:\nsamples 0.5% of pairs\nvs 9.2% for Real",
                 transform=axes[2].transAxes, ha="center", va="bottom",
                 fontsize=8, color=ORANGE, style="italic",
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                           edgecolor=ORANGE, alpha=.85, linewidth=.8))

    # --- D: 생성기 전수 감사값.  편향 없는 max 비교.
    data_d = [np.array(real["max_all"]), np.array(audit["exact"])]
    box_panel(axes[3], data_d,
              [f"Real\nn={n_real}", f"V15 exact\nn={n_audit}"], colors,
              "max |corr|  (exact audit)", "D",
              show_threshold=True)
    annotate_medians(axes[3], data_d)
    over = int((np.array(audit["exact"]) > THRESHOLD).sum())
    axes[3].text(.5, .045, f"{over} / {n_audit} exceed 0.95",
                 transform=axes[3].transAxes, ha="center", va="bottom",
                 fontsize=8.5, color=ORANGE, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                           edgecolor=ORANGE, alpha=.85, linewidth=.8))

    fig.suptitle(
        "Pairwise |corr| after the 0.95 filter:  real benchmarks vs the V15 generator",
        fontsize=13, fontweight="bold", x=.055, ha="left", y=.955)
    fig.text(.055, .885,
             "A and B are quantiles, unbiased under column subsampling, so the two "
             "columns are directly comparable.  C is not: the max of a subsample "
             "understates the true max,\nand V15 subsamples far more aggressively.  "
             "D replaces it with the exact per-shard maximum the generator records, "
             "which is the fair comparison.",
             fontsize=8.6, color="#333333", ha="left", va="top")

    out_png = args.out + ".png"
    fig.savefig(out_png, dpi=args.dpi)
    print(f"  저장: {out_png}")

    # ------------------------------------------------------------ 숫자 요약
    ex = np.array(audit["exact"])
    sa = np.array(audit["sampled"])
    print()
    print(f"{'':28s}{'Real':>12}{'V15':>12}")
    print("-" * 52)
    for key, name in (("median_all", "median |corr|"),
                      ("p95_all", "p95 |corr|"),
                      ("max_all", "max (subsampled)")):
        r = np.median(real[key]); v = np.median(v15[key])
        print(f"{name:28s}{r:12.4f}{v:12.4f}")
    print(f"{'max (exact audit)':28s}{np.median(real['max_all']):12.4f}"
          f"{np.median(ex):12.4f}")
    print("-" * 52)
    print(f"  V15 exact max   중앙 {np.median(ex):.4f}  "
          f"p95 {np.percentile(ex, 95):.4f}  최대 {ex.max():.4f}")
    print(f"  V15 sampled max 중앙 {np.median(sa):.4f}  "
          f"p95 {np.percentile(sa, 95):.4f}  최대 {sa.max():.4f}")
    print(f"  0.95 초과: exact {int((ex > .95).sum())}/{ex.size}, "
          f"sampled {int((sa > .95).sum())}/{sa.size}")


if __name__ == "__main__":
    main()
