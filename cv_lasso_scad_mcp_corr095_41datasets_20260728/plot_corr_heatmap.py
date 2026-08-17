#!/usr/bin/env python3
"""corr_heatmap_extract.py 가 만든 npz 를 히트맵 격자로 그린다.

두 가지 모드가 있다.

    상세 (패널이 소스당 12개 이하)
        패널마다 이름, (n, p), q99 를 적는다.  소수를 자세히 볼 때.

    개괄 (그보다 많을 때 자동 전환, --compact 로 강제)
        라벨을 지우고 격자만 채운다.  508개처럼 전수를 볼 때
        "구조가 있는 게 몇 개나 되는가" 를 한눈에 보는 용도다.
        패널을 q99 오름차순으로 정렬하므로, 왼쪽 위에서 오른쪽 아래로
        갈수록 상관이 강해진다.  real 과 나란히 놓으면 두 분포가
        어디서 겹치고 어디서 갈라지는지 위치로 읽힌다.

모든 패널이 같은 열 개수와 같은 색 눈금(0~1)을 쓰므로 패널끼리 직접 비교된다.

    python3 plot_corr_heatmap.py
    python3 plot_corr_heatmap.py --in corr_heatmap_all.npz --out corr_heatmap_all \
        --max-cols 26 --panel-size 0.42
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec


def layout_rows(count: int, max_cols: int) -> tuple[int, int]:
    """패널 count 개를 담을 (행, 열).  열은 max_cols 를 넘지 않는다."""
    cols = min(max_cols, count) if count else 1
    rows = math.ceil(count / cols) if cols else 1
    return rows, cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="corr_heatmap_data.npz")
    ap.add_argument("--out", default="corr_heatmap")
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--triangle", action="store_true",
                    help="하삼각만 그린다")
    ap.add_argument("--cmap", default="Blues")
    ap.add_argument("--panel-size", type=float, default=None,
                    help="패널 한 변의 인치.  생략하면 개수에 맞춰 정한다.")
    ap.add_argument("--max-cols", type=int, default=None,
                    help="한 줄에 놓을 패널 수 상한")
    ap.add_argument("--compact", action="store_true",
                    help="패널 라벨을 지운다 (많을 때 자동)")
    ap.add_argument("--sort", choices=["q99", "none"], default=None,
                    help="개괄 모드 기본은 q99 오름차순")
    args = ap.parse_args()

    z = np.load(args.inp, allow_pickle=False)
    meta = json.loads(str(z["meta"]))

    sources: list[str] = []
    for m in meta:
        if m["source"] not in sources:
            sources.append(m["source"])
    by_source = {s: [m for m in meta if m["source"] == s] for s in sources}
    biggest = max(len(v) for v in by_source.values())

    compact = args.compact or biggest > 12
    sort_mode = args.sort or ("q99" if compact else "none")
    if sort_mode == "q99":
        for s in sources:
            by_source[s].sort(key=lambda m: m["q99"])

    max_cols = args.max_cols or (min(26, biggest) if compact else biggest)
    ps = args.panel_size or (0.44 if compact else 2.05)

    # 소스마다 필요한 행 수를 구해 세로로 쌓는다.
    plan = {s: layout_rows(len(by_source[s]), max_cols) for s in sources}
    total_rows = sum(r for r, _ in plan.values())

    label_h = 0.0 if compact else 0.62      # 패널 제목/축라벨이 먹는 세로
    gap_h = 0.42                            # 소스 사이 여백
    header_in = 1.5
    body_h = total_rows * (ps + label_h) + gap_h * (len(sources) - 1)
    fig_h = header_in + body_h + 0.35
    fig_w = max_cols * ps + 1.7

    fig = plt.figure(figsize=(fig_w, fig_h))
    top_frac = 1.0 - header_in / fig_h
    bottom_frac = 0.35 / fig_h

    outer = gridspec.GridSpec(
        len(sources), 2, figure=fig,
        height_ratios=[plan[s][0] * (ps + label_h) for s in sources],
        width_ratios=[1, 0.055 * (26 / max_cols)],
        left=.045, right=.955, top=top_frac, bottom=bottom_frac,
        hspace=gap_h / (body_h / len(sources)) if len(sources) > 1 else .2,
        wspace=.02)

    im = None
    for si, s in enumerate(sources):
        rows, cols = plan[s]
        inner = gridspec.GridSpecFromSubplotSpec(
            rows, cols, subplot_spec=outer[si, 0],
            wspace=.10 if compact else .16,
            hspace=(.10 if compact else .34) + (0 if compact else .25))
        items = by_source[s]
        for i, m in enumerate(items):
            ax = fig.add_subplot(inner[i // cols, i % cols])
            C = np.array(z[f"{m['source']}|{m['label']}"], dtype=np.float32)
            if args.triangle:
                C = np.where(np.triu(np.ones_like(C), 1) > 0, np.nan, C)
            im = ax.imshow(C, cmap=args.cmap, vmin=0.0, vmax=1.0,
                           interpolation="nearest", rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(.35 if compact else .7)
                sp.set_color("#777777" if compact else "#555555")

            if not compact:
                label = m["label"]
                if len(label) > 22:
                    label = label[:20] + ".."
                ax.set_title(f"{label}\nn={m['n']}  p={m['p']}",
                             fontsize=8.2, pad=3.5, linespacing=1.25)
                ax.set_xlabel(f"q99 {m['q99']:.3f}", fontsize=8.4, labelpad=2.5)

            if i == 0:
                q = np.array([x["q99"] for x in items])
                extra = (f"   n={len(items)}   q99 {q.min():.3f}–{q.max():.3f}"
                         if compact else "")
                ax.set_ylabel(f"{s}{extra}", fontweight="bold",
                              fontsize=10.5 if compact else 11,
                              labelpad=7, ha="left", y=1.0,
                              rotation=0 if compact else 90,
                              va="bottom" if compact else "center")

        # 남는 칸 비우기
        for i in range(len(items), rows * cols):
            fig.add_subplot(inner[i // cols, i % cols]).axis("off")

    cax = fig.add_subplot(outer[:, 1])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("|corr|", fontsize=10)
    cb.outline.set_linewidth(.7)

    shown = meta[0]["shown"]
    fig.suptitle(
        "Design geometry after the 0.95 filter:  real benchmarks vs the V15 generator",
        fontsize=13.5, fontweight="bold", x=.045, ha="left",
        y=1.0 - 0.30 / fig_h)
    note = (f"The same {shown} columns and the same colour scale in every "
            f"panel.  ")
    note += ("Panels are sorted by q99, the 99th percentile of the "
             "off-diagonal |corr| within the panel,\nso correlation increases "
             "left to right and top to bottom within each row block."
             if sort_mode == "q99" else
             "q99 is the 99th percentile of the off-diagonal |corr| "
             "within the panel.")
    fig.text(.045, 1.0 - 0.58 / fig_h, note,
             fontsize=9, color="#333333", ha="left", va="top")

    out = args.out + ".png"
    fig.savefig(out, dpi=args.dpi)
    print(f"  저장: {out}   ({len(meta)}개 패널, "
          f"{fig_w:.1f}x{fig_h:.1f} in, {'개괄' if compact else '상세'} 모드)")

    print()
    print(f"{'source':8s}{'n':>6}{'q99 min':>10}{'q99 med':>10}"
          f"{'q99 max':>10}{'med |corr|':>12}")
    print("-" * 56)
    for s in sources:
        q = np.array([m["q99"] for m in by_source[s]])
        md = np.array([m["median"] for m in by_source[s]])
        print(f"{s:8s}{q.size:6d}{q.min():10.3f}{np.median(q):10.3f}"
              f"{q.max():10.3f}{np.median(md):12.3f}")


if __name__ == "__main__":
    main()
