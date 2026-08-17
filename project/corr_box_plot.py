#!/usr/bin/env python3
"""real 13 개 대 V14 대 V15 의 pairwise correlation 분포를 box plot 으로 비교한다.

lasso_path_gap.py 와 같은 진단 계열이다.  KALBI 를 전혀 쓰지 않고 X 만 본다.

전제: 이 프로젝트의 전체 파이프라인이 |corr| < 0.95 필터를 기준으로 한다
(폴더 이름 corr095_...).  real 은 infer_kalbi_13.preprocess() 가 train 기준
표준화 후 그리디 0.95 필터를 적용한 Ztr 을 쓴다.  V14/V15 npz 는 생성기의
SCM qfilter 가 이미 같은 0.95 문턱으로 골라낸 X 를 저장해 둔 것이므로
따로 필터링하지 않고 그대로 쓴다.  그래서 real 과 synthetic 모두
"0.95 필터를 통과한 뒤에 남아있는 상관"을 재는, 같은 잣대의 비교가 된다.

    max     필터 문턱에 가장 가까운 값.  세 소스 다 <0.95 이어야 정상이다.
    p95     상위 5% 쌍의 전형적인 강도.
    median  대다수 쌍의 전형적인 강도.  real 이 median 에서 높다면
            소수의 극단쌍만 필터링되고 나머지는 여전히 뭉쳐있다는 뜻이고,
            synthetic 이 median 에서 낮다면 필터를 통과하자마자 거의
            독립이 된다는 뜻이다.  이 차이가 V14 -> V15 로 좁혀졌는지가
            핵심 질문이다.

p 가 크면 상관행렬을 전부 못 만든다 (200k^2).  생성기 자체의 corr audit 과
같은 방식으로 열을 무작위 부분표본(기본 300개)해서 926*325 문 어림잡는다.

    python3 corr_box_plot.py \
        --real real_regression_data \
        --synth V14=out_v14_probe --synth V15=/workspace/v15_pool \
        --out corr_box
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, MUT = "#1F4E79", "#6E7B88"
CORR_THRESHOLD = 0.95


# ------------------------------------------------------------------ 상관 요약

def corr_summary(X: np.ndarray, rng: np.random.Generator,
                  max_features: int = 300) -> dict | None:
    """열을 (필요하면) 부분표본해서 |corr| 상삼각의 max/p95/median/mean 을 낸다."""
    X = np.asarray(X, dtype=np.float64)
    p = X.shape[1]
    if p < 2:
        return None
    if p > max_features:
        idx = rng.choice(p, size=max_features, replace=False)
        X = X[:, idx]
        p = max_features

    mu = X.mean(0)
    sd = X.std(0)
    keep = np.isfinite(sd) & (sd > 1e-12)
    if keep.sum() < 2:
        return None
    Z = (X[:, keep] - mu[keep]) / sd[keep]
    q = Z.shape[1]

    corr = np.clip((Z.T @ Z) / Z.shape[0], -1.0, 1.0)
    off = np.abs(corr[np.triu_indices(q, 1)])
    if off.size == 0:
        return None
    return dict(
        n_features=int(q), n_pairs=int(off.size),
        max=float(off.max()), p95=float(np.percentile(off, 95)),
        median=float(np.median(off)), mean=float(off.mean()),
    )


# ------------------------------------------------------------------ 로딩

def iter_real(root: Path, limit: int, max_features: int, seed: int):
    from infer_kalbi_13 import load_split, preprocess

    dirs = sorted(d for d in root.iterdir()
                  if d.is_dir() and not d.name.startswith("."))[:limit]
    for i, d in enumerate(dirs):
        rng = np.random.default_rng(seed + i)
        try:
            Ztr, _, _, _ = preprocess(*load_split(d), cache_dir=d)
            s = corr_summary(Ztr, rng, max_features)
        except Exception as exc:                                   # noqa: BLE001
            print(f"    {d.name[:34]:34s} 실패 {type(exc).__name__}")
            continue
        if s is not None:
            print(f"    {d.name[:34]:34s} OK  P={Ztr.shape[1]:5d}  "
                  f"max={s['max']:.3f}  median={s['median']:.3f}")
            yield d.name, s


def iter_npz(root: Path, limit: int, max_cells: int,
             max_features: int, seed: int):
    files = sorted(root.glob("batch_*.npz"))[:limit]
    for i, f in enumerate(files):
        rng = np.random.default_rng(seed + i)
        try:
            with np.load(f, allow_pickle=False) as z:
                X = np.asarray(z["X"][0], dtype=np.float64)
            if X.shape[0] * X.shape[1] > max_cells:
                continue
            s = corr_summary(X, rng, max_features)
        except Exception:                                          # noqa: BLE001
            continue
        if s is not None:
            yield f.stem, s


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--synth", action="append", default=[],
                    help="NAME=경로 형식, 여러 번 가능")
    ap.add_argument("--out", default="corr_box")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-cells", type=int, default=6_000_000)
    ap.add_argument("--max-features", type=int, default=300,
                    help="상관행렬을 만들 때 부분표본할 열 개수 상한")
    ap.add_argument("--seed", type=int, default=20260813)
    args = ap.parse_args()

    sources: dict[str, list] = {}
    labels: dict[str, list] = {}

    print("real 13")
    t0 = time.perf_counter()
    rl, rs = [], []
    for name, s in iter_real(Path(args.real), args.limit,
                             args.max_features, args.seed):
        rl.append(name); rs.append(s)
    sources["Real"] = rs
    labels["Real"] = rl
    print(f"  {len(rs)}개  {time.perf_counter()-t0:.0f}초\n")

    for spec in args.synth:
        if "=" not in spec:
            print(f"  형식 오류, NAME=경로 여야 함: {spec}")
            continue
        name, path = spec.split("=", 1)
        print(name)
        t0 = time.perf_counter()
        got = []
        gotl = []
        for stem, s in iter_npz(Path(path), args.limit, args.max_cells,
                                args.max_features, args.seed):
            got.append(s); gotl.append(stem)
        sources[name] = got
        labels[name] = gotl
        print(f"  {len(got)}개  {time.perf_counter()-t0:.0f}초\n")

    sources = {k: v for k, v in sources.items() if len(v) > 0}
    if "Real" not in sources:
        print("real 경로가 하나도 안 나왔다")
        return

    order = ["Real"] + [k for k in sources if k != "Real"]
    metrics = ("max", "p95", "median")

    # ------------------------------------------------------------ 요약
    print("=" * 90)
    print(f"{'source':10s}{'n':>5}"
          + "".join(f"{m + ' 중앙':>16}" for m in metrics))
    print("-" * 90)
    rows = []
    for k in order:
        vals = {m: np.array([s[m] for s in sources[k]]) for m in metrics}
        row = dict(source=k, n=len(sources[k]),
                   **{f"{m}_median": float(np.median(vals[m])) for m in metrics},
                   **{f"{m}_all": vals[m].tolist() for m in metrics})
        rows.append(row)
        print(f"{k:10s}{row['n']:5d}"
              + "".join(f"{row[m + '_median']:16.4f}" for m in metrics))
    print("-" * 90)
    print(f"  corr 필터 문턱 {CORR_THRESHOLD}  (max 는 다들 이 아래여야 정상)")

    Path(args.out + ".json").write_text(
        json.dumps({"rows": rows, "labels": labels}, indent=2,
                   ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------ 그림
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
    fig.subplots_adjust(left=.05, right=.98, top=.86, bottom=.10, wspace=.28)

    cmap = plt.get_cmap("tab10")
    titles = {"max": "max |corr|  (filter edge)",
              "p95": "95th pct |corr|",
              "median": "median |corr|  (typical pair)"}

    for ax, m, tag in zip(axes, metrics, "ABC"):
        data = [np.array([s[m] for s in sources[k]]) for k in order]
        colors = [BLUE if k == "Real" else cmap(i % 10)
                  for i, k in enumerate(order)]
        bp = ax.boxplot(data, labels=order, patch_artist=True, widths=.55,
                        showfliers=False, medianprops=dict(color="black", lw=1.5))
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col); patch.set_alpha(.35)
            patch.set_edgecolor(col)
        rng = np.random.default_rng(0)
        for i, (k, col, d) in enumerate(zip(order, colors, data)):
            jitter = rng.normal(0, .06, size=d.size)
            ax.scatter(np.full(d.size, i + 1) + jitter, d, s=14, color=col,
                       alpha=.6, edgecolor="none", zorder=3)
        if m == "max":
            ax.axhline(CORR_THRESHOLD, color=MUT, ls="--", lw=1.2)
            ax.text(len(order) + .45, CORR_THRESHOLD, " 0.95 filter",
                    color=MUT, fontsize=8, va="center", ha="right")
        ax.set_title(f"{tag}.  {titles[m]}", loc="left", fontweight="bold")
        ax.set_ylabel("|corr|")
        ax.grid(alpha=.2, axis="y")
        ax.set_ylim(-.02, 1.02)

    ns = "  ".join(f"{k} n={len(sources[k])}" for k in order)
    fig.suptitle(f"pairwise |corr| after the 0.95 filter, per dataset/task   ({ns})",
                fontsize=12.5, fontweight="bold", x=.05, ha="left", y=.97)
    fig.savefig(args.out + ".png", dpi=150)
    print(f"\n  저장: {args.out}.png , {args.out}.json")


if __name__ == "__main__":
    main()
