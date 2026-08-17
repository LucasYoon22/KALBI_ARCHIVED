#!/usr/bin/env python3
"""합성-real 간극을 exact LASSO 경로로 잰다.  V15 가 그 간극을 좁혔는지 본다.

교수님 figure1 의 재현 + V15 추가.  KALBI 를 전혀 쓰지 않는다.
데이터만 보는 진단이라 셀렉터/솔버 문제와 완전히 분리된 증거가 된다.

    capacity-normalized support  s(r) = nnz(lambda) / min(N, P)
    lambda = r * lambda_max ,   lambda_max = max_j |X_j^T y| / N
    r 을 1 에서 1e-2 까지 96 점 로그 격자

각 태스크가 96 차원 벡터 하나가 된다.  네 패널로 본다.

    A  중앙 경로 + 사분위 band
    B  96 차원 경로의 PCA
    C  real -> 합성 함수적 L2 거리
    D  real 데이터셋마다 가장 가까운 합성 소스

    python3.11 lasso_path_gap.py \
        --real real_regression_data \
        --synth V14=out_v14_probe --synth V15=/workspace/v15_pool \
        --out lasso_gap
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

GRID = 96
R_MAX, R_MIN = 1.0, 1e-2
BLUE, MUT = "#1F4E79", "#6E7B88"


# ------------------------------------------------------------------ 경로

def capacity_path(X: np.ndarray, y: np.ndarray, ratios: np.ndarray,
                  metric: str = "l1") -> np.ndarray:
    """표준화 후 exact LASSO 경로를 요약한다.

    metric="l1"   ||beta(lambda)||_1 / sqrt(p)
        연속량이고 축소 정도까지 반영한다.  X, y 를 표준화했으므로
        beta 가 무차원이라 태스크 간 비교가 된다.  sqrt(p) 로 나누는 것은
        차원 정규화다.  등방 신호에서 ||beta||_1 이 sqrt(p) 로 커지므로
        그 성장분을 빼면 남는 것이 순수한 밀도 신호다.

    metric="nnz"  nnz(lambda) / min(N, P)
        이산량이고 |beta_j| > 1e-10 이라는 임의 임계에 의존한다.
    """
    from sklearn.linear_model import lasso_path

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, p = X.shape

    mu = X.mean(0)
    sd = X.std(0)
    keep = np.isfinite(sd) & (sd > 1e-12)
    if keep.sum() < 2:
        return None
    Z = (X[:, keep] - mu[keep]) / sd[keep]
    yv = y - y.mean()
    ys = float(y.std())
    if not np.isfinite(ys) or ys < 1e-12:
        return None
    yv = yv / ys

    lam_max = float(np.abs(Z.T @ yv).max() / n)
    if not np.isfinite(lam_max) or lam_max <= 0:
        return None

    _, coefs, _ = lasso_path(Z, yv, alphas=ratios * lam_max,
                             tol=1e-4, max_iter=3000)
    pk = int(keep.sum())
    if metric == "nnz":
        nnz = (np.abs(coefs) > 1e-10).sum(0).astype(np.float64)
        return nnz / max(float(min(n, pk)), 1.0)
    return np.abs(coefs).sum(0).astype(np.float64) / np.sqrt(float(pk))


# ------------------------------------------------------------------ 로딩

def iter_real(root: Path, ratios, limit, metric):
    from infer_kalbi_13 import load_split, preprocess

    dirs = sorted(d for d in root.iterdir()
                  if d.is_dir() and not d.name.startswith("."))[:limit]
    for d in dirs:
        try:
            Ztr, ytr, _, _ = preprocess(*load_split(d), cache_dir=d)
            s = capacity_path(Ztr, ytr, ratios, metric)
        except Exception as exc:                                   # noqa: BLE001
            print(f"    {d.name[:34]:34s} 실패 {type(exc).__name__}")
            continue
        if s is not None:
            print(f"    {d.name[:34]:34s} OK  N={Ztr.shape[0]} P={Ztr.shape[1]}")
            yield d.name, s


def iter_npz(root: Path, ratios, limit, max_cells, metric):
    files = sorted(root.glob("batch_*.npz"))[:limit]
    for f in files:
        try:
            with np.load(f, allow_pickle=False) as z:
                X = np.asarray(z["X"][0], dtype=np.float64)
                y = np.asarray(z["y"][0], dtype=np.float64)
            if X.shape[0] * X.shape[1] > max_cells:
                continue
            s = capacity_path(X, y, ratios, metric)
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
    ap.add_argument("--out", default="lasso_gap")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-cells", type=int, default=6_000_000,
                    help="너무 큰 합성 태스크는 건너뛴다")
    ap.add_argument("--metric", choices=("l1", "nnz"), default="l1",
                    help="l1 = ||beta||_1/sqrt(p) (기본), nnz = support/min(N,P)")
    args = ap.parse_args()

    ratios = np.geomspace(R_MAX, R_MIN, GRID)
    sources: dict[str, list] = {}
    labels: dict[str, list] = {}

    print("real 13")
    t0 = time.perf_counter()
    rl, rs = [], []
    for name, s in iter_real(Path(args.real), ratios, args.limit, args.metric):
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
        got = [s for _, s in iter_npz(Path(path), ratios, args.limit,
                                      args.max_cells, args.metric)]
        sources[name] = got
        print(f"  {len(got)}개  {time.perf_counter()-t0:.0f}초\n")

    sources = {k: v for k, v in sources.items() if len(v) > 0}
    if "Real" not in sources:
        print("real 경로가 하나도 안 나왔다")
        return

    arrays = {k: np.stack(v) for k, v in sources.items()}
    real = arrays["Real"]
    real_med = np.median(real, axis=0)

    # ------------------------------------------------------------ 요약
    print("=" * 78)
    MLAB = ("||b||_1/sqrt(p)" if args.metric == "l1" else "nnz/min(N,P)")
    print(f"  지표  {MLAB}")
    print(f"{'source':12s}{'n':>5}{'끝값 중앙':>14}{'real 대비':>11}"
          f"{'L2 거리':>10}")
    print("-" * 78)
    rows = []
    for k, A in arrays.items():
        med = np.median(A, axis=0)
        end = float(med[-1])
        l2 = float(np.sqrt(np.mean((med - real_med) ** 2)))
        ratio = end / max(float(real_med[-1]), 1e-9)
        rows.append(dict(source=k, n=int(A.shape[0]), end=end,
                         ratio=ratio, l2=l2))
        print(f"{k:12s}{A.shape[0]:5d}{end:14.3f}{ratio:11.2f}x{l2:10.4f}")
    print("-" * 78)
    print(f"  끝값 = lambda 가 lambda_max 의 1% 일 때의 {MLAB}")
    print("  real 대비 1.0 에 가까울수록 real 과 닮았다")

    Path(args.out + ".json").write_text(
        json.dumps({"rows": rows, "ratios": ratios.tolist(),
                    "median": {k: np.median(A, 0).tolist()
                               for k, A in arrays.items()}},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------ 그림
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.subplots_adjust(left=.06, right=.98, top=.88, bottom=.14, wspace=.22)

    cmap = plt.get_cmap("tab10")
    order = ["Real"] + [k for k in arrays if k != "Real"]

    a = ax[0]
    for i, k in enumerate(order):
        A = arrays[k]
        med = np.median(A, axis=0)
        col = BLUE if k == "Real" else cmap(i % 10)
        lw = 2.6 if k == "Real" else 1.6
        a.plot(ratios, med, color=col, lw=lw, label=f"{k} (n={A.shape[0]})",
               zorder=5 if k == "Real" else 3)
        if A.shape[0] >= 4:
            lo = np.percentile(A, 25, axis=0)
            hi = np.percentile(A, 75, axis=0)
            a.fill_between(ratios, lo, hi, color=col, alpha=.13, lw=0)
    a.set_xscale("log"); a.invert_xaxis()
    a.set_xlabel(r"regularization ratio  $\lambda/\lambda_{\max}$  (decreasing $\rightarrow$)")
    a.set_ylabel(r"$\|\hat\beta(\lambda)\|_1/\sqrt{p}$" if args.metric == "l1"
                 else "capacity-normalized support")
    a.set_title("A.  median path and interquartile band",
                loc="left", fontweight="bold")
    a.legend(fontsize=8); a.grid(alpha=.2)

    b = ax[1]
    allA = np.concatenate([arrays[k] for k in order], axis=0)
    src = np.concatenate([[k] * arrays[k].shape[0] for k in order])
    C = allA - allA.mean(0)
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    pc = U[:, :2] * S[:2]
    var = S ** 2 / max(float((S ** 2).sum()), 1e-12)
    for i, k in enumerate(order):
        m = src == k
        col = BLUE if k == "Real" else cmap(i % 10)
        b.scatter(pc[m, 0], pc[m, 1], s=44 if k == "Real" else 16,
                  c=[col], alpha=.85 if k == "Real" else .5,
                  edgecolor="w" if k == "Real" else "none",
                  linewidth=.7, zorder=5 if k == "Real" else 3, label=k)
    b.set_xlabel(f"PC1 ({var[0]:.1%})")
    b.set_ylabel(f"PC2 ({var[1]:.1%})")
    b.set_title("B.  PCA of the 96-dimensional paths",
                loc="left", fontweight="bold")
    b.legend(fontsize=8); b.grid(alpha=.2)

    fig.suptitle("Synthetic-real gap in exact-LASSO paths  "
                 + ("($\\ell_1/\\sqrt{p}$)" if args.metric == "l1"
                    else "(capacity-normalized support)"),
                 fontsize=13, fontweight="bold", x=.06, ha="left", y=.97)
    fig.savefig(args.out + ".png", dpi=150)
    print(f"\n  저장: {args.out}.png , {args.out}.json")


if __name__ == "__main__":
    main()
