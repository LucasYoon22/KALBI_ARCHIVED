#!/usr/bin/env python3
"""새 13개 벤치마크의 CV LASSO 계수 크기와 조건수를 측정한다.

핵심 질문:  real 데이터의 ||beta|| 가 정말 1 근처인가, 아니면 폭발 케이스가 있는가?
            (V14 는 ||beta|| = 0.854 ~ 1.144, max|beta_j| <= 1.0000)

전처리는 번들의 41개 계약과 동일하게 outer-training rows 에서만 fit 한다.
  1. 비유한/상수 컬럼 제거
  2. train mean / population sd 로 표준화
  3. 원래 컬럼 순서대로 훑으며, 이미 채택된 모든 컬럼과 |corr| < 0.95 인 것만 채택
  4. 같은 변환을 test 에 적용

사용법:
    python3 measure_13_beta.py --data-dir ./real_regression_data
    python3 measure_13_beta.py --data-dir ./real_regression_data --only QSAR-TID-10331
    python3 measure_13_beta.py --data-dir ./real_regression_data --grid 60   # 빠르게
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import warnings

import numpy as np

warnings.filterwarnings("ignore")

CORR_THRESHOLD = 0.95
N_FOLDS = 5


# ---------------------------------------------------------------- 입출력

def load_split(d: str):
    """loader 가 저장한 dense(.npy) 또는 sparse(.npz) split 을 읽는다."""
    def _x(stem):
        npy, npz = os.path.join(d, stem + ".npy"), os.path.join(d, stem + ".npz")
        if os.path.exists(npy):
            return np.asarray(np.load(npy, allow_pickle=False), dtype=np.float64)
        if os.path.exists(npz):
            from scipy import sparse
            return np.asarray(sparse.load_npz(npz).todense(), dtype=np.float64)
        raise FileNotFoundError(f"{stem}.npy / .npz 없음: {d}")

    Xtr, Xte = _x("X_train"), _x("X_test")
    ytr = np.asarray(np.load(os.path.join(d, "y_train.npy")), dtype=np.float64).ravel()
    yte = np.asarray(np.load(os.path.join(d, "y_test.npy")), dtype=np.float64).ravel()
    return Xtr, ytr, Xte, yte


# ---------------------------------------------------------------- 전처리

def fit_preprocessing(Xtr: np.ndarray):
    """train 에서만 fit. (활성 컬럼 마스크, mean, sd) 를 돌려준다."""
    finite = np.isfinite(Xtr).all(axis=0)
    mean = np.where(finite, Xtr.mean(axis=0), 0.0)
    sd = np.where(finite, Xtr.std(axis=0), 0.0)          # population sd
    active = finite & (sd > 1e-10)
    sd = np.where(active, sd, 1.0)
    return active, mean, sd


def corr_filter(Z: np.ndarray, thr: float = CORR_THRESHOLD) -> np.ndarray:
    """원래 컬럼 순서대로 훑으며 |corr| < thr 인 것만 채택. 채택 인덱스를 돌려준다."""
    n, p = Z.shape
    U = (Z / np.linalg.norm(Z, axis=0, keepdims=True)).astype(np.float32)
    keep: list[int] = [0]
    K = U[:, [0]].copy()                       # (n, |keep|) 캐시
    for j in range(1, p):
        if np.abs(K.T @ U[:, j]).max() < thr:
            keep.append(j)
            K = np.concatenate([K, U[:, [j]]], axis=1)
    return np.asarray(keep, dtype=np.int64)


# ---------------------------------------------------------------- LASSO

def cv_lasso(Z: np.ndarray, y: np.ndarray, grid: int, seed: int = 0):
    """5-fold CV 로 lambda 를 고르고 전체 train 에 refit. (beta, ratio, nnz) 반환."""
    from sklearn.linear_model import lasso_path

    n = Z.shape[0]
    lmax = float(np.abs(Z.T @ y).max() / n)
    if lmax <= 0:
        return np.zeros(Z.shape[1]), 1.0, 0, lmax
    ratios = np.geomspace(1.0, 1e-6, grid)
    alphas = ratios * lmax

    idx = np.random.default_rng(seed).permutation(n)
    mse = np.zeros(grid)
    for f in range(N_FOLDS):
        te = idx[f::N_FOLDS]
        tr = np.setdiff1d(idx, te)
        _, coefs, _ = lasso_path(Z[tr], y[tr], alphas=alphas, tol=1e-4, max_iter=3000)
        mse += ((Z[te] @ coefs - y[te][:, None]) ** 2).mean(axis=0)
    best = int(np.argmin(mse))

    _, coefs, _ = lasso_path(Z, y, alphas=alphas[: best + 1], tol=1e-5, max_iter=10000)
    beta = coefs[:, best]
    return beta, float(ratios[best]), int((np.abs(beta) > 1e-8).sum()), lmax


# ---------------------------------------------------------------- 진단

def diagnose(label: str, ddir: str, grid: int):
    t0 = time.perf_counter()
    Xtr, ytr, Xte, yte = load_split(ddir)
    n_tr, p_raw = Xtr.shape

    active, mean, sd = fit_preprocessing(Xtr)
    Ztr = ((Xtr - mean) / sd)[:, active]
    Zte = ((Xte - mean) / sd)[:, active]
    p_const = int(Ztr.shape[1])

    keep = corr_filter(Ztr)
    Ztr, Zte = Ztr[:, keep], Zte[:, keep]
    p_after = int(Ztr.shape[1])

    ym, ys = ytr.mean(), ytr.std()
    ys = ys if ys > 1e-12 else 1.0
    ytr_s, yte_s = (ytr - ym) / ys, (yte - ym) / ys

    beta, ratio, nnz, lmax = cv_lasso(Ztr, ytr_s, grid)

    nb = float(np.linalg.norm(beta))
    mx = float(np.abs(beta).max())
    var_xb = float(np.var(Ztr @ beta))
    R = var_xb / nb ** 2 if nb > 1e-12 else np.nan

    # 선택된 support 의 부분 Gram 최소고유값
    S = np.flatnonzero(np.abs(beta) > 1e-8)
    lam_min = np.nan
    if 1 <= S.size <= 2000:
        G = Ztr[:, S].T @ Ztr[:, S] / n_tr
        lam_min = float(np.linalg.eigvalsh(G)[0])

    pred = Zte @ beta
    r2 = 1.0 - ((yte_s - pred) ** 2).sum() / max(((yte_s - yte_s.mean()) ** 2).sum(), 1e-12)

    return dict(label=label, n_tr=n_tr, p_raw=p_raw, p_const=p_const, p_after=p_after,
                removed=100 * (1 - p_after / max(p_const, 1)),
                ratio=ratio, nnz=nnz, norm=nb, maxabs=mx, R=R,
                lam_min=lam_min, r2=float(r2), secs=time.perf_counter() - t0)


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--only", default=None, help="한 개만 처리")
    ap.add_argument("--grid", type=int, default=100, help="lambda 그리드 점 수")
    ap.add_argument("--out", default="measure_13_beta.json")
    args = ap.parse_args()

    dirs = [d for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d)) and not d.startswith(".")]
    def _size(d):
        try:
            return sum(os.path.getsize(os.path.join(args.data_dir, d, f))
                       for f in os.listdir(os.path.join(args.data_dir, d)))
        except OSError:
            return 0
    dirs = sorted(dirs, key=_size)          # 작은 것부터
    if args.only:
        dirs = [d for d in dirs if args.only in d]
    if not dirs:
        print("데이터 폴더 없음. loader 를 --all 로 먼저 실행하십시오."); sys.exit(1)

    rows = []
    for d in dirs:
        print(f"  ... {d} 처리 시작", flush=True)
        try:
            r = diagnose(d, os.path.join(args.data_dir, d), args.grid)
            rows.append(r)
            print(f"  [{len(rows):2d}/{len(dirs)}] {d:34s} "
                  f"P {r['p_raw']:5d}->{r['p_after']:5d}  nnz {r['nnz']:5d}  "
                  f"||b|| {r['norm']:8.3f}  max|b| {r['maxabs']:7.3f}  "
                  f"({r['secs']:.0f}s)", flush=True)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  [건너뜀] {d}: {type(exc).__name__}: {exc}", flush=True)

    if not rows:
        return

    print()
    print("=" * 118)
    print(f"{'dataset':34s}{'N_tr':>7}{'P_raw':>7}{'P_after':>8}{'제거%':>7}"
          f"{'nnz':>6}{'lam_ratio':>11}{'||beta||':>10}{'max|b_j|':>10}"
          f"{'R(beta)':>11}{'lam_min':>11}{'test R2':>9}")
    print("-" * 118)
    for r in sorted(rows, key=lambda z: -z["norm"]):
        print(f"{r['label'][:34]:34s}{r['n_tr']:7d}{r['p_raw']:7d}{r['p_after']:8d}"
              f"{r['removed']:6.1f}%{r['nnz']:6d}{r['ratio']:11.2e}"
              f"{r['norm']:10.3f}{r['maxabs']:10.3f}{r['R']:11.2e}"
              f"{r['lam_min']:11.2e}{r['r2']:9.3f}")

    nb = np.array([r["norm"] for r in rows])
    mx = np.array([r["maxabs"] for r in rows])
    R = np.array([r["R"] for r in rows])
    print("-" * 118)
    print(f"  ||beta||   최소 {nb.min():.3f}   중앙 {np.median(nb):.3f}   최대 {nb.max():.3f}")
    print(f"  max|b_j|   최소 {mx.min():.3f}   중앙 {np.median(mx):.3f}   최대 {mx.max():.3f}")
    print(f"  R(beta)    최소 {np.nanmin(R):.2e}  중앙 {np.nanmedian(R):.2e}  최대 {np.nanmax(R):.2e}")
    print()
    print(f"  ||beta|| > 1.5 :  {int((nb > 1.5).sum())} / {len(rows)}")
    print(f"  ||beta|| > 10  :  {int((nb > 10).sum())} / {len(rows)}   <-- 폭발 케이스")
    print(f"  max|b_j| > 6   :  {int((mx > 6).sum())} / {len(rows)}   <-- Direct96 정의역 밖")
    print()
    print("  [비교] V14 32 태스크:  ||beta|| 0.854~1.144,  max|b_j| 최대 1.0000")
    print("  [비교] 이전 41개    :  ||beta|| 중앙 0.716, 12%가 >10, 최대 174.59")
    print()
    print("  판정 기준")
    print("    max|b_j| > 6 가 0 개  ->  Direct96 정의역 미탐색 문제는 현 벤치마크에서 무의미")
    print("    ||beta|| > 10 가 있음 ->  V14 가 못 만드는 병리 꼬리가 존재. 문제 1 유효")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    print(f"\n  저장: {args.out}")


if __name__ == "__main__":
    main()
