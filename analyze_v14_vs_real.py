#!/usr/bin/env python3
"""V14 생성 배치와 real 41개의 LASSO 진단량 분포를 비교한다.
 
사용법:
    python3 analyze_v14_vs_real.py --batch-dir /path/to/out_v14_test
 
측정 항목 (real 쪽은 results/cv_lasso_scad_mcp_41_wide.csv에서 읽음):
    lambda_max          = max_j |X_j^T y| / N   (표준화 후 = 최대 주변 상관)
    cv_lambda_ratio     = 5-fold CV 최소점의 lambda / lambda_max
    max_abs_corr        = 유지된 feature 쌍의 최대 |Pearson|
    max_abs_beta, l1    = CV 선택 지점의 계수 크기
    nnz / N             = 활성 계수 비율
"""
 
from __future__ import annotations
 
import argparse
import csv
import glob
import os
import statistics as st
 
import numpy as np
 
RATIOS = np.geomspace(1.0, 1e-6, 60)
 
 
def standardize(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    mu = a.mean(axis=0, keepdims=True)
    sd = a.std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return (a - mu) / sd
 
 
def lasso_path(x: np.ndarray, y: np.ndarray, alphas, tol=1e-6, max_epoch=500):
    """캐시된 Gram 좌표하강. bundle의 solver 계약과 동일한 파라미터화."""
    n, p = x.shape
    g = x.T @ x / n
    c = x.T @ y / n
    d = np.diag(g).copy()
    b = np.zeros(p)
    out = []
    for a in alphas:
        for _ in range(max_epoch):
            r = c - g @ b
            moved = 0.0
            for j in range(p):
                if d[j] <= 0:
                    continue
                z = r[j] + d[j] * b[j]
                nb = np.sign(z) * max(abs(z) - a, 0.0) / d[j]
                if nb != b[j]:
                    r -= g[:, j] * (nb - b[j])
                    moved = max(moved, abs(nb - b[j]))
                    b[j] = nb
            if moved < tol:
                break
        out.append(b.copy())
    return np.asarray(out).T
 
 
def diagnose(x: np.ndarray, y: np.ndarray, folds: int = 5) -> dict:
    x = standardize(x)
    y = standardize(y.reshape(-1, 1)).ravel()
    n, p = x.shape
    lmax = float(np.abs(x.T @ y).max() / n)
    if lmax <= 0:
        return {}
    alphas = RATIOS * lmax
 
    idx = np.random.default_rng(1).permutation(n)
    mse = np.zeros(len(RATIOS))
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        co = lasso_path(x[tr], y[tr], alphas)
        mse += ((x[te] @ co - y[te][:, None]) ** 2).mean(axis=0)
    best = int(np.argmin(mse))
 
    full = lasso_path(x, y, alphas[: best + 1])[:, best]
    nz = np.abs(full) > 1e-8
 
    if p > 1:
        z = x / np.linalg.norm(x, axis=0, keepdims=True)
        corr = np.abs(np.clip(z.T @ z, -1, 1))
        max_corr = float(corr[np.triu_indices(p, 1)].max())
    else:
        max_corr = 0.0
 
    return {
        "n": n,
        "p": p,
        "lambda_max": lmax,
        "cv_lambda_ratio": float(RATIOS[best]),
        "max_abs_corr": max_corr,
        "max_abs_beta": float(np.abs(full).max()),
        "l1_norm": float(np.abs(full).sum()),
        "nnz": int(nz.sum()),
        "nnz_over_n": float(nz.sum() / n),
        "cv_r2": float(1.0 - mse[best] / mse[0]),
    }
 
 
def summarize(name: str, recs: list[dict]) -> None:
    if not recs:
        print(f"{name}: 결과 없음")
        return
 
    def q(key):
        v = sorted(r[key] for r in recs if key in r)
        return v[len(v) // 2]
 
    ratios = np.array([r["cv_lambda_ratio"] for r in recs])
    print(f"\n=== {name}  (n_tasks={len(recs)}) ===")
    print(f"  median N / P            : {q('n'):.0f} / {q('p'):.0f}")
    print(f"  median lambda_max       : {q('lambda_max'):.4f}")
    print(f"  median cv_lambda_ratio  : {q('cv_lambda_ratio'):.5f}")
    print(f"    frac < 1e-2           : {float((ratios < 1e-2).mean()):.3f}")
    print(f"    frac < 1e-3           : {float((ratios < 1e-3).mean()):.3f}")
    print(f"  median max_abs_corr     : {q('max_abs_corr'):.6f}")
    print(f"  median max_abs_beta     : {q('max_abs_beta'):.4f}")
    print(f"  median l1_norm          : {q('l1_norm'):.4f}")
    print(f"  median nnz / N          : {q('nnz_over_n'):.3f}")
    print(f"  median cv_r2            : {q('cv_r2'):.3f}")
 
 
def load_real(bundle_dir: str) -> list[dict]:
    wide = os.path.join(bundle_dir, "results", "cv_lasso_scad_mcp_41_wide.csv")
    npz_dir = os.path.join(bundle_dir, "results", "coefficient_npz")
    recs = []
    for row in csv.DictReader(open(wide)):
        f = os.path.join(npz_dir, row["task_id"] + ".npz")
        if not os.path.exists(f):
            continue
        d = np.load(f)
        b = d["lasso_beta"]
        nz = np.abs(b) > 1e-8
        n = int(row["outer_train_rows"])
        m = d["lasso_mean_mse"]
        recs.append({
            "n": n,
            "p": int(row["P_after_corr095"]),
            "lambda_max": float(row["lasso_outer_train_lambda_max"]),
            "cv_lambda_ratio": float(row["lasso_selected_lambda_ratio"]),
            "max_abs_corr": float(row["retained_max_abs_correlation"]),
            "max_abs_beta": float(np.abs(b).max()),
            "l1_norm": float(np.abs(b).sum()),
            "nnz": int(nz.sum()),
            "nnz_over_n": float(nz.sum() / n),
            "cv_r2": float(1.0 - m.min() / m[0]),
        })
    return recs
 
 
def load_synth(batch_dir: str, limit: int) -> tuple[list[dict], list[dict]]:
    linear, tabicl = [], []
    files = sorted(glob.glob(os.path.join(batch_dir, "batch_*.npz")))[:limit]
    for f in files:
        d = np.load(f, allow_pickle=True)
        xs, ys = d["x"], d["y"]
        src = d["outcome_source_id"] if "outcome_source_id" in d.files else None
        for t in range(xs.shape[0]):
            rec = diagnose(np.asarray(xs[t], dtype=np.float64),
                           np.asarray(ys[t], dtype=np.float64))
            if not rec:
                continue
            is_lin = bool(src[t]) if src is not None else None
            (linear if is_lin else tabicl).append(rec)
        print(f"  처리 완료 {os.path.basename(f)}", flush=True)
    return linear, tabicl
 
 
def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", required=True,
                    help="V14 --out-dir (batch_*.npz 가 있는 폴더)")
    ap.add_argument("--bundle-dir", default=here)
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()
 
    print("real 41개 로드 중...")
    real = load_real(args.bundle_dir)
 
    print(f"synthetic 배치 로드 중: {args.batch_dir}")
    linear, tabicl = load_synth(args.batch_dir, args.limit)
 
    summarize("REAL 41 datasets", real)
    summarize("V14 synthetic - linear outcome", linear)
    summarize("V14 synthetic - tabicl (nonlinear) outcome", tabicl)
 
    print("\n해석 가이드")
    print("  max_abs_corr    real이 0.9499 천장, synthetic이 그보다 낮으면 design 공선성 부족")
    print("  cv_lambda_ratio real의 왼쪽 꼬리(<1e-3)를 synthetic이 못 만들면 과다 shrinkage 원인")
    print("  max_abs_beta    synthetic이 real보다 훨씬 작으면 계수 스케일 캘리브레이션 붕괴")
    print("  nnz/N           real은 0.85까지 감. synthetic이 0.33에서 잘리면 3k<N 가드 영향")
 
 
if __name__ == "__main__":
    main()