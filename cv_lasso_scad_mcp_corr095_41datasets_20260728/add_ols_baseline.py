#!/usr/bin/env python3
"""KALBI / CV LASSO / Ridge / 최소노름 OLS 를 같은 태스크에서 비교한다.

KALBI 와 LASSO 는 probe_*.json 에 이미 있으므로 재계산하지 않고,
남아 있는 npz 에서 Ridge 와 OLS 만 새로 구해 batch 이름으로 합친다.

읽는 법:
  최소노름 OLS 는 정규화가 없는 극단이다.  p > n 이면 보간해가 되고
  test R2 가 크게 음수가 난다.  KALBI 가 LASSO 쪽이 아니라 OLS 쪽에
  가깝다면, 과선택(nnz > N)이 성능 격차의 주범이라는 뜻이다.

사용법:
    cd ~/Desktop/KALBI/project
    python3.11 add_ols_baseline.py --data-dir ./out_v14_probe \
        --probe probe_out_v14_probe.json --out ~/Desktop/KALBI/baseline_compare
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from test_kalbi_on_synth import read_task, split_standardize

BINS = [(0, .3, "< 0.3"), (.3, .6, "0.3-0.6"), (.6, 1, "0.6-1"),
        (1, 2, "1-2"), (2, 4, "2-4"), (4, np.inf, "> 4")]


def min_norm_ols(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """정규화 없음.  p > n 이면 최소 L2 노름 보간해."""
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return np.asarray(beta, dtype=np.float64)


def ridge_cv(Z: np.ndarray, y: np.ndarray, folds: int = 5,
             seed: int = 0) -> tuple[np.ndarray, float]:
    """SVD 한 번으로 전체 alpha 격자를 훑는 5-fold CV."""
    n, p = Z.shape
    alphas = np.geomspace(1e-4, 1e4, 25) * n
    order = np.random.default_rng(seed).permutation(n)
    mse = np.zeros(alphas.size)

    for f in range(folds):
        te = order[f::folds]
        tr = np.setdiff1d(order, te)
        U, s, Vt = np.linalg.svd(Z[tr], full_matrices=False)
        Uty = U.T @ y[tr]
        for i, a in enumerate(alphas):
            coef = Vt.T @ (s / (s ** 2 + a) * Uty)
            mse[i] += float(((Z[te] @ coef - y[te]) ** 2).mean())
    best = float(alphas[int(np.argmin(mse))])

    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    beta = Vt.T @ (s / (s ** 2 + best) * (U.T @ y))
    return np.asarray(beta, dtype=np.float64), best / n


def stats(beta: np.ndarray, Ztr, ytr, Zte, yte) -> dict:
    nb = float(np.linalg.norm(beta))
    ss = max(float(((yte - yte.mean()) ** 2).sum()), 1e-12)
    return dict(
        r2=1.0 - float(((yte - Zte @ beta) ** 2).sum()) / ss,
        r2_train=1.0 - float(((ytr - Ztr @ beta) ** 2).sum())
        / max(float(((ytr - ytr.mean()) ** 2).sum()), 1e-12),
        nnz=int((np.abs(beta) > 1e-8).sum()),
        norm=nb,
        R=float(np.var(Ztr @ beta) / nb ** 2) if nb > 1e-12 else float("nan"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--probe", required=True, help="KALBI/LASSO 결과 json")
    ap.add_argument("--out", default="baseline_compare")
    ap.add_argument("--seed", type=int, default=20260809)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    probe = {r["batch"]: r for r in json.load(open(args.probe, encoding="utf-8"))}
    print(f"probe {len(probe)}개 로드")

    rows = []
    for path in sorted(Path(args.data_dir).glob("batch_*.npz")):
        if path.name not in probe:
            continue
        if args.limit and len(rows) >= args.limit:
            break
        task = read_task(path)
        if task is None:
            continue
        x, y, _ = task
        t0 = time.perf_counter()
        try:
            Ztr, ytr, Zte, yte = split_standardize(x, y, args.seed)
            Ztr64, Zte64 = Ztr.astype(np.float64), Zte.astype(np.float64)
            ytr64, yte64 = ytr.astype(np.float64), yte.astype(np.float64)

            ols = stats(min_norm_ols(Ztr64, ytr64), Ztr64, ytr64, Zte64, yte64)
            rb, ralpha = ridge_cv(Ztr64, ytr64, seed=args.seed)
            rid = stats(rb, Ztr64, ytr64, Zte64, yte64)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  [건너뜀] {path.name}: {type(exc).__name__}: {exc}")
            continue

        base = probe[path.name]
        rows.append(dict(
            batch=path.name, true_R=base["true_R"], true_k=base["true_k"],
            n_tr=base["n_tr"], p=base["p"], snr=base["snr"],
            kalbi_r2=base["r2"], kalbi_nnz=base["kalbi_nnz"],
            kalbi_norm=base["kalbi_norm"], kalbi_R=base.get("kalbi_R", np.nan),
            lasso_r2=base.get("cv_r2", np.nan),
            lasso_nnz=base.get("cv_nnz", np.nan),
            lasso_norm=base.get("cv_norm", np.nan),
            lasso_R=base.get("cv_R", np.nan),
            ridge_r2=rid["r2"], ridge_norm=rid["norm"], ridge_R=rid["R"],
            ridge_alpha=ralpha,
            ols_r2=ols["r2"], ols_r2_train=ols["r2_train"],
            ols_nnz=ols["nnz"], ols_norm=ols["norm"], ols_R=ols["R"],
            secs=time.perf_counter() - t0))
        if len(rows) % 25 == 0:
            print(f"  {len(rows)}개 완료 ...", flush=True)

    if not rows:
        print("겹치는 태스크가 없습니다."); return
    Path(args.out + ".json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    g = lambda k: np.array([r.get(k, np.nan) for r in rows], dtype=float)
    R = g("true_R")
    out = []
    def say(t=""):
        print(t); out.append(t)

    say("=" * 100)
    say(f"같은 태스크 {len(rows)}개에서 네 추정량 비교"
        f"   (N_train {int(np.median(g('n_tr')))}, P {int(np.median(g('p')))})")
    say("=" * 100)
    say(f"{'참 R':>10}{'n':>4} | {'KALBI':>7}{'LASSO':>8}{'Ridge':>8}{'OLS':>9}"
        f" | {'K nnz':>7}{'L nnz':>7}"
        f" | {'K ||b||':>9}{'L ||b||':>9}{'O ||b||':>9}")
    say("-" * 100)
    for lo, hi, label in BINS:
        m = (R >= lo) & (R < hi)
        if m.sum() < 2:
            continue
        med = lambda k: np.nanmedian(g(k)[m])
        say(f"{label:>10}{int(m.sum()):>4} | {med('kalbi_r2'):>7.3f}"
            f"{med('lasso_r2'):>8.3f}{med('ridge_r2'):>8.3f}{med('ols_r2'):>9.2f}"
            f" | {med('kalbi_nnz'):>7.0f}{med('lasso_nnz'):>7.0f}"
            f" | {med('kalbi_norm'):>9.2f}{med('lasso_norm'):>9.2f}"
            f"{med('ols_norm'):>9.2f}")
    say("-" * 100)
    say()

    names = [("kalbi", "KALBI"), ("lasso", "CV LASSO"),
             ("ridge", "CV Ridge"), ("ols", "min-norm OLS")]
    say(f"{'추정량':>14}{'test R2 중앙':>14}{'R2<0 비율':>11}"
        f"{'||b|| 중앙':>12}{'R(bhat) 중앙':>14}")
    say("-" * 100)
    for key, label in names:
        r2 = g(f"{key}_r2")
        say(f"{label:>14}{np.nanmedian(r2):>14.3f}"
            f"{np.nanmean(r2 < 0):>11.0%}{np.nanmedian(g(f'{key}_norm')):>12.2f}"
            f"{np.nanmedian(g(f'{key}_R')):>14.3f}")
    say("-" * 100)
    say()

    kr, lr, orr = g("kalbi_r2"), g("lasso_r2"), g("ols_r2")
    pos = np.clip(lr - orr, 1e-9, None)
    frac = np.clip((kr - orr) / pos, -0.5, 1.5)
    say("  KALBI 는 OLS 와 LASSO 중 어디에 가까운가")
    say("     0 = OLS 수준,  1 = LASSO 수준")
    for lo, hi, label in BINS:
        m = (R >= lo) & (R < hi) & np.isfinite(frac)
        if m.sum() >= 3:
            say(f"     {label:>10}  n={int(m.sum()):3d}   위치 {np.nanmedian(frac[m]):+.2f}")
    say(f"     {'전체':>10}  n={int(np.isfinite(frac).sum()):3d}   "
        f"위치 {np.nanmedian(frac):+.2f}")
    say()
    say(f"  OLS train R2 중앙 {np.nanmedian(g('ols_r2_train')):.3f} "
        f"(p > n 이면 1.0 = 완전 보간)")
    say(f"  저장: {args.out}.json , {args.out}.txt")
    Path(args.out + ".txt").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
