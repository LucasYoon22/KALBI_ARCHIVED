#!/usr/bin/env python3
"""새 V14(레일리 타깃팅) 합성 태스크에 기존 KALBI 를 씌워 본다.

핵심 질문:  real 13개에서 본 "R(beta) < 1 이면 test R2 붕괴" 가
            합성 데이터에서도 재현되는가?

재현되면 인과가 닫힌다 —
  V14 가 R ~ 1 만 생성  ->  셀렉터가 R != 1 을 못 배움  ->  real 에서 붕괴.

생성기 manifest 의 참 R(beta)(linear_rayleigh_quotient) 와
KALBI 가 낸 beta 의 test R2 를 짝지어 본다.

사용법 (서버):
    cd /workspace/minsuk/fm-lm-lambda
    PYTHONPATH=/workspace/minsuk/fm-lm-lambda /venv/main/bin/python \
        test_kalbi_on_synth.py --data-dir /workspace/minsuk/data/out_v14_probe
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from infer_kalbi_13 import load_kalbi

OUTCOME_LINEAR = 1


# ------------------------------------------------------------------ 매니페스트

def load_true_rayleigh(root: Path) -> dict[str, dict]:
    """batch 파일명 -> 그 배치의 선형 태스크 메타데이터."""
    path = root / "manifest.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))

    found: list[dict] = []

    def walk(node):
        if isinstance(node, dict):
            if "linear_rayleigh_quotient" in node:
                found.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)

    table: dict[str, dict] = {}
    for entry in found:
        name = None
        for key in ("relative_path", "path", "file", "filename", "batch_file"):
            if isinstance(entry.get(key), str):
                name = Path(entry[key]).name
                break
        if name:
            table[name] = entry
    if not table:                       # 파일명이 없으면 순서로 맞춘다
        table = {"__ordered__": found}  # type: ignore[dict-item]
    return table


# ------------------------------------------------------------------ 태스크

def read_task(path: Path):
    with np.load(path, allow_pickle=False) as payload:
        if int(np.asarray(payload["outcome_source_id"]).reshape(-1)[0]) != OUTCOME_LINEAR:
            return None
        x = np.asarray(payload["X"], dtype=np.float64)
        y = np.asarray(payload["y"], dtype=np.float64)
        extra = {
            key: float(np.asarray(payload[key]).reshape(-1)[0])
            for key in ("linear_support_size", "linear_realized_snr")
            if key in payload
        }
    if x.ndim == 3:
        x = x[0]
    return x, y.reshape(-1), extra


def cv_lasso(Z: np.ndarray, y: np.ndarray, grid: int = 40, folds: int = 5,
             seed: int = 0):
    """같은 train 분할에 CV LASSO. KALBI 실패가 '문제 난이도' 때문인지 가른다."""
    from sklearn.linear_model import lasso_path

    n = Z.shape[0]
    lmax = float(np.abs(Z.T @ y).max() / n)
    if not np.isfinite(lmax) or lmax <= 0:
        return np.zeros(Z.shape[1]), 1.0
    ratios = np.geomspace(1.0, 1e-4, grid)
    alphas = ratios * lmax

    order = np.random.default_rng(seed).permutation(n)
    mse = np.zeros(grid)
    for f in range(folds):
        te = order[f::folds]
        tr = np.setdiff1d(order, te)
        _, coefs, _ = lasso_path(Z[tr], y[tr], alphas=alphas,
                                 tol=1e-3, max_iter=1500)
        mse += ((Z[te] @ coefs - y[te][:, None]) ** 2).mean(axis=0)
    best = int(np.argmin(mse))
    _, coefs, _ = lasso_path(Z, y, alphas=alphas[: best + 1],
                             tol=1e-4, max_iter=4000)
    return coefs[:, best], float(ratios[best])


def split_standardize(x: np.ndarray, y: np.ndarray, seed: int):
    """학습 때와 같은 50/50 행 분할 + train 기준 표준화."""
    rows = x.shape[0]
    order = np.random.default_rng(seed).permutation(rows)
    tr, te = order[: rows // 2], order[rows // 2:]

    mean, sd = x[tr].mean(0, keepdims=True), x[tr].std(0, keepdims=True)
    active = sd > 1e-6
    ztr = np.where(active, (x[tr] - mean) / np.maximum(sd, 1e-8), 0.0)
    zte = np.where(active, (x[te] - mean) / np.maximum(sd, 1e-8), 0.0)

    ym, ys = float(y[tr].mean()), max(float(y[tr].std()), 1e-8)
    return (ztr.astype(np.float32), ((y[tr] - ym) / ys).astype(np.float32),
            zte.astype(np.float32), ((y[te] - ym) / ys).astype(np.float32))


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="test_kalbi_on_synth.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260809)
    ap.add_argument("--baseline", action="store_true",
                    help="같은 분할에 CV LASSO 도 돌려 비교")
    ap.add_argument("--baseline-cells", type=int, default=4_000_000,
                    help="N*P 가 이보다 크면 CV LASSO 생략")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device: {device}  |  torch {torch.__version__}")
    print("KALBI 로드 중 ...")
    from infer_kalbi_13 import kalbi_beta
    model, selector = load_kalbi(device)
    print()

    root = Path(args.data_dir)
    truth = load_true_rayleigh(root)
    ordered = truth.get("__ordered__") if isinstance(truth, dict) else None
    files = sorted(root.glob("batch_*.npz"))
    if args.limit:
        files = files[: args.limit]
    print(f"batch {len(files)}개  |  manifest 선형 항목 "
          f"{len(ordered) if ordered else len(truth)}개")
    print()

    rows, seen_linear = [], 0
    for path in files:
        task = read_task(path)
        if task is None:
            continue
        x, y, extra = task
        entry = None
        if ordered is not None:
            entry = ordered[seen_linear] if seen_linear < len(ordered) else None
        else:
            entry = truth.get(path.name)
        seen_linear += 1

        started = time.perf_counter()
        try:
            ztr, ytr, zte, yte = split_standardize(x, y, args.seed)
            beta, lam, lam_max, d0 = kalbi_beta(model, selector, ztr, ytr, device)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  [건너뜀] {path.name}: {type(exc).__name__}: {exc}",
                  flush=True)
            continue

        norm = float(np.linalg.norm(beta))
        pred = zte @ beta
        ss = max(float(((yte - yte.mean()) ** 2).sum()), 1e-12)
        row = dict(
            batch=path.name,
            n_tr=int(ztr.shape[0]), p=int(ztr.shape[1]),
            true_R=float(entry["linear_rayleigh_quotient"]) if entry else float("nan"),
            true_k=float(entry["linear_support_size"]) if entry else extra.get("linear_support_size", float("nan")),
            true_norm=float(entry.get("linear_beta_l2_norm", float("nan"))) if entry else float("nan"),
            mode=(entry or {}).get("linear_support_selection", "?"),
            targeted=bool((entry or {}).get("linear_rayleigh_targeted", False)),
            snr=float(entry.get("linear_realized_snr", float("nan"))) if entry else float("nan"),
            kalbi_nnz=int((np.abs(beta) > 1e-8).sum()),
            kalbi_norm=norm,
            kalbi_maxabs=float(np.abs(beta).max()),
            kalbi_R=float(np.var(ztr @ beta) / norm ** 2) if norm > 1e-12 else float("nan"),
            kalbi_clip6=int((np.abs(beta) > 5.999).sum()),
            lam_ratio=lam / max(lam_max, 1e-12),
            r2=1.0 - float(((yte - pred) ** 2).sum()) / ss,
            secs=time.perf_counter() - started,
        )
        if args.baseline and ztr.shape[0] * ztr.shape[1] <= args.baseline_cells:
            try:
                t1 = time.perf_counter()
                cb, cratio = cv_lasso(ztr.astype(np.float64),
                                      ytr.astype(np.float64), seed=args.seed)
                cn = float(np.linalg.norm(cb))
                cpred = zte @ cb
                row.update(
                    cv_r2=1.0 - float(((yte - cpred) ** 2).sum()) / ss,
                    cv_nnz=int((np.abs(cb) > 1e-8).sum()),
                    cv_norm=cn,
                    cv_ratio=cratio,
                    cv_R=float(np.var(ztr @ cb) / cn ** 2) if cn > 1e-12 else float("nan"),
                    cv_secs=time.perf_counter() - t1,
                )
            except Exception as exc:                               # noqa: BLE001
                row["cv_error"] = f"{type(exc).__name__}: {exc}"

        npz_k = extra.get("linear_support_size")
        if npz_k is not None and entry is not None:
            row["join_ok"] = bool(abs(npz_k - row["true_k"]) < 0.5)
        rows.append(row)
        if len(rows) % 10 == 0:
            print(f"  {len(rows):3d}개 완료 ...", flush=True)

    if not rows:
        print("선형 태스크를 찾지 못했습니다."); return

    Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                              encoding="utf-8")

    tR = np.array([r["true_R"] for r in rows])
    r2 = np.array([r["r2"] for r in rows])
    nnz = np.array([r["kalbi_nnz"] for r in rows], dtype=float)
    kn = np.array([r["kalbi_norm"] for r in rows])

    print()
    print("=" * 86)
    print(f"선형 태스크 {len(rows)}개")
    print()
    print(f"{'참 R(beta) 구간':>16}{'개수':>7}{'test R2 중앙':>14}"
          f"{'R2<0 비율':>11}{'KALBI nnz 중앙':>15}{'||b|| 중앙':>12}")
    print("-" * 86)
    edges = [0, .3, .6, 1.0, 2.0, 4.0, np.inf]
    labels = ["< 0.3", "0.3 - 0.6", "0.6 - 1", "1 - 2", "2 - 4", "> 4"]
    for label, lo, hi in zip(labels, edges[:-1], edges[1:]):
        m = (tR >= lo) & (tR < hi)
        if not m.any():
            print(f"{label:>16}{0:>7}{'-':>14}{'-':>11}{'-':>15}{'-':>12}")
            continue
        print(f"{label:>16}{int(m.sum()):>7}{np.median(r2[m]):>14.3f}"
              f"{(r2[m] < 0).mean():>11.0%}{np.median(nnz[m]):>15.0f}"
              f"{np.median(kn[m]):>12.3f}")
    print("-" * 86)

    checked = [r for r in rows if "join_ok" in r]
    if checked:
        bad = sum(1 for r in checked if not r["join_ok"])
        print(f"  매니페스트 정렬 검증 (npz의 k 대조) : "
              f"{len(checked) - bad}/{len(checked)} 일치"
              + ("   <-- 불일치! 참 R 신뢰 불가" if bad else ""))

    has_cv = [r for r in rows if "cv_r2" in r]
    if has_cv:
        cR = np.array([r["true_R"] for r in has_cv])
        kk = np.array([r["r2"] for r in has_cv])
        cc = np.array([r["cv_r2"] for r in has_cv])
        print()
        print(f"  === CV LASSO 대조 ({len(has_cv)}개) ===")
        print(f"{'참 R':>12}{'n':>5}{'KALBI R2':>11}{'LASSO R2':>11}{'차이':>9}")
        for lo, hi, lab in [(0, .3, "< 0.3"), (.3, .6, "0.3-0.6"),
                            (.6, 1, "0.6-1"), (1, 2, "1-2"),
                            (2, 4, "2-4"), (4, np.inf, "> 4")]:
            m = (cR >= lo) & (cR < hi)
            if m.sum() < 2:
                continue
            print(f"{lab:>12}{int(m.sum()):>5}{np.median(kk[m]):>11.3f}"
                  f"{np.median(cc[m]):>11.3f}"
                  f"{np.median(cc[m]) - np.median(kk[m]):>+9.3f}")

    valid = np.isfinite(tR) & np.isfinite(r2)
    if valid.sum() > 2:
        rho = np.corrcoef(np.log(np.clip(tR[valid], 1e-6, None)),
                          np.clip(r2[valid], -5, 1))[0, 1]
        print(f"  corr( log 참R , clip(test R2) ) = {rho:+.3f}"
              "    양수면 'R 낮을수록 붕괴' 재현")
    print(f"  전체 test R2  중앙 {np.median(r2):.3f}   "
          f"R2<0 {(r2 < 0).mean():.0%}   최저 {r2.min():.2f}")
    print(f"  KALBI clip6 총합 {sum(r['kalbi_clip6'] for r in rows)}")
    print()
    print("  [real 13개] R<1 인 5개 중 3개가 음수 R2 (-59.8, -3.31, -1.41), "
          "R>1 인 8개는 전부 0.31~0.69")
    print(f"\n  저장: {args.out}")


if __name__ == "__main__":
    main()
