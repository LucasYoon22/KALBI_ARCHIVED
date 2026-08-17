#!/usr/bin/env python3
"""corr 0.95 그리디 필터 결과를 한 번 계산해 디스크에 고정한다.

같은 함수가 실행 맥락에 따라 다른 P 를 낸 적이 있어(3972 vs 526),
필터를 캐시로 못 박아 모든 스크립트가 동일한 전처리를 쓰게 한다.
교수님 로더의 reported_p_after_corr095 와 대조해 검증까지 한다.

    python3.11 build_corr_cache.py --data-dir real_regression_data
    python3.11 build_corr_cache.py --data-dir real_regression_data --force
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CACHE_NAME = "corr095_keep.npy"
META_NAME = "corr095_keep.json"


def reference_table() -> dict[str, int]:
    """교수님 로더의 기준값. 없으면 빈 dict."""
    try:
        import dataclasses as dc
        import load_selected_real_regression as L
    except Exception:                                              # noqa: BLE001
        return {}
    out: dict[str, int] = {}
    for spec in L.SELECTED_DATASETS:
        d = dc.asdict(spec) if dc.is_dataclass(spec) else dict(spec._asdict())
        label = d.get("label") or d.get("openml_name")
        value = d.get("reported_p_after_corr095")
        if label and value:
            out[str(label)] = int(value)
    return out


def active_mask(Xtr: np.ndarray):
    finite = np.isfinite(Xtr).all(axis=0)
    mean = np.where(finite, Xtr.mean(axis=0), 0.0)
    sd = np.where(finite, Xtr.std(axis=0), 0.0)
    active = finite & (sd > 1e-10)
    return active, mean, np.where(active, sd, 1.0)


def greedy_keep(Z: np.ndarray, thr: float) -> np.ndarray:
    """float64 고정.  K 를 미리 할당해 concatenate 재할당을 없앤다."""
    Z = np.asarray(Z, dtype=np.float64)
    n, p = Z.shape
    U = Z / np.maximum(np.linalg.norm(Z, axis=0, keepdims=True), 1e-300)
    K = np.empty((n, p), dtype=np.float64)
    K[:, 0] = U[:, 0]
    keep = [0]
    for j in range(1, p):
        m = len(keep)
        if float(np.abs(K[:, :m].T @ U[:, j]).max()) < thr:
            K[:, m] = U[:, j]
            keep.append(j)
    return np.asarray(keep, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="real_regression_data")
    ap.add_argument("--thr", type=float, default=0.95)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from infer_kalbi_13 import load_split

    ref = reference_table()
    root = Path(args.data_dir)
    print(f"{'dataset':32s}{'raw':>7}{'active':>8}{'kept':>7}"
          f"{'기준':>7}{'판정':>7}")
    print("-" * 70)

    rows, bad = [], 0
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        cache = d / CACHE_NAME
        if cache.exists() and not args.force:
            keep = np.load(cache)
            status = "cached"
        else:
            try:
                Xtr, *_ = load_split(d)
            except Exception:                                      # noqa: BLE001
                continue
            active, mean, sd = active_mask(Xtr)
            Z = ((Xtr - mean) / sd)[:, active]
            local = greedy_keep(Z, args.thr)
            keep = np.flatnonzero(active)[local]        # 원본 열 인덱스
            np.save(cache, keep)
            (d / META_NAME).write_text(json.dumps(dict(
                threshold=args.thr, p_raw=int(Xtr.shape[1]),
                p_active=int(active.sum()), p_kept=int(keep.size),
                dtype="float64"), indent=2), encoding="utf-8")
            status = "new"

        expected = ref.get(d.name)
        mark = "-"
        if expected is not None:
            mark = "OK" if int(keep.size) == expected else "MISMATCH"
            if mark == "MISMATCH":
                bad += 1
        try:
            Xtr, *_ = load_split(d)
            raw, act = Xtr.shape[1], int(active_mask(Xtr)[0].sum())
        except Exception:                                          # noqa: BLE001
            raw = act = -1
        print(f"{d.name[:32]:32s}{raw:>7}{act:>8}{keep.size:>7}"
              f"{(expected if expected else '-'):>7}{mark:>9}  {status}")
        rows.append(dict(label=d.name, p_raw=raw, p_active=act,
                         p_kept=int(keep.size), expected=expected))

    print("-" * 70)
    print(f"  캐시 파일: <dataset>/{CACHE_NAME}  (원본 열 인덱스, float64 계산)")
    if bad:
        print(f"  기준값 불일치 {bad}개 — 확인 필요")
    else:
        print("  기준값 전부 일치")


if __name__ == "__main__":
    main()
