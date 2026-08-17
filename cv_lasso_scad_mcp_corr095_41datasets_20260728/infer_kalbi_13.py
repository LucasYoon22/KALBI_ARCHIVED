#!/usr/bin/env python3
"""KALBI (V6 beta solver + V24 selector) 를 13개 real 벤치마크에 추론한다.

서버(vast.ai)에서 실행한다.  전처리는 measure_13_beta.py 와 동일하게 맞춰
CV LASSO 결과와 직접 비교할 수 있게 한다.

  1. 비유한/상수 컬럼 제거
  2. train mean / population sd 로 표준화
  3. |corr| < 0.95 그리디 필터
  4. y 도 train 기준 표준화
  5. KALBI 로 beta 1개 (셀렉터가 고른 lambda/penalty) 추론

사용법:
    cd /workspace/minsuk/fm-lm-lambda
    PYTHONPATH=/workspace/minsuk/fm-lm-lambda /venv/main/bin/python infer_kalbi_13.py \
        --data-dir /workspace/minsuk/real_regression_data \
        --out /workspace/minsuk/infer_kalbi_13.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# 서버 기본값.  로컬에서는 환경변수로 덮어쓴다.
#   export KALBI_PROJECT=/Users/junseoyoon/Desktop/KALBI/project
#   export KALBI_CKPT=/Users/junseoyoon/Desktop/KALBI/checkpoint_selector_step030000.pt
PROJECT = Path(os.environ.get("KALBI_PROJECT", "/workspace/minsuk/fm-lm-lambda"))
PKG = os.environ.get(
    "KALBI_PKG",
    "kalbi_v6_exactprox10_eta4x_v24_additive512_traj3_k32s32_p100k_effbatch32",
)
CKPT = Path(os.environ.get(
    "KALBI_CKPT",
    str(PROJECT / "runs"
        / "kalbi_v24_selectoronly_v14_snr0p2to50_frozenbeta_step42700_effbatch32_20260728"
        / "checkpoint_selector_step030000.pt"),
))
CONFIG = Path(os.environ.get("KALBI_CONFIG", str(PROJECT / "fpk_v33_kkt0.yaml")))
V24_SCALER = PROJECT / PKG / "v24_scaler_rolling256_step42700.pt"
INPUT_SCALER = (
    PROJECT
    / "kalbi_v6_exactprox10_eta4x_v23_independentraw_k32s32_p100k_effbatch32"
    / "v22_input_scaler_train64.pt"
)

CORR_THRESHOLD = 0.95


# ------------------------------------------------------------------ 데이터

def load_split(d: Path):
    def _x(stem: str) -> np.ndarray:
        npy, npz = d / f"{stem}.npy", d / f"{stem}.npz"
        if npy.exists():
            return np.asarray(np.load(npy, allow_pickle=False), dtype=np.float64)
        if npz.exists():
            from scipy import sparse
            return np.asarray(sparse.load_npz(npz).todense(), dtype=np.float64)
        raise FileNotFoundError(f"{stem}.npy / .npz 없음: {d}")

    Xtr, Xte = _x("X_train"), _x("X_test")
    ytr = np.asarray(np.load(d / "y_train.npy"), dtype=np.float64).ravel()
    yte = np.asarray(np.load(d / "y_test.npy"), dtype=np.float64).ravel()
    return Xtr, ytr, Xte, yte


def corr_filter(Z: np.ndarray, thr: float = CORR_THRESHOLD) -> np.ndarray:
    U = (Z / np.linalg.norm(Z, axis=0, keepdims=True)).astype(np.float32)
    keep = [0]
    K = U[:, [0]].copy()
    for j in range(1, Z.shape[1]):
        if np.abs(K.T @ U[:, j]).max() < thr:
            keep.append(j)
            K = np.concatenate([K, U[:, [j]]], axis=1)
    return np.asarray(keep, dtype=np.int64)


def preprocess(Xtr, ytr, Xte, yte, cache_dir=None):
    """상수/비유한 제거 -> train 기준 표준화 -> corr 0.95 그리디 필터.

    cache_dir 에 corr095_keep.npy 가 있으면 그 인덱스를 쓴다.
    같은 필터가 실행 맥락에 따라 다른 결과를 낸 적이 있어
    build_corr_cache.py 로 한 번 고정해 두고 모두가 공유한다.
    """
    finite = np.isfinite(Xtr).all(axis=0)
    mean = np.where(finite, Xtr.mean(axis=0), 0.0)
    sd = np.where(finite, Xtr.std(axis=0), 0.0)
    active = finite & (sd > 1e-10)
    sd = np.where(active, sd, 1.0)

    cached = None
    if cache_dir is not None:
        path = Path(cache_dir) / "corr095_keep.npy"
        if path.exists():
            cached = np.load(path)                 # 원본 열 인덱스

    if cached is not None:
        Ztr = ((Xtr[:, cached] - mean[cached]) / sd[cached])
        Zte = ((Xte[:, cached] - mean[cached]) / sd[cached])
    else:
        Ztr = ((Xtr - mean) / sd)[:, active]
        Zte = ((Xte - mean) / sd)[:, active]
        keep = corr_filter(Ztr)
        Ztr, Zte = Ztr[:, keep], Zte[:, keep]

    ym = float(ytr.mean())
    ys = float(ytr.std()) or 1.0
    return (Ztr.astype(np.float32), ((ytr - ym) / ys).astype(np.float32),
            Zte.astype(np.float32), ((yte - ym) / ys).astype(np.float32))


# ------------------------------------------------------------------ 모델

def load_kalbi(device: torch.device):
    sys.path.insert(0, str(PROJECT))
    model_mod = importlib.import_module(f"{PKG}.model")
    sel_mod = importlib.import_module(f"{PKG}.selector_v24_additive512_trajectory")
    from train_fpk import _load_yaml

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    meta = ckpt.get("metadata", {})
    print(f"  checkpoint step      : {int(ckpt.get('completed_steps', -1))}")
    print(f"  training_mode        : {meta.get('training_mode')}")
    print(f"  source_joint_step    : {meta.get('source_joint_step')}")

    model = model_mod.build_model(_load_yaml(str(CONFIG)), device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()

    selector = sel_mod.DirectPenaltyMLPQuerySelectorV24Additive512(
        hidden=512,
        derivative_floor=1e-6,
        squareplus_scale=0.02,
        direct_head_init_std=1e-3,
    )
    selector.load_state_dict(ckpt["selector_state_dict"], strict=True)
    selector.to(device).eval()

    # state_dict 에 스케일러 버퍼가 있으면 그대로 쓰고, 없으면 캐시에서 채운다
    if not bool(selector.v24_scaler.cache_loaded.item()):
        selector.load_v24_scaler_cache(
            torch.load(V24_SCALER, map_location="cpu", weights_only=False))
        print("  v24 scaler           : 캐시에서 로드")
    if not bool(selector.input_scaler.cache_loaded.item()):
        selector.load_input_scaler_cache(
            torch.load(INPUT_SCALER, map_location="cpu", weights_only=False))
        print("  v22 input scaler     : 캐시에서 로드")

    return model, selector


@torch.no_grad()
def kalbi_beta(model, selector, Z: np.ndarray, y: np.ndarray, device):
    from train_fpk import _build_X_y_tensors

    x, yv = _build_X_y_tensors([Z], [y], device=device, dtype=torch.float32)
    encoded = model.encode_dataset(model.precompute(x, yv))
    query = selector(
        x, yv, encoded["lambda_max"].float(),
        solver_context=encoded, return_representation=True,
    )
    aux = model.decode_queries(
        encoded,
        lambdas=query["lambdas"],
        lambda_max=encoded["lambda_max"].float()[:, None],
        penalty_shape_features=query["penalty_shape_features"],
        penalty_d0=query["penalty_d0"],
        return_aux=True,
    )
    beta = aux["beta_q"][:, 0].float()[0].cpu().numpy().astype(np.float64)
    lam = float(query["lambdas"].reshape(-1)[0])
    lam_max = float(encoded["lambda_max"].float().reshape(-1)[0])
    d0 = float(query["penalty_d0"].reshape(-1)[0])
    return beta[: Z.shape[1]], lam, lam_max, d0


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="infer_kalbi_13.json")
    ap.add_argument("--only", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device: {device}  |  torch {torch.__version__}")
    print("KALBI 로드 중 ...")
    model, selector = load_kalbi(device)
    print()

    root = Path(args.data_dir)
    dirs = sorted((d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")),
                  key=lambda d: sum(f.stat().st_size for f in d.iterdir() if f.is_file()))
    if args.only:
        dirs = [d for d in dirs if args.only in d.name]

    rows = []
    for d in dirs:
        t0 = time.perf_counter()
        try:
            Xtr, ytr, Xte, yte = load_split(d)
            Ztr, ytr_s, Zte, yte_s = preprocess(Xtr, ytr, Xte, yte, cache_dir=d)
            beta, lam, lam_max, d0 = kalbi_beta(model, selector, Ztr, ytr_s, device)

            nnz = int((np.abs(beta) > 1e-8).sum())
            nb = float(np.linalg.norm(beta))
            mx = float(np.abs(beta).max())
            R = float(np.var(Ztr @ beta) / nb ** 2) if nb > 1e-12 else float("nan")
            pred = Zte @ beta
            ss = max(float(((yte_s - yte_s.mean()) ** 2).sum()), 1e-12)
            r2 = 1.0 - float(((yte_s - pred) ** 2).sum()) / ss

            row = dict(label=d.name, n_tr=int(Ztr.shape[0]), p_after=int(Ztr.shape[1]),
                       lam=lam, lam_max=lam_max, ratio=lam / max(lam_max, 1e-12),
                       d0=d0, nnz=nnz, norm=nb, maxabs=mx, R=R, r2=r2,
                       clipped=int((np.abs(beta) > 5.999).sum()),
                       secs=time.perf_counter() - t0)
            rows.append(row)
            print(f"  [{len(rows):2d}/{len(dirs)}] {d.name[:32]:32s} "
                  f"P {row['p_after']:5d}  nnz {nnz:5d}  ||b|| {nb:8.3f}  "
                  f"max|b| {mx:7.3f}  R2 {r2:6.3f}  ({row['secs']:.0f}s)", flush=True)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  [건너뜀] {d.name}: {type(exc).__name__}: {exc}", flush=True)

    if not rows:
        return

    print()
    print("=" * 112)
    print(f"{'dataset':32s}{'N_tr':>7}{'P':>7}{'nnz':>7}{'lam_ratio':>11}"
          f"{'||beta||':>10}{'max|b_j|':>10}{'clip6':>7}{'R(beta)':>11}{'test R2':>9}")
    print("-" * 112)
    for r in sorted(rows, key=lambda z: -z["norm"]):
        print(f"{r['label'][:32]:32s}{r['n_tr']:7d}{r['p_after']:7d}{r['nnz']:7d}"
              f"{r['ratio']:11.2e}{r['norm']:10.3f}{r['maxabs']:10.3f}"
              f"{r['clipped']:7d}{r['R']:11.2e}{r['r2']:9.3f}")

    nb = np.array([r["norm"] for r in rows])
    mx = np.array([r["maxabs"] for r in rows])
    r2 = np.array([r["r2"] for r in rows])
    k = np.array([r["nnz"] for r in rows], dtype=float)
    print("-" * 112)
    print(f"  ||beta||  최소 {nb.min():.3f}  중앙 {np.median(nb):.3f}  최대 {nb.max():.3f}")
    print(f"  max|b_j|  최소 {mx.min():.3f}  중앙 {np.median(mx):.3f}  최대 {mx.max():.3f}")
    print(f"  nnz       최소 {k.min():.0f}  중앙 {np.median(k):.0f}  최대 {k.max():.0f}")
    print(f"  test R2   최소 {r2.min():.3f}  중앙 {np.median(r2):.3f}  최대 {r2.max():.3f}")
    print()
    print("  [비교] CV LASSO 12개 : ||beta|| 중앙 0.499, max|b_j| 중앙 0.206, nnz 15~570")
    print("  [비교] V14 32 태스크 : ||beta|| 0.854~1.144, max|b_j| 최대 1.0000")

    Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\n  저장: {args.out}")


if __name__ == "__main__":
    main()
