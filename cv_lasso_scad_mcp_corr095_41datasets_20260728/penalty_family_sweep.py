#!/usr/bin/env python3
"""페널티 족 자체를 검증한다.  셀렉터도 잠재도 쓰지 않는다.

native196_from_derivative_knots 는 격자 위의 임의의 rho' 를 받아
shape_features 와 d0 를 만들어 준다.  그래서 LASSO / MCP / SCAD 를
해석적으로 구성해 model.decode_queries 에 직접 먹일 수 있다.

답하려는 질문:

    단조 비증가 족 안에 real 13개를 푸는 해가 존재하는가?

존재하는데 셀렉터가 못 찾는 거면  -> 재학습으로 해결된다
아예 존재하지 않으면              -> 솔버나 전처리가 문제다.  재학습 낭비

lambda 는 shape 마다 d0 가 다르므로 유효 임계를 맞춰 훑는다.

    lambda = r * lambda_max / d0        r 이 zero-KKT 비율

r = 1 이면 정의상 전부 0 이어야 한다.

    python3.11 penalty_family_sweep.py --data-dir real_regression_data
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from infer_kalbi_13 import load_kalbi, load_split, preprocess


# ------------------------------------------------------- 페널티 족 구성

def shape_library(knots: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """rho'(t) 를 격자 위에서 만든다.  전부 비증가."""
    t = knots
    out: list[tuple[str, np.ndarray]] = []

    out.append(("lasso", np.ones_like(t)))

    for gamma in (0.25, 0.5, 1.0, 2.0, 4.0):
        out.append((f"mcp{gamma:g}", np.maximum(1.0 - t / gamma, 0.0)))

    for a in (3.7, 6.0):
        d = np.ones_like(t)
        mid = (t > 1.0) & (t <= a)
        d[mid] = (a - t[mid]) / (a - 1.0)
        d[t > a] = 0.0
        out.append((f"scad{a:g}", d))

    # 거의 hard threshold.  현재 패치가 만들어내는 모양의 대표
    out.append(("spike0p1", np.maximum(1.0 - t / 0.1, 0.0)))

    trapz = getattr(np, "trapezoid", None) or np.trapz
    normed = []
    for name, d in out:
        d = np.clip(d, 1e-6, None)
        area = float(trapz(d, t))
        normed.append((name, (d / max(area, 1e-12)).astype(np.float64)))
    return normed


def encode_shapes(shapes, knots_t, cp_module):
    """(S, 196) shape_features 와 (S,) d0 를 만든다."""
    d = torch.as_tensor(np.stack([s for _, s in shapes]),
                        dtype=torch.float32, device=knots_t.device)
    enc = cp_module.native196_from_derivative_knots(d, knots_t)
    return enc["shape_features"], enc["d0"]


# ------------------------------------------------------------------ 경로

@torch.no_grad()
def sweep(model, Z, y, device, shapes, ratios, cp_module, chunk=64):
    from train_fpk import _build_X_y_tensors

    x, yv = _build_X_y_tensors([Z], [y], device=device, dtype=torch.float32)
    encoded = model.encode_dataset(model.precompute(x, yv))
    lam_max = encoded["lambda_max"].float().reshape(-1)[0]

    head_grid = cp_module.canonical_grid(device=device, dtype=torch.float32)
    feats, d0s = encode_shapes(shapes, head_grid, cp_module)      # (S,196), (S,)

    S, R = len(shapes), len(ratios)
    rt = torch.as_tensor(ratios, dtype=torch.float32, device=device)
    # lambda[s, r] = ratio_r * lambda_max / d0_s
    lam = (rt[None, :] * lam_max) / d0s[:, None].clamp_min(1e-12)
    lam_flat = lam.reshape(-1)                                     # (S*R,)
    feat_flat = feats[:, None, :].expand(S, R, feats.shape[-1]).reshape(S * R, -1)
    d0_flat = d0s[:, None].expand(S, R).reshape(-1)

    betas = []
    for start in range(0, S * R, chunk):
        stop = min(start + chunk, S * R)
        b = model.decode_queries(
            encoded,
            lambdas=lam_flat[start:stop][None],
            lambda_max=lam_max.expand(1, stop - start),
            penalty_shape_features=feat_flat[start:stop][None].contiguous(),
            penalty_d0=d0_flat[start:stop][None].contiguous())
        betas.append(b[0].float().cpu().numpy())
    B = np.concatenate(betas, axis=0)[:, : Z.shape[1]].astype(np.float64)
    return B.reshape(S, R, -1), float(lam_max)


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="penalty_family_sweep.json")
    ap.add_argument("--grid", type=int, default=40)
    ap.add_argument("--ratio-min", type=float, default=1e-3)
    ap.add_argument("--ratio-max", type=float, default=2.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_num_threads(4)
    model, _ = load_kalbi(device)

    import canonical_penalty_monotone as mono
    cp = mono._resolve()
    knots = cp.canonical_grid().numpy().astype(np.float64)
    shapes = shape_library(knots)
    ratios = np.geomspace(args.ratio_max, args.ratio_min, args.grid)
    print(f"\n  페널티 {len(shapes)}종 x lambda {len(ratios)}점 "
          f"= {len(shapes)*len(ratios)} 쿼리 / 데이터셋")
    print("  " + ", ".join(n for n, _ in shapes) + "\n")

    root = Path(args.data_dir)
    dirs = sorted(d for d in root.iterdir()
                  if d.is_dir() and not d.name.startswith("."))

    print(f"{'dataset':30s}{'N':>6}{'P':>6} | {'최적 shape':>10}{'r':>9}"
          f"{'nnz':>7}{'R2':>8} | {'lasso 최적':>11} | {'r=1 nnz':>9}{'초':>7}")
    print("-" * 112)

    rows = []
    for d in dirs:
        try:
            Ztr, ytr, Zte, yte = preprocess(*load_split(d), cache_dir=d)
        except Exception as exc:                                   # noqa: BLE001
            print(f"{d.name[:30]:30s} 로드 실패 {type(exc).__name__}")
            continue

        t0 = time.perf_counter()
        try:
            B, lam_max = sweep(model, Ztr, ytr, device, shapes, ratios, cp)
        except Exception as exc:                                   # noqa: BLE001
            print(f"{d.name[:30]:30s} 실패 {type(exc).__name__}: {exc}")
            continue

        ss = max(float(((yte - yte.mean()) ** 2).sum()), 1e-12)
        pred = np.einsum("srp,np->srn", B, Zte)
        r2 = 1.0 - ((yte[None, None] - pred) ** 2).sum(-1) / ss     # (S,R)
        nnz = (np.abs(B) > 1e-8).sum(-1)                            # (S,R)

        si, ri = np.unravel_index(int(np.argmax(r2)), r2.shape)
        li = [n for n, _ in shapes].index("lasso")
        lri = int(np.argmax(r2[li]))
        i1 = int(np.argmin(np.abs(ratios - 1.0)))

        row = dict(label=d.name, n_tr=int(Ztr.shape[0]), p=int(Ztr.shape[1]),
                   lambda_max=lam_max, shapes=[n for n, _ in shapes],
                   ratios=ratios.tolist(),
                   best_shape=shapes[si][0], best_ratio=float(ratios[ri]),
                   best_nnz=int(nnz[si, ri]), best_r2=float(r2[si, ri]),
                   lasso_ratio=float(ratios[lri]),
                   lasso_nnz=int(nnz[li, lri]), lasso_r2=float(r2[li, lri]),
                   nnz_at_r1=nnz[:, i1].tolist(),
                   r2_grid=r2.tolist(), nnz_grid=nnz.tolist(),
                   secs=time.perf_counter() - t0)
        rows.append(row)
        print(f"{d.name[:30]:30s}{row['n_tr']:6d}{row['p']:6d} | "
              f"{row['best_shape']:>10}{row['best_ratio']:9.1e}"
              f"{row['best_nnz']:7d}{row['best_r2']:8.3f} | "
              f"{row['lasso_r2']:11.3f} | "
              f"{int(np.max(nnz[:, i1])):9d}{row['secs']:7.0f}", flush=True)

    if not rows:
        return
    Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                              encoding="utf-8")

    g = lambda k: np.array([r[k] for r in rows], dtype=float)
    best, lasso = g("best_r2"), g("lasso_r2")
    print("-" * 112)
    print(f"  족 전체 최적 R2   중앙 {np.median(best):+.3f}   "
          f"음수 {int((best<0).sum())}/{len(rows)}")
    print(f"  LASSO 만          중앙 {np.median(lasso):+.3f}   "
          f"음수 {int((lasso<0).sum())}/{len(rows)}")
    from collections import Counter
    print(f"  최적 shape 분포   {dict(Counter(r['best_shape'] for r in rows))}")
    worst = int(max(max(r["nnz_at_r1"]) for r in rows))
    print(f"  r=1 에서의 최대 nnz  {worst}   (정의상 0 이어야 한다)")
    print()
    if (best < 0).sum() == 0:
        print("  판정: 족 안에 해가 존재한다.  셀렉터 재학습이 옳은 방향")
    else:
        print("  판정: 어떤 고전 페널티로도 안 풀린다.")
        print("        페널티가 아니라 솔버 / 전처리 / 데이터를 봐야 한다")
    print(f"\n  저장: {args.out}")


if __name__ == "__main__":
    main()
