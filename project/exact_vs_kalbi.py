#!/usr/bin/env python3
"""KALBI 가 고른 (lambda, rho) 를 그대로 두고 목적함수를 정확히 풀어본다.

    KALBI 10-step beta_hat  ->  test R2 = a
    같은 (lambda,rho) 정확해 ->  test R2 = b

  b >> a  : 솔버 문제.  선택은 괜찮은데 10 스텝이 못 따라감
  b ~= a  : 선택 문제.  셀렉터가 고른 lambda/rho 자체가 나쁨

nnz 도 같이 보면 과선택이 미수렴 탓인지 페널티 탓인지 갈린다.

사용법:
    cd ~/Desktop/KALBI/project
    export KALBI_PROJECT=$HOME/Desktop/KALBI/project
    export KALBI_CKPT=$HOME/Desktop/KALBI/ckpt_selector_step030000.pt
    export PYTHONPATH=$HOME/Desktop/KALBI/project
    python3.11 exact_vs_kalbi.py --data-dir ./out_v14_probe --limit 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from infer_kalbi_13 import PKG, PROJECT, load_kalbi
from test_kalbi_on_synth import read_task, split_standardize, load_true_rayleigh

OUTCOME_LINEAR = 1


# --------------------------------------------------------------------------
# eval_kalbi_v2_common.py 의 순수 numpy 부분만 옮겨 왔다 (모델 의존성 제거).
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class QuadraticSegment:
    lo: float
    hi: float
    c: float
    d: float
    h: float


def segments_from_knots(knots, rho, drho, *, tail_hi: float = 20.0):
    knots = np.asarray(knots, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    drho = np.asarray(drho, dtype=np.float64)
    if not (knots.ndim == rho.ndim == drho.ndim == 1
            and len(knots) == len(rho) == len(drho)):
        raise ValueError("canonical knot payload shape mismatch")
    out = []
    for i in range(len(knots) - 1):
        lo, hi = float(knots[i]), float(knots[i + 1])
        h = float((drho[i + 1] - drho[i]) / (hi - lo))
        d = float(drho[i] - h * lo)
        c = float(rho[i] - d * lo - 0.5 * h * lo * lo)
        out.append(QuadraticSegment(
            lo=lo, hi=float(tail_hi) if i == len(knots) - 2 else hi,
            c=c, d=d, h=h))
    return out


def rho_abs(value: float, segments) -> float:
    m = abs(float(value))
    seg = segments[-1]
    for cand in segments:
        if cand.lo <= m <= cand.hi:
            seg = cand
            break
    return float(seg.c + seg.d * m + 0.5 * seg.h * m * m)


def _coordinate_minimum_vectorized(diagonal, center, lam, segments,
                                   segment_arrays, current):
    lo, hi, c, d, h = segment_arrays
    best_x = float(current)
    best_value = (0.5 * diagonal * (best_x - center) ** 2
                  + lam * rho_abs(best_x, segments))
    zero_value = 0.5 * diagonal * center * center + lam * rho_abs(0.0, segments)
    if zero_value < best_value:
        best_x, best_value = 0.0, zero_value

    denominator = diagonal + lam * h
    valid = denominator > 1e-12
    for sign in (-1.0, 1.0):
        stationary = lo.copy()
        stationary[valid] = (diagonal * sign * center - lam * d[valid]) / denominator[valid]
        stationary = np.minimum(np.maximum(stationary, lo), hi)
        magnitudes = np.stack((lo, hi, stationary), axis=1)
        values = sign * magnitudes
        seg = np.clip(np.searchsorted(hi, magnitudes, side="left"), 0, hi.size - 1)
        penalty = (c[seg] + d[seg] * magnitudes
                   + 0.5 * h[seg] * magnitudes * magnitudes)
        objectives = (0.5 * diagonal * (values - center) * (values - center)
                      + lam * penalty)
        idx = int(np.argmin(objectives))
        if float(objectives.flat[idx]) < best_value:
            best_value = float(objectives.flat[idx])
            best_x = float(values.flat[idx])
    return 0.0 if abs(best_x) <= 1e-8 else best_x


def coordinate_descent_path(gram, linear, lambdas, segments, initial,
                            *, sweeps: int, tolerance: float):
    diagonal = np.diag(gram).astype(np.float64).clip(1e-12)
    segment_arrays = tuple(
        np.asarray([getattr(s, f) for s in segments], dtype=np.float64)
        for f in ("lo", "hi", "c", "d", "h"))
    out = []
    for q, lam in enumerate(lambdas):
        beta = np.asarray(initial[q], dtype=np.float64).copy()
        for _ in range(int(sweeps)):
            previous = beta.copy()
            for j in range(beta.size):
                center = (linear[j] - gram[j].dot(beta)
                          + diagonal[j] * beta[j]) / diagonal[j]
                beta[j] = _coordinate_minimum_vectorized(
                    float(diagonal[j]), float(center), float(lam),
                    segments, segment_arrays, float(beta[j]))
            if float(np.max(np.abs(beta - previous))) < float(tolerance):
                break
        out.append(beta.copy())
    return np.asarray(out)


def extract_penalty(selector, query, printed: list):
    """셀렉터가 고른 페널티의 knots / rho / drho 를 꺼낸다."""
    head = selector.penalty_parameter_head
    knots = head.penalty_grid.detach().cpu().numpy().astype(np.float64)
    k = knots.size

    def take(t):
        return t.detach().cpu().reshape(-1, k)[0].numpy().astype(np.float64)

    for rk, dk in (("penalty_rho", "penalty_drho"), ("rho", "drho")):
        if rk in query and dk in query:
            return knots, take(query[rk]), take(query[dk])

    latent = query.get("v24_corrected_penalty_latent")
    if latent is not None:
        enc = head.decode_direct_parameters(latent)
        if "rho" in enc and "drho" in enc:
            return knots, take(enc["rho"]), take(enc["drho"])

    if not printed:
        printed.append(1)
        print("  [키 목록] " + ", ".join(sorted(query)[:40]), flush=True)
    raise KeyError("penalty rho/drho 를 못 찾음")


@torch.no_grad()
def kalbi_query(model, selector, Z, y, device):
    from train_fpk import _build_X_y_tensors

    x, yv = _build_X_y_tensors([Z], [y], device=device, dtype=torch.float32)
    encoded = model.encode_dataset(model.precompute(x, yv))
    query = selector(x, yv, encoded["lambda_max"].float(),
                     solver_context=encoded, return_representation=True)
    aux = model.decode_queries(
        encoded, lambdas=query["lambdas"],
        lambda_max=encoded["lambda_max"].float()[:, None],
        penalty_shape_features=query["penalty_shape_features"],
        penalty_d0=query["penalty_d0"], return_aux=True)
    beta = aux["beta_q"][:, 0].float()[0].cpu().numpy().astype(np.float64)
    return beta[: Z.shape[1]], float(query["lambdas"].reshape(-1)[0]), query


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="exact_vs_kalbi.json")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--max-p", type=int, default=1300, help="CD 비용 상한")
    ap.add_argument("--sweeps", type=int, default=60)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=20260809)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device {device}   sweeps {args.sweeps}   max_p {args.max_p}")
    model, selector = load_kalbi(device)
    print()

    root = Path(args.data_dir)
    truth = load_true_rayleigh(root)
    ordered = truth.get("__ordered__") if isinstance(truth, dict) else None

    rows, seen, printed = [], 0, []
    for path in sorted(root.glob("batch_*.npz")):
        if len(rows) >= args.limit:
            break
        task = read_task(path)
        if task is None:
            continue
        entry = (ordered[seen] if ordered and seen < len(ordered)
                 else truth.get(path.name))
        seen += 1
        x, y, _ = task
        if x.shape[1] > args.max_p:
            continue

        t0 = time.perf_counter()
        try:
            Z, ytr, Zte, yte = split_standardize(x, y, args.seed)
            beta_k, lam, query = kalbi_query(model, selector, Z, ytr, device)
            knots, rho, drho = extract_penalty(selector, query, printed)
            segments = segments_from_knots(knots, rho, drho)

            Z64 = Z.astype(np.float64)
            n = Z64.shape[0]
            gram = Z64.T @ Z64 / n
            linear = Z64.T @ ytr.astype(np.float64) / n

            starts = np.stack([beta_k, np.zeros_like(beta_k)])
            cds = coordinate_descent_path(
                gram, linear, np.asarray([lam, lam]), segments, starts,
                sweeps=args.sweeps, tolerance=args.tol)

            def obj(b):
                r = ytr.astype(np.float64) - Z64 @ b
                pen = float(sum(rho_abs(v, segments) for v in b))
                return 0.5 * float(r @ r) / n + lam * pen

            beta_e = cds[int(np.argmin([obj(cds[0]), obj(cds[1])]))]

            ss = max(float(((yte - yte.mean()) ** 2).sum()), 1e-12)
            r2 = lambda b: 1.0 - float(((yte - Zte @ b) ** 2).sum()) / ss
            nz = lambda b: int((np.abs(b) > 1e-8).sum())

            rows.append(dict(
                batch=path.name, n_tr=int(n), p=int(Z.shape[1]),
                true_R=float(entry["linear_rayleigh_quotient"]) if entry else float("nan"),
                true_k=float(entry["linear_support_size"]) if entry else float("nan"),
                lam=lam,
                kalbi_r2=r2(beta_k), exact_r2=r2(beta_e),
                kalbi_nnz=nz(beta_k), exact_nnz=nz(beta_e),
                kalbi_norm=float(np.linalg.norm(beta_k)),
                exact_norm=float(np.linalg.norm(beta_e)),
                d0=float(drho[0]), max_drho=float(drho.max()),
                kappa=float(drho[0] / max(drho.max(), 1e-12)),
                secs=time.perf_counter() - t0))
            r = rows[-1]
            print(f"  [{len(rows):2d}] {path.name}  R={r['true_R']:.2f}  "
                  f"R2 {r['kalbi_r2']:+.3f} -> {r['exact_r2']:+.3f}   "
                  f"nnz {r['kalbi_nnz']} -> {r['exact_nnz']}   "
                  f"kappa {r['kappa']:.3f}  ({r['secs']:.0f}s)", flush=True)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  [건너뜀] {path.name}: {type(exc).__name__}: {exc}", flush=True)

    if not rows:
        print("결과 없음"); return
    Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                              encoding="utf-8")

    a = np.array([r["kalbi_r2"] for r in rows])
    b = np.array([r["exact_r2"] for r in rows])
    ka = np.array([r["kalbi_nnz"] for r in rows], float)
    kb = np.array([r["exact_nnz"] for r in rows], float)
    tk = np.array([r["true_k"] for r in rows], float)
    kp = np.array([r["kappa"] for r in rows])
    N = np.array([r["n_tr"] for r in rows], float)

    print()
    print("=" * 78)
    print(f"태스크 {len(rows)}개   (같은 lambda, 같은 penalty, 솔버만 교체)")
    print("=" * 78)
    print(f"  test R2   KALBI 10-step {np.median(a):+.3f}   "
          f"정확해 {np.median(b):+.3f}   차이 {np.median(b - a):+.3f}")
    print(f"  R2 개선된 태스크 : {int((b > a + 1e-4).sum())} / {len(rows)}")
    print()
    print(f"  nnz/N     KALBI {np.median(ka / N):.2f}   "
          f"정확해 {np.median(kb / N):.2f}   참값 {np.median(tk / N):.3f}")
    print(f"  nnz > N   KALBI {int((ka > N).sum())}   정확해 {int((kb > N).sum())}")
    print()
    print(f"  kappa = rho'(0)/max rho'   중앙 {np.median(kp):.3f}   "
          f"최소 {kp.min():.3f}   최대 {kp.max():.3f}")
    print("     kappa 가 작으면 lambda 를 키워도 0 을 못 만든다")
    print()
    gap = np.median(b - a)
    if gap > 0.05:
        print("  판정: 솔버 문제.  10 스텝이 부족하다")
    elif gap < 0.01:
        print("  판정: 선택 문제.  lambda/penalty 자체가 나쁘다")
    else:
        print("  판정: 애매.  표본을 늘리거나 sweeps 를 올려 재확인")


if __name__ == "__main__":
    main()
