#!/usr/bin/env python3
"""셀렉터가 태스크마다 어떤 Direct96 페널티를 고르는지 직접 들여다본다.

핵심 수치
    d0     = rho'(0)          zero-KKT 임계를 직접 곱으로 정한다
    kappa  = rho'(0)/max rho' 작으면 lambda 를 키워도 0 을 못 만든다
    ell    = log threshold multiplier   음수면 임계를 추가로 줄인다

    tau_j = lambda * d0 / g_j * exp(ell_j)        (model.py 의 실제 식)

LASSO 는 rho' = 1 상수라 kappa = 1.  SCAD/MCP 는 감소해서 kappa = 1 (0에서 최대).
kappa < 1 이면 0 근처 기울기가 최대가 아니라는 뜻이고, 희소성을 만들 힘이 약하다.

    python3.11 penalty_shape_probe.py --data-dir real_regression_data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from infer_kalbi_13 import load_kalbi, load_split, preprocess


def extract_penalty(selector, query, verbose=False):
    """knots / rho / drho / d0 / ell 을 꺼낸다. 여러 키 이름을 시도한다."""
    head = selector.penalty_parameter_head
    knots = head.penalty_grid.detach().cpu().numpy().astype(np.float64)
    k = knots.size
    take = lambda t: t.detach().cpu().reshape(-1, k)[0].numpy().astype(np.float64)

    rho = drho = None
    for rk, dk in (("penalty_rho", "penalty_drho"), ("rho", "drho")):
        if rk in query and dk in query:
            rho, drho = take(query[rk]), take(query[dk])
            break
    if drho is None:
        latent = query.get("v24_corrected_penalty_latent")
        if latent is not None:
            enc = head.decode_direct_parameters(latent)
            if "drho" in enc:
                drho = take(enc["drho"])
                rho = take(enc["rho"]) if "rho" in enc else None
    if drho is None:
        if verbose:
            print("  [키 목록]", ", ".join(sorted(query)[:60]))
        raise KeyError("penalty drho 를 못 찾음")

    d0 = float(query["penalty_d0"].reshape(-1)[0])
    return knots, rho, drho, d0


@torch.no_grad()
def probe(model, selector, Z, y, device):
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

    # 학습된 임계 배수
    ell = None
    for key in aux:
        if "threshold_log_multiplier" in key:
            ell = aux[key].detach().cpu().float().reshape(-1).numpy()
            break
    return query, aux, encoded, ell


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="penalty_shape")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_num_threads(4)
    model, selector = load_kalbi(device)
    print()

    root = Path(args.data_dir)
    dirs = sorted(d for d in root.iterdir()
                  if d.is_dir() and not d.name.startswith("."))

    head = ("dataset".ljust(30) + "d0".rjust(10) + "max drho".rjust(10)
            + "kappa".rjust(8) + "drho(6)".rjust(10) + "ell 중앙".rjust(10)
            + "exp(ell)".rjust(10) + "유효임계".rjust(11))
    print(head)
    print("-" * 100)

    rows, curves = [], []
    for d in dirs:
        try:
            Ztr, ytr, Zte, yte = preprocess(*load_split(d), cache_dir=d)
            query, aux, encoded, ell = probe(model, selector, Ztr, ytr, device)
            knots, rho, drho, d0 = extract_penalty(selector, query,
                                                   verbose=not rows)
        except Exception as exc:                                   # noqa: BLE001
            print(f"{d.name[:30]:30s} 실패 {type(exc).__name__}: {exc}")
            continue

        mx = float(drho.max())
        kappa = d0 / max(mx, 1e-12)
        em = float(np.median(np.exp(ell))) if ell is not None else float("nan")
        eff = d0 * em                     # tau = lambda * (d0 * exp(ell)) / g_j
        rows.append(dict(label=d.name, d0=d0, max_drho=mx, kappa=kappa,
                         drho_at_6=float(drho[-1]),
                         ell_median=float(np.median(ell)) if ell is not None else float("nan"),
                         exp_ell=em, effective=eff,
                         lambda_max=float(encoded["lambda_max"].float().reshape(-1)[0])))
        curves.append((d.name, knots, drho / max(mx, 1e-12)))
        print(f"{d.name[:30]:30s}{d0:10.4f}{mx:10.4f}{kappa:8.3f}"
              f"{float(drho[-1]):10.4f}{rows[-1]['ell_median']:10.3f}"
              f"{em:10.3f}{eff:11.4f}", flush=True)

    if not rows:
        return
    Path(args.out + ".json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    g = lambda k: np.array([r[k] for r in rows], dtype=float)
    print("-" * 100)
    print(f"  d0 = rho'(0)      중앙 {np.median(g('d0')):.4f}   "
          f"최소 {g('d0').min():.4f}   최대 {g('d0').max():.4f}")
    print(f"  kappa             중앙 {np.median(g('kappa')):.3f}   "
          f"최소 {g('kappa').min():.3f}")
    print(f"  exp(ell)          중앙 {np.nanmedian(g('exp_ell')):.3f}")
    print(f"  유효임계 d0*exp(ell)  중앙 {np.nanmedian(g('effective')):.4f}")
    print()
    print("  해석")
    print("    LASSO 는 rho' 상수 1 -> d0 = 1, kappa = 1")
    print("    d0 << 1 이면 lambda_max 에서도 계수가 안 죽는다")
    print(f"    lambda_max 에서 0 이 되려면 대략  d0*exp(ell) >= 1  이 필요")
    bad = int((g("effective") < 1.0).sum())
    print(f"    d0*exp(ell) < 1 인 데이터셋 : {bad} / {len(rows)}")

    # ------------------------------------------------------------- 그림
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.subplots_adjust(left=.07, right=.98, top=.86, bottom=.15, wspace=.22)

    ax = axes[0]
    for name, knots, nd in curves:
        ax.plot(knots, nd, lw=1.3, alpha=.8, label=name[:18])
    ax.axhline(1.0, color="#6E7B88", ls="--", lw=1.2)
    ax.set_xlim(0, 6); ax.set_ylim(0, 1.15)
    ax.set_xlabel(r"$|\beta_j|$"); ax.set_ylabel(r"$\rho'(t)\,/\,\max\rho'$")
    ax.set_title("(1)  selected penalty derivative", loc="left", fontweight="bold")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=.2)

    ax = axes[1]
    for name, knots, nd in curves:
        ax.plot(knots, nd, lw=1.3, alpha=.8)
    ax.axhline(1.0, color="#6E7B88", ls="--", lw=1.2)
    ax.set_xlim(0, 0.6); ax.set_ylim(0, 1.15)
    ax.set_xlabel(r"$|\beta_j|$  (zoom near 0)")
    ax.set_ylabel(r"$\rho'(t)\,/\,\max\rho'$")
    ax.set_title(r"(2)  near zero — $\kappa=\rho'(0)/\max\rho'$",
                 loc="left", fontweight="bold")
    ax.grid(alpha=.2)

    fig.suptitle("what penalty does the selector choose?  (13 real benchmarks)",
                 fontsize=13, fontweight="bold", x=.07, ha="left", y=.965)
    fig.savefig(args.out + ".png", dpi=150)
    print(f"\n  저장: {args.out}.json , {args.out}.png")


if __name__ == "__main__":
    main()
