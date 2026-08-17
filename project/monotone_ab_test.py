#!/usr/bin/env python3
"""단조 페널티 제약을 기존 체크포인트에 추론 시점에만 걸어 A/B 비교한다.

가중치는 전혀 안 바꾼다.  Direct96 잠재 -> rho 디코딩만 교체한다.
셀렉터가 이 제약 아래서 학습된 게 아니므로 최적은 아니지만,
kappa 와 d0 가 얼마나 올라가고 nnz / R2 가 어느 방향으로 움직이는지 본다.

    python3.11 monotone_ab_test.py --data-dir real_regression_data
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

from infer_kalbi_13 import load_kalbi, load_split, preprocess, kalbi_beta
from penalty_shape_probe import extract_penalty
import canonical_penalty_monotone as mono

BLUE, ACC, MUT = "#3A5A80", "#8C3A2E", "#6E7B88"


@torch.no_grad()
def evaluate(model, selector, Ztr, ytr, Zte, yte, device):
    from train_fpk import _build_X_y_tensors

    x, yv = _build_X_y_tensors([Ztr], [ytr], device=device, dtype=torch.float32)
    encoded = model.encode_dataset(model.precompute(x, yv))
    query = selector(x, yv, encoded["lambda_max"].float(),
                     solver_context=encoded, return_representation=True)
    knots, rho, drho, d0 = extract_penalty(selector, query)

    beta, lam, lam_max, _ = kalbi_beta(model, selector, Ztr, ytr, device)
    ss = max(float(((yte - yte.mean()) ** 2).sum()), 1e-12)
    nb = float(np.linalg.norm(beta))
    return dict(
        d0=d0, max_drho=float(drho.max()),
        kappa=d0 / max(float(drho.max()), 1e-12),
        drho_at_6=float(drho[-1]),
        nnz=int((np.abs(beta) > 1e-8).sum()),
        norm=nb, maxabs=float(np.abs(beta).max()),
        R=float(np.var(Ztr @ beta) / nb ** 2) if nb > 1e-12 else float("nan"),
        r2=1.0 - float(((yte - Zte @ beta) ** 2).sum()) / ss,
        lam_ratio=lam / max(lam_max, 1e-12),
    ), (knots, drho / max(float(drho.max()), 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="monotone_ab")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_num_threads(4)
    model, selector = load_kalbi(device)
    print()

    root = Path(args.data_dir)
    dirs = sorted(d for d in root.iterdir()
                  if d.is_dir() and not d.name.startswith("."))

    print("dataset".ljust(30) + "N".rjust(6) +
          "|  d0(현)   d0(단)  kappa현 kappa단 |"
          "  nnz현  nnz단 |   R2현    R2단   변화")
    print("-" * 118)

    rows, curves = [], []
    for d in dirs:
        try:
            Ztr, ytr, Zte, yte = preprocess(*load_split(d), cache_dir=d)
        except Exception as exc:                                   # noqa: BLE001
            print(f"{d.name[:30]:30s} 로드 실패 {type(exc).__name__}")
            continue

        try:
            mono.unpatch(verbose=False)
            base, cb = evaluate(model, selector, Ztr, ytr, Zte, yte, device)
            mono.patch(verbose=False)
            new, cn = evaluate(model, selector, Ztr, ytr, Zte, yte, device)
            mono.unpatch(verbose=False)
        except Exception as exc:                                   # noqa: BLE001
            mono.unpatch(verbose=False)
            print(f"{d.name[:30]:30s} 실패 {type(exc).__name__}: {exc}")
            continue

        rows.append(dict(label=d.name, n_tr=int(Ztr.shape[0]),
                         p=int(Ztr.shape[1]), base=base, mono=new))
        curves.append((d.name, cb, cn))
        print(f"{d.name[:30]:30s}{Ztr.shape[0]:6d}"
              f"|{base['d0']:8.3f}{new['d0']:9.3f}"
              f"{base['kappa']:8.3f}{new['kappa']:8.3f} |"
              f"{base['nnz']:7d}{new['nnz']:7d} |"
              f"{base['r2']:8.3f}{new['r2']:8.3f}"
              f"{new['r2'] - base['r2']:+8.3f}", flush=True)

    if not rows:
        return
    Path(args.out + ".json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    b = lambda k: np.array([r["base"][k] for r in rows], dtype=float)
    m = lambda k: np.array([r["mono"][k] for r in rows], dtype=float)
    N = np.array([r["n_tr"] for r in rows], dtype=float)

    print("-" * 118)
    print(f"  d0        현행 {np.median(b('d0')):.3f}  ->  단조 {np.median(m('d0')):.3f}")
    print(f"  kappa     현행 {np.median(b('kappa')):.3f}  ->  단조 {np.median(m('kappa')):.3f}")
    print(f"  rho'(6)   현행 {np.median(b('drho_at_6')):.4f}  ->  단조 {np.median(m('drho_at_6')):.4f}")
    print(f"  nnz/N     현행 {np.median(b('nnz')/N):.2f}  ->  단조 {np.median(m('nnz')/N):.2f}")
    print(f"  nnz > N   현행 {int((b('nnz')>N).sum())}/{len(rows)}  ->  "
          f"단조 {int((m('nnz')>N).sum())}/{len(rows)}")
    print(f"  ||beta||  현행 {np.median(b('norm')):.3f}  ->  단조 {np.median(m('norm')):.3f}")
    print(f"  test R2   현행 {np.median(b('r2')):+.3f}  ->  단조 {np.median(m('r2')):+.3f}")
    print(f"  R2 < 0    현행 {int((b('r2')<0).sum())}/{len(rows)}  ->  "
          f"단조 {int((m('r2')<0).sum())}/{len(rows)}")
    print()
    print("  주의: 셀렉터는 이 제약 아래서 학습된 적이 없다.")
    print("        kappa 와 d0 가 오르는지가 핵심이고, R2 는 재학습 후에 봐야 한다.")

    # ------------------------------------------------------------- 그림
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.subplots_adjust(left=.07, right=.98, top=.86, bottom=.15, wspace=.22)

    for ax, idx, title in ((axes[0], 1, "(1)  current — free hinge weights"),
                           (axes[1], 2, "(2)  patched — weights forced negative")):
        for item in curves:
            knots, nd = item[idx]
            ax.plot(knots, nd, lw=1.2, alpha=.8)
        ax.axhline(1.0, color=MUT, ls="--", lw=1.2)
        ax.set_xlim(0, 1.5); ax.set_ylim(0, 1.15)
        ax.set_xlabel(r"$|\beta_j|$")
        ax.set_ylabel(r"$\rho'(t)/\max\rho'$")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(alpha=.2)

    fig.suptitle("monotone constraint on the Direct96 penalty  (same checkpoint)",
                 fontsize=13, fontweight="bold", x=.07, ha="left", y=.965)
    fig.savefig(args.out + ".png", dpi=150)
    print(f"\n  저장: {args.out}.json , {args.out}.png")


if __name__ == "__main__":
    main()
