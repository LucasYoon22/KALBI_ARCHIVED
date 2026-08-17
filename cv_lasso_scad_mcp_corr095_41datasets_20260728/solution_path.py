#!/usr/bin/env python3
"""KALBI V6 + 셀렉터의 solution path 를 뽑고 CV LASSO 경로와 겹쳐 그린다.

셀렉터가 고른 페널티 rho 는 고정한 채 lambda 만 격자로 훑는다.
encode_dataset 은 한 번만 하고 decode_queries 에 lambda 벡터를 넣으므로
경로 전체가 한 번의 순전파로 나온다.

    python3.11 solution_path.py --real real_regression_data/QSAR-TID-10331
    python3.11 solution_path.py --npz out_v14_probe/batch_000001.npz
    python3.11 solution_path.py --real ... --out ~/Desktop/KALBI/path_TID10331

환경변수:
    KALBI_PROJECT, KALBI_CKPT  (infer_kalbi_13.py 와 동일)
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

from infer_kalbi_13 import load_kalbi, corr_filter, preprocess, load_split

BLUE, ACC, MUT, GREEN = "#3A5A80", "#8C3A2E", "#6E7B88", "#1D9E75"


# ------------------------------------------------------------------ 입력

def load_real(path: Path):
    Xtr, ytr, Xte, yte = load_split(path)
    return preprocess(Xtr, ytr, Xte, yte, cache_dir=path)


def load_npz(path: Path, seed: int):
    from test_kalbi_on_synth import read_task, split_standardize

    task = read_task(path)
    if task is None:
        raise ValueError("선형 태스크가 아닙니다 (outcome_source_id != 1)")
    x, y, _ = task
    return split_standardize(x, y, seed)


# ------------------------------------------------------------------ 경로

@torch.no_grad()
def kalbi_path(model, selector, Z, y, device, ratios, chunk=48):
    """셀렉터의 페널티를 고정하고 lambda 격자를 훑는다."""
    from train_fpk import _build_X_y_tensors

    x, yv = _build_X_y_tensors([Z], [y], device=device, dtype=torch.float32)
    encoded = model.encode_dataset(model.precompute(x, yv))
    lam_max = encoded["lambda_max"].float().reshape(-1)[0]

    query = selector(x, yv, encoded["lambda_max"].float(),
                     solver_context=encoded, return_representation=True)
    shape0 = query["penalty_shape_features"].reshape(1, -1)   # (1, F)
    d0_0 = query["penalty_d0"].reshape(1)                     # (1,)
    lam_sel = float(query["lambdas"].reshape(-1)[0])

    lambdas = torch.as_tensor(ratios, dtype=torch.float32,
                              device=device) * lam_max
    q = lambdas.numel()
    shape = shape0.expand(q, -1).contiguous()
    d0 = d0_0.expand(q).contiguous()

    betas = []
    for start in range(0, q, chunk):
        stop = min(start + chunk, q)
        beta = model.decode_queries(
            encoded,
            lambdas=lambdas[start:stop][None],
            lambda_max=lam_max.expand(1, stop - start),
            penalty_shape_features=shape[start:stop][None],
            penalty_d0=d0[start:stop][None])
        betas.append(beta[0].float().cpu().numpy())
    return (np.concatenate(betas, axis=0)[:, : Z.shape[1]].astype(np.float64),
            float(lam_max), lam_sel)


def lasso_path_ref(Z, y, ratios):
    from sklearn.linear_model import lasso_path

    n = Z.shape[0]
    lmax = float(np.abs(Z.T @ y).max() / n)
    _, coefs, _ = lasso_path(Z.astype(np.float64), y.astype(np.float64),
                             alphas=np.asarray(ratios) * lmax,
                             tol=1e-4, max_iter=5000)
    return coefs.T.astype(np.float64), lmax          # (Q, P)


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--real", help="real_regression_data/<name> 디렉터리")
    src.add_argument("--npz", help="합성 batch_XXXXXX.npz")
    ap.add_argument("--out", default="solution_path")
    ap.add_argument("--grid", type=int, default=96)
    ap.add_argument("--ratio-min", type=float, default=1e-4)
    ap.add_argument("--top", type=int, default=25, help="굵게 그릴 좌표 수")
    ap.add_argument("--seed", type=int, default=20260809)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    label = Path(args.real or args.npz).name
    Ztr, ytr, Zte, yte = (load_real(Path(args.real)) if args.real
                          else load_npz(Path(args.npz), args.seed))
    print(f"{label}   N_train={Ztr.shape[0]}  P={Ztr.shape[1]}")

    device = torch.device(args.device)
    model, selector = load_kalbi(device)
    ratios = np.geomspace(1.0, args.ratio_min, args.grid)

    print("KALBI 경로 ...", flush=True)
    kb, lam_max_k, lam_sel = kalbi_path(model, selector, Ztr, ytr, device, ratios)
    print("LASSO 경로 ...", flush=True)
    lb, lam_max_l = lasso_path_ref(Ztr, ytr, ratios)

    ss = max(float(((yte - yte.mean()) ** 2).sum()), 1e-12)
    r2 = lambda B: 1.0 - ((yte[None] - B @ Zte.T) ** 2).sum(1) / ss
    r2k, r2l = r2(kb), r2(lb)
    nzk = (np.abs(kb) > 1e-8).sum(1)
    nzl = (np.abs(lb) > 1e-8).sum(1)
    sel_ratio = lam_sel / max(lam_max_k, 1e-12)

    rows = dict(label=label, n_tr=int(Ztr.shape[0]), p=int(Ztr.shape[1]),
                lambda_max_kalbi=lam_max_k, lambda_max_lasso=lam_max_l,
                selector_ratio=sel_ratio,
                ratios=ratios.tolist(),
                kalbi_r2=r2k.tolist(), lasso_r2=r2l.tolist(),
                kalbi_nnz=nzk.tolist(), lasso_nnz=nzl.tolist())
    Path(args.out + ".json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(f"  lambda_max   KALBI {lam_max_k:.4g}   LASSO {lam_max_l:.4g}")
    print(f"  셀렉터가 고른 비율 r = {sel_ratio:.3e}")
    print(f"  그 지점  KALBI nnz {nzk[np.argmin(np.abs(ratios - sel_ratio))]}"
          f"   test R2 {r2k[np.argmin(np.abs(ratios - sel_ratio))]:+.3f}")
    print(f"  경로 최적  KALBI R2 {r2k.max():+.3f} (r={ratios[r2k.argmax()]:.2e})"
          f"   LASSO R2 {r2l.max():+.3f} (r={ratios[r2l.argmax()]:.2e})")
    print(f"  놓친 폭    {r2k.max() - r2k[np.argmin(np.abs(ratios - sel_ratio))]:+.3f}"
          "   <- 경로에 더 좋은 지점이 있었는데 셀렉터가 못 찾은 몫")

    # ------------------------------------------------------------- 그림
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4))
    fig.subplots_adjust(left=.075, right=.98, top=.90, bottom=.08,
                        wspace=.2, hspace=.32)

    order = np.argsort(-np.abs(lb).max(0))[: args.top]
    for ax, B, name in ((axes[0, 0], kb, "KALBI V6 + selector"),
                        (axes[0, 1], lb, "CV LASSO")):
        ax.axhline(0, color=MUT, lw=.8)
        ax.plot(ratios, B, color="#C9CDD2", lw=.5, alpha=.5)
        ax.plot(ratios, B[:, order], lw=1.3)
        ax.axvline(sel_ratio, color=ACC, ls="--", lw=1.5)
        ax.set_xscale("log"); ax.invert_xaxis()
        ax.set_xlabel(r"$\lambda / \lambda_{\max}$")
        ax.set_ylabel(r"$\hat\beta_j$")
        ax.set_title(f"{name}", loc="left", fontweight="bold")
        ax.grid(alpha=.2)
    axes[0, 0].text(sel_ratio, axes[0, 0].get_ylim()[1] * .92,
                    " selector", color=ACC, fontsize=8, va="top")

    ax = axes[1, 0]
    ax.plot(ratios, r2k, color=BLUE, lw=2, label="KALBI")
    ax.plot(ratios, r2l, color=GREEN, lw=2, ls="--", label="CV LASSO")
    ax.axvline(sel_ratio, color=ACC, ls="--", lw=1.5)
    ax.axhline(0, color=MUT, lw=.8)
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_ylim(max(-1.0, min(r2k.min(), r2l.min()) - .1), 1.02)
    ax.set_xlabel(r"$\lambda / \lambda_{\max}$"); ax.set_ylabel("test $R^2$")
    ax.set_title("test $R^2$ along the path", loc="left", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=.2)

    ax = axes[1, 1]
    ax.plot(ratios, np.maximum(nzk, .5), color=BLUE, lw=2, label="KALBI")
    ax.plot(ratios, np.maximum(nzl, .5), color=GREEN, lw=2, ls="--",
            label="CV LASSO")
    ax.axhline(Ztr.shape[0], color=MUT, ls=":", lw=1.4, label="N train")
    ax.axvline(sel_ratio, color=ACC, ls="--", lw=1.5)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel(r"$\lambda / \lambda_{\max}$"); ax.set_ylabel("nnz")
    ax.set_title("sparsity along the path", loc="left", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=.2)

    fig.suptitle(f"solution path — {label}   "
                 f"(N={Ztr.shape[0]}, P={Ztr.shape[1]})",
                 fontsize=13, fontweight="bold", x=.075, ha="left", y=.965)
    fig.savefig(args.out + ".png", dpi=150)
    print(f"\n  저장: {args.out}.png , {args.out}.json")


if __name__ == "__main__":
    main()
