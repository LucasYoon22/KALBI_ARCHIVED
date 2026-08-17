#!/usr/bin/env python3
"""KALBI 추론 결과를 CV LASSO 정답과 대조해 4가지 가설을 한 번에 검증한다.

    1. dense 여부        : KALBI nnz vs lasso_nnz_1e8
    2. 바운드 포화       : |beta_j| >= 5.9 인 계수 개수
    3. penalty collapse  : rho, rho', rho'(0)/max rho', TV(rho') 분포
    4. lambda 오차 방향  : 예측 r vs lasso_selected_lambda_ratio

입력 규약
---------
--pred-dir 안에 태스크별 NPZ가 있어야 한다. 파일명은 bundle의 task_id와 동일:

    <task_id>.npz

필수 키
    beta_std      (P_after_corr095,)  표준화 공간 계수.
                                      bundle의 lasso_beta와 같은 순서
                                      (retained_original_indices 순)
    lambda_ratio  scalar              V20이 낸 effective ratio r

선택 키 (있으면 해당 검사 수행)
    rho_grid      (96,)   Direct96 t 좌표
    rho           (96,)   rho(t)
    rho_prime     (96,)   rho'(t)
    lambda_hat    scalar  실제 사용된 lambda

사용법
------
    python3 check_kalbi_vs_cv.py --pred-dir /path/to/kalbi_preds
    python3 check_kalbi_vs_cv.py --pred-dir ... --plot   # matplotlib 있으면 그림 저장
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics as st

import numpy as np

BETA_BOUND = 6.0
SATURATION_THRESHOLD = 5.9
NNZ_TOL = 1e-8


# ---------------------------------------------------------------- 데이터 로딩

def load_reference(bundle_dir: str) -> dict[str, dict]:
    """bundle에서 CV LASSO 정답을 읽는다."""
    wide = os.path.join(bundle_dir, "results", "cv_lasso_scad_mcp_41_wide.csv")
    npz_dir = os.path.join(bundle_dir, "results", "coefficient_npz")
    ref = {}
    for row in csv.DictReader(open(wide)):
        tid = row["task_id"]
        f = os.path.join(npz_dir, tid + ".npz")
        entry = {
            "task_id": tid,
            "N_train": int(row["outer_train_rows"]),
            "P": int(row["P_after_corr095"]),
            "max_corr": float(row["retained_max_abs_correlation"]),
            "cv_ratio": float(row["lasso_selected_lambda_ratio"]),
            "lambda_max": float(row["lasso_outer_train_lambda_max"]),
            "cv_nnz": int(float(row["lasso_nnz_1e8"])),
            "cv_r2": float(row["lasso_test_r2"]),
        }
        if os.path.exists(f):
            b = np.load(f)["lasso_beta"].astype(np.float64)
            entry["cv_beta"] = b
            entry["cv_max_abs_beta"] = float(np.abs(b).max())
            entry["cv_l1"] = float(np.abs(b).sum())
        ref[tid] = entry
    return ref


def load_predictions(pred_dir: str) -> dict[str, dict]:
    preds = {}
    for name in sorted(os.listdir(pred_dir)):
        if not name.endswith(".npz"):
            continue
        d = np.load(os.path.join(pred_dir, name), allow_pickle=True)
        rec = {}
        if "beta_std" not in d.files:
            print(f"  [건너뜀] {name}: beta_std 키 없음 (있는 키: {list(d.files)})")
            continue
        rec["beta"] = np.asarray(d["beta_std"], dtype=np.float64).reshape(-1)
        for key, dst in [("lambda_ratio", "ratio"), ("lambda_hat", "lambda_hat")]:
            if key in d.files:
                rec[dst] = float(np.asarray(d[key]).reshape(-1)[0])
        for key in ("rho_grid", "rho", "rho_prime"):
            if key in d.files:
                rec[key] = np.asarray(d[key], dtype=np.float64).reshape(-1)
        preds[name[:-4]] = rec
    return preds


# ---------------------------------------------------------------- 검사 1 & 2

def check_density_and_bound(ref, preds) -> list[dict]:
    rows = []
    for tid, p in sorted(preds.items()):
        if tid not in ref:
            print(f"  [경고] bundle에 없는 task_id: {tid}")
            continue
        r = ref[tid]
        b = p["beta"]
        if b.size != r["P"]:
            print(f"  [경고] {tid}: 계수 길이 {b.size} != P {r['P']}"
                  f" — 순서/필터가 다를 수 있음")
        a = np.abs(b)
        rows.append({
            "task_id": tid,
            "N": r["N_train"],
            "P": r["P"],
            "kalbi_nnz": int((a > NNZ_TOL).sum()),
            "cv_nnz": r["cv_nnz"],
            "kalbi_max_abs": float(a.max()) if a.size else 0.0,
            "cv_max_abs": r.get("cv_max_abs_beta", float("nan")),
            "kalbi_l1": float(a.sum()),
            "cv_l1": r.get("cv_l1", float("nan")),
            "saturated": int((a >= SATURATION_THRESHOLD).sum()),
            "cv_over_bound": int((np.abs(r["cv_beta"]) > BETA_BOUND).sum())
            if "cv_beta" in r else -1,
            "max_corr": r["max_corr"],
        })
    return rows


def report_density_and_bound(rows) -> None:
    print("\n" + "=" * 100)
    print("검사 1 & 2 — dense 여부와 |beta| <= 6 바운드 포화")
    print("=" * 100)
    print(f"{'task_id':44s} {'N':>5} {'P':>5} "
          f"{'nnz':>10} {'max|b|':>16} {'sat':>4} {'CV>6':>5}")
    print(f"{'':44s} {'':>5} {'':>5} {'KALBI/CV':>10} {'KALBI/CV':>16}")
    print("-" * 100)
    for r in rows:
        print(f"{r['task_id'][:44]:44s} {r['N']:5d} {r['P']:5d} "
              f"{r['kalbi_nnz']:4d}/{r['cv_nnz']:<5d} "
              f"{r['kalbi_max_abs']:7.2f}/{r['cv_max_abs']:<8.2f} "
              f"{r['saturated']:4d} {r['cv_over_bound']:5d}")

    kn = np.array([r["kalbi_nnz"] for r in rows], dtype=float)
    cn = np.array([r["cv_nnz"] for r in rows], dtype=float)
    sat = np.array([r["saturated"] for r in rows])
    over = np.array([r["cv_over_bound"] for r in rows])

    ratio = kn / np.maximum(cn, 1)
    print("\n[검사 1 판정] KALBI nnz / CV nnz")
    print(f"  중앙값 {np.median(ratio):.2f}배,  KALBI가 더 dense한 태스크"
          f" {int((kn > cn).sum())}/{len(rows)}")
    if np.median(ratio) > 1.5:
        print("  => 문서 한계 #6 예측대로 KALBI 해가 CV LASSO보다 뚜렷하게 dense함. 가설 지지.")
    elif np.median(ratio) < 0.7:
        print("  => 오히려 더 sparse함. 한계 #6 가설은 기각.")
    else:
        print("  => 뚜렷한 차이 없음. 한계 #6는 주 원인이 아닐 가능성.")

    print("\n[검사 2 판정] |beta_j| >= 5.9 포화")
    print(f"  포화 계수가 있는 태스크 {int((sat > 0).sum())}/{len(rows)},"
          f" 총 포화 계수 {int(sat.sum())}개")
    print(f"  참고: CV 정답이 |beta|>6를 요구하는 태스크"
          f" {int((over > 0).sum())}/{len(rows)}")
    if (sat > 0).sum() > 0:
        print("  => 바운드가 실제로 걸리고 있음. 가설 지지.")
    elif (over > 0).sum() > 0:
        print("  => 바운드에 붙지도 못함. 계수가 작은 원인이 바운드가 아니라"
              " lambda/penalty 쪽일 가능성이 큼.")
    else:
        print("  => 바운드 무관.")


# ---------------------------------------------------------------- 검사 3

def check_penalty(preds) -> list[dict] | None:
    have = [t for t, p in preds.items() if "rho_prime" in p]
    if not have:
        print("\n[검사 3] rho_prime 키가 없어 건너뜀.")
        return None

    rows = []
    for tid in sorted(have):
        p = preds[tid]
        dp = p["rho_prime"]
        grid = p.get("rho_grid")
        if grid is None:
            grid = np.linspace(0.0, 6.0, dp.size)
        d0 = float(dp[0])
        dmax = float(dp.max())
        tv = float(np.abs(np.diff(dp)).sum())
        # 단조감소면 LASSO/SCAD/MCP-like, 단조증가면 ridge-like
        diffs = np.diff(dp)
        dec = float((diffs < 0).mean())
        rows.append({
            "task_id": tid,
            "rho_prime_0": d0,
            "max_rho_prime": dmax,
            "d0_over_max": d0 / dmax if dmax > 0 else float("nan"),
            "tv": tv,
            "frac_decreasing": dec,
            "grid": grid,
            "rho_prime": dp,
            "rho": p.get("rho"),
        })

    print("\n" + "=" * 100)
    print("검사 3 — penalty collapse 및 형태")
    print("=" * 100)
    print(f"{'task_id':44s} {'rho\'(0)':>10} {'max rho\'':>10} "
          f"{'d0/max':>8} {'TV':>10} {'감소비율':>8}")
    print("-" * 100)
    for r in rows:
        print(f"{r['task_id'][:44]:44s} {r['rho_prime_0']:10.4f} "
              f"{r['max_rho_prime']:10.4f} {r['d0_over_max']:8.4f} "
              f"{r['tv']:10.4f} {r['frac_decreasing']:8.2f}")

    stack = np.stack([r["rho_prime"] for r in rows])
    # 태스크 간 형태가 얼마나 같은가: 좌표별 std / 좌표별 mean
    coord_mean = stack.mean(axis=0)
    coord_std = stack.std(axis=0)
    rel = float(np.median(coord_std / np.maximum(np.abs(coord_mean), 1e-12)))
    d0max = np.array([r["d0_over_max"] for r in rows])
    dec = np.array([r["frac_decreasing"] for r in rows])

    print("\n[검사 3 판정]")
    print(f"  태스크 간 rho' 상대 변동(중앙값 std/mean): {rel:.4f}")
    if rel < 0.05:
        print("  => penalty가 태스크와 거의 무관하게 동일. COLLAPSE 확정.")
    elif rel < 0.20:
        print("  => 변동이 작음. 부분적 collapse 의심.")
    else:
        print("  => 태스크마다 충분히 다름. collapse 아님.")

    print(f"  d0/max rho' 중앙값: {np.median(d0max):.4f}")
    if np.median(d0max) < 0.2:
        print("  => 한계 #6대로 zero-KKT 도달이 어려움 (r=1에서도 0으로 못 보냄).")
    else:
        print("  => zero-KKT 스케일은 정상 범위.")

    print(f"  rho' 감소 구간 비율 중앙값: {np.median(dec):.2f}")
    if np.median(dec) < 0.4:
        print("  => rho'가 주로 증가. RIDGE-LIKE. 모든 계수를 조금씩 줄이는 형태.")
    else:
        print("  => rho'가 주로 감소. LASSO/SCAD/MCP-like.")
    return rows


# ---------------------------------------------------------------- 검사 4

def check_lambda(ref, preds) -> list[dict] | None:
    have = [t for t, p in preds.items() if "ratio" in p and t in ref]
    if not have:
        print("\n[검사 4] lambda_ratio 키가 없어 건너뜀.")
        return None

    rows = []
    for tid in sorted(have):
        pr = preds[tid]["ratio"]
        cv = ref[tid]["cv_ratio"]
        rows.append({
            "task_id": tid,
            "pred_ratio": pr,
            "cv_ratio": cv,
            "log10_err": math.log10(max(pr, 1e-12)) - math.log10(max(cv, 1e-12)),
            "P": ref[tid]["P"],
            "cv_r2": ref[tid]["cv_r2"],
        })

    print("\n" + "=" * 100)
    print("검사 4 — lambda ratio 오차 방향")
    print("=" * 100)
    print(f"{'task_id':44s} {'예측 r':>12} {'CV r':>12} {'log10차':>9} {'P':>6}")
    print("-" * 100)
    for r in sorted(rows, key=lambda z: -z["log10_err"]):
        print(f"{r['task_id'][:44]:44s} {r['pred_ratio']:12.6f} "
              f"{r['cv_ratio']:12.6f} {r['log10_err']:+9.2f} {r['P']:6d}")

    e = np.array([r["log10_err"] for r in rows])
    n_over = int((e > 0).sum())
    print("\n[검사 4 판정]")
    print(f"  log10(예측/CV) 중앙값 {np.median(e):+.3f}"
          f"  (10^{np.median(e):+.2f} = {10**np.median(e):.2f}배)")
    print(f"  예측이 더 큰 태스크 {n_over}/{len(rows)}")
    # 부호검정 근사 p값
    k, n = n_over, len(rows)
    p_two = min(1.0, 2 * sum(math.comb(n, i) for i in range(max(k, n - k), n + 1)) / 2**n)
    print(f"  부호검정 양측 p ≈ {p_two:.4f}")
    if np.median(e) > 0.3 and p_two < 0.05:
        print("  => lambda 과대예측이 통계적으로 확인됨. 과다 shrinkage 가설 지지.")
    elif np.median(e) < -0.3 and p_two < 0.05:
        print("  => 오히려 과소예측. 계수가 작은 원인은 lambda가 아님.")
    else:
        print("  => 뚜렷한 방향 편향 없음. 원인이 lambda 크기가 아니라"
              " penalty 형태나 바운드일 가능성.")

    lowp = [r for r in rows if r["P"] < 50]
    if lowp:
        el = np.array([r["log10_err"] for r in lowp])
        print(f"\n  P<50 서브셋({len(lowp)}개, 생성기가 못 만드는 영역)"
              f" log10 오차 중앙값 {np.median(el):+.3f}")
        print("  => 나머지보다 크면 생성기 커버리지 공백이 실제로 영향을 준 증거.")
    return rows


# ---------------------------------------------------------------- 그림

def make_plots(pen_rows, lam_rows, den_rows, out_dir) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[그림] matplotlib이 없어 건너뜀."
              "  설치: python3 -m pip install matplotlib")
        return

    if pen_rows:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for r in pen_rows:
            ax[0].plot(r["grid"], r["rho_prime"], lw=0.8, alpha=0.6)
            if r["rho"] is not None:
                ax[1].plot(r["grid"], r["rho"], lw=0.8, alpha=0.6)
        ax[0].set_xlabel("t"); ax[0].set_ylabel(r"$\rho'(t)$")
        ax[0].set_title("penalty derivative (41 tasks overlaid)")
        ax[0].set_xscale("symlog", linthresh=0.1)
        ax[1].set_xlabel("t"); ax[1].set_ylabel(r"$\rho(t)$")
        ax[1].set_title("penalty")
        fig.tight_layout()
        p = os.path.join(out_dir, "penalty_overlay.png")
        fig.savefig(p, dpi=140); plt.close(fig)
        print(f"[그림] 저장 {p}")

    if lam_rows:
        fig, ax = plt.subplots(figsize=(5.2, 5))
        x = [r["cv_ratio"] for r in lam_rows]
        y = [r["pred_ratio"] for r in lam_rows]
        c = ["tab:red" if r["P"] < 50 else "tab:blue" for r in lam_rows]
        ax.scatter(x, y, c=c, s=28, alpha=0.8)
        lo = min(min(x), min(y)) * 0.5
        hi = max(max(x), max(y)) * 2
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("CV LASSO selected ratio")
        ax.set_ylabel("V20 predicted ratio")
        ax.set_title("lambda ratio (red = P<50, 생성기 미커버)")
        fig.tight_layout()
        p = os.path.join(out_dir, "lambda_scatter.png")
        fig.savefig(p, dpi=140); plt.close(fig)
        print(f"[그림] 저장 {p}")

    if den_rows:
        fig, ax = plt.subplots(figsize=(5.2, 5))
        x = [r["cv_nnz"] for r in den_rows]
        y = [r["kalbi_nnz"] for r in den_rows]
        ax.scatter(x, y, s=28, alpha=0.8)
        hi = max(max(x), max(y)) * 1.1
        ax.plot([0, hi], [0, hi], "k--", lw=1)
        ax.set_xlabel("CV LASSO nnz"); ax.set_ylabel("KALBI nnz")
        ax.set_title("sparsity 비교")
        fig.tight_layout()
        p = os.path.join(out_dir, "nnz_scatter.png")
        fig.savefig(p, dpi=140); plt.close(fig)
        print(f"[그림] 저장 {p}")


# ---------------------------------------------------------------- main

def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True,
                    help="KALBI 추론 결과 NPZ 폴더 (<task_id>.npz)")
    ap.add_argument("--bundle-dir", default=here)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    out_dir = args.out_dir or args.pred_dir

    print("bundle 정답 로드 중...")
    ref = load_reference(args.bundle_dir)
    print(f"  {len(ref)}개 태스크")

    print(f"예측 로드 중: {args.pred_dir}")
    preds = load_predictions(args.pred_dir)
    print(f"  {len(preds)}개 예측")
    if not preds:
        print("예측이 없습니다. --pred-dir와 NPZ 키 이름을 확인하세요.")
        return

    den = check_density_and_bound(ref, preds)
    report_density_and_bound(den)
    pen = check_penalty(preds)
    lam = check_lambda(ref, preds)

    summary = {
        "n_tasks": len(den),
        "median_nnz_ratio": float(np.median(
            [r["kalbi_nnz"] / max(r["cv_nnz"], 1) for r in den])),
        "tasks_with_saturated_beta": int(sum(1 for r in den if r["saturated"] > 0)),
        "tasks_cv_needs_beta_over_6": int(sum(1 for r in den if r["cv_over_bound"] > 0)),
    }
    if lam:
        summary["median_log10_lambda_error"] = float(
            np.median([r["log10_err"] for r in lam]))
    if pen:
        summary["median_d0_over_max_rho_prime"] = float(
            np.median([r["d0_over_max"] for r in pen]))

    p = os.path.join(out_dir, "check_summary.json")
    json.dump(summary, open(p, "w"), indent=2, ensure_ascii=False)
    print(f"\n요약 저장: {p}")

    if args.plot:
        make_plots(pen, lam, den, out_dir)


if __name__ == "__main__":
    main()
