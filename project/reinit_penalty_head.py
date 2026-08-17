#!/usr/bin/env python3
"""체크포인트에서 페널티 잠재를 만드는 레이어만 다시 깐다.

왜 필요한가
-----------
현재 가중치는 "언덕 모양을 만들어라" 로 최적화돼 있다.  단조 디코더 아래서
그대로 resume 하면 d0 ~ 16 인 퇴화 지점 (거의 hard threshold) 에서 출발한다.
real 13 family sweep 이 찾은 좋은 해들은 d0 0.167~8 구간에 있다.

V24 의 최종 페널티는 두 경로의 합에서 나온다.

    corrected_latent = initial_encoded["latent_raw"] + delta_latent
                     = penalty_parameter_head.projection(penalty_mlp(u0))
                       + delta_penalty_latent_head(corrected_code)
    encoded = penalty_parameter_head.decode_direct_parameters(corrected_latent)

decode 는 proposal = parameter_mean + parameter_std * corrected_latent 이므로
두 경로가 모두 0 근처면 proposal ~ parameter_mean 이 되고,
캐시 통계 그대로라 d0 가 유용 구간 한가운데(~3.5)에서 시작한다.

건드리지 않는 것
    parameter_mean / parameter_std / calibrated / reference_*   캐시 보정값
    model_state_dict                                            동결 베타 솔버
    lambda 관련 헤드                                            --reset-lambda 로 선택

    python3.11 reinit_penalty_head.py <ckpt.pt> --list
    python3.11 reinit_penalty_head.py <ckpt.pt> --out <new.pt>
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import torch

DEFAULT_PKG = (
    "kalbi_v6_exactprox10_eta4x_v24_additive512_traj3_k32s32_p100k_effbatch32"
)
TARGET_PREFIXES = (
    "penalty_parameter_head.projection",
    "delta_penalty_latent_head",
)
LAMBDA_PREFIXES = (
    "delta_loglambda_head",
    "raw_lambda_head",
    "lambda_head",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--pkg", default=os.environ.get("KALBI_PKG", DEFAULT_PKG))
    ap.add_argument("--init-std", type=float, default=1e-3,
                    help="선택자 원래 direct_head_init_std 와 동일")
    ap.add_argument("--reset-lambda", action="store_true",
                    help="lambda 헤드도 같이 초기화")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--seed", type=int, default=20260811)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["selector_state_dict"]
    print(f"  {args.ckpt.name}")
    print(f"  step {int(ckpt.get('completed_steps', -1))}   "
          f"selector 텐서 {len(sd)}개")
    print()

    prefixes = TARGET_PREFIXES + (LAMBDA_PREFIXES if args.reset_lambda else ())
    hit = [k for k in sd if any(k.startswith(p) for p in prefixes)]

    if args.list:
        print("  페널티 잠재 경로 후보")
        for k in sorted(k for k in sd if "penalty" in k or "latent" in k
                        or "lambda" in k):
            mark = " <-- 초기화" if k in hit else ""
            print(f"    {k:60s} {tuple(sd[k].shape)}{mark}")
        return 0

    if not hit:
        print("  대상 키를 못 찾았다.  --list 로 실제 이름을 확인하라")
        return 1

    g = torch.Generator().manual_seed(args.seed)
    print("  초기화 대상")
    for k in sorted(hit):
        t = sd[k]
        before = float(t.float().norm())
        if k.endswith(".bias"):
            sd[k] = torch.zeros_like(t)
        else:
            sd[k] = (torch.randn(t.shape, generator=g) * args.init_std).to(t.dtype)
        print(f"    {k:60s} norm {before:10.4f} -> "
              f"{float(sd[k].float().norm()):8.4f}")

    # 새 출발점의 d0 를 실제로 계산한다
    print()
    try:
        sel = importlib.import_module(f"{args.pkg}.selector_v11_canonical")
        cp = importlib.import_module(f"{args.pkg}.canonical_penalty")
        mean = sd.get("penalty_parameter_head.parameter_mean")
        if mean is None:
            raise KeyError("parameter_mean 이 state_dict 에 없다")
        knots = cp.canonical_grid()
        kw = dict(derivative_floor=cp.DEFAULT_DERIVATIVE_FLOOR,
                  squareplus_scale=cp.DEFAULT_SQUAREPLUS_SCALE, knots=knots)
        free = cp.build_penalty(mean[None].float(), **kw)
        if hasattr(sel, "build_penalty_monotone"):
            mono = sel.build_penalty_monotone(mean[None].float(), **kw)
            d = mono["drho"][0]
            print(f"  새 출발점 (proposal = parameter_mean, 단조 디코더)")
            print(f"    d0 {float(mono['d0'][0]):.4f}   "
                  f"max rho' {float(d.max()):.4f}   "
                  f"kappa {float(d[0]/d.max()):.6f}")
            print(f"    참고: 자유 디코더였다면 d0 {float(free['d0'][0]):.4f}   "
                  f"kappa {float(free['drho'][0][0]/free['drho'][0].max()):.4f}")
            print(f"    유용 구간 [0.167, 8.0]  "
                  f"{'안' if 0.167 <= float(mono['d0'][0]) <= 8.0 else '밖'}")
        else:
            print("  selector_v11_canonical 에 단조 패치가 없다.")
            print("  apply_monotone_patch.py 를 먼저 적용하라")
    except Exception as exc:                                       # noqa: BLE001
        print(f"  d0 확인 생략: {type(exc).__name__}: {exc}")

    out = args.out or args.ckpt.with_name(
        args.ckpt.stem + "_penaltyreinit" + args.ckpt.suffix)
    ckpt["selector_state_dict"] = sd
    meta = ckpt.setdefault("metadata", {})
    meta["penalty_head_reinitialized"] = True
    meta["penalty_head_reinit_keys"] = sorted(hit)
    meta["penalty_head_reinit_std"] = args.init_std
    meta["monotone_hinge_weights"] = True
    torch.save(ckpt, out)
    print()
    print(f"  저장  {out}")
    print(f"  model_state_dict 는 건드리지 않았다 (동결 베타 그대로)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
