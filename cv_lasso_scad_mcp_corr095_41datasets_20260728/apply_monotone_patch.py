#!/usr/bin/env python3
"""셀렉터의 Direct96 헤드에만 단조 비증가 제약을 건다.

왜 canonical_penalty.py 가 아니라 여기인가
--------------------------------------------
canonical_penalty_cache.py 는 300k 캐시를 로드할 때 소스 파일들의
sha256 닫힌 집합을 검증한다.

    expected_sources = {
        "generator":              generate_canonical_penalty_cache.py,
        "canonical_builder":      canonical_penalty.py,      <- 여기 포함
        "penalty_sampler_v9":     penalty_sampler_v9.py,
        "penalty_relu_converter": penalty_relu_mlp_pwl_v33.py,
    }

canonical_penalty.py 를 한 바이트라도 고치면 캐시가 즉시 죽는다.
selector_v11_canonical.py 는 이 집합에 없다.  학습 스크립트가 해시를
기록하기는 하지만 provenance 용이고 하드 어서션이 아니다.

그리고 이쪽이 더 맞다
--------------------
캐시 페널티는 CanonicalPenaltyCache.encoded_indices 가 build_penalty 를
직접 부른다.  즉 베타 솔버가 학습 때 본 자유 형태 분포다.  베타는 동결이므로
그 분포를 건드리면 안 된다.

우리가 진단한 결함은 "셀렉터가 언덕 모양을 고른다" 하나뿐이다.
따라서 셀렉터의 제안에만 제약을 걸고 캐시는 그대로 두는 것이 정확하다.

    python3.11 apply_monotone_patch.py <PKG>/selector_v11_canonical.py
    python3.11 apply_monotone_patch.py <PKG>/selector_v11_canonical.py --revert
    python3.11 apply_monotone_patch.py <PKG>/selector_v11_canonical.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

FLAG = "MONOTONE_HINGE_WEIGHTS"

ANCHOR_HEAD = '''Tensor = torch.Tensor


class CanonicalDirect96Head(nn.Module):'''

INSERT_HEAD = '''Tensor = torch.Tensor


# --- 단조 패치 -------------------------------------------------------------
# rho'(t) = c + sum_k w_k * relu(t - t_k) 에서 relu 는 비감소이므로
# w_k <= 0 이면 rho' 가 비증가가 된다.  LASSO / SCAD / MCP 를 포함하는 족이고
# rho'(0) = max rho' 이 되어 kappa = 1 이 구조적으로 보장된다.
#
# selector_v20_yconditioned._lambda_reference_slope 의 docstring 이
#   "It equals d0 for the usual L1/SCAD/MCP-like decreasing derivative paths"
# 라고 이미 이 불변식을 가정하고 있었으나 어디서도 강제하지 않았다.
# selector_v24 는 zero_kkt_ratio(d0 기준)와 force_ratio(max drho 기준)를
# 따로 계산하는데, kappa = 1 이면 둘이 같아진다.
#
# 캐시 페널티(베타 솔버가 본 자유 형태)는 건드리지 않는다.
# CanonicalPenaltyCache.encoded_indices 는 canonical_penalty.build_penalty 를
# 직접 부르므로 이 패치의 영향을 받지 않는다.
MONOTONE_HINGE_WEIGHTS = True

# 감쇠 폭 손잡이.
# rho' 가 폭 w 로 감쇠하면 면적 정규화 후 d0 = 2/w 이고, w 는 총 기울기
# (활성 hinge 가중치 합)에 반비례한다.  따라서 d0 는 이 나눗수에 반비례한다.
#
# real 13 family sweep 승자들의 d0
#     lasso 0.167  scad6 0.286  scad3.7 0.426  mcp4 0.5
#     mcp2 1.0     mcp0.5 4.0   mcp0.25 8.0
# 캐시 통계로 디코딩한 d0 중앙이 이 구간에 오도록 맞춘다.
#     python3.11 verify_monotone.py --cache <penalty_cache...pt>
MONOTONE_HINGE_SCALE = 1.0


def build_penalty_monotone(
    direct_parameters: Tensor,
    *,
    derivative_floor: float,
    squareplus_scale: float,
    knots: Tensor,
) -> dict[str, Tensor]:
    """canonical_penalty.build_penalty 와 동일하되 hinge 를 음수로 강제."""
    from . import canonical_penalty as _cp

    if not MONOTONE_HINGE_WEIGHTS:
        return _cp.build_penalty(
            direct_parameters,
            derivative_floor=derivative_floor,
            squareplus_scale=squareplus_scale,
            knots=knots,
        )

    proposal = direct_parameters.float()
    if proposal.shape[-1] != _cp.DIRECT_PARAMETER_DIM:
        raise ValueError(f"Direct96 width {proposal.shape[-1]}")
    if not bool(torch.isfinite(proposal).all().item()):
        raise ValueError("Direct96 proposal contains nonfinite values")

    grid = knots.to(proposal)
    basis = _cp._basis(grid)

    hinge = -_cp.stable_squareplus(
        proposal[..., 1:], float(squareplus_scale)
    ) / float(MONOTONE_HINGE_SCALE)
    proposal_drho = proposal[..., :1] + torch.einsum(
        "...k,tk->...t", hinge, basis
    )

    shifted = proposal_drho - float(derivative_floor)
    positive_drho = float(derivative_floor) + _cp.stable_squareplus(
        shifted, float(squareplus_scale)
    )
    positivity_jacobian = _cp.stable_squareplus_jacobian(
        shifted, float(squareplus_scale)
    )
    if not bool(torch.isfinite(positivity_jacobian).all().item()) or bool(
        (positivity_jacobian <= 0.0).any().item()
    ):
        raise RuntimeError("canonical positivity map acquired a bad Jacobian")

    mass = torch.trapezoid(positive_drho, grid, dim=-1)
    if not bool(torch.isfinite(mass).all().item()) or bool(
        (mass <= 0.0).any().item()
    ):
        raise RuntimeError("canonical penalty has invalid positive mass")
    derivative = positive_drho / mass[..., None]

    encoded = _cp.native196_from_derivative_knots(derivative, grid)
    relu_b, relu_weights = _cp.derivative_knots_to_relu2(derivative, grid)
    encoded.update(
        {
            "direct_parameters": proposal,
            "proposal_drho": proposal_drho,
            "positive_drho": positive_drho,
            "positivity_jacobian": positivity_jacobian,
            "mass": mass,
            "relu_b": relu_b,
            "relu_weights": relu_weights,
            "knots": grid,
            "min_drho": derivative.amin(dim=-1),
            "rho6_error": (encoded["rho"][..., -1] - 1.0).abs(),
            "dead_last_weight_max_abs": relu_weights[..., -1].abs(),
            "monotone_hinge_weights": hinge,
            "monotone_violation_max": (
                derivative[..., 1:] - derivative[..., :-1]
            ).clamp_min(0.0).amax(dim=-1),
        }
    )
    return encoded


class CanonicalDirect96Head(nn.Module):'''

ANCHOR_CALL = '''        encoded = build_penalty(
            proposal,'''
INSERT_CALL = '''        encoded = build_penalty_monotone(
            proposal,'''


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=Path)
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    target: Path = args.target
    if not target.is_file():
        print(f"  없음: {target}")
        return 1
    text = target.read_text(encoding="utf-8")
    applied = FLAG in text

    if args.check:
        print(f"  {target}")
        print(f"  sha256   {sha(target)}")
        print(f"  단조패치 {'적용됨' if applied else '미적용'}")
        return 0

    backup = target.with_suffix(".py.premonotone")

    if args.revert:
        if not backup.is_file():
            print(f"  백업 없음: {backup}")
            return 1
        shutil.copy2(backup, target)
        print(f"  되돌림.  sha256 {sha(target)}")
        return 0

    if applied:
        print(f"  이미 적용됨.  sha256 {sha(target)}")
        return 0

    for anchor in (ANCHOR_HEAD, ANCHOR_CALL):
        if text.count(anchor) != 1:
            print(f"  앵커를 {text.count(anchor)}번 찾음 (1이어야 함):")
            print("  ---"); print(anchor); print("  ---")
            return 1

    if not backup.is_file():
        shutil.copy2(target, backup)
        print(f"  백업 생성  {backup.name}")

    text = text.replace(ANCHOR_HEAD, INSERT_HEAD, 1)
    text = text.replace(ANCHOR_CALL, INSERT_CALL, 1)
    target.write_text(text, encoding="utf-8")

    import py_compile
    try:
        py_compile.compile(str(target), doraise=True)
    except Exception as exc:                                       # noqa: BLE001
        shutil.copy2(backup, target)
        print(f"  컴파일 실패, 되돌림: {exc}")
        return 1

    print("  적용 완료")
    print(f"  sha256   {sha(target)}")
    print()
    print("  canonical_penalty.py 는 건드리지 않았다.  캐시 무결성 유지.")
    print("  검증:  python3.11 verify_monotone.py --cache <cache.pt>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
