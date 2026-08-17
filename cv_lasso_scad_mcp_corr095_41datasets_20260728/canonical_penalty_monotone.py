#!/usr/bin/env python3
"""Direct96 페널티에 단조 비증가 제약을 건다.

현재 계약 (canonical_penalty.build_penalty)

    rho'(t) = c + sum_k w_k * relu(t - t_k)          <- w_k 부호 자유
    positive = floor + squareplus(rho' - floor)
    derivative = positive / mass                      <- 면적만 1 로 정규화

hinge 기저 relu(t - t_k) 는 비감소이므로 w_k > 0 이면 rho' 가 증가한다.
정규화가 면적만 잡으므로 "가운데가 볼록한 언덕" 이 허용되고, 실제로
학습된 셀렉터가 그 모양을 골랐다.  결과:

    kappa = rho'(0)/max rho'  중앙 0.109   (LASSO/SCAD/MCP 는 1)
    rho'(6) = 0               13/13        (큰 계수 무벌금)
    d0*exp(ell) < 1           11/13        (lambda_max 에서도 0 이 안 됨)

수정: w_k <= 0 을 구조적으로 강제한다.

    w_k = -stable_squareplus(theta_k, scale)   <  0

relu(t - t_k) 가 비감소이고 계수가 음수이므로 합은 비증가.
squareplus 는 단조증가라 positivity 를 통과해도 비증가가 유지된다.
면적 정규화는 그대로 두므로 lambda 스케일 계약이 안 바뀐다.

따라서 자동으로 얻어지는 것

    rho' 비증가        -> rho 오목.  LASSO / SCAD / MCP 를 포함하는 정칙 족
    rho'(0) = max rho' -> kappa = 1
    integral = 1       -> rho'(0) >= 1/6.  실측상으로는 2~4 예상

canonical_penalty 는 패키지 안에 있고 상대 임포트로만 접근되므로
(from .canonical_penalty import ...) 최상위 이름으로 못 찾는다.
그래서 임포트 시점이 아니라 patch() 시점에 지연 해석한다.

사용법 (기존 체크포인트에 추론 시점에만 적용):

    import canonical_penalty_monotone as mono
    mono.patch()          # sys.modules 전체에서 build_penalty 를 교체
    ...                   # 평소대로 추론
    mono.unpatch()
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any

import torch

Tensor = torch.Tensor

MONOTONE_CONTRACT_VERSION = (
    "v13_canonical_direct96_monotone_nonincreasing_"
    "negative_hinge_weights_area_normalized_v1"
)

DEFAULT_PKG = (
    "kalbi_v6_exactprox10_eta4x_v24_additive512_traj3_k32s32_p100k_effbatch32"
)

_CP: Any = None
_ORIGINAL: Any = None
_PATCHED_MODULES: list[tuple[Any, str]] = []


def _resolve() -> Any:
    """canonical_penalty 모듈을 찾는다.  이미 임포트된 것 우선."""
    global _CP, _ORIGINAL
    if _CP is not None:
        return _CP
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        if name == "canonical_penalty" or name.endswith(".canonical_penalty"):
            if hasattr(module, "build_penalty"):
                _CP = module
                break
    if _CP is None:
        pkg = os.environ.get("KALBI_PKG", DEFAULT_PKG)
        _CP = importlib.import_module(f"{pkg}.canonical_penalty")
    _ORIGINAL = _CP.build_penalty
    return _CP


def build_penalty_monotone(
    direct_parameters: Tensor,
    *,
    derivative_floor: float | None = None,
    squareplus_scale: float | None = None,
    knots: Tensor | None = None,
) -> dict[str, Tensor]:
    """build_penalty 와 동일하되 hinge 가중치를 음수로 강제한다."""
    cp = _resolve()
    if derivative_floor is None:
        derivative_floor = cp.DEFAULT_DERIVATIVE_FLOOR
    if squareplus_scale is None:
        squareplus_scale = cp.DEFAULT_SQUAREPLUS_SCALE

    proposal = direct_parameters.float()
    if proposal.shape[-1] != cp.DIRECT_PARAMETER_DIM:
        raise ValueError(
            f"Direct96 width {proposal.shape[-1]} != {cp.DIRECT_PARAMETER_DIM}"
        )
    if not bool(torch.isfinite(proposal).all().item()):
        raise ValueError("Direct96 proposal contains nonfinite values")
    if float(derivative_floor) <= 0.0:
        raise ValueError("derivative floor must be positive")

    grid = (
        cp.canonical_grid(device=proposal.device, dtype=proposal.dtype)
        if knots is None
        else knots.to(proposal)
    )
    basis = cp._basis(grid)

    # ---- 유일한 변경점 --------------------------------------------------
    # 원본:  weights = proposal[..., 1:]                (부호 자유)
    # 변경:  weights = -squareplus(proposal[..., 1:])   (항상 음수)
    weights = -cp.stable_squareplus(proposal[..., 1:], float(squareplus_scale))
    # --------------------------------------------------------------------

    proposal_drho = proposal[..., :1] + torch.einsum(
        "...k,tk->...t", weights, basis
    )
    shifted = proposal_drho - float(derivative_floor)
    positive_drho = float(derivative_floor) + cp.stable_squareplus(
        shifted, float(squareplus_scale)
    )
    positivity_jacobian = cp.stable_squareplus_jacobian(
        shifted, float(squareplus_scale)
    )
    if not bool(torch.isfinite(positivity_jacobian).all().item()) or bool(
        (positivity_jacobian <= 0.0).any().item()
    ):
        raise RuntimeError("canonical positivity map acquired a zero/nonfinite Jacobian")

    mass = torch.trapezoid(positive_drho, grid, dim=-1)
    if not bool(torch.isfinite(mass).all().item()) or bool((mass <= 0.0).any().item()):
        raise RuntimeError("canonical penalty has invalid positive mass")
    derivative = positive_drho / mass[..., None]

    encoded = cp.native196_from_derivative_knots(derivative, grid)
    relu_b, relu_weights = cp.derivative_knots_to_relu2(derivative, grid)

    violation = (derivative[..., 1:] - derivative[..., :-1]).clamp_min(0.0)
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
            # 단조성 진단
            "monotone_hinge_weights": weights,
            "monotone_violation_max": violation.amax(dim=-1),
            "monotone_kappa": (
                derivative[..., 0] / derivative.amax(dim=-1).clamp_min(1e-12)
            ),
        }
    )
    return encoded


# ---------------------------------------------------------------- 패치

def patch(verbose: bool = True) -> int:
    """sys.modules 전체에서 원본 build_penalty 를 단조 버전으로 교체."""
    global _PATCHED_MODULES
    unpatch(verbose=False)
    _resolve()
    count = 0
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        try:
            target = getattr(module, "build_penalty", None)
        except Exception:                                          # noqa: BLE001
            continue
        if target is _ORIGINAL:
            setattr(module, "build_penalty", build_penalty_monotone)
            _PATCHED_MODULES.append((module, name))
            count += 1
    if verbose:
        print(f"  단조 페널티 패치 적용: {count}개 모듈")
        for _, name in _PATCHED_MODULES:
            print(f"    - {name}")
    if count == 0:
        raise RuntimeError(
            "build_penalty 를 참조하는 모듈을 못 찾았다. "
            "모델을 먼저 로드한 뒤 patch() 를 불러라."
        )
    return count


def unpatch(verbose: bool = True) -> int:
    global _PATCHED_MODULES
    for module, _ in _PATCHED_MODULES:
        setattr(module, "build_penalty", _ORIGINAL)
    count = len(_PATCHED_MODULES)
    _PATCHED_MODULES = []
    if verbose:
        print(f"  패치 해제: {count}개 모듈")
    return count


def is_patched() -> bool:
    return bool(_PATCHED_MODULES)
