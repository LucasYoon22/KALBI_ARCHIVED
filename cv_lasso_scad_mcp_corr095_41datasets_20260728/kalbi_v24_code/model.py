"""Kalbi V6 beta solver with matrix-free exact Direct96 proximal gradient.

Kalbi V6 preserves the complete V3 Krylov + A/B ISIS + learned-threshold path,
then applies ten exact composite proximal-gradient steps.  Direct96 represents
a piecewise-linear derivative and therefore a piecewise-quadratic penalty.  A
curvature-safe scalar prox is solved analytically after a batched interval
search.  The data gradient and line-search majorization test use only X beta and
X^T residual products; no P-by-P matrix is formed.
"""

from __future__ import annotations

import types
from typing import Any

import torch
import torch.nn as nn

import model_fpk
from model_fpk import FpkConfig, make_fpk_model
import shape_feature_normalization

from .exact_direct96_prox import (
    DEFAULT_CURVATURE_MARGIN,
    curvature_safe_step_cap,
    exact_direct96_prox,
    interpolate_direct96_penalty,
    unpack_direct96_penalty,
)
from .utils import (
    ABTrajectoryThresholdFiLMHead,
    EXPANDED_DIM,
    FEATURE_FILM_DIM,
    GRID_SIZE,
    LOCAL_ISIS_GRADIENT_CONTRACT,
    META_DIM,
    NATIVE_DIM,
    PROX_COND_DIM,
    ResidualMLP,
    ResidualPenaltyShapeEncoder,
    build_kalbi_ab_trajectory,
    make_penalty_grid,
    make_prox_film_cond,
    native196_to_expanded388,
    strip_convexity_flag,
)


Tensor = torch.Tensor

MODEL_VERSION = (
    "kalbi_v6_krylov_abfilm_threshold_exactproxpg10_eta4x_k32_stages32_v1"
)
V3_SOURCE_MODEL_VERSION = (
    "kalbi_v3_krylov_abfilm_learnedthreshold_k32_stages32_v1"
)
BASE_MODEL_VERSION = "global_krylov_ridge_native196_v1"
V56_BASE_MODEL_VERSION = "kalbi_v6_no_local_chebyshev"
PENALTY_CONTRACT_VERSION = (
    "native196_interleaved_rho_drho_meta4_expanded388_v1"
)
TRAINING_CONTRACT = (
    "kalbi_v6_abfilm_centeredcorr_disjoint_k32_stages32_ridge1e-6_"
    "learned_terminal_threshold_exactgrad_exactDirect96proxPG10_eta4x_"
    "curvatureSafe_majorizationBacktracking_noSTE_"
    "canonical_bound6_log1p_F_over_F0_v1"
)
LEGACY_BASE_MODEL_VERSION = "v50fast"
BASIS_VERSION = (
    "global_krylov_ridge_plus_ab_trajectory_threshold_plus_exactproxpg10_v6"
)
ISIS_SELECTION_VERSION = (
    "lambda_independent_centered_pearson_disjoint_topk32_stages32_v1"
)
SHARED_SUPPORT_CONTRACT = (
    "centered_corr_A_B_ridge1e-6_supportlocal_correction_global_threshold_"
    "fullP_exactgrad_exactDirect96proxPG10_eta4x_refinement_v6"
)

KALBI_TOPK = 32
KALBI_STAGES = 32
KALBI_RIDGE = 1e-6
KALBI_BETA_BOUND = 6.0
KALBI_THRESHOLD_LOG_MULTIPLIER_MIN = -6.0
KALBI_THRESHOLD_LOG_MULTIPLIER_MAX = 6.0
KALBI_EXACT_PG_STEPS = 10
# The eta sweep defines eta-factor = 2 / Ritz safety.  A value of 0.5
# therefore reproduces the tested eta4x initial-candidate convention.
KALBI_EXACT_PG_RITZ_SAFETY = 0.5
KALBI_EXACT_PG_DATA_MARGIN = 0.95
KALBI_EXACT_PG_CURVATURE_MARGIN = DEFAULT_CURVATURE_MARGIN
KALBI_EXACT_PG_BACKTRACK_FACTOR = 0.5
KALBI_EXACT_PG_MAX_BACKTRACKS = 6
KALBI_EXACT_PG_FEATURE_CHUNK = 4096

V58FAST_MODEL_VERSION = MODEL_VERSION
V58FAST_BASE_MODEL_VERSION = BASE_MODEL_VERSION
V58FAST_PENALTY_CONTRACT_VERSION = PENALTY_CONTRACT_VERSION
V58FAST_GRID_SIZE = GRID_SIZE
V58FAST_META_DIM = META_DIM
V58FAST_NATIVE_DIM = NATIVE_DIM
V58FAST_EXPANDED_DIM = EXPANDED_DIM
V58FAST_PROX_COND_DIM = PROX_COND_DIM
V58FAST_FEATURE_FILM_DIM = FEATURE_FILM_DIM

V58CHEB_MODEL_VERSION = MODEL_VERSION
V58CHEB_BASE_MODEL_VERSION = BASE_MODEL_VERSION
V58CHEB_PENALTY_CONTRACT_VERSION = PENALTY_CONTRACT_VERSION
V58CHEB_BASIS_VERSION = BASIS_VERSION


def make_fpk_config(config: dict[str, Any]) -> FpkConfig:
    """Build the stable FPK bootstrap used for global Krylov encoding."""
    model_config = config.get("model", {})
    shape_config = config.get("penalty_shape", {})
    return FpkConfig(
        krylov_dim=int(model_config.get("krylov_dim", 32)),
        num_decoder_channels=int(model_config.get("num_decoder_channels", 12)),
        use_bilinear_channels=bool(
            model_config.get("use_bilinear_channels", True)
        ),
        cond_dim=256,
        cond_hidden=256,
        query_mixer_hidden=256,
        feature_head_hidden=max(
            512, int(model_config.get("feature_head_hidden", 512))
        ),
        prox_hidden=max(512, int(model_config.get("prox_hidden", 512))),
        use_krylov_ridge_base=bool(
            model_config.get("use_krylov_ridge_base", True)
        ),
        base_head_hidden=max(
            256, int(model_config.get("base_head_hidden", 256))
        ),
        zero_init_query_mixer=True,
        zero_init_feature_head=True,
        zero_init_learned_prox=True,
        retain_U=bool(model_config.get("retain_U", True)),
        use_penalty_shape_encoding=True,
        shape_feature_dim=197,
        penalty_grid_size=int(shape_config.get("grid_size", 96)),
        penalty_grid_channels=2,
        penalty_shape_hidden=256,
        penalty_shape_code_dim=256,
        penalty_shape_depth=max(
            2, int(shape_config.get("penalty_shape_depth", 2))
        ),
    )


def _set_buffer(module: nn.Module, name: str, value: Tensor) -> None:
    if name in module._buffers:
        module._buffers[name] = value
    else:
        module.register_buffer(name, value)


def patch_model_kalbi_v6(
    model: nn.Module,
    device: str | torch.device | None = None,
    *,
    isis_topk: int = KALBI_TOPK,
    isis_stages: int = KALBI_STAGES,
    isis_ridge: float = KALBI_RIDGE,
    decoder_hidden: int = 512,
    film_hidden: int = 512,
    feature_chunk: int = 2048,
) -> nn.Module:
    """Install the active Kalbi V6 decoder on a fresh FPK model."""
    if int(isis_topk) <= 0 or int(isis_stages) <= 0:
        raise ValueError("isis_topk and isis_stages must be positive")
    if float(isis_ridge) != KALBI_RIDGE:
        raise ValueError(
            f"Kalbi V6 fixes ISIS ridge at {KALBI_RIDGE}, got {isis_ridge}"
        )
    target_device = (
        torch.device(device)
        if device is not None
        else next(model.parameters()).device
    )
    cfg = model.cfg
    cfg.retain_centered_data_in_encoding = True
    code_dim = int(cfg.cond_dim)

    _set_buffer(
        model.cond_embed,
        "_shape_mean",
        torch.zeros(NATIVE_DIM, dtype=torch.float32, device=target_device),
    )
    _set_buffer(
        model.cond_embed,
        "_shape_scale",
        torch.ones(NATIVE_DIM, dtype=torch.float32, device=target_device),
    )
    model.cond_embed.penalty_shape_encoder = ResidualPenaltyShapeEncoder(
        in_dim=NATIVE_DIM,
        out_dim=code_dim,
        hidden=256,
        depth=3,
        zero_init_last=True,
    ).to(target_device)
    numeric_dim = int(model.cond_embed.numeric_norm.normalized_shape[0])
    model.cond_embed.shape_film = ResidualMLP(
        in_dim=numeric_dim,
        hidden=256,
        out_dim=2 * code_dim,
        depth=3,
        zero_init_last=True,
        act=nn.GELU,
    ).to(target_device)
    condition_input_dim = numeric_dim + code_dim
    if getattr(model.cond_embed, "use_penalty_type_embed", False):
        condition_input_dim += int(model.cond_embed.pen_embed.embedding_dim)
    model.cond_embed.mlp = ResidualMLP(
        in_dim=condition_input_dim,
        hidden=256,
        out_dim=code_dim,
        depth=3,
        zero_init_last=False,
        act=nn.GELU,
    ).to(target_device)

    # Only log(alpha/lambda) is learned. The base is a pure global Krylov ridge
    # estimate; no diagonal blend and no local basis remain in Kalbi V6.
    model.base_head.net = ResidualMLP(
        in_dim=code_dim,
        hidden=max(256, int(getattr(cfg, "base_head_hidden", 256))),
        out_dim=1,
        depth=3,
        zero_init_last=True,
        act=nn.GELU,
    ).to(target_device)

    for stale_name in ("query_mixer", "feature_head", "learned_prox"):
        if hasattr(model, stale_name):
            delattr(model, stale_name)
    model.kalbi_ab_head = ABTrajectoryThresholdFiLMHead(
        stages=int(isis_stages),
        film_dim=FEATURE_FILM_DIM,
        hidden=int(decoder_hidden),
        film_hidden=int(film_hidden),
        feature_chunk=int(feature_chunk),
        use_checkpoint=True,
    ).to(target_device)

    _set_buffer(
        model,
        "_logt_grid",
        (torch.log(make_penalty_grid() + 1e-5) / 3.0).to(target_device),
    )
    model.register_buffer(
        "_kalbi_v6_penalty_grid",
        make_penalty_grid().to(target_device),
        persistent=False,
    )
    _set_buffer(
        model,
        "_shape_mean_388",
        torch.zeros(EXPANDED_DIM, dtype=torch.float32, device=target_device),
    )
    _set_buffer(
        model,
        "_shape_scale_388",
        torch.ones(EXPANDED_DIM, dtype=torch.float32, device=target_device),
    )
    for stale in ("_shape_mean_389", "_shape_scale_389"):
        if hasattr(model, stale):
            delattr(model, stale)

    cfg.shape_feature_dim = NATIVE_DIM
    cfg.meta_dim = META_DIM
    model.kalbi_isis_topk = int(isis_topk)
    model.kalbi_isis_stages = int(isis_stages)
    model.kalbi_isis_ridge = float(isis_ridge)
    model.kalbi_shared_support_contract = SHARED_SUPPORT_CONTRACT
    model.kalbi_model_version = MODEL_VERSION
    model.v58cheb_basis_version = BASIS_VERSION
    model.v58fast_model_version = MODEL_VERSION
    model.v58fast_base_model_version = BASE_MODEL_VERSION
    model.v58fast_penalty_contract_version = PENALTY_CONTRACT_VERSION
    model.v58fast_local_isis_gradient_contract = LOCAL_ISIS_GRADIENT_CONTRACT
    model.kalbi_refinement_contract = (
        "ten_matrix_free_exact_data_gradient_plus_curvature_safe_"
        "piecewise_quadratic_Direct96_exact_prox_majorization_backtracking_"
        "eta4x_initial_candidate_ordinary_unrolled_autograd_no_STE_v1"
    )
    model.decode_queries = types.MethodType(_decode_queries_kalbi_v6, model)
    return model


def build_model(
    config: dict[str, Any], device: str | torch.device = "cpu"
) -> nn.Module:
    """Build Kalbi V6 from scratch."""
    shape_feature_normalization.SHAPE_FEATURE_MEAN = [0.0] * 197
    shape_feature_normalization.SHAPE_FEATURE_SCALE = [1.0] * 197
    shape_feature_normalization.FEATURE_DIM = 197
    bootstrap_config = make_fpk_config(config)
    model, _ = make_fpk_model(bootstrap_config)
    patch_model_kalbi_v6(
        model,
        device=device,
        isis_topk=KALBI_TOPK,
        isis_stages=KALBI_STAGES,
        isis_ridge=KALBI_RIDGE,
        decoder_hidden=512,
        film_hidden=512,
        feature_chunk=2048,
    )
    if NATIVE_DIM != 196 or EXPANDED_DIM != 388:
        raise RuntimeError("native/expanded penalty contract drifted")
    if PROX_COND_DIM != 490 or FEATURE_FILM_DIM != 394:
        raise RuntimeError("FiLM conditioning dimensions drifted")
    return model.to(device)


def _resolve_query_context(
    self: nn.Module,
    encoded: dict,
    lambdas: Tensor,
    lambda_max: Tensor | None,
    penalty_type_id: Tensor | None,
    penalty_params: Tensor | None,
    penalty_shape_features: Tensor | None,
    penalty_d0: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    batch_size = int(encoded["B"])
    sample_count = int(encoded["N"])
    feature_count = int(encoded["P"])
    device = encoded["Xty"].device
    lam = model_fpk._as_2d_queries(
        lambdas,
        batch_size,
        "lambdas",
        device=device,
        dtype=torch.float32,
    )
    _, query_count = lam.shape
    lmx_source = encoded["lambda_max"] if lambda_max is None else lambda_max
    lmx = model_fpk._broadcast_queries_like(
        lmx_source,
        batch_size,
        query_count,
        "lambda_max",
        device=device,
        dtype=torch.float32,
    )

    if penalty_shape_features is None or penalty_d0 is None:
        if penalty_type_id is None:
            raise ValueError(
                "Kalbi V6 requires penalty_shape_features and penalty_d0 "
                "or penalty_type_id"
            )
        resolved_type = model_fpk._resolve_penalty_type_grid(
            penalty_type_id,
            B=batch_size,
            Q=query_count,
            device=device,
        )
        resolved_parameters = model_fpk._resolve_penalty_param_grid(
            penalty_params,
            B=batch_size,
            Q=query_count,
            device=device,
            dtype=torch.float32,
            out_dim=1,
        )
        features = model_fpk.make_penalty_features(
            penalty_type_id=resolved_type,
            penalty_params=resolved_parameters,
        )
        penalty_shape_features = features["shape_features"]
        penalty_d0 = features["d0"]
    penalty_shape_features = strip_convexity_flag(penalty_shape_features)
    type_for_embedding = (
        model_fpk._resolve_penalty_type_grid(
            penalty_type_id,
            B=batch_size,
            Q=query_count,
            device=device,
        )
        if penalty_type_id is not None
        else None
    )
    d0 = model_fpk._broadcast_queries_like(
        penalty_d0,
        batch_size,
        query_count,
        "penalty_d0",
        device=device,
        dtype=torch.float32,
    ).clamp_min(0.0)
    prox_condition = make_prox_film_cond(
        self, lam, lmx, d0, penalty_shape_features
    ).view(batch_size, query_count, -1)
    condition = self.cond_embed(
        lambdas=lam,
        lambda_max=lmx,
        N=float(sample_count),
        P=float(feature_count),
        penalty_type_id=type_for_embedding,
        penalty_shape_features=penalty_shape_features,
        penalty_d0=d0,
    )
    return lam, lmx, d0, penalty_shape_features, prox_condition, condition


def _global_krylov_ridge(
    self: nn.Module,
    encoded: dict,
    lam: Tensor,
    condition: Tensor,
) -> tuple[Tensor, Tensor]:
    if not self.cfg.use_krylov_ridge_base:
        raise RuntimeError("Kalbi V6 requires the global Krylov ridge base")
    device = encoded["Xty"].device
    batch_size, query_count = lam.shape
    base_raw = self.base_head(condition)
    log_alpha_scale = base_raw[..., 0].clamp(
        self.cfg.base_alpha_log_min, self.cfg.base_alpha_log_max
    )
    alpha = lam * torch.exp(log_alpha_scale) + self.cfg.diag_eps
    with model_fpk._maybe_disable_autocast(device.type):
        T = encoded["T_krylov"].float()
        Uy = encoded["Uy_krylov"].float()
        basis = encoded["basis_proj"].float()
        width = T.shape[-1]
        eye = torch.eye(width, device=device, dtype=torch.float32).view(
            1, 1, width, width
        )
        rhs = Uy[:, None, :, None].expand(
            batch_size, query_count, width, 1
        )
        system = (
            T[:, None, :, :]
            + alpha[..., None, None].float() * eye
            + self.cfg.diag_eps * eye
        )
        coefficients = torch.linalg.solve(system, rhs).squeeze(-1)
        base = torch.einsum("bqm,bmp->bqp", coefficients, basis)
    return base.to(condition.dtype), alpha


def _get_ab_trajectory(self: nn.Module, encoded: dict) -> dict[str, Tensor]:
    cached = encoded.get("_kalbi_v6_ab_trajectory")
    if cached is not None:
        return cached
    if "Xc" not in encoded or "yc" not in encoded:
        raise RuntimeError("Kalbi V6 encoding must retain centered X and y")
    with model_fpk._maybe_disable_autocast(encoded["Xc"].device.type):
        trajectory = build_kalbi_ab_trajectory(
            X=encoded["Xc"],
            y=encoded["yc"],
            feature_mask=encoded.get("feature_mask"),
            topk=int(self.kalbi_isis_topk),
            stages=int(self.kalbi_isis_stages),
            ridge=float(self.kalbi_isis_ridge),
        )
    encoded["_kalbi_v6_ab_trajectory"] = trajectory
    return trajectory


def _select_exact_pg_step(
    self: nn.Module,
    encoded: dict,
    beta0: Tensor,
    residual0: Tensor,
    data_gradient: Tensor,
    lam: Tensor,
    penalty_shape_features: Tensor,
) -> dict[str, Tensor]:
    """Choose a detached, curvature-safe step by exact quadratic majorization."""

    grid = self._kalbi_v6_penalty_grid.float()
    native = strip_convexity_flag(penalty_shape_features).detach().float()
    lambda_detached = lam.detach().float().clamp_min(0.0)
    pieces = unpack_direct96_penalty(native, grid)
    curvature_cap = curvature_safe_step_cap(
        lambda_detached,
        pieces.kappa,
        margin=KALBI_EXACT_PG_CURVATURE_MARGIN,
    )

    ritz_max = torch.linalg.eigvalsh(
        encoded["T_krylov"].float()
    ).amax(-1).clamp_min(1e-8)
    diagonal_max = encoded["gram_diag"].float().amax(-1).clamp_min(1e-8)
    initial_lipschitz = torch.maximum(
        KALBI_EXACT_PG_RITZ_SAFETY * ritz_max,
        diagonal_max,
    )
    step_size = (
        KALBI_EXACT_PG_DATA_MARGIN / initial_lipschitz[:, None]
    )
    step_size = torch.minimum(step_size, curvature_cap)
    if not bool(torch.isfinite(step_size).all().item()):
        raise RuntimeError("Kalbi V6 produced a nonfinite initial PG step")

    x_centered = encoded["Xc"].float()
    sample_count = max(1, int(x_centered.shape[1]))
    feature_mask = encoded.get("feature_mask")
    accepted = torch.zeros_like(step_size, dtype=torch.bool)
    backtracks = torch.zeros_like(step_size, dtype=torch.int64)
    accepted_ratio = torch.full_like(step_size, torch.inf)

    for attempt in range(KALBI_EXACT_PG_MAX_BACKTRACKS + 1):
        gamma = step_size * lambda_detached
        proposal = beta0.detach() - step_size[..., None] * data_gradient.detach()
        candidate, _ = exact_direct96_prox(
            proposal,
            gamma,
            native,
            grid,
            feature_chunk=KALBI_EXACT_PG_FEATURE_CHUNK,
            curvature_margin=KALBI_EXACT_PG_CURVATURE_MARGIN,
        )
        if feature_mask is not None:
            candidate = candidate * feature_mask[:, None, :].float()
        delta = candidate - beta0.detach()
        delta_energy = delta.square().sum(dim=-1)
        x_delta = torch.matmul(delta, x_centered.transpose(1, 2))
        directional_curvature = (
            x_delta.square().sum(dim=-1) / float(sample_count)
        ) / delta_energy.clamp_min(1e-20)
        ratio = step_size * directional_curvature
        ratio = torch.where(
            delta_energy <= 1e-20, torch.zeros_like(ratio), ratio
        )
        passed = (delta_energy <= 1e-20) | (
            torch.isfinite(ratio)
            & (ratio <= KALBI_EXACT_PG_DATA_MARGIN + 1e-6)
        )
        newly_accepted = (~accepted) & passed
        accepted_ratio = torch.where(
            newly_accepted, ratio.clamp_min(0.0), accepted_ratio
        )
        accepted = accepted | passed
        unresolved = ~accepted
        if not bool(unresolved.any().item()):
            break
        if attempt < KALBI_EXACT_PG_MAX_BACKTRACKS:
            step_size = torch.where(
                unresolved,
                step_size * KALBI_EXACT_PG_BACKTRACK_FACTOR,
                step_size,
            )
            backtracks = backtracks + unresolved.to(backtracks.dtype)

    unresolved = ~accepted
    trace_lipschitz = (
        encoded["gram_diag"].float().clamp_min(0.0).sum(dim=-1).clamp_min(1e-8)
    )
    fallback_step = (
        KALBI_EXACT_PG_DATA_MARGIN / trace_lipschitz[:, None]
    )
    fallback_step = torch.minimum(fallback_step, curvature_cap)
    step_size = torch.where(
        unresolved,
        torch.minimum(step_size, fallback_step),
        step_size,
    )
    if bool(unresolved.any().item()):
        fallback_gamma = step_size * lambda_detached
        fallback_proposal = (
            beta0.detach()
            - step_size[..., None] * data_gradient.detach()
        )
        fallback_candidate, _ = exact_direct96_prox(
            fallback_proposal,
            fallback_gamma,
            native,
            grid,
            feature_chunk=KALBI_EXACT_PG_FEATURE_CHUNK,
            curvature_margin=KALBI_EXACT_PG_CURVATURE_MARGIN,
        )
        if feature_mask is not None:
            fallback_candidate = (
                fallback_candidate * feature_mask[:, None, :].float()
            )
        fallback_delta = fallback_candidate - beta0.detach()
        fallback_x_delta = torch.matmul(
            fallback_delta, x_centered.transpose(1, 2)
        )
        candidate = torch.where(
            unresolved[..., None], fallback_candidate, candidate
        )
        x_delta = torch.where(
            unresolved[..., None], fallback_x_delta, x_delta
        )

    guaranteed_ratio = step_size * trace_lipschitz[:, None]
    majorization_ratio = torch.where(
        unresolved, guaranteed_ratio, accepted_ratio
    )
    rho_before, _, _ = interpolate_direct96_penalty(
        beta0.detach().abs(), pieces, grid
    )
    rho_after, _, _ = interpolate_direct96_penalty(
        candidate.abs(), pieces, grid
    )
    data_half_before = 0.5 * residual0.detach().square().mean(dim=-1)
    residual_after = residual0.detach() + x_delta
    data_half_after = 0.5 * residual_after.square().mean(dim=-1)
    penalty_before = lambda_detached * rho_before.sum(dim=-1)
    penalty_after = lambda_detached * rho_after.sum(dim=-1)
    objective_before = data_half_before + penalty_before
    objective_after = data_half_after + penalty_after
    return {
        "step_size": step_size.detach(),
        "kappa": pieces.kappa.detach(),
        "curvature_step_cap": curvature_cap.detach(),
        "initial_lipschitz": initial_lipschitz[:, None].expand_as(lam).detach(),
        "trace_lipschitz": trace_lipschitz[:, None].expand_as(lam).detach(),
        "majorization_ratio": majorization_ratio.detach(),
        "backtracks": backtracks.detach(),
        "trace_fallback": unresolved.float().detach(),
        "data_half_before": data_half_before.detach(),
        "data_half_after": data_half_after.detach(),
        "penalty_before": penalty_before.detach(),
        "penalty_after": penalty_after.detach(),
        "objective_before": objective_before.detach(),
        "objective_after": objective_after.detach(),
        "objective_delta": (objective_after - objective_before).detach(),
    }


def _refine_beta_kalbi_v6(
    self: nn.Module,
    encoded: dict,
    beta_v3: Tensor,
    lam: Tensor,
    d0: Tensor,
    penalty_shape_features: Tensor,
) -> dict[str, Tensor]:
    """Apply one matrix-free exact Direct96 composite PG refinement step."""

    if "Xc" not in encoded or "yc" not in encoded:
        raise RuntimeError("Kalbi V6 refinement requires retained centered data")
    device = beta_v3.device
    with model_fpk._maybe_disable_autocast(device.type):
        beta0 = beta_v3.float()
        x_centered = encoded["Xc"].float()
        y_centered = encoded["yc"].float()
        sample_count = int(x_centered.shape[1])
        prediction0 = torch.matmul(beta0, x_centered.transpose(1, 2))
        residual0 = prediction0 - y_centered[:, None, :]
        data_gradient = torch.matmul(residual0, x_centered) / float(
            max(1, sample_count)
        )
        if not bool(
            torch.isfinite(data_gradient).all().item()
            and torch.isfinite(beta0).all().item()
        ):
            raise RuntimeError("Kalbi V6 exact PG received nonfinite inputs")

        with torch.no_grad():
            step = _select_exact_pg_step(
                self,
                encoded,
                beta0,
                residual0,
                data_gradient,
                lam,
                penalty_shape_features,
            )
        step_size = step["step_size"]
        gamma = step_size * lam.float().clamp_min(0.0)
        proposal = beta0 - step_size[..., None] * data_gradient
        refined, prox_diagnostics = exact_direct96_prox(
            proposal,
            gamma,
            strip_convexity_flag(penalty_shape_features).float(),
            self._kalbi_v6_penalty_grid.float(),
            feature_chunk=KALBI_EXACT_PG_FEATURE_CHUNK,
            curvature_margin=KALBI_EXACT_PG_CURVATURE_MARGIN,
        )
        feature_mask = encoded.get("feature_mask")
        if feature_mask is not None:
            refined = refined * feature_mask[:, None, :].float()
        if not bool(torch.isfinite(refined).all().item()):
            raise RuntimeError("Kalbi V6 exact prox produced nonfinite coefficients")

        change = refined - beta0
        native_d0 = strip_convexity_flag(penalty_shape_features).float()[
            ..., 1
        ]
        return {
            # Keep the unrolled state in float32. Casting every iteration back
            # to autocast dtype breaks objective-path continuity and compounds
            # quantization over the ten exact updates.
            "beta": refined,
            "step_size": step_size,
            "gamma": gamma,
            "kappa": prox_diagnostics["kappa"],
            "minimum_curvature_denominator": prox_diagnostics[
                "minimum_denominator"
            ],
            "curvature_step_cap": step["curvature_step_cap"],
            "initial_lipschitz": step["initial_lipschitz"],
            "trace_lipschitz": step["trace_lipschitz"],
            "majorization_ratio": step["majorization_ratio"],
            "backtracks": step["backtracks"],
            "trace_fallback": step["trace_fallback"],
            "data_gradient_rms": data_gradient.square().mean(-1).sqrt(),
            "zero_threshold": prox_diagnostics["zero_threshold"],
            "change_rms": change.square().mean(-1).sqrt(),
            "active_fraction": (refined != 0.0).float().mean(-1),
            "saturated_fraction": prox_diagnostics["saturated_fraction"],
            "d0_contract_error": (native_d0 - d0.float()).abs(),
            "data_half_before": step["data_half_before"],
            "data_half_after": step["data_half_after"],
            "penalty_before": step["penalty_before"],
            "penalty_after": step["penalty_after"],
            "objective_before": step["objective_before"],
            "objective_after": step["objective_after"],
            "objective_delta": step["objective_delta"],
        }


def _decode_queries_kalbi_v6(
    self: nn.Module,
    encoded: dict,
    lambdas: Tensor,
    lambda_max: Tensor | None = None,
    penalty_type_id: Tensor | None = None,
    penalty_params: Tensor | None = None,
    penalty_shape_features: Tensor | None = None,
    penalty_d0: Tensor | None = None,
    return_aux: bool = False,
    return_row_state: bool = False,
    **_unused: Any,
):
    if return_row_state:
        raise ValueError("Kalbi V6 does not expose a local row-state tensor")
    lam, _lmx, d0, native_shape, prox_condition, condition = _resolve_query_context(
        self,
        encoded,
        lambdas,
        lambda_max,
        penalty_type_id,
        penalty_params,
        penalty_shape_features,
        penalty_d0,
    )
    base, alpha = _global_krylov_ridge(self, encoded, lam, condition)
    trajectory = _get_ab_trajectory(self, encoded)
    model_dtype = condition.dtype
    head_output = self.kalbi_ab_head(
        A=trajectory["A"].to(model_dtype),
        B=trajectory["B"].to(model_dtype),
        Xty=encoded["Xty"].to(model_dtype),
        gram_diag=encoded["gram_diag"].to(model_dtype),
        log_abs_Xty=encoded["log_abs_Xty"].to(model_dtype),
        base=base,
        support=trajectory["support_mask"],
        film_cond=prox_condition[..., :FEATURE_FILM_DIM].to(model_dtype),
    )
    correction = head_output["correction"].to(base.dtype)
    log_threshold_multiplier = head_output[
        "log_threshold_multiplier"
    ].float().clamp(
        KALBI_THRESHOLD_LOG_MULTIPLIER_MIN,
        KALBI_THRESHOLD_LOG_MULTIPLIER_MAX,
    )
    gram_diagonal = encoded["gram_diag"].float().clamp_min(1e-4)
    threshold_base = (
        lam.float()[..., None]
        * d0.float()[..., None]
        / gram_diagonal[:, None, :]
    )
    threshold = threshold_base * torch.exp(log_threshold_multiplier)
    threshold = torch.nan_to_num(
        threshold, nan=0.0, posinf=1e6, neginf=0.0
    ).clamp_min(0.0)
    prethreshold_beta = torch.nan_to_num(
        base.float() + correction.float(),
        nan=0.0,
        posinf=20.0,
        neginf=-20.0,
    )
    beta_unbounded = torch.sign(prethreshold_beta) * torch.relu(
        prethreshold_beta.abs() - threshold
    )
    beta_v3 = beta_unbounded.clamp(-KALBI_BETA_BOUND, KALBI_BETA_BOUND).to(
        base.dtype
    )
    feature_mask = encoded.get("feature_mask")
    if feature_mask is not None:
        beta_v3 = beta_v3 * feature_mask[:, None, :].to(beta_v3.dtype)
    beta_v3 = torch.nan_to_num(
        beta_v3,
        nan=0.0,
        posinf=KALBI_BETA_BOUND,
        neginf=-KALBI_BETA_BOUND,
    )
    refinement_history: list[dict[str, Tensor]] = []
    beta = beta_v3
    for _iteration in range(KALBI_EXACT_PG_STEPS):
        refinement = _refine_beta_kalbi_v6(
            self,
            encoded,
            beta,
            lam,
            d0,
            native_shape,
        )
        beta = refinement["beta"]
        refinement_history.append(refinement)
    if not return_aux:
        return beta

    refinement_paths = {
        name: torch.stack(
            [iteration[name] for iteration in refinement_history], dim=-1
        )
        for name in (
            "step_size",
            "gamma",
            "kappa",
            "minimum_curvature_denominator",
            "curvature_step_cap",
            "initial_lipschitz",
            "trace_lipschitz",
            "majorization_ratio",
            "backtracks",
            "trace_fallback",
            "data_gradient_rms",
            "zero_threshold",
            "change_rms",
            "active_fraction",
            "saturated_fraction",
            "d0_contract_error",
            "data_half_before",
            "data_half_after",
            "penalty_before",
            "penalty_after",
            "objective_before",
            "objective_after",
            "objective_delta",
        )
    }

    batch_size, query_count = lam.shape
    selected_indices = trajectory["selected_indices"][:, None].expand(
        batch_size,
        query_count,
        int(self.kalbi_isis_stages),
        int(self.kalbi_isis_topk),
    )
    stage_active = trajectory["stage_active"][:, None].expand(
        batch_size, query_count, int(self.kalbi_isis_stages)
    )
    result = {
        "beta_q": beta,
        "u_base_q": base,
        "u_hat_q": beta,
        "kalbi_v3_prethreshold_beta_q": prethreshold_beta,
        "kalbi_v3_correction_q": correction,
        "kalbi_v3_threshold_q": threshold,
        "kalbi_v3_threshold_base_q": threshold_base,
        "kalbi_v3_threshold_log_multiplier_q": log_threshold_multiplier,
        "kalbi_v3_active_fraction_q": (beta_v3 != 0.0).float().mean(dim=-1),
        "kalbi_v3_beta_bound": KALBI_BETA_BOUND,
        "kalbi_v6_beta0_q": beta_v3,
        "kalbi_v6_beta0_active_fraction_q": (
            beta_v3 != 0.0
        ).float().mean(dim=-1),
        "kalbi_v6_refined_active_fraction_q": refinement_paths[
            "active_fraction"
        ][..., -1],
        "kalbi_v6_refinement_steps": KALBI_EXACT_PG_STEPS,
        "kalbi_v6_refinement_contract": self.kalbi_refinement_contract,
        "alpha_q": alpha,
        "base_alpha_q": alpha,
        "cond_q": condition,
        "prox_film_cond_q": prox_condition,
        "kalbi_model_version": MODEL_VERSION,
        "kalbi_shared_support_contract": SHARED_SUPPORT_CONTRACT,
        "kalbi_v3_A": trajectory["A"],
        "kalbi_v3_B": trajectory["B"],
        "kalbi_v3_selected_indices": trajectory["selected_indices"],
        "kalbi_v3_stage_active": trajectory["stage_active"],
        "kalbi_v3_support_mask": trajectory["support_mask"],
        "kalbi_v3_residual_rms": trajectory["residual_rms"],
        "kalbi_v3_isis_ridge": trajectory["ridge"],
        "kalbi_support_identical_across_queries": True,
        "v56fast_isis_selected_indices": selected_indices,
        "v56fast_isis_stage_active": stage_active,
        "v56fast_isis_topk": int(self.kalbi_isis_topk),
        "v56fast_isis_requested_stages": int(self.kalbi_isis_stages),
        "v56fast_isis_effective_stage_counts": stage_active.sum(dim=-1),
        "v58fast_model_version": MODEL_VERSION,
        "v58fast_base_model_version": BASE_MODEL_VERSION,
        "v58fast_training_contract": TRAINING_CONTRACT,
        "v58cheb_model_version": MODEL_VERSION,
        "v58cheb_basis_version": BASIS_VERSION,
        "v58cheb_training_contract": TRAINING_CONTRACT,
    }
    diagnostic_names = {
        "step_size": "step_size",
        "gamma": "gamma",
        "kappa": "kappa",
        "min_curvature_denominator": "minimum_curvature_denominator",
        "curvature_step_cap": "curvature_step_cap",
        "initial_lipschitz": "initial_lipschitz",
        "trace_lipschitz": "trace_lipschitz",
        "majorization_ratio": "majorization_ratio",
        "backtracks": "backtracks",
        "trace_fallback": "trace_fallback",
        "data_gradient_rms": "data_gradient_rms",
        "zero_threshold": "zero_threshold",
        "change_rms": "change_rms",
        "active_fraction": "active_fraction",
        "saturated_fraction": "saturated_fraction",
        "d0_contract_error": "d0_contract_error",
        "data_half_before": "data_half_before",
        "data_half_after": "data_half_after",
        "penalty_before": "penalty_before",
        "penalty_after": "penalty_after",
        "objective_before": "objective_before",
        "objective_after": "objective_after",
        "objective_delta": "objective_delta",
    }
    for public_name, internal_name in diagnostic_names.items():
        path = refinement_paths[internal_name]
        result[f"kalbi_v6_refinement_{public_name}_path_q"] = path
        result[f"kalbi_v6_refinement_{public_name}_q"] = path[..., -1]
        result[f"kalbi_v6_refinement_{public_name}_mean_q"] = path.float().mean(
            dim=-1
        )
    result.update(
        {
            "kalbi_v6_refinement_min_curvature_denominator_min_q": (
                refinement_paths["minimum_curvature_denominator"].amin(dim=-1)
            ),
            "kalbi_v6_refinement_majorization_ratio_max_q": (
                refinement_paths["majorization_ratio"].amax(dim=-1)
            ),
            "kalbi_v6_refinement_backtracks_total_q": (
                refinement_paths["backtracks"].sum(dim=-1)
            ),
            "kalbi_v6_refinement_trace_fallback_fraction_q": (
                refinement_paths["trace_fallback"].float().mean(dim=-1)
            ),
            "kalbi_v6_refinement_objective_decrease_q": (
                refinement_paths["objective_before"][..., 0]
                - refinement_paths["objective_after"][..., -1]
            ),
        }
    )
    return result


# Compatibility entry points for launchers that previously imported older names.
patch_model_kalbi_v5 = patch_model_kalbi_v6
patch_model_v58_chebyshev = patch_model_kalbi_v6
patch_model_v58_chebyshevfast = patch_model_kalbi_v6
patch_model_v58cheb = patch_model_kalbi_v6


__all__ = [
    "BASE_MODEL_VERSION",
    "BASIS_VERSION",
    "EXPANDED_DIM",
    "FEATURE_FILM_DIM",
    "GRID_SIZE",
    "ISIS_SELECTION_VERSION",
    "KALBI_BETA_BOUND",
    "KALBI_RIDGE",
    "KALBI_EXACT_PG_BACKTRACK_FACTOR",
    "KALBI_EXACT_PG_CURVATURE_MARGIN",
    "KALBI_EXACT_PG_DATA_MARGIN",
    "KALBI_EXACT_PG_FEATURE_CHUNK",
    "KALBI_EXACT_PG_MAX_BACKTRACKS",
    "KALBI_EXACT_PG_RITZ_SAFETY",
    "KALBI_EXACT_PG_STEPS",
    "KALBI_STAGES",
    "KALBI_THRESHOLD_LOG_MULTIPLIER_MAX",
    "KALBI_THRESHOLD_LOG_MULTIPLIER_MIN",
    "KALBI_TOPK",
    "LOCAL_ISIS_GRADIENT_CONTRACT",
    "META_DIM",
    "MODEL_VERSION",
    "NATIVE_DIM",
    "PENALTY_CONTRACT_VERSION",
    "PROX_COND_DIM",
    "SHARED_SUPPORT_CONTRACT",
    "TRAINING_CONTRACT",
    "V58FAST_BASE_MODEL_VERSION",
    "V58FAST_EXPANDED_DIM",
    "V58FAST_FEATURE_FILM_DIM",
    "V58FAST_GRID_SIZE",
    "V58FAST_META_DIM",
    "V58FAST_MODEL_VERSION",
    "V58FAST_NATIVE_DIM",
    "V58FAST_PENALTY_CONTRACT_VERSION",
    "V58FAST_PROX_COND_DIM",
    "V58CHEB_BASE_MODEL_VERSION",
    "V58CHEB_BASIS_VERSION",
    "V58CHEB_MODEL_VERSION",
    "V58CHEB_PENALTY_CONTRACT_VERSION",
    "V3_SOURCE_MODEL_VERSION",
    "build_model",
    "patch_model_kalbi_v6",
    "patch_model_kalbi_v5",
    "make_fpk_config",
    "native196_to_expanded388",
    "strip_convexity_flag",
]
