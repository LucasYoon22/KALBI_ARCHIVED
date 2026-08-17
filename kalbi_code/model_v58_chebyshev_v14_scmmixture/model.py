"""Standalone V58-Chebyshev selector-gradient model and decoder.

The old implementation was a runtime patch chain

``V33 -> V50fast -> V55fast -> V56fast -> V57fast``.

This file preserves the final V57 topology and checkpoint schema directly,
while replacing local-ISIS backward semantics with a fixed-support
differentiable path and its local MGS basis with the exact row-bound,
eigensolve-free Chebyshev three-term recurrence used by V57-Chebyshev.  It imports only the
stable ``model_fpk`` bootstrap and the adjacent ``utils.py``; no versioned
model module is required at import time or runtime.
"""

from __future__ import annotations

import types
from typing import Any

import torch
import torch.nn as nn

import model_fpk
from model_fpk import FpkConfig, make_fpk_model
import shape_feature_normalization

from .utils import (
    EXPANDED_DIM,
    FEATURE_FILM_DIM,
    GRID_SIZE,
    LOCAL_ISIS_GRADIENT_CONTRACT,
    META_DIM,
    NATIVE_DIM,
    PROX_COND_DIM,
    ChunkedSharedFeatureEmissionHead,
    ResidualMLP,
    ResidualPenaltyShapeEncoder,
    ResidualQueryMixerGramAware,
    make_penalty_grid,
    make_prox_film_cond,
    native196_to_expanded388,
    strip_convexity_flag,
    support_memory_local_mixer_correction,
)


Tensor = torch.Tensor

MODEL_VERSION = (
    "v58cheb_native196_fixed_support_selector_gradient_"
    "chebyshev_rowbound_recurrence_v1"
)
BASE_MODEL_VERSION = "v57chebyshev_native196_recurrence_mixer_v1"
V56_BASE_MODEL_VERSION = "v56fast_native196_no_convexity_flag"
PENALTY_CONTRACT_VERSION = (
    "native196_interleaved_rho_drho_meta4_expanded388_v1"
)
TRAINING_CONTRACT = (
    "log1p_F_over_F0_lambda_floor_1e-6_top32_kkt_batched_global32_"
    "joint_v10_v1"
)
LEGACY_BASE_MODEL_VERSION = "v50fast"
BASIS_VERSION = "chebyshev_rowbound_recurrence_l2_mixer_only_v58cheb"
ISIS_SELECTION_VERSION = "global_disjoint_exact_topk_per_active_stage_v1"

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
    """Reproduce the V57 checkpoint bootstrap configuration exactly."""
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
        zero_init_query_mixer=bool(
            model_config.get("zero_init_query_mixer", True)
        ),
        zero_init_feature_head=bool(
            model_config.get("zero_init_feature_head", True)
        ),
        zero_init_learned_prox=bool(
            model_config.get("zero_init_learned_prox", True)
        ),
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


def patch_model_v58_chebyshev(
    model: nn.Module,
    device: str | torch.device | None = None,
    *,
    prox_hidden: int = 512,
    film_hidden: int = 256,
    isis_topk: int = 32,
    isis_steps: int = 32,
    isis_stages: int = 28,
    isis_scale: float = 0.38,
    delta_ratio: float = 0.24,
    delta_floor: float = 0.05,
    beta_abs_cap: float = 3.0,
    beta_bonus: float = 0.50,
    mixer_hidden: int = 512,
    learned_scale: float = 0.25,
    max_local_steps: int = 32,
) -> nn.Module:
    """Build V58-Chebyshev with the exact V57 parameter/state schema."""
    del prox_hidden  # retained only for API/checkpoint-launch compatibility
    if int(isis_topk) <= 0:
        raise ValueError(f"isis_topk must be positive, got {isis_topk}")
    if int(max_local_steps) <= 0:
        raise ValueError(
            f"max_local_steps must be positive, got {max_local_steps}"
        )
    if not 1 <= int(isis_steps) <= min(int(isis_topk), int(max_local_steps)):
        raise ValueError(
            "isis_steps must satisfy 1 <= steps <= min(topk,max_local_steps), got "
            f"steps={isis_steps}, topk={isis_topk}, "
            f"max_local_steps={max_local_steps}"
        )
    if int(isis_stages) < 0:
        raise ValueError(f"isis_stages must be nonnegative, got {isis_stages}")
    if float(isis_scale) < 0.0 or float(delta_ratio) < 0.0:
        raise ValueError("isis_scale and delta_ratio must be nonnegative")
    if float(delta_floor) < 0.0 or float(beta_abs_cap) <= 0.0:
        raise ValueError(
            "delta_floor must be nonnegative and beta_abs_cap positive"
        )
    if float(beta_bonus) < 0.0 or int(mixer_hidden) <= 0:
        raise ValueError(
            "beta_bonus must be nonnegative and mixer_hidden positive"
        )
    stale_chebyshev = [
        name for name in vars(model) if name.startswith("v51_cheb_")
    ]
    if stale_chebyshev:
        raise ValueError(
            "V58 must be patched from a fresh pre-Chebyshev model; found "
            f"stale attributes: {stale_chebyshev}"
        )

    target_device = (
        torch.device(device)
        if device is not None
        else next(model.parameters()).device
    )
    cfg = model.cfg
    cfg.retain_centered_data_in_encoding = True
    code_dim = int(cfg.cond_dim)

    # Final native196 condition path.
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

    model.base_head.net = ResidualMLP(
        in_dim=code_dim,
        hidden=max(256, int(getattr(cfg, "base_head_hidden", 256))),
        out_dim=2,
        depth=3,
        zero_init_last=True,
        act=nn.GELU,
    ).to(target_device)

    model.feature_head = ChunkedSharedFeatureEmissionHead(
        num_channels=int(cfg.num_decoder_channels),
        cond_dim=code_dim,
        hidden=int(cfg.feature_head_hidden),
        tau_scale_bound=getattr(cfg, "tau_scale_bound", 2.0),
        zero_init=bool(cfg.zero_init_feature_head),
        use_bilinear=bool(cfg.use_bilinear_channels),
        film_dim=FEATURE_FILM_DIM,
        film_hidden=max(512, int(film_hidden)),
        feature_chunk=4096,
        use_checkpoint=True,
    ).to(target_device)

    krylov_dim = int(cfg.krylov_dim)
    model.query_mixer = ResidualQueryMixerGramAware(
        cond_dim=PROX_COND_DIM,
        num_channels=int(cfg.num_decoder_channels),
        krylov_dim=krylov_dim,
        t_feat_dim=krylov_dim * (krylov_dim + 1) // 2,
        hidden=256,
        proj_dim=256,
        mix_scale=2.0,
        zero_init=bool(cfg.zero_init_query_mixer),
    ).to(target_device)

    model.v46_local_mixer = ResidualMLP(
        in_dim=9 + int(max_local_steps),
        hidden=int(mixer_hidden),
        out_dim=int(max_local_steps),
        depth=3,
        zero_init_last=True,
        act=nn.SiLU,
    ).to(target_device)

    _set_buffer(
        model,
        "_logt_grid",
        (torch.log(make_penalty_grid() + 1e-5) / 3.0).to(target_device),
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
    model.v46_isis_topk = int(isis_topk)
    model.v46_isis_steps = int(isis_steps)
    model.v46_isis_stages = int(isis_stages)
    model.v46_isis_scale = float(isis_scale)
    model.v46_delta_ratio = float(delta_ratio)
    model.v46_delta_floor = float(delta_floor)
    model.v46_beta_abs_cap = float(beta_abs_cap)
    model.v46_beta_bonus = float(beta_bonus)
    model.v46_learned_scale = float(learned_scale)
    model.v46_max_local_steps = int(max_local_steps)
    model.v58cheb_basis_version = BASIS_VERSION
    model.v55fast_base_model_version = LEGACY_BASE_MODEL_VERSION
    model.v55fast_basis_version = BASIS_VERSION
    model.v55fast_isis_selection_version = ISIS_SELECTION_VERSION
    model.v56fast_model_version = V56_BASE_MODEL_VERSION
    model.v56fast_penalty_contract_version = PENALTY_CONTRACT_VERSION
    model.v57fast_model_version = BASE_MODEL_VERSION
    model.v57fast_base_model_version = V56_BASE_MODEL_VERSION
    model.v57fast_penalty_contract_version = PENALTY_CONTRACT_VERSION
    model.v58fast_model_version = MODEL_VERSION
    model.v58fast_base_model_version = BASE_MODEL_VERSION
    model.v58fast_penalty_contract_version = PENALTY_CONTRACT_VERSION
    model.v58fast_local_isis_gradient_contract = LOCAL_ISIS_GRADIENT_CONTRACT
    model.v58cheb_model_version = MODEL_VERSION
    model.v58cheb_base_model_version = BASE_MODEL_VERSION
    model.v58cheb_penalty_contract_version = PENALTY_CONTRACT_VERSION
    model.v58cheb_local_isis_gradient_contract = LOCAL_ISIS_GRADIENT_CONTRACT
    model.decode_queries = types.MethodType(_decode_queries_v58, model)
    return model


def build_model(
    config: dict[str, Any], device: str | torch.device = "cpu"
) -> nn.Module:
    """Build V58 directly from the stable bootstrap configuration."""
    # The legacy V55 trainer installed this native197 bootstrap contract as an
    # import-time side effect before calling ``make_fpk_model``.  Keep the
    # necessary constructor setup local and explicit so a fresh process does
    # not depend on importing that trainer (or any versioned model module).
    shape_feature_normalization.SHAPE_FEATURE_MEAN = [0.0] * 197
    shape_feature_normalization.SHAPE_FEATURE_SCALE = [1.0] * 197
    shape_feature_normalization.FEATURE_DIM = 197
    bootstrap_config = make_fpk_config(config)
    model, _ = make_fpk_model(bootstrap_config)
    patch_model_v58_chebyshev(
        model,
        device=device,
        prox_hidden=512,
        film_hidden=256,
        isis_topk=32,
        isis_steps=32,
        isis_stages=28,
        isis_scale=0.38,
        delta_ratio=0.24,
        delta_floor=0.05,
        beta_abs_cap=3.0,
        beta_bonus=0.50,
        mixer_hidden=512,
        learned_scale=0.25,
        max_local_steps=32,
    )
    if NATIVE_DIM != 196 or EXPANDED_DIM != 388:
        raise RuntimeError("V58 native/expanded penalty contract drifted")
    if PROX_COND_DIM != 490 or FEATURE_FILM_DIM != 394:
        raise RuntimeError("V58 conditioning dimensions drifted")
    return model.to(device)


def _decode_queries_base(
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
):
    del return_row_state
    patched = self._patched_cfg
    if not patched.use_penalty_shape_encoding:
        return self.__class__.decode_queries(
            self,
            encoded,
            lambdas,
            lambda_max,
            penalty_type_id,
            penalty_params,
            return_aux,
            False,
        )
    batch_size, feature_count, sample_count = (
        encoded["B"],
        encoded["P"],
        encoded["N"],
    )
    device = encoded["Xty"].device
    lam = model_fpk._as_2d_queries(
        lambdas,
        batch_size,
        "lambdas",
        device=device,
        dtype=torch.float32,
    )
    _, query_count = lam.shape
    lambda_max_source = encoded["lambda_max"] if lambda_max is None else lambda_max
    lmx = model_fpk._broadcast_queries_like(
        lambda_max_source,
        batch_size,
        query_count,
        "lambda_max",
        device=device,
        dtype=torch.float32,
    )

    if penalty_shape_features is None or penalty_d0 is None:
        if penalty_type_id is None:
            raise ValueError(
                "V57 decode requires penalty_shape_features + penalty_d0 "
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
    )
    condition = self.cond_embed(
        lambdas=lam,
        lambda_max=lmx,
        N=float(sample_count),
        P=float(feature_count),
        penalty_type_id=type_for_embedding,
        penalty_shape_features=penalty_shape_features,
        penalty_d0=d0,
    )

    krylov_matrix = encoded["T_krylov"]
    krylov_dim = krylov_matrix.shape[-1]
    upper = torch.triu_indices(
        krylov_dim, krylov_dim, offset=0, device=krylov_matrix.device
    )
    krylov_features = krylov_matrix[:, upper[0], upper[1]].to(torch.float32)
    mixing = self.query_mixer(
        prox_condition.view(batch_size, query_count, -1), krylov_features
    )
    with model_fpk._maybe_disable_autocast(device.type):
        channels = torch.einsum(
            "bqcm,bmp->bqcp",
            mixing.float(),
            encoded["basis_proj"].float(),
        )

    model_dtype = condition.dtype
    channels = channels.to(model_dtype)
    Xty = encoded["Xty"].to(model_dtype)
    gram_diag = encoded["gram_diag"].to(model_dtype)
    log_abs_Xty = encoded["log_abs_Xty"].to(model_dtype)
    denominator = gram_diag.clamp_min(self.cfg.diag_eps)
    diagonal_solution = Xty[:, None, :] / denominator[:, None, :]
    tau_base = (
        lam.to(model_dtype).unsqueeze(-1)
        * d0.to(model_dtype).unsqueeze(-1)
        / denominator[:, None, :]
    )

    if self.cfg.use_krylov_ridge_base:
        base_raw = self.base_head(condition)
        log_alpha_scale = base_raw[..., 0].clamp(
            self.cfg.base_alpha_log_min, self.cfg.base_alpha_log_max
        )
        mix = torch.sigmoid(base_raw[..., 1])
        alpha = lam * torch.exp(log_alpha_scale) + self.cfg.diag_eps
        with model_fpk._maybe_disable_autocast(device.type):
            T = encoded["T_krylov"].float()
            Uy = encoded["Uy_krylov"].float()
            Phi = encoded["basis_proj"].float()
            width = T.shape[-1]
            eye = torch.eye(
                width, device=device, dtype=torch.float32
            ).view(1, 1, width, width)
            rhs = Uy[:, None, :, None].expand(
                batch_size, query_count, width, 1
            )
            system = (
                T[:, None, :, :]
                + alpha[..., None, None].float() * eye
                + self.cfg.diag_eps * eye
            )
            coefficients = torch.linalg.solve(system, rhs).squeeze(-1)
            krylov_solution = torch.einsum("bqm,bmp->bqp", coefficients, Phi)
        base_solution = mix.unsqueeze(-1) * krylov_solution.to(model_dtype) + (
            1.0 - mix.unsqueeze(-1)
        ) * diagonal_solution
    else:
        mix = torch.zeros(
            batch_size, query_count, device=device, dtype=model_dtype
        )
        alpha = lam.new_full((batch_size, query_count), float("nan"))
        base_solution = diagonal_solution

    residual_solution, log_tau_scale = self.feature_head(
        S=channels,
        Xty=Xty,
        gram_diag=gram_diag,
        log_abs_Xty=log_abs_Xty,
        cond=condition,
        film_cond=prox_condition.view(batch_size, query_count, -1)[
            ..., :FEATURE_FILM_DIM
        ],
        u_base=base_solution,
    )
    u_hat = base_solution + residual_solution
    tau_hat = tau_base * torch.exp(log_tau_scale)
    beta = torch.sign(u_hat) * torch.relu(
        u_hat.abs() - tau_hat.clamp_min(0.0)
    )
    feature_mask = encoded["feature_mask"]
    if feature_mask is not None:
        beta = beta * feature_mask[:, None, :].to(beta.dtype)
        u_hat = u_hat * feature_mask[:, None, :].to(u_hat.dtype)
    if not return_aux:
        return beta
    return {
        "beta_q": beta,
        "u_base_q": base_solution,
        "u_hat_q": u_hat,
        "tau_hat_q": tau_hat,
        "mix_q": mix,
        "alpha_q": alpha,
        "cond_q": condition,
        "prox_film_cond_q": prox_condition.view(
            batch_size, query_count, -1
        ),
    }


def _decode_queries_v58_core(
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
    isis_support_override: dict[str, Tensor] | None = None,
):
    aux = _decode_queries_base(
        self,
        encoded,
        lambdas,
        lambda_max,
        penalty_type_id,
        penalty_params,
        penalty_shape_features,
        penalty_d0,
        return_aux=True,
        return_row_state=False,
    )
    beta = aux["beta_q"]
    batch_size = encoded["B"]
    device = encoded["Xty"].device
    lam = model_fpk._as_2d_queries(
        lambdas,
        batch_size,
        "lambdas",
        device=device,
        dtype=torch.float32,
    )
    _, query_count = lam.shape
    if penalty_d0 is None:
        if penalty_type_id is None:
            raise ValueError("V58 decode requires penalty_d0 or penalty_type_id")
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
        penalty_d0 = model_fpk.make_penalty_features(
            penalty_type_id=resolved_type,
            penalty_params=resolved_parameters,
        )["d0"]
    d0 = model_fpk._broadcast_queries_like(
        penalty_d0,
        batch_size,
        query_count,
        "penalty_d0",
        device=device,
        dtype=torch.float32,
    ).clamp_min(0.0)

    topk = int(self.v46_isis_topk)
    steps = int(self.v46_isis_steps)
    stages = int(self.v46_isis_stages)
    scale = float(self.v46_isis_scale)
    delta_ratio = float(self.v46_delta_ratio)
    delta_floor = float(self.v46_delta_floor)
    beta_abs_cap = float(self.v46_beta_abs_cap)
    beta_bonus = float(self.v46_beta_bonus)
    feature_count = int(beta.shape[-1])
    if topk <= 0 or feature_count < topk:
        raise ValueError(f"invalid V58 TopK={topk} for P={feature_count}")
    feature_mask = encoded.get("feature_mask")
    if feature_mask is None:
        valid_mask = torch.ones(
            batch_size,
            query_count,
            feature_count,
            dtype=torch.bool,
            device=beta.device,
        )
    else:
        if feature_mask.shape != (batch_size, feature_count):
            raise ValueError(
                f"feature_mask must be (B,P), got {feature_mask.shape}"
            )
        valid_mask = feature_mask[:, None, :].to(torch.bool).expand(
            batch_size, query_count, feature_count
        )
    selected_mask = torch.zeros_like(valid_mask)
    stage_indices: list[Tensor] = []
    stage_active: list[Tensor] = []
    override_indices: Tensor | None = None
    override_active: Tensor | None = None
    if isis_support_override is not None:
        if not isinstance(isis_support_override, dict):
            raise TypeError("isis_support_override must be a mapping")
        override_indices = isis_support_override.get("selected_indices")
        override_active = isis_support_override.get("stage_active")
        if override_indices is None:
            override_indices = isis_support_override.get(
                "v56fast_isis_selected_indices"
            )
        if override_active is None:
            override_active = isis_support_override.get(
                "v56fast_isis_stage_active"
            )
        if not torch.is_tensor(override_indices) or not torch.is_tensor(
            override_active
        ):
            raise ValueError(
                "support override requires selected_indices and stage_active tensors"
            )
        expected_indices = (batch_size, query_count, stages, topk)
        expected_active = (batch_size, query_count, stages)
        if tuple(override_indices.shape) != expected_indices:
            raise ValueError(
                f"support override indices must be {expected_indices}, got "
                f"{tuple(override_indices.shape)}"
            )
        if tuple(override_active.shape) != expected_active:
            raise ValueError(
                f"support override active must be {expected_active}, got "
                f"{tuple(override_active.shape)}"
            )
        override_indices = override_indices.detach().to(
            device=beta.device, dtype=torch.long
        )
        override_active = override_active.detach().to(
            device=beta.device, dtype=torch.bool
        )

    with model_fpk._maybe_disable_autocast(device.type):
        for stage in range(max(0, stages)):
            can_run = (valid_mask & ~selected_mask).sum(dim=-1) >= topk
            if override_indices is None and not bool(can_run.any()):
                remaining = stages - stage
                inactive_indices = torch.full(
                    (batch_size, query_count, topk),
                    -1,
                    dtype=torch.long,
                    device=beta.device,
                )
                inactive = torch.zeros(
                    batch_size,
                    query_count,
                    dtype=torch.bool,
                    device=beta.device,
                )
                stage_indices.extend([inactive_indices] * remaining)
                stage_active.extend([inactive] * remaining)
                break
            correction, indices, active = support_memory_local_mixer_correction(
                self,
                encoded,
                beta,
                lam,
                d0,
                aux["alpha_q"],
                topk,
                steps,
                beta_bonus,
                stage_frac=(stage + 1.0) / max(1.0, float(stages)),
                valid_mask=valid_mask,
                selected_mask=selected_mask,
                fixed_indices=(
                    override_indices[:, :, stage, :]
                    if override_indices is not None
                    else None
                ),
                fixed_active=(
                    override_active[:, :, stage]
                    if override_active is not None
                    else None
                ),
            )
            new_mask = torch.zeros_like(selected_mask)
            new_mask.scatter_(-1, indices, True)
            selected_mask |= new_mask & active.unsqueeze(-1)
            stage_indices.append(
                indices.masked_fill(~active.unsqueeze(-1), -1)
            )
            stage_active.append(active)
            delta = scale * correction.to(beta.dtype)
            beta_norm = beta.float().norm(dim=-1, keepdim=True)
            delta_norm = delta.float().norm(dim=-1, keepdim=True).clamp_min(1e-8)
            max_delta = delta_ratio * (beta_norm + delta_floor)
            gate = (max_delta / delta_norm).clamp(max=1.0).to(delta.dtype)
            beta = torch.nan_to_num(
                beta + delta * gate,
                nan=0.0,
                posinf=beta_abs_cap,
                neginf=-beta_abs_cap,
            ).clamp(-beta_abs_cap, beta_abs_cap)

    if feature_mask is not None:
        beta = beta * feature_mask[:, None, :].to(beta.dtype)
    if not return_aux:
        return beta
    aux["beta_q"] = beta
    if stage_indices:
        selected_indices = torch.stack(stage_indices, dim=2)
        active_stages = torch.stack(stage_active, dim=2)
    else:
        selected_indices = torch.empty(
            batch_size,
            query_count,
            0,
            topk,
            dtype=torch.long,
            device=beta.device,
        )
        active_stages = torch.empty(
            batch_size,
            query_count,
            0,
            dtype=torch.bool,
            device=beta.device,
        )
    aux.update(
        {
            "v56fast_model_version": V56_BASE_MODEL_VERSION,
            "v56fast_penalty_contract_version": PENALTY_CONTRACT_VERSION,
            "v56fast_basis_version": BASIS_VERSION,
            "v56fast_isis_selection_version": ISIS_SELECTION_VERSION,
            "v56fast_isis_topk": topk,
            "v56fast_isis_requested_stages": stages,
            "v56fast_isis_effective_stage_counts": active_stages.sum(dim=-1),
            "v56fast_isis_selected_indices": selected_indices,
            "v56fast_isis_stage_active": active_stages,
            "v56fast_local_steps": steps,
            "v58fast_model_version": MODEL_VERSION,
            "v58fast_base_model_version": BASE_MODEL_VERSION,
            "v58fast_local_isis_gradient_contract": LOCAL_ISIS_GRADIENT_CONTRACT,
            "v58fast_isis_support_override_used": override_indices is not None,
            "v58cheb_model_version": MODEL_VERSION,
            "v58cheb_basis_version": BASIS_VERSION,
            "v58cheb_local_isis_gradient_contract": LOCAL_ISIS_GRADIENT_CONTRACT,
        }
    )
    return aux


def _decode_queries_v58(self: nn.Module, *args, **kwargs):
    result = _decode_queries_v58_core(self, *args, **kwargs)
    if isinstance(result, dict):
        result["v57fast_model_version"] = BASE_MODEL_VERSION
        result["v57fast_base_model_version"] = V56_BASE_MODEL_VERSION
        result["v57fast_training_contract"] = TRAINING_CONTRACT
        result["v58fast_model_version"] = MODEL_VERSION
        result["v58fast_base_model_version"] = BASE_MODEL_VERSION
        result["v58fast_training_contract"] = TRAINING_CONTRACT
        result["v58fast_local_isis_gradient_contract"] = (
            LOCAL_ISIS_GRADIENT_CONTRACT
        )
        result["v58cheb_model_version"] = MODEL_VERSION
        result["v58cheb_basis_version"] = BASIS_VERSION
        result["v58cheb_training_contract"] = TRAINING_CONTRACT
        result["v58cheb_local_isis_gradient_contract"] = (
            LOCAL_ISIS_GRADIENT_CONTRACT
        )
    return result


patch_model_v58_chebyshevfast = patch_model_v58_chebyshev
patch_model_v58cheb = patch_model_v58_chebyshev


__all__ = [
    "BASE_MODEL_VERSION",
    "EXPANDED_DIM",
    "FEATURE_FILM_DIM",
    "GRID_SIZE",
    "LOCAL_ISIS_GRADIENT_CONTRACT",
    "META_DIM",
    "MODEL_VERSION",
    "NATIVE_DIM",
    "PENALTY_CONTRACT_VERSION",
    "PROX_COND_DIM",
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
    "build_model",
    "make_fpk_config",
    "native196_to_expanded388",
    "patch_model_v58_chebyshev",
    "patch_model_v58_chebyshevfast",
    "patch_model_v58cheb",
    "strip_convexity_flag",
]
