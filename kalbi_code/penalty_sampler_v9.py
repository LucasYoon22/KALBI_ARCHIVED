"""Penalty generator V9: identical to V8 logic, but the flat-tail onset t_flat range is
[0.02, 5.0] (was V8 [0.1, 5.0]) -> lower bound 0.02 lets the penalty flatten from very small t.
Sampling of t_flat is LOG-UNIFORM (inherited from V8: t_flat = exp(uniform(log lo, log hi))),
so small onsets are well represented."""
from dataclasses import dataclass
from typing import Optional, Tuple
from penalty_sampler_v8 import SamplerConfigV8, PenaltySamplerV8


@dataclass
class SamplerConfigV9(SamplerConfigV8):
    flat_tail_t_range: Tuple[float, float] = (0.02, 5.0)


class PenaltySamplerV9(PenaltySamplerV8):
    def __init__(self, cfg: Optional[SamplerConfigV9] = None, seed: int = 42):
        super().__init__(cfg if cfg is not None else SamplerConfigV9(), seed=seed)


def make_v9(seed: int = 42, **cfg_kwargs) -> PenaltySamplerV9:
    return PenaltySamplerV9(SamplerConfigV9(**cfg_kwargs), seed=seed)
