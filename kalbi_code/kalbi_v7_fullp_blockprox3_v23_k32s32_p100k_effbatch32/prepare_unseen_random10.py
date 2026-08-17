#!/usr/bin/env python3
"""Freeze unseen tasks from the unconditioned rolling shape distribution."""

from __future__ import annotations

from . import prepare_paper_unseen_dimension5 as base


def main() -> None:
    base.DIMENSION_REQUIREMENTS["random"] = {
        "sampling": "unconditioned rolling shape distribution"
    }
    original = base.passes_dimension_regime

    def passes(n: int, p: int, regime: str) -> bool:
        if regime == "random":
            return True
        return original(n, p, regime)

    base.passes_dimension_regime = passes
    base.main()


if __name__ == "__main__":
    main()
