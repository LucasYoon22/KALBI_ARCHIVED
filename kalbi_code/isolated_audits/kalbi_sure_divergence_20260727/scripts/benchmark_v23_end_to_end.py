#!/usr/bin/env python3
"""Low-priority CPU benchmark for V23 plus conditional solver summaries."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

import run_sure_divergence_study as study


def timed(callable_, repeats: int = 3):
    durations = []
    value = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = callable_()
        durations.append(time.perf_counter() - start)
    return value, float(np.median(durations))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    selector = study.load_v23_selector(Path(args.checkpoint))
    rows = []
    for threads in (1, 16):
        torch.set_num_threads(threads)
        for n, p in ((200, 500), (500, 1000), (1000, 5000), (500, 20000)):
            rng = np.random.default_rng(20260727 + n + p)
            x = torch.from_numpy(study.make_design(rng, n, p, 0.3)).float()
            y = torch.from_numpy(rng.normal(size=n)).float()
            lambda_max = (x.T @ y / float(n)).abs().amax().reshape(1)

            def selector_call():
                with torch.no_grad():
                    return selector(
                        x[None],
                        y[None],
                        lambda_max,
                        return_representation=False,
                    )

            selector_result = selector_call()
            selector_result, selector_seconds = timed(selector_call)
            derivative = selector_result["penalty_drho"][0].float()
            raw_lambda = float(selector_result["raw_lambda"][0])

            vector = torch.ones(p)
            vector /= vector.norm().clamp_min(1e-12)
            for _ in range(12):
                vector = x.T @ (x @ vector) / float(n)
                vector /= vector.norm().clamp_min(1e-12)
            eta = (
                0.9
                / torch.dot(
                    vector, x.T @ (x @ vector) / float(n)
                ).clamp_min(1e-8)
            ).detach()

            base_call = lambda: study.solve_pg(
                x,
                y,
                derivative,
                raw_lambda,
                collect=False,
                eta=eta,
            )
            compact_call = lambda: study.solve_pg(
                x,
                y,
                derivative,
                raw_lambda,
                collect=True,
                compact=True,
                eta=eta,
            )
            base_call()
            compact_call()
            _, base_seconds = timed(base_call)
            compact_result, compact_seconds = timed(compact_call)
            total_base = selector_seconds + base_seconds
            total_compact = selector_seconds + compact_seconds
            rows.append(
                {
                    "n": n,
                    "p": p,
                    "threads": threads,
                    "selector_seconds": selector_seconds,
                    "refinement_seconds": base_seconds,
                    "compact_refinement_seconds": compact_seconds,
                    "end_to_end_base_seconds": total_base,
                    "end_to_end_compact_seconds": total_compact,
                    "end_to_end_incremental_fraction": (
                        total_compact / max(total_base, 1e-12) - 1.0
                    ),
                    "compact_summary_width": len(compact_result[1]),
                    "timing_repeats": 3,
                }
            )
    keys = list(rows[0])
    with (output / "v23_end_to_end_cpu.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (output / "v23_end_to_end_cpu.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
