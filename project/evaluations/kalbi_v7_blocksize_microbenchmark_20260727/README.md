# KALBI V7 Block-Size Microbenchmark

This directory contains the immutable six-task block-size benchmark.
The benchmark itself did not modify production training or rolling-data state.
An unrelated process stopped the V13 trainer and started V24 GPU work during
the timing window, so paired task/repetition ratios are the primary timing
result; absolute seconds have limited comparability.

## Frozen Design

- Six tasks selected before performance inspection from catalog epoch 66.
- Eight fixed raw-lambda/Direct96 queries per task.
- Five isolated timing repetitions with deterministically shuffled method order.
- Common standardized X/y and byte-identical frozen beta_initial per task.

## Decision Snapshot

- B=256 objective change vs B=4096: -2.684%.
- B=256 top-32 KKT change vs B=4096: -81.69%.
- B=256 forward-time multiple vs B=4096: 9.049.
- B=256 runtime multiple by P bin: low 4.934, mid 9.049, high 15.51.
- Adaptive retention of the B=256 objective gain: 6.02%.
- Block-3 p90 objective gap within 1% of PG10: block4096=no, block256=no, adaptive=no.
- Maximum sampled P: 36,241.
- Near-P=100K speed claim: not identified by the frozen six-task sample.
- Pareto frontier: pg10, block4096, block256, adaptive.

See `aggregate_summary.json`, the two CSV files, and `figures/` for the complete results.
