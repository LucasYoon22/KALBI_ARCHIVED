# KALBI V13 Mixed Linear Data Quality Audit

Date: 2026-07-27

## Verdict

**IMPLEMENTATION FAIL**

The controlled-linear equation, SNR calibration, mixture assignment, strict
`3k < N` guard, loader shape/dtype checks, and train/test preprocessing all
passed. The overall generator nevertheless fails two implementation contracts:

1. Full generator replay is not deterministic from its recorded seed unless
   `PYTHONHASHSEED` is fixed at process startup.
2. The final SCM-core correlation accumulator clips standardized features
   without restandardizing them before float16 emission.

There is also a separate **DISTRIBUTION DESIGN ISSUE**: conditional rejection
in the P-reference support sampler changes the nominal support-regime mixture
substantially, nearly removing its very-dense regime.

These findings are generator-contract issues, not conclusions drawn from KALBI
or baseline prediction performance.

## Scope

- Five retained catalogs, epochs 63-67: 12,392 metadata rows.
- Frozen full-array sample: 600 tasks selected before metrics by immutable task
  hash order.
- Strata: 200 linear/P, 200 linear/N, 200 TabICLv2.
- Nominal full-array bytes read: 23.75 GB.
- Support Monte Carlo: 200,000 draws.
- Noise-family tests: 200 draws per family.
- Controlled replay: 60 live-generator tasks plus 6 repeated tasks.
- No GPU used; production services were not changed or interrupted.

Frozen manifest SHA256:

`083e3337990858462f68f7348a218e88be220d6780e6b53913ae7283c6c851f8`

## What Passed

### Mixture and integrity

Active epoch 67 had:

| Stratum | Tasks | Task fraction | Byte fraction |
|---|---:|---:|---:|
| TabICLv2 | 1,236 | 49.98% | 50.50% |
| Linear/P | 618 | 24.99% | 24.09% |
| Linear/N | 619 | 25.03% | 25.41% |

- No duplicate task hashes.
- No duplicate recorded seeds.
- No assignment-rule mismatch.
- No branch fallback.
- No support-reference mismatch.
- Mixture proportions remained stable across epochs 63-67.

### Support and SNR

- Zero `3k >= N` violations in 6,200 linear metadata rows.
- Zero violations in the frozen 400 linear full-array tasks.
- Target and recorded realized SNR agreed to numerical precision.
- Counterfactual maximum SNR log-ratio error: `4.44e-16`.
- Raw linear identity maximum error: `8.88e-16`.
- Transformed/serialized identity maximum error: `3.60e-7`.
- All seven noise families were centered and scaled to unit sample variance.
- Fixed-X Monte Carlo showed monotone SNR behavior.

The five log-uniform SNR bands from `1e-2` to `1e3` are all represented.

### Arrays and response quality

All 600 frozen tasks passed:

- shape and dtype;
- finite values;
- feature variance/sigma metadata identity;
- feature-source inventory;
- branch metadata arrays;
- SNR metadata arrays;
- y finite/variance/unique/concentration/IQR gates;
- strict support guard.

Top-32 marginal-ridge R-squared was diverse:

| Stratum | Median | 25th percentile | 75th percentile |
|---|---:|---:|---:|
| Linear/P | 0.560 | 0.227 | 0.931 |
| Linear/N | 0.475 | 0.186 | 0.852 |
| TabICLv2 | 0.632 | 0.365 | 0.850 |

This does not indicate a collapsed all-easy or all-noise distribution.

### Loader and leakage

The live loader:

- reads the exact serialized task and converts float16 X to float32;
- preserves N, P, row order, and feature order;
- uses variable-shape task batch size one, so no padding enters statistics;
- checks response scale and branch metadata;
- uses disjoint train/test rows in selector preprocessing;
- computes test transforms from training rows only.

The sentinel test obtained:

- transformed linear identity maximum error: `2.07e-7`;
- train/test overlap: zero;
- test preprocessing error against training-only statistics: `2.38e-7`.

## Failure 1: Full-Generator Seed Reproducibility

Two separate live-generator processes were run with identical command line,
seed, start index, N, and P.

- 0/6 repeated tasks were array-exact.
- `X`, `y`, and correlation certificates differed.
- NumPy-controlled support, coefficient-family, noise-family, and SNR metadata
  remained equal.

The V13 script seeds Python, NumPy, and Torch for every batch, but the downstream
TabICLv2 graph path is still process-hash-sensitive. A one-task A/B replay with
`PYTHONHASHSEED=0` was exact for every NPZ array.

Consequence:

- the recorded seed does not uniquely reconstruct production X/y;
- task hashes still uniquely identify stored tasks;
- statistical validity is not invalidated, but reproducibility claims based on
  seed alone are false.

Minimal correction:

1. Set a fixed `PYTHONHASHSEED` in every producer supervisor environment before
   process startup.
2. Add a subprocess-level regression test that generates the same seed twice
   and compares every NPZ array.
3. Record `PYTHONHASHSEED`, external TabICL commit, and complete generator
   environment in each manifest.

## Failure 2: SCM-Core Feature Scale Drift

The exact correlation accumulator standardizes a candidate and then clips it,
but does not restandardize after clipping.

Across the frozen 600 tasks:

| Source | Features | std < 0.9 | std < 0.5 | std < 0.2 |
|---|---:|---:|---:|---:|
| SCM core | 1,226,965 | 3,573 (0.291%) | 87 (0.0071%) | 1 |
| BNN extension | 7,039,499 | 0 | 0 | 0 |

- 276/600 tasks had at least one core feature with std below 0.9.
- 53/600 had at least one core feature below 0.5.
- Minimum observed core feature std: 0.1793.
- Minimum observed BNN feature std: 0.99976.

The trainer standardizes X again, so model input columns return to unit scale.
However, controlled-linear y is constructed before trainer preprocessing.
Therefore low-scale core features receive smaller effective signal contribution
than the sampled beta magnitude suggests.

Minimal correction:

```text
standardize -> clip -> standardize -> float16 roundtrip
```

Then validate post-roundtrip mean/std and correlation before emission.

## Distribution Design Issue: P-Reference Conditional Rejection

Declared regime probabilities are:

```text
ultra_sparse 0.15
sparse       0.20
moderate     0.25
large        0.25
very_dense   0.15
```

The 200,000-draw sampler audit produced:

| Regime | P-reference | N-reference |
|---|---:|---:|
| Ultra sparse | 21.98% | 15.26% |
| Sparse | 26.83% | 20.09% |
| Moderate | 33.73% | 24.62% |
| Large | 16.40% | 24.85% |
| Very dense | 1.06% | 15.18% |

The live five-catalog census was even more extreme for P-reference very-dense
tasks: 0.84%.

This is the correct distribution induced by repeated conditional rejection,
not a coding accident. It does not preserve the declared regime weights after
conditioning on `k < N/3`.

Recommended redesign:

- choose the regime from probabilities already conditioned on N and P; or
- sample a target `k/k_max` distribution directly; and
- log both requested and realized regime distributions as explicit contracts.

## Provenance Gap

Production NPZ files omit true beta, support indices, signal, and noise. Thus
the following cannot be independently verified from historical shards:

- exact support/nonzero equality;
- exact `y = X beta + epsilon` reconstruction;
- residual-feature correlation;
- active-design condition number;
- coefficient sign and magnitude quantiles.

Counterfactual unit tests validate the current functions, but they are not a
substitute for production-shard identity checks.

Minimal correction: store a compact sidecar containing support indices,
nonzero beta, response transform, and deterministic noise seed. Full length-P
beta is unnecessary.

## Checkpoint Impact

The existing checkpoint has learned from:

- a mathematically calibrated and broad SNR/noise distribution;
- an underrepresented P-reference dense-support regime;
- a small fraction of mis-scaled SCM-core features before y construction;
- non-reproducible X/y for a given seed.

Immediate checkpoint deletion is not justified. The feature-scale issue affects
0.291% of core columns and is corrected at model input, while SNR/support guards
are valid. However:

- continuing after changing the generator creates a data-distribution boundary;
- a clean paper-grade run should start from scratch after generator V14 is
  frozen and replay-tested;
- for exploratory training, checkpoint continuation is technically possible if
  the transition step and old/new data mixture are recorded.

No correction was applied during this audit.

## Production Status

Audit start:

- trainer, curator, and 8 producers were RUNNING;
- active catalog epoch 67.

Audit end:

- trainer, curator, and 8 producers remained RUNNING;
- active catalog advanced to epoch 68;
- latest beta checkpoint reached step 42,800.

The audit did not stop, signal, modify, or reload production services.

## Figures

- `figures/mixture_support_snr.png` and `.pdf`
- `figures/response_feature_quality.png` and `.pdf`
- `figures/drift_filter_reproducibility.png` and `.pdf`

## Reproducibility Files

- `frozen_audit_manifest.csv/json`
- `metadata_census.csv`
- `task_metrics.csv/json`
- `distribution_summary.csv`
- `feature_scaling_task_metrics.csv`
- `feature_scaling_summary.json`
- `generator_unit_test_results.json`
- `loader_sentinel_results.json`
- `replay/replay_verification.json`
- exact audit scripts under `scripts/`
- `SHA256SUMS`
