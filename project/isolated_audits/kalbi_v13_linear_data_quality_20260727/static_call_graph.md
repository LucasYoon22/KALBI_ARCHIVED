# KALBI V13 Live Data Call Graph

Date: 2026-07-27

## Live Process

`supervisord`
-> `kalbi_v13_beta_trainer`
-> `/venv/main/bin/torchrun --nproc_per_node=8`
-> `kalbi_v6_exactprox10_eta4x_v23_independentraw_k32s32_p100k_effbatch32.train_beta_only_v23_supportmix_v13_transition_8gpu`

The live trainer uses:

- rolling root: `/users/sync_data/kalbi_tabiclv2_v13_mixed50_linear_pn50_strictn3_p100k_100GB_target3080_seed160076430`
- expected pipeline SHA256: `99f81c6a5bf5083a56f3268a56a0700cd64865b1ff0bba9b23386088d11a9559`
- effective batch size: 8 ranks x 1 task x 4 accumulation = 32
- rolling reload/checkpoint interval: 100 optimizer updates
- resume checkpoint: `checkpoint_step033800.pt`
- audit-end checkpoint: `checkpoint_step042800.pt`

## Rolling Publication

`kalbi_v13_beta_producer_00..07`
-> producer scratch NPZ
-> curator validation
-> immutable catalog
-> `rolling_control/CURRENT.json`
-> rank-sharded variable-shape stream

Audit-start frozen catalog:

- epoch: 67
- catalog SHA256: `9704470324f3ebcb383cce936225cdd71882ddf68d22b23b877c03f30d8cf0c0`
- tasks: 2,473
- bytes: 99,990,285,286

Audit-end live catalog:

- epoch: 68
- catalog SHA256: `57dbc17b5b4327e2337e1b4d6edd1e5267de18cd41ac6d42b9804aa24473c57b`

## Generator

Main generator:

`/users/minsuk/fm-lm-lambda/generate_tabiclv2_scm2048_qfilter_bnn128_mixedlinear_supportmix_v13.py`

SHA256: `c32b2b69e4f46366a65a738a618ded1dc388d595ccc03ca47079e7ae0cae0c3d`

Call graph:

1. Shape sampling: generator lines 977-1004.
2. TabICLv2 graph-SCM base: lines 1034-1041.
3. TabICL response quality filtering: lines 1043-1051; implementation lines 197-262.
4. Exact core correlation filtering: lines 1053-1069.
5. Per-feature Gaussian noise: lines 1071-1079; implementation lines 602-634.
6. Noisy-core correlation verification: lines 1080-1096.
7. BNN extension in 128-feature blocks: lines 1107-1140.
8. Feature permutation: lines 1146-1160.
9. Outcome and support-reference assignment: lines 1161-1164.
10. Controlled linear response: lines 1173-1184; implementation lines 494-599.
11. Final sampled correlation audit: lines 1212-1224.
12. Serialization: lines 1229-1267.

Dependencies:

- TabICLv2 wrapper: `/users/minsuk/fm-lm-lambda/generate_tabiclv2_graph_highdim_npz.py`
  - SHA256: `66ef1574b54cf186456dff8c3fc7f0dc9ad42dbc91b95bb63c87eed77122c4b3`
- correlation accumulator and `standardize`: `/users/minsuk/fm-lm-lambda/generate_tabpfn_v1_causal_scm_mlp512_groupdrop095_sparse_linear_v1.py`
  - SHA256: `846384ac4b4497515248da767325d3aedb0ee77a4844dbd92fa997597bed650a`

## Exact Mixture Contract

For each task:

```text
outcome_source_id = (batch_seed + task_index) & 1
support_reference_id = ((batch_seed + task_index) >> 1) & 1
```

This yields a deterministic 50% TabICL, 25% linear/P, 25% linear/N stream. A
60-consecutive-task live replay therefore yields 30/15/15, not 20/20/20.

## Controlled Linear Contract

Support:

```text
k_max = min(P, floor((N - 1) / 3))
support = sorted(choice(P, size=k, replace=False))
assert 3*k < N
```

The P-reference branch samples from the P-scale regime mixture and conditionally
rejects draws with `k > k_max`. The N-reference branch samples directly on the
N scale within `k_max`.

Signal and SNR:

```text
signal = X @ beta
signal = signal - mean(signal)
beta = beta / std(signal)
signal = signal / std(signal)

base_error = center_and_scale(realized_error)
noise = base_error / sqrt(target_snr)
y_raw = signal + noise
realized_snr = var(signal) / var(noise)
```

The final response is quality-filtered, clipped to `[-10, 10]`, centered, and
scaled to unit variance.

## Feature Contract and Scale Defect

Every feature draws:

```text
variance_j ~ Exponential(rate=1)
sigma_j = sqrt(variance_j)
error_ij ~ Normal(0, variance_j)
```

`add_featurewise_exp_variance_gaussian_noise` standardizes after clipping.
However, `ExactLowCorrelationAccumulator._prepare` then:

```text
candidate = standardize(candidate)
candidate = clip(candidate, -10, 10)
stored = float16(candidate)
```

without a second standardization. This affects SCM-core features only. BNN
extension features do not pass through this final accumulator.

## Loader and Trainer

Trainer:

`/users/minsuk/fm-lm-lambda/kalbi_v6_exactprox10_eta4x_v23_independentraw_k32s32_p100k_effbatch32/train_beta_only_v23_supportmix_v13_transition_8gpu.py`

SHA256: `2b96a57f04e951070cb270db4bb37900b42226edda5da48f99918414949e83a5`

NPZ reader:

- lines 620-825
- requires `X` float16 with shape `[1,N,P]`
- requires `y` float32 with shape `[1,N]`
- validates metadata arrays, SNR, support guard, correlation certificates, and response scale
- converts X and y to float32 before returning

Beta-training preprocessing:

- `make_canonical_query_payload`, lines 978-995
- calls `standardize_tasks` on the complete task
- `standardize_tasks`, `train.py` lines 441-448
- no padding is used because the variable-shape runtime batch is one task

Selector-only split code, although inactive in this beta-only run:

- `split_and_standardize_variable`, lines 937-975
- disjoint random half-split
- training rows alone determine X/y mean and scale
- test rows use training statistics

## Provenance Limitation

Production NPZ files do not serialize:

- support indices
- full beta
- signal
- realized noise

They serialize support size/hash, beta norms, SNR, and branch metadata. Exact
`y = X beta + epsilon` reconstruction and support/nonzero identity therefore
cannot be independently audited from production shards. Those identities were
tested against the live functions in isolated counterfactual tests.
