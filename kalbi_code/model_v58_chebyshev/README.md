# V58-Chebyshev + Lambda Selector V10

## Outcome

V58-Chebyshev is a standalone fixed-support selector-gradient successor to the
paired V57-Chebyshev model. It preserves the V57-Chebyshev hard forward and
checkpoint schema, replaces only the broken local-ISIS continuous backward
path, and jointly trains the beta solver with native196 Lambda Selector V10.

The production run is hosted on the replacement 8xRTX3090 server:

- package: `/users/minsuk/fm-lm-lambda/model_v58_chebyshev/`;
- run: `/users/minsuk/fm-lm-lambda/runs/v58chebyshev_v10_joint_N200_P400_30k_20260716_1547/`;
- supervisor: `v58chebyshev_v10_joint_30k_20260716_1547`;
- warm start: immutable V57-Chebyshev step 28,000, SHA256
  `8a59a26a1344501aee24fd01aa93e5e797a2f1da78ad42715efaed91dded4c20`.

The older V57 trainer, V57 solution-path watcher, and paired V57-Chebyshev
trainer were stopped before this run. Unrelated system services and Jupyter
kernels were not touched.

At the final handoff audit the production supervisor was still `RUNNING`.
Logging was changed in place at update 400: both JSON writing and the former
eight-rank monitor `all_gather_object` now run only every 100 updates. The run
resumed without changing model, selector, optimizer, scheduler, stream, or RNG
state. Its first new record was exactly update 500; no update record was
written for 401 through 499.

At update 500 all four beta gradient groups, all four selector head/trunk
groups, and all 48 penalty-latent coordinates were finite and nonzero on every
rank. The smallest sampled beta lambda was `1.1180e-6`, above the requested
floor, and effective-ratio compensation error was at most `7.45e-9`.
The first checkpoint written natively by the logging-100 source is update 600,
SHA256 `6473d7f468fce495178da630b92c2aa9387b96225093326e0ce7437fa31d749c`.
Its 133 beta keys, 71 selector keys, and eight rank states passed the same
finite/nonzero gates, so it is now the ordinary resumable checkpoint.

At update 3,600 the selector effective batch was deliberately changed from 8
to 32 tasks/update: each rank now evaluates four independent held-out tasks,
backpropagates each loss divided by four, skips DDP synchronization on the
first three micro-steps, and performs one synchronized optimizer update after
the fourth. The first complete 100-update window ended at update 3,700. All
eight ranks remained finite, all 32 beta and 32 selector gradient groups were
positive, and all four accumulated gradients were nonzero in every one of the
48 penalty-latent coordinates.

## Bundle

`model_v58_chebyshev/` contains:

- `model.py`: standalone native196 V58-Chebyshev decoder;
- `utils.py`: numerical/model utilities and differentiable fixed-support ISIS;
- `selector_v10.py`: native196/meta4 V10 selector;
- `train_joint_v10.py`: alternating beta and held-out-MSE V10 updates;
- `migrate_selector_accum_checkpoint.py`: state-exact V1-to-V2 selector-batch
  contract migration;
- `train.py`: beta-only trainer with the same `1e-6` lambda floor;
- `penalty_generator.py` and `synthetic_data_generator.py`;
- `gradient_audit.py`, `launch_joint_v10.sh`, and the production launcher.

The runtime package imports no `model_fpk_v*` or
`model_lambda_selector_v*` module. Stable project dependencies are limited to
the non-versioned bootstrap/data utilities required by the project.

## Model contract

- state schema: 133 keys, 4,909,208 trainable parameters, 4,910,640 state
  elements;
- penalty payload: native196, comprising 96 interleaved rho/drho pairs and
  meta4;
- expanded condition payload: 388 dimensions;
- convexity flag: absent;
- ISIS support: detached exact TopK indices and stage activity only;
- continuous fixed-support graph: residual, RHS, selected Gram, recurrence,
  exact ridge, mixer, lambda-d0 threshold, soft threshold, scatter, norm gate,
  and stage update are differentiable.

The mixer representation exactly matches V57-Chebyshev:

- safe operator: `Ahat = 2G / max_i(sum_j |G_ij|) - I`;
- recurrence: `t[j+1] = 2 Ahat t[j] - t[j-1]`;
- each emitted vector is L2-normalized;
- no eigendecomposition and no Gram-Schmidt projection;
- recurrence affects only learned mixer features;
- the production width-32 deterministic ridge path remains the exact direct
  local solve.

The actual CUDA timing gate measured 6.236 ms for the recurrence versus
38.781 ms for two-pass MGS on the pinned B=128, N=200, K=32 test, a 6.219x
basis-construction speedup. This is a basis-helper timing comparison, not an
end-to-end training speedup claim.

## Joint-training contract

Every update has two separate phases.

1. V58-Chebyshev is trained on all 200 task rows using the original
   `mean(log1p(F_hat/F_zero))` beta objective.
2. The updated beta model is frozen. A separate task is split 100/100 and
   standardized with train-half statistics only. V10 is trained through the
   fixed-support decoder using `mean(log1p(heldout_MSE/2))`.

The selector loss has no explicit lambda-times-penalty term. V10 emits the
normalized fixed-knot ReLU-squared penalty and lambda jointly, with lambda
compensated so `lambda*d0/lambda_max` equals the emitted effective ratio.

Pinned production settings:

- 8 DDP ranks;
- beta: one task per rank times four accumulation steps, global batch 32;
- selector: one held-out task per rank times four accumulation steps, global
  batch 32;
- beta learning rate `2e-4`, selector learning rate `3e-4`;
- `lambda_low_min_contract = 1e-6` in the beta sampler;
- V10 effective-ratio lower bound `1e-6`;
- 30,000 joint updates, checkpoints every 200 updates.
- telemetry gather and JSON logging every 100 updates; non-logging updates do
  not execute the former eight-rank `all_gather_object` monitor path;
- every `train_update` records the timed-step count, mean compute seconds per
  step across all eight ranks, and the slowest rank's mean compute seconds per
  step. Timing covers data loading plus beta and V10 computation but excludes
  telemetry serialization and checkpoint writing;
- the production launcher automatically resumes from the newest complete
  checkpoint in the run directory and verifies its SHA256 before loading.

## Logging-cadence migration

The production source was changed after the immutable update-400 checkpoint,
so the trainer's fail-closed source-hash guard correctly rejected a direct
resume. A logging-only checkpoint migration updated only the trainer source
hash and added explicit migration provenance. Recursive comparison proved all
non-metadata state exactly equal to the original checkpoint.

- original update-400 checkpoint SHA256:
  `03d96f27ed0c0a24c31b8e68fe242ecdf98030fd1ff20fd58cc4f80b550815c6`;
- logging-only resume checkpoint SHA256:
  `90607830b1ab26437edb14138e3eea5c3ae8fa3b74e118d5125119eea0c6b334`;
- post-resume records at the audit: updates 500 and 600 only;
- update-500 monitor: 8/8 ranks finite, 32/32 beta groups positive,
  32/32 selector groups positive, and 48/48 penalty-latent coordinates
  nonzero on every rank;
- observed time was `2.941 s/update` for updates 401--500 versus
  `2.904 s/update` for updates 301--400. This short comparison establishes no
  speed regression or speedup; the guaranteed change is removal of monitor
  serialization/communication on 99 of every 100 updates and roughly tenfold
  lower event-log cadence than the previous production setting of 10.
- first native post-migration checkpoint: `checkpoint_step000600.pt`, SHA256
  `6473d7f468fce495178da630b92c2aa9387b96225093326e0ce7437fa31d749c`.

The compute-time fields were enabled through a second state-exact controlled
resume at update 800. The first complete timing window, updates 801--900,
recorded:

- `compute_timed_steps = 100`;
- eight-rank mean `average_compute_seconds_per_step = 2.919673` seconds;
- slowest-rank `max_rank_average_compute_seconds_per_step = 2.919709`
  seconds;
- fastest-rank mean `2.919664` seconds.

The timing-only resume checkpoint SHA256 is
`87efff7cc93095776d7420c090463961457053e9658a398dd911f1298f304d23`.
All non-metadata state was recursively exact to the original update-800
checkpoint. At update 900 all gradient, finite, latent, lambda-floor, and
effective-ratio gates remained healthy.

## Selector effective-batch migration

The selector batch change was made at the complete update-3,600 checkpoint.
The migration changed only the serialized trainer/checkpoint contract and
source hash; recursive independent comparison proved the model, selector,
both optimizer states, beta scheduler, data stream, penalty sampler, RNG, split
generator, and every rank state exactly unchanged.

- original V1 update-3,600 checkpoint SHA256:
  `af314fc9105f718b5c2d251a4487fc438c0f2643461cf7405a63909096b18801`;
- V2 selector-accumulation migration checkpoint SHA256:
  `f69921e8f9297fff0085fdb129581f8214a24ccd70d504c97ae99c6f65349e56`;
- V2 trainer source SHA256:
  `476c7f413d5d0e9e708d9de8dcc99307012c314deb0d28f7c1e13ffafed22ca3`;
- microbatch: one task/rank; selector accumulation: four; global effective
  selector batch: `8 × 1 × 4 = 32 tasks/update`;
- update-3,700 compute time: `6.470226 s/update` across ranks and
  `6.470267 s/update` on the slowest rank, versus `2.886399 s/update` for the
  preceding 8-task window. The requested batch increase therefore made this
  measured joint update `2.242x` slower; no speed-neutral claim is made;
- 8/8 ranks, all numeric telemetry finite; 32/32 beta groups and 32/32
  selector groups positive; every DDP-averaged group norm and the total
  selector norm had exactly zero spread across ranks;
- penalty-latent telemetry: 192 elements/rank (`4 × 48`), with 48/48
  coordinates and all individual elements nonzero on every rank;
- smallest sampled beta lambda: `1.20873e-6`; effective-ratio maximum error:
  `7.45e-9`;
- four-task within-rank rho/drho variation was nonzero on every rank.
- first checkpoint written natively by V2:
  `checkpoint_step003800.pt`, SHA256
  `4d819e0c8f92ee6796af8f14c321f9c0ef1aa781590edb4cda2c3527414d4793`.
  It contains 133 beta keys, 71 selector keys, and all eight rank states. Its
  100-step mean was `6.410013 s/update`, and all gradient/latent/lambda gates
  passed again.

The pinned loss plot through update 3,700 is in
`/Users/minsukshin/.openclaw/workspace/artifacts/v58chebyshev_v10_joint_loss_through_step3700_selector_batch32_20260716/`.
Its red marker identifies the update-3,600 batch-contract transition.

The expanded loss/KKT plot through update 3,800 is in
`/Users/minsukshin/.openclaw/workspace/artifacts/v58chebyshev_v10_joint_loss_kkt_through_step3800_selector_batch32_20260716/`.
`kkt_top32` is a detached residual diagnostic, not a KKT training loss. It is
included in neither the beta loss nor the selector held-out-MSE loss. The
8-rank mean was `0.0555532`, `0.0748981`, and `0.0816336` at updates 3,600,
3,700, and 3,800, respectively. This recent rise should be monitored, but it
does not backpropagate and the sparse 100-step telemetry should not be read as
a smooth per-step trajectory.

## Equivalence and gradient evidence

The immutable step-28,000 checkpoint strict-loads with no missing or
unexpected keys. On four actual repaired N=200/P=400 training tasks:

- all encoded tensors were V57/V58 bitwise equal;
- beta, support indices, stage activity, and common aux tensors were bitwise
  equal at the unperturbed hard forward;
- natural and frozen-support V58 forwards were bitwise equal;
- fixed-support directional finite-difference cosine ranged from
  `0.9999830` to `0.9999942`;
- fixed-support relative error ranged from `0.345%` to `0.608%`;
- natural cosine was positive on every task, `0.8517860..0.9999913`, with the
  lower value explained by one support-changing perturbation;
- lambda and all native196 gradients were finite and nonzero.

The original V57-Chebyshev no-grad backward remained mismatched: its natural
finite-difference cosine ranged from `-0.0844` to `0.9863` on the same small
audit pack. V58 fixes the continuous path without pretending that hard TopK
support boundaries are differentiable.

## Two-update 8-GPU preflight

The final source-hash preflight completed two joint updates and wrote a strict
checkpoint. Across all eight ranks:

- beta shape/query/feature/local gradient groups were finite and nonzero;
- V10 penalty head, lambda head, and penalty trunk were finite and nonzero on
  update 1;
- the zero-initialized lambda head gave the expected lambda-trunk gradient 0
  on update 1, then opened to norm `0.0020930` on update 2;
- all 48 internal penalty-latent coordinates had nonzero gradient on both
  updates;
- effective-ratio compensation error was at most `7.45e-9`;
- observed beta lambda samples respected the `1e-6` floor;
- beta and V10 parameter hashes both changed and remained synchronized;
- after two V10 updates, mean rho L2 change was `0.020847`, maximum rho
  coordinate change `0.003385`, mean drho L2 change `0.163542`, and maximum
  drho coordinate change `0.034084`.

The original two-update preflight used one selector task per rank, so its
within-rank rho/drho standard deviation was expected to print zero. Production
after update 3,600 aggregates four selector tasks per rank and now reports
nonzero within-rank rho/drho variation. This is useful telemetry, while a fixed
multi-task evaluation pack remains the stronger specialization audit.

## Step-4,600 training-task solution paths and selector optimality

The immutable step-4,600 checkpoint was audited on ten deterministic tasks
drawn only from the first 117 actual `part_0` training files; the final eight
validation files were excluded. Every stored task retained its full
`N=200, P=400` shape and was split into fixed train-only-standardized 100/100
rows. V10 saw only the train half. Held-out test rows were used retrospectively
to measure the selector point and finite-grid oracles.

- V10 selected held-out MSE mean/median: `0.268228 / 0.232417`;
- V58 with the same V10 penalty, finite-grid oracle: `0.233397 / 0.184284`;
- V58 with SCAD, finite-grid oracle: `0.223748 / 0.182777`;
- coordinate same-penalty oracle: `0.219489 / 0.153237`;
- coordinate SCAD oracle: `0.219285 / 0.154401`;
- V10's aggregate mean MSE was `14.923%` above its V58 conditional oracle;
  taskwise regret median was `25.007%`, and 4/10 tasks were within 5%;
- the V10 effective ratio was nearly fixed at `0.103709..0.105280`;
- unit-origin-slope penalty curves had minimum pairwise correlation
  `0.999995611`, and raw curves had maximum pointwise task range `0.00539402`.

Thus the gradient path is alive, but by step 4,600 the selector has not yet
learned meaningful task-specific lambda or penalty-shape specialization on
this fixed ten-task pack. The test oracles are optimistic finite-grid audits,
and the nonconvex coordinate references are finite-sweep diagnostics rather
than global-optimum claims.

## Artifacts

- actual-checkpoint gradient audits:
  `/users/minsuk/fm-lm-lambda/artifacts/v58chebyshev_v10_joint_preflight_20260716/`;
- final two-step preflight:
  `/users/minsuk/fm-lm-lambda/runs/v58cheb_v10_joint_preflight2stepB_seed202607161405/`;
- local source bundle:
  `/Users/minsukshin/.openclaw/workspace/artifacts/model_v58_chebyshev_joint_v10_20260716/`.
- loss plot through update 3,700:
  `/Users/minsukshin/.openclaw/workspace/artifacts/v58chebyshev_v10_joint_loss_through_step3700_selector_batch32_20260716/loss_through_step3700_selector_batch32.png`;
- loss and detached KKT plot through update 3,800:
  `/Users/minsukshin/.openclaw/workspace/artifacts/v58chebyshev_v10_joint_loss_kkt_through_step3800_selector_batch32_20260716/loss_and_kkt_through_step3800_selector_batch32.png`;
- step-4,600 full-shape training-task selector/path audit:
  `/Users/minsukshin/.openclaw/workspace/artifacts/v58cheb_v10_step4600_train_selector_scad_paths_seed202607162115/`;
- Obsidian mirror:
  `/Users/minsukshin/Documents/Obsidian Vault/Research/FM for Linear Model/V58-Chebyshev and Lambda Selector V10 Joint Training 2026-07-16.md`.

`actual_validation_summary.json` is the compact machine-readable summary.
The four task-level actual-checkpoint reports are preserved under
`actual_checkpoint_gradient_audits/`, and the full 8-GPU two-update telemetry
is preserved under `preflight_2step/`.
