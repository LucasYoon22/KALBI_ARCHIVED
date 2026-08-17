# KALBI Immediate N201–999, P<2000 Evaluation

Date: 2026-07-28  
Status: **Complete and independently validated**

## Technical summary

The evaluation pinned the newest complete, strict-CPU-loadable checkpoints that existed at `2026-07-28 08:59:09 KST`, froze four first-accepted unseen V14 tasks satisfying `200 < N < 1000` and `P < 2000`, and evaluated a common 41-point raw-lambda grid from `1e-4` to `1`.

The four-task mean KALBI/CD coefficient-path correlation was `0.900036`, mean coefficient RMSE was `0.005810`, and mean explicit-objective gap was `0.009865`. The mean selected/oracle test-MSE ratio was `1.189032`. Full-P coordinate descent converged at every task/lambda point; the worst projected-KKT and fixed-point residuals were `2.67e-7` and `2.65e-7`.

The selected penalties were distinct on all four tasks. Pairwise penalty RMSE on `[0,1]` averaged `0.199977`, pairwise derivative RMSE averaged `2.218840`, and the taskwise `d0` coefficient of variation was `0.876960`.

## Immutable checkpoints

### Coefficient checkpoint

- Path: `/users/minsuk/fm-lm-lambda/runs/kalbi_v8_adaptive_t10_betaonly_supportmix_v14_target3080_effbatch32_to60k_20260728/checkpoint_step043000.pt`
- Completed step: `43000`
- Model version: `kalbi_v8_krylov_abfilm_threshold_adaptiveblockprox10_k32_stages32_v1`
- File SHA256: `36e38d01917f157919e0383841e99126a4b9eb8d5f98f5ea66fe9af16c23cc51`
- Model-state SHA256: `896d5a67c5d16a44788f5e0c180a53d7977506497f62a1453b645cb3bb81711c`
- Creation time: `2026-07-27T17:03:12.433795Z`
- Contract: adaptive full-P block-coordinate refinement, `T=10`, production block-size formula and alternating forward/reverse sweep order.

### Lambda/penalty checkpoint

- Path: `/users/minsuk/fm-lm-lambda/runs/kalbi_v24_selectoronly_v14_snr0p2to50_frozenbeta_step42700_effbatch32_20260728/checkpoint_selector_step009600.pt`
- Completed selector step: `9600`
- Selector version: `v24_v23_additive512_global_solvertrajectory3_proposal_residual4_independent_rawlambda_v1`
- Recorded solver model version: `kalbi_v6_krylov_abfilm_threshold_exactproxpg10_eta4x_k32_stages32_v1`
- File SHA256: `b058cd33ed6a72154312179781520a5b5ab2991def4b7a1e36ea65cb950d8bd3`
- Frozen-beta state SHA256: `63fc536c0899068e04b0c14f5438b88ad8d0403af3df520acace35b4c4741ad4`
- Creation time: `2026-07-27T23:52:18.987456Z`

The scheduled V24-on-V8 transition had **not** completed at pin time. The selector’s frozen-beta SHA differs from the newest coefficient checkpoint’s model-state SHA. The primary result therefore follows the requested independent pairing: the pinned selector supplies raw lambda and penalty, while the newest pinned coefficient model supplies the coefficient estimates. A state-preservation test confirmed that loading the coefficient state did not change selector parameters, selector buffers, cached scalers, or lambda/penalty heads.

## Frozen evaluation tasks

The generator source is `generate_tabiclv2_scm2048_qfilter_bnn128_mixedlinear_supportmix_v14.py`, SHA256 `d2548b21c63f220ec2d2c5f34bcf1a341e17b4440eb3db80f1ad3ad1029e812f`. The pipeline/config SHA256 is `4fd706b7b254eddb4a832b8107fa3379d128f152716c7ae9f142539064d7e707`. No explicit `generator_version` field was emitted, so V14 is recorded from the source/config contract rather than fabricated as a manifest field.

| Dataset | Branch | Seed | N | P | P/N | Context/Val/Test | Signal statistic | Target sparsity | Max abs feature corr. | Zero Val/Test MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | controlled linear | 2147100005 | 825 | 1,855 | 2.2485 | 495/165/165 | realized SNR 25.2437 | 1 | 0.90497 | 1.21985 / 1.17448 |
| 2 | controlled linear | 2147100007 | 801 | 1,693 | 2.1136 | 480/160/161 | realized SNR 13.6321 | 20 | 0.60814 | 0.95962 / 0.97375 |
| 3 | native | 2147200004 | 903 | 1,657 | 1.8350 | 541/180/182 | max context feature-target corr. 0.62016 | n/a | 0.85961 | 0.94251 / 1.18840 |
| 4 | native | 2147200006 | 772 | 1,645 | 2.1308 | 463/154/155 | max context feature-target corr. 0.25909 | n/a | 0.40339 | 1.04005 / 1.04535 |

Task hashes:

- Dataset 1: `79097d153e663e1a8afcc9f7d90c4e74488bae2c5e4cedba4388eda9c860c663`
- Dataset 2: `8a24626592db72972f96bb1f37417fe21558bb12c9d4f57ab07822fc7de3e13e`
- Dataset 3: `4d30f27a91c5cb1336ef34666968e55175951248e3b3a5cabcde4ee65812155d`
- Dataset 4: `ebcb5ae6b3d494df24b7ae5bbb0de79479030e955e18978e313b263b8343c1da`

The provenance audit scanned 548 JSON files over training catalogs, prior evaluations, validation, tuning/audit areas, and runs; it excluded 1,113 prior seeds and 613 prior task hashes. Tasks were the first deterministic unseen candidates accepted before inference, without performance-based replacement.

## Path and lambda-selection results

| Dataset | Path corr. | Coef. RMSE | Mean / max objective gap | Selected / oracle raw lambda | Selected / oracle validation MSE | Selected / oracle test MSE | Test ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.89513 | 0.006927 | 0.026252 / 0.544143 | 0.00134537 / 0.0630957 | 0.042794 / 0.036439 | 0.039191 / 0.037695 | 1.03969 |
| 2 | 0.89729 | 0.005899 | 0.009344 / 0.093524 | 0.000714966 / 0.00398107 | 0.119206 / 0.073404 | 0.101164 / 0.061719 | 1.63909 |
| 3 | 0.89245 | 0.004451 | 0.002432 / 0.016644 | 0.00778911 / 0.01 | 0.084716 / 0.084513 | 0.120865 / 0.120434 | 1.00358 |
| 4 | 0.91528 | 0.005961 | 0.001431 / 0.003686 | 0.0207812 / 0.0001 | 0.635959 / 0.598322 | 0.654049 / 0.609114 | 1.07377 |

Selected/oracle support sizes were `179/1`, `403/29`, `664/545`, and `832/1644`. The low-lambda endpoint NNZ values were `1331`, `1137`, `1648`, and `1644`; high-lambda endpoint NNZ values were `1`, `0`, `0`, and `0`.

## Penalty metrics

| Dataset | d0 | max rho-prime | Derivative TV | Concavity violations |
|---|---:|---:|---:|---:|
| 1 | 8.16492 | 8.16492 | 13.21791 | 16 |
| 2 | 8.52882 | 8.52882 | 14.17270 | 17 |
| 3 | 0.77202 | 3.65560 | 6.87715 | 32 |
| 4 | 0.33339 | 3.67402 | 7.01464 | 50 |

All penalties passed `rho(0)=0`, finite `rho`, finite `rho-prime`, finite `d0`, and finite exact-prox segment checks. Exact-representation maximum errors ranged from `4.95e-6` to `5.14e-5`.

## Validation and figure QA

- Context-only centering/scaling and disjoint splits: PASS.
- Validation/test targets excluded from selector and coefficient inference: PASS.
- Identical 41-point raw-lambda grid and penalty representation for KALBI and CD: PASS.
- Direct recomputation of every validation/test MSE: PASS; maximum saved-versus-recomputed difference `8.55e-9`.
- Independent objective recomputation: PASS; maximum absolute difference `4.83e-15`.
- Full-P deterministic multistart CD convergence: PASS at all 164 task/lambda points.
- Coefficient-vector nonreuse check: PASS; 36–41 unique path vectors per task, plus single-query cross-checks at grid points 0, 20, and 40.
- Deterministic CD top-12 feature ranking: PASS.
- Composite PNG, four detail PNGs, and PDF rasterization visually inspected: PASS.
- PDF: one page, embedded/subset DejaVu Sans, no clipping, and no prohibited internal implementation names in visible text.

The first one-task smoke run completed its numerical work but hit an empty-pair aggregation error in the one-task penalty-diversity summary. The reducer was corrected for the one-task case and `smoke_5point_retry` passed. This did not affect the primary four-task run.

## Runtime and service audit

- Full evaluation wall time: `633 s`.
- Sum of task runtimes: `626.607 s`; full-P CD accounted for `608.787 s`.
- KALBI inference: physical GPU 1 selected through `CUDA_VISIBLE_DEVICES=1`, reported inside the process as `cuda:0`.
- Peak evaluation GPU allocation: `937,137,152 bytes` (about 894 MiB).
- CD: CPU, 8 workers, 4 retries, 4,000 maximum sweeps, tolerance `1e-7`, zero tolerance `1e-11`.
- Initial disk: 751 GiB free; final disk: about 750 GiB free.

Initial audit at `2026-07-28 08:59:09 KST` and post-copy final audit at `2026-07-28 09:41:22 KST`:

- Selector trainer: RUNNING; logged step advanced `9686 → 10196` (+510).
- Coefficient trainer: STOPPED before evaluation and STOPPED after; latest logged step remained 43040.
- Curator: STOPPED before and after.
- Eight producers: all STOPPED before and after.
- Rolling catalog: epoch 1, SHA256 `ae5beda6c8349febf5cad431687cc695334ad13db6c6ac7d5a40069e308a232b`, rank ACK 8/8 before and after.

The evaluation did not start, stop, restart, or modify any service, checkpoint, optimizer state, rolling data, catalog, or service definition.

## Artifact locations

Server root:

`/users/minsuk/fm-lm-lambda/evaluation/latest_v8_v24_n201to999_punder2000_20260728_immediate`

Local Obsidian root:

`/Users/minsuk.shin/Documents/Aa/Research/FM for Linear Model/KALBI Latest Evaluation N201-999 Punder2000 2026-07-28 Immediate`

Primary figures:

- `figures/composite/kalbi_solution_path_composite.png`
- `figures/composite/kalbi_solution_path_composite.pdf`
- `figures/detail/dataset_1_detail.png` through `dataset_4_detail.png`

Machine-readable results:

- `metrics/metrics_summary.csv`
- `metrics/results.json`
- `metrics/penalty_diversity.json`
- `metrics/evaluation_manifest.json`
- `metrics/validation_report.json`
- `reproducibility_manifest.json`

## Exact evaluation commands

```bash
PROJECT=/users/minsuk/fm-lm-lambda
ROOT=/users/minsuk/fm-lm-lambda/evaluation/latest_v8_v24_n201to999_punder2000_20260728_immediate
PYTHON=$PROJECT/.venv-kalbi-v24-eval/bin/python
BETA=$PROJECT/runs/kalbi_v8_adaptive_t10_betaonly_supportmix_v14_target3080_effbatch32_to60k_20260728/checkpoint_step043000.pt
SELECTOR=$PROJECT/runs/kalbi_v24_selectoronly_v14_snr0p2to50_frozenbeta_step42700_effbatch32_20260728/checkpoint_selector_step009600.pt
CONFIG=$PROJECT/fpk_v33_kkt0.yaml
PIPELINE=$PROJECT/model_v58_chebyshev_v17_scmmixture_p100k/pipeline_config_tabicl_v14_snr0p2_50_supportmix_rolling100gb.json
ROLLING=/users/sync_data/kalbi_tabiclv2_v14_mixed50_linear_pn50_strictn3_snr0p2to50_p100k_100GB_target3080_seed160076430/rolling_control
export PYTHONPATH=$PROJECT

$PYTHON "$ROOT/source_snapshot/audit_checkpoints.py" \
  --beta "$BETA" \
  --selector "$SELECTOR" \
  --config "$CONFIG" \
  --pin-time-utc 2026-07-27T23:59:09Z \
  --output "$ROOT/checkpoints/checkpoint_audit.json"

$PYTHON "$ROOT/source_snapshot/prepare_tasks.py" \
  --output-root "$ROOT/data_manifest" \
  --pipeline-config "$PIPELINE" \
  --expected-config-sha256 4fd706b7b254eddb4a832b8107fa3379d128f152716c7ae9f142539064d7e707 \
  --project-root "$PROJECT" \
  --rolling-control-root "$ROLLING" \
  --python "$PYTHON" \
  --resume

$PYTHON "$ROOT/source_snapshot/test_contracts.py" \
  --evaluation-root "$ROOT" \
  --evaluator-source "$ROOT/source_snapshot/evaluate_latest.py"

CUDA_VISIBLE_DEVICES=1 $PYTHON "$ROOT/source_snapshot/evaluate_latest.py" \
  --evaluation-root "$ROOT" \
  --beta-checkpoint "$BETA" \
  --beta-sha256 36e38d01917f157919e0383841e99126a4b9eb8d5f98f5ea66fe9af16c23cc51 \
  --selector-checkpoint "$SELECTOR" \
  --selector-sha256 b058cd33ed6a72154312179781520a5b5ab2991def4b7a1e36ea65cb950d8bd3 \
  --config "$CONFIG" \
  --device cuda:0 \
  --grid-size 5 \
  --task-limit 1 \
  --tag smoke_5point_retry \
  --cd-workers 8 \
  --cd-max-sweeps 4000 \
  --cd-retries 4 \
  --cd-tolerance 1e-7 \
  --cd-zero-tolerance 1e-11

CUDA_VISIBLE_DEVICES=1 $PYTHON "$ROOT/source_snapshot/evaluate_latest.py" \
  --evaluation-root "$ROOT" \
  --beta-checkpoint "$BETA" \
  --beta-sha256 36e38d01917f157919e0383841e99126a4b9eb8d5f98f5ea66fe9af16c23cc51 \
  --selector-checkpoint "$SELECTOR" \
  --selector-sha256 b058cd33ed6a72154312179781520a5b5ab2991def4b7a1e36ea65cb950d8bd3 \
  --config "$CONFIG" \
  --device cuda:0 \
  --grid-size 41 \
  --task-limit 4 \
  --tag full_41point \
  --cd-workers 8 \
  --cd-max-sweeps 4000 \
  --cd-retries 4 \
  --cd-tolerance 1e-7 \
  --cd-zero-tolerance 1e-11

$PYTHON "$ROOT/source_snapshot/validate_artifacts.py" \
  --evaluation-root "$ROOT"
```

Figures were rendered locally because the server environment did not contain Matplotlib:

```bash
python /Users/minsuk.shin/kalbi_eval_work/plot_latest.py \
  --evaluation-root /Users/minsuk.shin/kalbi_render_work_v4_clean/latest_v8_v24_n201to999_punder2000_20260728_immediate
```

The complete command strings, hashes, task records, audit snapshots, figure hashes, and validation evidence are also preserved in `reproducibility_manifest.json`.
