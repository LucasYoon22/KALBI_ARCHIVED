#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR=/users/minsuk/fm-lm-lambda/kalbi_v7_fullp_blockprox3_v23_k32s32_p100k_effbatch32
PROJECT_DIR=/users/minsuk/fm-lm-lambda
APPROVAL_FILE="$PACKAGE_DIR/preflight/acceptance_status.json"
SELECTOR_CHECKPOINT="$PACKAGE_DIR/preflight/checkpoint_selector_step020300.pt"
SELECTOR_SHA256=e8bb3f0ed793bd557554d2252ff687e542ea913279f6a7e1c4fe73f6872b2b0a

: "${BETA_CHECKPOINT:?Set BETA_CHECKPOINT to a strict 100-step boundary checkpoint.}"
: "${BETA_CHECKPOINT_SHA256:?Set BETA_CHECKPOINT_SHA256.}"
: "${TRAINER_ID:?Set a unique TRAINER_ID.}"
: "${OUT_DIR:?Set an empty OUT_DIR.}"

if [[ "${KALBI_V7_TRANSITION_APPROVED:-0}" != "1" ]]; then
  echo "KALBI V7 transition is not explicitly approved." >&2
  exit 2
fi
if [[ ! -f "$APPROVAL_FILE" ]] || \
   ! grep -Eq '"approved"[[:space:]]*:[[:space:]]*true' "$APPROVAL_FILE"; then
  echo "KALBI V7 acceptance gate is not approved: $APPROVAL_FILE" >&2
  exit 3
fi
if [[ ! "$BETA_CHECKPOINT" =~ checkpoint_step[0-9]*00\\.pt$ ]]; then
  echo "BETA_CHECKPOINT is not a 100-step boundary checkpoint." >&2
  exit 4
fi
if [[ -e "$OUT_DIR" ]]; then
  echo "OUT_DIR must not already exist: $OUT_DIR" >&2
  exit 5
fi

printf '%s  %s\n' "$BETA_CHECKPOINT_SHA256" "$BETA_CHECKPOINT" | sha256sum -c -
printf '%s  %s\n' "$SELECTOR_SHA256" "$SELECTOR_CHECKPOINT" | sha256sum -c -

cd "$PROJECT_DIR"
exec /venv/main/bin/torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  -m kalbi_v7_fullp_blockprox3_v23_k32s32_p100k_effbatch32.train_joint_v7_blockprox3_v23_supportmix_v13_8gpu \
  --rolling-data-root /users/sync_data/kalbi_tabiclv2_v13_mixed50_linear_pn50_strictn3_p100k_100GB_target3080_seed160076430 \
  --rolling-control-root /users/sync_data/kalbi_tabiclv2_v13_mixed50_linear_pn50_strictn3_p100k_100GB_target3080_seed160076430/rolling_control \
  --expected-pipeline-sha256 99f81c6a5bf5083a56f3268a56a0700cd64865b1ff0bba9b23386088d11a9559 \
  --trainer-id "$TRAINER_ID" \
  --out-dir "$OUT_DIR" \
  --config "$PROJECT_DIR/fpk_v33_kkt0.yaml" \
  --penalty-cache "$PROJECT_DIR/penalty_cache_300k_v12_canonical_final.pt" \
  --penalty-cache-sha256 6a3fda8201de87d5e2078e6e56ad0c67e5ad2a27248865910cac8f0f36bfe1dc \
  --input-scaler-cache "$PROJECT_DIR/kalbi_v6_exactprox10_eta4x_v23_independentraw_k32s32_p100k_effbatch32/v22_input_scaler_train64.pt" \
  --input-scaler-cache-sha256 d574f1c50ed7fa9a46b42b012524cad48961ddc9ced33b50b8d0857281202cc1 \
  --transition-v7-beta-from "$BETA_CHECKPOINT" \
  --transition-v7-beta-sha256 "$BETA_CHECKPOINT_SHA256" \
  --transition-v7-selector-from "$SELECTOR_CHECKPOINT" \
  --transition-v7-selector-sha256 "$SELECTOR_SHA256" \
  --steps 60000 \
  --beta-lr 1e-4 \
  --feature-lr-scale 1.0 \
  --query-lr-scale 1.0 \
  --shape-lr-scale 1.0 \
  --local-lr-scale 1.0 \
  --other-lr-scale 1.0 \
  --clip-feature 20.0 \
  --clip-query 20.0 \
  --clip-shape 20.0 \
  --clip-local 20.0 \
  --clip-other 20.0 \
  --selector-lr 3e-4 \
  --direct-head-lr 3e-4 \
  --warmup-steps 500 \
  --batch-per-rank 1 \
  --grad-accum-steps 4 \
  --selector-grad-accum-steps 4 \
  --selector-update-every 4 \
  --selector-hidden 512 \
  --cache-files 2 \
  --reload-every 100 \
  --checkpoint-every 100 \
  --log-every 10 \
  --seed 202607232355
