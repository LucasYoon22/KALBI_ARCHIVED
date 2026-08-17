#!/usr/bin/env bash
# 8장이 전부 빌 때까지 기다렸다가 2000 스텝 파일럿을 띄운다.
#
# 왜 필요한가
#   3080 8장짜리 호스트를 다른 학습(PGKL)과 공유하고 있다.
#   그쪽이 들락날락하는데 우리 파일럿은 2.4시간 연속이 필요하다.
#   사람이 지켜보다 띄우는 대신 이 스크립트가 지켜본다.
#
# 사용법 (원격 3080에서, 반드시 tmux 안에서)
#   tmux new -s pilot
#   bash /workspace/kalbi/project/wait_and_launch_pilot.sh
#   Ctrl-b 떼고 d
#
# 상태 확인
#   tail -f /workspace/runs/pilot_watcher.log

set -uo pipefail

FREE_MIB=${FREE_MIB:-1000}      # 카드당 이 값 미만이면 "비었다"고 본다
STABLE_CHECKS=${STABLE_CHECKS:-3}  # 연속 몇 번 비어 있어야 띄우는가
INTERVAL=${INTERVAL:-60}        # 확인 주기 (초)

RUN_ID=v15_8gpu_pilot2k_$(date -u +%Y%m%d_%H%M%S)
OUT_DIR=/workspace/runs/${RUN_ID}
WATCH_LOG=/workspace/runs/pilot_watcher.log

SRC_CKPT=/workspace/runs/v15_8gpu_smoke_20260814m/checkpoint_selector_step030100.pt
CKPT_SHA=7ab049915d5849f35dbb468171b746f221d9c41eb95e68224e1d373e4a75299e
PKG=kalbi_v6_exactprox10_eta4x_v24_additive512_traj3_k32s32_p100k_effbatch32

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$WATCH_LOG"; }

if [[ ! -f "$SRC_CKPT" ]]; then
    log "스모크 체크포인트가 없다: $SRC_CKPT"
    exit 1
fi

log "감시 시작.  카드당 ${FREE_MIB} MiB 미만이 ${STABLE_CHECKS}회 연속이면 띄운다."

streak=0
while true; do
    busiest=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
              | sort -n | tail -1)
    if (( busiest < FREE_MIB )); then
        streak=$((streak + 1))
        log "비어 있음 (최대 사용 ${busiest} MiB) — ${streak}/${STABLE_CHECKS}"
        if (( streak >= STABLE_CHECKS )); then
            break
        fi
    else
        if (( streak > 0 )); then
            log "다시 점유됨 (최대 사용 ${busiest} MiB) — 카운터 초기화"
        fi
        streak=0
    fi
    sleep "$INTERVAL"
done

log "8장 확보.  파일럿을 띄운다: ${RUN_ID}"

mkdir -p "$OUT_DIR"
cp "$SRC_CKPT" "$OUT_DIR/" || { log "체크포인트 복사 실패"; exit 1; }

cd /workspace/kalbi/project || exit 1

torchrun --standalone --nproc_per_node=8 \
  -m ${PKG}.train_selector_only_v24_frozen_kalbi_v6_cachedtraj_v14_transition_8gpu \
  --rolling-data-root /workspace/rolling_v15_8gpu/data \
  --rolling-control-root /workspace/rolling_v15_8gpu/rolling_control \
  --expected-pipeline-sha256 826bf53977ec5d90c300544e5340e2e2b1f227f0ad0e66690ae661e70eba5065 \
  --trainer-id "${RUN_ID}" \
  --out-dir "${OUT_DIR}" \
  --config /workspace/kalbi/project/fpk_v33_kkt0.yaml \
  --penalty-cache /users/minsuk/fm-lm-lambda/penalty_cache_300k_v12_canonical_final.pt \
  --penalty-cache-sha256 6a3fda8201de87d5e2078e6e56ad0c67e5ad2a27248865910cac8f0f36bfe1dc \
  --input-scaler-cache /workspace/kalbi/project/${PKG}/v22_input_scaler_train64.pt \
  --input-scaler-cache-sha256 d574f1c50ed7fa9a46b42b012524cad48961ddc9ced33b50b8d0857281202cc1 \
  --v24-scaler-cache /workspace/kalbi/project/${PKG}/v24_scaler_rolling256_step42700.pt \
  --v24-scaler-cache-sha256 475e34d76c0ef936d83a43dbb3a616359a4d71213f7c42ec5a5aad06b78dbab8 \
  --resume "${OUT_DIR}/checkpoint_selector_step030100.pt" \
  --resume-sha256 "${CKPT_SHA}" \
  --steps 32100 \
  2>&1 | tee "/workspace/runs/${RUN_ID}.log"

status=${PIPESTATUS[0]}
if (( status == 0 )); then
    log "파일럿 완주.  로그: /workspace/runs/${RUN_ID}.log"
else
    log "파일럿 실패 (exit ${status}).  로그 확인: /workspace/runs/${RUN_ID}.log"
fi
