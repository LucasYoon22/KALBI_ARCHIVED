#!/usr/bin/env bash
# A4000 -> 3080 직접 전송.  노트북을 경유하지 않으므로 끊고 나가도 된다.
#
#   1. A4000 에 전용 키를 만들고 공개키를 3080 authorized_keys 에 넣는다
#   2. A4000 에서 nohup 으로 tar | ssh 를 띄운다
#   3. 노트북은 즉시 빠져도 된다
#
#   DST_PORT=4597 ./migrate_bg.sh
#   DST_PORT=4597 ./migrate_bg.sh --status      진행 확인
#
set -euo pipefail

SRC_HOST=${SRC_HOST:-root@38.29.145.27}
SRC_PORT=${SRC_PORT:-40801}
DST_HOST=${DST_HOST:-root@122.51.254.66}
DST_PORT=${DST_PORT:?"DST_PORT 를 지정하라"}
DST_IP=${DST_HOST#*@}

LOG=/workspace/minsuk/migrate.log
DONE=/workspace/minsuk/migrate.done

if [[ "${1:-}" == "--status" ]]; then
  echo "== A4000 쪽 진행"
  ssh -n -p "$SRC_PORT" "$SRC_HOST" \
    "tail -5 $LOG 2>/dev/null; echo; \
     if [ -f $DONE ]; then cat $DONE; else \
       pgrep -af 'tar czf' >/dev/null && echo '진행 중' || echo '프로세스 없음'; fi"
  echo
  echo "== 3080 쪽 수신량 (목표 ~470 MB)"
  ssh -n -p "$DST_PORT" "$DST_HOST" "du -sh /workspace/minsuk 2>/dev/null || echo '아직 없음'"
  exit 0
fi

echo "== 1. A4000 전용 키 생성 및 3080 등록"
ssh -n -p "$SRC_PORT" "$SRC_HOST" \
  "test -f ~/.ssh/id_ed25519 || ssh-keygen -t ed25519 -N '' -q -f ~/.ssh/id_ed25519 -C a4000-to-3080; cat ~/.ssh/id_ed25519.pub" \
  | ssh -p "$DST_PORT" "$DST_HOST" \
      "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && echo '  등록 완료'"

echo
echo "== 2. A4000 에서 3080 으로 붙는지 확인"
ssh -n -p "$SRC_PORT" "$SRC_HOST" \
  "ssh -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $DST_PORT $DST_HOST 'echo 3080 도달 OK' 2>&1 | tail -1"

echo
echo "== 3. 백그라운드 전송 시작"
ssh -n -p "$SRC_PORT" "$SRC_HOST" bash -s <<REMOTE
set -euo pipefail
rm -f $LOG $DONE
cd /workspace/minsuk
nohup bash -c '
  set -o pipefail
  cd /workspace/minsuk
  ssh -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -p $DST_PORT $DST_HOST "mkdir -p /workspace/minsuk"
  tar czf - \
    --exclude="fm-lm-lambda/runs" \
    --exclude="fm-lm-lambda/evaluations" \
    --exclude="fm-lm-lambda/logs" \
    --exclude="fm-lm-lambda/real_data_audit_20260728" \
    --exclude="fm-lm-lambda/isolated_audits" \
    --exclude="fm-lm-lambda/evaluation" \
    --exclude="fm-lm-lambda/kalbi_v7_fullp_blockprox3_v23_k32s32_p100k_effbatch32" \
    --exclude="fm-lm-lambda/kalbi_v7_blocksize_microbenchmark_20260727" \
    --exclude="*.pyc" --exclude="__pycache__" \
    fm-lm-lambda data \
  | ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -p $DST_PORT $DST_HOST "tar xzf - -C /workspace/minsuk"
  echo "전송 완료 \$(date)" > $DONE
' > $LOG 2>&1 &
echo "  pid \$!"
REMOTE

echo
echo "시작됐다.  노트북은 이제 빼도 된다."
echo "확인:  DST_PORT=$DST_PORT ./migrate_bg.sh --status"
