#!/usr/bin/env bash
# A4000 위에서 실행된다.  3080 으로 직접 tar 스트리밍.
#   bash remote_migrate.sh <DST_PORT> <DST_HOST>
#
# gzip 을 안 쓴다.  전체 470 MB 중 424 MB 가 체크포인트와 캐시라
# 이미 압축된 바이너리다.  CPU 만 쓰고 이득이 없다.

set -o pipefail

DST_PORT="${1:?DST_PORT}"
DST_HOST="${2:?DST_HOST}"

SSHOPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
SSHOPTS="$SSHOPTS -o ServerAliveInterval=15 -o ServerAliveCountMax=8"
SSHOPTS="$SSHOPTS -o ConnectTimeout=20"

cd /workspace/minsuk

echo "시작 $(date '+%F %T')"
echo "목적지 $DST_HOST:$DST_PORT"

if ! ssh -n $SSHOPTS -p "$DST_PORT" "$DST_HOST" "mkdir -p /workspace/minsuk"; then
  echo "실패: 목적지에 붙을 수 없다"
  exit 1
fi
echo "목적지 준비 완료"

tar cf - \
  --checkpoint=20000 --checkpoint-action=echo \
  --totals \
  --exclude='fm-lm-lambda/runs' \
  --exclude='fm-lm-lambda/evaluations' \
  --exclude='fm-lm-lambda/logs' \
  --exclude='fm-lm-lambda/real_data_audit_20260728' \
  --exclude='fm-lm-lambda/isolated_audits' \
  --exclude='fm-lm-lambda/evaluation' \
  --exclude='fm-lm-lambda/kalbi_v7_fullp_blockprox3_v23_k32s32_p100k_effbatch32' \
  --exclude='fm-lm-lambda/kalbi_v7_blocksize_microbenchmark_20260727' \
  --exclude='fm-lm-lambda/.migration_incoming' \
  --exclude='*.pyc' --exclude='__pycache__' \
  fm-lm-lambda data \
  | ssh $SSHOPTS -p "$DST_PORT" "$DST_HOST" "tar xf - -C /workspace/minsuk"

rc=$?
echo "종료코드 $rc   $(date '+%F %T')"
if [ "$rc" -eq 0 ]; then
  echo "전송 완료" > /workspace/minsuk/migrate.done
else
  echo "전송 실패 rc=$rc" > /workspace/minsuk/migrate.done
fi
