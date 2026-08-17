#!/usr/bin/env bash
# A4000 -> 8x3080 이관.  노트북을 경유해 스트리밍한다 (로컬 디스크 안 씀).
#
# 옮기는 것 (~470 MB)
#   패키지 v23 / v24, 루트 .py 의존성, config
#   penalty_cache_300k_v12_canonical_final.pt        110 MB
#   ckpt_step030000_monotone_reinit.pt               314 MB
#   /workspace/minsuk/data                           141 MB
#
# 안 옮기는 것 (~162 GB)
#   runs/ 157G   evaluations/ 2.2G   logs/ 1.4G
#   real_data_audit_20260728/ 726M   kalbi_v7_*/ 700M
#   isolated_audits/ 50M   evaluation/ 69M
#
# V15 rolling 데이터는 3080 에서 새로 생성한다.  전송보다 생성이 빠르고
# 어차피 V15 가 이번 실험의 핵심이라 옛 데이터는 쓸 일이 없다.
#
#   DST_PORT=<포트> ./migrate_to_3080.sh
#
set -euo pipefail

SRC_HOST=${SRC_HOST:-root@38.29.145.27}
SRC_PORT=${SRC_PORT:-40801}
DST_HOST=${DST_HOST:-root@122.51.254.66}
DST_PORT=${DST_PORT:?"DST_PORT 를 지정하라.  vast.ai 콘솔의 8x3080 카드에서 키 아이콘을 눌러 확인"}

SRC_ROOT=/workspace/minsuk
DST_ROOT=/workspace/minsuk

ssh_src() { ssh -n -p "$SRC_PORT" "$SRC_HOST" "$@"; }
ssh_dst() { ssh -n -p "$DST_PORT" "$DST_HOST" "$@"; }

echo "== 1. 목적지 확인"
ssh_dst "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader && df -h /workspace | tail -1"

echo
echo "== 2. 재초기화 체크포인트 존재 확인"
ssh_src "ls -lh $SRC_ROOT/fm-lm-lambda/ckpt_step030000_monotone_reinit.pt"

echo
echo "== 3. 목적지 디렉터리 생성"
ssh_dst "mkdir -p $DST_ROOT"

echo
echo "== 4. 코드 + 캐시 + 체크포인트 스트리밍"
ssh -n -p "$SRC_PORT" "$SRC_HOST" \
  "cd $SRC_ROOT && tar czf - \
     --exclude='fm-lm-lambda/runs' \
     --exclude='fm-lm-lambda/evaluations' \
     --exclude='fm-lm-lambda/logs' \
     --exclude='fm-lm-lambda/real_data_audit_20260728' \
     --exclude='fm-lm-lambda/isolated_audits' \
     --exclude='fm-lm-lambda/evaluation' \
     --exclude='fm-lm-lambda/kalbi_v7_fullp_blockprox3_v23_k32s32_p100k_effbatch32' \
     --exclude='fm-lm-lambda/kalbi_v7_blocksize_microbenchmark_20260727' \
     --exclude='*.pyc' --exclude='__pycache__' \
     fm-lm-lambda data" \
  | ssh -p "$DST_PORT" "$DST_HOST" "tar xzf - -C $DST_ROOT"

echo
echo "== 5. 이관 검증"
ssh_dst "cd $DST_ROOT/fm-lm-lambda && \
  ls -lh penalty_cache_300k_v12_canonical_final.pt ckpt_step030000_monotone_reinit.pt && \
  du -sh . && \
  echo --- && \
  python3 -c \"import hashlib,sys
for f in ['penalty_cache_300k_v12_canonical_final.pt']:
    h=hashlib.sha256(open(f,'rb').read()).hexdigest()
    print(f, h)\""

echo
echo "== 6. 원본 캐시 sha256 (위와 일치해야 한다)"
ssh_src "cd $SRC_ROOT/fm-lm-lambda && sha256sum penalty_cache_300k_v12_canonical_final.pt"

echo
echo "== 7. 목적지에서 단조 패치 상태 확인"
ssh_dst "cd $DST_ROOT/fm-lm-lambda && \
  ls \$(ls -d kalbi_v6_exactprox10_eta4x_v24*)/selector_v11_canonical.py && \
  grep -c MONOTONE_HINGE_WEIGHTS \$(ls -d kalbi_v6_exactprox10_eta4x_v24*)/selector_v11_canonical.py"

echo
echo "완료.  다음: 3080 에서 verify_monotone.py 재실행"
