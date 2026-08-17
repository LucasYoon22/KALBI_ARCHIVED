#!/bin/bash
# 새 V14(레일리 타깃팅) 합성 데이터 + 기존 KALBI 추론을 무인으로 돌린다.
# 한 단계가 실패해도 나머지는 계속 간다.  약 60~80분.
#
#   bash run_probe_suite.sh 2>&1 | tee ~/Desktop/KALBI/probe_suite.log

set +e

PROJ="$HOME/Desktop/KALBI/project"
export KALBI_PROJECT="$PROJ"
export KALBI_CKPT="$HOME/Desktop/KALBI/ckpt_selector_step030000.pt"
export PYTHONPATH="$PROJ:$HOME/Desktop/KALBI/external/tabicl/src"
GEN="$PROJ/generate_tabiclv2_scm2048_qfilter_bnn128_mixedlinear_supportmix_v14.py"
TAB="$HOME/Desktop/KALBI/external/tabicl"
PY=python3.11

cd "$PROJ" || exit 1
echo "시작 $(date '+%H:%M:%S')"

generate () {                       # $1=이름 $2=N $3=P $4=배치수 $5=oversample
  if [ -f "./$1/manifest.json" ]; then
    echo "[$(date '+%H:%M:%S')] $1 이미 있음 — 생성 건너뜀"
    return
  fi
  echo "[$(date '+%H:%M:%S')] 생성 $1  N=$2 P=$3 batches=$4"
  rm -rf "./$1"
  $PY "$GEN" --out-dir "./$1" --tabicl-src "$TAB" \
    --num-batches "$4" --p-min 3 --force-n "$2" --force-p "$3" \
    --core-oversample-factor "$5" --threads 4 > "/tmp/gen_$1.log" 2>&1
  echo "     -> $(ls ./$1/batch_*.npz 2>/dev/null | wc -l | tr -d ' ') 배치"
}

infer () {                          # $1=이름
  [ -f "./$1/manifest.json" ] || { echo "[$1] 데이터 없음 — 추론 건너뜀"; return; }
  echo "[$(date '+%H:%M:%S')] 추론 $1"
  $PY test_kalbi_on_synth.py --data-dir "./$1" \
    --out "./probe_$1.json" --device cpu > "/tmp/inf_$1.log" 2>&1
  tail -20 "/tmp/inf_$1.log"
}

# A: 이미 생성된 것.  추론만.
infer out_v14_probe

# B: real 벤치마크에 가까운 중간 크기
generate out_v14_probe_b 800 1000 100 3.0
infer out_v14_probe_b

# C: p >> n  고차원 희소 영역
generate out_v14_probe_c 200 1200 100 4.0
infer out_v14_probe_c

# D: 큰 N — real 의 QSAR-TID-10980(N=4612) 쪽
generate out_v14_probe_d 1600 800 60 3.0
infer out_v14_probe_d

echo
echo "[$(date '+%H:%M:%S')] 통합 분석"
$PY analyze_synth_probe.py --out "$HOME/Desktop/KALBI/synth_probe_report"
echo "끝 $(date '+%H:%M:%S')"
