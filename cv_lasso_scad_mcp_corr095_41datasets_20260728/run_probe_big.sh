#!/bin/bash
# 큰 shape 위주로 시간이 허락하는 만큼 계속 돌린다.
# 청크마다: 생성 -> 즉시 추론 -> npz 삭제(디스크 절약).
# 마감 시각을 넘기면 더 생성하지 않고 통합 분석으로 넘어간다.
#
#   nohup bash run_probe_big.sh > ~/Desktop/KALBI/probe_big.log 2>&1 &

set +e

MINUTES=${1:-75}                 # 생성에 쓸 시간 (분)
CHUNK=${2:-24}                   # 청크당 배치 수

PROJ="$HOME/Desktop/KALBI/project"
export KALBI_PROJECT="$PROJ"
export KALBI_CKPT="$HOME/Desktop/KALBI/ckpt_selector_step030000.pt"
export PYTHONPATH="$PROJ:$HOME/Desktop/KALBI/external/tabicl/src"
GEN="$PROJ/generate_tabiclv2_scm2048_qfilter_bnn128_mixedlinear_supportmix_v14.py"
TAB="$HOME/Desktop/KALBI/external/tabicl"
PY=python3.11

cd "$PROJ" || exit 1
DEADLINE=$(( $(date +%s) + MINUTES * 60 ))
echo "시작 $(date '+%H:%M:%S')   생성 마감 $(date -r $DEADLINE '+%H:%M:%S')   청크 ${CHUNK}배치"
echo

# 이미 완성된 폴더가 있으면 먼저 추론해 둔다
for d in out_v14_probe out_v14_probe_b out_v14_probe_c out_v14_probe_d; do
  [ -f "./$d/manifest.json" ] || continue
  [ -f "./probe_$d.json" ] && continue
  echo "[$(date '+%H:%M:%S')] 기존 데이터 추론: $d"
  $PY test_kalbi_on_synth.py --data-dir "./$d" --out "./probe_$d.json" \
      --device cpu > "/tmp/inf_$d.log" 2>&1
  grep -E "선형 태스크|R<0.6|corr\(" "/tmp/inf_$d.log" | head -3
done

# name  N     P     oversample     — P>1024 는 ISIS 스크리닝이 켜지는 영역
SHAPES=(
  "bigp   1500  3000  3.0"
  "hugep   800  4000  3.0"
  "bign   4600  1024  2.0"
  "midp    715  1500  3.0"
)

round=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  round=$((round + 1))
  for spec in "${SHAPES[@]}"; do
    set -- $spec
    NAME=$1; N=$2; P=$3; OS=$4
    [ "$(date +%s)" -lt "$DEADLINE" ] || break 2

    TAG="${NAME}_r${round}"
    SEED=$(( 88076230 + round * 100003 + N + P ))
    DIR="./big_${TAG}"
    echo "[$(date '+%H:%M:%S')] 생성 ${TAG}  N=$N P=$P seed=$SEED"
    t0=$(date +%s)
    rm -rf "$DIR"
    $PY "$GEN" --out-dir "$DIR" --tabicl-src "$TAB" \
      --num-batches "$CHUNK" --p-min 3 --force-n "$N" --force-p "$P" \
      --core-oversample-factor "$OS" --seed "$SEED" --threads 4 \
      > "/tmp/gen_${TAG}.log" 2>&1
    made=$(ls "$DIR"/batch_*.npz 2>/dev/null | wc -l | tr -d ' ')
    echo "     생성 ${made}배치  $(( $(date +%s) - t0 ))초"

    if [ "$made" -gt 0 ] && [ -f "$DIR/manifest.json" ]; then
      $PY test_kalbi_on_synth.py --data-dir "$DIR" \
          --out "./probe_${TAG}.json" --device cpu \
          > "/tmp/inf_${TAG}.log" 2>&1
      grep -E "선형 태스크|정렬 검증" "/tmp/inf_${TAG}.log" | head -2
      rm -f "$DIR"/batch_*.npz          # 디스크 절약, manifest 는 남김
    else
      echo "     생성 실패 — /tmp/gen_${TAG}.log 확인"
      tail -3 "/tmp/gen_${TAG}.log"
    fi
  done
done

echo
echo "[$(date '+%H:%M:%S')] 통합 분석"
$PY analyze_synth_probe.py --out "$HOME/Desktop/KALBI/synth_probe_report"
echo "끝 $(date '+%H:%M:%S')"
