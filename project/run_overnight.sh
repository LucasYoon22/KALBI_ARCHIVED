#!/bin/bash
# 밤새 돌리는 최종 검증.
#   - 개선된 V14 로 합성 태스크 생성
#   - 기존 KALBI 추론  +  같은 분할에 CV LASSO 대조
#   - 청크마다 추론 직후 npz 삭제 (디스크 절약)
#   - 마감 시각을 넘기면 생성을 멈추고 통합 분석
#
#   nohup bash run_overnight.sh 390 > ~/Desktop/KALBI/overnight.log 2>&1 &

set +e

MINUTES=${1:-390}                # 생성에 쓸 시간 (기본 6시간 30분)
CHUNK=${2:-30}

PROJ="$HOME/Desktop/KALBI/project"
export KALBI_PROJECT="$PROJ"
export KALBI_CKPT="$HOME/Desktop/KALBI/ckpt_selector_step030000.pt"
export PYTHONPATH="$PROJ:$HOME/Desktop/KALBI/external/tabicl/src"
export OMP_NUM_THREADS=4
GEN="$PROJ/generate_tabiclv2_scm2048_qfilter_bnn128_mixedlinear_supportmix_v14.py"
TAB="$HOME/Desktop/KALBI/external/tabicl"
PY=python3.11

cd "$PROJ" || exit 1
DEADLINE=$(( $(date +%s) + MINUTES * 60 ))
echo "시작 $(date '+%m-%d %H:%M')   생성 마감 $(date -r $DEADLINE '+%m-%d %H:%M')"
echo "청크 ${CHUNK}배치   CV LASSO 대조 켬"
echo

# 빠른 shape 를 많이 섞어 표본 수를 벌고, 큰 shape 로 범위를 덮는다
SHAPES=(
  "fast    400   620  4.0"
  "fast2   500   900  4.0"
  "mid     715  1500  3.0"
  "fast3   300   800  4.0"
  "big    1500  3000  3.0"
  "fast4   600  1000  4.0"
  "wide    800  4000  3.0"
  "tall   4600  1024  2.0"
)

round=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  round=$((round + 1))
  for spec in "${SHAPES[@]}"; do
    set -- $spec
    NAME=$1; N=$2; P=$3; OS=$4
    [ "$(date +%s)" -lt "$DEADLINE" ] || break 2

    TAG="${NAME}_n${round}"
    SEED=$(( 900000 + round * 7919 + N * 13 + P ))
    DIR="./on_${TAG}"
    printf "[%s] %-10s N=%-5s P=%-5s " "$(date '+%H:%M')" "$TAG" "$N" "$P"
    t0=$(date +%s)
    rm -rf "$DIR"
    $PY "$GEN" --out-dir "$DIR" --tabicl-src "$TAB" \
      --num-batches "$CHUNK" --p-min 3 --force-n "$N" --force-p "$P" \
      --core-oversample-factor "$OS" --seed "$SEED" --threads 4 \
      > "/tmp/g_${TAG}.log" 2>&1
    made=$(ls "$DIR"/batch_*.npz 2>/dev/null | wc -l | tr -d ' ')
    printf "gen %s배치 %ss  " "$made" "$(( $(date +%s) - t0 ))"

    if [ "$made" -gt 0 ] && [ -f "$DIR/manifest.json" ]; then
      $PY test_kalbi_on_synth.py --data-dir "$DIR" \
          --out "./probe_${TAG}.json" --device cpu --baseline \
          > "/tmp/i_${TAG}.log" 2>&1
      got=$(grep -o "선형 태스크 [0-9]*" "/tmp/i_${TAG}.log" | tail -1 | tr -d '선형 태스크 ')
      printf "-> %s태스크\n" "${got:-0}"
      rm -f "$DIR"/batch_*.npz
    else
      printf "-> 생성실패\n"
      tail -2 "/tmp/g_${TAG}.log"
    fi

    # 매 라운드 끝에 중간 집계를 남긴다
  done
  echo "  --- 라운드 ${round} 종료, 누적 $(ls probe_*.json 2>/dev/null | wc -l | tr -d ' ') 파일 ---"
  $PY describe_probe.py > "$HOME/Desktop/KALBI/overnight_progress.txt" 2>&1
done

echo
echo "[$(date '+%H:%M')] 통합 분석"
$PY describe_probe.py | tee "$HOME/Desktop/KALBI/overnight_summary.txt"
$PY analyze_synth_probe.py --out "$HOME/Desktop/KALBI/synth_probe_report"
echo "끝 $(date '+%m-%d %H:%M')"
