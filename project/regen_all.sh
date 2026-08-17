#!/bin/bash
set -e
export PYTHONPATH=/Users/junseoyoon/Desktop/KALBI/external/tabicl/src
GEN=generate_tabiclv2_scm2048_qfilter_bnn128_mixedlinear_supportmix_v14.py
TAB=/Users/junseoyoon/Desktop/KALBI/external/tabicl
PY=python3.11

echo "### baseline 확인"
N=$(grep -c "noise_rate=task_noise_rate" $GEN || true)
echo "  noise_rate 전달 횟수: $N   (0 이면 baseline)"
echo

echo "### 기존 폴더 삭제"
rm -rf out_v14_test out_v14_highp out_v14_bnn out_v14_natural
echo "  삭제 완료"
echo

echo "### 1/4  저차원  N=200 P=16   (20 batch)"
$PY $GEN --out-dir ./out_v14_test --tabicl-src $TAB \
  --num-batches 20 --p-min 3 --force-n 200 --force-p 16 --threads 4

echo "### 2/4  중차원  N=131 P=620  (25 batch)"
$PY $GEN --out-dir ./out_v14_highp --tabicl-src $TAB \
  --num-batches 25 --p-min 3 --force-n 131 --force-p 620 \
  --core-oversample-factor 4.0 --threads 4

echo "### 3/4  BNN경로 N=800 P=3000 (8 batch)"
$PY $GEN --out-dir ./out_v14_bnn --tabicl-src $TAB \
  --num-batches 8 --force-n 800 --force-p 3000 --threads 4

echo "### 4/4  자연shape (12 batch)"
$PY $GEN --out-dir ./out_v14_natural --tabicl-src $TAB \
  --num-batches 12 --threads 4

echo
echo "### 생성 완료"
for d in out_v14_test out_v14_highp out_v14_bnn out_v14_natural; do
  echo "  $d : $(ls $d/batch_*.npz 2>/dev/null | wc -l | tr -d ' ') 개"
done
