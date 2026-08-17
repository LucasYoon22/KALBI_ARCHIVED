---
title: Selected Real Regression Datasets — Corr40 and 0.2 to 0.9 R2 Filter
date: 2026-08-03
tags:
  - sparse-regression
  - real-data
  - OpenML
  - LASSO
  - EDA
  - reproducibility
---

# Selected Real Regression Datasets — Corr40 and 0.2 < R² < 0.9 Filter

## Technical summary

25개 OpenML 회귀 후보 중 아래 eligibility intersection을 적용해 **13개를 유지하고 12개를 제외**했다.

- `corr095_removed_pct ≤ 40%`
- 그리고 discovery `0.2 < CV-LASSO test R² < 0.9`

남은 13개의 중앙 corr 제거율은 **3.52%**, discovery test R² 중앙값은 **0.5351**이다. 13개 중 10개는 corr filter 후에도 `P > n_train`을 유지한다. 다만 이 R²는 후보 탐색에 사용된 단일 split 점수이므로 최종 benchmark 성능으로 간주하지 않는다.

최종 loader: [load_selected_real_regression.py](load_selected_real_regression.py)

## 두 조건의 교집합에 속하지 않는 12개가 제거됐다

아래 그림에서 파란 원은 유지, 금색 X는 제외된 데이터다. 세로선은 corr 제거율 40%, 가로선은 test R²의 0.2 및 0.9 cutoff다. Corr은 40%와 같으면 유지할 수 있지만, R²는 두 경계 모두 strict이므로 정확히 0.2 또는 0.9이면 제외한다.

![[selection_filter_corr40_r2_0p2_to_0p9.png]]

`mne-sample-meg-auditory`는 corr와 upper R² 조건을 동시에 위반한다. Corr 기준으로 이미 빠지던 8개 외에, lower R² cutoff가 `QSAR-TID-30028`, `Santander_transaction_value`, `CancerDrugResponse_copynumber`, `QSAR-TID-101557` 4개를 추가로 제외한다.

## 유지된 데이터 13개

`P after`는 archived screening split의 training rows에서 constant filter와 greedy `|corr| ≥ 0.95` filter를 적용한 뒤의 feature 수다. `P/n`의 분모도 outer-training row 수다.

| Dataset | OpenML ID | Target | N | P raw → after | Corr removed | P/n train | Discovery test R² |
|---|---:|---|---:|---:|---:|---:|---:|
| Acute_Myeloid_Leukemia | [46134](https://www.openml.org/d/46134) | MYC | 1,875 | 4,251 → 3,972 | 6.56% | 2.648 | 0.7289 |
| QSAR-TID-10331 | [3711](https://www.openml.org/d/3711) | MEDIAN_PXC50 | 303 | 1,024 → 689 | 32.71% | 2.847 | 0.6701 |
| QSAR-TID-17080 | [3790](https://www.openml.org/d/3790) | MEDIAN_PXC50 | 1,135 | 1,024 → 988 | 3.52% | 1.088 | 0.6358 |
| QSAR-TID-12666 | [3353](https://www.openml.org/d/3353) | MEDIAN_PXC50 | 2,243 | 1,024 → 1,024 | 0.00% | 0.571 | 0.5758 |
| QSAR-TID-10980 | [3277](https://www.openml.org/d/3277) | MEDIAN_PXC50 | 5,766 | 1,024 → 1,024 | 0.00% | 0.222 | 0.5565 |
| QSAR-TID-216 | [3515](https://www.openml.org/d/3515) | MEDIAN_PXC50 | 520 | 1,024 → 989 | 3.42% | 2.377 | 0.5442 |
| QSAR-TID-11575 | [4015](https://www.openml.org/d/4015) | MEDIAN_PXC50 | 1,490 | 1,024 → 1,000 | 2.34% | 0.839 | 0.5351 |
| CancerDrugResponse_methylation | [46139](https://www.openml.org/d/46139) | Methotrexate_GDSC_1008 | 475 | 808 → 768 | 4.95% | 2.021 | 0.3518 |
| QSAR-TID-10929 | [3161](https://www.openml.org/d/3161) | MEDIAN_PXC50 | 154 | 1,024 → 872 | 14.84% | 7.089 | 0.3482 |
| QSAR-TID-100097 | [4041](https://www.openml.org/d/4041) | MEDIAN_PXC50 | 894 | 1,024 → 1,024 | 0.00% | 1.432 | 0.3427 |
| CancerDrugResponse_expression | [46138](https://www.openml.org/d/46138) | Methotrexate_GDSC_1008 | 475 | 697 → 697 | 0.00% | 1.834 | 0.3415 |
| QSAR-TID-10547 | [4057](https://www.openml.org/d/4057) | MEDIAN_PXC50 | 622 | 1,024 → 934 | 8.79% | 1.879 | 0.2594 |
| mtp2 | [430](https://www.openml.org/d/430) | oz1143 | 274 | 1,142 → 779 | 31.79% | 3.557 | 0.2283 |

## 제거된 데이터 12개

| Removed dataset | OpenML ID | Corr removed | Test R² | 제거 사유 |
|---|---:|---:|---:|---|
| mne-sample-meg-auditory | [43884](https://www.openml.org/d/43884) | 44.40% | 0.9495 | corr > 40% 및 R² ≥ 0.9 |
| LimeSoda_H.138 | [47079](https://www.openml.org/d/47079) | 99.00% | 0.8218 | corr > 40% |
| QSAR-TID-11004 | [3133](https://www.openml.org/d/3133) | 58.69% | 0.7307 | corr > 40% |
| LimeSoda_UL.120 | [47099](https://www.openml.org/d/47099) | 99.42% | 0.6395 | corr > 40% |
| LimeSoda_SL.125 | [47094](https://www.openml.org/d/47094) | 99.52% | 0.4816 | corr > 40% |
| QSAR-TID-10654 | [4055](https://www.openml.org/d/4055) | 62.79% | 0.2049 | corr > 40% |
| Cancer_Drug_Response_full | [46137](https://www.openml.org/d/46137) | 94.98% | 0.0238 | corr > 40% 및 R² ≤ 0.2 |
| QSAR-TID-20128 | [3758](https://www.openml.org/d/3758) | 60.25% | −0.0302 | corr > 40% 및 R² ≤ 0.2 |
| QSAR-TID-30028 | [4096](https://www.openml.org/d/4096) | 0.20% | 0.1731 | R² ≤ 0.2 |
| Santander_transaction_value | [42572](https://www.openml.org/d/42572) | 9.90% | 0.1545 | R² ≤ 0.2 |
| CancerDrugResponse_copynumber | [46141](https://www.openml.org/d/46141) | 16.53% | 0.1471 | R² ≤ 0.2 |
| QSAR-TID-101557 | [3059](https://www.openml.org/d/3059) | 0.29% | −0.0982 | R² ≤ 0.2 |

## EDA 결과

### Feature family 구성은 QSAR 쪽으로 강하게 기운다

| Feature family | Count | Share |
|---|---:|---:|
| QSAR molecular fingerprints | 9 | 69.2% |
| Omics / drug response | 3 | 23.1% |
| Molecular descriptors | 1 | 7.7% |

이 필터는 smooth spectrum과 highly redundant full-expression view를 제거하면서 QSAR fingerprint를 대부분 유지한다. 따라서 최종 평균 성능은 dataset-level macro average만 보고 끝내지 말고, QSAR family가 aggregate를 지배하지 않도록 source-family-balanced score도 함께 계산해야 한다.

### 13개 중 3개는 더 이상 P > n이 아니다

이번 요청의 두 cutoff만 적용하면 아래 세 데이터는 유지되지만 corr filter 후 `P/n_train < 1`이다.

| Dataset | P after | P/n train | Test R² |
|---|---:|---:|---:|
| QSAR-TID-12666 | 1,024 | 0.571 | 0.5758 |
| QSAR-TID-10980 | 1,024 | 0.222 | 0.5565 |
| QSAR-TID-11575 | 1,000 | 0.839 | 0.5351 |

Foundation model의 core target이 반드시 high-dimensional sparse regression이라면 이 세 개는 삭제하기보다 **low-dimensional control track**으로 분리하는 편이 낫다. 나머지 10개는 `P > n_train`을 유지한다.

### 예측력은 넓게 분산되어 있다

- Discovery test R² 중앙값: **0.5351**
- Discovery test R² 범위: **0.2283–0.7289**
- Eligibility band 충족: **13/13**
- Corr 제거율 중앙값: **3.52%**
- Corr 제거율 범위: **0–32.71%**
- Corr filter 후 `P/n_train` 중앙값: **1.834**

Lower R² cutoff를 추가하면서 negative 및 weak-signal task가 모두 제거됐다. 따라서 이 13개만으로 평균 성능을 보고하면 survivorship bias가 커질 수 있으므로, 제거된 low-R² task는 별도 stress/control registry에 보존하는 것이 안전하다.

## 재현 가능한 train/test split contract

Loader는 다음 contract를 사용한다.

1. OpenML dataset name 대신 **data ID**를 사용해 version을 고정한다.
2. OpenML metadata의 MD5를 embedded registry와 비교하고 다르면 중단한다.
3. numeric target이 아닌 row만 제거한다. Predictor preprocessing은 적용하지 않는다.
4. `seed = 20260803`, `test_fraction = 0.20`을 사용한다.
5. `numpy.random.Generator(PCG64).permutation`으로 row position을 섞는다.
6. `n_train = floor(0.8 × N)`으로 고정하고 train/test position을 각각 정렬한다.
7. 원래 OpenML row index와 그 SHA-256을 `split_manifest.json`에 저장한다.
8. scaling, imputation, constant filter, corr filter는 downstream outer-training rows에서만 fit한다.

동일 seed와 OpenML data ID를 사용하면 같은 indices가 생성되며, 저장된 `.npy` index와 SHA-256이 최종 lock 역할을 한다.

## Loader 사용법

필수 패키지: `numpy`, `pandas`, `scipy`, `scikit-learn`.

```bash
# 데이터 목록만 출력
python load_selected_real_regression.py --list

# split invariant 검증
python load_selected_real_regression.py --self-test

# 한 개만 다운로드하고 split 저장
python load_selected_real_regression.py \
  --dataset QSAR-TID-10331 \
  --output-dir ./real_regression_data

# 13개 전부 다운로드 — 용량과 시간이 큼
python load_selected_real_regression.py \
  --all \
  --output-dir ./real_regression_data
```

각 dataset directory에는 다음 파일이 저장된다.

- Dense X: `X_train.npy`, `X_test.npy`
- Sparse X: `X_train.npz`, `X_test.npz`
- `y_train.npy`, `y_test.npy`
- `train_original_row_index.npy`, `test_original_row_index.npy`
- `feature_names.json`
- `split_manifest.json`

Root output directory에는 성공/실패 목록을 담은 `batch_manifest.json`이 저장된다. 기본 실행은 전체 데이터를 자동 다운로드하지 않고 registry만 출력한다.

## 검증 상태와 제한

- Python syntax compile과 internal split self-test를 통과했다.
- Sparse ARFF 경로는 `QSAR-TID-10331`에서 실제 download/save smoke test를 통과했다: `(N, P) = (303, 1024)`, train/test = `242/61`.
- Dense ARFF 경로는 `CancerDrugResponse_expression`에서 통과했다: `(N, P) = (475, 697)`, train/test = `380/95`.
- 나머지 15개는 OpenML metadata의 name, target, version, active status, MD5를 API로 검증했지만 이 문서 작성 시 전체 payload를 모두 재다운로드하지는 않았다.
- Discovery R²는 archived single 80/20 screen 결과다. Dataset filtering에 이미 사용했으므로 새 locked split에서 재계산한 결과를 최종 untouched benchmark 성능으로 부르면 안 된다.
- Cancer Drug Response expression, methylation, copy-number는 동일한 475개 row와 target을 공유하는 multi-view task다. 독립 dataset 세 개로 가중하면 안 된다.

## 다음 실행 권고

1. `--all`로 13개를 내려받고 생성된 `batch_manifest.json`을 version control 또는 artifact bundle에 보존한다.
2. Core high-dimensional track 10개와 low-dimensional control track 3개를 분리한다.
3. QSAR에는 random row split 외에 scaffold split manifest를 추가한다.
4. 최종 성능은 새 outer splits 또는 repeated nested CV에서 계산하고, 현재 discovery R²는 dataset-registry metadata로만 보존한다.

## Source notes

- Screening metrics: `candidate_25_corr_lasso_r2.csv`, reconstructed from `session-export-1785739335758.zip`.
- Dataset identity and target: OpenML JSON API and pinned dataset IDs.
- Loader API: scikit-learn `fetch_openml(data_id=..., as_frame=False)`.
