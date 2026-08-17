# V15 생성기 — V14 대비 변경점

파일: `generate_tabiclv2_scm2048_qfilter_bnn128_mixedlinear_rayleigh_v15.py`
기반: `..._supportmix_v14.py` (1505줄 → 1696줄)

목적: 레일리 몫 $R(\beta) = \mathrm{Var}(X\beta)/\|\beta\|^2$ 의 분포를 넓힌다.
V14는 구조적으로 $R \approx 1$ 인 과제만 생성하는데, 실제 데이터는
$R \in [0.056,\ 6.08]$ 로 100배 퍼져 있다.

---

## 신규 상수 (112–130행)

```python
MAX_SUPPORT_FRACTION_OF_N = 0.60
SUPPORT_SELECTION_NAMES = ("random", "correlated_block", "mixed")
SUPPORT_SELECTION_PROBABILITIES = [1/3, 1/3, 1/3]
RAYLEIGH_TARGET_PROBABILITY = 0.85
RAYLEIGH_TARGET_MIN = 0.03
RAYLEIGH_TARGET_MAX = 30.0
RAYLEIGH_EIGH_MAX_K = 768
RAYLEIGH_ISOTROPIC_MIX = 0.15
```

---

## 변경 1 — support 크기 상한 완화 (`sample_support_size`, 394행)

```diff
- max_support = min(p, (n - 1) // 3)
+ max_support = min(p, int(MAX_SUPPORT_FRACTION_OF_N * (n - 1)))
```

```diff
- if not 1 <= k <= p or not 3 * k < n:
+ if not 1 <= k <= p or k > max_support:
      raise AssertionError(...)
```

근거: $\mathrm{tr}(G_S) = k$ 이고 고유값이 모두 0 이상이므로 $R \le w_{\max} \le k$.
V14의 k 중앙값이 4였으므로 R이 4를 넘을 수 없었다.

기존 메타데이터 키는 전부 유지하고 `linear_support_strict_n_third_verified`
는 값만 `bool(3*k < n)` 으로 바뀐다 (강제에서 기록으로).
신규 키 `linear_support_max_fraction_of_n` 추가.

---

## 변경 2 — 상관 블록 support (`select_support_indices`, 442행 신규)

기존 호출부:

```diff
- support = np.sort(rng.choice(p, size=k, replace=False).astype(np.int64))
+ support, selection_metadata = select_support_indices(x64, k, rng)
```

세 모드를 확률 1/3씩 사용한다.

| 모드 | 동작 |
|---|---|
| `random` | 기존과 동일 |
| `correlated_block` | 씨앗 열 1개 + 절대상관 상위 $k-1$ 개 |
| `mixed` | 절반은 블록, 나머지는 무작위 |

큰 P에서도 $n \times p$ 복사가 없도록 `einsum` 으로 처리:

```python
cross = x64.T @ (x64[:, seed] - mean[seed])
norm_squared = np.einsum("ij,ij->j", x64, x64) - n * mean * mean
correlation = np.abs(cross) / np.sqrt(norm_squared * norm_squared[seed])
```

근거: 무작위 support의 부분 그람은 $w_{\max}$ 가 2–3에서 포화된다.
corr 0.95 필터는 제약이 아니다 — 게르슈고린으로 $w_{\max} \le 1 + (k-1)c$,
$c=0.95$, $k=15$ 이면 상한 14.3. 실제로 $R=6$ 은 15개 열이
등상관 $\rho \approx 0.36$ 만 공유하면 도달한다.

신규 메타데이터: `linear_support_selection`, `_selection_id`,
`_seed_column`, `_block_size`, `_block_mean_abs_corr`, `_block_min_abs_corr`.

---

## 변경 3 — 고유값 타깃팅 (`sample_rayleigh_targeted_coefficients`, 505행 신규)

호출부 (741행):

```python
if rng.random() < RAYLEIGH_TARGET_PROBABILITY:
    coefficient_values, rayleigh_metadata = (
        sample_rayleigh_targeted_coefficients(x64[:, support],
                                              coefficient_values, rng))
```

고유분해 $G_S = V \operatorname{diag}(w) V^\top$, $\beta_S = Vc$ 로 두면

$$R(\beta) = \frac{\sum_i c_i^2 w_i}{\sum_i c_i^2}$$

목표 $R^\star$ 를 감싸는 인접 고유값 쌍 $w_a \le R^\star \le w_b$ 에 대해

$$c_a^2 = \frac{w_b - R^\star}{w_b - w_a}, \qquad
  c_b^2 = \frac{R^\star - w_a}{w_b - w_a}$$

로 두면 $c_a^2 + c_b^2 = 1$ 이므로 $R(\beta) = R^\star$ 가 **정확히** 성립한다.
고유기저에서 rank-2 퇴화를 막기 위해 등방 성분 15%를 혼합하고,
R은 혼합 후에 다시 계산해 기록한다.

건너뛰는 경우 (기존 샘플러 그대로):

- 동전 던지기 15%
- $k < 2$ — $G_{SS}$ 가 스칼라 1이라 $R \equiv 1$, 타깃팅 불가
- $k > 768$ — 고유분해 비용
- 고유값 스프레드가 $10^{-9}$ 미만, 또는 `eigh` 실패

신규 메타데이터: `linear_rayleigh_targeted`, `_target`,
`_realized_presignal`, `linear_subgram_eig_min/max/mean`,
`_eig_index_low/high`, `_isotropic_mix`.

---

## 변경 4 — 진단 메타데이터 (`generate_controlled_linear_response`)

```python
"linear_rayleigh_quotient": float(np.var(x64 @ beta) / (beta @ beta)),
"linear_beta_nnz_over_n": float(k / n),
**selection_metadata,
**rayleigh_metadata,
```

생성기는 항상 `beta /= np.std(signal)` 로 $\mathrm{Var}(X\beta) = 1$ 을 맞추므로

$$R(\beta_{\mathrm{true}}) = \frac{1}{\|\beta_{\mathrm{true}}\|^2}$$

이 정확히 성립한다. 온전성 검사로 쓸 수 있다.

---

## 변경 5 — 계약 문자열

`strictn3` → `maxsupp0p6n`, 접미사에 `rq_v15` / `rq_v4` / `rq_v2`.

**주의:** 학습 런치 스크립트의 `--expected-pipeline-sha256` 이 바뀐다.
재학습 시 새 해시를 계산해 넣어야 한다.

---

## 검증 결과 (실제 V14 실행)

| | V14 | V15 |
|---|---|---|
| $R(\beta)$ 범위 | 전부 $\approx 1$ | **0.106 – 8.47** |
| $R < 1$ 비율 | $\approx 0\%$ | 45% |
| $R > 3$ 비율 | 0% | 13% |
| $\max \|\beta\|$ | 1.144 | 3.08 |
| $\max k$ | $N/3$ | $0.57N$ |
| $\max_j \|\beta_j\|$ | 1.000 | 1.767 (Direct96 상한 6 이내) |

실제 13개 벤치마크의 $[0.056,\ 6.08]$ 이 거의 전부 포함된다.

---

## 회귀 위험

타깃팅을 85%에만 적용하고 나머지는 기존 등방 샘플러를 그대로 둔다.
$k < 2$ 인 과제도 자동으로 기존 경로를 탄다.
따라서 V14 분포가 데이터에 상당 비율 남아 있으며,
재학습 시 기존 성능이 통째로 사라질 위험은 낮다.

## 확인되지 않은 것

- 이 데이터로 재학습하면 $R < 0.6$ 성능이 실제로 오르는가
- 참 β 벡터를 저장하지 않아 support 복원율을 측정할 수 없다
  (`linear_support_indices`, `linear_beta_values` 를 추가하면 가능)
