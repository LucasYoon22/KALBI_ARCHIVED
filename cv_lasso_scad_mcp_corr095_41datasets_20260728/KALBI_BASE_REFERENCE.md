# KALBI 기반 정리 — 생성기 · 아키텍처 · 학습 · 진단 · V15

작성 기준: 실제 코드에서 확인한 것만 적었습니다. 문서(교수님 아키텍처 md)와
코드가 어긋나는 곳은 **코드 쪽을 채택하고 차이를 명시**했습니다.
추측인 부분은 "미확인"이라고 따로 표시했습니다.

읽는 순서
1. **1부** 생성기가 무엇을 만드는가
2. **2부** KALBI가 무엇을 받아 무엇을 내는가
3. **3부** 학습과 "CV"에 해당하는 것
4. **4부** V14의 병목 — 진단 사슬
5. **5부** V15의 수정
6. **6부** 파일럿 스터디 전체 기록
7. **7부** 아는 것과 모르는 것
8. **8부** 내일 실행 순서

---

# 0. 기호

| 기호 | 뜻 |
|---|---|
| $N, P$ | 표본 수, 특징 수 |
| $X_c, y_c$ | 행평균을 뺀 설계행렬과 응답 |
| $g_j = \tfrac1N X_{c,j}^\top X_{c,j}$ | Gram 대각 |
| $c_j = \tfrac1N X_{c,j}^\top y_c$ | 특징-응답 내적 |
| $\lambda_{\max} = \max_j \lvert c_j\rvert$ | 모든 계수를 0으로 만드는 최소 $\lambda$ (LASSO 기준) |
| $G = \tfrac1N X^\top X$ | Gram 행렬 (모델은 실제로 만들지 않음) |
| $G_S$ | support $S$ 로 제한한 부분 Gram |
| $w_1 \le \cdots \le w_k$ | $G_S$ 의 고유값 |
| $\rho$ | 페널티 함수, $\rho'(0)$ 이 0에서의 기울기 |
| $R(\beta) = \dfrac{\beta^\top G\beta}{\lVert\beta\rVert^2} = \dfrac{\mathrm{Var}(X\beta)}{\lVert\beta\rVert^2}$ | 레일리 몫 |
| $k$ | 참 support 크기 = $\mathrm{nnz}(\beta_{\text{true}})$ |
| nnz | 추정 계수 중 0이 아닌 개수 |

**$R$ 은 스케일 불변**입니다. $\beta \to c\beta$ 해도 값이 같습니다.
따라서 $\beta$ 의 **방향**만이 $R$ 을 정합니다. 이 사실이 뒤에서 계속 쓰입니다.

---

# 1부. 생성기가 무엇을 만드는가

## 1.1 실행 단위

```
--num-batches N   →  배치 N 개
배치 1 개         →  --batch-size 태스크 (기본 1)  →  batch_XXXXXX.npz 하나
```

배치마다 시드가 결정됩니다.

```python
batch_seed = (seed + 1_000_003 * batch_index) % (2**32 - 1)
random.seed(batch_seed); np.random.seed(batch_seed); torch.manual_seed(batch_seed)
rng = np.random.default_rng(batch_seed)
```

**같은 `--seed` 와 같은 `batch_index` 면 완전히 같은 태스크가 나옵니다.** 재현 가능합니다.
그래서 npz 를 지워도 매니페스트만 있으면 언제든 다시 만들 수 있습니다.

## 1.2 shape 뽑기

`--force-n / --force-p` 를 주면 그대로 씁니다. 안 주면 사전분포에서 뽑습니다.

| 인자 | 기본값 | 뜻 |
|---|---|---|
| `--p-median` | 10000 | $P$ 의 로그정규 중앙값 |
| `--p-log-sigma` | 0.75 | |
| `--raw-n-median` | 5700 | $N$ 의 로그정규 중앙값 |
| `--n-log-sigma` | 0.70 | |
| `--small-n-probability` | 0.10 | 작은 $N$ 분기 확률 |
| `--small-n-max` | 199 | 그때 $N$ 상한 |
| `--main-n-min` | 200 | 일반 분기 $N$ 하한 |
| `--p-min` / `--p-max` | 50 / 200000 | |
| `--max-cells` | 24,000,000 | $N \times P$ 상한 |

생성 후 계약을 강제합니다.

```python
if not (args.n_min <= n <= args.n_max and args.p_min <= p <= args.p_max):
    raise ValueError(f"shape outside contract: N={n}, P={p}")
if n * p > args.max_cells:
    raise ValueError(f"shape exceeds max cells: N={n}, P={p}")
```

> **주의.** `--p-min` 기본값이 50 이라 $P < 50$ 인 태스크는 절대 안 나옵니다.
> 20,000회 사전분포 추출에서 $P$ 중앙값은 9,984 였고, 실제 벤치마크의
> $(N,P)$ 상자에 들어올 확률은 **0.08%** 였습니다. 즉 **shape 분포 자체가
> 실제 데이터와 거의 겹치지 않습니다.** 우리 실험에서는 `--force-n/-p` 로
> 강제했기 때문에 이 문제를 우회했습니다.

## 1.3 X 만들기 — 4단계

### ① TabICLv2 그래프 SCM 코어

`TabICLv2GraphGenerator` 로 최대 **2048** 노드짜리 인과 그래프를 만듭니다.
`--core-oversample-factor` 배만큼 후보를 뽑아 여유를 둡니다 (기본 1.5, 우리는 3~4 사용).

### ② 정확 상관 필터 (노이즈 전)

```python
core_acc = ExactLowCorrelationAccumulator(n, args.corr_threshold, corr_device, args.clip_value)
accepted = core_acc.add(candidate, limit=core_p)
```

**전쌍 Pearson 상관을 정확히 계산해서** $\lvert\mathrm{corr}\rvert < 0.95$ 인 것만 채택합니다.
목표 개수를 못 채우면 `RuntimeError` 로 배치를 통째로 재시도합니다.
생성 로그에 자주 보이던 게 이겁니다:

```
retry batch=6: RuntimeError('SCM qfilter retained too few core features:
               accepted=452, target=620, candidates=2480')
```

### ③ 특징별 이분산 노이즈

```python
signal = standardize(signal)
variance = rng.exponential(scale=1.0 / NOISE_VARIANCE_RATE, size=P)   # v_j ~ Exp(1)
sigma = sqrt(variance)
error = randn(N, P) * sigma[None, :]
noisy = standardize(signal + error)
noisy = clip(noisy, -10, 10);  noisy = standardize(noisy)
stored = noisy.astype(np.float16)
```

특징마다 **독립적인 분산** $v_j \sim \mathrm{Exp}(1)$ 을 뽑아 가우시안 노이즈를 더합니다.

> **이 단계가 4부·6부에서 결정적으로 중요해집니다.**
> 노이즈가 Gram 에 고유값 바닥을 깔아버립니다:
> $$G' = DGD + \operatorname{diag}\!\left(\frac{v_j}{1+v_j}\right)
>   \;\Longrightarrow\; \lambda_{\min}(G') \gtrsim \min_j \frac{v_j}{1+v_j}.$$
> 즉 **V14 는 구조적으로 잘 조건화된 $X$ 만 만듭니다.**

### ④ 노이즈 후 재검증 + BNN 확장

노이즈를 넣은 코어를 **다시 정확 필터로 검증**합니다 (`final_core_acc`).
여기까지가 코어 ≤ 2048 개.

$P > 2048$ 이면 부족분을 BNN 블록으로 채웁니다.

```python
while remaining > 0:
    out_dim = min(args.bnn_output_block, remaining)          # 기본 128
    signal, meta = random_bnn_block(filtered_core, rng,
        output_dim=out_dim, hidden_dim=args.bnn_hidden_dim,  # 기본 64
        parent_min=args.bnn_parent_min, parent_max=args.bnn_parent_max)  # 8 ~ 128
    noisy_block, ... = add_featurewise_exp_variance_gaussian_noise(signal, rng, clip_value=...)
```

코어에서 부모 8~128개를 골라 은닉 64짜리 무작위 신경망을 통과시켜 128개씩 만들고,
블록마다 다시 특징별 노이즈를 넣습니다.

마지막에 이어붙이고 **열 순서를 무작위로 섞습니다** (`perm = rng.permutation(p)`).

> **알아둘 한계.** 정확 전쌍 인증은 **코어에만** 적용됩니다.
> BNN 확장분은 인증 밖이고, `sampled_correlation_audit` 은 256개만 뽑아 보는
> **표본 감사**입니다 (계약 문자열에도 `not_full_pair_certificate` 라고 적혀 있습니다).
> $P > 2048$ 인 태스크는 전쌍 최대 상관이 0.95를 넘을 수 있습니다. **미확인**입니다.

## 1.4 y 만들기 — 50/50 갈림

```python
outcome_source_id  = int((batch_seed + task_index) & 1)          # 0=tabicl, 1=linear
support_reference_id = int(((batch_seed + task_index) >> 1) & 1) # 0=p_based, 1=n_based
```

**시드 홀짝으로 정확히 반반** 갈립니다.

| 분기 | y | 참 $\beta$ |
|---|---|---|
| `tabicl` (짝수) | SCM 이 직접 만든 비선형 응답 | **없음** |
| `linear` (홀수) | 통제된 선형 모델 | **있음** |

우리 분석은 전부 **linear 태스크만** 씁니다. tabicl 쪽은 참 $\beta$ 라는 개념이
존재하지 않기 때문입니다. 그래서 "300 배치 → 150 태스크"가 됩니다.

## 1.5 선형 분기 상세 — 여기가 V15에서 바뀐 곳

`generate_controlled_linear_response` 안에서 순서대로:

**① support 크기 $k$**

| 레짐 | 확률 | $k$ |
|---|---|---|
| ultra_sparse | 0.15 | $\mathrm{Uniform}\{1,\dots,4\}$ |
| sparse | 0.20 | 로그균등 $[5, 64]$ |
| moderate | 0.25 | $10^{U(-4,-2)} \times$ scale |
| large | 0.25 | $10^{U(\log_{10}0.01,\ \log_{10}0.2)} \times$ scale |
| very_dense | 0.15 | $\mathrm{Uniform}[0.2 \cdot n,\ k_{\max}]$ |

scale 은 `support_reference_id` 에 따라 $P$ 또는 $N$ 입니다.
상한은 **V14: $\lfloor (n-1)/3 \rfloor$ → V15: $\lfloor 0.6(n-1) \rfloor$**.

**② support 인덱스 선택** — V15 신규

| 모드 | 확률 | 동작 |
|---|---|---|
| `random` | 1/3 | 기존과 동일 (균등 무작위) |
| `correlated_block` | 1/3 | 씨앗 열 1개 + 절대상관 상위 $k-1$ 개 |
| `mixed` | 1/3 | 절반은 블록, 나머지는 무작위 |

**③ 계수 계열**

| 계열 | 확률 |
|---|---|
| gaussian | 0.30 |
| laplace | 0.20 |
| rademacher × 로그균등 크기 | 0.20 |
| student-t (df 3~10) | 0.15 |
| two-scale gaussian (20%가 3~30배) | 0.15 |

**④ 레일리 타깃팅** — V15 신규, 확률 0.85 (5부에서 상술)

**⑤ 정규화**

```python
signal = x64 @ beta
signal -= signal.mean()
beta /= np.std(signal)          # ⟹ Var(Xβ) = 1 정확히
```

이 한 줄 때문에 **참값에 대해 $R(\beta_{\text{true}}) = 1/\lVert\beta_{\text{true}}\rVert^2$ 가 정확히 성립**합니다.
온전성 검사로 쓸 수 있습니다.

**⑥ SNR** — 로그균등 $[0.2, 50]$
**⑦ 오차 계열** — 여러 종류
**⑧ y 표준화 + 품질 필터**

```
고유값 개수 ≥ 10,  최빈값 비율 ≤ 0.30,  표준화 IQR ≥ 0.20
```

통과 못 하면 `--max-retries 30` 까지 X 부터 다시 만듭니다.
생성 로그의 `unique count 5 < 10`, `dominant ratio 0.84 > 0.30` 이 그것입니다.

## 1.6 npz 에 무엇이 들어가는가

```
X, y,
feature_noise_variance, feature_noise_sigma, feature_noise_family,
feature_source, bnn_block_id,
corr_audit_feature_indices, sampled_corr_audit_max, core_exact_max_abs_corr,
scm_response_mean, scm_response_std, batch_seed, outcome_source_id,
linear_support_size, linear_support_fraction, linear_support_fraction_of_n,
linear_support_regime_id, linear_support_reference_id,
linear_coefficient_family_id, linear_snr_regime_id, linear_error_family_id,
linear_target_snr, linear_realized_snr
```

**참 $\beta$ 벡터와 support 인덱스는 저장되지 않습니다.** 요약 통계만 들어갑니다.
그래서 support 복원율(정밀도/재현율)이나 $\lVert\hat\beta - \beta\rVert$ 를 잴 수 없습니다.
지금은 test $R^2$ 로 간접 평가만 합니다.

**고치려면 두 줄이면 됩니다:**

```python
"linear_support_indices": support.tolist(),
"linear_beta_values": beta[support].tolist(),
```

`manifest.json` 에는 훨씬 자세한 메타데이터가 들어가고, V15 에서
`linear_rayleigh_quotient`, `linear_rayleigh_target`, `linear_subgram_eig_min/max`,
`linear_support_selection` 등이 추가됐습니다.

---

# 2부. KALBI 아키텍처 — 무엇을 받아 무엇을 내는가

## 2.1 최상위 계약

```
입력:  X (N×P),  y (N)
       λ (스칼라),  ρ (Direct96 페널티 함수 하나)
출력:  β (P)
```

$\lambda$ 와 $\rho$ 는 **셀렉터가 만들어 넣어줍니다.** 사람이 안 고릅니다.

호출 형태는 셋으로 나뉩니다.

```python
pre     = model.precompute(x, y)          # O(P) 통계
encoded = model.encode_dataset(pre)       # λ 와 무관한 인코딩 (한 번만)
beta    = model.decode_queries(encoded, lambdas=..., lambda_max=...,
                               penalty_shape_features=..., penalty_d0=...)
```

**핵심 설계:** `encode_dataset` 은 $\lambda$ 에 의존하지 않습니다.
그래서 **한 번 인코딩하고 $\lambda$ 격자 전체를 한 번의 순전파로 풀 수 있습니다.**
`solution_path.py` 가 정확히 이걸 씁니다.

## 2.2 precompute — $O(P)$ 통계만

```
X̃ = X − rowmean(X),   ỹ = y − mean(y)
g_j = X̃_j·X̃_j / N        (gram_diag)
c_j = X̃_j·ỹ  / N        (Xty)
λ_max = max_j |c_j|
+ log|Xty|, feature_mask, 그리고 centered X, y 자체
```

**$P \times P$ Gram 을 만들지 않습니다.** $P = 100{,}000$ 을 감당하는 이유입니다.
$O(P)$ 통계와 행렬-벡터 곱만 씁니다.

## 2.3 encode_dataset — 두 갈래

### ① Global Krylov encoder ($M = 32$)

**표본 공간**에서 크릴로프 기저를 만듭니다. 시작 벡터는 $q_0 = y_c / \lVert y_c\rVert$.
$y_c$ 가 퇴화하면 행별 특징 에너지를 대체 시드로 씁니다.
결과로 삼중대각 $T$ 와 기저 $\{\phi_m\}$ 을 얻습니다.

### ② A/B ISIS 궤적 ($K = 32$, $S = 32$)

```
K = 32 features per stage
S = 32 stages
최대 선택 support = K·S = 1024
고정 릿지 1e-6
중심화 피어슨 상관 스크리닝, 결정적 top-k, 비복원
torch.no_grad() 안에서 계산
```

단계 $s$ 에서 현재 잔차 $r_{s-1}$ 에 대해

$$\mathrm{score}_{j,s} = \frac{\lvert N^{-1} X_{c,j}^\top (r_{s-1} - \bar r_{s-1})\rvert}
{\mathrm{RMS}(X_{c,j})\,\mathrm{RMS}(r_{s-1} - \bar r_{s-1})}$$

상위 32개를 뽑아 $J_s$ 로 삼고, 두 궤적을 만듭니다.

- **A 궤적**: 새로 뽑은 $J_s$ 만으로 현재 잔차에 릿지 적합
- **B 궤적**: 누적 support 에 대한 적합

> **매우 중요.** 이 궤적은 $\lambda$ 나 $\rho$ 에 **의존하지 않습니다.**
> 같은 데이터셋의 모든 $\lambda$ 질의가 **정확히 같은 선택 support** 를 씁니다.
> $\lambda$ 에 따라 support 가 불연속으로 바뀌어 solution path 가 울퉁불퉁해지는 걸
> 막으려는 의도적 설계입니다.

> **오해 정정.** ISIS 의 1024 상한은 **최종 $\beta$ 의 support 를 제한하지 않습니다.**
> `support_mask` 는 A/B 헤드에 **입력 특징**으로 들어갈 뿐입니다.
> 실제로 우리는 $P = 3972$ 에서 $\mathrm{nnz} = 3360$ 을 관측했습니다.
> $\beta$ 는 처음부터 끝까지 **전체 $P$ 차원**입니다.

## 2.4 Direct96 페널티 표현

고정된 96점 **비균등** 격자 $0 = t_0 < t_1 < \cdots < t_{95} = 6$.
0 근처에 조밀하고, 대략 절반이 $[0, 0.3]$ 을 덮습니다.

각 격자점에서 $\rho(t_i)$ 와 $\rho'(t_i)$ 를 번갈아 저장하고 메타데이터 4개를 붙입니다:

$$\left[\rho'(0),\ \rho(1),\ \rho'(6),\ \mathrm{TV}(\rho')\right]$$

따라서 native 폭은 $2 \times 96 + 4 = 196$.

디코더는 96차원 무제약 제안 $\theta$ 를 내고, 첫 좌표는 도함수 절편,
나머지 95개는 hinge 기저 가중치로 씁니다. 조각별 이차식으로 해석됩니다.

**LASSO, SCAD, MCP 같은 고정 형태가 아니라 연속적인 페널티 공간에서 하나를 고르는 구조입니다.**

## 2.5 decode_queries — $\beta$ 를 만드는 5단계

### ① 조건 인코더

$(\lambda,\ \lambda_{\max},\ N,\ P,\ \rho\text{ 특징},\ d_0)$ 을 받아 조건 코드와
FiLM 파라미터를 만듭니다.

### ② Global Krylov 릿지 베이스

$$\alpha = \lambda \exp(s_\alpha) + \epsilon, \qquad (T + \alpha I + \epsilon I)a = U_y,
\qquad \beta_{\mathrm{Krylov}} = \sum_{m=0}^{31} a_m \phi_m$$

베이스 헤드는 폭 256 잔차 MLP, 마지막 스칼라 층은 0으로 초기화됩니다.

### ③ 특징별 A/B FiLM 디코더

A/B 토큰에 조건 FiLM 을 걸어 support-국소 보정 $\Delta\beta^{A/B}$ 를 만듭니다.

### ④ 학습된 종단 임계 초기화

$$\tau^{\mathrm{base}}_{j,q} = \frac{\lambda_q\, d_{0,q}}{\max(g_j,\ 10^{-4})},
\qquad \tau_{j,q} = \tau^{\mathrm{base}}_{j,q} \exp(\ell_{j,q}), \quad \ell \in [-6, 6]$$

$$u_{j,q} = \beta_{\mathrm{Krylov},j,q} + \Delta\beta^{A/B}_{j,q},
\qquad \beta^{(0)}_{j,q} = \mathrm{sign}(u)\max(\lvert u\rvert - \tau,\ 0)$$

마지막에 $[-6, 6]$ 로 클램프.

> **여기가 희소성이 결정되는 곳입니다.** $d_0 = \rho'(0)$ 이 임계값을 **직접 곱**으로
> 정합니다. $\rho'(0)$ 이 작으면 임계가 작아지고 0으로 밀리는 좌표가 없어집니다.
> 게다가 학습된 $\exp(\ell)$ 이 임계를 추가로 줄일 수 있습니다.
> 과선택 문제의 1차 용의자입니다.

### ⑤ 10-step exact composite proximal gradient

$t = 0, \dots, 9$ 에 대해 **전체 $P$** 에서

$$r^{(t)} = X_c\beta^{(t)} - y_c, \qquad
\nabla f = \tfrac1N X_c^\top r^{(t)}, \qquad
z^{(t)} = \beta^{(t)} - \eta_t \nabla f, \qquad
\beta^{(t+1)} = \mathrm{prox}_{\eta_t \lambda \rho}(z^{(t)})$$

스텝 크기는

$$L_{\mathrm{init}} = \max\{0.5\,\lambda_{\max}(T),\ \max_j g_j\},
\qquad \eta_{\mathrm{init}} = \frac{0.95}{L_{\mathrm{init}}}$$

에서 시작해 **eta4x** 규약(최종 채택 eta를 네 배), 음의 곡률 캡,
방향 우세화(majorization)와 백트래킹을 거칩니다.

prox 는 Direct96 조각별 이차식에 대한 **정확한 스칼라 prox** 입니다:

- $u < h_0 = \gamma\rho'(0)$ → **정확히 0**
- 내부 구간 → 닫힌 형태
- 상한 초과 → $t = 6$ 에 고정

**스텝 수는 10 으로 고정**입니다 (`KALBI_EXACT_PG_STEPS = 10`).

### 상수 정리

| 상수 | 값 |
|---|---|
| `KALBI_TOPK` | 32 |
| `KALBI_STAGES` | 32 |
| `KALBI_RIDGE` | $10^{-6}$ |
| `KALBI_EXACT_PG_STEPS` | 10 |
| `KALBI_BETA_BOUND` | 6.0 |
| Krylov $M$ | 32 |

## 2.6 셀렉터 — 실제로 도는 것은 V24

교수님 문서는 **V20** 을 설명하는데, 체크포인트에 실린 것은 **V24** 입니다.
차이가 중요합니다.

| 버전 | $\lambda$ 출력 |
|---|---|
| V20 | **비율** $r \in [10^{-6}, 1]$, $\lambda = r \cdot \lambda_{\max}$ |
| V23/V24 | **절대 $\lambda$** (raw). $\lambda_{\max}$ 로 변환하지 않음 |

계약 문자열이 그렇게 적혀 있습니다:
`independent_raw_lambda_head_no_d0_or_maxdrho_inverse_conversion_v1`.
클램프 범위는 `[V23_RAW_LAMBDA_MIN, V23_RAW_LAMBDA_MAX]`.

우리가 표에 적은 `lam_ratio` 는 $\lambda / \lambda_{\max}$ 로 **우리가 계산한 파생값**입니다.

### V24 = V23 + 가산 512 융합 + 솔버 궤적 탐침

```
V24_PROBE_STEPS      = 3
V24_TRAJECTORY_STATS = 14      →  탐침 궤적 3×14 = 42 차원
V24_GLOBAL_DIM       = 33
V24_PROPOSAL_DIM     = 21
```

계산 흐름:

```
표준화된 context-train X, y
   └─ 응답 분위 탐침 행 (최대 2048)
        └─ 전체 P 잔차 ISIS (K=32, S=32, 4096열 청크 스캔)
             ├─ 잔차 궤적 R (32×32 행 상호작용)
             ├─ 단계별 잔차 β / 주변 β / 누적 조인트 릿지 β
             └─ 단계 메타데이터
   └─ 직접 y 토큰 투영 + 대칭 풀링
        └─ 가산 융합 →  task code (폭 512)
             ├─ 초안 λ₀  (raw lambda 헤드)
             └─ 초안 ρ₀  (penalty MLP → Direct96 헤드)
                  └─ ★ 탐침: 실제 솔버를 β=0 에서 3스텝 돌려봄
                       └─ 14개 통계 × 3스텝
                            └─ 잔차 512 층들
                                 ├─ Δlog λ    →  λ = exp(log λ₀ + Δ)
                                 └─ Δlatent   →  ρ = decode(latent₀ + Δ)
```

**2단계 구조입니다.** 초안을 만들고, 그 초안으로 **솔버를 실제로 3스텝 돌려본 뒤**,
그 궤적을 보고 보정합니다.

> **셀렉터 입력에는 held-out $y$ 가 들어가지 않습니다.**
> 오직 표준화된 context-train $(X, y)$ 뿐입니다. held-out 은 손실 계산에만 씁니다.

---

# 3부. 학습 — "CV"에 해당하는 것

## 3.1 두 종류의 업데이트

### 베타 모델 업데이트

한 태스크에서

- Direct96 페널티 **4개** 무작위
- 페널티마다 $\lambda$ **8개** 무작위
- 합계 **32 질의**

를 한 번에 평가합니다. 각 질의의 목적함수와 손실은

$$\widehat F = \frac{1}{2N}\lVert y - X\widehat\beta\rVert^2 + \lambda\sum_j \rho(\lvert\widehat\beta_j\rvert),
\qquad F_0 = \frac{1}{2N}\lVert y\rVert^2$$

$$\mathcal L_\beta = \mathrm{mean}\left[\log\left(1 + \frac{\widehat F}{F_0}\right)\right]$$

**KKT top-32 는 로깅 진단일 뿐 손실에 안 들어갑니다.**

즉 베타 모델은 "주어진 $(\lambda, \rho)$ 에 대해 목적함수를 잘 최소화하라"만 배웁니다.
예측 성능은 안 봅니다.

### 셀렉터 업데이트 — **이것이 CV 역할**

베타 업데이트 **4번마다 1번** 돕니다.

```
1. 새 태스크의 행을 무작위 순열 후 절반씩 context-train / held-out 분할
      N_train = floor(N/2)
2. context-train 에서만 μ_X, σ_X, μ_y, σ_y 계산 → 양쪽에 적용
      분산 ~0 인 특징은 0으로 마스크
3. 셀렉터가 context-train (X, y) 만 보고 λ, ρ 출력
4. 얼린 KALBI V6 가 context-train 에서 β 계산
5. held-out X 에서 예측
6. held-out y 로 손실 계산
```

$$\mathrm{MSE}_{\mathrm{heldout}} = \frac{1}{N_{\mathrm{test}}}\lVert X_{\mathrm{test}}\widehat\beta - y_{\mathrm{test}}\rVert^2,
\qquad \mathcal L_{\mathrm{sel}} = \mathrm{mean}\left[\log\left(1 + \frac{\mathrm{MSE}}{2}\right)\right]$$

**셀렉터 손실에 없는 것:**

- lambda teacher
- oracle-lambda 지도
- penalty family 라벨
- ranking loss
- KKT loss
- penalty diversity 정규화
- sparsity 정규화

$\lambda$ 와 $\rho$ 는 **held-out 예측 성능만으로 결합 식별**됩니다.

## 3.2 그래서 "CV"와 무엇이 다른가

| | CV LASSO | KALBI 셀렉터 |
|---|---|---|
| 분할 | 5-fold | **1회 50/50**, 스텝마다 새로 뽑음 |
| $\lambda$ 후보 | 40~100개 격자 **전부 적합** | **예측 1회** |
| 선택 | held-out MSE 최소 **탐색** | 학습된 함수의 **상각 예측** |
| 총 적합 횟수 | $5 \times 40 = 200$ | **1** |
| 페널티 형태 | 고정 (L1) | **태스크마다 새로 생성** |

**핵심 차이는 "탐색 vs 상각 예측"입니다.**
CV 는 이 태스크에서 직접 최적을 찾습니다. 셀렉터는 수많은 태스크에서 배운 사상을
한 번 적용합니다. 학습이 완벽해도 상각 예측은 태스크별 탐색을 이기기 어렵습니다.
대신 **200배 빠릅니다.** 이게 설계상 감수한 절충입니다.

> 6부에서 CV LASSO 가 전 구간에서 이기는 결과가 나오는데,
> **속도를 함께 보고해야 공정합니다.** 정확도만 놓고 비교하면 손해입니다.

## 3.3 현재 체크포인트의 학습 설정

`launch_selector_only_v24_v14_snr0p2to50_resume.sh` 에서 확인:

| 항목 | 값 |
|---|---|
| 모드 | `selector_only_frozen_beta` |
| 베타 솔버 | joint step **42700** 에서 동결 |
| 셀렉터 스텝 | 30,000 |
| selector-lr / direct-head-lr | 3e-4 / 3e-4 |
| warmup | 500 |
| batch-per-rank | 1 |
| grad-accum | 4 (셀렉터도 4) |
| GPU | 8 → 유효 배치 32 |
| 데이터 | `..._v14_mixed50_linear_pn50_strictn3_snr0p2to50_p100k_100GB_target3080_...` |
| 설정 | `fpk_v33_kkt0.yaml` |
| 페널티 캐시 | `penalty_cache_300k_v12_canonical_final.pt` (sha 검증) |
| 입력 스케일러 | `v22_input_scaler_train64.pt` (sha 검증) |
| V24 스케일러 | `v24_scaler_rolling256_step42700.pt` (sha 검증) |

**체크포인트 파일 하나에 `model_state_dict`(얼린 베타)와 `selector_state_dict` 가 둘 다 들어 있습니다.**
`audit_selector_only_checkpoint.py` 가 그걸 검증합니다. 추론에는 파일 하나면 됩니다.

> 데이터 경로 이름에 **`v14`** 가 박혀 있습니다. 지금 체크포인트는 **V14 데이터로 학습된 것**입니다.

---

# 4부. V14 의 병목 — 진단 사슬

## 4.1 출발점과 기각된 가설들

보고된 증상: *"합성에서는 solution path 가 목표에 근접하는데 real 에서 망가진다.
계수가 작게 나온다."*

먼저 측정했습니다. 13개 벤치마크에 동일 전처리로 KALBI 를 돌린 결과:

| 가설 | 판정 | 근거 |
|---|---|---|
| 계수가 작다 | **기각** | $\lVert\hat\beta\rVert$ 중앙 0.540 > CV LASSO 0.499 |
| Direct96 정의역 미탐색 | **기각** | 상한 6 도달 0회, $\max\lvert\hat\beta_j\rvert = 0.447$ |
| $\lambda$ 과대 예측 | **기각** | 반대. $\mathrm{nnz} > N$ |
| 클램프가 예측을 망침 | **기각** | 박스 **안에서** 최적화, $R^2$ 손실 0.3% |
| 전처리 불일치 | **기각** | CV LASSO 와 동일 파이프라인 |
| 50/50 혼합 | **기각** | 선형 태스크만 봐도 동일 |
| `beta /= signal_std` | **기각** | y 재표준화가 흡수 (스케일 1/5/30 실험) |
| 쌍별 상관 격차 | **기각** | 조건수 1061 설계의 최대 상관이 0.59 |
| ISIS 스크리닝 (P>1024) | **기각** | 실패율 20% → 18% → **8%**, 반대 방향 |

**전제가 틀렸습니다.** 계수는 작지 않고, 정의역도 문제가 아닙니다.

## 4.2 진짜 지표 — 레일리 몫

실패한 것과 성공한 것을 가른 건 $R(\hat\beta)$ 였습니다. (표는 6부)

## 4.3 왜 V14 가 $R \approx 1$ 만 만드는가

**보조정리 (고유값 평균).**
$X$ 의 열이 표준화되어 있으면 $\mathrm{diag}(G) = \mathbf{1}$ 이므로, **임의의** 크기 $k$ 부분집합 $S$ 에 대해

$$\mathrm{tr}(G_S) = k \quad\Longrightarrow\quad \frac{1}{k}\sum_{i=1}^k w_i = 1$$

**조건수나 상관 구조와 무관하게** 성립합니다.

고유분해 $G_S = V\operatorname{diag}(w)V^\top$ 에서 $\beta_S = Vc$ 로 두면

$$R(\beta) = \frac{\sum_i c_i^2 w_i}{\sum_i c_i^2}$$

즉 **$R$ 은 고유값들의 가중평균**이고 가중치는 $\beta$ 의 방향이 정합니다.

**명제 (V14 의 구조적 제약).**
V14 는 support 를 균등 무작위로, 계수를 등방 분포에서 뽑습니다.
그러면 $\beta$ 방향이 고유기저에서 균등하므로 $\mathbb E[c_i^2] = 1/k$, 따라서

$$\mathbb E[R(\beta)] = \frac{1}{k}\sum_i w_i = 1$$

**귀결.** $R$ 이 1 에서 크게 벗어나려면 $\beta$ 가 특정 고유방향에 집중해야 하는데,
그 확률은 $P(c_1^2 > 1-\varepsilon) \sim \varepsilon^{(k-1)/2}$ 로 $k$ 에 대해 지수적으로 작습니다.

**측정 확인.** V14 가 만든 32개 선형 태스크에서
$\lVert\beta\rVert \in [0.854,\ 1.144]$, 중앙값 0.9998.
$k = 1$ 인 6개는 이론값 1 에 **소수점 여섯째 자리까지** 일치 ($1.000000 \pm 1.1\times10^{-4}$).

우연이 아니라 구조입니다.

## 4.4 세 가지 상한이 R 을 가둔다

$\mathrm{tr}(G_S) = k$ 이고 모든 고유값이 0 이상이므로

$$R \le w_{\max} \le k$$

**$k$ 중앙값이 4 였습니다. $R$ 이 4 를 넘을 수 없었습니다.**

정리하면 세 겹의 제약입니다.

```
R ≤ k                    ← support 크기 상한  (V14: N/3)
R ∈ [w₁, wₖ]             ← support 선택 방식  (무작위 → 스프레드 좁음)
R = Σcᵢ²wᵢ / Σcᵢ²        ← 계수 방향          (등방 → 평균 = 1)
```

**세 개가 동시에 작용해서 $R$ 을 1 에 못 박습니다.** 하나만 풀어서는 안 됩니다.

## 4.5 인과 사슬

```
무작위 support + 등방 β
   → R(β) → 1                          [보조정리, 수학]
   → 학습 데이터가 R ≈ 1 뿐             [32 태스크 측정]
   → 셀렉터·솔버가 R ≠ 1 을 본 적 없음
   → 출력 R̂ 하한 0.425, 추종 기울기 0.293  [합성 251 태스크]
   → 참 R < 0.6 문제는 도달 불가
   → 신호 회수 효율 0%                   [SNR 통제 후]
   → real 13개 중 3개 음수 R²
```

---

# 5부. V15 의 수정

목표: $R(\beta)$ 의 분포를 넓힌다. **세 제약을 각각 하나씩** 풉니다.

## 5.1 support 크기 상한 완화 — 천장을 올림

```python
max_support = min(p, int(MAX_SUPPORT_FRACTION_OF_N * (n - 1)))   # 0.60
```

`3k < N` 강제 assert 를 제거하고 `linear_support_strict_n_third_verified` 플래그로
**기록만** 남깁니다. 기존 매니페스트 키가 전부 살아 있어 다운스트림이 안 깨집니다.

**왜 0.6 인가.** real 13개의 $\mathrm{nnz}/N$ 이 최대 5.24 까지 갔지만,
CV LASSO 기준 $3k < N$ 위반은 12개 중 2개였습니다. 0.6 이면 대부분 덮으면서
$k > N$ 인 완전 병리 구간은 피합니다.

## 5.2 고유값 타깃팅 — 구간 안에서 정확히 조준

목표 $R^\star$ 를 로그균등 $[0.03, 30]$ 에서 뽑고, 감싸는 인접 고유값 쌍
$w_a \le R^\star \le w_b$ 를 찾아

$$c_a^2 = \frac{w_b - R^\star}{w_b - w_a}, \qquad c_b^2 = \frac{R^\star - w_a}{w_b - w_a}$$

**검산.** $c_a^2 + c_b^2 = 1$ 이므로

$$R = c_a^2 w_a + c_b^2 w_b
= \frac{(w_b - R^\star)w_a + (R^\star - w_a)w_b}{w_b - w_a}
= \frac{R^\star(w_b - w_a)}{w_b - w_a} = R^\star$$

**근사가 아니라 정확히 목표값**입니다.

**왜 등방 15% 를 섞나.** 순수하게 두 고유벡터만 쓰면 $\beta$ 가 고유기저에서
rank-2 가 됩니다. 실제 계수가 그런 모양일 리 없으니 비현실적 구조를 학습시킬 위험이 있습니다.
등방 성분을 15% 섞어 흐트러뜨리고, $R$ 은 혼합 **후에** 다시 계산해 기록합니다.

**왜 85% 에만 적용하나.** 15% 는 기존 등방 샘플러를 그대로 둡니다.
기존 분포가 데이터에 남아 있어야 재학습 시 회귀 위험이 없습니다.

**건너뛰는 경우:**

- 동전 던지기 15%
- $k < 2$ — $G_{SS}$ 가 스칼라 1 이라 $R \equiv 1$, 타깃팅이 **수학적으로 불가능**
- $k > 768$ — $k\times k$ 고유분해 비용
- 고유값 스프레드 $< 10^{-9}$, 또는 `eigh` 실패

> **$k=1$ 이면 $R$ 은 무조건 1 입니다.** 열이 하나면 부분 Gram 이 $1\times1$ 이고
> 표준화되어 있으므로 $G_{SS} = 1$, 따라서 $R = \beta_1^2 \cdot 1 / \beta_1^2 = 1$.
> $k=2$ 도 상관 $\rho$ 면 고유값이 $1\pm\rho$ 뿐이라 $R \in [1-\rho, 1+\rho]$ 로 갇힙니다.
> **$R$ 의 자유도는 $k$ 와 함께 커집니다.** 그림에 남아 있는 $R=1$ 세로 줄무늬의 정체입니다.

## 5.3 상관 블록 support — 구간을 넓힘

무작위 support 의 부분 Gram 은 $w_{\max}$ 가 2~3 에서 포화됩니다.
그러면 큰 $R^\star$ 를 뽑아도 $[w_1, w_k]$ 로 클리핑되어 잘립니다.

씨앗 열 하나를 잡고 절대상관 상위 $k-1$ 개를 붙입니다.

```python
cross = x64.T @ (x64[:, seed] - mean[seed])
norm_squared = np.einsum("ij,ij->j", x64, x64) - n * mean * mean
correlation = np.abs(cross) / np.sqrt(norm_squared * norm_squared[seed])
block = np.argpartition(-correlation, block_size - 1)[:block_size]
```

$n\times p$ 복사 없이 $O(np)$ 로 끝납니다. 모드는 확률 1/3 씩
무작위 / 블록 / 절반블록.

**측정: $w_{\max}$ 가 30.5 까지 올라갔습니다.**

### corr 0.95 필터가 제약이 아닌 이유

게르슈고린 원판 정리로

$$w_{\max} \le 1 + (k-1)c$$

$c = 0.95$, $k = 15$ 면 상한이 **14.3**. 목표 6 보다 두 배 넘게 여유가 있습니다.

더 중요한 직관: 등상관 $\rho$ 인 $k$ 개 열의 최대 고유값은 $1 + (k-1)\rho$ 입니다.

| $k$ | $R = 6$ 에 필요한 $\rho$ |
|---|---|
| 7 | 0.85 |
| **15** | **0.36** |
| 30 | 0.17 |

**높은 $R$ 은 "거의 똑같은 열"이 아니라 "완만한 공통 인자를 공유하는 무리"를 요구합니다.**
그리고 real 데이터가 이미 증명합니다 — **똑같은 0.95 필터를 거친 13개에서 $R$ 이 6.06 까지** 나왔습니다.

반대로 **저역**은 $w_{\min} = 1 - \rho$ 이므로 $R = 0.05$ 를 만들려면 $\rho \approx 0.95$ 가 필요합니다.
필터 경계에 정확히 걸립니다. **저역이 고역보다 어려운 구조적 이유**입니다.

## 5.4 진단 메타데이터 추가

```python
"linear_rayleigh_quotient": float(np.var(x64 @ beta) / (beta @ beta)),
"linear_beta_nnz_over_n": float(k / n),
**selection_metadata,     # linear_support_selection, _seed_column, _block_*_abs_corr
**rayleigh_metadata,      # linear_rayleigh_targeted, _target, _subgram_eig_min/max ...
```

계약 문자열도 `strictn3` → `maxsupp0p6n`, 접미사 `rq_v15`.

> **재학습 시 주의.** 학습 런치 스크립트가 `--expected-pipeline-sha256` 을 검증합니다.
> 계약이 바뀌었으므로 **새 해시를 계산해 넣어야** 시작됩니다.

## 5.5 결과

| | V14 | V15 |
|---|---|---|
| $R(\beta)$ 범위 | 전부 $\approx 1$ | **0.106 – 8.47** |
| $R < 1$ 비율 | $\approx 0\%$ | 45% |
| $R > 3$ 비율 | 0% | 13% |
| $\max\lVert\beta\rVert$ | 1.144 | 3.08 |
| $\max k$ | $N/3$ | $0.57N$ |
| $\max_j\lvert\beta_j\rvert$ | 1.000 | 1.767 (상한 6 이내) |

real 의 $[0.110,\ 6.06]$ 이 전부 포함됩니다. (캐시 기준 CPU 측정값. 캐시 이전 서버 측정에서는 하한이 0.056 이었는데, 6.6 에서 보듯 real 개별 수치는 백엔드에 민감합니다.)

### 개발 과정 — 세 번의 반복

| 차수 | 수정 | $R$ 범위 | $R>3$ |
|---|---|---|---|
| 1차 | 가드 + 고유값 타깃팅 (목표 $[0.03, 8]$) | 0.097 – 4.91 | 1% |
| 2차 | + 상관 블록 | 0.106 – 7.48 | 4% |
| 3차 | + 목표 상한 $8 \to 30$, 확률 $0.5 \to 0.85$ | **0.106 – 8.47** | **13%** |

**2차에서 막힌 이유가 배울 점입니다.** 상관 블록으로 $w_{\max}$ 는 30 까지 올렸는데
$R>3$ 이 4% 에 그쳤습니다. 원인은 클리핑이 아니라 **목표 분포 자체**였습니다:
$R^\star \sim \mathrm{LogUniform}(0.03, 8)$ 의 중앙값은 $\sqrt{0.03\times 8} = 0.49$.
**타깃의 절반이 0.49 아래**였습니다. 상한을 30 으로 올리자 로그균등 하에서
$P(R^\star > 3) = \ln(10)/\ln(1000) = 33\%$ 가 되어 해결됐습니다.

---

# 6부. 파일럿 스터디 — 전체 기록

## 6.1 real 13개 벤치마크 (최종, 캐시 기준, CPU)

전처리: 상수/비유한 열 제거 → train 기준 표준화 → corr 0.95 그리디 필터.
필터 인덱스는 배정밀도로 한 번 계산해 `<dataset>/corr095_keep.npy` 에 고정했고,
**결과 $P$ 가 교수님 로더의 `reported_p_after_corr095` 와 13개 전부 일치**합니다.

| dataset | $N$ | $P$ | nnz | $\lVert\hat\beta\rVert$ | $R(\hat\beta)$ | test $R^2$ |
|---|---|---|---|---|---|---|
| QSAR-TID-17080 | 908 | 988 | 946 | 3.525 | **0.110** | **−2.899** |
| QSAR-TID-10331 | 242 | 689 | 665 | 2.378 | **0.211** | **−2.084** |
| QSAR-TID-10547 | 497 | 934 | 714 | 1.364 | **0.426** | **−0.394** |
| QSAR-TID-11575 | 1192 | 1000 | 764 | 1.255 | 0.446 | 0.248 |
| mtp2 | 219 | 779 | 507 | 0.730 | 0.783 | 0.221 |
| QSAR-TID-216 | 416 | 989 | 769 | 0.641 | 2.13 | 0.527 |
| QSAR-TID-12666 | 1794 | 1024 | 779 | 0.540 | 2.62 | 0.580 |
| QSAR-TID-10980 | 4612 | 1024 | 565 | 0.523 | 1.88 | 0.553 |
| QSAR-TID-10929 | 123 | 872 | 696 | 0.513 | 3.37 | 0.281 |
| QSAR-TID-100097 | 715 | 1024 | 502 | 0.459 | 2.83 | 0.277 |
| Acute_Myeloid_Leukemia | 1500 | 3972 | 3360 | 0.310 | 5.40 | 0.684 |
| CancerDrugResponse_methylation | 380 | 768 | 255 | 0.250 | 5.51 | 0.357 |
| CancerDrugResponse_expression | 380 | 697 | 15 | 0.244 | 6.06 | 0.312 |

**$R(\hat\beta)$ 오름차순 정렬입니다. 음수 $R^2$ 셋이 정확히 하위 셋입니다.**

$$0.110 \to -2.90, \quad 0.211 \to -2.08, \quad 0.426 \to -0.39, \quad 0.446 \to +0.25$$

$R(\hat\beta) > 1$ 인 8개는 전부 $0.277 \sim 0.684$.

부수 관측: `clip6 = 0`, $\max_j\lvert\hat\beta_j\rvert \le 0.447$ — Direct96 정의역 $[0,6]$ 의
**7.5% 만** 씁니다.

## 6.2 합성 재현 — 956 태스크

13개는 인과라 하기엔 적습니다. **수정된 생성기로 956개를 만들고 기존 KALBI 체크포인트를
그대로 적용**했습니다. real 데이터는 안 씁니다.

규모: $N_{\mathrm{train}} \in [100, 2300]$, $P \in [620, 4000]$, 12개 실행.

**신호 회수 효율.** SNR $s$ 에서 달성 가능한 최대 $R^2$ 는 $s/(1+s)$ 이므로

$$\eta = \frac{R^2_{\text{test}}}{s/(1+s)}$$

| 참 $R$ | $n$ | $\eta$ 중앙 | $\eta<0$ | SNR 중앙 | 참 $k/N$ |
|---|---|---|---|---|---|
| < 0.3 | 79 | **−0.004** | 52% | **5.41** | 0.305 |
| 0.3 – 0.6 | 115 | 0.427 | 27% | 3.18 | 0.055 |
| 0.6 – 1 | 290 | 0.838 | 12% | 3.96 | 0.008 |
| 1 – 2 | 334 | 0.778 | 14% | 2.84 | 0.012 |
| 2 – 4 | 79 | 0.835 | 5% | 4.24 | 0.080 |
| > 4 | 59 | 0.833 | 7% | 2.40 | **0.280** |
| **< 0.6** | 194 | **0.168** | 37% | | |
| **≥ 0.6** | 762 | **0.820** | 12% | | |

**$R \ge 0.6$ 인 네 구간이 전부 0.78~0.84 로 평평하고, $R < 0.3$ 에서만 0 으로 떨어집니다.**
완만한 하락이 아니라 **문턱**입니다.

**표 안에 통제 두 개가 이미 들어 있습니다.**

1. **SNR** — 붕괴 구간의 SNR 중앙값이 **5.41** 로 정상 구간(2.84)보다 **높습니다.**
   잡음으로 설명 불가.
2. **support 크기** — $R<0.3$ 의 참 $k/N$ 이 0.305 로 큽니다. "$k$ 가 커서 실패한 것 아니냐"는
   반박이 가능한데, **$R>4$ 구간이 $k/N = 0.280$ 으로 비슷한데도 효율 0.833** 입니다.
   support 크기는 원인이 아닙니다.

## 6.3 난이도 통제 — CV LASSO 대조 (705 태스크)

같은 데이터, 같은 분할.

| 참 $R$ | $n$ | KALBI $R^2$ | LASSO $R^2$ | KALBI $R^2<0$ | LASSO $R^2<0$ |
|---|---|---|---|---|---|
| < 0.3 | 55 | 0.022 | 0.069 | 49% | 25% |
| **0.3 – 0.6** | 87 | **0.304** | **0.583** | **24%** | **8%** |
| 0.6 – 1 | 217 | 0.599 | 0.705 | 12% | 2% |
| 1 – 2 | 246 | 0.546 | 0.647 | 15% | 1% |
| 2 – 4 | 56 | 0.669 | 0.757 | 5% | 0% |
| > 4 | 44 | 0.579 | 0.636 | 2% | 0% |

**확실한 것:** $R \in [0.3, 0.6]$ 에서 LASSO 는 $R^2 = 0.583$, 실패율 8%.
KALBI 는 0.304, 24%. **이 구간의 실패는 문제 난이도가 아니라 모델 한계입니다.**

**정직하게 인정할 것:** $R < 0.3$ 에서는 LASSO 도 무너집니다 (0.069, 실패 25%).
그 구간은 실제로 식별이 어렵습니다. **주장은 $[0.3, 0.6]$ 으로 좁혀야 합니다.**

**예상 못 한 것:** KALBI 가 **전 구간에서** 집니다. LASSO 승률 저역 94%, 고역 90%.
$R\approx1$ 근처에서도 0.10 뒤집니다.

| 참 $R$ | 차이 (LASSO − KALBI) |
|---|---|
| > 4 | +0.057 |
| 2 – 4 | +0.088 |
| 1 – 2 | +0.101 |
| 0.6 – 1 | +0.106 |
| **0.3 – 0.6** | **+0.279** |

격차가 저역에서 **2.6배** 벌어지는 건 $R$ 가설과 일치합니다. 하지만 기저 열세는 별개 문제입니다.

> **부호검정.** $R\ge0.6$: $n=563$, 승률 90%, $z = 19$.
> $R<0.6$: $n=142$, 승률 94%, $z = 10.5$. **표본 부족이 아닙니다.**
>
> **다만 956개가 전부 새 생성기 데이터입니다.** $R\approx1$ 근처 태스크도 상관 블록
> support 에 $k$ 가 $0.6N$ 까지 갑니다. "학습 분포에서도 진다"로 읽으면 안 됩니다.
> **구 V14 대조가 아직 없습니다.**

## 6.4 메커니즘 — 출력 공간의 하한

KALBI 가 **낸** $\beta$ 의 레일리 몫을 참값과 비교했습니다.

| 참 $R$ | $R(\hat\beta)$ | 배율 |
|---|---|---|
| 0.188 | 1.634 | 8.7 |
| 0.424 | 1.640 | 3.9 |
| 0.952 | 1.421 | 1.5 |
| 6.057 | 5.444 | 0.9 |

$$\frac{d \log R(\hat\beta)}{d \log R(\beta_{\text{true}})} = 0.293 \qquad (\text{완전 추종이면 } 1)$$

그리고 **$R(\hat\beta)$ 의 최솟값이 0.425**. 참 $R$ 은 0.090 까지 내려갑니다.

**참 해가 거기 있으면 모델의 출력 공간에 그 점이 없습니다.** 못 맞히는 게 아니라 도달 불가입니다.

**같은 정리가 두 번 작용합니다.** 참 $R$ 이 낮은 구간에서 KALBI 는 계수를 얇게 펴 바릅니다:

| | 참 $R<0.3$ | 0.6–1 |
|---|---|---|
| nnz/N | 1.95 | 1.00 |
| $\max_j\lvert\hat\beta_j\rvert$ | 0.122 | 0.540 |

많은 좌표에 조금씩 = 등방 $\beta$ = 정의상 $R\approx1$.
**모델이 학습 데이터의 구조적 편향을 출력 형태로 복제하고 있습니다.**

## 6.5 왜 $R^2$ 가 음수가 되는가 — 증폭 논증

$$R^2 = 1 - \frac{\sum(y_i - \hat y_i)^2}{\sum(y_i - \bar y)^2}$$

분모는 "평균으로 예측"의 오차입니다. 음수면 **평균보다 못하다**는 뜻입니다.
훈련에서는 절편 있는 OLS 면 $R^2 \ge 0$ 이 보장되지만 **test 에는 그런 보장이 없습니다.**

$$\hat\beta^\top G_{\text{test}}\hat\beta
= \underbrace{\hat\beta^\top G_{\text{train}}\hat\beta}_{\text{의도한 신호}}
+ \underbrace{\hat\beta^\top(\Delta G)\hat\beta}_{\text{초과 오차}}$$

$$\lvert\hat\beta^\top \Delta G\,\hat\beta\rvert \lesssim \lVert\hat\beta\rVert^2 \lVert\Delta G\rVert
= \frac{1}{R(\hat\beta)}\lVert\Delta G\rVert$$

**$\lVert\beta\rVert^2 = 1/R$ 이 그대로 증폭 계수가 됩니다.**

| $R$ | 증폭 |
|---|---|
| 6 | 0.17 |
| 1 | 1 |
| 0.3 | 3.3 |
| 0.11 | 9.1 |

기하적으로: $R$ 이 낮다 = $\beta$ 가 $G_{\text{train}}$ 의 거의-영공간 방향.
train 에서는 그 방향으로 $X$ 가 거의 안 움직이니 같은 신호를 내려면 $\beta$ 를 키워야 하고,
test 에서 표본이 조금 달라지면 큰 $\beta$ 가 그 차이를 증폭합니다.

## 6.6 수치 식별성 — 합성과 real 이 갈린다

같은 체크포인트·같은 입력인데 CPU 와 CUDA 가 다른 답을 냈습니다
($\mathrm{nnz}$ 380 대 665, $R^2$ +0.67 대 −2.08). $\lambda$ 는 1% 만 달랐습니다.

**확인한 것들:**

- 로드된 17개 모듈 sha 가 서버와 **전부 일치**. 코드 불일치 아님
- `model_v58_chebyshev` 는 추론 경로에서 **로드조차 안 됨** (학습 `train.py` 전용)
- CPU 는 스레드 수 1/2/4/8 에서 **비트 단위로 동일**. CPU 내부 비결정성 아님

그래서 $Z$ 를 상대 $\varepsilon = 10^{-5}$ 로 흔들고 측정했습니다:

$$\text{증폭} = \frac{\lVert\Delta\hat\beta\rVert/\lVert\hat\beta\rVert}{\varepsilon}$$

| | 태스크 | 증폭 중앙 | $R^2$ 흔들림 중앙 |
|---|---|---|---|
| **real 13개** | 13 | $\mathbf{2.8\times10^4}$ | 0.047 |
| 합성 $R<0.6$ | 9 | $3.3\times10^2$ | 0.001 |
| 합성 $R\ge0.6$ | 31 | $1.5\times10^2$ | 0.000 |

**두 자릿수 배율 차이입니다.**

### 결론 ① 합성 증거는 견고하다

여섯 구간 전부 증폭 150~550, $R^2$ 흔들림 $\le 0.001$,
$\mathrm{corr}(\log R, \log\text{증폭}) = -0.09$ (관계 없음).

**6.2 의 붕괴는 진짜 추정 실패입니다.** 해가 정의되지 않아서가 아닙니다. **956개 실험은 유효합니다.**

### 결론 ② real 에는 별개의 병리가 있다

증폭 최대 $5.9\times10^5$. 13개 중 5개의 $\lambda_{\min}(G_S)$ 가 **수치적으로 음수** —
Gram 은 반양정부호이므로 이는 **참값이 0**, 즉 특이하다는 뜻입니다.
$\mathrm{corr}(\log\lambda_{\min}, \log\text{증폭}) = -0.74$.

$10^{-5}$ 섭동으로 부호가 뒤집힙니다:

| dataset | 기준 $R^2$ | 섭동 범위 |
|---|---|---|
| QSAR-TID-10331 | −2.084 | **−6.07 ~ +0.69** |
| QSAR-TID-17080 | −2.899 | **−2.90 ~ +0.54** |
| QSAR-TID-11575 | +0.248 | **−25.99 ~ +0.25** |

**real 개별 수치는 점추정으로 읽으면 안 됩니다.**

### 왜 생성기는 면역인가

1.3 ③ 의 특징별 노이즈가 Gram 에 바닥을 깝니다:

$$G' = DGD + \operatorname{diag}\!\left(\frac{v_j}{1+v_j}\right)
\;\Longrightarrow\; \lambda_{\min}(G') \gtrsim \min_j \frac{v_j}{1+v_j}$$

**V14 는 구조적으로 잘 조건화된 설계만 만듭니다. 실제 데이터엔 그런 바닥이 없습니다.**

이건 $R$ 과 **독립인 두 번째 생성기-현실 격차**이고, 손잡이가 이미 있습니다 —
`NOISE_VARIANCE_RATE` 가 1 로 고정이고 태스크별로 샘플링되지 않습니다.

> **반례 하나.** AML 은 $\mathrm{nnz}=3360 > N=1500$ 인데 증폭이 463 에 그칩니다.
> `nnz > N` 은 특이성을 **보장**하지만 불안정의 **충분조건은 아닙니다.**

## 6.7 부수 확정 사항

956 태스크 전체에서:

| | |
|---|---|
| $\mathrm{nnz} > N$ | **625 / 956 (65%)** |
| 과선택 배율 중앙 (참 $k$ 대비) | **34배** |
| KALBI nnz/N vs LASSO | 1.50 vs 0.05~0.17 |
| clip6 | **0**, $\max_j\lvert\hat\beta_j\rvert$ 최대 1.767 |
| ISIS 경계 실패율 | $P\le1024$ 20%, 1025–2500 18%, $>2500$ **8%** |

**$\lambda$ 는 과대가 아니라 과소입니다.** Direct96 정의역 상위 70% 는 놀고 있습니다.

### 과선택의 후보 원인 세 개

$$\text{threshold}_j = \frac{\lambda \cdot \rho'(0)}{G_{jj}}\cdot e^{m}$$

| 후보 | 확인 방법 | 상태 |
|---|---|---|
| $\rho'(0)$ 이 작다 ($\kappa = \rho'(0)/\max\rho'$) | `d0`, `max_drho` 기록 | **미측정** |
| 학습된 `log_threshold_multiplier` $m<0$ | 헤드 출력 기록 | **미측정** |
| 10 스텝이 못 밀어냄 | `exact_vs_kalbi.py` | 스크립트 준비됨, **미실행** |

## 6.8 미해결

**같은 `preprocess` 함수가 실행 맥락에 따라 다른 $P$ 를 냈습니다** (AML 3972 vs 526).
단독 호출은 3972 (정답), `sensitivity_probe.py` 안에서는 526.
코드·데이터·정밀도 전부 확인했지만 원인을 특정하지 못했습니다.
필터 결과를 디스크에 캐시해 문제 계열을 제거했습니다. **원인은 여전히 미상입니다.**

---

# 7부. 아는 것과 모르는 것

## 확립된 것

1. V14 는 보조정리와 명제에 의해 $R \approx 1$ 인 과제만 생성한다. 32개 태스크로 확인.
2. 그 결과 KALBI 출력의 $R$ 이 0.425 에서 하한을 갖고 추종 기울기가 0.293 이다.
3. $R \in [0.3, 0.6]$ 에서 CV LASSO 는 성공하고 KALBI 만 실패한다 (705 쌍대).
4. 생성기 수정으로 $R$ 커버리지를 $[0.106, 8.47]$ 로 확장했다.
5. 합성 태스크의 해는 안정적이다 (증폭 $10^2$). real 만 병적이다 ($10^4$–$10^5$).

## 확립되지 않은 것

1. **개입 실험이 없다.** 고친 데이터로 재학습하면 $R<0.6$ 성능이 오르는지 미확인.
   현 단계는 **원인 규명이고 처방은 예측**이다.
2. KALBI 가 전 구간에서 LASSO 에 뒤지는 게 근본 열세인지 분포 이동인지 미확인.
   **구 V14 대조 필요.**
3. $R<0.3$ 은 LASSO 도 저하되므로 주장에서 제외.
4. 참 $R$ 은 전체 $n$ 행에서, 출력 $R$ 은 train 절반에서 계산. 같은 Gram 이 아님.
5. real 개별 수치는 자릿수까지 재현되지 않음. 3개는 $10^{-5}$ 섭동에 $R^2$ 부호가 뒤집힘.
6. 과선택의 직접 원인 ($\kappa$ vs 임계 배수 vs 미수렴) 미측정.
7. $P > 2048$ 태스크의 BNN 확장분은 전쌍 상관 인증 밖.
8. 캐시 이전 `preprocess` 불일치의 원인 미상.

## 방법론적 교훈

이 조사에서 제가 자신 있게 말한 것 중 **세 개가 틀렸습니다**:
ISIS 상한이 출력을 자른다 (아님), `beta/=signal_std` 가 문제다 (아님),
$\beta$ 가 $G$ 와 독립이다 (방향만 독립).

**숫자로 확인되기 전엔 전부 가설로 두시는 게 맞습니다.**

---

# 8부. 내일 실행 순서

## 8.1 사전 점검 (5분)

```bash
# 서버 접속 후
cd /workspace/minsuk/fm-lm-lambda
nvidia-smi
ls -la runs/kalbi_v24_selectoronly_v14_snr0p2to50_frozenbeta_step42700_effbatch32_20260728/checkpoint_selector_step030000.pt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 8.2 V15 생성기 업로드

```bash
# Mac 에서 (rsync 는 vast.ai 에서 막힘, tar 파이프 사용)
cd ~/Desktop/KALBI/cv_lasso_scad_mcp_corr095_41datasets_20260728
tar czf - generate_tabiclv2_scm2048_qfilter_bnn128_mixedlinear_rayleigh_v15.py \
          solution_path.py test_kalbi_on_synth.py infer_kalbi_13.py \
  | ssh -p 40801 root@38.29.145.27 "cd /workspace/minsuk/fm-lm-lambda && tar xzf -"
```

## 8.3 생성 — 두 단계로

**먼저 소량 스모크 (5분).** 반드시 확인하고 넘어가세요.

```bash
cd /workspace/minsuk/fm-lm-lambda
export PYTHONPATH=/workspace/minsuk/fm-lm-lambda
python generate_tabiclv2_scm2048_qfilter_bnn128_mixedlinear_rayleigh_v15.py \
  --out-dir ./out_v15_smoke --tabicl-src ./external/tabicl \
  --num-batches 20 --p-min 3 --force-n 400 --force-p 620 \
  --core-oversample-factor 4.0 --threads 8 --seed 20260811
```

확인할 것:

```bash
python - <<'EOF'
import json, numpy as np
d = json.load(open('out_v15_smoke/manifest.json'))
rows=[]
def walk(o):
    if isinstance(o,dict):
        if 'linear_rayleigh_quotient' in o: rows.append(o)
        for v in o.values(): walk(v)
    elif isinstance(o,list):
        for v in o: walk(v)
walk(d)
R=np.array([r['linear_rayleigh_quotient'] for r in rows])
nb=np.array([r['linear_beta_l2_norm'] for r in rows])
print(len(R),'linear tasks')
print('R      ', np.round(np.quantile(R,[0,.25,.5,.75,1]),3))
print('||b||  ', np.round(np.quantile(nb,[0,.25,.5,.75,1]),3))
print('항등식 오차 (0 이어야 함):', np.abs(nb - 1/np.sqrt(R)).max())
print('타깃팅 적용', sum(bool(r.get('linear_rayleigh_targeted')) for r in rows), '/', len(rows))
EOF
```

- `항등식 오차` 가 $10^{-6}$ 이하 → 구현 정상
- `R` 이 1 한 점이 아니라 퍼져 있음 → 패치 작동
- 아니면 **여기서 멈추고** 원인부터

**그다음 본 생성.** shape 을 real 벤치마크에 맞춰 여러 개 돌리세요.

```bash
for spec in "400 620 4.0" "715 1024 3.0" "1200 1024 3.0" "1500 3000 3.0" "4600 1024 2.0"; do
  set -- $spec
  python generate_tabiclv2_scm2048_qfilter_bnn128_mixedlinear_rayleigh_v15.py \
    --out-dir ./out_v15_n$1_p$2 --tabicl-src ./external/tabicl \
    --num-batches 60 --p-min 3 --force-n $1 --force-p $2 \
    --core-oversample-factor $3 --threads 8 --seed $((20260811 + $1 + $2))
done
```

> $P > 2048$ 인 것(1500×3000)은 BNN 확장이 들어가고 전쌍 인증 밖입니다.
> 결과 해석 시 구분해 두세요.

## 8.4 KALBI 추론 + solution path

```bash
cd /workspace/minsuk/fm-lm-lambda
export PYTHONPATH=/workspace/minsuk/fm-lm-lambda

# 합성 태스크 하나의 경로
python solution_path.py --npz out_v15_n400_p620/batch_000001.npz \
  --out /workspace/minsuk/path_synth_R --device cuda

# real 벤치마크 — R 낮은 것과 높은 것 대비
python solution_path.py --real real_regression_data/QSAR-TID-17080 \
  --out /workspace/minsuk/path_TID17080 --device cuda      # R(β̂)=0.110, R²=−2.90
python solution_path.py --real real_regression_data/CancerDrugResponse_expression \
  --out /workspace/minsuk/path_CDRexpr --device cuda       # R(β̂)=6.06,  R²=+0.31
```

**볼 것 세 가지:**

1. **$\lambda = \lambda_{\max}$ 에서 nnz 가 0 인가.**
   정의상 0 이어야 합니다. 어제 TID-10331 에서 **150 개가 살아 있었습니다.**
   $\lvert X_j^\top y\rvert/N \le \lambda\rho'(0)$ 이 만족 안 된다는 뜻이고 $\rho'(0)$ 이 작다는 신호입니다.

2. **희소 구간이 존재하는가.**
   LASSO 는 nnz 가 0 → P 로 완만하게 갑니다. KALBI 는 $\lambda$ 를 조금만 낮춰도
   바로 P 전체로 포화됐습니다. **경로에 희소 구간 자체가 없었습니다.**

3. **"놓친 폭".** 스크립트가 출력합니다:
   경로 최적 $R^2$ − 셀렉터 지점 $R^2$.
   크면 **경로는 멀쩡한데 $\lambda$ 선택이 틀린 것**, 작으면 **경로 자체가 나쁜 것**.
   교수님 질문에 정확히 답하는 숫자입니다.

## 8.5 GPU vs CPU 대조를 반드시 넣으세요

6.6 에서 확인했듯 real 에서는 백엔드에 따라 답이 달라집니다.
**같은 데이터셋을 두 장치로 돌려 비교하세요.**

```bash
python infer_kalbi_13.py --data-dir real_regression_data --out /workspace/minsuk/infer_gpu.json --device cuda
python infer_kalbi_13.py --data-dir real_regression_data --out /workspace/minsuk/infer_cpu.json --device cpu
```

차이가 크면 **결과를 점추정이 아니라 범위로 보고**해야 합니다.

> `real_regression_data` 를 서버에 다시 올리실 때 **`corr095_keep.npy` 도 같이** 보내세요.
> 안 그러면 서버가 필터를 다시 계산해서 어제 같은 불일치가 재발할 수 있습니다.

## 8.6 여유가 되면

| | 무엇 | 왜 |
|---|---|---|
| A | `exact_vs_kalbi.py` 실행 | 과선택이 **미수렴** 탓인지 **페널티 형태** 탓인지 가름 |
| B | `d0`, `max_drho` 기록 → $\kappa$ | 임계값이 왜 작은지 |
| C | 구 V14 로 CV LASSO 대조 | 근본 열세 vs 분포 이동 |
| D | `linear_support_indices` 저장 | support 복원율 측정 가능해짐 |

A 와 C 가 논지를 가장 크게 바꿉니다.

---

## 한 장 요약

```
V14 :  무작위 support + 등방 β  +  k ≤ N/3  +  특징별 Exp(1) 노이즈
         ↓
       R(β) ≈ 1 만 생성,  잘 조건화된 X 만 생성
         ↓
       셀렉터·솔버가 R ≠ 1 과 병적 조건수를 본 적 없음
         ↓
       출력 R̂ 하한 0.425,  추종 기울기 0.293,  nnz > N 이 65%
         ↓
       참 R < 0.6 인 문제에서 신호 회수 효율 0        [합성 956, SNR·k 통제]
       real 13개 중 R(β̂) 하위 3개가 음수 R²           [벤치마크]

V15 :  가드 0.6N  +  상관 블록 support  +  고유값 타깃팅
         ↓
       R 커버리지 0.106 ~ 8.47  (real 범위 0.110 ~ 6.06 을 전부 포함)
         ↓
       ★ 재학습 효과는 아직 미확인 — 이것이 다음 단계
```
