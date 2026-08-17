# V14 데이터 생성기와 Lambda Selector V20 구조 정리

작성일: 2026-08-02
출처: `generate_tabiclv2_scm2048_qfilter_bnn128_mixedlinear_supportmix_v14.py`,
`KALBI V6 and Lambda Selector V20 Detailed Architecture 2026-07-24.md`

---

# 1부. V14는 데이터를 어떻게 생성하는가

## 1.1 한 줄 요약

V14는 **(X, y) 쌍만 생성**한다. λ도, β도, support 인덱스도 저장하지 않는다.
저장되는 것은 X, y와 생성 과정 메타데이터뿐이다.

## 1.2 태스크 1개를 만드는 전체 순서

```
[1] shape 샘플링        → (N, P) 결정
[2] SCM core 생성       → TabICL 공식 graph-SCM으로 후보 feature 생성
[3] feature 노이즈      → 컬럼별 v_j ~ Exp(1), eps ~ N(0, v_j)
[4] qfilter            → 모든 pair |corr| < 0.95 되도록 컬럼 선별
[5] BNN 확장           → P > 2048이면 128열 블록으로 랜덤 BNN feature 추가
[6] y 생성             → 50% 공식 SCM y / 50% 통제된 선형 y
[7] 품질 필터          → y가 퇴화면 거부하고 [2]부터 재시도 (최대 30회)
[8] 저장               → float16 X, float32 y, 메타데이터
```

## 1.3 각 단계 상세

### [1] Shape 샘플링

`exp122_shape_sampler.sample_shape_pair`가 (N, P)를 뽑는다. 기본 파라미터:

| 인자 | 기본값 | 의미 |
|---|---:|---|
| `--n-min` / `--n-max` | 32 / 40,000 | N 범위 |
| `--raw-n-median` | 5,700 | N의 log-normal 중앙값 |
| `--n-log-sigma` | 0.70 | N의 log 표준편차 |
| `--main-n-min` | 200 | "주 성분"의 N 하한 |
| `--small-n-max` | 199 | "소 성분"의 N 상한 |
| `--small-n-probability` | 0.10 | N < 200이 뽑힐 확률 |
| `--p-min` / `--p-max` | 50 / 200,000 | P 범위 |
| `--p-median` | 10,000 | P의 log-normal 중앙값 |
| `--p-log-sigma` | 0.75 | P의 log 표준편차 |
| `--max-cells` | 24,000,000 | N × P 상한 |

생성 후 1046행에서 하드 검증한다.

```python
if not (args.n_min <= n <= args.n_max and args.p_min <= p <= args.p_max):
    raise ValueError(f"shape outside contract: N={n}, P={p}")
```

`--force-n` / `--force-p`로 강제해도 이 검증은 통과해야 한다.

### [2] SCM core 생성

`ConfiguredTabICLv2GraphGenerator` (TabICLv2 공식 `PriorDataset`의
`graph_scm` 모드)로 feature를 생성한다.

- `--scm-core-cap 2048`: core feature 최대 2,048개 (961행에서 2048 고정 검증)
- `--core-oversample-factor 1.5`: 목표 P의 1.5배를 후보로 뽑음
  (qfilter에서 탈락할 것을 예상한 여유분)
- 커밋 고정: `EXPECTED_TABICL_COMMIT = "46b9196..."`, 워킹트리 clean 요구

### [3] Feature-wise 노이즈 주입

`add_featurewise_exp_variance_gaussian_noise` (642행)

```python
variance = rng.exponential(scale=1.0, size=P)      # v_j ~ Exp(rate=1), 컬럼마다 독립
sigma    = sqrt(variance)
error    = randn(N, P) * sigma[None, :]
noisy    = standardize(signal + error)
noisy    = clip(noisy, -10, 10); noisy = standardize(noisy)
stored   = noisy.astype(float16)
```

- 컬럼 j의 신호 보존율(attenuation) = `1 / sqrt(1 + v_j)`, 중앙값 약 0.77
- **v_j가 컬럼마다 i.i.d.** 라서 노이즈에 컬럼 간 구조가 없다
- 최종 저장은 float16

### [4] Correlation qfilter

`ExactLowCorrelationAccumulator`로 순차 탐색하며,
이미 채택된 모든 컬럼과의 |Pearson| < 0.95인 컬럼만 채택한다.

- P ≤ 2048: **전체 pair exact certificate**
- P > 2048: SCM core 2,048개만 exact, 이후 BNN 확장분은
  **256개 샘플 감사만** (`--corr-audit-features 256`)

```python
CORRELATION_AUDIT_CONTRACT = "deterministic_uniform_feature_sample_float16_
 pearson_not_full_pair_certificate_v1"
```

즉 고차원에서는 전체 pair가 0.95 미만임을 보장하지 않는다.

채택 수가 목표에 못 미치면 재시도한다:
`RuntimeError('SCM qfilter retained too few core features: accepted=..., target=...')`

### [5] BNN 확장 (P > 2048일 때만)

`random_bnn_block` (677행). qfiltered core를 부모로 삼아 랜덤 BNN을 통과시켜
128열씩 추가 feature를 만든다.

- 부모 수: 8~128에서 log-uniform (`--bnn-parent-min/max`)
- 은닉 폭 64 (`--bnn-hidden-dim`), 출력 블록 128 (`--bnn-output-block`)
- 활성함수: gelu / tanh / relu / sigmoid / sin_pi 중 랜덤
- 깊이: 1 (75%) 또는 2 (25%)
- 가중치 prior scale: log-uniform [0.60, 1.60] 등

### [6] y 생성 — 50/50 혼합

**certificate index의 홀짝으로 결정된다** (랜덤 아님).

```
짝수 → outcome_source = tabicl  (공식 SCM 응답 유지)
홀수 → outcome_source = linear  (통제된 선형 모델로 교체)
```

따라서 정확히 50%씩이다.

#### (a) tabicl 브랜치

TabICLv2 graph-SCM이 만든 y를 그대로 쓰고, 중심화·표준화·clip(±10)·재표준화만 한다.
**비선형이다.**

#### (b) linear 브랜치 — `generate_controlled_linear_response` (533행)

**support 크기 k 결정**

```python
max_support = min(p, (n - 1) // 3)          # 핵심 제약: 3k < N
```

support reference는 P기준/N기준 50:50. regime은 5종:

| regime | 확률 | k (scale = P 또는 N) |
|---|---:|---|
| ultra_sparse | 0.15 | 1 ~ 3 |
| sparse | 0.20 | 5 ~ 64 (log-uniform) |
| moderate | 0.25 | 1e-4 ~ 1e-2 × scale |
| large | 0.25 | 1e-2 ~ 0.2 × scale |
| very_dense | 0.15 | 0.2 ~ 1.0 × scale |

마지막에 `k = min(max(k,1), max_support)`로 잘린다.

**계수 값 샘플링** `sample_coefficient_values` (421행)

| family | 확률 | 분포 |
|---|---:|---|
| gaussian | 0.30 | N(0,1) |
| laplace | 0.20 | Laplace(0, 1/√2) |
| rademacher_loguniform | 0.20 | ± exp(U[log0.25, log4]) |
| student_t | 0.15 | t(df ~ U[3,10]) |
| two_scale_gaussian | 0.15 | 20%는 3~30배 강한 계수 |

**신호 정규화**

```python
signal = X @ beta
signal -= mean(signal)
signal_std = std(signal)
beta   /= signal_std        # Var(X beta) = 1 로 강제
signal /= signal_std
```

**노이즈 추가**

```python
target_snr = exp(U[log(0.2), log(50)])     # log-uniform [0.2, 50]
noise = base_error / sqrt(target_snr)
y = signal + noise
```

`base_error`는 7종 중 하나 (`sample_standardized_error`, 461행):
gaussian 0.25 / student_t 0.20 / laplace 0.15 / logistic 0.10 /
centered_lognormal 0.10 / contaminated_gaussian 0.10 /
heteroskedastic_gaussian 0.10

Var(Xβ)=1로 고정했으므로 **달성 가능한 R² = snr/(1+snr)** 이 되고,
k, P, N, 설계행렬 조건수와 무관하게 결정된다.

### [7] y 품질 필터 — `standardize_scm_response` (207행)

다음 중 하나라도 위반하면 `ValueError`를 던지고 태스크 전체를 재시도한다.

| 기준 | 기본값 |
|---|---|
| 유효 unique 값 개수 | ≥ max(10, 0.02 × N) |
| 최빈값 비율 | ≤ 0.30 |
| 표준화 IQR | ≥ 0.20 |
| 표준편차 | > 1e-7 |

`--max-retries 30`. 실행 로그를 보면 **거부는 거의 전부 tabicl 브랜치**에서
발생한다 (`unique count 3 < 10`, `dominant ratio 0.65`, `IQR 0.02` 등).
즉 살아남는 tabicl 태스크는 편향된 부분표본이다.

### [8] 저장 — `batch_*.npz`

| 키 | 내용 |
|---|---|
| `X` | float16, (batch, N, P) |
| `y` | float32, (batch, N) |
| `outcome_source_id` | 0=tabicl, 1=linear |
| `linear_support_size` | 정답 k (선형일 때만) |
| `linear_support_fraction`, `_of_n` | k/P, k/N |
| `linear_support_regime_id`, `_reference_id` | regime, P/N 기준 |
| `linear_coefficient_family_id`, `linear_error_family_id` | 분포 종류 |
| `linear_target_snr`, `linear_realized_snr` | SNR |
| `feature_noise_variance/sigma/family`, `feature_source`, `bnn_block_id` | X 생성 부산물 |
| `sampled_corr_audit_max`, `core_exact_max_abs_corr` | 상관 감사 |

**저장되지 않는 것: λ, β 값, support 인덱스(SHA만 남음).**

---

# 2부. V14 데이터로 어떻게 학습하는가

학습은 두 모델을 번갈아 업데이트한다.

## 2.1 Beta-model (KALBI V6) 업데이트

1. 태스크 하나의 전체 row를 taskwise Z-scale
2. **랜덤 Direct96 penalty 4개** × **penalty마다 랜덤 lambda 8개** = **32 쿼리**
3. 각 쿼리의 목적함수를 직접 평가

$$\widehat F = \frac{1}{2N}\lVert y - X\widehat\beta\rVert^2 + \lambda\sum_j\rho(|\widehat\beta_j|), \qquad F_0 = \frac{1}{2N}\lVert y\rVert^2$$

$$\mathcal L_\beta = \operatorname{mean}\left[\log\left(1 + \frac{\widehat F}{F_0}\right)\right]$$

**핵심: β 정답 라벨이 없다.** 목적함수 값 자체를 낮추는 self-supervised 방식이다.
KKT top-32는 로깅용이며 loss에 들어가지 않는다.

## 2.2 Selector (V20) 업데이트 — beta 업데이트 4회마다 1회

1. 새 태스크의 row를 랜덤 permutation 후 **절반씩 context-train / held-out** 분할
2. context-train 통계로 양쪽 표준화
3. **V20이 context-train (X, y)만 보고** λ와 penalty ρ 출력
4. **frozen KALBI V6**가 context-train에서 β 계산
5. held-out X로 예측
6. held-out y로 loss 계산

$$\mathcal L_{\text{selector}} = \operatorname{mean}\left[\log\left(1 + \frac{\operatorname{MSE}_{\text{heldout}}}{2}\right)\right]$$

문서가 명시적으로 나열한 **없는 것들**:

> lambda teacher / oracle-lambda supervision / penalty family label /
> ranking loss / KKT loss / penalty diversity regularizer / sparsity regularizer

> 즉, lambda와 penalty는 held-out prediction performance만으로 joint identification된다.

**이것이 가장 중요한 사실이다.** λ에는 정답 라벨이 없고,
"held-out MSE를 최소화하는 λ"가 학습 신호의 전부다.

---

# 3부. Lambda Selector V20은 어떻게 λ를 고르는가

## 3.1 입출력 계약

**입력:** 표준화된 context-train $(X_{\text{train}}, y_{\text{train}})$ 뿐.
KALBI의 β 출력도, candidate penalty set도 받지 않는다. held-out y는 볼 수 없다.

**출력:** 두 개.

1. scalar effective lambda ratio $r$
2. 연속 Direct96 penalty function $\rho$ 하나

파라미터 수: 21,749,345 (그중 16.84M이 cumulative projection 하나).

## 3.2 계산 흐름

```
표준화 context-train (X, y)
  ↓
[1] probe row 선택 (최대 2,048)
  ↓
[2] Full-P residual ISIS  (K=32 features/stage, S=32 stages, 최대 1,024 slot)
  ↓
[3] 4개 표현 분기 + y 직접 분기
      R        : residual trajectory      (N_probe × 32)
      B_res    : stage-local residual ridge   (32 × 32)
      B_marg   : stage-local marginal ridge   (32 × 32)
      B_cum    : cumulative joint ridge   (1024 × 32)
      h_y      : y token pooling          (512)
      metadata : stage 3좌표 × 32         (96)
  ↓
[4] 각각 512차원 code로 인코딩 → elementwise 합 → Fusion MLP
  ↓
[5] task code (512)
  ↓        ↓
lambda head   penalty head
```

## 3.3 각 단계

### [1] Probe row 선택

- $N_{\text{train}} \le 2048$: 전체 row 사용
- $N_{\text{train}} > 2048$: y를 정렬해 response quantile에 해당하는 row를 균등 선택

row attention도 fixed-N padding도 쓰지 않는다. (real 41개는 N이 작아 전체 row 사용)

### [2] Full-P residual ISIS

고정 candidate pool이 없다. **매 stage마다 아직 선택되지 않은 모든 feature를
현재 residual과 다시 비교**한다.

$$\operatorname{score}_{j,s} = \frac{|\langle x_j, r_{s-1}\rangle| / N_{\text{probe}}}{\sqrt{\langle x_j,x_j\rangle/N_{\text{probe}}}\cdot \operatorname{RMS}(r_{s-1})}$$

- 4,096열 chunk로 스캔하되 running global top-k 유지 → 전체 P를 한 번에 본 것과 동일
- 이전 stage에서 뽑힌 feature는 영구 마스크
- K=32, S=32 → 최대 1,024 slot

### [3] 세 coefficient 분기

stage s에서 뽑은 32개 feature의 Gram $G_s$, ridge $\alpha_s = \max(10^{-3}\overline{\operatorname{diag}G_s}, 10^{-5})$.

$$\beta_s^{\text{res}} = (G_s+\alpha_s I)^{-1}\frac{X_s^\top r_{s-1}}{N_{\text{probe}}} \quad\text{(residual 갱신에 사용)}$$

$$\beta_s^{\text{marg}} = (G_s+\alpha_s I)^{-1}\frac{X_s^\top y}{N_{\text{probe}}} \quad\text{(표현에만 사용)}$$

$$\beta_s^{\text{cum}} = \arg\min_b \frac{1}{2N_{\text{probe}}}\lVert y - X_{\mathcal S_s}b\rVert^2 + \frac{\alpha_{\text{cum}}}{2}\lVert b\rVert^2 \quad\text{(누적 prefix 전체 joint ridge refit)}$$

cumulative는 block-Cholesky expansion으로 증분 계산하며, 1,024 slot에 zero-padding.
**표현 분기일 뿐 screening residual을 바꾸지 않는다.**

### [4] Stage gain

$g_s = 1.005^s$ ($s = 0..31$, 마지막 ≈ 1.167)을 R, B_res, B_marg, B_cum에
사후 적용. screening 과정 자체는 바꾸지 않고 표현 스케일만 조정한다.

### [5] Residual interaction encoder

각 probe row의 residual 벡터 $R_i \in \mathbb R^{32}$에 두 MLP를 적용해
$q_i, k_i$를 만들고

$$A_R = \frac{1}{N_{\text{probe}}}\sum_i q_i k_i^\top \in \mathbb R^{32\times32}$$

를 flatten → 512로 projection. $N^2$ attention을 쓰지 않는다.

### [6] Direct y-conditioning (V19 → V20의 핵심 추가)

동일 probe index에서 표준화 $y_i$를 가져와

$$z_i = \operatorname{GELU}(W_y y_i + b_y) \in \mathbb R^{512}$$

row 방향 permutation-symmetric pooling:

$$p_y = [\operatorname{mean}_i z_i,\ \operatorname{std}_i z_i,\ \operatorname{max}_i z_i] \in \mathbb R^{1536} \rightarrow h_y \in \mathbb R^{512}$$

### [7] Fusion

$$h_0 = h_{\text{interaction}} + h_{\text{coefficient}} + h_{\text{metadata}} + h_y$$

width-512 three-linear residual MLP → $h_{\text{task}} \in \mathbb R^{512}$

### [8] Lambda head

$$z_\lambda = W_\lambda h_{\text{task}} + b_\lambda,\qquad u = \operatorname{sigmoid}(z_\lambda)$$

$$r = \exp\left[\log r_{\min} + u(\log r_{\max} - \log r_{\min})\right],\qquad r_{\min}=10^{-6},\ r_{\max}=1$$

**log scale에서 선형 보간**한다. 초기값은 $r = 0.1$.

$u$와 $r$의 대응:

| u | 0.0 | 0.25 | 0.5 | 0.757 | 0.833 | 1.0 |
|---|---|---|---|---|---|---|
| r | 1e-6 | 3.2e-5 | 1e-3 | 0.035 | 0.1 (초기값) | 1 |

### [9] λ 스케일 정의 (V20에서 변경)

V19까지: $r = \lambda\rho'(0)/\lambda_{\max}$

V20: reference slope를 $s(\rho) = \max_{0\le t\le 6}\rho'(t)$로 바꿔

$$r = \frac{\lambda\, s(\rho)}{\lambda_{\max}},\qquad \widehat\lambda = \frac{r\,\lambda_{\max}}{s(\rho)}$$

ridge-like penalty($\rho'(0)\simeq 0$)도 λ를 정의할 수 있게 하는 일반화다.
LASSO/SCAD/MCP처럼 derivative가 감소하는 penalty에서는 $s(\rho)=\rho'(0)$이라
이전 정의와 같다.

**한계 (문서가 직접 명시):** zero solution의 origin KKT scale은
$\lambda\rho'(0)/\lambda_{\max} = r\cdot\rho'(0)/s(\rho)$이므로,
$\rho'(0) \ll s(\rho)$인 penalty에서는 $r \le 1$이어도 zero solution에
도달하지 못한다.

> 이는 현재 KALBI-selected solution이 CV LASSO보다 훨씬 dense해질 수 있는
> 구조적 원인 중 하나다.

### [10] Penalty head

1. width-512 pre-norm Linear-GELU residual block 3개
2. final post-residual LayerNorm 없음
3. 512 → 96 Direct96 linear head
4. 300,000-penalty cache의 좌표별 mean/std로 proposal scale 복원
5. shared canonical Direct96 builder 적용

Direct96 decoder:

$$d_{\text{proposal}}(t) = \theta_0 + \sum_{k=0}^{94}\theta_{k+1}(t-t_k)_+$$

$$d_+(t) = \epsilon_d + \operatorname{squareplus}(d_{\text{proposal}}(t)-\epsilon_d),\qquad \rho'(t) = \frac{d_+(t)}{\int_0^6 d_+(u)du}$$

따라서 $\rho(0)=0$, $\rho(6)=1$, $\rho'>0$.

**강제되지 않는 것: convexity, concavity.**
LASSO/SCAD/MCP/ridge-like뿐 아니라 derivative가 오르내리는 형태도 표현 가능하다.

grid는 96점 비균등, 0 근처가 조밀 (약 절반이 [0, 0.3], 나머지가 [0.3, 6]).
grid 사이는 $\rho'$를 선형보간하므로 $\rho$는 구간별 정확한 이차함수이고,
V6가 scalar proximal problem을 구간별 closed form으로 정확히 푼다.

---

# 4부. 구조적으로 주목할 지점

교수님께 말씀드릴 때 근거가 되는 항목들. 앞의 두 자리가 아키텍처,
뒤의 두 자리가 생성기다.

## 4.1 계수 도메인 $|\beta_j| \le 6$

Direct96 grid가 $[0, 6]$이므로 coefficient domain이 $|\beta_j| \le 6$으로 제한된다.

표준화 공간에서 real 41개의 CV LASSO 계수를 확인한 결과:

| 데이터셋 | max\|β\| | \|β\|>6 개수 / nnz |
|---|---:|---:|
| mettlertoledo__glucose | 14.17 | 3 / 16 |
| tornado__acetate | 8.38 | 1 / 16 |
| tornado__glucose | 14.59 | 2 / 15 |
| greenhyperspectra__anth | **120.37** | 4 / 16 |
| greenhyperspectra__lai | 12.01 | 3 / 20 |

**41개 중 5개(12%)가 바운드 밖.** 이 데이터셋들은 λ와 무관하게 정답 도달이 불가능하다.

이것이 생성기와 연결되는 지점: synthetic X는 max_abs_corr가 0.47~0.76인데
real은 41/41 전부 0.9499(필터 천장)다. 잘 조건화된 설계에서는 β가 O(1)에
머물기 때문에 **$|\beta|\le 6$ 제약이 학습 중 한 번도 발동하지 않는다.**

## 4.2 zero-KKT 미도달 (문서 한계 #6, #7, #8)

- #6: $\rho'(0)/\max\rho'$가 작으면 $r=1$에서도 zero solution 불가 → dense해짐
- #7: penalty diversity/sparsity regularizer가 없어 penalty collapse 가능
- #8: Direct96는 concavity를 보장하지 않아 ridge-like 형태도 나올 수 있음

셋을 합치면 **"계수가 많이 살아있되 전부 작다"**는 증상과 정확히 일치한다.

## 4.3 생성기: 하드 가드로 생성 불가능한 영역

| 가드 | 위치 | 배제되는 real 데이터셋 |
|---|---|---|
| `--p-min 50` | parse_args + 1046행 검증 | **17 / 41** (P<50, 최소 P=3) |
| `max_support = min(p, (n-1)//3)` | sample_support_size | **6 / 41** (nnz/N > 1/3, 최대 0.85) |

`--force-p 16`을 줘도 1046행에서 거부된다. `--p-min 3`을 추가하면 통과한다
(확인 완료).

## 4.4 생성기: 난이도 이중봉

CV R² 중앙값 측정 (태스크 10개씩, N=200/P=16):

| | median cv_R² |
|---|---:|
| V14 linear 브랜치 | **0.908** |
| **Real 41** | **0.569** |
| V14 tabicl 브랜치 | **0.241** |

V14는 매우 쉽거나 거의 불가능한 두 봉우리만 만들고, real은 그 사이 골짜기에 있다.
`implied_r2 = snr/(1+snr)`가 측정 `cv_r2`와 일치하므로, 이 분포는 전적으로
`LINEAR_SNR_MIN = 0.2`, `LINEAR_SNR_MAX = 50.0` 두 상수가 결정한다.

---

# 5부. 아직 확인되지 않은 것

정직하게 구분해 둔다.

**확정된 것**
- V14가 생성 불가능한 영역이 존재하고 real 41개의 상당 부분이 거기 속한다 (코드의 하드 가드로 증명)
- real 5개가 $|\beta|\le6$ 밖에 있다 (bundle 계수 직접 측정)
- selector에 λ teacher가 없다 (문서 명시)

**확정되지 않은 것**
- 위 미스매치가 실제로 "coefficient가 작다"의 원인인지 (상관은 확인, 인과는 미확인)
- penalty가 실제로 collapse했는지 (미측정)
- KALBI 해가 실제로 CV LASSO보다 dense한지 (미측정)

**재학습 없이 추론 한 번으로 확인 가능한 것**
1. real 41개에 대한 KALBI의 nnz를 `lasso_nnz_1e8`과 비교 → dense 여부
2. 예측 β 중 $|\beta_j| \ge 5.9$ 개수 → 바운드 포화 여부
3. 41개 태스크의 penalty $\rho$, $\rho'$, $\rho'(0)$, $\max\rho'$, TV($\rho'$)를 겹쳐 그리기 → collapse 여부
4. 예측 λ ratio를 `lasso_selected_lambda_ratio`와 산점도 비교 → 오차 방향
