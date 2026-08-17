"""V14 학습 데이터의 max|beta| 가 1 을 넘지 못함 — 실측 + 독립재현 + 수식"""
import json, math, os
import numpy as np

print("=" * 72)
print("(A) 실측 — V14 가 생성 시점에 매니페스트에 기록한 값")
print("=" * 72)
rows = []
for d in ["out_v14_test", "out_v14_highp", "out_v14_bnn", "out_v14_natural"]:
    f = f"{d}/manifest.json"
    if not os.path.exists(f):
        print(f"  [없음] {d}"); continue
    for e in json.load(open(f))["files"]:
        for t in e["task_metadata"]:
            if t.get("outcome_source") != "linear":
                continue
            rows.append((d, t["n"], t["p"], t["linear_support_size"],
                         t["linear_beta_l2_norm"], t["linear_beta_max_abs"]))

print(f"\n{'폴더':<18}{'N':>7}{'P':>8}{'k':>7}{'||beta||':>11}{'max|beta_j|':>13}")
print("-" * 72)
for r in sorted(rows, key=lambda z: -z[5]):
    print(f"{r[0]:<18}{r[1]:>7}{r[2]:>8}{r[3]:>7}{r[4]:>11.4f}{r[5]:>13.4f}")

a = np.array([[r[4], r[5]] for r in rows])
print("-" * 72)
print(f"  linear 태스크 {len(rows)}개")
print(f"  ||beta||_2    중앙 {np.median(a[:,0]):.4f}   최소 {a[:,0].min():.4f}   최대 {a[:,0].max():.4f}")
print(f"  max|beta_j|   중앙 {np.median(a[:,1]):.4f}   최대 {a[:,1].max():.4f}")
print()
print(f"  max|beta_j| > 1 :  {int((a[:,1] > 1.0).sum())} / {len(rows)}")
print(f"  max|beta_j| > 6 :  {int((a[:,1] > 6.0).sum())} / {len(rows)}   <-- Direct96 정의역 상한")
print(f"  Direct96 정의역 [0,6] 중 실제 사용 구간: 0 ~ {a[:,1].max():.2f}  ({a[:,1].max()/6*100:.0f}%)")

k1 = [r for r in rows if r[3] == 1]
if k1:
    print(f"\n  k=1 인 태스크 {len(k1)}개 — 이론값 정확히 1.000000")
    for r in k1:
        print(f"     N={r[1]:6d} P={r[2]:7d}   ||beta||={r[4]:.6f}   max|beta|={r[5]:.6f}")

print()
print("=" * 72)
print("(B) 독립 재현 — V14 의 beta 생성 로직만 떼어내 재계산")
print("=" * 72)
rng = np.random.default_rng(0)
REG = ("ultra_sparse","sparse","moderate","large","very_dense"); RP = np.array([.15,.20,.25,.25,.15])
CF  = ("gaussian","laplace","rademacher_loguniform","student_t","two_scale_gaussian"); CP = np.array([.30,.20,.20,.15,.15])

def support_size(n, p, r):
    ms = min(p, (n - 1) // 3)
    g = REG[r.choice(5, p=RP)]
    if g == "ultra_sparse": return min(max(int(r.integers(1, min(4, ms) + 1)), 1), ms)
    if g == "sparse":
        lo, hi = min(5, ms), min(64, ms)
        k = hi if lo >= hi else int(round(math.exp(r.uniform(math.log(lo), math.log(hi)))))
    elif g == "moderate":   k = int(round(10 ** r.uniform(-4, -2) * p))
    elif g == "large":      k = int(round(10 ** r.uniform(-2, math.log10(.2)) * p))
    else:                   k = int(round(r.uniform(.2, 1.) * p))
    return min(max(k, 1), ms)

def coefs(k, r):
    f = CF[r.choice(5, p=CP)]
    if f == "gaussian":              return r.standard_normal(k)
    if f == "laplace":               return r.laplace(0, 1/math.sqrt(2), k)
    if f == "rademacher_loguniform": return r.choice([-1.,1.], k) * np.exp(r.uniform(math.log(.25), math.log(4.), k))
    if f == "student_t":             return r.standard_t(r.uniform(3, 10), k)
    s = r.random(k) < .20
    return r.standard_normal(k) * np.where(s, math.exp(r.uniform(math.log(3), math.log(30))), 1.)

print(f"\n{'(N, P)':>18}{'시행':>7}{'중앙 k':>8}{'중앙 max|b|':>13}{'최대 max|b|':>13}{'>6 개수':>9}")
print("-" * 72)
tot, over = 0, 0
for n, p in [(200,16), (131,620), (800,3000), (908,1500), (4499,1500)]:
    mx, ks = [], []
    for _ in range(300):
        X = rng.standard_normal((n, p)); X = (X - X.mean(0)) / X.std(0)
        k = support_size(n, p, rng)
        b = np.zeros(p); b[rng.choice(p, k, replace=False)] = coefs(k, rng)
        s = X @ b; s -= s.mean(); sd = s.std()
        if sd < 1e-10: continue
        b /= sd
        mx.append(np.abs(b).max()); ks.append(k)
    mx = np.array(mx); tot += len(mx); over += int((mx > 6).sum())
    print(f"{f'({n}, {p})':>18}{len(mx):>7}{np.median(ks):>8.0f}{np.median(mx):>13.4f}{mx.max():>13.4f}{int((mx>6).sum()):>9}")
print("-" * 72)
print(f"  재현 총 {tot}회 중 max|beta| > 6 :  {over} 개")

print()
print("=" * 72)
print("(C) 수식")
print("=" * 72)
print("""
  X 표준화  ->  Gram G 의 대각이 1
            ->  어떤 k개 부분집합이든  tr(G_SS) = k,  고유값 평균 = 1

  support 를 랜덤으로 뽑고 계수가 등방이면
        E[ beta^T G beta ] = (||beta||^2 / k) * tr(G_SS) = ||beta||^2
        =>  레일리 몫  R(beta) = Var(X beta) / ||beta||^2  ->  1

  ||beta||_2 = sqrt( Var(X beta) / R(beta) )
             = sqrt(   <= 1      /   ~1    )   =  1 이하

  * 설계행렬의 조건수와 무관. 랜덤 벡터가 G 의 저고유값 방향에
    정렬될 확률은 k 에 대해 지수적으로 작다.
  * k=1 이면 Var(X beta) = beta_1^2 * G_11 = beta_1^2 = 1  ->  |beta_1| = 1 정확히
""")
