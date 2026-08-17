import numpy as np, glob

def sub_min_eig(X, S):
    G = X[:, S].T @ X[:, S] / X.shape[0]
    return float(np.linalg.eigvalsh(G)[0])

for out in ["out_v14_test", "out_v14_highp", "out_v14_bnn"]:
    files = sorted(glob.glob(f"{out}/batch_*.npz"))[:12]
    if not files:
        print(f"{out}: 파일 없음"); continue
    rnd, grd, conds = [], [], []
    N = P = k = 0
    for f in files:
        X = np.asarray(np.load(f)["X"][0], dtype=np.float64)
        sd = X.std(0); sd[sd < 1e-12] = 1.0
        X = (X - X.mean(0)) / sd
        N, P = X.shape
        k = min(16, P, (N - 1) // 3)
        r = np.random.default_rng(0)
        conds.append(float(np.linalg.cond(X)))
        rnd += [sub_min_eig(X, r.choice(P, k, replace=False)) for _ in range(30)]
        C = np.abs(np.corrcoef(X.T)); np.fill_diagonal(C, 0.0); C = np.nan_to_num(C)
        S = [int(np.unravel_index(C.argmax(), C.shape)[0])]
        while len(S) < k:
            score = C[S].mean(0); score[S] = -np.inf
            S.append(int(np.argmax(score)))
        grd.append(sub_min_eig(X, np.array(S)))
        print(f"  {f} 처리 완료", flush=True)
    print(f"\n=== {out}   X: {N} x {P},  k={k} ===")
    print(f"  X 전체 조건수         중앙값 {np.median(conds):.1f}")
    print(f"  랜덤 support  min-eig 중앙값 {np.median(rnd):.3e}   최소 {np.min(rnd):.3e}")
    print(f"  탐욕 support  min-eig 중앙값 {np.median(grd):.3e}   최소 {np.min(grd):.3e}")
    print(f"  real 요구 수준        2e-05\n")
