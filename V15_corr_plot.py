#%%
import os

import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from tqdm import tqdm

# ---------------------------------------------------------
# 1. npz 파일 1개를 처리하는 함수
# ---------------------------------------------------------
def process_npz_file(filepath, max_features_shown=40):
    try:
        data = np.load(filepath, allow_pickle=False)
        X = data['X'][0]  # (1, N, P) -> (N, P)
        
        # 메타데이터 키 추출 (없을 경우 예외처리)
        sampled_max = float(data['sampled_corr_audit_max']) if 'sampled_corr_audit_max' in data else None
        exact_max = float(data['core_exact_max_abs_corr']) if 'core_exact_max_abs_corr' in data else None
        
        n, p = X.shape
        if p < 2 or n < 2:
            return None
            
        # 표준편차가 0인 피처(상수 열) 제거 예외처리
        std = np.std(X, axis=0)
        valid_cols = std > 1e-8
        X_valid = X[:, valid_cols]
        p_valid = X_valid.shape[1]
        
        if p_valid < 2:
            return None
            
        # 1. 상관계수 절댓값 계산
        corr = np.abs(np.corrcoef(X_valid, rowvar=False))
        np.nan_to_num(corr, copy=False, nan=0.0)
        
        # 상삼각행렬에서 통계량 추출
        upper_tri = corr[np.triu_indices_from(corr, k=1)]
        if len(upper_tri) == 0:
            return None
            
        q50, q90, q99 = np.percentile(upper_tri, [50, 90, 99])
        max_val = exact_max if exact_max is not None else np.max(upper_tri)
        spread = max_val - np.min(upper_tri)
        
        # 2. 계층적 군집화 (피처 정렬)
        dist = np.clip(1 - corr, 0, 2)
        np.fill_diagonal(dist, 0)
        
        link = linkage(squareform(dist), method='average')
        dendro = dendrogram(link, no_plot=True)
        idx = dendro['leaves']
        
        corr_sorted = corr[np.ix_(idx, idx)]
        
        # 피처 수가 많을 경우 상위 max_features_shown개만 표시
        shown_p = min(max_features_shown, p_valid)
        corr_display = corr_sorted[:shown_p, :shown_p]
        
        filename = os.path.basename(filepath).replace('.npz', '')
        
        return {
            'id': filename,
            'n': n,
            'p': p,
            'shown': shown_p,
            'q50': q50, 'q90': q90, 'q99': q99,
            'max': max_val,
            'spread': spread,
            'corr': corr_display
        }
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

# ---------------------------------------------------------
# 2. 508개 .npz 순회 및 Top 12 추출
# ---------------------------------------------------------
data_dir = "/workspace/rolling_v15_8gpu/data/tabicl_v2"
files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

results = []
print(f"Processing {len(files)} files...")
for f in tqdm(files):
    res = process_npz_file(f)
    if res is not None:
        results.append(res)

# Max 상관계수 기준 내림차순 정렬 후 상위 12개 선택
top12_tasks = sorted(results, key=lambda x: x['max'], reverse=True)[:12]

# ---------------------------------------------------------
# 3. 3x4 Heatmap 그리드 시각화 (원본 PDF 스타일)
# ---------------------------------------------------------
fig, axes = plt.subplots(3, 4, figsize=(16, 11))
fig.suptitle(
    "V15_tabicl_v2: max-correlation stress cases\n"
    "Tasks are selected by largest max |corr(Xj, Xk)|; shown features prioritize strongest correlated pairs.",
    fontsize=12, fontweight='bold', y=0.96
)

cmap = sns.color_palette("magma", as_cmap=True)

for idx, task in enumerate(top12_tasks):
    ax = axes[idx // 4, idx % 4]
    
    im = ax.imshow(task['corr'], cmap=cmap, vmin=0.0, vmax=1.0)
    ax.axis('off')
    
    line1 = f"{task['id']}\nn={task['n']}, p={task['p']}, shown={task['shown']}"
    line2 = f"q50={task['q50']:.2f} | q90={task['q90']:.2f} | q99={task['q99']:.2f} | max={task['max']:.2f} | spread={task['spread']:.2f}"
    
    ax.set_title(f"{line1}\n{line2}", fontsize=7.5, pad=4)

# 컬러바 배치
fig.subplots_adjust(right=0.88, hspace=0.38, wspace=0.15)
cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('|corr(Xj, Xk)|', fontsize=10)

# 저장 및 출력
output_pdf = "V15_tabicl_v2_max_corr_heatmaps.pdf"
plt.savefig(output_pdf, bbox_inches='tight', dpi=300)
print(f"\nSaved heatmap figure to {output_pdf}")
plt.show()
# %%
