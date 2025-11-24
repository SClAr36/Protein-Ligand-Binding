import numpy as np
from scipy.spatial.distance import cdist

# 按教材/原文定义的元素集合
PRO_ELEMENTS = ['C', 'N', 'O', 'S']
LIG_ELEMENTS = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']

N_PRO = len(PRO_ELEMENTS)
N_LIG = len(LIG_ELEMENTS)
FEATURE_DIM = N_PRO * N_LIG  # 4 * 9 = 36

# 简单 vdW 半径表（Å），你以后可以按需要细调
VDW_RADII = {
    'H': 1.20,
    'C': 1.70,
    'N': 1.55,
    'O': 1.52,
    'F': 1.47,
    'P': 1.80,
    'S': 1.80,
    'Cl': 1.75,
    'Br': 1.85,
    'I': 1.98,
}

def get_pair_index(pro_elem: str, lig_elem: str) -> int | None:
    """把 (蛋白元素, 配体元素) 映射到 [0,35] 的特征索引"""
    try:
        i = PRO_ELEMENTS.index(pro_elem)
        j = LIG_ELEMENTS.index(lig_elem)
        return i * N_LIG + j
    except ValueError:
        return None  # 如果不在 X/Y 集合里就跳过

def compute_RI_score_general(
    pro_coords: np.ndarray,
    pro_elems: np.ndarray,
    lig_coords: np.ndarray,
    lig_elems: np.ndarray,
    alpha: str,   # "exp" or "lor"
    beta: float,  # kappa (exp) or nu (lor)
    tau: float,
    cutoff: float,
) -> np.ndarray:
    """
    计算单个 complex 的 36 维 RI-score 特征。
    """
    RI = np.zeros(FEATURE_DIM, dtype=float)

    # 先算距离矩阵 [N_pro x N_lig]
    dist = cdist(pro_coords, lig_coords)

    # cutoff 只在物理距离上截断
    mask = dist <= cutoff

    # 遍历所有原子对
    for i, p_elem in enumerate(pro_elems):
        # 蛋白这个元素不在 X 集合里就跳过
        if p_elem not in PRO_ELEMENTS:
            continue
        r_i = VDW_RADII.get(p_elem)
        if r_i is None:
            continue

        for j, l_elem in enumerate(lig_elems):
            if not mask[i, j]:
                continue
            if l_elem not in LIG_ELEMENTS:
                continue
            r_j = VDW_RADII.get(l_elem)
            if r_j is None:
                continue

            idx = get_pair_index(p_elem, l_elem)
            if idx is None:
                continue

            d = dist[i, j]
            eta_ij = tau * (r_i + r_j)  # 关键修正：η_ij = τ (r_i + r_j)
            x = (d / eta_ij) ** beta

            if alpha == "exp":
                phi = np.exp(-x)
            elif alpha == "lor":
                phi = 1.0 / (1.0 + x)
            else:
                raise ValueError("alpha must be 'exp' or 'lor'.")

            RI[idx] += phi

    return RI
