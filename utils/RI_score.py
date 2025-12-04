import numpy as np
from scipy.spatial.distance import cdist

# 按教材/原文定义的元素集合
PRO_ELEMENTS = ['C', 'N', 'O', 'S']
LIG_ELEMENTS = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']

N_PRO = len(PRO_ELEMENTS)
N_LIG = len(LIG_ELEMENTS)
FEATURE_DIM = N_PRO * N_LIG  # 4 * 9 = 36

# van der Waals radii (单位：Å)
VDW_RADII = {
    'C': 1.70,
    'N': 1.55, # 1.63 for mbondi2
    'O': 1.52,
    'F': 1.47,
    'P': 1.80,
    'S': 1.80, # 1.85 for mbondi2
    'Cl': 1.75,
    'Br': 1.85,
    'I': 1.98,
}

# 为了加速，元素 -> 索引 做成 dict
PRO_INDEX = {e: i for i, e in enumerate(PRO_ELEMENTS)}
LIG_INDEX = {e: j for j, e in enumerate(LIG_ELEMENTS)}

# 构建 PRO x LIG 的 r_sum lookup: 长度 36
R_SUM_TABLE = np.zeros(FEATURE_DIM, dtype=np.float64)  # FEATURE_DIM = 36

for i, pe in enumerate(PRO_ELEMENTS):
    r_i = VDW_RADII.get(pe, np.nan)
    for j, le in enumerate(LIG_ELEMENTS):
        r_j = VDW_RADII.get(le, np.nan)
        idx = i * N_LIG + j
        R_SUM_TABLE[idx] = r_i + r_j


def precompute_phi_pairs(
    pro_coords: np.ndarray,
    pro_elems: np.ndarray,
    lig_coords: np.ndarray,
    lig_elems: np.ndarray,
    alpha: str,
    beta: float,
    tau: float,
    max_cutoff: float,
):
    """
    针对单个 complex 和一组 (alpha, beta, tau) 参数，预计算在
    d_ij <= max_cutoff 范围内所有原子对的 φ(d_ij) 以及对应的 36 维索引。

    返回:
        idx_valid : (K,) int16，每个 pair 属于 [0, 35] 的特征维度
        dist_valid: (K,) float32，各 pair 的距离
        phi_valid : (K,) float32，各 pair 的 kernel 值
    """
    max_cutoff = float(max_cutoff)

    # 距离矩阵 [N_pro, N_lig]
    dist = cdist(pro_coords, lig_coords)

    # 元素 -> 索引
    pro_idx = np.array([PRO_INDEX.get(e, -1) for e in pro_elems], dtype=np.int16)
    lig_idx = np.array([LIG_INDEX.get(e, -1) for e in lig_elems], dtype=np.int16)

    pi = pro_idx[:, None]   # [N_pro, 1]
    lj = lig_idx[None, :]   # [1, N_lig]

    # 只保留元素在集合里的 pair
    elem_valid = (pi >= 0) & (lj >= 0)

    # cutoff 限制
    cutoff_valid = dist <= max_cutoff

    # vdW 半径
    r_i = np.array([VDW_RADII.get(e, np.nan) for e in pro_elems])  # [N_pro]
    r_j = np.array([VDW_RADII.get(e, np.nan) for e in lig_elems])  # [N_lig]
    eta = tau * (r_i[:, None] + r_j[None, :])  # [N_pro, N_lig]

    eta_valid = np.isfinite(eta) & (eta > 0.0)

    valid = elem_valid & cutoff_valid & eta_valid

    if not np.any(valid):
        return (
            np.zeros((0,), dtype=np.int16),
            np.zeros((0,)),
            np.zeros((0,)),
        )

    # 只在 valid 的位置上计算 (d/eta)^beta
    x = np.zeros_like(dist)
    x[valid] = (dist[valid] / eta[valid]) ** beta

    if alpha == "exp":
        phi = np.zeros_like(dist)
        phi[valid] = np.exp(-x[valid])
    elif alpha == "lor":
        phi = np.zeros_like(dist)
        phi[valid] = 1.0 / (1.0 + x[valid])
    else:
        raise ValueError("alpha must be 'exp' or 'lor'.")

    # 元素或 cutoff 不合法的位置全部置零（虽然已经不在 valid 里）
    phi[~valid] = 0.0

    # 计算每个 pair 对应的 36 维 index = i * N_LIG + j
    pair_idx = pi * N_LIG + lj           # [N_pro, N_lig]
    pair_idx[~valid] = -1               # 无效 pair 标成 -1

    # 拉平，只保留有效 pair
    flat_valid = valid.ravel()
    flat_idx = pair_idx.ravel()[flat_valid]
    flat_dist = dist.ravel()[flat_valid]
    flat_phi = phi.ravel()[flat_valid]

    idx_valid = flat_idx.astype(np.int16)
    dist_valid = flat_dist
    phi_valid = flat_phi

    return idx_valid, dist_valid, phi_valid

def precompute_dist_ratio_pairs(
    pro_coords: np.ndarray,
    pro_elems: np.ndarray,
    lig_coords: np.ndarray,
    lig_elems: np.ndarray,
):
    """
    计算单个 complex 中有效的 protein-ligand 原子对：
        idx_valid   : (K,) int16
        dist_valid  : (K,) float64
        ratio_valid : (K,) float64 = dist / r_sum
    
    r_sum 根据 idx_valid 直接由 R_SUM_TABLE 得到。
    """

    # 距离矩阵 [N_pro, N_lig]
    dist = cdist(pro_coords, lig_coords)  # float64

    # protein / ligand 元素 -> 索引（-1 表示该元素不在 PRO / LIG 集）
    pro_idx = np.array([PRO_INDEX.get(e, -1) for e in pro_elems], dtype=np.int16)
    lig_idx = np.array([LIG_INDEX.get(e, -1) for e in lig_elems], dtype=np.int16)

    pi = pro_idx[:, None]      # [Np, 1]
    lj = lig_idx[None, :]      # [1, Nl]

    elem_valid = (pi >= 0) & (lj >= 0)

    # 计算 pair index = i*N_LIG + j
    pair_idx = pi * N_LIG + lj       # [Np, Nl], int16
    pair_idx[~elem_valid] = -1       # 无效元素设 -1

    # 拉平
    flat_idx = pair_idx.ravel()
    flat_dist = dist.ravel()

    # 只取有效 idx
    valid_mask = flat_idx >= 0

    idx_valid = flat_idx[valid_mask].astype(np.int16)
    dist_valid = flat_dist[valid_mask]

    # 查 lookup 表获得每个 pair 的 r_sum
    r_sum = R_SUM_TABLE[idx_valid]  # (K,)

    # ratio = dist / r_sum
    ratio_valid = dist_valid / r_sum

    return idx_valid, dist_valid, ratio_valid


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
    保留原有接口，只是内部改成向量化实现。

    注意：这里的 max_cutoff 就等于 cutoff，因此不会产生“预计算跨多个 cutoff 重用”的效果；
    真正跨 cutoff 重用是在 data_loader 里通过更大的 max_cutoff_global 来做的。
    """
    idx_valid, dist_valid, phi_valid = precompute_phi_pairs(
        pro_coords=pro_coords,
        pro_elems=pro_elems,
        lig_coords=lig_coords,
        lig_elems=lig_elems,
        alpha=alpha,
        beta=beta,
        tau=tau,
        max_cutoff=cutoff,
    )

    RI = np.zeros(FEATURE_DIM, dtype=float)

    if idx_valid.size == 0:
        return RI

    # 这里 precompute 已经用 cutoff 截过一次了，因此不需要再按 cutoff 做 mask
    np.add.at(RI, idx_valid, phi_valid)

    return RI
