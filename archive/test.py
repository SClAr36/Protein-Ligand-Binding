# import numpy as np

# df = np.load("data/interim/ri_pair_param_cache/3oy8_aexp_b2.5_t1_cmax50.npz")

# phi = df["phi_valid"]
# dist = df["dist_valid"]
# idx =df["idx_valid"]

# print(phi[0])
# # 同时查看数组形状和唯一值数量
# print("数组基本信息:")
# print(f"phi - 形状: {phi.shape}, 唯一值数量: {np.unique(phi).size}")
# print(f"dist - 形状: {dist.shape}, 唯一值数量: {np.unique(dist).size}")
# print(f"idx - 形状: {idx.shape}, 唯一值数量: {np.unique(idx).size}")

from utils.RI_score import precompute_phi_pairs
import numpy as np
from pathlib import Path

# 加载结构
root = Path("data")
data = np.load(root / "processed" / "refined_only" / "structures" / "1g48.npz")
pro_coords = data["pro_coords"]
pro_elems  = data["pro_elems"]
lig_coords = data["lig_coords"]
lig_elems  = data["lig_elems"]


idx, dist, phi = precompute_phi_pairs(
    pro_coords=pro_coords,
    pro_elems=pro_elems,
    lig_coords=lig_coords,
    lig_elems=lig_elems,
    alpha="exp",
    beta=2.5,
    tau=1.0,
    max_cutoff=50.0,
)
print(f"[exp] idx={idx.shape}, dist={dist.shape}, phi={phi.shape}, \n"
        f"unique(phi)={np.unique(phi).size}, min(phi)={phi.min():.3e}, max(phi)={phi.max():.3e}\n"
        f"unique(dist)={np.unique(dist).size}, min(dist)={dist.min():.3f}, max(dist)={dist.max():.3f}\n")
#print(f"  前10个 phi 值: {phi[:10]}")

lidx, ldist, lphi = precompute_phi_pairs(
    pro_coords=pro_coords,
    pro_elems=pro_elems,
    lig_coords=lig_coords,
    lig_elems=lig_elems,
    alpha="lor",
    beta=5,
    tau=1.0,
    max_cutoff=50.0,
)
print(f"[lor] idx={lidx.shape}, dist={ldist.shape}, phi={lphi.shape}, \n"
        f"unique(phi)={np.unique(lphi).size}, min(phi)={lphi.min():.3e}, max(phi)={lphi.max():.3e}\n"
        f"unique(dist)={np.unique(ldist).size}, min(dist)={ldist.min():.3f}, max(dist)={ldist.max():.3f}\n")