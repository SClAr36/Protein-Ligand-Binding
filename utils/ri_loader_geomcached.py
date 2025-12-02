import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# 从 RI_score 导入你已有的几何函数
from utils.RI_score import compute_dist_ratio_pairs, FEATURE_DIM

# ================== 路径设置 ==================
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"

PROCESSED = DATA_ROOT / "processed"
INTERIM = DATA_ROOT / "interim"
INTERIM.mkdir(exist_ok=True)

# 1) pair 级缓存：结构相关（geometry）
PAIR_CACHE_DIR = INTERIM / "ri_pair_geom_cache"
PAIR_CACHE_DIR.mkdir(exist_ok=True)

# 2) 最终特征改为：每个参数组合 → 一个 N×36 大矩阵
FEATURE_CACHE_DIR = PROCESSED / "feature_cache"
FEATURE_CACHE_DIR.mkdir(exist_ok=True)

FEATURE_CACHE_REFINED_DIR = FEATURE_CACHE_DIR / "refined_only"
FEATURE_CACHE_REFINED_DIR.mkdir(exist_ok=True)

FEATURE_CACHE_CORE_DIR = FEATURE_CACHE_DIR / "core"
FEATURE_CACHE_CORE_DIR.mkdir(exist_ok=True)


# ================== 小工具函数 ==================
def _fmt(x: float) -> str:
    """浮点格式化"""
    return f"{x:.6f}".rstrip("0").rstrip(".")


def _pair_cache_path(set_type: str, pdbid: str) -> Path:
    """几何 pair cache 路径"""
    subdir = PAIR_CACHE_DIR / set_type
    subdir.mkdir(exist_ok=True)
    return subdir / f"{pdbid}_pairs.npz"


# 新的数据文件结构：
# feature_cache/refined_only/aexp_b2.5_t1.0_c12.0.npy
def _bigmatrix_path(set_type: str, alpha: str, beta: float, tau: float, cutoff: float) -> Path:
    tag = f"a{alpha}_b{_fmt(beta)}_t{_fmt(tau)}_c{_fmt(cutoff)}.npy"
    subdir = FEATURE_CACHE_REFINED_DIR if set_type == "refined_only" else FEATURE_CACHE_CORE_DIR
    return subdir / tag


# ================== pair cache 读取/构建 ==================
def _load_or_build_pair_cache(set_type: str, pdbid: str, use_cache: bool = True):
    """
    - 若 pair cache 存在 → 直接加载
    - 否则 → 从 structures 重算 + 缓存
    """
    cache_file = _pair_cache_path(set_type, pdbid)

    if use_cache and cache_file.exists():
        try:
            cache = np.load(cache_file)
            return cache["idx_valid"], cache["dist_valid"], cache["ratio_valid"]
        except Exception:
            try: cache_file.unlink()
            except: pass

    # 重算
    npz_path = PROCESSED / set_type / "structures" / f"{pdbid}.npz"
    data = np.load(npz_path)

    idx_valid, dist_valid, ratio_valid = compute_dist_ratio_pairs(
        pro_coords=data["pro_coords"],
        pro_elems=data["pro_elems"],
        lig_coords=data["lig_coords"],
        lig_elems=data["lig_elems"],
    )

    if use_cache:
        np.savez_compressed(
            cache_file,
            idx_valid=idx_valid,
            dist_valid=dist_valid,
            ratio_valid=ratio_valid,
        )

    return idx_valid, dist_valid, ratio_valid


# ================== φ(alpha,beta,tau) ==================
def _compute_phi_from_ratio(ratio, alpha, beta, tau):
    """对给定 ratio 数组计算 phi（exp 或 lor）"""
    x = (ratio / float(tau)) ** float(beta)
    if alpha == "exp":
        return np.exp(-x)
    elif alpha == "lor":
        return 1.0 / (1.0 + x)
    else:
        raise ValueError(f"Unsupported alpha: {alpha}")


# ================== 单个 pdbid 的 36 维特征（用于构建大矩阵） ==================
def load_feature(
    pdbid: str,
    set_type: str,
    alpha: str,
    beta: float,
    tau: float,
    cutoff: float,
    use_cache: bool = True,
):
    """
    仅用于构建大矩阵，不再单独保存每个 pdbid 的 .npy。
    已优化：先 cutoff 筛选，再计算 phi。
    """

    idx_valid, dist_valid, ratio_valid = _load_or_build_pair_cache(
        set_type=set_type,
        pdbid=pdbid,
        use_cache=use_cache,
    )

    # Step 1 — cutoff mask
    mask = dist_valid <= float(cutoff)
    if not np.any(mask):
        return np.zeros(FEATURE_DIM, dtype=float)

    idx_m = idx_valid[mask]
    ratio_m = ratio_valid[mask]

    # Step 2 — compute phi only for valid pairs
    phi_m = _compute_phi_from_ratio(ratio_m, alpha, beta, tau)

    # Step 3 — aggregate into 36-dim RI
    RI = np.zeros(FEATURE_DIM, dtype=float)
    np.add.at(RI, idx_m, phi_m)

    return RI


# ================== 构建 N×36 大矩阵（新功能） ==================
def build_bigmatrix(
    set_type: str,
    alpha: str,
    beta: float,
    tau: float,
    cutoff: float,
    use_cache: bool = True,
):
    """
    将所有 pdbid 的 36 维特征合并成一个 N×36 大矩阵文件。
    同时保存 pdbid 顺序。
    """
    # 输出文件
    out_path = _bigmatrix_path(set_type, alpha, beta, tau, cutoff)
    out_dir = out_path.parent
    out_dir.mkdir(exist_ok=True)

    # csv 列表（顺序固定）
    csv_path = PROCESSED / set_type / f"{set_type}_set_list.csv"
    df = pd.read_csv(csv_path)
    pdbids = df["pdbid"].tolist()
    y = df["pkd"].values

    # 保存统一的 pdbids / pkd
    np.save(out_dir / f"{set_type}_pdbids.npy", np.array(pdbids))
    np.save(out_dir / f"{set_type}_pkd.npy", y)

    N = len(pdbids)
    X = np.zeros((N, FEATURE_DIM), dtype=float)

    for i, pdbid in tqdm(enumerate(pdbids), total=N,
                         desc=f"Building bigmatrix for {set_type}: a={alpha}, b={beta}, t={tau}, c={cutoff}"):
        X[i] = load_feature(
            pdbid=pdbid,
            set_type=set_type,
            alpha=alpha,
            beta=beta,
            tau=tau,
            cutoff=cutoff,
            use_cache=use_cache,
        )

    np.save(out_path, X)
    print(f"[完成] 保存大矩阵: {out_path} shape={X.shape}")

    return out_path


# ================== 新的 dataset 加载方式（最快） ==================
def load_dataset_bigmatrix(set_type: str, alpha: str, beta: float, tau: float, cutoff: float):
    """
    使用新结构：一次性读取整个 N×36 特征矩阵 + pkd + pdbids
    """
    base_dir = FEATURE_CACHE_REFINED_DIR if set_type == "refined_only" else FEATURE_CACHE_CORE_DIR
    tag = f"a{alpha}_b{_fmt(beta)}_t{_fmt(tau)}_c{_fmt(cutoff)}.npy"
    X = np.load(base_dir / tag)
    y = np.load(base_dir / f"{set_type}_pkd.npy")
    pdbids = np.load(base_dir / f"{set_type}_pdbids.npy").tolist()
    return X, y, pdbids
