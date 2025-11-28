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

# 2) 36 维 RI 特征缓存：现在直接复用旧目录 feature_cache
FEATURE_CACHE_DIR = PROCESSED / "feature_cache"
FEATURE_CACHE_DIR.mkdir(exist_ok=True)

FEATURE_CACHE_REFINED_DIR = FEATURE_CACHE_DIR / "refined_only"
FEATURE_CACHE_REFINED_DIR.mkdir(exist_ok=True)

FEATURE_CACHE_CORE_DIR = FEATURE_CACHE_DIR / "core"
FEATURE_CACHE_CORE_DIR.mkdir(exist_ok=True)


# ================== 小工具函数 ==================
def _fmt(x: float) -> str:
    """浮点格式化（避免科学计数法、去掉无意义零）"""
    return f"{x:.6f}".rstrip("0").rstrip(".")


def _pair_cache_path(set_type: str, pdbid: str) -> Path:
    """几何 pair cache 路径（interim）"""
    subdir = PAIR_CACHE_DIR / set_type
    subdir.mkdir(exist_ok=True)
    return subdir / f"{pdbid}_pairs.npz"


def _feature_cache_path(
    set_type: str,
    pdbid: str,
    alpha: str,
    beta: float,
    tau: float,
    cutoff: float,
) -> Path:
    """
    (alpha,beta,tau,cutoff) 对应的 36 维特征缓存路径
    —— 直接放入 processed/feature_cache 下
    """
    tag = f"a{alpha}_b{_fmt(beta)}_t{_fmt(tau)}_c{_fmt(cutoff)}"
    subdir = FEATURE_CACHE_REFINED_DIR if set_type == "refined_only" else FEATURE_CACHE_CORE_DIR
    return subdir / f"{pdbid}_{tag}.npy"


# ================== pair cache 读取/构建 ==================
def _load_or_build_pair_cache(
    set_type: str,
    pdbid: str,
    use_cache: bool = True,
):
    """
    - 若 pair cache 存在 → 直接加载
    - 否则 → 从 structures 重算 + 缓存
    """
    cache_file = _pair_cache_path(set_type, pdbid)

    if use_cache and cache_file.exists():
        try:
            cache = np.load(cache_file)
            return cache["idx_valid"], cache["dist_valid"], cache["ratio_valid"]
        except Exception as e:
            print(f"[警告] pair cache 损坏，将删除重算: {cache_file} ({e})")
            try: cache_file.unlink()
            except OSError: pass

    # 无缓存 → 从结构重算
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
def _compute_phi_from_ratio(ratio_valid, alpha, beta, tau):
    x = (ratio_valid / float(tau)) ** float(beta)
    if alpha == "exp":
        return np.exp(-x)
    elif alpha == "lor":
        return 1.0 / (1.0 + x)
    else:
        raise ValueError(f"Unsupported alpha: {alpha}")


# ================== 计算 + 缓存 36 维特征 ==================
def load_feature(
    pdbid: str,
    set_type: str,
    alpha: str,
    beta: float,
    tau: float,
    cutoff: float,
    use_cache: bool = True,
) -> np.ndarray:

    feat_cache_file = _feature_cache_path(set_type, pdbid, alpha, beta, tau, cutoff)

    # Step 1 — 最优先：尝试加载已存在的 36维特征
    if use_cache and feat_cache_file.exists():
        try:
            RI = np.load(feat_cache_file)
            if RI.size == FEATURE_DIM:
                return RI
        except:
            try: feat_cache_file.unlink()
            except: pass

    # Step 2 — pair 几何缓存（结构相关）
    idx_valid, dist_valid, ratio_valid = _load_or_build_pair_cache(
        set_type=set_type,
        pdbid=pdbid,
        use_cache=use_cache,
    )

    # Step 3 — 计算 φ
    phi_valid = _compute_phi_from_ratio(ratio_valid, alpha, beta, tau)

    # Step 4 — cutoff 聚合
    RI = np.zeros(FEATURE_DIM, dtype=float)
    mask = dist_valid <= float(cutoff)
    if np.any(mask):
        np.add.at(RI, idx_valid[mask], phi_valid[mask])

    # Step 5 — 写回特征缓存
    if use_cache:
        np.save(str(feat_cache_file), RI)

    return RI


# ================== 批量构建 dataset ==================
def build_dataset(
    set_type: str,
    alpha: str,
    beta: float,
    tau: float,
    cutoff: float,
    use_cache: bool = True,
):
    csv_file = PROCESSED / set_type / f"{set_type}_set_list.csv"
    df = pd.read_csv(csv_file)

    n = len(df)
    X = np.zeros((n, FEATURE_DIM))
    y = df["pkd"].values
    pdbids = df["pdbid"].tolist()

    for i, pdbid in tqdm(enumerate(pdbids), total=n,
                         desc=f"{set_type}: a={alpha},b={beta},t={tau},c={cutoff}"):
        X[i] = load_feature(
            pdbid=pdbid,
            set_type=set_type,
            alpha=alpha,
            beta=beta,
            tau=tau,
            cutoff=cutoff,
            use_cache=use_cache,
        )

    return X, y, pdbids
