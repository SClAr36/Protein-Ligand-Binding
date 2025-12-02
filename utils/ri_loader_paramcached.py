import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

from utils.RI_score import (
    precompute_phi_pairs,
    FEATURE_DIM,
)

# ================== 路径 ==================
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"

PROCESSED = DATA_ROOT / "processed"
INTERIM = DATA_ROOT / "interim"
INTERIM.mkdir(exist_ok=True)

# 只保留 pair cache，不再存单 pdbid feature 小文件
PAIR_CACHE_DIR = INTERIM / "ri_pair_param_cache"
PAIR_CACHE_DIR.mkdir(exist_ok=True)

# 新的大矩阵存储
FEATURE_CACHE_DIR = PROCESSED / "feature_cache"
FEATURE_CACHE_DIR.mkdir(exist_ok=True)


# ================== 辅助函数 ==================
def _fmt(x: float) -> str:
    """
    格式化参数，避免科学计数法，同时去掉无意义的尾随 0 和点号。
    """
    return f"{x:.6f}".rstrip("0").rstrip(".")


def _pair_cache_path(pdbid: str, alpha: str, beta: float, tau: float,
                     max_cutoff_global: float) -> Path:
    """
    生成某个复合物在指定参数下的 φ 预计算缓存路径。
    文件名显式带上 max_cutoff_global，避免未来更大 cutoff 时产生混淆。
    """
    tag = (
        f"a{alpha}_b{_fmt(beta)}_t{_fmt(tau)}"
        f"_cmax{_fmt(max_cutoff_global)}"
    )
    return PAIR_CACHE_DIR / f"{pdbid}_{tag}.npz"


def _bigmatrix_path(set_type: str, alpha: str, beta: float, tau: float, cutoff: float):
    """每个参数一个大矩阵文件"""
    tag = f"a{alpha}_b{_fmt(beta)}_t{_fmt(tau)}_c{_fmt(cutoff)}.npy"
    return FEATURE_CACHE_DIR / set_type / tag


# ================== 结构加载 ==================
def _load_structure_npz(set_type: str, pdbid: str):
    npz_path = PROCESSED / set_type / "structures" / f"{pdbid}.npz"
    return np.load(npz_path)


# ================== 不变：pair cache 逻辑 ==================
def _load_or_build_pair_cache(
    pdbid: str,
    set_type: str,
    alpha: str,
    beta: float,
    tau: float,
    max_cutoff_global: float,
    use_cache: bool = True,
):
    """
    对 (pdbid, alpha, beta, tau, max_cutoff_global)：
    - 若已经存在 pair 缓存，则直接读取；
    - 否则从结构 npz 中加载，并调用 precompute_phi_pairs 计算，再缓存。

    返回:
        idx_valid, dist_valid, phi_valid
    """
    pair_file = _pair_cache_path(pdbid, alpha, beta, tau, max_cutoff_global)

    if use_cache and pair_file.exists():
        cache = np.load(pair_file)
        return cache["idx_valid"], cache["dist_valid"], cache["phi_valid"]

    # 重算
    data = _load_structure_npz(set_type, pdbid)
    idx_valid, dist_valid, phi_valid = precompute_phi_pairs(
        pro_coords=data["pro_coords"],
        pro_elems=data["pro_elems"],
        lig_coords=data["lig_coords"],
        lig_elems=data["lig_elems"],
        alpha=alpha,
        beta=beta,
        tau=tau,
        max_cutoff=max_cutoff_global,
    )

    if use_cache:
        np.savez_compressed(
            pair_file,
            idx_valid=idx_valid,
            dist_valid=dist_valid,
            phi_valid=phi_valid,
        )

    return idx_valid, dist_valid, phi_valid


# ================== load_feature：计算逻辑完全不改，只去掉小文件 read/save ==================
def load_feature(
    pdbid: str,
    set_type: str,
    alpha: str,
    beta: float,
    tau: float,
    cutoff: float,
    use_cache: bool = True,
    max_cutoff_global: float | None = None,
) -> np.ndarray:
    """
    原本 load_feature 的运算逻辑保留：
    - 读取 pair cache
    - cutoff mask
    - 聚合成 36 dim 特征

    改动：
    - 不再读单 pdbid feature 缓存
    - 不再写单 pdbid feature 缓存
    """

    effective_max_cutoff = (
        float(max_cutoff_global) if max_cutoff_global is not None else float(cutoff)
    )

    idx_valid, dist_valid, phi_valid = _load_or_build_pair_cache(
        pdbid=pdbid,
        set_type=set_type,
        alpha=alpha,
        beta=beta,
        tau=tau,
        max_cutoff_global=effective_max_cutoff,
        use_cache=use_cache,
    )

    # cutoff mask（你原来的逻辑）
    RI = np.zeros(FEATURE_DIM, dtype=float)
    if idx_valid.size > 0:
        mask = dist_valid <= float(cutoff)
        if np.any(mask):
            np.add.at(RI, idx_valid[mask], phi_valid[mask])

    return RI


# ================== 新增：构建大矩阵 ==================
def build_bigmatrix(
    set_type: str,
    alpha: str,
    beta: float,
    tau: float,
    cutoff: float,
    use_cache: bool = True,
    max_cutoff_global: float | None = None,
):
    """
    构建 N×36 大矩阵，不改 paramcached 的计算逻辑。
    """

    csv_file = PROCESSED / set_type / f"{set_type}_set_list.csv"
    df = pd.read_csv(csv_file)
    pdbids = df["pdbid"].tolist()
    y = df["pkd"].values

    out_path = _bigmatrix_path(set_type, alpha, beta, tau, cutoff)
    out_dir = out_path.parent
    out_dir.mkdir(exist_ok=True, parents=True)

    # 保存顺序
    np.save(out_dir / f"{set_type}_pdbids.npy", np.array(pdbids))
    np.save(out_dir / f"{set_type}_pkd.npy", y)

    # 构建大矩阵
    N = len(pdbids)
    X = np.zeros((N, FEATURE_DIM), dtype=float)

    for i, pdbid in tqdm(
        enumerate(pdbids), total=N,
        desc=f"Bigmatrix(paramcached): a={alpha}, b={beta}, t={tau}, c={cutoff}"
    ):
        X[i] = load_feature(
            pdbid=pdbid,
            set_type=set_type,
            alpha=alpha,
            beta=beta,
            tau=tau,
            cutoff=cutoff,
            use_cache=use_cache,
            max_cutoff_global=max_cutoff_global,
        )

    np.save(out_path, X)
    print(f"[完成] 写入大矩阵: {out_path}, shape={X.shape}")

    return out_path


# ================== 新增：训练时读取大矩阵 ==================
def load_dataset_bigmatrix(
    set_type: str,
    alpha: str,
    beta: float,
    tau: float,
    cutoff: float,
):
    """
    一次性加载 X, y, pdbids (速度最快, 用于训练)
    """
    base = FEATURE_CACHE_DIR / set_type
    tag = f"a{alpha}_b{_fmt(beta)}_t{_fmt(tau)}_c{_fmt(cutoff)}.npy"

    X = np.load(base / tag)
    y = np.load(base / f"{set_type}_pkd.npy")
    pdbids = np.load(base / f"{set_type}_pdbids.npy").tolist()

    return X, y, pdbids


# ================== 并行预计算入口 ==================

def precompute_pair_cache_for_set(
    set_type: str,
    alpha: str,
    beta: float,
    tau: float,
    max_cutoff_global: float,
    use_cache: bool = True,
    n_jobs: int = 1,
):
    """
    为某个 set_type ('refined_only' 或 'core') + (alpha,beta,tau,max_cutoff_global)
    预先计算/刷新所有 pdbid 的 pair cache。

    这样后面无论是 build_bigmatrix 还是 load_feature，都可以直接命中 pair 缓存，
    避免重复从 structures 计算。
    """
    assert set_type in ["refined_only", "core"], "set_type 必须是 'refined_only' 或 'core'"

    csv_file = PROCESSED / set_type / f"{set_type}_set_list.csv"
    df = pd.read_csv(csv_file)
    pdbids = df["pdbid"].tolist()

    def _worker(pdbid: str):
        _load_or_build_pair_cache(
            pdbid=pdbid,
            set_type=set_type,
            alpha=alpha,
            beta=beta,
            tau=tau,
            max_cutoff_global=max_cutoff_global,
            use_cache=use_cache,
        )

    Parallel(n_jobs=n_jobs)(
        delayed(_worker)(pdbid) for pdbid in tqdm(
            pdbids,
            desc=f"Precomputing pair cache ({set_type}, alpha={alpha}, beta={beta}, tau={tau}, cmax={max_cutoff_global})"
        )
    )
