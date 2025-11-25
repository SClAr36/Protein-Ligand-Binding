import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

from utils.RI_score import (
    precompute_phi_pairs,
    FEATURE_DIM,
)

# ================== 路径设置 ==================
# 当前：所有东西都放在项目根目录下的 data/ 里
# 将来如果你想把 data 挂到外接 SSD，只需要改这一行：
#   比如改成：DATA_ROOT = Path("/mnt/data_ext/ProLig/data")
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"

PROCESSED = DATA_ROOT / "processed"
INTERIM = DATA_ROOT / "interim"
INTERIM.mkdir(exist_ok=True)

# 最终 36 维特征缓存目录：按 (pdbid, alpha, beta, tau, cutoff) 区分
FEATURE_CACHE_DIR = PROCESSED / "feature_cache"
FEATURE_CACHE_DIR.mkdir(exist_ok=True)

# pair 级别 φ 预计算缓存目录：按 (pdbid, alpha, beta, tau, max_cutoff_global) 区分
PAIR_CACHE_DIR = INTERIM / "ri_pair_cache"
PAIR_CACHE_DIR.mkdir(exist_ok=True)


def _feature_cache_path(pdbid: str, alpha: str, beta: float, tau: float, cutoff: float) -> Path:
    """
    生成某个复合物在特定 RI 参数 + cutoff 下的最终 36 维特征缓存路径。
    """
    tag = f"a{alpha}_b{beta:g}_t{tau:g}_c{cutoff:g}"
    return FEATURE_CACHE_DIR / f"{pdbid}_{tag}.npy"


def _pair_cache_path(
    pdbid: str,
    alpha: str,
    beta: float,
    tau: float,
    max_cutoff_global: float,
) -> Path:
    """
    生成某个复合物在 (alpha, beta, tau, max_cutoff_global) 下的 φ 预计算缓存路径。
    文件名显式带上 max_cutoff_global，避免未来更大 cutoff 时产生混淆。
    """
    tag = f"a{alpha}_b{beta:g}_t{tau:g}_cmax{max_cutoff_global:g}"
    return PAIR_CACHE_DIR / f"{pdbid}_{tag}.npz"


def _load_structure_npz(set_type: str, pdbid: str) -> dict:
    """
    读取单个 complex 的结构 npz 文件。
    """
    npz_path = PROCESSED / set_type / "structures" / f"{pdbid}.npz"
    data = np.load(npz_path)
    return data


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
    pair_cache_file = _pair_cache_path(pdbid, alpha, beta, tau, max_cutoff_global)

    if use_cache and pair_cache_file.exists():
        cache = np.load(pair_cache_file)
        idx_valid = cache["idx_valid"]
        dist_valid = cache["dist_valid"]
        phi_valid = cache["phi_valid"]
        return idx_valid, dist_valid, phi_valid

    # 没有缓存，重新计算
    data = _load_structure_npz(set_type, pdbid)

    pro_coords = data["pro_coords"]
    pro_elems = data["pro_elems"]
    lig_coords = data["lig_coords"]
    lig_elems = data["lig_elems"]

    idx_valid, dist_valid, phi_valid = precompute_phi_pairs(
        pro_coords=pro_coords,
        pro_elems=pro_elems,
        lig_coords=lig_coords,
        lig_elems=lig_elems,
        alpha=alpha,
        beta=beta,
        tau=tau,
        max_cutoff=max_cutoff_global,
    )

    if use_cache:
        # 压缩保存，避免磁盘占用过大
        np.savez_compressed(
            pair_cache_file,
            idx_valid=idx_valid,
            dist_valid=dist_valid,
            phi_valid=phi_valid,
        )

    return idx_valid, dist_valid, phi_valid


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
    对单个 complex 计算 36 维 RI 特征，带多层缓存。

    逻辑：
    1）先看是否已有最终 36 维特征缓存 (pdbid, alpha, beta, tau, cutoff)；
    2）否则，读取/构建 pair 级 φ 缓存 (pdbid, alpha, beta, tau, max_cutoff_global)；
    3）对 dist_valid <= cutoff 的 pair，用 np.add.at 聚合到 36 维向量；
    4）可选将这 36 维向量缓存起来。

    max_cutoff_global:
        - 如果为 None，则退化为 max_cutoff_global = cutoff（只对当前 cutoff 有效）；
        - 如果你在外部传入 max_cutoff_global = max(cutoff_list)，
          那么同一 (alpha,beta,tau) 这一轮 sweep 所有 cutoff 都可以重用同一份 pair 缓存。
    """
    # 1) 先看最终特征缓存
    feat_cache_file = _feature_cache_path(pdbid, alpha, beta, tau, cutoff)
    if use_cache and feat_cache_file.exists():
        return np.load(feat_cache_file)

    # 2) 确定有效的 max_cutoff_global
    if max_cutoff_global is None:
        effective_max_cutoff = float(cutoff)
    else:
        effective_max_cutoff = float(max_cutoff_global)

    # 读取 / 生成 pair 缓存
    idx_valid, dist_valid, phi_valid = _load_or_build_pair_cache(
        pdbid=pdbid,
        set_type=set_type,
        alpha=alpha,
        beta=beta,
        tau=tau,
        max_cutoff_global=effective_max_cutoff,
        use_cache=use_cache,
    )

    # 3) 针对当前 cutoff 聚合到 36 维
    RI = np.zeros(FEATURE_DIM, dtype=float)

    if idx_valid.size > 0:
        mask = dist_valid <= float(cutoff)
        if np.any(mask):
            np.add.at(RI, idx_valid[mask], phi_valid[mask])

    # 4) 写入最终特征缓存
    if use_cache:
        np.save(feat_cache_file, RI)

    return RI


def build_dataset(
    set_type: str,
    alpha: str,
    beta: float,
    tau: float,
    cutoff: float,
    use_cache: bool = True,
    max_cutoff_global: float | None = None,
):
    """
    生成某个 set (refined/core) 在给定 RI 参数下的 (X, y, pdbids)。

    X: [n_samples, 36]
    y: [n_samples]   (pKd)
    pdbids: list[str]

    max_cutoff_global:
        - None: 每次调用只针对本次 cutoff 做预计算（兼容旧用法）；
        - 非 None: 推荐设为本轮 sweep 的 max(cutoff_list)，
          这样同一组 (alpha,beta,tau) 下所有 cutoff 都可以重用 pair 缓存。
    """
    assert set_type in ["refined", "core"], "set_type 必须是 'refined' 或 'core'"

    csv_file = PROCESSED / set_type / f"{set_type}_set_list.csv"
    df = pd.read_csv(csv_file)

    n = len(df)
    X = np.zeros((n, FEATURE_DIM), dtype=float)
    y = df["pkd"].values
    pdbids = df["pdbid"].tolist()

    for k, pdbid in tqdm(
        enumerate(pdbids),
        total=n,
        desc=f"Building {set_type} set (alpha={alpha}, beta={beta}, tau={tau}, cutoff={cutoff})"
    ):
        X[k] = load_feature(
            pdbid=pdbid,
            set_type=set_type,
            alpha=alpha,
            beta=beta,
            tau=tau,
            cutoff=cutoff,
            use_cache=use_cache,
            max_cutoff_global=max_cutoff_global,
        )

    return X, y, pdbids


# ================== 可选：并行预计算入口 ==================

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
    可选工具函数：提前并行为某个 set (refined/core) + (alpha,beta,tau,max_cutoff_global)
    预计算/刷新所有 pair 缓存。你以后如果要扩展很多 alpha/beta 的组合，可以调这个函数来加速。

    举例：
        from utils.data_loader import precompute_pair_cache_for_set

        max_cutoff_global = max(cutoff_list)
        precompute_pair_cache_for_set(
            set_type="refined",
            alpha="exp",
            beta=2.5,
            tau=1.0,
            max_cutoff_global=max_cutoff_global,
            n_jobs=8,
        )

    然后再跑训练时，build_dataset/load_feature 都能直接命中 pair 缓存。
    """
    assert set_type in ["refined", "core"], "set_type 必须是 'refined' 或 'core'"

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
