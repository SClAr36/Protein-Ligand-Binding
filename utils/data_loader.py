import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from utils.RI_score import compute_RI_score_general, FEATURE_DIM

# project_root/utils/data_loader.py
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

# 特征缓存目录：按 (pdbid, alpha, beta, tau, cutoff) 区分
CACHE_DIR = PROCESSED / "feature_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _feature_cache_path(pdbid: str, alpha: str, beta: float, tau: float, cutoff: float) -> Path:
    """
    生成某个复合物在特定 RI 参数下的缓存文件路径。
    不区分 refined/core，因为同一个 pdbid 几何是一样的。
    """
    tag = f"a{alpha}_b{beta}_t{tau}_c{cutoff}"
    return CACHE_DIR / f"{pdbid}_{tag}.npy"


def load_feature(
    pdbid: str,
    set_type: str,
    alpha: str,
    beta: float,
    tau: float,
    cutoff: float,
    use_cache: bool = True,
) -> np.ndarray:
    """
    对单个 complex 计算 36 维 RI 特征，带缓存。
    set_type 用来决定从哪个目录加载 npz（refined/core），
    但缓存是以 pdbid 为键的，全局共享。
    """
    cache_file = _feature_cache_path(pdbid, alpha, beta, tau, cutoff)

    # 优先用缓存
    if use_cache and cache_file.exists():
        return np.load(cache_file)

    # 否则重新计算
    npz_path = PROCESSED / set_type / "structures" / f"{pdbid}.npz"
    data = np.load(npz_path)

    F = compute_RI_score_general(
        data["pro_coords"],
        data["pro_elems"],
        data["lig_coords"],
        data["lig_elems"],
        alpha, beta, tau, cutoff,
    )

    if use_cache:
        np.save(cache_file, F)

    return F


def build_dataset(
    set_type: str,
    alpha: str,
    beta: float,
    tau: float,
    cutoff: float,
    use_cache: bool = True,
):
    """
    生成某个 set (refined/core) 在给定 RI 参数下的 (X, y, pdbids)。

    X: [n_samples, 36]
    y: [n_samples]   (pKd)
    pdbids: list[str]
    """
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
        )

    return X, y, pdbids
