from pathlib import Path
import numpy as np
import pandas as pd

# =========================================
# 设置路径
# =========================================
ROOT = Path("data/interim/ri_pair_param_cache")   # 你可以修改为绝对路径
#SET_TYPE = "refined_only"                         # 或 "core"
CACHE_DIR = ROOT

# 自定义距离区间 (包含右，不包含左)
# 即 (0,5], (5,10], ..., (40,50]
BINS = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 40), (40, 50)]


# =========================================
# 工具函数：读取 npz / npy 中的 dist_valid
# =========================================
def load_dist_valid(path: Path):
    """自动支持 .npz 或 .npy"""
    if path.suffix == ".npz":
        data = np.load(path)
        if "dist_valid" in data:
            return data["dist_valid"]
        else:
            raise KeyError(f"{path} 中没有 dist_valid 键")
    elif path.suffix == ".npy":
        return np.load(path)
    else:
        raise ValueError(f"不支持的文件类型: {path}")


# =========================================
# 工具函数：统计一次距离数组在各区间的数量
# =========================================
def histogram_by_bins(dist_array):
    counts = []
    for lo, hi in BINS:
        mask = (dist_array > lo) & (dist_array <= hi)
        counts.append(mask.sum())
    return np.array(counts)


# =========================================
# 模式一：检查某个 pdbid 的距离分布
# =========================================
def inspect_single_pdb(pdbid: str):
    """抽查某 pdbid 的 dist_valid 落在各区间的数量和占比"""
    # 找文件
    matches = list(CACHE_DIR.glob(f"{pdbid}_aexp_b2.5_t1_cmax50.*"))
    if not matches:
        raise FileNotFoundError(f"找不到 {pdbid} 对应的缓存文件")

    path = matches[0]
    dist_valid = load_dist_valid(path)

    print(f"[INFO] 加载 {pdbid} 的 dist_valid, 数量={len(dist_valid)}")

    counts = histogram_by_bins(dist_valid)
    total = len(dist_valid)
    percents = counts / total * 100

    df = pd.DataFrame({
        "bin": [f"({lo},{hi}]" for lo, hi in BINS],
        "count": counts,
        "percent": percents,
    })

    print(df)
    return df


# =========================================
# 模式二：统计全部 pdbid 的合并分布
# =========================================
def inspect_all_pdb():
    """统计整个文件夹所有 pdb 的 dist_valid，总体分布占比"""
    files = list(CACHE_DIR.glob("*.np*"))
    if not files:
        raise FileNotFoundError(f"{CACHE_DIR} 里没有缓存文件")

    total_counts = np.zeros(len(BINS), dtype=int)
    total_n = 0

    for f in files:
        try:
            dist_valid = load_dist_valid(f)
        except Exception as e:
            print(f"[WARN] 跳过文件 {f}: {e}")
            continue

        counts = histogram_by_bins(dist_valid)
        total_counts += counts
        total_n += len(dist_valid)

    percents = total_counts / total_n * 100

    df = pd.DataFrame({
        "bin": [f"({lo},{hi}]" for lo, hi in BINS],
        "count": total_counts,
        "percent": percents,
    })

    print(f"[INFO] 总共处理 {len(files)} 个 pdb 文件，总 pair 数 = {total_n}")
    print(df)
    return df


# =========================================
# 主函数示例（你可以自由修改）
# =========================================
if __name__ == "__main__":
    mode = "all"   # "single" 或 "all"

    if mode == "single":
        pdbid = "1a1e"      # <- 在这里填你想检查的 pdbid
        inspect_single_pdb(pdbid)

    elif mode == "all":
        inspect_all_pdb()

    else:
        raise ValueError("mode 必须是 'single' 或 'all'")
