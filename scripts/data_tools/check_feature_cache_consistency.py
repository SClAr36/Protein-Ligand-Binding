# scripts/check_feature_cache_consistency.py
import numpy as np
from pathlib import Path


def load_first_column(csv_file):
    """只读取 CSV 第一列，跳过第一行表头，忽略其他列"""
    arr = np.loadtxt(
        csv_file,
        dtype=str,
        delimiter=",",
        usecols=[0],   # 只读第一列
        skiprows=1,    # 跳过第一行表头
    )
    # 保证是至少一维
    arr = np.atleast_1d(arr)
    return set(arr.tolist())


def extract_pdbid_from_feature(filename):
    """从 feature 文件名中解析 pdbid（约定为第一个 '_' 之前的内容）"""
    return filename.split("_")[0]


def main():
    DATA = Path("data")
    PROCESSED = DATA / "processed"
    FEATURE_CACHE = PROCESSED / "feature_cache"

    refined_csv = PROCESSED / "refined" / "refined_set_list.csv"
    core_csv = PROCESSED / "core" / "core_set_list.csv"

    refined_ids = load_first_column(refined_csv)
    core_ids = load_first_column(core_csv)

    print(f"[INFO] refined_set_list.csv 第一列共有 {len(refined_ids)} 条")
    print(f"[INFO] core_set_list.csv 第一列共有 {len(core_ids)} 条")
    print()

    refined_feat_dir = FEATURE_CACHE / "refined"
    core_feat_dir = FEATURE_CACHE / "core"

    refined_feat_ids = set(extract_pdbid_from_feature(f.name)
                           for f in refined_feat_dir.glob("*.npy"))
    core_feat_ids = set(extract_pdbid_from_feature(f.name)
                        for f in core_feat_dir.glob("*.npy"))

    print(f"[INFO] feature_cache/refined 中共有 {len(refined_feat_ids)} 个 pdbid")
    print(f"[INFO] feature_cache/core 中共有 {len(core_feat_ids)} 个 pdbid")
    print()

    # -------------------------------------------------------
    # 1) refined 中检查不一致
    # -------------------------------------------------------
    print("====== 检查 refined 集 ======")

    extra_refined = refined_feat_ids - refined_ids
    if extra_refined:
        print("[WARNING] refined feature_cache 中存在 csv 未列出的 pdbid：")
        print(len(extra_refined), "个额外 pdbid：")
        # for x in sorted(extra_refined):
        #     print("   +", x)
    else:
        print("[OK] refined feature_cache 与 refined csv 一致，没有额外 pdbid。")

    missing_refined = refined_ids - refined_feat_ids
    if missing_refined:
        print("[WARNING] refined csv 中存在未生成 feature 的 pdbid：")
        print(len(missing_refined), "个缺失 pdbid：")
        # for x in sorted(missing_refined):
        #     print("   -", x)
    else:
        print("[OK] refined csv 中所有 pdbid 都有 feature。")

    print()

    # -------------------------------------------------------
    # 2) core 中检查不一致
    # -------------------------------------------------------
    print("====== 检查 core 集 ======")

    extra_core = core_feat_ids - core_ids
    if extra_core:
        print("[WARNING] core feature_cache 中存在 csv 未列出的 pdbid：")
        print(len(extra_core), "个额外 pdbid：")
        # for x in sorted(extra_core):
        #     print("   +", x)
    else:
        print("[OK] core feature_cache 与 core csv 一致，没有额外 pdbid。")

    missing_core = core_ids - core_feat_ids
    if missing_core:
        print("[WARNING] core csv 中存在未生成 feature 的 pdbid：")
        print(len(missing_core), "个缺失 pdbid：")
        # for x in sorted(missing_core):
        #     print("   -", x)
    else:
        print("[OK] core csv 中所有 pdbid 都有 feature。")

    print()

    # -------------------------------------------------------
    # 3) refined / core 是否有重叠（通常应该无重复）
    # -------------------------------------------------------
    overlap = refined_ids & core_ids
    if overlap:
        print("[WARNING] refined_set_list 与 core_set_list 有重叠 pdbid：")
        print(len(overlap), "个重叠 pdbid：")
        # for x in sorted(overlap):
        #     print("   *", x)
    else:
        print("[OK] refined 与 core 集完全不重叠。")

    print()
    print("检查完成。")


if __name__ == "__main__":
    main()
