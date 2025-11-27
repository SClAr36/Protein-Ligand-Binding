# scripts/split_feature_cache_by_set.py
import shutil
from pathlib import Path
import pandas as pd


def load_pdbids_from_csv(csv_file: Path):
    """只读取 CSV 的 'pdbid' 列"""
    df = pd.read_csv(csv_file)
    if "pdbid" not in df.columns:
        raise ValueError(f"'pdbid' 列未找到: {csv_file}")
    return set(df["pdbid"].astype(str).str.strip())


def extract_pdbid_from_feature(filename: str) -> str:
    """
    从 feature 文件名中解析 pdbid。
    比如：
        '1abc_alphaE_beta2.0_tau1.5_c40.npy'
    pdbid = '1abc'
    """
    return filename.split("_")[0]


def main():
    DATA = Path("data")
    PROCESSED = DATA / "processed"
    FEATURE_CACHE = PROCESSED / "feature_cache"

    refined_csv = PROCESSED / "refined" / "refined_set_list.csv"
    core_csv = PROCESSED / "core" / "core_set_list.csv"

    refined_ids = load_pdbids_from_csv(refined_csv)
    core_ids = load_pdbids_from_csv(core_csv)

    print(f"[INFO] refined_set_list 中 pdbid 数量 = {len(refined_ids)}")
    print(f"[INFO] core_set_list 中 pdbid 数量 = {len(core_ids)}")

    # 创建 refined / core 目录
    refined_dir = FEATURE_CACHE / "refined"
    core_dir = FEATURE_CACHE / "core"
    refined_dir.mkdir(parents=True, exist_ok=True)
    core_dir.mkdir(parents=True, exist_ok=True)

    # feature_cache 根目录下搜 .npy 文件
    feat_files = list(FEATURE_CACHE.glob("*.npy"))
    print(f"[INFO] feature_cache 根目录发现 {len(feat_files)} 个 .npy feature 文件")

    moved = 0
    skipped = 0
    unknown = 0

    for feat_file in feat_files:
        pdbid = extract_pdbid_from_feature(feat_file.name)

        if pdbid in refined_ids:
            target = refined_dir / feat_file.name
        elif pdbid in core_ids:
            target = core_dir / feat_file.name
        else:
            print(f"[WARN] 未在 refined/core CSV 中找到 pdbid={pdbid} ; 文件={feat_file.name}")
            unknown += 1
            continue

        if target.exists():
            print(f"[Skip] 已存在: {target}")
            skipped += 1
            continue

        print(f"[Move] {feat_file} -> {target}")
        shutil.move(str(feat_file), str(target))
        moved += 1

    print(f"\n[SUMMARY] moved={moved}, skipped={skipped}, unknown={unknown}")


if __name__ == "__main__":
    main()
