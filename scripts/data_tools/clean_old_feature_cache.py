#!/usr/bin/env python3
"""
删除旧的 per-pdbid feature 缓存文件（即 feature_cache 下按 pdbid 命名的小文件）。

不会删除：
- pair cache (ri_pair_param_cache, ri_pair_geom_cache)
- 大矩阵文件（无 pdbid 前缀）
- pdbids.npy / pkd.npy 等标签文件

执行方式：
    python cleanup_old_feature_cache.py
"""

from pathlib import Path
import re

# === 项目根目录推断 ===
FEATURE_CACHE = Path("data/processed/feature_cache")

# === 匹配旧小文件的模式 ===
# 旧格式类似：
#   1abc_aexp_b2.5_t1.0_c12.npy
#   xxxx_alor_b5_t0.5_c20.npy
# 所以规则：文件名以 pdbid 开头，然后有 "_a" 或 "_lor" 等参数字段。
old_file_pattern = re.compile(r"^[0-9A-Za-z]{1,8}_a.*\.npy$")

def is_old_feature_file(file: Path) -> bool:
    """判断是否为旧的小文件（按 pdbid 命名）"""
    fname = file.name
    return bool(old_file_pattern.match(fname))

def main():
    if not FEATURE_CACHE.exists():
        print(f"Feature cache directory not found: {FEATURE_CACHE}")
        return

    print(f"Searching for old feature files in: {FEATURE_CACHE}")

    count = 0
    for set_dir in ["refined_only", "core"]:
        subdir = FEATURE_CACHE / set_dir
        if not subdir.exists():
            continue

        print(f"Checking {subdir} ...")

        for f in subdir.glob("*.npy"):
            # 跳过新大矩阵文件：它们是 "aexp_b2.5_t1.0_c12.npy"，文件名以"a"开头，不是 pdbid
            # 跳过标签文件："refined_only_pdbids.npy" / "refined_only_pkd.npy"
            if f.name.startswith(set_dir):
                continue
            if f.name.startswith("a"):
                continue
            if f.name.startswith("l"):
                continue

            # 判断是否旧格式的小文件
            if is_old_feature_file(f):
                print(f"  [删除] {f}")
                f.unlink()
                count += 1

    print(f"\n完成：共删除 {count} 个旧 feature 小文件。")


if __name__ == "__main__":
    main()
