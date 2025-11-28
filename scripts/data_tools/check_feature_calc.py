import numpy as np
from pathlib import Path
import random

root = Path("data/processed/feature_cache/refined_only")

# 随机挑 5 个 pdbid
files = list(root.glob("*.npy"))
random.shuffle(files)

def get_pid(name):
    # 假设 pdbid 是前4位，比如 1ABC 或 3XYZ
    return name[:4]

pdbids = []
for f in files:
    pid = get_pid(f.name)
    if pid not in pdbids:
        pdbids.append(pid)
    if len(pdbids) >= 5:
        break

print("抽查 pdbid：", pdbids)

for pid in pdbids:
    print("\n==== 检查 pdbid =", pid, "====")
    files_pid = sorted([f for f in root.glob(f"{pid}*.npy")])
    arrays = [(f.name, np.load(f)) for f in files_pid]

    # 简化：让你先看 mean 是否固定不变
    for name, arr in arrays:
        print(f"{name:30s} mean={arr.mean():.6f}, std={arr.std():.6f}")
