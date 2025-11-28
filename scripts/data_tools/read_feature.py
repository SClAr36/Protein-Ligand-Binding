from pathlib import Path
import numpy as np
import pandas as pd

FEATURE_CACHE_DIR = Path("data/processed/feature_cache")

alpha = "exp"
beta = 2.5
tau = 1
cutoff_list = list(range(5, 51))   # 5~50

def load_cutoff_series(pdbid, set_type="refined_only"):
    rows = []
    for c in cutoff_list:
        f = FEATURE_CACHE_DIR / set_type / f"{pdbid}_a{alpha}_b{beta:g}_t{tau:g}_c{c}.npy"
        if f.exists():
            arr = np.load(f)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            rows.append(arr[0])
        else:
            print(f"[WARNING] missing cutoff={c}, file={f}")
            rows.append(np.full(36, np.nan))  # 缺失时填 NaN，方便你看到哪里断裂

    X = np.vstack(rows)
    print(f"[INFO] Loaded feature series for {pdbid}: shape={X.shape}")
    return X

# ====== 你想看的 PDBID ======
pdbid = "4eu0"   # 替换成你自己的

X = load_cutoff_series(pdbid)

# 保存为 CSV（46 × 36），第一列是 cutoff
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
df.insert(0, "cutoff", cutoff_list)
df.to_csv(f"{pdbid}_all_cutoffs.csv", index=False)

print(f"Saved to {pdbid}_all_cutoffs.csv")
