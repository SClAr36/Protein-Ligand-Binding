import numpy as np
from pathlib import Path

FEATURE_CACHE_DIR = Path("data/processed/feature_cache")

print("Scanning for corrupted .npy files...")

bad_files = []

for f in FEATURE_CACHE_DIR.glob("*.npy"):
    try:
        arr = np.load(f)
        if arr is None or arr.size == 0:
            raise ValueError("empty content")
    except Exception as e:
        bad_files.append(f)
        print(f"[BAD] {f} ({e})")

print(f"\nFound {len(bad_files)} corrupted cache files.")
if bad_files:
    print("You can delete them by running:")
    for f in bad_files:
        print(f"rm \"{f}\"")
