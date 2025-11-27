import numpy as np
from pathlib import Path

FEATURE_CACHE_DIR = Path("data/processed/feature_cache")

def scan_and_cleanup_feature_cache(subdir: str):
    """
    Scan the feature_cache directory for corrupted or malformed .npy files,
    and delete them after user confirmation.
    """
    scan_dir = Path(FEATURE_CACHE_DIR / subdir)
    print(f"Scanning feature_cache for corrupted or malformed .npy files in {subdir.upper()}...\n")

    bad_files = []
    wrong_suffix_files = []

    # recursively scan
    for f in scan_dir.rglob("*"):
        if not f.is_file():
            continue

        name = f.name
        stem = f.stem

        # Check for malformed suffix
        if (f.suffix != ".npy" and ".npy" in name) or (
            f.suffix == ".npy" and ".npy" in stem
        ):
            wrong_suffix_files.append(f)
            continue

        # Validate valid-looking npy files
        if f.suffix == ".npy":
            try:
                arr = np.load(f)
                if arr is None or arr.size == 0:
                    raise ValueError("empty content")
            except Exception:
                bad_files.append(f)

    print("===== Scan Results =====")
    all_to_delete = wrong_suffix_files + bad_files

    if not all_to_delete:
        print("\nNo files to delete. Done.")
    else:
        print("\nFiles to delete:")
        for f in all_to_delete:
            print(f"  {f}")

    print(f"Malformed suffix files: {len(wrong_suffix_files)}")
    print(f"Corrupted .npy files:   {len(bad_files)}")
    
    # Confirmation
    ans = input("\nDelete these files? (y/N): ").strip().lower()
    if ans not in ("y", "yes"):
        print("\nAborted. No files were deleted.")
    else:
        # Perform deletion
        deleted = 0
        for f in all_to_delete:
            try:
                f.unlink()
                print(f"[DELETED] {f}")
                deleted += 1
            except Exception as e:
                print(f"[ERROR] Failed to delete {f}: {e}")

        print(f"\nCleanup completed. Deleted {deleted} files.")

if __name__ == "__main__":
    #scan_and_cleanup_feature_cache("core")
    scan_and_cleanup_feature_cache("refined_only")