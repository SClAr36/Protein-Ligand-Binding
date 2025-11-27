from pathlib import Path
import pandas as pd
import shutil

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data/processed"

FEATURE_CACHE = PROCESSED / "feature_cache"

REFINED_ONLY_DIR = FEATURE_CACHE / "refined_only"
CORE_DIR = FEATURE_CACHE / "core"

CORE_CSV = PROCESSED / "core" / "core_set_list.csv"


def move_core_feature_cache():
    # 1. Read core set list
    core_df = pd.read_csv(CORE_CSV)
    core_ids = set(core_df["pdbid"].str.lower())

    # 2. Prepare output directory
    CORE_DIR.mkdir(parents=True, exist_ok=True)

    moved_files = 0
    moved_complex_ids = set()
    skipped = 0
    failed = 0

    # 3. Iterate refined_only/ files
    for file in REFINED_ONLY_DIR.iterdir():
        if not file.is_file():
            continue

        # parse pdbid
        name = file.name
        try:
            pdbid = name.split("_")[0][:4].lower()
        except Exception:
            print(f"[WARNING] unexpected filename: {name}")
            failed += 1
            continue

        # check membership
        if pdbid in core_ids:
            # move file
            dst = CORE_DIR / file.name
            shutil.move(str(file), str(dst))

            moved_files += 1
            moved_complex_ids.add(pdbid)
        else:
            skipped += 1

    # 4. Print summary
    print("===== Core Feature Cache Movement =====")
    print(f"Core complexes (from CSV): {len(core_ids)}")
    print(f"Core complexes found in feature_cache: {len(moved_complex_ids)}")
    print(f"Total feature files moved: {moved_files}")
    print(f"Remaining in refined_only: {skipped}")
    print(f"Failed filename parse: {failed}")
    print(f"Core folder: {CORE_DIR}")


if __name__ == "__main__":
    move_core_feature_cache()
