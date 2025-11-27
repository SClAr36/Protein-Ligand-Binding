from pathlib import Path
import pandas as pd
import shutil

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

REFINED_ONLY_CSV = PROCESSED / "refined_only" / "refined_only_set_list.csv"

SRC = PROCESSED / "refined" / "structures"
DST = PROCESSED / "refined_only" / "structures"


def move_refined_only_npz():
    # 1. Load refined-only pdbid list
    refined_only_df = pd.read_csv(REFINED_ONLY_CSV)
    refined_only_ids = set(refined_only_df["pdbid"].str.lower())

    # 2. Ensure dst exists
    DST.mkdir(parents=True, exist_ok=True)

    moved = 0
    missing = 0

    # 3. Move matching npz files
    for pdbid in refined_only_ids:
        npz_name = f"{pdbid}.npz"
        src_file = SRC / npz_name
        dst_file = DST / npz_name

        if src_file.exists():
            shutil.move(str(src_file), str(dst_file))
            moved += 1
        else:
            print(f"[WARNING] missing structure: {npz_name}")
            missing += 1

    print("===== Move refined-only npz =====")
    print(f"Refined-only total: {len(refined_only_ids)}")
    print(f"Moved: {moved}")
    print(f"Missing in refined/structure: {missing}")
    print(f"Output dir: {DST}")


if __name__ == "__main__":
    move_refined_only_npz()
