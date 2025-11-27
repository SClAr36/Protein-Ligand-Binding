from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data/raw/v2016"
PROCESSED = ROOT / "data/processed"


def load_set(set_type: str) -> pd.DataFrame:
    """Load (pdbid, pkd) list for a given set (refined/core)."""
    index_file = f"INDEX_{set_type}_data.2016"
    index_path = RAW / "index" / index_file

    items = []
    with index_path.open() as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            pdbid = parts[0].lower()
            pkd = float(parts[3])
            exists = (RAW / pdbid).is_dir()
            if exists:
                items.append((pdbid, pkd))

    df = pd.DataFrame(items, columns=["pdbid", "pkd"])
    return df


def save_set(df: pd.DataFrame, set_name: str):
    """Save DataFrame to processed directory."""
    out_dir = PROCESSED / set_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{set_name}_set_list.csv"
    df.to_csv(out_file, index=False)

    print(f"{set_name}: {len(df)} complexes saved.")
    print(f"Saved listing: {out_file}")


def select_refined_only():
    """refined_only = refined_set - core_set"""
    refined_df = load_set("refined")
    core_df = load_set("core")

    core_ids = set(core_df["pdbid"])

    refined_only_df = refined_df[~refined_df["pdbid"].isin(core_ids)].reset_index(drop=True)

    save_set(refined_only_df, "refined_only")


if __name__ == "__main__":
    # 原有功能
    refined_df = load_set("refined")
    save_set(refined_df, "refined")

    core_df = load_set("core")
    save_set(core_df, "core")

    # 新功能
    select_refined_only()
