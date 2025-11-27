from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data/raw/v2016"
PROCESSED = ROOT / "data/processed"


def select_set(set_type: str):
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

    out_dir = PROCESSED / set_type
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{set_type}_set_list.csv"
    df.to_csv(out_file, index=False)

    print(f"Found {len(df)} valid complexes in {set_type} set.")
    print(f"Saved listing: {out_file}")


if __name__ == "__main__":
    select_set("refined")
    select_set("core")
