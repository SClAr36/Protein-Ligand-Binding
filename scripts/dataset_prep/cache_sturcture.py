from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data/raw/v2016"
PROCESSED = ROOT / "data/processed"


def load_protein_atoms(path: Path):
    coords, elems = [], []
    for line in path.read_text().splitlines():
        if line.startswith("ATOM"):
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            elems.append(line[76:78].strip())
    return np.array(coords), np.array(elems)


def load_ligand_atoms(path: Path):
    coords, elems = [], []
    read_atoms = False
    for line in path.read_text().splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            read_atoms = True
            continue
        if line.startswith("@<TRIPOS>BOND"):
            read_atoms = False
        if read_atoms:
            parts = line.split()
            coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
            elems.append(''.join(c for c in parts[1] if c.isalpha())[:2])
    return np.array(coords), np.array(elems)


def cache_structures(set_type: str):
    list_path = PROCESSED / set_type / f"{set_type}_set_list.csv"
    df = pd.read_csv(list_path)

    out_dir = PROCESSED / set_type / "structures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for pdbid in tqdm(df["pdbid"]):
        folder = RAW / pdbid
        pdb = folder / f"{pdbid}_protein.pdb"
        mol2 = folder / f"{pdbid}_ligand.mol2"

        pro_coords, pro_elems = load_protein_atoms(pdb)
        lig_coords, lig_elems = load_ligand_atoms(mol2)

        np.savez(
            out_dir / f"{pdbid}.npz",
            pro_coords=pro_coords,
            pro_elems=pro_elems,
            lig_coords=lig_coords,
            lig_elems=lig_elems
        )

    print(f"Cached structures saved to: {out_dir}")


if __name__ == "__main__":
    cache_structures("refined")
    cache_structures("core")
