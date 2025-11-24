from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

RAW = ROOT / "data" / "raw" / "v2016"

protein_suffix = "_protein.pdb"
ligand_suffix = "_ligand.mol2"

valid, missing_protein, missing_ligand = [], [], []

for complex_dir in RAW.iterdir():
    if not complex_dir.is_dir():
        continue

    pdbid = complex_dir.name.lower()

    protein = complex_dir / f"{pdbid}{protein_suffix}"
    ligand = complex_dir / f"{pdbid}{ligand_suffix}"

    has_protein = protein.exists()
    has_ligand = ligand.exists()

    if has_protein and has_ligand:
        valid.append(pdbid)
    else:
        if not has_protein:
            missing_protein.append(pdbid)
        if not has_ligand:
            missing_ligand.append(pdbid)

print(f"Total candidates: {len([d for d in RAW.iterdir() if d.is_dir()])}")
print(f"Valid complexes: {len(valid)}")
print(f"Missing protein: {len(missing_protein)}")
print(f"Missing ligand: {len(missing_ligand)}")
