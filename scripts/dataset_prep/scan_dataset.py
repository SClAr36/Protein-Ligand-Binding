from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "v2016"
INDEX_DIR = RAW / "index"

protein_suffix = "_protein.pdb"
ligand_suffix = "_ligand.mol2"


def load_index_set(set_type: str) -> set:
    """Load pdbid list from INDEX_refined/core_data.2016."""
    index_file = f"INDEX_{set_type}_data.2016"
    index_path = INDEX_DIR / index_file

    pdbids = set()
    with index_path.open() as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            pdbid = parts[0].lower()
            pdbids.add(pdbid)

    return pdbids


# ----------------------------
# Load refined & core sets
# ----------------------------
refined_set = load_index_set("refined")
core_set = load_index_set("core")

intersection = refined_set & core_set
refined_only = refined_set - core_set

# ----------------------------
# Scan real folders
# ----------------------------
valid = []
missing_protein = []
missing_ligand = []

all_dirs = [d for d in RAW.iterdir() if d.is_dir()]

for complex_dir in all_dirs:
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

# ----------------------------
# Print results
# ----------------------------
print("===== Dataset Overview =====")
print(f"Total directories in RAW: {len(all_dirs)}")

print("\n--- Index info ---")
print(f"Refined set: {len(refined_set)}")
print(f"Core set: {len(core_set)}")
print(f"Refined ∩ Core: {len(intersection)}")
print(f"Refined-only (refined - core): {len(refined_only)}")

print("\n--- File availability in RAW ---")
print(f"Valid complexes: {len(valid)}")
print(f"Missing protein: {len(missing_protein)}")
print(f"Missing ligand: {len(missing_ligand)}")

print("\nDone.")
