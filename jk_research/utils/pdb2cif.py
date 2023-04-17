"""Convert .pdb files to .cif files (or directories of those files) using biopython."""
import argparse
from glob import glob
import os
from tqdm import tqdm

from Bio.PDB import MMCIFParser, PDBIO, PDBParser, MMCIFIO


def main(pdb_file, cif_file, use_directories=False):

    if not use_directories:
        pdb2cif(pdb_file, cif_file)
    else:
        if not os.path.exists(cif_file):
            os.makedirs(cif_file, exist_ok=True)
        for pdb in tqdm(glob(os.path.join(pdb_file, "*.pdb"))):
            print(pdb)
            outfile = os.path.join(cif_file,
                                   os.path.basename(pdb).replace(".pdb", ".cif"))
            pdb2cif(pdb, outfile)


def pdb2cif(pdb_file, cif_file):
    """Convert a single .pdb file to a .cif file."""
    # parser = PDBParser()
    # structure = parser.get_structure("", pdb_file)
    # io = MMCIFIO()
    # io.set_structure(structure)
    # io.save(cif_file)
    # use gemmi CLI converter instead of Biopython
    chain = os.path.basename(pdb_file).split("_")[1]
    os.system(f"gemmi convert {pdb_file} {cif_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdb_file", help="Path to .pdb file")
    parser.add_argument("cif_file", help="Path to .cif file")
    parser.add_argument(
        "--use_directories",
        action="store_true",
        help="Treat input and output paths as directories for bulk conversion.")
    args = parser.parse_args()

    main(args.pdb_file, args.cif_file, args.use_directories)