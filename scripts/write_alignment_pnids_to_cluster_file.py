import os
import sidechainnet as scn
from sidechainnet.utils.download import get_pdbid_from_pnid
from tqdm import tqdm


def write_alignment_pnids_to_cluster_file():
    alignment_dir = "/scr/scn_roda/"

    subfolders = [f.path for f in os.scandir(alignment_dir) if f.is_dir()]

    with open("/scr/openfold/scn_clusters.txt", "w") as f:
        for subfolder in subfolders:
            # get folder name
            folder_name = os.path.basename(subfolder)

            scnid_info = folder_name.split("_")
            if len(scnid_info) == 2 or "#" in folder_name:
                continue
            scnid = f"{scnid_info[0]}_{scnid_info[2]}"
            f.write(f"{scnid}\n")


def copy_mmcif_files_to_dir_if_part_of_sidechainnet():
    """Copy mmcif files to a directory if they represent a structure in SidechainNet."""
    original_mmcif_dir1 = "/scr/alphafold_data/pdb_mmcif/mmcif_files_for_roda/"
    original_mmcif_dir2 = "/scr/alphafold_data/pdb_mmcif/mmcif_files/"
    target_mmcif_dir = "/scr/alphafold_data/pdb_mmcif/mmcif_files_for_scn/"
    # Load the complete sidechainnet dataset
    d = scn.load(local_scn_path="/home/jok120/scn221001/sidechainnet_casp12_100.pkl")
    pnids = d.get_pnids()
    del d
    already_copied = set()
    missing_pdbs = set()

    pdb_ids = (get_pdbid_from_pnid(p, return_chain=False, include_is_astral=False) for p in pnids if "TBM" not in p and "FM" not in p)
    pdb_ids = set(pdb_ids)
    for pdb_id in tqdm(pdb_ids, smoothing=0, dynamic_ncols=True):
        # Get the pdb_id and chain_id from the pnid
        # pdb_id, chain_id, is_astral = get_pdbid_from_pnid(p, return_chain=True, include_is_astral=True)
        # if pdb_id in already_copied:
        #     continue
        # Construct the cif file name
        original_cif_file1 = os.path.join(original_mmcif_dir1, f"{pdb_id.lower()}.cif")
        original_cif_file2 = os.path.join(original_mmcif_dir2, f"{pdb_id.lower()}.cif")
        # Copy the file if it exists. If it does not exist, print a warning
        if os.path.exists(original_cif_file1):
            os.system(f"cp {original_cif_file1} {target_mmcif_dir}")
            # append the pdb_id to the already_copied set
            already_copied.add(pdb_id)
        elif os.path.exists(original_cif_file2):
            os.system(f"cp {original_cif_file2} {target_mmcif_dir}")
            # append the pdb_id to the already
            already_copied.add(pdb_id)
        else:
            print(f"Warning: {pdb_id}.cif do not exist.")
            missing_pdbs.add(pdb_id)
            continue
    # Write the missing pdbs to a file
    with open("/scr/openfold/missing_pdbs.txt", "w") as f:
        for pdb_id in missing_pdbs:
            f.write(f"{pdb_id}\n")


def copy_missing_pdb_files_if_part_of_sidechainnet():
    missing_pdbs = []
    with open("/scr/openfold/missing_pdbs.txt", "r") as f:
        for line in f:
            missing_pdbs.append(line.strip())
    original_mmcif_dir1 = "/scr/alphafold_data/pdb_mmcif/mmcif_files_for_roda/"
    original_mmcif_dir2 = "/scr/alphafold_data/pdb_mmcif/mmcif_files/"
    target_mmcif_dir = "/scr/alphafold_data/pdb_mmcif/mmcif_files_for_scn/"

    for pdb_id in missing_pdbs:
        # Construct the cif file name
        original_cif_file1 = os.path.join(original_mmcif_dir1, f"{pdb_id.lower()}.pdb")
        original_cif_file2 = os.path.join(original_mmcif_dir2, f"{pdb_id.lower()}.pdb")
        # Copy the file if it exists. If it does not exist, print a warning
        if os.path.exists(original_cif_file1):
            os.system(f"cp {original_cif_file1} {target_mmcif_dir}")
        elif os.path.exists(original_cif_file2):
            os.system(f"cp {original_cif_file2} {target_mmcif_dir}")
        else:
            print(f"Warning: {pdb_id.lower()}.pdb do not exist.")
            continue


if __name__ == "__main__":
    # write_alignment_pnids_to_cluster_file()
    copy_mmcif_files_to_dir_if_part_of_sidechainnet()
    copy_missing_pdb_files_if_part_of_sidechainnet()