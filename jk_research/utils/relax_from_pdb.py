"""Performs AlphaFold-esque relaxation on a PDB file."""

import argparse
import os

import openfold.np.protein as protein
import openfold.utils.script_utils as script_utils
from openfold.config import model_config

CONFIG_PRESET = "finetuning_sidechainnet"
MOLPROBITY_PATH = "/net/pulsar/home/koes/jok120/build/phenix-install/phenix-1.20.1-4487/molprobity/cmdline/oneline-analysis"


def get_openfold_protein_from_path(path):
    with open(path, "r") as f:
        pdb_string = f.read()
    return protein.from_pdb_string(pdb_string)


def relax_single_protein_from_pdb_path(pdb_path, relaxed_pdb_path, model_device):
    config = model_config(CONFIG_PRESET)
    try:
        unrelaxed_protein = get_openfold_protein_from_path(pdb_path)
    except ValueError:
        print(f"Failed to parse {pdb_path}. Skipping.")
        return [(pdb_path, -1)]
    output_directory = os.path.dirname(relaxed_pdb_path)
    output_name = os.path.basename(pdb_path).split(".")[0]

    relax_time = script_utils.relax_protein(config,
                                            model_device,
                                            unrelaxed_protein,
                                            output_directory,
                                            output_name,
                                            return_time=True)
    return [(pdb_path, relax_time)]


def relax_proteins_from_pdb_directory(pdb_directory, relaxed_pdb_dir, model_device):
    paths_times = []
    for pdb_name in os.listdir(pdb_directory):
        pdb_path = os.path.join(pdb_directory, pdb_name)
        relaxed_pdb_path = os.path.join(relaxed_pdb_dir, pdb_name)
        paths_times.extend(
            relax_single_protein_from_pdb_path(pdb_path, relaxed_pdb_path, model_device))
    return paths_times


def create_results_csv(paths_times, output_csv_path):
    with open(output_csv_path, "w") as f:
        f.write("protein_name,relaxation_time\n")
        for path_time in paths_times:
            path = path_time[0]
            time = path_time[1]
            protein_name = os.path.basename(path).split(".")[0]
            f.write(f"{protein_name},{time}\n")


def run_molprobity_on_pdb_dir(pdb_dir):
    output_file = os.path.join(pdb_dir, "molprobity.out")
    cmd = f"{MOLPROBITY_PATH} {pdb_dir} > {output_file}"
    os.system(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Relax a PDB file or directory of PDB files.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdb_path", type=str, help="Path to a PDB file to relax.")
    group.add_argument("--pdb_directory",
                       type=str,
                       help="Path to a directory of PDB files to relax.")

    group2 = parser.add_mutually_exclusive_group(required=False)
    group2.add_argument("--relaxed_pdb_directory",
                        type=str,
                        help="Path to a directory to write the relaxed PDB files to.")
    group2.add_argument("--relaxed_pdb_path",
                        type=str,
                        help="Path to a PDB file to write the relaxed PDB file to.")

    parser.add_argument("--output_csv_path",
                        type=str,
                        help="Path to a CSV file to write the results to.")
    parser.add_argument("--device", type=str, help="Device to run the relaxation on.")
    parser.add_argument("--run_molprobity",
                        action="store_true",
                        help="Whether to run MolProbity on both "
                        "the unrelaxed and relaxed PDB files.")
    args = parser.parse_args()

    if args.run_molprobity and args.pdb_path:
        raise ValueError("Cannot run MolProbity on a single PDB file.")

    if args.pdb_directory and not args.relaxed_pdb_directory:
        args.relaxed_pdb_directory = args.pdb_directory + "_relaxed"


    print("Relaxing proteins...", end="")

    if args.pdb_path:
        times = relax_single_protein_from_pdb_path(args.pdb_path, args.relaxed_pdb_path,
                                                   args.device)
    elif args.pdb_directory:
        os.makedirs(args.relaxed_pdb_directory, exist_ok=True)
        times = relax_proteins_from_pdb_directory(args.pdb_directory,
                                                  args.relaxed_pdb_directory, args.device)
    print(" done.", end=" ")

    if not args.output_csv_path:
        args.output_csv_path = os.path.join(args.relaxed_pdb_directory, "relaxation_times.csv")
    create_results_csv(times, args.output_csv_path)
    print(f"Relaxation results written to {args.output_csv_path}")

    if args.run_molprobity:
        print("Running MolProbity...", end="")
        run_molprobity_on_pdb_dir(args.pdb_directory)
        run_molprobity_on_pdb_dir(args.relaxed_pdb_directory)
        print(f"MolProbity results written to {args.pdb_directory}/molprobity.out and "
              f"{args.relaxed_pdb_directory}/molprobity.out")


if __name__ == "__main__":
    main()
