""""The script takes a directory as input, which contains many subdirectories, each of which contains an evaluated run.
It produces on stdout a list of python commands to run the relaxation analysis on each of the subdirectories.

Example:
 python make_relaxation_cmds.py /net/pulsar/home/koes/jok120/openfold/out/evaluation/230629 > relaxation_cmds.sh
 
Each command looks like:
    python jk_research/utils/relax_from_pdb.py --pdb_directory="/net/pulsar/home/koes/jok120/openfold/out/evaluation/230627/30kpure_scnunmin_omm_00_eval_sOMM/pdbs/val/true" --device='cpu' --run_molprobity
"""

import os
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python make_relaxation_cmds.py <directory>")
        sys.exit(1)
    directory = sys.argv[1]
    for experiment in os.listdir(directory):
        experiment_path = os.path.join(directory, experiment)
        if os.path.isdir(experiment_path):
            for pdb_subdir in ['test/pred', 'val/pred', 'test/true', 'val/true']:
                pdb_dir_path = os.path.join(experiment_path, 'pdbs', pdb_subdir)
                print(f"python jk_research/utils/relax_from_pdb.py --pdb_directory='{pdb_dir_path}' --device='cpu' --run_molprobity")

if __name__ == '__main__':
    main()
    