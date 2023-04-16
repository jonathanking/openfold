"""This script will take a directory of structs minimized by SCN and filter them by RMSD.

Usage:
    filter_minimized_structs_by_rmsd.py <input_min_dir> <output_filt_dir> <log_file_dir> <rmsd>.
    
    <input_min_dir> = Directory containing minimized structures.
    <output_filt_dir> = Directory to write filtered structures to.
    <log_file_dir> = Directory containing log files. Each log file will report RMSD to original structure.
    <rmsd> = RMSD cutoff.

Example filtering the test set:

    python jk_research/utils/filter_minimized_structs_by_rmsd.py \
        ~/scnmin_evaltest230412/min ~/scnmin_evaltest230412/rmsd_filt_min/ \
        /net/pulsar/home/koes/jok120/repos/sidechainnet/sidechainnet/research/cluster/230413test 5

Example filtering the validation set:

    python jk_research/utils/filter_minimized_structs_by_rmsd.py \
        ~/scnmin_eval230412/min/ \
        ~/scnmin_eval230412/rmsd_filt_min/ \
        /net/pulsar/home/koes/jok120/repos/sidechainnet/sidechainnet/research/cluster/230413 5
"""

import argparse
import os
import glob
import shutil
from tqdm import tqdm


def main(input_dir, output_dir, log_file_dir, rmsd_cutoff=5):
    """Main function."""
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    minimized_structs = glob.glob(os.path.join(input_dir, '*.pdb'))

    # The directory containing the log files will have a log file for each minimized structure.
    # Each log file will report the RMSD to the original structure like so: 7vnb_A,RMSD_CA,0.2526460111767908,379.6502332687378
    # Extract all protein names and RMSDs from the log file dir and record as a dict.

    rmsd_lines = os.popen(f"grep -h \"RMSD_CA\" {log_file_dir}/*out").readlines()
    rmsd_dict = {}
    for line in rmsd_lines:
        line = line.split(',')
        rmsd_dict[line[0]] = float(line[2])

    passed_filter = 0
    for minimized_struct in tqdm(minimized_structs):
        struct_name = os.path.basename(minimized_struct)
        struct_name = struct_name.split('.')[0]

        rmsd = rmsd_dict[struct_name]

        if rmsd < rmsd_cutoff:
            shutil.copy(minimized_struct, output_dir)
            passed_filter += 1
    
    print(f"{passed_filter} structures passed filter out of {len(minimized_structs)} total structures.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        'This script will take a directory of minimized structures and filter them by RMSD.'
    )
    parser.add_argument('input_min_dir', help='Directory containing minimized structures.')
    parser.add_argument('output_filtered_dir', help='Directory to write filtered structures to.')
    parser.add_argument(
        'log_file_dir',
        help=
        'Directory containing log files. Each log file will report RMSD to original structure.'
    )
    parser.add_argument('rmsd', type=float, help='RMSD cutoff.')
    args = parser.parse_args()

    main(args.input_min_dir, args.output_filtered_dir, args.log_file_dir, args.rmsd)

