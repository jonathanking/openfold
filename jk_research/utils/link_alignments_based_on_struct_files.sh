#! /bin/bash
# This script will take a structure directory, an alignment directory and an outpyt directory
# and will create a new alignment directory containing symlinks to the relevant alignments
# Usage: ./link_alignments_based_on_struct_files.sh <structure_dir> <alignment_dir> <output_dir>
#
# Example for linking minimized test alignments:
# jk_research/utils/link_alignments_based_on_struct_files.sh /scr/jok120/yoda/test_data/cameo/20230103/minimized/data_dir/ /scr/jok120/yoda/test_data/cameo/20230103/alignments/ /scr/jok120/yoda/test_data/cameo/20230103/minimized/alignments

# Print usage if not enough arguments
if [ $# -ne 3 ]; then
    echo "Usage: ./link_alignments_based_on_struct_files.sh <structure_dir> <alignment_dir> <output_dir>"
    exit 1
fi

STRUCT_DIR=$1
ALIGN_DIR=$2
OUT_DIR=$3

mkdir -p ${OUT_DIR}

echo "Linking $(wc -l ${STRUCT_DIR}/*.pdb | tail -n 1 | awk '{print $1}') structures from ${STRUCT_DIR} to ${OUT_DIR} using alignments from ${ALIGN_DIR}..."

for struct in ${STRUCT_DIR}/*.pdb; do
    struct_name=$(basename ${struct} .pdb)
    echo "Linking ${struct_name}..."
    ln -s ${ALIGN_DIR}/${struct_name} ${OUT_DIR}/${struct_name}
done

