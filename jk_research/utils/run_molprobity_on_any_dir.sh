#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <dir_with_pdbs> <output_path>"
    exit 1
fi

# Set the evaluation top level dir
PDB_DIR=$1
OUT_PATH=$2
molprobity=/net/pulsar/home/koes/jok120/build/phenix-install/phenix-1.20.1-4487/molprobity/cmdline/oneline-analysis

$molprobity $PDB_DIR > $OUT_PATH &

echo "Done!"
