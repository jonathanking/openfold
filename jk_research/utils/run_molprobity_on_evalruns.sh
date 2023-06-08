#!/bin/bash

# Parse the command line arguments, the first being the evaluation top level dir
if [ $# -ne 1 ]; then
    echo "Usage: $0 <evaluation_top_level_dir>, i.e. ~/openfold/out/evaluation/230507"
    exit 1
fi

# Set the evaluation top level dir
eval_top_level_dir=$1  # i.e. ~/openfold/out/evaluation/230507


molprobity=/net/pulsar/home/koes/jok120/build/phenix-install/phenix-1.20.1-4487/molprobity/cmdline/oneline-analysis

# Analyze the predictions for each directory under openfold/out/evaluation/230507 excluding initial_training_eval_s0
for dir in `ls $eval_top_level_dir/ | grep -v initial_training_eval_s0`
do
    echo "Analyzing ${dir}..."
    $molprobity $eval_top_level_dir/$dir/pdbs/val/pred > ~/tmp/molpro/${dir}_val.out &
    $molprobity $eval_top_level_dir/$dir/pdbs/test/pred > ~/tmp/molpro/${dir}_test.out &
done
