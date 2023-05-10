#!/bin/bash

molprobity=/net/pulsar/home/koes/jok120/build/phenix-install/phenix-1.20.1-4487/molprobity/cmdline/oneline-analysis

# Analyze the labels
echo "Analyzing labels..."
$molprobity ~/openfold/out/evaluation/230507/initial_training_eval_s0/pdbs/true > ~/tmp/molpro/initial_training_eval_s0.out &

# Analyze the predictions for each directory under openfold/out/evaluation/230507 excluding initial_training_eval_s0
for dir in `ls ~/openfold/out/evaluation/230507/ | grep -v initial_training_eval_s0`
do
    echo "Analyzing ${dir}..."
    $molprobity ~/openfold/out/evaluation/230507/$dir/pdbs/pred > ~/tmp/molpro/${dir}.out &
done
