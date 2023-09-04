#!/bin/bash
#SBATCH --job-name=DL
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --time=28-00:00:00
#SBATCH --partition=dept_cpu
#SBATCH --output="out/%A_%6a.out"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jok120@pitt.edu
#SBATCH --ntasks-per-node=4

cd ~/openfold
source scripts/activate_conda_env.sh

python ~/openfold/jk_research/test_set.py > ~/openfold/jk_research/test_set_cmds.txt

while read p; do
    echo "$p"
    eval "$p"
    # Get the last part of the command which is a path
    p=${p##* }fasta_dir/
    # Remove the '~' from the first character in p
    p=${p#?}
    # Add homedir
    targetdir=${HOME}${p}
    # Replace the '~' with the full path to home
    # echo $( ls -1 ${targetdir}*.fasta | wc -l )
    # rm -rf ${p##* }
done < ~/openfold/jk_research/test_set_cmds.txt
