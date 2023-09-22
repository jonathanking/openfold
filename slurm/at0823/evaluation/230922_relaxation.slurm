#!/bin/bash
#SBATCH --job-name=relax
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH -c 16
#SBATCH --time=0-02:00:00
#SBATCH --partition=dept_cpu
#SBATCH --output="/net/pulsar/home/koes/jok120/openfold/out/%A_%6a.out"

############################
##       Description      ##
############################
echo "Running job ${SLURM_JOB_ID} with ${SLURM_NTASKS} workers on node ${SLURMD_NODENAME}."

############################
##       Environment      ##
############################
source scripts/activate_conda_env.sh

############################
##     Array Job Exec.    ##
############################

# Select the nth line from a command file, where n is the array task ID
cmd=$(sed -n ${SLURM_ARRAY_TASK_ID}p /net/pulsar/home/koes/jok120/openfold/jk_research/evaluations/aim3c/relaxation_cmds.txt)

# Execute the command
eval $cmd


############################
##     Copy Files Back    ##
############################
echo "done."
exit 0
