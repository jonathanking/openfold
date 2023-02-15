#!/bin/bash
#SBATCH --job-name=cachegen
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=64
#SBATCH --time=14-00:00:00
#SBATCH --partition=dept_cpu
#SBATCH --exclude=g122,g001
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
echo "Generating training data cache..."
python scripts/generate_chain_data_cache.py data/train_structs/pdb_files_for_scnmin/ data/caches/chain_data_cache.json --no_workers=${SLURM_NTASKS}	
echo "done."

echo "Generating template data cache..."
python scripts/generate_mmcif_cache.py data/template_structs/mmcif_files_for_scn/ data/caches/mmcif_cache.json --no_workers=${SLURM_NTASKS}	
echo "done."

# Print out the generated file paths
echo "Generated files:"
echo "    data/caches/chain_data_cache.json"
echo "    data/caches/mmcif_cache.json"



############################
##     Copy Files Back    ##
############################

exit 0

