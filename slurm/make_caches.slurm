#!/bin/bash
#SBATCH --job-name=cache10k
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=32
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
##          Vars          ##
############################
# For sidechainnet
# echo "Using sidechainnet data"
# TRAIN_STRUCTURES_DIR=data/train_structs/pdb_files_for_scnmin/
# TEMPLATE_STRUCTURES_DIR=data/template_structs/mmcif_files_for_scn/
# TRAIN_CACHE_OUT=data/caches/chain_data_cache_scn.json
# TEMPLATE_CACHE_OUT=data/caches/mmcif_cache_scn.json

# For typical openfold
echo "Using openfold data"
TRAIN_STRUCTURES_DIR=data/train_structs/random10k
TEMPLATE_STRUCTURES_DIR=data/template_structs/random10k
TRAIN_CACHE_OUT=data/caches/chain_data_cache_random10k.json
TEMPLATE_CACHE_OUT=data/caches/mmcif_cache_random10k.json

############################
##     Array Job Exec.    ##
############################
echo "Generating training data cache..."
python scripts/generate_chain_data_cache.py ${TRAIN_STRUCTURES_DIR} ${TRAIN_CACHE_OUT} --no_workers=${SLURM_NTASKS}	
echo "done."

echo "Generating template data cache..."
python scripts/generate_mmcif_cache.py ${TEMPLATE_STRUCTURES_DIR} ${TEMPLATE_CACHE_OUT} --no_workers=${SLURM_NTASKS}	
echo "done."

# Print out the generated file paths
echo "Generated files:"
echo "    data/caches/chain_data_cache.json"
echo "    data/caches/mmcif_cache.json"



############################
##     Copy Files Back    ##
############################

exit 0

