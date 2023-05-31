#!/bin/bash
#SBATCH --job-name=cache
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
##          Vars          ##
############################
# For sidechainnet
# echo "Using sidechainnet data"
# TRAIN_STRUCTURES_DIR=data/train_structs/pdb_files_for_scnmin/
# TEMPLATE_STRUCTURES_DIR=data/template_structs/mmcif_files_for_scn/
# TRAIN_CACHE_OUT=data/caches/chain_data_cache_scn.json
# TEMPLATE_CACHE_OUT=data/caches/mmcif_cache_scn.json

# # For typical openfold
# echo "Using openfold data"
# TRAIN_STRUCTURES_DIR=data/train_structs/roda_pdbs_snapshotted_flattened_do_not_overwrite
# TEMPLATE_STRUCTURES_DIR=data/template_structs/roda_pdbs_snapshotted_flattened_do_not_overwrite
# TRAIN_CACHE_OUT=data/caches/chain_data_cache_rodasnapshot_clustered.json
# TEMPLATE_CACHE_OUT=data/caches/mmcif_cache_rodasnapshot.json

# For combined_scnmin_alignments0510_random10k
# echo "Using combined_scnmin_alignments0510_random10k data"
# TRAIN_STRUCTURES_DIR=data/train_structs/combined_scnmin_structs0510_random10k
# TEMPLATE_STRUCTURES_DIR=data/template_structs/roda_pdbs_snapshotted_flattened_do_not_overwrite
# TRAIN_CACHE_OUT=data/caches/chain_data_cache_combined_scnmin_structs0510_random10k_clustered.json
# TEMPLATE_CACHE_OUT=data/caches/mmcif_cache_rodasnapshot.json

# For scnmin_alignments0530 data
echo "Using scnmin_alignments0530 data"
TRAIN_STRUCTURES_DIR=data/train_structs/scnmin_structs0530
# TEMPLATE_STRUCTURES_DIR=data/template_structs/roda_pdbs_snapshotted_flattened_do_not_overwrite
TRAIN_CACHE_OUT=data/caches/scnmin_structs0530_cache.json
# TEMPLATE_CACHE_OUT=data/caches/mmcif_cache_rodasnapshot.json

############################
##     Array Job Exec.    ##
############################
# echo "Generating training data cache..."
# python scripts/generate_chain_data_cache.py ${TRAIN_STRUCTURES_DIR} ${TRAIN_CACHE_OUT} \
#     --cluster_file=clusters-by-entity-40.txt --no_workers=${SLURM_NTASKS}	
# echo "done."

echo "Generating unclustered training data cache..."
python scripts/generate_chain_data_cache.py ${TRAIN_STRUCTURES_DIR} ${TRAIN_CACHE_OUT} \
    --no_workers=${SLURM_NTASKS}	
echo "done."


# echo "Generating template data cache..."
# python scripts/generate_mmcif_cache.py ${TEMPLATE_STRUCTURES_DIR} ${TEMPLATE_CACHE_OUT} --no_workers=${SLURM_NTASKS}	
# echo "done."

# Print out the generated file paths
echo "Generated files:"
echo "    ${TRAIN_CACHE_OUT}"
# echo "    ${TEMPLATE_CACHE_OUT}"



############################
##     Copy Files Back    ##
############################

exit 0

