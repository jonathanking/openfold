#!/bin/bash
#SBATCH --job-name=mmseq
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=32
#SBATCH --time=14-00:00:00
#SBATCH --partition=dept_gpu
#SBATCH --nodelist=g019
#SBATCH --output="/net/pulsar/home/koes/jok120/openfold/out/%A_%6a.out"

############################
##       Description      ##
############################
echo "Running job ${SLURM_JOB_ID} with ${SLURM_NTASKS} workers on node ${SLURMD_NODENAME}."
echo "Constructing CAMEO validation alignments. Output: data/validation/cameo/20220116/alignments."

############################
##       Environment      ##
############################
source scripts/activate_conda_env.sh

############################
##     Array Job Exec.    ##
############################

python scripts/precompute_alignments_mmseqs.py data/validation/cameo/20220116/input.fasta data/mmseqs_dbs/ \
    uniref30_2103_db \
    data/validation/cameo/20220116/alignments \
    /net/pulsar/home/koes/jok120/build/MMseqs2/build/bin/mmseqs \
    --hhsearch_binary_path=/net/pulsar/home/koes/jok120/openfold/lib/conda/envs/openfold_venv/bin/hhsearch \
    --env_db=colabfold_envdb_202108_db \
    --pdb70=/net/g019/scr/jok120/yoda/alphafold_data/pdb70 \
    --fasta_chunk_size=32
    

############################
##     Copy Files Back    ##
############################
echo "done."
exit 0
