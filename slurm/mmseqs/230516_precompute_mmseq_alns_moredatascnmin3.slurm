#!/bin/bash
#SBATCH --job-name=MSA3
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=64
#SBATCH --time=28-00:00:00
#SBATCH --partition=dept_cpu
#SBATCH --nodelist=n023
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jok120
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
##     Setup Scr Drive    ##
############################
SPLIT="3"
INPUT_FASTA=/scr/jok120/scnmin_alignment_generation_0516/input${SPLIT}.fasta
OUT=/scr/jok120/scnmin_alignment_generation_0516/alignments${SPLIT}
MMSEQS_DB=/scr/jok120/test_alignment_generation/mmseqs_dbs
PDB70=/scr/jok120/test_alignment_generation/pdb70/pdb70

echo "Constructing alignments for scnmin proteins."
echo "Input: ${INPUT_FASTA}"
echo "Output: ${OUT}"


############################
##     Array Job Exec.    ##
############################

python scripts/precompute_alignments_mmseqs.py $INPUT_FASTA $MMSEQS_DB \
    uniref30_2103_db \
    $OUT \
    /net/pulsar/home/koes/jok120/build/MMseqs2/build/bin/mmseqs \
    --hhsearch_binary_path=/net/pulsar/home/koes/jok120/openfold/lib/conda/envs/openfold_venv/bin/hhsearch \
    --env_db=colabfold_envdb_202108_db \
    --pdb70=$PDB70 \
    --fasta_chunk_size=64
    

############################
##     Copy Files Back    ##
############################
echo "done."
exit 0
