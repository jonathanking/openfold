#!/bin/bash
#SBATCH --job-name=hhsearch
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=64
#SBATCH --time=14-00:00:00
#SBATCH --partition=dept_cpu
#SBATCH --nodelist=n023
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jok120
#SBATCH --output="/net/pulsar/home/koes/jok120/openfold/out/%A_%6a.out"

############################
##       Description      ##
############################
echo "Running job ${SLURM_JOB_ID} with ${SLURM_NTASKS} workers on node ${SLURMD_NODENAME}."
echo "Constructing CAMEO test alignments. Output: data/test/cameo/20230103/alignments."

############################
##       Environment      ##
############################
source scripts/activate_conda_env.sh

############################
##     Setup Scr Drive    ##
############################
# Original locations
# INPUT_FASTA=/net/g019/scr/jok120/yoda/test_data/cameo/20230103/input.fasta
# MMSEQS_DB=/net/g019/scr/jok120/yoda/mmseqs_dbs/mmseqs_dbs
# OUT_ORIGINAL=/net/g019/scr/jok120/yoda/test_data/cameo/20230103
# PDB70=/net/g019/scr/jok120/yoda/alphafold_data/pdb70/pdb70

# Copy all files to scratch drive
# rsync -avh --progress $INPUT_FASTA /scr/jok120/test_alignment_generation/input.fasta
# rsync -avh --progress $MMSEQS_DB /scr/jok120/test_alignment_generation/mmseqs_dbs && rsync -avh --progress $PDB70 /scr/jok120/test_alignment_generation/pdb70
# rsync -avh --progress $PDB70 /scr/jok120/test_alignment_generation/pdb70

# Change locations to scratch drive
INPUT_FASTA=/scr/jok120/test_alignment_generation/input.fasta
MMSEQS_DB=/scr/jok120/test_alignment_generation/mmseqs_dbs
OUT=/scr/jok120/test_alignment_generation/alignments
PDB70=/scr/jok120/test_alignment_generation/pdb70/pdb70

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
# echo "Copying files back to original locations..."
# rsync -avh --progress $OUT $OUT_ORIGINAL
echo "done."
exit 0
