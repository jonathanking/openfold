#!/bin/bash
#SBATCH --job-name=openfold
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=a100
#SBATCH --qos=short
#SBATCH --cluster=gpu
#SBATCH --output="/ihome/dkoes/jok120/openfold/out/%A_%6a.out"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jok120@pitt.edu


############################
##       Description      ##
############################
echo "Running job ${SLURM_JOB_ID} with ${SLURM_NTASKS} workers on node ${SLURMD_NODENAME}."
echo "Running inference on validation set."

############################
##       Environment      ##
############################
source scripts/activate_conda_env.sh
module load cuda/11.3.0
module load gcc/8.2.0


############################
##     Array Job Exec.    ##
############################
OUTDIR=out/experiments/230319_infer/03allvalid/
mkdir -p ${OUTDIR}

TRAIN_STRUCTURES_DIR=data/train_structs/scnmin_structs/
ALIGNMENTS_DIR=data/alignments/scnmin_alignments/
TEMPLATE_STRUCTURES_DIR=data/template_structs/roda_pdbs_snapshotted_flattened_do_not_overwrite/
TRAIN_CACHE=data/caches/chain_data_cache_rodasnapshot_clustered.json
TEMPLATE_CACHE=data/caches/mmcif_cache_rodasnapshot.json
VALIDATION_STRUCTURES_DIR=data/validation/cameo/20220116/data_dir
VALIDATION_ALIGNMENTS_DIR=data/validation/cameo/20220116/alignments

FASTA_DIR=data/validation/cameo/20220116/fasta_dir

python3 run_pretrained_openfold.py \
    $FASTA_DIR \
    $TEMPLATE_STRUCTURES_DIR \
    --use_precomputed_alignments $VALIDATION_ALIGNMENTS_DIR \
    --output_dir $OUTDIR \
    --model_device "cuda" \
    --config_preset "finetuning_sidechainnet_inference" \
    --kalign_binary_path lib/conda/envs/openfold_venv/bin/kalign \
    --openfold_checkpoint_path openfold/resources/openfold_params/initial_training.pt \
    --obsolete_pdbs_path=data/obsolete_230310.dat \
    --save_outputs=True \
    --data_random_seed=1 \
    --trace_model=False \
    --skip_relaxation=False \
    --use_meta_fasta_file=data/validation/cameo/20220116/input.fasta

    

############################
##     Copy Files Back    ##
############################
echo "done."
exit 0
