#!/bin/bash
#SBATCH --job-name=baseln
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=16
#SBATCH --time=14-00:00:00
#SBATCH --partition=dept_gpu
#SBATCH --nodelist=g019
#SBATCH --output="/net/pulsar/home/koes/jok120/openfold/out/%A_%6a.out"

############################
##       Description      ##
############################
echo "Running job ${SLURM_JOB_ID} with ${SLURM_NTASKS} workers on node ${SLURMD_NODENAME}."
echo "Baseline on AlphaFold data."

############################
##       Environment      ##
############################
source scripts/activate_conda_env.sh

############################
##     Array Job Exec.    ##
############################
OUTDIR=out/experiments/230307/
mkdir -p ${OUTDIR}

# TRAIN_STRUCTURES_DIR=data/train_structs/mmcif_files_for_roda/
# TEMPLATE_STRUCTURES_DIR=data/template_structs/mmcif_files_for_roda/
# ALIGNMENTS_DIR=data/alignments/scn_roda2/
# TRAIN_CACHE=data/caches/chain_data_cache.json
# TEMPLATE_CACHE=data/caches/mmcif_cache.json

TRAIN_STRUCTURES_DIR=data/train_structs/mmcif_files_for_roda/
ALIGNMENTS_DIR=data/alignments/roda/
TEMPLATE_STRUCTURES_DIR=data/template_structs/mmcif_files_for_roda/
TRAIN_CACHE=data/caches/chain_data_cache.json
TEMPLATE_CACHE=data/caches/mmcif_cache.json

./train_openfold.py ${TRAIN_STRUCTURES_DIR} ${ALIGNMENTS_DIR} ${TEMPLATE_STRUCTURES_DIR} ${OUTDIR} 2021-10-10 \
    --train_chain_data_cache_path=${TRAIN_CACHE} \
    --template_release_dates_cache_path=${TEMPLATE_CACHE} \
    --obsolete_pdbs_file_path=data/obsolete.dat \
    --resume_from_ckpt=openfold/resources/openfold_params/initial_training.pt \
    --gpus=1 \
    --checkpoint_every_epoch \
    --wandb \
    --wandb_project=finetune-openfold-02 \
    --wandb_entity=koes-group \
    --precision=32 \
    --resume_model_weights_only=True \
    --train_epoch_len=100 \
    --max_epochs=500 \
    --debug \
    --add_struct_metrics \
    --openmm_weight=1 \
    --openmm_activation=None \
    --seed=0 \
    --debug \
    --num_workers=4 \
    --write_pdbs \
    --write_pdbs_every_n_steps=100 \
    --log_every_n_steps=10 \
    --config_preset=finetuning_sidechainnet \
    --grad_clip_val=0.1 \
    --violation_loss_weight=0 \
    --scheduler_base_lr=0 \
    --scheduler_warmup_no_steps=100 \
    --scheduler_max_lr=1e-6 \
    --use_scn_pdb_names=False \
    --use_openmm=False \
    --experiment=ftfd2-00-baseline-nonscn  \
    --wandb_tags="baseline,nonscn" \
    --wandb_note="Finetuning baseline on nonscn data. Violoation loss off. LR step at 0. Max LR = 1e-6."
    

############################
##     Copy Files Back    ##
############################
echo "done."
exit 0
