#!/bin/bash
#SBATCH --job-name=openfold
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=64
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
module load cuda


############################
##     Array Job Exec.    ##
############################
OUTDIR=out/experiments/230315/
mkdir -p ${OUTDIR}

TRAIN_STRUCTURES_DIR=data/train_structs/roda_pdbs_snapshotted_flattened_do_not_overwrite/
ALIGNMENTS_DIR=data/alignments/random10k_links/
TEMPLATE_STRUCTURES_DIR=data/template_structs/roda_pdbs_snapshotted_flattened_do_not_overwrite/
TRAIN_CACHE=data/caches/chain_data_cache_rodasnapshot_clustered.json
TEMPLATE_CACHE=data/caches/mmcif_cache_rodasnapshot.json
VALIDATION_STRUCTURES_DIR=data/validation/cameo/20220116/data_dir
VALIDATION_ALIGNMENTS_DIR=data/validation/cameo/20220116/alignments

./train_openfold.py ${TRAIN_STRUCTURES_DIR} ${ALIGNMENTS_DIR} ${TEMPLATE_STRUCTURES_DIR} ${OUTDIR} 2021-10-10 \
    --train_chain_data_cache_path=${TRAIN_CACHE} \
    --template_release_dates_cache_path=${TEMPLATE_CACHE} \
    --val_data_dir=${VALIDATION_STRUCTURES_DIR} \
    --val_alignment_dir=${VALIDATION_ALIGNMENTS_DIR} \
    --obsolete_pdbs_file_path=data/obsolete_230310.dat \
    --resume_from_ckpt=openfold/resources/openfold_params/initial_training.pt \
    --deepspeed_config_path=deepspeed_config_jk03.json \
    --gpus=4 \
    --batch_size=1 \
    --accumulate_grad_batches=32 \
    --checkpoint_every_epoch \
    --wandb \
    --wandb_project=finetune-openfold-02 \
    --wandb_entity=koes-group \
    --precision=bf16 \
    --resume_model_weights_only=True \
    --train_epoch_len=1000 \
    --max_epochs=500 \
    --debug \
    --add_struct_metrics \
    --openmm_weight=1 \
    --openmm_activation=None \
    --seed=0 \
    --debug \
    --num_workers=8 \
    --write_pdbs \
    --write_pdbs_every_n_steps=100 \
    --log_every_n_steps=1 \
    --config_preset=finetuning_sidechainnet \
    --violation_loss_weight=1 \
    --scheduler_base_lr=0 \
    --scheduler_warmup_no_steps=500 \
    --scheduler_max_lr=5e-4 \
    --use_scn_pdb_names=False \
    --use_openmm=False \
    --use_alphafold_sampling=True \
    --experiment=baseline  \
    --wandb_tags="baseline,nonscn,improved-grad,bf16" \
    --wandb_note="Baseline. MULTI GPU. Violation loss on. DeepSpeed. Trying to mimic OpenFold. Gradient accumulation effective size=32. Added val data." \
    --log_to_csv=False \
    --num_sanity_val_steps=0 \
    --checkpoint_every_n_train_steps=500

    

############################
##     Copy Files Back    ##
############################
echo "done."
exit 0
