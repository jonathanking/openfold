#!/bin/bash
#SBATCH --job-name=evalRN
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=28-00:00:00
#SBATCH --partition=dept_gpu
#SBATCH --output="out/%A_%6a.out"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jok120@pitt.edu
#SBATCH --nodelist=g019
#SBATCH --ntasks-per-node=32

############################
##       Description      ##
############################
EXPERIMENT_ID="COMPLETE-EVAL-AT-EPOCH35-SCNUNMIN-00"
# EXPERIMENT_ID="angletransformer-scnmin-00-AT-noOMM"
# EXPERIMENT_ID="angletransformer-scnmin-00-ResNet-noOMM"
# EXPERIMENT_ID="angletransformer-scnmin-00-ResNet-OMM"
echo "Running job ${SLURM_JOB_ID} with ${SLURM_NTASKS} workers on node ${SLURMD_NODENAME}."
echo "This will be called ${EXPERIMENT_ID}."
echo "Trying to finetune the angle transformer model."
echo "Fixing ntasks-per-node and cpus per task"

############################
##       Environment      ##
############################
# source scripts/activate_conda_env.sh
# module load cuda/11.5


############################
##     Array Job Exec.    ##
############################
OUTDIR=out/experiments/${EXPERIMENT_ID}/
mkdir -p ${OUTDIR}

# Remember, the aln dir is what determines the training set size! 
# The structure directory and caches can contain more structures than the aln dir.
# The train_structures_dir here contains only the scnmin structs as of 230530 (30k).

TRAIN_STRUCTURES_DIR=data/train_structs/scnunmin_structs0530/
ALIGNMENTS_DIR=data/alignments/scnmin_alignments0530/
TEMPLATE_STRUCTURES_DIR=data/template_structs/roda_pdbs_snapshotted_flattened_do_not_overwrite/
TRAIN_CACHE=data/caches/scnmin_structs0530_cache.json
TEMPLATE_CACHE=data/caches/mmcif_cache_rodasnapshot.json
VALIDATION_STRUCTURES_DIR=data/validation/cameo/20220116/data_dir
VALIDATION_ALIGNMENTS_DIR=data/validation/cameo/20220116/alignments
# VALIDATION_STRUCTURES_DIR=data/validation/cameo/20220116/data_dir
# VALIDATION_ALIGNMENTS_DIR=data/validation/cameo/20220116/alignments

# TRAIN_STRUCTURES_DIR=data/train_structs/scnunmin_structs0530/
# ALIGNMENTS_DIR=data/alignments/scnmin_alignments0530/  
# TEMPLATE_STRUCTURES_DIR=data/template_structs/roda_pdbs_snapshotted_flattened_do_not_overwrite/
# TRAIN_CACHE=data/caches/scnmin_structs0530_cache.json
# TEMPLATE_CACHE=data/caches/mmcif_cache_rodasnapshot.json
# VALIDATION_STRUCTURES_DIR=data/validation/cameo/20220116/data_dir
# VALIDATION_ALIGNMENTS_DIR=data/validation/cameo/20220116/alignments

CHECKPOINT=openfold/resources/openfold_params/finetuning_5.pt
AT_CHECKPOINT="/net/pulsar/home/koes/jok120/angletransformer/out/experiments/angletransformer_solo01/pw0rl5yo/at-epoch=35.ckpt"
# PREV_EXPERIMENT_OUTDIR="out/experiments/angletransformer-unmin-noomm-09d"
# CHECKPOINT=$(ls -td ${PREV_EXPERIMENT_OUTDIR}/finetune-openfold-02/*/checkpoints/* | head -n 1)
echo "Uses checkpoint $CHECKPOINT"

if [[ "$CHECKPOINT" == *"initial_training.pt"* ]] || [[ "$CHECKPOINT" == *"finetuning_"* ]]; then
    RESUME_MODEL_WEIGHTS_ONLY=True
else
    RESUME_MODEL_WEIGHTS_ONLY=False
fi

export CUDA_LAUNCH_BLOCKING=1


./train_openfold.py ${TRAIN_STRUCTURES_DIR} ${ALIGNMENTS_DIR} ${TEMPLATE_STRUCTURES_DIR} ${OUTDIR} 2021-10-10 \
    --train_chain_data_cache_path=${TRAIN_CACHE} \
    --template_release_dates_cache_path=${TEMPLATE_CACHE} \
    --val_data_dir=${VALIDATION_STRUCTURES_DIR} \
    --val_alignment_dir=${VALIDATION_ALIGNMENTS_DIR} \
    --obsolete_pdbs_file_path=data/obsolete_230310.dat \
    --resume_from_ckpt=${CHECKPOINT} \
    --deepspeed_config_path=deepspeed_config_jk03.json \
    --replace_sampler_ddp=True \
    --gpus=1 \
    --batch_size=1 \
    --checkpoint_every_epoch \
    --wandb \
    --wandb_project=finetune-openfold-02 \
    --wandb_entity=koes-group \
    --precision=bf16 \
    --resume_model_weights_only=True \
    --train_epoch_len=500 \
    --max_epochs=50000 \
    --debug \
    --add_struct_metrics \
    --openmm_weight=0.01 \
    --openmm_activation=None \
    --seed=1 \
    --num_workers=6 \
    --write_pdbs \
    --write_pdbs_every_n_steps=300 \
    --log_every_n_steps=1 \
    --config_preset=finetuning_sidechainnet \
    --violation_loss_weight=1 \
    --scheduler_base_lr=2e-5 \
    --scheduler_warmup_no_steps=1 \
    --scheduler_max_lr=2e-5 \
    --use_scn_pdb_names=True \
    --use_scn_pdb_names_val=True \
    --use_openmm=True \
    --use_alphafold_sampling=True \
    --experiment=${EXPERIMENT_ID}  \
    --wandb_tags="0831_AT" \
    --wandb_note="Trying to finetune the AT with hthe whole AF2." \
    --log_to_csv=True \
    --openmm_squashed_loss=False \
    --openmm_modified_sigmoid=5,1000000,300000,5 \
    --run_validate_first=False \
    --auto_slurm_resubmit=False \
    --use_openmm_warmup=True \
    --openmm_warmup_steps=1000 \
    --train_only_angle_predictor=True \
    --use_angle_transformer=True \
    --angle_transformer_layers=4 \
    --angle_transformer_heads=2 \
    --angle_transformer_hidden=2048 \
    --angle_transformer_dff=256 \
    --angle_transformer_dropout=0.15 \
    --angle_transformer_activation=gelu \
    --chi_weight=0.05 \
    --angle_loss_only=False \
    --angle_like_loss_only=True \
    --angle_transformer_checkpoint=${AT_CHECKPOINT} \
    --add_relu_to_omm_loss=True \
    --trainer_mode=validate-val \
    # --limit_val_batches=12
    # --accumulate_grad_batches=4 \



    

############################
##     Copy Files Back    ##
############################
echo "done."
exit 0
