#!/bin/bash
#SBATCH --job-name=combined-better3A
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --output="out/%A_%6a.out"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jok120@pitt.edu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=a100_multi
#SBATCH --nodes=2
#SBATCH --cluster=gpu
#SBATCH --time=6-00:00:00
#SBATCH --qos=long
# --exclude=gpu-n28,gpu-n32,gpu-n29,gpu-n41



############################
##       Description      ##
############################
EXPERIMENT_ID="toastyC0_repeat_noconvB-FullAF2"
# EXPERIMENT_ID="angletransformer-scnmin-00-AT-noOMM"
# EXPERIMENT_ID="angletransformer-scnmin-00-ResNet-noOMM"
# EXPERIMENT_ID="angletransformer-scnmin-00-ResNet-OMM"
echo "Running job ${SLURM_JOB_ID} with ${SLURM_NTASKS} workers on node ${SLURMD_NODENAME}."
echo "This will be called ${EXPERIMENT_ID}."
echo "Trying to finetune the angle transformer model."
AT_CHECKPOINT="/ihome/dkoes/jok120/chkpts_angletransformer/toastyC0_repeat.ckpt"
AT_LAYERS=42
AT_HEADS=1
AT_HIDDEN=64
AT_DFF=1024
AT_DROPOUT=0.018609913167811645
AT_ACTIVATION=gelu
AT_CONV_ENC=False
USE_AT=True
AIM_TAG="aim3A"
FORCE_LOAD_AT_WEIGHTS=True


############################
##       Environment      ##
############################
source scripts/activate_conda_env.sh
module load cuda/11.3.0
module load gcc/8.2.0


############################
##     Array Job Exec.    ##
############################
OUTDIR=out/experiments/${EXPERIMENT_ID}/
mkdir -p ${OUTDIR}

# Remember, the aln dir is what determines the training set size! 
# The structure directory and caches can contain more structures than the aln dir.
# The train_structures_dir here contains only the scnmin structs as of 230530 (30k).

TRAIN_STRUCTURES_DIR=data/train_structs/scnmin_structs0530/
ALIGNMENTS_DIR=data/alignments/scnmin_alignments0530/
TEMPLATE_STRUCTURES_DIR=data/template_structs/roda_pdbs_snapshotted_flattened_do_not_overwrite/
TRAIN_CACHE=data/caches/scnmin_structs0530_cache.json
TEMPLATE_CACHE=data/caches/mmcif_cache_rodasnapshot.json
VALIDATION_STRUCTURES_DIR=data/validation/cameo/20220116/minimized/data_dir
VALIDATION_ALIGNMENTS_DIR=data/validation/cameo/20220116/minimized/alignments

CHECKPOINT=openfold/resources/openfold_params/finetuning_5.pt
# PREV_EXPERIMENT_OUTDIR="out/experiments/angletransformer-unmin-noomm-09d"
# CHECKPOINT=$(ls -td ${PREV_EXPERIMENT_OUTDIR}/finetune-openfold-02/*/checkpoints/* | head -n 1)
echo "Uses checkpoint $CHECKPOINT"

if [[ "$CHECKPOINT" == *"initial_training.pt"* ]] || [[ "$CHECKPOINT" == *"finetuning_"* ]]; then
    RESUME_MODEL_WEIGHTS_ONLY=True
else
    RESUME_MODEL_WEIGHTS_ONLY=False
fi

export CUDA_LAUNCH_BLOCKING=1


srun -N 1 -n 1 -c 18 python train_openfold.py ${TRAIN_STRUCTURES_DIR} ${ALIGNMENTS_DIR} ${TEMPLATE_STRUCTURES_DIR} ${OUTDIR} 2021-10-10 \
    --train_chain_data_cache_path=${TRAIN_CACHE} \
    --template_release_dates_cache_path=${TEMPLATE_CACHE} \
    --val_data_dir=${VALIDATION_STRUCTURES_DIR} \
    --val_alignment_dir=${VALIDATION_ALIGNMENTS_DIR} \
    --obsolete_pdbs_file_path=data/obsolete_230310.dat \
    --resume_from_ckpt=${CHECKPOINT} \
    --deepspeed_config_path=deepspeed_config_jk03.json \
    --replace_sampler_ddp=True \
    --gpus=4 \
    --batch_size=1 \
    --accumulate_grad_batches=4 \
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
    --openmm_weight=0.01 \
    --openmm_activation=None \
    --seed=1 \
    --debug \
    --num_workers=4 \
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
    --use_openmm=False \
    --use_alphafold_sampling=True \
    --experiment=${EXPERIMENT_ID}  \
    --wandb_tags="0913_AT,0913_RN,0924_extend_good_runs,${AIM_TAG}" \
    --wandb_note="Trying to finetune the AT with the whole AF2." \
    --log_to_csv=False \
    --openmm_squashed_loss=False \
    --openmm_modified_sigmoid=5,1000000,300000,5 \
    --run_validate_first=False \
    --auto_slurm_resubmit=False \
    --use_openmm_warmup=True \
    --openmm_warmup_steps=1000 \
    --use_angle_transformer=${USE_AT} \
    --train_only_angle_predictor=True \
    --angle_transformer_layers=${AT_LAYERS} \
    --angle_transformer_heads=${AT_HEADS} \
    --angle_transformer_hidden=${AT_HIDDEN} \
    --angle_transformer_dff=${AT_DFF} \
    --angle_transformer_dropout=${AT_DROPOUT} \
    --angle_transformer_activation=${AT_ACTIVATION} \
    --chi_weight=0.5 \
    --angle_loss_only=False \
    --angle_like_loss_only=True \
    --angle_transformer_checkpoint=${AT_CHECKPOINT} \
    --add_relu_to_omm_loss=True \
    --angle_transformer_conv_encoder=${AT_CONV_ENC} \
    --force_load_angle_transformer_weights=${FORCE_LOAD_AT_WEIGHTS} \
    --angle_transformer_conv_encoder=True &

############################
##       Description      ##
############################
EXPERIMENT_ID="resnet_baseline_not_AF2_toastyRN1-FullAF2"
AT_LAYERS=2
AT_HEADS=1
AT_HIDDEN=128
AT_DFF=512
AT_DROPOUT=0.09362803755294724
AT_ACTIVATION=gelu
AT_CONV_ENC=False
USE_AT=False
AIM_TAG="aim3A"

FORCE_LOAD_AT_WEIGHTS=True


echo "Running job ${SLURM_JOB_ID} with ${SLURM_NTASKS} workers on node ${SLURMD_NODENAME}."
echo "This will be called ${EXPERIMENT_ID}."
echo "Trying to finetune the angle transformer model."


############################
##     Array Job Exec.    ##
############################
OUTDIR=out/experiments/${EXPERIMENT_ID}/
mkdir -p ${OUTDIR}

# Remember, the aln dir is what determines the training set size! 
# The structure directory and caches can contain more structures than the aln dir.
# The train_structures_dir here contains only the scnmin structs as of 230530 (30k).

TRAIN_STRUCTURES_DIR=data/train_structs/scnmin_structs0530/
ALIGNMENTS_DIR=data/alignments/scnmin_alignments0530/
TEMPLATE_STRUCTURES_DIR=data/template_structs/roda_pdbs_snapshotted_flattened_do_not_overwrite/
TRAIN_CACHE=data/caches/scnmin_structs0530_cache.json
TEMPLATE_CACHE=data/caches/mmcif_cache_rodasnapshot.json
VALIDATION_STRUCTURES_DIR=data/validation/cameo/20220116/minimized/data_dir
VALIDATION_ALIGNMENTS_DIR=data/validation/cameo/20220116/minimized/alignments

CHECKPOINT=openfold/resources/openfold_params/finetuning_5.pt
AT_CHECKPOINT="/ihome/dkoes/jok120/chkpts_angletransformer/restnet_baselinet_not_AF2.ckpt"
# PREV_EXPERIMENT_OUTDIR="out/experiments/angletransformer-unmin-noomm-09d"
# CHECKPOINT=$(ls -td ${PREV_EXPERIMENT_OUTDIR}/finetune-openfold-02/*/checkpoints/* | head -n 1)
echo "Uses checkpoint $CHECKPOINT"

if [[ "$CHECKPOINT" == *"initial_training.pt"* ]] || [[ "$CHECKPOINT" == *"finetuning_"* ]]; then
    RESUME_MODEL_WEIGHTS_ONLY=True
else
    RESUME_MODEL_WEIGHTS_ONLY=False
fi

export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"


srun -N 1 -n 1 -c 18 python train_openfold.py ${TRAIN_STRUCTURES_DIR} ${ALIGNMENTS_DIR} ${TEMPLATE_STRUCTURES_DIR} ${OUTDIR} 2021-10-10 \
    --train_chain_data_cache_path=${TRAIN_CACHE} \
    --template_release_dates_cache_path=${TEMPLATE_CACHE} \
    --val_data_dir=${VALIDATION_STRUCTURES_DIR} \
    --val_alignment_dir=${VALIDATION_ALIGNMENTS_DIR} \
    --obsolete_pdbs_file_path=data/obsolete_230310.dat \
    --resume_from_ckpt=${CHECKPOINT} \
    --deepspeed_config_path=deepspeed_config_jk03.json \
    --replace_sampler_ddp=True \
    --gpus=4 \
    --batch_size=1 \
    --accumulate_grad_batches=4 \
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
    --openmm_weight=0.01 \
    --openmm_activation=None \
    --seed=1 \
    --debug \
    --num_workers=4 \
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
    --use_openmm=False \
    --use_alphafold_sampling=True \
    --experiment=${EXPERIMENT_ID}  \
    --wandb_tags="0913_AT,0913_RN,0924_extend_good_runs,${AIM_TAG}" \
    --wandb_note="Trying to finetune the AT with the whole AF2." \
    --log_to_csv=False \
    --openmm_squashed_loss=False \
    --openmm_modified_sigmoid=5,1000000,300000,5 \
    --run_validate_first=False \
    --auto_slurm_resubmit=False \
    --use_openmm_warmup=True \
    --openmm_warmup_steps=1000 \
    --use_angle_transformer=${USE_AT} \
    --train_only_angle_predictor=True \
    --angle_transformer_layers=${AT_LAYERS} \
    --angle_transformer_heads=${AT_HEADS} \
    --angle_transformer_hidden=${AT_HIDDEN} \
    --angle_transformer_dff=${AT_DFF} \
    --angle_transformer_dropout=${AT_DROPOUT} \
    --angle_transformer_activation=${AT_ACTIVATION} \
    --chi_weight=0.5 \
    --angle_loss_only=False \
    --angle_like_loss_only=True \
    --angle_transformer_checkpoint=${AT_CHECKPOINT} \
    --angle_transformer_conv_encoder=${AT_CONV_ENC} \
    --force_load_angle_transformer_weights=${FORCE_LOAD_AT_WEIGHTS} \
    --add_relu_to_omm_loss=True \
    --angle_transformer_conv_encoder=True &

wait




    

############################
##     Copy Files Back    ##
############################
echo "done."
exit 0
