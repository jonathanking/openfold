#!/bin/bash
#SBATCH --job-name=RN-OMM-long-40g-non323941
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --output="out/%A_%6a.out"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jok120@pitt.edu
#SBATCH --ntasks-per-node=18
#SBATCH --partition=a100
#SBATCH --cluster=gpu
#SBATCH --time=3-00:00:00
#SBATCH --qos=normal
#SBATCH --exclude=gpu-n32,gpu-n29,gpu-n41



############################
##       Description      ##
############################
EXPERIMENT_ID="angletransformer-scnmin-00-RN-OMM-toasty-long-a100only"
# EXPERIMENT_ID="angletransformer-scnmin-00-AT-noOMM"
# EXPERIMENT_ID="angletransformer-scnmin-00-ResNet-noOMM"
# EXPERIMENT_ID="angletransformer-scnmin-00-ResNet-OMM"
echo "Running job ${SLURM_JOB_ID} with ${SLURM_NTASKS} workers on node ${SLURMD_NODENAME}."
echo "This will be called ${EXPERIMENT_ID}."
echo "Trying to finetune the angle transformer model."

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
AT_CHECKPOINT="/ihome/dkoes/jok120/chkpts_angletransformer/best_resnet.ckpt"
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
    --use_openmm=True \
    --use_alphafold_sampling=True \
    --experiment=${EXPERIMENT_ID}  \
    --wandb_tags="0913_AT,0913_RN" \
    --wandb_note="Trying to finetune the AT with the whole AF2." \
    --log_to_csv=False \
    --openmm_squashed_loss=False \
    --openmm_modified_sigmoid=5,1000000,300000,5 \
    --run_validate_first=False \
    --auto_slurm_resubmit=False \
    --use_openmm_warmup=True \
    --openmm_warmup_steps=1000 \
    --use_angle_transformer=False \
    --force_load_angle_transformer_weights=True \
    --train_only_angle_predictor=True \
    --angle_transformer_layers=42 \
    --angle_transformer_heads=1 \
    --angle_transformer_hidden=64 \
    --angle_transformer_dff=1024 \
    --angle_transformer_dropout=0.018609913167811645 \
    --angle_transformer_activation=gelu \
    --chi_weight=13.012551860064766 \
    --angle_loss_only=False \
    --angle_like_loss_only=True \
    --angle_transformer_checkpoint=${AT_CHECKPOINT} \
    --add_relu_to_omm_loss=True \
    --angle_transformer_conv_encoder=False \


    

############################
##     Copy Files Back    ##
############################
echo "done."
exit 0
