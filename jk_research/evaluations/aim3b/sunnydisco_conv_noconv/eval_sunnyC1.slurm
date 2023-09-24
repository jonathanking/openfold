#!/bin/bash
#SBATCH --job-name=eval_sunnyC1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=dept_gpu
#SBATCH --time=6-00:00:00
#SBATCH --output="/net/pulsar/home/koes/jok120/openfold/out/%A_%6a.out"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jok120@pitt.edu
#SBATCH --nodelist=g019

###########################
## UPDATE ME!! UPDATE ME!##
###########################
EXPERIMENT_ID="eval_sunny-disco-172-g4gpu-extended3-conv-4gpu"
AT_CHECKPOINT="/net/pulsar/home/koes/jok120/angletransformer/out/experiments/angletransformer_solo01/80hdru8i/checkpoints/"
AT_CHECKPOINT="${AT_CHECKPOINT}$(ls -t ${AT_CHECKPOINT} | head -1)"
echo "Using checkpoint ${AT_CHECKPOINT}"
AT_LAYERS=4
AT_HEADS=2
AT_HIDDEN=2048
AT_DFF=256
AT_DROPOUT=0.15
AT_ACTIVATION=gelu
AT_CONV_ENC=True
USE_AT=True
AIM_TAG="aim3B"


############################
##       Description      ##
############################
echo "Running job ${SLURM_JOB_ID} with ${SLURM_NTASKS} workers on node ${SLURMD_NODENAME}."
echo "This will be called ${EXPERIMENT_ID}."


############################
##       Environment      ##
############################
source scripts/activate_conda_env.sh
module load cuda/11.5



############################
##     Array Job Exec.    ##
############################
OUTDIR=out/evaluations/${AIM_TAG}/${EXPERIMENT_ID}/
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
    --accumulate_grad_batches=1 \
    --checkpoint_every_epoch \
    --wandb \
    --wandb_project=finetune-openfold-02 \
    --wandb_entity=koes-group \
    --precision=bf16 \
    --resume_model_weights_only=${RESUME_MODEL_WEIGHTS_ONLY} \
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
    --write_pdbs_every_n_steps=1 \
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
    --wandb_tags="eval,evalAT,${AIM_TAG}" \
    --wandb_note="Evaluating ${EXPERIMENT_ID}" \
    --log_to_csv=True \
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
    --angle_like_loss_only=False \
    --angle_transformer_checkpoint=${AT_CHECKPOINT} \
    --add_relu_to_omm_loss=True \
    --angle_transformer_conv_encoder=${AT_CONV_ENC} \
    --force_load_angle_transformer_weights=True \
    --trainer_mode=validate-val-test \
    --test_data_dir=data/test/cameo/20230103/minimized/data_dir \
    --test_alignment_dir=data/test/cameo/20230103/minimized/alignments



############################
##     Copy Files Back    ##
############################
echo "done."
exit 0
