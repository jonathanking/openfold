#!/bin/bash
#SBATCH --job-name=AngTran
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=6-00:00:00
#SBATCH --partition=a100_nvlink
#SBATCH --qos=long
#SBATCH --cluster=gpu
#SBATCH --output="out/%A_%6a.out"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jok120@pitt.edu
#SBATCH --ntasks-per-node=64
#SBATCH --constraint=80g


############################
##       Description      ##
############################
EXPERIMENT_ID="angletransformer-unmin-noomm-01"
echo "Running job ${SLURM_JOB_ID} with ${SLURM_NTASKS} workers on node ${SLURMD_NODENAME}."
echo "Starts training from 'finetuning_5.pt'. Resume previous job."
echo "This will be called ${EXPERIMENT_ID}."
echo "Trying to finetune the angle transformer model. Unminimized structures. No OpenMM."

############################
##       Environment      ##
############################
source scripts/activate_conda_env.sh
module load cuda/11.3.0
module load gcc/8.2.0
# Trying to debug NCCL hanging
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export PL_TORCH_DISTRIBUTED_BACKEND=gloo
# export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL


############################
##     Array Job Exec.    ##
############################
OUTDIR=out/experiments/${EXPERIMENT_ID}/
mkdir -p ${OUTDIR}

# Remember, the aln dir is what determines the training set size! 
# The structure directory and caches can contain more structures than the aln dir.
# The train_structures_dir here contains only the scnmin structs as of 230530 (30k).

TRAIN_STRUCTURES_DIR=data/train_structs/roda_pdbs_snapshotted_flattened_do_not_overwrite/
ALIGNMENTS_DIR=data/alignments/random10k_links/
TEMPLATE_STRUCTURES_DIR=data/template_structs/roda_pdbs_snapshotted_flattened_do_not_overwrite/
TRAIN_CACHE=data/caches/chain_data_cache_rodasnapshot_clustered.json
TEMPLATE_CACHE=data/caches/mmcif_cache_rodasnapshot.json
VALIDATION_STRUCTURES_DIR=data/validation/cameo/20220116/data_dir
VALIDATION_ALIGNMENTS_DIR=data/validation/cameo/20220116/alignments

# CHECKPOINT=openfold/resources/openfold_params/finetuning_5.pt
CHECKPOINT=/zfs1/dkoes/jok120/openfold_out3/experiments/angletransformer-unmin-noomm-01/finetune-openfold-02/1myrv5zj/checkpoints/13-321.ckpt
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
    --accumulate_grad_batches=32 \
    --checkpoint_every_epoch \
    --wandb \
    --wandb_project=finetune-openfold-02 \
    --wandb_entity=koes-group \
    --precision=bf16 \
    --resume_model_weights_only=${RESUME_MODEL_WEIGHTS_ONLY} \
    --train_epoch_len=9000 \
    --max_epochs=500 \
    --debug \
    --add_struct_metrics \
    --openmm_weight=0.01 \
    --openmm_activation=None \
    --seed=1 \
    --debug \
    --num_workers=7 \
    --write_pdbs \
    --write_pdbs_every_n_steps=300 \
    --log_every_n_steps=1 \
    --config_preset=finetuning_sidechainnet \
    --violation_loss_weight=1 \
    --scheduler_base_lr=0 \
    --scheduler_warmup_no_steps=1 \
    --scheduler_max_lr=5e-4 \
    --use_scn_pdb_names=False \
    --use_scn_pdb_names_val=False \
    --use_openmm=False \
    --use_alphafold_sampling=True \
    --experiment=${EXPERIMENT_ID}  \
    --wandb_tags="atf,encoder-only" \
    --wandb_note="A new noomm baseline using finetuning_5.pt. Resume." \
    --log_to_csv=False \
    --openmm_squashed_loss=False \
    --openmm_modified_sigmoid=5,1000000,300000,5 \
    --run_validate_first=True \
    --auto_slurm_resubmit=False \
    --use_openmm_warmup=True \
    --openmm_warmup_steps=1000 \
    --use_angle_transformer=True \
    --train_only_angle_predictor=True


    

############################
##     Copy Files Back    ##
############################
echo "done."
exit 0
