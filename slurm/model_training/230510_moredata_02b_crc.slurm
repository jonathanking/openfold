#!/bin/bash
#SBATCH --job-name=moredata
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=6-00:00:00
#SBATCH --partition=a100_nvlink
#SBATCH --constraint=80g
#SBATCH --qos=long
#SBATCH --cluster=gpu
#SBATCH --output="/ihome/dkoes/jok120/openfold/out/%A_%6a.out"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jok120@pitt.edu
#SBATCH --ntasks-per-node=64



############################
##       Description      ##
############################
EXPERIMENT_ID=moredata_scnmin_omm_02b
echo "Running job ${SLURM_JOB_ID} with ${SLURM_NTASKS} workers on node ${SLURMD_NODENAME}."
echo "The biggest change here is using a larger training set."
echo "Warmup OpenMM over 1000 steps. 4 gpus x 1 nodes."
echo "Uses minimized val!"
echo "Starts training from 'initial_training.pt'."
echo "This will be called ${EXPERIMENT_ID}."
echo "OMM loss is scaled by 0.01."
echo "Skips warmup for standard lr."
echo "Seed 1"

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

TRAIN_STRUCTURES_DIR=data/train_structs/scnmin_structs0510/
ALIGNMENTS_DIR=data/alignments/scnmin_alignments0510/
# TRAIN_STRUCTURES_DIR=data/train_structs/scnmin_structs/
# ALIGNMENTS_DIR=data/alignments/scnmin_alignments/
TEMPLATE_STRUCTURES_DIR=data/template_structs/roda_pdbs_snapshotted_flattened_do_not_overwrite/
TRAIN_CACHE=data/caches/chain_data_cache_rodasnapshot_clustered.json
TEMPLATE_CACHE=data/caches/mmcif_cache_rodasnapshot.json
VALIDATION_STRUCTURES_DIR=data/validation/cameo/20220116/minimized/data_dir
VALIDATION_ALIGNMENTS_DIR=data/validation/cameo/20220116/minimized/alignments

CHECKPOINT=/ihome/dkoes/jok120/openfold/out/experiments/moredata_scnmin_omm_02/finetune-openfold-02/hgcdeupr/checkpoints/50-356.ckpt
echo "Uses checkpoint $CHECKPOINT"

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
    --resume_model_weights_only=False \
    --train_epoch_len=1000 \
    --max_epochs=500 \
    --debug \
    --add_struct_metrics \
    --openmm_weight=0.01 \
    --openmm_activation=None \
    --seed=1 \
    --num_workers=4 \
    --write_pdbs \
    --write_pdbs_every_n_steps=200 \
    --log_every_n_steps=1 \
    --config_preset=finetuning_sidechainnet \
    --violation_loss_weight=1 \
    --scheduler_base_lr=0 \
    --scheduler_warmup_no_steps=1 \
    --scheduler_max_lr=5e-4 \
    --use_scn_pdb_names=True \
    --use_scn_pdb_names_val=True \
    --use_openmm=True \
    --use_alphafold_sampling=True \
    --experiment=${EXPERIMENT_ID}  \
    --wandb_tags="omm,scnmin,omm-warmup,valmin,more-data" \
    --wandb_note="More training data. Start from initial_training.pt. Valmin. OMM warmup over 1000 steps. Skips standard warmup. Early stopping." \
    --log_to_csv=False \
    --openmm_squashed_loss=False \
    --openmm_modified_sigmoid=5,1000000,300000,5 \
    --run_validate_first=False \
    --auto_slurm_resubmit=False \
    --use_openmm_warmup=True \
    --openmm_warmup_steps=1000 \
    --early_stopping=True \
    --min_delta=0 \
    --patience=20



    

############################
##     Copy Files Back    ##
############################
echo "done."
exit 0
