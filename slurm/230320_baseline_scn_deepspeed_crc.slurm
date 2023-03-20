#!/bin/bash
#SBATCH --job-name=ofbaseline
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=64
#SBATCH --time=1-00:00:00
#SBATCH --partition=a100_multi
#SBATCH --qos=short
#SBATCH --cluster=gpu
#SBATCH --output="/ihome/dkoes/jok120/openfold/out/%A_%6a.out"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jok120@pitt.edu
#SBATCH --signal=SIGUSR1@90

############################
##       Description      ##
############################
echo "Running job ${SLURM_JOB_ID} with ${SLURM_NTASKS} workers on node ${SLURMD_NODENAME}."
echo "Baseline on AlphaFold data, but better checkpointing, etc. Will try auto-resubmit."

############################
##       Environment      ##
############################
source scripts/activate_conda_env.sh
module load cuda/11.3.0
module load gcc/8.2.0


############################
##     Array Job Exec.    ##
############################
OUTDIR=out/experiments/230320/
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
    --gpus=16 \
    --num_nodes=4 \
    --batch_size=1 \
    --accumulate_grad_batches=8 \
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
    --openmm_weight=0 \
    --openmm_activation=None \
    --seed=0 \
    --debug \
    --num_workers=4 \
    --write_pdbs \
    --write_pdbs_every_n_steps=200 \
    --log_every_n_steps=1 \
    --config_preset=finetuning_sidechainnet \
    --violation_loss_weight=1 \
    --scheduler_base_lr=0 \
    --scheduler_warmup_no_steps=500 \
    --scheduler_max_lr=5e-4 \
    --use_scn_pdb_names=False \
    --use_openmm=True \
    --use_alphafold_sampling=True \
    --experiment=baseline-01  \
    --wandb_tags="baseline,non-scn" \
    --wandb_note="Baseline with better checkpointing and openmm measurement." \
    --log_to_csv=False \
    --openmm_squashed_loss=False \
    --openmm_modified_sigmoid=5,1000000,300000,5 \
    --run_validate_first=True \
    --auto_slurm_resubmit=True


    

############################
##     Copy Files Back    ##
############################
echo "done."
exit 0
