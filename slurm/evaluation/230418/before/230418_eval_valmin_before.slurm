#!/bin/bash
#SBATCH --job-name=evalval0
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=a100
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --cluster=gpu
#SBATCH --output="/ihome/dkoes/jok120/openfold/out/%A_%6a.out"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jok120@pitt.edu


############################
##    Description/Flags   ##
############################
EVALUATE="valid"
EXPERIMENT_ID="eval-valmin-before-00"
DESCR="inference on valmin at initial_training.pt. fixed bug with valmin"
OUTDIR=out/evaluation/230418/${EXPERIMENT_ID//-/_}  # the exp id with dashes replaced with underscores
USE_SCN_PDB_NAMES="True"
CKPT="openfold/resources/openfold_params/initial_training.pt"


############################
##      Flag Handling     ##
############################
echo "Running job ${SLURM_JOB_ID} with ${SLURM_NTASKS} workers on node ${SLURMD_NODENAME}."
echo "${EXPERIMENT_ID}"
echo "${DESCR}"

if [ $EVALUATE == "valid" ]; then
    VALIDATION_STRUCTURES_DIR=data/validation/cameo/20220116/minimized/data_dir
    VALIDATION_ALIGNMENTS_DIR=data/validation/cameo/20220116/minimized/alignments
elif [ $EVALUATE == "test" ]; then
    VALIDATION_STRUCTURES_DIR=data/test/cameo/20230103/minimized/data_dir
    VALIDATION_ALIGNMENTS_DIR=data/test/cameo/20230103/minimized/alignments
fi


############################
##       Environment      ##
############################
source scripts/activate_conda_env.sh
module load cuda/11.3.0
module load gcc/8.2.0


############################
##     Array Job Exec.    ##
############################
mkdir -p ${OUTDIR}

TRAIN_STRUCTURES_DIR=data/train_structs/scnmin_structs/
ALIGNMENTS_DIR=data/alignments/scnmin_alignments/
TEMPLATE_STRUCTURES_DIR=data/template_structs/roda_pdbs_snapshotted_flattened_do_not_overwrite/
TRAIN_CACHE=data/caches/chain_data_cache_rodasnapshot_clustered.json
TEMPLATE_CACHE=data/caches/mmcif_cache_rodasnapshot.json

./train_openfold.py ${TRAIN_STRUCTURES_DIR} ${ALIGNMENTS_DIR} ${TEMPLATE_STRUCTURES_DIR} ${OUTDIR} 2021-10-10 \
    --train_chain_data_cache_path=${TRAIN_CACHE} \
    --template_release_dates_cache_path=${TEMPLATE_CACHE} \
    --val_data_dir=${VALIDATION_STRUCTURES_DIR} \
    --val_alignment_dir=${VALIDATION_ALIGNMENTS_DIR} \
    --obsolete_pdbs_file_path=data/obsolete_230310.dat \
    --resume_from_ckpt=${CKPT} \
    --deepspeed_config_path=deepspeed_config_jk03.json \
    --gpus=1 \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
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
    --num_workers=2 \
    --write_pdbs \
    --write_pdbs_every_n_steps=1 \
    --log_every_n_steps=1 \
    --config_preset=finetuning_sidechainnet \
    --violation_loss_weight=1 \
    --scheduler_base_lr=0 \
    --scheduler_warmup_no_steps=500 \
    --scheduler_max_lr=5e-4 \
    --use_scn_pdb_names=${USE_SCN_PDB_NAMES} \
    --use_scn_pdb_names_val=${USE_SCN_PDB_NAMES} \
    --use_openmm=True \
    --use_alphafold_sampling=True \
    --experiment=${EXPERIMENT_ID} \
    --wandb_tags="eval,valmin,before" \
    --wandb_note="${DESCR}" \
    --log_to_csv=True \
    --openmm_squashed_loss=False \
    --openmm_modified_sigmoid=5,1000000,300000,5 \
    --run_validate_first=True \
    --auto_slurm_resubmit=False \
    --use_openmm_warmup=True \
    --openmm_warmup_steps=1000 \
    --trainer_mode=validate-val

    

############################
##     Copy Files Back    ##
############################
echo "done."
exit 0
