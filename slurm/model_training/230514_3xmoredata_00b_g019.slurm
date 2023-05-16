#!/bin/bash
#SBATCH --job-name=4xdata
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --nodelist=g019
#SBATCH --time=14-00:00:00
#SBATCH --partition=dept_gpu
#SBATCH --output="out/%A_%6a.out"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jok120@pitt.edu
#SBATCH --ntasks-per-node=32



############################
##       Description      ##
############################
EXPERIMENT_ID="4xmoredata_scnmin_omm_00b"
echo "Running job ${SLURM_JOB_ID} with ${SLURM_NTASKS} workers on node ${SLURMD_NODENAME}."
echo "The biggest change here is using a 3x larger training set."
echo "Warmup OpenMM over 1000 steps. 4 gpus x 1 nodes."
echo "Uses minimized val!"
echo "Starts training from 'initial_training.pt'."
echo "This will be called ${EXPERIMENT_ID}."
echo "OMM loss is scaled by 0.01."
echo "Skips warmup for standard lr."

############################
##       Environment      ##
############################
source scripts/activate_conda_env.sh
module load cuda/11.5


############################
##     Array Job Exec.    ##
############################
OUTDIR=out/experiments/${EXPERIMENT_ID}/
mkdir -p ${OUTDIR}

# Remember, the aln dir is what determines the training set size! 
# The structure directory and caches can contain more structures than the aln dir.
# The train_structures_dir here contains all of the scnmin structs as of 230510 as well
# as all of the roda structures!

TRAIN_STRUCTURES_DIR=data/train_structs/combined_scnmin_structs0510/
ALIGNMENTS_DIR=data/alignments/combined_scnmin_alignments0510_random30k/  
TEMPLATE_STRUCTURES_DIR=data/template_structs/roda_pdbs_snapshotted_flattened_do_not_overwrite/
TRAIN_CACHE=data/caches/chain_data_cache_rodasnapshot_clustered.json
TEMPLATE_CACHE=data/caches/mmcif_cache_rodasnapshot.json
VALIDATION_STRUCTURES_DIR=data/validation/cameo/20220116/minimized/data_dir
VALIDATION_ALIGNMENTS_DIR=data/validation/cameo/20220116/minimized/alignments

CHECKPOINT=/net/pulsar/home/koes/jok120/openfold/out/experiments/4xmoredata_scnmin_omm_00/finetune-openfold-02/rcectmbe/checkpoints/10-76.ckpt
echo "Uses checkpoint $CHECKPOINT"

./train_openfold.py ${TRAIN_STRUCTURES_DIR} ${ALIGNMENTS_DIR} ${TEMPLATE_STRUCTURES_DIR} ${OUTDIR} 2021-10-10 \
    --train_chain_data_cache_path=${TRAIN_CACHE} \
    --template_release_dates_cache_path=${TEMPLATE_CACHE} \
    --val_data_dir=${VALIDATION_STRUCTURES_DIR} \
    --val_alignment_dir=${VALIDATION_ALIGNMENTS_DIR} \
    --obsolete_pdbs_file_path=data/obsolete_230310.dat \
    --resume_from_ckpt=${CHECKPOINT} \
    --deepspeed_config_path=deepspeed_config_jk03.json \
    --replace_sampler_ddp=True \
    --gpus=3 \
    --batch_size=1 \
    --accumulate_grad_batches=43 \
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
    --seed=0 \
    --debug \
    --num_workers=6 \
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
    --wandb_note="3x more training data. Start from initial_training.pt. Valmin. OMM warmup over 1000 steps. Skips standard warmup. Early stopping." \
    --log_to_csv=False \
    --openmm_squashed_loss=False \
    --openmm_modified_sigmoid=5,1000000,300000,5 \
    --run_validate_first=True \
    --auto_slurm_resubmit=False \
    --use_openmm_warmup=True \
    --openmm_warmup_steps=1000 \
    --early_stopping=True \
    --min_delta=0 \
    --patience=7
    # --val_check_interval=0.15


    

############################
##     Copy Files Back    ##
############################
echo "done."
exit 0
