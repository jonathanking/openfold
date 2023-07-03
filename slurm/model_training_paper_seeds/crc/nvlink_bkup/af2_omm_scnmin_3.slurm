#!/bin/bash
#SBATCH --job-name=sm3omm
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
EXPERIMENT_ID="30kpure-scnmin-omm-03-bkup"
echo "Running job ${SLURM_JOB_ID} with ${SLURM_NTASKS} workers on node ${SLURMD_NODENAME}."
echo "The biggest change here is using a much larger training set after generating MSAs."
echo "Warmup OpenMM over 1000 steps. 4 gpus x 1 nodes."
echo "Uses minimized val!"
echo "Starts training from 'finetuning_5.pt'. Resumes 30kpure_00."
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
# The train_structures_dir here contains only the scnmin structs as of 230530 (30k).

TRAIN_STRUCTURES_DIR=data/train_structs/scnmin_structs0530/
ALIGNMENTS_DIR=data/alignments/scnmin_alignments0530_prefer_old_alns/  
TEMPLATE_STRUCTURES_DIR=data/template_structs/roda_pdbs_snapshotted_flattened_do_not_overwrite/
TRAIN_CACHE=data/caches/scnmin_structs0530_cache.json
TEMPLATE_CACHE=data/caches/mmcif_cache_rodasnapshot.json
VALIDATION_STRUCTURES_DIR=data/validation/cameo/20220116/minimized/data_dir
VALIDATION_ALIGNMENTS_DIR=data/validation/cameo/20220116/minimized/alignments

# CHECKPOINT=/net/pulsar/home/koes/jok120/research_openfold/experiments/30kpure-scnmin-omm-00/finetune-openfold-02/etvqd7tt/checkpoints/89-629.ckpt
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
    --gpus=4 \
    --batch_size=1 \
    --accumulate_grad_batches=32 \
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
    --seed=3 \
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
    --use_scn_pdb_names=True \
    --use_scn_pdb_names_val=True \
    --use_openmm=True \
    --use_alphafold_sampling=True \
    --experiment=${EXPERIMENT_ID}  \
    --wandb_tags="omm,scnmin,omm-warmup,valmin,more-data,more-pure-data" \
    --wandb_note="Seed 3. More 'pure' training data. Resume from finetuning_5. Valmin. OMM warmup over 1000 steps. Skips standard warmup." \
    --log_to_csv=False \
    --openmm_squashed_loss=False \
    --openmm_modified_sigmoid=5,1000000,300000,5 \
    --run_validate_first=True \
    --auto_slurm_resubmit=False \
    --use_openmm_warmup=True \
    --openmm_warmup_steps=1000


    

############################
##     Copy Files Back    ##
############################
echo "done."
exit 0
