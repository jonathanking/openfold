#!/bin/bash
#SBATCH --job-name=eval2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=a100
#SBATCH --time=03:00:00
#SBATCH --output="/ihome/dkoes/jok120/openfold/out/%A_%6a.out"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jok120@pitt.edu
#SBATCH --qos=short
#SBATCH --cluster=gpu


############################
##       Description      ##
############################
echo "Running job ${SLURM_JOB_ID} with ${SLURM_NTASKS} workers on node ${SLURMD_NODENAME}."
echo "Experiment: 30kpureReLU_scnmin_omm_04_eval_sOMM"

############################
##       Environment      ##
############################
source scripts/activate_conda_env.sh
module load cuda/11.3.0
module load gcc/8.2.0


############################
##     Array Job Exec.    ##
############################
mkdir -p out/evaluation/230719/30kpureReLU_scnmin_omm_04_eval_sOMM


./train_openfold.py data/train_structs/scnmin_structs/ data/alignments/scnmin_alignments/ data/template_structs/roda_pdbs_snapshotted_flattened_do_not_overwrite/ out/evaluation/230719/30kpureReLU_scnmin_omm_04_eval_sOMM 2021-10-10 \
    --train_chain_data_cache_path=data/caches/chain_data_cache_rodasnapshot_clustered.json \
    --template_release_dates_cache_path=data/caches/mmcif_cache_rodasnapshot.json \
    --val_data_dir=data/validation/cameo/20220116/minimized/data_dir \
    --val_alignment_dir=data/validation/cameo/20220116/minimized/alignments \
    --obsolete_pdbs_file_path=data/obsolete_230310.dat \
    --resume_from_ckpt=out/experiments/30kpureReLU-scnmin-omm-04/finetune-openfold-02/3qobnh62/checkpoints/35-1403-bestopenmm.ckpt \
    --deepspeed_config_path=deepspeed_config_jk03.json \
    --gpus=1 \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
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
    --openmm_weight=1 \
    --openmm_activation=None \
    --seed=0 \
    --num_workers=2 \
    --write_pdbs \
    --write_pdbs_every_n_steps=1 \
    --log_every_n_steps=1 \
    --config_preset=finetuning_sidechainnet \
    --violation_loss_weight=1 \
    --scheduler_base_lr=0 \
    --scheduler_warmup_no_steps=500 \
    --scheduler_max_lr=5e-4 \
    --use_scn_pdb_names=True \
    --use_scn_pdb_names_val=True \
    --use_openmm=True \
    --use_alphafold_sampling=True \
    --experiment=30kpureReLU_scnmin_omm_04_eval_sOMM \
    --wandb_tags="eval" \
    --wandb_note="Evaluating 30kpureReLU-scnmin-omm-04 at step OMM." \
    --log_to_csv=True \
    --openmm_squashed_loss=False \
    --openmm_modified_sigmoid=5,1000000,300000,5 \
    --run_validate_first=True \
    --auto_slurm_resubmit=False \
    --use_openmm_warmup=False \
    --openmm_warmup_steps=1000 \
    --trainer_mode=validate-val-test \
    --test_data_dir=data/test/cameo/20230103/minimized/data_dir \
    --test_alignment_dir=data/test/cameo/20230103/minimized/alignments


############################
##     Copy Files Back    ##
############################
echo "done."
exit 0
