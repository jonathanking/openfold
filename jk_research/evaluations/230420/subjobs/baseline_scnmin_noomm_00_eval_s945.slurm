#!/bin/bash
#SBATCH --job-name=eval1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=dept_gpu
#SBATCH --time=6-00:00:00
#SBATCH --output="/net/pulsar/home/koes/jok120/openfold/out/%A_%6a.out"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jok120@pitt.edu
#SBATCH --nodelist=g019


############################
##       Description      ##
############################
echo "Running job ${SLURM_JOB_ID} with ${SLURM_NTASKS} workers on node ${SLURMD_NODENAME}."
echo "Experiment: baseline_scnmin_noomm_00_eval_s945"

############################
##       Environment      ##
############################
source scripts/activate_conda_env.sh
module load cuda/11.5


############################
##     Array Job Exec.    ##
############################
mkdir -p out/evaluation/230420/baseline_scnmin_noomm_00_eval_s945


./train_openfold.py data/train_structs/scnmin_structs/ data/alignments/scnmin_alignments/ data/template_structs/roda_pdbs_snapshotted_flattened_do_not_overwrite/ out/evaluation/230420/baseline_scnmin_noomm_00_eval_s945 2021-10-10 \
    --train_chain_data_cache_path=data/caches/chain_data_cache_rodasnapshot_clustered.json \
    --template_release_dates_cache_path=data/caches/mmcif_cache_rodasnapshot.json \
    --val_data_dir=data/validation/cameo/20220116/minimized/data_dir \
    --val_alignment_dir=data/validation/cameo/20220116/minimized/alignments \
    --obsolete_pdbs_file_path=data/obsolete_230310.dat \
    --resume_from_ckpt=out/experiments/230319/finetune-openfold-02/3lzb4s3o/checkpoints/62-944.ckpt \
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
    --experiment=baseline_scnmin_noomm_00_eval_s945 \
    --wandb_tags="eval" \
    --wandb_note="Evaluating baseline-scnmin-noomm-00 at step 945." \
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
