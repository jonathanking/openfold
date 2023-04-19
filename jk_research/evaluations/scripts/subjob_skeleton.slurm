#!/bin/bash
{preamble}


############################
##       Description      ##
############################
echo "Running job ${{SLURM_JOB_ID}} with ${{SLURM_NTASKS}} workers on node ${{SLURMD_NODENAME}}."
echo "Experiment: {EXPERIMENT_NAME}"

############################
##       Environment      ##
############################
source scripts/activate_conda_env.sh
{module_loads}


############################
##     Array Job Exec.    ##
############################
mkdir -p {OUTDIR}


./train_openfold.py {TRAIN_STRUCTURES_DIR} {ALIGNMENTS_DIR} {TEMPLATE_STRUCTURES_DIR} {OUTDIR} 2021-10-10 \
    --train_chain_data_cache_path={TRAIN_CACHE} \
    --template_release_dates_cache_path={TEMPLATE_CACHE} \
    --val_data_dir={VALIDATION_STRUCTURES_DIR} \
    --val_alignment_dir={VALIDATION_ALIGNMENTS_DIR} \
    --obsolete_pdbs_file_path=data/obsolete_230310.dat \
    --resume_from_ckpt={CHECKPOINT} \
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
    --experiment={EXPERIMENT_NAME} \
    --wandb_tags="eval" \
    --wandb_note="{NOTES}" \
    --log_to_csv=True \
    --openmm_squashed_loss=False \
    --openmm_modified_sigmoid=5,1000000,300000,5 \
    --run_validate_first=True \
    --auto_slurm_resubmit=False \
    --use_openmm_warmup=True \
    --openmm_warmup_steps=1000 \
    --trainer_mode=validate-val-test \
    --test_data_dir={TEST_STRUCTURES_DIR} \
    --test_alignment_dir={TEST_ALIGNMENTS_DIR}


############################
##     Copy Files Back    ##
############################
echo "done."
exit 0
