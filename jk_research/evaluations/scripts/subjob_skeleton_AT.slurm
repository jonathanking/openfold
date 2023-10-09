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
    --replace_sampler_ddp=True \
    --gpus=1 \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --checkpoint_every_epoch \
    --wandb \
    --wandb_project=finetune-openfold-02 \
    --wandb_entity=koes-group \
    --precision=bf16 \
    --resume_model_weights_only={resume_model_weights_only} \
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
    --scheduler_base_lr=0 \
    --scheduler_warmup_no_steps=1 \
    --scheduler_max_lr=2e-5 \
    --use_scn_pdb_names=True \
    --use_scn_pdb_names_val=True \
    --use_openmm=True \
    --use_alphafold_sampling=True \
    --experiment={EXPERIMENT_NAME} \
    --wandb_tags="eval,eval_AT" \
    --wandb_note="{NOTES}" \
    --log_to_csv=True \
    --openmm_squashed_loss=False \
    --openmm_modified_sigmoid=5,1000000,300000,5 \
    --run_validate_first=True \
    --auto_slurm_resubmit=False \
    --use_openmm_warmup=False \
    --openmm_warmup_steps=1000 \
    --use_angle_transformer=False \
    --force_load_angle_transformer_weights=True \
    --train_only_angle_predictor=True \
    --angle_transformer_layers=42 \
    --angle_transformer_heads=1 \
    --angle_transformer_hidden=64 \
    --angle_transformer_dff=1024 \
    --angle_transformer_dropout=0.018609913167811645 \
    --angle_transformer_activation=gelu \
    --chi_weight=13.012551860064766 \
    --angle_loss_only=False \
    --angle_like_loss_only=True \
    --angle_transformer_checkpoint=${AT_CHECKPOINT} \
    --add_relu_to_omm_loss=True \
    --angle_transformer_conv_encoder=True \
    --trainer_mode=validate-val-test \
    --test_data_dir={TEST_STRUCTURES_DIR} \
    --test_alignment_dir={TEST_ALIGNMENTS_DIR}



############################
##     Copy Files Back    ##
############################
echo "done."
exit 0
