#!/bin/bash
#SBATCH --job-name=db-baseline-lrstep
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=16
#SBATCH --time=14-00:00:00
#SBATCH --partition=dept_gpu
#SBATCH --nodelist=g019
#SBATCH --output="/net/pulsar/home/koes/jok120/openfold/out/%A_%6a.out"

############################
##       Description      ##
############################
echo "Running job ${SLURM_JOB_ID} with ${SLURM_NTASKS} workers on node ${SLURMD_NODENAME}."

############################
##       Environment      ##
############################
source scripts/activate_conda_env.sh

############################
##     Array Job Exec.    ##
############################
OUTDIR=out/experiments/230214/
mkdir -p ${OUTDIR}


./train_openfold.py data/train_structs/pdb_files_for_scnmin/ data/alignments/scn_roda2/ data/template_structs/mmcif_files_for_scn/ ${OUTDIR} 2021-10-10 --gpus=1 --checkpoint_every_epoch --train_chain_data_cache_path=data/caches/chain_data_cache_scn.json --obsolete_pdbs_file_path=data/obsolete.dat --config_preset=finetuning_sidechainnet --wandb --wandb_project=finetune-openfold-01 --wandb_entity=jonathanking --num_workers=4 --log_every_n_steps=1 --precision=32 --resume_from_ckpt=openfold/resources/openfold_params/initial_training.pt --resume_model_weights_only=True --train_epoch_len=100 --max_epochs=500 --debug --use_openmm=False --add_struct_metrics --openmm_weight=1 --openmm_activation=None --use_scn_pdb_names --write_pdbs --write_pdbs_every_n_steps=10 --seed=0 --wandb_note="Baseline on scn data. violoation loss on. LR set to 70k." --experiment=ftfd-00-baseline-70klr --template_release_dates_cache_path=data/caches/mmcif_cache_scn.json --set_lr_step=70000 --debug --scheduler_last_epoch=70000 --scheduler_max_lr=1e-4 --scheduler_decay_every_n_steps=1000

############################
##     Copy Files Back    ##
############################
echo "done."
exit 0
