#!/bin/bash
#SBATCH --job-name=RtoastC3
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=28-00:00:00
#SBATCH --partition=dept_gpu
#SBATCH --output="out/%A_%6a.out"
#SBATCH --ntasks-per-node=24
#SBATCH --exclude=g019

# Setup
cd ~/openfold
source scripts/activate_conda_env.sh
module load cuda/11.5


if [ ! -d /scr/jok120/angletransformer/ ]; then
    echo "The directory /scr/jok120/angletransformer/ does not exist"
    rsync -azL --timeout=180 --info=progress2 ~/angletransformer/data/val/ /scr/jok120/angletransformer/data/val/ &
    rsync -azL --timeout=180 --info=progress2 ~/angletransformer/data/train/ /scr/jok120/angletransformer/data/train/
else
    echo "The directory /scr/jok120/angletransformer/ exists, continuing"
    # Check if the placeholder file is there
    if [ ! -f /scr/jok120/angletransformer/data/these_files_were_checked.txt   ]; then
        echo "Removing files that shouldn't be there"
        for f in ~/angletransformer/data/val/*; do rm /scr/jok120/angletransformer/data/train/$(basename $f); done
        for f in ~/angletransformer/data/train/*; do rm /scr/jok120/angletransformer/data/val/$(basename $f); done
        echo "Done removing excess files"
        echo "Done removing excess files" > /scr/jok120/angletransformer/data/these_files_were_checked.txt
    else
        echo "The placeholder file is there, continuing"
    fi
fi

cd ~/angletransformer

python train.py \
    --activation=gelu \
    --c_hidden=64 \
    --chi_weight=13.012551860064766 \
    --d_ff=1024 \
    --dropout=0.018609913167811645 \
    --is_sweep=False \
    --no_blocks=42 \
    --no_heads=1 \
    --opt_lr=0.00037082862954881073 \
    --opt_lr_scheduling=plateau \
    --opt_lr_scheduling_metric=val/angle_mae \
    --opt_n_warmup_steps=10000 \
    --opt_name=adamw \
    --opt_noam_lr_factor=1.925 \
    --replace_sampler_ddp=True \
    --train_data=data/train/ \
    --val_data=data/val/ \
    --output_dir=out/experiments/ \
    --num_workers=5 \
    --wandb_tags="aim3b,toastyconv" \
    --batch_size=1 \
    --val_check_interval=2500 \
    --experiment_name="toastyRC3" \
    --opt_patience=20 \
    --run_resnet_with_conv_encoder=True \
    --skip_loading_resnet_weights=True \
    --use_resnet_baseline=True \
    --conv_encoder=True \
    --seed=3 \

