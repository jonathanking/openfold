RDIR="/scr"
FASTA_DIR="${RDIR}/experiments/221101/fastas"
MMCIF_FILES="/scr/alphafold_data/pdb_mmcif/mmcif_files_for_roda/"
DATA_ROOT="/scr/alphafold_data"
OUT_DIR="${RDIR}/experiments/221205/out1"
JACKHMMER="/home/jok120/anaconda3/envs/openfold_venv/bin/jackhmmer"
HHBLITS="/home/jok120/anaconda3/envs/openfold_venv/bin/hhblits"
HHSEARCH="/home/jok120/anaconda3/envs/openfold_venv/bin/hhsearch"
KALIGN="/home/jok120/anaconda3/envs/openfold_venv/bin/kalign"
ALIGN_DIR="/scr/scn_roda"


python3 run_pretrained_openfold.py \
    $FASTA_DIR \
    $MMCIF_FILES \
    --output_dir $OUT_DIR \
    --config_preset "model_1_ptm" \
    --use_precomputed_alignments $ALIGN_DIR \
    --uniref90_database_path $DATA_ROOT/uniref90/uniref90.fasta \
    --mgnify_database_path $DATA_ROOT/mgnify/mgy_clusters_2018_12.fa \
    --pdb70_database_path $DATA_ROOT/pdb70/pdb70 \
    --uniclust30_database_path $DATA_ROOT/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
    --bfd_database_path $DATA_ROOT/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
    --model_device "cuda:0" \
    --jackhmmer_binary_path $JACKHMMER \
    --hhblits_binary_path $HHBLITS \
    --hhsearch_binary_path $HHSEARCH \
    --kalign_binary_path $KALIGN 



    # --save_outputs \
    # --openfold_checkpoint_path /scr/openfold/openfold/resources/openfold_params/finetuning_ptm_2.pt \

python3 train_openfold.py $MMCIF_FILES $ALIGN_DIR $MMCIF_FILES $OUT_DIR/ \
    2021-10-10 \ 
    --template_release_dates_cache_path mmcif_cache.json \ 
    --precision bf16 \
    --gpus 1 --replace_sampler_ddp=True \
    --seed 4242022 \ # in multi-gpu settings, the seed must be specified
    --deepspeed_config_path deepspeed_config.json \
    --checkpoint_every_epoch \
    --resume_from_ckpt ckpt_dir/ \
    --train_chain_data_cache_path chain_data_cache.json \
    --obsolete_pdbs_file_path obsolete.dat