# make a list of names

# for each name, copy over the checkpoint

aim3names=(
    # 0. AT OMM
    # "angletransformer-scnmin-00-AT-OMM-toasty-long-a100-RESUME"                
    # 1. AT noOMM
    # "angletransformer-scnmin-00-AT-noOMM-toasty"                             
    # 2. RN OMM
    # "angletransformer-scnmin-00-RN-OMM-toasty-long-a100"                
    # 3. RN noOMM
    # "angletransformer-scnmin-00-RN-noOMM-toasty-long-a100"         
    # 4. AT OMM chw0.5 omm0.1
    # "angletransformer-scnmin-00-AT-OMM-toasty-long-a100-chw0.5-omm0.1"  
    # 5. AT OMM wholemodel
    # "angletransformer-scnmin-00-AT-OMM-toasty-long-a100-wholemodel"     
    # 6. RN OMM wholemodel
    # "angletransformer-scnmin-00-RN-OMM-toasty-long-a100-wholemodel"     
    # 7. RN noOMM wholemodel
    # "angletransformer-scnmin-00-RN-noOMM-toasty-long-a100-wholemodel"   
    # Needs angletransformer-scnmin-00-AT-noOMM-toasty-long-a100-wholemodel
    # 8. AT noOMM wholemodel
    # "angletransformer-scnmin-00-AT-noOMM-toasty-long-a100-wholemodel"
    # 9. AT Franken OMM
    # "AT-frankenstein-OMM-toasty"
    # 10. AT Franken noOMM
    # "AT-frankenstein-noOMM-toasty"
    # 11. AT noOMM whole model
    "angletransformer-scnmin-00-AT-noOMM-toasty-long-a100-wholemodel-RESUME2"
    # 12. AT OMM whole model
    "angletransformer-scnmin-00-AT-OMM-toasty-long-a100-wholemodel-RESUME2"
    # 13. RN OMM whole model
    "angletransformer-scnmin-00-RN-OMM-toasty-long-a100-wholemodel-RESUME2"
    # 14. RN noOMM whole model
    "angletransformer-scnmin-00-RN-noOMM-toasty-long-a100-wholemodel-RESUME2"
)


# Baseline (no checkpoint), see file: COMPLETE-EVAL-RESNET-SCNMIN-00

# loop through and print names
# for name in "${aim3names[@]}"
# do
#     checkpoint_basedir="/ihome/dkoes/jok120/openfold/out/experiments/${name}/checkpoints/"
#     mkdir -p "out/experiments/${name}/checkpoints/"
#     echo "Copying over checkpoints from ${checkpoint_basedir}"
#     # The checkpoint is the file that ends with bestopenmm.ckpt, copy it over
#     # to the local directory
#     scp -r h2p:${checkpoint_basedir}/*bestopenmm.ckpt out/experiments/${name}/checkpoints/ &

# done


for name in "${aim3names[@]}"
do
    checkpoint_basedir="/ihome/dkoes/jok120/openfold/out/experiments/${name}/checkpoints/"
    local_dir="out/experiments/${name}/checkpoints/"
    mkdir -p ${local_dir}
    echo "Copying over checkpoints from ${checkpoint_basedir}"
    
    if [ "$name" == "angletransformer-scnmin-00-AT-OMM-toasty-long-a100-wholemodel-RESUME2" ]; then
        # For this specific model, manually set the best file
        latest_checkpoint="${checkpoint_basedir}31-1983-v1.ckpt"
    else
        # For all other models, find the highest epoch checkpoint
        remote_cmd="find ${checkpoint_basedir} -name '*.ckpt' | sort -V | tail -n 1"
        latest_checkpoint=$(ssh jok120@h2p "${remote_cmd}")
    fi
    
    # SCP the latest checkpoint over to the local directory
    scp -r jok120@h2p:${latest_checkpoint}/ ${local_dir} &
done


# BEST AT OMM CHECKPOINT
best_at_omm="/ihome/dkoes/jok120/openfold/out/experiments/angletransformer-scnmin-00-AT-OMM-toasty-long-a100-wholemodel-RESUME2/checkpoints/31-1983-v1.ckpt"
best_at_nommm="/ihome/dkoes/jok120/openfold/out/experiments/angletransformer-scnmin-00-AT-OMM-toasty-long-a100-wholemodel-RESUME2/checkpoints/53-3347.ckpt"

# copy them in to the correct local experiment directories
