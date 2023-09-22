# make a list of names

# for each name, copy over the checkpoint

aim3names=(
    # 0. AT OMM
    "angletransformer-scnmin-00-AT-OMM-toasty-long-a100"                
    # 1. AT noOMM
    "angletransformer-scnmin-00-AT-noOMM-toasty"                             
    # 2. RN OMM
    "angletransformer-scnmin-00-RN-OMM-toasty-long-a100"                
    # 3. RN noOMM
    "angletransformer-scnmin-00-RN-noOMM-toasty-long-a100"         
    # 4. AT OMM chw0.5 omm0.1
    "angletransformer-scnmin-00-AT-OMM-toasty-long-a100-chw0.5-omm0.1"  
    # 5. AT OMM wholemodel
    "angletransformer-scnmin-00-AT-OMM-toasty-long-a100-wholemodel"     
    # 6. RN OMM wholemodel
    "angletransformer-scnmin-00-RN-OMM-toasty-long-a100-wholemodel"     
    # 7. RN noOMM wholemodel
    "angletransformer-scnmin-00-RN-noOMM-toasty-long-a100-wholemodel"   
    # Needs angletransformer-scnmin-00-AT-noOMM-toasty-long-a100-wholemodel
    # 8. AT noOMM wholemodel
    # "angletransformer-scnmin-00-AT-noOMM-toasty-long-a100-wholemodel"
)


# Baseline (no checkpoint), see file: COMPLETE-EVAL-RESNET-SCNMIN-00

# loop through and print names
for name in "${aim3names[@]}"
do
    checkpoint_basedir="/ihome/dkoes/jok120/openfold/out/experiments/${name}/checkpoints/"
    mkdir -p "out/experiments/${name}/checkpoints/"
    echo "Copying over checkpoints from ${checkpoint_basedir}"
    # The checkpoint is the file that ends with bestopenmm.ckpt, copy it over
    # to the local directory
    scp -r h2p:${checkpoint_basedir}/*bestopenmm.ckpt out/experiments/${name}/checkpoints/ &

done