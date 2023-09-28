# Use bash to run all jobs in jk_research/evaluations/aim3c/slurm/*.slurm that don't contain the word "whole"

# echo "WARNING!! NOT RUNNING WHOLE MODEL FT!!"
# exit 0

# for file in jk_research/evaluations/aim3c/slurm/*.slurm; do
#     if [[ $file != *"whole"* ]]; then
#         echo "Running $file"
#         bash $file
#     fi
# done


echo "Running RN-OMM"
bash jk_research/evaluations/aim3c/slurm/2-RN-OMM.slurm

echo "running RN-noOMM"
bash jk_research/evaluations/aim3c/slurm/3-RN-noOMM.slurm
