# Use bash to run all jobs in jk_research/evaluations/aim3c/slurm/*.slurm that don't start with 8-

for file in jk_research/evaluations/aim3c/slurm/*.slurm; do
    if [[ $file != *"8-AT-noOMM-whole"* ]]; then
        echo "Running $file"
        bash $file
    fi
done

