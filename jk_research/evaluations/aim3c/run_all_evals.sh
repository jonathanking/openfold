# Use bash to run all jobs in jk_research/evaluations/aim3c/slurm/*.slurm that don't start with 8-

echo "WARNING!! NOT RUNNING PART 8!!"
exit 0

for file in jk_research/evaluations/aim3c/slurm/*.slurm; do
    if [[ $file != *"8-AT-noOMM-whole"* ]]; then
        echo "Running $file"
        bash $file
    fi
done

