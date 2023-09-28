echo "MAKE SURE YOU UPDATE THE CHECKPOINT PATHS IN THE CORRESPONDING SCRIPTS FIRST!!"
# exit 0

# # Baseline 1
# echo "Running jk_research/evaluations/aim3a/slurm/resnet_baseline.slurm"
# bash jk_research/evaluations/aim3a/slurm/resnet_baseline.slurm
# # Baseline 2
# echo "Running jk_research/evaluations/aim3a/slurm/resnet_baseline_not_AF2_toastyRN1.slurm"
# bash jk_research/evaluations/aim3a/slurm/resnet_baseline_not_AF2_toastyRN1.slurm

# # Maybe the best non-conv model
# echo "Running jk_research/evaluations/aim3a/slurm/best-toastyC0_repeat_noconvB.slurm"
# bash jk_research/evaluations/aim3a/slurm/best-toastyC0_repeat_noconvB.slurm

# # A good conv model
# echo "Running jk_research/evaluations/aim3a/slurm/young_sweep_159.slurm"
# bash jk_research/evaluations/aim3a/slurm/young_sweep_159.slurm

# echo "Running jk_research/evaluations/aim3a/slurm/atXL.slurm"
# bash jk_research/evaluations/aim3a/slurm/atXL.slurm

echo "Running jk_research/evaluations/aim3a/slurm/best-toastyC0_repeat_noconvB_finetuned.slurm"
bash jk_research/evaluations/aim3a/slurm/best-toastyC0_repeat_noconvB_finetuned.slurm

echo "Running jk_research/evaluations/aim3a/slurm/resnet_baseline_not_AF2_toastyRN1_finetuned.slurm"
bash jk_research/evaluations/aim3a/slurm/resnet_baseline_not_AF2_toastyRN1_finetuned.slurm