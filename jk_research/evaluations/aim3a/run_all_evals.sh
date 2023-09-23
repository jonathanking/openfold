echo "MAKE SURE YOU UPDATE THE CHECKPOINT PATHS IN THE CORRESPONDING SCRIPTS FIRST!!"
exit 0

# Baseline 1
bash jk_research/evaluations/aim3a/slurm/resnet_baseline.slurm
# Baseline 2
bash jk_research/evaluations/aim3a/slurm/resnet_baseline_not_AF2_toastyRN1.slurm
# Maybe the best non-conv model
bash jk_research/evaluations/aim3a/slurm/best-toastyC0_repeat_noconvB.slurm
# A good conv model
bash jk_research/evaluations/aim3a/slurm/young_sweep_159.slurm
