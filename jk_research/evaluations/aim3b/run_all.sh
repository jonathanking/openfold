
# # Toasty AT Conv noConv
# for file in jk_research/evaluations/aim3b/toast_sweep_conv_noconv/*.slurm; do
#     echo $file
#     bash $file
# done

# # ResNet Conv noConv
# for file in jk_research/evaluations/aim3b/resnet_conv_noconv/*.slurm; do
#     echo $file
#     bash $file
# done

echo "Skipping toasty and resnet conv/noconv jobs because they're already complete."

# Large AT (sunnydisco) Conv noConv
for file in jk_research/evaluations/aim3b/sunnydisco_conv_noconv/*.slurm; do
    echo $file
    bash $file
done