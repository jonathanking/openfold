# Use scp to copy only the most recent file from /ihome/dkoes/jok120/openfold/out/experiments/toastyC0_repeat_noconvB-FullAF2/checkpoints/*ckpt
# to /ihome/dkoes/jok120/openfold/out/experiments/toastyC0_repeat_noconvB-FullAF2/checkpoints/

name=toastyC0_repeat_noconvB-FullAF2
mkdir -p ~/openfold/out/experiments/${name}/checkpoints/
scp -r h2p:/ihome/dkoes/jok120/openfold/out/experiments/${name}/checkpoints/70-4401.ckpt/ ~/openfold/out/experiments/${name}/checkpoints/

name=resnet_baseline_not_AF2_toastyRN1-FullAF2
mkdir -p ~/openfold/out/experiments/${name}/checkpoints/
scp -r h2p:/ihome/dkoes/jok120/openfold/out/experiments/${name}/checkpoints/80-5021.ckpt/ ~/openfold/out/experiments/${name}/checkpoints/

name=sunny-disco-172-g4gpu-extended3
# mkdir -p ~/openfold/out/experiments/${name}/checkpoints/
# scp -r h2p:/ihome/dkoes/jok120/openfold/out/experiments/${name}/checkpoints/80-5021.ckpt/ ~/openfold/out/experiments/${name}/checkpoints/
checkpoint="/net/pulsar/home/koes/jok120/angletransformer/out/experiments/angletransformer_solo01/mjvwhtai/checkpoints/at-epoch=41.ckpt"