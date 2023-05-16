#!/bin/bash
#SBATCH --job-name=COPYDATA
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=12
#SBATCH --time=28-00:00:00
#SBATCH --partition=dept_cpu
#SBATCH --nodelist=n023
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jok120
#SBATCH --output="/net/pulsar/home/koes/jok120/openfold/out/%A_%6a.out"

############################
##       Description      ##
############################
echo "Running job ${SLURM_JOB_ID} with ${SLURM_NTASKS} workers on node ${SLURMD_NODENAME}."
echo "Copying data to scratch drives on nodes n021, n022, n024, and n025."

cd /scr/jok120/

rsync -ah --info=progress2 --exclude alignments/ test_alignment_generation/ /net/n021/scr/jok120/ &
rsync -ah --info=progress2 --exclude alignments/ test_alignment_generation/ /net/n022/scr/jok120/ &
rsync -ah --info=progress2 --exclude alignments/ test_alignment_generation/ /net/n024/scr/jok120/ &
rsync -ah --info=progress2 --exclude alignments/ test_alignment_generation/ /net/n025/scr/jok120/ &
wait

############################
##     Copy Files Back    ##
############################
echo "done."
exit 0
