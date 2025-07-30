#!/bin/bash
#SBATCH --time=0-7:30:0  # Time: D-H:m:S
#SBATCH --account=rrg-keli # Account 1/8, rrg 7/8
#SBATCH --mem=80G           # Memory in total
#SBATCH --nodes=1          # Number of nodes requested.
#SBATCH --tasks-per-node=8
#SBATCH --gres=gpu:a100:1 # 32G V100
#SBATCH --output=/scratch/qmd/results/new_imle/celeba/preproc/log_out.log
##SBATCH -e slurm.%N.%j.err    # STDERR

# Below sets the email notification, swap to your email to receive notifications
#SBATCH --mail-user=qmd@sfu.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
# Print some info for context.
pwd
hostname
date
echo "Starting job..."

cd /project/def-keli/qmd/AdaIMLE/scripts
module purge
module load StdEnv/2023 gcc cuda arrow faiss/1.8.0 python/3.11.5
# module load gcc cuda faiss/1.7.4 python/3.10
# module load scipy-stack/2024a


source ~/py311/bin/activate
# scp /project/6054857/cva19/clean-fid/weights/inception-2015-12-05.pt /tmp/
# Python will buffer output of your script unless you set this.
# If you’re not using python, figure out how to turn off output
# buffering when stdout is a file, or else when watching your output
# script you’ll only get updated every several lines printed.
#pip download -i https://test.pypi.org/simple/ dciknn-cuda==0.1.15

python divide.py

python extract.py

