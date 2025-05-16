#!/bin/bash
#SBATCH --time=0-23:30:0  # Time: D-H:m:S
#SBATCH --account=def-keli # Account
#SBATCH --mem=80G           # Memory in total
#SBATCH --nodes=1          # Number of nodes requested.
#SBATCH --tasks-per-node=8
#SBATCH --gres=gpu:v100l:2 # 32G V100

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
echo “Starting job...”

cd $projects/AdaIMLE
module purge
module load StdEnv/2023 gcc cuda arrow faiss/1.8.0 python/3.11.5
# module load gcc cuda faiss/1.7.4 python/3.10
# module load scipy-stack/2024a

export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_HANDLING=1
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo “r$SLURM_NODEID master: $MASTER_ADDR”
echo “r$SLURM_NODEID Launching python script”

source ~/py311/bin/activate
# scp /project/6054857/cva19/clean-fid/weights/inception-2015-12-05.pt /tmp/
# Python will buffer output of your script unless you set this.
# If you’re not using python, figure out how to turn off output
# buffering when stdout is a file, or else when watching your output
# script you’ll only get updated every several lines printed.
#pip download -i https://test.pypi.org/simple/ dciknn-cuda==0.1.15
export EXP_NAME=test
export save_dir=“/scratch/qmd/results/new_imle/flowers_i/${EXP_NAME}”
export load_point=“latest”
#!/bin/bash
set -ex
echo “Running at $(date)”
exec torchrun --nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}') --standalone train.py --hps fewshot \
    --save_dir ${save_dir} \
    --data_root /scratch/qmd/datasets/flowers_i/ \
    --dataset flowers102-i \
    --wandb_name new_imle \
    --force_factor 5 \
    --imle_force_resample 5  \
    --lr 0.0002 \
    --iters_per_ckpt 100000 --iters_per_images 5000 --iters_per_save 1000 \
    --search_type 'lpips' \
    --n_batch 8 \
    --num_epochs 500 \
    --fid_freq 10 \
    --imle_batch 32 \
    --compile True \
    --use_multi_res True \
    --multi_res_scales '32,64,128' \
    --dec_blocks '1x2,4m1,4x3,8m4,8x4,16m8,16x9,32m16,32x21,64m32,64x13,128m64,128x7,256m128'
