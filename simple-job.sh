#!/bin/bash
#SBATCH --account=def-keli
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:v100l:1
#SBATCH --time=24:0:0    
#SBATCH --mail-user=qmd@sfu.ca
#SBATCH --mail-type=ALL

cd $projects/rs-imle
module purge
module load StdEnv/2023 gcc cuda arrow faiss/1.8.0 python/3.11.5
source ~/py311/bin/activate

python train.py --hps fewshot \
    --save_dir /scratch/qmd/imle/results/flowers_i/ \
    --data_root /scratch/qmd/datasets/flowers_i/ \
    --dataset flowers102-i \
    --wandb_name rs_imle_ca \
    --num_epochs 100 \
    --n_batch 16 \
    --imle_force_resample 10 \
    --fid_freq 10 \
    --lr 0.0002 

