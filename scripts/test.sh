cd /project/def-keli/qmd/AdaIMLE
module purge
module load StdEnv/2023 gcc cuda arrow faiss/1.8.0 python/3.11.5

export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.
export MASTER_PORT=$((10000 + RANDOM % 50000))

source ~/py311/bin/activate

exec torchrun --nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}') --standalone train.py --hps fewshot \
    --save_dir /scratch/qmd/results/test \
    --data_root /scratch/qmd/datasets/flowers_t \
    --dataset flowers102-t \
    --use_wandb 0 \
    --force_factor 0.01 \
    --imle_force_resample 2  \
    --lr 0.0002 \
    --iters_per_ckpt 100000 --iters_per_images 5000 --iters_per_save 1000 \
    --search_type 'lpips' \
    --n_batch 4 \
    --num_epochs 10 \
    --fid_freq 10 \
    --imle_batch 32 \
    --compile True \
    --use_clip_loss True \
    --merge_concat True \
    --use_multi_res True \
    --style_gan_merge True \
    --rep_text_emb True \
    --multi_res_scales '32,64,128' \
    --dec_blocks '1x2,4m1,4x3,8m4,8x4,16m8,16x9,32m16,32x21,64m32,64x13,128m64,128x7,256m128' 
