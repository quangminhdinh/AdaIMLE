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

export EXP_NAME=text_base_9_force_2_rep
export save_dir="/scratch/qmd/results/new_imle/flowers_t/${EXP_NAME}"
export load_point="latest"

exec torchrun --nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}') --standalone train.py --hps fewshot \
    --save_dir ${save_dir} \
    --data_root /scratch/qmd/datasets/flowers_t \
    --dataset flowers102-t \
    --use_wandb 0 \
    --force_factor 0.01 \
    --imle_force_resample 2  \
    --lr 0.0002 \
    --iters_per_ckpt 100000 --iters_per_images 5000 --iters_per_save 1000 \
    --search_type 'lpips' \
    --n_batch 4 \
    --num_epochs 4000 \
    --fid_freq 10 \
    --imle_batch 16 \
    --imle_db_size 512 \
    --compile True \
    --use_multi_res True \
    --rep_text_emb True \
    --multi_res_scales '32,64,128' \
    --dec_blocks '1x2,4m1,4x3,8m4,8x4,16m8,16x9,32m16,32x21,64m32,64x13,128m64,128x7,256m128' \
    --restore_path ${save_dir}/train/${load_point}-model.th \
    --restore_ema_path ${save_dir}/train/${load_point}-model-ema.th \
    --restore_optimizer_path ${save_dir}/train/${load_point}-opt.th \
    --restore_scaler_path ${save_dir}/train/${load_point}-scaler.th \
    --restore_scheduler_path ${save_dir}/train/${load_point}-sched.th \
    --restore_log_path ${save_dir}/train/${load_point}-log.jsonl \
    --mode eval_text
