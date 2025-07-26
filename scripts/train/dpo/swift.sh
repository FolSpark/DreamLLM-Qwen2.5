# after you install the swift package, you can use the swift command to run swift code
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

rpo_alpha=1
learning_rate=5e-7
output_dir=work_dirs/dpo/debug_1
mkdir -p $output_dir

export PYTHONPATH=.

# VIDEO_SEGMENTS=8 \
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
NPROC_PER_NODE=7 \
swift rlhf \
    --rlhf_type dpo \
    --custom_register_path projects/dpo/register_model.py \
    --model work_dirs/DreamLLM-Qwen2.5-InternVit-SD3.5 \
    --dataset datasets/MM-RLHF/tmp/mmrlhf_v1_image.jsonl \
    --beta 0.2 \
    --rpo_alpha $rpo_alpha \
    --train_type full \
    --deepspeed zero2_offload \
    --freeze_parameters stable_diffusion_head model.dream_embedding model.vision_encoder \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate $learning_rate \
    --freeze_vit true \
    --eval_steps 3 \
    --save_steps 5 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --max_length 2048 \
    --output_dir $output_dir \
    --warmup_ratio 0.03 \
    --system "You are a helpful assistant." \
    --dataloader_num_workers 8 \
    --report_to none 2>&1 | tee $output_dir/train.log