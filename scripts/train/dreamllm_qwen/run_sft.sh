#!/bin/bash
set -e
set -x


MASTER_ADDR=`ifconfig bond1 | grep inet | grep -v inet6 | awk '{print $2}'`
MASTER_PORT=20028
NNODES=1
NODE_RANK=0
NPROC_PER_NODE=8
# export CUDA_VISIBLE_DEVICES=0,1,4,5,6,7

DIR=`pwd`
# export WANDB_BASE_URL=https://api.wandb.ai
wandb online
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY=
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1


MODEL_NAME_OR_PATH="work_dirs/stage2_20241121_from_stage1_20241119"
MODEL_CLIP_PATH="models/clip-vit-large-patch14"
MODEL_SD_PATH="models/stable-diffusion-2-1-base"
PRETRAIN_CLIP_MODEL_PATH="work_dirs/stage2_20241121_from_stage1_20241119"
PRETRAIN_SD_MODEL_PATH="work_dirs/stage2_20241121_from_stage1_20241119"
PRETRAIN_DREAM_EMBEDDING_PATH="work_dirs/stage2_20241121_from_stage1_20241119"

RUN_NAME="sft_20241122_from_stage2_20241121"
SEED=42
TRAIN_EPOCH=1
DEVICE_BATCH_SIZE=2
GRADIENT_ACC_STEPS=8

mkdir -p ./work_dirs/${RUN_NAME}

# "data.datasets=['llavav1.5_instruct']" \
# "data.size_list=['665298']" \
# "data.datasets_approx_sizes=['665298']" \
# "data.datasets=['mmc4_instruct_filtered224', 'instruct_blip_laion', 'llava_instruct_filter']" \
# "data.size_list=['20K', '20K', '80K']" \
# "data.datasets_approx_sizes=['54K', '1M', '80K']" \
torchrun --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank $NODE_RANK --nnodes $NNODES --nproc_per_node $NPROC_PER_NODE \
-m projects.dreamllm_qwen.train \
--config_file projects/dreamllm_qwen/configs/sft/base.py \
"data.comprehension_only=False" \
"data.creation_only=False" \
"data.datasets=['llavav1.5_instruct']" \
"data.size_list=['665298']" \
"data.datasets_approx_sizes=['665298']" \
"data.datasets_init_kwargs.seed=${SEED}" \
"data.datasets_init_kwargs.ignore_image=False" \
"data.datasets_init_kwargs.instruction_image_num=0" \
"data.skip_long_sample=True" \
"data.max_image_length=30" \
"data.disable_system_prompt=False" \
"data.conv_template_name='Qwen2.5'" \
"training.seed=${SEED}" \
"training.save_steps=2000" \
"training.save_strategy='epoch'" \
"training.save_total_limit=3" \
"training.vit_llrd=False" \
"training.llm_llrd=False" \
"training.unfreeze_vit=False" \
"training.per_device_train_batch_size=${DEVICE_BATCH_SIZE}" \
"training.gradient_accumulation_steps=${GRADIENT_ACC_STEPS}" \
"training.max_grad_norm=1.0" \
"training.num_train_epochs=${TRAIN_EPOCH}" \
"training.output_dir='./work_dirs/${RUN_NAME}/'" \
"training.learning_rate=4e-5" \
"training.fsdp=''" \
"training.deepspeed=projects/configs/deepspeed/stage2_bf16.json" \
"training.run_project='dreamllm_qwen'" \
"training.run_name='${RUN_NAME}'" \
"training.report_to=['wandb']" \
"model.loss_weight_vm=10.0" \
"model.loss_weight_lm=1.0" \
"model.diffusion_bs=0" \
"model.model_max_length=2048" \
"model.model_name_or_path=${MODEL_NAME_OR_PATH}" \
"model.plugins_config_init_kwargs.clip_vision_embedding.pretrained_model_name_or_path='${PRETRAIN_CLIP_MODEL_PATH}'" \
"model.plugins_config_init_kwargs.clip_vision_embedding.clip_vision_model_name_or_path='${MODEL_CLIP_PATH}'" \
"model.plugins_config_init_kwargs.stable_diffusion_head.pretrained_model_name_or_path='${PRETRAIN_SD_MODEL_PATH}'" \
"model.plugins_config_init_kwargs.stable_diffusion_head.diffusion_name_or_path='${MODEL_SD_PATH}'" \
"model.plugins_config_init_kwargs.stable_diffusion_head.freeze_unet=True" \
"model.plugins_config_init_kwargs.stable_diffusion_head.random_flip=False" \
"model.plugins_config_init_kwargs.dream_embedding.num_dream_queries=64" \
"model.plugins_config_init_kwargs.dream_embedding.pretrained_model_name_or_path='${PRETRAIN_DREAM_EMBEDDING_PATH}'" 2>&1 | tee "./work_dirs/${RUN_NAME}/training-torchrun.log"

