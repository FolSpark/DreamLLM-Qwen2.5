#!/bin/bash
set -e
set -x


MASTER_ADDR=`ifconfig bond1 | grep inet | grep -v inet6 | awk '{print $2}'`
MASTER_PORT=20028
NNODES=1
NODE_RANK=0
NPROC_PER_NODE=3
export CUDA_VISIBLE_DEVICES=0,1,2

DIR=`pwd`
# export WANDB_BASE_URL=https://api.wandb.ai
wandb offline
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY=
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

MODEL_CLIP_PATH="models/clip-vit-large-patch14"
MODEL_SD_PATH="models/stable-diffusion-2-1-base"
# MODEL_NAME_OR_PATH="models/models--lmsys--vicuna-7b-v1.1/snapshots/694d39416b1744d939760f5257c4bb08b2fae400"
MODEL_NAME_OR_PATH="models/Qwen2.5-1.5B-Instruct"
# MODEL_NAME_OR_PATH="models/Qwen2.5-1.5B-Instruct"

# PRETRAIN_CLIP_MODEL_PATH="work_dirs/sft_20241122_from_stage2_20241121"
# PRETRAIN_SD_MODEL_PATH="work_dirs/sft_20241122_from_stage2_20241121"
# PRETRAIN_DREAM_EMBEDDING_PATH="work_dirs/sft_20241122_from_stage2_20241121"
PRETRAIN_CLIP_MODEL_PATH="NotSpecified"
PRETRAIN_SD_MODEL_PATH="NotSpecified"
PRETRAIN_DREAM_EMBEDDING_PATH="NotSpecified"

RUN_NAME="test"
SEED=42
TRAIN_EPOCH=1
DEVICE_BATCH_SIZE=16
GRADIENT_ACC_STEPS=2

mkdir -p ./work_dirs/${RUN_NAME}

torchrun --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank $NODE_RANK --nnodes $NNODES --nproc_per_node $NPROC_PER_NODE \
-m projects.dreamllm_qwen.train \
--config_file projects/dreamllm_qwen/configs/stage1/base.py \
"data.comprehension_only=False" \
"data.creation_only=True" \
"data.datasets=['blip_laion']" \
"data.size_list=['5.8M']" \
"data.datasets_approx_sizes=['5.8M']" \
"data.datasets_init_kwargs.seed=${SEED}" \
"data.datasets_init_kwargs.ignore_image=False" \
"data.datasets_init_kwargs.instruction_image_num=0" \
"data.skip_long_sample=True" \
"data.max_image_length=30" \
"data.disable_system_prompt=True" \
"data.conv_template_name='Qwen2.5'" \
"training.seed=${SEED}" \
"training.adam_beta2=0.95" \
"training.save_steps=10000" \
"training.save_total_limit=3" \
"training.logging_steps=10" \
"training.vit_llrd=False" \
"training.llm_llrd=False" \
"training.unfreeze_vit=False" \
"training.per_device_train_batch_size=${DEVICE_BATCH_SIZE}" \
"training.gradient_accumulation_steps=${GRADIENT_ACC_STEPS}" \
"training.max_grad_norm=1.0" \
"training.num_train_epochs=${TRAIN_EPOCH}" \
"training.output_dir='./work_dirs/${RUN_NAME}/'" \
"training.learning_rate=2e-3" \
"training.fsdp=''" \
"training.deepspeed=projects/configs/deepspeed/stage2_bf16.json" \
"training.run_project='dreamllm_qwen_creation_test'" \
"training.run_name='${RUN_NAME}'" \
"training.report_to=['wandb']" \
"model.loss_weight_vm=1.0" \
"model.loss_weight_lm=0.0" \
"model.diffusion_bs=0" \
"model.model_max_length=4096" \
"model.model_name_or_path=${MODEL_NAME_OR_PATH}" \
"model.plugins_config_init_kwargs.clip_vision_embedding.pretrained_model_name_or_path=${PRETRAIN_CLIP_MODEL_PATH}" \
"model.plugins_config_init_kwargs.clip_vision_embedding.clip_vision_model_name_or_path='${MODEL_CLIP_PATH}'" \
"model.plugins_config_init_kwargs.stable_diffusion_head.pretrained_model_name_or_path=${PRETRAIN_SD_MODEL_PATH}" \
"model.plugins_config_init_kwargs.stable_diffusion_head.diffusion_name_or_path='${MODEL_SD_PATH}'" \
"model.plugins_config_init_kwargs.stable_diffusion_head.freeze_unet=True" \
"model.plugins_config_init_kwargs.stable_diffusion_head.random_flip=False" \
"model.plugins_config_init_kwargs.dream_embedding.pretrained_model_name_or_path=${PRETRAIN_DREAM_EMBEDDING_PATH}" 2>&1 | tee "./work_dirs/${RUN_NAME}/training-torchrun.log"


