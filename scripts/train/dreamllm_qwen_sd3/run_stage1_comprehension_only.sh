#!/bin/bash
set -e
set -x


MASTER_ADDR=`ifconfig bond1 | grep inet | grep -v inet6 | awk '{print $2}'`
MASTER_PORT=20028
NNODES=1
NODE_RANK=0
NPROC_PER_NODE=1
# export CUDA_VISIBLE_DEVICES=0,1,4,5,6,7

DIR=`pwd`
# export WANDB_BASE_URL=https://api.wandb.ai
wandb offline
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY=
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

MODEL_CLIP_PATH="models/clip-vit-large-patch14"
MODEL_SD_PATH="models/stable-diffusion-3.5-medium"
MODEL_NAME_OR_PATH="models/Qwen2.5-1.5B-Instruct"

PRETRAIN_CLIP_MODEL_PATH="NotSpecified"
PRETRAIN_SD_MODEL_PATH="NotSpecified"
PRETRAIN_DREAM_EMBEDDING_PATH="NotSpecified"

RUN_NAME="test"
SEED=42
TRAIN_EPOCH=1
DEVICE_BATCH_SIZE=8
GRADIENT_ACC_STEPS=1

mkdir -p ./work_dirs/${RUN_NAME}

torchrun --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank $NODE_RANK --nnodes $NNODES --nproc_per_node $NPROC_PER_NODE \
-m projects.dreamllm_qwen_sd3.train \
--config_file projects/dreamllm_qwen_sd3/configs/stage1/base.py \
"data.comprehension_only=True" \
"data.creation_only=False" \
"data.datasets=['mmc4_core']" \
"data.size_list=['1M']" \
"data.datasets_approx_sizes=['5.8M']" \
"data.datasets_init_kwargs.seed=${SEED}" \
"data.datasets_init_kwargs.ignore_image=False" \
"data.datasets_init_kwargs.instruction_image_num=0" \
"data.skip_long_sample=True" \
"data.max_image_length=30" \
"data.disable_system_prompt=True" \
"data.conv_template_name='Qwen2.5'" \
"training.seed=${SEED}" \
"training.save_steps=5000" \
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
"training.run_project='dreamllm_qwen_sd3'" \
"training.run_name='${RUN_NAME}'" \
"training.report_to=['wandb']" \
"model.loss_weight_vm=0.0" \
"model.loss_weight_lm=1.0" \
"model.diffusion_bs=0" \
"model.model_max_length=2048" \
"model.model_name_or_path=${MODEL_NAME_OR_PATH}" \
"model.plugins_config_init_kwargs.clip_vision_embedding.pretrained_model_name_or_path=${PRETRAIN_CLIP_MODEL_PATH}" \
"model.plugins_config_init_kwargs.clip_vision_embedding.clip_vision_model_name_or_path='${MODEL_CLIP_PATH}'" \
"model.plugins_config_init_kwargs.stable_diffusion3_head.pretrained_model_name_or_path=${PRETRAIN_SD_MODEL_PATH}" \
"model.plugins_config_init_kwargs.stable_diffusion3_head.diffusion_name_or_path='${MODEL_SD_PATH}'" \
"model.plugins_config_init_kwargs.stable_diffusion3_head.freeze_transformer=True" \
"model.plugins_config_init_kwargs.stable_diffusion3_head.random_flip=False" \
"model.plugins_config_init_kwargs.stable_diffusion3_head.resolution=512" \
"model.plugins_config_init_kwargs.dream_embedding.pretrained_model_name_or_path=${PRETRAIN_DREAM_EMBEDDING_PATH}" 2>&1 | tee "./work_dirs/${RUN_NAME}/training-torchrun.log"
