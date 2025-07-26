#!/bin/bash
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECKS_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_DEBUG=info
export export WANDB_BASE_URL=https://api.wandb.ai

MLP_WORKER_0_HOST=${MLP_WORKER_0_HOST:-127.0.0.1}
MLP_WORKER_0_PORT=${MLP_WORKER_0_PORT:-20023}
MLP_ROLE_INDEX=${MLP_ROLE_INDEX:-0}
MLP_WORKER_NUM=${MLP_WORKER_NUM:-1}

MODEL_CLIP_PATH=
MODEL_SD_PATH=
MODEL_NAME_OR_PATH=
DATASET_PATH=
DATASET_SIZE=
RUN_NAME=
SEED=42

torchrun --master_addr $MLP_WORKER_0_HOST --master_port $MLP_WORKER_0_PORT --node_rank $MLP_ROLE_INDEX --nnodes $MLP_WORKER_NUM --nproc_per_node 8 \
-m projects.dreamllm.train \
--config_file projects/dreamllm/configs/sft/base.py \
"data.comprehension_only=False" \
"data.creation_only=False" \
"data.datasets=['penpot']" \
"data.size_list=['${DATASET_SIZE}']" \
"data.datasets_approx_sizes=['${DATASET_SIZE}']" \
"data.datasets_shard_lists=[${DATASET_PATH}]" \
"data.datasets_init_kwargs.seed=${SEED}" \
"data.skip_long_sample=True" \
"training.seed=${SEED}" \
"training.save_steps=1000" \
"training.save_total_limit=10" \
"training.vit_llrd=False" \
"training.llm_llrd=False" \
"training.unfreeze_vit=False" \
"training.per_device_train_batch_size=2" \
"training.gradient_accumulation_steps=1" \
"training.num_train_epochs=25" \
"training.output_dir='./work_dirs/${RUN_NAME}/'" \
"training.learning_rate=4e-5" \
"training.fsdp=''" \
"training.deepspeed=projects/configs/deepspeed/stage2_bf16.json" \
"training.run_project='dreamllm'" \
"training.run_name='${RUN_NAME}'" \
"training.report_to=['wandb']" \
"training.use_lora=True" \
"training.lora_config.r=16" \
"training.lora_config.lora_alpha=32" \
"training.lora_config.lora_dropout=0.05" \
"training.lora_config.target_modules='model.layers.*.*.(q_proj|v_proj|k_proj|o_proj|gate_proj|down_proj|up_proj)$'" \
"model.loss_weight_vm=10.0" \
"model.loss_weight_lm=1.0" \
"model.model_max_length=8192" \
"model.model_name_or_path=${MODEL_NAME_OR_PATH}" \
"model.plugins_config_init_kwargs.clip_vision_embedding.pretrained_model_name_or_path='${MODEL_NAME_OR_PATH}'" \
"model.plugins_config_init_kwargs.clip_vision_embedding.clip_vision_model_name_or_path='${MODEL_CLIP_PATH}'" \
"model.plugins_config_init_kwargs.stable_diffusion_head.pretrained_model_name_or_path='${MODEL_NAME_OR_PATH}'" \
"model.plugins_config_init_kwargs.stable_diffusion_head.diffusion_name_or_path='${MODEL_SD_PATH}'" \
"model.plugins_config_init_kwargs.dream_embedding.pretrained_model_name_or_path='${MODEL_NAME_OR_PATH}'"

