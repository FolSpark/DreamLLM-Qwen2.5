#!/bin/bash
set -e
set -x
export PATH=/opt/conda/bin:${PATH}
export PYTHONPATH=${DIR}

HOSTS=("" "" "" "")
SLOTS=8
HOSTFILE=./hostfiles/_hostfile
> $HOSTFILE
for HOST in "${HOSTS[@]}"; do
  echo "$HOST   slots=$SLOTS" >> $HOSTFILE
done
echo "hostfile:"
cat $HOSTFILE

NP=$(( `grep -v '#' $HOSTFILE | grep -v ^$ | wc -l` * 8 ))
NN=$(( `grep -v '#' $HOSTFILE | grep -v ^$ | wc -l`))
MASTER_ADDR=`ifconfig bond1 | sed -nr '2s/.*inet ([0-9.]+) .*/\1/p'`
MASTER_ADDR=`ifconfig bond1 | grep inet | grep -v inet6 | awk '{print $2}'`
MASTER_PORT=20028
DIR=`pwd`
wandb online
WANDB_BASE_URL=https://api.wandb.ai
WANDB_API_KEY=

MODEL_InternViT_PATH="models/InternViT-300M-448px-V2_5"
MODEL_SD_PATH="models/stable-diffusion-3.5-medium"

MODEL_NAME_OR_PATH="work_dirs/InternViT_sd3.5_512_q256_stage2_20250220_from_stage1_20250216"

PRETRAIN_InternViT_MODEL_PATH="work_dirs/InternViT_sd3.5_512_q256_stage2_20250220_from_stage1_20250216"
PRETRAIN_SD_MODEL_PATH="work_dirs/InternViT_sd3.5_512_q256_stage2_20250220_from_stage1_20250216"
PRETRAIN_DREAM_EMBEDDING_PATH="work_dirs/InternViT_sd3.5_512_q256_stage2_20250220_from_stage1_20250216"

RUN_NAME="InternViT_sd3.5_512_q256_stage3_full_20250309_from_stage2_20250220"
SEED=42
TRAIN_EPOCH=1
DEVICE_BATCH_SIZE=2
GRADIENT_ACC_STEPS=16

mkdir -p ./work_dirs/${RUN_NAME}

mpirun --allow-run-as-root -np $NP \
        -hostfile $HOSTFILE \
        -mca plm_rsh_args "-p 8150"  \
        --tag-output \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_SL=3 \
        -x NCCL_CHECKS_DISABLE=1 \
        -x NCCL_P2P_DISABLE=0 \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_LL_THRESHOLD=16384 \
        -x NCCL_IB_CUDA_SUPPORT=1 \
        -x NCCL_SOCKET_IFNAME=bond1 \
        -x UCX_NET_DEVICES=bond1 \
        -x NCCL_IB_HCA=mlx5 \
        -x NCCL_COLLNET_ENABLE=1 \
        -x SHARP_COLL_ENABLE_SAT=0 \
        -x NCCL_NET_GDR_LEVEL=2 \
        -x NCCL_IB_QPS_PER_CONNECTION=4 \
        -x NCCL_IB_TC=160 \
        -x GLOO_SOCKET_IFNAME=bond1 \
        -x NCCL_DEBUG=info \
        -x WANDB_BASE_URL=${WANDB_BASE_URL} \
        -x WANDB_API_KEY=${WANDB_API_KEY} \
        -x PATH=/opt/conda/bin:${PATH} \
        -x PYTHONPATH=${DIR} \
        -x MASTER_ADDR=$MASTER_ADDR \
        -x MASTER_PORT=$MASTER_PORT \
    python ${DIR}/projects/dreamllm_qwen2/train.py \
        --config_file projects/dreamllm_qwen2/configs_intern/sft/base.py \
        "data.comprehension_only=False" \
        "data.creation_only=False" \
        "data.datasets=['mmc4_instruct_filtered224', 'instruct_laion2B-en-aesthetic', 'llavav1.5_instruct']" \
        "data.size_list=['60K', '60K', '1995894']" \
        "data.datasets_approx_sizes=['54K', '6M', '665298']" \
        "data.datasets_init_kwargs.seed=${SEED}" \
        "data.datasets_init_kwargs.ignore_image=False" \
        "data.datasets_init_kwargs.instruction_image_num=0" \
        "data.skip_long_sample=True" \
        "data.max_image_length=30" \
        "data.disable_system_prompt=False" \
        "data.conv_template_name='Qwen2.5'" \
        "training.seed=${SEED}" \
        "training.save_steps=700" \
        "training.save_total_limit=6" \
        "training.logging_steps=1" \
        "training.vit_llrd=False" \
        "training.llm_llrd=False" \
        "training.unfreeze_vit=False" \
        "training.per_device_train_batch_size=${DEVICE_BATCH_SIZE}" \
        "training.gradient_accumulation_steps=${GRADIENT_ACC_STEPS}" \
        "training.max_grad_norm=1.0" \
        "training.num_train_epochs=${TRAIN_EPOCH}" \
        "training.output_dir='./work_dirs/${RUN_NAME}/'" \
        "training.learning_rate=4e-6" \
        "training.fsdp=''" \
        "training.deepspeed=projects/configs/deepspeed/stage2_bf16_disable_overlap_comm.json" \
        "training.run_project='dreamllm_qwen_sd3'" \
        "training.run_name='${RUN_NAME}'" \
        "training.report_to=['wandb']" \
        "model.attn_implementation='flash_attention_2'" \
        "model.loss_weight_vm=10.0" \
        "model.loss_weight_lm=1.0" \
        "model.diffusion_bs=0" \
        "model.model_max_length=4096" \
        "model.model_name_or_path=${MODEL_NAME_OR_PATH}" \
        "model.plugins_config_init_kwargs.vision_encoder.pretrained_model_name_or_path=${PRETRAIN_InternViT_MODEL_PATH}" \
        "model.plugins_config_init_kwargs.vision_encoder.intern_vit_model_name_or_path='${MODEL_InternViT_PATH}'" \
        "model.plugins_config_init_kwargs.stable_diffusion_head.pretrained_model_name_or_path=${PRETRAIN_SD_MODEL_PATH}" \
        "model.plugins_config_init_kwargs.stable_diffusion_head.diffusion_name_or_path='${MODEL_SD_PATH}'" \
        "model.plugins_config_init_kwargs.stable_diffusion_head.freeze_transformer=True" \
        "model.plugins_config_init_kwargs.stable_diffusion_head.random_flip=False" \
        "model.plugins_config_init_kwargs.stable_diffusion_head.resolution=512" \
        "model.plugins_config_init_kwargs.dream_embedding.num_dream_queries=256" \
        "model.plugins_config_init_kwargs.dream_embedding.pretrained_model_name_or_path=${PRETRAIN_DREAM_EMBEDDING_PATH}" 2>&1 | tee "./work_dirs/${RUN_NAME}/training-mpirun.log"


