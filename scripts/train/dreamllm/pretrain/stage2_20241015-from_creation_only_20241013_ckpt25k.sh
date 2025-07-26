#!/bin/bash
set -e
set -x
export PATH=/opt/conda/bin:${PATH}
export PYTHONPATH=${DIR}

HOSTS=("" "")
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
export WANDB_BASE_URL=https://api.wandb.ai
WANDB_API_KEY=
MODEL_CLIP_PATH="./models/clip-vit-large-patch14"
MODEL_SD_PATH="./models/stable-diffusion-2-1-base"
MODEL_NAME_OR_PATH="./models/models--lmsys--vicuna-7b-v1.1/snapshots/694d39416b1744d939760f5257c4bb08b2fae400"
PRETRAIN_CLIP_MODEL_PATH="work_dirs/stage1_creation_only_20241013-from_compre_only_20241013_ckpt25k/checkpoint-25000"
PRETRAIN_SD_MODEL_PATH="work_dirs/stage1_creation_only_20241013-from_compre_only_20241013_ckpt25k/checkpoint-25000"
PRETRAIN_DREAM_EMBEDDING_PATH="work_dirs/stage1_creation_only_20241013-from_compre_only_20241013_ckpt25k/checkpoint-25000"
RUN_NAME="stage2_20241015-from_creation_only_20241013_ckpt25k"
SEED=42
DEVICE_BATCH_SIZE=16
GRADIENT_ACC_STEPS=4

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
    python ${DIR}/projects/dreamllm/train.py \
        --config_file projects/dreamllm/configs/stage2/base.py \
        "data.comprehension_only=False" \
        "data.creation_only=False" \
        "data.datasets=['blip_laion', 'mmc4_core']" \
        "data.size_list=['2M', '2M']" \
        "data.datasets_approx_sizes=['15M', '2.8M']" \
        "data.datasets_init_kwargs.seed=${SEED}" \
        "data.datasets_init_kwargs.ignore_image=False" \
        "data.datasets_init_kwargs.instruction_image_num=0" \
        "data.skip_long_sample=True" \
        "data.max_image_length=30" \
        "data.disable_system_prompt=True" \
        "training.seed=${SEED}" \
        "training.save_steps=1000" \
        "training.save_total_limit=6" \
        "training.vit_llrd=False" \
        "training.llm_llrd=False" \
        "training.unfreeze_vit=False" \
        "training.per_device_train_batch_size=${DEVICE_BATCH_SIZE}" \
        "training.gradient_accumulation_steps=${GRADIENT_ACC_STEPS}" \
        "training.max_grad_norm=1.0" \
        "training.num_train_epochs=1" \
        "training.output_dir='./work_dirs/${RUN_NAME}/'" \
        "training.learning_rate=2e-5" \
        "training.fsdp=''" \
        "training.deepspeed=projects/configs/deepspeed/stage2_bf16.json" \
        "training.run_project='pretrain'" \
        "training.run_name='${RUN_NAME}'" \
        "training.report_to=['wandb']" \
        "model.loss_weight_vm=1.0" \
        "model.loss_weight_lm=10.0" \
        "model.diffusion_bs=0" \
        "model.model_max_length=2048" \
        "model.model_name_or_path=${MODEL_NAME_OR_PATH}" \
        "model.plugins_config_init_kwargs.clip_vision_embedding.pretrained_model_name_or_path='${PRETRAIN_CLIP_MODEL_PATH}'" \
        "model.plugins_config_init_kwargs.clip_vision_embedding.clip_vision_model_name_or_path='${MODEL_CLIP_PATH}'" \
        "model.plugins_config_init_kwargs.stable_diffusion_head.pretrained_model_name_or_path='${PRETRAIN_SD_MODEL_PATH}'" \
        "model.plugins_config_init_kwargs.stable_diffusion_head.diffusion_name_or_path='${MODEL_SD_PATH}'" \
        "model.plugins_config_init_kwargs.stable_diffusion_head.freeze_unet=True" \
        "model.plugins_config_init_kwargs.stable_diffusion_head.random_flip=False" \
        "model.plugins_config_init_kwargs.dream_embedding.pretrained_model_name_or_path='${PRETRAIN_DREAM_EMBEDDING_PATH}'" 2>&1 | tee "./work_dirs/${RUN_NAME}/training-mpirun.log"
