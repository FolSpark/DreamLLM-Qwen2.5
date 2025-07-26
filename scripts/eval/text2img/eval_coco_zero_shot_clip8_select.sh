#!/bin/bash
# out_dir=$1
# model_name_or_path=$2
set -e
set -x
export PYTHONPATH=.
# export CUDA_VISIBLE_DEVICES=2,4,5,6,7
NUM_PROCESSES=8

out_dir="vicuna"
model_name_or_path="work_dirs/DreamLLM-Vicuna"
# out_dir="1.5B_stage1_20241127"
# model_name_or_path="work_dirs/stage1_creation_only_20241127"

COCO_ROOT="datasets/coco_fid_files/annotations_trainval2014/annotations"
n_samples=30000
batch_size_per_device=10

if [ ! -d "./samples/${out_dir}" ]; then
    mkdir -p ./samples/${out_dir}
fi

seed=(42 43 44 45 46 47 48 49)
# 抽取30000个样本
# PYTHONPATH=path2dreamllm/ \
accelerate launch --num_processes ${NUM_PROCESSES} omni/eval/text2img/ddp_sample_coco.py \
--type caption \
--coco_root ${COCO_ROOT} \
--ann_file captions_val2014.json \
--n_samples ${n_samples} \
--batch_size_per_device ${batch_size_per_device} \
--out_data_info_path "samples/${out_dir}/data_info.json" \


for ((i=0; i<=0; i++))
do
    # PYTHONPATH=path2dreamllm/ \
    accelerate launch --num_processes=${NUM_PROCESSES} omni/eval/text2img/ddp_sample_coco.py \
    --type dreamllm \
    --model_name_or_path $model_name_or_path \
    --diffusion_model_name_or_path models/stable-diffusion-2-1-base \
    --clip_model_name_or_path models/clip-vit-large-patch14 \
    --coco_root ${COCO_ROOT} \
    --ann_file captions_val2014.json \
    --n_samples ${n_samples}  \
    --batch_size_per_device ${batch_size_per_device} \
    --out_dir "samples/${out_dir}/seed${seed[i]}" \
    --num_inference_steps 100 \
    --guidance_scale 2 \
    --local_files_only \
    --seed ${seed[$i]} > samples/${out_dir}/seed${seed[i]}.log
# done


# for ((i=6; i<=7; i++))
# do
    python -m pytorch_fid ${COCO_ROOT}/fid_stats_mscoco256_val.npz samples/${out_dir}/seed${seed[i]} > samples/${out_dir}/seed${seed[i]}_fid.log 2>&1

    tail -n 1 samples/${out_dir}/seed${seed[i]}_fid.log
done



# PYTHONPATH=path2dreamllm/ \
# accelerate launch --num_processes ${NUM_PROCESSES} omni/eval/text2img/ddp_sample_coco.py \
#     --type select \
#     --data_info_path "samples/${out_dir}/data_info.json" \
#     --clip_for_similarity_model_name_or_path models/clip-vit-large-patch14 \
#     --base_dir samples/${out_dir} \
#     --dirs_name seed \
#     --out_dir final_res



# python -m pytorch_fid ${COCO_ROOT}/fid_stats_mscoco256_val.npz samples/${out_dir}/final_res  > samples/${out_dir}/final_fid.log 2>&1
