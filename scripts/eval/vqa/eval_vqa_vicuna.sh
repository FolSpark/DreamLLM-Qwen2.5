# set -e
set -x
export PYTHONPATH=.
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
num_gpus=8

# path2model="work_dirs/sft_llava_20241105"
# out_dir="eval_results_qwen_sft_llava_20241105"
path2model="work_dirs/vicuna/our_train/stage3_20241029-from_stage2_20241016"
out_dir="eval_results_vicuna_stage3_20241029-from_stage2_20241016"
# path2model="models/dreamllm-7b-chat-v1.0"
# out_dir="eval_results_dreamllm-7b-chat-v1.0"

if [ ! -d "./samples/${out_dir}" ]; then
    mkdir -p ./samples/${out_dir}
fi

# 注意这里的system_prompt是在user之后的，真正的system_prompt是代码中的模版
# post_prompt是提示在assistant之后的

# -------------------- Accuracy Metrics Tasks --------------------
# CoCO Caption test         # TODO
python -m omni.eval.vqa_vicuna.eval_dreamllm \
    --model_name ${path2model} \
    --gtfile_path ./datasets/eval_vqa/omni_comprehension_eval_format_files/OMNI_format_coco_caption_test.json \
    --image_path ./datasets/eval_vqa/COCO/val2014 \
    --out_path ./samples/${out_dir}/coco_cap \
    --num-chunks ${num_gpus} \
    --datatype COCO-Captions \
    --prompt 'Please summarize object in one sentence.' \
    --post_prompt "The image depicts: " \
    --evaltype "all" \
    --img_aug none \
    --beamsearch "True_short" \
    --system_prompt "Based on the image, give the image caption briefly." > ./samples/${out_dir}/CoCO_Caption_test.log 2>&1

# NoCaps
# python -m omni.eval.vqa_vicuna.eval_dreamllm \
#     --model_name ${path2model} \
#     --gtfile_path ./datasets/eval_vqa/omni_comprehension_eval_format_files/OMNI_format_nocaps_caption.json \
#     --image_path ./datasets/eval_vqa/nocaps/images \
#     --out_path ./samples/${out_dir}/nocaps \
#     --num-chunks ${num_gpus} \
#     --datatype NoCaps \
#     --prompt 'Please summarize object briefly in one sentence within 5 words.' \
#     --evaltype "all" \
#     --img_aug none \
#     --beamsearch True \

# Image2Paragraph           # TODO
python -m omni.eval.vqa_vicuna.eval_dreamllm \
    --model_name ${path2model} \
    --gtfile_path ./datasets/eval_vqa/omni_comprehension_eval_format_files/OMNI_format_image_paragraph.json \
    --image_path ./datasets/eval_vqa/paragraph-captioning/VG_100K \
    --out_path ./samples/${out_dir}/image_para \
    --num-chunks ${num_gpus} \
    --datatype Image-Paragraph \
    --prompt 'Please describe the image in detail.' \
    --post_prompt "The image depicts: " \
    --system_prompt "Based on the image, please describe the image in detail." \
    --evaltype "all" \
    --img_aug none \
    --beamsearch True > ./samples/${out_dir}/Image2Paragraph.log 2>&1

# VQAv2                 # TODO
python -m omni.eval.vqa_vicuna.eval_dreamllm \
    --model_name ${path2model} \
    --gtfile_path ./datasets/eval_vqa/omni_comprehension_eval_format_files/OMNI_format_VQAv2_sub_val.json \
    --image_path ./datasets/eval_vqa/COCO/val2014 \
    --out_path ./samples/${out_dir}/vqav2 \
    --num-chunks ${num_gpus} \
    --datatype VQAv2 \
    --img_aug none \
    --beamsearch True \
    --evaltype "all" \
    --system_prompt "Based on the image, please answer the question." \
    --clip True \
    --prompt "Please provide an accurate answer within one word or phrase." \
    --post_prompt "The answer is:" > ./samples/${out_dir}/VQAv2.log 2>&1

# VQAv2-test-dev
# python -m omni.eval.vqa_vicuna.eval_dreamllm \
#     --model_name ${path2model} \
#     --gtfile_path ./datasets/eval_vqa/omni_comprehension_eval_format_files/OMNI_format_VQAv2_test_dev.json \
#     --image_path ./datasets/eval_vqa/VQAv2/test2015 \
#     --out_path ./samples/${out_dir}/vqav2 \
#     --num-chunks ${num_gpus} \
#     --datatype VQAv2 \
#     --img_aug none \
#     --beamsearch True \
#     --system_prompt "Based on the image, please answer the question." \
#     --prompt "Please provide an accurate answer within one word." \
#     --post_prompt "The short answer '\(one word\)' is:" \
#     --clip True
    # --post_prompt "The answer is:" \

# TextVQA                       # TODO
python -m omni.eval.vqa_vicuna.eval_dreamllm \
    --model_name ${path2model}  \
    --gtfile_path ./datasets/eval_vqa/omni_comprehension_eval_format_files/OMNI_format_TextVQA_val.json \
    --image_path ./datasets/eval_vqa/TextVQA/train_images \
    --out_path ./samples/${out_dir}/textvqa \
    --num-chunks ${num_gpus} \
    --datatype TextVQA \
    --img_aug none \
    --beamsearch True \
    --evaltype "all" \
    --system_prompt "Based on the image, please answer the question." \
    --prompt "Please provide an accurate answer within one word." \
    --post_prompt "The answer is:" \
    --clip True > ./samples/${out_dir}/TextVQA.log 2>&1

# MMBench                       # TODO
python -m omni.eval.vqa_vicuna.eval_dreamllm \
    --model_name ${path2model} \
    --gtfile_path ./datasets/eval_vqa/omni_comprehension_eval_format_files/mmbench_dev_20230712.tsv \
    --image_path none \
    --out_path ./samples/${out_dir}/mmbench \
    --num-chunks ${num_gpus} \
    --datatype mmbench \
    --img_aug none \
    --beamsearch True \
    --evaltype "all" \
    --prompt "Please provide an accurate and detailed answer." \
    --system_prompt "This is an exam, please answer according to the image, hint and question." > ./samples/${out_dir}/MMBench.log 2>&1

# MM-Vet (Submit to https://huggingface.co/spaces/whyu/MM-Vet_Evaluator)        # TODO
python -m omni.eval.vqa_vicuna.eval_dreamllm \
    --gtfile_path ./datasets/eval_vqa/MM-VET/MMGPT_mm-vet.json \
    --model_name ${path2model} \
    --image_path ./datasets/eval_vqa/MM-VET/images \
    --out_path ./samples/${out_dir}/mm-vet \
    --num-chunks ${num_gpus} \
    --datatype mmvet \
    --img_aug none \
    --beamsearch True \
    --evaltype "all" \
    --prompt "Please provide an accurate, detailed and comprehensive answer." \
    --system_prompt "This is an exam, please answer according to the image and question." > ./samples/${out_dir}/MM-Vet.log 2>&1

# # VizWizVQA                 # TODO
python -m omni.eval.vqa_vicuna.eval_dreamllm \
    --model_name ${path2model} \
    --gtfile_path ./datasets/eval_vqa/omni_comprehension_eval_format_files/OMNI_format_VizWizVQA_val.json \
    --image_path ./datasets/eval_vqa/VizWiz-VQA/val \
    --out_path ./samples/${out_dir}/VizWizVQA \
    --num-chunks ${num_gpus} \
    --datatype VizWizVQA \
    --img_aug none \
    --beamsearch True > ./samples/${out_dir}/VizWizVQA.log 2>&1

# VizWizVQA-test
# python -m omni.eval.vqa_vicuna.eval_dreamllm \
#     --model_name ${path2model} \
#     --gtfile_path ./datasets/eval_vqa/omni_comprehension_eval_format_files/OMNI_format_VizWizVQA_test.json \
#     --image_path ./datasets/eval_vqa/VizWiz-VQA/test \
#     --out_path ./samples/${out_dir}/VizWizVQA \
#     --num-chunks ${num_gpus} \
#     --datatype VizWizVQA \
#     --img_aug none \
#     --beamsearch True \
#     --prompt "Please provide an accurate answer within one word." \
#     --post_prompt "The answer is:" \
#     --system_prompt "Based on the image, please answer the question." \
#     --clip True

# OKVQA                     # TODO
python -m omni.eval.vqa_vicuna.eval_dreamllm \
    --model_name ${path2model} \
    --gtfile_path ./datasets/eval_vqa/eval_format_files/OMNI_format_OKVQA_val.json \
    --image_path ./datasets/eval_vqa/COCO/val2014 \
    --out_path ./samples/${out_dir}/OKVQA/ \
    --num-chunks ${num_gpus} \
    --datatype OKVQA \
    --img_aug none \
    --beamsearch True \
    --prompt "Please provide an accurate and brief answer within one word." \
    --post_prompt "The short answer '\(one word\)' is:" \
    --evaltype "all" \
    --clip True > ./samples/${out_dir}/OKVQA.log 2>&1


# -------------------- ANLS Metrics Tasks --------------------

# DocVQA
# python -m omni.eval.vqa_vicuna.eval_dreamllm \
#     --model_name ${path2model} \
#     --gtfile_path ./datasets/eval_vqa/omni_comprehension_eval_format_files/OMNI_format_DocVQA_val.json \
#     --image_path ./datasets/eval_vqa/DocVQA/val/documents \
#     --out_path ./samples/${out_dir}/DocVQA \
#     --num-chunks ${num_gpus} \
#     --datatype DocVQA \
#     --img_aug padding_square_resize \
#     --beamsearch True \

# InfographicVQA
# python -m omni.eval.vqa_vicuna.eval_dreamllm \
#     --model_name ${path2model} \
#     --gtfile_path ./datasets/eval_vqa/omni_comprehension_eval_format_files/OMNI_format_InfographicsVQA_val.json \
#     --image_path ./datasets/eval_vqa/InfographicsVQA/infographicVQA_val_v1.0_images \
#     --out_path ./samples/${out_dir}/InfographicVQA \
#     --num-chunks ${num_gpus} \
#     --datatype InfographicVQA \
#     --img_aug padding_square_resize \
#     --beamsearch True \


# -------------------- OCR Tasks ---------------------
# python -m omni.eval.vqa_vicuna.eval_dreamllm \
#     --model_name ${path2model} \
#     --gtfile_path ./datasets/eval_vqa/omni_comprehension_eval_format_files/OMNI_format_OCR_val.json \
#     --image_path ./datasets/eval_vqa/OCR/IC13_857 \
#     --out_path ./samples/${out_dir}/OCR \
#     --num-chunks ${num_gpus} \
#     --datatype OCR \
#     --img_aug padding_square_resize \
#     --beamsearch True \


# -------------------- POPE Tasks --------------------
# python -m omni.eval.vqa_vicuna.eval_dreamllm \
    --model_name ${path2model} \
    --gtfile_path ./datasets/eval_vqa/omni_comprehension_eval_format_files/OMNI_format_coco_pope_random.json \
    --image_path ./datasets/eval_vqa/COCO/val2014 \
    --out_path ./samples/${out_dir}/POPE_random \
    --num-chunks ${num_gpus} \
    --img_aug none \
    --beamsearch False \
    --datatype POPE_random \
    --evaltype all \
    --system_prompt "Based on the image, please objectively and accurately indicate whether the object exists." \
    --post_prompt "The answer is:" > ./samples/${out_dir}/POPE_random.log 2>&1

python -m omni.eval.vqa_vicuna.eval_dreamllm \
    --model_name ${path2model} \
    --gtfile_path ./datasets/eval_vqa/omni_comprehension_eval_format_files/OMNI_format_coco_pope_popular.json \
    --image_path ./datasets/eval_vqa/COCO/val2014 \
    --out_path ./samples/${out_dir}/POPE_popular \
    --num-chunks ${num_gpus} \
    --img_aug none \
    --beamsearch False \
    --datatype POPE_popular \
    --system_prompt "Based on the image, please objectively and accurately indicate whether the object exists." \
    --post_prompt "The answer is:" > ./samples/${out_dir}/POPE_popular.log 2>&1

python -m omni.eval.vqa_vicuna.eval_dreamllm \
    --model_name ${path2model} \
    --gtfile_path ./datasets/eval_vqa/omni_comprehension_eval_format_files/OMNI_format_coco_pope_adversarial.json \
    --image_path ./datasets/eval_vqa/COCO/val2014 \
    --out_path ./samples/${out_dir}/POPE_adversarial \
    --num-chunks ${num_gpus} \
    --img_aug none \
    --beamsearch False \
    --datatype POPE_adversarial \
    --system_prompt "Based on the image, please objectively and accurately indicate whether the object exists." \
    --post_prompt "The answer is:" > ./samples/${out_dir}/POPE_adversarial.log 2>&1

