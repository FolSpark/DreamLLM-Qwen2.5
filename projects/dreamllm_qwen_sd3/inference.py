import os

import cv2
import numpy as np
import PIL.Image
import torch
from accelerate.utils import set_seed
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
# from omni.models.dreamllm_qwen_sd3.tokenization_dreamllm import DreamLLMTokenizer as AutoTokenizer

from omni.constants import MODEL_ZOOS
from omni.models.dreamllm_qwen_sd3.configuration_dreamllm import DreamLLMConfig
from omni.models.dreamllm_qwen_sd3.modeling_dreamllm import DreamLLMForCausalMLM as AutoModelForCausalLM
from omni.utils.image_utils import load_image, save_image
from omni.utils.profiler import FunctionProfiler

set_seed(42, device_specific=False)

output_dir = "samples/images_qwen_sd3/sd3.5_stage1_creation_only_20241216/"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# model_name_or_path = "models/dreamllm-7b-chat-aesthetic-v1.0"
# model_name_or_path = "work_dirs/stage1_creation_only_20241130_from_sft_1122"
model_name_or_path = "work_dirs/sd3.5_stage1_creation_only_20241216"


local_files_only = True

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    local_files_only=local_files_only,
    padding_side="right",
)
 
config = DreamLLMConfig.from_pretrained(
    model_name_or_path,
    local_files_only=local_files_only,
)
with FunctionProfiler("AutoModelForCausalLM.from_pretrained"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        tokenizer=tokenizer,
        config=config,
        local_files_only=local_files_only,
        cache_dir=None,
        reset_plugin_model_name_or_path=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )


model.eval()
model = torch.compile(model)

prompt = [
    "A photo of a black dog.",
    "A polar bear in the forest.",
    "An astronaut riding a horse.",
    "An espresso machine that makes coffee, art station.",
    "orange fruit",
    "(painting,fantasy) A tiger with a suit, fantasy painting.",
    "A polar bear by the river.",
    "A photo of a frog reading the newspaper named “Toaday” written on it.",
    "A panda is reading the newspaper.",
    
    # "(best quality,extremely detailed) Cafe Latte in Round Red Cup and Saucer",
    # "Cafe Latte in Round Red Cup and Saucer",
    # "Trail: Empire Link, Park City, Utah. Rider: The man himself, Chips Chippendale of Singletrack Magazine. Photo: Jeff.",
    # "recently married couple characters vector illustration design",
    # "Grilled salmon and tomato, lemon, rosemary on the wooden background.",
    # "At the Seashore by Edward Henry Potthast",
    # "Image of OUR LADY OF GRACE TRYPTYCH",
    # "1965 ford mustang coupe",
    # "Stoli-bloody-mary-newest"
    "(best quality,extremely detailed) Grilled salmon and tomato, lemon, rosemary on the wooden background.",
]


# positive_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\nPlease generate a photo:"
# generation_prompt = "<|im_end|><|im_start|>assistant\n"
# prompt = [f"{positive_prompt} {p} {generation_prompt}" for p in prompt]

positive_prompt = "(best quality,extremely detailed)"
prompt = [f"{positive_prompt} {p}" for p in prompt]
negative_prompt = "ugly,duplicate,morbid,mutilated,tranny,mutated hands,poorly drawn hands,blurry,bad anatomy,bad proportions,extra limbs,cloned face,disfigured,missing arms,extra legs,mutated hands,fused fingers,too many fingers,unclear eyes,lowers,bad anatomy,bad hands,text,error,missing fingers,extra digit,fewer digits,cropped,worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,username,blurry,bad feet,gettyimages"
negative_prompt = [negative_prompt] * len(prompt)
# negative_prompt = None

images = model.stable_diffusion3_pipeline(
    tokenizer=tokenizer,
    width=512,
    height=512,
    prompt = prompt,
    # prompt=["A photo of a cat"]*10,
    # prompt=["a photo of a cat, cat, cat, cat, cat, cat, cat, cat, cat, cat, cat, cat"]*10,
    negative_prompt=negative_prompt,
    guidance_scale=7,
    num_inference_steps=28,
)

for i, image in enumerate(images):
    save_image(images[i], path=output_dir + f"512_test_{i}.jpg", force_overwrite=True)

