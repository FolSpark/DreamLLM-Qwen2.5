import os

import cv2
import numpy as np
import PIL.Image
import torch
from accelerate.utils import set_seed
from huggingface_hub import snapshot_download
from transformers import LlamaTokenizer

from omni.constants import MODEL_ZOOS
from omni.models.dreamllm.configuration_dreamllm import DreamLLMConfig
from omni.models.dreamllm.modeling_dreamllm import DreamLLMForCausalMLM as AutoModelForCausalLM
from omni.utils.image_utils import load_image, save_image
from omni.utils.profiler import FunctionProfiler

set_seed(1234, device_specific=False)

output_dir = "samples/images/stage3_20241029/"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# This is for DreamLLM-Controlnet, which is not fully supported at the present
# download an image
# image = load_image("https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png")
# image = np.array(image)

# # get canny image
# image = cv2.Canny(image, 100, 200)
# image = image[:, :, None]
# image = np.concatenate([image, image, image], axis=2)
# canny_image = PIL.Image.fromarray(image)

# NOTE: download the model to the specified local directory
# FIXME: directly download to the huggingface cache would possibly lead to wrong model loading (bug)
model_name_or_path = "work_dirs/stage3_20241029-from_stage2_20241016"
# model_name_or_path = "models/dreamllm-7b-chat-v1.0"      # stage1_creation_only_20241114_3B_mlp
# snapshot_download("RunpeiDong/dreamllm-7b-chat-aesthetic-v1.0", local_dir=model_name_or_path)

local_files_only = True

tokenizer = LlamaTokenizer.from_pretrained(
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
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    # .to(dtype=torch.float16).cuda()
# config.update_plugins(
#     dict(
#         _class_=ControlNetHead,
#         _name_="controlnet_head",
#         _plugin_type_="head",
#         controlnet_model_name_or_path=MODEL_ZOOS["lllyasviel/control_v11p_sd15_canny"],
#         diffusion_name_or_path=MODEL_ZOOS["runwayml/stable-diffusion-v1-5"],
#         pretrained_model_name_or_path=model_name_or_path,
#         local_files_only=True,
#     )
# )
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
    
    # # "(best quality,extremely detailed) Cafe Latte in Round Red Cup and Saucer",
    # # "Cafe Latte in Round Red Cup and Saucer",
    # # "Trail: Empire Link, Park City, Utah. Rider: The man himself, Chips Chippendale of Singletrack Magazine. Photo: Jeff.",
    # # "recently married couple characters vector illustration design",
    # # "Grilled salmon and tomato, lemon, rosemary on the wooden background.",
    # # "At the Seashore by Edward Henry Potthast",
    # # "Image of OUR LADY OF GRACE TRYPTYCH",
    # # "1965 ford mustang coupe",
    # # "Stoli-bloody-mary-newest"
    # "(best quality,extremely detailed) Grilled salmon and tomato, lemon, rosemary on the wooden background.",
    # "Pikachu wearing a gray hat, standing on a wooden floor.",
    # "The image shows a stylized, animated character with large, expressive eyes and a simple, cartoonish design. The character has red hair with a bow on top, a small nose, and a smiling mouth. The character is wearing a striped top with a pink and white color scheme. The background is a solid, dark red color.",
    # "The image shows an animated character with a joyful expression, wearing a hooded jacket with the hood up, revealing only the eyes. The character has blonde hair and is smiling broadly, with their mouth open in a cheerful manner. The background suggests an indoor setting with a television screen and what appears to be a wall with decorative elements. The character's pose and expression convey a sense of happiness or excitement.",
    # "A cartoon character wearing a blue coat with a hood, smiling and emitting a cloud of steam.",
    # "The image features an illustration of a character resembling Pepe the Frog, a popular internet meme, depicted in a stylized manner. The character is shown with a serious expression and is holding a handgun to its mouth, suggesting a sense of danger or threat. The background is a simple, solid color, which helps to focus attention on the character. ",
    # "The image shows a stylized, cartoon-like depiction of a character with a very expressive face. The character has a yellow body with orange highlights, a red mouth open wide, and blue eyes with a surprised or shocked expression. The character's hands are raised to the sides of the head, with fingers splayed out as if in a gesture of confusion or frustration. The overall impression is one of a strong emotional reaction, possibly to a sudden or unexpected event."
]

# positive_prompt = "(best quality,extremely detailed)"
# prompt = [f"{positive_prompt} {p}" for p in prompt]
negative_prompt = "ugly,duplicate,morbid,mutilated,tranny,mutated hands,poorly drawn hands,blurry,bad anatomy,bad proportions,extra limbs,cloned face,disfigured,missing arms,extra legs,mutated hands,fused fingers,too many fingers,unclear eyes,lowers,bad anatomy,bad hands,text,error,missing fingers,extra digit,fewer digits,cropped,worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,username,blurry,bad feet,gettyimages"
negative_prompt = [negative_prompt] * len(prompt)
# negative_prompt = None

images = model.stable_diffusion_pipeline(
    tokenizer=tokenizer,
    prompt = prompt,
    width=512,
    height=512,
    # prompt=["A photo of a cat"]*10,
    # prompt=["a photo of a cat, cat, cat, cat, cat, cat, cat, cat, cat, cat, cat, cat"]*10,
    # negative_prompt=negative_prompt,
    guidance_scale=7.5,
    num_inference_steps=100,
)
for i, image in enumerate(images):
    save_image(images[i], path=output_dir + f"0_test_{i}.jpg", force_overwrite=True)

# images = model.controlnet_pipeline(
#     tokenizer=tokenizer,
#     prompt=prompt,
#     image=canny_image,
#     guidance_scale=7.5,
#     num_inference_steps=100,
# )

# for i, image in enumerate(images):
#     save_image(images[i], path=f"{i}.jpg", force_overwrite=False)
