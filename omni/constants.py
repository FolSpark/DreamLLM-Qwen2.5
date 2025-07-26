from omni.utils.import_utils import is_volc_mlplatform_available

DEBUG = False
# DEBUG = True
LOGDIR = "./work_dirs/"
USE_HF_LOCAL_FILES = True

HUGGINGFACE_CO_RESOLVE_ENDPOINT = "https://huggingface.co"

# fmt: off
MODEL_ZOOS = {
    "decapoda-research/llama-7b-hf"           : "huggingface_cache/hub/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348",
    "meta-llama/Llama-2-7b-hf"                : "huggingface_cache/hub/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9",
    "meta-llama/Llama-2-7b-chat-hf"           : "huggingface_cache/hub/llama2-7b-chat",
    "meta-llama/Llama-2-70b-hf"               : "huggingface_cache/hub/models--meta-llama--Llama-2-70b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf"          : "huggingface_cache/hub/models--meta-llama--Llama-2-70b-hf",
    "lmsys/vicuna-7b-v1.1"                    : "huggingface_cache/hub/llama-vicuna-7b-v1.1",
    "lmsys/vicuna-7b-v1.3"                    : "huggingface_cache/hub/llama-vicuna-7b-v1.3",
    "lmsys/vicuna-7b-v1.5"                    : "huggingface_cache/hub/models--lmsys--vicuna-7b-v15",
    "lmsys/vicuna-13b-v1.3"                   : "huggingface_cache/hub/llama-vicuna-13b-v1.3",
    "lmsys/vicuna-33b-v1.3"                   : "huggingface_cache/hub/models--lmsys--vicuna-33b-v1.3",
    "stabilityai/stable-diffusion-2-1-base"   : "huggingface_cache/hub/models--stabilityai--stable-diffusion-2-1-base/snapshots/dcd3ee64f0c1aba2eb9e0c0c16041c6cae40d780",
    "stabilityai/stable-diffusion-xl-base-0.9": "huggingface_cache/hub/models--stabilityai--stable-diffusion-xl-base-0.9/snapshots/ccb3e0a2bfc06b2c27b38c54684074972c365258",
    "stabilityai/stable-diffusion-xl-base-1.0": "huggingface_cache/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/bf714989e22c57ddc1c453bf74dab4521acb81d8",
    "stabilityai/stable-diffusion-3.5-medium" : "models/stable-diffusion-3.5-medium",
    "madebyollin/sdxl-vae-fp16-fix"           : "huggingface_cache/hub/models--madebyollin--sdxl-vae-fp16-fix/snapshots/4df413ca49271c25289a6482ab97a433f8117d15",
    "openai/clip-vit-large-patch14"           : "models/clip-vit-large-patch14",
    "google/siglip-so400m-patch14-384"        : "models/siglip-so400m-patch14-384",
    "OpenGVLab/InternViT-300M-448px"          : "models/InternViT-300M-448px-V2_5",
    "Qwen/Qwen2.5-1.5B-Instruct"              : "models/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct"                : "models/Qwen2.5-7B-Instruct",
    
}
if not is_volc_mlplatform_available():
    MODEL_ZOOS = {key: key if "/" in key else value for key, value in MODEL_ZOOS.items()}
# fmt: on

MODEL_ZOOS["fid_weights"] = (
    "huggingface_cache/pt_inception-2015-12-05-6726825d.pth"
    if is_volc_mlplatform_available()
    else "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth",
)

# # model config
# ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION = True
# IGNORE_INDEX = -100
# MAX_TOKEN_LENGTH = 2048
# LLM_HIDDEN_DIM = 4096
# MM_HIDDEN_DIM = 1024
# LDM_HIDDEN_DIM = 1024
# REORDER_ATTENTION = False
# VISION_HIDDEN_DIM = 256
# NUM_DREAM_QUERIES = 64

# # special token
# DEFAULT_BOS_TOKEN = "<s>"
# DEFAULT_EOS_TOKEN = "</s>"
# DEFAULT_UNK_TOKEN = "<unk>"
# DEFAULT_PAD_TOKEN = "[PAD]"

# # additional special token
# DEFAULT_IMAGE_TOKEN = "<image>"
# DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
# DEFAULT_IMAGE_START_TOKEN = "<im_start>"
# DEFAULT_IMAGE_END_TOKEN = "<im_end>"

# DEFAULT_DREAM_TOKEN = "<dream>"
# DEFAULT_DREAM_START_TOKEN = "<dream_start>"
# DEFAULT_DREAM_END_TOKEN = "<dream_end>"

# # worker config
# CONTROLLER_HEART_BEAT_EXPIRATION = 30
# WORKER_HEART_BEAT_INTERVAL = 15


# qwen2.5-1.5B model config
ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION = True
IGNORE_INDEX = -100
MAX_TOKEN_LENGTH = 32768
LLM_HIDDEN_DIM = 1536
MM_HIDDEN_DIM = 1024
LDM_HIDDEN_DIM = 1024
REORDER_ATTENTION = False
VISION_HIDDEN_DIM = 256
NUM_DREAM_QUERIES = 64

# special token
# DEFAULT_BOS_TOKEN = "<s>"
# DEFAULT_EOS_TOKEN = "</s>"
# DEFAULT_UNK_TOKEN = "<unk>"
# DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_BOS_TOKEN = "<|im_start|>"
DEFAULT_EOS_TOKEN = "<|im_end|>"
# DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_PAD_TOKEN = "<|endoftext|>"

# additional special token
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<|image_pad|>"
DEFAULT_IMAGE_START_TOKEN = "<|vision_start|>"
DEFAULT_IMAGE_END_TOKEN = "<|vision_end|>"

DEFAULT_DREAM_TOKEN = "<dream>"
DEFAULT_DREAM_START_TOKEN = "<dream_start>"
DEFAULT_DREAM_END_TOKEN = "<dream_end>"

# worker config
CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15
