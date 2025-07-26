import os
import json
from omni.constants import MODEL_ZOOS
from omni.models.dreamllm_qwen_sd3.configuration_dreamllm import ConfigAndInitKwargs, create_config_init_kwargs
from omni.models.dreamllm_qwen_sd3.modeling_plugins import CLIPVisionEmbedding, DreamEmbedding, StableDiffusion3Head
from omni.utils.import_utils import is_volc_mlplatform_available

if is_volc_mlplatform_available():
    local_files_only = True
else:
    local_files_only = False

model_name_or_path = 'models/Qwen2.5-1.5B-Instruct'
with open(os.path.join(model_name_or_path, "config.json")) as f:
    model_config = json.load(f)
hidden_size = model_config["hidden_size"]

# NOTE: don't forget add `embed_hidden_size` kwargs
dream_embedding_config_init_kwargs = create_config_init_kwargs(
    ConfigAndInitKwargs(
        _class_=DreamEmbedding,
        _name_="dream_embedding",
        _plugin_type_="embedding",
        pretrained_model_name_or_path=None,
        num_dream_queries=64,
        embed_hidden_size=hidden_size,
        freeze_dream_queries=False,
    )
)
clip_vision_embedding_config_init_kwargs = create_config_init_kwargs(
    ConfigAndInitKwargs(
        _class_=CLIPVisionEmbedding,
        _name_="clip_vision_embedding",
        _plugin_type_="embedding",
        projector_type="linear",
        projector_depth=1,
        clip_vision_model_name_or_path=MODEL_ZOOS["openai/clip-vit-large-patch14"],
        pretrained_model_name_or_path=None,
        embed_hidden_size=hidden_size,
        use_additional_post_layernorm=False,
        select_layer=-2,
        freeze_clip_vision_model=True,
        freeze_embedding_layers=True,
        freeze_projector=False,
        local_files_only=local_files_only,
    )
)
stable_diffusion3_head_config_init_kwargs = create_config_init_kwargs(
    ConfigAndInitKwargs(
        _class_=StableDiffusion3Head,
        _name_="stable_diffusion3_head",
        _plugin_type_="head",
        projector_type="linear",
        projector_depth=1,
        diffusion_name_or_path=MODEL_ZOOS["stabilityai/stable-diffusion-xl-base-1.0"],
        pretrained_model_name_or_path=None,
        embed_hidden_size=hidden_size,
        freeze_vae=True,
        freeze_transformer=True,
        freeze_projector=False,
        local_files_only=local_files_only,
    )
)
