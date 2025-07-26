# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional, Tuple, Type

import torch
from transformers import AutoConfig, BitsAndBytesConfig, PreTrainedTokenizerBase
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.tokenization_auto import get_tokenizer_config

from swift.llm import TemplateType
from swift.utils import get_device_count, get_dist_setting, get_env_args, get_logger
from swift.llm.model.constant import LLMModelType, MLLMModelType, RMModelType
from swift.llm.model.model_arch import ModelArch
from swift.llm.model.patcher import patch_fixed_device, patch_output_clone, patch_output_to_input_device
from swift.llm.model.register import (Model, ModelGroup, ModelMeta, get_model_tokenizer_multimodal, get_model_tokenizer_reward_model,
                        get_model_tokenizer_with_flash_attn, register_model)
from swift.llm.model.utils import AttnImpl, ModelInfo, use_submodel_func


import template

logger = get_logger()
dtype_mapping = {torch.float16: 'fp16', torch.bfloat16: 'bf16', torch.float32: 'fp32'}


def log_trainable_parameters(model):
    # TODO: make it more accurate
    total_params = sum([p.numel() for p in model.parameters()])
    train_params = sum([p.numel() if p.requires_grad else 0 for p in model.parameters()])
    logger.info(f">> Total params: {total_params / 1.e6}M")
    logger.info(f">> Train params: {train_params / 1.e6}M, Ratio {train_params / total_params * 100.:.2f}%")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"Parameter: {name}, dtype: {param.dtype}, requires_grad: {param.requires_grad} ")
    #     logger.info(f"Parameter: {name}, dtype: {param.dtype}, requires_grad: {param.requires_grad} ")



def get_model_tokenizer_qwen(model_dir: str,
                             model_info: ModelInfo,
                             model_kwargs: Dict[str, Any],
                             load_model: bool = True,
                             model_config=None,
                             **kwargs):
    if model_config is None:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if model_info.torch_dtype is not None:
        k_true = dtype_mapping[model_info.torch_dtype]
        for k in dtype_mapping.values():
            setattr(model_config, k, k == k_true)

    quantization_config = model_kwargs.get('quantization_config')
    if not isinstance(quantization_config, BitsAndBytesConfig):
        # not bnb quant
        model_config.torch_dtype = None
    use_flash_attn = AttnImpl.to_use_flash_attn(kwargs.pop('attn_impl', None), 'auto')
    model_config.use_flash_attn = use_flash_attn
    kwargs['model_config'] = model_config
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    try:
        # fix mp+ddp bug
        model.transformer.registered_causal_mask = model.transformer.registered_causal_mask.cuda()
        logger.info('registered_causal_mask to cuda')
    except AttributeError:
        pass
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.eod_id
    return model, tokenizer


def fix_qwen_inplace_bug(model) -> None:
    # qwen-vl, qwen-audio
    first_drop = model.transformer.drop
    if first_drop.p == 0.:
        # fix in-place operation bug
        patch_output_clone(first_drop)


def _qwen_vl_visual_block_forward(
    self,
    q_x: torch.Tensor,
    k_x: Optional[torch.Tensor] = None,
    v_x: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
):
    k_x = self.ln_1_kv(k_x) if hasattr(self, 'ln_1_kv') and k_x is not None else None
    v_x = self.ln_1_kv(v_x) if hasattr(self, 'ln_1_kv') and v_x is not None else None

    x = q_x + self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
    z = self.mlp(self.ln_2(x))
    x = x.to(z.device) + z  # FIX
    return x


def patch_qwen_vl_utils(vision_process):
    if hasattr(vision_process, '_patch'):
        return
    for key in [
            'image_factor', 'min_pixels', 'max_pixels', 'max_ratio', 'video_min_pixels', 'video_max_pixels',
            'video_total_pixels', 'frame_factor', 'fps', 'fps_min_frames', 'fps_max_frames'
    ]:
        type_func = float if key == 'fps' else int
        setattr(vision_process, key.upper(), get_env_args(key, type_func, getattr(vision_process, key.upper())))
    _read_video_decord = vision_process._read_video_decord

    def _new_read_video_decord(ele: dict):
        from swift.llm import load_file
        ele['video'] = load_file(ele['video'])
        return _read_video_decord(ele)

    vision_process.VIDEO_READER_BACKENDS['decord'] = _new_read_video_decord
    vision_process._patch = True


def get_model_tokenizer_qwen2_vl(*args, **kwargs):
    from transformers import Qwen2VLForConditionalGeneration
    kwargs['automodel_class'] = kwargs['automodel_class'] or Qwen2VLForConditionalGeneration
    model, tokenizer = get_model_tokenizer_multimodal(*args, **kwargs)
    if model is not None and hasattr(model.model, 'embed_tokens'):
        patch_output_clone(model.model.embed_tokens)
        patch_output_to_input_device(model.model.embed_tokens)

    from qwen_vl_utils import vision_process
    patch_qwen_vl_utils(vision_process)
    return model, tokenizer


def get_model_tokenizer_qwen2_5_vl(*args, **kwargs):
    from transformers import Qwen2_5_VLForConditionalGeneration
    kwargs['automodel_class'] = kwargs['automodel_class'] or Qwen2_5_VLForConditionalGeneration
    return get_model_tokenizer_qwen2_vl(*args, **kwargs)


# register_model(
#     ModelMeta(
#         MLLMModelType.qwen2_5_vl,
#         [
#             ModelGroup([
#                 Model('Qwen/Qwen2.5-VL-3B-Instruct', 'Qwen/Qwen2.5-VL-3B-Instruct'),
#             ])
#         ],
#         TemplateType.qwen2_5_vl,
#         get_model_tokenizer_qwen2_5_vl,
#         model_arch=ModelArch.qwen2_vl,
#         architectures=['Qwen2_5_VLForConditionalGeneration'],
#         requires=['transformers>=4.49', 'qwen_vl_utils>=0.0.6', 'decord'],
#         tags=['vision', 'video']))


def get_model_tokenizer_dreamllm_qwen2(*args, **kwargs):
    # import ipdb
    # ipdb.set_trace()
    from omni.models.dreamllm_qwen2.configuration_dreamllm import DreamLLMConfig
    from omni.models.dreamllm_qwen2.modeling_dreamllm import DreamLLMForCausalMLM
    # from transformers import AutoProcessor
    from omni.models.dreamllm_qwen2.processing_dreamllm import DreamLLMProcessor
    kwargs['automodel_class'] = kwargs['automodel_class'] or DreamLLMForCausalMLM
    # model, tokenizer = get_model_tokenizer_multimodal(*args, **kwargs)
    tokenizer = DreamLLMProcessor.from_pretrained(
        args[0], trust_remote_code=True,)
    # model = DreamLLMForCausalMLM.from_pretrained(
    #     args[0], trust_remote_code=True, tokenizer=tokenizer, **kwargs)
    config = DreamLLMConfig.from_pretrained(
        args[0],
        local_files_only=True,
    )
    model = DreamLLMForCausalMLM.from_pretrained(
        args[0],
        trust_remote_code=True,
        config=config,
        tokenizer=tokenizer.tokenizer,
        local_files_only=True,
        cache_dir=None,
        reset_plugin_model_name_or_path=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        # **kwargs
    )
    model.to('cuda', dtype=torch.bfloat16)
    model.stable_diffusion_head.to('cpu')
    # log_trainable_parameters(model)
    
    return model, tokenizer


register_model(
    ModelMeta(
        "dreamllm",
        [
            ModelGroup([
                Model('DreamLLMForCausalMLM')
            ])
        ],
        "dreamllm",
        get_model_tokenizer_dreamllm_qwen2,
        model_arch='dreamllm',
        architectures=['DreamLLMForCausalMLM'],
        is_multimodal=True,
    )
)