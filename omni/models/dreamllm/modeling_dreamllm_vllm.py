# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inference-only DreamLLM model compatible with HuggingFace weights."""
import numpy as np
import torch
import torch.backends.cuda
import torch.utils.checkpoint
from torch import nn
from PIL import Image
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Callable, Literal

from omni.utils.loguru import logger
from omni.utils.profiler import pretty_format
from omni.config.instantiate import deep_instantiate
from omni.models.dreamllm.configuration_dreamllm import DreamLLMConfig
from omni.models.dreamllm.tokenization_dreamllm import (
    DEFAULT_BOS_TOKEN,
    DEFAULT_DREAM_END_TOKEN,
    DEFAULT_DREAM_START_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_IMAGE_END_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
)

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig, ModelConfig, VisionLanguageConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, kv_cache_scales_loader)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput, SequenceData
from vllm.utils import is_hip, print_warning_once
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalData
from vllm.multimodal.image import ImagePixelData, get_dummy_image_data


def _get_dummy_image_data(
    seq_len: int,
    model_config: ModelConfig,
    vlm_config: VisionLanguageConfig,
) -> Tuple[SequenceData, MultiModalData]:
    seq_data, fake_mm_data = get_dummy_image_data(seq_len, model_config,
                                                  vlm_config)
    return seq_data, fake_mm_data

def _image_pixel_processor(
    data: ImagePixelData,
    model_config: ModelConfig,
    vlm_config: VisionLanguageConfig,
) -> Dict[str, torch.Tensor]:
    image = data.image
    if isinstance(image, Image.Image):
        image = torch.tensor(np.array(image))[None, :]
    return {'images': image}


class DreamLLMMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config)
        self.down_proj = RowParallelLinear(input_size=intermediate_size,
                                           output_size=hidden_size,
                                           bias=bias,
                                           quant_config=quant_config)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x

class DreamLLMAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class DreamLLMDecoderLayer(nn.Module):

    def __init__(
        self,
        config: DreamLLMConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        self.self_attn = DreamLLMAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
        )
        self.mlp = DreamLLMMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual



class DreamLLMModel(nn.Module):
    """DreamLLM Model (https://dreamllm.github.io/)
    """
    def __init__(
        self,
        config: DreamLLMConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )
        self.layers = nn.ModuleList([
            DreamLLMDecoderLayer(config=config,
                              cache_config=cache_config,
                              quant_config=quant_config)
            for idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def init_plugin_modules(self):
        p = next(self.parameters())
        device, dtype = p.device, p.dtype          
        for name, init_kwargs in self.config.plugins_init_kwargs.items():
            if self.config.plugins_type[name] == "embedding":
                logger.info(f"Initialized embedding `{name}` with kwargs:\n{pretty_format(init_kwargs)}")
                setattr(self, name, deep_instantiate(init_kwargs).to(device, dtype=dtype))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def replace_dream_image_embeddings(self, images, input_ids, inputs_embeds):
        assert inputs_embeds is not None
        image_start_id = self.config.special_tokens2ids_dict["additional_special_tokens"][DEFAULT_IMAGE_START_TOKEN]
        dream_start_id = self.config.special_tokens2ids_dict["additional_special_tokens"][DEFAULT_DREAM_START_TOKEN]
        is_empty_image = images is None or torch.all(images == 0).item()
        if images is not None and inputs_embeds is not None:
            images = images.to(inputs_embeds.device)

        # replace dream embedding
        if (
            (not is_empty_image)
            or (input_ids is not None and torch.any(input_ids == dream_start_id).item())
        ):
            query_embedding = self.dream_embedding()[0]
            dream_start_pos = torch.where(input_ids == dream_start_id)[0]
            cur_new_input_embeds = inputs_embeds
            for _dream_start_pos in dream_start_pos:
                cur_new_input_embeds = torch.cat(
                    [
                        cur_new_input_embeds[: _dream_start_pos + 1],
                        query_embedding,
                        cur_new_input_embeds[_dream_start_pos + self.dream_embedding.embed_len + 1 :],
                    ],
                    dim=0,
                ).contiguous()
            assert cur_new_input_embeds.shape[0] == inputs_embeds.shape[0]
            inputs_embeds = cur_new_input_embeds

        # replace image embedding
        if not is_empty_image:
            # HACK: not elegent, how can we auto select `clip_vision_embedding`
            image_features = self.clip_vision_embedding(images)
            cur_image_idx = 0
            image_start_pos = torch.where(input_ids == image_start_id)[0]
            cur_new_input_embeds = inputs_embeds
            for _image_start_pos in image_start_pos:
                if cur_image_idx >= image_features.shape[0]:
                    print(
                        input_ids.shape,
                        torch.where(input_ids == image_start_id),
                        image_features.shape,
                        images.shape,
                        _image_start_pos,
                        image_start_pos,
                    )
                if cur_image_idx >= image_features.shape[0]:
                    break
                cur_image_features = image_features[cur_image_idx]
                num_patches = cur_image_features.shape[0]
                assert _image_start_pos + num_patches + 1 <= inputs_embeds.shape[0]
                assert cur_image_idx <= image_features.size(0)

                cur_image_features = cur_image_features.to(device=inputs_embeds.device)
                cur_new_input_embeds = torch.cat(
                    (
                        cur_new_input_embeds[: _image_start_pos + 1],
                        cur_image_features,
                        cur_new_input_embeds[_image_start_pos + num_patches + 1 :],
                    ),
                    dim=0,
                ).contiguous()
                cur_image_idx += 1
            assert cur_new_input_embeds.shape[0] == inputs_embeds.shape[0]
            inputs_embeds = cur_new_input_embeds

        return inputs_embeds


    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.Tensor] = None,
        images: torch.FloatTensor | None = None
    ) -> torch.Tensor:   
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)

        hidden_states = self.replace_dream_image_embeddings(images, input_ids, hidden_states)

        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
                residual,
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    # `DreamEmbedding`
    def prepare_dream_queries_with_special_token(self, batch_size: int = 1):
        dream_start_id = self.config.special_tokens2ids_dict["additional_special_tokens"][DEFAULT_DREAM_START_TOKEN]
        dream_end_id = self.config.special_tokens2ids_dict["additional_special_tokens"][DEFAULT_DREAM_END_TOKEN]
        dream_special_ids = [dream_start_id, dream_end_id]
        dream_special_embeds = self.embed_tokens(torch.as_tensor([dream_special_ids], device=self.device))
        dream_queries = torch.cat(
            [dream_special_embeds[..., :1, :], self.dream_embedding(), dream_special_embeds[..., 1:, :]], 1
        )
        return dream_queries.repeat(batch_size, 1, 1).to(self.device)


@MULTIMODAL_REGISTRY.register_image_pixel_input(_image_pixel_processor)
@MULTIMODAL_REGISTRY.register_dummy_data(_get_dummy_image_data)
class DreamLLMForCausalMLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "embed_tokens",
        "lm_head"
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: DreamLLMConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = DreamLLMModel(config,
                                cache_config,
                                quant_config,
                                lora_config=lora_config)
        
        self.vocab_size = config.vocab_size
        self.loss_weight_lm = config.loss_weight_lm
        self.loss_weight_vm = config.loss_weight_vm
        self.diffusion_bs = config.diffusion_bs

        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)
        self.sampler = Sampler()

    def init_plugin_modules(self):
        # define multimodal encoders and decoders
        p = next(self.parameters())
        device, dtype = p.device, p.dtype    
        self.model.init_plugin_modules()
        for name, init_kwargs in self.config.plugins_init_kwargs.items():
            if self.config.plugins_type[name] == "head":
                logger.info(f"Initialized head `{name}` with kwargs:\n{pretty_format(init_kwargs)}")
                setattr(self, name, deep_instantiate(init_kwargs).to(device, dtype=dtype))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        images=None,
        **kwargs
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, images=images)
        self.hidden_states = hidden_states
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                if name.endswith("kv_scale"):
                    remapped_kv_scale_name = name.replace(
                        ".kv_scale", ".attn.kv_scale")
                    if remapped_kv_scale_name not in params_dict:
                        print_warning_once(
                            f"Found kv scale in the checkpoint (e.g. {name}), "
                            "but not found the expected name in the model "
                            f"(e.g. {remapped_kv_scale_name}). kv-scale is "
                            "not loaded.")
                        continue
                    else:
                        name = remapped_kv_scale_name
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
        
        # initilaize plugins
        self.init_plugin_modules()

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for layer_idx, scaling_factor in kv_cache_scales_loader(
                quantization_param_path, tp_rank, tp_size,
                self.config.num_hidden_layers,
                self.config.__class__.model_type):
            layer_self_attn = self.model.layers[layer_idx].self_attn

            if is_hip():
                # The scaling factor convention we are assuming is
                # quantized_value * scaling_factor ~= true_value
                # which is consistent with the practice of setting
                # scaling_factor = tensor_amax / FPtype_max
                scaling_factor *= 2
            if hasattr(layer_self_attn, "kv_scale"):
                layer_self_attn.attn._kv_scale = scaling_factor
            else:
                raise RuntimeError("Self attention has no KV cache scaling "
                                   "factor attribute!")

    # Diffusion utility functions
    def check_inputs(
        self,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def encode_prompt(
        self,
        num_images_per_prompt,
        do_classifier_free_guidance,
        prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        batch_size, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is not None:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    @torch.no_grad()
    def stable_diffusion_pipeline(
        self,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int | None = 1,
        eta: float = 0.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.FloatTensor | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
        output_type: Literal["latent", "pt", "np", "pil"] | None = "pil",
        callback: Callable[[int, int, torch.FloatTensor], None] | None = None,
        callback_steps: int = 1,
        cross_attention_kwargs: dict[str, Any] | None = None,
        guidance_rescale: float = 0.0,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt_embeds, negative_prompt_embeds)

        # 2. Define call parameters
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            num_images_per_prompt,
            do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # NOTE: 4. DreamLLMForCausalMLM only acts as a text encoder and leaves the rest to the stable diffusion head
        return self.stable_diffusion_head.pipeline(
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            output_type=output_type,
            callback=callback,
            callback_steps=callback_steps,
            cross_attention_kwargs=cross_attention_kwargs,
            guidance_rescale=guidance_rescale,
        )
