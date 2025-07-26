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
""" PyTorch DreamLLM model."""
import math
import os
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Tuple

import numpy as np
import torch
import torch.backends.cuda
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
# from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.generation import GenerationMixin

from omni.config.instantiate import deep_instantiate
from omni.conversation.multimodal import MultimodalContent
from omni.models.dreamllm_qwen.configuration_dreamllm import DreamLLMConfig
from omni.models.dreamllm_qwen.modeling_plugins import PipelineImageType
from omni.models.dreamllm_qwen.tokenization_dreamllm import (
    DEFAULT_BOS_TOKEN,
    DEFAULT_DREAM_END_TOKEN,
    DEFAULT_DREAM_START_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_IMAGE_END_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
)
from omni.utils.fsdp_utils import FSDPMixin
from omni.utils.import_utils import _flash_attn_version, is_flash_attn_2_available, is_torch_fx_available, is_flash_attn_greater_or_equal_2_10
from omni.utils.loguru import logger
from omni.utils.profiler import pretty_format
from omni.utils.pytorch_utils import ALL_LAYERNORM_LAYERS

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


class DreamLLMRMSNorm(nn.Module):
# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Qwen2
# class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class RotaryEmbedding(nn.Module):
# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Qwen2
# class Qwen2RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[DreamLLMConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`Qwen2RotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DreamLLMMLP(nn.Module):
# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Qwen2
# class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class DreamLLMAttention(nn.Module):
# class Qwen2Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: DreamLLMConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class DreamLLMFlashAttention2(DreamLLMAttention):
# class Qwen2FlashAttention2(Qwen2Attention):
    """
    Qwen2 flash attention module, following Qwen2 attention module. This module inherits from `Qwen2Attention`
    as the weights of the module stays untouched. The only required change would be on the forward pass
    where it needs to correctly call the public API of flash attention and deal with padding tokens
    in case the input contains any of them. Additionally, for sliding window attention, we apply SWA only to the bottom
    config.max_window_layers layers.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            # cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            # kv_seq_len = key_states.shape[-2] + cache_position[0]
            # if (
            #     getattr(self.config, "sliding_window", None) is not None
            #     and kv_seq_len > self.config.sliding_window
            #     and cache_has_contents
            # ):
            #     slicing_tokens = 1 - self.config.sliding_window

            #     past_key = past_key_value[self.layer_idx][0]
            #     past_value = past_key_value[self.layer_idx][1]

            #     past_key = past_key[:, :, slicing_tokens:, :].contiguous()
            #     past_value = past_value[:, :, slicing_tokens:, :].contiguous()

            #     if past_key.shape[-2] != self.config.sliding_window - 1:
            #         raise ValueError(
            #             f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
            #             f" {past_key.shape}"
            #         )

            #     if attention_mask is not None:
            #         attention_mask = attention_mask[:, slicing_tokens:]
            #         attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class DreamLLMSdpaAttention(DreamLLMAttention):
# class Qwen2SdpaAttention(DreamLLMAttention):
    """
    Qwen2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from Qwen2Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


QWEN2_ATTENTION_CLASSES = {
    "eager": DreamLLMAttention,
    "flash_attention_2": DreamLLMFlashAttention2,
    "sdpa": DreamLLMSdpaAttention,
}


class DreamLLMDecoderLayer(nn.Module):
# class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: DreamLLMConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = DreamLLMMLP(config)
        self.input_layernorm = DreamLLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DreamLLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


QWEN2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`DreamLLMConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
# class DreamLLMPreTrainedModel(PreTrainedModel, FSDPMixin):
class DreamLLMPreTrainedModel(PreTrainedModel):
# class Qwen2PreTrainedModel(PreTrainedModel):
    config_class = DreamLLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DreamLLMDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    
    # a list of `state_dict` keys to ignore when saving the model (useful for keys that aren't
    # trained, but which are either deterministic or tied variables)
    _keys_to_ignore_on_save = []

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

QWEN2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


# Copy from modeling_qwen2.py: end


@dataclass
class BaseModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: tuple[tuple[torch.FloatTensor]] | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    additional_log_info: dict[str, float] | None = None


@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class DreamLLMModel(DreamLLMPreTrainedModel):
    """DreamLLM Model (https://dreamllm.github.io/)
    """
    def __init__(self, config: DreamLLMConfig):
        super().__init__(config)
        logger.debug(f"self.config._attn_implementation: {self.config._attn_implementation}")
        # if getattr(self.config, "_flash_attn_2_enabled", False):
        if self.config._attn_implementation == "flash_attention_2":
            logger.info("DreamLLMModel is using Flash Attention 2")

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DreamLLMDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = DreamLLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def init_plugin_modules(self):
        for name, init_kwargs in self.config.plugins_init_kwargs.items():
            if self.config.plugins_type[name] == "embedding":
                logger.info(f"Initialized embedding `{name}` with kwargs:\n{pretty_format(init_kwargs)}")
                setattr(self, name, deep_instantiate(init_kwargs).to(self.device, dtype=self.dtype))

                keys_to_ignore = list(getattr(self, name).state_dict().keys())
                keys_to_ignore = [f"model.{name}.{key}" for key in keys_to_ignore]
                self._keys_to_ignore_on_save.extend(keys_to_ignore)
                logger.info(f"Added the prefix keys of `model.{name}` to the list of keys to ignore on save.")

    def fsdp_ignored_modules(self) -> list:
        ignored_modules = []
        for name, _ in self.config.plugins_init_kwargs.items():
            if self.config.plugins_type[name] == "embedding":
                ignored_modules += getattr(self, name).fsdp_ignored_modules()
        return ignored_modules

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.phi3.modeling_phi3.Phi3Model._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    # Copied from transformers.models.mistral.modeling_mistral.MistralModel._prepare_4d_causal_attention_mask_with_cache_position with Mistral->Qwen2
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: DreamLLMConfig,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen2Config`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask
    
    
    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def _forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> tuple | BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        log_attentions = self.config.log_attentions
        log_hidden_states = self.config.log_hidden_states
        
        if log_attentions:
            output_attentions = True
            
        if self.config._attn_implementation == "flash_attention_2":
            output_attentions = False
            logger.warning_once(f"Flash Attention 2.0 is enabled. `output_attentions` is set to `False`.")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        
        additional_log_info = {}

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if log_hidden_states:
                additional_log_info[f"hidden_states_layer{idx - 1}"] = torch.max(torch.abs(hidden_states)).cpu().item()


            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if log_attentions:
                    additional_log_info[f"attentions_layer{idx}"] = torch.max(torch.abs(layer_outputs[1])).cpu().item()

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if log_hidden_states:
            additional_log_info[f"hidden_states_layer{len(self.layers) - 1}"] = torch.max(torch.abs(hidden_states)).cpu().item()

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            additional_log_info=additional_log_info,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        images: torch.FloatTensor | None = None,
        images_dm: torch.FloatTensor | dict | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> tuple | BaseModelOutputWithPast:
        # Not update original embedding parameters
        embed_tokens_backup = getattr(self, "embed_tokens_backup", None)
        if embed_tokens_backup is not None:
            logger.warning_once(f"Only update last num_added_tokens {self.num_added_tokens} of the embedding parameters.")
            with torch.no_grad():
                self.embed_tokens.weight[: -self.num_added_tokens] = embed_tokens_backup[: -self.num_added_tokens].data

        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        image_start_id = self.config.special_tokens2ids_dict["additional_special_tokens"][DEFAULT_IMAGE_START_TOKEN]
        dream_start_id = self.config.special_tokens2ids_dict["additional_special_tokens"][DEFAULT_DREAM_START_TOKEN]

        # for evaluation, avoid extract the clip features repeatly
        if (
            (images is not None)
            and (not self.training)
            and (past_key_values is not None)
            and (input_ids is None or torch.where(input_ids.reshape(-1) == image_start_id)[0].shape[0] == 0)
        ):
            images = None

        # replace diffusion query token
        if (
            (images is not None) 
            or (input_ids is not None and torch.any(input_ids == dream_start_id).item())
        ):
            assert inputs_embeds is not None
            query_embedding = self.dream_embedding(inputs_embeds.shape[0])
            new_input_embeds = []
            # logger.debug(f"query_embedding.shape: {query_embedding.shape}\ninput_ids.shape: {input_ids.shape}\ninputs_embeds.shape: {inputs_embeds.shape}")
            for cur_input_ids, cur_input_embeds, cur_query_embedding in zip(input_ids, inputs_embeds, query_embedding):
                # logger.debug(f"cur_input_ids.shape: {cur_input_ids.shape}\ncur_input_embeds.shape: {cur_input_embeds.shape}\ncur_query_embedding.shape: {cur_query_embedding.shape}")
                dream_start_pos = torch.where(cur_input_ids == dream_start_id)[0]
                cur_new_input_embeds = cur_input_embeds
                for _dream_start_pos in dream_start_pos:
                    cur_new_input_embeds = torch.cat(
                        [
                            cur_new_input_embeds[: _dream_start_pos + 1],
                            cur_query_embedding,
                            cur_new_input_embeds[_dream_start_pos + self.dream_embedding.embed_len + 1 :],
                        ],
                        dim=0,
                    ).contiguous()
                assert cur_new_input_embeds.shape[0] == cur_input_embeds.shape[0], f"cur_new_input_embeds.shape[0]: {cur_new_input_embeds.shape[0]}, cur_input_embeds.shape[0]: {cur_input_embeds.shape[0]}"
                new_input_embeds.append(cur_new_input_embeds)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        # HACK: not elegent, how can we auto select `vision_encoder`
        image_features = self.vision_encoder(images)

        if inputs_embeds is not None:
            if images is not None:
                # replace image token
                new_input_embeds = []
                cur_image_idx = 0
                for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                    image_start_pos = torch.where(cur_input_ids == image_start_id)[0]
                    cur_new_input_embeds = cur_input_embeds
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
                        assert _image_start_pos + num_patches + 1 <= cur_input_embeds.shape[0]
                        assert cur_image_idx <= image_features.size(0)

                        cur_image_features = cur_image_features.to(device=cur_input_embeds.device)
                        cur_new_input_embeds = torch.cat(
                            (
                                cur_new_input_embeds[: _image_start_pos + 1],
                                cur_image_features,
                                cur_new_input_embeds[_image_start_pos + num_patches + 1 :],
                            ),
                            dim=0,
                        ).contiguous()
                        cur_image_idx += 1
                    assert cur_new_input_embeds.shape[0] == cur_input_embeds.shape[0]
                    new_input_embeds.append(cur_new_input_embeds)
                inputs_embeds = torch.stack(new_input_embeds, dim=0)
            elif self.training:
                # dummy image features for unused parameters bug
                inputs_embeds = inputs_embeds + image_features
        else:
            inputs_embeds = image_features.unsqueeze(0)

        return self._forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

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


@dataclass
class CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    past_key_values: tuple[tuple[torch.FloatTensor]] | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    additional_log_info: dict[str, float] | None = None


class DreamLLMForCausalMLM(DreamLLMPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: DreamLLMConfig):
        super().__init__(config)
        self.model = DreamLLMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.loss_weight_lm = config.loss_weight_lm
        self.loss_weight_vm = config.loss_weight_vm
        self.diffusion_bs = config.diffusion_bs

        # initialize weights and apply final processing
        self.post_init()

    def init_plugin_modules(self):
        # define multimodal encoders and decoders
        self.model.init_plugin_modules()
        for name, init_kwargs in self.config.plugins_init_kwargs.items():
            if self.config.plugins_type[name] == "head":
                logger.info(f"Initialized head `{name}` with kwargs:\n{pretty_format(init_kwargs)}")
                setattr(self, name, deep_instantiate(init_kwargs).to(self.device, dtype=self.dtype))

                keys_to_ignore = list(getattr(self, name).state_dict().keys())
                keys_to_ignore = [f"{name}.{key}" for key in keys_to_ignore]
                self._keys_to_ignore_on_save.extend(keys_to_ignore)
                logger.info(f"Added the prefix keys of `{name}` to the list of keys to ignore on save.")

    def fsdp_ignored_modules(self) -> list:
        ignored_modules = self.model.fsdp_ignored_modules()
        for name, _ in self.config.plugins_init_kwargs.items():
            if self.config.plugins_type[name] == "head":
                ignored_modules += getattr(self, name).fsdp_ignored_modules()
        return ignored_modules

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike | None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        *model_args,
        config: PretrainedConfig | str | os.PathLike | None = None,
        cache_dir: str | os.PathLike | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool = None,
        reset_plugin_model_name_or_path: bool = False,
        **kwargs,
    ):
        """
        Newly add a `tokenizer` argument to support `from_pretrained` method.
        Ensure the consistency of tokenizer and model vocabulary length.
        """
        assert tokenizer is not None, "tokenizer should not be None"

        resume_download = kwargs.get("resume_download", False)
        proxies = kwargs.get("proxies", None)
        from_pipeline = kwargs.get("_from_pipeline", None)
        from_auto_class = kwargs.get("_from_auto", False)
        subfolder = kwargs.get("subfolder", "")

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, _ = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )

        # use initial `LlamaConfig` or exported `DreamLLMConfig` behind
        model = super(DreamLLMForCausalMLM, cls).from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs,
        )

        if reset_plugin_model_name_or_path:
            config.reset_plugins_init_kwargs()

        if len(tokenizer) > model.config.vocab_size:
            logger.info(
                f"The tokenizer vocabulary size {len(tokenizer)} is larger than the model vocabulary size {model.config.vocab_size}. "
                f"Resizing token embedding of model..."
            )
            model.resize_token_embeddings(len(tokenizer))
        elif len(tokenizer) < model.config.vocab_size:
            logger.warning(
                f"The tokenizer vocabulary size {len(tokenizer)} is smaller than the model vocabulary size {model.config.vocab_size}. "
                f"Carefully check the configuration to avoid potential issues."
            )

        logger.info(f"Now, the tokenizer and model vocabulary sizes are both {len(tokenizer)}.")


        # print('=' * 100)
        # print(f"self.config=   {config.plugins_init_kwargs}")
        # HACK: add all pretrained plugins path if not specified
        for key, init_kwargs in config.plugins_init_kwargs.items():
            if init_kwargs["pretrained_model_name_or_path"] == 'NotSpecified':
                init_kwargs["pretrained_model_name_or_path"] = None
            elif init_kwargs["pretrained_model_name_or_path"] is None:
                # load pre-aligned plugin models instead of random initialization during training
                init_kwargs["pretrained_model_name_or_path"] = pretrained_model_name_or_path

        # print('-' * 100)
        # print(f"self.config=   {config.plugins_init_kwargs}")
        model.config.plugins_init_kwargs = config.plugins_init_kwargs
        # BUG: parameters of pretrained plugin modules will be influenced by `_init_weights` if put this in `__init__` or before `model.resize_token_embeddings`
        model.init_plugin_modules()

        return model

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        images: torch.FloatTensor | None = None,
        images_dm: torch.FloatTensor | dict | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> tuple | CausalLMOutputWithPast:
        if input_ids is not None:
            assert (
                input_ids.shape[1] <= self.config.max_position_embeddings
            ), f"the sequence length should be less than model max length {self.config.max_position_embeddings}"

        # assert images is not None
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            images=images,
            images_dm=images_dm,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]

        dream_start_id = self.config.special_tokens2ids_dict["additional_special_tokens"][DEFAULT_DREAM_START_TOKEN]
        dream_end_id = self.config.special_tokens2ids_dict["additional_special_tokens"][DEFAULT_DREAM_END_TOKEN]
        dream_patch_id = self.config.special_tokens2ids_dict["additional_special_tokens"][DEFAULT_IMAGE_PATCH_TOKEN]
        bos_id = self.config.special_tokens2ids_dict[DEFAULT_BOS_TOKEN]
        eos_id = self.config.special_tokens2ids_dict[DEFAULT_EOS_TOKEN]

        # Let's train diffusion!
        vm_loss = 0.0
        if self.training and images_dm is not None:
            # get `encoder_hidden_states`
            images_dm_len = sum([img['images_dm'].shape[0] for img in images_dm.values()]) if isinstance(images_dm, dict) else images_dm.shape[0]
            encoder_hidden_states = []
            for _idx, cur_hidden_states in enumerate(hidden_states):
                cur_input_ids = input_ids[_idx]
                dream_start_pos = torch.where(
                    cur_input_ids
                    == self.model.config.special_tokens2ids_dict["additional_special_tokens"][DEFAULT_DREAM_START_TOKEN]
                )[0]
                for _dream_start_pos in dream_start_pos:
                    assert len(encoder_hidden_states) < images_dm_len, f"len(encoder_hidden_states)({len(encoder_hidden_states)}) >= images_dm_len({images_dm_len})"
                    encoder_hidden_states.append(
                        cur_hidden_states[_dream_start_pos + 1 : _dream_start_pos + 1 + self.model.dream_embedding.embed_len]
                    )
                    if encoder_hidden_states[-1].shape[0] != self.model.dream_embedding.embed_len:
                        logger.error(_idx, _dream_start_pos, hidden_states.shape)
            encoder_hidden_states = torch.stack(encoder_hidden_states, 0)

            # get `u_encoder_hidden_states` if use classifier free guidance
            u_encoder_hidden_states = None
            if self.stable_diffusion_head.drop_prob is not None:
                u_outputs = self.model(
                    input_ids=torch.tensor(
                        [
                            [bos_id, dream_start_id]
                            + [dream_patch_id] * self.model.dream_embedding.embed_len
                            + [dream_end_id, eos_id]
                        ]
                    ).to(input_ids.device),
                    attention_mask=torch.ones(1, self.model.dream_embedding.embed_len + 4, device=encoder_hidden_states.device),
                    past_key_values=None,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                u_encoder_hidden_states = u_outputs[0][:, 2 : 2 + self.model.dream_embedding.embed_len, :]
                u_encoder_hidden_states = u_encoder_hidden_states.repeat(encoder_hidden_states.shape[0], 1, 1)

            assert encoder_hidden_states.shape[0] == images_dm_len, f"encoder_hidden_states.shape[0]({encoder_hidden_states.shape[0]}) != images_dm_len({images_dm_len})"
            if isinstance(images_dm, dict):
                for key, cur_images_dm in images_dm.items():
                    cur_bs = key[2]
                    data_idx = cur_images_dm['data_idx']
                    cur_images = cur_images_dm['images_dm']
                    cur_encoder_hidden_states = encoder_hidden_states[data_idx]
                    cur_u_encoder_hidden_states = u_encoder_hidden_states and u_encoder_hidden_states[data_idx]

                    if cur_bs > 0:
                        for diffusion_idx in range(-(-cur_images.shape[0] // cur_bs)):
                            start_idx = diffusion_idx * cur_bs
                            end_idx = start_idx + cur_bs
                            cur_mini_bs = cur_images[start_idx: end_idx].shape[0]
                            vm_loss += cur_mini_bs * self.stable_diffusion_head(cur_images[start_idx: end_idx],
                                                                                cur_encoder_hidden_states[start_idx: end_idx],
                                                                                cur_u_encoder_hidden_states and cur_u_encoder_hidden_states[start_idx: end_idx])
                    else:
                        vm_loss += cur_images.shape[0] * self.stable_diffusion_head(cur_images, cur_encoder_hidden_states, cur_u_encoder_hidden_states)
                vm_loss /= images_dm_len
            else:
                if self.diffusion_bs > 0:
                    for diffusion_idx in range(-(-images_dm_len // self.diffusion_bs)):
                        start_idx = diffusion_idx * self.diffusion_bs
                        end_idx = start_idx + self.diffusion_bs
                        cur_mini_bs = images_dm[start_idx: end_idx].shape[0]
                        vm_loss += cur_mini_bs * self.stable_diffusion_head(images_dm[start_idx: end_idx], 
                                                                            encoder_hidden_states[start_idx: end_idx],
                                                                            u_encoder_hidden_states and u_encoder_hidden_states[start_idx: end_idx])
                    vm_loss /= images_dm_len
                else:
                    vm_loss = self.stable_diffusion_head(images_dm, encoder_hidden_states, u_encoder_hidden_states)
        elif self.training:
            # HACK: dummy forward to avoid `find_unused_parameters` error
            # logger.warning("use dummy input for diffusion due to empty images list.")
            vm_loss = self.stable_diffusion_head(images_dm, None, None, self.model.dream_embedding())

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
        logits = logits.float()

        lm_loss = 0.0
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="none")
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            valid_labels =(shift_labels !=-100).bool()
            if valid_labels.sum() > 0:
                lm_loss = (loss_fct(shift_logits, shift_labels) * valid_labels).sum() / valid_labels.sum()
            else:
                lm_loss = loss_fct(shift_logits, shift_labels).mean()

        if self.config.loss_scale_schedule == "l1_norm":
            loss_scale = self.loss_weight_lm + self.loss_weight_vm
        elif self.config.loss_scale_schedule == "l2_norm":
            loss_scale = math.sqrt(self.loss_weight_lm**2 + self.loss_weight_vm**2)
        else:
            loss_scale = 1

        if self.training and images is not None and np.isnan(lm_loss.cpu().item()):
            logger.warning("lm_loss is NaN!")
            loss = vm_loss * self.loss_weight_vm + lm_loss * 0.0
        elif self.training and images_dm is not None and np.isnan(vm_loss.cpu().item()):
            logger.warning("vm_loss is NaN!")
            loss = vm_loss * 0.0 + lm_loss * self.loss_weight_lm
        else:
            # For comprehension_only or creation_only, set another loss weight to 0
            loss = vm_loss * self.loss_weight_vm + lm_loss * self.loss_weight_lm
        loss = loss / loss_scale

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        additional_log_info = {
            "lm_loss": lm_loss.cpu().item() if not isinstance(lm_loss, float) else lm_loss,
            "vm_loss": vm_loss.cpu().item() if not isinstance(vm_loss, float) else vm_loss,
        }
        additional_log_info.update(outputs.additional_log_info)
        if len(additional_log_info) == 0:
            additional_log_info = None

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            additional_log_info=additional_log_info,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Prepare the model inputs for generation. In includes operations like computing the 4D attention mask or
        slicing inputs given the existing cache.

        See the forward pass in the model documentation for expected arguments (different models might have different
        requirements for e.g. `past_key_values`). This function should work as is for most LLMs.
        """

        # 1. Handle BC:
        model_inputs = {}
        # - some models don't have `Cache` support (which implies they don't expect `cache_position` in `forward`)
        if self._supports_cache_class:
            model_inputs["cache_position"] = cache_position
        # - `cache_position` was not a mandatory input in `prepare_inputs_for_generation` for those models, and this
        #   function may be called outside of `generate`. Handle most use cases by creating `cache_position` on the fly
        #   (this alternative is not as robust as calling `generate` and letting it create `cache_position`)
        elif cache_position is None:
            past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            cache_position = torch.arange(past_length, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        
        # 2. Generic cache-dependent input preparation
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
            if inputs_embeds is not None or cache_position[-1] >= input_ids.shape[1]:  # Exception 1 or Exception 3
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]
                
        # 3. Prepare base model inputs
        input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if not self.config.is_encoder_decoder:
            if inputs_embeds is not None and cache_position[0] == 0:
                model_inputs[input_ids_key] = None
                model_inputs["inputs_embeds"] = inputs_embeds
            else:
                # `clone` calls in this function ensure a consistent stride. See #32227
                model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
                model_inputs["inputs_embeds"] = None
        else:
            model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)

        # 4. Create missing `position_ids` on the fly
        if (
            attention_mask is not None
            and kwargs.get("position_ids") is None
            and "position_ids" in set(inspect.signature(self.forward).parameters.keys())
        ):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            kwargs["position_ids"] = position_ids  # placed in kwargs for further processing (see below)

        # 5. Slice model inputs if it's an input that should have the same length as `input_ids`
        for model_input_name in ["position_ids", "token_type_ids"]:
            model_input = kwargs.get(model_input_name)
            if model_input is not None:
                if past_key_values:
                    model_input = model_input[:, -input_ids.shape[1] :]
                    model_input = model_input.clone(memory_format=torch.contiguous_format)
                model_inputs[model_input_name] = model_input
                
        # 6. Create 4D attention mask is we are using a `StaticCache` (important for performant compiled forward pass)
        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs[input_ids_key].shape
                device = model_inputs[input_ids_key].device

            # Create the causal mask with fixed shape in advance, to reduce recompilations. If the function to create
            # the 4D causal mask exists, it should be present in the base model (XXXModel class).
            base_model = getattr(self, self.base_model_prefix, None)
            if base_model is None:
                causal_mask_creation_function = getattr(
                    self, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )
            else:
                causal_mask_creation_function = getattr(
                    base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )
            if causal_mask_creation_function is None:
                logger.warning_once(
                    f"{self.__class__.__name__} has no `_prepare_4d_causal_attention_mask_with_cache_position` method "
                    "defined in its base modeling class. Compiled forward passes will be sub-optimal. If you're "
                    "writing code, see Llama for an example implementation. If you're a user, please report this "
                    "issue on GitHub."
                )
            else:
                attention_mask = causal_mask_creation_function(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_cache_shape(),
                    dtype=self.dtype,
                    device=device,
                    cache_position=cache_position,
                    batch_size=batch_size,
                    config=self.config,
                    past_key_values=past_key_values,
                )
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask

        # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        # 8. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
        model_inputs.pop("labels", None)

        # 9. Return model inputs

        model_inputs.update(
            {
                "images": kwargs.pop("images", None),
            }
        )
        return model_inputs

    # @staticmethod
    # def _reorder_cache(past_key_values, beam_idx):
    #     reordered_past = ()
    #     for layer_past in past_key_values:
    #         reordered_past += (tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),)
    #     return reordered_past

    # Diffusion utility functions
    def check_inputs(
        self,
        tokenizer,
        prompt,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise ValueError(
                "The `tokenizer` argument should be a `PreTrainedTokenizerBase` instance. Make sure to pass the tokenizer as `tokenizer=your_tokenizer`."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, MultimodalContent) and not isinstance(prompt, list)
        ):
            raise ValueError(f"`prompt` has to be of type `str`, `MultimodalContent` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def get_prompt_embeds(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompt: str | list[str] | MultimodalContent,
        device: str | torch.device,
    ):
        if isinstance(prompt, str) or isinstance(prompt, MultimodalContent):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)

        if isinstance(prompt, str) or isinstance(prompt, list):
            text_inputs = tokenizer(prompt, padding=True, return_tensors="pt")
            text_input_ids = text_inputs.input_ids.to(device)
            text_input_attention_mask = text_inputs.attention_mask.to(device)

            images_input_pt = None

        elif isinstance(prompt, MultimodalContent):
            # FIXME: noqa, not debug yet
            image_placeholder_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
            image_start_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_START_TOKEN)
            image_patch_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_PATCH_TOKEN)
            image_end_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_END_TOKEN)

            text_inputs = tokenizer(prompt.text)
            text_input_ids = text_inputs.input_ids

            text_input_ids_start = torch.where(torch.as_tensor(text_input_ids) == image_placeholder_id)[0].tolist()
            text_input_ids_start = [-1] + text_input_ids_start
            text_input_ids_end = text_input_ids_start[1:] + [len(text_input_ids)]

            image_patch_ids = [image_start_id] + [image_patch_id] * 256 + [image_end_id]

            text_input_ids_list = []
            for start_index, end_index in zip(text_input_ids_start, text_input_ids_end):
                text_input_ids_list.append(text_input_ids[start_index + 1 : end_index])

            text_input_ids = text_input_ids_list[0]

            for i in range(1, len(text_input_ids_list)):
                text_input_ids = text_input_ids + image_patch_ids + text_input_ids_list[i]

            text_input_ids = torch.as_tensor([text_input_ids], device=device)
            text_input_attention_mask = torch.ones(batch_size, text_input_ids.shape[1], device=device)

            images_input_pt = self.model.vision_encoder.processor.preprocess(
                prompt.mm_content_list[0],
                return_tensors="pt",
            )["pixel_values"]
            images_input_pt = images_input_pt.to(device)

        text_out = self.forward(
            input_ids=text_input_ids,
            images=images_input_pt,
            attention_mask=text_input_attention_mask,
            use_cache=True,
        )

        dream_query = self.model.prepare_dream_queries_with_special_token(batch_size=batch_size)

        dream_query_attention_mask = torch.ones(batch_size, dream_query.shape[1], device=device)
        attention_mask = torch.cat([text_input_attention_mask, dream_query_attention_mask], dim=1)

        out = self.forward(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=text_out.past_key_values,
            inputs_embeds=dream_query,
            use_cache=True,
            output_hidden_states=True,
        )

        prompt_embeds = out.hidden_states[-1][:, 1:-1, :]

        return prompt_embeds

    def encode_prompt(
        self,
        tokenizer,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
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
        if prompt is not None and isinstance(prompt, str) or isinstance(prompt, MultimodalContent):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self.get_prompt_embeds(tokenizer, prompt, device)

        prompt_embeds_dtype = self.dtype

        prompt_embeds = prompt_embeds.to(device, dtype=prompt_embeds_dtype)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: list[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            negative_prompt_embeds = self.get_prompt_embeds(tokenizer, uncond_tokens, device)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    @torch.no_grad()
    def stable_diffusion_pipeline(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompt: str | list[str] | MultimodalContent = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: str | list[str] | MultimodalContent | None = None,
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
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
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
        self.check_inputs(tokenizer, prompt, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # 2. Define call parameters
        device = self.device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            tokenizer,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
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

    @torch.no_grad()
    def controlnet_pipeline(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompt: str | list[str] | MultimodalContent = None,
        image: PipelineImageType = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: str | list[str] | MultimodalContent | None = None,
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
        controlnet_conditioning_scale: float | list[float] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: float | list[float] = 0.0,
        control_guidance_end: float | list[float] = 1.0,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                specified as `torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be
                accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If height
                and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in
                `init`, images must be passed as a list such that each element of the list can be correctly batched for
                input to a single ControlNet.
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
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                the corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                The ControlNet encoder tries to recognize the content of the input image even if you remove all
                prompts. A `guidance_scale` value between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(tokenizer, prompt, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # 2. Define call parameters
        device = self.device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            tokenizer,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # NOTE: 4. DreamLLMForCausalMLM only acts as a text encoder and leaves the rest to the stable diffusion head
        return self.controlnet_head.pipeline(
            image=image,
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
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guess_mode=guess_mode,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
        )
