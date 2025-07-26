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
""" DreamLLM model configuration"""

from typing import Any, NewType, TypeAlias

from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizerBase
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation

from omni.config.lazy import LazyCall as L
from omni.models.dreamllm_qwen.modeling_plugins import PluginType
from omni.utils.import_utils import is_flash_attn_2_available
from omni.utils.loguru import logger

SpecialToken = NewType("SpecialToken", str)
SpecialTokenID = NewType("SpecialTokenID", int)

CLASS_KEY = "_class_"
NAME_KEY = "_name_"
PLUGIN_TYPE_KEY = "_plugin_type_"

_class_ = NewType(CLASS_KEY, type)
_name_ = NewType(NAME_KEY, str)
_plugin_type_ = NewType(PLUGIN_TYPE_KEY, PluginType)
InitKwargs = NewType("InitKwargs", str)
ConfigAndInitKwargs: TypeAlias = dict[_class_ | _name_ | _plugin_type_ | InitKwargs, Any]


def create_config_init_kwargs(config_init_kwargs: ConfigAndInitKwargs) -> DictConfig:
    config_class_ = config_init_kwargs.get("_class_", None)
    config_name_ = config_init_kwargs.get("_name_", None)
    config_plugin_type_ = config_init_kwargs.get("_plugin_type_", None)

    if not isinstance(config_class_, type):
        raise ValueError(f"`config_init_kwargs` must have `_class_` field of type `{type}`, got `{type(config_class_)}`.")
    if not isinstance(config_name_, str):
        raise ValueError(f"`config_init_kwargs` must have `_name_` field of type `{str}`, got `{type(config_name_)}`.")
    if not isinstance(config_plugin_type_, str):
        raise ValueError(
            f"`config_init_kwargs` must have `_plugin_type_` field of type `{str}`, got `{type(config_plugin_type_)}`."
        )

    return OmegaConf.create(config_init_kwargs, flags={"allow_objects": True})


class DreamLLMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2Model`]. It is used to instantiate a
    Qwen2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen2-7B-beta [Qwen/Qwen2-7B-beta](https://huggingface.co/Qwen/Qwen2-7B-beta).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen2Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers that use SWA (Sliding Window Attention). The bottom layers use SWA while the top use full attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import Qwen2Model, Qwen2Config

    >>> # Initializing a Qwen2 style configuration
    >>> configuration = Qwen2Config()

    >>> # Initializing a model from the Qwen2-7B style configuration
    >>> model = Qwen2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "dreamllm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        attention_bias=False,
        special_tokens2ids_dict={},
        plugins_init_kwargs={},
        plugins_type={},
        loss_weight_lm=1.0,
        loss_weight_vm=10.0,
        loss_scale_schedule="none",
        log_attentions=False,
        log_hidden_states=False,
        diffusion_bs=0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        self._rope_scaling_validation()
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # special token to id or `additional_special_tokens` dict, which contains additional special tokens and their ids
        self.special_tokens2ids_dict: dict[
            SpecialToken, SpecialTokenID | dict[SpecialToken | SpecialTokenID]
        ] = special_tokens2ids_dict

        self.plugins_init_kwargs: dict[_name_, dict] = plugins_init_kwargs
        self.plugins_type: dict[_name_, _plugin_type_] = plugins_type

        self.loss_weight_lm = loss_weight_lm
        self.loss_weight_vm = loss_weight_vm
        self.loss_scale_schedule = loss_scale_schedule

        # whether to log attentions and hidden states for better monitoring LLM training
        self.log_attentions = log_attentions
        self.log_hidden_states = log_hidden_states

        self.diffusion_bs = diffusion_bs

    def update_special_tokens2ids_dict(self, tokens_dict: dict, tokenizer: PreTrainedTokenizerBase):
        for key, token in tokens_dict.items():
            if isinstance(token, list):
                ids = tokenizer.convert_tokens_to_ids(token)
                if key not in self.special_tokens2ids_dict.keys():
                    self.special_tokens2ids_dict[key] = {}
                for _token, id in zip(token, ids):
                    self.special_tokens2ids_dict[key][_token] = id
            else:
                id = tokenizer.convert_tokens_to_ids(token)
                self.special_tokens2ids_dict[token] = id

    def update_plugins(self, init_kwargs: ConfigAndInitKwargs):
        cls = init_kwargs.pop(CLASS_KEY, None)
        name = init_kwargs.pop(NAME_KEY, None)
        plugin_type = init_kwargs.pop(PLUGIN_TYPE_KEY, None)
        assert (
            cls is not None and name is not None and plugin_type is not None
        ), f"`init_kwargs` must have `{CLASS_KEY}`, `{NAME_KEY}` and `{PLUGIN_TYPE_KEY}` fields"

        lazy_init = L(cls)(**init_kwargs)
        lazy_init = OmegaConf.to_container(lazy_init, resolve=True)

        if name not in self.plugins_init_kwargs.keys():
            self.plugins_init_kwargs[name] = lazy_init
        else:
            self.plugins_init_kwargs[name].update(lazy_init)

        self.plugins_type[name] = plugin_type

        return name

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, " f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}")
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be an float > 1, got {rope_scaling_factor}")

    def reset_plugins_init_kwargs(self, pretrained_plugin_model_name_or_path: str = None):
        for plugin_name in self.plugins_init_kwargs.keys():
            self.plugins_init_kwargs[plugin_name]["pretrained_model_name_or_path"] = pretrained_plugin_model_name_or_path
        logger.warning(f"reset all pretrained_model_name_or_path of plugin modules to {pretrained_plugin_model_name_or_path}")
