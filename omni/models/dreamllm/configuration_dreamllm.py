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

from omni.config.lazy import LazyCall as L
from omni.models.dreamllm.modeling_plugins import PluginType
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
    This is the configuration class to store the configuration of a [`LlamaModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LlamaModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens,
            Llama 2 up to 4096, CodeLlama up to 16384.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be an float greater than 1. The expected format
            is `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.


    ```python
    >>> from transformers import LlamaModel, LlamaConfig

    >>> # Initializing a LLaMA llama-7b style configuration
    >>> configuration = LlamaConfig()

    >>> # Initializing a model from the llama-7b style configuration
    >>> model = LlamaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "dreamllm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
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
        self._rope_scaling_validation()
        self.attention_bias = attention_bias

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
