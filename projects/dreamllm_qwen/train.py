# ------------------------------------------------------------------------------------------------
# Copyright (c) 2023-2024 DreamLLM Authors. All rights reserved.
# ------------------------------------------------------------------------------------------------
import os
import pathlib
from dataclasses import dataclass, field

import torch
# from transformers import AutoTokenizer
from omni.models.dreamllm_qwen.tokenization_dreamllm import DreamLLMTokenizer

from omni.config.arg_parser import LazyAguments, LazyArgumentParser
from omni.data.builders.builder_dreamllm_qwen import DataCollatorForDreamLLMDataset, DreamLLMDataset
from omni.models.dreamllm_qwen.configuration_dreamllm import ConfigAndInitKwargs, DreamLLMConfig
# from omni.models.dreamllm_qwen.modeling_dreamllm import DreamLLMForCausalMLM
from omni.models.dreamllm_qwen.modeling_dreamllm_qwen import DreamLLMForCausalMLM
from omni.train.dreamllm_trainer import DreamLLMTrainer
from omni.train.training_args import TrainingArguments
from omni.utils.loguru import logger
from omni.utils.profiler import FunctionProfiler, pretty_format
from omni.utils.tokenizer_utils import average_init_token_embeddings
from omni.utils.lora_utils import build_lora_model
from peft import LoraConfig

@dataclass
class ModelArguments:
    model_name_or_path: str | None = field(default="lmsys/vicuna-13b-delta-v0")
    model_max_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."}
    )
    use_fast_tokenizer: bool = field(default=True, metadata={"help": "Whether to use fast tokenizer."})
    local_files_only: bool = field(default=False)
    use_flash_attention_2: bool = field(default=True)
    attn_implementation: str = field(default="flash_attention_2")
    special_tokens_dict: dict = field(default_factory=dict)
    average_init_embed_tokens: bool = field(default=True)
    freeze_embed_tokens: bool = field(default=False)
    freeze_lm_model: bool = field(default=False)
    freeze_lm_head: bool = field(default=False)
    plugins_config_init_kwargs: dict[str, ConfigAndInitKwargs] = field(default_factory=dict)
    loss_weight_lm: float = field(default=1.0)
    loss_weight_vm: float = field(default=10.0)
    loss_scale_schedule: str = field(
        default="none", metadata={"help": "The schedule of loss scale, choose from ['none', 'l1_norm', 'l2_norm']"}
    )
    log_attentions: bool = field(default=False, metadata={"help": "Whether to log attentions when training."})
    log_hidden_states: bool = field(default=False, metadata={"help": "Whether to log hidden states when training."})
    diffusion_bs: int = field(default=0, metadata={"help": "The batch size for training diffusion."})

@dataclass
class DataArguments:
    datasets: list[str] = field(default_factory=list, metadata={"help": "Which datasets are used? (default: [])"})
    datasets_approx_sizes: list[str] = field(default_factory=list, metadata={"help": "Which datasets are used? (default: [])"})
    datasets_shard_lists: list[str] = field(default_factory=list, metadata={"help": "Which datasets are used? (default: [])"})
    datasets_init_kwargs: dict = field(default_factory=dict, metadata={"help": "The init kwargs of datasets."})
    size_list: list[str | int] = field(default_factory=list, metadata={"help": "The size of each dataset."})
    comprehension_only: bool = field(default=False, metadata={"help": "Whether only load the understand mode data"})
    creation_only: bool = field(default=False, metadata={"help": "Whether only load the generation mode data"})
    skip_long_sample: bool = field(default=False, metadata={"help": "Whether to skip long sample"})
    max_image_length: int = field(default=0, metadata={"help": "Whether to skip long image list sample"})
    conv_template_name: str = field(default="vicuna_v1.1", metadata={"help": "The template of the conversation dataset."})
    use_bucket: bool = field(default=False, metadata={"help": "Whether to use bucket."})
    bucket_config: dict = field(default_factory=lambda:{'512': (1.0, 16)}, metadata={"help": "The init kwargs of datasets."})
    disable_system_prompt: bool = field(default=False, metadata={"help": "Whether to disable system prompt."})
    use_rgba: bool = field(default=False, metadata={"help": "Whether to use rgba."})

@dataclass
class DreamLLMTrainingArguments(TrainingArguments):
    use_lora :bool = field(default=False, metadata={"help": "Whether to use lora model"})
    lora_config : LoraConfig = field(default_factory=LoraConfig)

@dataclass
class Arguments(LazyAguments):
    model: ModelArguments = field(default_factory=ModelArguments)
    data: DataArguments = field(default_factory=DataArguments)
    training: DreamLLMTrainingArguments = field(default_factory=DreamLLMTrainingArguments)

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

def setup_mpi_launcher():
    # launch from mpi
    if int(os.getenv('OMPI_COMM_WORLD_SIZE', '0')) > 0:
        os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
        os.environ['LOCAL_RANK'] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
        os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']   

@logger.catch
def train():
    setup_mpi_launcher() 
    config = LazyArgumentParser(Arguments)

    dtype = torch.float32
    if config.training.fp16:
        dtype = torch.float16
    elif config.training.bf16:
        dtype = torch.bfloat16
    logger.info(f"Training with precision {dtype}...")

    # define tokenizer
    tokenizer = DreamLLMTokenizer.from_pretrained(
        config.model.model_name_or_path,
        local_files_only=config.model.local_files_only,
        model_max_length=config.model.model_max_length,
        padding_side="right",
        use_fast=config.model.use_fast_tokenizer,
    )
    logger.info(
        f"Successfully load ({'fast' if config.model.use_fast_tokenizer else 'slow'}) tokenizer from {config.model.model_name_or_path}, "
        f"with `len(tokenizer)` {len(tokenizer)}, `model_max_length` {config.model.model_max_length} and padding right."
    )

    # add special tokens
    num_added_tokens = 0
    if len(config.model.special_tokens_dict) > 0:
        num_added_tokens = tokenizer.add_special_tokens(config.model.special_tokens_dict)
    if num_added_tokens > 0:
        logger.info(
            f"Successfully add {num_added_tokens} special tokens, which are:\n{pretty_format(config.model.special_tokens_dict)}"
        )
    else:
        logger.info("No special tokens need to be added.")
    logger.info(f"The final tokenizer is:\n{pretty_format(tokenizer)}")

    # define `DreamLLMConfig`
    dreamllm_config = DreamLLMConfig.from_pretrained(
        config.model.model_name_or_path,
        local_files_only=config.model.local_files_only,
        max_position_embeddings=config.model.model_max_length,
        max_sequence_length=config.model.model_max_length,
        loss_weight_lm=config.model.loss_weight_lm,
        loss_weight_vm=config.model.loss_weight_vm,
        loss_scale_schedule=config.model.loss_scale_schedule,
        log_attentions=config.model.log_attentions,
        log_hidden_states=config.model.log_hidden_states,
        diffusion_bs=config.model.diffusion_bs,
        attn_implementation=config.model.attn_implementation,
    )
    logger.info(f"Successfully load `DreamLLMConfig` from {config.model.model_name_or_path}.")

    # add special tokens and ids to `DreamLLMConfig`
    if num_added_tokens > 0:
        dreamllm_config.update_special_tokens2ids_dict(config.model.special_tokens_dict, tokenizer)
        logger.info(
            f"Successfully update special tokens above to `DreamLLMConfig`, now the `special_tokens2ids_dict` is :\n"
            f"{pretty_format(dreamllm_config.special_tokens2ids_dict)}"
        )
    else:
        logger.info(f"No special tokens need to be added to `DreamLLMConfig`.")

    # When freezing the `lm_model`, do not affect the `requires_grad` status of the plugin.
    plugin_modules_names = []

    # add `init_kwargs` of plugins to `dreamllm_config`
    for _, plugin_config_init_kwargs in config.model.plugins_config_init_kwargs.items():
        name = dreamllm_config.update_plugins(plugin_config_init_kwargs)
        logger.info(f"Successfully update plugin `{name}` with `ConfigAndInitKwargs` to `DreamLLMConfig`.")
        plugin_modules_names.append(name)

    if config.model.use_flash_attention_2:
        attn_implementation="flash_attention_2"
    else:
        attn_implementation="sdpa"
    # define model
    with torch.device(config.training.device):
        with FunctionProfiler("DreamLLMForCausalMLM.from_pretrained"):
            model = DreamLLMForCausalMLM.from_pretrained(
                config.model.model_name_or_path,
                tokenizer,
                config=dreamllm_config,
                local_files_only=config.model.local_files_only,
                attn_implementation=attn_implementation,
                torch_dtype=dtype,
            )

    # set the `requires_grad` status of `embed_tokens`
    if config.model.average_init_embed_tokens and num_added_tokens > 0:
        average_init_token_embeddings(model, num_added_tokens)

        # if average initialization is used, the `requires_grad` status of **newly added** `embed_tokens` should be set to `True`
        model.get_input_embeddings().requires_grad_(True)
        logger.info("Setting `requires_grad` status of `embed_tokens` to `True`")

        # HACK: This is a hack method for newly added tokens that can be trained
        if config.model.freeze_embed_tokens:
            model.get_decoder().embed_tokens_backup = (
                model.get_input_embeddings().weight.data.clone().to(model.device, dtype=dtype)
            )
            model.get_decoder().num_added_tokens = num_added_tokens
            logger.info(f"Freeze original `embed_tokens`, only train newly added tokens.")
    else:
        model.get_input_embeddings().requires_grad_(not config.model.freeze_embed_tokens)

    # set the `requires_grad` status of `lm_model`
    if config.model.freeze_lm_model:
        # Loop over all named parameters in the model
        for name, param in model.named_parameters():
            # Only change the requires_grad attribute if the parameter name does not contain any of the specified substrings
            if all(_name not in name for _name in plugin_modules_names + ["lm_head", "embed_tokens"]):
                param.requires_grad_(False)

    # set the `requires_grad` status of `lm_head`
    model.get_output_embeddings().requires_grad_(not config.model.freeze_lm_head)

    log_trainable_parameters(model)

    if config.training.use_lora == True:
        logger.info("Building LoRA model.")
        model = build_lora_model(model=model, training_args=config.training)
        logger.info(f"LoRA model:\n{model}")
        logger.info(f">> LoRA trainable parameters:")
        log_trainable_parameters(model)
        # for name, param in model.named_parameters():
        #     logger.info(f"Parameter: {name}, dtype: {param.dtype}, requires_grad: {param.requires_grad} ")

        if config.training.gradient_checkpointing:
            model.enable_input_require_grads()

    model.to(config.training.device, dtype=dtype)
    logger.info(f">> device: {config.training.device}")

    # data
    if config.data.use_bucket:
        logger.info(f"Use bucket in diffusion training.")
        data_collator = DataCollatorForDreamLLMDataset(tokenizer, config.data.bucket_config, getattr(model, "stable_diffusion_head").processor_size)
    else:
        data_collator = DataCollatorForDreamLLMDataset(tokenizer)
    train_dataset = DreamLLMDataset(
        datasets=config.data.datasets,
        datasets_init_kwargs=config.data.datasets_init_kwargs,
        size_list=config.data.size_list,
        tokenizer=tokenizer,
        clip_vision_embedding_processor=getattr(model.get_decoder(), "clip_vision_embedding").processor,
        stable_diffusion_head_processor=getattr(model, "stable_diffusion_head").processor,
        clip_vision_embedding_len=getattr(model.get_decoder(), "clip_vision_embedding").embed_len,
        dream_embedding_len=getattr(model.get_decoder(), "dream_embedding").embed_len,
        comprehension_only=config.data.comprehension_only,
        creation_only=config.data.creation_only,
        use_image_start_and_end=True,
        use_dream_start_and_end=True,
        skip_long_sample=config.data.skip_long_sample,
        max_image_length=config.data.max_image_length,
        approx_sizes=config.data.datasets_approx_sizes,
        shard_lists=config.data.datasets_shard_lists,
        conv_template_name=config.data.conv_template_name,
        use_bucket=config.data.use_bucket,
        disable_system_prompt=config.data.disable_system_prompt,
        use_rgba=config.data.use_rgba,
    )
    eval_dataset = None

    # train
    trainer = DreamLLMTrainer(
        model=model,
        args=config.training,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    if list(pathlib.Path(config.training.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_model(output_dir=config.training.output_dir)
    trainer.save_state()


if __name__ == "__main__":
    train()
