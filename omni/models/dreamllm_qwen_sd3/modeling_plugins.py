import inspect
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Literal

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import is_torch_xla_available
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from torch import nn
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModel

from omni.models.projector.builder import build_projector
from omni.utils.fsdp_utils import FSDPMixin
from omni.utils.import_utils import is_xformers_available
from omni.utils.loguru import logger
from omni.utils.misc import check_path_and_file
from omni.utils.modeling_utils import get_model_device, get_model_dtype
from omni.utils.torch_utils import is_compiled_module, randn_tensor
from omni.models.transparent_vae import TransparentVAE
from safetensors.torch import load_file

PluginType = Literal["embedding", "head"]
PipelineImageType = (
    PIL.Image.Image | np.ndarray | torch.FloatTensor | list[PIL.Image.Image] | list[torch.FloatTensor] | list[np.ndarray]
)


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


def state_dict_to_cpu(state_dict):
    return {k: v.cpu() for k, v in state_dict.items()}


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class PluginBase(ABC, nn.Module, FSDPMixin):
    initializer_range: float = 0.02
    plugin_type: PluginType | None = None

    def _init_weights(self, module):
        std = self.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=std)

    @property
    def device(self):
        return get_model_device(self)

    @property
    def dtype(self):
        return get_model_dtype(self)

    @property
    @abstractmethod
    def processor(self):
        """
        In `MultimodalEmbedding`, it is similar to text tokenizer, it processes signals into a form that can be understood by `MultimodalEmbedding`.
        In `MultimodalHead`, it will process signals into a form that can be **trained** by `MultimodalHead`.
        """
        pass

    @property
    @abstractmethod
    def config(self):
        pass

    @abstractmethod
    def save_model(self, output_dir: str):
        pass

    @abstractmethod
    def load_model(self, output_dir: str):
        pass

    @abstractmethod
    def forward(self):
        pass


class MultimodalEmbedding(PluginBase):
    initializer_range: float = 0.02
    plugin_type: PluginType | None = "embedding"

    @property
    @abstractmethod
    def embed_len(self):
        """
        The length a signal after being processed by MultimodalEmbedding.
        """
        pass

    @property
    @abstractmethod
    def embed_dim(self):
        """
        The dimension of the embedding.
        """
        pass


class MultimodalHead(PluginBase):
    initializer_range: float = 0.02
    plugin_type: PluginType | None = "head"

    @abstractmethod
    @torch.no_grad()
    def pipeline(self):
        pass


# embedding modules
class DreamEmbedding(MultimodalEmbedding):
    def __init__(
        self,
        pretrained_model_name_or_path: str | None = None,
        num_dream_queries: int = 64,
        embed_hidden_size: int = 4096,
        freeze_dream_queries: bool = False,
    ):
        super().__init__()
        self.save_model_name = "dream_embedding"
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.num_dream_queries = num_dream_queries
        self.embed_hidden_size = embed_hidden_size
        self.freeze_dream_queries = freeze_dream_queries

        self.dream_queries = nn.Parameter(torch.zeros(1, self.num_dream_queries, self.embed_hidden_size))
        self._init_weights(self.dream_queries)

        if pretrained_model_name_or_path is not None:
            self.load_model(pretrained_model_name_or_path)

        self.dream_queries.requires_grad_(not freeze_dream_queries)

    def fsdp_ignored_modules(self) -> list:
        ignored_modules = []
        if self.freeze_dream_queries:
            ignored_modules.append(self)
        return ignored_modules

    @property
    def processor(self):
        return None

    @property
    def embed_len(self):
        return self.num_dream_queries

    @property
    def embed_dim(self):
        return self.embed_hidden_size

    @property
    def config(self):
        return dict(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            num_dream_queries=self.num_dream_queries,
            embed_len=self.embed_len,
            embed_dim=self.embed_dim,
            freeze_dream_queries=self.freeze_dream_queries,
        )

    def save_model(self, output_dir: str):
        logger.info(f"saving `DreamEmbedding`...")
        torch.save(state_dict_to_cpu(self.state_dict()), os.path.join(output_dir, f"{self.save_model_name}.bin"))

    def load_model(self, output_dir: str):
        if check_path_and_file(output_dir, f"{self.save_model_name}.bin"):
            logger.info(f">>> loading `DreamEmbedding`... from {output_dir}")
            loading_status = self.load_state_dict(torch.load(os.path.join(output_dir, f"{self.save_model_name}.bin"), map_location="cpu"))
            logger.info(f"{loading_status}")
        # HACK: For compatibility
        if check_path_and_file(output_dir, f"dream_queries.pt"):
            self.dream_queries = torch.load(os.path.join(output_dir, "dream_queries.pt"), map_location="cpu")

    def forward(self, batch_size: int = 1):
        return self.dream_queries.repeat(batch_size, 1, 1)


class CLIPVisionEmbedding(MultimodalEmbedding):
    def __init__(
        self,
        clip_vision_model_name_or_path: str,
        projector_type: str = "linear",
        projector_depth: int = 1,
        projector_name_or_path: str = None,
        pretrained_model_name_or_path: str | None = None,
        use_additional_post_layernorm: bool = False,
        select_layer: int = -2,
        embed_hidden_size: int = 4096,
        freeze_clip_vision_model: bool = True,
        freeze_embedding_layers: bool = True,
        freeze_projector: bool = False,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.save_model_name = "clip_vision_embedding"
        self.clip_vision_model_name_or_path = clip_vision_model_name_or_path
        self.projector_type = projector_type
        self.projector_depth = projector_depth
        self.projector_name_or_path = projector_name_or_path
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.use_additional_post_layernorm = use_additional_post_layernorm
        self.select_layer = select_layer
        self.embed_hidden_size = embed_hidden_size
        self.freeze_clip_vision_model = freeze_clip_vision_model
        self.freeze_embedding_layers = freeze_embedding_layers
        self.freeze_projector = freeze_projector

        self.clip_image_processor = CLIPImageProcessor.from_pretrained(
            clip_vision_model_name_or_path, local_files_only=local_files_only
        )
        self.clip_vision_model = CLIPVisionModel.from_pretrained(
            clip_vision_model_name_or_path, local_files_only=local_files_only
        )

        projector_cfg = dict(
            projector=projector_type,
            freeze_projector=freeze_projector,
            depth=projector_depth,
            save_model_name=self.save_model_name,
            model_name_or_path=None,
        )
        self.projector = build_projector(
            projector_cfg, in_hidden_size=self.clip_vision_model.config.hidden_size, out_hidden_size=embed_hidden_size, bias=True
        )

        self._init_weights(self.projector)

        self.post_layernorm = (
            nn.LayerNorm(embed_hidden_size, eps=self.clip_vision_model.config.layer_norm_eps)
            if use_additional_post_layernorm
            else nn.Identity()
        )

        self.image_embed_len = (self.clip_vision_model.config.image_size // self.clip_vision_model.config.patch_size) ** 2

        if pretrained_model_name_or_path is not None:
            self.load_model(pretrained_model_name_or_path)

        if self.projector.load_model(projector_name_or_path):
            logger.info(f">>> loading `CLIPVisionEmbedding` projector from {projector_name_or_path}")

        self.clip_vision_model.requires_grad_(not freeze_clip_vision_model)
        if not freeze_clip_vision_model:
            for i in range(select_layer + 1, 0):
                self.clip_vision_model.vision_model.encoder.layers[i].requires_grad_(False)
            self.clip_vision_model.vision_model.post_layernorm.requires_grad_(False)

        # NOTE: must be set after `self.clip_vision_model.requires_grad_(not freeze_clip_vision_model)`
        self.clip_vision_model.vision_model.embeddings.requires_grad_(not freeze_embedding_layers)

        self.projector.requires_grad_(not freeze_projector)

    @property
    def processor(self):
        return self.clip_image_processor

    @property
    def embed_len(self):
        return self.image_embed_len

    @property
    def embed_dim(self):
        return self.embed_hidden_size

    @property
    def config(self) -> dict:
        return dict(
            clip_vision_model_name_or_path=self.clip_vision_model_name_or_path,
            clip_vision_model_config=self.clip_vision_model.config.to_dict(),
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            select_layer=self.select_layer,
            embed_len=self.embed_len,
            embed_dim=self.embed_dim,
            freeze_clip_vision_model=self.freeze_clip_vision_model,
            freeze_embedding_layers=self.freeze_embedding_layers,
            freeze_projector=self.freeze_projector,
        )

    def fsdp_ignored_modules(self) -> list:
        ignored_modules = []
        if self.freeze_clip_vision_model:
            ignored_modules.append(self.clip_vision_model)
        if self.freeze_projector:
            ignored_modules.append(self.projector)
        return ignored_modules

    def save_model(self, output_dir: str):
        logger.info(f"saving `CLIPVisionEmbedding`...")
        torch.save(state_dict_to_cpu(self.state_dict()), os.path.join(output_dir, f"{self.save_model_name}.bin"))

    def load_model(self, output_dir: str):
        if check_path_and_file(output_dir, f"{self.save_model_name}.bin"):
            logger.info(f">>> loading `CLIPVisionEmbedding`... from {output_dir}")
            loading_status = self.load_state_dict(torch.load(os.path.join(output_dir, f"{self.save_model_name}.bin"), map_location="cpu"))
            logger.info(f"{loading_status}")
        # HACK: For compatibility
        elif check_path_and_file(output_dir, f"clip_vision_model_projector.pt"):
            self.projector.load_state_dict(
                torch.load(os.path.join(output_dir, "clip_vision_model_projector.pt"), map_location="cpu")
            )
        else:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id=output_dir, filename=f"{self.save_model_name}.bin")
            logger.info(f">>> loading `CLIPVisionEmbedding`... from {output_dir}")
            loading_status = self.load_state_dict(torch.load(model_path, map_location="cpu"))
            logger.info(f"{loading_status}")

    def forward(self, images: torch.FloatTensor | None = None):
        # HACK: dummy forward to avoid `find_unused_parameters` error
        is_dummy = images is None
        if is_dummy:
            c, h, w = 3, self.processor.crop_size["height"], self.processor.crop_size["width"]
            images = torch.zeros(1, c, h, w, device=self.clip_vision_model.device, dtype=self.clip_vision_model.dtype)

        output = self.clip_vision_model(images, output_hidden_states=True)
        hidden_state = output.hidden_states[self.select_layer]
        image_features = hidden_state[:, 1:]

        image_embeds = self.projector(image_features)[-1]
        image_embeds = self.post_layernorm(image_embeds)

        if is_dummy:
            return (0.0 * image_embeds).sum()
        else:
            return image_embeds


# head modules
class StableDiffusion3Head(MultimodalHead):
    def __init__(
        self,
        diffusion_name_or_path: str,
        projector_type="linear",
        projector_depth: int = 1,
        projector_name_or_path: str = None,
        pretrained_model_name_or_path: str = None,
        embed_hidden_size: int = 4096,
        drop_prob: float | None = None,
        input_perturbation: float = 0.0,
        snr_gamma: float | None = None,
        resolution: int = 1024,
        center_crop: bool = True,
        random_flip: bool = True,
        freeze_vae: bool = True,
        freeze_transformer: bool = True,
        freeze_projector: bool = False,
        local_files_only: bool = False,
        weighting_scheme: Literal["sigma_sqrt", "logit_normal", "mode", "cosmap"] | None = "logit_normal",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        mode_scale: float = 1.29,
    ):
        super().__init__()
        self.save_model_name = "stable_diffusion3_head"
        self.diffusion_name_or_path = diffusion_name_or_path
        self.projector_type = projector_type
        self.projector_depth = projector_depth
        self.projector_name_or_path = projector_name_or_path
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.embed_hidden_size = embed_hidden_size
        self.drop_prob = drop_prob  # Recommended value is 0.1
        self.input_perturbation = input_perturbation  # Recommended value is 0.1
        self.snr_gamma = snr_gamma  # Recommended value is 5.0
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.freeze_vae = freeze_vae
        self.freeze_transformer = freeze_transformer
        self.freeze_projector = freeze_projector
        self.weighting_scheme = weighting_scheme
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.mode_scale = mode_scale

        self.vae = AutoencoderKL.from_pretrained(diffusion_name_or_path, subfolder="vae", local_files_only=local_files_only)
        
        self.transformer = SD3Transformer2DModel.from_pretrained(
            diffusion_name_or_path, subfolder="transformer", local_files_only=local_files_only
        )

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            diffusion_name_or_path, subfolder="scheduler", local_files_only=local_files_only
        )
        projector_cfg = dict(
            projector=projector_type,
            freeze_projector=freeze_projector,
            depth=projector_depth,
            save_model_name=self.save_model_name,
            model_name_or_path=None,
        )
        self.projector = build_projector(
            projector_cfg, in_hidden_size=embed_hidden_size, out_hidden_size=self.transformer.config.joint_attention_dim, bias=False
        )
        self._init_weights(self.projector)
        pooled_projector_cfg = dict(
            projector=projector_type,
            freeze_projector=freeze_projector,
            depth=projector_depth,
            save_model_name=self.save_model_name,
            model_name_or_path=None,
        )
        self.pooled_projector = build_projector(
            pooled_projector_cfg, in_hidden_size=embed_hidden_size, out_hidden_size=self.transformer.config.pooled_projection_dim, bias=False
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.default_sample_size = self.transformer.config.sample_size
        self.patch_size = self.transformer.config.patch_size
        self._callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "negative_pooled_prompt_embeds"]

        if pretrained_model_name_or_path is not None:
            self.load_model(pretrained_model_name_or_path)
            
        if self.projector.load_model(projector_name_or_path):
            logger.info(">>> loading `StableDiffusion3Head` projector from {projector_name_or_path}")
        if self.pooled_projector.load_model(projector_name_or_path):
            logger.info(">>> loading `StableDiffusion3Head` pooled_projector from {projector_name_or_path}")

        self.vae.requires_grad_(not freeze_vae)
        self.transformer.requires_grad_(not freeze_transformer)
        self.projector.requires_grad_(not freeze_projector)
        self.pooled_projector.requires_grad_(not freeze_projector)

    def processor_size(self, size):
        return transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if self.center_crop else transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip() if self.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    @property
    def processor(self):
        return transforms.Compose(
            [
                transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.resolution) if self.center_crop else transforms.RandomCrop(self.resolution),
                transforms.RandomHorizontalFlip() if self.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    @property
    def config(self):
        return dict(
            diffusion_name_or_path=self.diffusion_name_or_path,
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            embed_hidden_size=self.embed_hidden_size,
            drop_prob=self.drop_prob,
            input_perturbation=self.input_perturbation,
            snr_gamma=self.snr_gamma,
            freeze_vae=self.freeze_vae,
            freeze_transformer=self.freeze_transformer,
            freeze_projector=self.freeze_projector,
        )

    def fsdp_ignored_modules(self) -> list:
        ignored_modules = []
        if self.freeze_vae:
            ignored_modules.append(self.vae)
        if self.freeze_transformer:
            ignored_modules.append(self.transformer)
        if self.freeze_projector:
            ignored_modules.append(self.projector)
            ignored_modules.append(self.pooled_projector)
        return ignored_modules

    def save_model(self, output_dir: str):
        logger.info(f"saving `StableDiffusionHead`...")
        torch.save(state_dict_to_cpu(self.state_dict()), os.path.join(output_dir, f"{self.save_model_name}.bin"))

    def load_model(self, output_dir: str):
        if check_path_and_file(output_dir, f"{self.save_model_name}.bin"):
            logger.info(f">>> loading `StableDiffusionHead`... from {output_dir}")
            loading_status = self.load_state_dict(torch.load(os.path.join(output_dir, f"{self.save_model_name}.bin"), map_location="cpu"), strict=False)
            logger.info(f"{loading_status}")
            # ckpt = torch.load(os.path.join(output_dir, f"{self.save_model_name}.bin"), map_location="cpu")
            # self.projector.projector.weight.data = ckpt["projector.projector.weight"].float()
        # HACK: For compatibility
        elif check_path_and_file(output_dir, f"unet_projector.pt"):
            self.projector.load_state_dict(torch.load(os.path.join(output_dir, "unet_projector.pt"), map_location="cpu"))
        else:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id=output_dir, filename=f"{self.save_model_name}.bin")
            logger.info(f">>> loading `StableDiffusionHead`... from {output_dir}")
            loading_status = self.load_state_dict(torch.load(model_path, map_location="cpu"))
            logger.info(f"{loading_status}")

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.scheduler.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def forward(
        self,
        images: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        u_encoder_hidden_states: torch.FloatTensor | None = None,
        dream_embeddings: torch.FloatTensor | None = None,
    ):
        # HACK: avoid `find_unused_parameters` error
        is_dummy = images == None
        if is_dummy:
            assert dream_embeddings is not None, "You must provide `dream_embeddings` when dummy forward."
            dummy_image_features = torch.zeros(
                1, dream_embeddings.shape[1], self.embed_hidden_size, device=self.device, dtype=self.dtype
            )
            dummy_image_features = self.projector(dummy_image_features)[-1]
            
            if self.freeze_unet:
                return (0.0 * dummy_image_features).sum() + (0.0 * dream_embeddings).sum()
            else:
                dummy_noisy_latents = torch.zeros(1, 4, 64, 64, device=self.device, dtype=self.dtype)
                dummy_timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (1,), device=self.device).long()
                dummy_model_pred = self.unet(dummy_noisy_latents, dummy_timesteps, dummy_image_features).sample
                return (0.0 * dummy_model_pred).sum() + (0.0 * dream_embeddings).sum()

        # Convert images to latent space
        latents = self.vae.encode(images).latent_dist.sample()
        latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        
        assert (
            encoder_hidden_states.shape[0] == latents.shape[0]
        ), f"encoder_hidden_states.shape[0]: {encoder_hidden_states.shape[0]} != latents.shape[0]: {latents.shape[0]}"
        bsz = latents.shape[0]

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)

        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=bsz,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            mode_scale=self.mode_scale,
        )
        indices = (u * self.scheduler.config.num_train_timesteps).long()
        timesteps = self.scheduler.timesteps[indices].to(device=latents.device)

        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
        noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
        
        prompt_embeds = self.projector(encoder_hidden_states)[-1]
        global_encoder_hidden_states = encoder_hidden_states.mean(dim=1)
        pooled_prompt_embeds = self.pooled_projector(global_encoder_hidden_states)[-1]
        
        model_pred = self.transformer(
                hidden_states=noisy_model_input,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]
        
        # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
        # Preconditioning of the model outputs.
        # if args.precondition_outputs:
        model_pred = model_pred * (-sigmas) + noisy_model_input
        
        # these weighting schemes use a uniform timestep sampling
        # and instead post-weight the loss
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.weighting_scheme, sigmas=sigmas)
        
        # flow matching loss
        # if args.precondition_outputs:
        target = latents
        # else:
        #     target = noise - model_input
        
        # Compute loss
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(
                target.shape[0], -1
            ),
            1,
        )
        loss = loss.mean()

        if is_dummy:
            loss = 0.0 * loss

        return loss

    def check_inputs(
        self,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if (
            height % (self.vae_scale_factor * self.patch_size) != 0
            or width % (self.vae_scale_factor * self.patch_size) != 0
        ):
            raise ValueError(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * self.patch_size} but are {height} and {width}."
                f"You can use height {height - height % (self.vae_scale_factor * self.patch_size)} and width {width - width % (self.vae_scale_factor * self.patch_size)}."
            )
        
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt_embeds is None:
            raise ValueError("Provide `prompt_embeds`. Cannot leave `prompt_embeds` undefined.")

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
        
        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )
        
        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.noise_scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.noise_scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs

    @property
    def guidance_scale(self):
        return self._guidance_scale
    
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def pipeline(
        self,
        # prompt: str | list[str] | MultimodalContent = None,
        # prompt_2: str | list[str] | MultimodalContent | None = None,
        # prompt_3: str | list[str] | MultimodalContent | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 28,
        timesteps: list[int] = None,
        guidance_scale: float = 7.0,
        # negative_prompt: str | list[str] | MultimodalContent | None = None,
        # negative_prompt_2: str | list[str] | MultimodalContent | None = None,
        # negative_prompt_3: str | list[str] | MultimodalContent | None = None,
        num_images_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.FloatTensor | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
        pooled_prompt_embeds: torch.FloatTensor | None = None,
        negative_pooled_prompt_embeds: torch.FloatTensor | None = None,
        output_type: Literal["latent", "pt", "np", "pil"] | None = "pil",
        joint_attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used instead
            negative_prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used instead
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 256): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """
        # NOTE: The decoder only considers embedding inputs, so there is no raw prompt.
        # 0. Default height and width to transformer
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            height,
            width,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )
        
        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        batch_size = prompt_embeds.shape[0]

        device = self.device

        # 3. Encode input prompt
        # NOTE: After mounting to LLM, LLM takes on the task.
        assert prompt_embeds is not None, "`prompt_embeds` must be provided by LLM."
        pooled_prompt_embeds = prompt_embeds.mean(1)
        prompt_embeds = self.projector(prompt_embeds)[-1]
        pooled_prompt_embeds = self.pooled_projector(pooled_prompt_embeds)[-1]

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            assert (
                negative_prompt_embeds is not None
            ), "When using classifier free guidance, `negative_prompt_embeds` must be provided by LLM."
            pooled_negative_prompt_embeds = negative_prompt_embeds.mean(1)
            negative_prompt_embeds = self.projector(negative_prompt_embeds)[-1]
            negative_pooled_prompt_embeds = self.pooled_projector(pooled_negative_prompt_embeds)[-1]
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        # self.maybe_free_model_hooks()

        return image

