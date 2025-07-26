# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn.functional as F

from swift.llm import to_device
from swift.utils import get_env_args, is_deepspeed_enabled
from swift.llm.template.base import Template
from swift.llm.template.constant import LLMTemplateType, MLLMTemplateType
from swift.llm.template.register import register_template
from swift.llm.template.template_inputs import StdTemplateInputs
from swift.llm.template.template_meta import TemplateMeta
from swift.llm.template.utils import Context, Word, findall
from swift.llm.template.template.utils import DEFAULT_SYSTEM, ChatmlTemplateMeta


@dataclass
class QwenTemplateMeta(ChatmlTemplateMeta):
    default_system: Optional[str] = DEFAULT_SYSTEM
    auto_add_bos: bool = False
    stop_words: List[Word] = field(default_factory=lambda: ['<|endoftext|>'])


@dataclass
class Qwen2_5TemplateMeta(QwenTemplateMeta):
    default_system: Optional[str] = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'


@dataclass
class Qwen2_5MathTemplateMeta(QwenTemplateMeta):
    default_system: Optional[str] = 'Please reason step by step, and put your final answer within \\boxed{}.'


class Qwen2VLTemplate(Template):
    image_token_id = 151655
    video_token_id = 151656
    placeholder_tokens = ['<|image_pad|>', '<|video_pad|>']
    version = 'v2'

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        from qwen_vl_utils import fetch_image, fetch_video
        assert media_type in {'image', 'video'}
        if media_type == 'image':
            inputs.images[index] = fetch_image({'image': inputs.images[index]})
            if self.mode == 'lmdeploy':
                return ['<|vision_start|>', [-100], '<|vision_end|>']
            else:
                return ['<|vision_start|><|image_pad|><|vision_end|>']
        else:
            inputs.videos[index] = fetch_video({'video': inputs.videos[index]}).to(torch.uint8)
            return ['<|vision_start|><|video_pad|><|vision_end|>']

    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [f'<|object_ref_start|>{ref}<|object_ref_end|>']

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [f'<|box_start|>{self._get_bbox_str(bbox)}<|box_end|>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        images = inputs.images
        videos = inputs.videos
        for media_type in ['images', 'videos']:
            if locals()[media_type]:
                if media_type == 'images':
                    media_token = self.image_token_id
                    media_inputs = processor.image_processor(
                        images=images, videos=None, return_tensors='pt', do_resize=False)
                    media_grid_thw = media_inputs['image_grid_thw']
                else:
                    media_inputs = processor.image_processor(
                        images=None, videos=videos, return_tensors='pt', do_resize=False)
                    media_grid_thw = media_inputs['video_grid_thw']
                    media_token = self.video_token_id
                    if self.version == 'v2_5':
                        from qwen_vl_utils import vision_process
                        media_inputs['second_per_grid_ts'] = [
                            processor.image_processor.temporal_patch_size / vision_process.FPS
                        ] * len(media_grid_thw)
                idx_list = findall(input_ids, media_token)
                merge_length = processor.image_processor.merge_size**2

                def _get_new_tokens(i):
                    token_len = (media_grid_thw[i].prod() // merge_length)
                    return [media_token] * token_len

                input_ids, labels = self._extend_tokens(input_ids, labels, idx_list, _get_new_tokens)
                encoded.update(media_inputs)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_training:
            return inputs
        input_ids = inputs['input_ids']
        _model = model.model
        if not hasattr(_model, 'embed_tokens'):
            _model = _model.model  # LoRA
        pixel_values = inputs.get('pixel_values')
        pixel_values_videos = inputs.get('pixel_values_videos')
        image_grid_thw = inputs.get('image_grid_thw')
        video_grid_thw = inputs.get('video_grid_thw')
        second_per_grid_ts = inputs.get('second_per_grid_ts')

        inputs_embeds = _model.embed_tokens(input_ids)

        dtype = model.visual.get_dtype() if self.version == 'v2' else model.visual.dtype
        if pixel_values is None and pixel_values_videos is None:  # plain-text
            if is_deepspeed_enabled():
                from PIL import Image
                images = [Image.new('RGB', (32, 32), (0, 0, 0))]
                media_inputs = self.processor.image_processor(images=images, videos=None, return_tensors='pt')
                device = input_ids.device
                media_inputs = to_device(media_inputs, device)
                pixel_values = media_inputs['pixel_values'].type(dtype)
                image_embeds = model.visual(pixel_values, grid_thw=media_inputs['image_grid_thw'])
                inputs_embeds += image_embeds.mean() * 0.
        else:
            if pixel_values is not None:
                pixel_values = pixel_values.type(dtype)
                image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(dtype)
                video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (input_ids == model.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # fix https://github.com/huggingface/transformers/pull/33487
        kwargs = {}
        if self.version == 'v2_5':
            kwargs = {'second_per_grid_ts': second_per_grid_ts}
        position_ids, _ = model.get_rope_index(
            input_ids, image_grid_thw, video_grid_thw, attention_mask=inputs['attention_mask'], **kwargs)
        return {'inputs_embeds': inputs_embeds, 'position_ids': position_ids.contiguous()}

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        second_per_grid_ts = self.gather_list(batch, 'second_per_grid_ts')
        if second_per_grid_ts:
            res['second_per_grid_ts'] = second_per_grid_ts
        for media_type in ['image', 'video']:
            grid_thw = [b[f'images'] for b in batch if b.get(f'images') is not None]
            if grid_thw:
                res[f'images'] = torch.concat(grid_thw)
        return res



class DreamLLMTemplate(Template):
    image_token_id = 151655
    placeholder_tokens = ['<|image_pad|>']

    def replace_tag(self, media_type: Literal['image'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        from qwen_vl_utils import fetch_image
        assert media_type in {'image'}
        if media_type == 'image':
            inputs.images[index] = fetch_image({'image': inputs.images[index]})
            if self.mode == 'lmdeploy':
                return ['<|vision_start|>', [-100]*256, '<|vision_end|>']
            else:
                return ['<|vision_start|>' + '<|image_pad|>'*256 + '<|vision_end|>']
        else:
            raise ValueError(f'Unsupported media type: {media_type}')

    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [f'<|object_ref_start|>{ref}<|object_ref_end|>']

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [f'<|box_start|>{self._get_bbox_str(bbox)}<|box_end|>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        images = inputs.images
        for media_type in ['images']:
            if locals()[media_type]:
                if media_type == 'images':
                    media_inputs = processor.image_processor(
                        images=images, return_tensors='pt')["pixel_values"]
                
                encoded.update({
                    "images": media_inputs,
                    "images_dm": None
                })

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # print('\n\n', '='*100, '\n', inputs, '\n', '='*100, '\n\n')
        return inputs
        
        if not self.is_training:
            return inputs
        input_ids = inputs['input_ids']
        _model = model.model
        if not hasattr(_model, 'embed_tokens'):
            _model = _model.model  # LoRA
        pixel_values = inputs.get('images')

        inputs_embeds = _model.embed_tokens(input_ids)

        dtype = model.model.vision_encoder.dtype
        if pixel_values is None:  # plain-text
            if is_deepspeed_enabled():
                from PIL import Image
                images = [Image.new('RGB', (448, 448), (0, 0, 0))]
                media_inputs = self.processor.image_processor(images=images, return_tensors='pt')
                device = input_ids.device
                media_inputs = to_device(media_inputs, device)
                pixel_values = media_inputs['pixel_values'].type(dtype)
                image_embeds = model.model.vision_encoder(pixel_values)
                inputs_embeds += image_embeds.mean() * 0.
        else:
            if pixel_values is not None:
                print('pixel_values', pixel_values.shape)
                pixel_values = pixel_values.type(dtype)
                image_embeds = model.model.vision_encoder(pixel_values)
                image_mask = (input_ids == self.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        return {'inputs_embeds': inputs_embeds}

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        # print('\n\n', '='*100, '\n', res, '\n', '='*100, '\n\n')
        # second_per_grid_ts = self.gather_list(batch, 'second_per_grid_ts')
        # if second_per_grid_ts:
        #     res['second_per_grid_ts'] = second_per_grid_ts
        for media_type in ['image']:
            grid_thw = [b[f'images'] for b in batch if b.get(f'images') is not None]
            if grid_thw:
                res[f'images'] = torch.concat(grid_thw)
        return res


register_template(
    QwenTemplateMeta(
        "dreamllm",
        default_system=('You are a helpful assistant.'),
        template_cls=DreamLLMTemplate,
    ))