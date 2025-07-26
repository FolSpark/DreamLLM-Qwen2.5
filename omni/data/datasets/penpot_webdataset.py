import random
import base64
import re
import pickle
from typing import Callable

import webdataset as wds
from io import BytesIO
from PIL import Image

from omni.conversation.conversation import Dialog, Message
from omni.conversation.multimodal import ModalType, MultimodalContent, Unimodal
from omni.utils.loguru import logger

from ..manager.dataset_info import WebDatasetInfo
from ..manager.dataset_type import InterleavedImageTextReturnType, InstructInterleavedImageTextReturnType
from .base_dataset import InstructInterleavedImageTextDataset


def filter_no_text_or_no_image(sample):
    return (b"text_list" in sample["json"]) and (b"image_info" in sample["json"])

penpot_instruction_prompt = [
    "Please generate a {}",
    "Please create a {}",
    "Generate a {}",
    "I need a {}",
    "Produce a {}",
    "Can you design a {}",
    "Please provide a {}",
    "Create a {}",
    "I would like a {}",
    "Could you generate a {}",
    "Please design a {}",
    "I need you to create a {}",
    "Generate a {}",
    "Can you produce a {}",
    "Design a {}",
    "Please create a {}",
    "I would like you to make a {}",
    "Create a {}",
    "Could you make a {}",
    "Please generate a {}",
    "Produce a {}",
]

penpot_text_prompt = [
    ", with the words of {}.",
    ", including the text {}.",
    ", featuring the text {}.",
    ", with the wording {}.",
    ", containing the words {}.",
    ", with the following text: {}.",
    ", that says {}.",
    ", using the words {}.",
    ", incorporating the words {}.",
    ", including the text {}.",
    ", with the phrase {}.",
    ", that includes {}.",
    ", with the words {}.",
    ", featuring {}?",
    ", containing the text {}.",
    ", with the words {}.",
    ", featuring {}.",
    ", using {} as the text.",
    ", with {}?",
    ", incorporating the words {}.",
    ", including {} as the text.",
]

penpot_industry_prompt = [
    " The industry category is {}.",
    " Industry: {}.",
    " The industry type is {}.",
    " Industry category: {}.",
    " Relevant industry: {}.",
    " Industry: {}.",
    " Industry type: {}.",
    " Specify industry: {}.",
    " Industry category: {}.",
    " Industry: {}.",
    " Industry field: {}.",
    " Industry category is {}.",
    " Industry: {}.",
    " The industry category is {}.",
    " Industry sector: {}.",
    " The industry type is {}.",
    " Industry: {}.",
    " Industry category: {}.",
    " Industry type: {}.",
    " The industry is {}.",
    " The industry category: {}.",
]

penpot_channel_prompt = [
    " The channel category is {}.",
    " Channel: {}.",
    " The channel is {}.",
    " Channel category: {}.",
    " Relevant channel: {}.",
    " Channel: {}.",
    " Channel type: {}.",
    " Specify channel: {}.",
    " Channel category: {}.",
    " Channel: {}.",
    " Channel field: {}.",
    " Channel category is {}.",
    " Channel: {}.",
    " The channel category is {}.",
    " Channel sector: {}.",
    " The channel type is {}.",
    " Channel: {}.",
    " Channel category: {}.",
    " Channel type: {}.",
    " The channel is {}.",
    " The industry category: {}.",
]

penpot_design_prompt = [
    " The design category is {}.",
    " Design: {}.",
    " The design style is {}.",
    " Design category: {}.",
    " Relevant design: {}.",
    " Design: {}.",
    " Design type: {}.",
    " Specify design: {}.",
    " Design category: {}.",
    " Design: {}.",
    " Design field: {}.",
    " Design category is {}.",
    " Design: {}.",
    " The design category is {}.",
    " Design sector: {}.",
    " The design type is {}.",
    " Design: {}.",
    " Design category: {}.",
    " Design type: {}.",
    " The design is {}.",
    " The design category: {}.",
]

penpot_instruction_image_prompt = [
    " Contains the following images: ---instrcution_image_place_holder---.",
]

penpot_prompt_response = [
    "Here is a {}.",
    "Here is a {}.",
	"This is a {}.",
	"Here's your {}.",
	"Here you go, a {}.",
	"This is the {}.",
	"Here is the {} you requested.",
	"Here's a {}.",
	"Here is your {}.",
	"This is your {}.",
	"Here is a {} for you.",
	"Here's the {}.",
	"Here's the {} you asked for.",
	"This is a {}.",
	"Here you have a {}.",
	"Here's the {}.",
	"This is the {}.",
	"Here is a {}.",
	"Here's your {}.",
	"Here is the {}.",
	"This is a {}.",
]

class PenpotWebdataset(InstructInterleavedImageTextDataset):
    def __init__(self, dataset_info: WebDatasetInfo, **kwargs):
        """
        A unified Interleaved Image Text Pair Webdataset.

        Args:
            dataset_info (WebDatasetInfo): The dataset info.
            image_processor (Callable, optional): Post process image. Defaults to `lambda x: x`.
            text_processor (Callable, optional): Post process text. Defaults to `lambda x: x`.
            seed (int, optional): Seed used to scramble the dataset. Defaults to 42.
        """
        super().__init__(dataset_info)
        self.image_processor: Callable = kwargs.get("image_processor", lambda x: x)
        self.text_processor: Callable = kwargs.get("text_processor", lambda x: x)
        self.seed: int = kwargs.get("seed", 42)
        self.ignore_image: bool = kwargs.get("ignore_image", False)
        self.instruction_image_num: int = kwargs.get("instruction_image_num", 0)
        self.rng = random.Random(self.seed)

        self.roles_map = {
            "user": "user",
            "human": "user",
            "assistant": "assistant",
            "gpt": "assistant",
            "obj365": "assistant",
            "vg": "assistant",
        }

        self.data_pipeline = wds.DataPipeline(
            wds.ResampledShards(self.dataset_info.name, self.dataset_info.shard_list, deterministic=True),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, rng=random.Random(self.seed), handler=wds.warn_and_continue),
            # wds.map(self.to_return_type, handler=wds.warn_and_continue),
            wds.map(self.to_return_type),
        )

        self.inner_iter = iter(self.data_pipeline)

        logger.info(
            f"Interleaved Image-Text WebDataset {self.dataset_info.name} loaded. Approximate total number of samples: {self.dataset_info.approx_size}"
        )

    def to_return_type(self, sample):
        data = pickle.loads(sample['pkl'])
        indx = self.rng.randint(0, len(penpot_instruction_prompt)-1)

        json_file_name = data['json_file_name']
        page_id = data['page_id']
        board_id = data['board_id']
        text = data['text']
        meta_info = data['meta']
        answer = data['content']
        answer = penpot_prompt_response[indx].format(meta_info.get('type', 'poster')) + " " + answer    

        image_ids = re.findall(r'<image_id>(.*?)</image_id>', answer)
        if len(image_ids) > 0:
            text_list = re.split('|'.join(image_ids), answer)
            matched_text_index = list(range(len(image_ids)))
            matched_sim = [None] * len(image_ids)
            image_list = [base64.b64decode(data['images'][image_id]) for image_id in image_ids]
            # image_list = [Image.open(BytesIO(image)).convert("RGB") for image in image_list]
            image_list = [Image.open(BytesIO(image)).convert("RGBA") for image in image_list]
            # Filter images with h or w == 1
            new_size = [tuple(map(lambda x: max(x), zip(image.size, (2, 2)))) for image in image_list]
            image_list = [image if image.size == new_size[idx] else image.resize(new_size[idx]) for idx, image in enumerate(image_list)]

            sequential_index = range(len(matched_text_index))
            zipped_lists = zip(matched_text_index, sequential_index, image_list, matched_sim)
            sorted_pairs = sorted(zipped_lists)
            matched_text_index, _, image_list, matched_sim = zip(*sorted_pairs)
            matched_text_index = list(matched_text_index)
            image_list = list(image_list)
            matched_sim = list(matched_sim)
        else:
            text_list = [answer]
            matched_text_index = []
            image_list = []
            matched_sim = []
        
        prob_thresh = 0.5
        self.rng.shuffle(text)
        text_content =  ', '.join(["\"{}\"".format(t) for t in text])
        instruction = penpot_instruction_prompt[indx].format(meta_info.get('type', 'poster')) + penpot_text_prompt[indx].format(text_content) \
                    if len(text_content) > 0 else penpot_instruction_prompt[indx].format(meta_info.get('type', 'poster')) + "."
        if len(meta_info.get('industry_category', '')) > 0:
            instruction += penpot_industry_prompt[indx].format(meta_info['industry_category']) if self.rng.random() > prob_thresh else ''
        if len(meta_info.get('channel_category', '')) > 0:
            instruction += penpot_channel_prompt[indx].format(meta_info['channel_category']) if self.rng.random() > prob_thresh else ''
        if len(meta_info.get('design_category', '')) > 0:
            instruction += penpot_design_prompt[indx].format(meta_info['design_category']) if self.rng.random() > prob_thresh else ''

        # instruction_image_num = int(0.3 * len(image_list))
        instruction_image_num = min(len(image_list), self.instruction_image_num)
        instruction_image_idx = self.rng.sample(list(range(len(image_list))), instruction_image_num)
        instruction_image_idx.sort()
        if instruction_image_idx:
            instruction += penpot_instruction_image_prompt[0]

        # construct multimodal dialog
        dialog: Dialog = []
        # instruction
        dialog.append(
            Message(
                role=self.roles_map["user"],
                content=MultimodalContent(text=instruction),
            )
        )

        dialog.append(
            Message(
                role=self.roles_map["assistant"],
                content=None,
            )
        )

        return InstructInterleavedImageTextReturnType(
            dataset_type=self.dataset_type,
            image_list=[] if self.ignore_image else image_list,
            instruction_image_idx=[] if self.ignore_image else instruction_image_idx,
            text_list=text_list,
            matched_text_index=matched_text_index,
            matched_sim=matched_sim,
            instruction=instruction,
            dialog=dialog,
        )

    def __len__(self):
        return self.dataset_info.approx_size

    def __getitem__(self, idx) -> InterleavedImageTextReturnType:
        return next(self.inner_iter)
