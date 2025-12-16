from typing import Any, Dict, List, Optional
import torch
from collections import defaultdict
import numpy as np
from transformers import PreTrainedTokenizer, ProcessorMixin

import os
from jinja2 import Template
from datasets import load_dataset
import json
import base64
from PIL import Image

from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils import torch_functional as VF
from verl.utils.dataset import RLHFDataset, process_image, process_video

def encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image object to Base64 string"""
    from io import BytesIO
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.empty(len(value), dtype=object)
        non_tensors[key][:] = value

    return {**tensors, **non_tensors}

class ReplanDataset(RLHFDataset):
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        task_name: str = "",
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        use_image_base64: bool = True,
        video_key: str = "videos",
        extra_info_key: str = "extra_info",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
        sampling_rate: float = 1.0,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.task_name = task_name
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.extra_info_key = extra_info_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.use_image_base64 = use_image_base64

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # When using dataset builder, always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # Load remote dataset from HuggingFace Hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

        if sampling_rate < 1.0:
            total_samples = len(self.dataset)
            num_samples = int(total_samples * sampling_rate)
            if num_samples > 0:
                import random
                indices = random.sample(range(total_samples), num_samples)
                indices.sort() 
                self.dataset = self.dataset.select(indices)
                print(f"Sampled data size: {num_samples}/{total_samples} (sampling rate: {sampling_rate})")
            else:
                print(f"Warning: Sampling rate {sampling_rate} results in 0 samples, keeping at least 1 sample")
                self.dataset = self.dataset.select([0])

    def _filter_overlong_prompts(self, example: Dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            if not isinstance(images, list):
                images = [images]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # Image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # Text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))
            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if not isinstance(videos, list):
                videos = [videos]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # Video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(process_video(video, self.min_pixels, self.max_pixels, self.video_fps))

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length


    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            if "detection" in self.task_name:
                prompt_str = "Please locate: s" + prompt_str
            if self.image_key in example and "<image>" not in prompt_str:
                prompt_str = "<image>" + "\n" + prompt_str
            prompt_str = format_prompt.render(content=prompt_str)
        # if not ("<image>" in prompt_str or "<video>" in prompt_str):
        #     raise RuntimeError("Prompt must contain either '<image>' or '<video>' placeholder")

        # if "<image>" in prompt_str and (self.image_key not in example):
        #     raise RuntimeError(f"Image key {self.image_key} not found in example")

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})
                
                # if len(prompt_str.split("<image>")) <= 1:
                #     raise RuntimeError("Image prompt must contain at least one '<image>' placeholder")

            return [{"role": "user", "content": content_list}]
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    
    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            if not isinstance(images, list):
                images = [images]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # Image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # Text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            
            # if len(images) == 0:
            #     raise RuntimeError("Images list cannot be empty when image_key is present")

            example["multi_modal_data"] = {"images": images}  # NOTE: Support various image formats from different datasets, but may cause communication overhead
            if self.use_image_base64:
                example["input_image_base64"] = [encode_image_to_base64(image) for image in images]
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if not isinstance(videos, list):
                videos = [videos]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # Video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}  # NOTE: Support various video formats from different datasets, but may cause communication overhead
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # Qwen2VL mrope
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key, None)
        example["task_name"] = self.task_name
        extra_info_str = example.pop(self.extra_info_key, None)
        if extra_info_str is not None:
            extra_info = json.loads(extra_info_str.strip())
            example["extra_info"] = extra_info
        else:
            example["extra_info"] = None
        return example

