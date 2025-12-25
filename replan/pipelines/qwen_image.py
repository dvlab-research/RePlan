# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
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

import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Union
from PIL import Image
import numpy as np
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import QwenImageLoraLoaderMixin
from diffusers.models import AutoencoderKLQwenImage
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput

from transformers.generation.utils import GenerationConfig
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel, QwenDoubleStreamAttnProcessor2_0


from replan.pipelines.flex_attn import prepare_flex_attention_inputs, QwenFlexAttentionProcessor, create_flex_block_mask
from replan.pipelines.replan import generate_default_attention_rules

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from PIL import Image
        >>> from diffusers import QwenImageEditPipeline
        >>> from diffusers.utils import load_image

        >>> pipe = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
        ... ).convert("RGB")
        >>> prompt = (
        ...     "Make Pikachu hold a sign that says 'Qwen Edit is awesome', yarn art style, detailed, vibrant colors"
        ... )
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(image, prompt, num_inference_steps=50).images[0]
        >>> image.save("qwenimage_edit.png")
        ```
"""


# Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
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


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height, None


class MultiRegionQwenImageEditPipeline(DiffusionPipeline, QwenImageLoraLoaderMixin):
    r"""
    The Qwen-Image-Edit pipeline for image editing.

    Args:
        transformer ([`QwenImageTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`Qwen2.5-VL-7B-Instruct`]):
            [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the
            [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.
        tokenizer (`QwenTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLQwenImage,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: Qwen2Tokenizer,
        processor: Qwen2VLProcessor,
        transformer: QwenImageTransformer2DModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=processor,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.latent_channels = self.vae.config.z_dim if getattr(self, "vae", None) else 16
        # QwenImage latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.vl_processor = processor
        self.tokenizer_max_length = 1024

        self.prompt_template_encode = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 64
        self.default_sample_size = 128

    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline._extract_masked_hidden
    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

    def encode_prompts_batch(
        self,
        prompts: List[str],
        image: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 1024,
        keep_image_tokens_flags: Optional[List[bool]] = None,  # New parameter
    ):
        """
        Encode multiple prompts in a single batch forward pass.
        
        Args:
            prompts (`List[str]`): List of prompts to encode (can include main prompt and hints).
            image (`torch.Tensor`, *optional*): Image tensor for encoding.
            device (`torch.device`, *optional*): Device to use.
            num_images_per_prompt (`int`): Number of images per prompt.
            max_sequence_length (`int`): Maximum sequence length.
            keep_image_tokens_flags (`List[bool]`, *optional*): 
                Flags for each prompt. True=keep image tokens (for main prompt), False=remove (for hints).
            
        Returns:
            Tuple: A tuple containing:
                - `prompt_embeds_list` (`List[torch.FloatTensor]`): List of embeddings for each prompt.
                - `prompt_embeds_mask_list` (`List[torch.FloatTensor]`): List of masks for each prompt.
        """
        device = device or self._execution_device
        
        # If not specified, default to keeping image tokens for all
        if keep_image_tokens_flags is None:
            keep_image_tokens_flags = [True] * len(prompts)
        
        all_embeds = []
        all_masks = []
        for i in range(len(prompts)):
            embeds, masks = self._get_qwen_prompt_embeds(
                prompt=prompts[i],
                image=image,
                device=device,
                keep_image_tokens=keep_image_tokens_flags[i],  # Pass flag
            )
            all_embeds.append(embeds)
            all_masks.append(masks)

        return all_embeds, all_masks

    def _create_attention_mask(
        self,
        attention_rules,
        num_patches,
        total_text_len,
        prompt_len,
        hint_lens,
        image_patch_indices_list,
        main_prompt_indices,
        image_prompt_indices,
        batch_size,
        num_images_per_prompt,
        device,
        mask_main_prompt_influence=False,
        symmetric_masking=False,
        delete_main_prompt=False,
    ):
        total_seq_len = total_text_len + 2 * num_patches
        attention_mask = torch.zeros(
            num_images_per_prompt * batch_size,
            self.transformer.config.num_attention_heads,
            total_seq_len,
            total_seq_len,
            device=device,
            dtype=torch.bool,
        )

        if attention_rules:
            # 1. Define component indices
            num_regions = len(image_patch_indices_list)
            
            # Get indices for text components
            text_indices = {}
            text_indices['Main Prompt'] = main_prompt_indices
            if image_prompt_indices:
                text_indices['Image Prompt'] = image_prompt_indices
            
            hint_start_idx = prompt_len
            for i in range(num_regions):
                hint_len = hint_lens[i]
                text_indices[f'Hint {i+1}'] = list(range(hint_start_idx, hint_start_idx + hint_len))
                hint_start_idx += hint_len
            
            # Get indices for image patch components
            all_bbox_patches = set()
            for indices in image_patch_indices_list:
                all_bbox_patches.update(indices)
            
            all_patches = set(range(num_patches))
            bg_patches = all_patches - all_bbox_patches

            patch_indices = {}
            for i, indices in enumerate(image_patch_indices_list):
                patch_indices[f'BBox {i+1}'] = list(indices)
            patch_indices['Background'] = list(bg_patches)

            # Helper to get all indices for a component name
            def get_indices(comp_name):
                if comp_name in text_indices:
                    return text_indices[comp_name]
                
                # Distinguish between Noise patches and Image patches
                if 'Noise' in comp_name:
                    base_comp_name = comp_name.replace('Noise ', '')
                    if base_comp_name in patch_indices:
                        # Noise patches are in the first group after text
                        return [total_text_len + i for i in patch_indices[base_comp_name]]
                elif 'Image' in comp_name:
                    base_comp_name = comp_name.replace('Image ', '')
                    if base_comp_name in patch_indices:
                        # Image patches are in the second group
                        return [total_text_len + num_patches + i for i in patch_indices[base_comp_name]]
                # Fallback for old component names (returns both noise and image)
                elif comp_name in patch_indices:
                    noise_indices = [total_text_len + i for i in patch_indices[comp_name]]
                    image_indices = [total_text_len + num_patches + i for i in patch_indices[comp_name]]
                    return noise_indices + image_indices
                    
                return []

            # 2. Populate attention_mask based on rules
            for (q_comp, k_comp), allowed in attention_rules.items():
                if allowed:
                    q_indices = get_indices(q_comp)
                    k_indices = get_indices(k_comp)
                    if q_indices and k_indices:
                        q_indices_tensor = torch.tensor(q_indices, device=device, dtype=torch.long)
                        k_indices_tensor = torch.tensor(k_indices, device=device, dtype=torch.long)
                        attention_mask[:, :, q_indices_tensor.view(-1, 1), k_indices_tensor.view(1, -1)] = True
            
        else:
            # Original attention mask logic
            # 1. Text-to-Text attention:
            # Main prompt attends to itself
            if prompt_len > 0:
                attention_mask[:, :, :prompt_len, :prompt_len] = True
            # Each hint attends to itself
            hint_start_idx = prompt_len
            for hint_len in hint_lens:
                attention_mask[:, :, hint_start_idx : hint_start_idx + hint_len, hint_start_idx : hint_start_idx + hint_len] = True
                hint_start_idx += hint_len

            # 2. Patch-to-Patch attention: all patches attend to each other (both noise and image)
            attention_mask[:, :, total_text_len:, total_text_len:] = True

            # 3. Patch-to-Text attention (Region Guidance)
            if not mask_main_prompt_influence and prompt_len > 0:
                # All patches attend to the main prompt
                attention_mask[:, :, total_text_len:, :prompt_len] = True

            # Specific patch regions attend to their corresponding hints (apply to both noise and image patches)
            hint_start_idx = prompt_len
            for i, patch_indices in enumerate(image_patch_indices_list):
                hint_len = hint_lens[i]
                if patch_indices:
                    for p_idx in patch_indices:
                        # Noise patches
                        attention_mask[:, :, total_text_len + p_idx, hint_start_idx : hint_start_idx + hint_len] = True
                        # Image patches
                        attention_mask[:, :, total_text_len + num_patches + p_idx, hint_start_idx : hint_start_idx + hint_len] = True
                hint_start_idx += hint_len

            # 4. (Optional) Symmetric Text-to-Patch attention
            if symmetric_masking:
                # Main prompt attends to all patches
                if not mask_main_prompt_influence and prompt_len > 0:
                    attention_mask[:, :, :prompt_len, total_text_len:] = True

                # Regional hints attend to their corresponding patches (both noise and image)
                hint_start_idx = prompt_len
                for i, patch_indices in enumerate(image_patch_indices_list):
                    hint_len = hint_lens[i]
                    if patch_indices:
                        for p_idx in patch_indices:
                            # Noise patches
                            attention_mask[:, :, hint_start_idx : hint_start_idx + hint_len, total_text_len + p_idx] = True
                            # Image patches
                            attention_mask[:, :, hint_start_idx : hint_start_idx + hint_len, total_text_len + num_patches + p_idx] = True
                    hint_start_idx += hint_len
            else:
                # Main prompt attends to all image patches
                if not mask_main_prompt_influence:
                    attention_mask[:, :, :prompt_len, total_text_len:] = True
                
                # All Regional hints attend to ALL image patches
                # hints range: [prompt_len : total_text_len]
                # image patches range: [total_text_len : end]
                attention_mask[:, :, prompt_len:total_text_len, total_text_len:] = True

        return attention_mask

    def process_region_guidance(
        self,
        prompt_embeds: torch.FloatTensor,
        prompt_embeds_mask: torch.FloatTensor,
        region_guidance: List[Dict],
        width: int,  # resized width
        height: int,  # resized height
        original_height: int,
        original_width: int,
        dtype: torch.dtype,
        device: torch.device,
        num_images_per_prompt: Optional[int] = 1,
        max_sequence_length: Optional[int] = 1024,
        mask_main_prompt_influence: bool = False,
        delete_main_prompt: bool = False,
        symmetric_masking: bool = False,
        attention_rules: Optional[Dict] = None,
        prompt_image: Optional[torch.Tensor] = None,
        main_prompt: Optional[Union[str, List[str]]] = None,  # Add main_prompt parameter
        return_attention_mask: bool = True,
    ):
        """
        Processes regional guidance to generate combined text embeddings and a self-attention mask.

        This function takes a list of regional guidance specifications (bounding boxes and text hints)
        and integrates them with the main prompt. It computes patch indices corresponding to each bounding
        box and constructs a custom self-attention mask for a sequence structured as `[text, image]`.

        Args:
            prompt_embeds (`torch.FloatTensor`): Embeddings for the main prompt (can be ignored if main_prompt is provided).
            prompt_embeds_mask (`torch.FloatTensor`): Mask for the main prompt embeddings.
            region_guidance (`List[Dict]`): A list where each dict contains a 'bbox' and a 'hint'.
            width (`int`): The target width of the image being generated.
            height (`int`): The target height of the image being generated.
            original_height (`int`): The original height provided by the user.
            original_width (`int`): The original width provided by the user.
            dtype (`torch.dtype`): The data type for new tensors.
            device (`torch.device`): The device for new tensors.
            num_images_per_prompt (`int`, *optional*): Number of images per prompt. Defaults to 1.
            max_sequence_length (`int`, *optional*): Max sequence length for text encoder. Defaults to 1024.
            mask_main_prompt_influence (`bool`, *optional*): If True, prevents image patches from attending to the main prompt. Defaults to False.
            symmetric_masking (`bool`, *optional*): If True, makes the text-image attention mask symmetric. Defaults to False.
            attention_rules (`Dict`, *optional*): Custom attention rules dictionary.
            prompt_image (`torch.Tensor`, *optional*): Image tensor for encoding hints.
            main_prompt (`str` or `List[str]`, *optional*): Main prompt text from __call__. If provided, will batch encode with hints.

        Returns:
            Tuple: A tuple containing:
                - `final_prompt_embeds` (`torch.FloatTensor`): Concatenated embeddings of the main prompt and all hints.
                - `final_prompt_embeds_mask` (`torch.FloatTensor`): Concatenated mask.
                - `attention_mask` (`torch.FloatTensor`): The 4D self-attention mask for the combined sequence.
                - `prompt_len` (`int`): The sequence length of the original prompt.
                - `hint_lens` (`List[int]`): A list of sequence lengths for each hint.
                - `image_patch_indices_list` (`List[List[int]]`): List of patch indices for each region.
        """
        if prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0] // num_images_per_prompt
        elif main_prompt is not None:
            if isinstance(main_prompt, str):
                batch_size = 1
            else:
                batch_size = len(main_prompt)
        else:
            batch_size = 1
        
        # Collect all prompts for batch encoding: [main_prompt] + [hint1, hint2, ...]
        all_prompts = []
        if main_prompt is not None:
            # Use provided main_prompt for batch encoding
            if isinstance(main_prompt, str):
                all_prompts.append(main_prompt)
            else:
                # If main_prompt is a list, assume it's a single batch item
                all_prompts.extend(main_prompt)
        
        # Add all hints
        hint_prompts = [guidance['hint'] for guidance in region_guidance]
        all_prompts.extend(hint_prompts)
        
        # Before calling encode_prompts_batch

        # Batch encode all prompts (main + hints) in one forward pass
        # Create keep_image_tokens_flags:
        #   - main prompt: True (always keep image tokens)
        #   - hints: False (remove image tokens, keep only text)
        if main_prompt is not None:
            num_main_prompts = 1 if isinstance(main_prompt, str) else len(main_prompt)
        else:
            num_main_prompts = 0

        keep_image_tokens_flags = [True] * num_main_prompts + [False] * len(region_guidance)

        all_embeds_list, all_masks_list = self.encode_prompts_batch(
            prompts=all_prompts,
            image=prompt_image,
            device=device,
            num_images_per_prompt=1,  # Start with 1, will expand later
            max_sequence_length=max_sequence_length,
            keep_image_tokens_flags=keep_image_tokens_flags,  # Pass flags
        )
        
        # Split results: first is main prompt, rest are hints
        if main_prompt is not None:
            main_prompt_count = 1 if isinstance(main_prompt, str) else len(main_prompt)
            main_embeds = all_embeds_list[0] if main_prompt_count == 1 else torch.cat(all_embeds_list[:main_prompt_count], dim=0)
            main_masks = all_masks_list[0] if main_prompt_count == 1 else torch.cat(all_masks_list[:main_prompt_count], dim=0)
            hint_embeds_list = all_embeds_list[main_prompt_count:]
            hint_masks_list = all_masks_list[main_prompt_count:]
        else:
            # Use the provided prompt_embeds
            main_embeds = prompt_embeds
            main_masks = prompt_embeds_mask
            hint_embeds_list = all_embeds_list
            hint_masks_list = all_masks_list
        
        # Handle batch_size and num_images_per_prompt expansion
        # Expand main prompt
        if main_prompt is not None:
            if batch_size > 1:
                main_embeds = main_embeds.repeat(batch_size, 1, 1)
                main_masks = main_masks.repeat(batch_size, 1)
            if num_images_per_prompt > 1:
                _, seq_len, hidden_dim = main_embeds.shape
                main_embeds = main_embeds.repeat(1, num_images_per_prompt, 1)
                main_embeds = main_embeds.view(batch_size * num_images_per_prompt, seq_len, hidden_dim)
                main_masks = main_masks.repeat(1, num_images_per_prompt)
                main_masks = main_masks.view(batch_size * num_images_per_prompt, -1)
        
        # Expand hints
        expanded_hint_embeds_list = []
        expanded_hint_masks_list = []
        for hint_embeds_i, hint_mask_i in zip(hint_embeds_list, hint_masks_list):
            if batch_size > 1:
                hint_embeds_i = hint_embeds_i.repeat(batch_size, 1, 1)
                hint_mask_i = hint_mask_i.repeat(batch_size, 1)
            
            if num_images_per_prompt > 1:
                _, seq_len, hidden_dim = hint_embeds_i.shape
                hint_embeds_i = hint_embeds_i.repeat(1, num_images_per_prompt, 1)
                hint_embeds_i = hint_embeds_i.view(batch_size * num_images_per_prompt, seq_len, hidden_dim)
                hint_mask_i = hint_mask_i.repeat(1, num_images_per_prompt)
                hint_mask_i = hint_mask_i.view(batch_size * num_images_per_prompt, -1)
            
            expanded_hint_embeds_list.append(hint_embeds_i)
            expanded_hint_masks_list.append(hint_mask_i)

        grid_h = height // self.vae_scale_factor // 2
        grid_w = width // self.vae_scale_factor // 2
        num_patches = grid_h * grid_w

        image_patch_indices_list = []
        for guidance in region_guidance:
            bbox = guidance["bbox"]
            x1, y1, x2, y2 = bbox
            scale_x = width / original_width
            scale_y = height / original_height

            start_col = int(math.floor(x1 * scale_x / self.vae_scale_factor / 2))
            end_col = int(math.ceil(x2 * scale_x / self.vae_scale_factor / 2))
            start_row = int(math.floor(y1 * scale_y / self.vae_scale_factor / 2))
            end_row = int(math.ceil(y2 * scale_y / self.vae_scale_factor / 2))

            start_col = max(0, start_col)
            end_col = min(grid_w, end_col)
            start_row = max(0, start_row)
            end_row = min(grid_h, end_row)

            patch_indices = []
            for r in range(start_row, end_row):
                for c in range(start_col, end_col):
                    patch_indices.append(r * grid_w + c)
            image_patch_indices_list.append(patch_indices)

        if delete_main_prompt:
            final_prompt_embeds = torch.cat(expanded_hint_embeds_list, dim=1)
            final_prompt_embeds_mask = torch.cat(expanded_hint_masks_list, dim=1)
            prompt_len = 0
        else:
            final_prompt_embeds = torch.cat([main_embeds] + expanded_hint_embeds_list, dim=1)
            final_prompt_embeds_mask = torch.cat([main_masks] + expanded_hint_masks_list, dim=1)
            prompt_len = main_embeds.shape[1]

        hint_lens = [h.shape[1] for h in expanded_hint_embeds_list]
        total_text_len = final_prompt_embeds.shape[1]
        
        # Calculate Main Prompt and Image Prompt indices if applicable
        main_prompt_indices = list(range(prompt_len))
        image_prompt_indices = []
        
        if not delete_main_prompt and main_embeds is not None and prompt_image is not None and main_prompt is not None:
             # Logic to split Main Prompt and Image Prompt
             vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
             vision_end_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
             
             # Use the first prompt as reference
             template = self.prompt_template_encode
             drop_idx = self.prompt_template_encode_start_idx
             txt = template.format(main_prompt[0] if isinstance(main_prompt, list) else main_prompt)
             model_inputs = self.processor(
                 text=[txt],
                 images=prompt_image,
                 padding=True,
                 return_tensors="pt",
             ).to(device)
             
             input_ids_sample = model_inputs.input_ids[0]
             attention_mask_sample = model_inputs.attention_mask[0]
             input_ids_sample = input_ids_sample[attention_mask_sample.bool()]
             
             vision_start_idx = None
             vision_end_idx = None
             for j, token_id in enumerate(input_ids_sample):
                 if token_id == vision_start_id:
                     vision_start_idx = j
                 elif token_id == vision_end_id and vision_start_idx is not None:
                     vision_end_idx = j + 1
                     break
             
             if vision_start_idx is not None and vision_end_idx is not None:
                 vision_start_idx = max(0, vision_start_idx - drop_idx)
                 vision_end_idx = max(0, vision_end_idx - drop_idx)
                 
                 image_prompt_indices = list(range(vision_start_idx, vision_end_idx))
                 main_prompt_indices = list(range(0, vision_start_idx)) + list(range(vision_end_idx, prompt_len))

        if not return_attention_mask:
             return final_prompt_embeds, final_prompt_embeds_mask, None, prompt_len, hint_lens, image_patch_indices_list, main_prompt_indices, image_prompt_indices

        # Self-attention mask for [text, noise_patches, image_patches] sequence
        attention_mask = self._create_attention_mask(
            attention_rules,
            num_patches,
            total_text_len,
            prompt_len,
            hint_lens,
            image_patch_indices_list,
            main_prompt_indices,
            image_prompt_indices,
            batch_size,
            num_images_per_prompt,
            device,
            mask_main_prompt_influence,
            symmetric_masking,
            delete_main_prompt,
        )

        return final_prompt_embeds, final_prompt_embeds_mask, attention_mask, prompt_len, hint_lens, image_patch_indices_list, main_prompt_indices, image_prompt_indices

    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        keep_image_tokens: bool = True,  # Default True - used for main prompt
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(e) for e in prompt]
        model_inputs = self.processor(
            text=txt,
            images=image,
            padding=True,
            return_tensors="pt",
        ).to(device)

        # All prompts keep image tokens during encoding (text encoder needs full context)
        outputs = self.text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        
        # Only when keep_image_tokens=False (hints), remove hidden states corresponding to image tokens
        if not keep_image_tokens and image is not None:
            # Find image tokens positions
            vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            vision_end_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
            
            new_split_hidden_states = []
            for i, (hidden, input_ids_sample) in enumerate(zip(split_hidden_states, model_inputs.input_ids)):
                # Keep only valid token ids (based on attention mask)
                input_ids_sample = input_ids_sample[model_inputs.attention_mask[i].bool()]
                
                # Find vision tokens range
                vision_start_idx = None
                vision_end_idx = None
                for j, token_id in enumerate(input_ids_sample):
                    if token_id == vision_start_id:
                        vision_start_idx = j
                    elif token_id == vision_end_id and vision_start_idx is not None:
                        vision_end_idx = j + 1  # Include vision_end token
                        break
                
                # If vision tokens found, remove their corresponding hidden states
                if vision_start_idx is not None and vision_end_idx is not None:
                    before_vision = hidden[:vision_start_idx]
                    after_vision = hidden[vision_end_idx:]
                    hidden = torch.cat([before_vision, after_vision], dim=0)
                
                new_split_hidden_states.append(hidden)
            split_hidden_states = new_split_hidden_states
        
        # drop_idx truncation (remove template prefix)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, encoder_attention_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        image: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 1024,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            image (`torch.Tensor`, *optional*):
                image to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt, image, device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        return prompt_embeds, prompt_embeds_mask

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds_mask=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
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
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and prompt_embeds_mask is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `prompt_embeds_mask` also have to be passed. Make sure to generate `prompt_embeds_mask` from the same text encoder that was used to generate `prompt_embeds`."
            )
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_prompt_embeds_mask` also have to be passed. Make sure to generate `negative_prompt_embeds_mask` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 1024:
            raise ValueError(f"`max_sequence_length` cannot be greater than 1024 but is {max_sequence_length}")

    @staticmethod
    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline._pack_latents
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline._unpack_latents
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)

        return latents

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i], sample_mode="argmax")
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        image_latents = (image_latents - latents_mean) / latents_std

        return image_latents

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def prepare_latents(
        self,
        image,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, 1, num_channels_latents, height, width)

        image_latents = None
        if image is not None:
            image = image.to(device=device, dtype=dtype)
            if image.shape[1] != self.latent_channels:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            else:
                image_latents = image
            if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                # expand init_latents for batch_size
                additional_image_per_prompt = batch_size // image_latents.shape[0]
                image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
            elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                )
            else:
                image_latents = torch.cat([image_latents], dim=0)

            image_latent_height, image_latent_width = image_latents.shape[3:]
            image_latents = self._pack_latents(
                image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
            )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, image_latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    def set_attn_processor(self, attn_processor_class, **kwargs):
        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor_class, **kwargs):
            if hasattr(module, "set_processor"):
                module.set_processor(processor_class(**kwargs))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor_class, **kwargs)

        fn_recursive_attn_processor("transformer", self.transformer, attn_processor_class, **kwargs)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        region_guidance: Optional[List[Dict]] = None, # [{'bbox': [x1, y1, x2, y2], 'hint': 'xxx'}, ...]
        mask_main_prompt_influence: bool = False,
        symmetric_masking: bool = False,
        delete_main_prompt: bool = False,
        attention_rules: Optional[Dict] = None,
        enable_flex_attn: bool = True,
        flex_attn_use_bitmask: bool = True,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                When > 1.0 and a provided `negative_prompt`, enables true classifier-free guidance.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.

                This parameter in the pipeline is there to support future guidance-distilled models when they come up.
                Note that passing `guidance_scale` to the pipeline is ineffective. To enable classifier-free guidance,
                please pass `true_cfg_scale` and `negative_prompt` (even an empty negative prompt like " ") should
                enable classifier-free guidance computations.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.qwenimage.QwenImagePipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
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
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.
            region_guidance (`List[Dict]`, *optional*): Regional guidance specifications.
            visualize_attention_path (`str`, *optional*): Path to save attention mask visualization.
            mask_main_prompt_influence (`bool`, *optional*): If True, prevents image patches from attending to main prompt.
            symmetric_masking (`bool`, *optional*): If True, makes text-image attention symmetric.
            delete_main_prompt (`bool`, *optional*): If True, removes main prompt from attention.
            attention_rules (`Dict`, *optional*): Custom attention rules dictionary.

        Examples:

        Returns:
            [`~pipelines.qwenimage.QwenImagePipelineOutput`] or `tuple`:
            [`~pipelines.qwenimage.QwenImagePipelineOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is a list with the generated images.
        """
        image_size = image[0].size if isinstance(image, list) else image.size
        original_height, original_width = height or image_size[1], width or image_size[0]
        calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
        height = height or calculated_height
        width = width or calculated_width

        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of
        image = image.resize((width, height))
        calculated_width = width
        calculated_height = height


        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # 3. Preprocess image
        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            image = self.image_processor.resize(image, calculated_height, calculated_width)
            prompt_image = image
            image = self.image_processor.preprocess(image, calculated_height, calculated_width)
            image = image.unsqueeze(2)
        else:
            prompt_image = None

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        
        attention_mask = None
        hint_lens = []
        image_patch_indices_list = []
        prompt_len = 0

        step_attention_rules = {}
        has_image_prompt = prompt_image is not None
        if attention_rules is None:
            attention_rules = generate_default_attention_rules(region_guidance or [], delete_main_prompt=delete_main_prompt, bboxes_attend_to_each_other=True, has_image_prompt=has_image_prompt, symmetric_masking=symmetric_masking)
        elif isinstance(attention_rules, dict) and any(isinstance(k, int) for k in attention_rules.keys()):
            step_attention_rules = attention_rules
            attention_rules = generate_default_attention_rules(region_guidance or [], delete_main_prompt=delete_main_prompt, bboxes_attend_to_each_other=True, has_image_prompt=has_image_prompt, symmetric_masking=symmetric_masking)

        if region_guidance:
            # When region_guidance is provided, batch encode main prompt + hints together
            # Skip the first encode_prompt call to avoid redundant encoding
            prompt_embeds, prompt_embeds_mask, attention_mask, prompt_len, hint_lens, image_patch_indices_list, main_prompt_indices, image_prompt_indices = self.process_region_guidance(
                prompt_embeds=None,  # Not used when main_prompt is provided
                prompt_embeds_mask=None,  # Not used when main_prompt is provided
                region_guidance=region_guidance,
                width=width,
                height=height,
                original_height=original_height,
                original_width=original_width,
                dtype=self.text_encoder.dtype if self.text_encoder else torch.float32,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                mask_main_prompt_influence=mask_main_prompt_influence,
                symmetric_masking=symmetric_masking,
                delete_main_prompt=delete_main_prompt,
                attention_rules=attention_rules,
                prompt_image=prompt_image,
                main_prompt=prompt,  # Pass the original prompt to batch encode with hints
                return_attention_mask=not enable_flex_attn,
            )
        else:
            # When no region_guidance, encode prompt normally
            prompt_embeds, prompt_embeds_mask = self.encode_prompt(
                image=prompt_image,
                prompt=prompt,
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
            prompt_len = prompt_embeds.shape[1]
            # Initialize indices for consistency if needed or handle appropriately
            main_prompt_indices = list(range(prompt_len))
            image_prompt_indices = []

        total_text_len = prompt_embeds.shape[1]


        indices_map = None
        total_seq_len = 0

        if enable_flex_attn and region_guidance is not None:
            self.set_attn_processor(QwenFlexAttentionProcessor)
            
            grid_h = height // self.vae_scale_factor // 2
            grid_w = width // self.vae_scale_factor // 2
            num_patches = grid_h * grid_w
            
            total_seq_len = total_text_len + 2 * num_patches
            
            # Construct indices_map
            indices_map = {}
            
            # Text Components
            if not delete_main_prompt:
                indices_map['Main Prompt'] = main_prompt_indices
            
            if image_prompt_indices:
                indices_map['Image Prompt'] = image_prompt_indices
                
            # Hints
            current_txt_idx = prompt_len
            for i, h_len in enumerate(hint_lens):
                indices_map[f'Hint {i+1}'] = list(range(current_txt_idx, current_txt_idx + h_len))
                current_txt_idx += h_len
                
            # Image Components
            img_start_idx = total_text_len
            all_bbox_patches = set()
            for indices in image_patch_indices_list:
                all_bbox_patches.update(indices)
            
            all_patches = set(range(num_patches))
            bg_patches = list(all_patches - all_bbox_patches)
            
            def get_combined_indices(patch_indices):
                res = []
                for p in patch_indices:
                    res.append(img_start_idx + p)
                    res.append(img_start_idx + num_patches + p)
                return res

            def get_noise_indices(patch_indices):
                return [img_start_idx + p for p in patch_indices]

            def get_image_indices(patch_indices):
                return [img_start_idx + num_patches + p for p in patch_indices]

            indices_map['Background'] = get_combined_indices(bg_patches)
            indices_map['Noise Background'] = get_noise_indices(bg_patches)
            indices_map['Image Background'] = get_image_indices(bg_patches)
            
            for i, indices in enumerate(image_patch_indices_list):
                indices_map[f'BBox {i+1}'] = get_combined_indices(indices)
                indices_map[f'Noise BBox {i+1}'] = get_noise_indices(indices)
                indices_map[f'Image BBox {i+1}'] = get_image_indices(indices)
            
            block_mask = create_flex_block_mask(
                indices_map=indices_map,
                total_seq_len=total_seq_len,
                attention_rules=attention_rules,
                device=device,
                use_bitmask=flex_attn_use_bitmask
            )
            
            if self.attention_kwargs is None:
                self._attention_kwargs = {}
            self._attention_kwargs["flex_block_mask"] = block_mask
        else:
            self.set_attn_processor(QwenDoubleStreamAttnProcessor2_0)

        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                image=prompt_image,
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents = self.prepare_latents(
            image,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        img_shapes = [
            [
                (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
                (1, calculated_height // self.vae_scale_factor // 2, calculated_width // self.vae_scale_factor // 2),
            ]
        ] * batch_size

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )

        # 6. Denoising loop
        self.scheduler.set_begin_index(0)
        active_rules = attention_rules

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # Check for step-specific attention rules
                current_rules = step_attention_rules.get(i, attention_rules)
                if current_rules is not active_rules:
                    if enable_flex_attn and region_guidance is not None and indices_map is not None:
                        block_mask = create_flex_block_mask(
                            indices_map=indices_map,
                            total_seq_len=total_seq_len,
                            attention_rules=current_rules,
                            device=device,
                            use_bitmask=flex_attn_use_bitmask
                        )
                        if self.attention_kwargs is None:
                            self._attention_kwargs = {}
                        self._attention_kwargs["flex_block_mask"] = block_mask
                    elif region_guidance is not None:
                         attention_mask = self._create_attention_mask(
                            current_rules,
                            num_patches,
                            total_text_len,
                            prompt_len,
                            hint_lens,
                            image_patch_indices_list,
                            main_prompt_indices,
                            image_prompt_indices,
                            batch_size,
                            num_images_per_prompt,
                            device,
                            mask_main_prompt_influence,
                            symmetric_masking,
                            delete_main_prompt,
                        )
                    active_rules = current_rules

                self._current_timestep = t

                latent_model_input = latents
                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                
                # Prepare attention kwargs with region mask if available
                current_attention_kwargs = self.attention_kwargs.copy() if self.attention_kwargs else {}
                if attention_mask is not None:
                    current_attention_kwargs["attention_mask"] = attention_mask
                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        attention_kwargs=current_attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred[:, : latents.size(1)]

                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes,
                            txt_seq_lens=negative_txt_seq_lens,
                            attention_kwargs={},
                            return_dict=False,
                        )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

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

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return QwenImagePipelineOutput(images=[img.resize((original_width, original_height)) for img in image])



if __name__ == "__main__":
    image_path = "assets/crowd.png"
    image = Image.open(image_path).convert("RGB")
    global_prompt = "keep remaining part of image unchanged."
    region_guidance = [{"bbox_2d": [446, 98, 542, 356], "point_2d": [498, 180], "hint": "change the color of her shoes to red"}]
    pipeline = MultiRegionQwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")

    pipeline.to(torch.bfloat16)
    pipeline.to("cuda")

    attention_rules = generate_default_attention_rules(
        region_guidance, 
        delete_main_prompt=False, 
        bboxes_attend_to_each_other=True,
        has_image_prompt=True,
        symmetric_masking=False
    )

    inputs = {
        "image": image,
        "prompt": global_prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 50,
        "region_guidance": region_guidance,
        "attention_rules": attention_rules,
    }
    image = pipeline(**inputs).images[0]
    image.save("output_image.png")