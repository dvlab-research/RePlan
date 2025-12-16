import os
import json
import re
import time
from datetime import datetime
from typing import Optional
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from qwen_vl_utils import process_vision_info


def generate_default_attention_rules(
    bbox_data, 
    delete_main_prompt=False, 
    bboxes_attend_to_each_other=True, 
    has_image_prompt=False,
    symmetric_masking=False
):
    """Generate default attention rules dictionary"""
    num_regions = len(bbox_data) if bbox_data else 0
    components = []
    
    if not delete_main_prompt:
        components.append('Main Prompt')
    if has_image_prompt:
        components.append('Image Prompt')
    
    hint_components = [f'Hint {i+1}' for i in range(num_regions)]
    bbox_components = [f'BBox {i+1}' for i in range(num_regions)]
    
    components.extend(hint_components)
    components.extend(bbox_components)
    components.append('Background')
    
    rules = { (q, k): False for q in components for k in components }

    # 1. Self Attention (Basic)
    if not delete_main_prompt:
        rules[('Main Prompt', 'Main Prompt')] = True
    for comp in hint_components:
        rules[(comp, comp)] = True
        
    if has_image_prompt:
        # Image Prompt attends to itself
        rules[('Image Prompt', 'Image Prompt')] = True
        
        # Image Prompt 和所有其他组件互相 attend
        for comp in components:
            if comp != 'Image Prompt':
                rules[('Image Prompt', comp)] = True
                rules[(comp, 'Image Prompt')] = True

    # 2. Image Internal
    img_components = bbox_components + ['Background']
    if bboxes_attend_to_each_other:
        for q_comp in img_components:
            for k_comp in img_components:
                rules[(q_comp, k_comp)] = True
    else:
        # Only need to see Background
        for q_comp in img_components:
            rules[(q_comp, 'Background')] = True
            if q_comp == 'Background':
                for k in img_components: rules[('Background', k)] = True
            else:
                rules[(q_comp, q_comp)] = True # Self
    
    # 3. Cross Component
    if not delete_main_prompt:
        # Image sees Main Prompt
        for q_comp in img_components:
            rules[(q_comp, 'Main Prompt')] = True
        # Main Prompt sees Image
        for k_comp in img_components:
            rules[('Main Prompt', k_comp)] = True

    for i in range(num_regions):
        rules[(f'BBox {i+1}', f'Hint {i+1}')] = True
        
    if symmetric_masking:
        for i in range(num_regions):
            rules[(f'Hint {i+1}', f'BBox {i+1}')] = True
    else:
        # Text sees all image patches
        text_components = []
        if not delete_main_prompt:
            text_components.append('Main Prompt')
        text_components.extend(hint_components)
        
        for q_comp in text_components:
            for k_comp in img_components:
                rules[(q_comp, k_comp)] = True
    
    return rules


class RePlanPipeline:
    """
    Regional editing evaluator class
    Encapsulates regional editing functionality for batch evaluation
    """

    def __init__(
        self,
        vlm_ckpt_path: str = 'TainU/RePlan-Qwen2.5-VL-7B',
        diffusion_model_name: Optional[str] = None,
        pipeline_type: str = "flux",
        output_dir: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        vlm_prompt_template_path: str = "replan.txt",
        lora_path: Optional[str] = None,
        init_vlm: bool = True,
        sync_cuda_for_timing: bool = True,
        empty_cuda_cache: bool = False,
        image_dir: Optional[str] = None,
    ):
        """
        Initialize evaluator

        Args:
            diffusion_model_name: Diffusion model name (use this first)
            flux_model_name: FLUX model name (compatibility parameter)
            qwen_model_name: Qwen Image Edit model name (compatibility parameter)
            pipeline_type: Diffusion model type, 'flux' or 'qwen'
            vlm_ckpt_path: VLM model checkpoint path. Required if init_vlm is True.
            image_dir: Root directory of input image dataset, used for resolving relative paths
            output_dir: Root output directory for evaluation results
            device: Device type
            torch_dtype: Data type
            vlm_prompt_template_path: VLM prompt template path
            lora_path: LoRA weights path (optional)
            init_vlm: Whether to initialize VLM
            sync_cuda_for_timing: Whether timing code calls torch.cuda.synchronize()
            empty_cuda_cache: Whether to call torch.cuda.empty_cache() after each sample
        """
        self.device = device
        self.torch_dtype = torch_dtype
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.init_vlm = init_vlm
        self.pipeline_type = pipeline_type.lower()
        self.sync_cuda_for_timing = sync_cuda_for_timing
        self.empty_cuda_cache = empty_cuda_cache

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

        # Prompt template
        if self.init_vlm:
            if not vlm_ckpt_path:
                raise ValueError("When init_vlm=True, vlm_ckpt_path must be provided.")
            if not vlm_prompt_template_path:
                raise ValueError("When init_vlm=True, vlm_prompt_template_path must be provided.")
            if os.path.isabs(vlm_prompt_template_path):
                vlm_prompt_template_path = vlm_prompt_template_path
            else:   
                current_file_dir = os.path.dirname(os.path.abspath(__file__))
                vlm_prompt_template_path = os.path.join(current_file_dir, "prompt_template", vlm_prompt_template_path)
            with open(vlm_prompt_template_path, 'r', encoding='utf-8') as f:
                self.template = f.read()
        else:
            self.template = None

        self.vlm_ckpt_path = vlm_ckpt_path

        # These objects are lazily initialized
        self.vlm_model = None
        self.processor = None
        self.diffusion_pipe = None
        self.attn_processor_class = None

        # Initialize immediately (if you don't want immediate loading, change these two lines to lazy loading mode)
        if self.init_vlm:
            self._init_vlm()

        self._init_diffusion(diffusion_model_name, lora_path)

    # -------------------------------------------------------------------------
    # Initialization related
    # -------------------------------------------------------------------------

    def _init_vlm(self):
        """Initialize VLM model and processor"""
        if self.vlm_model is not None and self.processor is not None:
            return

        print("Initializing VLM model...")
        self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.vlm_ckpt_path,
            torch_dtype=self.torch_dtype,
            attn_implementation="flash_attention_2",
        ).to(self.device)

        print("Initializing VLM processor...")
        self.processor = AutoProcessor.from_pretrained(self.vlm_ckpt_path)
        print("VLM initialization complete!")

    def _init_diffusion(self, diffusion_model_name, lora_path=None):
        """Initialize diffusion model based on pipeline type"""
        if self.diffusion_pipe is not None:
            return

        if self.pipeline_type == "flux":
            from .flux_kontext import MultiRegionFluxKontextPipeline
            diffusion_model_name = diffusion_model_name or "black-forest-labs/FLUX.1-Kontext-dev"
            self.diffusion_pipe = MultiRegionFluxKontextPipeline.from_pretrained(
                diffusion_model_name,
                torch_dtype=self.torch_dtype
            ).to(self.device)
        elif self.pipeline_type == "qwen":
            from .qwen_image import MultiRegionQwenImageEditPipeline
            diffusion_model_name = diffusion_model_name or "Qwen/Qwen-Image-Edit"
            self.diffusion_pipe = MultiRegionQwenImageEditPipeline.from_pretrained(
                diffusion_model_name,
                torch_dtype=self.torch_dtype
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported pipeline type: {self.pipeline_type}. Supported types: 'flux', 'qwen'")

        if lora_path:
            print(f"Loading LoRA weights: {lora_path}")
            self.diffusion_pipe.load_lora_weights(lora_path, adapter_name="default")

        print("Diffusion model initialization complete!")

    def _ensure_vlm_initialized(self):
        if not self.init_vlm:
            raise RuntimeError("VLM not initialized. Please set init_vlm=True when initializing RePlanPipeline.")
        if self.vlm_model is None or self.processor is None:
            self._init_vlm()

    # -------------------------------------------------------------------------
    # VLM input/output and parsing
    # -------------------------------------------------------------------------

    def build_messages(self, instruction, image):
        """Build message format"""
        self._ensure_vlm_initialized()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": self.template + '\n' + instruction},
                ],
            }
        ]
        return messages

    def extract_bbox_from_response(self, response_text):
        """Extract bbox information from response"""
        try:
            region_pattern = r'<region>(.*?)</region>'
            match = re.search(region_pattern, response_text, re.DOTALL)

            if match:
                region_content = match.group(1).strip()
                bbox_data = json.loads(region_content)
                return bbox_data
            else:
                print("<region> tag not found")
                return None
        except Exception as e:
            print(f"Error parsing bbox information: {e}")
            return None

    def extract_gen_image_prompt(self, response_text):
        """Extract content from <gen_image> tag in response"""
        try:
            gen_image_pattern = r'<gen_image>(.*?)</gen_image>'
            match = re.search(gen_image_pattern, response_text, re.DOTALL)

            if match:
                return match.group(1).strip()
            else:
                print("<gen_image> tag not found")
                return None
        except Exception as e:
            print(f"Error parsing <gen_image> information: {e}")
            return None

    def get_vlm_response(self, instruction, image):
        """
        Get VLM response

        Args:
            instruction: User instruction
            image: Image input, can be a path (str) or PIL.Image object
        Returns:
            response_text (str), image_input_path_or_pil
        """
        self._ensure_vlm_initialized()

        image_input = image
        if isinstance(image, str):
            if not os.path.isabs(image):
                if self.image_dir is None:
                    raise ValueError("When using relative image paths, image_dir must be provided during initialization.")
                image_input = os.path.join(self.image_dir, image)
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Unsupported image type: {type(image)}. Supported types are str (path) and PIL.Image.")

        messages = self.build_messages(instruction, image_input)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.vlm_model.device)

        generated_ids = self.vlm_model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text_list = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        # Simplification: internally use str uniformly
        response_text = output_text_list[0] if isinstance(output_text_list, list) else output_text_list
        return response_text, image_input

    # -------------------------------------------------------------------------
    # bbox / drawing
    # -------------------------------------------------------------------------

    def expand_bbox(self, bbox, image_size, expand_value=0.15, expand_mode='ratio'):
        """Expand bbox boundaries"""
        if len(bbox) != 4:
            raise ValueError(f"bbox length should be 4, received: {bbox}")
        x1, y1, x2, y2 = map(float, bbox)
        width, height = image_size

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox: {bbox}")

        if expand_mode == 'pixels':
            expand_pixels = float(expand_value)
            x1_expanded = max(0, x1 - expand_pixels)
            y1_expanded = max(0, y1 - expand_pixels)
            x2_expanded = min(width, x2 + expand_pixels)
            y2_expanded = min(height, y2 + expand_pixels)

        elif expand_mode == 'ratio':
            bbox_width = x2 - x1
            bbox_height = y2 - y1

            expand_pixels_x = bbox_width * float(expand_value)
            expand_pixels_y = bbox_height * float(expand_value)

            x1_expanded = max(0, x1 - expand_pixels_x)
            y1_expanded = max(0, y1 - expand_pixels_y)
            x2_expanded = min(width, x2 + expand_pixels_x)
            y2_expanded = min(height, y2 + expand_pixels_y)

        else:
            raise ValueError("expand_mode must be 'pixels' or 'ratio'")

        return [int(x1_expanded), int(y1_expanded), int(x2_expanded), int(y2_expanded)]

    def draw_bbox_on_image(self, image, bbox, label="", color='red', width=3):
        """
        Draw a single bbox on the image
        """
        image_with_bbox = image.copy()
        draw = ImageDraw.Draw(image_with_bbox)

        # Try to load font, use default if failed
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20
            )
        except Exception:
            font = ImageFont.load_default()

        x1, y1, x2, y2 = bbox

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        # Draw label (if exists)
        if label:
            text_y = max(0, y1 - 25)  # Avoid text exceeding image top
            text_bbox = draw.textbbox((x1, text_y), label, font=font)
            draw.rectangle(
                [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
                fill=color,
                outline=color,
            )
            draw.text((x1, text_y), label, fill='white', font=font)

        return image_with_bbox

    def _get_components_for_rules(self, bbox_data, delete_main_prompt, has_image_prompt):
        num_regions = len(bbox_data) if bbox_data else 0
        components = []
        if not delete_main_prompt:
            components.append('Main Prompt')
        if has_image_prompt:
            components.append('Image Prompt')
        components.extend([f'Hint {i+1}' for i in range(num_regions)])
        components.extend([f'BBox {i+1}' for i in range(num_regions)])
        components.append('Background')
        return components

    def region_edit_with_attention(
        self,
        image,
        instruction,
        response,
        delete_main_prompt=False,
        replace_global_prompt=False,
        custom_global_prompt_text="keep remaining parts of this image unchanged",
        expand_value=0.15,
        expand_mode='ratio',
        attention_rules=None,
        visualize_attention=False,
        symmetric_masking=False,
        output_dir=None,
        bboxes_attend_to_each_other=True,
        only_save_image=False,
        skip_save=False,
        enable_flex_attn=True,
        flex_attn_use_bitmask=True,
        pipeline_kwargs=None
    ):
        """
        Perform regional editing using attention mechanism
        """

        # Parse input image
        if isinstance(image, str):
            origin_image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            origin_image = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}. Supported types are str (path) and PIL.Image.")

        image_size = origin_image.size
        width, height = image_size

        pipeline_kwargs = pipeline_kwargs or {}

        # Set output directory
        edit_folder = None
        if not skip_save:
            if output_dir:
                edit_folder = output_dir
            elif self.output_dir is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                edit_folder = os.path.join(self.output_dir, f"attn_edit_{timestamp}")
            else:
                raise ValueError("Both output_dir and self.output_dir are empty")
            os.makedirs(edit_folder, exist_ok=True)


        # Save configuration and text information
        response_text = response[0] if isinstance(response, list) else response
        bbox_data = self.extract_bbox_from_response(response_text)

        # Auto-generate attention rules
        if attention_rules is None and bbox_data:
            attention_rules = generate_default_attention_rules(
                bbox_data, 
                delete_main_prompt, 
                bboxes_attend_to_each_other, 
                symmetric_masking=False,
                has_image_prompt=(self.pipeline_type == "qwen")
            )

        if not only_save_image and not skip_save:
            info_path = os.path.join(edit_folder, "config.json")
            config_info = {
                "instruction": instruction,
                "response": response_text,
                "delete_main_prompt": delete_main_prompt,
                "replace_global_prompt": replace_global_prompt,
                "custom_global_prompt_text": custom_global_prompt_text,
                "expand_value": expand_value,
                "expand_mode": expand_mode,
                "enable_flex_attn": enable_flex_attn,
                "flex_attn_use_bitmask": flex_attn_use_bitmask,
            }
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(config_info, f, ensure_ascii=False, indent=2, sort_keys=True)

            text_info_path = os.path.join(edit_folder, "instruction_and_response.txt")
            with open(text_info_path, 'w', encoding='utf-8') as f:
                f.write("--- User Instruction ---\n")
                f.write(instruction + "\n\n")
                f.write("--- VLM Response ---\n")
                f.write(response_text + "\n\n")

                if attention_rules:
                    f.write("--- Attention Rules ---\n")
                    components = self._get_components_for_rules(
                        bbox_data, delete_main_prompt, has_image_prompt=(self.pipeline_type == "qwen")
                    )

                    col_width = 15
                    header = f"{' ':<{col_width}}" + "".join(
                        [f"{comp:<{col_width}}" for comp in components]
                    )
                    f.write(header + "\n")
                    f.write("-" * len(header) + "\n")

                    for q_comp in components:
                        row_str = f"{q_comp:<{col_width}}"
                        for k_comp in components:
                            state = "✅" if attention_rules.get((q_comp, k_comp)) else "❌"
                            row_str += f"{state:<{col_width}}"
                        f.write(row_str + "\n")
        else:
            config_info = {
                "instruction": instruction,
                "response": response_text,
                "delete_main_prompt": delete_main_prompt,
                "replace_global_prompt": replace_global_prompt,
                "custom_global_prompt_text": custom_global_prompt_text,
                "expand_value": expand_value,
                "expand_mode": expand_mode,
                "enable_flex_attn": enable_flex_attn,
                "flex_attn_use_bitmask": flex_attn_use_bitmask,
            }

        # Save original image
        original_path = None
        if not only_save_image and not skip_save:
            original_path = os.path.join(edit_folder, "original.png")
            origin_image.save(original_path)

        # Extract global prompt
        global_prompt = self.extract_gen_image_prompt(response_text)
        if replace_global_prompt:
            global_prompt = custom_global_prompt_text
        if not global_prompt:
            global_prompt = " "

        # Prepare region guidance
        region_guidance = None
        if bbox_data:
            region_guidance = []
            for bbox_info in bbox_data:
                original_bbox = bbox_info.get('bbox_2d')
                hint = bbox_info.get('hint')
                if not original_bbox or not hint:
                    continue

                # Expansion
                expanded_bbox = self.expand_bbox(
                    original_bbox,
                    image_size,
                    expand_value,
                    expand_mode
                )
                region_guidance.append(
                    {'bbox': expanded_bbox, 'hint': hint}
                )

        # Call diffusion pipeline
        attention_map_path = (
            os.path.join(edit_folder, "attention_map_visualization.png")
            if visualize_attention and not only_save_image and not skip_save
            else None
        )

        pipe_kwargs = {
            "prompt": global_prompt,
            "image": origin_image,
            "region_guidance": region_guidance,
            "delete_main_prompt": delete_main_prompt,
            "symmetric_masking": symmetric_masking,
            "attention_rules": attention_rules,
            "enable_flex_attn": enable_flex_attn,
            "flex_attn_use_bitmask": flex_attn_use_bitmask,
        }

        if self.pipeline_type == "flux":
            pipe_kwargs["guidance_scale"] = pipeline_kwargs.get("guidance_scale", 2.5)
            pipe_kwargs["height"] = pipeline_kwargs.get("height", height)
            pipe_kwargs["width"] = pipeline_kwargs.get("width", width)
        elif self.pipeline_type == "qwen":
            pipe_kwargs["negative_prompt"] = pipeline_kwargs.get("negative_prompt", " ")
            pipe_kwargs["generator"] = torch.manual_seed(0)
            pipe_kwargs["true_cfg_scale"] = pipeline_kwargs.get("true_cfg_scale", 4.0)
            pipe_kwargs["num_inference_steps"] = pipeline_kwargs.get("num_inference_steps", 50)

        edited_image = self.diffusion_pipe(**pipe_kwargs).images[0]

        # Save results
        final_path = None
        if not skip_save:
            if only_save_image:
                if isinstance(image, str):
                    original_name = os.path.splitext(os.path.basename(image))[0]
                    final_filename = f"{original_name}_edited.png"
                else:
                    final_filename = "edited_image.png"
                final_path = os.path.join(edit_folder, final_filename)
            else:
                final_path = os.path.join(edit_folder, "final_result.png")

            edited_image.save(final_path)

        # Draw bbox on final image (using region_guidance to ensure consistency)
        if region_guidance and not only_save_image and not skip_save:
            final_with_all_bbox = edited_image.copy()
            for i, region in enumerate(region_guidance):
                bbox = region['bbox']
                hint = region.get('hint', '')
                bbox_label = f"{hint}"
                final_with_all_bbox = self.draw_bbox_on_image(
                    final_with_all_bbox, bbox, bbox_label, 'green', 2
                )
            all_bbox_path = os.path.join(
                edit_folder, "final_result_with_all_bbox.png"
            )
            final_with_all_bbox.save(all_bbox_path)

        results = {
            "edited_image": edited_image,
            "edit_folder": edit_folder,
            "instruction": instruction,
            "vlm_response": response_text,
            "original_path": original_path,
            "final_path": final_path,
            "attention_map_path": attention_map_path,
            "bbox_data": bbox_data,
            "global_prompt": global_prompt,
            "region_guidance": region_guidance,
            "config": config_info,
        }

        return edited_image, edit_folder, results

    def edit_single(
        self,
        image,
        instruction,
        id=None,
        delete_main_prompt=False,
        replace_global_prompt=False,
        custom_global_prompt_text="keep remaining parts of this image unchanged",
        expand_value=0.15,
        expand_mode='ratio',
        attention_rules=None,
        bboxes_attend_to_each_other=True,
        symmetric_masking=False,
        output_dir=None,
        only_save_image=False,
        skip_save=False,
        enable_flex_attn=True,
        flex_attn_use_bitmask=True
    ):
        """
        Evaluate a single sample (internally calls VLM first, then diffusion)
        """

        # Determine the final directory for this single evaluation
        final_output_dir = None
        if not skip_save:
            if output_dir is not None:
                final_output_root = output_dir
            elif self.output_dir is not None:
                final_output_root = self.output_dir
            else:
                raise ValueError("Both output_dir and self.output_dir are empty")
            if only_save_image:
                final_output_dir = final_output_root
            elif id:
                final_output_dir = os.path.join(final_output_root, str(id))
            else:
                final_output_dir = final_output_root
            os.makedirs(final_output_dir, exist_ok=True)

        response_text, processed_image = self.get_vlm_response(instruction, image)
        
        edited_image, edit_folder, results = self.region_edit_with_attention(
            processed_image,
            instruction,
            response_text,
            delete_main_prompt=delete_main_prompt,
            replace_global_prompt=replace_global_prompt,
            custom_global_prompt_text=custom_global_prompt_text,
            expand_value=expand_value,
            expand_mode=expand_mode,
            attention_rules=attention_rules,
            symmetric_masking=symmetric_masking,
            output_dir=final_output_dir,
            bboxes_attend_to_each_other=bboxes_attend_to_each_other,
            only_save_image=only_save_image,
            skip_save=skip_save,
            enable_flex_attn=enable_flex_attn,
            flex_attn_use_bitmask=flex_attn_use_bitmask
        )

        return results

    def edit_batch(
        self,
        samples,
        output_dir=None,
        **kwargs
    ):
        """
        Batch evaluate samples

        Args:
            samples: List of samples, each sample is a dict containing image, instruction, and optional id
            output_dir: Output subdirectory name for batch evaluation (optional)
            **kwargs: Other parameters passed to evaluate_single_sample

        Returns:
            results_list: List of results for all samples
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.output_dir is not None:
                batch_output_dir = os.path.join(
                    self.output_dir, f"batch_evaluation_{timestamp}"
                )
            else:
                raise ValueError("Both output_dir and self.output_dir are empty")
        else:
            batch_output_dir = os.path.join(output_dir)

        os.makedirs(batch_output_dir, exist_ok=True)

        results_list = []

        for i, sample in enumerate(samples):
            sample_id_str = sample.get("id", f"sample_{i+1}")
            try:
                result = self.evaluate_single_sample(
                    image=sample["image"],
                    instruction=sample["instruction"],
                    id=sample_id_str,
                    output_dir=batch_output_dir,
                    **kwargs
                )
                results_list.append(result)
            except Exception as e:
                print(f"Error processing sample {i+1} (ID: {sample_id_str}): {type(e).__name__}: {e}")
                results_list.append({
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "sample": sample,
                    "index": i
                })

        # Save batch evaluation results
        batch_results_path = os.path.join(batch_output_dir, "batch_results.json")
        with open(batch_results_path, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, ensure_ascii=False, indent=2, sort_keys=True)

        return results_list


