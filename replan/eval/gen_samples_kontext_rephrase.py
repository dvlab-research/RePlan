import sys
import os

# Correctly set the project root and add it to sys.path
# This allows for absolute imports from the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import torch
import random
import numpy as np
import argparse
from tqdm import tqdm
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from omegaconf import OmegaConf
from dataclasses import dataclass, field
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from datasets import load_dataset

# adapted from https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/random.py#L31
def set_seed(seed, rank, device_specific=True):
    """Sets the seed for reproducibility."""
    if device_specific:
        seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

@dataclass
class QwenConfig:
    vlm_ckpt_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    prompt_template_path: str = "rephrase.txt"

@dataclass
class KontextGenConfig:
    flux_model_name: str = "black-forest-labs/FLUX.1-Kontext-dev"
    output_dir: str = "./output/kontext_rephrase"
    seed: int = 42
    torch_dtype: str = "bfloat16"
    guidance_scale: float = 2.5
    rank_id: int = 0
    world: int = 1
    qwen: QwenConfig = field(default_factory=QwenConfig)

def load_iv_edit_dataset():
    dataset = load_dataset("TainU/IV-Edit", split="test")
    data_list = []
    for sample in dataset:
        instruction = sample["prompt"]
        image = sample["image"]
        image_id = json.loads(sample["extra_info"])["image_id"]
        data_list.append([instruction, image, image_id])
    return data_list

def rephrase_instruction_with_qwen(vlm_model, processor, prompt_template, instruction):
    prompt = prompt_template + '\n' + instruction
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(vlm_model.device)
        
        generated_ids = vlm_model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text_list = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        rephrased_instruction = output_text_list[0]
        
        clean_instruction = rephrased_instruction.strip()
        return clean_instruction
    except Exception as e:
        return instruction

def main(conf: KontextGenConfig):
    """Main function to generate images."""
    set_seed(conf.seed, rank=conf.rank_id, device_specific=True)
    
    torch_dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    flux_torch_dtype = torch_dtype_map.get(conf.torch_dtype, torch.float32)
    if conf.torch_dtype == "bfloat16" and not torch.cuda.is_available():
        flux_torch_dtype = torch.float32

    # --- 初始化 Qwen-VL 模型 ---
    print("Initializing Qwen-VL model...")
    vlm_torch_dtype = torch.bfloat16 # Qwen-VL prefers bfloat16 for performance
    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        conf.qwen.vlm_ckpt_path,
        torch_dtype=vlm_torch_dtype,
        attn_implementation="flash_attention_2",
    ).to(f"cuda")
    processor = AutoProcessor.from_pretrained(conf.qwen.vlm_ckpt_path)
    
    with open(conf.qwen.prompt_template_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    print("Qwen-VL model initialized.")

    print(f"Loading FLUX model {conf.flux_model_name}...")
    pipe = FluxKontextPipeline.from_pretrained(conf.flux_model_name, torch_dtype=flux_torch_dtype)
    pipe.to(f'cuda')
    print("FLUX model loaded successfully.")

    data = load_iv_edit_dataset()
    data = data[conf.rank_id::conf.world]
    os.makedirs(conf.output_dir, exist_ok=True)
    
    for editing_instruction, input_image_path, sample_id in tqdm(data, desc=f"Rank {conf.rank_id} generating images"):
        sample_dir = os.path.join(conf.output_dir, str(sample_id))
        os.makedirs(sample_dir, exist_ok=True)
        output_path = os.path.join(sample_dir, "final_result.png")
        
        if os.path.exists(output_path):
            continue
        
        rephrased_instruction = rephrase_instruction_with_qwen(
            vlm_model, processor, prompt_template, editing_instruction, conf.rank_id
        )
        
        input_image = load_image(input_image_path)
        
        input_image.save(os.path.join(sample_dir, "original_image.png"))
        with open(os.path.join(sample_dir, "instruction.txt"), "w", encoding="utf-8") as f:
            f.write(editing_instruction)
            
        with open(os.path.join(sample_dir, "rephrased_instruction.txt"), "w", encoding="utf-8") as f:
            f.write(rephrased_instruction)
            
        try:
            width, height = input_image.size
            image = pipe(
                image=input_image,
                prompt=rephrased_instruction,
                guidance_scale=conf.guidance_scale,
                height=height,
                width=width
            ).images[0]
            image = image.resize((width, height))
            image.save(output_path)
        except Exception as e:
            print(f"Failed to process sample {sample_id} on rank {conf.rank_id} due to error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using FluxKontextPipeline with Qwen-VL rephrasing.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override the output directory from the config file.")
    parser.add_argument("--rank_id", type=int, default=None, help="Rank ID for distributed processing.")
    parser.add_argument("--world", type=int, default=None, help="Total number of processes for distributed processing.")
    
    cli_args = parser.parse_args()

    schema = OmegaConf.structured(KontextGenConfig)
    file_conf = OmegaConf.load(cli_args.config)
    conf = OmegaConf.merge(schema, file_conf)
    
    if cli_args.output_dir is not None:
        conf.output_dir = cli_args.output_dir
    if cli_args.rank_id is not None:
        conf.rank_id = cli_args.rank_id
    if cli_args.world is not None:
        conf.world = cli_args.world
        
    main(conf)
