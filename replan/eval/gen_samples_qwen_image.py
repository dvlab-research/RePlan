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
from diffusers import QwenImageEditPipeline, QwenImageEditPlusPipeline
from diffusers.utils import load_image
from omegaconf import OmegaConf
from dataclasses import dataclass
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
class QwenGenConfig:
    qwen_model_name: str = "Qwen/Qwen-Image-Edit"
    output_dir: str = "output/qwen_image_edit"
    seed: int = 42
    torch_dtype: str = "bfloat16"
    true_cfg_scale: float = 4.0
    num_inference_steps: int = 50
    negative_prompt: str = " "
    rank_id: int = 0
    world: int = 1

def load_iv_edit_dataset():
    dataset = load_dataset("TainU/IV-Edit", split="test")
    data_list = []
    for sample in dataset:
        instruction = sample["prompt"]
        image = sample["image"]
        image_id = json.loads(sample["extra_info"])["image_id"]
        data_list.append([instruction, image, image_id])
    return data_list
    
def main(conf: QwenGenConfig):
    """Main function to generate images."""
    set_seed(conf.seed, rank=conf.rank_id, device_specific=True)
    
    # Set torch dtype
    torch_dtype = torch.bfloat16 if conf.torch_dtype == "bfloat16" and torch.cuda.is_available() else torch.float32

    # Load pipeline
    print(f"Loading model {conf.qwen_model_name}...")
    if conf.qwen_model_name == "Qwen/Qwen-Image-Edit-2509":
        pipe = QwenImageEditPlusPipeline.from_pretrained(conf.qwen_model_name, torch_dtype=torch_dtype)
    else:
        pipe = QwenImageEditPipeline.from_pretrained(conf.qwen_model_name, torch_dtype=torch_dtype)
    pipe.to('cuda')
    print("Model loaded successfully.")

    # Load data
    data = load_iv_edit_dataset()

    # Distribute data across ranks for parallel processing
    data = data[conf.rank_id::conf.world]
    
    # Create the output directory if it doesn't exist
    os.makedirs(conf.output_dir, exist_ok=True)
    
    # Create a generator for reproducibility
    seed = conf.seed
    if conf.world > 1:
        seed += conf.rank_id
    generator = torch.Generator(device="cuda").manual_seed(seed)

    for editing_instruction, input_image_path, sample_id in tqdm(data, desc=f"Rank {conf.rank_id} generating images"):
        sample_dir = os.path.join(conf.output_dir, f"{sample_id}")
        os.makedirs(sample_dir, exist_ok=True)
        output_path = os.path.join(sample_dir, "final_result.png")
        
        if os.path.exists(output_path):
            continue
        
        input_image = load_image(input_image_path).convert("RGB")
        
        input_image.save(os.path.join(sample_dir, "original_image.png"))
        with open(os.path.join(sample_dir, "instruction.txt"), "w", encoding="utf-8") as f:
            f.write(editing_instruction)
            
        try:
            inputs = {
                "image": input_image,
                "prompt": editing_instruction,
                "generator": generator,
                "true_cfg_scale": conf.true_cfg_scale,
                "negative_prompt": conf.negative_prompt,
                "num_inference_steps": conf.num_inference_steps,
            }

            with torch.inference_mode():
                image = pipe(**inputs).images[0]
            
            image.save(output_path)
        except Exception as e:
            print(f"Failed to process sample {sample_id} on rank {conf.rank_id} due to error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using QwenImageEditPipeline from a config file.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override the output directory from the config file.")
    parser.add_argument("--rank_id", type=int, default=None, help="Rank ID for distributed processing.")
    parser.add_argument("--world", type=int, default=None, help="Total number of processes for distributed processing.")
    
    cli_args = parser.parse_args()

    # Load base config from YAML
    schema = OmegaConf.structured(QwenGenConfig)
    file_conf = OmegaConf.load(cli_args.config)
    conf = OmegaConf.merge(schema, file_conf)
    
    # Merge CLI overrides
    if cli_args.output_dir is not None:
        conf.output_dir = cli_args.output_dir
    if cli_args.rank_id is not None:
        conf.rank_id = cli_args.rank_id
    if cli_args.world is not None:
        conf.world = cli_args.world
        
    main(conf) 