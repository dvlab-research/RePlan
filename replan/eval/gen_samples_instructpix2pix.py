import sys
import os
# This allows for absolute imports from the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from glob import glob
import json
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf, MISSING
from dataclasses import dataclass
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from typing import Optional
from datasets import load_dataset

def set_seed(seed, rank, device_specific=True):
    if device_specific:
        seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_iv_edit_dataset():
    dataset = load_dataset("TainU/IV-Edit", split="test")
    data_list = []
    for sample in dataset:
        instruction = sample["prompt"]
        image = sample["image"]
        image_id = json.loads(sample["extra_info"])["image_id"]
        data_list.append([instruction, image, image_id])
    return data_list

class InstructPix2PixEvaluator:
    def __init__(self, model_id, device, torch_dtype):
        self.device = device
        self.torch_dtype = torch_dtype
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, safety_checker=None)
        pipe.to(device)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        self.pipe = pipe

    def generate(self, image, instruction, seed, image_guidance_scale=1.5, guidance_scale=7.5, num_inference_steps=20):
        generator = torch.Generator(self.device).manual_seed(seed)
        
        edited_image = self.pipe(
            instruction,
            image=image,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        return edited_image

@dataclass
class InstructPix2PixEvalConfig:
    model_id: Optional[str] = None
    output_dir: Optional[str] = None
    seed: int = 42
    allow_tf32: bool = False
    torch_dtype: str = "bfloat16"
    rank_id: int = 0
    world: int = 1
    image_guidance_scale: float = 1.5
    guidance_scale: float = 7.5
    num_inference_steps: int = 20

def main(args: InstructPix2PixEvalConfig):
    torch.backends.cuda.matmul.allow_tf32 = False 
    torch.backends.cudnn.allow_tf32 = False
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    set_seed(args.seed, rank=args.rank_id, device_specific=True)
    device = torch.cuda.current_device()

    os.makedirs(args.output_dir, exist_ok=True)

    evaluator = InstructPix2PixEvaluator(
        model_id=args.model_id,
        device=device,
        torch_dtype=torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float32,
    )

    data = load_iv_edit_dataset()
    data = data[args.rank_id::args.world]
    
    for editing_instruction, image_path, sample_id in tqdm(data, desc=f"Rank {args.rank_id} generating images"):
        sample_dir = os.path.join(args.output_dir, str(sample_id))
        os.makedirs(sample_dir, exist_ok=True)
        output_path = os.path.join(sample_dir, "final_result.png")
        
        if os.path.exists(output_path):
            continue
            
        try:
            input_image = Image.open(image_path).convert("RGB")
            
            # Save original image and instruction
            input_image.save(os.path.join(sample_dir, "original_image.png"))
            with open(os.path.join(sample_dir, "instruction.txt"), "w", encoding="utf-8") as f:
                f.write(editing_instruction)

            edited_image = evaluator.generate(
                input_image, 
                editing_instruction,
                seed=args.seed + args.rank_id,
                image_guidance_scale=args.image_guidance_scale,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
            )
            
            edited_image.save(output_path)
        except Exception as e:
            print(f"Failed to process sample {sample_id} on rank {args.rank_id} due to error: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--output_dir", type=str, default=None, required=False)
    parser.add_argument("--rank_id", type=int, default=None, required=False)
    parser.add_argument("--world", type=int, default=None, required=False)
    
    cli_args = parser.parse_args()

    schema = OmegaConf.structured(InstructPix2PixEvalConfig)
    file_conf = OmegaConf.load(cli_args.config)
    conf = OmegaConf.merge(schema, file_conf)
    
    if cli_args.output_dir is not None:
        conf.output_dir = cli_args.output_dir
    if cli_args.rank_id is not None:
        conf.rank_id = cli_args.rank_id
    if cli_args.world is not None:
        conf.world = cli_args.world
        
    main(conf)
