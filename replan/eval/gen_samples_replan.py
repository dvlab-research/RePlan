import sys
import os
# Correctly set the project root and add it to sys.path
# This allows for absolute imports from the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from glob import glob
import json
import torch
import random
import subprocess
import numpy as np
import torch.distributed as dist
import pandas as pd
import argparse
import torch
import os
from PIL import Image
from tqdm import tqdm
import torch.distributed as dist
from datasets import load_dataset
import re

import torch


from replan.pipelines.replan import RePlanPipeline
from replan.eval.configuration_eval import RegionEditEvalConfig

# adapted from https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/random.py#L31
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


def main(args):

    set_seed(args.seed, rank=args.rank_id, device_specific=True)
    device = torch.cuda.current_device()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    evaluator = RePlanPipeline(
        vlm_ckpt_path=args.vlm_ckpt_path,
        pipeline_type=args.pipeline_type,
        diffusion_model_name=args.diffusion_model_name,
        output_dir=args.output_dir,
        device=device,
        torch_dtype=torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float32,
        vlm_prompt_template_path=args.vlm_prompt_template_path
    )

    data = load_iv_edit_dataset()

    # Distribute data across ranks for parallel processing
    data = data[args.rank_id::args.world]
    
    for editing_instruction, input_image, sample_id in tqdm(data, desc="生成编辑图像"):
        output_path = os.path.join(args.output_dir, f"{sample_id}", "final_result.png")
        
        # Skip if already processed
        if os.path.exists(output_path):
            continue
            
        # Generate edited image
        result = evaluator.edit_single(
            input_image, editing_instruction, 
            delete_main_prompt=args.delete_main_prompt,
            replace_global_prompt=args.replace_global_prompt,
            custom_global_prompt_text=args.custom_global_prompt_text,
            expand_value=args.expand_value,
            expand_mode=args.expand_mode,
            bboxes_attend_to_each_other=args.bboxes_attend_to_each_other,
            id=sample_id,
            output_dir=args.output_dir,
            enable_flex_attn=True
        )


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--vlm_ckpt_path", type=str, default=None, required=False)
    parser.add_argument("--output_dir", type=str, default=None, required=False)
    parser.add_argument("--rank_id", type=int, default=0, required=False)
    parser.add_argument("--world", type=int, default=1, required=False)
    
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    schema = OmegaConf.structured(RegionEditEvalConfig)
    conf = OmegaConf.merge(schema, config)
    if args.vlm_ckpt_path is not None:
        conf.vlm_ckpt_path = args.vlm_ckpt_path
    if args.output_dir is not None:
        conf.output_dir = args.output_dir
    if args.rank_id is not None:
        conf.rank_id = args.rank_id
    if args.world is not None:
        conf.world = args.world
    main(conf) 