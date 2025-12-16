import sys
import os


import json
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf, MISSING
from dataclasses import dataclass, field
from typing import Optional, List
import os
from copy import deepcopy
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)
import requests
from io import BytesIO

from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

bagel_root = "/path/to/bagel"
if project_root not in sys.path:
    sys.path.insert(0, bagel_root)
from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae
from safetensors.torch import load_file

try:
    from inferencer import InterleaveInferencer
except ImportError as e:
    from BAGEL.inferencer import InterleaveInferencer

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

MODEL_PATH = '/path/to/your/model'

def load_iv_edit_dataset():
    dataset = load_dataset("TainU/IV-Edit", split="test")
    data_list = []
    for sample in dataset:
        instruction = sample["prompt"]
        image = sample["image"]
        image_id = json.loads(sample["extra_info"])["image_id"]
        data_list.append([instruction, image, image_id])
    return data_list

class BagelEvaluator:
    def __init__(self, model_id, device, torch_dtype):
        self.device = device
        self.torch_dtype = torch_dtype
    
        # LLM config preparing
        llm_config = Qwen2Config.from_json_file(os.path.join(MODEL_PATH, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        # ViT config preparing
        vit_config = SiglipVisionConfig.from_json_file(os.path.join(MODEL_PATH, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        # VAE loading
        vae_model, vae_config = load_ae(local_path=os.path.join(MODEL_PATH, "ae.safetensors"))

        # Bagel config preparing
        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config, 
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act='gelu_pytorch_tanh',
            latent_patch_size=2,
            max_latent_size=64,
        )

        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model      = SiglipVisionModel(vit_config)
            model          = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        # Tokenizer Preparing
        tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_PATH)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        # Image Transform Preparing
        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)

        max_mem_per_gpu = "90GiB"  # Modify it according to your GPU setting. On an A100, 80 GiB is sufficient to load on a single GPU.

        device_map = infer_auto_device_map(
            model,
            max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )
        print(device_map)

        same_device_modules = [
            'language_model.model.embed_tokens',
            'time_embedder',
            'latent_pos_embed',
            'vae2llm',
            'llm2vae',
            'connector',
            'vit_pos_embed'
        ]

        if torch.cuda.device_count() == 1:
            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device
                else:
                    device_map[k] = "cuda:0"
        else:
            first_device = device_map.get(same_device_modules[0])
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device

        # Thanks @onion-liu: https://github.com/ByteDance-Seed/Bagel/pull/8
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(MODEL_PATH, "ema.safetensors"),
            device_map=device_map,
            dtype=torch.bfloat16,
            force_hooks=True,
        )

        model = model.eval()
        print('Model loaded')

        if model is None:
            print("Model not loaded. Please implement model loading logic in BagelEvaluator.")
            sys.exit(1)

        self.inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids
        )

    def generate(self, image, instruction, seed, **kwargs):
        torch.manual_seed(seed)

        inference_hyper=dict(
            max_think_token_n=1000,
            do_sample=False,
            # text_temperature=0.3,
            cfg_text_scale=4.0,
            cfg_img_scale=2.0,
            cfg_interval=[0.0, 1.0],
            timestep_shift=3.0,
            num_timesteps=50,
            cfg_renorm_min=0.0,
            cfg_renorm_type="text_channel",
        )


        #result = self.inferencer(image=image, text="edit this image: " + instruction, think=True, **inference_hyper)
        result = self.inferencer(image=image, text="edit this image: " + instruction, think=False, **inference_hyper)
        return result['image'], result['text']

@dataclass
class BagelEvalConfig:
    model_id: str = MISSING
    output_dir: str = MISSING
    seed: int = 42
    allow_tf32: bool = False
    torch_dtype: str = "bfloat16"
    rank_id: int = 0
    world: int = 1
    # Bagel specific generation parameters
    cfg_text_scale: float = 3.0
    cfg_img_scale: float = 1.5
    num_timesteps: int = 50
    think: bool = False
    cfg_interval: List[float] = field(default_factory=lambda: [0.4, 1.0])

def main(args: BagelEvalConfig):
    torch.backends.cuda.matmul.allow_tf32 = False 
    torch.backends.cudnn.allow_tf32 = False
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    set_seed(args.seed, rank=args.rank_id, device_specific=True)
    device = torch.cuda.current_device()

    os.makedirs(args.output_dir, exist_ok=True)

    evaluator = BagelEvaluator(
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
            
            input_image.save(os.path.join(sample_dir, "original_image.png"))
            with open(os.path.join(sample_dir, "instruction.txt"), "w", encoding="utf-8") as f:
                f.write(editing_instruction)

            edited_image, edited_text = evaluator.generate(
                input_image, 
                editing_instruction,
                seed=args.seed + args.rank_id,
                cfg_text_scale=args.cfg_text_scale,
                cfg_img_scale=args.cfg_img_scale,
                num_timesteps=args.num_timesteps,
                think=args.think,
                cfg_interval=list(args.cfg_interval)
            )
            
            if edited_image:
                edited_image.save(output_path)
            if edited_text:
                with open(os.path.join(sample_dir, "edited_thought.txt"), "w", encoding="utf-8") as f:
                    f.write(edited_text)
            else:
                print(f"Failed to generate image for sample {sample_id} on rank {args.rank_id}")

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

    schema = OmegaConf.structured(BagelEvalConfig)
    file_conf = OmegaConf.load(cli_args.config)
    conf = OmegaConf.merge(schema, file_conf)
    
    if cli_args.output_dir is not None:
        conf.output_dir = cli_args.output_dir
    if cli_args.rank_id is not None:
        conf.rank_id = cli_args.rank_id
    if cli_args.world is not None:
        conf.world = cli_args.world
        
    main(conf)
