from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class RegionEditEvalConfig:
    vlm_ckpt_path: str = 'TainU/RePlan-Qwen2.5-VL-7B'
    diffusion_model_name: str = "black-forest-labs/FLUX.1-Kontext-dev"
    pipeline_type: str = "flux"
    
    # device
    device: str = "cuda"
    torch_dtype: str = "bfloat16" 
    
    # base
    seed: int = 42
    output_dir: str = "./region_edit_output"
    vlm_prompt_template_path: str = "replan.txt"
    
    # region edit
    delete_main_prompt: bool = False
    replace_global_prompt: bool = False
    custom_global_prompt_text: str = "keep remaining parts of this image unchanged"
    expand_value: float = 0.15
    expand_mode: str = 'ratio'  # 'ratio' or 'pixels'
    
    # attention rules
    bboxes_attend_to_each_other: bool = True

    # multi-GPU
    local_rank: int = 0
    world_size: int = 1
    
    rank_id: int = 0
    world: int = 1

    lora_path: Optional[str] = None

