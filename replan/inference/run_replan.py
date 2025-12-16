import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import torch

from replan.pipelines.replan import RePlanPipeline


def _parse_torch_dtype(dtype_str: str) -> torch.dtype:
    s = (dtype_str or "").lower().strip()
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16", "half"}:
        return torch.float16
    if s in {"fp32", "float32", "full"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}. Use one of: bf16, fp16, fp32.")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RePlan inference: input image + edit instruction -> save edited outputs."
    )

    # Image and instruction, if not provided, an interactive mode will be launched.
    parser.add_argument("--image", type=str, help="Input image path.")
    parser.add_argument("--instruction", type=str, help="Editing instruction.")
    parser.add_argument("--output_dir", type=str, default='./output/inference', help="Directory to save outputs.")

    # Pipeline / model
    parser.add_argument("--pipeline_type", type=str, default="flux", choices=["flux", "qwen"])
    parser.add_argument("--vlm_ckpt_path", type=str, default="TainU/RePlan-Qwen2.5-VL-7B")
    parser.add_argument("--diffusion_model_name", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--vlm_prompt_template_path", type=str, default="replan.txt")

    # Runtime
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", help="bf16|fp16|fp32")
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Root directory for resolving relative image paths (optional).",
    )

    # Saving / naming
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="Run identifier (subfolder name). Default: <image_stem>_<timestamp>.",
    )
    parser.add_argument("--only_save_image", action="store_true", help="Only save final edited image.", default=False)

    # Editing options (forwarded to edit_single)
    parser.add_argument("--delete_main_prompt", action="store_true", default=False)
    parser.add_argument("--replace_global_prompt", action="store_true", default=False)
    parser.add_argument(
        "--custom_global_prompt_text",
        type=str,
        default="keep remaining parts of this image unchanged",
    )
    parser.add_argument("--expand_value", type=float, default=0.15)
    parser.add_argument("--expand_mode", type=str, default="ratio", choices=["ratio", "pixels"])
    parser.add_argument("--bboxes_attend_to_each_other", action="store_true", default=True)
    parser.add_argument(
        "--no_bboxes_attend_to_each_other",
        action="store_false",
        dest="bboxes_attend_to_each_other",
    )
    parser.add_argument("--symmetric_masking", action="store_true", default=False)
    parser.add_argument("--enable_flex_attn", action="store_true", default=True)
    parser.add_argument("--disable_flex_attn", action="store_false", dest="enable_flex_attn")
    parser.add_argument("--flex_attn_use_bitmask", action="store_true", default=True)
    parser.add_argument("--flex_attn_use_densemask", action="store_false", dest="flex_attn_use_bitmask")

    return parser


def interactive_mode(args, pipeline: RePlanPipeline):
    print("Entering interactive mode...")
    while True:
        try:
            image = input("Enter image path: ")
            instruction = input("Enter editing instruction: ")

            run_id = args.id
            if not run_id:
                run_id = f"{Path(image).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            results = pipeline.edit_single(
                image=image, instruction=instruction,
                delete_main_prompt=args.delete_main_prompt,
                replace_global_prompt=args.replace_global_prompt,
                custom_global_prompt_text=args.custom_global_prompt_text,
                expand_value=args.expand_value,
                expand_mode=args.expand_mode,
                bboxes_attend_to_each_other=args.bboxes_attend_to_each_other,
                symmetric_masking=args.symmetric_masking,
                output_dir=args.output_dir,
                only_save_image=args.only_save_image,
                enable_flex_attn=args.enable_flex_attn,
                flex_attn_use_bitmask=args.flex_attn_use_bitmask,
            )
            print(f"Results: {results.get('edit_folder')}")
            print(f"Final image: {results.get('final_path')}")
        except Exception as e:
            print(f"Error: {e}")
        if input("Continue? (y/n): ") == "n":
            break
    return 0

def main(argv=None) -> int:
    # Make `python replan/inference/run_replan.py ...` robust (without requiring editable installs).
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    args = build_argparser().parse_args(argv)

    if args.image_dir is None:
        args.image_dir = project_root

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)


    pipeline = RePlanPipeline(
        vlm_ckpt_path=args.vlm_ckpt_path,
        diffusion_model_name=args.diffusion_model_name,
        pipeline_type=args.pipeline_type,
        output_dir=output_dir,
        device=args.device,
        torch_dtype=_parse_torch_dtype(args.dtype),
        vlm_prompt_template_path=args.vlm_prompt_template_path,
        lora_path=args.lora_path,
        init_vlm=True,
        image_dir=args.image_dir,
    )

    if args.image is None or args.instruction is None:
        print("No image or instruction provided, launching interactive mode...")
        interactive_mode(args, pipeline)
        return 0


    run_id = args.id
    if not run_id:
        run_id = f"{Path(args.image).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    results = pipeline.edit_single(
        image=args.image,
        instruction=args.instruction,
        id=run_id,
        delete_main_prompt=args.delete_main_prompt,
        replace_global_prompt=args.replace_global_prompt,
        custom_global_prompt_text=args.custom_global_prompt_text,
        expand_value=args.expand_value,
        expand_mode=args.expand_mode,
        bboxes_attend_to_each_other=args.bboxes_attend_to_each_other,
        symmetric_masking=args.symmetric_masking,
        output_dir=output_dir,
        only_save_image=args.only_save_image,
        enable_flex_attn=args.enable_flex_attn,
        flex_attn_use_bitmask=args.flex_attn_use_bitmask,
    )

    print(f"Saved results to: {results.get('edit_folder')}")
    print(f"Final image: {results.get('final_path')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


