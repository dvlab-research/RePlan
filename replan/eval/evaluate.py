import argparse
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import logging
from typing import Dict, Any, List, Optional
import os
import sys
import concurrent.futures
from functools import partial
import re


# assume image_qa.py and GeminiImageQA class are in the same directory or in PYTHONPATH
from replan.eval.image_qa import GeminiImageQA
from datasets import load_dataset

def _setup_logging(log_level: str = "INFO") -> None:
    """
    Configure global logging.
    - This script uses logging.info extensively to print statistics;
    - Python's default logging level is WARNING, which would hide these outputs;
    - Here we set it to stdout to avoid being swallowed by tqdm/cluster log collection.
    """
    level = getattr(logging, (log_level or "INFO").upper(), logging.INFO)
    root = logging.getLogger()
    
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
    else:
        has_stdout = any(
            isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout
            for h in root.handlers
        )
        if not has_stdout:
            root.addHandler(logging.StreamHandler(sys.stdout))
        for h in root.handlers:
            try:
                h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
            except Exception:
                pass
    root.setLevel(level)

def render_template(template_string: str, data: Dict[str, any]) -> str:
    def replace_func(match):
        key = match.group(1).strip()
        return str(data.get(key, ''))

    return re.sub(r'{{{([^{}]+)}}}', replace_func, template_string)


def load_iv_edit_dataset(dataset_name: str = "TainU/IV-Edit", split: str = "test") -> List[Dict[str, Any]]:
    logging.info(f"Loading dataset from HuggingFace: {dataset_name} (split: {split})...")
    dataset = load_dataset(dataset_name, split=split)
    
    data_list = []
    for sample in tqdm(dataset, desc="Loading dataset and converting format"):
        try:
            # extract fields from dataset
            prompt = sample.get("prompt", "")
            image = sample.get("image")
            extra_info_str = sample.get("extra_info", "{}")
            
            # parse extra_info
            try:
                extra_info = json.loads(extra_info_str) if isinstance(extra_info_str, str) else extra_info_str
            except (json.JSONDecodeError, TypeError) as e:
                logging.warning(f"Failed to parse extra_info, using empty dictionary: {e}")
                extra_info = {}
            
            image_id = extra_info.get("image_id", None)
            if image_id is None:
                logging.warning(f"Sample missing image_id, skipping")
                continue
            
            if image is None:
                logging.warning(f"Sample {image_id} missing image, skipping")
                continue
            
            # ensure RGB mode
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # build data format expected by evaluation code
            item = {
                "image_id": str(image_id),
                "original_image": image, # directly store PIL Image object
                "instruction_details": {
                    "editing_instruction": prompt,
                    "referring_expression": extra_info.get("referring_expression", ""),
                },
            }
            
            # add other fields that may be needed
            if "status" in sample:
                item["status"] = sample["status"]
            
            # ignore data with status "filtered"
            if item.get("status") == "filtered":
                continue
            
            data_list.append(item)
        except Exception as e:
            logging.error(f"Error processing sample: {e}")
            continue
    
    logging.info(f"Successfully loaded {len(data_list)} data samples")
    return data_list


def get_edited_image_path(edited_images_dir: str, image_id: str) -> Path:
    """
    Build the path to the edited image based on the directory and image ID of the edited images.
    For example: <edited_images_dir>/image/final_result.png
    """
    return Path(edited_images_dir) / str(image_id) / "final_result.png"

def evaluate_quality(qa_model: GeminiImageQA, edited_image: Image.Image, ref_exp: str, template: str) -> Dict[str, Any]:
    """Evaluate the quality of the edited image."""
    width, height = edited_image.size
    # normalized_bbox = [
    #     round(bbox[1] / height * 1000),
    #     round(bbox[0] / width * 1000),
    #     round(bbox[3] / height * 1000),
    #     round(bbox[2] / width * 1000),
    # ]
    prompt = render_template(template, {"ref_exp": ref_exp})
    # insert image into prompt
    parts = prompt.rsplit("**Edited Image**:", 1)
    inputs = [parts[0] + "**Edited Image**:", edited_image, parts[1]]

    schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string", "description": "A brief explanation for the score."},
            "quality_score": {"type": "integer", "description": "A numerical score from 1 to 5."}
        }, "required": ["quality_score", "reasoning"]}
    try:
        return qa_model.ask(inputs=inputs, response_schema=schema)
    except Exception as e:
        logging.error(f"Error evaluating 'quality': {e}")
        return {"reasoning": None, "quality_score": None}

def evaluate_target_following(qa_model: GeminiImageQA, original_image: Image.Image, edited_image: Image.Image, instruction: str, ref_exp: str, template: str) -> Dict[str, Any]:
    """Evaluate if the edited image follows the target."""

    prompt = render_template(template, {"instruct": instruction, "ref_exp": ref_exp})
    # insert image into prompt in order
    p1, rest = prompt.split("**Original Image**:", 1)
    p2, p3 = rest.split("**Edited Image**:", 1)
    inputs = [p1 + "**Original Image**:", original_image, p2 + "**Edited Image**:", edited_image, p3]

    schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string", "description": "A brief explanation for the score."},
            "target_score": {"type": "integer", "description": "A numerical score from 1 to 5."}
        }, "required": ["target_score", "reasoning"]}
    try:
        return qa_model.ask(inputs=inputs, response_schema=schema)
    except Exception as e:
        logging.error(f"Error evaluating 'target_following': {e}")
        return {"reasoning": None, "target_score": None}

def evaluate_effect(qa_model: GeminiImageQA, original_image: Image.Image, edited_image: Image.Image, instruction: str, ref_exp: str, template: str) -> Dict[str, Any]:
    """Evaluate if the edited image follows the instruction."""
    prompt = render_template(template, {"instruct": instruction, "ref_exp": ref_exp})
    # insert image into prompt in order
    p1, rest = prompt.split("**Original Image**:", 1)
    p2, p3 = rest.split("**Edited Image**:", 1)
    inputs = [p1 + "**Original Image**:", original_image, p2 + "**Edited Image**:", edited_image, p3]

    schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string", "description": "A brief explanation for the score."},
            "effect_score": {"type": "integer", "description": "A numerical score from 1 to 5."}
        }, "required": ["effect_score", "reasoning"]}
    try:
        return qa_model.ask(inputs=inputs, response_schema=schema)
    except Exception as e:
        logging.error(f"Error evaluating 'effect': {e}")
        return {"reasoning": None, "effect_score": None}

def evaluate_consistency(qa_model: GeminiImageQA, original_image: Image.Image, edited_image: Image.Image, instruction: str, ref_exp: str, template: str) -> Dict[str, Any]:
    """Evaluate the consistency of the edited image."""
    prompt = render_template(template, {"instruct": instruction, "ref_exp": ref_exp})
    # insert image into prompt in order
    p1, rest = prompt.split("**Original Image**:", 1)
    p2, p3 = rest.split("**Edited Image**:", 1)
    inputs = [p1 + "**Original Image**:", original_image, p2 + "**Edited Image**:", edited_image, p3]

    schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string", "description": "A brief explanation for the score."},
            "consistency_score": {"type": "integer", "description": "A numerical score from 1 to 5."}
        }, "required": ["consistency_score", "reasoning"]}
    try:
        return qa_model.ask(inputs=inputs, response_schema=schema)
    except Exception as e:
        logging.error(f"Error evaluating 'consistency': {e}")
        return {"reasoning": None, "consistency_score": None}

def process_single_item(item: Dict[str, Any], edited_images_dir: str, qa_model: GeminiImageQA, prompt_templates: Dict[str, str]) -> Dict[str, Any]:
    """Process the evaluation of a single data item."""
    image_id = item['image_id']
    edited_image_path = get_edited_image_path(edited_images_dir, image_id)

    # check if already evaluated
    individual_output_path = edited_image_path.parent / "gemini_evaluation_result.json"
    if individual_output_path.exists():
        try:
            with open(individual_output_path, 'r', encoding='utf-8') as f:
                result_item = json.load(f)
            logging.info(f"Skipping already evaluated sample: {image_id}")
            return result_item
        except Exception as e:
            pass

    if not edited_image_path.exists():
        logging.warning(f"Cannot find edited image {edited_image_path}, skipping")
        return None

    try:
        original_image = item.get('original_image')
        if original_image is None:
             logging.error(f"Item {image_id} missing original_image object")
             return None
        edited_image = Image.open(edited_image_path).convert('RGB')
    except Exception as e:
        logging.error(f"Error opening image ({image_id} or {edited_image_path}): {e}")
        return None

    instruction = item['instruction_details']['editing_instruction']
    ref_exp = item['instruction_details']['referring_expression']
    #bbox = item['qwen_targets'][0] # assume item contains bbox

    # multi-dimensional evaluation
    dimensional_evals = {
        'quality': evaluate_quality(qa_model, edited_image, ref_exp, prompt_templates['quality']),
        'target_following': evaluate_target_following(qa_model, original_image, edited_image, instruction, ref_exp, prompt_templates['target_following']),
        'effect': evaluate_effect(qa_model, original_image, edited_image, instruction, ref_exp, prompt_templates['effect']),
        'consistency': evaluate_consistency(qa_model, original_image, edited_image, instruction, ref_exp, prompt_templates['consistency'])
    }

    # combine results
    result_item = item.copy()
    # remove Image object to allow JSON serialization
    if 'original_image' in result_item:
        del result_item['original_image']
        
    result_item['evaluation'] = {
        'dimensional_evals': dimensional_evals
    }
    
    # save single evaluation result to its subdirectory
    individual_output_path = edited_image_path.parent / "gemini_evaluation_result.json"
    try:
        with open(individual_output_path, 'w', encoding='utf-8') as f:
            json.dump(result_item, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Cannot save single evaluation result to {individual_output_path}: {e}")

    return result_item


def main():
    """Main function to parse arguments and drive the evaluation process."""
    parser = argparse.ArgumentParser(description="Evaluate image editing results using Gemini API.")
    parser.add_argument("--dataset_name", type=str, default="TainU/IV-Edit", help="HuggingFace dataset name.")
    parser.add_argument("--dataset_split", type=str, default="test", help="HuggingFace dataset split (e.g. 'test', 'train').")
    parser.add_argument("--edited_images_dir", type=str, default="/apdcephfs_sh2/share_300000800/user/leike/interns/tianyuan/research/output/reason_gen/reasonseg_coco_ours", help="Root directory for edited images.")
    parser.add_argument("--output_json", type=str, default="/apdcephfs_sh2/share_300000800/user/leike/interns/tianyuan/research/output/reason_gen/reasonseg_coco_ours_eval.json", help="Output JSON file path for evaluation results.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N samples, if not specified evaluate all samples.")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of worker threads for parallel evaluation.")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level: DEBUG/INFO/WARNING/ERROR")
    args = parser.parse_args()

    _setup_logging(args.log_level)
    # load prompt templates
    prompt_dir = Path(__file__).parent / "prompt_templates"
    prompt_templates = {}
    if prompt_dir.exists() and prompt_dir.is_dir():
        for prompt_file in prompt_dir.glob("*.txt"):
            dimension = prompt_file.stem
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_templates[dimension] = f.read()
        logging.info(f"Loaded {len(prompt_templates)} evaluation dimensions: {list(prompt_templates.keys())}")
    else:
        logging.error(f"Prompt template directory does not exist: {prompt_dir}")
        return

    model_name = "gemini-2.5-pro"
    logging.info("Initializing GeminiImageQA...")
    qa_model = GeminiImageQA(
        model_name=model_name,
    )

    # load dataset from HuggingFace
    logging.info("Loading dataset from HuggingFace...")
    data = load_iv_edit_dataset(
        dataset_name=args.dataset_name,
        split=args.dataset_split
    )
    
    logging.info(f"After loading and filtering, {len(data)} valid samples remaining.")

    if args.limit and args.limit > 0:
        data = data[:args.limit]
        logging.info(f"Limiting evaluation to the first {args.limit} samples.")

    process_func = partial(process_single_item, edited_images_dir=args.edited_images_dir, qa_model=qa_model, prompt_templates=prompt_templates)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # use tqdm to display progress bar
        results_iterator = executor.map(process_func, data)
        evaluation_results = list(tqdm(results_iterator, total=len(data), desc="Evaluating images"))

    all_dimensional_scores = {
        'quality': [], 'target_following': [], 'effect': [], 'consistency': []
    }
    score_keys = {
        'quality': 'quality_score', 'target_following': 'target_score',
        'effect': 'effect_score', 'consistency': 'consistency_score'
    }
    all_overall_scores = []
    all_weighted_overall_scores = [] # new: for storing weighted total scores

    for item in evaluation_results:
        if item and item.get('evaluation'):
            dimensional_evals = item['evaluation'].get('dimensional_evals', {})
            item_scores = []
            item_weighted_scores = [] # new: for storing weighted scores for each item

            # get effect score for calculating weight
            effect_eval = dimensional_evals.get('effect', {})
            instruction_score = effect_eval.get('effect_score') if effect_eval else None
            weight = (instruction_score / 5.0) if instruction_score is not None else 0

            for dim, scores_list in all_dimensional_scores.items():
                score_key = score_keys[dim]
                eval_result = dimensional_evals.get(dim, {})
                score = eval_result.get(score_key)
                if score is None:
                    score = 0
                scores_list.append(score)
                item_scores.append(score)

                # calculate weighted score
                weighted_score = score
                if dim in ['quality', 'consistency']:
                    weighted_score *= weight
                item_weighted_scores.append(weighted_score)
            
            if item_scores and len(item_scores) > 0:
                item_avg = sum(item_scores) / len(item_scores)
                all_overall_scores.append(item_avg)
                item['evaluation']['overall_average_score'] = item_avg
            else:
                all_overall_scores.append(0)

            # calculate and save weighted average score
            if item_weighted_scores:
                item_weighted_avg = sum(item_weighted_scores) / len(item_weighted_scores)
                all_weighted_overall_scores.append(item_weighted_avg)
                item['evaluation']['weighted_overall_average_score'] = item_weighted_avg
            else:
                all_weighted_overall_scores.append(0)
        else:
            # for samples that failed to process or have no evaluation results, all scores are set to 0
            for dim, scores_list in all_dimensional_scores.items():
                scores_list.append(0)
            all_overall_scores.append(0)
            all_weighted_overall_scores.append(0)

    logging.info(f"Evaluation completed. Saving results to {args.output_json}...")
    with open(args.output_json, 'w', encoding='utf-8') as f:
        # filter out None values to avoid JSON serialization errors
        json.dump([res for res in evaluation_results if res is not None], f, indent=4, ensure_ascii=False)
    
    # calculate and print average scores for each dimension
    for dimension, scores in all_dimensional_scores.items():
        if scores:
            average_score = sum(scores) / len(scores)
            logging.info(f"Total {len(scores)} samples ({dimension} dimension).")
            logging.info(f"'{dimension.replace('_', ' ').capitalize()}' dimension average score (1-5): {average_score:.2f}")
        else:
            logging.warning(f"No scores available for calculating '{dimension}' dimension average score.")

    # calculate and print overall average score
    if all_overall_scores:
        average_overall_score = sum(all_overall_scores) / len(all_overall_scores)
        logging.info(f"Total {len(all_overall_scores)} samples overall average score.")
        logging.info(f"Overall average score after averaging all dimensions (1-5): {average_overall_score:.2f}")

    # calculate and print weighted overall average score
    if all_weighted_overall_scores:
        average_weighted_overall_score = sum(all_weighted_overall_scores) / len(all_weighted_overall_scores)
        logging.info(f"Weighted overall average score (1-5): {average_weighted_overall_score:.2f}")

    logging.info("All operations completed!")


if __name__ == "__main__":
    main() 

