import base64
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List
import re
import time

from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError
from PIL import Image
import os

logger = logging.getLogger(__name__)

# OpenAI-compatible API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"
MLLM_MAX_WORKERS = int(os.getenv("MLLM_MAX_WORKERS", 8))
MAX_RETRIES = 3
RETRY_DELAY = 5

# OpenAI client init
try:
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
    )
except Exception as e:
    logger.error("Failed to initialize OpenAI client: %s", e)
    client = None

# Prompt templates (loaded from files)
PROMPT_TEMPLATES = {}
try:
    prompt_dir = Path(__file__).resolve().parent / 'prompt_templates'
    
    prompt_files = {
        'target_following': 'target_following.txt',
        'effect': 'effect.txt',
        'consistency': 'consistency.txt',
    }

    for dim, filename in prompt_files.items():
        filepath = prompt_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                PROMPT_TEMPLATES[dim] = f.read()
        else:
            logger.warning("Missing prompt template: %s", filepath)
except Exception as e:
    logger.error("Failed to load prompt templates: %s", e)
    PROMPT_TEMPLATES = {
        'target_following': '',
        'effect': '',
        'consistency': '',
    }

def encode_image_to_base64(image: Image.Image) -> str:
    """Encode a PIL Image to a base64 PNG string."""
    from io import BytesIO
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_qwen_vl_response(inputs: List[Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call Qwen-VL via an OpenAI-compatible API.

    Args:
        inputs: a list of strings and/or PIL.Image objects
        schema: expected JSON output schema

    Returns:
        Dict with "output_text" on success, or "error"/"message" on failure.
    """
    if not client:
        return {"error": "OpenAI client not initialized"}

    content = []
    for item in inputs:
        if isinstance(item, str):
            content.append({"type": "text", "text": item})
        elif isinstance(item, Image.Image):
            base64_image = encode_image_to_base64(item)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            })

    json_prompt = (
        f"\nPlease provide your response in a JSON format that adheres to the following schema:\n"
        f"```json\n{json.dumps(schema, indent=2)}\n```\n"
        f"Your response MUST be a JSON object enclosed in ```json and ```."
    )
    content.append({"type": "text", "text": json_prompt})
    
    messages = [{"role": "user", "content": content}]

    for attempt in range(MAX_RETRIES):
        try:
            chat_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=1024,
                temperature=0.0,
            )
            output_text = chat_response.choices[0].message.content
            return {"output_text": output_text}
        except (APIConnectionError, APITimeoutError, RateLimitError) as e:
            logger.warning("Qwen-VL request failed (attempt %s/%s): %s", attempt + 1, MAX_RETRIES, e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                logger.error("Qwen-VL request failed: max retries reached.")
                return {"error": "API Error", "message": str(e)}
        except Exception as e:
            logger.error("Qwen-VL request failed (non-retryable): %s", e)
            return {"error": "API Error", "message": str(e)}
            
    return {"error": "API Error", "message": "All retry attempts failed."}

def render_template(template_string: str, data: Dict[str, any]) -> str:
    """
    Replace {{{key}}} placeholders in a template string with values from data.
    """
    def replace_func(match):
        key = match.group(1).strip()
        return str(data.get(key, ''))

    return re.sub(r'{{{([^{}]+)}}}', replace_func, template_string)

def evaluate_dimension(
    dimension: str,
    schema: Dict[str, Any],
    original_image: Image.Image = None,
    edited_image: Image.Image = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Evaluate a single dimension (target_following/effect/consistency) and parse the model output.
    """
    template = PROMPT_TEMPLATES.get(dimension)
    if not template:
        return {"error": f"Missing prompt template for dimension '{dimension}'."}

    prompt = render_template(template, kwargs)
    
    inputs = []
    parts = re.split(r"(\*\*Original Image\*\*|\*\*Edited Image\*\*:)", prompt)
    
    i = 0
    while i < len(parts):
        text = parts[i]
        inputs.append(text)
        if i + 1 < len(parts):
            marker = parts[i+1]
            if "Original Image" in marker and original_image:
                inputs.append(original_image)
            elif "Edited Image" in marker and edited_image:
                inputs.append(edited_image)
        i += 2

    # Derive score/reasoning keys from schema.
    prop_keys = list(schema.get("properties", {}).keys())
    score_key = prop_keys[0] if len(prop_keys) > 0 else "score"
    reasoning_key = prop_keys[1] if len(prop_keys) > 1 else "reasoning"

    try:
        response = get_qwen_vl_response(inputs=inputs, schema=schema)
        if "error" in response:
            return {score_key: None, reasoning_key: f"API Error: {response.get('message', 'Unknown')}"}

        output_text = response.get("output_text", "")
        
        # 1) Prefer parsing a ```json ... ``` block.
        match = re.search(r"```json\s*(\{.*?\})\s*```", output_text, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON code block; falling back. json=%r", json_str)

        # 2) Try parsing the entire output as JSON.
        try:
            return json.loads(output_text)
        except json.JSONDecodeError:
            pass

        # 3) Fallback: extract a 1-5 score from text.
        score_match = re.search(r'\b([1-5])\b', output_text)
        if score_match:
            score = int(score_match.group(1))
            result = {
                score_key: score,
                reasoning_key: f"JSON parse failed; extracted score '{score}' from text."
            }
            logger.info("JSON parse failed; extracted score=%s.", score)
            return result

        raise ValueError(f"Could not parse JSON or extract a valid score. output={output_text!r}")

    except Exception as e:
        logger.error("Failed to evaluate '%s': %s", dimension, e)
        return {score_key: None, reasoning_key: f"Exception: {e}"}

def compute_score(reward_input: Dict[str, Any], sharp_reward: int = 4) -> Dict[str, Any]:
    """
    Compute reward scores for one sample.

    Expects:
      reward_input["extra_info"]["original_image_base64"]
      reward_input["extra_info"]["edited_image_base64"]
      reward_input["extra_info"]["instruction_details"]
    """
    zero_scores = {
        'target_following': 0,
        'effect': 0,
        'consistency': 0,
        'overall': 0.0
    }
    extra_info = reward_input["extra_info"]

    # Image data (base64).
    original_image_base64 = extra_info.get('original_image_base64')
    edited_image_base64 = extra_info.get('edited_image_base64')

    instruction = extra_info.get('instruction_details', {}).get('editing_instruction')
    ref_exp = extra_info.get('instruction_details', {}).get('referring_expression')

    zero_results = {
        "scores": zero_scores,
        "details": {},
        "original_image_path": "base64_data",
        "edited_image_path": "base64_data",
    }
    
    if not instruction or not ref_exp:
        logger.warning("Missing instruction/ref_exp; skip scoring. extra_info_keys=%s", list(extra_info.keys()))
        return zero_results

    try:
        from io import BytesIO
        
        original_image = None
        if original_image_base64:
             try:
                if isinstance(original_image_base64, list):
                    original_image_base64 = original_image_base64[0]
                original_image = Image.open(BytesIO(base64.b64decode(original_image_base64))).convert('RGB')
             except Exception as e:
                logger.error("Original image decode failed: %s", e)

        edited_image = None
        if edited_image_base64:
            try:
                if isinstance(edited_image_base64, list):
                    edited_image_base64 = edited_image_base64[0]
                edited_image = Image.open(BytesIO(base64.b64decode(edited_image_base64))).convert('RGB')
            except Exception as e:
                logger.error("Edited image decode failed: %s", e)
        
        if original_image is None or edited_image is None:
            logger.error("Failed to load images. original=%s edited=%s", type(original_image), type(edited_image))
            return zero_results

    except Exception as e:
        logger.error("Failed to load images: %s", e)
        import traceback
        traceback.print_exc()
        return zero_results

    # Schemas per dimension.
    schemas = {
        'target_following': {"type": "object", "properties": {"target_score": {"type": "integer"}, "reasoning": {"type": "string"}}, "required": ["target_score", "reasoning"]},
        'effect': {"type": "object", "properties": {"effect_score": {"type": "integer"}, "reasoning": {"type": "string"}}, "required": ["effect_score", "reasoning"]},
        'consistency': {"type": "object", "properties": {"consistency_score": {"type": "integer"}, "reasoning": {"type": "string"}}, "required": ["consistency_score", "reasoning"]},
    }
    
    eval_params = {"instruction": instruction, "instruct": instruction, "ref_exp": ref_exp}
    
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                'target_following': executor.submit(evaluate_dimension, 'target_following', schemas['target_following'], original_image=original_image, edited_image=edited_image, **eval_params),
                'effect': executor.submit(evaluate_dimension, 'effect', schemas['effect'], original_image=original_image, edited_image=edited_image, **eval_params),
                'consistency': executor.submit(evaluate_dimension, 'consistency', schemas['consistency'], original_image=original_image, edited_image=edited_image, **eval_params),
            }
            results = {dim: future.result() for dim, future in futures.items()}

        
        # Extract numeric scores robustly.
        scores = {}
        for dim, res in results.items():
            schema = schemas[dim]
            score_key = list(schema.get("properties", {}).keys())[0]

            if isinstance(res, dict):
                score = res.get(score_key)
                if isinstance(score, (int, float)):
                    scores[dim] = float(score)
                else:
                    logger.warning("Invalid/missing score for '%s'; defaulting to 0. res=%s", dim, res)
                    scores[dim] = 0.0
            else:
                logger.warning("Invalid eval result for '%s'; defaulting to 0. res=%s", dim, res)
                scores[dim] = 0.0
        
        if not scores:
            return zero_results
        
        if sharp_reward:
            target_score = scores.get("target_following", 0.0)
            instruction_score = scores.get("effect", 0.0)
            consistency_score = scores.get("consistency", 0.0)

            scores["target_following"] = 1.0 if target_score >= sharp_reward else 0.0
            scores["effect"] = 1.0 if instruction_score >= sharp_reward else 0.0
            scores["consistency"] = 1.0 if consistency_score >= sharp_reward else 0.0
            weighted_scores = [scores["target_following"], scores["effect"], scores["consistency"] * scores["effect"]]
            scores["overall"] = sum(weighted_scores) / 3.0
        else:
            instruction_score = results.get('effect', {}).get('effect_score')
            weight = (max(instruction_score - 3.0, 0.0) / 5.0) if instruction_score is not None else 0.0
            weighted_scores = []
            for dim, score in scores.items():
                if dim == 'consistency':
                    weighted_scores.append(score * weight)
                else:
                    weighted_scores.append(score)

            if not weighted_scores:
                logger.warning("Evaluation failed: empty score list.")
                return zero_results
                
            scores["overall"] = sum(weighted_scores) / len(weighted_scores) / 5.0
    except:
        import traceback
        traceback.print_exc()
        return zero_results
    
    detailed_results = {
        "scores": scores,
        "details": results,
        "original_image_path": "base64_data",
        "edited_image_path": "base64_data",
    }

    return detailed_results


def compute_score_batch(reward_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute reward scores for a batch in parallel."""
    with ThreadPoolExecutor(max_workers=MLLM_MAX_WORKERS) as executor:
        futures = [
            executor.submit(compute_score, reward_input)
            for reward_input in reward_inputs
        ]
        results = [future.result() for future in futures]
    return results

if __name__ == '__main__':
    # Minimal local smoke test.
    try:
        Path("dummy_data").mkdir(exist_ok=True)
        Image.new('RGB', (100, 100), color = 'red').save('dummy_data/original.png')
        Image.new('RGB', (100, 100), color = 'blue').save('dummy_data/edited.png')
        
        with open("dummy_data/original.png", "rb") as f:
            original_b64 = base64.b64encode(f.read()).decode("utf-8")
        with open("dummy_data/edited.png", "rb") as f:
            edited_b64 = base64.b64encode(f.read()).decode("utf-8")

        reward_input = {
            "extra_info": {
                "original_image_base64": original_b64,
                "edited_image_base64": edited_b64,
                "instruction_details": {
                    "editing_instruction": "change the background to blue",
                    "referring_expression": "the background",
                },
            }
        }

        print("--- Single sample ---")
        print(compute_score(reward_input))

        print("\n--- Batch ---")
        print(compute_score_batch([reward_input, reward_input]))

    except ImportError:
        logger.warning("Install Pillow (pip install Pillow) to run this test.")
    except Exception as e:
        logger.error("Smoke test failed: %s", e)