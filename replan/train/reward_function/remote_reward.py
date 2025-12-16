# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

import copy


KONTEXT_SERVER_URL = os.getenv("KONTEXT_SERVER_URL", "http://localhost:8001/v1/images/generations")
KONTEXT_MAX_WORKERS = int(os.getenv("KONTEXT_MAX_WORKERS", 8))
MLLM_MAX_WORKERS = int(os.getenv("MLLM_MAX_WORKERS", 24))

import replan.train.reward_function.qwen_vl_reward as qwen_vl_reward


logger = logging.getLogger(__name__)


def generate_edited_image(image_base64: str, instruction: str, model_response: str):
    """
    Call the Kontext server to generate an edited image.

    Returns:
        The edited image as a base64 string, or None on failure.
    """

    payload = {
        "image_base64": image_base64,
        "instruction": instruction,
        "response": model_response,
    }

    for attempt in range(qwen_vl_reward.MAX_RETRIES):
        try:

            http_resp = requests.post(KONTEXT_SERVER_URL, json=payload, timeout=600)
            http_resp.raise_for_status()  # raise on 4xx/5xx
            result = http_resp.json()

            edited_image_base64 = result.get("data", [{}])[0].get("final_image_base64")

            if edited_image_base64:
                logger.info("Kontext edit succeeded (base64_len=%s).", len(edited_image_base64))
                return edited_image_base64

            logger.error("Kontext response missing 'final_image_base64': %s", result)
            return None

        except (requests.exceptions.ConnectionError, requests.exceptions.ProxyError, requests.exceptions.Timeout) as e:
            logger.warning(
                "Kontext request failed (attempt %s/%s): %s",
                attempt + 1,
                qwen_vl_reward.MAX_RETRIES,
                e,
            )
        except requests.exceptions.HTTPError as e:
            if 500 <= e.response.status_code < 600:
                logger.warning(
                    "Kontext server error %s (attempt %s/%s).",
                    e.response.status_code,
                    attempt + 1,
                    qwen_vl_reward.MAX_RETRIES,
                )
            else:
                logger.error("Kontext client error %s: %s", e.response.status_code, e)
                return None
        except requests.exceptions.JSONDecodeError:
            logger.error(
                "Failed to decode JSON from Kontext response (status=%s, body=%r).",
                http_resp.status_code,
                http_resp.text,
            )
            return None
        except (KeyError, IndexError) as e:
            logger.error("Unexpected Kontext response format: %s (result=%s)", e, result)
            return None

        if attempt < qwen_vl_reward.MAX_RETRIES - 1:
            time.sleep(qwen_vl_reward.RETRY_DELAY)
        else:
            logger.error("Kontext request failed: max retries reached.")

    return None


def compute_score(reward_input):
    """
    Main reward pipeline:
      1) Call Kontext to generate the edited image.
      2) Call Qwen-VL to score the edit.
    """
    response_str = reward_input["response"]
    ground_truth = reward_input["ground_truth"]
    extra_info = reward_input["extra_info"]
    input_image_base64 = reward_input["input_image_base64"]
    if isinstance(input_image_base64, list):
        input_image_base64 = input_image_base64[0]

    zero_scores = {
        'target_following': 0,
        'effect': 0,
        'consistency': 0,
        'overall': 0.0
    }

    # Validate response
    if not isinstance(response_str, str) or not response_str.strip():
        logger.error("Invalid response string.")
        return zero_scores
    response_str = response_str.strip()

    # Pull required fields
    instruction = extra_info.get("instruction_details", {}).get("editing_instruction")


    if not input_image_base64:
        raise ValueError("Missing input_image_base64.")

    # Generate edited image via Kontext
    edited_image_base64 = generate_edited_image(input_image_base64, instruction, response_str)

    if not edited_image_base64:
        logger.error("Failed to generate edited image via Kontext.")
        return zero_scores

    # Score via Qwen-VL
    qwen_extra_info = {**extra_info}
    qwen_extra_info["original_image_base64"] = input_image_base64
    qwen_extra_info["edited_image_base64"] = edited_image_base64

    reward = qwen_vl_reward.compute_score(
        reward_input={
            "response": response_str,
            "ground_truth": ground_truth,
            "extra_info": qwen_extra_info
        }
    )
    
    return reward


def compute_score_batch(reward_inputs, format_compute_func=None):
    """
    Compute rewards for a batch with a two-stage pipeline:
      1) Filter by format/region quality and generate edited images via Kontext (parallel).
      2) Score successfully generated edits via Qwen-VL (parallel).
    """
    if format_compute_func is None:
        from replan.train.reward_function.replan_reward import compute_score_editing as format_compute_func

    # Stage 0: compute format scores and select valid samples
    scores_format = format_compute_func(reward_inputs)

    final_logs = []
    zero_score = {
        "target_following": 0.0,
        "effect": 0.0,
        "consistency": 0.0,
        "overall": 0.0,
    }
    for i, reward_input in enumerate(reward_inputs):
        final_logs.append({
            "response": reward_input.get("response"),
            "scores": copy.deepcopy(zero_score),
            "details": None,
        })

    planning_gen_args = []
    planning_indices = []  # indices in the original batch
    skipped_invalid = 0
    skipped_missing_image = 0
    for i, score_f in enumerate(scores_format):
        if score_f.get("format") == 1.0 and score_f.get("region", 0.0) >= 0.7:
            reward_input = reward_inputs[i]
            response_str = reward_input.get("response", "").strip()
            extra_info = reward_input.get("extra_info", {})
            input_image_base64 = reward_input.get("input_image_base64")
            if isinstance(input_image_base64, list):
                input_image_base64 = input_image_base64[0]
            

            instruction = extra_info.get("instruction_details", {}).get("editing_instruction")
            if not input_image_base64 or not instruction or not response_str:
                raise ValueError("Missing input_image_base64.")


            planning_gen_args.append((input_image_base64, instruction, response_str))
            planning_indices.append(i)
        else:
            skipped_invalid += 1

    if skipped_invalid or skipped_missing_image:
        logger.info(
            "Batch filter: total=%s, selected=%s, skipped_invalid=%s, skipped_missing_image=%s",
            len(reward_inputs),
            len(planning_gen_args),
            skipped_invalid,
            skipped_missing_image,
        )

    # Stage 1: generate images for selected samples
    if planning_gen_args:
        with ThreadPoolExecutor(max_workers=KONTEXT_MAX_WORKERS) as executor:
            edited_image_results = list(executor.map(lambda args: generate_edited_image(*args), planning_gen_args))
        for i, base64_data in enumerate(edited_image_results):
            original_index = planning_indices[i]
            if base64_data:
                final_logs[original_index]["edited_image_path"] = "base64_data_in_memory"
    else:
        edited_image_results = []

    # Prepare Qwen-VL inputs
    qwen_inputs = []
    valid_qwen_indices = []  # indices in the original batch

    for i, (original_index, edited_base64) in enumerate(zip(planning_indices, edited_image_results)):
        if edited_base64:
            reward_input = reward_inputs[original_index]

            qwen_extra_info = {**reward_input["extra_info"]}
            if "original_image_base64" not in qwen_extra_info and reward_input.get("input_image_base64"):
                qwen_extra_info["original_image_base64"] = reward_input.get("input_image_base64")
                if isinstance(qwen_extra_info["original_image_base64"], list):
                    qwen_extra_info["original_image_base64"] = qwen_extra_info["original_image_base64"][0]
            if isinstance(edited_base64, list):
                edited_base64 = edited_base64[0]
            qwen_extra_info["edited_image_base64"] = edited_base64
            qwen_extra_info["edited_image_path"] = "base64_image"  # placeholder

            qwen_input = {
                "response": reward_input["response"],
                "ground_truth": reward_input["ground_truth"],
                "extra_info": qwen_extra_info
            }
            qwen_inputs.append(qwen_input)
            valid_qwen_indices.append(original_index)
        else:
            logger.warning("Kontext generation failed for sample %s; score=0.", original_index)

    # Stage 2: score successfully generated edits
    final_results = [copy.deepcopy(zero_score) for _ in range(len(reward_inputs))]
    if qwen_inputs:
        with ThreadPoolExecutor(max_workers=MLLM_MAX_WORKERS) as executor:
            qwen_results = list(executor.map(qwen_vl_reward.compute_score, qwen_inputs))

        for i, res in enumerate(qwen_results):
            original_index = valid_qwen_indices[i]
            final_results[original_index] = res["scores"]
            final_logs[original_index].update(res)

    return final_results, final_logs