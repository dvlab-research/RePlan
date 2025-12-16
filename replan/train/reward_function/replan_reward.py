# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import re
import json
from typing import Any, Dict, List
from collections import defaultdict
import copy
import numpy as np
from scipy.optimize import linear_sum_assignment

from replan.train.reward_function.remote_reward import compute_score_batch as compute_score_batch_planning

logger = logging.getLogger(__name__)

def validate_region_format(region_str: str) -> bool:
    """Validate the JSON format inside the <region>...</region> tag."""
    try:
        region_data = json.loads(region_str.strip())
        if not isinstance(region_data, list):
            return False
        
        for item in region_data:
            if not isinstance(item, dict):
                return False
            if "bbox_2d" not in item or "hint" not in item:
                return False
            if not isinstance(item["bbox_2d"], list) or len(item["bbox_2d"]) != 4:
                return False
            if not all(isinstance(coord, (int, float)) for coord in item["bbox_2d"]):
                return False
            if not isinstance(item["hint"], str):
                return False
        return True
    except (json.JSONDecodeError, KeyError, TypeError):
        return False


def format_reward(response: str) -> float:
    """
    Check whether the response follows the required tag format.

    Supported formats:
      - With region: <think>...</think><gen_image>...</gen_image><region>...</region>
      - Without region: <think>...</think><gen_image>...</gen_image>
    """
    has_region = bool(re.search(r"<region>.*?</region>", response, re.DOTALL))
    if has_region:
        pattern = re.compile(r"<think>.*?</think>\s*<gen_image>.*?</gen_image>\s*<region>.*?</region>", re.DOTALL)
    else:
        pattern = re.compile(r"<think>.*?</think>\s*<gen_image>.*?</gen_image>", re.DOTALL)
    format_match = re.fullmatch(pattern, response.strip())
    return 1.0 if format_match else 0.0


def extract_components(response: str) -> Dict[str, str]:
    """Extract tagged components from a response string."""
    components = {"think": "", "region": "", "gen_image": "", "answer": ""}
    
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if think_match:
        components["think"] = think_match.group(1).strip()
    
    region_match = re.search(r"<region>(.*?)</region>", response, re.DOTALL)
    if region_match:
        components["region"] = region_match.group(1).strip()
    
    gen_image_match = re.search(r"<gen_image>(.*?)</gen_image>", response, re.DOTALL)
    if gen_image_match:
        components["gen_image"] = gen_image_match.group(1).strip()
    
    return components


def _calculate_length_reward(length: int, min_len: int, max_len: int, reward_value: float) -> float:
    """Linear interpolation reward based on a length range."""
    if length >= max_len:
        return reward_value
    if length <= min_len:
        return 0.0
    if max_len <= min_len:
        return 0.0
    return reward_value * (length - min_len) / (max_len - min_len)


def content_quality_reward(response: str) -> float:
    """Heuristic content quality reward (think/gen_image + optional region validity)."""
    components = extract_components(response)
    
    quality_score = 0.0
    
    think_len = len(components["think"])
    quality_score += _calculate_length_reward(think_len, 0, 50, 0.25)

    if components["think"] == "Reasoning process":
        quality_score -= 20 
    elif components["think"] == "Your reasoning process to understand the user's instruction based on the image. Decompose the instruction into global and local edits, identifying regions for editing.":
        quality_score -= 20
    elif components["think"] == "First, state the user's main goal based on their instruction. Then, detail the global and local edits required to achieve this goal, explaining your reasoning for each.":
        quality_score -= 20
        
    if think_len > 512:
        quality_score -= 0.25
    
    if components["region"]:
        if validate_region_format(components["region"]):
            quality_score += 0.25
        else:
            quality_score -= 0.25
    
    gen_image_len = len(components["gen_image"])
    quality_score += _calculate_length_reward(gen_image_len, 0, 20, 0.25)
    
    if components["gen_image"] == "Detailed prompt for the image generator":
        quality_score -= 20
    elif components["gen_image"] == "Global edit instruction":
        quality_score -= 20
    elif components["gen_image"] == "":
        quality_score -= 20
    
    if gen_image_len > 512:
        quality_score -= 0.25
    return quality_score


def region_quality_reward(response: str) -> float:
    """Heuristic quality reward for the <region> JSON content."""
    components = extract_components(response)
    
    if not components["region"]:
        return 0.0
    
    region_score = 0.0
    
    if validate_region_format(components["region"]):
        region_score += 0.5
        
        try:
            region_data = json.loads(components["region"])
            
            # Encourage a reasonable number of regions.
            if 1 <= len(region_data) <= 5:
                region_score += 0.2
            
            # Heuristics for hints/bboxes.
            valid_hints = 0
            for item in region_data:
                hint = item.get("hint", "")
                if hint in ["how to edit for this region", "", "change the color of this one apple to blue", "keep this one apple unchanged"]:
                    region_score -= 1.0
                else:
                    valid_hints += 1
                bbox_2d = item.get("bbox_2d", [])
                if len(bbox_2d) != 4:
                    region_score -= 1.0
                if bbox_2d == [10,150,150,210] or bbox_2d == [150,50,200,150]:
                    region_score -= 1.0
            
            if valid_hints == len(region_data) and len(region_data) > 0:
                region_score += 0.2
                
        except (json.JSONDecodeError, KeyError):
            region_score -= 0.5
    else:
        region_score -= 0.5
    
    return min(region_score, 1.0)


def compute_score_editing(reward_inputs: List[Dict[str, Any]], 
                 format_weight: float = 0.5, 
                 quality_weight: float = 0.3,
                 region_weight: float = 0.2) -> List[Dict[str, float]]:
    """Compute a combined score for editing tasks."""
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for replan reward function.")

    scores = []
    for reward_input in reward_inputs:
        # Normalize whitespace around tags (e.g. Qwen2.5-VL variants).
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        
        format_score = format_reward(response)
        
        quality_score = content_quality_reward(response)
        
        region_score = region_quality_reward(response)
        
        overall_score = (format_weight * format_score + 
                        quality_weight * quality_score + 
                        region_weight * region_score)
        
        scores.append(
            {
                "overall": overall_score,
                "format": format_score,
                "quality": quality_score,
                "region": region_score,
            }
        )

    return scores


def compute_score_generation(reward_inputs: List[Dict[str, Any]], 
                                format_weight: float = 0.6, 
                                quality_weight: float = 0.4) -> List[Dict[str, float]]:
    """Compute a combined score for generation tasks (ignores region)."""
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for replan reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        
        format_score = format_reward_without_region(response)
        
        quality_score = content_quality_reward_without_region(response)
        
        overall_score = (format_weight * format_score + 
                        quality_weight * quality_score)
        
        scores.append(
            {
                "overall": overall_score,
                "format": format_score,
                "quality": quality_score,
            }
        )

    return scores


def format_reward_without_region(response: str) -> float:
    """Check tag format, ignoring <region>."""
    pattern = re.compile(r"<think>.*?</think>\s*<gen_image>.*?</gen_image>", re.DOTALL)
    
    format_match = re.fullmatch(pattern, response.strip())
    return 1.0 if format_match else 0.0


def content_quality_reward_without_region(response: str) -> float:
    """Content quality reward (ignores <region>)."""
    components = extract_components(response)
    
    quality_score = 0.0
    
    think_len = len(components["think"])
    quality_score += _calculate_length_reward(think_len, 0, 40, 0.33)

    if components["think"] == "Reasoning process":
        quality_score -= 20 
    
    if think_len > 512:
        quality_score -= 0.33
    
    gen_image_len = len(components["gen_image"])
    quality_score += _calculate_length_reward(gen_image_len, 0, 5, 0.33)
    
    if components["gen_image"] == "Detailed prompt for the image generator":
        quality_score -= 20
    elif components["gen_image"] == "Detailed prompt for the image editor":
        quality_score -= 20
    
    if gen_image_len > 512:
        quality_score -= 0.33
    return quality_score

# ==============================================================================
# Functions for VQA/Detection Task
# ==============================================================================

def detection_format_reward(predict_str: str) -> float:
    """Format check for VQA/Detection: expects <think>...</think><region>...</region>."""
    pattern = r"<think>.*?</think>\s*<region>.*?</region>"
    match = re.fullmatch(pattern, predict_str, re.DOTALL)
    if not match:
        return 0.0

    components = extract_components(predict_str)
    if components["think"] == "Reasoning process":
        return -20.0

    try:
        region_match = re.search(r'<region>\s*(.*?)\s*</region>', predict_str, re.DOTALL)
        if not region_match:
            return 0.0
        
        data = json.loads(region_match.group(1))
        
        if not isinstance(data, list):
            return 0.5
        
        for item in data:
            if not isinstance(item, dict) or "bbox_2d" not in item or "point_2d" not in item or "label" not in item:
                return 0.75
                
    except json.JSONDecodeError:
        return 0.25
    
    return 1.0

def detection_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    max_accuracy_reward = 0.0
    MAX_OBJECTS = 120
    
    try:
        gt_data = json.loads(ground_truth)
        gt_bboxes = [item['bbox_2d'] for item in gt_data]
        gt_points = [item.get('point_2d', [0, 0]) for item in gt_data]
            
        json_match = re.search(r'<region>\s*(.*?)\s*</region>', predict_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            pred_bboxes = [item['bbox_2d'] for item in data]
            pred_points = [item.get('point_2d', [0, 0]) for item in data]
            
            if len(pred_bboxes) > MAX_OBJECTS:
                pred_bboxes = pred_bboxes[:MAX_OBJECTS]
                pred_points = pred_points[:MAX_OBJECTS]
            
            if len(gt_bboxes) > MAX_OBJECTS:
                gt_bboxes = gt_bboxes[:MAX_OBJECTS]
                gt_points = gt_points[:MAX_OBJECTS]
            
            pred_bboxes = np.array(pred_bboxes)
            pred_points = np.array(pred_points)
            gt_bboxes = np.array(gt_bboxes)
            gt_points = np.array(gt_points)
            
            iou_matrix = batch_iou(pred_bboxes, gt_bboxes)
            l1_matrix = batch_l1_distance(pred_bboxes, gt_bboxes)
            points_dist_matrix = batch_points_distance(pred_points, gt_points)
            points_in_box = batch_points_in_box(pred_points, pred_bboxes)
            
            iou_reward = (iou_matrix > 0.5).astype(float)
            bbox_l1_reward = (l1_matrix < 10).astype(float)
            point_reward = ((points_dist_matrix < 30) & points_in_box[:,np.newaxis]).astype(float)
            
            cost_matrix = 3.0 - (iou_reward + bbox_l1_reward + point_reward)
            
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            total_reward = len(row_indices) * 3.0 - cost_matrix[row_indices, col_indices].sum()
            
            max_length = max(len(pred_bboxes), len(gt_bboxes))
            if max_length > 0:
                max_accuracy_reward = total_reward / max_length
            
    except Exception:
        pass
    return max_accuracy_reward

def detection_non_repeat_reward(predict_str: str) -> float:
    non_repeat_reward = 1.0
    try:
        sentences = predict_str.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        seen = set()
        repeats = 0
        
        for sentence in sentences:
            if sentence in seen:
                repeats += 1
            if repeats >=2:
                non_repeat_reward = 0
                break
            seen.add(sentence)
            
    except Exception:
        pass
    
    return non_repeat_reward

def batch_iou(boxes1, boxes2):
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)
    
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    
    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
    box1Area = (x12 - x11 + 1) * (y12 - y11 + 1)
    box2Area = (x22 - x21 + 1) * (y22 - y21 + 1)
    
    unionArea = box1Area + np.transpose(box2Area) - interArea
    iou = interArea / np.maximum(unionArea, 1e-6)
    return iou

def batch_l1_distance(boxes1, boxes2):
    boxes1 = boxes1[:, np.newaxis, :]
    boxes2 = boxes2[np.newaxis, :, :]
    return np.mean(np.abs(boxes1 - boxes2), axis=2)

def batch_points_distance(points1, points2):
    points1 = points1[:, np.newaxis, :]
    points2 = points2[np.newaxis, :, :]
    dist = np.sqrt(np.sum((points1 - points2)**2, axis=2))
    return dist

def batch_points_in_box(points, boxes):
    x_check = (points[:,0] >= boxes[:,0]) & (points[:,0] <= boxes[:,2])
    y_check = (points[:,1] >= boxes[:,1]) & (points[:,1] <= boxes[:,3])
    return x_check & y_check

def compute_score_detection(reward_inputs: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for replan reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        ground_truth = reward_input.get("ground_truth", "[]")
        
        format_reward = detection_format_reward(response)
        accuracy_reward = detection_accuracy_reward(response, ground_truth)
        non_repeat_reward = detection_non_repeat_reward(response)

        overall_score = format_reward + accuracy_reward + non_repeat_reward
        
        scores.append(
            {
                "overall": overall_score,
                "format": format_reward,
                "accuracy": accuracy_reward,
                "non_repeat": non_repeat_reward,
            }
        )
    return scores

def compute_score_edit_region_planning(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.2) -> List[Dict[str, float]]:
    """Combine planning score (Kontext+Qwen-VL) with format score."""
    scores_plan, logs_plan = compute_score_batch_planning(reward_inputs)
    scores_format = compute_score_editing(reward_inputs)
    for i in range(len(scores_format)):
        scores_format[i]["overall"] = scores_format[i]["overall"] * format_weight
    scores_format_copy = copy.deepcopy(scores_format)
    scores_combined = []
    try:
        for i in range(len(scores_format)):
            try:
                scores_combined.append({
                    "overall_planning": scores_plan[i]["overall"],
                    "overall_format": scores_format[i]["overall"],
                    "overall": scores_plan[i].pop("overall") + scores_format[i].pop("overall"),
                })
                scores_combined[i].update(scores_plan[i])
                scores_combined[i].update(scores_format[i])
            except:
                import traceback
                traceback.print_exc()
                scores_combined.append({
                    "overall_planning": 0.0,
                    "overall_format": scores_format[i]["overall"],
                    "overall": scores_format[i].pop("overall"),
                })
                if "overall" in scores_plan[i]:
                    scores_plan[i].pop("overall")
                scores_combined[i].update(scores_plan[i])
                scores_combined[i].update(scores_format[i])
    except:
        import traceback
        traceback.print_exc()
        return scores_format_copy, logs_plan

    return scores_combined, logs_plan


def compute_score_edit_region_planning_nothink(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.2) -> List[Dict[str, float]]:
    """Like compute_score_edit_region_planning, but uses the no-think format scorer."""
    scores_plan, logs_plan = compute_score_batch_planning(reward_inputs, format_compute_func=compute_score_editing_nothink)
    scores_format = compute_score_editing_nothink(reward_inputs)
    for i in range(len(scores_format)):
        scores_format[i]["overall"] = scores_format[i]["overall"] * format_weight
    scores_format_copy = copy.deepcopy(scores_format)
    scores_combined = []
    try:
        for i in range(len(scores_format)):
            try:
                scores_combined.append({
                    "overall_planning": scores_plan[i]["overall"],
                    "overall_format": scores_format[i]["overall"],
                    "overall": scores_plan[i].pop("overall") + scores_format[i].pop("overall"),
                })
                scores_combined[i].update(scores_plan[i])
                scores_combined[i].update(scores_format[i])
            except:
                # error traceback
                import traceback
                traceback.print_exc()
                scores_combined.append({
                    "overall_planning": 0.0,
                    "overall_format": scores_format[i]["overall"],
                    "overall": scores_format[i].pop("overall"),
                })
                if "overall" in scores_plan[i]:
                    scores_plan[i].pop("overall")
                scores_combined[i].update(scores_plan[i])
                scores_combined[i].update(scores_format[i])
    except:
        import traceback
        traceback.print_exc()
        return scores_format_copy, logs_plan

    return scores_combined, logs_plan


def format_reward_nothink(response: str) -> float:
    """Format check without <think> (optional <region>)."""
    has_region = bool(re.search(r"<region>.*?</region>", response, re.DOTALL))
    
    if has_region:
        pattern = re.compile(r"<gen_image>.*?</gen_image>\s*<region>.*?</region>", re.DOTALL)
    else:
        pattern = re.compile(r"<gen_image>.*?</gen_image>", re.DOTALL)
    
    format_match = re.fullmatch(pattern, response.strip())
    return 1.0 if format_match else 0.0


def content_quality_reward_nothink(response: str) -> float:
    """Content quality reward without <think>."""
    components = extract_components(response)
    
    quality_score = 0.0
    
    if components["region"]:
        if validate_region_format(components["region"]):
            quality_score += 0.5
        else:
            quality_score -= 0.5
    
    gen_image_len = len(components["gen_image"])
    quality_score += _calculate_length_reward(gen_image_len, 0, 5, 0.5)
    
    if components["gen_image"] == "Detailed prompt for the image generator":
        quality_score -= 20
    elif components["gen_image"] == "Global edit instruction":
        quality_score -= 20
    elif components["gen_image"] == "":
        quality_score -= 20
    
    if gen_image_len > 512:
        quality_score -= 0.5
        
    return quality_score


def compute_score_editing_nothink(reward_inputs: List[Dict[str, Any]], 
                 format_weight: float = 0.5, 
                 quality_weight: float = 0.3,
                 region_weight: float = 0.2) -> List[Dict[str, float]]:
    """Compute a combined score for editing tasks without <think>."""
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for replan reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        
        # 格式分数：检查是否严格遵循格式
        format_score = format_reward_nothink(response)
        
        # 内容质量分数：检查各部分内容的质量
        quality_score = content_quality_reward_nothink(response)
        
        # region质量分数：专门评估region部分
        region_score = region_quality_reward(response)
        
        # 综合分数
        overall_score = (format_weight * format_score + 
                        quality_weight * quality_score + 
                        region_weight * region_score)
        
        scores.append(
            {
                "overall": overall_score,
                "format": format_score,
                "quality": quality_score,
                "region": region_score,
            }
        )

    return scores


def compute_score_unified(reward_inputs: List[Dict[str, Any]], task_configs: Dict[str, Dict] = None) -> List[Dict[str, float]]:
    """
    Unified entry that supports multiple tasks with per-task compute functions.

    Args:
        reward_inputs: list of inputs
        task_configs: optional task config mapping, e.g.
        {
            "task_name": {
                "compute_func": callable,
                "prefix": optional metric name prefix (defaults to task_name)
            }
        }
    """
    default_configs = {
        "text_refinement": {
            "compute_func": compute_score_generation,
            "prefix": "text_refinement"
        },
        "text_refinement_edit": {
            "compute_func": compute_score_editing,
            "prefix": "text_refinement_edit"
        },
        "text_refinement_edit_nothink": {
            "compute_func": compute_score_editing_nothink,
            "prefix": "text_refinement_edit_nothink"
        },
        "detection": {
            "compute_func": compute_score_detection,
            "prefix": "detection"
        },
        "edit_region_planning": {
            "compute_func": compute_score_edit_region_planning,
            "prefix": "edit_region_planning"
        },
        "edit_region_planning_nothink": {
            "compute_func": compute_score_edit_region_planning_nothink,
            "prefix": "edit_region_planning_nothink"
        }
    }
    
    if task_configs:
        default_configs.update(task_configs)

    # Group by task_name while preserving original indices.
    tasks_to_process = defaultdict(list)
    for i, reward_input in enumerate(reward_inputs):
        task_name = reward_input["task_name"]
        tasks_to_process[task_name].append((i, reward_input))

    scores = [None] * len(reward_inputs)
    logs = [None] * len(reward_inputs)

    # Process per task group.
    for task_name, indexed_inputs in tasks_to_process.items():
        if task_name in default_configs:
            config = default_configs[task_name]
            compute_func = config["compute_func"]
            prefix = config.get("prefix", task_name)
            
            original_indices = [item[0] for item in indexed_inputs]
            inputs_for_task = [item[1] for item in indexed_inputs]

            # Compute function may return (scores, logs) or just scores.
            results = compute_func(inputs_for_task)
            if isinstance(results, tuple) and len(results) == 2:
                task_scores_list, task_logs_list = results
            else:
                task_scores_list = results
                task_logs_list = [None] * len(task_scores_list)


            if len(task_scores_list) != len(inputs_for_task):
                raise ValueError(
                    f"任务 '{task_name}' 的计算函数返回值数量 "
                    f"({len(task_scores_list)}) 与输入数量 "
                    f"({len(inputs_for_task)}) 不匹配。"
                )

            for i, task_scores in enumerate(task_scores_list):
                try:
                    final_score = {"overall": task_scores["overall"]}
                except:
                    final_score = {"overall": 0.0}
                    logger.error("Invalid task_scores: %s", task_scores)
                for key, value in task_scores.items():
                    if key != "overall":
                        final_score[f"{prefix}_{key}"] = value
                
                
                original_index = original_indices[i]
                scores[original_index] = final_score
                logs[original_index] = task_logs_list[i]
        else:
            raise ValueError(f"Unknown task_name: {task_name}")
            
    if any(s is None for s in scores):
        raise RuntimeError("Some reward_inputs were not processed; please check task routing.")

    return scores, logs
