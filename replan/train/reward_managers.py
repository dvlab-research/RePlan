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

import importlib.util
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, TypedDict
from typing import Any

import torch
from transformers import PreTrainedTokenizer

from verl.protocol import DataProto
from verl.workers.reward.function import FunctionRewardManager

class RewardInput(TypedDict):
    task_name: str
    response: str
    response_length: int
    ground_truth: str


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[List[RewardInput]], List[RewardScore]]


class SequentialFunctionRewardManager(FunctionRewardManager):
    reward_fn: SequentialRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        for i in range(len(data)):
            valid_response_ids = response_ids[i][: response_length[i]]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            score = self.reward_fn(
                {
                    "task_name": data.non_tensor_batch["task_name"][i],
                    "response": response_str,
                    "response_length": response_length[i],
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                    "input_image_base64": data.non_tensor_batch["input_image_base64"][i] if "input_image_base64" in data.non_tensor_batch else None,
                    "extra_info": data.non_tensor_batch["extra_info"][i] if "extra_info" in data.non_tensor_batch else None,
                }
            )
            reward_tensor[i, response_length[i] - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class BatchFunctionRewardManager(FunctionRewardManager):
    reward_fn: BatchRewardFunction

    def compute_reward(self, data: DataProto, return_logs: bool = False) -> Tuple[torch.Tensor, Dict[str, List[float]], List[Dict[str, Any]]]:
        reward_inputs = []
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        for i in range(len(data)):
            valid_response_ids = response_ids[i][: response_length[i]]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            reward_inputs.append(
                {
                    "task_name": data.non_tensor_batch["task_name"][i],
                    "response": response_str,
                    "response_length": response_length[i],
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                    "input_image_base64": data.non_tensor_batch["input_image_base64"][i] if "input_image_base64" in data.non_tensor_batch else None,
                    "extra_info": data.non_tensor_batch["extra_info"][i] if "extra_info" in data.non_tensor_batch else None,
                }
            )

        # reward_fn now returns scores and logs
        scores, logs = self.reward_fn(reward_inputs)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            reward_tensor[i, response_length[i] - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        # Return three values
        if return_logs:
            return reward_tensor, reward_metrics, logs
        else:
            return reward_tensor, reward_metrics
