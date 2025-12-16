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
"""
PPO config
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple, Union, Dict, List

from verl.workers.config import WorkerConfig
from omegaconf import MISSING


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))

@dataclass
class DataConfig:
    """Configuration for a single dataset and its loading parameters."""
    # task name
    task_name: str = ""

    # task specific
    train_files: str = ""
    val_files: Optional[str] = None
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    video_key: str = "videos"
    extra_info_key: str = "extra_info"
    image_dir: Optional[str] = None
    format_prompt: Optional[str] = None
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 16
    train_sampling_rate: float = 1.0
    val_sampling_rate: float = 1.0

    # global
    override_chat_template: Optional[str] = None
    video_fps: float = 2.0
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: int = -1
    shuffle: bool = True
    shuffle_between_tasks: bool = False
    seed: int = 1

    def post_init(self):
        if self.image_dir is not None:
            if os.path.exists(self.image_dir):  # ray job uses absolute path
                self.image_dir = os.path.abspath(self.image_dir)
            else:
                print(f"Image directory {self.image_dir} not found.")
                self.image_dir = None

        if self.format_prompt is not None:
            if os.path.exists(self.format_prompt):  # ray job uses absolute path
                self.format_prompt = os.path.abspath(self.format_prompt)
            else:
                print(f"Format prompt file {self.format_prompt} not found.")
                self.format_prompt = None
        



@dataclass
class MultiTaskDataConfig:
    """
    Configuration for multi-task data loading.

    This class holds a dictionary of DataConfig objects, one for each task,
    and provides global settings that can be shared across all tasks.
    """
    
    # Dictionary of DataConfig objects, each task requires a unique key name
    tasks: Dict[str, DataConfig] = field(default_factory=dict)
    
    # Global configuration settings (can be overridden by individual tasks)
    override_chat_template: Optional[str] = None
    max_prompt_length: int = 1300
    max_response_length: int = 2000
    rollout_batch_size: int = 16
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: int = -1
    shuffle: bool = True
    video_fps: float = 2.0
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    seed: int = 42
    shuffle_between_tasks: bool = False
    
    def post_init(self):
        # Validate tasks
        if not isinstance(self.tasks, dict) or len(self.tasks) == 0:
            raise ValueError("tasks must be a non-empty Dict[str, DataConfig]")
        
        # Apply global configuration as defaults for each task (if not specified by the task)
        for task_name, task in self.tasks.items():
            if not isinstance(task, DataConfig):
                raise ValueError(f"Task '{task_name}' must be a DataConfig instance")
            
            # Apply global configuration as defaults
            self._apply_global_defaults(task)
        
        # Call post_init for each DataConfig
        for task_name, task_config in self.tasks.items():
            if hasattr(task_config, 'post_init'):
                task_config.post_init()
                if task_config.task_name == "":
                    task_config.task_name = task_name
    
    def _apply_global_defaults(self, task_config: DataConfig):
        """Apply global configuration to task configuration (if not specified by the task)"""
        # Create field mapping, only apply global config when task config uses default values
        global_defaults = {
            'max_prompt_length': self.max_prompt_length,
            'max_response_length': self.max_response_length, 
            'rollout_batch_size': self.rollout_batch_size,
            'shuffle': self.shuffle,
            'seed': self.seed,
            'max_pixels': self.max_pixels,
            'min_pixels': self.min_pixels,
        }
        
        # Get default values from DataConfig
        default_task = DataConfig()
        
        for field_name, global_value in global_defaults.items():
            current_value = getattr(task_config, field_name)
            default_value = getattr(default_task, field_name)
            
            # Apply global config if task config uses default value
            if current_value == default_value:
                setattr(task_config, field_name, global_value)
    
    def get_task_names(self) -> List[str]:
        """Get all task names"""
        return list(self.tasks.keys())
    
    def get_task_config(self, task_name: str) -> DataConfig:
        """Get configuration for a specific task"""
        if task_name not in self.tasks:
            raise ValueError(f"Task '{task_name}' not found")
        return self.tasks[task_name]
    
    def get_effective_config(self, task_name: str) -> DataConfig:
        """Get effective task configuration with global settings applied"""
        return self.get_task_config(task_name)


@dataclass
class AlgorithmConfig:
    """Configuration for the PPO algorithm hyperparameters."""
    gamma: float = 1.0
    """discount factor for ppo gae advantage estimator"""
    lam: float = 1.0
    """lambda value for ppo gae advantage estimator"""
    adv_estimator: str = "grpo"
    """advantage estimator, support `gae`, `grpo`, `reinforce_plus_plus`, `remax`, `rloo`"""
    disable_kl: bool = False
    """disable reference model"""
    use_kl_loss: bool = False
    """use kl loss instead of kl in reward"""
    kl_penalty: str = "kl"
    """kl penalty type, support `kl`, `abs`, `mse`, `low_var_kl`, `full`"""
    kl_coef: float = 1e-3
    """kl coefficient"""
    kl_type: str = "fixed"
    """kl controller type, support `fixed`, `adaptive`"""
    kl_horizon: float = 10000.0
    """kl horizon for adaptive kl controller"""
    kl_target: float = 0.1
    """target kl for adaptive kl controller"""
    online_filtering: bool = False
    """use online filtering"""
    filter_key: str = "overall"
    """reward key for filtering samples"""
    filter_low: float = 0.01
    """filter out low reward samples if online filtering"""
    filter_high: float = 0.99
    """filter out high reward samples if online filtering"""


@dataclass
class TrainerConfig:
    """Configuration for the training process, including logging, hardware, and checkpointing."""
    total_epochs: int = 15
    """total epochs for training"""
    max_steps: Optional[int] = None
    """max steps for training, if specified, total_epochs is ignored"""
    project_name: str = "easy_r1"
    """project name for logger"""
    experiment_name: str = "demo"
    """experiment name for logger"""
    logger: Tuple[str] = ("console", "wandb")
    """logger type, support `console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`"""
    nnodes: int = 1
    """number of nodes for training"""
    n_gpus_per_node: int = 8
    """number of gpus per node for training"""
    max_try_make_batch: int = 20
    """max number of generations for online filtering, -1 means no limit"""
    critic_warmup: int = 0
    """critic warmup steps"""
    val_freq: int = -1
    """validation frequency, -1 means no validation"""
    val_before_train: bool = True
    """validate before training"""
    val_only: bool = False
    """validate only, skip training"""
    val_generations_to_log: int = 0
    """number of generations to log for validation"""
    train_generations_to_log: int = 0
    """number of generations to log for training"""
    save_freq: int = -1
    """save frequency, -1 means no saving"""
    save_limit: int = -1
    """max number of checkpoints to save, -1 means no limit"""
    save_model_only: bool = False
    """save model only, no optimizer state dict"""
    save_checkpoint_path: Optional[str] = None
    """save checkpoint path, if not specified, use `checkpoints/project_name/experiment_name`"""
    load_checkpoint_path: Optional[str] = None
    """load checkpoint path"""

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.join(os.path.abspath(self.save_checkpoint_path), self.project_name, self.experiment_name)  # ray job uses absolute path
        if self.load_checkpoint_path is not None:
            if os.path.exists(self.load_checkpoint_path):  # ray job uses absolute path
                self.load_checkpoint_path = os.path.abspath(self.load_checkpoint_path)
            else:
                print(f"Model checkpoint {self.load_checkpoint_path} not found.")
                self.load_checkpoint_path = None


@dataclass
class ReplanConfig:
    """
    Top-level configuration class for the Replan project.

    This class aggregates all other configuration objects (Data, Worker,
    Algorithm, Trainer) into a single unified structure. It uses structured
    dataclasses to provide type safety and easy instantiation from YAML files
    via OmegaConf.
    """
    data: MultiTaskDataConfig = field(default_factory=MultiTaskDataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def post_init(self):
        # Determine max lengths based on all tasks
        max_prompt_length = max(task.max_prompt_length for task in self.data.tasks.values())
        max_response_length = max(task.max_response_length for task in self.data.tasks.values())
        self.worker.rollout.prompt_length = max_prompt_length
        self.worker.rollout.response_length = max_response_length
        
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
    
    def get_task_configs(self) -> Dict[str, DataConfig]:
        """Get all task configurations"""
        return self.data.tasks


