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

import re
from typing import Optional, Union, List, Dict
import random
import math

import torch
from torch.utils.data import RandomSampler, SequentialSampler, ConcatDataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from replan.train.dataset import ReplanDataset, collate_fn
from replan.train.config import DataConfig, MultiTaskDataConfig

def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0



def create_replan_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]) -> None:
    train_dataset = ReplanDataset(
        data_path=config.train_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,
        extra_info_key=config.extra_info_key,
        image_dir=config.image_dir,
        video_fps=config.video_fps,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
        filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
    )
    # use sampler for better ckpt resume
    if config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(config.seed)
        sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=train_dataset)

    if config.mini_rollout_batch_size is not None:
        train_batch_size = config.mini_rollout_batch_size
    else:
        train_batch_size = config.rollout_batch_size

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    val_dataset = ReplanDataset(
        data_path=config.val_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        extra_info_key=config.extra_info_key,
        image_dir=config.image_dir,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
    )

    if config.val_batch_size == -1:
        val_batch_size = len(val_dataset)
    else:
        val_batch_size = config.val_batch_size

    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    assert len(train_dataloader) >= 1
    assert len(val_dataloader) >= 1
    print(f"Size of train dataloader: {len(train_dataloader)}")
    print(f"Size of val dataloader: {len(val_dataloader)}")
    return train_dataloader, val_dataloader


class TaskGroupedSampler(Sampler):
    """
    Sampler that ensures each batch contains data from only one task
    """
    def __init__(self, 
                 datasets: List, 
                 batch_size: int,
                 shuffle_within_task: bool = True,
                 shuffle_between_tasks: bool = True,
                 drop_last: bool = True,
                 task_weights: Optional[Dict[str, float]] = None,
                 generator: Optional[torch.Generator] = None):
        """
        Args:
            datasets: List of original datasets (before concatenation)
            batch_size: Size of each batch
            shuffle_within_task: Whether to shuffle data within each task
            shuffle_between_tasks: Whether to shuffle batch order across different tasks
            drop_last: Whether to drop the last incomplete batch
            task_weights: Sampling weights for each task, if None then uniform sampling
            generator: Random number generator
        """
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle_within_task = shuffle_within_task
        self.shuffle_between_tasks = shuffle_between_tasks
        self.drop_last = drop_last
        self.generator = generator
        
        # Record index ranges for each task in the concatenated dataset
        self.task_ranges = []
        start_idx = 0
        for dataset in datasets:
            end_idx = start_idx + len(dataset)
            self.task_ranges.append((start_idx, end_idx))
            start_idx = end_idx
        
        self.total_size = sum(len(dataset) for dataset in datasets)
        
        # Set task weights
        if task_weights is None:
            # Default: allocate weights proportional to dataset size
            self.task_weights = [len(dataset) / self.total_size for dataset in datasets]
        else:
            # Get weights based on task names (ensure consistent ordering)
            self.task_weights = []
            for i, dataset in enumerate(datasets):
                task_name = getattr(dataset, 'task_name', f'task_{i}')
                weight = task_weights.get(task_name, 1.0)
                self.task_weights.append(weight)
            
            # Normalize weights
            total_weight = sum(self.task_weights)
            self.task_weights = [w / total_weight for w in self.task_weights]
        
        # Calculate number of batches for each task
        self._calculate_batches_per_task()
    
    def _calculate_batches_per_task(self):
        """Calculate the number of batches each task should produce"""
        self.batches_per_task = []
        total_batches = 0
        
        for i, (dataset, weight) in enumerate(zip(self.datasets, self.task_weights)):
            if self.drop_last:
                max_batches_for_task = len(dataset) // self.batch_size
            else:
                max_batches_for_task = math.ceil(len(dataset) / self.batch_size)
            
            # Calculate expected number of batches based on weight
            expected_batches = max_batches_for_task * weight
            actual_batches = min(int(expected_batches), max_batches_for_task)
            
            self.batches_per_task.append(actual_batches)
            total_batches += actual_batches
        
        self.total_batches = total_batches
    
    def __iter__(self):
        # Generate batch indices for each task
        all_batches = []
        
        for task_idx, (start_idx, end_idx) in enumerate(self.task_ranges):
            dataset_size = end_idx - start_idx
            task_indices = list(range(start_idx, end_idx))
            
            # Shuffle within task
            if self.shuffle_within_task:
                if self.generator is not None:
                    # Use same generator to ensure reproducibility
                    g = torch.Generator()
                    g.manual_seed(self.generator.initial_seed() + task_idx)
                    indices = torch.randperm(dataset_size, generator=g).tolist()
                else:
                    random.shuffle(task_indices)
                    indices = [idx - start_idx for idx in task_indices]
                
                # Convert back to global indices
                task_indices = [start_idx + idx for idx in indices]
            
            # Generate all batches for this task
            num_batches = self.batches_per_task[task_idx]
            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, dataset_size)
                
                if self.drop_last and (batch_end - batch_start) < self.batch_size:
                    continue
                
                batch_indices = task_indices[batch_start:batch_end]
                all_batches.append(batch_indices)
        
        # Shuffle batches across tasks
        if self.shuffle_between_tasks:
            if self.generator is not None:
                g = torch.Generator()
                g.manual_seed(self.generator.initial_seed() + len(self.datasets))
                batch_order = torch.randperm(len(all_batches), generator=g).tolist()
            else:
                batch_order = list(range(len(all_batches)))
                random.shuffle(batch_order)
            
            all_batches = [all_batches[i] for i in batch_order]
        
        # Return all indices
        for batch in all_batches:
            for idx in batch:
                yield idx
    
    def __len__(self):
        return self.total_batches * self.batch_size if self.drop_last else self.total_size


def create_multi_task_replan_dataloader(config: Union[DataConfig, MultiTaskDataConfig], tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]) -> None:
    train_datasets = []
    val_datasets = []
    
    for task_name, task_config in config.tasks.items():
        print(f"Loading task: {task_name}")
        train_dataset = ReplanDataset(
            data_path=task_config.train_files,
            tokenizer=tokenizer,
            processor=processor,
            task_name=task_config.task_name,
            prompt_key=task_config.prompt_key,
            answer_key=task_config.answer_key,
            image_key=task_config.image_key,
            video_key=task_config.video_key,
            extra_info_key=task_config.extra_info_key,
            image_dir=task_config.image_dir,
            video_fps=task_config.video_fps,
            max_prompt_length=task_config.max_prompt_length,
            truncation="right",
            format_prompt=task_config.format_prompt,
            min_pixels=task_config.min_pixels,
            max_pixels=task_config.max_pixels,
            filter_overlong_prompts=task_config.filter_overlong_prompts,
            filter_overlong_prompts_workers=task_config.filter_overlong_prompts_workers,
            sampling_rate=task_config.train_sampling_rate,
        )
        train_datasets.append(train_dataset)
        print(f"Task '{task_name}' train dataset size: {len(train_dataset)}")

        if task_config.val_files is not None:
            val_dataset = ReplanDataset(
                data_path=task_config.val_files,
                tokenizer=tokenizer,
                processor=processor,
                task_name=task_config.task_name,
                prompt_key=task_config.prompt_key,
                answer_key=task_config.answer_key,
                image_key=task_config.image_key,
                extra_info_key=task_config.extra_info_key,
                image_dir=task_config.image_dir,
                max_prompt_length=task_config.max_prompt_length,
                truncation="right",
                format_prompt=task_config.format_prompt,
                min_pixels=task_config.min_pixels,  
                max_pixels=task_config.max_pixels,
                filter_overlong_prompts=task_config.filter_overlong_prompts,
                sampling_rate=task_config.val_sampling_rate,
            )
            val_datasets.append(val_dataset)
            print(f"Task '{task_name}' val dataset size: {len(val_dataset)}")

    # Combine multiple datasets
    combined_train_dataset = ConcatDataset(train_datasets)
    print(f"Combined train dataset size: {len(combined_train_dataset)}")

    if len(val_datasets) > 0:
        combined_val_dataset = ConcatDataset(val_datasets)
        print(f"Combined val dataset size: {len(combined_val_dataset)}")    
    else:
        combined_val_dataset = None

    if config.mini_rollout_batch_size is not None:
        train_batch_size = config.mini_rollout_batch_size
    else:
        train_batch_size = config.rollout_batch_size

    # use sampler for better ckpt resume
    train_dataloader_generator = torch.Generator()
    train_dataloader_generator.manual_seed(config.seed)
    sampler = TaskGroupedSampler(
        datasets=train_datasets, 
        batch_size=train_batch_size, 
        shuffle_within_task=config.shuffle, 
        shuffle_between_tasks=config.shuffle_between_tasks, 
        drop_last=True, 
        generator=train_dataloader_generator
    )


    train_dataloader = StatefulDataLoader(
        dataset=combined_train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )
    

    if len(val_datasets) > 0:
        if config.val_batch_size == -1:
            val_batch_size = len(combined_val_dataset)
        else:
            val_batch_size = config.val_batch_size
        
        val_sampler = TaskGroupedSampler(
            datasets=val_datasets, 
            batch_size=val_batch_size, 
            shuffle_within_task=False, 
            shuffle_between_tasks=False, 
            drop_last=False, 
        )

        val_dataloader = StatefulDataLoader(
            dataset=combined_val_dataset,
            batch_size=val_batch_size,
            sampler=val_sampler,
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
        )
    else:
        val_dataloader = None

    assert len(train_dataloader) >= 1
    #assert len(val_dataloader) >= 1
    print(f"Size of train dataloader: {len(train_dataloader)}")
    if val_dataloader is not None:
        print(f"Size of val dataloader: {len(val_dataloader)}")
    return train_dataloader, val_dataloader