"""调度模块的测试用例。"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from src.scheduling.task_based_scheduler import TaskBasedScheduler
from src.scheduling.task_allocator import TaskAllocator
from src.scheduling.token_based_scheduler import TokenBasedScheduler
from src.scheduling.base_allocator import BaseAllocator
from typing import List, Dict, Any

# 测试配置
TEST_CONFIG = {
    "hardware_config": {
        "apple_m1_pro": {
            "device_type": "m1_pro",
            "idle_power": 10.0,
            "sample_interval": 200
        },
        "nvidia_rtx4050": {
            "device_type": "rtx4050",
            "idle_power": 15.0,
            "sample_interval": 200
        },
        "nvidia_rtx4090": {
            "device_type": "rtx4090",
            "idle_power": 20.0,
            "sample_interval": 200
        }
    },
    "model_config": {
        "models": {
            "tinyllama": {
                "model_name": "tinyllama",
                "model_path": "path/to/model",
                "mode": "local",
                "batch_size": 1,
                "max_length": 128
            }
        }
    }
}

@pytest.fixture
def task_scheduler():
    """创建 TaskBasedScheduler 实例的 fixture。"""
    return TaskBasedScheduler(TEST_CONFIG)

@pytest.fixture
def task_allocator():
    """创建 TaskAllocator 实例的 fixture。"""
    return TaskAllocator(
        hardware_config=TEST_CONFIG["hardware_config"],
        model_config=TEST_CONFIG["model_config"]
    )

@pytest.fixture
def token_scheduler():
    """创建 TokenBasedScheduler 实例的 fixture。"""
    return TokenBasedScheduler(TEST_CONFIG)

def test_task_scheduler_init(task_scheduler):
    """测试 TaskBasedScheduler 初始化。"""
    assert task_scheduler.tasks == []
    assert task_scheduler.device_queues == {"apple_m1_pro": [], "nvidia_rtx4050": []}
    assert task_scheduler.gpu_cache == {}
    assert task_scheduler.cpu_cache == {}
    assert not task_scheduler.is_warmed_up
    assert task_scheduler.device_affinity == {}
    assert task_scheduler.affinity_threshold == 0.7
    assert task_scheduler.batch_size == 4
    assert task_scheduler.current_batch == {"apple_m1_pro": [], "nvidia_rtx4050": []}
    assert task_scheduler.batch_results == {}

def test_task_scheduler_add_task(task_scheduler):
    """测试 TaskBasedScheduler 添加任务。"""
    task = {"instruction": "test", "input": "test input"}
    task_scheduler.add_task(task)
    assert len(task_scheduler.tasks) == 1
    assert task_scheduler.tasks[0] == task

def test_task_scheduler_schedule_tasks(task_scheduler):
    """测试 TaskBasedScheduler 调度任务。"""
    tasks = [
        {
            "query": {
                "input_tokens": 500,
                "output_tokens": 50,
                "prompt": "test input 1"
            },
            "model": "tinyllama"
        },
        {
            "query": {
                "input_tokens": 600,
                "output_tokens": 50,
                "prompt": "test input 2"
            },
            "model": "tinyllama"
        }
    ]

    for task in tasks:
        task_scheduler.add_task(task)

    scheduled_tasks = task_scheduler.schedule(tasks)
    assert isinstance(scheduled_tasks, list)
    assert len(scheduled_tasks) == 2

def test_task_scheduler_warmup(task_scheduler):
    """测试 TaskBasedScheduler 预热。"""
    task_scheduler.warmup()
    assert task_scheduler.is_warmed_up

def test_task_allocator_init(task_allocator):
    """测试 TaskAllocator 初始化。"""
    assert task_allocator.hardware_config == TEST_CONFIG["hardware_config"]
    assert task_allocator.model_config == TEST_CONFIG["model_config"]
    assert task_allocator.initialized

def test_task_allocator_init_invalid_config():
    """测试 TaskAllocator 初始化时的无效配置。"""
    invalid_configs = [
        (None, None),           # 空配置
        ({}, {}),            # 空字典
        ({"hardware_config": {}}, {}),  # 空硬件配置
        ({}, {"model_config": {}}),     # 空模型配置
    ]
    
    for hardware_config, model_config in invalid_configs:
        with pytest.raises(ValueError):
            TaskAllocator(hardware_config=hardware_config, model_config=model_config)

def test_task_allocator_allocate(task_allocator):
    """测试 TaskAllocator 分配任务。"""
    tasks = [
        {
            "query": {
                "input_tokens": 500,
                "output_tokens": 50,
                "prompt": "test input 1"
            },
            "model": "tinyllama"
        }
    ]

    allocations = task_allocator.allocate(tasks, model_name="tinyllama")
    assert isinstance(allocations, list)
    assert len(allocations) == 1
    assert "metrics" in allocations[0]
    assert "hardware" in allocations[0]

def test_token_scheduler_init(token_scheduler):
    """测试 TokenBasedScheduler 初始化。"""
    assert token_scheduler.initialized
    assert token_scheduler.token_threshold == 1000

def test_token_scheduler_schedule(token_scheduler):
    """测试 TokenBasedScheduler 调度任务。"""
    tasks = [
        {
            "decoded_text": "这是一个短文本",
            "input_tokens": 500,
            "tokens": 500
        },   # 小任务
        {
            "decoded_text": "这是一个长文本，需要更多的处理能力",
            "input_tokens": 1500,
            "tokens": 1500
        }   # 大任务
    ]
    
    scheduled_tasks = token_scheduler.schedule(tasks)
    assert len(scheduled_tasks) == 2
    assert scheduled_tasks[0]["model"] == "tinyllama"
    assert scheduled_tasks[0]["hardware"] == "nvidia_rtx4050"
    assert scheduled_tasks[1]["model"] == "tinyllama"
    assert scheduled_tasks[1]["hardware"] == "nvidia_rtx4050"

def test_token_scheduler_schedule_invalid_tasks(token_scheduler):
    """测试 TokenBasedScheduler 调度无效任务。"""
    invalid_tasks = [
        None,           # 空任务
        {},            # 空字典
        {"tokens": -1}, # 负令牌数
        {"tokens": "invalid"} # 非数字令牌数
    ]
    
    for task in invalid_tasks:
        with pytest.raises((ValueError, TypeError)):
            token_scheduler.schedule([task])

class TestAllocator(BaseAllocator):
    """测试用的任务分配器。"""
    
    def _init_allocator(self) -> None:
        """初始化任务分配器。"""
        self.initialized = True
    
    def allocate(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """分配任务。"""
        if not self.initialized:
            raise RuntimeError("分配器未初始化")
        if not tasks:
            return []
            
        allocated_tasks = []
        for task in tasks:
            allocated_tasks.append({
                "task": task,
                "hardware": "apple_m1_pro",
                "model": "tinyllama"
            })
        return allocated_tasks

def test_base_allocator_init():
    """测试 BaseAllocator 初始化。"""
    allocator = TestAllocator(TEST_CONFIG)
    assert allocator.config == TEST_CONFIG
    assert not allocator.initialized

def test_base_allocator_cleanup():
    """测试 BaseAllocator 资源清理。"""
    allocator = TestAllocator(TEST_CONFIG)
    allocator.initialize()
    assert allocator.initialized
    allocator.cleanup()
    assert not allocator.initialized 