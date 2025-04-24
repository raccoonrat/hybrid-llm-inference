"""任务分配器测试模块。"""

import pytest
from src.scheduling.task_allocator import TaskAllocator
from typing import Dict, Any, List

# 测试配置
HARDWARE_CONFIG = {
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
}

MODEL_CONFIG = {
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

@pytest.fixture
def allocator():
    """创建 TaskAllocator 实例的 fixture。"""
    return TaskAllocator(HARDWARE_CONFIG, MODEL_CONFIG)

def test_initialization(allocator):
    """测试初始化。"""
    assert allocator.hardware_config == HARDWARE_CONFIG
    assert allocator.model_config == MODEL_CONFIG
    assert len(allocator.available_hardware) == 3
    assert "apple_m1_pro" in allocator.available_hardware
    assert "nvidia_rtx4050" in allocator.available_hardware
    assert "nvidia_rtx4090" in allocator.available_hardware

def test_allocate_small_task(allocator):
    """测试小任务分配。"""
    task = {
        "input_tokens": 500,
        "output_tokens": 50,
        "model": "tinyllama"
    }
    
    allocation = allocator.allocate(task)
    assert allocation["hardware"] == "apple_m1_pro"
    assert allocation["input_tokens"] == 500
    assert allocation["output_tokens"] == 50
    assert allocation["model"] == "tinyllama"

def test_allocate_medium_task(allocator):
    """测试中等任务分配。"""
    task = {
        "input_tokens": 1500,
        "output_tokens": 150,
        "model": "tinyllama"
    }
    
    allocation = allocator.allocate(task)
    assert allocation["hardware"] == "nvidia_rtx4050"
    assert allocation["input_tokens"] == 1500
    assert allocation["output_tokens"] == 150
    assert allocation["model"] == "tinyllama"

def test_allocate_large_task(allocator):
    """测试大任务分配。"""
    task = {
        "input_tokens": 2500,
        "output_tokens": 250,
        "model": "tinyllama"
    }
    
    allocation = allocator.allocate(task)
    assert allocation["hardware"] == "nvidia_rtx4090"
    assert allocation["input_tokens"] == 2500
    assert allocation["output_tokens"] == 250
    assert allocation["model"] == "tinyllama"

def test_allocate_multiple_tasks(allocator):
    """测试多任务分配。"""
    tasks = [
        {
            "input_tokens": 500,
            "output_tokens": 50,
            "model": "tinyllama"
        },
        {
            "input_tokens": 1500,
            "output_tokens": 150,
            "model": "tinyllama"
        },
        {
            "input_tokens": 2500,
            "output_tokens": 250,
            "model": "tinyllama"
        }
    ]
    
    allocations = allocator.allocate_multiple(tasks)
    assert len(allocations) == 3
    assert allocations[0]["hardware"] == "apple_m1_pro"
    assert allocations[1]["hardware"] == "nvidia_rtx4050"
    assert allocations[2]["hardware"] == "nvidia_rtx4090"
    
    for i, task in enumerate(tasks):
        assert allocations[i]["input_tokens"] == task["input_tokens"]
        assert allocations[i]["output_tokens"] == task["output_tokens"]
        assert allocations[i]["model"] == task["model"]

def test_allocate_boundary_conditions(allocator):
    """测试任务分配边界条件。"""
    boundary_tasks = [
        {
            "input_tokens": 999,
            "output_tokens": 99,
            "model": "tinyllama"
        },
        {
            "input_tokens": 1000,
            "output_tokens": 100,
            "model": "tinyllama"
        },
        {
            "input_tokens": 1001,
            "output_tokens": 101,
            "model": "tinyllama"
        }
    ]
    
    allocations = allocator.allocate_multiple(boundary_tasks)
    assert len(allocations) == 3
    assert allocations[0]["hardware"] == "apple_m1_pro"
    assert allocations[1]["hardware"] == "nvidia_rtx4050"
    assert allocations[2]["hardware"] == "nvidia_rtx4090"
    
    for i, task in enumerate(boundary_tasks):
        assert allocations[i]["input_tokens"] == task["input_tokens"]
        assert allocations[i]["output_tokens"] == task["output_tokens"]
        assert allocations[i]["model"] == task["model"]

def test_allocate_edge_cases(allocator):
    """测试任务分配边缘情况。"""
    # 测试极小任务
    tiny_task = {
        "input_tokens": 1,
        "output_tokens": 1,
        "model": "tinyllama"
    }
    
    allocation = allocator.allocate(tiny_task)
    assert allocation["hardware"] == "apple_m1_pro"
    assert allocation["input_tokens"] == 1
    assert allocation["output_tokens"] == 1
    assert allocation["model"] == "tinyllama"
    
    # 测试极大任务
    huge_task = {
        "input_tokens": 1000000,
        "output_tokens": 100000,
        "model": "tinyllama"
    }
    
    allocation = allocator.allocate(huge_task)
    assert allocation["hardware"] == "nvidia_rtx4090"
    assert allocation["input_tokens"] == 1000000
    assert allocation["output_tokens"] == 100000
    assert allocation["model"] == "tinyllama"

def test_error_handling():
    """测试错误处理。"""
    # 测试无效的硬件配置
    invalid_hardware_configs = [
        {},  # 空配置
        {"invalid": {}},  # 无效的硬件类型
        {"apple_m1_pro": {"device_type": "m1_pro"}},  # 缺少必要参数
        {"apple_m1_pro": {"idle_power": 10.0}},  # 缺少必要参数
        {"apple_m1_pro": {"sample_interval": 200}},  # 缺少必要参数
        {"apple_m1_pro": {"device_type": "m1_pro", "idle_power": -1.0}},  # 负的 idle_power
        {"apple_m1_pro": {"device_type": "m1_pro", "sample_interval": 0}}  # 零 sample_interval
    ]
    
    for config in invalid_hardware_configs:
        with pytest.raises((ValueError, TypeError)):
            TaskAllocator(config, MODEL_CONFIG)
    
    # 测试无效的模型配置
    invalid_model_configs = [
        {},  # 空配置
        {"models": {}},  # 空模型列表
        {"models": {"invalid": {}}},  # 无效的模型配置
        {"models": {"tinyllama": {"model_name": "tinyllama"}}}  # 缺少必要参数
    ]
    
    for config in invalid_model_configs:
        with pytest.raises((ValueError, TypeError)):
            TaskAllocator(HARDWARE_CONFIG, config)
    
    # 测试无效的任务
    allocator = TaskAllocator(HARDWARE_CONFIG, MODEL_CONFIG)
    invalid_tasks = [
        None,  # 空任务
        {},  # 缺少必要参数
        {"input_tokens": 100},  # 缺少必要参数
        {"output_tokens": 50},  # 缺少必要参数
        {"model": "tinyllama"},  # 缺少必要参数
        {"input_tokens": -1, "output_tokens": 50, "model": "tinyllama"},  # 负的 input_tokens
        {"input_tokens": 100, "output_tokens": -1, "model": "tinyllama"},  # 负的 output_tokens
        {"input_tokens": 100, "output_tokens": 50, "model": "invalid"}  # 无效的模型
    ]
    
    for task in invalid_tasks:
        with pytest.raises((ValueError, TypeError, KeyError)):
            allocator.allocate(task)

def test_config_validation():
    """测试配置验证。"""
    # 测试缺少必要参数
    with pytest.raises(ValueError):
        TaskAllocator({}, {})
    
    # 测试无效的硬件配置类型
    with pytest.raises(TypeError):
        TaskAllocator("invalid", MODEL_CONFIG)
    
    # 测试无效的模型配置类型
    with pytest.raises(TypeError):
        TaskAllocator(HARDWARE_CONFIG, "invalid")
    
    # 测试无效的硬件类型
    invalid_config = HARDWARE_CONFIG.copy()
    invalid_config["apple_m1_pro"]["device_type"] = 123
    with pytest.raises(TypeError):
        TaskAllocator(invalid_config, MODEL_CONFIG)
    
    # 测试无效的模型类型
    invalid_config = MODEL_CONFIG.copy()
    invalid_config["models"]["tinyllama"]["model_name"] = 123
    with pytest.raises(TypeError):
        TaskAllocator(HARDWARE_CONFIG, invalid_config) 