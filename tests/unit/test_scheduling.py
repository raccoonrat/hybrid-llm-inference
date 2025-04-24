# hybrid-llm-inference/tests/unit/test_scheduling.py
import pytest
import os
from scheduling.token_based_scheduler import TokenBasedScheduler
from scheduling.task_allocator import TaskAllocator
from typing import Dict, Any, List

@pytest.fixture
def scheduler_config():
    return {
        "hardware_map": {
            "m1_pro": "apple_m1_pro",
            "a100": "nvidia_a100"
        }
    }

@pytest.fixture
def hardware_config():
    """返回硬件配置。"""
    return {
        "m1_pro": {
            "device_type": "m1_pro",
            "idle_power": 10.0,
            "sample_interval": 200
        },
        "rtx4050": {
            "device_type": "rtx4050",
            "idle_power": 15.0,
            "sample_interval": 200
        },
        "a100": {
            "device_type": "a100",
            "idle_power": 20.0,
            "sample_interval": 200
        }
    }

@pytest.fixture
def model_config():
    """返回模型配置。"""
    return {
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

# 测试配置
TEST_CONFIG = {
    "thresholds": {
        "T_in": 1000,
        "T_out": 100
    },
    "hardware_map": {
        "small": "apple_m1_pro",
        "medium": "nvidia_rtx4050",
        "large": "nvidia_rtx4090"
    }
}

@pytest.fixture
def scheduler():
    """创建 TokenBasedScheduler 实例的 fixture。"""
    return TokenBasedScheduler(TEST_CONFIG)

def test_token_based_scheduler(scheduler):
    """测试基于令牌的调度器。"""
    # 测试小任务
    small_task = {
        "input_tokens": 500,
        "output_tokens": 50
    }
    allocation = scheduler.schedule([small_task])[0]
    assert allocation["hardware"] == "apple_m1_pro"
    assert allocation["input_tokens"] == 500
    assert allocation["output_tokens"] == 50
    
    # 测试中等任务
    medium_task = {
        "input_tokens": 1500,
        "output_tokens": 150
    }
    allocation = scheduler.schedule([medium_task])[0]
    assert allocation["hardware"] == "nvidia_rtx4050"
    assert allocation["input_tokens"] == 1500
    assert allocation["output_tokens"] == 150
    
    # 测试大任务
    large_task = {
        "input_tokens": 2500,
        "output_tokens": 250
    }
    allocation = scheduler.schedule([large_task])[0]
    assert allocation["hardware"] == "nvidia_rtx4090"
    assert allocation["input_tokens"] == 2500
    assert allocation["output_tokens"] == 250

def test_token_based_scheduler_empty_data(scheduler):
    """测试基于令牌的调度器处理空数据。"""
    # 测试空任务列表
    allocations = scheduler.schedule([])
    assert len(allocations) == 0
    
    # 测试无效任务
    with pytest.raises((ValueError, TypeError, KeyError)):
        scheduler.schedule([None])
    
    with pytest.raises((ValueError, TypeError, KeyError)):
        scheduler.schedule([{}])
    
    with pytest.raises((ValueError, TypeError, KeyError)):
        scheduler.schedule([{"input_tokens": -1, "output_tokens": 50}])
    
    with pytest.raises((ValueError, TypeError, KeyError)):
        scheduler.schedule([{"input_tokens": 100, "output_tokens": -1}])

def test_token_based_scheduler_boundary_conditions(scheduler):
    """测试基于令牌的调度器边界条件。"""
    # 测试边界值任务
    boundary_tasks = [
        {"input_tokens": 999, "output_tokens": 99},  # 小任务边界
        {"input_tokens": 1000, "output_tokens": 100},  # 中等任务边界
        {"input_tokens": 1001, "output_tokens": 101}  # 大任务边界
    ]
    
    allocations = scheduler.schedule(boundary_tasks)
    assert len(allocations) == 3
    assert allocations[0]["hardware"] == "apple_m1_pro"
    assert allocations[1]["hardware"] == "nvidia_rtx4050"
    assert allocations[2]["hardware"] == "nvidia_rtx4090"
    
    for i, task in enumerate(boundary_tasks):
        assert allocations[i]["input_tokens"] == task["input_tokens"]
        assert allocations[i]["output_tokens"] == task["output_tokens"]

def test_token_based_scheduler_edge_cases(scheduler):
    """测试基于令牌的调度器边缘情况。"""
    # 测试极小任务
    tiny_task = {
        "input_tokens": 1,
        "output_tokens": 1
    }
    
    allocation = scheduler.schedule([tiny_task])[0]
    assert allocation["hardware"] == "apple_m1_pro"
    assert allocation["input_tokens"] == 1
    assert allocation["output_tokens"] == 1
    
    # 测试极大任务
    huge_task = {
        "input_tokens": 1000000,
        "output_tokens": 100000
    }
    
    allocation = scheduler.schedule([huge_task])[0]
    assert allocation["hardware"] == "nvidia_rtx4090"
    assert allocation["input_tokens"] == 1000000
    assert allocation["output_tokens"] == 100000

def test_token_based_scheduler_multiple_tasks(scheduler):
    """测试基于令牌的调度器多任务调度。"""
    tasks = [
        {"input_tokens": 500, "output_tokens": 50},
        {"input_tokens": 1500, "output_tokens": 150},
        {"input_tokens": 2500, "output_tokens": 250}
    ]
    
    allocations = scheduler.schedule(tasks)
    assert len(allocations) == 3
    assert allocations[0]["hardware"] == "apple_m1_pro"
    assert allocations[1]["hardware"] == "nvidia_rtx4050"
    assert allocations[2]["hardware"] == "nvidia_rtx4090"
    
    for i, task in enumerate(tasks):
        assert allocations[i]["input_tokens"] == task["input_tokens"]
        assert allocations[i]["output_tokens"] == task["output_tokens"]

def test_task_allocator(hardware_config, model_config):
    """测试任务分配器。"""
    allocator = TaskAllocator(
        hardware_config=hardware_config,
        model_config=model_config
    )
    assert allocator is not None
    
def test_task_allocator_invalid_hardware(hardware_config, model_config):
    """测试无效硬件配置。"""
    invalid_config = hardware_config.copy()
    invalid_config["invalid"] = {
        "device_type": "invalid",
        "idle_power": 10.0,
        "sample_interval": 200
    }
    
    with pytest.raises(ValueError):
        TaskAllocator(
            hardware_config=invalid_config,
            model_config=model_config
        )

# 测试配置
HARDWARE_CONFIG = {
    "nvidia_rtx4050": {
        "device_type": "rtx4050",
        "idle_power": 15.0,
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

def test_task_allocator(allocator):
    """测试任务分配器。"""
    # 验证初始化
    assert allocator.hardware_config == HARDWARE_CONFIG
    assert allocator.model_config == MODEL_CONFIG
    assert len(allocator.available_hardware) == 1
    assert "nvidia_rtx4050" in allocator.available_hardware
    
    # 测试任务分配
    task = {
        "input_tokens": 100,
        "output_tokens": 50,
        "model": "tinyllama"
    }
    
    allocation = allocator.allocate(task)
    assert allocation["hardware"] == "nvidia_rtx4050"
    assert allocation["input_tokens"] == 100
    assert allocation["output_tokens"] == 50
    assert allocation["model"] == "tinyllama"

def test_task_allocator_invalid_hardware():
    """测试无效硬件配置。"""
    # 测试空配置
    with pytest.raises(ValueError):
        TaskAllocator({}, MODEL_CONFIG)
    
    # 测试无效的硬件类型
    invalid_config = HARDWARE_CONFIG.copy()
    invalid_config["nvidia_rtx4050"]["device_type"] = 123
    with pytest.raises(TypeError):
        TaskAllocator(invalid_config, MODEL_CONFIG)
    
    # 测试无效的 idle_power
    invalid_config = HARDWARE_CONFIG.copy()
    invalid_config["nvidia_rtx4050"]["idle_power"] = -1.0
    with pytest.raises(ValueError):
        TaskAllocator(invalid_config, MODEL_CONFIG)
    
    # 测试无效的 sample_interval
    invalid_config = HARDWARE_CONFIG.copy()
    invalid_config["nvidia_rtx4050"]["sample_interval"] = 0
    with pytest.raises(ValueError):
        TaskAllocator(invalid_config, MODEL_CONFIG)

def test_task_allocator_invalid_model():
    """测试无效模型配置。"""
    # 测试空配置
    with pytest.raises(ValueError):
        TaskAllocator(HARDWARE_CONFIG, {})
    
    # 测试无效的模型类型
    invalid_config = MODEL_CONFIG.copy()
    invalid_config["models"]["tinyllama"]["model_name"] = 123
    with pytest.raises(TypeError):
        TaskAllocator(HARDWARE_CONFIG, invalid_config)
    
    # 测试无效的 batch_size
    invalid_config = MODEL_CONFIG.copy()
    invalid_config["models"]["tinyllama"]["batch_size"] = -1
    with pytest.raises(ValueError):
        TaskAllocator(HARDWARE_CONFIG, invalid_config)
    
    # 测试无效的 max_length
    invalid_config = MODEL_CONFIG.copy()
    invalid_config["models"]["tinyllama"]["max_length"] = 0
    with pytest.raises(ValueError):
        TaskAllocator(HARDWARE_CONFIG, invalid_config)

def test_task_allocator_invalid_tasks(allocator):
    """测试无效任务。"""
    # 测试无效的输入令牌
    invalid_tasks = [
        {"input_tokens": -1, "output_tokens": 50, "model": "tinyllama"},
        {"input_tokens": "100", "output_tokens": 50, "model": "tinyllama"},
        {"input_tokens": None, "output_tokens": 50, "model": "tinyllama"}
    ]
    
    for task in invalid_tasks:
        with pytest.raises((ValueError, TypeError)):
            allocator.allocate(task)
    
    # 测试无效的输出令牌
    invalid_tasks = [
        {"input_tokens": 100, "output_tokens": -1, "model": "tinyllama"},
        {"input_tokens": 100, "output_tokens": "50", "model": "tinyllama"},
        {"input_tokens": 100, "output_tokens": None, "model": "tinyllama"}
    ]
    
    for task in invalid_tasks:
        with pytest.raises((ValueError, TypeError)):
            allocator.allocate(task)
    
    # 测试无效的模型
    invalid_tasks = [
        {"input_tokens": 100, "output_tokens": 50, "model": 123},
        {"input_tokens": 100, "output_tokens": 50, "model": "invalid"},
        {"input_tokens": 100, "output_tokens": 50, "model": None}
    ]
    
    for task in invalid_tasks:
        with pytest.raises((ValueError, TypeError, KeyError)):
            allocator.allocate(task)

def test_task_allocator_multiple_tasks(allocator):
    """测试多任务分配。"""
    tasks = [
        {"input_tokens": 100, "output_tokens": 50, "model": "tinyllama"},
        {"input_tokens": 200, "output_tokens": 100, "model": "tinyllama"},
        {"input_tokens": 300, "output_tokens": 150, "model": "tinyllama"}
    ]
    
    allocations = allocator.allocate_multiple(tasks)
    assert len(allocations) == len(tasks)
    
    for i, allocation in enumerate(allocations):
        assert allocation["hardware"] == "nvidia_rtx4050"
        assert allocation["input_tokens"] == tasks[i]["input_tokens"]
        assert allocation["output_tokens"] == tasks[i]["output_tokens"]
        assert allocation["model"] == tasks[i]["model"]

def test_task_allocator_edge_cases(allocator):
    """测试边缘情况。"""
    # 测试极小任务
    tiny_task = {
        "input_tokens": 1,
        "output_tokens": 1,
        "model": "tinyllama"
    }
    
    allocation = allocator.allocate(tiny_task)
    assert allocation["hardware"] == "nvidia_rtx4050"
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
    assert allocation["hardware"] == "nvidia_rtx4050"
    assert allocation["input_tokens"] == 1000000
    assert allocation["output_tokens"] == 100000
    assert allocation["model"] == "tinyllama"
