"""基于令牌的调度器测试模块。"""

import pytest
from src.scheduling.token_based_scheduler import TokenBasedScheduler
from typing import Dict, Any, List

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

SCHEDULER_CONFIG = {
    "thresholds": {
        "T_in": 1000,
        "T_out": 100
    },
    "hardware_map": {
        "small": "nvidia_rtx4050",
        "medium": "nvidia_rtx4050",
        "large": "nvidia_rtx4050"
    }
}

@pytest.fixture
def scheduler():
    """创建 TokenBasedScheduler 实例的 fixture。"""
    return TokenBasedScheduler(HARDWARE_CONFIG, MODEL_CONFIG, SCHEDULER_CONFIG)

def test_scheduler_initialization(scheduler):
    """测试调度器初始化。"""
    # 验证初始化
    assert scheduler.hardware_config == HARDWARE_CONFIG
    assert scheduler.model_config == MODEL_CONFIG
    assert scheduler.scheduler_config == SCHEDULER_CONFIG
    assert len(scheduler.available_hardware) == 1
    assert "nvidia_rtx4050" in scheduler.available_hardware

def test_schedule_tasks(scheduler):
    """测试任务调度。"""
    # 测试小任务
    small_task = {
        "input_tokens": 500,
        "output_tokens": 50,
        "model": "tinyllama"
    }
    allocation = scheduler.schedule([small_task])[0]
    assert allocation["hardware"] == "nvidia_rtx4050"
    assert allocation["input_tokens"] == 500
    assert allocation["output_tokens"] == 50
    assert allocation["model"] == "tinyllama"
    
    # 测试中等任务
    medium_task = {
        "input_tokens": 1500,
        "output_tokens": 150,
        "model": "tinyllama"
    }
    allocation = scheduler.schedule([medium_task])[0]
    assert allocation["hardware"] == "nvidia_rtx4050"
    assert allocation["input_tokens"] == 1500
    assert allocation["output_tokens"] == 150
    assert allocation["model"] == "tinyllama"
    
    # 测试大任务
    large_task = {
        "input_tokens": 2500,
        "output_tokens": 250,
        "model": "tinyllama"
    }
    allocation = scheduler.schedule([large_task])[0]
    assert allocation["hardware"] == "nvidia_rtx4050"
    assert allocation["input_tokens"] == 2500
    assert allocation["output_tokens"] == 250
    assert allocation["model"] == "tinyllama"

def test_invalid_config():
    """测试无效配置。"""
    # 测试空配置
    with pytest.raises(ValueError):
        TokenBasedScheduler({}, MODEL_CONFIG, SCHEDULER_CONFIG)
    
    with pytest.raises(ValueError):
        TokenBasedScheduler(HARDWARE_CONFIG, {}, SCHEDULER_CONFIG)
    
    with pytest.raises(ValueError):
        TokenBasedScheduler(HARDWARE_CONFIG, MODEL_CONFIG, {})
    
    # 测试无效的阈值
    invalid_config = SCHEDULER_CONFIG.copy()
    invalid_config["thresholds"]["T_in"] = -1
    with pytest.raises(ValueError):
        TokenBasedScheduler(HARDWARE_CONFIG, MODEL_CONFIG, invalid_config)
    
    # 测试无效的硬件映射
    invalid_config = SCHEDULER_CONFIG.copy()
    invalid_config["hardware_map"]["small"] = "invalid"
    with pytest.raises(ValueError):
        TokenBasedScheduler(HARDWARE_CONFIG, MODEL_CONFIG, invalid_config)

def test_empty_task_list(scheduler):
    """测试空任务列表。"""
    allocations = scheduler.schedule([])
    assert len(allocations) == 0

def test_invalid_tasks(scheduler):
    """测试无效任务。"""
    # 测试无效的输入令牌
    invalid_tasks = [
        {"input_tokens": -1, "output_tokens": 50, "model": "tinyllama"},
        {"input_tokens": "100", "output_tokens": 50, "model": "tinyllama"},
        {"input_tokens": None, "output_tokens": 50, "model": "tinyllama"}
    ]
    
    for task in invalid_tasks:
        with pytest.raises((ValueError, TypeError)):
            scheduler.schedule([task])
    
    # 测试无效的输出令牌
    invalid_tasks = [
        {"input_tokens": 100, "output_tokens": -1, "model": "tinyllama"},
        {"input_tokens": 100, "output_tokens": "50", "model": "tinyllama"},
        {"input_tokens": 100, "output_tokens": None, "model": "tinyllama"}
    ]
    
    for task in invalid_tasks:
        with pytest.raises((ValueError, TypeError)):
            scheduler.schedule([task])
    
    # 测试无效的模型
    invalid_tasks = [
        {"input_tokens": 100, "output_tokens": 50, "model": 123},
        {"input_tokens": 100, "output_tokens": 50, "model": "invalid"},
        {"input_tokens": 100, "output_tokens": 50, "model": None}
    ]
    
    for task in invalid_tasks:
        with pytest.raises((ValueError, TypeError, KeyError)):
            scheduler.schedule([task])

def test_edge_cases(scheduler):
    """测试边缘情况。"""
    # 测试极小任务
    tiny_task = {
        "input_tokens": 1,
        "output_tokens": 1,
        "model": "tinyllama"
    }
    
    allocation = scheduler.schedule([tiny_task])[0]
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
    
    allocation = scheduler.schedule([huge_task])[0]
    assert allocation["hardware"] == "nvidia_rtx4050"
    assert allocation["input_tokens"] == 1000000
    assert allocation["output_tokens"] == 100000
    assert allocation["model"] == "tinyllama"

def test_performance_metrics(scheduler):
    """测试性能指标。"""
    tasks = [
        {"input_tokens": 100, "output_tokens": 50, "model": "tinyllama"},
        {"input_tokens": 200, "output_tokens": 100, "model": "tinyllama"},
        {"input_tokens": 300, "output_tokens": 150, "model": "tinyllama"}
    ]
    
    allocations = scheduler.schedule(tasks)
    assert len(allocations) == len(tasks)
    
    for i, allocation in enumerate(allocations):
        assert allocation["hardware"] == "nvidia_rtx4050"
        assert allocation["input_tokens"] == tasks[i]["input_tokens"]
        assert allocation["output_tokens"] == tasks[i]["output_tokens"]
        assert allocation["model"] == tasks[i]["model"] 