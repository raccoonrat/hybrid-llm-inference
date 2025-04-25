"""基于令牌的调度器测试模块。"""

import pytest
from src.scheduling.token_based_scheduler import TokenBasedScheduler
from typing import Dict, Any, List

# 测试配置
TEST_CONFIG = {
    "token_threshold": 100,
    "hardware_config": {
        "nvidia_rtx4050": {
            "device_type": "rtx4050",
            "idle_power": 15.0,
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
def scheduler():
    """创建 TokenBasedScheduler 实例的 fixture。"""
    return TokenBasedScheduler(TEST_CONFIG)

def test_scheduler_initialization(scheduler):
    """测试调度器初始化。"""
    # 验证初始化
    assert scheduler.token_threshold == TEST_CONFIG["token_threshold"]
    assert scheduler.hardware_config == TEST_CONFIG["hardware_config"]
    assert scheduler.model_config == TEST_CONFIG["model_config"]
    assert scheduler.initialized

def test_schedule_tasks(scheduler):
    """测试任务调度。"""
    # 测试小任务
    small_task = {
        "tokens": 50
    }
    allocation = scheduler.schedule([small_task])[0]
    assert allocation["hardware"] == "apple_m1_pro"
    assert allocation["tokens"] == 50
    assert allocation["model"] == "tinyllama"
    
    # 测试大任务
    large_task = {
        "tokens": 150
    }
    allocation = scheduler.schedule([large_task])[0]
    assert allocation["hardware"] == "rtx4050"
    assert allocation["tokens"] == 150
    assert allocation["model"] == "llama3"

def test_invalid_config():
    """测试无效配置。"""
    # 测试空配置
    with pytest.raises(ValueError):
        TokenBasedScheduler({})
    
    # 测试无效的令牌阈值
    invalid_config = TEST_CONFIG.copy()
    invalid_config["token_threshold"] = -1
    with pytest.raises(ValueError):
        TokenBasedScheduler(invalid_config)
    
    # 测试无效的硬件配置
    invalid_config = TEST_CONFIG.copy()
    invalid_config["hardware_config"] = "invalid"
    with pytest.raises(ValueError):
        TokenBasedScheduler(invalid_config)
    
    # 测试无效的模型配置
    invalid_config = TEST_CONFIG.copy()
    invalid_config["model_config"] = "invalid"
    with pytest.raises(ValueError):
        TokenBasedScheduler(invalid_config)

def test_empty_task_list(scheduler):
    """测试空任务列表。"""
    allocations = scheduler.schedule([])
    assert len(allocations) == 0

def test_invalid_tasks(scheduler):
    """测试无效任务。"""
    # 测试无效的令牌
    invalid_tasks = [
        {"tokens": -1},
        {"tokens": "100"},
        {"tokens": None},
        None
    ]
    
    for task in invalid_tasks:
        with pytest.raises((ValueError, TypeError, AttributeError)):
            scheduler.schedule([task])

def test_edge_cases(scheduler):
    """测试边缘情况。"""
    # 测试极小任务
    tiny_task = {
        "tokens": 1
    }
    
    allocation = scheduler.schedule([tiny_task])[0]
    assert allocation["hardware"] == "apple_m1_pro"
    assert allocation["tokens"] == 1
    assert allocation["model"] == "tinyllama"
    
    # 测试极大任务
    huge_task = {
        "tokens": 1000000
    }
    
    allocation = scheduler.schedule([huge_task])[0]
    assert allocation["hardware"] == "rtx4050"
    assert allocation["tokens"] == 1000000
    assert allocation["model"] == "llama3"

def test_performance_metrics(scheduler):
    """测试性能指标。"""
    tasks = [
        {"tokens": 50},
        {"tokens": 150},
        {"tokens": 250}
    ]
    
    allocations = scheduler.schedule(tasks)
    assert len(allocations) == len(tasks)
    
    for i, allocation in enumerate(allocations):
        assert allocation["tokens"] == tasks[i]["tokens"]
        if tasks[i]["tokens"] <= scheduler.token_threshold:
            assert allocation["hardware"] == "apple_m1_pro"
            assert allocation["model"] == "tinyllama"
        else:
            assert allocation["hardware"] == "rtx4050"
            assert allocation["model"] == "llama3" 