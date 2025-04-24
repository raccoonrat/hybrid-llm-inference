"""基于令牌的调度器测试模块。"""

import pytest
from src.scheduling.token_based_scheduler import TokenBasedScheduler
from typing import Dict, Any, List

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

def test_initialization(scheduler):
    """测试初始化。"""
    assert scheduler.config == TEST_CONFIG
    assert scheduler.thresholds["T_in"] == 1000
    assert scheduler.thresholds["T_out"] == 100
    assert scheduler.hardware_map["small"] == "apple_m1_pro"
    assert scheduler.hardware_map["medium"] == "nvidia_rtx4050"
    assert scheduler.hardware_map["large"] == "nvidia_rtx4090"

def test_schedule_small_task(scheduler):
    """测试小任务调度。"""
    tasks = [{
        "input_tokens": 500,
        "output_tokens": 50
    }]
    
    allocations = scheduler.schedule(tasks)
    assert len(allocations) == 1
    assert allocations[0]["hardware"] == "apple_m1_pro"
    assert allocations[0]["input_tokens"] == 500
    assert allocations[0]["output_tokens"] == 50

def test_schedule_medium_task(scheduler):
    """测试中等任务调度。"""
    tasks = [{
        "input_tokens": 1500,
        "output_tokens": 150
    }]
    
    allocations = scheduler.schedule(tasks)
    assert len(allocations) == 1
    assert allocations[0]["hardware"] == "nvidia_rtx4050"
    assert allocations[0]["input_tokens"] == 1500
    assert allocations[0]["output_tokens"] == 150

def test_schedule_large_task(scheduler):
    """测试大任务调度。"""
    tasks = [{
        "input_tokens": 2500,
        "output_tokens": 250
    }]
    
    allocations = scheduler.schedule(tasks)
    assert len(allocations) == 1
    assert allocations[0]["hardware"] == "nvidia_rtx4090"
    assert allocations[0]["input_tokens"] == 2500
    assert allocations[0]["output_tokens"] == 250

def test_schedule_multiple_tasks(scheduler):
    """测试多任务调度。"""
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

def test_schedule_boundary_conditions(scheduler):
    """测试调度边界条件。"""
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

def test_schedule_edge_cases(scheduler):
    """测试调度边缘情况。"""
    # 测试极小任务
    tiny_task = {
        "input_tokens": 1,
        "output_tokens": 1
    }
    
    allocations = scheduler.schedule([tiny_task])
    assert len(allocations) == 1
    assert allocations[0]["hardware"] == "apple_m1_pro"
    assert allocations[0]["input_tokens"] == 1
    assert allocations[0]["output_tokens"] == 1
    
    # 测试极大任务
    huge_task = {
        "input_tokens": 1000000,
        "output_tokens": 100000
    }
    
    allocations = scheduler.schedule([huge_task])
    assert len(allocations) == 1
    assert allocations[0]["hardware"] == "nvidia_rtx4090"
    assert allocations[0]["input_tokens"] == 1000000
    assert allocations[0]["output_tokens"] == 100000

def test_error_handling():
    """测试错误处理。"""
    # 测试无效配置
    invalid_configs = [
        {},  # 空配置
        {"thresholds": {}},  # 缺少阈值
        {"hardware_map": {}},  # 缺少硬件映射
        {"thresholds": {"T_in": 1000}, "hardware_map": {}},  # 缺少 T_out
        {"thresholds": {"T_out": 100}, "hardware_map": {}},  # 缺少 T_in
        {"thresholds": {"T_in": 1000, "T_out": 100}, "hardware_map": {"small": "apple_m1_pro"}},  # 缺少 medium
        {"thresholds": {"T_in": 1000, "T_out": 100}, "hardware_map": {"medium": "nvidia_rtx4050"}},  # 缺少 small
        {"thresholds": {"T_in": 1000, "T_out": 100}, "hardware_map": {"small": "apple_m1_pro", "medium": "nvidia_rtx4050"}}  # 缺少 large
    ]
    
    for config in invalid_configs:
        with pytest.raises(ValueError):
            TokenBasedScheduler(config)
    
    # 测试无效的阈值
    invalid_thresholds = [
        {"T_in": -1, "T_out": 100},  # 负的 T_in
        {"T_in": 1000, "T_out": -1},  # 负的 T_out
        {"T_in": 0, "T_out": 100},  # 零 T_in
        {"T_in": 1000, "T_out": 0}  # 零 T_out
    ]
    
    for thresholds in invalid_thresholds:
        config = TEST_CONFIG.copy()
        config["thresholds"] = thresholds
        with pytest.raises(ValueError):
            TokenBasedScheduler(config)
    
    # 测试无效的任务
    scheduler = TokenBasedScheduler(TEST_CONFIG)
    invalid_tasks = [
        {"input_tokens": -1, "output_tokens": 50},
        {"input_tokens": 100, "output_tokens": -1},
        {"input_tokens": "500", "output_tokens": 50},
        {"input_tokens": 500, "output_tokens": "50"},
        {},
        None
    ]
    
    for task in invalid_tasks:
        with pytest.raises((ValueError, TypeError, KeyError)):
            scheduler.schedule([task])

def test_config_validation():
    """测试配置验证。"""
    # 测试缺少必要参数
    with pytest.raises(ValueError):
        TokenBasedScheduler({})
    
    # 测试无效的阈值类型
    invalid_config = TEST_CONFIG.copy()
    invalid_config["thresholds"]["T_in"] = "1000"
    with pytest.raises(TypeError):
        TokenBasedScheduler(invalid_config)
    
    # 测试无效的硬件映射类型
    invalid_config = TEST_CONFIG.copy()
    invalid_config["hardware_map"] = "invalid"
    with pytest.raises(TypeError):
        TokenBasedScheduler(invalid_config)
    
    # 测试无效的硬件类型
    invalid_config = TEST_CONFIG.copy()
    invalid_config["hardware_map"]["small"] = 123
    with pytest.raises(TypeError):
        TokenBasedScheduler(invalid_config) 