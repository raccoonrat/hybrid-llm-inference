"""任务分配器测试模块。"""

import pytest
from src.scheduling.task_allocator import TaskAllocator
from typing import Dict, Any

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
    },
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
    }
}

@pytest.fixture
def allocator():
    """创建 TaskAllocator 实例的 fixture。"""
    return TaskAllocator(TEST_CONFIG)

def test_initialization(allocator):
    """测试初始化。"""
    assert allocator.config == TEST_CONFIG
    assert len(allocator.profilers) == 3
    assert "apple_m1_pro" in allocator.profilers
    assert "nvidia_rtx4050" in allocator.profilers
    assert "nvidia_rtx4090" in allocator.profilers

def test_allocate_small_task(allocator):
    """测试小任务分配。"""
    tasks = [{
        "input_tokens": 500,
        "output_tokens": 50
    }]
    
    allocations = allocator.allocate(tasks)
    assert len(allocations) == 1
    assert allocations[0]["hardware"] == "apple_m1_pro"
    assert allocations[0]["input_tokens"] == 500
    assert allocations[0]["output_tokens"] == 50
    assert allocations[0]["profiler"] is not None

def test_allocate_medium_task(allocator):
    """测试中等任务分配。"""
    tasks = [{
        "input_tokens": 1500,
        "output_tokens": 150
    }]
    
    allocations = allocator.allocate(tasks)
    assert len(allocations) == 1
    assert allocations[0]["hardware"] == "nvidia_rtx4050"
    assert allocations[0]["input_tokens"] == 1500
    assert allocations[0]["output_tokens"] == 150
    assert allocations[0]["profiler"] is not None

def test_allocate_large_task(allocator):
    """测试大任务分配。"""
    tasks = [{
        "input_tokens": 2500,
        "output_tokens": 250
    }]
    
    allocations = allocator.allocate(tasks)
    assert len(allocations) == 1
    assert allocations[0]["hardware"] == "nvidia_rtx4090"
    assert allocations[0]["input_tokens"] == 2500
    assert allocations[0]["output_tokens"] == 250
    assert allocations[0]["profiler"] is not None

def test_allocate_multiple_tasks(allocator):
    """测试多任务分配。"""
    tasks = [
        {"input_tokens": 500, "output_tokens": 50},
        {"input_tokens": 1500, "output_tokens": 150},
        {"input_tokens": 2500, "output_tokens": 250}
    ]
    
    allocations = allocator.allocate(tasks)
    assert len(allocations) == 3
    assert allocations[0]["hardware"] == "apple_m1_pro"
    assert allocations[1]["hardware"] == "nvidia_rtx4050"
    assert allocations[2]["hardware"] == "nvidia_rtx4090"
    
    for allocation in allocations:
        assert allocation["profiler"] is not None

def test_invalid_config():
    """测试无效配置。"""
    # 测试缺少阈值配置
    with pytest.raises(ValueError):
        TaskAllocator({})
    
    # 测试缺少 T_in
    invalid_config = TEST_CONFIG.copy()
    del invalid_config["thresholds"]["T_in"]
    with pytest.raises(ValueError):
        TaskAllocator(invalid_config)
    
    # 测试缺少 T_out
    invalid_config = TEST_CONFIG.copy()
    del invalid_config["thresholds"]["T_out"]
    with pytest.raises(ValueError):
        TaskAllocator(invalid_config)
    
    # 测试无效的 T_in 值
    invalid_config = TEST_CONFIG.copy()
    invalid_config["thresholds"]["T_in"] = -1
    with pytest.raises(ValueError):
        TaskAllocator(invalid_config)
    
    # 测试无效的 T_out 值
    invalid_config = TEST_CONFIG.copy()
    invalid_config["thresholds"]["T_out"] = 0
    with pytest.raises(ValueError):
        TaskAllocator(invalid_config)
    
    # 测试缺少硬件映射
    invalid_config = TEST_CONFIG.copy()
    del invalid_config["hardware_map"]
    with pytest.raises(ValueError):
        TaskAllocator(invalid_config)
    
    # 测试缺少 small 硬件类型
    invalid_config = TEST_CONFIG.copy()
    del invalid_config["hardware_map"]["small"]
    with pytest.raises(ValueError):
        TaskAllocator(invalid_config)
    
    # 测试缺少 medium 硬件类型
    invalid_config = TEST_CONFIG.copy()
    del invalid_config["hardware_map"]["medium"]
    with pytest.raises(ValueError):
        TaskAllocator(invalid_config)
    
    # 测试缺少 large 硬件类型
    invalid_config = TEST_CONFIG.copy()
    del invalid_config["hardware_map"]["large"]
    with pytest.raises(ValueError):
        TaskAllocator(invalid_config)
    
    # 测试缺少硬件配置
    invalid_config = TEST_CONFIG.copy()
    del invalid_config["hardware_config"]
    with pytest.raises(ValueError):
        TaskAllocator(invalid_config)
    
    # 测试缺少硬件类型配置
    invalid_config = TEST_CONFIG.copy()
    del invalid_config["hardware_config"]["apple_m1_pro"]
    with pytest.raises(ValueError):
        TaskAllocator(invalid_config)

def test_invalid_tasks(allocator):
    """测试无效任务。"""
    # 测试缺少 input_tokens
    with pytest.raises(KeyError):
        allocator.allocate([{"output_tokens": 50}])
    
    # 测试缺少 output_tokens
    with pytest.raises(KeyError):
        allocator.allocate([{"input_tokens": 500}])
    
    # 测试无效的 input_tokens 类型
    with pytest.raises(TypeError):
        allocator.allocate([{"input_tokens": "500", "output_tokens": 50}])
    
    # 测试无效的 output_tokens 类型
    with pytest.raises(TypeError):
        allocator.allocate([{"input_tokens": 500, "output_tokens": "50"}])

def test_cleanup(allocator):
    """测试资源清理。"""
    # 分配一些任务
    tasks = [
        {"input_tokens": 500, "output_tokens": 50},
        {"input_tokens": 1500, "output_tokens": 150},
        {"input_tokens": 2500, "output_tokens": 250}
    ]
    allocations = allocator.allocate(tasks)
    
    # 清理资源
    allocator.cleanup()
    
    # 验证所有性能分析器已被清理
    for profiler in allocator.profilers.values():
        assert profiler.handle is None
        assert profiler.nvml is None 