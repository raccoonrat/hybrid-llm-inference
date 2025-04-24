"""RTX4050 集成测试模块。"""

import pytest
import time
import threading
from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from src.scheduling.task_allocator import TaskAllocator
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
def profiler():
    """创建 RTX4050Profiler 实例的 fixture。"""
    return RTX4050Profiler(TEST_CONFIG["hardware_config"]["nvidia_rtx4050"])

@pytest.fixture
def allocator():
    """创建 TaskAllocator 实例的 fixture。"""
    return TaskAllocator(TEST_CONFIG)

def test_profiler_integration(profiler, allocator):
    """测试性能分析器集成。"""
    # 创建任务
    tasks = [
        {"input_tokens": 500, "output_tokens": 50},
        {"input_tokens": 1500, "output_tokens": 150},
        {"input_tokens": 2500, "output_tokens": 250}
    ]
    
    # 分配任务
    allocations = allocator.allocate(tasks)
    
    # 验证分配结果
    assert len(allocations) == 3
    assert allocations[0]["hardware"] == "apple_m1_pro"
    assert allocations[1]["hardware"] == "nvidia_rtx4050"
    assert allocations[2]["hardware"] == "nvidia_rtx4090"
    
    # 验证性能分析器
    for allocation in allocations:
        assert allocation["profiler"] is not None
        assert allocation["profiler"].device_type in ["m1_pro", "rtx4050", "rtx4090"]
        assert allocation["profiler"].idle_power > 0
        assert allocation["profiler"].sample_interval > 0

def test_measurement_integration(profiler, allocator):
    """测试测量集成。"""
    # 创建任务
    tasks = [
        {"input_tokens": 500, "output_tokens": 50},
        {"input_tokens": 1500, "output_tokens": 150},
        {"input_tokens": 2500, "output_tokens": 250}
    ]
    
    # 分配任务
    allocations = allocator.allocate(tasks)
    
    # 执行测量
    for allocation in allocations:
        profiler = allocation["profiler"]
        task = allocation["task"]
        
        # 开始测量
        profiler.start_measurement()
        try:
            # 执行任务
            metrics = profiler.measure(task)
            
            # 验证指标
            assert metrics["runtime"] > 0
            assert metrics["power"] >= 0.0
            assert metrics["energy"] >= 0.0
            assert metrics["throughput"] > 0
            assert metrics["energy_per_token"] >= 0.0
            
            # 验证吞吐量计算
            total_tokens = task["input_tokens"] + task["output_tokens"]
            expected_throughput = total_tokens / metrics["runtime"]
            assert abs(metrics["throughput"] - expected_throughput) < 0.1
            
            # 验证每令牌能耗计算
            expected_energy_per_token = metrics["energy"] / total_tokens
            assert abs(metrics["energy_per_token"] - expected_energy_per_token) < 0.1
        finally:
            # 停止测量
            profiler.stop_measurement()

def test_concurrent_measurements(profiler, allocator):
    """测试并发测量。"""
    def measure_task(allocation: Dict[str, Any]) -> Dict[str, float]:
        """执行测量任务。"""
        profiler = allocation["profiler"]
        task = allocation["task"]
        
        profiler.start_measurement()
        try:
            return profiler.measure(task)
        finally:
            profiler.stop_measurement()
    
    # 创建任务
    tasks = [
        {"input_tokens": 500, "output_tokens": 50},
        {"input_tokens": 1500, "output_tokens": 150},
        {"input_tokens": 2500, "output_tokens": 250}
    ]
    
    # 分配任务
    allocations = allocator.allocate(tasks)
    
    # 创建线程
    threads = []
    results = []
    
    for allocation in allocations:
        thread = threading.Thread(
            target=lambda a: results.append(measure_task(a)),
            args=(allocation,)
        )
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    # 验证结果
    assert len(results) == len(allocations)
    for metrics in results:
        assert metrics["runtime"] > 0
        assert metrics["power"] >= 0.0
        assert metrics["energy"] >= 0.0
        assert metrics["throughput"] > 0
        assert metrics["energy_per_token"] >= 0.0

def test_resource_management(profiler, allocator):
    """测试资源管理。"""
    # 创建任务
    tasks = [
        {"input_tokens": 500, "output_tokens": 50},
        {"input_tokens": 1500, "output_tokens": 150},
        {"input_tokens": 2500, "output_tokens": 250}
    ]
    
    # 分配任务
    allocations = allocator.allocate(tasks)
    
    # 执行测量
    for allocation in allocations:
        profiler = allocation["profiler"]
        task = allocation["task"]
        
        # 开始测量
        profiler.start_measurement()
        try:
            # 执行任务
            metrics = profiler.measure(task)
            
            # 验证指标
            assert metrics["runtime"] > 0
            assert metrics["power"] >= 0.0
            assert metrics["energy"] >= 0.0
            assert metrics["throughput"] > 0
            assert metrics["energy_per_token"] >= 0.0
        finally:
            # 停止测量
            profiler.stop_measurement()
    
    # 清理资源
    allocator.cleanup()
    
    # 验证资源已被清理
    for profiler in allocator.profilers.values():
        assert profiler.handle is None
        assert profiler.nvml is None

def test_error_handling(profiler, allocator):
    """测试错误处理。"""
    # 测试无效任务
    invalid_tasks = [
        {"input_tokens": -1, "output_tokens": 50},
        {"input_tokens": 100, "output_tokens": -1},
        {"input_tokens": "500", "output_tokens": 50},
        {"input_tokens": 500, "output_tokens": "50"},
        {},
        None
    ]
    
    for task in invalid_tasks:
        with pytest.raises((KeyError, TypeError, ValueError)):
            allocator.allocate([task])
    
    # 测试无效配置
    invalid_config = TEST_CONFIG.copy()
    invalid_config["thresholds"]["T_in"] = -1
    with pytest.raises(ValueError):
        TaskAllocator(invalid_config)
    
    # 测试无效的硬件配置
    invalid_config = TEST_CONFIG.copy()
    invalid_config["hardware_config"]["nvidia_rtx4050"]["idle_power"] = -1.0
    with pytest.raises(ValueError):
        RTX4050Profiler(invalid_config["hardware_config"]["nvidia_rtx4050"])
    
    # 测试无效的硬件类型
    invalid_config = TEST_CONFIG.copy()
    invalid_config["hardware_map"]["medium"] = "invalid"
    with pytest.raises(ValueError):
        TaskAllocator(invalid_config)
    
    # 测试缺少硬件配置
    invalid_config = TEST_CONFIG.copy()
    del invalid_config["hardware_config"]["nvidia_rtx4050"]
    with pytest.raises(ValueError):
        TaskAllocator(invalid_config) 