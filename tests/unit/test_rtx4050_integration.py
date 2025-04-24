"""RTX4050 集成测试模块。"""

import pytest
import time
import threading
from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from src.scheduling.task_allocator import TaskAllocator
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

@pytest.fixture
def profiler():
    """创建 RTX4050Profiler 实例的 fixture。"""
    profiler = RTX4050Profiler(HARDWARE_CONFIG["nvidia_rtx4050"])
    profiler.initialize()
    yield profiler
    profiler.cleanup()

@pytest.fixture
def allocator():
    """创建 TaskAllocator 实例的 fixture。"""
    return TaskAllocator(HARDWARE_CONFIG, MODEL_CONFIG)

def test_profiler_integration(profiler, allocator):
    """测试分析器集成。"""
    # 验证分析器初始化
    assert profiler.is_initialized
    assert profiler.config == HARDWARE_CONFIG["nvidia_rtx4050"]
    
    # 验证基本功能
    power = profiler.measure_power()
    assert isinstance(power, float)
    assert power >= 0.0
    
    memory_info = profiler.get_memory_info()
    assert isinstance(memory_info, dict)
    assert "total" in memory_info
    assert "used" in memory_info
    assert "free" in memory_info
    
    temperature = profiler.get_temperature()
    assert isinstance(temperature, float)
    assert temperature >= 0.0

def test_measurement_integration(profiler, allocator):
    """测试测量集成。"""
    # 创建任务
    task = {
        "input_tokens": 100,
        "output_tokens": 50,
        "model": "tinyllama"
    }
    
    # 分配任务
    allocation = allocator.allocate(task)
    assert allocation["hardware"] == "nvidia_rtx4050"
    assert allocation["input_tokens"] == 100
    assert allocation["output_tokens"] == 50
    assert allocation["model"] == "tinyllama"
    
    # 执行测量
    profiler.start_measurement()
    try:
        metrics = profiler.measure(task)
        assert isinstance(metrics, dict)
        assert "runtime" in metrics
        assert "power" in metrics
        assert "energy" in metrics
        assert "throughput" in metrics
        assert "energy_per_token" in metrics
        assert metrics["runtime"] > 0
        assert metrics["power"] >= 0.0
        assert metrics["energy"] >= 0.0
        assert metrics["throughput"] > 0
        assert metrics["energy_per_token"] >= 0.0
    finally:
        profiler.stop_measurement()

def test_concurrent_measurements(profiler, allocator):
    """测试并发测量。"""
    def measure_task(task):
        try:
            # 分配任务
            allocation = allocator.allocate(task)
            assert allocation["hardware"] == "nvidia_rtx4050"
            
            # 执行测量
            profiler.start_measurement()
            try:
                metrics = profiler.measure(task)
                assert isinstance(metrics, dict)
                assert metrics["runtime"] > 0
                assert metrics["power"] >= 0.0
                assert metrics["energy"] >= 0.0
            finally:
                profiler.stop_measurement()
        except Exception as e:
            pytest.fail(f"并发测量失败: {str(e)}")
    
    # 创建多个任务
    tasks = [
        {"input_tokens": 100, "output_tokens": 50, "model": "tinyllama"},
        {"input_tokens": 200, "output_tokens": 100, "model": "tinyllama"},
        {"input_tokens": 300, "output_tokens": 150, "model": "tinyllama"}
    ]
    
    # 创建线程
    threads = [threading.Thread(target=measure_task, args=(task,)) for task in tasks]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

def test_resource_management(profiler, allocator):
    """测试资源管理。"""
    # 创建多个任务
    tasks = [
        {"input_tokens": 100, "output_tokens": 50, "model": "tinyllama"},
        {"input_tokens": 200, "output_tokens": 100, "model": "tinyllama"},
        {"input_tokens": 300, "output_tokens": 150, "model": "tinyllama"}
    ]
    
    # 分配和执行任务
    for task in tasks:
        allocation = allocator.allocate(task)
        assert allocation["hardware"] == "nvidia_rtx4050"
        
        profiler.start_measurement()
        try:
            metrics = profiler.measure(task)
            assert isinstance(metrics, dict)
            assert metrics["runtime"] > 0
            assert metrics["power"] >= 0.0
            assert metrics["energy"] >= 0.0
        finally:
            profiler.stop_measurement()
    
    # 验证资源状态
    assert profiler.is_initialized
    memory_info = profiler.get_memory_info()
    assert memory_info["total"] > 0
    assert memory_info["used"] >= 0
    assert memory_info["free"] >= 0

def test_error_handling(profiler, allocator):
    """测试错误处理。"""
    # 测试无效任务
    invalid_tasks = [
        {"input_tokens": -1, "output_tokens": 50, "model": "tinyllama"},
        {"input_tokens": 100, "output_tokens": -1, "model": "tinyllama"},
        {"input_tokens": "100", "output_tokens": 50, "model": "tinyllama"},
        {"input_tokens": 100, "output_tokens": "50", "model": "tinyllama"},
        {"input_tokens": 100, "output_tokens": 50, "model": "invalid"},
        {},
        None
    ]
    
    for task in invalid_tasks:
        with pytest.raises((ValueError, TypeError, KeyError)):
            allocator.allocate(task)
    
    # 测试无效的硬件配置
    invalid_hardware_config = HARDWARE_CONFIG.copy()
    invalid_hardware_config["nvidia_rtx4050"]["idle_power"] = -1.0
    with pytest.raises(ValueError):
        TaskAllocator(invalid_hardware_config, MODEL_CONFIG)
    
    # 测试无效的模型配置
    invalid_model_config = MODEL_CONFIG.copy()
    invalid_model_config["models"]["tinyllama"]["batch_size"] = -1
    with pytest.raises(ValueError):
        TaskAllocator(HARDWARE_CONFIG, invalid_model_config)

def test_cleanup(profiler, allocator):
    """测试资源清理。"""
    # 清理分析器资源
    profiler.cleanup()
    assert not profiler.is_initialized
    
    # 验证清理后的行为
    with pytest.raises(RuntimeError):
        profiler.measure_power()
    
    with pytest.raises(RuntimeError):
        profiler.get_memory_info()
    
    with pytest.raises(RuntimeError):
        profiler.get_temperature()
    
    with pytest.raises(RuntimeError):
        profiler.start_measurement()
    
    with pytest.raises(RuntimeError):
        profiler.stop_measurement()
    
    with pytest.raises(RuntimeError):
        profiler.measure({"input_tokens": 100, "output_tokens": 50, "model": "tinyllama"}) 