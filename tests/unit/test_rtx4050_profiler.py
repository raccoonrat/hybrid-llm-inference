"""RTX4050性能分析器的详细测试。"""

import os
import pytest
import time
import torch
import pynvml
from unittest.mock import patch, MagicMock

from ...src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from ...src.toolbox.logger import get_logger

logger = get_logger(__name__)

# 测试配置
TEST_CONFIG = {
    "device_id": 0,
    "device_type": "RTX4050",
    "idle_power": 20.0,
    "sample_interval": 0.1,
    "memory_limit": 6 * 1024 * 1024 * 1024,  # 6GB
    "tdp": 115.0,  # 115W
    "log_level": "DEBUG"
}

@pytest.fixture(scope="function")
def profiler():
    """创建RTX4050Profiler实例。"""
    prof = RTX4050Profiler(TEST_CONFIG)
    yield prof
    prof.cleanup()

@pytest.fixture
def mock_nvml(monkeypatch):
    """模拟NVML交互。"""
    mock_handle = MagicMock()
    
    def mock_init():
        return None
    
    def mock_device_get_handle(index):
        return mock_handle
    
    def mock_device_get_name(handle):
        return "NVIDIA GeForce RTX 4050"
    
    def mock_device_get_power_usage(handle):
        return 50000  # 50W in milliwatts
    
    def mock_device_get_memory_info(handle):
        class MemoryInfo:
            def __init__(self):
                self.total = 6 * 1024 * 1024 * 1024  # 6GB
                self.used = 2 * 1024 * 1024 * 1024   # 2GB
                self.free = 4 * 1024 * 1024 * 1024   # 4GB
        return MemoryInfo()
    
    def mock_device_get_utilization_rates(handle):
        class UtilizationRates:
            def __init__(self):
                self.gpu = 75
                self.memory = 50
        return UtilizationRates()
    
    monkeypatch.setattr(pynvml, "nvmlInit", mock_init)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetHandleByIndex", mock_device_get_handle)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetName", mock_device_get_name)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetPowerUsage", mock_device_get_power_usage)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetMemoryInfo", mock_device_get_memory_info)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetUtilizationRates", mock_device_get_utilization_rates)
    
    return mock_handle

def test_initialization(profiler, mock_nvml):
    """测试初始化过程。"""
    assert profiler is not None
    assert profiler.config == TEST_CONFIG
    assert profiler.device_type == "RTX4050"
    assert profiler.idle_power == 20.0
    assert profiler.sample_interval == 0.1
    assert profiler.initialized

def test_power_measurement(profiler, mock_nvml):
    """测试功率测量功能。"""
    power = profiler.measure_power()
    assert isinstance(power, float)
    assert power > 0
    
    # 测试多次测量
    powers = [profiler.measure_power() for _ in range(5)]
    assert all(isinstance(p, float) for p in powers)
    assert all(p > 0 for p in powers)

def test_memory_measurement(profiler, mock_nvml):
    """测试内存测量功能。"""
    memory_info = profiler.get_memory_usage()
    assert isinstance(memory_info, dict)
    assert "total" in memory_info
    assert "used" in memory_info
    assert "free" in memory_info
    assert memory_info["total"] == 6 * 1024 * 1024 * 1024
    assert memory_info["used"] == 2 * 1024 * 1024 * 1024
    assert memory_info["free"] == 4 * 1024 * 1024 * 1024

def test_gpu_utilization(profiler, mock_nvml):
    """测试GPU利用率测量。"""
    utilization = profiler.get_gpu_utilization()
    assert isinstance(utilization, float)
    assert 0 <= utilization <= 100

def test_monitoring(profiler, mock_nvml):
    """测试监控功能。"""
    profiler.start_monitoring()
    time.sleep(0.2)  # 等待一些数据收集
    profiler.stop_monitoring()
    
    metrics = profiler.get_metrics()
    assert isinstance(metrics, dict)
    assert "power" in metrics
    assert "memory" in metrics
    assert "utilization" in metrics

def test_error_handling(profiler, mock_nvml):
    """测试错误处理。"""
    # 测试NVML初始化错误
    def mock_init_error():
        raise pynvml.NVMLError("NVML initialization failed")
    mock_nvml.monkeypatch.setattr(pynvml, "nvmlInit", mock_init_error)
    with pytest.raises(RuntimeError):
        RTX4050Profiler(TEST_CONFIG)
    
    # 测试设备获取错误
    def mock_device_error():
        raise pynvml.NVMLError("Device not found")
    mock_nvml.monkeypatch.setattr(pynvml, "nvmlDeviceGetHandleByIndex", mock_device_error)
    with pytest.raises(RuntimeError):
        RTX4050Profiler(TEST_CONFIG)

def test_resource_cleanup(profiler, mock_nvml):
    """测试资源清理。"""
    profiler.cleanup()
    assert not profiler.initialized
    assert profiler.device is None

def test_boundary_conditions(profiler, mock_nvml):
    """测试边界条件。"""
    # 测试零功耗情况
    def mock_zero_power(handle):
        return 0.0
    mock_nvml.monkeypatch.setattr(pynvml, "nvmlDeviceGetPowerUsage", mock_zero_power)
    assert profiler.measure_power() == 0.0
    
    # 测试最大功耗情况
    def mock_max_power(handle):
        return profiler.config["tdp"] * 1000  # 转换为毫瓦
    mock_nvml.monkeypatch.setattr(pynvml, "nvmlDeviceGetPowerUsage", mock_max_power)
    assert profiler.measure_power() == profiler.config["tdp"]

def test_concurrent_operations(profiler, mock_nvml):
    """测试并发操作。"""
    import threading
    
    def measure_thread():
        for _ in range(10):
            power = profiler.measure_power()
            assert isinstance(power, float)
            assert power >= 0
    
    threads = [threading.Thread(target=measure_thread) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

def test_performance_metrics(profiler, mock_nvml):
    """测试性能指标计算。"""
    # 启动监控
    profiler.start_monitoring()
    
    # 模拟一些负载
    time.sleep(0.5)
    
    # 停止监控
    metrics = profiler.stop_monitoring()
    
    # 验证指标
    assert isinstance(metrics, dict)
    assert all(key in metrics for key in ["power", "memory", "utilization"])
    assert all(isinstance(metrics[key], list) for key in metrics)
    assert all(len(metrics[key]) > 0 for key in metrics)
    
    # 验证统计数据
    stats = profiler.get_statistics()
    assert isinstance(stats, dict)
    assert all(key in stats for key in ["avg_power", "max_power", "min_power"])
    assert all(isinstance(stats[key], float) for key in stats) 