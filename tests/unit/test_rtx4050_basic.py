"""RTX4050基本功能测试。"""

import os
import pytest
import time
import threading
import pynvml
from unittest.mock import patch, MagicMock

from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from src.toolbox.logger import get_logger

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

@pytest.fixture(scope="function")
def mock_nvml():
    """模拟NVML交互。"""
    with patch("pynvml.nvmlInit") as mock_init, \
         patch("pynvml.nvmlDeviceGetHandleByIndex") as mock_get_handle, \
         patch("pynvml.nvmlDeviceGetName") as mock_get_name, \
         patch("pynvml.nvmlDeviceGetPowerUsage") as mock_get_power, \
         patch("pynvml.nvmlDeviceGetMemoryInfo") as mock_get_memory, \
         patch("pynvml.nvmlDeviceGetUtilizationRates") as mock_get_util, \
         patch("pynvml.nvmlShutdown") as mock_shutdown:
        
        # 设置模拟返回值
        mock_handle = MagicMock()
        mock_get_handle.return_value = mock_handle
        mock_get_name.return_value = b"NVIDIA GeForce RTX 4050"
        mock_get_power.return_value = 50000  # 50W in milliwatts
        
        class MemoryInfo:
            def __init__(self):
                self.total = 6 * 1024 * 1024 * 1024  # 6GB
                self.used = 2 * 1024 * 1024 * 1024   # 2GB
                self.free = 4 * 1024 * 1024 * 1024   # 4GB
        mock_get_memory.return_value = MemoryInfo()
        
        class UtilizationRates:
            def __init__(self):
                self.gpu = 75
                self.memory = 50
        mock_get_util.return_value = UtilizationRates()
        
        yield {
            "init": mock_init,
            "get_handle": mock_get_handle,
            "get_name": mock_get_name,
            "get_power": mock_get_power,
            "get_memory": mock_get_memory,
            "get_util": mock_get_util,
            "shutdown": mock_shutdown,
            "handle": mock_handle
        }

def test_basic_measurement(profiler, mock_nvml):
    """测试基本性能测量。"""
    result = profiler.measure(100)
    assert result["runtime"] > 0
    assert result["power"] > 0
    assert result["energy"] > 0
    assert result["throughput"] > 0

def test_boundary_conditions(profiler, mock_nvml):
    """测试边界条件。"""
    # 测试最小输入值
    result = profiler.measure(1)
    assert result["runtime"] > 0
    
    # 测试极大输入值
    with pytest.raises(MemoryError):
        profiler.measure(1000000000)
    
    # 测试零和负值
    with pytest.raises(ValueError):
        profiler.measure(0)
    with pytest.raises(ValueError):
        profiler.measure(-1)

def test_error_handling(profiler, mock_nvml):
    """测试错误处理。"""
    # 测试设备未初始化
    profiler.device = None
    with pytest.raises(RuntimeError):
        profiler.measure(100)
    
    # 测试NVML错误
    def mock_error():
        raise pynvml.NVMLError("NVML error")
    mock_nvml.monkeypatch.setattr(pynvml, "nvmlDeviceGetPowerUsage", mock_error)
    with pytest.raises(RuntimeError):
        profiler.measure(100)

def test_resource_cleanup(profiler, mock_nvml):
    """测试资源清理。"""
    profiler.measure(100)
    profiler.cleanup()
    assert profiler.device is None
    
def test_concurrent_operations(profiler, mock_nvml):
    """测试并发操作。"""
    def measure_task():
        result = profiler.measure(100)
        assert result["runtime"] > 0
    
    threads = []
    for _ in range(5):
        t = threading.Thread(target=measure_task)
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()

def test_performance_metrics(profiler, mock_nvml):
    """测试性能指标。"""
    result = profiler.measure(100)
    assert "runtime" in result
    assert "power" in result
    assert "energy" in result
    assert "throughput" in result
    assert "memory_usage" in result
    assert "gpu_utilization" in result 