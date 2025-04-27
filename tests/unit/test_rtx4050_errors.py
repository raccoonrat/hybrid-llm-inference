"""RTX4050错误处理测试。"""

import os
import pytest
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

def test_nvml_init_error(profiler, mock_nvml):
    """测试NVML初始化错误。"""
    # 模拟NVML初始化失败
    mock_nvml["init"].side_effect = pynvml.NVMLError("NVML initialization failed")
    
    with pytest.raises(RuntimeError) as exc_info:
        profiler.initialize()
    assert "NVML initialization failed" in str(exc_info.value)

def test_device_not_found_error(profiler, mock_nvml):
    """测试设备未找到错误。"""
    # 模拟设备获取失败
    mock_nvml["get_handle"].side_effect = pynvml.NVMLError("Device not found")
    
    with pytest.raises(RuntimeError) as exc_info:
        profiler.initialize()
    assert "Device not found" in str(exc_info.value)

def test_power_measurement_error(profiler, mock_nvml):
    """测试功率测量错误。"""
    # 模拟功率测量失败
    mock_nvml["get_power"].side_effect = pynvml.NVMLError("Power measurement failed")
    
    with pytest.raises(RuntimeError) as exc_info:
        profiler.measure_power()
    assert "Power measurement failed" in str(exc_info.value)

def test_memory_measurement_error(profiler, mock_nvml):
    """测试内存测量错误。"""
    # 模拟内存测量失败
    mock_nvml["get_memory"].side_effect = pynvml.NVMLError("Memory measurement failed")
    
    with pytest.raises(RuntimeError) as exc_info:
        profiler.get_memory_usage()
    assert "Memory measurement failed" in str(exc_info.value)

def test_utilization_measurement_error(profiler, mock_nvml):
    """测试利用率测量错误。"""
    # 模拟利用率测量失败
    mock_nvml["get_util"].side_effect = pynvml.NVMLError("Utilization measurement failed")
    
    with pytest.raises(RuntimeError) as exc_info:
        profiler.get_gpu_utilization()
    assert "Utilization measurement failed" in str(exc_info.value)

def test_invalid_config_error():
    """测试无效配置错误。"""
    invalid_configs = [
        {"device_id": "invalid"},  # 非整数 device_id
        {"device_type": 123},      # 非字符串 device_type
        {"idle_power": -1.0},      # 非正数 idle_power
        {"sample_interval": 0},    # 非正整数 sample_interval
        {"memory_limit": -1},      # 非正数 memory_limit
        {"tdp": 0.0}              # 非正数 tdp
    ]
    
    for config in invalid_configs:
        with pytest.raises(ValueError):
            RTX4050Profiler({**TEST_CONFIG, **config})

def test_resource_cleanup_error(profiler, mock_nvml):
    """测试资源清理错误。"""
    # 模拟清理失败
    mock_nvml["shutdown"].side_effect = pynvml.NVMLError("Cleanup failed")
    
    with pytest.raises(RuntimeError) as exc_info:
        profiler.cleanup()
    assert "Cleanup failed" in str(exc_info.value)

def test_concurrent_error_handling(profiler, mock_nvml):
    """测试并发错误处理。"""
    import threading
    
    def error_task():
        try:
            # 模拟随机错误
            if threading.current_thread().ident % 2 == 0:
                mock_nvml["get_power"].side_effect = pynvml.NVMLError("Random error")
                profiler.measure_power()
            else:
                mock_nvml["get_memory"].side_effect = pynvml.NVMLError("Random error")
                profiler.get_memory_usage()
        except RuntimeError as e:
            assert "Random error" in str(e)
    
    threads = [threading.Thread(target=error_task) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join() 