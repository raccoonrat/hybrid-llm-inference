"""硬件性能分析模块的测试用例。"""

import os
import pytest
import torch
import pynvml
from unittest.mock import patch, MagicMock

from ..src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from ..src.hardware_profiling.a800_profiling import A800Profiler
from ..src.hardware_profiling.m1_profiler import M1Profiler
from ..src.hardware_profiling import get_profiler

# 测试配置
TEST_CONFIG = {
    "device_id": 0,
    "device_type": "gpu",
    "idle_power": 15.0,
    "sample_interval": 200,
    "log_level": "DEBUG"
}

@pytest.fixture(scope="function")
def mock_nvml():
    """模拟 NVML 的 fixture。"""
    with patch("pynvml.nvmlInit") as mock_init, \
         patch("pynvml.nvmlDeviceGetHandleByIndex") as mock_get_handle, \
         patch("pynvml.nvmlDeviceGetName") as mock_get_name, \
         patch("pynvml.nvmlDeviceGetPowerUsage") as mock_get_power, \
         patch("pynvml.nvmlShutdown") as mock_shutdown:
        
        # 设置模拟返回值
        mock_get_name.return_value = b"NVIDIA RTX 4050"
        mock_get_power.return_value = 15000  # 15W
        
        yield {
            "init": mock_init,
            "get_handle": mock_get_handle,
            "get_name": mock_get_name,
            "get_power": mock_get_power,
            "shutdown": mock_shutdown
        }

@pytest.fixture(scope="function")
def mock_torch():
    """模拟 PyTorch 的 fixture。"""
    with patch("torch.cuda.is_available") as mock_is_available, \
         patch("torch.cuda.device_count") as mock_device_count:
        
        mock_is_available.return_value = True
        mock_device_count.return_value = 1
        
        yield {
            "is_available": mock_is_available,
            "device_count": mock_device_count
        }

def test_rtx4050_profiler_init(mock_nvml, mock_torch):
    """测试 RTX4050Profiler 初始化。"""
    profiler = RTX4050Profiler(TEST_CONFIG)
    assert profiler.device_id == TEST_CONFIG["device_id"]
    assert profiler.device_type == TEST_CONFIG["device_type"]
    assert profiler.idle_power == TEST_CONFIG["idle_power"]
    assert profiler.sample_interval == TEST_CONFIG["sample_interval"]

def test_rtx4050_profiler_init_invalid_config():
    """测试 RTX4050Profiler 初始化时的无效配置。"""
    invalid_configs = [
        {"device_id": "invalid"},  # 非整数 device_id
        {"device_type": 123},      # 非字符串 device_type
        {"idle_power": -1.0},      # 非正数 idle_power
        {"sample_interval": 0}     # 非正整数 sample_interval
    ]
    
    for config in invalid_configs:
        with pytest.raises(ValueError):
            RTX4050Profiler({**TEST_CONFIG, **config})

def test_rtx4050_profiler_measure_power(mock_nvml, mock_torch):
    """测试 RTX4050Profiler 的功率测量功能。"""
    profiler = RTX4050Profiler(TEST_CONFIG)
    power = profiler.measure_power()
    assert isinstance(power, float)
    assert power >= 0.0

def test_rtx4050_profiler_measure(mock_nvml, mock_torch):
    """测试 RTX4050Profiler 的性能测量功能。"""
    profiler = RTX4050Profiler(TEST_CONFIG)
    
    def mock_task():
        """模拟任务。"""
        pass
    
    metrics = profiler.measure(mock_task, input_tokens=10, output_tokens=20)
    
    assert isinstance(metrics, dict)
    assert "energy" in metrics
    assert "runtime" in metrics
    assert "throughput" in metrics
    assert "energy_per_token" in metrics
    assert metrics["energy"] >= 0
    assert metrics["runtime"] >= 0
    assert metrics["throughput"] >= 0
    assert metrics["energy_per_token"] >= 0

def test_rtx4050_profiler_cleanup(mock_nvml, mock_torch):
    """测试 RTX4050Profiler 的资源清理功能。"""
    profiler = RTX4050Profiler(TEST_CONFIG)
    profiler.cleanup()
    assert not profiler.initialized
    assert profiler.handle is None
    assert profiler.device is None

def test_a800_profiler_init(mock_nvml, mock_torch):
    """测试 A800Profiler 初始化。"""
    profiler = A800Profiler(TEST_CONFIG)
    assert profiler.device_id == TEST_CONFIG["device_id"]
    assert profiler.sample_interval == TEST_CONFIG["sample_interval"]
    assert profiler.idle_power == 50.0  # A800 的默认空闲功率

def test_a800_profiler_measure(mock_nvml, mock_torch):
    """测试 A800Profiler 的性能测量功能。"""
    profiler = A800Profiler(TEST_CONFIG)
    
    def mock_task():
        """模拟任务。"""
        pass
    
    metrics = profiler.measure(mock_task, input_tokens=10, output_tokens=20)
    
    assert isinstance(metrics, dict)
    assert "energy" in metrics
    assert "runtime" in metrics
    assert "throughput" in metrics
    assert "energy_per_token" in metrics

def test_m1_profiler_init():
    """测试 M1Profiler 初始化。"""
    profiler = M1Profiler(TEST_CONFIG)
    assert profiler.device_id == TEST_CONFIG["device_id"]
    assert profiler.sample_interval == TEST_CONFIG["sample_interval"]
    assert profiler.idle_power == TEST_CONFIG["idle_power"]

def test_m1_profiler_measure():
    """测试 M1Profiler 的性能测量功能。"""
    profiler = M1Profiler(TEST_CONFIG)
    
    def mock_task():
        """模拟任务。"""
        pass
    
    metrics = profiler.measure(mock_task, input_tokens=10, output_tokens=20)
    
    assert isinstance(metrics, dict)
    assert "energy" in metrics
    assert "runtime" in metrics
    assert "throughput" in metrics
    assert "energy_per_token" in metrics

def test_get_profiler():
    """测试获取性能分析器。"""
    # 测试获取 RTX4050 分析器
    rtx4050_profiler = get_profiler("rtx4050", TEST_CONFIG)
    assert isinstance(rtx4050_profiler, RTX4050Profiler)
    
    # 测试获取 A800 分析器
    a800_profiler = get_profiler("a800", TEST_CONFIG)
    assert isinstance(a800_profiler, A800Profiler)
    
    # 测试获取 M1 分析器
    m1_profiler = get_profiler("m1_pro", TEST_CONFIG)
    assert isinstance(m1_profiler, M1Profiler)
    
    # 测试获取不支持的设备类型
    with pytest.raises(ValueError, match="不支持的设备类型"):
        get_profiler("invalid_device", TEST_CONFIG) 