"""RTX4050 性能分析器测试模块。"""

import pytest
import time
import torch
import pynvml
from unittest.mock import patch, MagicMock
from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from typing import Dict, Any, List

# 测试配置
TEST_CONFIG = {
    "device_id": 0,
    "device_type": "gpu",
    "idle_power": 15.0,
    "sample_interval": 200
}

@pytest.fixture
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

@pytest.fixture
def profiler(mock_nvml):
    """创建 RTX4050Profiler 实例的 fixture。"""
    profiler = RTX4050Profiler(TEST_CONFIG)
    yield profiler
    profiler.cleanup()

def test_initialization(profiler, mock_nvml):
    """测试初始化。"""
    assert profiler.config == TEST_CONFIG
    assert profiler.device_type == "gpu"
    assert profiler.idle_power == 15.0
    assert profiler.sample_interval == 200
    assert profiler.initialized
    assert profiler.handle is not None
    assert profiler.device is not None
    assert profiler.device.type == "cuda"
    assert profiler.device.index == 0

def test_config_validation():
    """测试配置验证。"""
    # 测试无效的 device_id
    with pytest.raises(ValueError, match="device_id 必须是整数"):
        RTX4050Profiler({"device_id": "0"})
    
    # 测试无效的 device_type
    with pytest.raises(ValueError, match="device_type 必须是字符串"):
        RTX4050Profiler({"device_id": 0, "device_type": 123})
    
    # 测试无效的 idle_power
    with pytest.raises(ValueError, match="idle_power 必须是正数"):
        RTX4050Profiler({"device_id": 0, "device_type": "gpu", "idle_power": -1})
    
    # 测试无效的 sample_interval
    with pytest.raises(ValueError, match="sample_interval 必须是正整数"):
        RTX4050Profiler({"device_id": 0, "device_type": "gpu", "sample_interval": 0})

def test_cuda_availability():
    """测试 CUDA 可用性检查。"""
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="CUDA 不可用"):
            RTX4050Profiler(TEST_CONFIG)
    
    with patch("torch.cuda.device_count", return_value=0):
        with pytest.raises(ValueError, match="设备 ID 0 无效"):
            RTX4050Profiler(TEST_CONFIG)

def test_device_type_validation(mock_nvml):
    """测试设备类型验证。"""
    mock_nvml["get_name"].return_value = b"NVIDIA GTX 1080"
    with pytest.warns(UserWarning, match="当前设备不是 RTX 4050"):
        profiler = RTX4050Profiler(TEST_CONFIG)
        assert profiler.initialized

def test_power_measurement(profiler, mock_nvml):
    """测试功率测量。"""
    # 测试正常情况
    power = profiler.measure_power()
    assert isinstance(power, float)
    assert power >= 0.0
    
    # 测试 NVML 错误
    mock_nvml["get_power"].side_effect = pynvml.NVMLError(1)
    power = profiler.measure_power()
    assert power == profiler.idle_power

def test_measurement(profiler, mock_nvml):
    """测试性能测量。"""
    def mock_task():
        time.sleep(0.1)
        return "result"
    
    # 测试正常情况
    metrics = profiler.measure(mock_task, input_tokens=10, output_tokens=20)
    assert isinstance(metrics, dict)
    assert "energy" in metrics
    assert "runtime" in metrics
    assert "throughput" in metrics
    assert "energy_per_token" in metrics
    assert metrics["runtime"] >= 0.1
    assert metrics["energy"] >= 0.0
    assert metrics["throughput"] > 0
    assert metrics["energy_per_token"] >= 0.0
    
    # 测试任务执行错误
    def failing_task():
        raise RuntimeError("任务执行失败")
    
    with pytest.raises(RuntimeError, match="任务执行失败"):
        profiler.measure(failing_task, input_tokens=10, output_tokens=20)

def test_cleanup(profiler, mock_nvml):
    """测试资源清理。"""
    profiler.cleanup()
    assert not profiler.initialized
    assert profiler.handle is None
    mock_nvml["shutdown"].assert_called_once()

def test_destructor(mock_nvml):
    """测试析构函数。"""
    profiler = RTX4050Profiler(TEST_CONFIG)
    del profiler
    mock_nvml["shutdown"].assert_called_once()

def test_measurement_with_zero_tokens(profiler):
    """测试零令牌输入/输出的测量。"""
    def mock_task():
        time.sleep(0.1)
    
    metrics = profiler.measure(mock_task, input_tokens=0, output_tokens=0)
    assert metrics["throughput"] == 0
    assert metrics["energy_per_token"] == 0

def test_measurement_with_high_tokens(profiler):
    """测试大量令牌的测量。"""
    def mock_task():
        time.sleep(0.1)
    
    metrics = profiler.measure(mock_task, input_tokens=1000, output_tokens=1000)
    assert metrics["throughput"] > 0
    assert metrics["energy_per_token"] > 0

def test_concurrent_measurements(profiler):
    """测试并发测量。"""
    import threading
    
    def mock_task():
        time.sleep(0.1)
    
    def run_measurement():
        metrics = profiler.measure(mock_task, input_tokens=10, output_tokens=10)
        assert metrics["runtime"] >= 0.1
    
    threads = [threading.Thread(target=run_measurement) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

def test_error_handling():
    """测试错误处理。"""
    # 测试初始化错误
    with patch("pynvml.nvmlInit", side_effect=pynvml.NVMLError(1)):
        with pytest.raises(RuntimeError):
            RTX4050Profiler(TEST_CONFIG)
    
    # 测试获取句柄错误
    with patch("pynvml.nvmlDeviceGetHandleByIndex", side_effect=pynvml.NVMLError(1)):
        with pytest.raises(RuntimeError):
            RTX4050Profiler(TEST_CONFIG)
    
    # 测试未初始化状态下的测量
    profiler = RTX4050Profiler(TEST_CONFIG)
    profiler.initialized = False
    with pytest.raises(RuntimeError, match="性能分析器未初始化"):
        profiler.measure(lambda: None, input_tokens=10, output_tokens=10) 