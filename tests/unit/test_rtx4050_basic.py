"""RTX4050 基本功能测试模块。"""

import pytest
import time
import torch
import pynvml
from unittest.mock import patch, MagicMock
from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler

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

def test_basic_initialization(profiler, mock_nvml):
    """测试基本初始化。"""
    assert profiler is not None
    assert profiler.config == TEST_CONFIG
    assert profiler.device_type == "gpu"
    assert profiler.idle_power == 15.0
    assert profiler.sample_interval == 200
    assert profiler.initialized
    assert profiler.handle is not None
    assert profiler.device is not None
    assert profiler.device.type == "cuda"
    assert profiler.device.index == 0

def test_power_measurement_basic(profiler, mock_nvml):
    """测试基本功率测量。"""
    # 测试正常情况
    power = profiler.measure_power()
    assert isinstance(power, float)
    assert power >= 0.0
    
    # 测试多次测量
    powers = [profiler.measure_power() for _ in range(5)]
    assert all(isinstance(p, float) for p in powers)
    assert all(p >= 0.0 for p in powers)
    
    # 测试 NVML 错误
    mock_nvml["get_power"].side_effect = pynvml.NVMLError(1)
    power = profiler.measure_power()
    assert power == profiler.idle_power

def test_basic_measurement(profiler, mock_nvml):
    """测试基本性能测量。"""
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
    
    # 测试多次测量
    metrics_list = [profiler.measure(mock_task, input_tokens=10, output_tokens=20) for _ in range(5)]
    assert all(isinstance(m, dict) for m in metrics_list)
    assert all(m["runtime"] >= 0.1 for m in metrics_list)
    assert all(m["energy"] >= 0.0 for m in metrics_list)

def test_basic_cleanup(profiler, mock_nvml):
    """测试基本资源清理。"""
    profiler.cleanup()
    assert not profiler.initialized
    assert profiler.handle is None
    mock_nvml["shutdown"].assert_called_once()
    
    # 测试清理后再次初始化
    profiler._init_profiler()
    assert profiler.initialized
    assert profiler.handle is not None

def test_basic_error_handling():
    """测试基本错误处理。"""
    # 测试无效配置
    with pytest.raises(ValueError):
        RTX4050Profiler({})
    
    # 测试 CUDA 不可用
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="CUDA 不可用"):
            RTX4050Profiler(TEST_CONFIG)
    
    # 测试无效设备 ID
    with patch("torch.cuda.device_count", return_value=0):
        with pytest.raises(ValueError, match="设备 ID 0 无效"):
            RTX4050Profiler(TEST_CONFIG)

def test_basic_device_validation(mock_nvml):
    """测试基本设备验证。"""
    # 测试非 RTX 4050 设备
    mock_nvml["get_name"].return_value = b"NVIDIA GTX 1080"
    with pytest.warns(UserWarning, match="当前设备不是 RTX 4050"):
        profiler = RTX4050Profiler(TEST_CONFIG)
        assert profiler.initialized
    
    # 测试设备名称解码错误
    mock_nvml["get_name"].return_value = b"\xff\xfe"  # 无效的 UTF-8
    with pytest.raises(UnicodeDecodeError):
        RTX4050Profiler(TEST_CONFIG)

def test_basic_measurement_edge_cases(profiler):
    """测试基本测量的边界情况。"""
    def mock_task():
        time.sleep(0.1)
    
    # 测试零令牌
    metrics = profiler.measure(mock_task, input_tokens=0, output_tokens=0)
    assert metrics["throughput"] == 0
    assert metrics["energy_per_token"] == 0
    
    # 测试大量令牌
    metrics = profiler.measure(mock_task, input_tokens=1000, output_tokens=1000)
    assert metrics["throughput"] > 0
    assert metrics["energy_per_token"] > 0
    
    # 测试极短任务
    def very_short_task():
        pass
    metrics = profiler.measure(very_short_task, input_tokens=10, output_tokens=10)
    assert metrics["runtime"] >= 0
    assert metrics["throughput"] > 0

def test_basic_resource_management(profiler, mock_nvml):
    """测试基本资源管理。"""
    # 测试多次初始化和清理
    for _ in range(3):
        profiler.cleanup()
        assert not profiler.initialized
        assert profiler.handle is None
        profiler._init_profiler()
        assert profiler.initialized
        assert profiler.handle is not None
    
    # 测试析构函数
    del profiler
    mock_nvml["shutdown"].assert_called()

def test_memory_info(profiler):
    """测试内存信息获取。"""
    memory_info = profiler.get_memory_info()
    assert isinstance(memory_info, dict)
    assert "total" in memory_info
    assert "free" in memory_info
    assert "used" in memory_info
    assert all(isinstance(v, int) for v in memory_info.values())
    assert memory_info["total"] > 0
    assert memory_info["free"] >= 0
    assert memory_info["used"] >= 0
    assert memory_info["used"] <= memory_info["total"]
    
    # 测试多次获取
    memory_infos = [profiler.get_memory_info() for _ in range(5)]
    assert all(isinstance(info, dict) for info in memory_infos)
    assert all("total" in info for info in memory_infos)

def test_temperature(profiler):
    """测试温度获取。"""
    temp = profiler.get_temperature()
    assert isinstance(temp, float)
    assert temp >= 0.0
    
    # 测试多次获取
    temps = [profiler.get_temperature() for _ in range(5)]
    assert all(isinstance(t, float) for t in temps)
    assert all(t >= 0.0 for t in temps)

def test_measurement_lifecycle(profiler):
    """测试测量生命周期。"""
    # 开始测量
    profiler.start_measurement()
    assert profiler.is_measuring
    
    # 执行一些计算
    start_time = time.time()
    while time.time() - start_time < 0.1:  # 运行 100ms
        _ = [i * i for i in range(1000)]
    
    # 停止测量
    metrics = profiler.stop_measurement()
    assert not profiler.is_measuring
    assert isinstance(metrics, dict)
    assert "runtime" in metrics
    assert "power" in metrics
    assert "energy" in metrics
    assert metrics["runtime"] > 0
    assert metrics["power"] >= TEST_CONFIG["idle_power"]
    assert metrics["energy"] > 0

def test_performance_metrics(profiler):
    """测试性能指标测量。"""
    task = {
        "input": "This is a test input for performance measurement.",
        "max_tokens": 20
    }
    
    metrics = profiler.measure_performance(task)
    assert isinstance(metrics, dict)
    assert "runtime" in metrics
    assert "power" in metrics
    assert "energy" in metrics
    assert "throughput" in metrics
    assert "energy_per_token" in metrics
    
    assert metrics["runtime"] > 0
    assert metrics["power"] >= TEST_CONFIG["idle_power"]
    assert metrics["energy"] > 0
    assert metrics["throughput"] > 0
    assert metrics["energy_per_token"] > 0

def test_cleanup(profiler):
    """测试资源清理。"""
    profiler.cleanup()
    assert profiler.handle is None
    assert profiler.device is None
    
    # 测试清理后使用
    with pytest.raises(RuntimeError):
        profiler.measure_power()
    
    with pytest.raises(RuntimeError):
        profiler.get_memory_info()
    
    with pytest.raises(RuntimeError):
        profiler.get_temperature()

def test_error_handling():
    """测试错误处理。"""
    # 测试无效配置
    invalid_configs = [
        {},  # 空配置
        {"device_type": 123},  # 无效的设备类型
        {"idle_power": "invalid"},  # 无效的空闲功率
        {"sample_interval": -1},  # 无效的采样间隔
        {"device_type": "nvidia", "idle_power": -1},  # 无效的空闲功率
        {"device_type": "nvidia", "sample_interval": 0}  # 无效的采样间隔
    ]
    
    for config in invalid_configs:
        with pytest.raises((ValueError, TypeError)):
            RTX4050Profiler(config)

def test_real_mode():
    """测试真实模式。"""
    profiler = RTX4050Profiler(TEST_CONFIG)
    try:
        # 验证基本功能
        power = profiler.measure_power()
        assert isinstance(power, float)
        assert power >= 0.0
        
        memory_info = profiler.get_memory_info()
        assert isinstance(memory_info, dict)
        assert "total" in memory_info
        assert memory_info["total"] > 0
        
        temp = profiler.get_temperature()
        assert isinstance(temp, float)
        assert temp >= 0.0
        
        # 验证性能指标
        task = {
            "input": "Test input",
            "max_tokens": 10
        }
        metrics = profiler.measure_performance(task)
        assert isinstance(metrics, dict)
        assert "runtime" in metrics
        assert metrics["runtime"] > 0
    finally:
        profiler.cleanup()

def test_config_validation():
    """测试配置验证。"""
    # 测试缺少必要参数
    missing_configs = [
        {"idle_power": 15.0, "sample_interval": 200},  # 缺少 device_id
        {"device_type": "gpu", "sample_interval": 200},  # 缺少 idle_power
        {"device_type": "gpu", "idle_power": 15.0}  # 缺少 sample_interval
    ]
    
    for config in missing_configs:
        with pytest.raises(ValueError):
            RTX4050Profiler(config)
    
    # 测试类型错误
    type_error_configs = [
        {"device_type": 123, "idle_power": 15.0, "sample_interval": 200},
        {"device_type": "gpu", "idle_power": "15.0", "sample_interval": 200},
        {"device_type": "gpu", "idle_power": 15.0, "sample_interval": "200"}
    ]
    
    for config in type_error_configs:
        with pytest.raises(TypeError):
            RTX4050Profiler(config) 