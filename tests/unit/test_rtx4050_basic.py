"""RTX4050 基本功能测试模块。"""

import pytest
import time
from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from typing import Dict, Any

# 测试配置
TEST_CONFIG = {
    "device_type": "rtx4050",
    "idle_power": 15.0,
    "sample_interval": 200
}

@pytest.fixture
def profiler():
    """创建 RTX4050Profiler 实例的 fixture。"""
    profiler = RTX4050Profiler(TEST_CONFIG)
    profiler.initialize()
    yield profiler
    profiler.cleanup()

def test_initialization(profiler):
    """测试初始化。"""
    assert profiler.is_initialized
    assert profiler.config == TEST_CONFIG
    assert profiler.handle is not None
    assert profiler.nvml is not None

def test_measure_power(profiler):
    """测试功率测量。"""
    power = profiler.measure_power()
    assert isinstance(power, float)
    assert power >= 0.0
    
    # 测试多次测量
    powers = [profiler.measure_power() for _ in range(10)]
    assert all(isinstance(p, float) for p in powers)
    assert all(p >= 0.0 for p in powers)
    assert len(set(powers)) > 1  # 确保不是固定值

def test_get_memory_info(profiler):
    """测试获取内存信息。"""
    memory_info = profiler.get_memory_info()
    assert isinstance(memory_info, dict)
    assert "total" in memory_info
    assert "used" in memory_info
    assert "free" in memory_info
    assert memory_info["total"] > 0
    assert memory_info["used"] >= 0
    assert memory_info["free"] >= 0
    assert memory_info["total"] == memory_info["used"] + memory_info["free"]
    
    # 测试多次获取
    memory_infos = [profiler.get_memory_info() for _ in range(10)]
    assert all(isinstance(m, dict) for m in memory_infos)
    assert all(m["total"] > 0 for m in memory_infos)
    assert all(m["used"] >= 0 for m in memory_infos)
    assert all(m["free"] >= 0 for m in memory_infos)
    assert all(m["total"] == m["used"] + m["free"] for m in memory_infos)

def test_get_temperature(profiler):
    """测试获取温度。"""
    temperature = profiler.get_temperature()
    assert isinstance(temperature, float)
    assert temperature >= 0.0
    
    # 测试多次获取
    temperatures = [profiler.get_temperature() for _ in range(10)]
    assert all(isinstance(t, float) for t in temperatures)
    assert all(t >= 0.0 for t in temperatures)
    assert len(set(temperatures)) > 1  # 确保不是固定值

def test_measurement_lifecycle(profiler):
    """测试测量生命周期。"""
    # 开始测量
    profiler.start_measurement()
    assert profiler.is_measuring
    
    # 执行一些计算
    start_time = time.time()
    while time.time() - start_time < 0.1:
        _ = [i * i for i in range(1000)]
    
    # 停止测量
    results = profiler.stop_measurement()
    assert not profiler.is_measuring
    assert isinstance(results, dict)
    assert "runtime" in results
    assert "power" in results
    assert "energy" in results
    assert results["runtime"] >= 0.1
    assert results["power"] >= 0.0
    assert results["energy"] >= 0.0

def test_measure(profiler):
    """测试性能指标测量。"""
    task = {
        "input_tokens": 100,
        "output_tokens": 50
    }
    
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
    
    # 验证吞吐量计算
    total_tokens = task["input_tokens"] + task["output_tokens"]
    expected_throughput = total_tokens / metrics["runtime"]
    assert abs(metrics["throughput"] - expected_throughput) < 0.1
    
    # 验证每令牌能耗计算
    expected_energy_per_token = metrics["energy"] / total_tokens
    assert abs(metrics["energy_per_token"] - expected_energy_per_token) < 0.1

def test_cleanup(profiler):
    """测试资源清理。"""
    # 清理资源
    profiler.cleanup()
    assert not profiler.is_initialized
    assert profiler.handle is None
    assert profiler.nvml is None
    
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
        profiler.measure({"input_tokens": 100, "output_tokens": 50})

def test_error_handling():
    """测试错误处理。"""
    # 测试无效配置
    invalid_configs = [
        {},  # 空配置
        {"device_type": "rtx4050"},  # 缺少 idle_power
        {"idle_power": 15.0},  # 缺少 device_type
        {"sample_interval": 200},  # 缺少 device_type 和 idle_power
        {"device_type": "rtx4050", "idle_power": -1.0},  # 负的 idle_power
        {"device_type": "rtx4050", "sample_interval": 0},  # 零 sample_interval
        {"device_type": "invalid", "idle_power": 15.0},  # 无效的 device_type
        {"device_type": "rtx4050", "idle_power": "15.0"},  # 字符串 idle_power
        {"device_type": "rtx4050", "sample_interval": "200"}  # 字符串 sample_interval
    ]
    
    for config in invalid_configs:
        with pytest.raises((ValueError, TypeError)):
            RTX4050Profiler(config)
    
    # 测试无效任务
    profiler = RTX4050Profiler(TEST_CONFIG)
    profiler.initialize()
    try:
        invalid_tasks = [
            {"input_tokens": -1, "output_tokens": 50},
            {"input_tokens": 100, "output_tokens": -1},
            {"input_tokens": "100", "output_tokens": 50},
            {"input_tokens": 100, "output_tokens": "50"},
            {},
            None
        ]
        
        for task in invalid_tasks:
            with pytest.raises((ValueError, TypeError, KeyError)):
                profiler.measure(task)
    finally:
        profiler.cleanup()

def test_real_mode():
    """测试真实模式。"""
    profiler = RTX4050Profiler(TEST_CONFIG)
    profiler.initialize()
    try:
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
        
        # 测试测量
        task = {
            "input_tokens": 100,
            "output_tokens": 50
        }
        
        metrics = profiler.measure(task)
        assert isinstance(metrics, dict)
        assert metrics["runtime"] > 0
        assert metrics["power"] >= 0.0
        assert metrics["energy"] >= 0.0
        assert metrics["throughput"] > 0
        assert metrics["energy_per_token"] >= 0.0
    finally:
        profiler.cleanup()

def test_config_validation():
    """测试配置验证。"""
    # 测试缺少必要参数
    with pytest.raises(ValueError):
        RTX4050Profiler({})
    
    # 测试无效的 device_type 类型
    invalid_config = TEST_CONFIG.copy()
    invalid_config["device_type"] = 123
    with pytest.raises(TypeError):
        RTX4050Profiler(invalid_config)
    
    # 测试无效的 idle_power 类型
    invalid_config = TEST_CONFIG.copy()
    invalid_config["idle_power"] = "15.0"
    with pytest.raises(TypeError):
        RTX4050Profiler(invalid_config)
    
    # 测试无效的 sample_interval 类型
    invalid_config = TEST_CONFIG.copy()
    invalid_config["sample_interval"] = "200"
    with pytest.raises(TypeError):
        RTX4050Profiler(invalid_config) 