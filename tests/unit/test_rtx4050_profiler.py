"""RTX4050性能分析器测试模块。"""

import os
import time
import pytest
from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from typing import Dict, Any

# 测试配置
TEST_CONFIG = {
    "device_type": "rtx4050",
    "idle_power": 30.0,
    "sample_interval": 200
}

@pytest.fixture
def profiler():
    """创建 RTX4050Profiler 实例的 fixture。"""
    return RTX4050Profiler(TEST_CONFIG, skip_nvml=True)

def test_initialization(profiler):
    """测试初始化。"""
    assert profiler.device_type == "rtx4050"
    assert profiler.idle_power == 30.0
    assert profiler.sample_interval == 200
    assert profiler.skip_nvml is True
    assert profiler.nvml_initialized is False

def test_measure_power(profiler):
    """测试功率测量。"""
    power = profiler.measure_power()
    assert power == 30.0  # 在测试模式下应该返回空闲功率

def test_get_memory_info(profiler):
    """测试内存信息获取。"""
    memory_info = profiler.get_memory_info()
    assert memory_info["used"] == 1000
    assert memory_info["total"] == 8000

def test_get_temperature(profiler):
    """测试温度获取。"""
    temperature = profiler.get_temperature()
    assert temperature == 50.0  # 在测试模式下应该返回模拟温度

def test_measurement_lifecycle(profiler):
    """测试测量生命周期。"""
    # 开始测量
    profiler.start_measurement()
    assert profiler.start_time is not None
    assert profiler.start_power is not None
    
    # 等待一小段时间
    time.sleep(0.1)
    
    # 结束测量
    profiler.end_measurement()
    assert profiler.start_time is None
    assert profiler.start_power is None

def test_measure(profiler):
    """测试性能指标测量。"""
    def mock_task():
        time.sleep(0.1)
        return "test result"
    
    metrics = profiler.measure(mock_task, input_tokens=100, output_tokens=50)
    
    assert "energy" in metrics
    assert "runtime" in metrics
    assert "throughput" in metrics
    assert "energy_per_token" in metrics
    
    # 在测试模式下，这些值应该是固定的
    assert metrics["energy"] == 10.0
    assert metrics["runtime"] == 2.0
    assert metrics["throughput"] == 75.0  # (100 + 50) / 2
    assert metrics["energy_per_token"] == 0.1

def test_cleanup(profiler):
    """测试资源清理。"""
    profiler.cleanup()
    assert profiler.nvml_initialized is False

def test_error_handling(profiler):
    """测试错误处理。"""
    # 测试空任务
    with pytest.raises(TypeError):
        profiler.measure(None, 100, 50)
    
    # 测试无效的token数量
    with pytest.raises(ValueError):
        profiler.measure(lambda: None, -1, 50)
    with pytest.raises(ValueError):
        profiler.measure(lambda: None, 100, -1)

def test_real_mode():
    """测试真实模式（需要实际硬件支持）。"""
    if os.environ.get("TEST_MODE") == "true":
        pytest.skip("跳过真实模式测试")
    
    real_profiler = RTX4050Profiler(TEST_CONFIG, skip_nvml=False)
    try:
        # 测试基本功能
        power = real_profiler.measure_power()
        assert power >= 0
        
        memory_info = real_profiler.get_memory_info()
        assert memory_info["used"] >= 0
        assert memory_info["total"] > 0
        
        temperature = real_profiler.get_temperature()
        assert temperature >= 0
        
        # 测试性能测量
        def real_task():
            time.sleep(0.1)
            return "test result"
        
        metrics = real_profiler.measure(real_task, 100, 50)
        assert metrics["runtime"] > 0
        assert metrics["energy"] >= 0
        assert metrics["throughput"] > 0
        assert metrics["energy_per_token"] >= 0
        
    finally:
        real_profiler.cleanup()

def test_config_validation():
    """测试配置验证。"""
    # 测试缺少必要配置
    with pytest.raises(ValueError):
        RTX4050Profiler({})
    
    # 测试无效的配置值
    invalid_config = TEST_CONFIG.copy()
    invalid_config["idle_power"] = -1
    with pytest.raises(ValueError):
        RTX4050Profiler(invalid_config)
    
    invalid_config = TEST_CONFIG.copy()
    invalid_config["sample_interval"] = 0
    with pytest.raises(ValueError):
        RTX4050Profiler(invalid_config) 