import pytest
import os
from hardware_profiling.rtx4050_profiler import RTX4050Profiler

def test_profiler_initialization():
    """测试分析器初始化"""
    config = {
        "device_id": 0,
        "idle_power": 15.0,
        "sample_interval": 200
    }
    
    # 在测试模式下运行
    os.environ["TEST_MODE"] = "true"
    profiler = RTX4050Profiler(config)
    
    assert profiler.device_id == 0
    assert profiler.idle_power == 15.0
    assert profiler.sample_interval == 0.2  # 200ms转换为秒
    assert profiler.is_test_mode is True

def test_power_measurement():
    """测试功率测量"""
    config = {
        "device_id": 0,
        "idle_power": 15.0,
        "sample_interval": 200
    }
    
    # 在测试模式下运行
    os.environ["TEST_MODE"] = "true"
    profiler = RTX4050Profiler(config)
    
    # 测试模式下应该返回100W
    power = profiler.measure_power()
    assert power == 100.0

def test_invalid_config():
    """测试无效配置"""
    # 测试缺少必要参数
    with pytest.raises(ValueError):
        RTX4050Profiler({})
    
    # 测试无效的设备ID
    config = {
        "device_id": -1,
        "idle_power": 15.0,
        "sample_interval": 200
    }
    with pytest.raises(ValueError):
        RTX4050Profiler(config)

def test_rtx4050_measure(monkeypatch):
    """测试 RTX4050 的测量功能"""
    profiler = RTX4050Profiler(test_mode=True)
    
    def mock_measure(task, input_tokens, output_tokens):
        task()
        return {
            "energy": 8.0,
            "runtime": 1.5,
            "throughput": (input_tokens + output_tokens) / 1.5,
            "energy_per_token": 8.0 / (input_tokens + output_tokens),
            "total_tasks": 1
        }
    
    monkeypatch.setattr(profiler, "measure", mock_measure) 