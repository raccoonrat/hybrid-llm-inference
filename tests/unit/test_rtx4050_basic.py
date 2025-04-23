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

def test_rtx4050_measure():
    """测试 RTX4050 功率测量"""
    # 设置测试模式环境变量
    os.environ['TEST_MODE'] = '1'
    
    profiler = RTX4050Profiler(
        device_id=0,
        idle_power=15.0,
        sample_interval=200
    )
    
    # 测试空闲状态功率测量
    power = profiler.measure_power()
    assert power >= 0, "功率测量值不应为负"
    
    # 测试测量方法
    def test_task():
        return "test"
    
    metrics = profiler.measure(test_task, input_tokens=1, output_tokens=1)
    assert "energy" in metrics
    assert "runtime" in metrics
    assert metrics["energy"] >= 0
    assert metrics["runtime"] >= 0 