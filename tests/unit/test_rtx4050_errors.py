import pytest
import os
from hardware_profiling.rtx4050_profiler import RTX4050Profiler

def test_nvml_initialization_failure():
    """测试NVML初始化失败的情况"""
    config = {
        "device_id": 0,
        "idle_power": 15.0,
        "sample_interval": 200
    }
    
    # 不在测试模式下运行
    if "TEST_MODE" in os.environ:
        del os.environ["TEST_MODE"]
    
    # 应该抛出异常
    with pytest.raises(RuntimeError):
        RTX4050Profiler(config)

def test_invalid_device():
    """测试无效设备的情况"""
    config = {
        "device_id": 999,  # 不存在的设备ID
        "idle_power": 15.0,
        "sample_interval": 200
    }
    
    # 在测试模式下运行
    os.environ["TEST_MODE"] = "true"
    
    # 应该抛出异常
    with pytest.raises(ValueError):
        RTX4050Profiler(config)

def test_task_failure():
    """测试任务执行失败的情况"""
    config = {
        "device_id": 0,
        "idle_power": 15.0,
        "sample_interval": 200
    }
    
    os.environ["TEST_MODE"] = "true"
    profiler = RTX4050Profiler(config)
    
    # 定义一个会失败的任务
    def failing_task():
        raise RuntimeError("Task failed")
    
    # 应该抛出异常
    with pytest.raises(RuntimeError):
        profiler.measure(failing_task, input_tokens=10, output_tokens=20)

def test_invalid_power_measurement():
    """测试功率测量失败的情况"""
    config = {
        "device_id": 0,
        "idle_power": 15.0,
        "sample_interval": 200
    }
    
    os.environ["TEST_MODE"] = "true"
    profiler = RTX4050Profiler(config)
    
    # 模拟功率测量失败
    profiler.handle = None
    
    # 在测试模式下应该返回0
    power = profiler.measure_power()
    assert power == 0.0 