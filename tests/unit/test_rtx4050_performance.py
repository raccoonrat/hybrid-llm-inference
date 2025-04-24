"""RTX4050 性能测试模块。"""

import pytest
import time
import threading
from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from typing import Dict, Any, List

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

def test_measurement_accuracy(profiler):
    """测试测量准确性。"""
    # 测试功率测量
    power = profiler.measure_power()
    assert isinstance(power, float)
    assert power >= 0.0
    
    # 测试多次测量的一致性
    powers = [profiler.measure_power() for _ in range(10)]
    assert all(isinstance(p, float) for p in powers)
    assert all(p >= 0.0 for p in powers)
    assert len(set(powers)) > 1  # 确保不是固定值
    
    # 测试内存信息
    memory_info = profiler.get_memory_info()
    assert isinstance(memory_info, dict)
    assert "total" in memory_info
    assert "used" in memory_info
    assert "free" in memory_info
    assert memory_info["total"] > 0
    assert memory_info["used"] >= 0
    assert memory_info["free"] >= 0
    assert memory_info["total"] == memory_info["used"] + memory_info["free"]
    
    # 测试温度
    temperature = profiler.get_temperature()
    assert isinstance(temperature, float)
    assert temperature >= 0.0

def test_concurrent_measurements(profiler):
    """测试并发测量。"""
    def measure_task():
        try:
            profiler.start_measurement()
            time.sleep(0.1)
            results = profiler.stop_measurement()
            assert isinstance(results, dict)
            assert "runtime" in results
            assert "power" in results
            assert "energy" in results
            assert results["runtime"] >= 0.1
            assert results["power"] >= 0.0
            assert results["energy"] >= 0.0
        except Exception as e:
            pytest.fail(f"并发测量失败: {str(e)}")
    
    # 创建多个线程进行并发测量
    threads = [threading.Thread(target=measure_task) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

def test_resource_usage(profiler):
    """测试资源使用情况。"""
    # 测试内存使用
    initial_memory = profiler.get_memory_info()
    
    # 执行一些计算密集型任务
    profiler.start_measurement()
    try:
        start_time = time.time()
        while time.time() - start_time < 0.1:
            _ = [i * i for i in range(1000)]
    finally:
        profiler.stop_measurement()
    
    # 验证内存使用变化
    final_memory = profiler.get_memory_info()
    assert final_memory["total"] == initial_memory["total"]
    assert final_memory["used"] >= initial_memory["used"]
    assert final_memory["free"] <= initial_memory["free"]

def test_long_running_measurements(profiler):
    """测试长时间运行测量。"""
    # 开始长时间测量
    profiler.start_measurement()
    try:
        # 执行长时间任务
        start_time = time.time()
        while time.time() - start_time < 1.0:
            _ = [i * i for i in range(1000)]
            time.sleep(0.1)
    finally:
        results = profiler.stop_measurement()
    
    # 验证结果
    assert isinstance(results, dict)
    assert "runtime" in results
    assert "power" in results
    assert "energy" in results
    assert results["runtime"] >= 1.0
    assert results["power"] >= 0.0
    assert results["energy"] >= 0.0

def test_boundary_conditions(profiler):
    """测试边界条件。"""
    # 测试极小任务
    tiny_task = {
        "input_tokens": 1,
        "output_tokens": 1
    }
    
    metrics = profiler.measure(tiny_task)
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
    
    # 测试极大任务
    huge_task = {
        "input_tokens": 1000000,
        "output_tokens": 100000
    }
    
    metrics = profiler.measure(huge_task)
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

def test_cleanup(profiler):
    """测试资源清理。"""
    # 清理资源
    profiler.cleanup()
    
    # 验证资源已被清理
    assert not profiler.is_initialized
    
    # 尝试使用已清理的分析器
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