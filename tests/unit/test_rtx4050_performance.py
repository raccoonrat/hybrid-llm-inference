"""RTX4050 性能测试模块。"""

import pytest
import time
import threading
from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from typing import Dict, Any, List

# 测试配置
HARDWARE_CONFIG = {
    "device_type": "nvidia",
    "idle_power": 10.0,
    "sample_interval": 0.1
}

@pytest.fixture
def profiler():
    """创建 RTX4050Profiler 实例的 fixture。"""
    return RTX4050Profiler(HARDWARE_CONFIG)

def test_measurement_accuracy(profiler):
    """测试测量准确性。"""
    # 测试短时间测量
    task = {
        "input": "Short test input",
        "max_tokens": 5
    }
    
    start_time = time.time()
    metrics = profiler.measure_performance(task)
    end_time = time.time()
    
    assert metrics["runtime"] > 0
    assert abs(metrics["runtime"] - (end_time - start_time)) < 0.1  # 允许 100ms 误差
    
    # 测试长时间测量
    task = {
        "input": "Long test input" * 100,
        "max_tokens": 100
    }
    
    start_time = time.time()
    metrics = profiler.measure_performance(task)
    end_time = time.time()
    
    assert metrics["runtime"] > 0
    assert abs(metrics["runtime"] - (end_time - start_time)) < 0.1  # 允许 100ms 误差

def test_concurrent_measurements(profiler):
    """测试并发测量。"""
    def run_measurement():
        task = {
            "input": "Concurrent test input",
            "max_tokens": 10
        }
        return profiler.measure_performance(task)
    
    # 创建多个线程同时运行测量
    threads = []
    results = []
    
    for _ in range(5):
        thread = threading.Thread(target=lambda: results.append(run_measurement()))
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    # 验证所有测量结果
    assert len(results) == 5
    for result in results:
        assert isinstance(result, dict)
        assert "runtime" in result
        assert "power" in result
        assert "energy" in result
        assert result["runtime"] > 0
        assert result["power"] >= HARDWARE_CONFIG["idle_power"]
        assert result["energy"] > 0

def test_measurement_resources(profiler):
    """测试测量资源使用。"""
    # 获取初始内存使用
    initial_memory = profiler.get_memory_info()["used"]
    
    # 运行多个测量
    for _ in range(10):
        task = {
            "input": "Resource test input",
            "max_tokens": 10
        }
        profiler.measure_performance(task)
    
    # 获取最终内存使用
    final_memory = profiler.get_memory_info()["used"]
    
    # 验证内存使用没有显著增加
    memory_increase = final_memory - initial_memory
    assert memory_increase < 100 * 1024 * 1024  # 允许增加不超过 100MB

def test_long_running_measurement(profiler):
    """测试长时间运行测量。"""
    # 运行长时间测量
    task = {
        "input": "Long running test input" * 1000,
        "max_tokens": 1000
    }
    
    start_time = time.time()
    metrics = profiler.measure_performance(task)
    end_time = time.time()
    
    # 验证长时间测量的准确性
    assert metrics["runtime"] > 0
    assert abs(metrics["runtime"] - (end_time - start_time)) < 0.1  # 允许 100ms 误差
    
    # 验证性能指标
    assert metrics["power"] >= HARDWARE_CONFIG["idle_power"]
    assert metrics["energy"] > 0
    assert metrics["throughput"] > 0
    assert metrics["energy_per_token"] > 0

def test_measurement_boundary_conditions(profiler):
    """测试测量边界条件。"""
    # 测试最小输入
    task = {
        "input": "",
        "max_tokens": 1
    }
    
    metrics = profiler.measure_performance(task)
    assert metrics["runtime"] > 0
    assert metrics["power"] >= HARDWARE_CONFIG["idle_power"]
    
    # 测试最大输入
    task = {
        "input": "A" * 10000,
        "max_tokens": 1000
    }
    
    metrics = profiler.measure_performance(task)
    assert metrics["runtime"] > 0
    assert metrics["power"] >= HARDWARE_CONFIG["idle_power"]
    
    # 测试零 token
    task = {
        "input": "Test input",
        "max_tokens": 0
    }
    
    with pytest.raises(ValueError):
        profiler.measure_performance(task)

def test_measurement_error_handling(profiler):
    """测试测量错误处理。"""
    # 测试无效输入
    invalid_tasks = [
        {"input": 123, "max_tokens": 10},  # 无效的输入类型
        {"input": "Test", "max_tokens": -1},  # 无效的 token 数量
        {"input": None, "max_tokens": 10},  # None 输入
        {"max_tokens": 10},  # 缺少输入
        {"input": "Test"},  # 缺少 token 数量
        None,  # None 任务
        {}  # 空任务
    ]
    
    for task in invalid_tasks:
        with pytest.raises((ValueError, TypeError, KeyError)):
            profiler.measure_performance(task)
    
    # 测试测量状态错误
    profiler.start_measurement()
    with pytest.raises(RuntimeError):
        profiler.start_measurement()  # 重复开始测量
    
    profiler.stop_measurement()
    with pytest.raises(RuntimeError):
        profiler.stop_measurement()  # 重复停止测量

def test_measurement_cleanup(profiler):
    """测试测量资源清理。"""
    # 运行多个测量
    for _ in range(5):
        task = {
            "input": "Cleanup test input",
            "max_tokens": 10
        }
        profiler.measure_performance(task)
    
    # 清理资源
    profiler.cleanup()
    
    # 验证清理后的状态
    assert profiler.handle is None
    assert profiler.nvml is None
    
    # 验证清理后使用
    with pytest.raises(RuntimeError):
        profiler.measure_performance({
            "input": "Test after cleanup",
            "max_tokens": 10
        })

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
    invalid_config = HARDWARE_CONFIG.copy()
    invalid_config["device_type"] = 123
    with pytest.raises(TypeError):
        RTX4050Profiler(invalid_config)
    
    # 测试无效的 idle_power 类型
    invalid_config = HARDWARE_CONFIG.copy()
    invalid_config["idle_power"] = "10.0"
    with pytest.raises(TypeError):
        RTX4050Profiler(invalid_config)
    
    # 测试无效的 sample_interval 类型
    invalid_config = HARDWARE_CONFIG.copy()
    invalid_config["sample_interval"] = "0.1"
    with pytest.raises(TypeError):
        RTX4050Profiler(invalid_config) 