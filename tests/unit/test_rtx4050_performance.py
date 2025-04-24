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
    return RTX4050Profiler(TEST_CONFIG)

def test_measurement_accuracy(profiler):
    """测试测量准确性。"""
    # 创建不同规模的任务
    tasks = [
        {"input_tokens": 100, "output_tokens": 50},
        {"input_tokens": 1000, "output_tokens": 100},
        {"input_tokens": 10000, "output_tokens": 1000}
    ]
    
    for task in tasks:
        # 开始测量
        profiler.start_measurement()
        try:
            # 执行任务
            metrics = profiler.measure(task)
            
            # 验证指标
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
        finally:
            profiler.stop_measurement()

def test_concurrent_measurements(profiler):
    """测试并发测量。"""
    def measure_task(task: Dict[str, int]) -> Dict[str, float]:
        """执行测量任务。"""
        profiler.start_measurement()
        try:
            return profiler.measure(task)
        finally:
            profiler.stop_measurement()
    
    # 创建多个任务
    tasks = [
        {"input_tokens": 100, "output_tokens": 50},
        {"input_tokens": 200, "output_tokens": 100},
        {"input_tokens": 300, "output_tokens": 150}
    ]
    
    # 创建线程
    threads = []
    results = []
    
    for task in tasks:
        thread = threading.Thread(
            target=lambda t: results.append(measure_task(t)),
            args=(task,)
        )
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    # 验证结果
    assert len(results) == len(tasks)
    for metrics in results:
        assert metrics["runtime"] > 0
        assert metrics["power"] >= 0.0
        assert metrics["energy"] >= 0.0
        assert metrics["throughput"] > 0
        assert metrics["energy_per_token"] >= 0.0

def test_measurement_resources(profiler):
    """测试测量资源使用。"""
    # 获取初始内存使用
    initial_memory = profiler.get_memory_info()["used"]
    
    # 执行多个测量
    for _ in range(10):
        profiler.start_measurement()
        try:
            task = {
                "input_tokens": 100,
                "output_tokens": 50
            }
            profiler.measure(task)
        finally:
            profiler.stop_measurement()
    
    # 获取最终内存使用
    final_memory = profiler.get_memory_info()["used"]
    
    # 验证内存使用没有显著增加
    memory_increase = final_memory - initial_memory
    assert memory_increase < 1024 * 1024  # 小于 1MB

def test_long_running_measurement(profiler):
    """测试长时间运行测量。"""
    # 开始测量
    profiler.start_measurement()
    try:
        # 执行长时间运行的任务
        start_time = time.time()
        while time.time() - start_time < 1.0:  # 运行 1 秒
            task = {
                "input_tokens": 100,
                "output_tokens": 50
            }
            metrics = profiler.measure(task)
            
            # 验证指标
            assert metrics["runtime"] > 0
            assert metrics["power"] >= 0.0
            assert metrics["energy"] >= 0.0
            assert metrics["throughput"] > 0
            assert metrics["energy_per_token"] >= 0.0
    finally:
        profiler.stop_measurement()

def test_measurement_boundary_conditions(profiler):
    """测试测量边界条件。"""
    # 测试极小任务
    tiny_task = {
        "input_tokens": 1,
        "output_tokens": 1
    }
    
    profiler.start_measurement()
    try:
        metrics = profiler.measure(tiny_task)
        assert metrics["runtime"] > 0
        assert metrics["power"] >= 0.0
        assert metrics["energy"] >= 0.0
        assert metrics["throughput"] > 0
        assert metrics["energy_per_token"] >= 0.0
    finally:
        profiler.stop_measurement()
    
    # 测试极大任务
    huge_task = {
        "input_tokens": 1000000,
        "output_tokens": 100000
    }
    
    profiler.start_measurement()
    try:
        metrics = profiler.measure(huge_task)
        assert metrics["runtime"] > 0
        assert metrics["power"] >= 0.0
        assert metrics["energy"] >= 0.0
        assert metrics["throughput"] > 0
        assert metrics["energy_per_token"] >= 0.0
    finally:
        profiler.stop_measurement()

def test_measurement_error_handling(profiler):
    """测试测量错误处理。"""
    # 测试无效任务
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
    
    # 测试未开始测量
    task = {
        "input_tokens": 100,
        "output_tokens": 50
    }
    with pytest.raises(RuntimeError):
        profiler.measure(task)
    
    # 测试重复开始测量
    profiler.start_measurement()
    with pytest.raises(RuntimeError):
        profiler.start_measurement()
    profiler.stop_measurement()
    
    # 测试重复停止测量
    profiler.start_measurement()
    profiler.stop_measurement()
    with pytest.raises(RuntimeError):
        profiler.stop_measurement()

def test_measurement_cleanup(profiler):
    """测试测量资源清理。"""
    # 执行多个测量
    for _ in range(5):
        profiler.start_measurement()
        try:
            task = {
                "input_tokens": 100,
                "output_tokens": 50
            }
            profiler.measure(task)
        finally:
            profiler.stop_measurement()
    
    # 清理资源
    profiler.cleanup()
    
    # 验证资源已被清理
    assert profiler.handle is None
    assert profiler.nvml is None
    
    # 尝试使用已清理的分析器
    with pytest.raises(RuntimeError):
        profiler.start_measurement()
    
    with pytest.raises(RuntimeError):
        profiler.stop_measurement()
    
    with pytest.raises(RuntimeError):
        profiler.measure({"input_tokens": 100, "output_tokens": 50}) 