"""RTX4050性能测试。"""

import os
import pytest
import time
import numpy as np
import pynvml
from unittest.mock import patch, MagicMock

from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from src.toolbox.logger import get_logger

logger = get_logger(__name__)

# 测试配置
TEST_CONFIG = {
    "device_id": 0,
    "device_type": "RTX4050",
    "idle_power": 20.0,
    "sample_interval": 0.1,
    "memory_limit": 6 * 1024 * 1024 * 1024,  # 6GB
    "tdp": 115.0,  # 115W
    "log_level": "DEBUG"
}

@pytest.fixture(scope="function")
def profiler():
    """创建RTX4050Profiler实例。"""
    prof = RTX4050Profiler(TEST_CONFIG)
    yield prof
    prof.cleanup()

@pytest.fixture(scope="function")
def mock_nvml():
    """模拟NVML交互。"""
    with patch("pynvml.nvmlInit") as mock_init, \
         patch("pynvml.nvmlDeviceGetHandleByIndex") as mock_get_handle, \
         patch("pynvml.nvmlDeviceGetName") as mock_get_name, \
         patch("pynvml.nvmlDeviceGetPowerUsage") as mock_get_power, \
         patch("pynvml.nvmlDeviceGetMemoryInfo") as mock_get_memory, \
         patch("pynvml.nvmlDeviceGetUtilizationRates") as mock_get_util, \
         patch("pynvml.nvmlShutdown") as mock_shutdown:
        
        # 设置模拟返回值
        mock_handle = MagicMock()
        mock_get_handle.return_value = mock_handle
        mock_get_name.return_value = b"NVIDIA GeForce RTX 4050"
        mock_get_power.return_value = 50000  # 50W in milliwatts
        
        class MemoryInfo:
            def __init__(self):
                self.total = 6 * 1024 * 1024 * 1024  # 6GB
                self.used = 2 * 1024 * 1024 * 1024   # 2GB
                self.free = 4 * 1024 * 1024 * 1024   # 4GB
        mock_get_memory.return_value = MemoryInfo()
        
        class UtilizationRates:
            def __init__(self):
                self.gpu = 75
                self.memory = 50
        mock_get_util.return_value = UtilizationRates()
        
        yield {
            "init": mock_init,
            "get_handle": mock_get_handle,
            "get_name": mock_get_name,
            "get_power": mock_get_power,
            "get_memory": mock_get_memory,
            "get_util": mock_get_util,
            "shutdown": mock_shutdown,
            "handle": mock_handle
        }

def test_long_running_measurement(profiler, mock_nvml):
    """测试长时间运行的性能测量。"""
    # 启动监控
    profiler.start_monitoring()
    
    # 模拟长时间运行
    time.sleep(1.0)
    
    # 停止监控并获取指标
    metrics = profiler.stop_monitoring()
    
    # 验证指标
    assert isinstance(metrics, dict)
    assert all(key in metrics for key in ["power", "memory", "utilization"])
    assert all(isinstance(metrics[key], list) for key in metrics)
    assert all(len(metrics[key]) > 0 for key in metrics)
    
    # 验证统计数据
    stats = profiler.get_statistics()
    assert isinstance(stats, dict)
    assert all(key in stats for key in ["avg_power", "max_power", "min_power"])
    assert all(isinstance(stats[key], float) for key in stats)

def test_high_load_performance(profiler, mock_nvml):
    """测试高负载性能。"""
    def high_load_task():
        # 创建大型矩阵
        matrix_size = 1000
        matrix_a = np.random.rand(matrix_size, matrix_size)
        matrix_b = np.random.rand(matrix_size, matrix_size)
        # 执行矩阵乘法
        result = np.dot(matrix_a, matrix_b)
        return result
    
    # 测量性能
    metrics = profiler.measure(high_load_task, input_tokens=1000, output_tokens=1000)
    
    # 验证性能指标
    assert metrics["runtime"] > 0
    assert metrics["power"] > TEST_CONFIG["idle_power"]
    assert metrics["energy"] > 0
    assert metrics["throughput"] > 0

def test_memory_intensive_performance(profiler, mock_nvml):
    """测试内存密集型性能。"""
    def memory_intensive_task():
        # 分配大量内存
        size = 1024 * 1024 * 1024  # 1GB
        data = np.zeros(size, dtype=np.float32)
        # 执行一些操作
        data += 1.0
        return data
    
    # 测量性能
    metrics = profiler.measure(memory_intensive_task, input_tokens=1000, output_tokens=1000)
    
    # 验证性能指标
    assert metrics["runtime"] > 0
    assert metrics["memory_usage"] > 0
    assert metrics["throughput"] > 0

def test_performance_stability(profiler, mock_nvml):
    """测试性能稳定性。"""
    results = []
    
    # 执行多次测量
    for _ in range(10):
        metrics = profiler.measure(lambda: time.sleep(0.1), input_tokens=100, output_tokens=100)
        results.append(metrics)
    
    # 计算统计指标
    runtimes = [r["runtime"] for r in results]
    powers = [r["power"] for r in results]
    throughputs = [r["throughput"] for r in results]
    
    # 验证稳定性
    assert np.std(runtimes) / np.mean(runtimes) < 0.2  # 运行时间变异系数小于20%
    assert np.std(powers) / np.mean(powers) < 0.2  # 功率变异系数小于20%
    assert np.std(throughputs) / np.mean(throughputs) < 0.2  # 吞吐量变异系数小于20%

def test_performance_under_stress(profiler, mock_nvml):
    """测试压力下的性能。"""
    def stress_task():
        # 执行计算密集型操作
        result = 0
        for i in range(1000000):
            result += i * i
        return result
    
    # 在监控下执行任务
    profiler.start_monitoring()
    result = stress_task()
    metrics = profiler.stop_monitoring()
    
    # 验证结果
    assert result > 0
    assert metrics["power"] > TEST_CONFIG["idle_power"]
    assert metrics["gpu_utilization"] > 0
    assert metrics["memory_usage"] > 0

def test_concurrent_measurements(profiler):
    """测试并发测量。"""
    def mock_task():
        time.sleep(0.1)
        return "result"
    
    results = []
    def run_measurement():
        metrics = profiler.measure(mock_task, input_tokens=10, output_tokens=10)
        results.append(metrics)
    
    # 创建多个线程并发执行
    threads = [threading.Thread(target=run_measurement) for _ in range(5)]
    start_time = time.time()
    
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 验证结果
    assert len(results) == 5
    for metrics in results:
        assert metrics["runtime"] >= 0.1
        assert metrics["energy"] > 0
        assert metrics["throughput"] > 0
    
    # 验证并发效率
    avg_runtime = np.mean([m["runtime"] for m in results])
    assert total_time < avg_runtime * 5  # 验证并发执行效率

def test_long_running_measurements(profiler, mock_nvml):
    """测试长时间运行的测量。"""
    def long_task():
        time.sleep(1.0)
        return "result"
    
    metrics = profiler.measure(long_task, input_tokens=100, output_tokens=100)
    assert metrics["runtime"] >= 1.0
    assert metrics["energy"] > 0
    assert metrics["throughput"] > 0
    assert metrics["energy_per_token"] > 0

def test_high_load_measurements(profiler, mock_nvml):
    """测试高负载测量。"""
    def high_load_task():
        # 模拟高负载计算
        for _ in range(1000000):
            pass
        return "result"
    
    metrics = profiler.measure(high_load_task, input_tokens=1000, output_tokens=1000)
    assert metrics["runtime"] > 0
    assert metrics["energy"] > 0
    assert metrics["throughput"] > 0
    assert metrics["energy_per_token"] > 0

def test_memory_intensive_measurements(profiler):
    """测试内存密集型测量。"""
    def memory_intensive_task():
        # 分配大量内存
        data = [np.random.rand(1000, 1000) for _ in range(5)]
        time.sleep(0.1)
        return "result"
    
    # 记录初始内存
    initial_memory = profiler.get_memory_info()["used"]
    
    # 执行测量
    metrics = profiler.measure(memory_intensive_task, input_tokens=100, output_tokens=100)
    
    # 验证内存使用
    final_memory = profiler.get_memory_info()["used"]
    memory_diff = final_memory - initial_memory
    
    assert metrics["runtime"] >= 0.1
    assert metrics["energy"] > 0
    assert metrics["throughput"] > 0
    assert memory_diff < 1 * 1024 * 1024 * 1024  # 内存增长不超过1GB

def test_mixed_workload_measurements(profiler):
    """测试混合工作负载测量。"""
    def mixed_workload_task():
        # CPU密集型操作
        for _ in range(100000):
            _ = np.random.rand() * np.random.rand()
        
        # 内存操作
        data = [np.random.rand(100, 100) for _ in range(10)]
        
        # IO操作
        time.sleep(0.1)
        return "result"
    
    metrics = profiler.measure(mixed_workload_task, input_tokens=50, output_tokens=50)
    
    # 验证性能指标
    assert metrics["runtime"] > 0
    assert metrics["energy"] > 0
    assert metrics["throughput"] > 0
    assert metrics["energy_per_token"] > 0
    
    # 验证资源使用
    assert profiler.get_memory_info()["used"] > 0
    assert profiler.measure_power() > TEST_CONFIG["idle_power"]

def test_measurement_accuracy(profiler, mock_nvml):
    """测试测量准确性。"""
    def accurate_task():
        time.sleep(0.5)
        return "result"
    
    # 多次测量取平均值
    measurements = []
    for _ in range(10):
        metrics = profiler.measure(accurate_task, input_tokens=10, output_tokens=10)
        measurements.append(metrics["runtime"])
    
    avg_runtime = sum(measurements) / len(measurements)
    assert abs(avg_runtime - 0.5) < 0.1  # 允许10%的误差

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

def test_measurement_boundary_conditions(profiler):
    """测试测量边界条件。"""
    # 测试最小输入
    task = {
        "input": "",
        "max_tokens": 1
    }
    
    metrics = profiler.measure_performance(task)
    assert metrics["runtime"] > 0
    assert metrics["power"] >= TEST_CONFIG["idle_power"]
    
    # 测试最大输入
    task = {
        "input": "A" * 10000,
        "max_tokens": 1000
    }
    
    metrics = profiler.measure_performance(task)
    assert metrics["runtime"] > 0
    assert metrics["power"] >= TEST_CONFIG["idle_power"]
    
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
    invalid_config["idle_power"] = "10.0"
    with pytest.raises(TypeError):
        RTX4050Profiler(invalid_config)
    
    # 测试无效的 sample_interval 类型
    invalid_config = TEST_CONFIG.copy()
    invalid_config["sample_interval"] = "0.1"
    with pytest.raises(TypeError):
        RTX4050Profiler(invalid_config)

def test_performance_benchmark(profiler):
    """测试性能基准。"""
    # 定义不同大小的任务
    tasks = [
        {"input": "Short input", "max_tokens": 10},
        {"input": "Medium input" * 10, "max_tokens": 100},
        {"input": "Long input" * 100, "max_tokens": 1000}
    ]
    
    # 运行基准测试
    results = []
    for task in tasks:
        metrics = profiler.measure_performance(task)
        results.append(metrics)
    
    # 验证基准测试结果
    assert len(results) == 3
    for result in results:
        assert isinstance(result, dict)
        assert "runtime" in result
        assert "power" in result
        assert "energy" in result
        assert "throughput" in result
        assert "energy_per_token" in result
    
    # 验证性能趋势
    assert results[0]["runtime"] < results[1]["runtime"] < results[2]["runtime"]
    assert results[0]["energy"] < results[1]["energy"] < results[2]["energy"]
    assert results[0]["throughput"] > results[1]["throughput"] > results[2]["throughput"]

def test_resource_utilization(profiler):
    """测试资源利用率。"""
    # 运行多个任务并监控资源使用
    tasks = [
        {"input": "Task 1", "max_tokens": 10},
        {"input": "Task 2", "max_tokens": 20},
        {"input": "Task 3", "max_tokens": 30}
    ]
    
    # 开始监控
    profiler.start_monitoring()
    
    # 执行任务
    for task in tasks:
        profiler.measure_performance(task)
        time.sleep(0.1)
    
    # 停止监控并获取结果
    metrics = profiler.stop_monitoring()
    
    # 验证监控结果
    assert isinstance(metrics, dict)
    assert "energy_consumption" in metrics
    assert "runtime" in metrics
    assert "avg_power" in metrics
    assert "avg_memory" in metrics
    assert "avg_utilization" in metrics
    
    # 验证资源使用趋势
    assert metrics["avg_power"] >= TEST_CONFIG["idle_power"]
    assert metrics["avg_memory"] >= 0
    assert metrics["avg_utilization"] >= 0

def test_stress_test(profiler):
    """测试压力测试。"""
    # 创建大量并发任务
    tasks = [
        {"input": f"Stress test input {i}", "max_tokens": 10}
        for i in range(100)
    ]
    
    # 运行压力测试
    results = []
    for task in tasks:
        metrics = profiler.measure_performance(task)
        results.append(metrics)
    
    # 验证压力测试结果
    assert len(results) == 100
    assert all(isinstance(r, dict) for r in results)
    assert all(r["runtime"] > 0 for r in results)
    assert all(r["power"] >= TEST_CONFIG["idle_power"] for r in results)
    assert all(r["energy"] > 0 for r in results)
    
    # 验证性能稳定性
    runtimes = [r["runtime"] for r in results]
    assert np.std(runtimes) < np.mean(runtimes) * 0.5  # 运行时标准差小于平均值的50%

def test_energy_efficiency(profiler):
    """测试能效。"""
    # 测试不同大小的任务
    tasks = [
        {"input": "Small task", "max_tokens": 10},
        {"input": "Medium task" * 10, "max_tokens": 100},
        {"input": "Large task" * 100, "max_tokens": 1000}
    ]
    
    # 测量能效
    efficiencies = []
    for task in tasks:
        metrics = profiler.measure_performance(task)
        efficiency = metrics["energy_per_token"]
        efficiencies.append(efficiency)
    
    # 验证能效趋势
    assert efficiencies[0] > efficiencies[1] > efficiencies[2]  # 大任务通常更高效

def test_performance_metrics_validation(profiler):
    """测试性能指标验证。"""
    def validation_task():
        time.sleep(0.1)
        return "result"
    
    # 收集多次测量结果
    metrics_list = []
    for _ in range(10):
        metrics = profiler.measure(validation_task, input_tokens=100, output_tokens=100)
        metrics_list.append(metrics)
    
    # 计算统计指标
    runtimes = [m["runtime"] for m in metrics_list]
    powers = [m["power"] for m in metrics_list]
    throughputs = [m["throughput"] for m in metrics_list]
    
    # 验证指标稳定性
    assert np.std(runtimes) / np.mean(runtimes) < 0.2  # 运行时间变异系数小于20%
    assert np.std(powers) / np.mean(powers) < 0.2  # 功率变异系数小于20%
    assert np.std(throughputs) / np.mean(throughputs) < 0.2  # 吞吐量变异系数小于20%
    
    # 验证指标合理性
    assert np.mean(powers) >= TEST_CONFIG["idle_power"]
    assert np.mean(powers) <= TEST_CONFIG["tdp"]
    assert np.mean(throughputs) > 0

def test_resource_efficiency(profiler):
    """测试资源使用效率。"""
    def efficiency_task():
        # 执行计算密集型操作
        matrix_size = 500
        matrix_a = np.random.rand(matrix_size, matrix_size)
        matrix_b = np.random.rand(matrix_size, matrix_size)
        result = np.dot(matrix_a, matrix_b)
        return result
    
    # 记录初始状态
    initial_memory = profiler.get_memory_info()["used"]
    initial_power = profiler.measure_power()
    
    # 执行多次测量
    for _ in range(5):
        metrics = profiler.measure(efficiency_task, input_tokens=1000, output_tokens=1000)
        
        # 验证资源使用效率
        current_memory = profiler.get_memory_info()["used"]
        current_power = profiler.measure_power()
        
        # 验证内存效率
        memory_increase = current_memory - initial_memory
        assert memory_increase < 500 * 1024 * 1024  # 内存增长不超过500MB
        
        # 验证功率效率
        power_increase = current_power - initial_power
        assert power_increase >= 0  # 功率应该增加
        assert current_power <= TEST_CONFIG["tdp"]  # 不超过TDP
        
        # 验证性能效率
        assert metrics["throughput"] > 0
        assert metrics["energy_per_token"] > 0

def test_performance_stability(profiler):
    """测试性能稳定性。"""
    def stability_task():
        time.sleep(0.1)
        return "result"
    
    # 执行长时间测试
    start_time = time.time()
    end_time = start_time + 60  # 1分钟测试
    
    metrics_list = []
    while time.time() < end_time:
        metrics = profiler.measure(stability_task, input_tokens=100, output_tokens=100)
        metrics_list.append(metrics)
        
        # 验证实时性能
        assert metrics["runtime"] >= 0.1
        assert metrics["power"] >= TEST_CONFIG["idle_power"]
        assert metrics["power"] <= TEST_CONFIG["tdp"]
        assert metrics["throughput"] > 0
    
    # 分析性能趋势
    runtimes = [m["runtime"] for m in metrics_list]
    powers = [m["power"] for m in metrics_list]
    throughputs = [m["throughput"] for m in metrics_list]
    
    # 验证性能没有显著退化
    runtime_trend = np.polyfit(range(len(runtimes)), runtimes, 1)[0]
    throughput_trend = np.polyfit(range(len(throughputs)), throughputs, 1)[0]
    
    assert abs(runtime_trend) < 0.001  # 运行时间趋势应该很小
    assert abs(throughput_trend) < 0.1  # 吞吐量趋势应该很小 