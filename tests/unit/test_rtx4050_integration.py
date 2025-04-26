"""RTX4050 集成测试模块。"""

import os
import pytest
import time
import threading
import numpy as np

from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from src.scheduling.task_allocator import TaskAllocator
from src.toolbox.logger import get_logger
from typing import Dict, Any, List

logger = get_logger(__name__)

# 硬件配置
HARDWARE_CONFIG = {
    "nvidia_rtx4050": {
        "device_type": "rtx4050",
        "idle_power": 20.0,
        "sample_interval": 0.1,
        "memory_limit": 6 * 1024 * 1024 * 1024,  # 6GB
        "tdp": 115.0,  # 115W
        "log_level": "DEBUG"
    }
}

# 模型配置
MODEL_CONFIG = {
    "models": {
        "tinyllama": {
            "model_name": "tinyllama",
            "model_path": "path/to/model",
            "mode": "local",
            "batch_size": 1,
            "max_length": 128
        },
        "falcon": {
            "model_name": "falcon",
            "model_path": "path/to/falcon",
            "mode": "local",
            "batch_size": 2,
            "max_length": 256
        }
    }
}

@pytest.fixture(scope="function")
def profiler():
    """创建 RTX4050Profiler 实例的 fixture。"""
    prof = RTX4050Profiler(HARDWARE_CONFIG["nvidia_rtx4050"])
    prof.initialize()
    yield prof
    prof.cleanup()

@pytest.fixture(scope="function")
def allocator():
    """创建 TaskAllocator 实例的 fixture。"""
    alloc = TaskAllocator(HARDWARE_CONFIG, MODEL_CONFIG)
    yield alloc
    alloc.cleanup()

def test_system_initialization(profiler, allocator):
    """测试系统初始化。"""
    # 验证分析器初始化
    assert profiler.is_initialized
    assert profiler.config == HARDWARE_CONFIG["nvidia_rtx4050"]
    
    # 验证分配器初始化
    assert allocator.hardware_config == HARDWARE_CONFIG
    assert allocator.model_config == MODEL_CONFIG
    
    # 验证基本功能
    power = profiler.measure_power()
    assert isinstance(power, float)
    assert power >= 0.0
    
    memory_info = profiler.get_memory_info()
    assert isinstance(memory_info, dict)
    assert all(key in memory_info for key in ["total", "used", "free"])
    assert memory_info["total"] > 0
    assert memory_info["used"] >= 0
    assert memory_info["free"] > 0
    
    temperature = profiler.get_temperature()
    assert isinstance(temperature, float)
    assert temperature >= 0.0

def test_task_allocation_and_execution(profiler, allocator):
    """测试任务分配和执行。"""
    tasks = [
        {
            "input_tokens": 100,
            "output_tokens": 50,
            "model": "tinyllama",
            "priority": "high"
        },
        {
            "input_tokens": 200,
            "output_tokens": 100,
            "model": "falcon",
            "priority": "normal"
        }
    ]
    
    for task in tasks:
        # 分配任务
        allocation = allocator.allocate(task)
        assert allocation["hardware"] == "nvidia_rtx4050"
        assert allocation["model"] == task["model"]
        assert allocation["input_tokens"] == task["input_tokens"]
        assert allocation["output_tokens"] == task["output_tokens"]
        
        # 执行任务并测量性能
        def execute_task():
            time.sleep(0.1)  # 模拟任务执行
            return "result"
        
        metrics = profiler.measure(execute_task, 
                                 input_tokens=task["input_tokens"],
                                 output_tokens=task["output_tokens"])
        
        # 验证性能指标
        assert metrics["runtime"] > 0
        assert metrics["power"] >= HARDWARE_CONFIG["nvidia_rtx4050"]["idle_power"]
        assert metrics["power"] <= HARDWARE_CONFIG["nvidia_rtx4050"]["tdp"]
        assert metrics["energy"] > 0
        assert metrics["throughput"] > 0

def test_concurrent_task_processing(profiler, allocator):
    """测试并发任务处理。"""
    def process_task(task_config):
        try:
            # 分配任务
            allocation = allocator.allocate(task_config)
            assert allocation["hardware"] == "nvidia_rtx4050"
            
            # 执行任务
            def task():
                time.sleep(0.1)  # 模拟计算
                return "result"
            
            metrics = profiler.measure(task,
                                     input_tokens=task_config["input_tokens"],
                                     output_tokens=task_config["output_tokens"])
            
            return metrics
        except Exception as e:
            logger.error(f"任务处理失败: {str(e)}")
            raise
    
    # 创建多个并发任务
    tasks = [
        {"input_tokens": 100, "output_tokens": 50, "model": "tinyllama"},
        {"input_tokens": 150, "output_tokens": 75, "model": "falcon"},
        {"input_tokens": 200, "output_tokens": 100, "model": "tinyllama"}
    ]
    
    # 并发执行任务
    threads = []
    results = []
    
    def run_task(task):
        try:
            result = process_task(task)
            results.append(result)
        except Exception as e:
            logger.error(f"线程执行失败: {str(e)}")
            raise
    
    for task in tasks:
        thread = threading.Thread(target=run_task, args=(task,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # 验证结果
    assert len(results) == len(tasks)
    for metrics in results:
        assert metrics["runtime"] > 0
        assert metrics["power"] > 0
        assert metrics["throughput"] > 0

def test_resource_management_under_load(profiler, allocator):
    """测试负载下的资源管理。"""
    initial_memory = profiler.get_memory_info()["used"]
    initial_power = profiler.measure_power()
    
    # 创建高负载任务
    def heavy_task():
        # 执行计算密集型操作
        matrix_size = 1000
        matrix_a = np.random.rand(matrix_size, matrix_size)
        matrix_b = np.random.rand(matrix_size, matrix_size)
        result = np.dot(matrix_a, matrix_b)
        time.sleep(0.1)
        return result
    
    # 执行多个任务
    for _ in range(5):
        task = {
            "input_tokens": 500,
            "output_tokens": 250,
            "model": "tinyllama"
        }
        
        allocation = allocator.allocate(task)
        metrics = profiler.measure(heavy_task,
                                 input_tokens=task["input_tokens"],
                                 output_tokens=task["output_tokens"])
        
        # 检查资源使用
        current_memory = profiler.get_memory_info()["used"]
        current_power = profiler.measure_power()
        
        # 验证资源约束
        assert current_memory - initial_memory < 2 * 1024 * 1024 * 1024  # 内存增长不超过2GB
        assert current_power <= HARDWARE_CONFIG["nvidia_rtx4050"]["tdp"]
        assert metrics["throughput"] > 0

def test_system_stability(profiler, allocator):
    """测试系统稳定性。"""
    start_time = time.time()
    test_duration = 60  # 1分钟测试
    
    # 记录初始状态
    initial_memory = profiler.get_memory_info()["used"]
    initial_power = profiler.measure_power()
    
    # 持续执行任务
    while time.time() - start_time < test_duration:
        task = {
            "input_tokens": 100,
            "output_tokens": 50,
            "model": "tinyllama"
        }
        
        allocation = allocator.allocate(task)
        metrics = profiler.measure(lambda: time.sleep(0.1),
                                 input_tokens=task["input_tokens"],
                                 output_tokens=task["output_tokens"])
        
        # 验证系统状态
        current_memory = profiler.get_memory_info()["used"]
        current_power = profiler.measure_power()
        
        assert current_memory - initial_memory < 1 * 1024 * 1024 * 1024  # 内存增长不超过1GB
        assert current_power <= HARDWARE_CONFIG["nvidia_rtx4050"]["tdp"]
        assert metrics["runtime"] > 0
        assert metrics["throughput"] > 0

def test_error_handling_and_recovery(profiler, allocator):
    """测试错误处理和恢复。"""
    def error_task():
        raise RuntimeError("模拟任务错误")
    
    def normal_task():
        time.sleep(0.1)
        return "result"
    
    # 测试错误任务
    task = {
        "input_tokens": 100,
        "output_tokens": 50,
        "model": "tinyllama"
    }
    
    allocation = allocator.allocate(task)
    
    # 执行错误任务
    with pytest.raises(RuntimeError):
        profiler.measure(error_task,
                        input_tokens=task["input_tokens"],
                        output_tokens=task["output_tokens"])
    
    # 验证系统恢复
    metrics = profiler.measure(normal_task,
                             input_tokens=task["input_tokens"],
                             output_tokens=task["output_tokens"])
    
    assert metrics["runtime"] > 0
    assert metrics["power"] > 0
    assert metrics["throughput"] > 0

def test_system_cleanup(profiler, allocator):
    """测试系统清理。"""
    # 执行一些任务
    for _ in range(3):
        task = {
            "input_tokens": 100,
            "output_tokens": 50,
            "model": "tinyllama"
        }
        
        allocation = allocator.allocate(task)
        profiler.measure(lambda: time.sleep(0.1),
                        input_tokens=task["input_tokens"],
                        output_tokens=task["output_tokens"])
    
    # 执行清理
    profiler.cleanup()
    
    # 验证清理后的状态
    assert not profiler.is_initialized
    
    # 验证无法执行新任务
    with pytest.raises(RuntimeError):
        profiler.measure(lambda: "result",
                        input_tokens=10,
                        output_tokens=10)

def test_performance_monitoring(profiler, allocator):
    """测试性能监控。"""
    metrics_history = []
    
    # 执行一系列任务并收集性能指标
    for _ in range(10):
        task = {
            "input_tokens": 100,
            "output_tokens": 50,
            "model": "tinyllama"
        }
        
        allocation = allocator.allocate(task)
        metrics = profiler.measure(lambda: time.sleep(0.1),
                                 input_tokens=task["input_tokens"],
                                 output_tokens=task["output_tokens"])
        metrics_history.append(metrics)
    
    # 分析性能趋势
    runtimes = [m["runtime"] for m in metrics_history]
    powers = [m["power"] for m in metrics_history]
    throughputs = [m["throughput"] for m in metrics_history]
    
    # 计算统计指标
    avg_runtime = np.mean(runtimes)
    avg_power = np.mean(powers)
    avg_throughput = np.mean(throughputs)
    
    std_runtime = np.std(runtimes)
    std_power = np.std(powers)
    std_throughput = np.std(throughputs)
    
    # 验证性能稳定性
    assert std_runtime / avg_runtime < 0.2  # 运行时间变异系数小于20%
    assert std_power / avg_power < 0.2  # 功率变异系数小于20%
    assert std_throughput / avg_throughput < 0.2  # 吞吐量变异系数小于20% 