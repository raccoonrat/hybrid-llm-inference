"""RTX4050 稳定性和基准测试模块。"""

import pytest
import time
import threading
import psutil
import numpy as np
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
    "tdp": 115.0  # 115W
}

@pytest.fixture
def profiler():
    """创建RTX4050Profiler实例。"""
    profiler = RTX4050Profiler(TEST_CONFIG)
    yield profiler
    profiler.cleanup()

def test_performance_baseline(profiler):
    """测试基准性能表现"""
    # 定义基准任务
    def benchmark_task():
        # 模拟标准计算负载
        matrix_size = 1000
        matrix_a = np.random.rand(matrix_size, matrix_size)
        matrix_b = np.random.rand(matrix_size, matrix_size)
        result = np.dot(matrix_a, matrix_b)
        return result
    
    # 收集性能指标
    metrics_list = []
    for _ in range(10):  # 运行10次获取平均值
        metrics = profiler.measure(benchmark_task, input_tokens=1000, output_tokens=1000)
        metrics_list.append(metrics)
    
    # 计算平均值和标准差
    avg_runtime = np.mean([m["runtime"] for m in metrics_list])
    avg_power = np.mean([m["power"] for m in metrics_list])
    avg_throughput = np.mean([m["throughput"] for m in metrics_list])
    
    std_runtime = np.std([m["runtime"] for m in metrics_list])
    std_power = np.std([m["power"] for m in metrics_list])
    std_throughput = np.std([m["throughput"] for m in metrics_list])
    
    # 记录基准测试结果
    logger.info(f"基准测试结果:")
    logger.info(f"平均运行时间: {avg_runtime:.3f}s ± {std_runtime:.3f}s")
    logger.info(f"平均功率: {avg_power:.2f}W ± {std_power:.2f}W")
    logger.info(f"平均吞吐量: {avg_throughput:.2f} tokens/s ± {std_throughput:.2f} tokens/s")
    
    # 验证性能稳定性
    assert std_runtime / avg_runtime < 0.1  # 运行时间变异系数小于10%
    assert std_power / avg_power < 0.1  # 功率变异系数小于10%
    assert std_throughput / avg_throughput < 0.1  # 吞吐量变异系数小于10%

def test_long_term_stability(profiler):
    """测试长期运行的稳定性"""
    test_duration = 300  # 5分钟测试
    check_interval = 10  # 每10秒检查一次
    start_time = time.time()
    
    initial_memory = profiler.get_memory_info()["used"]
    initial_power = profiler.measure_power()
    
    def monitoring_task():
        while time.time() - start_time < test_duration:
            # 执行标准负载
            metrics = profiler.measure(lambda: time.sleep(1), input_tokens=100, output_tokens=100)
            
            # 检查关键指标
            current_memory = profiler.get_memory_info()["used"]
            current_power = profiler.measure_power()
            
            # 验证内存使用
            memory_increase = current_memory - initial_memory
            assert memory_increase < 100 * 1024 * 1024  # 内存增长不超过100MB
            
            # 验证功率稳定性
            power_diff = abs(current_power - initial_power)
            assert power_diff < 20.0  # 功率波动不超过20W
            
            time.sleep(check_interval)
    
    # 启动监控线程
    monitoring_thread = threading.Thread(target=monitoring_task)
    monitoring_thread.start()
    monitoring_thread.join()
    
    # 验证最终状态
    final_memory = profiler.get_memory_info()["used"]
    memory_diff = final_memory - initial_memory
    assert memory_diff < 200 * 1024 * 1024  # 整体内存增长不超过200MB

def test_resource_leak(profiler):
    """测试资源泄漏情况"""
    def get_process_memory():
        """获取当前进程的内存使用"""
        process = psutil.Process()
        return process.memory_info().rss
    
    def get_gpu_handles():
        """获取GPU句柄数量"""
        return len(profiler._active_handles) if hasattr(profiler, '_active_handles') else 0
    
    # 记录初始状态
    initial_memory = get_process_memory()
    initial_handles = get_gpu_handles()
    
    # 执行多次测量
    for _ in range(100):
        metrics = profiler.measure(lambda: time.sleep(0.1), input_tokens=10, output_tokens=10)
        
        # 每10次测量检查一次资源使用
        if _ % 10 == 0:
            current_memory = get_process_memory()
            current_handles = get_gpu_handles()
            
            # 验证内存增长
            memory_increase = current_memory - initial_memory
            assert memory_increase < 50 * 1024 * 1024  # 内存增长不超过50MB
            
            # 验证句柄数量
            assert current_handles <= initial_handles + 2  # 允许最多增加2个句柄
    
    # 执行清理
    profiler.cleanup()
    
    # 验证最终状态
    final_memory = get_process_memory()
    final_handles = get_gpu_handles()
    
    # 验证资源释放
    assert final_memory <= initial_memory * 1.1  # 允许10%的内存增长
    assert final_handles == 0  # 所有句柄都应该被释放

def test_recovery_after_failure(profiler):
    """测试故障后的恢复能力"""
    def failing_task():
        raise RuntimeError("模拟任务失败")
    
    # 记录初始状态
    initial_memory = profiler.get_memory_info()["used"]
    
    # 执行失败任务
    with pytest.raises(RuntimeError):
        profiler.measure(failing_task, input_tokens=10, output_tokens=10)
    
    # 验证恢复状态
    assert not profiler.is_measuring  # 确保测量状态已重置
    
    # 尝试正常任务
    metrics = profiler.measure(lambda: time.sleep(0.1), input_tokens=10, output_tokens=10)
    assert metrics["runtime"] > 0
    assert metrics["power"] >= 0
    
    # 验证资源状态
    current_memory = profiler.get_memory_info()["used"]
    memory_diff = current_memory - initial_memory
    assert memory_diff < 50 * 1024 * 1024  # 内存增长不超过50MB

def test_performance_degradation(profiler):
    """测试性能退化检测"""
    def benchmark_task():
        time.sleep(0.1)
        return "result"
    
    # 初始性能基准
    initial_metrics = []
    for _ in range(5):
        metrics = profiler.measure(benchmark_task, input_tokens=100, output_tokens=100)
        initial_metrics.append(metrics)
    
    initial_avg_throughput = np.mean([m["throughput"] for m in initial_metrics])
    
    # 模拟持续负载
    for _ in range(20):
        metrics = profiler.measure(benchmark_task, input_tokens=100, output_tokens=100)
        current_throughput = metrics["throughput"]
        
        # 验证性能没有显著退化
        assert current_throughput >= initial_avg_throughput * 0.8  # 允许20%的性能波动 