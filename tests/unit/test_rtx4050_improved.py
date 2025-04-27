"""RTX4050 改进测试模块。"""

import pytest
import time
import threading
import pynvml
from unittest.mock import patch, MagicMock
from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from typing import Dict, Any, List

# 测试配置
TEST_CONFIG = {
    "device_id": 0,
    "device_type": "gpu",
    "idle_power": 15.0,
    "sample_interval": 200,
    "memory_limit": 6144,  # 6GB
    "tdp": 115  # Watts
}

@pytest.fixture
def profiler():
    """创建 RTX4050Profiler 实例的 fixture。"""
    return RTX4050Profiler(TEST_CONFIG)

@pytest.fixture
def mock_nvml():
    """模拟 NVML 的 fixture。"""
    with patch("pynvml.nvmlInit") as mock_init, \
         patch("pynvml.nvmlDeviceGetHandleByIndex") as mock_get_handle, \
         patch("pynvml.nvmlDeviceGetName") as mock_get_name, \
         patch("pynvml.nvmlDeviceGetPowerUsage") as mock_get_power, \
         patch("pynvml.nvmlDeviceGetMemoryInfo") as mock_get_memory, \
         patch("pynvml.nvmlShutdown") as mock_shutdown:
        
        # 设置模拟返回值
        mock_get_name.return_value = b"NVIDIA RTX 4050"
        mock_get_power.return_value = 15000  # 15W
        mock_get_memory.return_value = (6144 * 1024 * 1024,  # total
                                       2048 * 1024 * 1024,  # free
                                       4096 * 1024 * 1024)  # used
        
        yield {
            "init": mock_init,
            "get_handle": mock_get_handle,
            "get_name": mock_get_name,
            "get_power": mock_get_power,
            "get_memory": mock_get_memory,
            "shutdown": mock_shutdown
        }

def test_error_recovery(profiler, mock_nvml):
    """测试错误恢复机制。"""
    # 模拟设备错误
    mock_nvml["get_power"].side_effect = pynvml.NVMLError
    power = profiler.measure_power()
    assert power == TEST_CONFIG["idle_power"]
    
    # 模拟内存获取错误
    mock_nvml["get_memory"].side_effect = pynvml.NVMLError
    memory_info = profiler.get_memory_info()
    assert memory_info["total"] == 0
    assert memory_info["used"] == 0
    assert memory_info["free"] == 0

def test_memory_limit(profiler, mock_nvml):
    """测试内存限制。"""
    # 测试接近内存限制的情况
    large_task = {
        "input_tokens": 1000000,
        "output_tokens": 500000
    }
    
    # 模拟内存使用接近限制
    mock_nvml["get_memory"].return_value = (6144 * 1024 * 1024,  # total
                                           100 * 1024 * 1024,    # free
                                           6044 * 1024 * 1024)   # used
    
    with pytest.raises(MemoryError):
        profiler.measure(large_task)

def test_hardware_limits(profiler, mock_nvml):
    """测试硬件限制。"""
    # 验证功耗不超过TDP
    mock_nvml["get_power"].return_value = 120000  # 120W
    power = profiler.measure_power()
    assert power <= TEST_CONFIG["tdp"]
    
    # 验证显存使用
    memory_info = profiler.get_memory_info()
    assert memory_info["total"] <= TEST_CONFIG["memory_limit"]
    assert memory_info["used"] <= TEST_CONFIG["memory_limit"]
    assert memory_info["free"] >= 0

def test_long_term_stability(profiler, mock_nvml):
    """测试长期稳定性。"""
    # 运行长时间测试
    start_time = time.time()
    metrics_list = []
    
    while time.time() - start_time < 5.0:  # 运行5秒
        task = {
            "input_tokens": 100,
            "output_tokens": 50
        }
        metrics = profiler.measure(task)
        metrics_list.append(metrics)
        time.sleep(0.1)
    
    # 验证稳定性
    assert len(metrics_list) > 0
    assert all(isinstance(m, dict) for m in metrics_list)
    assert all(m["runtime"] > 0 for m in metrics_list)
    assert all(m["power"] >= 0 for m in metrics_list)
    assert all(m["energy"] >= 0 for m in metrics_list)

def test_concurrent_stress(profiler, mock_nvml):
    """测试并发压力。"""
    def run_measurement():
        task = {
            "input_tokens": 100,
            "output_tokens": 50
        }
        return profiler.measure(task)
    
    # 创建多个线程同时运行测量
    threads = []
    results = []
    
    for _ in range(10):  # 增加并发数
        thread = threading.Thread(target=lambda: results.append(run_measurement()))
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    # 验证所有测量结果
    assert len(results) == 10
    assert all(isinstance(r, dict) for r in results)
    assert all(r["runtime"] > 0 for r in results)
    assert all(r["power"] >= 0 for r in results)
    assert all(r["energy"] >= 0 for r in results)

def test_resource_monitoring(profiler, mock_nvml):
    """测试资源监控。"""
    # 开始监控
    profiler.start_monitoring()
    
    # 模拟资源使用变化
    mock_nvml["get_power"].return_value = 80000  # 80W
    mock_nvml["get_memory"].return_value = (6144 * 1024 * 1024,  # total
                                           2048 * 1024 * 1024,   # free
                                           4096 * 1024 * 1024)   # used
    
    # 等待一段时间
    time.sleep(1.0)
    
    # 停止监控并获取结果
    metrics = profiler.stop_monitoring()
    
    # 验证监控结果
    assert isinstance(metrics, dict)
    assert "energy_consumption" in metrics
    assert "runtime" in metrics
    assert "avg_power" in metrics
    assert "avg_memory" in metrics
    assert "avg_utilization" in metrics
    assert metrics["runtime"] >= 1.0
    assert metrics["avg_power"] >= 0.0
    assert metrics["avg_memory"] >= 0.0
    assert metrics["avg_utilization"] >= 0.0 