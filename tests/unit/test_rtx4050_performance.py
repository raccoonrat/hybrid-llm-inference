import pytest
import os
import time
from hardware_profiling.rtx4050_profiler import RTX4050Profiler

def test_task_measurement():
    """测试任务测量"""
    config = {
        "device_id": 0,
        "idle_power": 15.0,
        "sample_interval": 200
    }
    
    # 在测试模式下运行
    os.environ["TEST_MODE"] = "true"
    profiler = RTX4050Profiler(config)
    
    # 定义一个简单的任务
    def simple_task():
        time.sleep(0.5)  # 模拟500ms的任务
        return "task completed"
    
    # 测量任务
    metrics = profiler.measure(simple_task, input_tokens=10, output_tokens=20)
    
    # 验证指标
    assert metrics["energy"] > 0
    assert metrics["runtime"] >= 0.5  # 至少500ms
    assert metrics["throughput"] > 0
    assert metrics["energy_per_token"] > 0
    assert metrics["result"] == "task completed"

def test_zero_tokens():
    """测试零token的情况"""
    config = {
        "device_id": 0,
        "idle_power": 15.0,
        "sample_interval": 200
    }
    
    os.environ["TEST_MODE"] = "true"
    profiler = RTX4050Profiler(config)
    
    def simple_task():
        return "task completed"
    
    # 测试零输入token
    metrics = profiler.measure(simple_task, input_tokens=0, output_tokens=10)
    assert metrics["throughput"] > 0
    assert metrics["energy_per_token"] > 0
    
    # 测试零输出token
    metrics = profiler.measure(simple_task, input_tokens=10, output_tokens=0)
    assert metrics["throughput"] > 0
    assert metrics["energy_per_token"] > 0

def test_short_tasks():
    """测试短任务"""
    config = {
        "device_id": 0,
        "idle_power": 15.0,
        "sample_interval": 200
    }
    
    os.environ["TEST_MODE"] = "true"
    profiler = RTX4050Profiler(config)
    
    def very_short_task():
        return "done"
    
    # 测试非常短的任务
    metrics = profiler.measure(very_short_task, input_tokens=1, output_tokens=1)
    assert metrics["runtime"] > 0
    assert metrics["energy"] > 0

def test_rtx4050_performance_measure():
    """测试RTX4050分析器的性能测量"""
    # 设置测试模式
    os.environ["TEST_MODE"] = "true"
    
    config = {
        "device_id": 0,
        "idle_power": 15.0,
        "sample_interval": 200
    }
    profiler = RTX4050Profiler(config)
    
    def mock_measure():
        """模拟测量函数"""
        return {
            "energy": 12.0,
            "runtime": 2.0,
            "throughput": 15.0,  # 基于输入和输出token计算
            "energy_per_token": 0.4,  # 基于总能耗和token数计算
            "total_tasks": 1
        }
    
    # 测试不同的输入输出token组合
    test_cases = [
        (10, 20),  # 小规模
        (100, 200),  # 中等规模
        (1000, 2000)  # 大规模
    ]
    
    for input_tokens, output_tokens in test_cases:
        metrics = profiler.measure(mock_measure, input_tokens, output_tokens)
        
        # 验证指标
        assert isinstance(metrics, dict), "返回值应该是字典类型"
        assert "energy" in metrics, "应包含energy指标"
        assert "runtime" in metrics, "应包含runtime指标"
        assert "throughput" in metrics, "应包含throughput指标"
        assert "energy_per_token" in metrics, "应包含energy_per_token指标"
        
        # 验证数值
        assert metrics["energy"] > 0, "能耗应该大于0"
        assert metrics["runtime"] > 0, "运行时间应该大于0"
        assert metrics["throughput"] > 0, "吞吐量应该大于0"
        assert metrics["energy_per_token"] > 0, "每token能耗应该大于0" 