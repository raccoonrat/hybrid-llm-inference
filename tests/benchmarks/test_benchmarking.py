"""基准测试模块。"""

import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, List
import re
from unittest.mock import patch, MagicMock
from src.benchmarking.system_benchmarking import SystemBenchmarking
from src.benchmarking.model_benchmarking import ModelBenchmarking
from src.benchmarking.base_benchmarking import BaseBenchmarking
from src.benchmarking.report_generator import ReportGenerator
import json
import shutil
import multiprocessing
import psutil

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from src.model_zoo.base_model import BaseModel
from toolbox.logger import get_logger
from .test_benchmark_classes import TestSystemBenchmarking, TestModelBenchmarking

logger = get_logger(__name__)

# 设置测试模式
os.environ['TEST_MODE'] = 'true'

@pytest.fixture
def mock_dataset(tmp_path) -> str:
    """创建模拟数据集。"""
    dataset_path = tmp_path / "mock_dataset.json"
    dataset_content = [
        {
            "input": "测试输入1",
            "output": "测试输出1",
            "tokens": 10
        },
        {
            "input": "测试输入2",
            "output": "测试输出2",
            "tokens": 8
        }
    ]
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_content, f, ensure_ascii=False, indent=4)
    return str(dataset_path)

@pytest.fixture
def test_config(tmp_path, mock_dataset):
    """创建测试配置的fixture。"""
    return {
        "dataset_path": mock_dataset,
        "hardware_config": {
            "device_id": 0,
            "device_type": "cuda",
            "idle_power": 100,
            "sample_interval": 0.1
        },
        "model_config": {
            "model_name": "mock_model",
            "batch_size": 1,
            "max_length": 512,
            "mode": "inference"
        },
        "scheduler_config": {
            "strategy": "round_robin",
            "batch_size": 1,
            "num_workers": 1
        },
        "output_dir": str(tmp_path / "output")
    }

@pytest.fixture
def system_benchmarking(test_config):
    """系统基准测试fixture。"""
    return TestSystemBenchmarking(test_config)

@pytest.fixture
def model_benchmarking(test_config):
    """模型基准测试fixture。"""
    return TestModelBenchmarking(test_config)

@pytest.fixture
def report_generator(tmp_path):
    """创建报告生成器实例。"""
    return ReportGenerator(str(tmp_path / "output"))

def test_system_benchmarking_initialization(test_config):
    """测试系统基准测试类的初始化。"""
    benchmarking = TestSystemBenchmarking(test_config)
    assert benchmarking.config == test_config
    assert benchmarking.dataset_path == test_config["dataset_path"]
    assert benchmarking.output_dir == test_config["output_dir"]

def test_system_benchmarking_run_benchmark(system_benchmarking):
    """测试系统基准测试的运行。"""
    results = system_benchmarking.run_benchmarks()
    assert "metrics" in results
    assert "tradeoff_results" in results
    assert "weights" in results["tradeoff_results"]
    assert "values" in results["tradeoff_results"]

def test_model_benchmarking_initialization(test_config):
    """测试模型基准测试类的初始化。"""
    benchmarking = TestModelBenchmarking(test_config)
    assert benchmarking.config == test_config
    assert benchmarking.dataset_path == test_config["dataset_path"]
    assert benchmarking.output_dir == test_config["output_dir"]

def test_model_benchmarking_run_benchmark(model_benchmarking):
    """测试 ModelBenchmarking 运行基准测试。"""
    results = model_benchmarking.run_benchmarks()
    assert "metrics" in results
    assert "tradeoff_results" in results
    assert "weights" in results["tradeoff_results"]
    assert "values" in results["tradeoff_results"]

def test_system_benchmarking_cleanup(test_config):
    """测试系统基准测试的清理。"""
    benchmarking = TestSystemBenchmarking(test_config)
    benchmarking.cleanup()
    assert not os.path.exists(test_config["output_dir"])

def test_model_benchmarking_cleanup(test_config):
    """测试模型基准测试的清理。"""
    benchmarking = TestModelBenchmarking(test_config)
    benchmarking.cleanup()
    assert not os.path.exists(test_config["output_dir"])

def test_system_benchmarking_small_dataset(test_config):
    """测试小数据集系统基准测试。"""
    benchmarking = TestSystemBenchmarking(test_config)
    results = benchmarking.run_benchmarks()
    assert "metrics" in results
    assert "tradeoff_results" in results

def test_model_benchmarking_small_dataset(test_config):
    """测试小数据集模型基准测试。"""
    benchmarking = TestModelBenchmarking(test_config)
    results = benchmarking.run_benchmarks()
    assert "metrics" in results
    assert "tradeoff_results" in results

def test_system_benchmarking_large_sample_size(test_config):
    """测试大样本量系统基准测试。"""
    benchmarking = TestSystemBenchmarking(test_config)
    results = benchmarking.run_benchmarks()
    assert "metrics" in results
    assert "tradeoff_results" in results

def test_system_benchmarking_empty_dataset(test_config):
    """测试空数据集的情况。"""
    # 创建空数据集
    empty_dataset = []
    with open("empty_dataset.json", "w", encoding="utf-8") as f:
        json.dump(empty_dataset, f)

    # 创建测试实例并设置属性
    benchmark = TestSystemBenchmarking(test_config)
    benchmark.dataset_path = "empty_dataset.json"
    benchmark.output_dir = "test_output"

    # 验证在_init_components时抛出异常
    with pytest.raises(ValueError, match="数据集不能为空"):
        benchmark._init_components()

    # 清理
    os.remove("empty_dataset.json")

def test_system_benchmarking_invalid_dataset(test_config):
    """测试无效数据集的情况。"""
    # 创建无效数据集
    invalid_dataset = [{"input": "test", "output": "test"}]
    with open("invalid_dataset.json", "w", encoding="utf-8") as f:
        json.dump(invalid_dataset, f)

    # 创建测试实例并设置属性
    benchmark = TestSystemBenchmarking(test_config)
    benchmark.dataset_path = "invalid_dataset.json"
    benchmark.output_dir = "test_output"

    # 验证在_init_components时抛出异常
    with pytest.raises(ValueError, match="数据集中的每个项目必须包含input、output和tokens字段"):
        benchmark._init_components()

    # 清理
    os.remove("invalid_dataset.json")

def test_model_benchmarking_invalid_dataset(test_config):
    """测试模型基准测试的无效数据集情况。"""
    # 创建无效数据集
    invalid_dataset = [{"input": "test", "output": "test"}]
    with open("invalid_dataset.json", "w", encoding="utf-8") as f:
        json.dump(invalid_dataset, f)

    # 创建测试实例并设置属性
    benchmark = TestModelBenchmarking(test_config)
    benchmark.dataset_path = "invalid_dataset.json"
    benchmark.output_dir = "test_output"

    # 验证在_init_components时抛出异常
    with pytest.raises(ValueError, match="数据集中的每个项目必须包含input、output和tokens字段"):
        benchmark._init_components()

    # 清理
    os.remove("invalid_dataset.json")

def test_cleanup(tmp_path):
    """测试清理功能。"""
    output_dir = tmp_path / "output"
    os.makedirs(output_dir)
    assert os.path.exists(output_dir)
    shutil.rmtree(output_dir)
    assert not os.path.exists(output_dir)

def test_report_generator_valid_output(tmp_path):
    """测试报告生成器的有效输出。"""
    output_dir = tmp_path / "output"
    os.makedirs(output_dir)
    generator = ReportGenerator(str(output_dir))
    metrics = {
        "mock_model": {
            "throughput": 100.0,
            "latency": 0.1,
            "energy": 50.0,
            "runtime": 1.0,
            "summary": {
                "average_throughput": 100.0,
                "average_latency": 0.1,
                "average_energy": 50.0,
                "avg_energy_per_token": 0.5,
                "avg_runtime": 1.0
            }
        }
    }
    tradeoff_results = {
        "weights": [0.2, 0.3, 0.5],
        "values": [
            {"throughput": 100.0, "latency": 0.1, "energy": 50.0, "runtime": 1.0},
            {"throughput": 200.0, "latency": 0.2, "energy": 100.0, "runtime": 2.0},
            {"throughput": 300.0, "latency": 0.3, "energy": 150.0, "runtime": 3.0}
        ]
    }
    generator.generate_report(metrics, tradeoff_results)
    assert os.path.exists(output_dir / "report.html")
    assert os.path.exists(output_dir / "energy_per_token.png")
    assert os.path.exists(output_dir / "runtime.png")
    assert os.path.exists(output_dir / "tradeoff_curve.png")

def test_report_generator_invalid_tradeoff_results(tmp_path):
    """测试报告生成器的无效权衡结果。"""
    output_dir = tmp_path / "output"
    os.makedirs(output_dir)
    generator = ReportGenerator(str(output_dir))
    metrics = {
        "throughput": 100.0,
        "latency": 0.1,
        "energy": 50.0,
        "runtime": 1.0,
        "summary": {
            "average_throughput": 100.0,
            "average_latency": 0.1,
            "average_energy": 50.0
        }
    }
    tradeoff_results = {
        "weights": [0.2, 0.3, 0.5],
        "values": "invalid"
    }
    with pytest.raises(ValueError):
        generator.generate_report(metrics, tradeoff_results)

def test_model_benchmarking_config_validation(tmp_path):
    """测试模型基准测试的配置验证。"""
    # 创建有效的数据集文件
    dataset_path = tmp_path / "test.json"
    valid_dataset = [{"input": "test", "output": "result", "tokens": 10}]
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(valid_dataset, f)
    
    # 测试无效的model_config
    invalid_config = {
        "dataset_path": str(dataset_path),
        "model_config": "invalid",  # 不是字典
        "hardware_config": {"device_id": 0},
        "output_dir": str(tmp_path / "output")
    }
    with pytest.raises(ValueError, match="model_config 必须是 dict 类型"):
        ModelBenchmarking(invalid_config)
    
    # 测试空的model_config
    empty_config = {
        "dataset_path": str(dataset_path),
        "model_config": {},  # 空字典
        "hardware_config": {"device_id": 0},
        "output_dir": str(tmp_path / "output")
    }
    with pytest.raises(ValueError, match="model_config 不能为空"):
        ModelBenchmarking(empty_config)

def test_model_benchmarking_dataset_loading(tmp_path):
    """测试模型基准测试的数据集加载。"""
    # 创建测试数据集
    dataset_path = tmp_path / "test_dataset.json"
    valid_dataset = [
        {"input": "test1", "output": "result1", "tokens": 10},
        {"input": "test2", "output": "result2", "tokens": 20}
    ]
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(valid_dataset, f)
    
    # 测试有效数据集
    config = {
        "dataset_path": str(dataset_path),
        "model_config": {"model_name": "test_model"},
        "hardware_config": {"device_id": 0},
        "output_dir": str(tmp_path / "output")
    }
    benchmark = ModelBenchmarking(config)
    assert len(benchmark.dataset) == 2
    
    # 测试无效JSON格式
    with open(dataset_path, 'w', encoding='utf-8') as f:
        f.write("invalid json")
    with pytest.raises(ValueError, match="数据集 .* 不是有效的JSON格式"):
        ModelBenchmarking(config)
    
    # 测试空数据集
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump([], f)
    with pytest.raises(ValueError, match="数据集 .* 为空"):
        ModelBenchmarking(config)

def test_model_benchmarking_component_initialization(tmp_path):
    """测试模型基准测试组件初始化失败的情况。"""
    test_config = {
        "dataset_path": str(tmp_path / "test_dataset.json"),
        "model_config": {"model_name": "test_model"},
        "hardware_config": {"device": "cpu"},
        "output_dir": "/invalid/path"
    }

    # 创建测试数据集
    dataset = [{"input": "test", "output": "test", "tokens": 10}]
    with open(test_config["dataset_path"], "w", encoding="utf-8") as f:
        json.dump(dataset, f)

    # 模拟os.makedirs抛出PermissionError
    with patch("os.makedirs") as mock_makedirs:
        mock_makedirs.side_effect = PermissionError("权限不足")

        with pytest.raises(PermissionError, match="权限不足"):
            ModelBenchmarking(test_config)

    # 清理测试文件
    os.remove(test_config["dataset_path"])

def test_model_benchmarking_run_benchmarks(tmp_path):
    """测试模型基准测试的运行。"""
    # 创建测试数据集
    dataset_path = tmp_path / "test_dataset.json"
    valid_dataset = [{"input": "test", "output": "result", "tokens": 10}]
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(valid_dataset, f)
    
    # 测试正常运行
    config = {
        "dataset_path": str(dataset_path),
        "model_config": {"model_name": "test_model"},
        "hardware_config": {"device_id": 0},
        "output_dir": str(tmp_path / "output")
    }
    benchmark = ModelBenchmarking(config)
    results = benchmark.run_benchmarks()
    
    assert "metrics" in results
    assert "tradeoff_results" in results
    assert "weights" in results["tradeoff_results"]
    assert "values" in results["tradeoff_results"]
    
    metrics = results["metrics"]
    assert "throughput" in metrics
    assert "latency" in metrics
    assert "energy" in metrics
    assert "runtime" in metrics
    assert "summary" in metrics
    
    # 测试未初始化的情况
    benchmark.initialized = False
    with pytest.raises(RuntimeError, match="基准测试未初始化"):
        benchmark.run_benchmarks()

def test_model_benchmarking_cleanup(tmp_path):
    """测试模型基准测试的资源清理。"""
    # 创建测试数据集
    dataset_path = tmp_path / "test_dataset.json"
    valid_dataset = [{"input": "test", "output": "result", "tokens": 10}]
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(valid_dataset, f)
    
    # 创建输出目录并添加一些文件
    output_dir = tmp_path / "output"
    os.makedirs(output_dir)
    with open(output_dir / "test.txt", 'w') as f:
        f.write("test")
    
    # 测试正常清理
    config = {
        "dataset_path": str(dataset_path),
        "model_config": {"model_name": "test_model"},
        "hardware_config": {"device_id": 0},
        "output_dir": str(output_dir)
    }
    benchmark = ModelBenchmarking(config)
    
    # 确保目录存在
    assert os.path.exists(config["output_dir"])
    # 执行清理
    benchmark.cleanup()
    # 验证目录已被删除
    assert not os.path.exists(config["output_dir"])
    
    # 测试清理不存在的目录
    benchmark.cleanup()  # 应该不会抛出异常

def test_system_benchmarking_resource_monitoring(test_config):
    """测试系统资源监控功能。"""
    with patch('src.hardware_profiling.rtx4050_profiler.RTX4050Profiler') as mock_profiler:
        # 模拟资源监控数据
        mock_profiler.return_value.get_power_usage.return_value = 100.0
        mock_profiler.return_value.get_memory_usage.return_value = 8000
        mock_profiler.return_value.get_gpu_utilization.return_value = 80.0
        
        benchmarking = TestSystemBenchmarking(test_config)
        results = benchmarking.run_benchmarks()
        
        assert "metrics" in results
        assert "hardware_metrics" in results
        assert results["hardware_metrics"]["power_usage"] > 0
        assert results["hardware_metrics"]["memory_usage"] > 0
        assert results["hardware_metrics"]["gpu_utilization"] > 0

def test_system_benchmarking_multiprocessing(test_config):
    """测试多进程处理功能。"""
    test_config["scheduler_config"]["num_workers"] = 4
    benchmarking = TestSystemBenchmarking(test_config)
    
    with patch('multiprocessing.Pool') as mock_pool:
        mock_pool.return_value.map.return_value = [
            {"throughput": 100.0, "latency": 0.1},
            {"throughput": 110.0, "latency": 0.12},
            {"throughput": 95.0, "latency": 0.09},
            {"throughput": 105.0, "latency": 0.11}
        ]
        
        results = benchmarking.run_benchmarks()
        
        assert "metrics" in results
        assert "parallel_metrics" in results
        assert len(results["parallel_metrics"]) == 4
        assert mock_pool.return_value.map.called

def test_system_benchmarking_scheduling_strategy(test_config):
    """测试不同调度策略的效果。"""
    strategies = ["round_robin", "token_based", "dynamic"]
    for strategy in strategies:
        test_config["scheduler_config"]["strategy"] = strategy
        benchmarking = TestSystemBenchmarking(test_config)
        results = benchmarking.run_benchmarks()
        
        assert "metrics" in results
        assert "scheduling_metrics" in results
        assert results["scheduling_metrics"]["strategy"] == strategy
        assert "avg_wait_time" in results["scheduling_metrics"]
        assert "avg_queue_length" in results["scheduling_metrics"]

def test_system_benchmarking_error_handling(test_config):
    """测试异常情况处理。"""
    # 测试硬件错误
    with patch('src.hardware_profiling.rtx4050_profiler.RTX4050Profiler.get_power_usage',
              side_effect=RuntimeError("硬件错误")):
        benchmarking = TestSystemBenchmarking(test_config)
        with pytest.raises(RuntimeError, match="硬件错误"):
            benchmarking.run_benchmarks()
    
    # 测试调度错误
    test_config["scheduler_config"]["strategy"] = "invalid_strategy"
    benchmarking = TestSystemBenchmarking(test_config)
    with pytest.raises(ValueError, match="无效的调度策略"):
        benchmarking.run_benchmarks()
    
    # 测试内存不足
    with patch('psutil.virtual_memory') as mock_memory:
        mock_memory.return_value.available = 100  # 模拟内存不足
        benchmarking = TestSystemBenchmarking(test_config)
        with pytest.raises(MemoryError, match="可用内存不足"):
            benchmarking.run_benchmarks()
