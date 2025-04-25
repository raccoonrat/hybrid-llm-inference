"""基准测试模块。"""

import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, List

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.benchmarking.report_generator import ReportGenerator
from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from src.model_zoo.base_model import BaseModel
from toolbox.logger import get_logger
from .test_benchmark_classes import TestSystemBenchmarking, TestModelBenchmarking

logger = get_logger(__name__)

# 设置测试模式
os.environ['TEST_MODE'] = 'true'

@pytest.fixture
def mock_dataset(tmp_path):
    """创建模拟数据集"""
    data = {
        "input": ["测试输入1", "测试输入2", "测试输入3"],
        "output": ["测试输出1", "测试输出2", "测试输出3"]
    }
    df = pd.DataFrame(data)
    dataset_path = tmp_path / "dataset.csv"
    df.to_csv(dataset_path, index=False)
    return str(dataset_path)

@pytest.fixture
def mock_hardware_config():
    """创建模拟硬件配置。"""
    return {
        "device_type": "nvidia_rtx4050",
        "device_id": 0,
        "max_batch_size": 4
    }

@pytest.fixture
def mock_model_config():
    """创建模拟模型配置。"""
    return {
        "model_name": "test_model",
        "model_path": "path/to/model",
        "device": "cuda",
        "dtype": "float16"
    }

@pytest.fixture
def mock_scheduler_config():
    """创建模拟调度器配置。"""
    return {
        "type": "token_based",
        "threshold": 100
    }

@pytest.fixture
def hardware_config() -> Dict[str, Any]:
    """硬件配置"""
    return {
        "device_type": "rtx4050",
        "idle_power": 30.0,
        "sample_interval": 200
    }

@pytest.fixture
def model_config() -> Dict[str, Any]:
    """模型配置"""
    return {
        "model_name": "mock_model",
        "model_path": "models/mock_model",
        "mode": "local",
        "batch_size": 1,
        "max_length": 2048
    }

@pytest.fixture
def scheduler_config():
    """Mock scheduler configuration."""
    return {
        "hardware_map": {
            "m1_pro": "m1_pro",
            "a100": "a100"
        }
    }

@pytest.fixture
def output_dir(tmp_path):
    """创建输出目录"""
    return tmp_path / "output"

@pytest.fixture
def system_benchmarking(hardware_config: Dict[str, Any], model_config: Dict[str, Any]) -> TestSystemBenchmarking:
    """创建系统基准测试实例"""
    config = {
        "hardware_config": hardware_config,
        "model_config": model_config,
        "dataset_path": "mock_dataset.csv",
        "output_dir": "mock_output"
    }
    return TestSystemBenchmarking(config)

@pytest.fixture
def model_benchmarking(hardware_config: Dict[str, Any], model_config: Dict[str, Any]) -> TestModelBenchmarking:
    """创建模型基准测试实例"""
    config = {
        "hardware_config": hardware_config,
        "model_config": model_config,
        "dataset_path": "mock_dataset.csv",
        "output_dir": "mock_output"
    }
    return TestModelBenchmarking(config)

def test_system_benchmarking_initialization(system_benchmarking: TestSystemBenchmarking):
    """测试系统基准测试初始化"""
    assert system_benchmarking is not None
    assert system_benchmarking.hardware_config is not None
    assert system_benchmarking.model_config is not None

def test_system_benchmarking_run_benchmark(system_benchmarking: TestSystemBenchmarking):
    """测试系统基准测试运行"""
    tasks = ["测试任务1", "测试任务2"]
    result = system_benchmarking.run_benchmarks()
    assert result is not None
    assert "energy" in result
    assert "runtime" in result
    assert "throughput" in result
    assert "energy_per_token" in result
    assert "total_tasks" in result

def test_system_benchmarking_cleanup(system_benchmarking: TestSystemBenchmarking):
    """测试系统基准测试清理"""
    system_benchmarking.cleanup()
    assert True  # 如果清理成功，不会抛出异常

def test_model_benchmarking_initialization(model_benchmarking: TestModelBenchmarking):
    """测试模型基准测试初始化"""
    assert model_benchmarking is not None
    assert model_benchmarking.hardware_config is not None
    assert model_benchmarking.model_config is not None

def test_model_benchmarking_run_benchmark(model_benchmarking: TestModelBenchmarking):
    """测试模型基准测试运行"""
    tasks = ["测试任务1", "测试任务2"]
    result = model_benchmarking.run_benchmarks()
    assert result is not None
    assert "mock_model" in result
    assert "energy" in result["mock_model"]
    assert "runtime" in result["mock_model"]
    assert "throughput" in result["mock_model"]
    assert "energy_per_token" in result["mock_model"]
    assert "total_tasks" in result["mock_model"]

def test_model_benchmarking_cleanup(model_benchmarking: TestModelBenchmarking):
    """测试模型基准测试清理"""
    model_benchmarking.cleanup()
    assert True  # 如果清理成功，不会抛出异常

def test_system_benchmarking_small_dataset(mock_dataset, hardware_config, model_config, output_dir):
    """测试小数据集系统基准测试"""
    # 设置测试模式环境变量
    os.environ['TEST_MODE'] = '1'
    
    config = {
        "dataset_path": mock_dataset,
        "hardware_config": hardware_config,
        "model_config": model_config,
        "output_dir": str(output_dir)
    }
    benchmarker = TestSystemBenchmarking(config)
    
    results = benchmarker.run_benchmarks()
    
    assert isinstance(results, dict)
    assert "energy" in results
    assert "runtime" in results
    assert "total_tasks" in results
    assert results["total_tasks"] > 0

def test_model_benchmarking_small_dataset(mock_dataset, hardware_config, model_config, output_dir):
    """测试小数据集模型基准测试"""
    # 设置测试模式环境变量
    os.environ['TEST_MODE'] = '1'
    
    config = {
        "dataset_path": mock_dataset,
        "hardware_config": hardware_config,
        "model_config": model_config,
        "output_dir": str(output_dir)
    }
    benchmarker = TestModelBenchmarking(config)
    
    results = benchmarker.run_benchmarks()
    
    assert isinstance(results, dict)
    assert "metrics" in results
    assert "summary" in results
    assert "avg_energy" in results["summary"]
    assert "avg_runtime" in results["summary"]
    assert "avg_throughput" in results["summary"]
    assert "avg_energy_per_token" in results["summary"]

def test_system_benchmarking_large_sample_size(mock_dataset, hardware_config, model_config, output_dir):
    """测试大样本量系统基准测试"""
    # 设置测试模式环境变量
    os.environ['TEST_MODE'] = '1'
    
    config = {
        "dataset_path": mock_dataset,
        "hardware_config": hardware_config,
        "model_config": model_config,
        "output_dir": str(output_dir)
    }
    benchmarker = TestSystemBenchmarking(config)
    
    results = benchmarker.run_benchmarks(sample_size=100)
    
    assert isinstance(results, dict)
    assert "energy" in results
    assert "runtime" in results
    assert "total_tasks" in results
    assert results["total_tasks"] > 0

def test_system_benchmarking_empty_dataset(tmp_path, hardware_config, model_config, scheduler_config, output_dir, monkeypatch):
    """Test system benchmarking with an empty dataset."""
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("[]")
    
    def mock_measure(task, input_tokens, output_tokens): task()
    monkeypatch.setattr("hardware_profiling.base_profiler.HardwareProfiler.measure", mock_measure)
    monkeypatch.setattr("model_zoo.base_model.BaseModel.infer", lambda self, prompt: "Mock response")
    monkeypatch.setattr("model_zoo.base_model.BaseModel.get_token_count", lambda self, text: 10)
    
    config = {
        "dataset_path": str(empty_file),
        "hardware_config": hardware_config,
        "model_config": model_config,
        "scheduler_config": scheduler_config,
        "output_dir": str(output_dir)
    }
    benchmarker = TestSystemBenchmarking(config)
    thresholds = {"T_in": 32, "T_out": 32}
    
    with pytest.raises(ValueError, match="Dataset is empty"):
        benchmarker.run_benchmarks(thresholds, model_name="mock_model", sample_size=3)

def test_system_benchmarking_invalid_thresholds(mock_dataset, hardware_config, model_config, scheduler_config, output_dir, monkeypatch):
    """Test system benchmarking with invalid thresholds."""
    def mock_measure(task, input_tokens, output_tokens):
        task()
        return {"energy": 10.0, "runtime": 2.0, "throughput": 15.0, "energy_per_token": 0.5}
    monkeypatch.setattr("hardware_profiling.base_profiler.HardwareProfiler.measure", mock_measure)
    monkeypatch.setattr("model_zoo.base_model.BaseModel.infer", lambda self, prompt: "Mock response")
    monkeypatch.setattr("model_zoo.base_model.BaseModel.get_token_count", lambda self, text: 10)
    
    config = {
        "dataset_path": mock_dataset,
        "hardware_config": hardware_config,
        "model_config": model_config,
        "scheduler_config": scheduler_config,
        "output_dir": str(output_dir)
    }
    benchmarker = TestSystemBenchmarking(config)
    thresholds = {"T_in": -1, "T_out": 32}
    
    with pytest.raises(ValueError, match="Thresholds must be positive"):
        benchmarker.run_benchmarks(thresholds, model_name="mock_model", sample_size=3)

def test_report_generator_valid_output(mock_dataset, hardware_config, model_config, scheduler_config, output_dir, monkeypatch):
    """Test report generation with valid benchmark results."""
    benchmark_results = {
        "hybrid": {
            "metrics": [
                {"energy": 10.0, "runtime": 2.0, "throughput": 15.0, "energy_per_token": 0.5},
                {"energy": 12.0, "runtime": 2.5, "throughput": 12.0, "energy_per_token": 0.6}
            ],
            "summary": {
                "avg_energy": 11.0,
                "avg_runtime": 2.25,
                "avg_throughput": 13.5,
                "avg_energy_per_token": 0.55
            }
        },
        "a100": {
            "metrics": [
                {"energy": 15.0, "runtime": 1.5, "throughput": 20.0, "energy_per_token": 0.75},
                {"energy": 14.0, "runtime": 1.8, "throughput": 18.0, "energy_per_token": 0.7}
            ],
            "summary": {
                "avg_energy": 14.5,
                "avg_runtime": 1.65,
                "avg_throughput": 19.0,
                "avg_energy_per_token": 0.725
            }
        }
    }
    tradeoff_results = {
        0.0: {"energy": 12.0, "runtime": 2.5},
        0.5: {"energy": 11.0, "runtime": 2.0},
        1.0: {"energy": 10.0, "runtime": 1.8}
    }

    generator = ReportGenerator(output_dir=output_dir)
    generator.generate_report(benchmark_results, tradeoff_results)

def test_report_generator_empty_results(output_dir):
    """Test report generation with empty benchmark results."""
    generator = ReportGenerator(output_dir=output_dir)
    benchmark_results = {}

    with pytest.raises(ValueError, match="基准测试结果不能为空"):
        generator.generate_report(benchmark_results)

def test_report_generator_invalid_tradeoff_results(output_dir):
    """Test report generation with invalid tradeoff results."""
    benchmark_results = {
        "hybrid": {
            "metrics": [{"energy": 10.0, "runtime": 2.0, "throughput": 15.0, "energy_per_token": 0.5}],
            "summary": {
                "avg_energy": 10.0,
                "avg_runtime": 2.0,
                "avg_throughput": 15.0,
                "avg_energy_per_token": 0.5
            }
        }
    }
    tradeoff_results = {0.5: {"energy": -1.0, "runtime": 2.0}}

    generator = ReportGenerator(output_dir=output_dir)
    with pytest.raises(ValueError, match="Invalid tradeoff results"):
        generator.generate_report(benchmark_results, tradeoff_results)
