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
from src.model_zoo.mock_model import MockModel
import json
import shutil
import multiprocessing
import psutil
import random
import torch
import tempfile

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from src.model_zoo.base_model import BaseModel
from toolbox.logger import get_logger
from .test_benchmark_classes import TestSystemBenchmarking, TestModelBenchmarking

logger = get_logger(__name__)

# 设置测试模式
os.environ['TEST_MODE'] = '1'

class TensorEncoder(json.JSONEncoder):
    """自定义 JSON 编码器，用于处理 PyTorch Tensor 和 NumPy 数组。"""
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

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
def sample_dataset(tmp_path):
    """创建示例数据集。"""
    dataset = [
        {
            "input": "测试输入1",
            "output": "测试输出1",
            "tokens": 10
        },
        {
            "input": "测试输入2",
            "output": "测试输出2",
            "tokens": 15
        }
    ]
    dataset_path = tmp_path / "sample_dataset.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f)
    return str(dataset_path)

@pytest.fixture
def test_config(tmp_path):
    """创建测试配置。"""
    # 创建测试数据集
    dataset_path = tmp_path / "sample_dataset.json"
    test_dataset = [
        {
            "input_data": "test input 1",
            "output": "test output 1",
            "tokens": 10
        },
        {
            "input_data": "test input 2",
            "output": "test output 2",
            "tokens": 20
        }
    ]
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(test_dataset, f)

    # 创建测试模型
    model_path = tmp_path / "model.bin"
    mock_model = MockModel(model_path=str(model_path))

    # 创建输出目录
    output_dir = tmp_path / "output"
    os.makedirs(output_dir, exist_ok=True)

    return {
        "dataset_path": str(dataset_path),
        "model_config": {
            "model_type": "test_model",
            "model_path": str(model_path)
        },
        "hardware_config": {
            "device": "cpu",
            "device_id": 0,
            "num_workers": 1
        },
        "scheduler_config": {
            "scheduler_type": "token_based",
            "batch_size": 2,
            "max_tokens": 100
        },
        "output_dir": str(output_dir)
    }

@pytest.fixture
def mock_model_files(tmp_path):
    """创建模拟的模型文件。"""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    (model_dir / "pytorch_model.bin").write_text("mock data")
    return str(model_dir)

@pytest.fixture
def mock_model_file(tmp_path):
    """创建模拟的模型文件。"""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    (model_dir / "pytorch_model.bin").write_text("mock data")
    return str(model_dir)

@pytest.fixture
def mock_dataset():
    """创建模拟的数据集。"""
    return [
        {"input": "测试输入1", "expected_output": "测试输出1"},
        {"input": "测试输入2", "expected_output": "测试输出2"}
    ]

@pytest.fixture
def benchmark_config(mock_model_file):
    """创建基准测试配置。"""
    return {
        "model_config": {
            "model_path": mock_model_file,
            "device": "cpu",
            "dtype": "float32"
        },
        "batch_size": 1,
        "num_iterations": 2,
        "warmup_iterations": 1,
        "metrics": ["latency", "throughput", "memory_usage"],
        "output_dir": str(tempfile.mkdtemp())
    }

@pytest.fixture
def benchmarking_instance(benchmark_config, mock_dataset):
    """创建基准测试实例。"""
    os.environ['TEST_MODE'] = '1'  # 避免实际加载模型
    instance = SystemBenchmarking(
        config=benchmark_config,
        dataset=mock_dataset
    )
    yield instance
    os.environ.pop('TEST_MODE', None)

@pytest.fixture
def system_benchmarking(test_config, mock_model_files):
    """系统基准测试fixture。"""
    test_config["model_config"]["model_path"] = mock_model_files
    with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model:
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_model.return_value = MagicMock()
            mock_tokenizer.return_value = MagicMock()
            return TestSystemBenchmarking(test_config)

@pytest.fixture
def model_benchmarking(test_config):
    """模型基准测试fixture。"""
    return TestModelBenchmarking(test_config)

@pytest.fixture
def report_generator(tmp_path):
    """创建报告生成器实例。"""
    return ReportGenerator(str(tmp_path / "output"))

@pytest.fixture
def sample_config(tmp_path, sample_dataset):
    """创建测试配置"""
    # 创建临时数据集文件
    dataset_file = tmp_path / "sample_dataset.json"
    with open(dataset_file, "w", encoding="utf-8") as f:
        json.dump(sample_dataset, f, ensure_ascii=False)

    # 创建临时模型文件
    model_file = tmp_path / "sample_model.pt"
    model = torch.nn.Linear(10, 10)
    torch.save(model.state_dict(), model_file)

    return {
        "dataset_path": str(dataset_file),
        "model_config": {
            "model_type": "test_model",
            "model_path": str(model_file)
        },
        "hardware_config": {
            "device": "cpu",
            "device_id": 0,
            "num_workers": 1
        },
        "scheduler_config": {
            "scheduler_type": "token_based",
            "max_batch_size": 32,
            "max_queue_size": 100
        },
        "output_dir": str(tmp_path / "output")
    }

@pytest.fixture
def benchmark_instance(test_config):
    """创建基准测试实例。"""
    # 设置测试模式
    os.environ['TEST_MODE'] = '1'
    instance = SystemBenchmarking(test_config)
    yield instance
    # 清理
    if os.path.exists("benchmark_state.json"):
        os.remove("benchmark_state.json")
    os.environ.pop('TEST_MODE', None)

def test_system_benchmarking_initialization(benchmark_instance):
    """测试系统基准测试的初始化。"""
    assert benchmark_instance.initialized
    assert isinstance(benchmark_instance.model, MockModel)
    assert benchmark_instance.dataset

def test_system_benchmarking_run_benchmarks(benchmark_instance):
    """测试系统基准测试运行功能。"""
    # 准备测试数据
    test_tasks = [
        {
            "input_data": "test input 1",
            "output": "test output 1",
            "tokens": 10
        },
        {
            "input_data": "test input 2",
            "output": "test output 2",
            "tokens": 20
        }
    ]
    
    # 运行基准测试
    results = benchmark_instance.run_benchmarks(test_tasks)
    
    # 验证结果
    assert isinstance(results, dict)
    assert len(results) == len(test_tasks)
    for task_id, result in results.items():
        assert "task" in result
        assert "result" in result
        assert "execution_time" in result
        assert isinstance(result["execution_time"], float)
        assert result["execution_time"] > 0

def test_system_benchmarking_run_benchmarks_with_dataset(benchmark_instance):
    """测试使用数据集的系统基准测试运行功能"""
    # 运行基准测试（不提供任务列表，使用数据集）
    results = benchmark_instance.run_benchmarks()
    
    # 验证结果
    assert isinstance(results, dict)
    assert len(results) == len(benchmark_instance.dataset)
    for task_id, result in results.items():
        assert "task" in result
        assert "result" in result
        assert "execution_time" in result
        assert isinstance(result["execution_time"], float)
        assert result["execution_time"] > 0

def test_system_benchmarking_cleanup(benchmark_instance):
    """测试系统基准测试的清理。"""
    benchmark_instance.cleanup()
    assert not benchmark_instance.initialized

def test_system_benchmarking_empty_dataset(test_config):
    """测试空数据集系统基准测试。"""
    # 创建空数据集
    empty_dataset = []
    with open(test_config["dataset_path"], "w", encoding="utf-8") as f:
        json.dump(empty_dataset, f)

    # 设置测试模式
    os.environ['TEST_MODE'] = '1'
    benchmarking = SystemBenchmarking(test_config)
    results = benchmarking.run_benchmarks()
    assert isinstance(results, dict)
    assert len(results) == 0

def test_system_benchmarking_invalid_dataset(test_config):
    """测试无效数据集的情况。"""
    # 创建无效数据集
    invalid_dataset = [{"output": "test"}]  # 缺少输入数据
    with open(test_config["dataset_path"], "w", encoding="utf-8") as f:
        json.dump(invalid_dataset, f)

    # 设置测试模式
    os.environ['TEST_MODE'] = '1'
    benchmarking = SystemBenchmarking(test_config)
    with pytest.raises(ValueError, match="任务必须包含输入数据"):
        benchmarking.run_benchmarks()

def test_system_benchmarking_resource_monitoring(benchmark_instance):
    """测试系统基准测试的资源监控。"""
    # 准备测试数据
    test_tasks = [
        {
            "input_data": "test input 1",
            "output": "test output 1",
            "tokens": 10
        }
    ]
    results = benchmark_instance.run_benchmarks(test_tasks)
    assert isinstance(results, dict)
    assert len(results) == 1

def test_system_benchmarking_multiprocessing(benchmark_instance):
    """测试系统基准测试的多进程支持。"""
    # 准备测试数据
    test_tasks = [
        {
            "input_data": "test input 1",
            "output": "test output 1",
            "tokens": 10
        }
    ]
    # 测试多进程支持
    with multiprocessing.Pool(2) as pool:
        results = pool.map(benchmark_instance.run_benchmarks, [test_tasks])
    assert isinstance(results[0], dict)
    assert len(results[0]) == 1

def test_system_benchmarking_scheduling_strategy(test_config):
    """测试不同调度策略的效果。"""
    # 测试token_based策略
    test_config["scheduler_config"] = {
        "scheduler_type": "token_based",
        "max_batch_size": 32,
        "max_queue_size": 100
    }

    # 设置测试模式
    os.environ['TEST_MODE'] = '1'
    benchmarking = SystemBenchmarking(test_config)
    results = benchmarking.run_benchmarks()
    assert isinstance(results, dict)
    assert len(results) > 0

def test_system_benchmarking_error_handling(test_config):
    """测试错误处理机制。"""
    # 设置测试模式
    os.environ['TEST_MODE'] = '1'
    benchmarking = SystemBenchmarking(test_config)
    
    # 测试无效任务
    invalid_tasks = [{"invalid_field": "test"}]
    with pytest.raises(ValueError, match="任务必须包含输入数据"):
        benchmarking.run_benchmarks(invalid_tasks)

def test_system_benchmarking_large_sample_size(benchmark_instance):
    """测试系统基准测试的大样本支持。"""
    # 准备大量测试数据
    test_tasks = [
        {
            "input_data": f"test input {i}",
            "output": f"test output {i}",
            "tokens": 10 + i
        }
        for i in range(100)
    ]
    results = benchmark_instance.run_benchmarks(test_tasks)
    assert isinstance(results, dict)
    assert len(results) == 100

def test_model_benchmarking_small_dataset(test_config):
    """测试小数据集模型基准测试。"""
    # 设置测试模式
    os.environ['TEST_MODE'] = '1'
    benchmarking = ModelBenchmarking(test_config)
    results = benchmarking.run_benchmarks()
    os.environ.pop('TEST_MODE', None)
    
    assert "metrics" in results
    assert "tradeoff_results" in results

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
                "avg_throughput": 100.0,
                "avg_latency": 0.1,
                "avg_energy_per_token": 0.5,
                "avg_runtime": 1.0
            }
        }
    }
    report_path = generator.generate_report(metrics, output_format="json")
    assert os.path.exists(report_path)
    assert report_path.endswith(".json")

def test_report_generator_invalid_tradeoff_results(tmp_path):
    """测试报告生成器的无效权衡结果。"""
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
                "avg_throughput": 100.0,
                "avg_latency": 0.1,
                "avg_energy_per_token": 0.5,
                "avg_runtime": 1.0
            }
        }
    }
    with pytest.raises(ValueError):
        generator.generate_report(metrics, output_format="json", include_visualizations=True)

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
    with pytest.raises(ValueError, match="model_config 必须是字典类型"):
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

def test_base_benchmarking_resource_cleanup(test_config):
    """测试基准测试基类的资源清理功能。"""
    class TestBaseBenchmarking(BaseBenchmarking):
        def _validate_config(self): pass
        def _init_components(self): pass
        def run_benchmarks(self, tasks): pass
        def get_metrics(self): pass
        
        def allocate_resources(self):
            self.temp_file = "temp.txt"
            with open(self.temp_file, "w") as f:
                f.write("test")
            self.temp_dir = "temp_dir"
            os.makedirs(self.temp_dir, exist_ok=True)
    
    benchmark = TestBaseBenchmarking(test_config)
    benchmark.allocate_resources()
    
    assert os.path.exists(benchmark.temp_file)
    assert os.path.exists(benchmark.temp_dir)
    
    benchmark.cleanup()
    
    assert not os.path.exists(benchmark.temp_file)
    assert not os.path.exists(benchmark.temp_dir)

def test_base_benchmarking_state_recovery(test_config):
    """测试基准测试基类的状态恢复功能。"""
    class TestBaseBenchmarking(BaseBenchmarking):
        def _validate_config(self): pass
        def _init_components(self): 
            self.state = {"initialized": True, "count": 0}
            self.save_state()
        def run_benchmarks(self, tasks): pass
        def get_metrics(self): pass
        
        def save_state(self):
            with open("benchmark_state.json", "w") as f:
                json.dump(self.state, f)
        
        def load_state(self):
            try:
                with open("benchmark_state.json", "r") as f:
                    self.state = json.load(f)
                return True
            except FileNotFoundError:
                return False
    
    # 测试正常状态保存和恢复
    benchmark = TestBaseBenchmarking(test_config)
    benchmark._init_components()
    assert os.path.exists("benchmark_state.json")
    
    # 测试状态恢复
    new_benchmark = TestBaseBenchmarking(test_config)
    assert new_benchmark.load_state()
    assert new_benchmark.state["initialized"]
    assert new_benchmark.state["count"] == 0
    
    # 清理
    os.remove("benchmark_state.json")

def test_base_benchmarking_config_validation(test_config):
    """测试基准测试基类的配置验证功能。"""
    class TestBaseBenchmarking(BaseBenchmarking):
        def _validate_config(self):
            if "custom_field" not in self.config:
                raise ValueError("缺少custom_field配置")
        def _init_components(self): pass
        def run_benchmarks(self, tasks): pass
        def get_metrics(self): pass

    # 测试必需字段验证
    invalid_config = {}
    with pytest.raises(ValueError, match="dataset_path 不能为空"):
        TestBaseBenchmarking(invalid_config)

    # 测试字段类型验证
    invalid_config = {
        "dataset_path": 123,  # 应该是字符串
        "hardware_config": {},
        "model_config": {},
        "output_dir": ""
    }
    with pytest.raises(ValueError, match="model_path 必须是字符串类型"):
        TestBaseBenchmarking(invalid_config)

    # 测试自定义配置验证
    valid_base_config = test_config.copy()
    valid_base_config["custom_field"] = "test"  # 添加自定义字段
    TestBaseBenchmarking(valid_base_config)  # 应该不会抛出异常

    # 测试缺少自定义字段
    invalid_config = test_config.copy()
    with pytest.raises(ValueError, match="缺少custom_field配置"):
        TestBaseBenchmarking(invalid_config)

def test_report_generator_custom_plots(tmp_path):
    """测试报告生成器的自定义图表生成。"""
    output_dir = tmp_path / "output"
    os.makedirs(output_dir)
    generator = ReportGenerator(str(output_dir))

    # 准备测试数据
    data = [100.0, 110.0, 90.0]
    title = "吞吐量时间序列"
    xlabel = "时间"
    ylabel = "吞吐量"
    output_path = str(tmp_path / "throughput_time_series.png")

    # 测试时间序列图
    generator.plot_time_series(data, title, xlabel, ylabel, output_path)
    assert os.path.exists(output_path)

def test_report_generator_visualization(tmp_path):
    """测试报告生成器的可视化功能。"""
    output_dir = tmp_path / "output"
    os.makedirs(output_dir)
    generator = ReportGenerator(str(output_dir))

    # 准备测试数据
    metrics = {
        "metrics": {
            "throughput": [100.0, 110.0, 90.0],
            "latency": [0.1, 0.12, 0.09],
            "memory_usage": [1024.0, 1024.5, 1023.5],
            "power_usage": [75.0, 76.0, 74.0]
        }
    }

    # 生成报告和图表
    report_path = generator.generate_report(metrics, include_visualizations=True)
    assert os.path.exists(report_path)

    # 验证图表文件是否存在
    for metric in ["throughput", "latency", "memory_usage", "power_usage"]:
        chart_path = os.path.join(os.path.dirname(report_path), f"{metric}_time_series.png")
        assert os.path.exists(chart_path)

def test_report_generator_invalid_data_handling(tmp_path):
    """测试报告生成器的无效数据处理。"""
    output_dir = tmp_path / "output"
    os.makedirs(output_dir)
    generator = ReportGenerator(str(output_dir))

    # 测试空指标数据
    with pytest.raises(ValueError, match="基准测试结果不能为空"):
        generator.generate_report({})

def test_report_generator_custom_options(tmp_path):
    """测试报告生成器的自定义配置选项。"""
    output_dir = tmp_path / "output"
    os.makedirs(output_dir)

    # 测试自定义图表样式
    custom_style = {
        "figure.figsize": (12, 8),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "lines.linewidth": 2,
        "lines.markersize": 8
    }
    generator = ReportGenerator(str(output_dir), style_config=custom_style)

    # 测试自定义输出格式
    metrics = {
        "model1": {
            "throughput": 100.0,
            "latency": 0.1,
            "energy": 50.0,
            "runtime": 1.0,
            "summary": {
                "avg_throughput": 100.0,
                "avg_latency": 0.1,
                "avg_energy_per_token": 0.5,
                "avg_runtime": 1.0
            }
        }
    }

    # 生成JSON报告
    report_path = generator.generate_report(metrics, output_format="json")
    assert os.path.exists(report_path)
    assert report_path.endswith(".json")
