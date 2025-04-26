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
def test_config(tmp_path):
    """创建测试配置。"""
    # 创建临时数据集文件
    dataset_file = tmp_path / "test_dataset.json"
    with open(dataset_file, 'w', encoding='utf-8') as f:
        json.dump([
            {"input": "测试输入1", "expected_output": "测试输出1"},
            {"input": "测试输入2", "expected_output": "测试输出2"}
        ], f)
    
    # 创建临时模型文件
    model_path = tmp_path / "test_model.pt"
    model = torch.nn.Linear(10, 10)
    torch.save(model.state_dict(), model_path)
    
    return {
        "hardware_config": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "num_workers": 4
        },
        "model_config": {
            "model_path": str(model_path),
            "model_type": "test_model"
        },
        "scheduler_config": {
            "scheduler_type": "token_based",
            "batch_size": 32
        },
        "dataset_path": str(dataset_file),
        "output_dir": str(tmp_path / "test_output")
    }

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
def sample_dataset():
    """创建测试数据集"""
    return {
        "inputs": [
            {
                "input": "测试输入1",
                "output": "测试输出1",
                "tokens": [1, 2, 3, 4, 5]
            },
            {
                "input": "测试输入2",
                "output": "测试输出2",
                "tokens": [6, 7, 8, 9, 10]
            }
        ]
    }

@pytest.fixture
def sample_config(tmp_path, sample_dataset):
    """创建测试配置"""
    # 创建临时数据集文件
    dataset_file = tmp_path / "sample_dataset.json"
    with open(dataset_file, "w", encoding="utf-8") as f:
        json.dump(sample_dataset, f, ensure_ascii=False)

    return {
        "model_path": str(tmp_path / "sample_model.pt"),
        "dataset_path": str(dataset_file),
        "output_dir": str(tmp_path / "output"),
        "model_config": {
            "model_type": "test_model"
        },
        "hardware_config": {
            "device": "cpu"
        }
    }

@pytest.fixture
def benchmark_instance(sample_config, sample_dataset):
    """创建基准测试实例。"""
    # 创建测试模型文件
    model = torch.nn.Linear(10, 10)
    os.makedirs("tests/data", exist_ok=True)
    torch.save(model.state_dict(), "tests/data/sample_model.pt")
    
    # 创建测试数据集文件
    with open("tests/data/sample_dataset.json", "w") as f:
        json.dump(sample_dataset, f, cls=TensorEncoder)
    
    # 设置测试模式
    os.environ['TEST_MODE'] = '1'
    instance = SystemBenchmarking(sample_config, sample_dataset)
    yield instance
    os.environ.pop('TEST_MODE', None)

def test_system_benchmarking_initialization(benchmark_instance):
    assert benchmark_instance.initialized
    assert isinstance(benchmark_instance.model, torch.nn.Module)
    assert benchmark_instance.dataset

def test_system_benchmarking_run_benchmark(benchmark_instance):
    results = benchmark_instance.run_benchmarks(benchmark_instance.dataset)
    assert results
    assert len(results) == len(benchmark_instance.dataset)
    for task_id, result in results.items():
        assert "task" in result
        assert "result" in result
        assert "execution_time" in result

def test_system_benchmarking_cleanup(benchmark_instance):
    benchmark_instance.cleanup()
    assert not benchmark_instance.initialized

def test_system_benchmarking_small_dataset(benchmark_instance):
    small_dataset = [benchmark_instance.dataset[0]]
    benchmark_instance.dataset = small_dataset
    results = benchmark_instance.run_benchmarks(small_dataset)
    assert len(results) == 1

def test_system_benchmarking_large_sample_size(benchmark_instance):
    large_dataset = benchmark_instance.dataset * 10
    benchmark_instance.dataset = large_dataset
    results = benchmark_instance.run_benchmarks(large_dataset)
    assert len(results) == len(large_dataset)

def test_system_benchmarking_empty_dataset(benchmark_instance):
    benchmark_instance.dataset = []
    with pytest.raises(RuntimeError, match="数据集为空"):
        benchmark_instance.run_benchmarks([])

def test_system_benchmarking_invalid_dataset(benchmark_instance):
    invalid_dataset = [{"task_id": "task1"}]  # 缺少 input_data
    benchmark_instance.dataset = invalid_dataset
    with pytest.raises(ValueError, match="任务必须包含输入数据"):
        benchmark_instance.run_benchmarks(invalid_dataset)

def test_system_benchmarking_resource_monitoring(benchmark_instance):
    results = benchmark_instance.run_benchmarks(benchmark_instance.dataset)
    metrics = benchmark_instance.get_metrics()
    assert "average_execution_time" in metrics
    assert "max_execution_time" in metrics
    assert "min_execution_time" in metrics
    assert "throughput" in metrics

def test_system_benchmarking_multiprocessing(benchmark_instance):
    # 测试多进程支持
    with multiprocessing.Pool(2) as pool:
        results = pool.map(benchmark_instance.run_benchmarks, [benchmark_instance.dataset])
    assert len(results) == 1
    assert len(results[0]) == len(benchmark_instance.dataset)

def test_system_benchmarking_scheduling_strategy(benchmark_instance):
    # 测试不同的调度策略
    benchmark_instance.scheduler_config["scheduler_type"] = "task_based"
    benchmark_instance._init_scheduler()
    results = benchmark_instance.run_benchmarks(benchmark_instance.dataset)
    assert results

def test_system_benchmarking_error_handling(benchmark_instance):
    # 测试错误处理
    benchmark_instance.model = None
    with pytest.raises(AttributeError):
        benchmark_instance.run_benchmarks(benchmark_instance.dataset)

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
                "avg_throughput": 100.0,
                "avg_latency": 0.1,
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
    report_path = generator.generate_report(metrics, tradeoff_results)
    assert os.path.exists(report_path)
    assert report_path.endswith(".markdown")

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

def test_system_benchmarking_scheduling_strategy(test_config):
    """测试不同调度策略的效果。"""
    # 测试FIFO策略
    test_config["scheduler"] = "fifo"
    benchmarking = SystemBenchmarking(test_config)
    results_fifo = benchmarking.run_benchmark()
    
    # 测试Round Robin策略
    test_config["scheduler"] = "round_robin"
    benchmarking = SystemBenchmarking(test_config)
    results_rr = benchmarking.run_benchmark()
    
    assert "scheduling_stats" in results_fifo
    assert "scheduling_stats" in results_rr
    assert results_fifo["scheduling_stats"]["strategy"] == "fifo"
    assert results_rr["scheduling_stats"]["strategy"] == "round_robin"

def test_system_benchmarking_error_handling(test_config):
    """测试错误处理机制。"""
    benchmarking = SystemBenchmarking(test_config)
    
    # 测试硬件错误
    with patch('torch.cuda.is_available', return_value=False):
        with pytest.raises(RuntimeError, match="GPU不可用"):
            benchmarking.run_benchmark()
    
    # 测试内存不足
    with patch('torch.cuda.memory_allocated', return_value=float('inf')):
        with pytest.raises(MemoryError, match="GPU内存不足"):
            benchmarking.run_benchmark()
    
    # 测试无效配置
    invalid_config = test_config.copy()
    invalid_config["batch_size"] = -1
    with pytest.raises(ValueError, match="批处理大小必须为正数"):
        SystemBenchmarking(invalid_config)
    
    # 测试GPU错误
    with patch('torch.cuda.current_device', side_effect=RuntimeError("GPU错误")):
        with pytest.raises(RuntimeError, match="GPU错误"):
            benchmarking.run_benchmark()

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
    with pytest.raises(ValueError, match="dataset_path 必须是 str 类型"):
        TestBaseBenchmarking(invalid_config)
    
    # 测试自定义配置验证
    valid_base_config = test_config.copy()
    with pytest.raises(ValueError, match="缺少custom_field配置"):
        TestBaseBenchmarking(valid_base_config)
    
    # 测试有效配置
    valid_base_config["custom_field"] = "test"
    benchmark = TestBaseBenchmarking(valid_base_config)
    assert benchmark.config["custom_field"] == "test"

def test_report_generator_custom_plots(tmp_path):
    """测试报告生成器的自定义图表生成。"""
    output_dir = tmp_path / "output"
    os.makedirs(output_dir)
    generator = ReportGenerator(str(output_dir))
    
    # 准备测试数据
    metrics = {
        "model1": {
            "throughput": [100.0, 110.0, 90.0],
            "latency": [0.1, 0.12, 0.09],
            "energy": [50.0, 55.0, 45.0],
            "runtime": [1.0, 1.1, 0.9],
            "summary": {
                "avg_throughput": 100.0,
                "avg_latency": 0.1,
                "avg_energy_per_token": 0.5,
                "avg_runtime": 1.0
            }
        },
        "model2": {
            "throughput": [200.0, 220.0, 180.0],
            "latency": [0.2, 0.22, 0.18],
            "energy": [100.0, 110.0, 90.0],
            "runtime": [2.0, 2.2, 1.8],
            "summary": {
                "avg_throughput": 200.0,
                "avg_latency": 0.2,
                "avg_energy_per_token": 1.0,
                "avg_runtime": 2.0
            }
        }
    }
    
    # 测试时间序列图
    generator.plot_time_series(metrics, "throughput")
    assert os.path.exists(output_dir / "throughput_time_series.png")
    
    # 测试箱线图
    generator.plot_boxplot(metrics, "latency")
    assert os.path.exists(output_dir / "latency_boxplot.png")
    
    # 测试散点图
    generator.plot_scatter(metrics, "energy", "runtime")
    assert os.path.exists(output_dir / "energy_vs_runtime_scatter.png")
    
    # 测试热力图
    generator.plot_heatmap(metrics)
    assert os.path.exists(output_dir / "metrics_heatmap.png")

def test_report_generator_invalid_data_handling(tmp_path):
    """测试报告生成器的无效数据处理。"""
    output_dir = tmp_path / "output"
    os.makedirs(output_dir)
    generator = ReportGenerator(str(output_dir))

    # 测试空指标数据
    with pytest.raises(ValueError, match="指标数据不能为空"):
        generator.generate_report({})

    # 测试缺失必需字段
    invalid_metrics = {
        "model1": {
            "throughput": 100.0,
            # 缺少 latency
            "energy": 50.0,
            "runtime": 1.0
        }
    }
    with pytest.raises(ValueError, match="模型 model1 缺少必要的性能指标"):
        generator.generate_report(invalid_metrics)

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

    # 生成PDF报告
    report_path = generator.generate_report(metrics, output_format="pdf")
    assert os.path.exists(report_path)
    assert report_path.endswith(".pdf")
