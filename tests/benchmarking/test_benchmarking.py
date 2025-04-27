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
def mock_dataset(tmp_path) -> List[Dict[str, Any]]:
    """创建模拟数据集。"""
    dataset = [
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
    return dataset

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
def test_config():
    """创建测试配置。"""
    temp_dir = tempfile.mkdtemp(prefix="test_model_benchmarking_")
    model_path = os.path.join(temp_dir, "model.bin")
    dataset_path = os.path.join(temp_dir, "dataset.json")
    output_dir = os.path.join(temp_dir, "output")
    
    # 创建测试数据集
    dataset = [{"input": "test input", "output": "test output"} for _ in range(5)]
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    with open(dataset_path, "w") as f:
        json.dump(dataset, f)
    
    # 创建空模型文件
    with open(model_path, "w") as f:
        f.write("")
    
    return {
        "model_name": "test_model",
        "batch_size": 1,
        "dataset_path": dataset_path,
        "model_config": {
            "device": "cpu",
            "model_path": model_path
        },
        "hardware_config": {
            "device": "cpu",
            "device_id": 0
        },
        "output_dir": output_dir
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
def benchmarking_instance(test_config):
    """创建基准测试实例。"""
    os.environ['TEST_MODE'] = '1'  # 避免实际加载模型
    instance = SystemBenchmarking(test_config)
    yield instance
    instance.cleanup()
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
        "model_name": "test_model",
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

def test_system_benchmarking_initialization(benchmarking_instance):
    """测试系统基准测试初始化。"""
    assert benchmarking_instance.initialized
    assert benchmarking_instance.dataset is not None
    assert len(benchmarking_instance.dataset) > 0
    assert benchmarking_instance.model is not None
    assert benchmarking_instance.scheduler is not None

def test_system_benchmarking_run_benchmarks(benchmarking_instance):
    """测试运行基准测试。"""
    results = benchmarking_instance.run_benchmarks()
    assert isinstance(results, dict)
    assert len(results) > 0
    for task_id, task_result in results.items():
        assert "task" in task_result
        assert "result" in task_result
        assert "execution_time" in task_result

def test_system_benchmarking_run_benchmarks_with_dataset(benchmarking_instance, mock_dataset):
    """测试使用指定数据集运行基准测试。"""
    results = benchmarking_instance.run_benchmarks(tasks=mock_dataset)
    assert isinstance(results, dict)
    assert len(results) == len(mock_dataset)
    for task_id, task_result in results.items():
        assert "task" in task_result
        assert "result" in task_result
        assert "execution_time" in task_result

def test_system_benchmarking_cleanup(test_config):
    """测试系统基准测试的资源清理。"""
    # 创建系统基准测试实例
    benchmark = SystemBenchmarking(test_config)
    
    # 运行基准测试
    benchmark.run()
    
    # 执行清理
    benchmark.cleanup()
    
    # 验证资源已被清理
    assert benchmark.dataset is None, "数据集未被清理"
    assert benchmark.report_generator is None, "报告生成器未被清理"
    assert benchmark.model is None, "模型未被清理"
    assert benchmark.initialized is False, "初始化状态未被重置"

def test_system_benchmarking_empty_dataset():
    """测试空数据集。"""
    config = {
        "model_name": "test_model",
        "batch_size": 32,
        "dataset": [],
        "model_config": {
            "model_type": "mock",
            "model_path": "mock_path"
        },
        "hardware_config": {
            "device": "cpu",
            "device_id": 0
        },
        "scheduler_config": {
            "scheduler_type": "token_based",
            "batch_size": 2,
            "max_tokens": 100
        }
    }
    with pytest.raises(ValueError, match="数据集不能为空"):
        SystemBenchmarking(config)

def test_system_benchmarking_invalid_dataset():
    """测试无效数据集。"""
    config = {
        "model_name": "test_model",
        "batch_size": 32,
        "dataset": ["invalid", "dataset"],  # 无效的数据集格式
        "model_config": {
            "model_type": "mock",
            "model_path": "mock_path"
        },
        "hardware_config": {
            "device": "cpu",
            "device_id": 0
        },
        "scheduler_config": {
            "scheduler_type": "token_based",
            "batch_size": 2,
            "max_tokens": 100
        }
    }
    with pytest.raises(ValueError, match="数据集中的每个项目必须是字典类型"):
        SystemBenchmarking(config)

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
    
    # 添加必需的配置字段
    test_config["batch_size"] = 32
    test_config["model_name"] = "test_model"
    test_config["dataset"] = [
        {"input": "test1", "output": "output1"},
        {"input": "test2", "output": "output2"}
    ]
    
    benchmarking = SystemBenchmarking(test_config)
    results = benchmarking.run_benchmarks()
    assert isinstance(results, dict)
    assert len(results) > 0

def test_system_benchmarking_error_handling(test_config):
    """测试错误处理机制。"""
    # 设置测试模式
    os.environ['TEST_MODE'] = '1'
    test_config["dataset"] = [
        {"input": "test1", "output": "output1"},
        {"input": "test2", "output": "output2"}
    ]
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
    """测试小数据集的基准测试。"""
    # 创建一个只有一个样本的小数据集
    dataset = [{"input": "test input", "output": "test output"}]
    dataset_path = test_config["dataset_path"]
    with open(dataset_path, "w") as f:
        json.dump(dataset, f)
    
    benchmarking = ModelBenchmarking(test_config)
    results = benchmarking.run_benchmarks()
    
    # 验证结果
    assert isinstance(results, dict)
    assert "metrics" in results
    assert results["metrics"]["latency"] > 0
    assert results["metrics"]["throughput"] > 0

def test_cleanup(tmp_path):
    """测试清理功能。"""
    output_dir = tmp_path / "output"
    os.makedirs(output_dir)
    assert os.path.exists(output_dir)
    shutil.rmtree(output_dir)
    assert not os.path.exists(output_dir)

def test_report_generator_valid_output(tmp_path):
    """测试报告生成器生成有效输出。"""
    metrics = {
        "metrics": {
            "latency": {
                "mean": 100.0,
                "std": 10.0,
                "min": 80.0,
                "max": 120.0
            },
            "energy": {
                "mean": 50.0,
                "std": 5.0,
                "min": 40.0,
                "max": 60.0
            },
            "throughput": 1000.0
        }
    }
    
    report_generator = ReportGenerator(str(tmp_path))
    report_generator.generate_report(metrics)
    
    # 验证输出文件是否存在
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "visualizations").exists()

def test_report_generator_visualization(tmp_path):
    """测试报告生成器的可视化功能。"""
    metrics = {
        "metrics": {
            "latency": {
                "mean": 100.0,
                "std": 10.0,
                "min": 80.0,
                "max": 120.0,
                "distribution": [95.0, 98.0, 102.0, 105.0]
            },
            "energy": {
                "mean": 50.0,
                "std": 5.0,
                "min": 40.0,
                "max": 60.0,
                "distribution": [45.0, 48.0, 52.0, 55.0]
            }
        }
    }
    
    report_generator = ReportGenerator(str(tmp_path))
    report_generator.generate_report(metrics)
    
    # 验证可视化文件是否存在
    vis_dir = tmp_path / "visualizations"
    assert (vis_dir / "latency_distribution.png").exists()
    assert (vis_dir / "energy_distribution.png").exists()
    assert (vis_dir / "latency_energy_tradeoff.png").exists()

def test_report_generator_custom_options(tmp_path):
    """测试报告生成器的自定义选项。"""
    metrics = {
        "metrics": {
            "latency": {
                "mean": 100.0,
                "distribution": [95.0, 98.0, 102.0, 105.0]
            },
            "energy": {
                "mean": 50.0,
                "distribution": [45.0, 48.0, 52.0, 55.0]
            },
            "custom_metric": {
                "value": 75.0,
                "distribution": [70.0, 73.0, 77.0, 80.0]
            }
        }
    }
    
    options = {
        "plot_style": "seaborn",
        "figure_size": (10, 6),
        "dpi": 150
    }
    
    report_generator = ReportGenerator(str(tmp_path), options=options)
    report_generator.generate_report(metrics)
    
    # 验证输出文件是否存在
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "visualizations" / "custom_metric_distribution.png").exists()

def test_model_benchmarking_config_validation():
    """测试模型基准测试配置验证。"""
    # 测试缺少model_config
    with pytest.raises(ValueError, match="配置缺少必需字段: model_config"):
        ModelBenchmarking({
            "model_name": "test_model",
            "batch_size": 1,
            "dataset_path": "path/to/dataset",
            "hardware_config": {
                "device": "cpu",
                "device_id": 0
            }
        })

    # 测试model_config必须是字典类型
    with pytest.raises(ValueError, match="model_config 必须是字典类型"):
        ModelBenchmarking({
            "model_name": "test_model",
            "batch_size": 1,
            "dataset_path": "path/to/dataset",
            "model_config": "invalid",
            "hardware_config": {
                "device": "cpu",
                "device_id": 0
            }
        })

def test_model_benchmarking_dataset_loading():
    """测试模型基准测试数据集加载。"""
    # 测试数据集为空
    with pytest.raises(ValueError, match="数据集路径不存在"):
        ModelBenchmarking({
            "model_name": "test_model",
            "batch_size": 1,
            "dataset_path": "path/to/empty/dataset",
            "model_config": {
                "device": "cpu",
                "model_path": "path/to/model"
            },
            "hardware_config": {
                "device": "cpu",
                "device_id": 0
            }
        })

def test_model_benchmarking_component_initialization(test_config):
    """测试模型基准测试组件初始化。"""
    benchmarking = ModelBenchmarking(test_config)
    benchmarking._init_components()
    assert benchmarking.model is not None
    assert benchmarking.report_generator is not None
    assert benchmarking.dataset is not None

def test_model_benchmarking_run_benchmarks(test_config):
    """测试模型基准测试的运行。"""
    benchmarking = ModelBenchmarking(test_config)
    results = benchmarking.run_benchmarks()
    
    # 验证结果格式
    assert isinstance(results, dict)
    assert "metrics" in results
    assert "latency" in results["metrics"]
    assert "throughput" in results["metrics"]
    assert "memory" in results["metrics"]
    assert "runtime" in results["metrics"]
    assert "summary" in results

def test_model_benchmarking_cleanup(test_config):
    """测试模型基准测试的资源清理。"""
    benchmarking = ModelBenchmarking(test_config)
    benchmarking._init_components()
    benchmarking.run_benchmarks()
    benchmarking.cleanup()
    
    # 验证清理
    assert benchmarking.model is None
    assert benchmarking.dataset is None
    assert benchmarking.report_generator is None

def test_base_benchmarking_resource_cleanup():
    """测试基础基准测试的资源清理。"""
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, "model.bin")
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建模拟模型文件
    with open(model_path, "w") as f:
        f.write("mock model data")
    
    # 创建数据集文件
    dataset_path = os.path.join(temp_dir, "dataset.json")
    dataset = [
        {"input": "test1", "output": "result1"},
        {"input": "test2", "output": "result2"}
    ]
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f)
    
    config = {
        "model_name": "test_model",
        "batch_size": 1,
        "dataset_path": dataset_path,
        "model_config": {
            "device": "cpu",
            "model_path": model_path
        },
        "hardware_config": {
            "device": "cpu",
            "device_id": 0
        },
        "output_dir": output_dir
    }
    
    try:
        # 设置测试模式
        os.environ['TEST_MODE'] = '1'
        benchmark = ModelBenchmarking(config)  # 使用具体实现类
        
        # 运行基准测试
        benchmark.run_benchmarks()
        
        # 执行清理
        benchmark.cleanup()
        
        # 验证清理结果
        assert benchmark.model is None
        assert benchmark.dataset is None
        assert benchmark.report_generator is None
    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir)

def test_base_benchmarking_state_recovery():
    """测试基础基准测试的状态恢复功能。"""
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, "model.bin")
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建模拟模型文件
    with open(model_path, "w") as f:
        f.write("mock model data")
    
    # 创建数据集文件
    dataset_path = os.path.join(temp_dir, "dataset.json")
    dataset = [
        {"input": "test1", "output": "result1"},
        {"input": "test2", "output": "result2"}
    ]
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f)
    
    config = {
        "model_name": "test_model",
        "batch_size": 1,
        "dataset_path": dataset_path,
        "model_config": {
            "device": "cpu",
            "model_path": model_path
        },
        "hardware_config": {
            "device": "cpu",
            "device_id": 0
        },
        "output_dir": output_dir
    }
    
    try:
        # 设置测试模式
        os.environ['TEST_MODE'] = '1'
        benchmark = ModelBenchmarking(config)  # 使用具体实现类
        
        # 运行基准测试
        results = benchmark.run_benchmarks()
        
        # 保存状态
        state_file = os.path.join(temp_dir, "state.json")
        with open(state_file, "w") as f:
            json.dump(results, f)
        
        # 清理并重新加载
        benchmark.cleanup()
        with open(state_file, "r") as f:
            loaded_results = json.load(f)
        
        # 验证状态恢复
        assert loaded_results == results
    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir)

def test_base_benchmarking_config_validation():
    """测试基础基准测试配置验证。"""
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, "model.bin")
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建模拟模型文件
    with open(model_path, "w") as f:
        f.write("mock model data")
    
    # 创建数据集文件
    dataset_path = os.path.join(temp_dir, "dataset.json")
    dataset = [
        {"input": "test1", "output": "result1"}
    ]
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f)
    
    try:
        # 测试空配置
        with pytest.raises(ValueError, match="配置缺少必需字段: model_name"):
            ModelBenchmarking({})  # 使用具体实现类
        
        # 测试缺少必需字段的配置
        invalid_config = {
            "batch_size": 1
        }
        with pytest.raises(ValueError, match="配置缺少必需字段: model_name"):
            ModelBenchmarking(invalid_config)
        
        # 测试有效配置
        valid_config = {
            "model_name": "test_model",
            "batch_size": 1,
            "dataset_path": dataset_path,
            "model_config": {
                "device": "cpu",
                "model_path": model_path
            },
            "hardware_config": {
                "device": "cpu",
                "device_id": 0
            },
            "output_dir": output_dir
        }
        # 设置测试模式
        os.environ['TEST_MODE'] = '1'
        benchmark = ModelBenchmarking(valid_config)
        assert benchmark is not None
    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir)

def test_hardware_config_validation():
    """测试硬件配置验证。"""
    # 测试缺少hardware_config
    invalid_config = {
        "model_name": "test_model",
        "batch_size": 1,
        "dataset": [{"input": "test"}],
        "model_config": {"model_type": "mock"}
    }
    with pytest.raises(ValueError, match="配置缺少必需字段: hardware_config"):
        SystemBenchmarking(invalid_config)

    # 测试无效的device
    invalid_config = {
        "model_name": "test_model",
        "batch_size": 1,
        "dataset": [{"input": "test"}],
        "model_config": {"model_type": "mock"},
        "hardware_config": {"device": "invalid_device"}
    }
    with pytest.raises(ValueError, match="不支持的设备类型: invalid_device"):
        SystemBenchmarking(invalid_config)

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

def test_report_generator_invalid_data_handling(tmp_path):
    """测试报告生成器的无效数据处理。"""
    output_dir = tmp_path / "output"
    os.makedirs(output_dir)
    generator = ReportGenerator(str(output_dir))

    # 测试空指标数据
    with pytest.raises(ValueError, match="基准测试结果不能为空"):
        generator.generate_report({})
