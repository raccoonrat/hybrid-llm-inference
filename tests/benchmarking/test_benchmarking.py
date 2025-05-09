"""基准测试模块。"""

import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, List
import re
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
import yaml
import torch.nn as nn

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from src.model_zoo.base_model import BaseModel
from toolbox.logger import get_logger
from .test_benchmark_classes import TestSystemBenchmarking, TestModelBenchmarking

logger = get_logger(__name__)

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
    """完全采用真实配置文件, 自动补全所有常用字段, 仅动态生成数据集和输出路径。"""
    # 读取真实配置
    with open("configs/model_config.yaml", "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)
    with open("configs/hardware_config.yaml", "r", encoding="utf-8") as f:
        hardware_config = yaml.safe_load(f)
    with open("configs/scheduler_config.yaml", "r", encoding="utf-8") as f:
        scheduler_config = yaml.safe_load(f)

    # 自动获取第一个模型名和batch_size、model_path
    first_model_name = next(iter(model_config["models"]))
    first_model_cfg = model_config["models"][first_model_name]
    batch_size = first_model_cfg.get("batch_size", 1)
    model_path = first_model_cfg.get("model_path", "")

    # 补全 model_config 顶层 model_path 字段（兼容部分代码直接访问）
    model_config["model_path"] = model_path

    # 自动补全 hardware_config 顶层 device 字段
    if "device" not in hardware_config:
        if "devices" in hardware_config and isinstance(hardware_config["devices"], dict):
            first_device_name = next(iter(hardware_config["devices"]))
            hardware_config["device"] = first_device_name
            # 补充 device_id
            hardware_config["device_id"] = hardware_config["devices"][first_device_name].get("device_id", 0)
        else:
            hardware_config["device"] = "cuda"
            hardware_config["device_id"] = 0

    # 统一 device 字段为 'cuda' 或 'cpu'
    gpu_names = ["rtx", "a100", "cuda", "nvidia", "gpu", "m1_pro"]
    device_val = hardware_config["device"].lower()
    if any(name in device_val for name in gpu_names):
        hardware_config["device"] = "cuda"
    else:
        hardware_config["device"] = "cpu"

    # 动态生成数据集
    temp_dir = tempfile.mkdtemp()
    dataset_path = os.path.join(temp_dir, "dataset.json")
    dataset = [
        {"input": "测试输入1", "output": "测试输出1", "tokens": 10},
        {"input": "测试输入2", "output": "测试输出2", "tokens": 15}
    ]
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False)

    output_dir = "output/benchmark_test"
    test_config = {
        "model_name": first_model_name,
        "batch_size": batch_size,
        "model_path": model_path,
        "model_config": model_config,
        "hardware_config": hardware_config,
        "scheduler_config": scheduler_config,
        "dataset_path": dataset_path,
        "output_dir": output_dir
    }

    yield test_config

    shutil.rmtree(temp_dir)

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
    instance = SystemBenchmarking(test_config)
    yield instance
    instance.cleanup()

@pytest.fixture
def system_benchmarking(test_config, mock_model_files):
    """系统基准测试fixture。"""
    test_config["model_config"]["model_path"] = mock_model_files
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
    instance = SystemBenchmarking(test_config)
    yield instance
    # 清理
    if os.path.exists("benchmark_state.json"):
        os.remove("benchmark_state.json")

# ============= SystemBenchmarking Tests =============

def test_system_benchmarking_initialization(test_config):
    """测试系统基准测试初始化。"""
    benchmark = SystemBenchmarking(test_config)
    assert benchmark.dataset is not None
    assert len(benchmark.dataset) > 0
    assert benchmark.model is not None
    assert benchmark.scheduler is not None

def test_system_benchmarking_run_benchmarks(test_config):
    """测试运行基准测试。"""
    benchmark = SystemBenchmarking(test_config)
    results = benchmark.run_benchmarks()
    assert isinstance(results, dict)
    assert len(results) > 0
    for task_id, task_result in results.items():
        assert isinstance(task_result, dict)
        assert "latency" in task_result
        assert "throughput" in task_result
        assert "memory" in task_result
        assert "runtime" in task_result
        assert "energy" in task_result

def test_system_benchmarking_cleanup(test_config):
    """测试系统基准测试的资源清理。"""
    benchmark = SystemBenchmarking(test_config)
    benchmark.cleanup()
    # 验证资源是否被正确清理
    assert benchmark.model is None
    assert benchmark.scheduler is None
    assert benchmark.profiler is None

def test_system_benchmarking_empty_dataset(test_config):
    """测试空数据集的处理。"""
    test_config["dataset_path"] = os.path.join(os.path.dirname(test_config["dataset_path"]), "empty.json")
    with open(test_config["dataset_path"], "w") as f:
        json.dump([], f)
    benchmark = SystemBenchmarking(test_config)
    with pytest.raises(ValueError, match="数据集为空"):
        benchmark.run_benchmarks()

def test_system_benchmarking_scheduling_strategy(test_config):
    """测试不同调度策略的效果。"""
    benchmark = SystemBenchmarking(test_config)
    results = benchmark.run_benchmarks()
    assert isinstance(results, dict)
    assert len(results) > 0

def test_system_benchmarking_error_handling(test_config):
    """测试错误处理机制。"""
    benchmark = SystemBenchmarking(test_config)
    
    # 测试无效任务
    with pytest.raises(ValueError, match="任务格式无效"):
        benchmark.run_benchmarks([{"invalid_field": "test"}])
    
    # 测试无效设备
    test_config["hardware_config"]["device"] = "invalid_device"
    with pytest.raises(ValueError, match="不支持的设备类型"):
        SystemBenchmarking(test_config)

def test_real_benchmark_report_written(test_config):
    """真实基准测试后自动生成报告文件到 output/benchmark_test。"""
    benchmark = SystemBenchmarking(test_config)
    results = benchmark.run_benchmarks()
    generator = ReportGenerator(test_config["output_dir"])
    report_path = generator.generate_report(results)
    assert os.path.exists(report_path), f"报告文件未生成: {report_path}"

# ============= ModelBenchmarking Tests =============

def test_model_benchmarking_initialization(test_config):
    """测试模型基准测试初始化。"""
    benchmark = ModelBenchmarking(test_config)
    assert benchmark.dataset is not None
    assert len(benchmark.dataset) > 0
    assert benchmark.model is not None

def test_model_benchmarking_run_benchmarks(test_config):
    """测试模型基准测试的运行。"""
    benchmark = ModelBenchmarking(test_config)
    results = benchmark.run_benchmarks()
    
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
    benchmark = ModelBenchmarking(test_config)
    benchmark.cleanup()
    assert benchmark.model is None
    assert benchmark.dataset is None
    assert benchmark.report_generator is None

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

# ============= ReportGenerator Tests =============

def test_report_generator_initialization(tmp_path):
    """测试报告生成器初始化。"""
    output_dir = tmp_path / "output"
    os.makedirs(output_dir)
    generator = ReportGenerator(str(output_dir))
    assert generator.output_dir == str(output_dir)

def test_report_generator_valid_output(tmp_path):
    """测试报告生成器生成有效输出。"""
    metrics = {
        "metrics": {
            "latency": {
                "value": 100.0,
                "mean": 100.0,
                "std": 10.0,
                "min": 80.0,
                "max": 120.0,
                "unit": "ms"
            },
            "energy": {
                "value": 50.0,
                "mean": 50.0,
                "std": 5.0,
                "min": 40.0,
                "max": 60.0,
                "unit": "J"
            },
            "throughput": {
                "value": 1000.0,
                "unit": "tokens/s"
            }
        }
    }
    
    output_dir = tmp_path / "output"
    os.makedirs(output_dir)
    
    generator = ReportGenerator(str(output_dir))
    generator.generate_report(metrics)
    
    # 验证输出文件是否存在
    report_files = list(output_dir.glob("benchmark_report_*.json"))
    assert len(report_files) > 0, "报告文件未生成"
    assert (output_dir / "visualizations").exists()

def test_report_generator_visualization(tmp_path):
    """测试报告生成器的可视化功能。"""
    output_dir = tmp_path / "output"
    os.makedirs(output_dir)
    generator = ReportGenerator(str(output_dir))

    metrics = {
        "metrics": {
            "latency": {
                "value": 100.0,
                "mean": 100.0,
                "std": 10.0,
                "min": 80.0,
                "max": 120.0,
                "distribution": [95.0, 98.0, 102.0, 105.0],
                "unit": "ms"
            },
            "energy": {
                "value": 50.0,
                "mean": 50.0,
                "std": 5.0,
                "min": 40.0,
                "max": 60.0,
                "distribution": [45.0, 48.0, 52.0, 55.0],
                "unit": "J"
            }
        }
    }
    
    generator.generate_report(metrics)
    
    # 验证可视化文件是否存在
    vis_dir = output_dir / "visualizations"
    assert vis_dir.exists()
    assert len(list(vis_dir.glob("*.png"))) > 0, "可视化文件未生成"

def test_report_generator_invalid_data_handling(tmp_path):
    """测试报告生成器的无效数据处理。"""
    output_dir = tmp_path / "output"
    os.makedirs(output_dir)
    generator = ReportGenerator(str(output_dir))

    # 测试空指标数据
    with pytest.raises(ValueError, match="基准测试结果不能为空"):
        generator.generate_report({})
