"""测试模拟基准测试类模块。"""

import os
import pytest
from src.benchmarking.mock_benchmarking import MockBenchmarking

@pytest.fixture
def valid_config(tmp_path):
    """创建有效的配置字典。"""
    # 创建临时数据集目录
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    
    # 创建临时输出目录
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    return {
        "model_name": "test_model",
        "dataset_path": str(dataset_dir),
        "batch_size": 32,
        "num_threads": 4,
        "device": "cpu",
        "metrics": ["latency", "energy", "throughput"],
        "output_dir": str(output_dir),
        "hardware_config": {
            "cpu": {
                "cores": 4,
                "frequency": 2.5,
                "memory": 8
            },
            "gpu": {
                "name": "test_gpu",
                "memory": 4
            }
        },
        "model_config": {
            "name": "test_model",
            "type": "transformer",
            "parameters": {
                "num_layers": 12,
                "hidden_size": 768,
                "num_attention_heads": 12
            }
        }
    }

@pytest.fixture
def mock_benchmarking(valid_config):
    """创建 MockBenchmarking 实例。"""
    return MockBenchmarking(valid_config)

def test_mock_benchmarking_initialization(valid_config):
    """测试 MockBenchmarking 初始化。"""
    benchmarking = MockBenchmarking(valid_config)
    assert benchmarking.config == valid_config
    assert benchmarking.initialized
    assert "latency" in benchmarking.metrics
    assert "energy" in benchmarking.metrics
    assert "throughput" in benchmarking.metrics

def test_mock_benchmarking_run(mock_benchmarking):
    """测试运行基准测试。"""
    mock_benchmarking.run()
    assert len(mock_benchmarking.metrics["latency"]) == mock_benchmarking.config["batch_size"]
    assert len(mock_benchmarking.metrics["energy"]) == mock_benchmarking.config["batch_size"]
    assert isinstance(mock_benchmarking.metrics["throughput"], float)

def test_mock_benchmarking_collect_metrics(mock_benchmarking):
    """测试收集性能指标。"""
    mock_benchmarking.run()
    metrics = mock_benchmarking.collect_metrics()
    assert "latency" in metrics
    assert "energy" in metrics
    assert "throughput" in metrics
    assert len(metrics["latency"]) == mock_benchmarking.config["batch_size"]
    assert len(metrics["energy"]) == mock_benchmarking.config["batch_size"]
    assert isinstance(metrics["throughput"], float)

def test_mock_benchmarking_run_benchmarks(mock_benchmarking):
    """测试运行基准测试任务。"""
    tasks = [{"task_id": i} for i in range(10)]
    results = mock_benchmarking.run_benchmarks(tasks)
    assert len(results["latency"]) == len(tasks)
    assert len(results["energy"]) == len(tasks)
    assert isinstance(results["throughput"], float)

def test_mock_benchmarking_get_metrics(mock_benchmarking):
    """测试获取性能指标。"""
    mock_benchmarking.run()
    metrics = mock_benchmarking.get_metrics()
    assert "latency" in metrics
    assert "energy" in metrics
    assert "throughput" in metrics
    assert len(metrics["latency"]) == mock_benchmarking.config["batch_size"]
    assert len(metrics["energy"]) == mock_benchmarking.config["batch_size"]
    assert isinstance(metrics["throughput"], float)

def test_mock_benchmarking_invalid_dataset():
    """测试无效的数据集路径。"""
    with pytest.raises(ValueError, match="数据集路径不存在"):
        config = {
            "model_name": "test_model",
            "dataset_path": "/path/to/nonexistent/dataset",
            "batch_size": 32,
            "num_threads": 4,
            "device": "cpu",
            "metrics": ["latency", "energy", "throughput"]
        }
        benchmarking = MockBenchmarking(config)
        benchmarking.run()

def test_mock_benchmarking_cleanup(mock_benchmarking):
    """测试清理资源。"""
    mock_benchmarking.run()
    mock_benchmarking.cleanup()
    assert not mock_benchmarking.initialized
    assert not mock_benchmarking.resources

def test_mock_benchmarking_metrics_range(mock_benchmarking):
    """测试性能指标的范围。"""
    mock_benchmarking.run()
    metrics = mock_benchmarking.get_metrics()
    
    # 验证延迟范围
    for latency in metrics["latency"]:
        assert 0.1 <= latency <= 0.5
    
    # 验证能耗范围
    for energy in metrics["energy"]:
        assert 50.0 <= energy <= 100.0
    
    # 验证吞吐量范围
    assert 80.0 <= metrics["throughput"] <= 120.0

def test_mock_benchmarking_empty_tasks(mock_benchmarking):
    """测试空任务列表。"""
    results = mock_benchmarking.run_benchmarks([])
    assert len(results["latency"]) == 0
    assert len(results["energy"]) == 0
    assert isinstance(results["throughput"], float)

def test_mock_benchmarking_multiple_runs(mock_benchmarking):
    """测试多次运行基准测试。"""
    # 第一次运行
    mock_benchmarking.run()
    first_metrics = mock_benchmarking.get_metrics()
    
    # 第二次运行
    mock_benchmarking.run()
    second_metrics = mock_benchmarking.get_metrics()
    
    # 验证两次运行的结果不同
    assert first_metrics["latency"] != second_metrics["latency"]
    assert first_metrics["energy"] != second_metrics["energy"]
    assert first_metrics["throughput"] != second_metrics["throughput"]

def test_mock_benchmarking_invalid_config():
    """测试无效的配置。"""
    # 测试缺少必需字段
    with pytest.raises(ValueError, match="配置缺少必需字段"):
        config = {
            "model_name": "test_model",
            "batch_size": 32
        }
        MockBenchmarking(config)
    
    # 测试无效的批处理大小
    with pytest.raises(ValueError, match="batch_size 必须大于 0"):
        config = {
            "model_name": "test_model",
            "dataset_path": "/path/to/dataset",
            "batch_size": -1
        }
        MockBenchmarking(config)

def test_mock_benchmarking_resource_registration(mock_benchmarking):
    """测试资源注册。"""
    # 注册一个资源
    resource_path = "/path/to/resource"
    mock_benchmarking.register_resource(resource_path)
    assert resource_path in mock_benchmarking.resources
    
    # 清理资源
    mock_benchmarking.cleanup()
    assert resource_path not in mock_benchmarking.resources

def test_mock_benchmarking_validate_metrics(mock_benchmarking):
    """测试指标验证。"""
    # 测试有效的指标
    valid_metrics = {
        "latency": [0.1, 0.2, 0.3],
        "energy": [50.0, 51.0, 49.0],
        "throughput": 100.0
    }
    assert mock_benchmarking.validate_metrics(valid_metrics) == valid_metrics
    
    # 测试缺少必需指标
    invalid_metrics = {
        "latency": [0.1, 0.2, 0.3],
        "throughput": 100.0
    }
    with pytest.raises(ValueError, match="缺少必需的指标: energy"):
        mock_benchmarking.validate_metrics(invalid_metrics)
    
    # 测试指标类型错误
    invalid_metrics = {
        "latency": "not a list",
        "energy": [50.0, 51.0, 49.0],
        "throughput": 100.0
    }
    with pytest.raises(ValueError, match="latency 指标必须是列表或元组类型"):
        mock_benchmarking.validate_metrics(invalid_metrics) 