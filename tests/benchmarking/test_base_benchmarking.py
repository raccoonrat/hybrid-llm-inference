import pytest
import os
from src.benchmarking.base_benchmarking import BaseBenchmarking
from src.benchmarking.mock_benchmarking import MockBenchmarking

def test_base_benchmarking_config_validation():
    """测试基准测试配置验证"""
    # 测试缺少必需字段
    invalid_config = {
        "model_name": "test_model",
        "batch_size": 32
    }
    with pytest.raises(ValueError, match=r".*dataset_path.*"):
        MockBenchmarking(invalid_config)

    # 测试无效的批处理大小
    invalid_batch_config = {
        "model_name": "test_model",
        "dataset_path": "/path/to/dataset",
        "batch_size": -1
    }
    with pytest.raises(ValueError, match=r".*batch_size.*"):
        MockBenchmarking(invalid_batch_config)

    # 测试有效配置
    valid_config = {
        "model_name": "test_model",
        "dataset_path": "/path/to/dataset",
        "batch_size": 32,
        "num_threads": 4,
        "device": "cpu",
        "metrics": ["latency", "energy", "throughput"],
        "model_config": {
            "model_type": "mock",
            "model_path": "mock_path"
        },
        "hardware_config": {
            "device": "cpu",
            "device_id": 0
        }
    }
    benchmarking = MockBenchmarking(valid_config)
    assert benchmarking.config == valid_config

def test_base_benchmarking_run():
    """测试基准测试运行方法"""
    config = {
        "model_name": "test_model",
        "dataset_path": "/path/to/dataset",
        "batch_size": 32,
        "num_threads": 4,
        "device": "cpu",
        "metrics": ["latency", "energy", "throughput"],
        "model_config": {
            "model_type": "mock",
            "model_path": "mock_path"
        },
        "hardware_config": {
            "device": "cpu",
            "device_id": 0,
            "num_threads": 4
        }
    }
    benchmarking = MockBenchmarking(config)
    results = benchmarking.run()
    assert isinstance(results, dict)
    assert "metrics" in results

def test_base_benchmarking_collect_metrics():
    """测试基准测试指标收集方法"""
    config = {
        "model_name": "test_model",
        "dataset_path": "/path/to/dataset",
        "batch_size": 32,
        "num_threads": 4,
        "device": "cpu",
        "metrics": ["latency", "energy", "throughput"],
        "model_config": {
            "model_type": "mock",
            "model_path": "mock_path"
        },
        "hardware_config": {
            "device": "cpu",
            "device_id": 0,
            "num_threads": 4
        }
    }
    benchmarking = MockBenchmarking(config)
    metrics = benchmarking.collect_metrics()
    assert isinstance(metrics, dict)
    assert "latency" in metrics
    assert "energy" in metrics

def test_base_benchmarking_validate_metrics():
    """测试基准测试指标验证"""
    config = {
        "model_name": "test_model",
        "dataset_path": "/path/to/dataset",
        "batch_size": 32,
        "num_threads": 4,
        "device": "cpu",
        "metrics": ["latency", "energy", "throughput"],
        "model_config": {
            "model_type": "mock",
            "model_path": "mock_path"
        },
        "hardware_config": {
            "device": "cpu",
            "device_id": 0,
            "num_threads": 4
        }
    }
    benchmarking = MockBenchmarking(config)

    # 测试无效的指标数据
    invalid_metrics = {
        "latency": [0.1, 0.2, 0.3],
        # 缺少必需的 energy 指标
        "throughput": 100.0
    }
    with pytest.raises(ValueError, match=r".*energy.*"):
        benchmarking.validate_metrics(invalid_metrics)

    # 测试有效的指标数据
    valid_metrics = {
        "latency": [0.1, 0.2, 0.3],
        "energy": [50.0, 51.0, 49.0],
        "throughput": 100.0
    }
    # 验证不应该抛出异常
    benchmarking.validate_metrics(valid_metrics) 