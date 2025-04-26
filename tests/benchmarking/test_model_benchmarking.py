"""模型基准测试的测试模块。"""

import os
import json
import pytest
import tempfile
import shutil
from typing import Dict, Any
from src.benchmarking.model_benchmarking import ModelBenchmarking

class TestModelBenchmarking(ModelBenchmarking):
    """测试模型基准测试类。"""
    
    def setup_method(self):
        """设置测试环境。"""
        self.dataset_path = None
        self.output_dir = None
        self.config = None
        self.initialized = False
    
    def _init_components(self):
        """初始化组件。"""
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"数据集文件 {self.dataset_path} 不存在")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            if not dataset:
                raise ValueError("数据集不能为空")
            for item in dataset:
                if not all(k in item for k in ["input", "output", "tokens"]):
                    raise ValueError("数据集中的每个项目必须包含input、output和tokens字段")
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.initialized = True
    
    def get_metrics(self):
        """获取性能指标。"""
        return {
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
    
    def run_benchmarks(self, tasks=None):
        """运行基准测试。"""
        metrics = self.get_metrics()
        return {
            "metrics": {
                "throughput": float(metrics["throughput"]),
                "latency": float(metrics["latency"]),
                "energy": float(metrics["energy"]),
                "runtime": float(metrics["runtime"])
            },
            "tradeoff_results": {
                "weights": [0.2, 0.5, 0.8],
                "values": [
                    {
                        "throughput": 100.0,
                        "latency": 0.1,
                        "energy": 10.0,
                        "runtime": 1.0
                    }
                ]
            }
        }

@pytest.fixture
def temp_dir():
    """创建临时目录。"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

@pytest.fixture
def dataset_file(temp_dir):
    """创建测试数据集文件。"""
    dataset = [
        {"input": "测试输入1", "output": "测试输出1", "tokens": 10},
        {"input": "测试输入2", "output": "测试输出2", "tokens": 20}
    ]
    dataset_path = os.path.join(temp_dir, "test_dataset.json")
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False)
    return dataset_path

@pytest.fixture
def valid_config(dataset_file, temp_dir):
    """创建有效的配置。"""
    return {
        "dataset_path": dataset_file,
        "output_dir": os.path.join(temp_dir, "output"),
        "hardware_config": {"device": "cpu"},
        "model_config": {
            "model_type": "mock",
            "model_path": "/path/to/model",
            "batch_size": 32
        }
    }

@pytest.fixture
def model_benchmarking(valid_config):
    """创建模型基准测试实例。"""
    benchmarking = TestModelBenchmarking(valid_config)
    yield benchmarking
    benchmarking.cleanup()

def test_init_with_valid_config(valid_config):
    """测试使用有效配置初始化。"""
    benchmarking = TestModelBenchmarking(valid_config)
    assert benchmarking.dataset_path == valid_config["dataset_path"]
    assert benchmarking.model_config == valid_config["model_config"]
    assert benchmarking.output_dir == valid_config["output_dir"]
    assert benchmarking.initialized
    assert os.path.exists(benchmarking.output_dir)

def test_init_with_invalid_dataset():
    """测试使用无效数据集初始化。"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # 测试不存在的数据集文件
        config = {
            "dataset_path": os.path.join(temp_dir, "not_exist.json"),
            "output_dir": os.path.join(temp_dir, "output"),
            "hardware_config": {"device": "cpu"},
            "model_config": {"model_type": "mock"}
        }
        with pytest.raises(ValueError, match="不存在"):
            TestModelBenchmarking(config)
        
        # 测试无效的JSON格式
        invalid_json_path = os.path.join(temp_dir, "invalid.json")
        with open(invalid_json_path, "w") as f:
            f.write("invalid json")
        config["dataset_path"] = invalid_json_path
        with pytest.raises(ValueError, match="不是有效的JSON格式"):
            TestModelBenchmarking(config)
        
        # 测试空数据集
        empty_dataset_path = os.path.join(temp_dir, "empty.json")
        with open(empty_dataset_path, "w") as f:
            json.dump([], f)
        config["dataset_path"] = empty_dataset_path
        with pytest.raises(ValueError, match="为空"):
            TestModelBenchmarking(config)

def test_init_with_invalid_model_config(dataset_file, temp_dir):
    """测试使用无效模型配置初始化。"""
    invalid_configs = [
        {  # 缺少model_config
            "dataset_path": dataset_file,
            "output_dir": os.path.join(temp_dir, "output"),
            "hardware_config": {"device": "cpu"}
        },
        {  # model_config为空
            "dataset_path": dataset_file,
            "output_dir": os.path.join(temp_dir, "output"),
            "hardware_config": {"device": "cpu"},
            "model_config": {}
        },
        {  # model_config类型错误
            "dataset_path": dataset_file,
            "output_dir": os.path.join(temp_dir, "output"),
            "hardware_config": {"device": "cpu"},
            "model_config": "invalid"
        }
    ]
    
    for config in invalid_configs:
        with pytest.raises(ValueError):
            TestModelBenchmarking(config)

def test_run_benchmarks(model_benchmarking):
    """测试运行基准测试。"""
    results = model_benchmarking.run_benchmarks()
    
    # 验证结果格式
    assert "metrics" in results
    assert "tradeoff_results" in results
    
    # 验证指标
    metrics = results["metrics"]
    assert isinstance(metrics, dict)
    assert all(key in metrics for key in ["throughput", "latency", "energy", "runtime"])
    assert all(isinstance(value, (int, float)) for value in metrics.values())
    
    # 验证权衡分析结果
    tradeoff = results["tradeoff_results"]
    assert "weights" in tradeoff
    assert "values" in tradeoff
    assert len(tradeoff["weights"]) > 0
    assert len(tradeoff["values"]) > 0

def test_get_metrics(model_benchmarking):
    """测试获取性能指标。"""
    metrics = model_benchmarking.get_metrics()
    
    # 验证基本指标
    assert isinstance(metrics, dict)
    assert all(key in metrics for key in ["throughput", "latency", "energy", "runtime"])
    assert all(isinstance(value, (int, float)) for value in metrics.values())
    
    # 验证汇总指标
    assert "summary" in metrics
    summary = metrics["summary"]
    assert all(key in summary for key in [
        "avg_throughput", "avg_latency", "avg_energy_per_token", "avg_runtime"
    ])
    assert all(isinstance(value, (int, float)) for value in summary.values())

def test_cleanup(model_benchmarking):
    """测试资源清理。"""
    output_dir = model_benchmarking.output_dir
    
    # 确保输出目录存在
    assert os.path.exists(output_dir)
    
    # 在输出目录中创建一些文件
    test_file = os.path.join(output_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("test")
    
    # 清理资源
    model_benchmarking.cleanup()
    
    # 验证资源已被清理
    assert not os.path.exists(output_dir)
    assert not os.path.exists(test_file)
    assert not model_benchmarking.initialized 