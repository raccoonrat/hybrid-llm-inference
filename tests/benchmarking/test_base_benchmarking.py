"""基准测试基类的测试模块。"""

import os
import json
import pytest
import tempfile
import shutil
from typing import Dict, Any, List
from src.benchmarking.base_benchmarking import BaseBenchmarking

class MockBaseBenchmarking(BaseBenchmarking):
    """用于测试的模拟基准测试类。"""
    
    def _validate_config(self) -> None:
        """验证配置。"""
        if "mock_field" not in self.config:
            raise ValueError("mock_field 不能为空")
    
    def _init_components(self) -> None:
        """初始化组件。"""
        self.initialized = True
        self.temp_file = tempfile.mktemp()
        with open(self.temp_file, 'w') as f:
            f.write("test")
        self.register_resource(self.temp_file)
        
        self.temp_dir = tempfile.mkdtemp()
        self.register_resource(self.temp_dir)
    
    def run_benchmarks(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """运行基准测试。"""
        return {
            "mock_metric": 100.0,
            "tasks_count": len(tasks)
        }
    
    def get_metrics(self) -> Dict[str, float]:
        """获取性能指标。"""
        return {
            "mock_metric": 100.0
        }

@pytest.fixture
def temp_dir():
    """创建临时目录。"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

@pytest.fixture
def valid_config(temp_dir):
    """创建有效的配置。"""
    return {
        "dataset_path": os.path.join(temp_dir, "dataset"),
        "output_dir": os.path.join(temp_dir, "output"),
        "hardware_config": {"device": "cpu"},
        "model_config": {"model_type": "mock"},
        "mock_field": "test"
    }

@pytest.fixture
def mock_benchmarking(valid_config):
    """创建模拟基准测试实例。"""
    benchmarking = MockBaseBenchmarking(valid_config)
    yield benchmarking
    benchmarking.cleanup()

def test_init_with_valid_config(valid_config):
    """测试使用有效配置初始化。"""
    benchmarking = MockBaseBenchmarking(valid_config)
    assert benchmarking.dataset_path == valid_config["dataset_path"]
    assert benchmarking.output_dir == valid_config["output_dir"]
    assert benchmarking.hardware_config == valid_config["hardware_config"]
    assert benchmarking.model_config == valid_config["model_config"]
    assert benchmarking.initialized

def test_init_with_invalid_config():
    """测试使用无效配置初始化。"""
    invalid_configs = [
        {},  # 空配置
        {"dataset_path": 123},  # 错误类型
        {
            "dataset_path": "path",
            "output_dir": "path",
            "hardware_config": "invalid",  # 错误类型
            "model_config": {}
        }
    ]
    
    for config in invalid_configs:
        with pytest.raises(ValueError):
            MockBaseBenchmarking(config)

def test_cleanup(mock_benchmarking):
    """测试资源清理。"""
    # 获取临时文件和目录的路径
    temp_file = mock_benchmarking.temp_file
    temp_dir = mock_benchmarking.temp_dir
    
    # 确保临时资源存在
    assert os.path.exists(temp_file)
    assert os.path.exists(temp_dir)
    
    # 清理资源
    mock_benchmarking.cleanup()
    
    # 验证资源已被清理
    assert not os.path.exists(temp_file)
    assert not os.path.exists(temp_dir)
    assert not mock_benchmarking.initialized
    assert len(mock_benchmarking.resources) == 0

def test_register_resource(mock_benchmarking, temp_dir):
    """测试资源注册。"""
    test_file = os.path.join(temp_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("test")
    
    # 注册资源
    mock_benchmarking.register_resource(test_file)
    assert test_file in mock_benchmarking.resources
    
    # 清理时应该删除注册的资源
    mock_benchmarking.cleanup()
    assert not os.path.exists(test_file)

def test_run_benchmarks(mock_benchmarking):
    """测试运行基准测试。"""
    tasks = [
        {"input": "test1"},
        {"input": "test2"}
    ]
    
    results = mock_benchmarking.run_benchmarks(tasks)
    assert results["mock_metric"] == 100.0
    assert results["tasks_count"] == len(tasks)

def test_get_metrics(mock_benchmarking):
    """测试获取性能指标。"""
    metrics = mock_benchmarking.get_metrics()
    assert isinstance(metrics, dict)
    assert metrics["mock_metric"] == 100.0 