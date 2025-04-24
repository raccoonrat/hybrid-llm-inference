# hybrid-llm-inference/tests/unit/test_dataset_manager.py
"""数据集管理模块测试。"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
import json
import pandas as pd
import numpy as np
from src.dataset_manager.alpaca_loader import AlpacaLoader
from src.dataset_manager.data_processor import DataProcessor
from src.dataset_manager.token_distribution import TokenDistribution

@pytest.fixture(autouse=True)
def setup_test_env():
    """设置测试环境。"""
    os.environ["PYTHONPATH"] = str(Path.cwd())
    os.environ["TEST_MODE"] = "true"
    yield
    if "TEST_MODE" in os.environ:
        del os.environ["TEST_MODE"]

@pytest.fixture
def mock_dataset(tmp_path):
    """创建模拟数据集。"""
    data = [
        {"instruction": "写一个故事", "input": "", "output": "从前..."},
        {"instruction": "解释AI", "input": "", "output": "AI是..."}
    ]
    file_path = tmp_path / "test.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return file_path

@pytest.fixture
def sample_alpaca_data():
    """创建示例Alpaca数据。"""
    return pd.DataFrame({
        'instruction': ['写一个故事', '解释AI'],
        'input': ['', ''],
        'output': ['从前...', 'AI是...']
    })

@pytest.fixture
def alpaca_loader(mock_dataset):
    """创建AlpacaLoader实例。"""
    return AlpacaLoader(str(mock_dataset))

@pytest.fixture
def data_processor():
    """创建DataProcessor实例。"""
    return DataProcessor()

@pytest.fixture
def token_distribution(sample_alpaca_data):
    """创建TokenDistribution实例。"""
    return TokenDistribution(sample_alpaca_data, {"llama3": None})

def test_alpaca_loader_initialization(mock_dataset):
    """测试AlpacaLoader初始化。"""
    loader = AlpacaLoader(str(mock_dataset))
    assert loader.dataset_path == Path(mock_dataset)
    assert loader.data is None

def test_alpaca_loader_load(alpaca_loader):
    """测试AlpacaLoader的数据加载功能。"""
    data = alpaca_loader.load()
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 2
    assert "instruction" in data.columns
    assert "input" in data.columns
    assert "output" in data.columns

def test_alpaca_loader_empty_file(tmp_path):
    """测试空文件加载。"""
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("[]")
    
    with pytest.raises(ValueError, match="Dataset is empty"):
        AlpacaLoader(str(empty_file)).load()

def test_alpaca_loader_invalid_file(tmp_path):
    """测试无效文件加载。"""
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text("invalid json")
    
    with pytest.raises(json.JSONDecodeError):
        AlpacaLoader(str(invalid_file)).load()

def test_data_processor_initialization():
    """测试DataProcessor初始化。"""
    processor = DataProcessor()
    assert processor.logger is not None

def test_data_processing(data_processor):
    """测试数据处理功能。"""
    data = [
        {"text": "测试1", "label": 0},
        {"text": "测试2", "label": 1}
    ]
    processed = data_processor.process(data)
    assert isinstance(processed, list)
    assert len(processed) == 2
    for item in processed:
        assert "text" in item
        assert "length" in item
        assert "processed" in item

def test_data_processor_invalid_data(data_processor):
    """测试无效数据处理。"""
    invalid_data = [
        {"invalid": "data"},
        None,
        "string"
    ]
    processed = data_processor.process(invalid_data)
    assert len(processed) == 0

def test_token_distribution_initialization(sample_alpaca_data):
    """测试TokenDistribution初始化。"""
    distribution = TokenDistribution(sample_alpaca_data, {"llama3": None})
    assert distribution.data.equals(sample_alpaca_data)
    assert "llama3" in distribution.models
    assert distribution.distribution is None
    assert distribution.stats is None

def test_token_distribution_analyze(token_distribution):
    """测试token分布分析功能。"""
    distribution, stats = token_distribution.analyze("llama3")
    assert isinstance(distribution, dict)
    assert "input_distribution" in distribution
    assert "output_distribution" in distribution
    assert isinstance(stats, dict)
    assert "input" in stats
    assert "output" in stats

def test_token_distribution_visualization(token_distribution, tmp_path):
    """测试token分布可视化功能。"""
    token_distribution.output_dir = tmp_path
    token_distribution.analyze("llama3")
    plot_path = tmp_path / "token_distribution.png"
    assert plot_path.exists()

def test_token_distribution_invalid_model(token_distribution):
    """测试无效模型分析。"""
    with pytest.raises(ValueError, match="Model invalid_model not found"):
        token_distribution.analyze("invalid_model")
