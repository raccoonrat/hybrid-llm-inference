"""数据集管理模块的测试用例。"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
from src.dataset_manager.alpaca_loader import AlpacaLoader
from src.dataset_manager.data_processor import DataProcessor
from src.dataset_manager.token_distribution import TokenDistribution

@pytest.fixture
def mock_alpaca_data():
    """创建模拟Alpaca数据集。"""
    return [
        {
            "instruction": "写一个故事",
            "input": "",
            "output": "从前有座山，山里有座庙..."
        },
        {
            "instruction": "解释AI",
            "input": "",
            "output": "人工智能是..."
        }
    ]

@pytest.fixture
def mock_alpaca_file(tmp_path, mock_alpaca_data):
    """创建模拟Alpaca数据文件。"""
    file_path = tmp_path / "test_alpaca.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(mock_alpaca_data, f, ensure_ascii=False)
    return file_path

@pytest.fixture
def mock_models():
    """创建模拟模型。"""
    class MockModel:
        def get_token_count(self, text):
            return len(text.split())
    return {"llama3": MockModel()}

@pytest.fixture
def mock_dataframe():
    """创建模拟DataFrame。"""
    return pd.DataFrame({
        "instruction": ["写一个故事", "解释AI"],
        "input": ["", ""],
        "output": ["从前有座山...", "人工智能是..."]
    })

def test_alpaca_loader_init(mock_alpaca_file):
    """测试AlpacaLoader初始化。"""
    loader = AlpacaLoader(str(mock_alpaca_file))
    assert loader.dataset_path == Path(mock_alpaca_file)
    assert loader.data is None

def test_alpaca_loader_load_valid_file(mock_alpaca_file):
    """测试加载有效的Alpaca数据文件。"""
    loader = AlpacaLoader(str(mock_alpaca_file))
    data = loader.load()
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 2
    assert all(col in data.columns for col in ["instruction", "input", "output"])

def test_alpaca_loader_load_nonexistent_file(tmp_path):
    """测试加载不存在的文件。"""
    loader = AlpacaLoader(str(tmp_path / "nonexistent.json"))
    with pytest.raises(FileNotFoundError):
        loader.load()

def test_alpaca_loader_load_invalid_json(tmp_path):
    """测试加载无效的JSON文件。"""
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text("{invalid json}")
    loader = AlpacaLoader(str(invalid_file))
    with pytest.raises(json.JSONDecodeError):
        loader.load()

def test_alpaca_loader_load_empty_file(tmp_path):
    """测试加载空文件。"""
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("[]")
    loader = AlpacaLoader(str(empty_file))
    with pytest.raises(ValueError, match="Dataset is empty"):
        loader.load()

def test_alpaca_loader_get_data(mock_alpaca_file):
    """测试获取已加载的数据。"""
    loader = AlpacaLoader(str(mock_alpaca_file))
    assert loader.get_data().empty  # 未加载时返回空DataFrame
    loader.load()
    assert not loader.get_data().empty  # 加载后返回数据

def test_data_processor_init():
    """测试DataProcessor初始化。"""
    processor = DataProcessor()
    assert processor.logger is not None

def test_data_processor_process_valid_data():
    """测试处理有效数据。"""
    processor = DataProcessor()
    data = [
        {"text": "测试文本1"},
        {"text": "测试文本2"}
    ]
    processed = processor.process(data)
    assert isinstance(processed, list)
    assert len(processed) == 2
    for item in processed:
        assert "text" in item
        assert "length" in item
        assert "processed" in item

def test_data_processor_process_invalid_data():
    """测试处理无效数据。"""
    processor = DataProcessor()
    invalid_data = [
        {"invalid": "data"},
        None,
        "string"
    ]
    processed = processor.process(invalid_data)
    assert len(processed) == 0

def test_data_processor_process_empty_text():
    """测试处理空文本。"""
    processor = DataProcessor()
    data = [
        {"text": ""},
        {"text": "   "}
    ]
    processed = processor.process(data)
    assert len(processed) == 0

def test_token_distribution_init(mock_dataframe, mock_models, tmp_path):
    """测试TokenDistribution初始化。"""
    distribution = TokenDistribution(mock_dataframe, mock_models, str(tmp_path))
    assert distribution.data.equals(mock_dataframe)
    assert distribution.models == mock_models
    assert distribution.output_dir == tmp_path

def test_token_distribution_analyze(mock_dataframe, mock_models, tmp_path):
    """测试token分布分析。"""
    distribution = TokenDistribution(mock_dataframe, mock_models, str(tmp_path))
    result = distribution.analyze("llama3")
    assert isinstance(result, dict)
    assert "input_distribution" in result
    assert "output_distribution" in result

def test_token_distribution_analyze_invalid_model(mock_dataframe, mock_models, tmp_path):
    """测试使用无效模型进行分析。"""
    distribution = TokenDistribution(mock_dataframe, mock_models, str(tmp_path))
    with pytest.raises(ValueError, match="Model invalid_model not found"):
        distribution.analyze("invalid_model")

def test_token_distribution_save_distribution(mock_dataframe, mock_models, tmp_path):
    """测试保存分布结果。"""
    distribution = TokenDistribution(mock_dataframe, mock_models, str(tmp_path))
    distribution.analyze("llama3")
    save_path = tmp_path / "distribution.pkl"
    distribution.save_distribution(str(save_path))
    assert save_path.exists()

def test_token_distribution_load_distribution(mock_dataframe, mock_models, tmp_path):
    """测试加载分布结果。"""
    distribution = TokenDistribution(mock_dataframe, mock_models, str(tmp_path))
    distribution.analyze("llama3")
    save_path = tmp_path / "distribution.pkl"
    distribution.save_distribution(str(save_path))
    
    loaded_distribution = TokenDistribution(mock_dataframe, mock_models, str(tmp_path))
    loaded_distribution.load_distribution(str(save_path))
    assert loaded_distribution.distribution is not None 