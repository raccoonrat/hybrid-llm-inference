# hybrid-llm-inference/tests/unit/test_dataset_manager.py
import pytest
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from src.dataset_manager.alpaca_loader import AlpacaLoader
from src.dataset_manager.data_processor import DataProcessor
from src.dataset_manager.token_distribution import TokenDistributionAnalyzer

@pytest.fixture
def mock_dataset(tmp_path):
    """创建模拟数据集"""
    data = [
        {"instruction": "test1", "input": "", "output": "response1"},
        {"instruction": "test2", "input": "", "output": "response2"}
    ]
    file_path = tmp_path / "test.json"
    with open(file_path, "w") as f:
        json.dump(data, f)
    return file_path

@pytest.fixture
def sample_alpaca_data():
    return pd.DataFrame({
        'instruction': ['写一个故事', '解释AI'],
        'input': ['', ''],
        'output': ['从前...', 'AI是...']
    })

@pytest.fixture
def alpaca_loader():
    return AlpacaLoader()

@pytest.fixture
def token_analyzer():
    """创建TokenDistributionAnalyzer实例。"""
    return TokenDistributionAnalyzer()

@pytest.fixture
def data_processor():
    """创建DataProcessor实例。"""
    return DataProcessor()

def test_alpaca_loader_empty_file(tmp_path):
    """测试空文件加载"""
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("[]")
    
    with pytest.raises(ValueError, match="Dataset is empty"):
        AlpacaLoader(empty_file).load()

def test_data_processing(data_processor):
    """测试数据处理功能。"""
    data = [{"text": "测试1"}, {"text": "测试2"}]
    processed = data_processor.process(data)
    assert isinstance(processed, list)
    assert len(processed) == 2

def test_data_processing_no_response(mock_dataset):
    """测试无响应数据处理"""
    # 设置测试模式环境变量
    os.environ['TEST_MODE'] = '1'
    
    processor = DataProcessor(
        model_name="tinyllama",
        model_path="models/TinyLlama-1.1B-Chat-v1.0",
        mode="local"
    )
    
    data = AlpacaLoader(mock_dataset).load()
    # 移除响应
    for item in data:
        item["output"] = ""
    
    processed_data = processor.process(data)
    assert len(processed_data) > 0
    assert all(item["output_tokens"] == 0 for item in processed_data)

def test_alpaca_loader(alpaca_loader, sample_alpaca_data, tmp_path):
    # Save test data
    data_path = tmp_path / "alpaca_data.json"
    sample_alpaca_data.to_json(data_path, orient='records', force_ascii=False)
    
    # Test data loading
    loaded_data = alpaca_loader.load_data(data_path)
    assert isinstance(loaded_data, pd.DataFrame)
    assert len(loaded_data) == len(sample_alpaca_data)
    
    # Test data preprocessing
    processed_data = alpaca_loader.preprocess(loaded_data)
    assert isinstance(processed_data, pd.DataFrame)
    assert len(processed_data) == len(sample_alpaca_data)

def test_token_distribution(token_analyzer):
    """测试token分布分析功能。"""
    texts = ["这是一个测试句子", "这是另一个测试句子"]
    distribution = token_analyzer.analyze(texts)
    assert isinstance(distribution, dict)
    assert "mean" in distribution
    assert "std" in distribution
