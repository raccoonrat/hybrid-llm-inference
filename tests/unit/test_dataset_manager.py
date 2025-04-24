# hybrid-llm-inference/tests/unit/test_dataset_manager.py
import pytest
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from src.dataset_manager.alpaca_loader import AlpacaLoader
from dataset_manager.data_processor import DataProcessor
from src.dataset_manager.token_distribution import TokenDistribution

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
def token_distribution():
    return TokenDistribution()

def test_alpaca_loader_empty_file(tmp_path):
    """测试空文件加载"""
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("[]")
    
    with pytest.raises(ValueError, match="Dataset is empty"):
        AlpacaLoader(empty_file).load()

def test_data_processing(mock_dataset):
    """测试数据处理"""
    # 设置测试模式环境变量
    os.environ['TEST_MODE'] = '1'
    
    processor = DataProcessor(
        model_name="tinyllama",
        model_path="models/TinyLlama-1.1B-Chat-v1.0",
        mode="local"
    )
    
    data = AlpacaLoader(mock_dataset).load()
    processed_data = processor.process(data)
    
    assert len(processed_data) > 0
    assert "input_tokens" in processed_data[0]
    assert "output_tokens" in processed_data[0]

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

def test_token_distribution(token_distribution, sample_alpaca_data, tmp_path):
    # Test token distribution analysis
    distribution = token_distribution.analyze(sample_alpaca_data['instruction'])
    assert isinstance(distribution, dict)
    assert 'mean' in distribution
    assert 'std' in distribution
    assert 'min' in distribution
    assert 'max' in distribution
    
    # Test distribution visualization
    output_path = tmp_path / "token_distribution.png"
    token_distribution.visualize(distribution, output_path)
    assert output_path.exists()
    
    # Test distribution saving
    save_path = tmp_path / "token_distribution.pkl"
    token_distribution.save(distribution, save_path)
    assert save_path.exists()
    
    # Test distribution loading
    loaded_distribution = token_distribution.load(save_path)
    assert isinstance(loaded_distribution, dict)
    assert loaded_distribution['mean'] == distribution['mean']
