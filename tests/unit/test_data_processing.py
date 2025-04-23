# hybrid-llm-inference/tests/unit/test_data_processing.py
import pytest
import json
import os
import pandas as pd
from pathlib import Path
from data_processing.data_loader import DataLoader
from data_processing.token_processor import TokenProcessor
from model_zoo import get_model

@pytest.fixture
def mock_data(tmp_path):
    """创建模拟数据"""
    data = [
        {"prompt": "test1", "response": "response1"},
        {"prompt": "test2", "response": "response2"}
    ]
    file_path = tmp_path / "test.json"
    with open(file_path, "w") as f:
        json.dump(data, f)
    return file_path

@pytest.fixture
def model():
    """Create a mock model for testing."""
    config = {"model_name": "meta-llama/Llama-3-8B", "mode": "local", "max_length": 512}
    return get_model("llama3", "local", config)

def test_data_loader(mock_data):
    """测试数据加载"""
    loader = DataLoader(mock_data)
    data = loader.load()
    
    assert len(data) == 2
    assert data[0]["prompt"] == "test1"
    assert data[0]["response"] == "response1"

def test_token_processing(mock_data):
    """测试 token 处理"""
    # 设置测试模式环境变量
    os.environ['TEST_MODE'] = '1'
    
    processor = TokenProcessor(
        model_name="tinyllama",
        model_path="models/TinyLlama-1.1B-Chat-v1.0",
        mode="local"
    )
    
    data = DataLoader(mock_data).load()
    processed_data = processor.process(data)
    
    assert len(processed_data) == 2
    assert "input_tokens" in processed_data[0]
    assert "output_tokens" in processed_data[0]
    assert processed_data[0]["input_tokens"] > 0
    assert processed_data[0]["output_tokens"] > 0

def test_data_loader(sample_data):
    loader = DataLoader(sample_data)
    data = loader.load()
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 2
    assert list(data.columns) == ['prompt', 'response']

def test_token_processing(sample_data, model, tmp_path):
    # Load data
    loader = DataLoader(sample_data)
    data = loader.load()

    # Process tokens
    processor = TokenProcessor(
        model_name="tinyllama",
        model_path="models/TinyLlama-1.1B-Chat-v1.0",
        mode="local"
    )
    processed_data = processor.process(data)
    
    assert isinstance(processed_data, pd.DataFrame)
    assert len(processed_data) == 2
    assert set(processed_data.columns) == {'prompt', 'response', 'input_tokens', 'output_tokens'}
    assert all(processed_data['input_tokens'] > 0)
    assert all(processed_data['output_tokens'] > 0)

    # Compute distribution
    distribution = processor.compute_distribution()
    assert 'input_distribution' in distribution
    assert 'output_distribution' in distribution
    assert (tmp_path / 'token_distribution.pkl').exists()
    assert (tmp_path / 'token_distribution.png').exists()
