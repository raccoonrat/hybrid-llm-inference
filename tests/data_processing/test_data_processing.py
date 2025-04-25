"""数据处理模块的测试用例。"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
from src.data_processing.data_loader import DataLoader
from src.data_processing.token_processor import TokenProcessor, MockTokenizer
from src.data_processing.token_processing import TokenProcessing

@pytest.fixture
def mock_data():
    """创建模拟数据。"""
    return [
        {"prompt": "Hello, how are you?", "response": "I'm fine, thank you."},
        {"prompt": "What is your name?", "response": "My name is AI."},
        {"prompt": "Tell me a joke.", "response": "Why did the chicken cross the road?"}
    ]

@pytest.fixture
def mock_data_file(tmp_path, mock_data):
    """创建模拟数据文件。"""
    file_path = tmp_path / "test_data.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(mock_data, f)
    return file_path

@pytest.fixture
def mock_dataframe():
    """创建模拟DataFrame。"""
    return pd.DataFrame({
        "prompt": ["Hello", "Hi", "Hey"],
        "response": ["World", "There", "You"]
    })

@pytest.fixture
def mock_models():
    """创建模拟模型。"""
    class MockModel:
        def get_token_count(self, text):
            return len(text.split())
    return {"llama3": MockModel()}

def test_data_loader_load_valid_file(mock_data_file):
    """测试加载有效的数据文件。"""
    loader = DataLoader()
    data = loader.load(mock_data_file)
    assert len(data) == 3
    assert isinstance(data, list)
    assert all(isinstance(item, dict) for item in data)

def test_data_loader_load_nonexistent_file(tmp_path):
    """测试加载不存在的文件。"""
    loader = DataLoader()
    with pytest.raises(FileNotFoundError):
        loader.load(tmp_path / "nonexistent.json")

def test_data_loader_load_invalid_json(tmp_path):
    """测试加载无效的JSON文件。"""
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text("{invalid json}")
    loader = DataLoader()
    with pytest.raises(json.JSONDecodeError):
        loader.load(invalid_file)

def test_token_processor_init():
    """测试TokenProcessor初始化。"""
    os.environ["TEST_MODE"] = "true"
    processor = TokenProcessor()
    assert isinstance(processor.tokenizer, MockTokenizer)
    os.environ.pop("TEST_MODE", None)

def test_token_processor_process():
    """测试TokenProcessor的process方法。"""
    os.environ["TEST_MODE"] = "true"
    processor = TokenProcessor()
    tokens = processor.process("Hello world")
    assert isinstance(tokens, list)
    assert len(tokens) == 11  # "Hello world" 的字符数
    os.environ.pop("TEST_MODE", None)

def test_token_processor_encode_decode():
    """测试TokenProcessor的编码和解码。"""
    os.environ["TEST_MODE"] = "true"
    processor = TokenProcessor()
    text = "Hello world"
    encoded = processor.encode(text)
    decoded = processor.decode(encoded)
    assert isinstance(encoded, list)
    assert isinstance(decoded, str)
    os.environ.pop("TEST_MODE", None)

def test_token_processor_batch_process():
    """测试TokenProcessor的批量处理。"""
    os.environ["TEST_MODE"] = "true"
    processor = TokenProcessor()
    texts = ["Hello", "World"]
    results = processor.batch_process(texts)
    assert isinstance(results, list)
    assert len(results) == 2
    os.environ.pop("TEST_MODE", None)

def test_token_processing_init(mock_dataframe, mock_models):
    """测试TokenProcessing初始化。"""
    processor = TokenProcessing(mock_dataframe, mock_models)
    assert processor.data.equals(mock_dataframe)
    assert processor.models == mock_models

def test_token_processing_process_tokens(mock_dataframe, mock_models):
    """测试TokenProcessing的process_tokens方法。"""
    processor = TokenProcessing(mock_dataframe, mock_models)
    token_data = processor.process_tokens("llama3")
    assert isinstance(token_data, pd.DataFrame)
    assert "input_tokens" in token_data.columns
    assert "output_tokens" in token_data.columns

def test_token_processing_compute_distribution(mock_dataframe, mock_models, tmp_path):
    """测试TokenProcessing的compute_distribution方法。"""
    processor = TokenProcessing(mock_dataframe, mock_models, output_dir=tmp_path)
    processor.process_tokens("llama3")
    distribution = processor.compute_distribution()
    assert isinstance(distribution, dict)
    assert "input_distribution" in distribution
    assert "output_distribution" in distribution
    assert (tmp_path / "token_distribution.pkl").exists()
    assert (tmp_path / "token_distribution.png").exists()

def test_token_processing_get_token_data(mock_dataframe, mock_models):
    """测试TokenProcessing的get_token_data方法。"""
    processor = TokenProcessing(mock_dataframe, mock_models)
    assert processor.get_token_data().empty
    processor.process_tokens("llama3")
    assert not processor.get_token_data().empty 