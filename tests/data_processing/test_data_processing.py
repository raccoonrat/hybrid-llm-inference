import os
os.environ['TEST_MODE'] = '1'

"""数据处理模块的测试用例。"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
from src.data_processing.data_loader import DataLoader
from src.data_processing.token_processor import TokenProcessor, MockTokenizer
from src.data_processing.token_processing import TokenProcessing

@pytest.fixture
def model_path(tmp_path):
    """创建测试用的模型路径。"""
    model_dir = tmp_path / "models" / "TinyLlama-1.1B-Chat-v1.0"
    model_dir.mkdir(parents=True)
    return str(model_dir)

@pytest.fixture
def mock_data():
    """创建模拟数据。"""
    return [
        {
            "instruction": "Hello, how are you?",
            "input": "",
            "output": "I'm fine, thank you."
        },
        {
            "instruction": "What is your name?",
            "input": "",
            "output": "My name is AI."
        },
        {
            "instruction": "Tell me a joke.",
            "input": "",
            "output": "Why did the chicken cross the road?"
        }
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
        "response": ["World", "There", "You"],
        "input_tokens": [[1, 2, 3], [4, 5], [6, 7, 8]],
        "decoded_text": ["Hello", "Hi", "Hey"]
    })

@pytest.fixture
def mock_models():
    """创建模拟模型。"""
    class MockModel:
        def get_token_count(self, text):
            return len(text.split())
            
        def encode(self, text):
            return [ord(c) for c in text]
            
        def decode(self, tokens):
            return ''.join(chr(t) for t in tokens)
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

def test_token_processor_init(model_path):
    """测试TokenProcessor初始化。"""
    processor = TokenProcessor(model_path=model_path)
    assert processor.model_path == model_path
    assert isinstance(processor.tokenizer, MockTokenizer)

def test_token_processor_process(model_path):
    """测试TokenProcessor的process方法。"""
    processor = TokenProcessor(model_path=model_path)
    tokens = processor.process("Hello world")
    assert isinstance(tokens, list)
    assert len(tokens) == 11  # "Hello world" 的字符数

def test_token_processor_encode_decode(model_path):
    """测试TokenProcessor的编码和解码。"""
    processor = TokenProcessor(model_path=model_path)
    text = "Hello world"
    encoded = processor.encode(text)
    decoded = processor.decode(encoded)
    assert isinstance(encoded, list)
    assert isinstance(decoded, str)
    assert decoded == text

def test_token_processor_batch_process(model_path):
    """测试TokenProcessor的批量处理。"""
    processor = TokenProcessor(model_path=model_path)
    texts = ["Hello", "World"]
    results = processor.batch_process(texts)
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, list) for r in results)

def test_token_processing_init(model_path):
    """测试 TokenProcessing 初始化。"""
    processor = TokenProcessing(model_path)
    assert isinstance(processor.processor, TokenProcessor)
    assert processor.processor.model_path == model_path

def test_token_processing_process_tokens(model_path):
    """测试 TokenProcessing 的 process_tokens 方法。"""
    processor = TokenProcessing(model_path)
    texts = ["Hello", "World"]
    df = processor.process_tokens(texts)
    
    assert isinstance(df, pd.DataFrame)
    assert "input_tokens" in df.columns
    assert "decoded_text" in df.columns
    assert len(df) == len(texts)

def test_token_processing_compute_distribution(model_path, mock_dataframe):
    """测试 TokenProcessing 的 compute_distribution 方法。"""
    processor = TokenProcessing(model_path)
    distribution = processor.compute_distribution(mock_dataframe)
    
    assert isinstance(distribution, dict)
    assert all(isinstance(k, str) for k in distribution.keys())
    assert all(isinstance(v, float) for v in distribution.values())
    assert sum(distribution.values()) == pytest.approx(1.0)

def test_token_processing_compute_distribution_with_save(model_path, mock_dataframe, tmp_path):
    """测试 TokenProcessing 的 compute_distribution 方法（带保存图表）。"""
    processor = TokenProcessing(model_path)
    save_path = str(tmp_path / "distribution.png")
    distribution = processor.compute_distribution(mock_dataframe, save_path)
    
    assert isinstance(distribution, dict)
    assert Path(save_path).exists()

def test_token_processing_get_token_data_dataframe(model_path, mock_dataframe):
    """测试 TokenProcessing 的 get_token_data 方法（DataFrame 格式）。"""
    processor = TokenProcessing(model_path)
    token_data = processor.get_token_data(mock_dataframe)
    
    assert isinstance(token_data, pd.DataFrame)
    assert "input_tokens" in token_data.columns
    assert "decoded_text" in token_data.columns
    assert len(token_data) == len(mock_dataframe)

def test_token_processing_get_token_data_dict(model_path, mock_dataframe):
    """测试 TokenProcessing 的 get_token_data 方法（字典格式）。"""
    processor = TokenProcessing(model_path)
    token_data = processor.get_token_data(mock_dataframe, format='dict')
    
    assert isinstance(token_data, dict)
    assert "input_tokens" in token_data
    assert "decoded_text" in token_data
    assert isinstance(token_data["input_tokens"], list)
    assert isinstance(token_data["decoded_text"], list)
    assert len(token_data["input_tokens"]) == len(mock_dataframe)

def test_token_processing_get_token_data_empty(model_path):
    """测试 TokenProcessing 的 get_token_data 方法（空数据）。"""
    processor = TokenProcessing(model_path)
    empty_df = pd.DataFrame()
    
    # 测试 DataFrame 格式
    token_data_df = processor.get_token_data(empty_df)
    assert isinstance(token_data_df, pd.DataFrame)
    assert token_data_df.empty
    
    # 测试字典格式
    token_data_dict = processor.get_token_data(empty_df, format='dict')
    assert isinstance(token_data_dict, dict)
    assert not token_data_dict 