"""测试模拟模型。"""

import pytest
from src.model_zoo.mock_model import MockModel

@pytest.fixture
def mock_model():
    """创建模拟模型实例。"""
    config = {
        "model_name": "mock_model",
        "model_path": "/path/to/model",
        "mode": "local",
        "batch_size": 1,
        "max_length": 512
    }
    return MockModel(config)

def test_mock_model_initialization(mock_model):
    """测试模拟模型初始化。"""
    assert mock_model.response_text == "这是一个模拟的响应。"
    assert mock_model.token_multiplier == 1.5
    assert mock_model.config is not None

def test_mock_model_inference(mock_model):
    """测试模拟模型推理。"""
    # 测试正常输入
    input_text = "测试输入"
    response = mock_model.infer(input_text)
    assert response == mock_model.response_text
    
    # 测试空输入
    empty_response = mock_model.infer("")
    assert empty_response == ""

def test_mock_model_token_count(mock_model):
    """测试模拟模型token计数。"""
    # 测试正常输入
    input_text = "测试输入"
    token_count = mock_model.get_token_count(input_text)
    assert token_count == int(len(input_text) * mock_model.token_multiplier)
    
    # 测试空输入
    empty_token_count = mock_model.get_token_count("")
    assert empty_token_count == 0
    
    # 测试长文本
    long_text = "这是一个很长的测试文本" * 10
    long_token_count = mock_model.get_token_count(long_text)
    assert long_token_count == int(len(long_text) * mock_model.token_multiplier)

def test_mock_model_do_inference(mock_model):
    """测试模拟模型的内部推理方法。"""
    input_text = "测试输入"
    response = mock_model._do_inference(input_text)
    assert response == mock_model.response_text

def test_mock_model_with_different_configs():
    """测试不同配置下的模拟模型。"""
    configs = [
        {
            "model_name": "mock_model",
            "model_path": "/path/to/model",
            "mode": "local",
            "batch_size": 1,
            "max_length": 512
        },
        {
            "model_name": "mock_model",
            "model_path": "/path/to/model",
            "mode": "local",
            "batch_size": 4,
            "max_length": 1024,
            "device": "cuda",
            "dtype": "float16"
        },
        {
            "model_name": "mock_model",
            "model_path": "/path/to/model",
            "mode": "local",
            "batch_size": 8,
            "max_length": 2048,
            "device": "cpu",
            "dtype": "float32"
        }
    ]
    
    for config in configs:
        model = MockModel(config)
        assert model.config == config
        assert model.response_text == "这是一个模拟的响应。"
        assert model.token_multiplier == 1.5 