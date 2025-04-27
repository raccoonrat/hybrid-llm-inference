"""模型库模块的测试用例。"""

import pytest
import os
from unittest.mock import patch, MagicMock
from src.model_zoo import get_model
from src.model_zoo.base_model import BaseModel
from src.model_zoo.mock_model import MockModel
from src.model_zoo.tinyllama import TinyLlama
from src.model_zoo.mistral import LocalMistral, APIMistral
from src.model_zoo.falcon import FalconModel
from src.model_zoo.llama3 import LocalLlama3, APILlama3

# 测试配置
TEST_CONFIG = {
    "model_name": "test_model",
    "model_path": "/path/to/model",
    "mode": "local",
    "batch_size": 1,
    "max_length": 100,
    "device": "cuda",
    "dtype": "float16"
}

@pytest.fixture
def mock_model():
    """模拟模型对象的 fixture。"""
    model = MagicMock()
    model.generate.return_value = {
        "generated_text": "This is a test response.",
        "input_ids": [1, 2, 3],
        "output_ids": [4, 5, 6]
    }
    return model

@pytest.fixture
def mock_tokenizer():
    """模拟分词器的 fixture。"""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.decode.return_value = "This is a test response."
    return tokenizer

def test_base_model_init():
    """测试 BaseModel 初始化。"""
    model = BaseModel(TEST_CONFIG)
    assert model.config == TEST_CONFIG
    assert not model.initialized

def test_base_model_init_invalid_config():
    """测试 BaseModel 初始化时的无效配置。"""
    invalid_configs = [
        None,           # 空配置
        "invalid",      # 非字典配置
        {},            # 空字典
    ]
    
    for config in invalid_configs:
        with pytest.raises(ValueError):
            BaseModel(config)

def test_mock_model():
    """测试 MockModel。"""
    model = MockModel(TEST_CONFIG)
    
    # 测试推理
    result = model.infer("test input")
    assert result == "这是一个模拟的响应。"
    
    # 测试 token 计数
    token_count = model.get_token_count("test input")
    assert token_count == int(len("test input") * 1.5)

def test_get_model():
    """测试 get_model 函数。"""
    # 测试获取 TinyLlama
    tinyllama = get_model({**TEST_CONFIG, "model_name": "tinyllama"})
    assert isinstance(tinyllama, TinyLlama)
    
    # 测试获取 Mistral
    local_mistral = get_model({**TEST_CONFIG, "model_name": "mistral", "mode": "local"})
    assert isinstance(local_mistral, LocalMistral)
    
    api_mistral = get_model({**TEST_CONFIG, "model_name": "mistral", "mode": "remote"})
    assert isinstance(api_mistral, APIMistral)
    
    # 测试获取 Falcon
    falcon = get_model({**TEST_CONFIG, "model_name": "falcon", "mode": "local"})
    assert isinstance(falcon, FalconModel)
    
    # 测试获取 Llama3
    local_llama3 = get_model({**TEST_CONFIG, "model_name": "llama3", "mode": "local"})
    assert isinstance(local_llama3, LocalLlama3)
    
    api_llama3 = get_model({**TEST_CONFIG, "model_name": "llama3", "mode": "remote"})
    assert isinstance(api_llama3, APILlama3)
    
    # 测试无效模型名称
    with pytest.raises(ValueError, match="不支持的模型"):
        get_model({**TEST_CONFIG, "model_name": "invalid_model"})

def test_get_model_invalid_mode():
    """测试 get_model 函数中的无效模式。"""
    # 测试 Falcon 的无效模式
    with pytest.raises(ValueError, match="Falcon 只支持本地模式"):
        get_model({**TEST_CONFIG, "model_name": "falcon", "mode": "remote"})
    
    # 测试 Mistral 的无效模式
    with pytest.raises(ValueError, match="Mistral 只支持本地和远程模式"):
        get_model({**TEST_CONFIG, "model_name": "mistral", "mode": "invalid"})
    
    # 测试 Llama3 的无效模式
    with pytest.raises(ValueError, match="Llama3 只支持本地和远程模式"):
        get_model({**TEST_CONFIG, "model_name": "llama3", "mode": "invalid"})

def test_get_model_missing_config():
    """测试 get_model 函数中的缺失配置。"""
    required_fields = ["model_name", "model_path", "mode", "batch_size", "max_length"]
    
    for field in required_fields:
        config = TEST_CONFIG.copy()
        del config[field]
        with pytest.raises(ValueError, match=f"缺少必需的配置字段: {field}"):
            get_model(config)

@patch("os.getenv")
def test_get_model_test_mode(mock_getenv):
    """测试 get_model 函数在测试模式下的行为。"""
    mock_getenv.return_value = "true"
    model = get_model(TEST_CONFIG)
    assert isinstance(model, MockModel)

def test_model_cleanup(mock_model, mock_tokenizer):
    """测试模型的资源清理功能。"""
    # 测试 TinyLlama
    tinyllama = TinyLlama(TEST_CONFIG)
    tinyllama.model = mock_model
    tinyllama.tokenizer = mock_tokenizer
    tinyllama.cleanup()
    
    # 测试 Mistral
    mistral = LocalMistral(TEST_CONFIG)
    mistral.model = mock_model
    mistral.tokenizer = mock_tokenizer
    mistral.cleanup()
    
    # 测试 Falcon
    falcon = FalconModel(TEST_CONFIG)
    falcon.model = mock_model
    falcon.tokenizer = mock_tokenizer
    falcon.cleanup()
    
    # 测试 Llama3
    llama3 = LocalLlama3(TEST_CONFIG)
    llama3.model = mock_model
    llama3.tokenizer = mock_tokenizer
    llama3.cleanup() 