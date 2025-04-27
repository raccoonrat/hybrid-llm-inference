"""BaseModel 类的测试用例。"""

import pytest
import os
from unittest.mock import patch, MagicMock
from src.model_zoo.base_model import BaseModel
from src.toolbox.logger import get_logger

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

@patch("torch.cuda.is_available")
@patch("torch.cuda.device_count")
@patch("torch.cuda.get_device_name")
@patch("torch.cuda.get_device_properties")
def test_base_model_initialization(mock_properties, mock_name, mock_count, mock_available):
    """测试 BaseModel 的基本初始化。"""
    mock_available.return_value = True
    mock_count.return_value = 1
    mock_name.return_value = "NVIDIA GeForce RTX 4050"
    mock_properties.return_value = MagicMock(total_memory=1024 * 1024 * 1024)
    
    model = BaseModel(TEST_CONFIG)
    assert model.config == TEST_CONFIG
    assert not model.initialized
    assert model.model is None
    assert model.tokenizer is None

@patch("torch.cuda.is_available")
@patch("torch.cuda.device_count")
@patch("torch.cuda.get_device_name")
@patch("torch.cuda.get_device_properties")
def test_base_model_invalid_config(mock_properties, mock_name, mock_count, mock_available):
    """测试 BaseModel 初始化时的无效配置。"""
    mock_available.return_value = True
    mock_count.return_value = 1
    mock_name.return_value = "NVIDIA GeForce RTX 4050"
    mock_properties.return_value = MagicMock(total_memory=1024 * 1024 * 1024)
    
    invalid_configs = [
        None,
        "invalid",
        {},
        {"model_name": "test"},  # 缺少必需字段
        {"model_path": "/path"},  # 缺少必需字段
    ]
    
    for config in invalid_configs:
        with pytest.raises(ValueError):
            BaseModel(config)

@patch("torch.cuda.is_available")
@patch("torch.cuda.device_count")
@patch("torch.cuda.get_device_name")
@patch("torch.cuda.get_device_properties")
def test_base_model_initialize(mock_properties, mock_name, mock_count, mock_available, mock_model, mock_tokenizer):
    """测试模型初始化。"""
    mock_available.return_value = True
    mock_count.return_value = 1
    mock_name.return_value = "NVIDIA GeForce RTX 4050"
    mock_properties.return_value = MagicMock(total_memory=1024 * 1024 * 1024)
    
    model = BaseModel(TEST_CONFIG)
    model.model = mock_model
    model.tokenizer = mock_tokenizer
    
    model.initialize()
    assert model.initialized
    mock_model.to.assert_called_once_with(TEST_CONFIG["device"])

@patch("torch.cuda.is_available")
@patch("torch.cuda.device_count")
@patch("torch.cuda.get_device_name")
@patch("torch.cuda.get_device_properties")
def test_base_model_get_token_count(mock_properties, mock_name, mock_count, mock_available, mock_tokenizer):
    """测试获取 token 数量。"""
    mock_available.return_value = True
    mock_count.return_value = 1
    mock_name.return_value = "NVIDIA GeForce RTX 4050"
    mock_properties.return_value = MagicMock(total_memory=1024 * 1024 * 1024)
    
    model = BaseModel(TEST_CONFIG)
    model.tokenizer = mock_tokenizer
    
    text = "This is a test."
    token_count = model.get_token_count(text)
    assert token_count == 3  # mock_tokenizer.encode 返回 [1, 2, 3]
    mock_tokenizer.encode.assert_called_once_with(text)

@patch("torch.cuda.is_available")
@patch("torch.cuda.device_count")
@patch("torch.cuda.get_device_name")
@patch("torch.cuda.get_device_properties")
def test_base_model_infer(mock_properties, mock_name, mock_count, mock_available, mock_model, mock_tokenizer):
    """测试模型推理。"""
    mock_available.return_value = True
    mock_count.return_value = 1
    mock_name.return_value = "NVIDIA GeForce RTX 4050"
    mock_properties.return_value = MagicMock(total_memory=1024 * 1024 * 1024)
    
    model = BaseModel(TEST_CONFIG)
    model.model = mock_model
    model.tokenizer = mock_tokenizer
    model.initialized = True
    
    prompt = "Test prompt"
    max_tokens = 50
    result = model.infer(prompt, max_tokens)
    
    assert result == "This is a test response."
    mock_tokenizer.assert_called_once_with(prompt, return_tensors="pt")
    mock_model.generate.assert_called_once()

@patch("torch.cuda.is_available")
@patch("torch.cuda.device_count")
@patch("torch.cuda.get_device_name")
@patch("torch.cuda.get_device_properties")
def test_base_model_infer_not_initialized(mock_properties, mock_name, mock_count, mock_available):
    """测试未初始化模型时的推理。"""
    mock_available.return_value = True
    mock_count.return_value = 1
    mock_name.return_value = "NVIDIA GeForce RTX 4050"
    mock_properties.return_value = MagicMock(total_memory=1024 * 1024 * 1024)
    
    model = BaseModel(TEST_CONFIG)
    with pytest.raises(RuntimeError, match="模型未初始化"):
        model.infer("test")

@patch("torch.cuda.is_available")
@patch("torch.cuda.device_count")
@patch("torch.cuda.get_device_name")
@patch("torch.cuda.get_device_properties")
def test_base_model_generate(mock_properties, mock_name, mock_count, mock_available, mock_model, mock_tokenizer):
    """测试文本生成。"""
    mock_available.return_value = True
    mock_count.return_value = 1
    mock_name.return_value = "NVIDIA GeForce RTX 4050"
    mock_properties.return_value = MagicMock(total_memory=1024 * 1024 * 1024)
    
    model = BaseModel(TEST_CONFIG)
    model.model = mock_model
    model.tokenizer = mock_tokenizer
    model.initialized = True
    
    prompt = "Test prompt"
    max_length = 50
    result = model.generate(prompt, max_length)
    
    assert result == "This is a test response."
    mock_tokenizer.assert_called_once_with(prompt, return_tensors="pt")
    mock_model.generate.assert_called_once()

@patch("torch.cuda.is_available")
@patch("torch.cuda.device_count")
@patch("torch.cuda.get_device_name")
@patch("torch.cuda.get_device_properties")
def test_base_model_generate_test_mode(mock_properties, mock_name, mock_count, mock_available):
    """测试测试模式下的文本生成。"""
    mock_available.return_value = True
    mock_count.return_value = 1
    mock_name.return_value = "NVIDIA GeForce RTX 4050"
    mock_properties.return_value = MagicMock(total_memory=1024 * 1024 * 1024)
    
    with patch.dict(os.environ, {"TEST_MODE": "1"}):
        model = BaseModel(TEST_CONFIG)
        result = model.generate("test")
        assert result == "This is a mock response."

@patch("torch.cuda.is_available")
@patch("torch.cuda.device_count")
@patch("torch.cuda.get_device_name")
@patch("torch.cuda.get_device_properties")
def test_base_model_get_metrics(mock_properties, mock_name, mock_count, mock_available):
    """测试获取性能指标。"""
    mock_available.return_value = True
    mock_count.return_value = 1
    mock_name.return_value = "NVIDIA GeForce RTX 4050"
    mock_properties.return_value = MagicMock(total_memory=1024 * 1024 * 1024)
    
    model = BaseModel(TEST_CONFIG)
    metrics = model.get_metrics()
    
    assert isinstance(metrics, dict)
    assert "total_tokens" in metrics
    assert "total_time" in metrics
    assert "avg_tokens_per_second" in metrics
    assert "avg_time_per_call" in metrics

@patch("torch.cuda.is_available")
@patch("torch.cuda.device_count")
@patch("torch.cuda.get_device_name")
@patch("torch.cuda.get_device_properties")
def test_base_model_reset_metrics(mock_properties, mock_name, mock_count, mock_available):
    """测试重置性能指标。"""
    mock_available.return_value = True
    mock_count.return_value = 1
    mock_name.return_value = "NVIDIA GeForce RTX 4050"
    mock_properties.return_value = MagicMock(total_memory=1024 * 1024 * 1024)
    
    model = BaseModel(TEST_CONFIG)
    model.total_tokens = 100
    model.total_time = 10.0
    model.call_count = 5
    
    model.reset_metrics()
    
    assert model.total_tokens == 0
    assert model.total_time == 0.0
    assert model.call_count == 0

@patch("torch.cuda.is_available")
@patch("torch.cuda.device_count")
@patch("torch.cuda.get_device_name")
@patch("torch.cuda.get_device_properties")
def test_base_model_cleanup(mock_properties, mock_name, mock_count, mock_available, mock_model, mock_tokenizer):
    """测试资源清理。"""
    mock_available.return_value = True
    mock_count.return_value = 1
    mock_name.return_value = "NVIDIA GeForce RTX 4050"
    mock_properties.return_value = MagicMock(total_memory=1024 * 1024 * 1024)
    
    model = BaseModel(TEST_CONFIG)
    model.model = mock_model
    model.tokenizer = mock_tokenizer
    
    model.cleanup()
    
    assert model.model is None
    assert model.tokenizer is None
    assert not model.initialized

@patch("torch.cuda.is_available")
@patch("torch.cuda.device_count")
@patch("torch.cuda.get_device_name")
@patch("torch.cuda.get_device_properties")
def test_base_model_validate_config(mock_properties, mock_name, mock_count, mock_available):
    """测试配置验证。"""
    mock_available.return_value = True
    mock_count.return_value = 1
    mock_name.return_value = "NVIDIA GeForce RTX 4050"
    mock_properties.return_value = MagicMock(total_memory=1024 * 1024 * 1024)
    
    model = BaseModel(TEST_CONFIG)
    with pytest.raises(NotImplementedError):
        model._validate_config()

@patch("torch.cuda.is_available")
@patch("torch.cuda.device_count")
@patch("torch.cuda.get_device_name")
@patch("torch.cuda.get_device_properties")
def test_base_model_init_model(mock_properties, mock_name, mock_count, mock_available):
    """测试模型初始化。"""
    mock_available.return_value = True
    mock_count.return_value = 1
    mock_name.return_value = "NVIDIA GeForce RTX 4050"
    mock_properties.return_value = MagicMock(total_memory=1024 * 1024 * 1024)
    
    model = BaseModel(TEST_CONFIG)
    with pytest.raises(NotImplementedError):
        model._init_model()

@patch("torch.cuda.is_available")
@patch("torch.cuda.device_count")
@patch("torch.cuda.get_device_name")
@patch("torch.cuda.get_device_properties")
def test_base_model_do_inference(mock_properties, mock_name, mock_count, mock_available):
    """测试实际推理操作。"""
    mock_available.return_value = True
    mock_count.return_value = 1
    mock_name.return_value = "NVIDIA GeForce RTX 4050"
    mock_properties.return_value = MagicMock(total_memory=1024 * 1024 * 1024)
    
    model = BaseModel(TEST_CONFIG)
    with pytest.raises(NotImplementedError):
        model._do_inference("test") 