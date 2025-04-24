"""Mistral模型测试模块。"""

import os
import pytest
from typing import Generator
from unittest.mock import patch, MagicMock

from src.model_zoo.mistral import APIMistral

@pytest.fixture
def api_config():
    """提供测试用的API配置。"""
    return {
        "api_key": "test_key",
        "model_name": "mistralai/Mistral-7B-v0.1",
        "max_length": 100,
        "max_retries": 2,
        "retry_delay": 0.1
    }

@pytest.fixture
def test_model(api_config):
    """提供测试用的APIMistral实例。"""
    with patch.dict(os.environ, {"TEST_MODE": "true"}):
        return APIMistral(api_config)

def test_initialization(api_config):
    """测试模型初始化。"""
    with patch.dict(os.environ, {"TEST_MODE": "true"}):
        model = APIMistral(api_config)
        assert model.api_key == "test_key"
        assert model.max_retries == 2
        assert model.retry_delay == 0.1

def test_inference(test_model):
    """测试基本推理功能。"""
    input_text = "测试输入"
    output = test_model.inference(input_text)
    assert output == f"测试输出: {input_text}"

def test_batch_inference(test_model):
    """测试批量推理功能。"""
    input_texts = ["测试输入1", "测试输入2"]
    outputs = test_model.batch_inference(input_texts)
    assert len(outputs) == 2
    assert outputs[0] == f"测试输出: {input_texts[0]}"
    assert outputs[1] == f"测试输出: {input_texts[1]}"

def test_stream_inference(test_model):
    """测试流式推理功能。"""
    input_text = "测试输入"
    stream = test_model.stream_inference(input_text)
    assert isinstance(stream, Generator)
    output = next(stream)
    assert output == f"测试输出: {input_text}"

def test_token_count(test_model):
    """测试token计数功能。"""
    text = "这是一个测试句子"
    count = test_model.get_token_count(text)
    assert count == len(text.split())

def test_cleanup(test_model):
    """测试资源清理功能。"""
    test_model.cleanup()
    assert not hasattr(test_model, "tokenizer")

@patch("requests.post")
def test_api_request_retry(mock_post, api_config):
    """测试API请求重试机制。"""
    # 模拟前两次请求失败，第三次成功
    mock_post.side_effect = [
        MagicMock(status_code=500),
        MagicMock(status_code=500),
        MagicMock(status_code=200, json=lambda: {"generated_text": "成功"})
    ]
    
    with patch.dict(os.environ, {"TEST_MODE": "true"}):
        model = APIMistral(api_config)
        response = model._make_api_request({"inputs": "测试"})
        assert response.status_code == 200
        assert mock_post.call_count == 3

def test_invalid_input(test_model):
    """测试无效输入处理。"""
    with pytest.raises(ValueError):
        test_model.inference("")
        
    with pytest.raises(ValueError):
        test_model.batch_inference([])

@patch("requests.post")
def test_api_error_handling(mock_post, api_config):
    """测试API错误处理。"""
    mock_post.side_effect = Exception("API错误")
    
    with patch.dict(os.environ, {"TEST_MODE": "true"}):
        model = APIMistral(api_config)
        with pytest.raises(RuntimeError):
            model._make_api_request({"inputs": "测试"})

def test_config_validation():
    """测试配置验证。"""
    with pytest.raises(ValueError):
        APIMistral({})
        
    with pytest.raises(ValueError):
        APIMistral({"api_key": None}) 