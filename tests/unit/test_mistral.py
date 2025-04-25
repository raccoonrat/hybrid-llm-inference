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
def mock_response():
    """提供模拟的API响应。"""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "choices": [{
            "message": {
                "content": "这是一个测试响应"
            }
        }]
    }
    return response

@pytest.fixture
def mock_models_response():
    """提供模拟的模型列表响应。"""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "data": [
            {"id": "mistralai/Mistral-7B-v0.1"},
            {"id": "mistralai/Mistral-7B-Instruct-v0.1"}
        ]
    }
    return response

@pytest.fixture
def test_model(api_config, mock_response, mock_models_response):
    """提供测试用的APIMistral实例。"""
    with patch("requests.get", return_value=mock_models_response), \
         patch("requests.post", return_value=mock_response):
        return APIMistral(api_config)

def test_initialization(api_config, mock_models_response):
    """测试模型初始化。"""
    with patch("requests.get", return_value=mock_models_response):
        model = APIMistral(api_config)
        assert model.api_key == "test_key"
        assert model.max_retries == 2
        assert model.retry_delay == 0.1
        assert model.initialized

def test_inference(test_model, mock_response):
    """测试推理功能。"""
    with patch("requests.post", return_value=mock_response):
        result = test_model.inference("测试输入")
        assert result == "这是一个测试响应"

def test_batch_inference(test_model, mock_response):
    """测试批量推理功能。"""
    with patch("requests.post", return_value=mock_response):
        inputs = ["测试输入1", "测试输入2"]
        results = [test_model.inference(input) for input in inputs]
        assert len(results) == 2
        assert all(result == "这是一个测试响应" for result in results)

def test_stream_inference(test_model, mock_response):
    """测试流式推理功能。"""
    with patch("requests.post", return_value=mock_response):
        result = test_model.inference("测试输入", stream=True)
        assert result == "这是一个测试响应"

def test_token_count(test_model, mock_response):
    """测试令牌计数功能。"""
    with patch("requests.post", return_value=mock_response):
        count = test_model.get_token_count("测试文本")
        assert isinstance(count, int)
        assert count > 0

def test_cleanup(test_model):
    """测试资源清理。"""
    test_model.cleanup()
    assert not test_model.initialized

def test_invalid_input(test_model):
    """测试无效输入处理。"""
    with pytest.raises(ValueError):
        test_model.inference("")

@patch("requests.post")
def test_api_request_retry(mock_post, api_config):
    """测试API请求重试机制。"""
    # 模拟前两次请求失败，第三次成功
    mock_post.side_effect = [
        MagicMock(status_code=500),
        MagicMock(status_code=500),
        MagicMock(status_code=200, json=lambda: {"choices": [{"message": {"content": "成功"}}]})
    ]
    
    with patch("requests.get", return_value=MagicMock(status_code=200, json=lambda: {"data": []})):
        model = APIMistral(api_config)
        response = model.inference("测试")
        assert response == "成功"
        assert mock_post.call_count == 3

@patch("requests.post")
def test_api_error_handling(mock_post, api_config):
    """测试API错误处理。"""
    mock_post.side_effect = Exception("API错误")
    
    with patch("requests.get", return_value=MagicMock(status_code=200, json=lambda: {"data": []})):
        model = APIMistral(api_config)
        with pytest.raises(RuntimeError):
            model.inference("测试")

def test_config_validation():
    """测试配置验证。"""
    with pytest.raises(ValueError):
        APIMistral({})
        
    with pytest.raises(ValueError):
        APIMistral({"api_key": None})
        
    with pytest.raises(ValueError):
        APIMistral({"api_key": "test", "max_length": -1})
        
    with pytest.raises(ValueError):
        APIMistral({"api_key": "test", "max_retries": -1})
        
    with pytest.raises(ValueError):
        APIMistral({"api_key": "test", "retry_delay": -1}) 