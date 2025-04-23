# hybrid-llm-inference/tests/unit/test_model_zoo.py
import pytest
import os
from model_zoo import get_model

@pytest.fixture
def model_config():
    return {
        "model_name": "meta-llama/Llama-3-8B",
        "mode": "local",
        "max_length": 512,
        "api_key": "dummy_key"
    }

def test_local_tinyllama():
    """测试本地 TinyLlama 模型"""
    # 设置测试模式环境变量
    os.environ['TEST_MODE'] = '1'
    
    model = get_model(
        model_name="tinyllama",
        model_path="models/TinyLlama-1.1B-Chat-v1.0",
        mode="local"
    )
    
    # 测试推理
    response = model.infer("Hello")
    assert isinstance(response, str)
    assert len(response) > 0
    
    # 测试 token 计数
    token_count = model.get_token_count("Hello")
    assert token_count > 0

def test_api_tinyllama():
    """测试 API 模式 TinyLlama 模型"""
    # 设置测试模式环境变量
    os.environ['TEST_MODE'] = '1'
    
    model = get_model(
        model_name="tinyllama",
        model_path="models/TinyLlama-1.1B-Chat-v1.0",
        mode="api"
    )
    
    # 测试推理
    response = model.infer("Hello")
    assert isinstance(response, str)
    assert len(response) > 0
    
    # 测试 token 计数
    token_count = model.get_token_count("Hello")
    assert token_count > 0

def test_invalid_model():
    """测试无效模型"""
    with pytest.raises(ValueError, match="Unsupported model"):
        get_model("invalid_model", "local", {})
