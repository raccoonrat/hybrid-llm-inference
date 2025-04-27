"""模型推理模块的测试用例。"""

import pytest
from unittest.mock import patch, MagicMock
from src.model_inference.model_inference import ModelInference

# 测试配置
TEST_CONFIG = {
    "model_name": "test_model",
    "model_path": "/path/to/model",
    "device": "cuda",
    "batch_size": 1,
    "max_length": 100
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

def test_model_inference_init(mock_model, mock_tokenizer):
    """测试 ModelInference 初始化。"""
    inference = ModelInference(TEST_CONFIG, mock_model, mock_tokenizer)
    assert inference.model_name == TEST_CONFIG["model_name"]
    assert inference.model_path == TEST_CONFIG["model_path"]
    assert inference.device == TEST_CONFIG["device"]
    assert inference.batch_size == TEST_CONFIG["batch_size"]
    assert inference.max_length == TEST_CONFIG["max_length"]

def test_model_inference_init_invalid_config():
    """测试 ModelInference 初始化时的无效配置。"""
    invalid_configs = [
        {"model_name": 123},        # 非字符串 model_name
        {"model_path": 456},        # 非字符串 model_path
        {"device": "invalid"},      # 无效的设备类型
        {"batch_size": 0},          # 非正整数 batch_size
        {"max_length": -1}          # 非正整数 max_length
    ]
    
    for config in invalid_configs:
        with pytest.raises(ValueError):
            ModelInference({**TEST_CONFIG, **config}, MagicMock(), MagicMock())

def test_model_inference_infer(mock_model, mock_tokenizer):
    """测试 ModelInference 的推理功能。"""
    inference = ModelInference(TEST_CONFIG, mock_model, mock_tokenizer)
    
    # 测试正常输入
    input_text = "This is a test input."
    result = inference.infer(input_text)
    
    assert isinstance(result, dict)
    assert "generated_text" in result
    assert "input_tokens" in result
    assert "output_tokens" in result
    assert "runtime" in result
    assert result["generated_text"] == "This is a test response."
    assert result["input_tokens"] == 3
    assert result["output_tokens"] == 3
    assert result["runtime"] >= 0

def test_model_inference_infer_invalid_input(mock_model, mock_tokenizer):
    """测试 ModelInference 的无效输入处理。"""
    inference = ModelInference(TEST_CONFIG, mock_model, mock_tokenizer)
    
    # 测试空输入
    with pytest.raises(ValueError, match="输入文本不能为空"):
        inference.infer("")
    
    # 测试非字符串输入
    with pytest.raises(ValueError, match="输入文本必须是字符串"):
        inference.infer(123)

def test_model_inference_infer_error_handling(mock_model, mock_tokenizer):
    """测试 ModelInference 的错误处理。"""
    inference = ModelInference(TEST_CONFIG, mock_model, mock_tokenizer)
    
    # 模拟模型生成时抛出异常
    mock_model.generate.side_effect = Exception("Model generation failed")
    
    with pytest.raises(Exception, match="Model generation failed"):
        inference.infer("This is a test input.")

def test_model_inference_cleanup(mock_model, mock_tokenizer):
    """测试 ModelInference 的资源清理功能。"""
    inference = ModelInference(TEST_CONFIG, mock_model, mock_tokenizer)
    inference.cleanup()
    
    # 验证模型和分词器是否被正确清理
    mock_model.cpu.assert_called_once()
    if TEST_CONFIG["device"] == "cuda":
        mock_model.cuda.assert_called_once() 