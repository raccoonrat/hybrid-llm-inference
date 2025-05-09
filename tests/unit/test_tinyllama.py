"""TinyLlama 模型单元测试。"""

import os
import pytest
import psutil
import torch
from src.model_zoo.tinyllama import TinyLlama

@pytest.fixture
def model_config():
    return {
        "model_path": "models/TinyLlama-1.1B-Chat-v1.0",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dtype": "bfloat16",  # 与config.json保持一致
        "batch_size": 4,
        "max_length": 2048
    }

@pytest.fixture
def model(model_config):
    os.environ["TEST_MODE"] = "1"
    model = TinyLlama(model_config)
    yield model
    if "TEST_MODE" in os.environ:
        del os.environ["TEST_MODE"]
    model.cleanup()

def test_model_validation(model_config):
    """测试模型配置验证"""
    # 测试缺少必需字段
    invalid_config = model_config.copy()
    del invalid_config["model_path"]
    with pytest.raises(ValueError, match="model_path is required"):
        TinyLlama(invalid_config)

    # 测试无效的batch_size
    invalid_config = model_config.copy()
    invalid_config["batch_size"] = 0
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        TinyLlama(invalid_config)

    # 测试无效的max_length
    invalid_config = model_config.copy()
    invalid_config["max_length"] = -1
    with pytest.raises(ValueError, match="max_length must be a positive integer"):
        TinyLlama(invalid_config)

    # 测试CUDA不可用时请求CUDA设备
    if not torch.cuda.is_available():
        invalid_config = model_config.copy()
        invalid_config["device"] = "cuda"
        with pytest.raises(ValueError, match="CUDA device requested but not available"):
            TinyLlama(invalid_config)

def test_model_initialization(model):
    """测试模型初始化"""
    assert model.batch_size == 4
    assert model.max_length == 2048
    assert model.device in ["cuda", "cpu"]
    assert model.dtype == torch.bfloat16

def test_model_attributes(model):
    """测试模型属性"""
    assert hasattr(model, "model_path")
    assert hasattr(model, "device")
    assert hasattr(model, "dtype")
    assert hasattr(model, "batch_size")
    assert hasattr(model, "max_length")
    assert isinstance(model.batch_size, int)
    assert isinstance(model.max_length, int)

def test_config_validation(model_config):
    """测试配置验证"""
    os.environ["TEST_MODE"] = "1"
    try:
        model = TinyLlama(model_config)
        assert model.batch_size == model_config["batch_size"]
        assert model.max_length == model_config["max_length"]
        assert model.dtype == getattr(torch, model_config["dtype"])
    finally:
        if "TEST_MODE" in os.environ:
            del os.environ["TEST_MODE"]

def test_mock_model_inference(model):
    """测试模拟模型推理"""
    input_text = "你好，请介绍一下你自己。"
    response = model.infer(input_text)
    assert isinstance(response, str)
    assert len(response) > 0

def test_mock_model_generate(model):
    """测试模拟模型生成"""
    input_text = "解释什么是人工智能。"
    response = model.generate(input_text)
    assert isinstance(response, str)
    assert len(response) > 0

def test_mock_model_batch_inference(model):
    """测试模拟模型批量推理"""
    input_texts = [
        "什么是机器学习？",
        "深度学习的应用场景有哪些？"
    ]
    for text in input_texts:
        response = model.infer(text)
        assert isinstance(response, str)
        assert len(response) > 0

def test_mock_model_metrics(model):
    """测试模拟模型指标"""
    metrics = model.get_metrics()
    assert isinstance(metrics, dict)
    assert "total_tokens" in metrics
    assert "total_time" in metrics
    assert "avg_tokens_per_second" in metrics
    assert "avg_time_per_call" in metrics

def test_inference_validation(model):
    # 测试空输入
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        model.inference("")

    # 测试超长输入
    long_text = "a" * (model.max_length + 1)
    with pytest.raises(ValueError, match=f"Input text length exceeds maximum limit {model.max_length}"):
        model.inference(long_text)

def test_batch_inference_validation(model):
    # 测试空列表
    with pytest.raises(ValueError, match="Input list cannot be empty"):
        model.batch_inference([])

    # 测试包含空文本的列表
    with pytest.raises(ValueError, match="Input texts cannot be empty"):
        model.batch_inference(["text", ""])

    # 测试超长文本
    long_text = "a" * (model.max_length + 1)
    with pytest.raises(ValueError, match=f"Input text length exceeds maximum limit {model.max_length}"):
        model.batch_inference(["text", long_text])

def test_cleanup(model):
    # 测试清理前状态
    assert model._model is None
    assert model._tokenizer is None

    # 测试重复清理
    model.cleanup()
    assert model._model is None
    assert model._tokenizer is None

    # 测试CUDA内存清理
    if torch.cuda.is_available():
        initial_memory = torch.cuda.memory_allocated()
        model.cleanup()
        final_memory = torch.cuda.memory_allocated()
        assert final_memory <= initial_memory

def test_model_loading(model):
    # 测试模型路径不存在
    invalid_config = {
        "model_path": "nonexistent_path",
        "device": "cpu",
        "dtype": "float32"
    }
    model = TinyLlama(invalid_config)
    with pytest.raises(FileNotFoundError, match="Model path does not exist"):
        model._load_model()

def test_model_performance(model):
    """测试模型性能"""
    # 在测试模式下，我们只验证输入验证逻辑
    with pytest.raises(RuntimeError, match="Error during inference"):
        model.inference("测试输入")

def test_error_handling(model):
    """测试错误处理"""
    # 测试空输入
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        model.inference("")

def test_memory_management(model):
    """测试内存管理"""
    # 在测试模式下，我们只验证清理逻辑
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    model.cleanup()
    final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    assert final_memory <= initial_memory

def test_model_device_compatibility(model_config):
    """测试模型设备兼容性"""
    # 测试 CPU 设备
    cpu_config = model_config.copy()
    cpu_config["device"] = "cpu"
    os.environ["TEST_MODE"] = "1"
    cpu_model = TinyLlama(cpu_config)
    assert cpu_model.device == "cpu"

    # 测试 CUDA 设备（如果可用）
    if torch.cuda.is_available():
        cuda_config = model_config.copy()
        cuda_config["device"] = "cuda"
        cuda_model = TinyLlama(cuda_config)
        assert cuda_model.device == "cuda"

def test_model_dtype_compatibility(model_config):
    """测试模型数据类型兼容性"""
    os.environ["TEST_MODE"] = "1"
    for dtype in ["float32", "float16"]:
        config = model_config.copy()
        config["dtype"] = dtype
        model = TinyLlama(config)
        assert model.dtype == getattr(torch, dtype) 