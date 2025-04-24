"""模型推理性能测试模块。"""

import pytest
import time
from src.model_inference import ModelInference
from src.model_zoo import get_model
from typing import Dict, Any

# 测试配置
MODEL_CONFIG = {
    "models": {
        "tinyllama": {
            "model_name": "tinyllama",
            "model_path": "path/to/model",
            "mode": "local",
            "batch_size": 1,
            "max_length": 128
        }
    }
}

@pytest.fixture
def model():
    """创建模型实例的 fixture。"""
    return get_model(MODEL_CONFIG["models"]["tinyllama"])

@pytest.fixture
def model_inference():
    """创建模型推理实例的 fixture。"""
    return ModelInference(MODEL_CONFIG)

def test_initialization(model):
    """测试初始化。"""
    assert model is not None
    assert model.model_name == "tinyllama"
    assert model.model_path == "path/to/model"
    assert model.mode == "local"
    assert model.batch_size == 1
    assert model.max_length == 128

def test_inference_speed(model_inference):
    """测试推理速度。"""
    # 测试小任务
    small_task = {
        "input_tokens": 100,
        "output_tokens": 50,
        "model": "tinyllama"
    }
    
    start_time = time.time()
    result = model_inference.infer(small_task)
    end_time = time.time()
    
    assert result is not None
    assert "output" in result
    assert "metrics" in result
    assert result["metrics"]["runtime"] > 0
    assert result["metrics"]["runtime"] <= end_time - start_time

def test_memory_usage(model_inference):
    """测试内存使用。"""
    # 测试不同大小的任务
    tasks = [
        {"input_tokens": 100, "output_tokens": 50, "model": "tinyllama"},
        {"input_tokens": 200, "output_tokens": 100, "model": "tinyllama"},
        {"input_tokens": 300, "output_tokens": 150, "model": "tinyllama"}
    ]
    
    for task in tasks:
        result = model_inference.infer(task)
        assert result["metrics"]["memory_used"] > 0
        assert result["metrics"]["memory_used"] <= result["metrics"]["memory_total"]

def test_precision_impact(model_inference):
    """测试精度影响。"""
    # 测试不同精度的任务
    tasks = [
        {"input_tokens": 100, "output_tokens": 50, "model": "tinyllama", "precision": "fp32"},
        {"input_tokens": 100, "output_tokens": 50, "model": "tinyllama", "precision": "fp16"},
        {"input_tokens": 100, "output_tokens": 50, "model": "tinyllama", "precision": "int8"}
    ]
    
    for task in tasks:
        result = model_inference.infer(task)
        assert result["metrics"]["runtime"] > 0
        assert result["metrics"]["memory_used"] > 0
        assert result["metrics"]["throughput"] > 0

def test_error_handling(model_inference):
    """测试错误处理。"""
    # 测试无效任务
    invalid_tasks = [
        {"input_tokens": -1, "output_tokens": 50, "model": "tinyllama"},
        {"input_tokens": 100, "output_tokens": -1, "model": "tinyllama"},
        {"input_tokens": 100, "output_tokens": 50, "model": "invalid"},
        {"input_tokens": 100, "output_tokens": 50},  # 缺少模型
        {"input_tokens": 100, "model": "tinyllama"},  # 缺少输出令牌
        {"output_tokens": 50, "model": "tinyllama"},  # 缺少输入令牌
        None,
        {}
    ]
    
    for task in invalid_tasks:
        with pytest.raises((ValueError, TypeError, KeyError)):
            model_inference.infer(task)

def test_cleanup(model_inference):
    """测试资源清理。"""
    model_inference.cleanup()
    
    # 测试清理后使用
    with pytest.raises(RuntimeError):
        model_inference.infer({
            "input_tokens": 100,
            "output_tokens": 50,
            "model": "tinyllama"
        })

def test_model_inference_initialization():
    """测试模型推理初始化。"""
    # 测试空配置
    with pytest.raises(ValueError):
        ModelInference({})
    
    # 测试无效的模型配置
    invalid_config = MODEL_CONFIG.copy()
    invalid_config["models"]["tinyllama"]["model_name"] = 123
    with pytest.raises(TypeError):
        ModelInference(invalid_config)
    
    # 测试无效的模型路径
    invalid_config = MODEL_CONFIG.copy()
    invalid_config["models"]["tinyllama"]["model_path"] = None
    with pytest.raises(ValueError):
        ModelInference(invalid_config)

def test_model_inference_performance(model_inference):
    """测试模型推理性能。"""
    # 测试连续推理
    tasks = [
        {"input_tokens": 100, "output_tokens": 50, "model": "tinyllama"},
        {"input_tokens": 200, "output_tokens": 100, "model": "tinyllama"},
        {"input_tokens": 300, "output_tokens": 150, "model": "tinyllama"}
    ]
    
    start_time = time.time()
    for task in tasks:
        result = model_inference.infer(task)
        assert result["metrics"]["runtime"] > 0
        assert result["metrics"]["memory_used"] > 0
        assert result["metrics"]["throughput"] > 0
    end_time = time.time()
    
    # 验证总执行时间
    assert end_time - start_time > 0 