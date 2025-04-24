"""混合推理性能测试模块。"""

import pytest
import time
from src.hybrid_inference import HybridInference
from src.profiler import get_profiler
from typing import Dict, Any

# 测试配置
HARDWARE_CONFIG = {
    "nvidia_rtx4050": {
        "device_type": "rtx4050",
        "idle_power": 15.0,
        "sample_interval": 200
    }
}

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
def profiler():
    """创建性能分析器实例的 fixture。"""
    return get_profiler(HARDWARE_CONFIG)

@pytest.fixture
def hybrid_inference():
    """创建混合推理实例的 fixture。"""
    return HybridInference(HARDWARE_CONFIG, MODEL_CONFIG)

def test_initialization(profiler):
    """测试初始化。"""
    assert profiler is not None
    assert profiler.hardware_config == HARDWARE_CONFIG
    assert profiler.device_type == "rtx4050"
    assert profiler.idle_power == 15.0
    assert profiler.sample_interval == 200

def test_inference(hybrid_inference):
    """测试推理功能。"""
    # 测试小任务
    small_task = {
        "input_tokens": 500,
        "output_tokens": 50,
        "model": "tinyllama"
    }
    
    result = hybrid_inference.infer(small_task)
    assert result is not None
    assert "output" in result
    assert "metrics" in result
    assert result["metrics"]["runtime"] > 0
    assert result["metrics"]["power"] >= 15.0
    assert result["metrics"]["energy"] > 0
    assert result["metrics"]["throughput"] > 0
    assert result["metrics"]["energy_per_token"] > 0

def test_error_handling(hybrid_inference):
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
            hybrid_inference.infer(task)

def test_performance_measurement(hybrid_inference):
    """测试性能测量。"""
    tasks = [
        {"input_tokens": 100, "output_tokens": 50, "model": "tinyllama"},
        {"input_tokens": 200, "output_tokens": 100, "model": "tinyllama"},
        {"input_tokens": 300, "output_tokens": 150, "model": "tinyllama"}
    ]
    
    for task in tasks:
        result = hybrid_inference.infer(task)
        assert result["metrics"]["runtime"] > 0
        assert result["metrics"]["power"] >= 15.0
        assert result["metrics"]["energy"] > 0
        assert result["metrics"]["throughput"] > 0
        assert result["metrics"]["energy_per_token"] > 0

def test_cleanup(hybrid_inference):
    """测试资源清理。"""
    hybrid_inference.cleanup()
    
    # 测试清理后使用
    with pytest.raises(RuntimeError):
        hybrid_inference.infer({
            "input_tokens": 100,
            "output_tokens": 50,
            "model": "tinyllama"
        })

def test_hybrid_inference_initialization():
    """测试混合推理初始化。"""
    # 测试空配置
    with pytest.raises(ValueError):
        HybridInference({}, MODEL_CONFIG)
    
    with pytest.raises(ValueError):
        HybridInference(HARDWARE_CONFIG, {})
    
    # 测试无效的硬件配置
    invalid_config = HARDWARE_CONFIG.copy()
    invalid_config["nvidia_rtx4050"]["device_type"] = 123
    with pytest.raises(TypeError):
        HybridInference(invalid_config, MODEL_CONFIG)
    
    # 测试无效的模型配置
    invalid_config = MODEL_CONFIG.copy()
    invalid_config["models"]["tinyllama"]["model_name"] = 123
    with pytest.raises(TypeError):
        HybridInference(HARDWARE_CONFIG, invalid_config)

def test_hybrid_inference_inference(hybrid_inference):
    """测试混合推理功能。"""
    # 测试边界条件
    boundary_tasks = [
        {"input_tokens": 1, "output_tokens": 1, "model": "tinyllama"},
        {"input_tokens": 1000, "output_tokens": 100, "model": "tinyllama"},
        {"input_tokens": 10000, "output_tokens": 1000, "model": "tinyllama"}
    ]
    
    for task in boundary_tasks:
        result = hybrid_inference.infer(task)
        assert result is not None
        assert "output" in result
        assert "metrics" in result
        assert result["metrics"]["runtime"] > 0
        assert result["metrics"]["power"] >= 15.0
        assert result["metrics"]["energy"] > 0
        assert result["metrics"]["throughput"] > 0
        assert result["metrics"]["energy_per_token"] > 0

def test_hybrid_inference_cleanup(hybrid_inference):
    """测试混合推理资源清理。"""
    # 测试多次清理
    hybrid_inference.cleanup()
    hybrid_inference.cleanup()  # 应该不会抛出异常
    
    # 测试清理后初始化
    with pytest.raises(RuntimeError):
        hybrid_inference.infer({
            "input_tokens": 100,
            "output_tokens": 50,
            "model": "tinyllama"
        })

def test_hybrid_inference_performance(hybrid_inference):
    """测试混合推理性能。"""
    # 测试连续推理
    tasks = [
        {"input_tokens": 100, "output_tokens": 50, "model": "tinyllama"},
        {"input_tokens": 200, "output_tokens": 100, "model": "tinyllama"},
        {"input_tokens": 300, "output_tokens": 150, "model": "tinyllama"}
    ]
    
    start_time = time.time()
    for task in tasks:
        result = hybrid_inference.infer(task)
        assert result["metrics"]["runtime"] > 0
        assert result["metrics"]["power"] >= 15.0
        assert result["metrics"]["energy"] > 0
        assert result["metrics"]["throughput"] > 0
        assert result["metrics"]["energy_per_token"] > 0
    end_time = time.time()
    
    # 验证总执行时间
    assert end_time - start_time > 0 