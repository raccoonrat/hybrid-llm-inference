"""
混合推理性能测试模块。
"""
import pytest
import time
import os
from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from src.model_inference.hybrid_inference import HybridInference

# 测试配置
HARDWARE_CONFIG = {
    "device_type": "nvidia",
    "idle_power": 10.0,
    "sample_interval": 100  # 修改为 100 毫秒
}

MODEL_CONFIG = {
    "model_name": "tinyllama",
    "model_path": "D:/Dev/cursor/github.com/hybrid-llm-inference/models/TinyLlama-1.1B-Chat-v1.0",
    "device": "cuda",
    "mode": "local",
    "batch_size": 1,
    "dtype": "float32",
    "scheduler_config": {
        "hardware_config": HARDWARE_CONFIG
    }
}

@pytest.fixture
def profiler():
    """创建 RTX4050Profiler 实例的 fixture。"""
    return RTX4050Profiler(HARDWARE_CONFIG)

@pytest.fixture
def hybrid_inference():
    """创建 HybridInference 实例的 fixture。"""
    # 设置测试模式
    os.environ['TEST_MODE'] = '1'
    return HybridInference(MODEL_CONFIG, test_mode=True)

def test_initialization(hybrid_inference):
    """测试初始化。"""
    assert hybrid_inference.profiler is not None
    assert hybrid_inference.model_name == MODEL_CONFIG["model_name"]
    assert hybrid_inference.model_path == MODEL_CONFIG["model_path"]
    assert hybrid_inference.is_initialized

def test_inference_functionality(hybrid_inference):
    """测试推理功能。"""
    task = {
        "input": "Hello, how are you?",
        "max_tokens": 10
    }
    
    result = hybrid_inference.infer(task)
    assert result is not None
    assert "output" in result
    assert "metrics" in result
    assert isinstance(result["output"], str)
    assert isinstance(result["metrics"], dict)
    assert "latency" in result["metrics"]
    assert "energy" in result["metrics"]

def test_error_handling(hybrid_inference):
    """测试错误处理。"""
    # 测试无效任务
    with pytest.raises(ValueError):
        hybrid_inference.infer(None)
    
    with pytest.raises(TypeError):
        hybrid_inference.infer("invalid task")
    
    with pytest.raises(ValueError):
        hybrid_inference.infer({"input": "missing max_tokens"})
    
    with pytest.raises(ValueError):
        hybrid_inference.infer({"max_tokens": 10})  # missing input
    
    # 测试无效配置
    with pytest.raises(ValueError):
        HybridInference({})  # empty config
    
    with pytest.raises(ValueError):
        HybridInference({"model_path": "path/to/model"})  # missing model_name
    
    with pytest.raises(ValueError):
        HybridInference({"model_name": "model"})  # missing model_path

def test_cleanup(hybrid_inference):
    """测试资源清理。"""
    hybrid_inference.cleanup()
    assert not hybrid_inference.is_initialized
    
    # 测试清理后使用
    task = {
        "input": "Test after cleanup",
        "max_tokens": 5
    }
    
    with pytest.raises(RuntimeError):
        hybrid_inference.infer(task)

def test_generate_text(hybrid_inference):
    """测试文本生成。"""
    # 测试正常生成
    result = hybrid_inference.generate("Test input", 10)
    assert isinstance(result, str)
    assert len(result) > 0
    
    # 测试空输入
    result = hybrid_inference.generate("", 10)
    assert isinstance(result, str)
    
    # 测试最大长度为0
    result = hybrid_inference.generate("Test input", 0)
    assert isinstance(result, str)

def test_performance_measurement(hybrid_inference):
    """测试性能测量。"""
    task = {
        "input": "This is a test input for performance measurement.",
        "max_tokens": 20
    }
    
    start_time = time.time()
    result = hybrid_inference.infer(task)
    end_time = time.time()
    
    assert result is not None
    assert "metrics" in result
    metrics = result["metrics"]
    
    # 验证性能指标
    assert metrics["runtime"] > 0
    assert metrics["power"] >= HARDWARE_CONFIG["idle_power"]
    assert metrics["energy"] > 0
    assert metrics["throughput"] > 0
    assert metrics["energy_per_token"] > 0
    
    # 验证实际运行时间
    actual_runtime = end_time - start_time
    assert abs(metrics["runtime"] - actual_runtime) < 0.1  # 允许 100ms 的误差

def test_boundary_conditions(hybrid_inference):
    """测试边界条件。"""
    # 测试空输入
    empty_task = {
        "input": "",
        "max_tokens": 1
    }
    
    result = hybrid_inference.infer(empty_task)
    assert result is not None
    assert "output" in result
    assert len(result["output"]) > 0
    
    # 测试最大 token 限制
    large_task = {
        "input": "Test with large token count",
        "max_tokens": 1000
    }
    
    result = hybrid_inference.infer(large_task)
    assert result is not None
    assert "metrics" in result
    assert result["metrics"]["throughput"] > 0 