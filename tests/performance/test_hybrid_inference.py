"""
混合推理性能测试模块。
"""
import pytest
import time
from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from src.model_inference.hybrid_inference import HybridInference

# 测试配置
HARDWARE_CONFIG = {
    "device_type": "nvidia",
    "idle_power": 10.0,
    "sample_interval": 0.1
}

MODEL_CONFIG = {
    "name": "tinyllama",
    "size": "1.1B",
    "precision": "int8"
}

@pytest.fixture
def profiler():
    """创建 RTX4050Profiler 实例的 fixture。"""
    return RTX4050Profiler(HARDWARE_CONFIG)

@pytest.fixture
def hybrid_inference(profiler):
    """创建 HybridInference 实例的 fixture。"""
    return HybridInference(profiler, MODEL_CONFIG)

def test_initialization(hybrid_inference):
    """测试初始化。"""
    assert hybrid_inference.profiler is not None
    assert hybrid_inference.model_config == MODEL_CONFIG
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
    assert all(key in result["metrics"] for key in ["runtime", "power", "energy", "throughput", "energy_per_token"])

def test_error_handling(hybrid_inference):
    """测试错误处理。"""
    # 测试无效任务
    invalid_task = {
        "input": 123,  # 无效的输入类型
        "max_tokens": -1  # 无效的 token 数量
    }
    
    with pytest.raises(ValueError):
        hybrid_inference.infer(invalid_task)
    
    # 测试无效配置
    invalid_config = {
        "name": 123,  # 无效的模型名称类型
        "size": "invalid",  # 无效的模型大小
        "precision": "invalid"  # 无效的精度
    }
    
    with pytest.raises(ValueError):
        HybridInference(hybrid_inference.profiler, invalid_config)

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