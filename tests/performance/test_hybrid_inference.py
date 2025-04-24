"""混合LLM推理测试模块。"""

import os
import pytest
from pathlib import Path
import sys
from typing import Dict, Any, List

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.hybrid_inference import HybridInference
from src.hardware_profiling import get_profiler, BaseProfiler
from src.model_zoo.base_model import BaseModel

# 设置测试模式
os.environ['TEST_MODE'] = 'true'

class MockModel(BaseModel):
    """用于测试的模拟模型类。"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化模拟模型。"""
        super().__init__(config)
        self.response_text = "这是一个模拟的响应。"
        self.token_multiplier = 1.5  # 用于模拟token计数
        
    def _do_inference(self, text: str) -> str:
        """执行模拟推理。"""
        return self.response_text
        
    def infer(self, text: str) -> str:
        """执行推理。"""
        return self._do_inference(text)
        
    def get_token_count(self, text: str) -> int:
        """获取文本的token数量。"""
        return int(len(text) * self.token_multiplier)

class MockProfiler:
    """模拟性能分析器类"""
    def __init__(self, device_id: int = 0, skip_nvml: bool = True):
        self.device_id = device_id
        self.skip_nvml = skip_nvml
        
    def measure_power(self) -> float:
        return 100.0
        
    def get_memory_info(self) -> Dict[str, int]:
        return {"used": 1000, "total": 8000}
        
    def get_temperature(self) -> float:
        return 50.0
        
    def measure(self, task, input_tokens: int, output_tokens: int) -> Dict[str, float]:
        return {
            "energy": 10.0,
            "runtime": 0.5,
            "throughput": 100.0
        }
        
    def cleanup(self):
        pass

@pytest.fixture
def hybrid_inference():
    """创建HybridInference测试实例。"""
    # 设置测试模式
    os.environ["TEST_MODE"] = "true"
    
    # 配置参数
    config = {
        "hardware_config": {
            "device_id": 0,
            "skip_nvml": True
        },
        "model_config": {
            "model_name": "mock_model",
            "model_path": "/path/to/mock",
            "model_type": "mock",
            "mode": "local",
            "batch_size": 1,
            "max_length": 2048
        }
    }
    
    # 创建实例
    inference = HybridInference(config)
    inference.model = MockModel(config["model_config"])  # 使用测试文件中的MockModel
    inference.profiler = MockProfiler()
    
    yield inference
    
    # 清理
    inference.cleanup()
    del os.environ["TEST_MODE"]

def test_initialization(hybrid_inference):
    """测试初始化。"""
    assert hybrid_inference is not None
    assert hybrid_inference.model is not None
    assert hybrid_inference.profiler is not None

def test_inference(hybrid_inference):
    """测试推理功能。"""
    prompt = "这是一个测试提示。"
    response = hybrid_inference.infer(prompt)
    assert response is not None
    assert isinstance(response, str)
    assert response == "这是一个模拟的响应。"

def test_performance_measurement(hybrid_inference):
    """测试性能测量。"""
    prompt = "这是一个测试提示。"
    metrics = hybrid_inference.measure_performance(prompt)
    assert metrics is not None
    assert "energy" in metrics
    assert "runtime" in metrics

def test_error_handling(hybrid_inference):
    """测试错误处理。"""
    with pytest.raises(ValueError):
        hybrid_inference.infer("")

def test_cleanup(hybrid_inference):
    """测试资源清理。"""
    hybrid_inference.cleanup()
    # 验证清理后的状态
    assert True  # 如果清理成功，不会抛出异常

def test_hybrid_inference_initialization(hybrid_inference: HybridInference):
    """测试混合推理初始化。"""
    assert hybrid_inference is not None
    assert isinstance(hybrid_inference.model, MockModel)
    assert isinstance(hybrid_inference.profiler, MockProfiler)

def test_hybrid_inference_inference(hybrid_inference: HybridInference):
    """测试混合推理推理功能。"""
    prompt = "Hello, world!"
    response = hybrid_inference.infer(prompt)
    assert response is not None
    assert isinstance(response, str)
    assert response == "这是一个模拟的响应。"

def test_hybrid_inference_cleanup(hybrid_inference: HybridInference):
    """测试混合推理清理功能。"""
    hybrid_inference.model.cleanup()
    hybrid_inference.profiler.cleanup()

def test_hybrid_inference_performance(hybrid_inference: HybridInference):
    """测试混合推理性能测量。"""
    prompt = "Test performance measurement"
    metrics = hybrid_inference.measure_performance(prompt)
    
    assert isinstance(metrics, dict)
    assert "power" in metrics
    assert "temperature" in metrics
    assert "memory_used" in metrics
    assert "total_tokens" in metrics
    assert metrics["power"] > 0
    assert metrics["temperature"] > 0
    assert metrics["memory_used"] > 0
    assert metrics["total_tokens"] > 0

def test_hybrid_inference_error_handling(hybrid_inference: HybridInference):
    """测试混合推理错误处理。"""
    with pytest.raises(ValueError):
        hybrid_inference.infer("")  # 空输入应该引发错误 