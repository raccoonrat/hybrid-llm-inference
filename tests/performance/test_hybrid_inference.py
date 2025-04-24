"""混合LLM推理性能测试。"""

import os
import pytest
from pathlib import Path
import sys
from typing import Dict, Any

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.hybrid_inference import HybridInference
from src.hardware_profiling import get_profiler
from src.model_zoo.base_model import BaseModel

# 设置测试模式
os.environ['TEST_MODE'] = '1'

@pytest.fixture
def hardware_config() -> Dict[str, Any]:
    """硬件配置"""
    return {
        'device_id': 0,
        'skip_nvml': True,
        'idle_power': 15.0,
        'max_power': 115.0,
        'sample_interval': 0.1
    }

@pytest.fixture
def model_config() -> Dict[str, Any]:
    """模型配置"""
    return {
        'model_name': 'test_model',
        'model_path': 'test_path',
        'model_type': 'test_type',
        'mode': 'local',
        'batch_size': 1,
        'max_length': 128
    }

@pytest.fixture
def hybrid_inference(hardware_config: Dict[str, Any], model_config: Dict[str, Any]) -> HybridInference:
    """混合推理实例"""
    return HybridInference(hardware_config, model_config)

class MockModel(BaseModel):
    """模拟模型类"""
    def __init__(self):
        super().__init__(
            model_name="mock_model",
            config={
                "model_path": "mock_path",
                "mode": "local",
                "batch_size": 1,
                "max_length": 128
            }
        )
        self.device = "cuda"
    
    def infer(self, prompt: str) -> str:
        """模拟推理"""
        return "Mock response"
    
    def get_token_count(self, text: str) -> int:
        """模拟token计数"""
        return len(text.split())

def test_hybrid_inference_initialization(hybrid_inference: HybridInference):
    """测试混合推理初始化"""
    assert hybrid_inference is not None
    assert hybrid_inference.profiler is not None
    assert hybrid_inference.model is not None
    assert hybrid_inference.profiler.config['device_id'] == 0
    assert hybrid_inference.profiler.config['skip_nvml'] is True

def test_hybrid_inference_inference(hybrid_inference: HybridInference):
    """测试混合推理功能"""
    # 测试正常输入
    input_text = "测试输入"
    output = hybrid_inference.inference(input_text)
    assert output is not None
    assert isinstance(output, str)
    
    # 测试空输入
    empty_output = hybrid_inference.inference("")
    assert empty_output is not None
    assert isinstance(empty_output, str)
    
    # 测试长输入
    long_input = "测试" * 1000
    long_output = hybrid_inference.inference(long_input)
    assert long_output is not None
    assert isinstance(long_output, str)

def test_hybrid_inference_cleanup(hybrid_inference: HybridInference):
    """测试混合推理清理"""
    # 测试正常清理
    hybrid_inference.cleanup()
    assert hybrid_inference.profiler is None
    assert hybrid_inference.model is None
    
    # 测试重复清理
    hybrid_inference.cleanup()
    assert hybrid_inference.profiler is None
    assert hybrid_inference.model is None

def test_hybrid_inference_performance(hybrid_inference: HybridInference):
    """测试混合推理性能指标"""
    # 测试正常推理
    result = hybrid_inference.inference("Test prompt")
    assert isinstance(result, str)
    
    # 测试性能测量
    metrics = hybrid_inference.measure_performance("Test prompt")
    assert isinstance(metrics, dict)
    assert "energy" in metrics
    assert "runtime" in metrics
    assert "throughput" in metrics
    assert "energy_per_token" in metrics
    
    # 验证性能指标的有效性
    assert metrics["energy"] >= 0
    assert metrics["runtime"] >= 0
    assert metrics["throughput"] >= 0
    assert metrics["energy_per_token"] >= 0

def test_hybrid_inference_error_handling(hybrid_inference: HybridInference):
    """测试混合推理错误处理"""
    # 测试无效输入
    with pytest.raises(ValueError, match="输入文本不能为None"):
        hybrid_inference.inference(None)
    
    # 测试无效配置
    with pytest.raises(ValueError):
        invalid_config = {"invalid": "config"}
        HybridInference(invalid_config, invalid_config)
        
    # 测试性能测量的无效输入
    with pytest.raises(ValueError, match="输入文本不能为None"):
        hybrid_inference.measure_performance(None) 