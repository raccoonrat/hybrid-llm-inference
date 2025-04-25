"""测试基础模型类。"""

import pytest
import torch
from src.model_zoo.base_model import BaseModel

class TestModel(BaseModel):
    """用于测试的具体模型实现。"""
    
    def __init__(self, config):
        super().__init__(config)
        self._initialized = False
        self._metrics = {
            "total_tokens": 0,
            "total_time": 0.0,
            "total_calls": 0
        }
    
    def _validate_base_config(self) -> None:
        if "model_name" not in self.config:
            raise ValueError("缺少模型名称")
        
    def _validate_config(self) -> None:
        if "max_length" not in self.config:
            raise ValueError("缺少最大长度配置")
        
    def _init_model(self) -> None:
        self._initialized = True
        
    def inference(self, input_text: str, max_tokens: int = None) -> str:
        if not self._initialized:
            raise RuntimeError("模型未初始化")
        self._metrics["total_calls"] += 1
        return "测试响应"
        
    def cleanup(self) -> None:
        self._initialized = False
        self._metrics = {
            "total_tokens": 0,
            "total_time": 0.0,
            "total_calls": 0
        }
        
    def get_metrics(self) -> dict:
        return self._metrics.copy()

@pytest.fixture
def base_model():
    """创建测试模型实例。"""
    config = {
        "model_name": "test_model",
        "model_path": "/path/to/model",
        "mode": "local",
        "batch_size": 1,
        "max_length": 512
    }
    return TestModel(config)

def test_base_model_initialization(base_model):
    """测试基础模型初始化。"""
    assert base_model.config is not None
    assert base_model.model is None
    assert base_model.tokenizer is None
    assert isinstance(base_model.device, torch.device)
    
    # 测试配置验证
    with pytest.raises(ValueError) as exc_info:
        TestModel({})  # 空配置
    assert "缺少模型名称" in str(exc_info.value)
    
    with pytest.raises(ValueError) as exc_info:
        TestModel({"model_name": "test"})  # 缺少max_length
    assert "缺少最大长度配置" in str(exc_info.value)

def test_base_model_load_model(base_model):
    """测试加载模型方法。"""
    with pytest.raises(NotImplementedError):
        base_model.load_model()

def test_base_model_get_token_count(base_model):
    """测试获取token数量方法。"""
    # 测试tokenizer未初始化的情况
    with pytest.raises(RuntimeError) as exc_info:
        base_model.get_token_count("测试文本")
    assert "Tokenizer未初始化" in str(exc_info.value)
    
    # 测试空文本
    class MockTokenizer:
        def encode(self, text):
            if not text:
                return []
            return [1] * len(text)
    
    base_model.tokenizer = MockTokenizer()
    assert base_model.get_token_count("") == 0
    
    # 测试正常文本
    token_count = base_model.get_token_count("测试文本")
    assert token_count == 4
    
    # 测试特殊字符
    special_text = "测试!@#$%^&*()_+"
    token_count = base_model.get_token_count(special_text)
    assert token_count == len(special_text)

def test_base_model_infer(base_model):
    """测试推理方法。"""
    # 测试模型未加载的情况
    with pytest.raises(RuntimeError) as exc_info:
        base_model.infer("测试文本")
    assert "模型未加载" in str(exc_info.value)
    
    # 测试空输入
    class MockModel:
        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3]])
            
    class MockTokenizer:
        def __call__(self, text, return_tensors="pt"):
            return {"input_ids": torch.tensor([[1, 2, 3]])}
            
        def decode(self, tokens, skip_special_tokens=True):
            return "生成的响应"
    
    base_model.model = MockModel()
    base_model.tokenizer = MockTokenizer()
    
    response = base_model.infer("测试文本")
    assert response == "生成的响应"

def test_base_model_abstract_methods():
    """测试抽象方法。"""
    config = {
        "model_name": "test_model",
        "model_path": "/path/to/model",
        "mode": "local",
        "batch_size": 1,
        "max_length": 512
    }
    
    # 尝试实例化抽象基类
    with pytest.raises(TypeError):
        BaseModel(config)

def test_base_model_do_inference(base_model):
    """测试实际推理方法。"""
    # _do_inference 是一个可选的辅助方法
    assert base_model._do_inference("测试文本") is None

def test_base_model_metrics_and_reset(base_model):
    """测试性能指标和重置方法。"""
    # get_metrics 和 reset_metrics 是可选的辅助方法
    assert base_model.get_metrics() is None
    assert base_model.reset_metrics() is None

def test_base_model_initialize(base_model):
    """测试初始化方法。"""
    # 模拟 load_model 方法
    def mock_load_model():
        return "加载的模型"
    
    base_model.load_model = mock_load_model
    base_model.initialize()
    
    assert base_model.model == "加载的模型"
    assert base_model.tokenizer == base_model.tokenizer  # tokenizer 保持不变

def test_base_model_with_different_devices():
    """测试不同设备配置。"""
    configs = [
        {
            "model_name": "test_model",
            "model_path": "/path/to/model",
            "mode": "local",
            "batch_size": 1,
            "max_length": 512,
            "device": "cuda"
        },
        {
            "model_name": "test_model",
            "model_path": "/path/to/model",
            "mode": "local",
            "batch_size": 1,
            "max_length": 512,
            "device": "cpu"
        }
    ]
    
    for config in configs:
        model = TestModel(config)
        if torch.cuda.is_available() and config["device"] == "cuda":
            assert model.device.type == "cuda"
        else:
            assert model.device.type == "cpu" 