"""数据处理模块测试。"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
import json
from src.data_processing.token_processor import TokenProcessor
from src.data_processing.data_loader import DataLoader

@pytest.fixture(autouse=True)
def setup_test_env():
    """设置测试环境。"""
    os.environ["PYTHONPATH"] = str(Path.cwd())
    os.environ["TEST_MODE"] = "true"
    yield
    if "TEST_MODE" in os.environ:
        del os.environ["TEST_MODE"]

@pytest.fixture
def mock_model_dir(tmp_path):
    """创建模拟的模型目录。
    
    Args:
        tmp_path: pytest提供的临时目录
        
    Returns:
        Path: 模型目录路径
    """
    model_dir = tmp_path / "models" / "TinyLlama-1.1B-Chat-v1.0"
    model_dir.mkdir(parents=True)
    return model_dir

@pytest.fixture
def token_processor(mock_model_dir):
    """创建TokenProcessor实例。
    
    Args:
        mock_model_dir: 模拟的模型目录
        
    Returns:
        TokenProcessor: 处理器实例
    """
    return TokenProcessor(model_path=str(mock_model_dir))

@pytest.fixture
def data_loader():
    """创建DataLoader实例。
    
    Returns:
        DataLoader: 加载器实例
    """
    return DataLoader()

@pytest.fixture
def sample_data():
    """创建示例数据。
    
    Returns:
        list: 示例数据列表
    """
    return [
        {"text": "这是第一个测试文本", "label": 0},
        {"text": "这是第二个测试文本", "label": 1}
    ]

def test_token_processor_initialization(mock_model_dir):
    """测试TokenProcessor初始化。"""
    # 测试默认参数
    processor = TokenProcessor()
    assert processor.model_name == TokenProcessor.DEFAULT_MODEL_NAME
    assert processor.model_path == TokenProcessor.DEFAULT_MODEL_PATH
    
    # 测试自定义参数
    custom_name = "custom-model"
    custom_path = str(mock_model_dir / "custom")
    processor = TokenProcessor(model_name=custom_name, model_path=custom_path)
    assert processor.model_name == custom_name
    assert processor.model_path == custom_path

def test_token_processor_test_mode():
    """测试TokenProcessor的测试模式。"""
    processor = TokenProcessor()
    # 在测试模式下应该使用bert-base-chinese
    assert processor.tokenizer is not None

def test_token_processing(token_processor):
    """测试token处理功能。"""
    text = "这是一个测试句子"
    
    # 测试分词
    tokens = token_processor.process(text)
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    
    # 测试编码
    encoded = token_processor.encode(text)
    assert isinstance(encoded, list)
    assert len(encoded) > 0
    
    # 测试解码
    decoded = token_processor.decode(encoded)
    assert isinstance(decoded, str)
    assert len(decoded) > 0

def test_token_processor_error_handling(setup_test_env):
    """测试TokenProcessor的错误处理。"""
    # 临时禁用测试模式
    if "TEST_MODE" in os.environ:
        del os.environ["TEST_MODE"]
        
    # 测试无效的模型路径
    with pytest.raises(Exception) as exc_info:
        TokenProcessor(model_path="/not/exist/path", model_name="invalid-model")
        
    # 恢复测试模式
    os.environ["TEST_MODE"] = "true"

def test_vocab_size(token_processor):
    """测试词汇表大小获取。"""
    vocab_size = token_processor.get_vocab_size()
    assert isinstance(vocab_size, int)
    assert vocab_size > 0

def test_data_loading(data_loader, tmp_path):
    """测试数据加载功能。"""
    # 创建测试数据文件
    test_data = [{"text": "测试1"}, {"text": "测试2"}]
    data_file = tmp_path / "test_data.json"
    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False)
    
    # 测试加载
    loaded_data = data_loader.load(str(data_file))
    assert isinstance(loaded_data, list)
    assert len(loaded_data) == 2
    assert loaded_data[0]["text"] == "测试1"

def test_token_processing_with_sample_data(token_processor, sample_data):
    """测试使用示例数据进行token处理。"""
    for item in sample_data:
        tokens = token_processor.process(item["text"])
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        
        encoded = token_processor.encode(item["text"])
        assert isinstance(encoded, list)
        assert len(encoded) > 0
        
        decoded = token_processor.decode(encoded)
        assert isinstance(decoded, str)
        assert len(decoded) > 0
