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
    """创建模拟的模型目录。"""
    model_dir = tmp_path / "models" / "TinyLlama-1.1B-Chat-v1.0"
    model_dir.mkdir(parents=True)
    
    # 创建模拟的tokenizer文件
    config = {
        "model_type": "tinyllama",
        "vocab_size": 32000
    }
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)
        
    return model_dir

@pytest.fixture
def token_processor(mock_model_dir):
    """创建TokenProcessor实例。"""
    processor = TokenProcessor(model_path=str(mock_model_dir))
    yield processor
    processor.cleanup()

@pytest.fixture
def data_loader():
    """创建DataLoader实例。"""
    return DataLoader()

@pytest.fixture
def sample_data():
    """创建示例数据。"""
    return [
        {"text": "这是第一个测试文本", "label": 0},
        {"text": "这是第二个测试文本", "label": 1},
        {"text": "这是第三个测试文本", "metadata": {"source": "test"}}
    ]

@pytest.fixture
def large_sample_data():
    """创建大量示例数据。"""
    return [
        {"text": f"测试文本{i}", "label": i % 2}
        for i in range(100)
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
    
    # 测试无效路径
    with pytest.raises(Exception):
        TokenProcessor(model_path="/invalid/path")

def test_token_processor_test_mode():
    """测试TokenProcessor的测试模式。"""
    # 测试正常测试模式
    processor = TokenProcessor()
    assert processor.tokenizer is not None
    
    # 测试禁用测试模式
    os.environ["TEST_MODE"] = "false"
    with pytest.raises(Exception):
        TokenProcessor(model_path="/invalid/path")
    os.environ["TEST_MODE"] = "true"

def test_token_processing(token_processor):
    """测试token处理功能。"""
    # 测试基本文本
    text = "这是一个测试句子"
    tokens = token_processor.process(text)
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    
    # 测试空文本
    empty_text = ""
    empty_tokens = token_processor.process(empty_text)
    assert isinstance(empty_tokens, list)
    assert len(empty_tokens) == 0
    
    # 测试特殊字符
    special_text = "测试!@#$%^&*()_+"
    special_tokens = token_processor.process(special_text)
    assert isinstance(special_tokens, list)
    assert len(special_tokens) > 0
    
    # 测试长文本
    long_text = "测试" * 1000
    long_tokens = token_processor.process(long_text)
    assert isinstance(long_tokens, list)
    assert len(long_tokens) > 0

def test_token_processor_batch_processing(token_processor, sample_data):
    """测试批量处理功能。"""
    texts = [item["text"] for item in sample_data]
    
    # 测试批量分词
    batch_tokens = token_processor.batch_process(texts)
    assert isinstance(batch_tokens, list)
    assert len(batch_tokens) == len(texts)
    assert all(isinstance(tokens, list) for tokens in batch_tokens)
    
    # 测试空列表
    empty_batch = token_processor.batch_process([])
    assert isinstance(empty_batch, list)
    assert len(empty_batch) == 0
    
    # 测试包含无效数据的批量处理
    invalid_texts = ["正常文本", None, "", "另一个正常文本"]
    with pytest.raises(Exception):
        token_processor.batch_process(invalid_texts)

def test_token_processor_error_handling(token_processor):
    """测试错误处理。"""
    # 测试None输入
    with pytest.raises(Exception):
        token_processor.process(None)
    
    # 测试无效类型
    with pytest.raises(Exception):
        token_processor.process(123)
    
    # 测试无效的token ID
    with pytest.raises(Exception):
        token_processor.decode([-1, 999999])

def test_data_loader_functionality(data_loader, tmp_path, sample_data):
    """测试DataLoader功能。"""
    # 测试正常JSON文件
    json_file = tmp_path / "test.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False)
    
    loaded_data = data_loader.load(json_file)
    assert isinstance(loaded_data, list)
    assert len(loaded_data) == len(sample_data)
    
    # 测试大文件
    large_file = tmp_path / "large.json"
    with open(large_file, "w", encoding="utf-8") as f:
        json.dump([{"id": i} for i in range(1000)], f)
    
    large_data = data_loader.load(large_file)
    assert isinstance(large_data, list)
    assert len(large_data) == 1000
    
    # 测试格式错误的JSON
    invalid_file = tmp_path / "invalid.json"
    with open(invalid_file, "w") as f:
        f.write("invalid json content")
    
    with pytest.raises(json.JSONDecodeError):
        data_loader.load(invalid_file)
    
    # 测试空文件
    empty_file = tmp_path / "empty.json"
    with open(empty_file, "w") as f:
        f.write("")
    
    with pytest.raises(json.JSONDecodeError):
        data_loader.load(empty_file)

def test_data_loader_error_handling(data_loader):
    """测试DataLoader错误处理。"""
    # 测试不存在的文件
    with pytest.raises(FileNotFoundError):
        data_loader.load("nonexistent.json")
    
    # 测试无效的文件路径
    with pytest.raises(Exception):
        data_loader.load(None)
    
    # 测试无效的文件类型
    with pytest.raises(Exception):
        data_loader.load(123)

def test_integration(token_processor, data_loader, tmp_path, sample_data):
    """集成测试。"""
    # 准备测试数据
    data_file = tmp_path / "integration_test.json"
    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False)
    
    # 加载数据
    loaded_data = data_loader.load(data_file)
    assert len(loaded_data) == len(sample_data)
    
    # 处理每个数据项
    for item in loaded_data:
        # 文本处理
        tokens = token_processor.process(item["text"])
        assert isinstance(tokens, list)
        
        # 编码和解码
        encoded = token_processor.encode(item["text"])
        decoded = token_processor.decode(encoded)
        assert isinstance(decoded, str)
        
        # 元数据处理（如果存在）
        if "metadata" in item:
            assert isinstance(item["metadata"], dict)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
