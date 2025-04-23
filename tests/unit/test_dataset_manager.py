# hybrid-llm-inference/tests/unit/test_dataset_manager.py
import pytest
import json
import os
from dataset_manager.alpaca_loader import AlpacaLoader
from dataset_manager.data_processor import DataProcessor

@pytest.fixture
def mock_dataset(tmp_path):
    """创建模拟数据集"""
    data = [
        {"instruction": "test1", "input": "", "output": "response1"},
        {"instruction": "test2", "input": "", "output": "response2"}
    ]
    file_path = tmp_path / "test.json"
    with open(file_path, "w") as f:
        json.dump(data, f)
    return file_path

def test_alpaca_loader_empty_file(tmp_path):
    """测试空文件加载"""
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("[]")
    
    with pytest.raises(ValueError, match="Dataset is empty"):
        AlpacaLoader(empty_file).load()

def test_data_processing(mock_dataset):
    """测试数据处理"""
    # 设置测试模式环境变量
    os.environ['TEST_MODE'] = '1'
    
    processor = DataProcessor(
        model_name="tinyllama",
        model_path="models/TinyLlama-1.1B-Chat-v1.0",
        mode="local"
    )
    
    data = AlpacaLoader(mock_dataset).load()
    processed_data = processor.process(data)
    
    assert len(processed_data) > 0
    assert "input_tokens" in processed_data[0]
    assert "output_tokens" in processed_data[0]

def test_data_processing_no_response(mock_dataset):
    """测试无响应数据处理"""
    # 设置测试模式环境变量
    os.environ['TEST_MODE'] = '1'
    
    processor = DataProcessor(
        model_name="tinyllama",
        model_path="models/TinyLlama-1.1B-Chat-v1.0",
        mode="local"
    )
    
    data = AlpacaLoader(mock_dataset).load()
    # 移除响应
    for item in data:
        item["output"] = ""
    
    processed_data = processor.process(data)
    assert len(processed_data) > 0
    assert all(item["output_tokens"] == 0 for item in processed_data)
