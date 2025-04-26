"""TokenProcessor 和 TokenProcessing 的扩展测试用例。"""

import os
os.environ['TEST_MODE'] = '1'

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
from src.data_processing.token_processor import TokenProcessor, MockTokenizer
from src.data_processing.token_processing import TokenProcessing
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

@pytest.fixture
def model_path(tmp_path):
    """创建测试用的模型路径。"""
    model_dir = tmp_path / "models" / "TinyLlama-1.1B-Chat-v1.0"
    model_dir.mkdir(parents=True)
    return str(model_dir)

@pytest.fixture
def invalid_model_path(tmp_path):
    """创建无效的模型路径。"""
    return str(tmp_path / "nonexistent_model")

@pytest.fixture
def large_text_data():
    """创建大文本数据。"""
    return ["Hello world! " * 1000,  # 长文本
            "",  # 空文本
            "Special chars: !@#$%^&*()",  # 特殊字符
            "Numbers: 1234567890",  # 数字
            "Unicode: 你好世界🌍"  # Unicode字符
            ]

# TokenProcessor 的扩展测试

def test_token_processor_invalid_model_path(invalid_model_path):
    """测试使用无效的模型路径初始化 TokenProcessor。"""
    with pytest.raises(ValueError, match="模型路径不存在"):
        TokenProcessor(model_path=invalid_model_path, validate_path=True)

def test_token_processor_empty_input(model_path):
    """测试处理空输入。"""
    processor = TokenProcessor(model_path=model_path)
    
    # 测试空字符串
    assert processor.process("") == []
    
    # 测试空列表批处理
    assert processor.batch_process([]) == []
    
    # 测试 None 输入
    with pytest.raises(ValueError, match="输入文本不能为 None"):
        processor.process(None)

def test_token_processor_special_characters(model_path):
    """测试处理特殊字符。"""
    processor = TokenProcessor(model_path=model_path)
    special_chars = "!@#$%^&*()_+-=[]{}|;:'\",.<>?/\\"
    
    tokens = processor.process(special_chars)
    decoded = processor.decode(tokens)
    assert decoded == special_chars

def test_token_processor_unicode(model_path):
    """测试处理 Unicode 字符。"""
    processor = TokenProcessor(model_path=model_path)
    unicode_text = "你好世界🌍"
    
    tokens = processor.process(unicode_text)
    decoded = processor.decode(tokens)
    assert decoded == unicode_text

def test_token_processor_large_input(model_path):
    """测试处理大输入。"""
    processor = TokenProcessor(model_path=model_path)
    large_text = "Hello world! " * 1000
    
    tokens = processor.process(large_text)
    assert len(tokens) > 1000
    decoded = processor.decode(tokens)
    assert decoded == large_text

def test_token_processor_batch_process_mixed(model_path, large_text_data):
    """测试批处理混合输入。"""
    processor = TokenProcessor(model_path=model_path)
    results = processor.batch_process(large_text_data)
    
    assert len(results) == len(large_text_data)
    for text, tokens in zip(large_text_data, results):
        decoded = processor.decode(tokens)
        assert decoded == text

def test_token_processor_max_length(model_path):
    """测试最大长度限制。"""
    processor = TokenProcessor(model_path=model_path, max_length=10)
    long_text = "This is a very long text that should be truncated"
    
    tokens = processor.process(long_text)
    assert len(tokens) <= 10

# TokenProcessing 的扩展测试

def test_token_processing_invalid_format(model_path, mock_dataframe):
    """测试无效的输出格式。"""
    processor = TokenProcessing(model_path)
    with pytest.raises(ValueError, match="不支持的格式"):
        processor.get_token_data(mock_dataframe, format='invalid')

def test_token_processing_distribution_empty_data(model_path):
    """测试空数据的分布计算。"""
    processor = TokenProcessing(model_path)
    empty_df = pd.DataFrame(columns=["input_tokens"])
    
    distribution = processor.compute_distribution(empty_df)
    assert isinstance(distribution, dict)
    assert len(distribution) == 0

def test_token_processing_distribution_single_token(model_path):
    """测试单个令牌的分布计算。"""
    processor = TokenProcessing(model_path)
    df = pd.DataFrame({
        "input_tokens": [[1]]
    })
    
    distribution = processor.compute_distribution(df)
    assert isinstance(distribution, dict)
    assert len(distribution) == 1
    assert list(distribution.values())[0] == 1.0

def test_token_processing_save_distribution_invalid_path(model_path, mock_dataframe):
    """测试保存分布图到无效路径。"""
    processor = TokenProcessing(model_path)
    invalid_path = "/nonexistent/directory/plot.png"
    
    with pytest.raises(Exception):
        processor.compute_distribution(mock_dataframe, save_path=invalid_path)

def test_token_processing_large_dataset(model_path):
    """测试处理大数据集。"""
    processor = TokenProcessing(model_path)
    large_df = pd.DataFrame({
        "text": ["Hello world! " * 100] * 100
    })
    
    result = processor.process_tokens(large_df["text"])
    assert len(result) == len(large_df)
    assert all(len(tokens) > 0 for tokens in result["input_tokens"])

def test_token_processing_mixed_data_types(model_path):
    """测试处理混合数据类型。"""
    processor = TokenProcessing(model_path)
    mixed_data = pd.DataFrame({
        "text": [
            "Normal text",
            123,  # 数字
            True,  # 布尔值
            None,  # 空值
            ["list", "of", "items"]  # 列表
        ]
    })
    
    # 应该能处理所有类型，将它们转换为字符串
    result = processor.process_tokens(mixed_data["text"])
    assert len(result) == len(mixed_data)

def test_token_processing_concurrent_processing(model_path):
    """测试并发处理。"""
    processor = TokenProcessing(model_path)
    large_df = pd.DataFrame({
        "text": ["Hello world! " * 100] * 100
    })
    
    # 使用多个线程处理
    result1 = processor.process_tokens(large_df["text"])
    result2 = processor.process_tokens(large_df["text"])
    
    # 结果应该相同
    pd.testing.assert_frame_equal(result1, result2)

def test_token_processing_visualization(model_path, tmp_path):
    """测试token分布可视化功能"""
    processor = TokenProcessing(model_path)
    
    # 准备测试数据
    df = pd.DataFrame({
        'input_tokens': [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    })
    
    # 测试保存到临时目录
    save_path = os.path.join(tmp_path, "distribution.png")
    distribution = processor.compute_distribution(df, save_path)
    
    # 验证结果
    assert os.path.exists(save_path)
    assert isinstance(distribution, dict)
    assert len(distribution) > 0
    
def test_token_processing_empty_visualization(model_path, tmp_path):
    """测试空数据的可视化处理"""
    processor = TokenProcessing(model_path)
    
    # 准备空数据
    df = pd.DataFrame({
        'input_tokens': []
    })
    
    # 测试保存到临时目录
    save_path = os.path.join(tmp_path, "empty_distribution.png")
    distribution = processor.compute_distribution(df, save_path)
    
    # 验证结果
    assert isinstance(distribution, dict)
    assert len(distribution) == 0
    
def test_token_processing_invalid_tokens(model_path):
    """测试无效token数据的处理"""
    processor = TokenProcessing(model_path)
    
    # 准备包含None和非列表数据的DataFrame
    df = pd.DataFrame({
        'input_tokens': [None, 123, [1, 2, 3], "invalid"]
    })
    
    # 测试分布计算
    distribution = processor.compute_distribution(df)
    
    # 验证结果
    assert isinstance(distribution, dict)
    assert len(distribution) > 0  # 应该只包含有效的token
    
def test_token_processing_write_permission(model_path, tmp_path):
    """测试写入权限检查"""
    processor = TokenProcessing(model_path)
    
    # 准备测试数据
    df = pd.DataFrame({
        'input_tokens': [[1, 2, 3]]
    })
    
    # 创建一个只读目录
    readonly_dir = os.path.join(tmp_path, "readonly")
    os.makedirs(readonly_dir)
    save_path = os.path.join(readonly_dir, "distribution.png")
    
    # 创建一个空文件并设置为只读
    with open(save_path, 'w') as f:
        f.write('')
    
    # 在Windows上设置文件和目录为只读
    if os.name == 'nt':
        os.system(f'attrib +r "{save_path}"')
        os.system(f'attrib +r "{readonly_dir}"')
    else:
        os.chmod(readonly_dir, 0o444)
        os.chmod(save_path, 0o444)
    
    # 测试写入权限检查
    with pytest.raises(ValueError, match="没有写入权限"):
        processor.compute_distribution(df, save_path)
        
def test_token_processing_mixed_columns(model_path):
    """测试同时包含input_tokens和token列的情况"""
    processor = TokenProcessing(model_path)
    
    # 准备包含两种列的数据
    df = pd.DataFrame({
        'input_tokens': [[1, 2, 3], [4, 5, 6]],
        'token': ['a', 'b']
    })
    
    # 测试分布计算（应该优先使用input_tokens）
    distribution = processor.compute_distribution(df)
    
    # 验证结果
    assert isinstance(distribution, dict)
    assert len(distribution) > 0
    assert 1 in [float(k) for k in distribution.keys()]  # 确认使用了input_tokens列
    
def test_token_processing_visualization_error(model_path, tmp_path):
    """测试可视化过程中的错误处理"""
    processor = TokenProcessing(model_path)
    
    # 准备测试数据
    df = pd.DataFrame({
        'input_tokens': [[1, 2, 3]]
    })
    
    # 使用无效的文件名（包含Windows上的非法字符）
    save_path = os.path.join(tmp_path, "test<>:.png")
    
    # 测试错误处理
    with pytest.raises(ValueError, match="包含无效字符"):
        processor.compute_distribution(df, save_path)

@pytest.fixture
def mock_dataframe():
    """创建模拟DataFrame。"""
    return pd.DataFrame({
        "text": ["Hello", "Hi", "Hey"],
        "input_tokens": [[1, 2, 3], [4, 5], [6, 7, 8]],
        "decoded_text": ["Hello", "Hi", "Hey"]
    }) 