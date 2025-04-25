# hybrid-llm-inference/tests/unit/test_dataset_manager.py
"""数据集管理模块测试。"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
import json
import pandas as pd
import numpy as np
from src.dataset_manager.alpaca_loader import AlpacaLoader
from src.dataset_manager.data_processor import DataProcessor
from src.dataset_manager.token_distribution import TokenDistribution

@pytest.fixture(autouse=True)
def setup_test_env():
    """设置测试环境。"""
    os.environ["PYTHONPATH"] = str(Path.cwd())
    os.environ["TEST_MODE"] = "true"
    yield
    if "TEST_MODE" in os.environ:
        del os.environ["TEST_MODE"]

@pytest.fixture
def mock_dataset(tmp_path):
    """创建模拟数据集。"""
    data = [
        {
            "instruction": "写一个故事",
            "input": "",
            "output": "从前...",
            "metadata": {"category": "creative"}
        },
        {
            "instruction": "解释AI",
            "input": "什么是人工智能？",
            "output": "AI是...",
            "metadata": {"category": "technical"}
        },
        {
            "instruction": "翻译句子",
            "input": "Hello world",
            "output": "你好世界",
            "metadata": {"category": "translation"}
        }
    ]
    file_path = tmp_path / "test.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return file_path

@pytest.fixture
def sample_alpaca_data():
    """创建示例Alpaca数据。"""
    return pd.DataFrame({
        'instruction': ['写一个故事', '解释AI', '翻译句子'],
        'input': ['', '什么是人工智能？', 'Hello world'],
        'output': ['从前...', 'AI是...', '你好世界'],
        'metadata': [
            {'category': 'creative'},
            {'category': 'technical'},
            {'category': 'translation'}
        ]
    })

@pytest.fixture
def large_alpaca_data():
    """创建大量Alpaca数据。"""
    return pd.DataFrame({
        'instruction': [f'指令{i}' for i in range(100)],
        'input': [f'输入{i}' if i % 2 == 0 else '' for i in range(100)],
        'output': [f'输出{i}' for i in range(100)],
        'metadata': [{'category': 'test', 'id': i} for i in range(100)]
    })

@pytest.fixture
def alpaca_loader(mock_dataset):
    """创建AlpacaLoader实例。"""
    return AlpacaLoader(str(mock_dataset))

@pytest.fixture
def data_processor():
    """创建DataProcessor实例。"""
    return DataProcessor()

@pytest.fixture
def token_distribution(sample_alpaca_data):
    """创建TokenDistribution实例。"""
    return TokenDistribution(sample_alpaca_data, {"llama3": None})

def test_alpaca_loader_initialization(mock_dataset):
    """测试AlpacaLoader初始化。"""
    # 测试正常初始化
    loader = AlpacaLoader(str(mock_dataset))
    assert loader.dataset_path == Path(mock_dataset)
    assert loader.data is None
    
    # 测试无效路径
    with pytest.raises(ValueError):
        AlpacaLoader("")
    
    # 测试None路径
    with pytest.raises(ValueError):
        AlpacaLoader(None)

def test_alpaca_loader_load(alpaca_loader, mock_dataset):
    """测试AlpacaLoader的数据加载功能。"""
    # 测试基本加载
    data = alpaca_loader.load()
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 3
    assert all(col in data.columns for col in ["instruction", "input", "output", "metadata"])
    
    # 测试数据类型
    assert all(isinstance(x, str) for x in data["instruction"])
    assert all(isinstance(x, str) for x in data["input"])
    assert all(isinstance(x, str) for x in data["output"])
    assert all(isinstance(x, dict) for x in data["metadata"])
    
    # 测试重复加载
    data2 = alpaca_loader.load()
    pd.testing.assert_frame_equal(data, data2)

def test_alpaca_loader_data_validation(tmp_path):
    """测试数据验证。"""
    # 测试缺少必要字段
    invalid_data = [{"instruction": "test"}]
    invalid_file = tmp_path / "invalid_fields.json"
    with open(invalid_file, "w") as f:
        json.dump(invalid_data, f)
    
    loader = AlpacaLoader(str(invalid_file))
    with pytest.raises(ValueError, match="Missing required fields"):
        loader.load()
    
    # 测试字段类型错误
    wrong_type_data = [{
        "instruction": 123,
        "input": "",
        "output": "test"
    }]
    wrong_type_file = tmp_path / "wrong_type.json"
    with open(wrong_type_file, "w") as f:
        json.dump(wrong_type_data, f)
    
    loader = AlpacaLoader(str(wrong_type_file))
    with pytest.raises(ValueError, match="Invalid data type"):
        loader.load()

def test_data_processor_initialization():
    """测试DataProcessor初始化。"""
    processor = DataProcessor()
    assert processor.logger is not None
    assert hasattr(processor, "process")

def test_data_processing(data_processor, sample_alpaca_data):
    """测试数据处理功能。"""
    # 测试基本处理
    processed = data_processor.process(sample_alpaca_data)
    assert isinstance(processed, pd.DataFrame)
    assert len(processed) == len(sample_alpaca_data)
    
    # 测试新增字段
    assert "total_length" in processed.columns
    assert "input_tokens" in processed.columns
    assert "output_tokens" in processed.columns
    
    # 测试计算正确性
    for _, row in processed.iterrows():
        assert row["total_length"] == len(row["instruction"]) + len(row["input"]) + len(row["output"])
        assert isinstance(row["input_tokens"], list)
        assert isinstance(row["output_tokens"], list)

def test_data_processor_error_handling(data_processor):
    """测试数据处理错误处理。"""
    # 测试None输入
    with pytest.raises(ValueError):
        data_processor.process(None)
    
    # 测试空DataFrame
    with pytest.raises(ValueError):
        data_processor.process(pd.DataFrame())
    
    # 测试缺少必要列
    invalid_df = pd.DataFrame({"wrong_column": [1, 2, 3]})
    with pytest.raises(ValueError):
        data_processor.process(invalid_df)

def test_token_distribution_initialization(sample_alpaca_data):
    """测试TokenDistribution初始化。"""
    # 测试正常初始化
    distribution = TokenDistribution(sample_alpaca_data, {"llama3": None})
    assert distribution.data.equals(sample_alpaca_data)
    assert "llama3" in distribution.models
    assert distribution.distribution is None
    assert distribution.stats is None
    
    # 测试无效模型配置
    with pytest.raises(ValueError):
        TokenDistribution(sample_alpaca_data, {})
    
    # 测试无效数据
    with pytest.raises(ValueError):
        TokenDistribution(None, {"llama3": None})

def test_token_distribution_analyze(token_distribution):
    """测试token分布分析功能。"""
    # 测试基本分析
    distribution, stats = token_distribution.analyze("llama3")
    assert isinstance(distribution, dict)
    assert "input_distribution" in distribution
    assert "output_distribution" in distribution
    assert isinstance(stats, dict)
    assert "input" in stats
    assert "output" in stats
    
    # 验证统计信息
    for key in ["mean", "std", "min", "max"]:
        assert key in stats["input"]
        assert key in stats["output"]
        assert isinstance(stats["input"][key], (int, float))
        assert isinstance(stats["output"][key], (int, float))
    
    # 测试分布计算
    assert all(isinstance(x, (int, float)) for x in distribution["input_distribution"])
    assert all(isinstance(x, (int, float)) for x in distribution["output_distribution"])
    assert sum(distribution["input_distribution"]) > 0
    assert sum(distribution["output_distribution"]) > 0

def test_token_distribution_visualization(token_distribution, tmp_path):
    """测试token分布可视化功能。"""
    # 设置输出目录
    token_distribution.output_dir = tmp_path
    
    # 生成可视化
    token_distribution.analyze("llama3")
    token_distribution.visualize()
    
    # 检查输出文件
    expected_files = [
        "token_distribution.png",
        "token_stats.json",
        "distribution_data.csv"
    ]
    for file in expected_files:
        assert (tmp_path / file).exists()
    
    # 验证JSON输出
    with open(tmp_path / "token_stats.json") as f:
        stats = json.load(f)
        assert isinstance(stats, dict)
        assert "input" in stats
        assert "output" in stats

def test_token_distribution_large_dataset(large_alpaca_data):
    """测试大数据集的token分布分析。"""
    distribution = TokenDistribution(large_alpaca_data, {"llama3": None})
    dist, stats = distribution.analyze("llama3")
    
    # 验证大数据集处理
    assert len(dist["input_distribution"]) > 0
    assert len(dist["output_distribution"]) > 0
    assert all(isinstance(x, (int, float)) for x in dist["input_distribution"])
    assert all(isinstance(x, (int, float)) for x in dist["output_distribution"])
    
    # 验证统计信息的合理性
    assert stats["input"]["mean"] > 0
    assert stats["output"]["mean"] > 0
    assert stats["input"]["max"] >= stats["input"]["min"]
    assert stats["output"]["max"] >= stats["output"]["min"]

def test_integration(mock_dataset, tmp_path):
    """集成测试。"""
    # 1. 加载数据
    loader = AlpacaLoader(str(mock_dataset))
    data = loader.load()
    assert isinstance(data, pd.DataFrame)
    
    # 2. 处理数据
    processor = DataProcessor()
    processed_data = processor.process(data)
    assert isinstance(processed_data, pd.DataFrame)
    assert len(processed_data) == len(data)
    
    # 3. 分析token分布
    distribution = TokenDistribution(processed_data, {"llama3": None})
    distribution.output_dir = tmp_path
    dist, stats = distribution.analyze("llama3")
    
    # 验证完整流程
    assert isinstance(dist, dict)
    assert isinstance(stats, dict)
    assert all(key in dist for key in ["input_distribution", "output_distribution"])
    assert all(key in stats for key in ["input", "output"])
    
    # 验证可视化输出
    distribution.visualize()
    assert (tmp_path / "token_distribution.png").exists()
    assert (tmp_path / "token_stats.json").exists()

if __name__ == "__main__":
    pytest.main(["-v", __file__])
