import pytest
import os
import json
from pathlib import Path
from src.optimization_engine.tradeoff_analyzer import TradeoffAnalyzer

@pytest.fixture
def test_config():
    """测试配置"""
    hardware_config = {
        "device_type": "rtx4050",  # 使用 RTX4050 作为测试设备
        "device_id": "cuda:0",     # 支持字符串格式的设备ID
        "max_batch_size": 32,
        "max_memory": "16GB"
    }
    
    model_config = {
        "models": {
            "TinyLlama-1.1B-Chat-v1.0": {
                "model_path": "models/TinyLlama-1.1B-Chat-v1.0",
                "model_type": "tinyllama",
                "max_length": 2048,
                "device": "cuda:0"
            }
        }
    }
    
    return {
        "hardware_config": hardware_config,
        "model_config": model_config,
        "token_distribution_path": "data/processed/token_distribution.pkl",
        "output_dir": "tests/output/tradeoff"
    }

@pytest.fixture
def mock_distribution(tmp_path):
    """创建模拟的分布数据文件"""
    distribution = {
        "distribution": {
            "input_distribution": {
                128: 0.2,
                256: 0.3,
                512: 0.3,
                1024: 0.15,
                2048: 0.05
            },
            "output_distribution": {
                256: 0.3,
                512: 0.4,
                1024: 0.2,
                2048: 0.1
            }
        }
    }
    
    dist_path = tmp_path / "token_distribution.pkl"
    with open(dist_path, 'wb') as f:
        import pickle
        pickle.dump(distribution, f)
    
    return str(dist_path)

def test_tradeoff_analyzer_with_valid_distribution(test_config, mock_distribution):
    """测试使用有效分布数据的 TradeoffAnalyzer"""
    # 使用模拟的分布数据路径
    test_config["token_distribution_path"] = mock_distribution
    
    # 确保输出目录存在
    output_dir = Path(test_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建分析器实例
    analyzer = TradeoffAnalyzer(
        token_distribution_path=test_config["token_distribution_path"],
        hardware_config=test_config["hardware_config"],
        model_config=test_config["model_config"],
        output_dir=test_config["output_dir"]
    )
    
    # 运行分析
    results = analyzer.analyze(model_name="TinyLlama-1.1B-Chat-v1.0")
    
    # 验证结果
    assert results is not None
    assert len(results) > 0
    
    # 检查每个 lambda 值的结果
    for lambda_param, metrics in results.items():
        assert "energy" in metrics
        assert "runtime" in metrics
        assert metrics["energy"] > 0
        assert metrics["runtime"] > 0
        
    # 检查输出文件
    assert (output_dir / "tradeoff_results.json").exists()
    assert (output_dir / "tradeoff_curve.png").exists()
    
    # 验证 JSON 结果文件的内容
    with open(output_dir / "tradeoff_results.json", 'r') as f:
        saved_results = json.load(f)
    assert saved_results == results

def test_tradeoff_analyzer_with_missing_distribution(test_config):
    """测试缺失分布数据时的行为"""
    # 使用不存在的分布数据路径
    test_config["token_distribution_path"] = "nonexistent/path.pkl"
    
    # 确保输出目录存在
    output_dir = Path(test_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建分析器实例
    analyzer = TradeoffAnalyzer(
        token_distribution_path=test_config["token_distribution_path"],
        hardware_config=test_config["hardware_config"],
        model_config=test_config["model_config"],
        output_dir=test_config["output_dir"]
    )
    
    # 运行分析
    results = analyzer.analyze(model_name="TinyLlama-1.1B-Chat-v1.0")
    
    # 验证使用了默认分布
    assert results is not None
    assert len(results) > 0
    
    # 检查输出文件
    assert (output_dir / "tradeoff_results.json").exists()
    assert (output_dir / "tradeoff_curve.png").exists()

def test_tradeoff_analyzer_with_invalid_model(test_config, mock_distribution):
    """测试使用无效模型名称时的行为"""
    # 使用模拟的分布数据路径
    test_config["token_distribution_path"] = mock_distribution
    
    # 创建分析器实例
    analyzer = TradeoffAnalyzer(
        token_distribution_path=test_config["token_distribution_path"],
        hardware_config=test_config["hardware_config"],
        model_config=test_config["model_config"],
        output_dir=test_config["output_dir"]
    )
    
    # 验证使用无效模型名称时抛出异常
    with pytest.raises(ValueError, match="Model invalid_model not found"):
        analyzer.analyze(model_name="invalid_model") 