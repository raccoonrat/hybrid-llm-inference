# hybrid-llm-inference/tests/unit/test_optimization_engine.py
import pytest
import os
from optimization_engine.cost_function import CostFunction
from optimization_engine.threshold_optimizer import ThresholdOptimizer
from optimization_engine.tradeoff_analyzer import TradeoffAnalyzer

@pytest.fixture
def mock_measure():
    """模拟测量函数"""
    def measure(task, input_tokens, output_tokens, device_id):
        task()
        return {
            "energy": 10.0,
            "runtime": 1.0,
            "throughput": (input_tokens + output_tokens) / 1.0,
            "energy_per_token": 10.0 / (input_tokens + output_tokens)
        }
    return measure

def test_cost_function(mock_measure):
    """测试成本函数"""
    # 设置测试模式环境变量
    os.environ['TEST_MODE'] = '1'
    
    cost_fn = CostFunction(
        model_name="tinyllama",
        model_path="models/TinyLlama-1.1B-Chat-v1.0",
        mode="local",
        measure_fn=mock_measure
    )
    
    cost = cost_fn(32, 32)
    assert cost >= 0

def test_threshold_optimizer(mock_measure):
    """测试阈值优化器"""
    # 设置测试模式环境变量
    os.environ['TEST_MODE'] = '1'
    
    optimizer = ThresholdOptimizer(
        model_name="tinyllama",
        model_path="models/TinyLlama-1.1B-Chat-v1.0",
        mode="local",
        measure_fn=mock_measure
    )
    
    thresholds = optimizer.optimize()
    assert "T_in" in thresholds
    assert "T_out" in thresholds
    assert thresholds["T_in"] > 0
    assert thresholds["T_out"] > 0

def test_tradeoff_analyzer(mock_measure):
    """测试权衡分析器"""
    # 设置测试模式环境变量
    os.environ['TEST_MODE'] = '1'
    
    analyzer = TradeoffAnalyzer(
        model_name="tinyllama",
        model_path="models/TinyLlama-1.1B-Chat-v1.0",
        mode="local",
        measure_fn=mock_measure
    )
    
    tradeoff = analyzer.analyze()
    assert "energy" in tradeoff
    assert "runtime" in tradeoff
    assert tradeoff["energy"] >= 0
    assert tradeoff["runtime"] >= 0

