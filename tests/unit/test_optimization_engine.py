# hybrid-llm-inference/tests/unit/test_optimization_engine.py
import pytest
import pickle
from pathlib import Path
from optimization_engine.cost_function import CostFunction
from optimization_engine.threshold_optimizer import ThresholdOptimizer
from optimization_engine.tradeoff_analyzer import TradeoffAnalyzer

@pytest.fixture
def mock_distribution(tmp_path):
    dist = {
        'distribution': {
            'input_distribution': {10: 100, 20: 50},
            'output_distribution': {30: 80, 40: 70}
        }
    }
    dist_path = tmp_path / "token_distribution.pkl"
    with open(dist_path, 'wb') as f:
        pickle.dump(dist, f)
    return dist_path

@pytest.fixture
def hardware_config():
    return {
        "m1_pro": {"type": "cpu_gpu", "idle_power": 10.0},
        "a100": {"type": "gpu", "device_id": 0}
    }

@pytest.fixture
def model_config():
    return {
        "models": {
            "llama3": {"model_name": "meta-llama/Llama-3-8B", "mode": "local", "max_length": 512}
        }
    }

def test_cost_function(hardware_config, model_config, monkeypatch):
    def mock_measure(task, input_tokens, output_tokens):
        task()
        return {"energy": 10.0, "runtime": 2.0, "throughput": 15.0, "energy_per_token": 0.5}
    monkeypatch.setattr("hardware_profiling.base_profiler.HardwareProfiler.measure", mock_measure)
    
    cost_fn = CostFunction(lambda_param=0.5, hardware_config=hardware_config)
    cost = cost_fn.compute(task=lambda: None, input_tokens=10, output_tokens=20, system="m1_pro")
    
    assert cost == 0.5 * 10.0 + 0.5 * 2.0
    assert isinstance(cost, float)

def test_cost_function_invalid_system(hardware_config, model_config):
    cost_fn = CostFunction(lambda_param=0.5, hardware_config=hardware_config)
    with pytest.raises(ValueError, match="System invalid not supported"):
        cost_fn.compute(task=lambda: None, input_tokens=10, output_tokens=20, system="invalid")

def test_threshold_optimizer(mock_distribution, hardware_config, model_config, monkeypatch):
    def mock_measure(task, input_tokens, output_tokens):
        task()
        return {"energy": 10.0 if input_tokens <= 32 else 20.0, "runtime": 2.0, "throughput": 15.0, "energy_per_token": 0.5}
    monkeypatch.setattr("hardware_profiling.base_profiler.HardwareProfiler.measure", mock_measure)
    def mock_infer(prompt): pass
    monkeypatch.setattr("model_zoo.base_model.BaseModel.infer", mock_infer)
    
    optimizer = ThresholdOptimizer(mock_distribution, hardware_config, model_config)
    thresholds = optimizer.optimize(lambda_param=0.5, model_name="llama3")
    
    assert "T_in" in thresholds
    assert "T_out" in thresholds
    assert isinstance(thresholds["T_in"], int)
    assert isinstance(thresholds["T_out"], int)

def test_tradeoff_analyzer(mock_distribution, hardware_config, model_config, tmp_path, monkeypatch):
    def mock_measure(task, input_tokens, output_tokens):
        task()
        return {"energy": 10.0 if input_tokens <= 32 else 20.0, "runtime": 2.0, "throughput": 15.0, "energy_per_token": 0.5}
    monkeypatch.setattr("hardware_profiling.base_profiler.HardwareProfiler.measure", mock_measure)
    def mock_infer(prompt): pass
    monkeypatch.setattr("model_zoo.base_model.BaseModel.infer", mock_infer)
    
    analyzer = TradeoffAnalyzer(mock_distribution, hardware_config, model_config, output_dir=tmp_path)
    results = analyzer.analyze(model_name="llama3")
    
    assert len(results) == 11
    assert all("energy" in res and "runtime" in res for res in results.values())
    assert (tmp_path / "tradeoff_results.json").exists()
    assert (tmp_path / "tradeoff_curve.png").exists()

