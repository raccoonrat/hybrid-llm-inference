# hybrid-llm-inference/tests/unit/test_benchmarking.py
import pytest
import pandas as pd
from pathlib import Path
from benchmarking.system_benchmarking import SystemBenchmarking
from benchmarking.report_generator import ReportGenerator

@pytest.fixture
def mock_dataset(tmp_path):
    """Create a mock Alpaca dataset."""
    data = pd.DataFrame([
        {"prompt": "Write a story", "response": "Once upon a time"},
        {"prompt": "Explain AI", "response": "AI is..."}
    ])
    dataset_path = tmp_path / "alpaca_prompts.json"
    data.to_json(dataset_path, orient="records")
    return dataset_path

@pytest.fixture
def hardware_config():
    """Mock hardware configuration."""
    return {
        "m1_pro": {"type": "cpu_gpu", "idle_power": 10.0},
        "a100": {"type": "gpu", "device_id": 0}
    }

@pytest.fixture
def model_config():
    """Mock model configuration."""
    return {
        "models": {
            "llama3": {"model_name": "meta-llama/Llama-3-8B", "mode": "local", "max_length": 512}
        }
    }

@pytest.fixture
def scheduler_config():
    """Mock scheduler configuration."""
    return {
        "hardware_map": {
            "m1_pro": "m1_pro",
            "a100": "a100"
        }
    }

def test_system_benchmarking(mock_dataset, hardware_config, model_config, scheduler_config, tmp_path, monkeypatch):
    # Mock dependencies
    def mock_measure(task, input_tokens, output_tokens):
        task()
        return {"energy": 10.0, "runtime": 2.0, "throughput": 15.0, "energy_per_token": 0.5}
    monkeypatch.setattr("hardware_profiling.base_profiler.HardwareProfiler.measure", mock_measure)
    def mock_infer(prompt): return "Mock response"
    monkeypatch.setattr("model_zoo.base_model.BaseModel.infer", mock_infer)
    def mock_get_token_count(text): return 10
    monkeypatch.setattr("model_zoo.base_model.BaseModel.get_token_count", mock_get_token_count)
    
    benchmarker = SystemBenchmarking(mock_dataset, hardware_config, model_config, scheduler_config, output_dir=tmp_path)
    thresholds = {"T_in": 32, "T_out": 32}
    results = benchmarker.run_benchmarks(thresholds, model_name="llama3", sample_size=2)
    
    assert "hybrid" in results
    assert "a100" in results
    assert "m1_pro" in results
    assert all("metrics" in res and "summary" in res for res in results.values())
    assert results["hybrid"]["summary"]["total_tasks"] == 2
    assert results["hybrid"]["summary"]["avg_energy"] == 10.0

def test_report_generator(tmp_path, monkeypatch):
    # Mock benchmark results
    benchmark_results = {
        "hybrid": {
            "metrics": [
                {"energy": 10.0, "runtime": 2.0, "throughput": 15.0, "energy_per_token": 0.5},
                {"energy": 12.0, "runtime": 2.5, "throughput": 12.0, "energy_per_token": 0.6}
            ],
            "summary": {"avg_energy": 11.0, "avg_runtime": 2.25}
        },
        "a100": {
            "metrics": [
                {"energy": 15.0, "runtime": 1.5, "throughput": 20.0, "energy_per_token": 0.75},
                {"energy": 14.0, "runtime": 1.8, "throughput": 18.0, "energy_per_token": 0.7}
            ],
            "summary": {"avg_energy": 14.5, "avg_runtime": 1.65}
        }
    }
    tradeoff_results = {
        0.0: {"energy": 12.0, "runtime": 2.5},
        0.5: {"energy": 11.0, "runtime": 2.0},
        1.0: {"energy": 10.0, "runtime": 1.8}
    }
    
    generator = ReportGenerator(output_dir=tmp_path)
    generator.generate_report(benchmark_results, tradeoff_results)
    
    assert (tmp_path / "benchmark_summary.json").exists()
    assert (tmp_path / "energy_per_token.png").exists()
    assert (tmp_path / "runtime.png").exists()
    assert (tmp_path / "tradeoff_curve.png").exists()
