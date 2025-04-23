# hybrid-llm-inference/tests/benchmarks/test_benchmarking.py
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from benchmarking.system_benchmarking import SystemBenchmarking
from benchmarking.model_benchmarking import ModelBenchmarking
from benchmarking.report_generator import ReportGenerator
import json

# 设置测试模式
os.environ["TEST_MODE"] = "true"

@pytest.fixture
def mock_dataset(tmp_path):
    """Create a mock Alpaca dataset with varying token counts."""
    data = pd.DataFrame([
        {"prompt": "Write a story", "response": "Once upon a time"},
        {"prompt": "Explain AI", "response": "AI is..." * 10},
        {"prompt": "Long prompt " * 100, "response": "Long response " * 100}
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
            "tinyllama": {
                "model_name": "\\\\wsl.localhost\\Ubuntu-24.04\\home\\mpcblock\\models\\TinyLlama-1.1B-Chat-v1.0",
                "mode": "local",
                "max_length": 512,
                "local_files_only": True
            }
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

@pytest.fixture
def output_dir(tmp_path):
    """Create output directory for benchmark results."""
    return tmp_path / "benchmarks"

def test_system_benchmarking_small_dataset(mock_dataset, hardware_config, model_config, scheduler_config, output_dir, monkeypatch):
    """Test system benchmarking with a small dataset."""
    def mock_measure(task, input_tokens, output_tokens):
        task()
        # Simulate lower energy for M1 Pro on small tasks
        energy = 10.0 if input_tokens <= 32 and output_tokens <= 32 else 15.0
        return {
            "energy": energy,
            "runtime": 2.0,
            "throughput": (input_tokens + output_tokens) / 2.0,
            "energy_per_token": energy / (input_tokens + output_tokens)
        }
    monkeypatch.setattr("hardware_profiling.base_profiler.HardwareProfiler.measure", mock_measure)
    def mock_infer(prompt): return "Mock response"
    monkeypatch.setattr("model_zoo.base_model.BaseModel.infer", mock_infer)
    def mock_get_token_count(text): return min(len(text.split()), 50)  # Simulate token counts
    monkeypatch.setattr("model_zoo.base_model.BaseModel.get_token_count", mock_get_token_count)
    
    benchmarker = SystemBenchmarking(mock_dataset, hardware_config, model_config, scheduler_config, output_dir=output_dir)
    thresholds = {"T_in": 32, "T_out": 32}
    results = benchmarker.run_benchmarks(thresholds, model_name="tinyllama", sample_size=3)
    
    assert "hybrid" in results
    assert "a100" in results
    assert "m1_pro" in results
    assert results["hybrid"]["summary"]["total_tasks"] == 3
    assert results["hybrid"]["summary"]["avg_energy"] <= results["a100"]["summary"]["avg_energy"]
    
    # Validate 7.5% energy reduction
    hybrid_energy = results["hybrid"]["summary"]["avg_energy"]
    a100_energy = results["a100"]["summary"]["avg_energy"]
    energy_reduction = (a100_energy - hybrid_energy) / a100_energy * 100
    assert energy_reduction >= 7.5, f"Energy reduction {energy_reduction:.2f}% is less than 7.5%"

def test_model_benchmarking_small_dataset(mock_dataset, hardware_config, model_config, output_dir, monkeypatch):
    """Test model benchmarking with a small dataset."""
    def mock_measure(task, input_tokens, output_tokens):
        task()
        energy = 10.0 if input_tokens <= 32 and output_tokens <= 32 else 15.0
        return {
            "energy": energy,
            "runtime": 2.0,
            "throughput": (input_tokens + output_tokens) / 2.0,
            "energy_per_token": energy / (input_tokens + output_tokens)
        }
    monkeypatch.setattr("hardware_profiling.base_profiler.HardwareProfiler.measure", mock_measure)
    def mock_infer(prompt): return "Mock response"
    monkeypatch.setattr("model_zoo.base_model.BaseModel.infer", mock_infer)
    def mock_get_token_count(text): return min(len(text.split()), 50)
    monkeypatch.setattr("model_zoo.base_model.BaseModel.get_token_count", mock_get_token_count)
    
    benchmarker = ModelBenchmarking(mock_dataset, hardware_config, model_config, output_dir=output_dir)
    results = benchmarker.run_benchmarks(sample_size=3)
    
    assert "tinyllama" in results
    assert "falcon" in results
    assert "m1_pro" in results["tinyllama"]
    assert "a100" in results["tinyllama"]
    assert results["tinyllama"]["m1_pro"]["summary"]["total_tasks"] == 3
    assert results["tinyllama"]["m1_pro"]["summary"]["avg_energy"] <= results["tinyllama"]["a100"]["summary"]["avg_energy"]
    
    assert (output_dir / "model_benchmarks.json").exists()
    with open(output_dir / "model_benchmarks.json", "r") as f:
        saved_results = json.load(f)
    assert saved_results["tinyllama"]["m1_pro"]["summary"]["avg_energy"] == results["tinyllama"]["m1_pro"]["summary"]["avg_energy"]

def test_model_benchmarking_empty_dataset(tmp_path, hardware_config, model_config, output_dir, monkeypatch):
    """Test model benchmarking with an empty dataset."""
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("[]")
    
    def mock_measure(task, input_tokens, output_tokens): task()
    monkeypatch.setattr("hardware_profiling.base_profiler.HardwareProfiler.measure", mock_measure)
    monkeypatch.setattr("model_zoo.base_model.BaseModel.infer", lambda self, prompt: "Mock response")
    monkeypatch.setattr("model_zoo.base_model.BaseModel.get_token_count", lambda self, text: 10)
    
    benchmarker = ModelBenchmarking(empty_file, hardware_config, model_config, output_dir=output_dir)
    
    with pytest.raises(ValueError, match="Dataset is empty"):
        benchmarker.run_benchmarks(sample_size=3)

def test_model_benchmarking_invalid_sample_size(mock_dataset, hardware_config, model_config, output_dir, monkeypatch):
    """Test model benchmarking with an invalid sample size."""
    def mock_measure(task, input_tokens, output_tokens): task()
    monkeypatch.setattr("hardware_profiling.base_profiler.HardwareProfiler.measure", mock_measure)
    monkeypatch.setattr("model_zoo.base_model.BaseModel.infer", lambda self, prompt: "Mock response")
    monkeypatch.setattr("model_zoo.base_model.BaseModel.get_token_count", lambda self, text: 10)
    
    benchmarker = ModelBenchmarking(mock_dataset, hardware_config, model_config, output_dir=output_dir)
    
    with pytest.raises(ValueError, match="Sample size must be positive"):
        benchmarker.run_benchmarks(sample_size=0)

def test_system_benchmarking_empty_dataset(tmp_path, hardware_config, model_config, scheduler_config, output_dir, monkeypatch):
    """Test system benchmarking with an empty dataset."""
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("[]")
    
    def mock_measure(task, input_tokens, output_tokens): task()
    monkeypatch.setattr("hardware_profiling.base_profiler.HardwareProfiler.measure", mock_measure)
    monkeypatch.setattr("model_zoo.base_model.BaseModel.infer", lambda self, prompt: "Mock response")
    monkeypatch.setattr("model_zoo.base_model.BaseModel.get_token_count", lambda self, text: 10)
    
    benchmarker = SystemBenchmarking(empty_file, hardware_config, model_config, scheduler_config, output_dir=output_dir)
    thresholds = {"T_in": 32, "T_out": 32}
    
    with pytest.raises(ValueError, match="Dataset is empty"):
        benchmarker.run_benchmarks(thresholds, model_name="tinyllama", sample_size=3)

def test_system_benchmarking_invalid_thresholds(mock_dataset, hardware_config, model_config, scheduler_config, output_dir, monkeypatch):
    """Test system benchmarking with invalid thresholds."""
    def mock_measure(task, input_tokens, output_tokens):
        task()
        return {"energy": 10.0, "runtime": 2.0, "throughput": 15.0, "energy_per_token": 0.5}
    monkeypatch.setattr("hardware_profiling.base_profiler.HardwareProfiler.measure", mock_measure)
    monkeypatch.setattr("model_zoo.base_model.BaseModel.infer", lambda self, prompt: "Mock response")
    monkeypatch.setattr("model_zoo.base_model.BaseModel.get_token_count", lambda self, text: 10)
    
    benchmarker = SystemBenchmarking(mock_dataset, hardware_config, model_config, scheduler_config, output_dir=output_dir)
    thresholds = {"T_in": -1, "T_out": 32}
    
    with pytest.raises(ValueError, match="Thresholds must be positive"):
        benchmarker.run_benchmarks(thresholds, model_name="tinyllama", sample_size=3)

def test_system_benchmarking_large_sample_size(mock_dataset, hardware_config, model_config, scheduler_config, output_dir, monkeypatch):
    """Test system benchmarking with a large sample size (simulated)."""
    def mock_measure(task, input_tokens, output_tokens):
        task()
        energy = 10.0 if input_tokens <= 32 and output_tokens <= 32 else 15.0
        return {
            "energy": energy,
            "runtime": 2.0,
            "throughput": (input_tokens + output_tokens) / 2.0,
            "energy_per_token": energy / (input_tokens + output_tokens)
        }
    monkeypatch.setattr("hardware_profiling.base_profiler.HardwareProfiler.measure", mock_measure)
    monkeypatch.setattr("model_zoo.base_model.BaseModel.infer", lambda self, prompt: "Mock response")
    monkeypatch.setattr("model_zoo.base_model.BaseModel.get_token_count", lambda self, text: min(len(text.split()), 50))
    
    benchmarker = SystemBenchmarking(mock_dataset, hardware_config, model_config, scheduler_config, output_dir=output_dir)
    thresholds = {"T_in": 32, "T_out": 32}
    results = benchmarker.run_benchmarks(thresholds, model_name="tinyllama", sample_size=100)
    
    assert results["hybrid"]["summary"]["total_tasks"] == 3  # Limited by dataset size
    assert results["hybrid"]["summary"]["avg_energy"] <= results["a100"]["summary"]["avg_energy"]

def test_report_generator_valid_output(mock_dataset, hardware_config, model_config, scheduler_config, output_dir, monkeypatch):
    """Test report generation with valid benchmark results."""
    benchmark_results = {
        "hybrid": {
            "metrics": [
                {"energy": 10.0, "runtime": 2.0, "throughput": 15.0, "energy_per_token": 0.5},
                {"energy": 12.0, "runtime": 2.5, "throughput": 12.0, "energy_per_token": 0.6}
            ],
            "summary": {"avg_energy": 11.0, "avg_runtime": 2.25, "avg_throughput": 13.5, "avg_energy_per_token": 0.55}
        },
        "a100": {
            "metrics": [
                {"energy": 15.0, "runtime": 1.5, "throughput": 20.0, "energy_per_token": 0.75},
                {"energy": 14.0, "runtime": 1.8, "throughput": 18.0, "energy_per_token": 0.7}
            ],
            "summary": {"avg_energy": 14.5, "avg_runtime": 1.65, "avg_throughput": 19.0, "avg_energy_per_token": 0.725}
        },
        "m1_pro": {
            "metrics": [
                {"energy": 8.0, "runtime": 3.0, "throughput": 10.0, "energy_per_token": 0.4},
                {"energy": 9.0, "runtime": 3.2, "throughput": 9.0, "energy_per_token": 0.45}
            ],
            "summary": {"avg_energy": 8.5, "avg_runtime": 3.1, "avg_throughput": 9.5, "avg_energy_per_token": 0.425}
        }
    }
    tradeoff_results = {
        0.0: {"energy": 12.0, "runtime": 2.5},
        0.5: {"energy": 11.0, "runtime": 2.0},
        1.0: {"energy": 10.0, "runtime": 1.8}
    }
    
    generator = ReportGenerator(output_dir=output_dir)
    generator.generate_report(benchmark_results, tradeoff_results)
    
    assert (output_dir / "benchmark_summary.json").exists()
    assert (output_dir / "energy_per_token.png").exists()
    assert (output_dir / "runtime.png").exists()
    assert (output_dir / "tradeoff_curve.png").exists()
    
    with open(output_dir / "benchmark_summary.json", "r") as f:
        summary = json.load(f)
    assert summary["hybrid"]["avg_energy"] == 11.0
    assert summary["a100"]["avg_energy"] == 14.5

def test_report_generator_empty_results(output_dir):
    """Test report generation with empty benchmark results."""
    generator = ReportGenerator(output_dir=output_dir)
    benchmark_results = {}
    
    with pytest.raises(ValueError, match="Benchmark results are empty"):
        generator.generate_report(benchmark_results)

def test_report_generator_invalid_tradeoff_results(output_dir):
    """Test report generation with invalid tradeoff results."""
    benchmark_results = {
        "hybrid": {
            "metrics": [{"energy": 10.0, "runtime": 2.0, "throughput": 15.0, "energy_per_token": 0.5}],
            "summary": {"avg_energy": 10.0, "avg_runtime": 2.0}
        }
    }
    tradeoff_results = {0.5: {"energy": -1.0, "runtime": 2.0}}
    
    generator = ReportGenerator(output_dir=output_dir)
    with pytest.raises(ValueError, match="Invalid tradeoff results"):
        generator.generate_report(benchmark_results, tradeoff_results)
