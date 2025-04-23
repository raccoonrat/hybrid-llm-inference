# hybrid-llm-inference/tests/unit/test_hardware_profiling.py
import pytest
from hardware_profiling import get_profiler
from hardware_profiling.rtx4050_profiling import RTX4050Profiler
from hardware_profiling.a800_profiling import A800Profiler
from toolbox.logger import get_logger

@pytest.fixture
def hardware_config():
    """Mock hardware configuration."""
    return {
        "m1_pro": {"type": "cpu_gpu", "idle_power": 10.0, "sample_interval": 200},
        "a100": {"type": "gpu", "device_id": 0, "idle_power": 40.0, "sample_interval": 200},
        "rtx4050": {"type": "gpu", "device_id": 1, "idle_power": 15.0, "sample_interval": 200},
        "a800": {"type": "gpu", "device_id": 2, "idle_power": 50.0, "sample_interval": 200}
    }

def test_rtx4050_profiler_initialization(hardware_config, monkeypatch):
    """Test RTX4050Profiler initialization."""
    def mock_nvml_init(): pass
    def mock_nvml_device_get_handle(index): return "handle"
    monkeypatch.setattr("pynvml.nvmlInit", mock_nvml_init)
    monkeypatch.setattr("pynvml.nvmlDeviceGetHandleByIndex", mock_nvml_device_get_handle)
    
    profiler = RTX4050Profiler(hardware_config["rtx4050"])
    assert profiler.device_id == 1
    assert profiler.idle_power == 15.0
    assert profiler.sample_interval == 0.2

def test_a800_profiler_initialization(hardware_config, monkeypatch):
    """Test A800Profiler initialization."""
    def mock_nvml_init(): pass
    def mock_nvml_device_get_handle(index): return "handle"
    monkeypatch.setattr("pynvml.nvmlInit", mock_nvml_init)
    monkeypatch.setattr("pynvml.nvmlDeviceGetHandleByIndex", mock_nvml_device_get_handle)
    
    profiler = A800Profiler(hardware_config["a800"])
    assert profiler.device_id == 2
    assert profiler.idle_power == 50.0
    assert profiler.sample_interval == 0.2

def test_get_profiler_rtx4050(hardware_config):
    """Test get_profiler for RTX 4050."""
    profiler = get_profiler("rtx4050", hardware_config["rtx4050"])
    assert isinstance(profiler, RTX4050Profiler)

def test_get_profiler_a800(hardware_config):
    """Test get_profiler for A800."""
    profiler = get_profiler("a800", hardware_config["a800"])
    assert isinstance(profiler, A800Profiler)

# hybrid-llm-inference/tests/benchmarks/test_benchmarking.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from benchmarking.system_benchmarking import SystemBenchmarking
from benchmarking.model_benchmarking import ModelBenchmarking
from benchmarking.report_generator import ReportGenerator
import json

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
        "a100": {"type": "gpu", "device_id": 0, "idle_power": 40.0},
        "rtx4050": {"type": "gpu", "device_id": 1, "idle_power": 15.0},
        "a800": {"type": "gpu", "device_id": 2, "idle_power": 50.0}
    }

@pytest.fixture
def model_config():
    """Mock model configuration."""
    return {
        "models": {
            "llama3": {"model_name": "meta-llama/Llama-3-8B", "mode": "local", "max_length": 512},
            "falcon": {"model_name": "tiiuae/falcon-7b", "mode": "local", "max_length": 512}
        }
    }

@pytest.fixture
def scheduler_config():
    """Mock scheduler configuration."""
    return {
        "hardware_map": {
            "m1_pro": "m1_pro",
            "a100": "a100",
            "rtx4050": "rtx4050",
            "a800": "a800"
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
        # Simulate lower energy for RTX 4050 and M1 Pro on small tasks
        energy = {
            "rtx4050": 8.0,
            "m1_pro": 10.0,
            "a100": 15.0,
            "a800": 20.0
        }.get(task.__self__.__class__.__name__.lower().replace("profiler", ""), 15.0)
        if input_tokens > 32 or output_tokens > 32:
            energy *= 1.5  # Higher energy for large tasks
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
    results = benchmarker.run_benchmarks(thresholds, model_name="llama3", sample_size=3)
    
    assert "hybrid" in results
    assert "a100" in results
    assert "m1_pro" in results
    assert "rtx4050" in results
    assert "a800" in results
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
        energy = {
            "rtx4050": 8.0,
            "m1_pro": 10.0,
            "a100": 15.0,
            "a800": 20.0
        }.get(task.__self__.__class__.__name__.lower().replace("profiler", ""), 15.0)
        if input_tokens > 32 or output_tokens > 32:
            energy *= 1.5
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
    
    assert "llama3" in results
    assert "falcon" in results
    assert "m1_pro" in results["llama3"]
    assert "a100" in results["llama3"]
    assert "rtx4050" in results["llama3"]
    assert "a800" in results["llama3"]
    assert results["llama3"]["rtx4050"]["summary"]["total_tasks"] == 3
    assert results["llama3"]["rtx4050"]["summary"]["avg_energy"] <= results["llama3"]["a100"]["summary"]["avg_energy"]
    
    assert (output_dir / "model_benchmarks.json").exists()
    with open(output_dir / "model_benchmarks.json", "r") as f:
        saved_results = json.load(f)
    assert saved_results["llama3"]["rtx4050"]["summary"]["avg_energy"] == results["llama3"]["rtx4050"]["summary"]["avg_energy"]

# Existing tests (e.g., test_model_benchmarking_empty_dataset, test_report_generator_valid_output) remain unchanged
