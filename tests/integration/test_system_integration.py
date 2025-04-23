# hybrid-llm-inference/tests/integration/test_system_integration.py
import pytest
import pandas as pd
import yaml
from pathlib import Path
from dataset_manager.alpaca_loader import AlpacaLoader
from dataset_manager.data_processing import DataProcessing
from optimization_engine.threshold_optimizer import ThresholdOptimizer
from optimization_engine.tradeoff_analyzer import TradeoffAnalyzer
from scheduling.token_based_scheduler import TokenBasedScheduler
from scheduling.task_allocator import TaskAllocator
from benchmarking.system_benchmarking import SystemBenchmarking
from benchmarking.report_generator import ReportGenerator
from toolbox.config_manager import ConfigManager

@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path

@pytest.fixture
def mock_dataset(tmp_dir):
    data = pd.DataFrame([
        {"prompt": "Write a story", "response": "Once upon a time"},
        {"prompt": "Explain AI", "response": "AI is..."}
    ])
    dataset_path = tmp_dir / "alpaca_prompts.json"
    data.to_json(dataset_path, orient="records")
    return dataset_path

@pytest.fixture
def mock_configs(tmp_dir):
    config_dir = tmp_dir / "configs"
    config_dir.mkdir()
    
    hardware_config = {
        "m1_pro": {"type": "cpu_gpu", "idle_power": 10.0},
        "a100": {"type": "gpu", "device_id": 0}
    }
    model_config = {
        "models": {
            "llama3": {"model_name": "meta-llama/Llama-3-8B", "mode": "local", "max_length": 512}
        }
    }
    scheduler_config = {
        "hardware_map": {
            "m1_pro": "m1_pro",
            "a100": "a100"
        }
    }
    
    with open(config_dir / "hardware_config.yaml", "w") as f:
        yaml.dump(hardware_config, f)
    with open(config_dir / "model_config.yaml", "w") as f:
        yaml.dump(model_config, f)
    with open(config_dir / "scheduler_config.yaml", "w") as f:
        yaml.dump(scheduler_config, f)
    
    return config_dir

@pytest.fixture
def mock_distribution(tmp_dir):
    dist = {
        'distribution': {
            'input_distribution': {10: 100, 20: 50},
            'output_distribution': {30: 80, 40: 70}
        }
    }
    dist_path = tmp_dir / "token_distribution.pkl"
    with open(dist_path, 'wb') as f:
        pickle.dump(dist, f)
    return dist_path

def test_full_system_pipeline(mock_dataset, mock_configs, mock_distribution, tmp_dir, monkeypatch):
    # Mock dependencies
    def mock_measure(task, input_tokens, output_tokens):
        task()
        energy = 10.0 if input_tokens <= 32 else 15.0  # Simulate lower energy for M1 Pro
        return {"energy": energy, "runtime": 2.0, "throughput": 15.0, "energy_per_token": energy / (input_tokens + output_tokens)}
    monkeypatch.setattr("hardware_profiling.base_profiler.HardwareProfiler.measure", mock_measure)
    def mock_infer(prompt): return "Mock response"
    monkeypatch.setattr("model_zoo.base_model.BaseModel.infer", mock_infer)
    def mock_get_token_count(text): return 10
    monkeypatch.setattr("model_zoo.base_model.BaseModel.get_token_count", mock_get_token_count)
    
    # Load configurations
    config_manager = ConfigManager(mock_configs)
    hardware_config = config_manager.load_config("hardware_config.yaml")
    model_config = config_manager.load_config("model_config.yaml")
    scheduler_config = config_manager.load_config("scheduler_config.yaml")
    
    # Step 1: Load and process dataset
    loader = AlpacaLoader(mock_dataset)
    processor = DataProcessing(loader, model_config["models"]["llama3"])
    token_data = processor.get_token_data()
    
    assert len(token_data) == 2
    assert token_data[0]["input_tokens"] == 10
    
    # Step 2: Optimize thresholds
    optimizer = ThresholdOptimizer(mock_distribution, hardware_config, model_config)
    thresholds = optimizer.optimize(lambda_param=0.5, model_name="llama3")
    
    assert "T_in" in thresholds
    assert "T_out" in thresholds
    
    # Step 3: Schedule tasks
    scheduler = TokenBasedScheduler(thresholds, scheduler_config)
    allocations = scheduler.schedule(token_data)
    
    assert len(allocations) == 2
    assert allocations[0]["hardware"] in ["m1_pro", "a100"]
    
    # Step 4: Allocate and execute tasks
    allocator = TaskAllocator(hardware_config, model_config)
    results = allocator.allocate(allocations, model_name="llama3")
    
    assert len(results) == 2
    assert results[0]["metrics"]["energy"] == 10.0  # Below threshold, uses M1 Pro
    
    # Step 5: Run benchmarks
    benchmarker = SystemBenchmarking(mock_dataset, hardware_config, model_config, scheduler_config, output_dir=tmp_dir)
    benchmark_results = benchmarker.run_benchmarks(thresholds, model_name="llama3", sample_size=2)
    
    assert "hybrid" in benchmark_results
    assert "a100" in benchmark_results
    assert benchmark_results["hybrid"]["summary"]["avg_energy"] <= benchmark_results["a100"]["summary"]["avg_energy"]
    
    # Step 6: Generate tradeoff analysis
    analyzer = TradeoffAnalyzer(mock_distribution, hardware_config, model_config, output_dir=tmp_dir)
    tradeoff_results = analyzer.analyze(model_name="llama3")
    
    assert len(tradeoff_results) == 11
    assert (tmp_dir / "tradeoff_results.json").exists()
    
    # Step 7: Generate report
    generator = ReportGenerator(output_dir=tmp_dir)
    generator.generate_report(benchmark_results, tradeoff_results)
    
    assert (tmp_dir / "benchmark_summary.json").exists()
    assert (tmp_dir / "energy_per_token.png").exists()
    assert (tmp_dir / "runtime.png").exists()
    assert (tmp_dir / "tradeoff_curve.png").exists()
    
    # Validate 7.5% energy reduction
    hybrid_energy = benchmark_results["hybrid"]["summary"]["avg_energy"]
    a100_energy = benchmark_results["a100"]["summary"]["avg_energy"]
    energy_reduction = (a100_energy - hybrid_energy) / a100_energy * 100
    assert energy_reduction >= 7.5, f"Energy reduction {energy_reduction:.2f}% is less than 7.5%"
