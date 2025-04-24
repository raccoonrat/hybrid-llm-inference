"""基准测试模块。"""

import os
import pytest
from pathlib import Path
import sys
from typing import Dict, Any, List

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.benchmarking.system_benchmarking import SystemBenchmarking
from src.benchmarking.model_benchmarking import ModelBenchmarking
from src.model_zoo.base_model import BaseModel

# 设置测试模式
os.environ['TEST_MODE'] = 'true'

@pytest.fixture
def mock_dataset(tmp_path):
    """创建模拟数据集"""
    data = pd.DataFrame([
        {"prompt": "test1", "response": "response1"},
        {"prompt": "test2", "response": "response2"}
    ])
    dataset_path = tmp_path / "test.json"
    data.to_json(dataset_path, orient="records")
    return dataset_path

@pytest.fixture
def hardware_config() -> Dict[str, Any]:
    """硬件配置"""
    return {
        "device_type": "rtx4050",
        "idle_power": 30.0,
        "sample_interval": 200
    }

@pytest.fixture
def model_config() -> Dict[str, Any]:
    """模型配置"""
    return {
        "model_name": "mock_model",
        "model_path": "models/mock_model",
        "mode": "local",
        "batch_size": 1,
        "max_length": 2048
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
    """创建输出目录"""
    return tmp_path / "output"

@pytest.fixture
def system_benchmarking(hardware_config: Dict[str, Any], model_config: Dict[str, Any]) -> SystemBenchmarking:
    """创建系统基准测试实例"""
    return SystemBenchmarking(hardware_config, model_config)

@pytest.fixture
def model_benchmarking(hardware_config: Dict[str, Any], model_config: Dict[str, Any]) -> ModelBenchmarking:
    """创建模型基准测试实例"""
    return ModelBenchmarking(hardware_config, model_config)

def test_system_benchmarking_initialization(system_benchmarking: SystemBenchmarking):
    """测试系统基准测试初始化"""
    assert system_benchmarking is not None
    assert system_benchmarking.hardware_config is not None
    assert system_benchmarking.model_config is not None

def test_system_benchmarking_run_benchmark(system_benchmarking: SystemBenchmarking):
    """测试系统基准测试运行"""
    tasks = ["测试任务1", "测试任务2"]
    result = system_benchmarking.run_benchmark(tasks)
    assert result is not None
    assert "tasks" in result
    assert "hardware_config" in result
    assert "model_config" in result

def test_system_benchmarking_cleanup(system_benchmarking: SystemBenchmarking):
    """测试系统基准测试清理"""
    system_benchmarking.cleanup()
    assert True  # 如果清理成功，不会抛出异常

def test_model_benchmarking_initialization(model_benchmarking: ModelBenchmarking):
    """测试模型基准测试初始化"""
    assert model_benchmarking is not None
    assert model_benchmarking.hardware_config is not None
    assert model_benchmarking.model_config is not None

def test_model_benchmarking_run_benchmark(model_benchmarking: ModelBenchmarking):
    """测试模型基准测试运行"""
    tasks = ["测试任务1", "测试任务2"]
    result = model_benchmarking.run_benchmark(tasks)
    assert result is not None
    assert "tasks" in result
    assert "hardware_config" in result
    assert "model_config" in result

def test_model_benchmarking_cleanup(model_benchmarking: ModelBenchmarking):
    """测试模型基准测试清理"""
    model_benchmarking.cleanup()
    assert True  # 如果清理成功，不会抛出异常

def test_system_benchmarking_small_dataset(mock_dataset, hardware_config, model_config, output_dir):
    """测试小数据集系统基准测试"""
    # 设置测试模式环境变量
    os.environ['TEST_MODE'] = '1'
    
    benchmarker = SystemBenchmarking(
        dataset_path=mock_dataset,
        hardware_config=hardware_config,
        model_config=model_config,
        output_dir=output_dir
    )
    
    results = benchmarker.run_benchmarks()
    
    assert isinstance(results, dict)
    assert "energy" in results
    assert "runtime" in results
    assert "total_tasks" in results
    assert results["total_tasks"] > 0

def test_model_benchmarking_small_dataset(mock_dataset, hardware_config, model_config, output_dir):
    """测试小数据集模型基准测试"""
    # 设置测试模式环境变量
    os.environ['TEST_MODE'] = '1'
    
    benchmarker = ModelBenchmarking(
        dataset_path=mock_dataset,
        hardware_config=hardware_config,
        model_config=model_config,
        output_dir=output_dir
    )
    
    results = benchmarker.run_benchmarks()
    
    assert isinstance(results, dict)
    assert "mock_model" in results
    assert "energy" in results["mock_model"]
    assert "runtime" in results["mock_model"]
    assert "total_tasks" in results["mock_model"]
    assert results["mock_model"]["total_tasks"] > 0

def test_system_benchmarking_large_sample_size(mock_dataset, hardware_config, model_config, output_dir):
    """测试大样本量系统基准测试"""
    # 设置测试模式环境变量
    os.environ['TEST_MODE'] = '1'
    
    benchmarker = SystemBenchmarking(
        dataset_path=mock_dataset,
        hardware_config=hardware_config,
        model_config=model_config,
        output_dir=output_dir
    )
    
    results = benchmarker.run_benchmarks(sample_size=100)
    
    assert isinstance(results, dict)
    assert "energy" in results
    assert "runtime" in results
    assert "total_tasks" in results
    assert results["total_tasks"] > 0

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
        benchmarker.run_benchmarks(thresholds, model_name="mock_model", sample_size=3)

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
        benchmarker.run_benchmarks(thresholds, model_name="mock_model", sample_size=3)

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
