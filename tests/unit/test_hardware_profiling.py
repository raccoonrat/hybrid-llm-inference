# hybrid-llm-inference/tests/unit/test_hardware_profiling.py
import pytest
import pandas as pd
from pathlib import Path
from src.hardware_profiling import get_profiler
from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from src.hardware_profiling.a800_profiling import A800Profiler
from src.hardware_profiling.a100_profiler import A100Profiler
from src.hardware_profiling.m1_pro_profiler import M1ProProfiler
from src.hardware_profiling.base_profiler import HardwareProfiler
from src.benchmarking.system_benchmarking import SystemBenchmarking
from src.benchmarking.model_benchmarking import ModelBenchmarking
from src.toolbox.logger import get_logger
import pynvml
import os

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

@pytest.fixture
def hardware_config():
    """Mock hardware configuration."""
    return {
        "rtx4050": {"type": "gpu", "device_id": 0, "idle_power": 15.0, "sample_interval": 200}
    }

def test_rtx4050_profiler_initialization():
    """测试 RTX4050Profiler 初始化"""
    # 设置测试模式环境变量
    os.environ['TEST_MODE'] = '1'
    
    profiler = RTX4050Profiler(
        device_id=0,
        idle_power=15.0,
        sample_interval=200
    )
    
    assert profiler.device_id == 0
    assert profiler.idle_power == 15.0
    assert profiler.sample_interval == 200

def test_get_profiler_rtx4050():
    """测试获取 RTX4050 性能分析器"""
    # 设置测试模式环境变量
    os.environ['TEST_MODE'] = '1'
    
    config = {
        "type": "gpu",
        "device_id": 0,
        "idle_power": 15.0,
        "sample_interval": 200
    }
    
    profiler = get_profiler("rtx4050", config)
    assert isinstance(profiler, RTX4050Profiler)
    assert profiler.sample_interval == 200
    
def test_get_profiler_invalid():
    """测试获取无效的分析器类型"""
    with pytest.raises(ValueError):
        get_profiler("invalid_type", {})
        
def test_rtx4050_profiler_measure():
    """测试RTX 4050分析器的测量功能"""
    config = {
        "device_id": 0,
        "idle_power": 15.0,
        "sample_interval": 200
    }
    profiler = RTX4050Profiler(config)
    
    def mock_task():
        """模拟任务"""
        pass
    
    metrics = profiler.measure(mock_task, input_tokens=10, output_tokens=20)
    
    assert isinstance(metrics, dict)
    assert "energy" in metrics
    assert "runtime" in metrics
    assert "throughput" in metrics
    assert "energy_per_token" in metrics
    assert metrics["energy"] >= 0
    assert metrics["runtime"] >= 0
    assert metrics["throughput"] >= 0
    assert metrics["energy_per_token"] >= 0

def setup_nvml_mocks(monkeypatch):
    """设置NVML函数的模拟"""
    def mock_nvml_init():
        return None
    def mock_nvml_get_handle(index):
        return f"handle_{index}"
    def mock_nvml_get_power(handle):
        return 100000  # 100W in milliwatts
    def mock_nvml_get_name(handle):
        return "NVIDIA GeForce RTX 4050 Laptop GPU"
    def mock_nvml_shutdown():
        return None
    
    monkeypatch.setattr(pynvml, "nvmlInit", mock_nvml_init)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetHandleByIndex", mock_nvml_get_handle)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetPowerUsage", mock_nvml_get_power)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetName", mock_nvml_get_name)
    monkeypatch.setattr(pynvml, "nvmlShutdown", mock_nvml_shutdown)

def test_system_benchmarking_small_dataset(mock_dataset, hardware_config, model_config, output_dir, monkeypatch):
    """Test system benchmarking with a small dataset."""
    # 设置NVML模拟
    setup_nvml_mocks(monkeypatch)

    def mock_measure(task, input_tokens, output_tokens):
        task()
        # 返回固定的测试值
        return {
            "energy": 10.0,
            "runtime": 2.0,
            "throughput": (input_tokens + output_tokens) / 2.0,
            "energy_per_token": 10.0 / (input_tokens + output_tokens),
            "total_tasks": 1
        }
    monkeypatch.setattr("hardware_profiling.base_profiler.HardwareProfiler.measure", mock_measure)
    
    # 使用与模型基准测试相同的模拟模型类
    class MockModel:
        def __init__(self, *args, **kwargs):
            pass
        
        def infer(self, prompt):
            return "Mock response"
            
        def get_token_count(self, text):
            return min(len(text.split()), 50)
            
    # 替换实际的模型类
    monkeypatch.setattr("model_zoo.llama3.Llama3Model", MockModel)
    monkeypatch.setattr("model_zoo.falcon.FalconModel", MockModel)

    # 设置离线模式
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    # 修改测试名称以触发基础分析器的使用
    monkeypatch.setattr("src.scheduling.task_allocator.__name__", "test_module")

    benchmarker = SystemBenchmarking(mock_dataset, hardware_config, model_config, output_dir=output_dir)
    results = benchmarker.run_benchmarks(
        hybrid_threshold=0.5,
        homogeneous_threshold=0.5,
        model_name="llama3",
        sample_size=10
    )

    # 验证结果
    assert isinstance(results, dict)
    assert "hybrid" in results
    assert "homogeneous" in results
    assert all(isinstance(results[mode]["energy"], float) for mode in ["hybrid", "homogeneous"])
    assert all(isinstance(results[mode]["runtime"], float) for mode in ["hybrid", "homogeneous"])
    assert results["hybrid"]["energy"] <= results["homogeneous"]["energy"]

def test_model_benchmarking_small_dataset(mock_dataset, hardware_config, model_config, output_dir, monkeypatch):
    """Test model benchmarking with a small dataset."""
    # 首先设置NVML模拟
    setup_nvml_mocks(monkeypatch)

    def mock_measure(task, input_tokens, output_tokens):
        task()
        # 模拟RTX 4050的能耗
        energy = 8.0
        if input_tokens > 32 or output_tokens > 32:
            energy *= 1.5
        return {
            "energy": energy,
            "runtime": 2.0,
            "throughput": (input_tokens + output_tokens) / 2.0,
            "energy_per_token": energy / (input_tokens + output_tokens)
        }
    monkeypatch.setattr("hardware_profiling.base_profiler.HardwareProfiler.measure", mock_measure)
    
    # 模拟模型类
    class MockModel:
        def __init__(self, *args, **kwargs):
            pass
        
        def infer(self, prompt):
            return "Mock response"
            
        def get_token_count(self, text):
            return min(len(text.split()), 50)
            
    # 替换实际的模型类
    monkeypatch.setattr("model_zoo.llama3.Llama3Model", MockModel)
    monkeypatch.setattr("model_zoo.falcon.FalconModel", MockModel)

    # 设置离线模式
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    # 修改测试名称以触发基础分析器的使用
    monkeypatch.setattr("src.scheduling.task_allocator.__name__", "test_module")

    benchmarker = ModelBenchmarking(mock_dataset, hardware_config, model_config, output_dir=output_dir)
    results = benchmarker.run_benchmarks(sample_size=10)  # 减小样本大小以加快测试

    # 验证结果
    assert isinstance(results, dict)
    assert "falcon" in results
    assert "llama3" in results
    assert all(isinstance(results[model]["energy"], float) for model in results)
    assert all(isinstance(results[model]["runtime"], float) for model in results)
