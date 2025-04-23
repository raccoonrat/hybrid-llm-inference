# hybrid-llm-inference/tests/unit/test_scheduling.py
import pytest
from scheduling.token_based_scheduler import TokenBasedScheduler
from scheduling.task_allocator import TaskAllocator

@pytest.fixture
def scheduler_config():
    return {
        "hardware_map": {
            "m1_pro": "apple_m1_pro",
            "a100": "nvidia_a100"
        }
    }

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

def test_token_based_scheduler(scheduler_config):
    thresholds = {"T_in": 32, "T_out": 32}
    scheduler = TokenBasedScheduler(thresholds, scheduler_config)
    
    token_data = [
        {"prompt": "Write a story", "response": "Once upon a time", "input_tokens": 10, "output_tokens": 20},
        {"prompt": "Explain AI", "response": "AI is...", "input_tokens": 50, "output_tokens": 60}
    ]
    
    allocations = scheduler.schedule(token_data)
    
    assert len(allocations) == 2
    assert allocations[0]["hardware"] == "apple_m1_pro"
    assert allocations[1]["hardware"] == "nvidia_a100"
    assert allocations[0]["query"] == token_data[0]

def test_token_based_scheduler_empty_data(scheduler_config):
    thresholds = {"T_in": 32, "T_out": 32}
    scheduler = TokenBasedScheduler(thresholds, scheduler_config)
    allocations = scheduler.schedule([])
    
    assert len(allocations) == 0

def test_task_allocator(hardware_config, model_config, monkeypatch):
    def mock_measure(task, input_tokens, output_tokens):
        task()
        return {"energy": 10.0, "runtime": 2.0, "throughput": 15.0, "energy_per_token": 0.5}
    monkeypatch.setattr("hardware_profiling.base_profiler.HardwareProfiler.measure", mock_measure)
    def mock_infer(prompt): return "Mock response"
    monkeypatch.setattr("model_zoo.base_model.BaseModel.infer", mock_infer)
    
    allocator = TaskAllocator(hardware_config, model_config)
    
    allocations = [
        {"query": {"prompt": "Write a story", "input_tokens": 10, "output_tokens": 20}, "hardware": "m1_pro"},
        {"query": {"prompt": "Explain AI", "input_tokens": 50, "output_tokens": 60}, "hardware": "a100"}
    ]
    
    results = allocator.allocate(allocations, model_name="llama3")
    
    assert len(results) == 2
    assert results[0]["metrics"]["energy"] == 10.0
    assert results[0]["query"] == allocations[0]["query"]

def test_task_allocator_invalid_hardware(hardware_config, model_config, monkeypatch):
    def mock_measure(task, input_tokens, output_tokens):
        task()
        return {"energy": 10.0, "runtime": 2.0, "throughput": 15.0, "energy_per_token": 0.5}
    monkeypatch.setattr("hardware_profiling.base_profiler.HardwareProfiler.measure", mock_measure)
    def mock_infer(prompt): return "Mock response"
    monkeypatch.setattr("model_zoo.base_model.BaseModel.infer", mock_infer)
    
    allocator = TaskAllocator(hardware_config, model_config)
    
    allocations = [{"query": {"prompt": "Test", "input_tokens": 10, "output_tokens": 20}, "hardware": "invalid"}]
    results = allocator.allocate(allocations, model_name="llama3")
    
    assert len(results) == 0
